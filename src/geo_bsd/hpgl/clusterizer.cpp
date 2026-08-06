#include "stdafx.h"

#include "bs_assert.h"
#include "clusterizer.h"
#include "sugarbox_grid.h"
#include "hpgl_exception.h"
#include <memory>
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace hpgl
{
	class cluster_t
	{
		//non copyable
		cluster_t(const cluster_t &);
		cluster_t & operator=(const cluster_t &);	
	public:
		cluster_t (size_t limit)
		:	m_nodes(new std::vector<node_index_t>()),
			m_limit(limit),
			m_limit_exceeded(false)
		{
			m_nodes->reserve(limit);
		}		

		void add_node(node_index_t node)
		{
			if (m_limit_exceeded)
				return;
			if (m_nodes->size() >= m_limit)
			{
				m_limit_exceeded = true;
				m_nodes.reset();
				return;
			}
			m_nodes->push_back(node);
		}
		inline bool limit_exceeded() const { return m_limit_exceeded;}
		inline size_t count() const
		{
			if (m_limit_exceeded)
				{ fprintf(stderr, "HPGL FATAL: cluster_t::count: limit exceeded\n"); abort(); }
			return m_nodes->size();
		}
		const std::vector<node_index_t> & nodes() const{ return *m_nodes;}

	private:
		std::unique_ptr<std::vector<node_index_t> > m_nodes;
		size_t m_limit;
		bool m_limit_exceeded;
	};
	

	struct clusterizer_t::state
	{
		//typedef std::vector<node_index_t> cluster_t;
		// F-N11: clusters are created lazily on first add_node — the vector
		// holds null pointers for empty cells, so heap objects scale with
		// actual cluster occupancy instead of the cluster-grid volume.
		std::vector<std::unique_ptr<cluster_t> > m_clusters;
		rect_3d_t<int> m_cluster_box;
		int m_x;
		int m_y;
		int m_z;
		
		const sugarbox_grid_t * m_geometry;
		size_t m_limit;
	};
	
	clusterizer_t::clusterizer_t()
		: m_state(std::make_unique<state>())
	{
		throw hpgl_exception("clusterizer_t",
			"Default construction is not supported — use the parameterized constructor.");
	}

	clusterizer_t::clusterizer_t(
			const sugarbox_grid_t * grid, 			
			const sugarbox_search_ellipsoid_t & ellipsoid,
			size_t limit)
		: m_state(std::make_unique<state>())
	{		
		m_state->m_limit = limit;
		m_ellipsoid = ellipsoid;
		if (m_ellipsoid[0] == 0) m_ellipsoid[0] = 1;
		if (m_ellipsoid[1] == 0) m_ellipsoid[1] = 1;
		if (m_ellipsoid[2] == 0) m_ellipsoid[2] = 1;
		m_state->m_geometry = grid;
		
		int gx, gy, gz;
		grid->get_dimensions(gx, gy, gz);
		
		m_state->m_x = gx / m_ellipsoid[0] + 2;
		m_state->m_y = gy / m_ellipsoid[1] + 2;
		m_state->m_z = gz / m_ellipsoid[2] + 2;

		m_state->m_cluster_box = rect_3d_t<int>(0, 0, 0, m_state->m_x, m_state->m_y, m_state->m_z);
		// Prevent signed integer overflow: cast to size_t before
		// multiplying three ints (gx/rx+2)*(gy/ry+2)*(gz/rz+2).
		size_t total_volume = static_cast<size_t>(m_state->m_x)
		                    * static_cast<size_t>(m_state->m_y)
		                    * static_cast<size_t>(m_state->m_z);
		// F-N11: the previous cap (1e9, e791f6b) still admitted a legal
		// 998³ grid + radius 1, which allocated ~1e9 cluster_t heap
		// objects (each with its own reserve()'d vector) and hung before
		// kriging started.  Two prior attempts (b017bd9 size_t cast,
		// e791f6b 1e9 cap) only tweaked the arithmetic/cap.  Two-part fix:
		//   (1) Lazy cluster creation (add_node allocates on first touch)
		//       removes the per-cell heap-allocation amplifier entirely.
		//   (2) The cap bounds the up-front pointer vector.
		// M-12: the cap must bound the EFFECTIVE memory, not just the
		// pointer-vector count. The old 1e8 cap admitted fully-dense grids
		// (~464³) that lazily allocate ~1e8 cluster_t heap objects (~24-32 B
		// each plus a nested vector) → 2.4-3.2 GB — the previous "800MB
		// worst case" comment counted only the 8 B/slot pointer vector.
		// Per-cell worst case at fully-dense occupancy:
		//   8  B pointer slot          (std::unique_ptr in m_clusters)
		//  24  B cluster_t object      (unique_ptr + limit + flag + padding)
		//  24  B nested std::vector    (heap object created by the ctor)
		//   >=8 B reserve()'d buffer   (limit * sizeof(node_index_t), min 8)
		// Cap the product at 1 GiB so a fully-dense admitted grid stays
		// memory-safe (radius-1 dense grids up to ~1.7e7 cells ≈ 256³, far
		// beyond typical use — the largest test grid is 50³; larger grids
		// are served by increasing the search radius, which shrinks the
		// cluster grid quadratically).
		size_t reserve_bytes = m_state->m_limit * sizeof(node_index_t);
		size_t per_cell_cost = 8 + 24 + 24 + (reserve_bytes < 8 ? 8 : reserve_bytes);
		const size_t max_cells = (1024ULL * 1024 * 1024) / per_cell_cost;
		if (total_volume > max_cells)
			throw hpgl_exception("clusterizer_t::clusterizer_t",
				"Cluster grid volume exceeds the memory-safe limit of "
				+ std::to_string(max_cells)
				+ " cells — check grid dimensions and search radii.");
		m_state->m_clusters.resize(total_volume);
	}

	clusterizer_t::~clusterizer_t()
	{
	}


	int clusterizer_t::get_index(int cx, int cy, int cz)const
	{
		if (!m_state->m_cluster_box.has(cx, cy, cz))
			return -1;		
		return cz * m_state->m_x * m_state->m_y + cy * m_state->m_x + cx;
	}

	void clusterizer_t::add_node(node_index_t idx)
	{
		sugarbox_location_t loc = m_state->m_geometry->operator[](idx);	
		
		int cluster_idx = get_index_from_grid_point(loc);
		if (cluster_idx >= 0)
		{
			// F-N11: lazy cluster creation — allocate the cluster_t only
			// when a node actually lands in this cell.  Heap objects now
			// scale with occupancy, not with the cluster-grid volume.
			if (!m_state->m_clusters[cluster_idx])
				m_state->m_clusters[cluster_idx] = std::make_unique<cluster_t>(m_state->m_limit);
			m_state->m_clusters[cluster_idx]->add_node(idx);
		}
	}
	
	int clusterizer_t::get_nearby_harddata_count(node_index_t idx)const
	{
		sugarbox_location_t loc = m_state->m_geometry->operator[](idx);
		int cx = loc[0] / m_ellipsoid[0];
		int cy = loc[1] / m_ellipsoid[1];
		int cz = loc[2] / m_ellipsoid[2];

		int result = 0;
			
		for (int k = cz - 1; k <= cz + 1; ++k)
			for (int j = cy - 1; j	<= cy + 1; ++j)
				for (int i = cx - 1; i <= cx + 1; ++i)
				{
					int cluster_idx = get_index(i, j, k);
					if (cluster_idx >= 0)
					{
						// F-N11: lazily-created clusters may be null — skip
						// cells that never received a node.
						if (!m_state->m_clusters[cluster_idx])
							continue;
						if (m_state->m_clusters[cluster_idx]->limit_exceeded())
							{ fprintf(stderr, "HPGL FATAL: clusterizer_t: cluster limit exceeded\n"); abort(); }
			   			size_t cluster_size = m_state->m_clusters[cluster_idx]->count();
						result += cluster_size;				
					}
				}
		return result;
	}

	bool clusterizer_t::get_nearby_harddata(node_index_t idx, std::vector<node_index_t> & neighbours)const
	{
		sugarbox_location_t loc = m_state->m_geometry->operator[](idx);
		int cx = loc[0] / m_ellipsoid[0];
		int cy = loc[1] / m_ellipsoid[1];
		int cz = loc[2] / m_ellipsoid[2];

		// E-M65 (CONFIRMED MEDIUM): bound the per-node candidate COPY, not
		// just the caller's post-collection truncation. Pre-fix this routine
		// copied ALL nodes of the 27 neighbour clusters (27 × m_limit — a
		// legal radius-644 workflow reached ~925K candidates/node) and the
		// caller then ran its distance pass + nth_element over the full pool
		// inside the OpenMP kriging loop. The copy cap below stops the copy
		// as soon as the cap is reached — bounding the real work before any
		// distance math.
		// R-12 (CONFIRMED MEDIUM): the cap is 2 × NEIGHBOUR_WORK_CAP, NOT
		// equal to it. The indexed caller (sugarbox_indexed_neighbour_lookup.h)
		// keeps its pool RANKING bound at NEIGHBOUR_WORK_CAP (SCAN_LIMIT —
		// the E2-145 count cap shared with the plain lookup); if the copy
		// cap equaled that bound, the caller's covariance-order pool
		// selection (E-M69) could never fire: the pool was always the first
		// 10000 candidates in cluster-visit order and high-covariance
		// anisotropic candidates beyond the cap were silently dropped. The
		// 2× headroom (still 46× below the pre-E-M65 ~925K worst case)
		// makes the selection reachable on dense configs while keeping the
		// per-node copy bounded. The plain lookup path never uses the
		// clusterizer, so this cap affects only the indexed path.
		// Because the old cluster-visit order (z-major) visited the center
		// cluster 14th of 27, a plain prefix cap would drop the CLOSEST data
		// in dense regions (F-21/R-03). Clusters are therefore visited
		// center-first (by |cluster offset|²) so the cap keeps the closest
		// clusters' data.
		static const size_t WORK_CAP = 20000;  // 2 × NEIGHBOUR_WORK_CAP (sugarbox_neighbour_lookup.h) — see R-12 note above
		// The 27 cluster offsets of the 3×3×3 box, ordered by |offset|²
		// (center cluster first: 1 center, 6 face, 12 edge, 8 corner).
		static const int visit_offsets[27][3] = {
			{ 0,  0,  0},
			{ 1,  0,  0}, {-1,  0,  0}, { 0,  1,  0}, { 0, -1,  0}, { 0,  0,  1}, { 0,  0, -1},
			{ 1,  1,  0}, { 1, -1,  0}, {-1,  1,  0}, {-1, -1,  0},
			{ 1,  0,  1}, { 1,  0, -1}, {-1,  0,  1}, {-1,  0, -1},
			{ 0,  1,  1}, { 0,  1, -1}, { 0, -1,  1}, { 0, -1, -1},
			{ 1,  1,  1}, { 1,  1, -1}, { 1, -1,  1}, { 1, -1, -1},
			{-1,  1,  1}, {-1,  1, -1}, {-1, -1,  1}, {-1, -1, -1}
		};
		neighbours.clear();
		size_t reserve_cap = 27 * m_state->m_limit;
		if (reserve_cap > WORK_CAP)
			reserve_cap = WORK_CAP;
		neighbours.reserve(reserve_cap);

		for (size_t o = 0; o < 27; ++o)
		{
			int i = cx + visit_offsets[o][0];
			int j = cy + visit_offsets[o][1];
			int k = cz + visit_offsets[o][2];
			int cluster_idx = get_index(i, j, k);
			if (cluster_idx >= 0)
			{
				// F-N11: lazily-created clusters may be null — skip
				// cells that never received a node.
				if (!m_state->m_clusters[cluster_idx])
					continue;
				if (m_state->m_clusters[cluster_idx]->limit_exceeded())
					{ fprintf(stderr, "HPGL FATAL: clusterizer_t: cluster limit exceeded\n"); abort(); }
				const std::vector<node_index_t> & nodes = m_state->m_clusters[cluster_idx]->nodes();
				for (size_t n = 0, end_n = nodes.size(); n != end_n; ++n)
				{
					if (neighbours.size() >= WORK_CAP)
						return true;
					neighbours.push_back(nodes[n]);
				}
			}
		}
		return true;
	}
	
	bool clusterizer_t::limit_exceeded(node_index_t idx)const		
	{
		sugarbox_location_t loc = m_state->m_geometry->operator[](idx);
		int cx = loc[0] / m_ellipsoid[0];
		int cy = loc[1] / m_ellipsoid[1];
		int cz = loc[2] / m_ellipsoid[2];

		for (int k = cz - 1; k <= cz + 1; ++k)
			for (int j = cy - 1; j	<= cy + 1; ++j)
				for (int i = cx - 1; i <= cx + 1; ++i)
				{
					int cluster_idx = get_index(i, j, k);
					if (cluster_idx >= 0)
					{
						// F-N11: lazily-created clusters may be null — skip
						// cells that never received a node.
						if (!m_state->m_clusters[cluster_idx])
							continue;
						if (m_state->m_clusters[cluster_idx]->limit_exceeded())
							return true;			   		
					}
				}
		return false;
	}
		
}
