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

		// Pre-compute total capacity to avoid chain reallocations.
		// At most 27 clusters (3×3×3 neighbour box), each capped at m_limit.
		// M-13: cap the reserve. 27*m_limit scales with the search-box volume
		// (m_limit = offsets/250; a legal radius-644 workflow reaches
		// ~8.57M → ~925 MB/thread reserve inside the OpenMP kriging loop),
		// and the caller only ever consumes up to its configured
		// max_neighbours (bounded by MAX_NEIGHBOURS_UPPER_BOUND = 100000).
		// std::vector grows on demand past the reserve, so capping only
		// sacrifices a reallocation in pathological cases — never
		// correctness.
		const size_t MAX_NEIGHBOUR_RESERVE = 100000;
		size_t reserve_cap = m_state->m_limit > MAX_NEIGHBOUR_RESERVE / 27
			? MAX_NEIGHBOUR_RESERVE : 27 * m_state->m_limit;
		neighbours.clear();
		neighbours.reserve(reserve_cap);

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
					const std::vector<node_index_t> & nodes = m_state->m_clusters[cluster_idx]->nodes();
					std::copy(nodes.begin(), nodes.end(), std::back_inserter(neighbours));					
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
