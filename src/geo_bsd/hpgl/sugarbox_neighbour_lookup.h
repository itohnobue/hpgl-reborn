#ifndef __SUGARBOX_NEIGHBOUR_LOOKUP_H__B31AEF89_3C7A_48C8_97DD_F2983ED6A237
#define __SUGARBOX_NEIGHBOUR_LOOKUP_H__B31AEF89_3C7A_48C8_97DD_F2983ED6A237

#include "typedefs.h"
#include "covariance_field.h"
#include "neighbourhood_param.h"
#include "sugarbox_grid.h"
#include "neighbourhood_lookup.h"
#include <algorithm>
#include <memory>
#include <utility>

namespace hpgl
{
	// E2-145 (CONFIRMED MEDIUM): absolute per-node work cap for neighbour
	// scanning on BOTH sugarbox lookup paths. max_neighbours is honored only
	// up to this cap — the effective count bound is
	// min(max_neighbours, NEIGHBOUR_WORK_CAP) — replacing the old
	// SCAN_LIMIT=4913 that silently truncated every request above 4913.
	// The value is coordinated with the api.cpp MAX_NEIGHBOURS cap (E-M85,
	// target ~10000; another fix agent lowers the C API bound there).
	// clusterizer.cpp uses the same value for its candidate-copy bound
	// (E-M65) — keep the three sites in sync.
	constexpr int NEIGHBOUR_WORK_CAP = 10000;

	// R-15 (CONFIRMED MEDIUM): absolute per-node EVALUATION cap for the
	// covariance-ordered main scan in neighbour_lookup_t::find(). The
	// offset list is radius-bounded and covariance-threshold-filtered at
	// construction (calc_cov_field keeps only offsets with cov > C(0)/100),
	// so in dense regions the count early-exit fires after <= effective_max
	// admissions (<= NEIGHBOUR_WORK_CAP evaluations). In sparse regions the
	// count stays low and the scan would walk the ENTIRE list — up to
	// ~1e8 offsets at the E2-139 volume cap -> ~1e14 evaluations on a
	// 1M-node grid (hang class). The cap bounds per-node evaluations at
	// 100000: 14x the E-H2 correctness case (the distance-12 datum ranked
	// 7,154 in the covariance order), so far data with non-negligible
	// covariance are still admitted — only offsets ranked beyond the cap
	// are left unevaluated (their covariance sits in the tail of the
	// admission band, within an order of magnitude of the C(0)/100
	// threshold). Dense regions never reach the cap (the count early-exit
	// fires first).
	constexpr int NEIGHBOUR_SCAN_WORK_CAP = 100000;

	// R2-07 (CONFIRMED MEDIUM): the pure-nugget box fallback in find() scans
	// the full (2r+1)³ box whenever the heap window never fills (fewer than
	// effective_max pred-passing candidates in the whole box — sparse
	// regions, pure-nugget models). At r=231 that is 99,252,847 offsets per
	// find() call -> ~1e14 evaluations on a 1M-node grid (hang class); the
	// R-15 cap bounds only the MAIN scan. The fallback is bounded here to
	// the NEIGHBOUR_SCAN_WORK_CAP NEAREST lattice offsets: fallback_d2_cap()
	// returns the largest squared distance whose in-box lattice count fits
	// the cap (or -1 when the whole box fits — every reachable config keeps
	// the exact E-M66 full-radius admission). The scan then skips offsets
	// beyond the cap with the same monotone layer/row/column break/continue
	// used for the window-full pruning, so the evaluated set stays
	// distance-ordered nearest-first (E2-141). A flat evaluation counter in
	// the raw z,y,x scan order would truncate at the FARTHEST corner and
	// re-introduce the farthest-first line bias E2-141 fixed — hence the
	// distance cap, not a counter.
	// Exact count of box offsets (|dx|<=rx, |dy|<=ry, |dz|<=rz) whose
	// squared distance from the origin is <= d2_limit. O(lattice count
	// within the limit) — used by the binary search in fallback_d2_cap.
	inline long long count_box_offsets_within_d2(int rx, int ry, int rz, long long d2_limit)
	{
		long long n = 0;
		for (int dz = -rz; dz <= rz; ++dz)
		{
			const long long dz2 = static_cast<long long>(dz) * dz;
			if (dz2 > d2_limit)
				continue;
			for (int dy = -ry; dy <= ry; ++dy)
			{
				const long long ddy = dz2 + static_cast<long long>(dy) * dy;
				if (ddy > d2_limit)
					continue;
				for (int dx = -rx; dx <= rx; ++dx)
				{
					if (ddy + static_cast<long long>(dx) * dx <= d2_limit)
						++n;
				}
			}
		}
		return n;
	}

	// Largest squared distance whose in-box lattice count fits
	// NEIGHBOUR_SCAN_WORK_CAP; -1 when the whole box fits (no truncation).
	// Callers must pass radii already validated by calc_cov_field (the
	// E2-139 volume cap bounds the box product at 1e8, so the arithmetic
	// below cannot overflow on legal inputs).
	inline long long fallback_d2_cap(int rx, int ry, int rz)
	{
		const unsigned long long volume = (2ULL * static_cast<unsigned int>(rx) + 1ULL)
			* (2ULL * static_cast<unsigned int>(ry) + 1ULL)
			* (2ULL * static_cast<unsigned int>(rz) + 1ULL);
		if (volume <= static_cast<unsigned long long>(NEIGHBOUR_SCAN_WORK_CAP))
			return -1;
		const long long max_d2 = static_cast<long long>(rx) * rx
			+ static_cast<long long>(ry) * ry
			+ static_cast<long long>(rz) * rz;
		long long lo = 0, hi = max_d2;
		while (lo < hi)
		{
			const long long mid = lo + (hi - lo + 1) / 2;
			if (count_box_offsets_within_d2(rx, ry, rz, mid) <= NEIGHBOUR_SCAN_WORK_CAP)
				lo = mid;
			else
				hi = mid - 1;
		}
		return lo;
	}

	template<typename covariances_t>
	class neighbour_lookup_t<sugarbox_grid_t, covariances_t>
	{
		template <typename grid_t, typename cov_t> friend class indexed_neighbour_lookup_t;
		std::shared_ptr<std::vector<sugarbox_vector_t>> m_vectors;
		const sugarbox_grid_t * m_grid;
		int m_max_neighbours;
		// Stored search ellipsoid — used by the pure-nugget fallback in
		// find() (2-M-32) to admit radius-bounded nodes when the covariance
		// threshold produced an empty offset list.
		sugarbox_search_ellipsoid_t m_radiuses;
		// R2-07: squared-distance bound for the pure-nugget fallback scan
		// (see fallback_d2_cap). -1 = whole box fits the eval budget; the
		// fallback then scans the full radius exactly as before. Computed in
		// the ctor body AFTER calc_cov_field (radii validated there).
		long long m_fallback_d2_cap;
	public:		
		typedef sugarbox_location_t coord_t;
		typedef sugarbox_grid_t grid_t;

		neighbour_lookup_t(
				const sugarbox_grid_t * grid,				
				const covariances_t * cov, 
				const neighbourhood_param_t & nb_param
				)
			: m_vectors(new std::vector<sugarbox_vector_t>()),
			  m_grid(grid), 
			  m_max_neighbours((int)nb_param.m_max_neighbours),
			  m_radiuses(nb_param.m_radiuses),
			  m_fallback_d2_cap(-1)
		{
			// E2-139: radius-vs-grid sanity validation. The covariance box
			// below is O((2r+1)³); its MEMORY bound is enforced by
			// validate_covariance_radiuses_or_throw inside calc_cov_field
			// (COVARIANCE_FIELD_VOLUME_CAP = 1e8 cells ≈ 0.8 GB doubles +
			// ≤ 1.2 GB offset vectors worst case).
			// R-21 (CONFIRMED MEDIUM): the former "radius ≤ 10x grid
			// extent" heuristic is REMOVED. It rejected shipped workflows
			// on the assumption that a radius > 10x an axis extent is
			// always a units error: book 7.3/2_var.py (grid 10x10x1,
			// radiuses (160,40,1) = 16x in x -> box (321·81·3) = 78,003
			// cells) and sample-scripts/sk_test.py (grid 286x10x1,
			// radiuses (20,20,20) -> box 41³ = 68,921 cells) both threw,
			// and both are far below the volume cap. Data cannot exist
			// beyond the grid, so an oversized radius only materializes a
			// bounded field — the volume cap alone bounds the memory, and
			// radiuses up to it remain legal whole-grid searches
			// (test_legacy_migrated.py:626 uses radius 100 on a 20x20x10
			// grid and still constructs).
			calc_cov_field<covariances_t, sugarbox_location_t>(nb_param.m_radiuses, *cov, *m_vectors);
			// R2-07: bound the fallback scan once per lookup. Computed
			// after calc_cov_field so radii are already E2-139-validated.
			m_fallback_d2_cap = fallback_d2_cap(
				static_cast<int>(nb_param.m_radiuses[0]),
				static_cast<int>(nb_param.m_radiuses[1]),
				static_cast<int>(nb_param.m_radiuses[2]));
		}

		template<typename filter_pred_t>
		void find(node_index_t node, const filter_pred_t & pred, coord_t & node_coord, std::vector<node_index_t> & indices, std::vector<coord_t> & coords)const
		{
			indices.clear();
			indices.reserve(m_max_neighbours);
			coords.clear();
			coords.reserve(m_max_neighbours);
			const std::vector<sugarbox_vector_t> & vectors = *m_vectors;
			sugarbox_location_t center = (*m_grid)[node];
			node_coord = coord_t(center[0], center[1], center[2]);
			// II-13: max_neighbours == 0 is the documented "unconditional
			// simulation" mode — an empty neighbourhood by contract. The
			// pure-nugget fallback below would otherwise fire on count == 0
			// and silently convert unconditional simulation into 1-neighbour
			// conditioned kriging (live probe: mean 4.7493 conditioned vs
			// 0 unconditional). node_coord is set above so callers keep the
			// pre-fix behaviour for the empty result.
			if (m_max_neighbours <= 0)
				return;
			// E2-145: the effective count bound honors the configured
			// max_neighbours up to the absolute work cap (NEIGHBOUR_WORK_CAP,
			// coordinated with the api.cpp E-M85 cap). Pre-fix the
			// SCAN_LIMIT=4913 index cap silently truncated the returned
			// count for every max_neighbours ∈ (4913, 100000] on both
			// lookup paths.
			const int effective_max = m_max_neighbours < NEIGHBOUR_WORK_CAP ? m_max_neighbours : NEIGHBOUR_WORK_CAP;
			int count = 0;
			if (!vectors.empty())
			{
				const sugarbox_vector_t * vec = &vectors[0];
				// E-H2 (CONFIRMED HIGH): scan the FULL covariance-ordered
				// offset list — no fixed index cap. The list is already
				// radius-bounded (search ellipsoid) and covariance-threshold
				// filtered (cov > C(0)/100) at construction (calc_cov_field),
				// so every offset it contains can carry a non-negligible
				// covariance. The old SCAN_LIMIT=4913 silently collapsed the
				// effective search radius to ~10.6 cells (the 4913th lattice
				// offset: a distance-12 datum ranks 7,154 > 4,913) and
				// SGS/SIS/cokriging mean/marginal-filled on sparse data +
				// radius ≥ 10 + range ≥ 10; the R-6 "negligible weight"
				// justification was FALSE (range-20/distance-12 cov =
				// 0.208·sill = 20× the C(0)/100 threshold). The per-node
				// work stays bounded by the count early-exit in dense
				// regions and by the R-15 evaluation cap
				// (NEIGHBOUR_SCAN_WORK_CAP, below) in sparse regions — the
				// same full-radius admission the indexed sibling applies in
				// sparse regions, minus the unevaluated tail beyond the cap.
				int evals = 0;
				for (int idx = 0, end_idx = (int) vectors.size(); idx < end_idx && count < effective_max && evals < NEIGHBOUR_SCAN_WORK_CAP; ++idx, ++vec)
				{
					++evals;
					sugarbox_location_t point = center + *vec; //vectors[idx];
					int index = m_grid->get_index(point);
					if (index >= 0)
					{		
						if (pred(index)) 
						{
							indices.push_back(index);
							coords.push_back(coord_t(point[0], point[1], point[2]));						
							++count;
						}
					}
				}
			}
			if (count == 0)
			{
				// 2-M-32 / E-M66 / E2-141: pure-nugget fallback. calc_cov_field
				// builds the offset list by covariance threshold (cov > C(0)/100);
				// for a pure-nugget model (nugget==sill) every h>0 covariance is 0,
				// so only the self-offset (0,0,0) survives — and the center is
				// never a valid conditioning datum (it is the node being
				// estimated). The neighbourhood would then be EMPTY → every
				// consumer of this plain lookup (cokriging, SGS, SIS) silently
				// mean-fills / falls back, while the indexed lookup
				// (sugarbox_indexed_neighbour_lookup.h) has a pure-nugget
				// fallback that admits radius-bounded nodes. GSLIB's search is
				// radius-based — data inside the search ellipsoid are admitted
				// and covariance only ranks.
				// E-M66 (CONFIRMED MEDIUM, test-pinned divergence): the old
				// FALLBACK_WINDOW=8 admitted only min(radius,8) cells per
				// direction while the indexed sibling admits the FULL radius —
				// the same neighbourhood_param_t produced different
				// neighbourhoods for plain consumers (SGS/SIS/cokriging) vs
				// indexed consumers (OK/IK/median-IK). The aligned behavior is
				// FULL radius.
				// E2-141 (CONFIRMED MEDIUM): the old window also admitted
				// neighbours in raw z-major/x-minor scan order starting at the
				// FARTHEST corner (farthest-first, line-biased). The fallback
				// now collects the radius-bounded candidates with their squared
				// distances, keeps the nearest effective_max, and admits them
				// distance-sorted nearest-first — mirroring the indexed sibling
				// (isotropic-distance-ordered pool) and GSLIB semantics.
				// R-6 / R-13 (CONFIRMED MEDIUM): this fallback runs PER
				// find() call. The pre-fix FALLBACK_WINDOW=8 bounded the
				// work but re-introduced the window cap E-M66 forbids, so
				// the work bound must come from the admission process
				// itself. Two bounds, both preserving the E-M66 full-radius
				// window and the E2-141 nearest-first distance-sorted
				// admission:
				//  - Memory: a thread_local scratch (E2-129 pattern) that
				//    never grows beyond effective_max entries — a max-heap
				//    window keeps only the nearest effective_max candidates
				//    seen so far (16 B x effective_max, <= 160 KB at the
				//    NEIGHBOUR_WORK_CAP ceiling) instead of a per-call
				//    vector of up to (2r+1)³ pairs (3.6 MB at r=30, 1.59 GB
				//    at r=231) inside the OpenMP kriging loops.
				//  - Work: once the window is full, an offset whose squared
				//    distance is >= the current worst admitted distance can
				//    never enter the window, so the scan prunes by
				//    layer/row/column and terminates as soon as every
				//    remaining offset is at least as far as the worst
				//    admitted candidate. In dense regions this bounds the
				//    scan at O(effective_max) offsets instead of (2r+1)³.
				// R2-07 (CONFIRMED MEDIUM): in sparse regions (fewer than
				// effective_max candidates in the whole box) the window
				// never fills, worst_d2 stays -1 and the pruning above
				// never engages — the full (2r+1)³ box was scanned per
				// find() call (99,252,847 offsets at r=231 -> ~1e14 on a
				// 1M-node grid). The scan is now bounded by
				// m_fallback_d2_cap (fallback_d2_cap): the NEAREST
				// NEIGHBOUR_SCAN_WORK_CAP lattice offsets are evaluated
				// (distance cap, not a flat counter — a counter would
				// truncate at the farthest corner and re-introduce the
				// E2-141 farthest-first bias). Offsets beyond the cap are
				// skipped with the same monotone layer/row/column
				// break/continue as the window pruning, so the admitted
				// set stays nearest-first; -1 (whole box fits the budget)
				// keeps the exact E-M66 full-radius scan for every
				// reachable config (box volume <= 1e5 cells).
				const int rx = static_cast<int>(m_radiuses[0]);
				const int ry = static_cast<int>(m_radiuses[1]);
				const int rz = static_cast<int>(m_radiuses[2]);
				thread_local std::vector<std::pair<int, long long> > fallback;
				fallback.clear();
				fallback.reserve(static_cast<size_t>(effective_max));
				// Max-heap window: push_heap/pop_heap keep the worst
				// admitted candidate at the front. Ties (equal squared
				// distance) keep the earlier-scanned candidate — a
				// deterministic total order for the fixed z,y,x scan order.
				auto farther = [](const std::pair<int, long long> & a, const std::pair<int, long long> & b)
				{
					return a.second < b.second;
				};
				const size_t cap = static_cast<size_t>(effective_max);
				long long worst_d2 = -1; // < 0 while the window is not full (no pruning yet)
				for (int dz = -rz; dz <= rz; ++dz)
				{
					const long long dz2 = static_cast<long long>(dz) * dz;
					if (m_fallback_d2_cap >= 0 && dz2 > m_fallback_d2_cap)
					{
						if (dz >= 0)
							break;   // remaining z-layers are all beyond the eval budget
						continue;    // this layer is beyond the budget; closer layers may not be
					}
					if (fallback.size() == cap && dz2 >= worst_d2)
					{
						if (dz >= 0)
							break;   // remaining z-layers are all at least as far as the worst admitted candidate
						continue;    // this layer cannot improve the window; closer layers may
					}
					for (int dy = -ry; dy <= ry; ++dy)
					{
						const long long row_min_d2 = dz2 + static_cast<long long>(dy) * dy;
						if (m_fallback_d2_cap >= 0 && row_min_d2 > m_fallback_d2_cap)
						{
							if (dy >= 0)
								break;   // remaining y-rows are all beyond the eval budget
							continue;    // this row is beyond the budget; closer rows may not be
						}
						if (fallback.size() == cap && row_min_d2 >= worst_d2)
						{
							if (dy >= 0)
								break;   // remaining y-rows are all at least as far
							continue;    // this row cannot improve; closer rows may
						}
						for (int dx = -rx; dx <= rx; ++dx)
						{
							const long long d2 = row_min_d2 + static_cast<long long>(dx) * dx;
							if (m_fallback_d2_cap >= 0 && d2 > m_fallback_d2_cap)
							{
								if (dx >= 0)
									break;   // remaining x-columns are all beyond the eval budget
								continue;
							}
							if (fallback.size() == cap && d2 >= worst_d2)
							{
								if (dx >= 0)
									break;   // remaining x-columns are all at least as far
								continue;
							}
							sugarbox_location_t point = center + sugarbox_vector_t(dx, dy, dz);
							int index = m_grid->get_index(point);
							if (index >= 0)
							{
								if (pred(index))
								{
									if (fallback.size() < cap)
									{
										fallback.push_back(std::make_pair(index, d2));
										std::push_heap(fallback.begin(), fallback.end(), farther);
										if (fallback.size() == cap)
											worst_d2 = fallback.front().second;
									}
									else if (d2 < worst_d2)
									{
										std::pop_heap(fallback.begin(), fallback.end(), farther);
										fallback.back() = std::make_pair(index, d2);
										std::push_heap(fallback.begin(), fallback.end(), farther);
										worst_d2 = fallback.front().second;
									}
								}
							}
						}
					}
				}
				if (!fallback.empty())
				{
					// Admit the nearest effective_max candidates
					// nearest-first (E2-141).
					std::sort(fallback.begin(), fallback.end(), farther);
					for (size_t i = 0; i < fallback.size(); ++i)
					{
						sugarbox_location_t point = (*m_grid)[fallback[i].first];
						indices.push_back(fallback[i].first);
						coords.push_back(coord_t(point[0], point[1], point[2]));
						++count;
					}
				}
			}
		}		
	};

}		

#endif //__SUGARBOX_NEIGHBOUR_LOOKUP_H__B31AEF89_3C7A_48C8_97DD_F2983ED6A237
