#ifndef __SUGARBOX_INDEXED_NEIGHBOUR_LOOKUP_H__9A00552C_FE2E_4B73_8F3D_18289B1492E6
#define __SUGARBOX_INDEXED_NEIGHBOUR_LOOKUP_H__9A00552C_FE2E_4B73_8F3D_18289B1492E6

#include "clusterizer.h"
#include "sugarbox_neighbour_lookup.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <utility>

namespace hpgl
{
	namespace detail
	{
		const int MAGIC_NUMBER_2 = 250;
		// E2-144 (CONFIRMED MEDIUM): minimum clusterizer cluster limit.
		// The old formula max(offsets/250, 1) collapses to 1 for pure-nugget
		// models (covariance field = single self offset), so any cluster
		// with ≥ 2 nodes trips limit_exceeded and the indexed fast path
		// (and its full-radius fallback) is disabled in dense regions —
		// silently routing indexed consumers (OK/IK/median-IK) to the plain
		// path. The floor keeps the fast path functional for degenerate
		// models while leaving the normal-model value (offsets/250, typically
		// ≥ 100) unchanged.
		// R-14 (CONFIRMED MEDIUM): the original floor of 32 priced the
		// clusterizer memory cap at 184 B/cell (8 B ptr slot + 24 B
		// cluster_t + 24 B nested vector + 128 B reserve for limit 32) →
		// max cluster-grid volume 5,835,553 cells, rejecting previously-
		// legal fine grids (200³ + radius 1 = 202³ = 8,242,408 cells
		// threw "memory-safe limit"). Lowered to 8: reserve 32 B →
		// 88 B/cell → max 12,201,611 cells, so 200³ + radius 1 constructs
		// again. The floor still keeps the indexed fast path functional
		// for pure-nugget models where it binds (radius ≤ 2: a cluster
		// cell holds at most rx·ry·rz = 8 nodes, exactly covered by limit
		// 8; radius-1 cells hold at most 1). Larger-radius degenerate
		// models trip the limit and take the (correct) dense-region
		// fallback, exactly as before.
		const size_t MIN_CLUSTER_LIMIT = 8;

		struct entry_t
		{
			node_index_t index;
			double cov_value;
			double distance2;       // squared isotropic distance from the estimation node (E-M67 tie-break key)
			sugarbox_location_t coord;  // cached grid coordinate (avoids redundant operator[] calls)
			// E-M67 (CONFIRMED MEDIUM): deterministic tie-break. Covariance
			// ties are routine on regular grids (48 lattice positions at
			// h²=14); the old covariance-only comparators over different
			// input permutations (cluster traversal vs box loop + nth_element
			// partition) picked tied candidates at the max_neighbours
			// boundary differently between OK (indexed) and SGS (plain) —
			// different kriging results for identical input. Order:
			// covariance desc, then squared distance asc, then node index
			// asc. index is unique, so this is a strict total order —
			// permutation-independent sort results.
			bool operator<(const entry_t & e)const
			{
					if (cov_value != e.cov_value)
						return cov_value < e.cov_value;
					if (distance2 != e.distance2)
						return distance2 < e.distance2;
					return index < e.index;
			}
			bool operator>(const entry_t & e)const
			{
					if (cov_value != e.cov_value)
						return cov_value > e.cov_value;
					if (distance2 != e.distance2)
						return distance2 > e.distance2;
					return index > e.index;
			}
			bool operator<=(const entry_t & e)const
			{
					return !(*this > e);
			}
			bool operator>=(const entry_t & e)const
			{
					return !(*this < e);
			}
			bool operator==(const entry_t & e)const
			{
					return cov_value == e.cov_value && distance2 == e.distance2 && index == e.index;
			}
		};

		// Named comparator for descending covariance sort (highest first)
		// with deterministic tie-break (E-M67): equal covariance → smaller
		// squared distance → smaller node index (strict total order).
		struct desc_cov_compare
		{
			bool operator()(const entry_t & a, const entry_t & b)const
			{
				if (a.cov_value != b.cov_value)
					return a.cov_value > b.cov_value;
				if (a.distance2 != b.distance2)
					return a.distance2 < b.distance2;
				return a.index < b.index;
			}
		};

		// E-M69 (CONFIRMED MEDIUM): per-candidate scan record for the bounded
		// pool. The pool is selected by COVARIANCE order (desc) — not by
		// isotropic distance, which is not monotone with covariance for
		// anisotropic models — with squared distance as the deterministic
		// tie-break (nearest first; also yields nearest-first admission for
		// the degenerate pure-nugget case where every covariance is 0).
		struct pool_entry_t
		{
			node_index_t index;
			double covar;
			double distance2;
		};

		// Named comparator for POOL selection (E-M69): descending
		// covariance, then squared distance ascending, then node index
		// ascending — the same strict total order as desc_cov_compare over
		// pool_entry_t. Used by the SCAN_LIMIT pool truncation AND the
		// pure-nugget fallback admission (R-12) so both select the
		// highest-covariance candidates deterministically.
		struct pool_order_compare
		{
			bool operator()(const pool_entry_t & a, const pool_entry_t & b)const
			{
				if (a.covar != b.covar)
					return a.covar > b.covar;
				if (a.distance2 != b.distance2)
					return a.distance2 < b.distance2;
				return a.index < b.index;
			}
		};

	}

	template <typename covariances_t>
	class indexed_neighbour_lookup_t<sugarbox_grid_t, covariances_t>
	{
		neighbour_lookup_t<sugarbox_grid_t, covariances_t> m_nlookup;
		std::shared_ptr<clusterizer_t> m_clusterizer;
		const covariances_t * m_cov;
		size_t m_max_neighbours;
		sugarbox_search_ellipsoid_t m_radiuses;
		const sugarbox_grid_t * m_grid;
	public:
		typedef sugarbox_grid_t grid_t;
		typedef sugarbox_location_t coord_t;

		// E-M68 / E-M70 / E2-129: shared octant-diversity admission pass
		// (GSLIB noct equivalent), used by BOTH the fast path (covariance
		// threshold-filtered pool) and the dense-region fallback (plain
		// lookup result) so the octant rule is consistent across regimes
		// (E-M68). Input must be sorted by descending covariance (total
		// order from desc_cov_compare). Pass 1 picks the best
		// (highest-covariance) candidate of each occupied octant; Pass 2
		// fills the remaining slots from the best-remaining by covariance.
		// E-M70: the ENTIRE list is sorted before this pass — the old code
		// partial_sorted only [0, result_size) and then scanned the full
		// vector, so weak-octant representatives came from the arbitrary
		// unsorted tail rather than the best member of their octant.
		// E2-129: `chosen` is a thread_local scratch buffer (matching the
		// file's ncandidates/temp_sort_vector/cand_ordered pattern) instead
		// of a per-node heap allocation inside the OpenMP kriging loops.
		static void apply_octant_diversity(
				const std::vector<detail::entry_t> & sorted_entries,
				const coord_t & center,
				size_t max_nb,
				std::vector<node_index_t> & indices,
				std::vector<coord_t> & coords)
		{
			indices.clear();
			coords.clear();
			const size_t n = sorted_entries.size();

			// Record the best (highest-covariance) candidate in each octant.
			std::array<int, 8> octant_best;
			octant_best.fill(-1);
			for (size_t i = 0; i < n; ++i)
			{
				const coord_t & c = sorted_entries[i].coord;
				const int oct = (c[0] >= center[0] ? 1 : 0)
				              | (c[1] >= center[1] ? 2 : 0)
				              | (c[2] >= center[2] ? 4 : 0);
				if (octant_best[oct] < 0)
					octant_best[oct] = static_cast<int>(i);
			}

			thread_local std::vector<bool> chosen;
			chosen.assign(n, false);

			// Pass 1: pick one best candidate from each occupied octant.
			for (size_t i = 0; i < n && indices.size() < max_nb; ++i)
			{
				const coord_t & c = sorted_entries[i].coord;
				const int oct = (c[0] >= center[0] ? 1 : 0)
				              | (c[1] >= center[1] ? 2 : 0)
				              | (c[2] >= center[2] ? 4 : 0);
				if (octant_best[oct] == static_cast<int>(i))
				{
					indices.push_back(sorted_entries[i].index);
					coords.push_back(c);
					chosen[i] = true;
				}
			}

			// Pass 2: fill remaining slots from best-remaining by covariance.
			for (size_t i = 0; i < n && indices.size() < max_nb; ++i)
			{
				if (!chosen[i])
				{
					indices.push_back(sorted_entries[i].index);
					coords.push_back(sorted_entries[i].coord);
				}
			}
		}
		indexed_neighbour_lookup_t(
				const sugarbox_grid_t * grid,
				const covariances_t * cov,
				const neighbourhood_param_t & nb_param
				)
			:	m_nlookup(grid, cov, nb_param), 
			m_clusterizer(new clusterizer_t(grid, nb_param.m_radiuses, std::max(m_nlookup.m_vectors->size() / detail::MAGIC_NUMBER_2, detail::MIN_CLUSTER_LIMIT))), 
					m_cov(cov), 
					m_max_neighbours(nb_param.m_max_neighbours),
					m_radiuses(nb_param.m_radiuses),
					m_grid(grid)
		{
		}

		template<typename defineds_t>
		void find(node_index_t node, const defineds_t & defineds, coord_t & node_coord, std::vector<node_index_t> & indices, std::vector<coord_t> & coords)const
		{
			if (m_clusterizer->limit_exceeded(node))
			{
				// F-M26 / E-M68 (CONFIRMED MEDIUM): dense-region fallback.
				// This branch delegates to the shared plain
				// neighbour_lookup_t::find — radius-bounded, covariance
				// threshold-filtered at construction (calc_cov_field keeps
				// only offsets with cov > C(0)/100), covariance-ranked up to
				// min(max_neighbours, NEIGHBOUR_WORK_CAP). E-M68: the octant
				// diversity rule is now applied here TOO (shared
				// apply_octant_diversity below), so both regimes enforce the
				// same GSLIB-noct-equivalent rule. The plain find already
				// caps the returned set at the count bound, so the octant
				// pass reorders (octant representatives first) without
				// changing membership; the fast path additionally selects
				// from its full 10000-candidate pool, so weak octants can
				// substitute a lower-covariance member there but not here —
				// the residual difference is the pool-vs-returned-set
				// design, not the admission rules (threshold, radius bound
				// and count cap are identical).
				m_nlookup.find(node, defineds, node_coord, indices, coords);
				// Re-apply the octant rule over the plain result (already in
				// covariance order). thread_local scratch — no per-node heap
				// allocation (E2-129 pattern).
				thread_local std::vector<detail::entry_t> fallback_entries;
				fallback_entries.clear();
				fallback_entries.reserve(indices.size());
				for (size_t i = 0, end_i = indices.size(); i != end_i; ++i)
				{
					fallback_entries.push_back(detail::entry_t{indices[i], 0.0, 0.0, coords[i]});
				}
				apply_octant_diversity(fallback_entries, node_coord, m_max_neighbours, indices, coords);
			}
			else
			{
				indices.clear();
				thread_local std::vector<node_index_t> ncandidates;
				ncandidates.clear();
				coord_t center = (*m_grid)[node];
				node_coord = coord_t(center[0], center[1], center[2]);

				m_clusterizer->get_nearby_harddata(node, ncandidates);

				thread_local std::vector<detail::entry_t> temp_sort_vector;
				temp_sort_vector.clear();
				//top_only_container_t<entry_t> temp_sort_vector(params.m_max_neighbours);

				double threshold = (*m_cov)(center, center) / 100;

				// F-21 / R-03 / E-M65 / E2-145: bound the per-node candidate
				// scan. The clusterizer returns up to 2 × NEIGHBOUR_WORK_CAP
				// (= 20000) candidates (copy capped at the source,
				// clusterizer.cpp E-M65 — the old path could reach ~925K
				// candidates/node for a legal radius-644 workflow). The
				// collection bound is deliberately 2× the RANKING bound:
				// SCAN_LIMIT below is NEIGHBOUR_WORK_CAP = 10000, the same
				// count cap the sibling plain lookup uses (E2-145 raised
				// the old 4913 cap so max_neighbours ∈ (4913, 100000] is
				// honored). The headroom makes the covariance-order pool
				// selection below reachable (R-12) — without it the copy
				// cap equaled SCAN_LIMIT, the nth_element was dead code,
				// and the pool was the first 10000 candidates in cluster
				// visit order. The covariance sort and octant
				// passes below then run on a pool of at most SCAN_LIMIT
				// entries — constant per-node work regardless of search
				// radius. (No max_neighbours early-exit on the collection:
				// the octant pass needs candidates from every octant.)
				// E-M69 (CONFIRMED MEDIUM): the pool is selected by
				// COVARIANCE order, not isotropic distance. For ANISOTROPIC
				// models the isotropic squared distance is NOT monotone with
				// covariance (probe, ranges 30,3,3: Euclidean 20 along the
				// major axis has HIGHER covariance 0.135 than Euclidean 3
				// along the minor axis, 0.050), so a distance-ordered pool
				// would drop high-covariance candidates exactly where the
				// plain sibling (which truncates in true covariance order
				// via calc_cov_field) keeps them. Ranking by the actual
				// covariance removes the anisotropic truncation tradeoff
				// (the old F-21/R-03 accepted-tradeoff). The radius admission
				// (F-09) is applied BEFORE the covariance evaluation so no
				// covariance work is spent on beyond-radius nodes — they can
				// never enter the neighbourhood (infinite-support models)
				// regardless of their covariance. Ties break by squared
				// distance then node index (E-M67) — a strict total order,
				// so the pool selection is permutation-independent.
				thread_local std::vector<detail::pool_entry_t> cand_ordered;
				cand_ordered.clear();
				cand_ordered.reserve(ncandidates.size());
				for (size_t idx = 0, end_idx = ncandidates.size(); idx != end_idx; ++idx)
				{
					const node_index_t candidate = ncandidates[idx];
					const coord_t c = (*m_grid)[candidate];
					if (std::abs(c[0] - center[0]) > m_radiuses[0]
						|| std::abs(c[1] - center[1]) > m_radiuses[1]
						|| std::abs(c[2] - center[2]) > m_radiuses[2])
					{
						continue;
					}
					const double dx = c[0] - center[0];
					const double dy = c[1] - center[1];
					const double dz = c[2] - center[2];
					cand_ordered.push_back(detail::pool_entry_t{candidate, (*m_cov)(center, c), dx * dx + dy * dy + dz * dz});
				}
				// R-12 (CONFIRMED MEDIUM): this branch is now REACHABLE.
				// The clusterizer's E-M65 copy cap was raised to
				// 2 × NEIGHBOUR_WORK_CAP (clusterizer.cpp), so on dense
				// configs cand_ordered can exceed SCAN_LIMIT and the pool
				// is truncated in true covariance order instead of cluster
				// visit order. (Pre-fix the copy cap equaled SCAN_LIMIT —
				// cand_ordered.size() ≤ 10000 made this branch dead code,
				// the E-M69 anisotropic truncation persisted, and the
				// pure-nugget fallback below admitted clusterizer
				// insertion order.)
				const size_t SCAN_LIMIT = static_cast<size_t>(NEIGHBOUR_WORK_CAP);
				if (cand_ordered.size() > SCAN_LIMIT)
				{
					// Pool selection in covariance order (E-M69): keep the
					// highest-covariance candidates; ties break by squared
					// distance then index (E-M67) — strict total order.
					std::nth_element(cand_ordered.begin(), cand_ordered.begin() + SCAN_LIMIT, cand_ordered.end(),
						detail::pool_order_compare());
					cand_ordered.resize(SCAN_LIMIT);
				}

				for (size_t idx = 0, end_idx = cand_ordered.size(); idx != end_idx; ++idx)
				{
					const node_index_t candidate = cand_ordered[idx].index;
					coord_t c = (*m_grid)[candidate];
					// F-09 radius admission and the covariance evaluation
					// were already applied at pool build (E-M69 restructure):
					// every pool member is inside the search radius, so the
					// infinite-support-model divergence from the
					// radius-bounded fallback cannot occur.
					if (cand_ordered[idx].covar > threshold)
					{
						detail::entry_t entry = { candidate, cand_ordered[idx].covar, cand_ordered[idx].distance2, c };
						temp_sort_vector.push_back(entry);
						//temp_sort_vector.add(entry);
					}
				}

				// Pure-nugget fallback: when nugget==sill (or a degenerate
				// covariance model), cov(h>0) = 0 for every candidate, so the
				// covar > sill/100 threshold filter above rejects ALL of them
				// and the neighbourhood would be empty. GSLIB's search is
				// radius-based: data inside the search ellipsoid are admitted
				// and the covariance is only used for ranking/weighting. The
				// fallback path neighbour_lookup_t::find (calc_cov_field) has
				// no covariance threshold and admits radius-bounded nodes, so
				// the fast path must mirror it in the degenerate case —
				// otherwise ordinary_kriging with nugget==sill produces an
				// empty neighbourhood → NaN output (F-46 regression tests:
				// test_ok_nugget_sweep[1.0], test_ok_with_nugget, test_minimal_radius).
				// R-03 / R-12 (CONFIRMED MEDIUM): the fallback admits the
				// best m_max_neighbours candidates from the pool in
				// COVARIANCE order (E-M69), selected DIRECTLY here — it
				// must not rely on the SCAN_LIMIT nth_element above having
				// ordered the pool (that branch only fires when the pool
				// exceeds SCAN_LIMIT; the ranking was dead code pre-R-12
				// and the admission loop regressed to clusterizer
				// insertion order). For a pure-nugget model every
				// covariance is equal (0), so the E-M67 tie-break (squared
				// distance asc, index asc) admits the NEAREST candidates
				// first — the indexed sibling's nearest-first semantics
				// (E2-141), restoring the pre-fix nearest-bounded
				// admission. The scan is still bounded by the
				// SCAN_LIMIT-truncated pool (plus its own m_max_neighbours
				// stop condition).
				if (temp_sort_vector.empty())
				{
					const size_t want = cand_ordered.size() < m_max_neighbours
						? cand_ordered.size() : m_max_neighbours;
					if (want > 0 && cand_ordered.size() > want)
					{
						std::nth_element(cand_ordered.begin(), cand_ordered.begin() + want, cand_ordered.end(),
							detail::pool_order_compare());
						cand_ordered.resize(want);
					}
					for (size_t idx = 0, end_idx = cand_ordered.size(); idx != end_idx; ++idx)
					{
						const node_index_t candidate = cand_ordered[idx].index;
						coord_t c = (*m_grid)[candidate];
						detail::entry_t entry = { candidate, cand_ordered[idx].covar, cand_ordered[idx].distance2, c };
						temp_sort_vector.push_back(entry);
					}
				}
				
				// E-M70 (CONFIRMED MEDIUM): FULL sort, not partial. The
				// octant pass below needs the best candidate per octant over
				// the WHOLE list; with the old partial_sort only
				// [0, result_size) was ordered and weak-octant
				// representatives came from the arbitrary unsorted tail
				// rather than the best member of their octant.
				// desc_cov_compare is a strict total order (covariance desc,
				// squared distance asc, index asc — E-M67), so the sort is
				// deterministic and permutation-independent.
				std::sort(temp_sort_vector.begin(), temp_sort_vector.end(), detail::desc_cov_compare());

				// Octant diversity enforcement (GSLIB noct equivalent),
				// shared with the dense-region fallback (E-M68): both
				// regimes apply the same rule. After sorting by covariance,
				// ensure at least 1 neighbour per spatial octant (8 octants
				// in 3D) before filling remaining slots. This prevents all
				// neighbours from concentrating in one direction when data
				// is clustered on one side of the estimation point.
				apply_octant_diversity(temp_sort_vector, center, m_max_neighbours, indices, coords);

				if (indices.size() > m_max_neighbours)
					HPGL_CHECK(false, "find_neighbours_with_clusterizer: indices.size() exceeds max_neighbours");
			}
		}

		void add_node(node_index_t node)
		{
			m_clusterizer->add_node(node);
		}

	};
	
}

#endif //__SUGARBOX_INDEXED_NEIGHBOUR_LOOKUP_H__9A00552C_FE2E_4B73_8F3D_18289B1492E6
