#ifndef __SUGARBOX_INDEXED_NEIGHBOUR_LOOKUP_H__9A00552C_FE2E_4B73_8F3D_18289B1492E6
#define __SUGARBOX_INDEXED_NEIGHBOUR_LOOKUP_H__9A00552C_FE2E_4B73_8F3D_18289B1492E6

#include "clusterizer.h"
#include "sugarbox_neighbour_lookup.h"
#include <array>
#include <cstdlib>
#include <memory>

namespace hpgl
{
	namespace detail
	{
		const int MAGIC_NUMBER_2 = 250;

		struct entry_t
		{
			node_index_t index;
			double cov_value;
			sugarbox_location_t coord;  // cached grid coordinate (avoids redundant operator[] calls)
			bool operator<(const entry_t & e)const
			{
					return cov_value < e.cov_value;
			}
			bool operator>(const entry_t & e)const
			{
					return cov_value > e.cov_value;
			}
			bool operator<=(const entry_t & e)const
			{
					return cov_value <= e.cov_value;
			}
			bool operator>=(const entry_t & e)const
			{
					return cov_value >= e.cov_value;
			}
			bool operator==(const entry_t & e)const
			{
					return cov_value == e.cov_value;
			}
		};

		// Named comparator for descending covariance sort (highest first)
		struct desc_cov_compare
		{
			bool operator()(const entry_t & a, const entry_t & b)const
			{
				return a.cov_value > b.cov_value;
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
		indexed_neighbour_lookup_t(
				const sugarbox_grid_t * grid,
				const covariances_t * cov,
				const neighbourhood_param_t & nb_param
				)
			:	m_nlookup(grid, cov, nb_param), 
			m_clusterizer(new clusterizer_t(grid, nb_param.m_radiuses, std::max(m_nlookup.m_vectors->size() / detail::MAGIC_NUMBER_2, static_cast<size_t>(1)))), 
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
				m_nlookup.find(node, defineds, node_coord, indices, coords);				
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
				
				for (size_t idx = 0, end_idx = ncandidates.size();	idx != end_idx; ++idx)
				{
					coord_t c = (*m_grid)[ncandidates[idx]];
					// F-09: the clusterizer fast path collects the whole
					// 3×3×3 cluster box (~2×radius); only candidates within
					// the configured search radius may enter the
					// neighbourhood. Without this check, infinite-support
					// covariance models (exponential/gaussian) admit
					// beyond-radius nodes on the ordinary-kriging path
					// (use_new_cov=true), diverging from the strictly
					// radius-bounded fallback neighbour_lookup_t::find.
					if (std::abs(c[0] - center[0]) > m_radiuses[0]
						|| std::abs(c[1] - center[1]) > m_radiuses[1]
						|| std::abs(c[2] - center[2]) > m_radiuses[2])
					{
						continue;
					}
					double covar = (*m_cov)(center, c);
					if (covar > threshold)
					{
						detail::entry_t entry = { ncandidates[idx], covar, c };
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
				if (temp_sort_vector.empty())
				{
					for (size_t idx = 0, end_idx = ncandidates.size();
						idx != end_idx && temp_sort_vector.size() < m_max_neighbours; ++idx)
					{
						coord_t c = (*m_grid)[ncandidates[idx]];
						if (std::abs(c[0] - center[0]) > m_radiuses[0]
							|| std::abs(c[1] - center[1]) > m_radiuses[1]
							|| std::abs(c[2] - center[2]) > m_radiuses[2])
						{
							continue;
						}
						double covar = (*m_cov)(center, c);
						detail::entry_t entry = { ncandidates[idx], covar, c };
						temp_sort_vector.push_back(entry);
					}
				}
				
				size_t result_size = //temp_sort_vector.get_them().size();
					m_max_neighbours > temp_sort_vector.size() ? temp_sort_vector.size() : m_max_neighbours;
				
				// Partial sort: only need top result_size entries by covariance
				// desc_cov_compare sorts descending (highest covariance first)
				if (result_size < temp_sort_vector.size())
					std::partial_sort(temp_sort_vector.begin(), temp_sort_vector.begin() + result_size, temp_sort_vector.end(), detail::desc_cov_compare());
				else
					std::sort(temp_sort_vector.begin(), temp_sort_vector.end(), detail::desc_cov_compare());

				// Octant diversity enforcement (GSLIB noct equivalent):
				// After sorting by covariance, ensure at least 1 neighbour per
				// spatial octant (8 octants in 3D) before filling remaining slots.
				// This prevents all neighbours from concentrating in one direction
				// when data is clustered on one side of the estimation point.
				indices.clear();
				coords.clear();
				const size_t max_nb = m_max_neighbours;

				// Record the best (highest-covariance) candidate in each octant
				std::array<int, 8> octant_best;
				octant_best.fill(-1);
				for (size_t i = 0; i < temp_sort_vector.size(); ++i) {
					const coord_t & c = temp_sort_vector[i].coord;
					int oct = (c[0] >= center[0] ? 1 : 0)
					        | (c[1] >= center[1] ? 2 : 0)
					        | (c[2] >= center[2] ? 4 : 0);
					if (octant_best[oct] < 0)
						octant_best[oct] = static_cast<int>(i);
				}

				// Pass 1: pick one best candidate from each occupied octant
				std::vector<bool> chosen(temp_sort_vector.size(), false);
				for (size_t i = 0; i < temp_sort_vector.size() && indices.size() < max_nb; ++i) {
					const coord_t & c = temp_sort_vector[i].coord;
					int oct = (c[0] >= center[0] ? 1 : 0)
					        | (c[1] >= center[1] ? 2 : 0)
					        | (c[2] >= center[2] ? 4 : 0);
					if (octant_best[oct] == static_cast<int>(i)) {
						indices.push_back(temp_sort_vector[i].index);
						coords.push_back(c);
						chosen[i] = true;
					}
				}

				// Pass 2: fill remaining slots from best-remaining by covariance
				for (size_t i = 0; i < temp_sort_vector.size() && indices.size() < max_nb; ++i) {
					if (!chosen[i]) {
						indices.push_back(temp_sort_vector[i].index);
						coords.push_back(temp_sort_vector[i].coord);
					}
				}

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
