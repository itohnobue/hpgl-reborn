#ifndef __SUGARBOX_INDEXED_NEIGHBOUR_LOOKUP_H__9A00552C_FE2E_4B73_8F3D_18289B1492E6
#define __SUGARBOX_INDEXED_NEIGHBOUR_LOOKUP_H__9A00552C_FE2E_4B73_8F3D_18289B1492E6

#include "clusterizer.h"
#include "sugarbox_neighbour_lookup.h"
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
				std::vector<node_index_t> ncandidates;
				coord_t center = (*m_grid)[node];
				node_coord = coord_t(center[0], center[1], center[2]);

				m_clusterizer->get_nearby_harddata(node, ncandidates);

				std::vector<detail::entry_t> temp_sort_vector;
				//top_only_container_t<entry_t> temp_sort_vector(params.m_max_neighbours);

				double threshold = (*m_cov)(center, center) / 100;
				
				for (size_t idx = 0, end_idx = ncandidates.size();	idx != end_idx; ++idx)
				{
					double covar = (*m_cov)(center, (*m_grid)[ncandidates[idx]]);
					if (covar > threshold)
					{
						detail::entry_t entry = { ncandidates[idx], covar };
						temp_sort_vector.push_back(entry);
						//temp_sort_vector.add(entry);
					}
				}
				
				size_t count = 0;	
				size_t result_size = //temp_sort_vector.get_them().size();
					m_max_neighbours > temp_sort_vector.size() ? temp_sort_vector.size() : m_max_neighbours;
				
			// Partial sort: only need top result_size entries by covariance
			// desc_cov_compare sorts descending (highest covariance first)
			if (result_size < temp_sort_vector.size())
				std::partial_sort(temp_sort_vector.begin(), temp_sort_vector.begin() + result_size, temp_sort_vector.end(), detail::desc_cov_compare());
			else
				std::sort(temp_sort_vector.begin(), temp_sort_vector.end(), detail::desc_cov_compare());

				
				
				indices.resize(result_size);
				coords.resize(result_size);

				const auto& entries = temp_sort_vector;//temp_sort_vector.get_them();

				for (size_t idx = 0, end_idx = entries.size();
						idx != end_idx; ++idx)
				{
					indices[idx] = entries[idx].index;	
					coords[idx] = (*m_grid)[entries[idx].index];
					count++;
					if (count >= m_max_neighbours)
						break;
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
