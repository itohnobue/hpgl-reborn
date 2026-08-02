#ifndef __SUGARBOX_NEIGHBOUR_LOOKUP_H__B31AEF89_3C7A_48C8_97DD_F2983ED6A237
#define __SUGARBOX_NEIGHBOUR_LOOKUP_H__B31AEF89_3C7A_48C8_97DD_F2983ED6A237

#include "typedefs.h"
#include "covariance_field.h"
#include "neighbourhood_param.h"
#include "sugarbox_grid.h"
#include "neighbourhood_lookup.h"
#include <memory>

namespace hpgl
{
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
			  m_radiuses(nb_param.m_radiuses)
		{
			calc_cov_field<covariances_t, sugarbox_location_t>(nb_param.m_radiuses, *cov, *m_vectors);
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
			int count = 0;
			if (!vectors.empty())
			{
				const sugarbox_vector_t * vec = &vectors[0];
				for (int idx = 0, end_idx = (int) vectors.size(); idx < end_idx && count < m_max_neighbours; ++idx, ++vec)
				{
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
				// 2-M-32 (pure-nugget fallback): calc_cov_field builds the
				// offset list by covariance threshold (cov > C(0)/100). For a
				// pure-nugget model (nugget==sill) every h>0 covariance is 0,
				// so only the self-offset (0,0,0) survives — and the center is
				// never a valid conditioning datum (it is the node being
				// estimated). The neighbourhood would then be EMPTY → every
				// consumer of this plain lookup (cokriging, SGS, SIS) silently
				// mean-fills / falls back, while the indexed lookup
				// (sugarbox_indexed_neighbour_lookup.h) has a pure-nugget
				// fallback that admits radius-bounded nodes. GSLIB's search is
				// radius-based — data inside the search ellipsoid are admitted
				// and covariance only ranks. Mirror the indexed fast path's
				// fallback here so the same neighbourhood_param_t produces
				// consistent neighbour selection across consumers. This also
				// covers nodes whose neighbours all fall below the covariance
				// threshold (sparse / infinite-support models), matching the
				// indexed path's behavior.
				// R-6 (performance bound): this fallback runs PER find() call,
				// not once at construction. Iterating the FULL radius box per
				// node is O((2r+1)³) — for a pure-nugget model with a large
				// radius (r≈50 → 1e6 cells) on a sparse grid this is an
				// effectively-infinite loop (the round-1 fix comment conflated
				// the construction-time calc_cov_field scan with this find-time
				// scan). The fallback only fires when EVERY offset in the box
				// has covariance ≤ C(0)/100 (pure-nugget / extreme short-range),
				// so any admitted neighbour has negligible weight regardless of
				// distance — bounding the window to a modest fixed radius is
				// numerically sound. Cap the per-direction window so the
				// per-node scan is ≤ (2·8+1)³ = 4913 cells.
				const int FALLBACK_WINDOW = 8;  // cells per direction
				const int rx = static_cast<int>(m_radiuses[0]);
				const int ry = static_cast<int>(m_radiuses[1]);
				const int rz = static_cast<int>(m_radiuses[2]);
				const int wrx = rx < FALLBACK_WINDOW ? rx : FALLBACK_WINDOW;
				const int wry = ry < FALLBACK_WINDOW ? ry : FALLBACK_WINDOW;
				const int wrz = rz < FALLBACK_WINDOW ? rz : FALLBACK_WINDOW;
				bool done = false;
				for (int dz = -wrz; dz <= wrz && !done; ++dz)
				{
					for (int dy = -wry; dy <= wry && !done; ++dy)
					{
						for (int dx = -wrx; dx <= wrx && !done; ++dx)
						{
							sugarbox_location_t point = center + sugarbox_vector_t(dx, dy, dz);
							int index = m_grid->get_index(point);
							if (index >= 0)
							{
								if (pred(index))
								{
									indices.push_back(index);
									coords.push_back(coord_t(point[0], point[1], point[2]));
									++count;
									if (count >= m_max_neighbours)
										done = true;
								}
							}
						}
					}
				}
			}
		}		
	};

}		

#endif //__SUGARBOX_NEIGHBOUR_LOOKUP_H__B31AEF89_3C7A_48C8_97DD_F2983ED6A237
