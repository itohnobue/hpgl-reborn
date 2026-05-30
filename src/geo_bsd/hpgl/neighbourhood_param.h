#ifndef __NEIGHBOURHOOD_PARAM_H__BDE0925A_D03A_4288_8306_3ADE14B0004E__
#define __NEIGHBOURHOOD_PARAM_H__BDE0925A_D03A_4288_8306_3ADE14B0004E__

#include "typedefs.h"

namespace hpgl
{
	class neighbourhood_param_t
	{
	public:
		neighbourhood_param_t();
		neighbourhood_param_t(const sugarbox_search_ellipsoid_t & radiuses, int max_neighbours);
		sugarbox_search_ellipsoid_t m_radiuses;
		size_t m_max_neighbours;
		// NOTE: radiuses are stored as doubles (search_ellipsoid_t) but the public
		// setter accepts size_t. Negative int values from C API params undergo
		// unsigned wrap to large positive values. Callers should ensure non-negative
		// inputs — this is validated at the Python wrapper layer.
		void set_radiuses(size_t radius1,
			size_t radius2, size_t radius3);
	};
}

#endif //__NEIGHBOURHOOD_PARAM_H__BDE0925A_D03A_4288_8306_3ADE14B0004E__
