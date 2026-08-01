#include "stdafx.h"
#include "neighbourhood_param.h"
#include "hpgl_exception.h"
#include <limits>
#include <sstream>

namespace hpgl
{
	neighbourhood_param_t::neighbourhood_param_t()
	{
		m_max_neighbours = 0;
		set_radiuses(0, 0, 0);
	}

	void neighbourhood_param_t::set_radiuses(size_t radius1, size_t radius2, size_t radius3)
	{
		// M23: C API passes signed int[3]; negative values undergo unsigned
		// wrap to near-SIZE_MAX. Reject values exceeding int range.
		const size_t max_radius = static_cast<size_t>(std::numeric_limits<int>::max());
		if (radius1 > max_radius || radius2 > max_radius || radius3 > max_radius)
			throw hpgl_exception("neighbourhood_param_t::set_radiuses",
				"Radius exceeds maximum allowed (must be non-negative int)");
		m_radiuses[0] = radius1;
		m_radiuses[1] = radius2;
		m_radiuses[2] = radius3;
	}

	void validate_kriging_radiuses_or_throw(
			const sugarbox_search_ellipsoid_t & radiuses,
			const char * context)
	{
		// A zero search radius yields an empty neighbourhood on kriging paths:
		// every node reports KI_NO_NEIGHBOURS and kriging silently degrades to
		// mean/noise fill (F-34). Simulation paths (SGS zero-radius CDF draw)
		// are intentionally exempt — callers on those paths must not call this.
		if (radiuses[0] <= 0 && radiuses[1] <= 0 && radiuses[2] <= 0)
		{
			std::ostringstream oss;
			oss << "search radius must be positive (got "
			    << radiuses[0] << ", " << radiuses[1] << ", " << radiuses[2] << ")";
			throw hpgl_exception(context, oss.str());
		}
	}
}
