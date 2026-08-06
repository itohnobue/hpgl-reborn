#include "stdafx.h"
#include "calc_mean.h"
#include "property_array.h"
#include <cmath>


	double hpgl::calc_mean(const cont_property_array_t & property, bool * success)
	{	
		int count_points = 0;
		double sum = 0;

		// Mean computation procedure
		for(int idx = 0; idx < property.size(); ++idx)
		{
			if( property.is_informed(idx) )
			{
				double val = property.get_at(idx);
				if (!std::isfinite(val))
				{
					*success = false;
					return NAN;
				}
				sum += val;
				count_points += 1;
			}
		}		
		if (count_points == 0)
		{
			*success = false;
			return 0;
		}
		else
		{
			*success = true;
			return sum / count_points;
		}
	}

namespace hpgl
{

	bool calc_mean(const cont_property_array_t * property, double * mean)
	{		
		if (!property || ! mean)
			return false;

		int count_points = 0;
		double sum = 0;
		// Mean computation procedure
		for(int idx = 0; idx < property->size(); ++idx)
		{
			if( property->is_informed(idx) )
			{
				// E-M62: reject non-finite input like the reference
				// overload (:18-22) and the Python mirror (geo.py:1519-
				// 1521).  Without this gate a NaN sum produced a NaN
				// mean reported as success=true.
				double val = property->get_at(idx);
				if (!std::isfinite(val))
					return false;
				sum += val;
				count_points += 1;
			}
		}
		if (count_points == 0)
		{
			return false;
		}
		*mean = sum / count_points;		
		return true;
	}
}