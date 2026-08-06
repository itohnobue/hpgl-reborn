#ifndef INCLUDED_MEDIAN_IK_H_IN_SOME_BLUE_SKY_PROJECT_SDFLSDKJFLSDFJLSDKFJLSDF
#define INCLUDED_MEDIAN_IK_H_IN_SOME_BLUE_SKY_PROJECT_SDFLSDKJFLSDFJLSDKFJLSDF

#include "typedefs.h"
#include "sugarbox_grid.h"
#include "property_array.h"
#include "ok_params.h"

namespace hpgl
{

	struct median_ik_params : public ok_params_t
	{			
		// E2-126: default member initializers — previously m_marginal_probs
		// was indeterminate (no ctor/member init), so a direct-C++ caller
		// that forgot to set it got NaN marginals → NaN <= 0.5 is false →
		// choose_indicator silently returned category 1 for EVERY node.
		// Equal-probability prior (0.5/0.5); the C API (api.cpp) and the
		// Python wrapper always overwrite with validated values, and the
		// C++ entry (median_ik.cpp) now validates at the chokepoint.
		double m_marginal_probs[2] = {0.5, 0.5};
	private:
		//indicator_value_t m_values[2];	
	};

void median_ik_for_two_indicators(
		const median_ik_params &, 
		const sugarbox_grid_t & grid,
		const indicator_property_array_t & input_property,
		indicator_property_array_t & output_property
);

}

#endif // INCLUDED_MEDIAN_IK_H_IN_SOME_BLUE_SKY_PROJECT_SDFLSDKJFLSDFJLSDKFJLSDF

