#include "stdafx.h"
#include "typedefs.h"
#include "property_array.h"
#include "sugarbox_grid.h"
#include "ik_params.h"
#include "indicator_kriging.h"
#include "pretty_printer.h"
#include "mean_provider.h"
#include "sugarbox_indexed_neighbour_lookup.h"
#include "api.h"

namespace hpgl
{
void indicator_kriging(
	const indicator_property_array_t & input,
	const sugarbox_grid_t & grid,  //TODO: Сделать тип grid_size_t задающий размеры грида, и все функции перевода координат сделать свободными
	const ik_params_t & params,
	indicator_property_array_t & output)
{
	// F-194: mirror the two C-API gates for this entry at the C++ entry, so a
	// direct-C++ consumer (the headers in hpgl_core.h are installed) gets the
	// same clean rejection the C boundary produces instead of a silently
	// biased result.
	//
	// (a) per-category marginal probability — the C gate is
	// api_helpers.cpp:163-166 (m_marginal_prob in [0, 1]); the members are
	// consumed verbatim by create_means below and by the failure fallback, so
	// an out-of-range or NaN value silently corrupts category selection
	// (NaN <= 0.5 is false -> the top category).
	for (size_t c = 0; c < params.m_marginal_probs.size(); ++c)
	{
		const double p = static_cast<double>(params.m_marginal_probs[c]);
		if (!(p >= 0.0 && p <= 1.0))
		{
			std::ostringstream oss;
			oss << "m_marginal_probs[" << c << "] = " << p
			    << " is outside [0, 1] — indicator kriging consumes it"
			       " directly as a probability";
			throw hpgl_exception("indicator_kriging", oss.str());
		}
	}
	// (b) per-category search size — the C gate is
	// validate_kriging_max_neighbours_or_throw (api.cpp), which rejects
	// < 1 and > HPGL_MAX_NEIGHBOURS_UPPER_BOUND. The default
	// neighbourhood_param_t carries m_max_neighbours == 0, and a zero limit
	// makes every node's neighbourhood empty: the kernel then silently
	// mean-fills / returns all-undefined output instead of kriging.
	for (size_t c = 0; c < params.m_neighbour_limits.size(); ++c)
	{
		const size_t limit = params.m_neighbour_limits[c];
		if (limit < 1 || limit > static_cast<size_t>(HPGL_MAX_NEIGHBOURS_UPPER_BOUND))
		{
			std::ostringstream oss;
			oss << "indicator " << c << ": max_neighbours " << limit
			    << " is out of range [1, " << HPGL_MAX_NEIGHBOURS_UPPER_BOUND << "]";
			throw hpgl_exception("indicator_kriging", oss.str());
		}
	}

	std::vector<single_mean_t> single_means;
	create_means(params.m_marginal_probs, single_means);
	do_indicator_kriging(input, grid, params, single_means, output);
}

}
