#include "stdafx.h"

#include "typedefs.h"
#include "property_array.h"
#include "sugarbox_grid.h"
#include "ok_params.h"
#include "pretty_printer.h"
#include "cont_kriging.h"
#include "mean_provider.h"
#include "my_kriging_weights.h"
#include "sugarbox_indexed_neighbour_lookup.h"
#include "hpgl_exception.h"
#include "api.h"

namespace hpgl
{

void lvm_kriging(
		const cont_property_array_t & input,
		const mean_t * mean_data,
		const sugarbox_grid_t & grid,
		const ok_params_t & params,
		cont_property_array_t & output)
{
	print_algo_name("LVM Kriging");
	print_params(params);
	
	// F-194 gate 4 (F2): mirror the C-API kriging gate at this C++ entry — the
	// boundary applies validate_kriging_max_neighbours_or_throw (api.cpp), but
	// a direct-C++ consumer (hpgl_core.h is installed) reaches this function
	// with no predicate at all. m_max_neighbours == 0 makes every node's
	// neighbourhood empty (find() early-returns), so the kernel mean-fills
	// every cell (mean_on_failure below) with no error; >
	// HPGL_MAX_NEIGHBOURS_UPPER_BOUND asks for the unbounded system the cap
	// exists to stop. One copy of the number: api.h.
	if (params.m_max_neighbours < 1
		|| params.m_max_neighbours > static_cast<size_t>(HPGL_MAX_NEIGHBOURS_UPPER_BOUND))
	{
		std::ostringstream oss;
		oss << "max_neighbours " << params.m_max_neighbours
		    << " is out of range [1, " << HPGL_MAX_NEIGHBOURS_UPPER_BOUND
		    << "]; LVM kriging requires at least one conditioning neighbour";
		throw hpgl_exception("lvm_kriging", oss.str());
	}
	
	progress_reporter_t reporter(grid.size());
	kriging_stats_t stats;

	typedef precalculated_covariances_t covariances_t;
	covariances_t pcov(cov_model_t(params), params.m_radiuses);

	hpgl::cont_kriging(input, grid, params, mean_data, pcov,
		sk_weight_calculator_t(),
		output, reporter, stats, kriging_failure_handling::mean_on_failure);

	set_kriging_stats(stats);

	std::ostringstream oss;
	oss << stats;
	write(oss.str());
}

}
