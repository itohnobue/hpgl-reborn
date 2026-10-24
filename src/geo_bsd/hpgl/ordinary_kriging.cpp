#include "stdafx.h"
#include "property_array.h"
#include "sugarbox_grid.h"
#include "ok_params.h"
#include "pretty_printer.h"
#include "cont_kriging.h"
#include "mean_provider.h"
#include "my_kriging_weights.h"
#include "sugarbox_indexed_neighbour_lookup.h"
#include "kriging_stats.h"
#include "hpgl_exception.h"
#include "api.h"

namespace hpgl
{
	void ordinary_kriging(
		const cont_property_array_t & input,
		const sugarbox_grid_t & grid,
		const ok_params_t & params,		
		cont_property_array_t & output,
		bool use_new_cov)
	{			
		print_algo_name("Ordinary Kriging");
		print_params(params);
		
		// F-194 gate 4 (F2): mirror the C-API kriging gate at this C++ entry,
		// so a direct-C++ consumer (hpgl_core.h is installed) gets the same
		// clean rejection api.cpp applies instead of a silently degraded run.
		// The C gate is validate_kriging_max_neighbours_or_throw (api.cpp),
		// which rejects < 1 and > HPGL_MAX_NEIGHBOURS_UPPER_BOUND; the bound
		// comes from api.h precisely so this mirror keeps one copy of the
		// number. The default neighbourhood_param_t carries
		// m_max_neighbours == 0, and a zero limit makes every node's
		// neighbourhood empty (find() early-returns): with
		// undefined_on_failure below that leaves EVERY cell undefined, with no
		// error reaching the caller.
		if (params.m_max_neighbours < 1
			|| params.m_max_neighbours > static_cast<size_t>(HPGL_MAX_NEIGHBOURS_UPPER_BOUND))
		{
			std::ostringstream oss;
			oss << "max_neighbours " << params.m_max_neighbours
			    << " is out of range [1, " << HPGL_MAX_NEIGHBOURS_UPPER_BOUND
			    << "]; ordinary kriging requires at least one conditioning neighbour";
			throw hpgl_exception("ordinary_kriging", oss.str());
		}
		
		kriging_stats_t stats;
		progress_reporter_t report(grid.size());

		if (use_new_cov)
		{
			cov_model_t covariance(params);

			hpgl::cont_kriging(input, grid, params, no_mean_t(), covariance,
				ok_weight_calculator_t(), 
				output, report, stats, kriging_failure_handling::undefined_on_failure);
		}
		else
		{	
			typedef precalculated_covariances_t covariances_t;
			covariances_t pcov(cov_model_t(params), params.m_radiuses);

			hpgl::cont_kriging(input, grid, params, no_mean_t(), pcov,
				ok_weight_calculator_t(),
				output, report, stats, kriging_failure_handling::undefined_on_failure);
		}

		set_kriging_stats(stats);

		std::ostringstream oss;
		oss << stats;
		write(oss.str());
	}
}
