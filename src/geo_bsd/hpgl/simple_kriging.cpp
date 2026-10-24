#include "stdafx.h"

#include "property_array.h"
#include "sugarbox_grid.h"
#include "sk_params.h"
#include "pretty_printer.h"
#include "cont_kriging.h"
#include "calc_mean.h"
#include "mean_provider.h"
#include "sugarbox_indexed_neighbour_lookup.h"
#include "my_kriging_weights.h"
#include "kriging_stats.h"
#include "api.h"
#include "output.h"
#include "hpgl_exception.h"

namespace hpgl
{

            void simple_kriging_weights(
                const covariance_param_t * cov_params,
                real_location_t center_point,
                std::vector<real_location_t> neighbourhoods_coords,
                std::vector<kriging_weight_t> & weights,
                double & variance)
            {
                cov_model_t cov(*cov_params);
               bool result = sk_kriging_weights_3<cov_model_t, true, real_location_t>(
					center_point,
					neighbourhoods_coords,
					cov,					
					weights,
					variance);

               if(result == false)
               {
                   // F-38: surface a meaningful kriging-failure error instead
                   // of silently emptying `weights` — the C API then
                   // dereferenced the empty vector and leaked the standard
                   // library's internal out-of-range text ("vector" on
                   // libc++, "vector::_M_range_check..." on libstdc++) to the
                   // caller (trigger: all-identical neighbour points → singular
                   // covariance matrix, F-206). Throwing lets the C API's
                   // exception handler produce a clean message.
                   throw hpgl_exception("simple_kriging_weights",
                       "kriging solve failed (check for identical or degenerate neighbour points)");
               }

            }

	void simple_kriging(
			const cont_property_array_t & input,
			const sugarbox_grid_t & grid,
			const sk_params_t & params,
			cont_property_array_t & output)
	{
		print_algo_name("Simple Kriging");
		print_params(params);

		// F-194 gate 4 (F2): mirror the C-API kriging gate at this C++ entry —
		// the boundary applies validate_kriging_max_neighbours_or_throw
		// (api.cpp), but a direct-C++ consumer (hpgl_core.h is installed)
		// reaches this function with no predicate at all. m_max_neighbours ==
		// 0 makes every node's neighbourhood empty (find() early-returns), so
		// the kernel mean-fills every cell (mean_on_failure below) with no
		// error; > HPGL_MAX_NEIGHBOURS_UPPER_BOUND asks for the unbounded
		// system the cap exists to stop. One copy of the number: api.h.
		if (params.m_max_neighbours < 1
			|| params.m_max_neighbours > static_cast<size_t>(HPGL_MAX_NEIGHBOURS_UPPER_BOUND))
		{
			std::ostringstream oss;
			oss << "max_neighbours " << params.m_max_neighbours
			    << " is out of range [1, " << HPGL_MAX_NEIGHBOURS_UPPER_BOUND
			    << "]; simple kriging requires at least one conditioning neighbour";
			throw hpgl_exception("simple_kriging", oss.str());
		}

		double mean;
		if (params.m_calculate_mean)
		{	
			bool valid_mean;
			mean = calc_mean(input, &valid_mean);
			if (!valid_mean)
			{
				LOGWARNING("No data to calculate mean. Defaulting to 0.\n");
				mean = 0.0;
			}
			print_param("Calculated mean: ", mean);		
		}
		else
			mean = params.mean();

		kriging_stats_t stats;			
		progress_reporter_t report(grid.size());

		cov_model_t covariance(params);		
		typedef precalculated_covariances_t covariances_t;
		covariances_t pcov(covariance, params.m_radiuses);

		hpgl::cont_kriging(input, grid, params, single_mean_t(mean), pcov,
			sk_weight_calculator_t(),
			output, report, stats, kriging_failure_handling::mean_on_failure);

		set_kriging_stats(stats);

		std::ostringstream oss;
		oss << stats;
		write(oss.str());
	}
}
