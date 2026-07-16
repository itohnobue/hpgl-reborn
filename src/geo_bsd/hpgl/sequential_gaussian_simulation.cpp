#include "stdafx.h"
#include "api.h"
#include "property_array.h"
#include "sugarbox_grid.h"
#include "sgs_params.h"
#include "pretty_printer.h"
#include "sequential_simulation.h"
#include "calc_mean.h"
#include "mean_provider.h"
#include "my_kriging_weights.h"
#include "sugarbox_indexed_neighbour_lookup.h"
#include "lvm_utils.h"
#include "gaussian_distribution.h"
#include "non_parametric_cdf.h"

namespace hpgl
{
	void sequential_gaussian_simulation(
		const sugarbox_grid_t& grid,
		const sgs_params_t& params,
		cont_property_array_t& output,
		const hpgl_non_parametric_cdf_t* cdf,
		const unsigned char* mask)
	{
		print_algo_name("Sequential Gaussian Simulation");
		print_params(params);

		if (output.size() != grid.size())
		{
			std::ostringstream oss;
			oss << "Input property size: " << output.size() << ". Grid size: " << grid.size() << ". Must be equal.";
			throw hpgl_exception("sequential_gaussian_simulation", oss.str());
		}

		if (cdf != nullptr)
		{
			non_parametric_cdf_2_t ncdf(cdf);
			if (ncdf.is_empty())
			{
				LOGWARNING("Non-parametric CDF is empty — skipping forward transformation.\n");
			}
			else
			{
				transform_cdf_p(output, ncdf, gaussian_cdf_t());
			}
		}

		if (params.m_kriging_kind == KRIG_SIMPLE)
		{
			double mean;
			if (params.m_calculate_mean)
			{
			bool valid_mean;
			mean = calc_mean(output, &valid_mean);
			if (!valid_mean)
			{
				LOGWARNING("No data to calculate mean. Defaulting to 0.\n");
				mean = 0.0;
			}
			}
			else
				mean = params.mean();

			if (mask != nullptr)
			{
				do_sequential_gausian_simulation(output, grid, params,
					single_mean_t(mean),
					sk_weight_calculator_t(),
					mask);
			}
			else
			{
				do_sequential_gausian_simulation(output, grid, params,
					single_mean_t(mean),
					sk_weight_calculator_t(),
					no_mask_t());

			}
		}
		else {
			if (mask != nullptr)
			{
				do_sequential_gausian_simulation(output, grid, params,
					no_mean_t(),
					ok_weight_calculator_t(),
					mask);
			}
			else {
				do_sequential_gausian_simulation(output, grid, params,
					no_mean_t(),
					ok_weight_calculator_t(),
					no_mask_t());
			}
		}

		if (cdf != nullptr)
		{
			non_parametric_cdf_2_t ncdf(cdf);
			if (ncdf.is_empty())
			{
				LOGWARNING("Non-parametric CDF is empty — skipping back-transformation.\n");
			}
			else
			{
				transform_cdf_p(output, gaussian_cdf_t(), ncdf);
			}
		}
	}

	void sequential_gaussian_simulation_lvm(
		const sugarbox_grid_t& grid,
		const sgs_params_t& params,
		const mean_t* mean_data,
		cont_property_array_t& output,
		const hpgl_non_parametric_cdf_t* cdf,
		const unsigned char* mask
	)
	{
		print_algo_name("Sequential Gaussian Simulation with Local Varying Mean");
		print_params(params);

		if (output.size() != grid.size())
		{
			std::ostringstream oss;
			oss << "Input property size: " << output.size() << ". Grid size: " << grid.size() << ". Must be equal.";
			throw hpgl_exception("sequential_gaussian_simulation_lvm", oss.str());
		}

		if (mean_data == nullptr)
		{
			throw hpgl_exception("sequential_gaussian_simulation_lvm",
				"Null mean_data pointer");
		}

		std::vector<mean_t> mean_data_vec;
		mean_data_vec.assign(mean_data, mean_data + output.size());

		if (cdf != nullptr)
		{
			non_parametric_cdf_2_t new_cdf(cdf);
			if (new_cdf.is_empty())
			{
				LOGWARNING("Non-parametric CDF is empty — skipping forward transformation.\n");
			}
			else
			{
				transform_cdf_p(output, new_cdf, gaussian_cdf_t());
				transform_cdf_ptr(mean_data, mean_data_vec, new_cdf, gaussian_cdf_t());
			}
		}

		if (mask != nullptr)
		{
			do_sequential_gausian_simulation( output, grid, params,
						mean_data_vec,
						sk_weight_calculator_t(),
						mask);
		}
		else
		{
			do_sequential_gausian_simulation( output, grid, params,
						mean_data_vec,
						sk_weight_calculator_t(),
						no_mask_t());

		}

		if (cdf != nullptr)
		{
			non_parametric_cdf_2_t new_cdf(cdf);
			if (new_cdf.is_empty())
			{
				LOGWARNING("Non-parametric CDF is empty — skipping back-transformation.\n");
			}
			else
			{
				transform_cdf_p(output, gaussian_cdf_t(), new_cdf);
			}
		}
	}
}
