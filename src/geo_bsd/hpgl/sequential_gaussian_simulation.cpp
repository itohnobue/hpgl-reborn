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

		// III-37: the data were forward-transformed to normal-score space
		// above, but the user-supplied scalar stationary mean is in DATA space.
		// Using it raw pins simulated cells to the CDF's max datum (a
		// data-space mean of 50 maps to a huge normal score). Transform it
		// through the CDF the same way the LVM path transforms mean_data at
		// the end of its forward block (:189,
		// transform_cdf_ptr(mean_data, ..., new_cdf, gaussian_cdf_t())).
		// The auto-computed mean (calc_mean on the already-transformed data)
		// is already in normal-score space and must NOT be transformed.
		auto transform_stationary_mean = [&](double m) -> double
		{
			if (cdf == nullptr)
				return m;
			non_parametric_cdf_2_t mean_cdf(cdf);
			if (mean_cdf.is_empty())
				return m;
			return transform_cdf_s(m, mean_cdf, gaussian_cdf_t());
		};

		if (params.m_kriging_kind == KRIG_SIMPLE)
		{
			double mean;
			// 2-M-1(c): consult m_mean_kind (the descriptive mean-mode field
			// set by the C API) rather than the redundant m_calculate_mean
			// flag — previously m_mean_kind was set, printed, but never read
			// by any algorithm. e_mean_stationary_auto → calculate from the
			// transformed data; e_mean_stationary → use the user-supplied
			// stationary mean (params.set_mean). The two fields were always
			// set consistently by api.cpp; this wiring makes the documented
			// contract (sgs.py docstring) match the actual behavior.
			if (params.m_mean_kind == mean_kind_t::e_mean_stationary_auto)
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
				mean = transform_stationary_mean(params.mean());

			if (mask != nullptr)
			{
				do_sequential_gausian_simulation(output, grid, params,
					single_mean_t(mean),
					single_mean_t(mean),
					sk_weight_calculator_t(),
					mask);
			}
			else
			{
				do_sequential_gausian_simulation(output, grid, params,
					single_mean_t(mean),
					single_mean_t(mean),
					sk_weight_calculator_t(),
					no_mask_t());

			}
		}
		else {
			// M-29 + R-5: GSLIB's OK-mode failure fallback draws N(gmean, 1.0)
			// (sgsim.for: `cmean = gmean; cstdev = 1.0`), NOT N(0,1). The
			// mean is computed the same way as the KRIG_SIMPLE path above
			// (user-supplied mean, or calculated from data). R-5: the OK
			// kriging path uses no_mean_t() (zero means) so the kriged
			// estimate is Σλᵢzᵢ with NO mean term on either branch — the
			// n≥4 OK estimate (Σλ=1 ⇒ Σλᵢzᵢ + (1−Σλᵢ)·gmean = Σλᵢzᵢ anyway)
			// and the n<4 SK-downgraded estimate (M-3, ok_sgs_weight_
			// calculator_t) — matching GSLIB's zero-mean normal-score
			// semantics and removing the n=4 mean-pull discontinuity. The
			// user/computed mean is passed SEPARATELY as the fallback mean
			// provider so the failure fallback still draws N(gmean, 1.0).
			double mean;
			// 2-M-1(c): same m_mean_kind wiring as the KRIG_SIMPLE branch —
			// the descriptive field selects auto-calculated vs user-supplied
			// stationary mean (2-M-1(a): the OK branch honors the user mean).
			if (params.m_mean_kind == mean_kind_t::e_mean_stationary_auto)
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
				mean = transform_stationary_mean(params.mean());

			if (mask != nullptr)
			{
				do_sequential_gausian_simulation(output, grid, params,
					no_mean_t(),
					single_mean_t(mean),
					ok_sgs_weight_calculator_t(),
					mask);
			}
			else {
				do_sequential_gausian_simulation(output, grid, params,
					no_mean_t(),
					single_mean_t(mean),
					ok_sgs_weight_calculator_t(),
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

		// 2-M-1(b): the LVM path intentionally uses the simple-kriging weight
		// calculator against the local varying mean (GSLIB sgsim ktype=3 LVM
		// semantics — "LVM kernel performs simple kriging against it", sgs.py
		// docstring). m_kriging_kind is deliberately NOT consulted here: LVM
		// is a separate C API entry point (hpgl_sgs_lvm_simulation), not a
		// kriging_kind value — the enum admits only KRIG_ORDINARY(0) and
		// KRIG_SIMPLE(1), both of which use the varying mean on this path.
		if (mask != nullptr)
		{
			do_sequential_gausian_simulation( output, grid, params,
						mean_data_vec,
						mean_data_vec,
						sk_weight_calculator_t(),
						mask);
		}
		else
		{
			do_sequential_gausian_simulation( output, grid, params,
						mean_data_vec,
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
