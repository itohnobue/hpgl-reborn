#include "stdafx.h"

#include "mean_provider.h"
#include "ik_params.h"
#include "random_node_enumerator.h"
#include "bs_random_generator.h"
#include "sample.h"
#include "my_kriging_weights.h"
#include "pretty_printer.h"
#include "progress_reporter.h"
#include "precalculated_covariance.h"
#include "kriging_interpolation.h"
#include "sugarbox_indexed_neighbour_lookup.h"
#include "is_informed_predicate.h"
#include "cov_model.h"
#include "hpgl_exception.h"
#include "indicator_kriging.h"

namespace hpgl
{

namespace
{	

template<typename grid_t, typename marginal_probs_t, typename weight_calculator_t, typename mask_t>
void do_sis(
		indicator_property_array_t & property,
		const grid_t & grid,
		const ik_params_t & params,
		int64_t seed,
		const marginal_probs_t & marginal_probs,		
		progress_reporter_t & reporter,
		const weight_calculator_t & weight_calculator_sis,
		const mask_t & mask)
{
	if (params.m_category_count == 0)
		return;

	// Defense-in-depth: indicator_index_t is unsigned char (max 255).
	// category_count > 255 causes infinite wrap-around at line ~102.
	if (params.m_category_count > 255)
		throw hpgl_exception("do_sis",
			"indicator count exceeds indicator_index_t max (255)");

	typedef precalculated_covariances_t cov_t;
	typedef neighbour_lookup_t<grid_t, cov_t> nl_t;
	std::vector<cov_t> 							covariances(params.m_category_count);	
	std::vector<nl_t> 							nblookups;
	std::vector<indicator_array_adapter_t> 		ind_props;

	for (size_t i = 0; i < params.m_category_count; ++i)
	{		
		covariances[i].init(cov_model_t(params.m_cov_params[i]), params.m_radiuses[i]);
		nblookups.push_back(nl_t(&grid, &covariances[i], params.m_nb_params[i]));
		ind_props.push_back(indicator_array_adapter_t(&property, i));
	}

	random_path_generator_t path(property.size(), seed);
	
	mt_random_generator_t gen(seed);
	reporter.start();

	kriging_ws_t<indicator_value_t, sugarbox_location_t> ws;

	// Pre-allocated probs vector: reused across all node iterations
	// via clear() to avoid 1M+ heap allocations in the hot loop.
	std::vector<indicator_probability_t> probs;

	if(params.m_category_count == 2)
	{
		write("Only 2 indicators found, Median SIS will be performed.");
	}

	while (!path.end_of_path())	
	{
		node_index_t node = path.get_next();
		reporter.next_lap();
		if (property.is_informed(node))
			continue;

		// Bounds-guard: validate node index before mask access.
		// node_index_t is signed int; path generator should produce valid
		// indices but double-check in case mask is undersized.
		if (node < 0 || node >= property.size())
			continue;

		if (mask[node] != 1)
			continue;

		probs.clear();

		// median SIS
			if(params.m_category_count == 2)
		{
				int idx = 0;

				double prob;
				
				ki_result_t ki_result = kriging_interpolation_ws(
					ind_props[idx], is_informed_predicate_t<indicator_property_array_t>(property), 
					node, covariances[idx], marginal_probs[idx], nblookups[idx], weight_calculator_sis, prob, ws);

				if (ki_result != ki_result_t::KI_SUCCESS)
				{
					prob = marginal_probs[idx][node];
				}

				// Clamp kriged probability to [0,1] before computing the
				// complement so both probabilities are well-formed.
				// SK weights are unconstrained — poorly conditioned matrices,
				// sparse neighbours, or extreme anisotropy can push the
				// combine() result outside [0,1].
				if (prob < 0.0) prob = 0.0;
				else if (prob > 1.0) prob = 1.0;
				probs.push_back(prob);
				probs.push_back(1.0 - prob);

		}
		else
		{
			for (indicator_index_t idx = 0; idx < params.m_category_count; ++idx)
			{
				double prob;
				
				ki_result_t ki_result = kriging_interpolation_ws(
					ind_props[idx], is_informed_predicate_t<indicator_property_array_t>(property), 
					node, covariances[idx], marginal_probs[idx], nblookups[idx], weight_calculator_sis, prob, ws);

				if (ki_result != ki_result_t::KI_SUCCESS)
				{
					prob = marginal_probs[idx][node];
				}
				probs.push_back(prob);
			}
		}
		// Enforce monotonicity and [0,1] bounds on indicator probabilities
		// before sampling. Only applicable for multi-category (3+) SIS since
		// 2-category SIS uses complementary [prob, 1-prob] — monotonicity
		// correction would destroy the distribution.
		if (params.m_category_count > 2)
		{
			// indicator_array_adapter_t uses exclusive encoding
			// (== m_value ? 1 : 0); kriged probabilities are per-category
			// PMF values. Convert to cumulative CDF before order relations
			// correction, which assumes monotonic [0,1]-bounded values.
			for (size_t i = 1; i < probs.size(); ++i)
				probs[i] += probs[i - 1];

			detail::correct_order_relations(probs);
			// Convert cumulative CDF values to class-level probabilities for
			// the PMF sampler. probs currently holds [P(Z≤0), P(Z≤1), ..., 1.0];
			// backward differencing produces [p_0, p_1, ..., p_{K-1}] where
			// p_0 = P(Z≤0) and p_i = P(Z≤z_i) - P(Z≤z_{i-1}) for i > 0.
			for (size_t i = probs.size() - 1; i > 0; --i)
				probs[i] -= probs[i - 1];
		}
		property.set_at(node, sample(gen, probs));
	}

	reporter.stop();	
}

struct no_mask_t
{
	int operator[](node_index_t /*index*/)const
	{
		return 1;
	}
};

} //namespace


void sequential_indicator_simulation(
			indicator_property_array_t & property,
			const sugarbox_grid_t & grid,
			const ik_params_t & params,
			int64_t seed,
			progress_reporter_t & report,
			const unsigned char * mask)
{
	print_algo_name("Sequential Indicator Simulation");
	print_params(params);
	if (property.size() != grid.size())
	{
		std::ostringstream oss;
		oss << "Property size '" << property.size() << "' is not equal to grid size '" << grid.size() << "'";
		throw hpgl_exception("sequential_indicator_simulation", oss.str());
	}
	
	std::vector<single_mean_t> single_means;
	create_means(params.m_marginal_probs, single_means);	
	if (mask == NULL)	
		do_sis(property, grid, params, seed, single_means, report, sk_weight_calculator_t(), no_mask_t());
	else
		do_sis(property, grid, params, seed, single_means, report, sk_weight_calculator_t(), mask);

}
	
void sequential_indicator_simulation_lvm(
		indicator_property_array_t & property,
		const sugarbox_grid_t & grid,
		const ik_params_t & params,
		int64_t seed,
		const mean_t ** mean_data,		
		progress_reporter_t & report,
		bool use_corellogram,
		const unsigned char * mask)
{
	print_algo_name("Sequential Indicator Simulation");
	print_params(params);
	print_param("LVM", "on");

	if (property.size() != grid.size())
	{
		std::ostringstream oss;
		oss << "Property size '" << property.size() << "' is not equal to grid size '" << grid.size() << "'";
		throw hpgl_exception("sequential_indicator_simulation_lvm", oss.str());
	}

	if (mean_data == nullptr)
	{
		throw hpgl_exception("sequential_indicator_simulation_lvm",
			"Null mean_data pointer-to-pointer");
	}
		
	if(use_corellogram)
	{
		print_param("Corellogram", "on");
		if (mask == NULL)
			do_sis(property, grid, params, seed, mean_data, report, corellogram_weight_calculator_t(), no_mask_t());
		else
			do_sis(property, grid, params, seed, mean_data, report, corellogram_weight_calculator_t(), mask);
	} else {
		print_param("Corellogram", "off");
		if (mask == NULL)
			do_sis(property, grid, params, seed, mean_data, report, sk_weight_calculator_t(), no_mask_t());
		else
			do_sis(property, grid, params, seed, mean_data, report, sk_weight_calculator_t(), mask);
	}
}

} //namespace hpgl
