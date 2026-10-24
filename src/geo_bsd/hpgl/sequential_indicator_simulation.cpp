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
#include "kriging_stats.h"
#include "api.h"

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
	// category_count > 255 causes infinite wrap-around in the
	// indicator_index_t casts below.
	if (params.m_category_count > 255)
		throw hpgl_exception("do_sis",
			"indicator count exceeds indicator_index_t max (255)");

	// G-08: the per-category structures built below (covariances[K],
	// nblookups[K]) stay resident for the whole run, and each is validated only
	// against its OWN box cap (COVARIANCE_FIELD_VOLUME_CAP = 1e8 cells per
	// category, enforced per entry in precalculated_covariances_t::init and
	// neighbourhood_param_t::set_radiuses). Nothing bounds the aggregate: at
	// K=255 with r=231 that is ~204 GB of covariance tables alone (measured
	// 24-33 B/cell, so 20 B/cell is a floor of the true residency) — a
	// configuration BOTH layers admit (validation.py MAX_INDICATORS = 255,
	// api.cpp indicator_count > 255). Sum the per-category box volumes here and
	// reject before the first allocation.
	//
	// F1: the ceiling must not reject a configuration the two documented layers
	// admit ONE OBJECT AT A TIME — the per-object cap is 1e8 cells
	// (covariance_field.h COVARIANCE_FIELD_VOLUME_CAP, mirrored by
	// validation.py MAX_COVARIANCE_FIELD_VOLUME), i.e. 2e9 B at the measured
	// 20 B/cell. The ceiling is that same worst case (2 GiB, identical to the
	// H-01 sibling's MAX_AGGREGATE_COVARIANCE_BYTES in indicator_kriging.h), so
	// no single legal category and no small legal group fails: the previous
	// 0.5 GiB ceiling admitted only 2.68e7 cells and rejected, e.g., a legal
	// 2-category run at radius 119 (2 x 239^3 = 2.73e7 cells). It is still an
	// aggregate bound — K x 1e8 cells stays refused.
	//
	// F1: each category's residency is the LARGER of the two boxes the run
	// allocates — the covariance table (m_radiuses[i], built below) and the
	// lookup's own box (m_nb_params[i].m_radiuses, built inside nl_t). The
	// H-01 twin takes max(cov_box, nb_box) for the same reason; counting only
	// the covariance box under-counted the residency whenever nb > cov.
	{
		static const double MAX_SIS_AGGREGATE_BYTES = 2.0 * 1024.0 * 1024.0 * 1024.0;
		static const double BYTES_PER_BOX_CELL = 20.0; // 8 V table + <= 12 V offsets
		double aggregate_cells = 0.0;
		for (size_t i = 0; i < params.m_category_count; ++i)
		{
			const size_t cov_box = validate_covariance_radiuses_or_throw(
				params.m_radiuses[i], "do_sis");
			const size_t nb_box = validate_covariance_radiuses_or_throw(
				params.m_nb_params[i].m_radiuses, "do_sis");
			aggregate_cells += static_cast<double>(cov_box > nb_box ? cov_box : nb_box);
		}
		const double aggregate_bytes = aggregate_cells * BYTES_PER_BOX_CELL;
		if (aggregate_bytes > MAX_SIS_AGGREGATE_BYTES)
		{
			std::ostringstream oss;
			oss << "SIS aggregate covariance memory " << (aggregate_bytes / (1024.0 * 1024.0 * 1024.0))
			    << " GB across " << params.m_category_count
			    << " categories exceeds the " << (MAX_SIS_AGGREGATE_BYTES / (1024.0 * 1024.0 * 1024.0))
			    << " GB limit; reduce the per-category radiuses or the category count.";
			throw hpgl_exception("do_sis", oss.str());
		}
	}

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

	// Track kriging failures for visibility — SIS silently substitutes
	// marginal probabilities on failure, which can bias results if
	// failures are widespread.
	unsigned long kriging_failures = 0;
	unsigned long kriging_skipped = 0;
	unsigned long points_calculated = 0;
	unsigned long points_without_neighbours = 0;
	unsigned long points_singularity = 0;
	unsigned long nodes_processed = 0;
	double sum_categories = 0;

	if(params.m_category_count == 2)
	{
		write("Only 2 indicators found, Median SIS will be performed.");
	}

	while (!path.end_of_path())	
	{
		if (reporter.cancelled())
		{
			// Match cancellation behaviour of parallel kriging loops
			// (cont_kriging.h, indicator_kriging.h, median_ik.cpp) —
			// those check report.cancelled(); single-threaded SIS
			// previously lacked this check.
			break;
		}
		node_index_t node = path.get_next();
		reporter.next_lap();
		if (property.is_informed(node))
		{
			++kriging_skipped;
			continue;
		}

		// Bounds-guard: validate node index before mask access.
		// node_index_t is signed int; path generator should produce valid
		// indices but double-check in case mask is undersized.
		if (node < 0 || node >= property.size())
		{
			++kriging_skipped;
			continue;
		}

		if (mask[node] != 1)
		{
			++kriging_skipped;
			continue;
		}

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
						++kriging_failures;
					}
					switch (ki_result)
					{
					case ki_result_t::KI_SUCCESS: ++points_calculated; break;
					case ki_result_t::KI_NO_NEIGHBOURS: ++points_without_neighbours; break;
					case ki_result_t::KI_SINGULARITY: ++points_singularity; break;
					}

					// Clamp kriged probability to [0,1] before computing the
					// complement so both probabilities are well-formed (F-37).
					// SK weights are unconstrained — poorly conditioned matrices,
					// sparse neighbours, or extreme anisotropy can push the
					// combine() result outside [0,1]. NaN-safe (II-11): NaN
					// bypasses the relational clamps (NaN < 0.0 is false), so
					// non-finite values fall back to the marginal probability
					// exactly like the KI-failure path above.
					prob = detail::sanitize_probability(prob, marginal_probs[idx][node]);
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
					++kriging_failures;
				}
				switch (ki_result)
				{
				case ki_result_t::KI_SUCCESS: ++points_calculated; break;
				case ki_result_t::KI_NO_NEIGHBOURS: ++points_without_neighbours; break;
				case ki_result_t::KI_SINGULARITY: ++points_singularity; break;
				}
				// F-37: the multi-category branch pushed the RAW kriged
				// probability with no clamp. A value outside [0,1] — or NaN,
				// which survives correct_order_relations below because NaN
				// comparisons are false — silently biased the CDF and made
				// the sampler return category 0. Sanitize each probability
				// (finite + [0,1], marginal fallback) before the cumulative
				// sum, mirroring the 2-category branch.
				prob = detail::sanitize_probability(prob, marginal_probs[idx][node]);
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
		indicator_value_t sampled = sample(gen, probs);
		property.set_at(node, sampled);
		sum_categories += static_cast<double>(sampled);
		++nodes_processed;
	}

	reporter.stop();
	// F-M6: surface SIS kriging failures via stats (previously the counter
	// below was stderr-only; the Python wrapper could not observe SIS
	// solver failures and geo._last_kriging_stats stayed None). Counters
	// are per-category kriging evaluations; mean is the average category.
	{
		kriging_stats_t stats;
		stats.m_points_calculated = points_calculated;
		stats.m_points_without_neighbours = points_without_neighbours;
		stats.m_points_singularity = points_singularity;
		stats.m_mean = nodes_processed > 0 ? sum_categories / static_cast<double>(nodes_processed) : 0;
		stats.m_speed_nps = reporter.iterations_per_second();
		set_kriging_stats(stats);
	}
	if (kriging_failures > 0)
	{
		// G-23: documented output channel (write() → registered output
		// handler, stdout fallback), not raw stderr — the SGS sibling sites
		// in sequential_simulation.h were converted for the same reason: a
		// handler-based consumer never saw this aggregate.
		std::ostringstream oss;
		oss << "HPGL: SIS kriging failures: " << kriging_failures
		    << " indicator-estimate failures fell back to marginal probabilities (of "
		    << static_cast<unsigned long>(property.size()) << " nodes, "
		    << kriging_skipped << " skipped).\n";
		write(oss.str());
	}
	// P-02: no-output guard mirroring the SGS analog
	// (sequential_simulation.h:443-449). SIS has no ndmin gate, so every path
	// node reaches exactly one of {kriging_skipped, nodes_processed} — an
	// all-masked / fully-informed run previously returned the unchanged input
	// clone with no signal at all (the only stderr path above requires
	// kriging_failures > 0). Warn when every node was skipped.
	if (kriging_skipped >= static_cast<unsigned long>(property.size()))
	{
		// G-23: handler-visible (see the kriging-failures message above).
		std::ostringstream oss;
		oss << "HPGL: SIS produced no output — all "
		    << static_cast<unsigned long>(property.size())
		    << " nodes were either already informed, masked out, "
		       "or out of bounds. Check that the output property grid contains uninformed cells, "
		       "and the mask permits processing.\n";
		write(oss.str());
	}
	// F-379 (CONFIRMED MEDIUM): a cancelled run used to fall through to the
	// success path — sequential_indicator_simulation / _lvm returned normally,
	// so the partially simulated property was indistinguishable from a
	// completed simulation. Cancellation is a control-flow outcome, so it must
	// reach the caller: throwing here makes the C API catch (api.cpp
	// hpgl_sis_simulation / hpgl_sis_simulation_lvm) record the message the FFI
	// layer turns into a Python RuntimeError. The caller's buffer stays
	// partially written (unavoidable — it is the working buffer), but it can no
	// longer be mistaken for a finished run. SIS writes no "Done" line, so this
	// is the position the SGS fix used: after the counters, stats and warnings.
	// Block direction: the loop still stops promptly on cancel (the
	// `if (reporter.cancelled()) break;` above is unchanged); the pre-existing
	// stats are still published, so a caller inspecting them sees the real
	// zero/short counters.
	if (reporter.cancelled())
		throw hpgl_exception("do_sis",
			"cancelled by progress handler; the output property is only partially simulated");
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
		size_t mean_data_count,
		size_t mean_row_size,
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

	// E2-118: the raw pointer-to-pointer carries no row count and the
	// rows no length — do_sis dereferences mean_data[idx][node] for every
	// category (idx < m_category_count) and every node (node <
	// property.size()), a heap OOB for direct-C++ callers with too few
	// rows or shorter rows. The count/row-length are now explicit contract
	// parameters, validated here (mirroring the C API gate
	// api.cpp:1601-1609 which requires indicator_count rows of exactly
	// grid-volume size).
	if (mean_data_count != static_cast<size_t>(params.m_category_count))
	{
		std::ostringstream oss;
		oss << "mean_data row count: " << mean_data_count
		    << " != category count: " << params.m_category_count;
		throw hpgl_exception("sequential_indicator_simulation_lvm", oss.str());
	}
	if (mean_row_size != static_cast<size_t>(property.size()))
	{
		std::ostringstream oss;
		oss << "mean_data row size: " << mean_row_size
		    << " != property size: " << property.size();
		throw hpgl_exception("sequential_indicator_simulation_lvm", oss.str());
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
