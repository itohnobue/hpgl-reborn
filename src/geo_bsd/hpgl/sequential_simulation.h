#ifndef SEQUENTIAL_SIMULATION_H_INCLUDED_DEU0932RMNPDVCPDZFIG03249MROPWEUFSDHVIAERT0934NTFMD0GN2304UNLKWEJOFDSZF0G32R
#define SEQUENTIAL_SIMULATION_H_INCLUDED_DEU0932RMNPDVCPDZFIG03249MROPWEUFSDHVIAERT0934NTFMD0GN2304UNLKWEJOFDSZF0G32R

#include "cdf_utils.h"
#include "path_random_generator.h"
#include "bs_random_generator.h"
#include "typedefs.h"
#include "sample.h"
#include "covariance_field.h"
#include "progress_reporter.h"
#include "combiner.h"
#include "kriging_interpolation.h"
#include "precalculated_covariance.h"
#include "neighbourhood_lookup.h"
#include "is_informed_predicate.h"
#include "cov_model.h"
#include "kriging_stats.h"
#include "api.h"
#include <sstream>

namespace hpgl
{

	struct no_mask_t
	{
		int operator[](node_index_t /*node*/)const
		{
			return 1;
		}
	};

	template <typename grid_t, typename mean_provider_t, typename fallback_mean_provider_t, typename weight_calculator_t, typename mask_t>
	void do_sequential_gausian_simulation(
		cont_property_array_t & property,
		const grid_t & grid,
		const sgs_params_t & params,
		const mean_provider_t & mp,
		// R-5: separate fallback mean provider. The kriging estimate and the
		// failure fallback draw can need DIFFERENT means: the SGS-OK path
		// passes no_mean_t() as mp (GSLIB zero-mean normal-score semantics —
		// no (1−Σλᵢ)·mean term on the n<4 SK-downgraded estimate) while the
		// fallback must still draw N(gmean, 1.0) per GSLIB
		// (sgsim.for: `cmean = gmean; cstdev = 1.0`; M-29). KRIG_SIMPLE and
		// LVM pass the same provider for both.
		const fallback_mean_provider_t & fallback_mp,
		const weight_calculator_t & weight_calculator_sgs,
		const mask_t & mask)
	{

		if (property.size() != grid.size())
		{
			std::ostringstream oss;
			oss << "Property size '" << property.size() << "' is not equal to grid size '" << grid.size() << "'";
			throw hpgl_exception("do_sequential_gausian_simulation", oss.str());
		}

		mt_random_generator_t gen;
		gen.seed(params.m_seed);
		path_random_generator_t path_gen(property.size(), params.m_seed);

		typedef precalculated_covariances_t cov_t;
		cov_t pcov(cov_model_t(params), params.m_radiuses);
		typedef /*indexed_*/ neighbour_lookup_t<grid_t, cov_t> nl_t;
		nl_t neighbour_lookup(&grid, &pcov,	params);
		
		/*for (node_index_t node = 0; node < property.size(); ++node)
		{
			if (property.is_informed(node))
				neighbour_lookup.add_node(node);
		}*/
		
		progress_reporter_t report(property.size());	
		report.start();

		// M-11 (GSLIB ndmin semantics): the ndmin gate must count ORIGINAL
		// conditioning data only (sgsim.for: "If there are fewer than ndmin
		// data points the node is not simulated" — ndmin counts original
		// data, never previously simulated nodes). The plain neighbour
		// lookup applies the live informed-mask predicate at call time, and
		// set_at marks each simulated node informed, so previously simulated
		// nodes re-enter conditioning and would inflate ws.indices.size().
		// Snapshot which cells were originally informed so the gate can
		// exclude simulated nodes from the count. This is a deliberate
		// GSLIB-compliance change: it changes covariance reproduction between
		// simulated cells only through the ndmin skip threshold — the
		// conditioning set itself still includes simulated nodes per HPGL's
		// established (non-GSLIB) design.
		std::vector<unsigned char> originally_informed(property.size(), 0);
		for (node_index_t i = 0; i < property.size(); ++i)
			originally_informed[i] = property.is_informed(i) ? 1 : 0;

		node_index_t node;
		kriging_ws_t<cont_value_t, sugarbox_location_t> ws;
		unsigned long kriging_failures = 0;
		unsigned long kriging_skipped = 0;
		unsigned long kriging_ndmin_skipped = 0;
		unsigned long points_calculated = 0;
		unsigned long points_without_neighbours = 0;
		unsigned long points_singularity = 0;
		// 2-M-33: count every node that received a simulated value (success
		// OR failure fallback) so m_mean uses a numerator==denominator pair
		// like the four sibling consumers (cont_kriging, indicator_kriging,
		// cokriging, SIS). Previously the denominator was points_calculated
		// (KI_SUCCESS only) while sum_simulated accumulated over ALL
		// simulated nodes — a biased mean on partial failure.
		unsigned long nodes_simulated = 0;
		double sum_simulated = 0;
		for (node_index_t counter = 0, counter_end = property.size(); counter < counter_end; ++counter, report.next_lap())		
		{
			node = path_gen.next();
			if(property.is_informed(node)) 
			{
				++kriging_skipped;
				continue;
			}

			// Bounds-guard: validate node index before mask access
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

			double variance = 0.0;						
			sugarbox_location_t loc = grid[node];			

			cont_value_t mean = 0.0f;
			ki_result_t ki_result = kriging_interpolation_ws(property, is_informed_predicate_t<cont_property_array_t>(property), node, pcov, mp, 
				neighbour_lookup, weight_calculator_sgs, mean, variance, ws);			

			// GSLIB ndmin semantics (F-14 + M-11): when fewer than
			// m_min_neighbours ORIGINAL conditioning data are available, leave
			// the node unsimulated instead of simulating from the marginal
			// distribution. GSLIB sgsim.for counts original data only
			// (`if(nclose.lt.ndmin) go to 5` — nclose comes from srchsupr, the
			// input-data search; previously simulated nodes are searched
			// separately by srchnd and do NOT count toward ndmin). The plain
			// lookup applies the live informed-mask predicate at call time, so
			// previously simulated nodes re-enter conditioning and inflate
			// ws.indices.size(). The gate therefore counts ORIGINAL data only:
			// when m_min_neighbours > 0, skip iff the number of originally
			// informed conditioning cells is < m_min_neighbours — with NO outer
			// total-precondition. A node with fewer than ndmin original data is
			// skipped even when previously simulated neighbours push the total
			// count above the threshold. R-3: the round-1 fix nested this check
			// inside `ws.indices.size() < min`, where original_count <= size
			// always holds — making the gate byte-identical to the pre-fix total
			// gate. The outer precondition is removed here.
			if (params.m_min_neighbours > 0)
			{
				size_t original_data_count = 0;
				for (size_t k = 0; k < ws.indices.size(); ++k)
				{
					if (originally_informed[ws.indices[k]])
						++original_data_count;
				}
				if (original_data_count < static_cast<size_t>(params.m_min_neighbours))
				{
					++kriging_ndmin_skipped;
					continue;
				}
			}

			if (ki_result != ki_result_t::KI_SUCCESS)
				++kriging_failures;

			switch (ki_result)
			{
			case ki_result_t::KI_SUCCESS: ++points_calculated; break;
			case ki_result_t::KI_NO_NEIGHBOURS: ++points_without_neighbours; break;
			case ki_result_t::KI_SINGULARITY: ++points_singularity; break;
			}

			double value = ki_result == ki_result_t::KI_SUCCESS
				? sample(gen, gaussian_cdf_t(mean, variance))
				: sample(gen, gaussian_cdf_t(fallback_mp[node], 1.0));
			
			property.set_at(node, value);
			sum_simulated += value;
			++nodes_simulated;
			//neighbour_lookup.add_node(node);
		}
		report.stop();
		// F-M6: surface kriging failures via stats (previously the counters
		// below were stderr-only; the Python wrapper could not observe SGS
		// solver failures and geo._last_kriging_stats stayed None). The
		// points_calculated semantics mirror cont_kriging: KI_SUCCESS cells.
		{
			kriging_stats_t stats;
			stats.m_points_calculated = points_calculated;
			stats.m_points_without_neighbours = points_without_neighbours;
			stats.m_points_singularity = points_singularity;
			// 2-M-33: divide by ALL simulated nodes (nodes_simulated), not
			// the success-only points_calculated — sum_simulated includes
			// failure-fallback draws.
			stats.m_mean = nodes_simulated > 0 ? sum_simulated / static_cast<double>(nodes_simulated) : 0;
			stats.m_speed_nps = report.iterations_per_second();
			set_kriging_stats(stats);
		}
		if (kriging_failures > 0)
		{
			fprintf(stderr,
				"HPGL: SGS kriging failures: %lu nodes fell back to marginal mean (of %lu total, %lu skipped).\n",
				kriging_failures, static_cast<unsigned long>(property.size()),
				kriging_skipped);
		}
		if (kriging_ndmin_skipped > 0)
		{
			fprintf(stderr,
				"HPGL: SGS ndmin: %lu nodes left unsimulated (fewer than %d conditioning data).\n",
				kriging_ndmin_skipped, params.m_min_neighbours);
		}
		if (kriging_skipped >= static_cast<unsigned long>(property.size()))
		{
			fprintf(stderr,
				"HPGL: SGS produced no output — all %lu nodes were either already informed or masked out. "
				"Check that the output property grid contains uninformed cells and the mask permits processing.\n",
				static_cast<unsigned long>(property.size()));
		}
		{
			std::ostringstream oss;
			oss << "Done. Average speed: " << report.iterations_per_second() << " point/sec.\n";
			write(oss.str());
		}
	}

	template <typename grid_t, typename mean_provider_t, typename weight_calculator_t>
	void do_sequential_gausian_simulation_in_points(
		cont_property_array_t & property,
		const grid_t & grid,
		const sgs_params_t & params,
		const mean_provider_t & mp,		
		const weight_calculator_t & weight_calculator_sgs,
		const std::vector<int> & points_indexes)
	{
		if (property.size() != grid.size())
		{
			std::ostringstream oss;
			oss << "Property size '" << property.size() << "' is not equal to grid size '" << grid.size() << "'";
			throw hpgl_exception("do_sequential_gausian_simulation_in_points", oss.str());
		}

		mt_random_generator_t gen;
		gen.seed(params.m_seed);
		path_random_generator_t path_gen(points_indexes.size(), params.m_seed);

		typedef precalculated_covariances_t cov_t;
		//typedef precalculated_covariances_t<cov_model_t, sugarbox_location_t> cov_t;
		cov_t pcov(cov_model_t(params), params.m_radiuses);
		typedef /*indexed_*/ neighbour_lookup_t<grid_t, cov_t> nl_t;	
		nl_t neighbour_lookup(&grid, &pcov,	params);
		
		/*for (node_index_t node = 0; node < property.size(); ++node)
		{
			if (property.is_informed(node))
				neighbour_lookup.add_node(node);
		}*/
		
		progress_reporter_t report(points_indexes.size());	
		report.start();
		node_index_t node;
		kriging_ws_t<cont_value_t, sugarbox_location_t> ws;
		unsigned long kriging_failures = 0;
		unsigned long kriging_skipped = 0;
		// report.next_lap() in the increment expression ensures laps are
		// counted even when is_informed() causes a 'continue' (matching
		// the regular do_sequential_gausian_simulation variant at line 68).
		for (node_index_t counter = 0, counter_end = points_indexes.size(); counter < counter_end; ++counter, report.next_lap())		
		{
			node = points_indexes[path_gen.next()];
			if(property.is_informed(node)) 
			{
				++kriging_skipped;
				continue;
			}

			double variance = 0.0;						
			sugarbox_location_t loc = grid[node];			

			cont_value_t mean = 0.0f;
			ki_result_t ki_result = kriging_interpolation_ws(property, is_informed_predicate_t<cont_property_array_t>(property), node, pcov, mp, 
				neighbour_lookup, weight_calculator_sgs, mean, variance, ws);			

			if (ki_result != KI_SUCCESS)
				++kriging_failures;

			double value = ki_result == KI_SUCCESS 
				? sample(gen, gaussian_cdf_t(mean, variance))
				: sample(gen, gaussian_cdf_t(mp[node], 1.0));		
			
			property.set_at(node, value);
			//neighbour_lookup.add_node(node);
		}
		report.stop();
		if (kriging_failures > 0)
		{
			fprintf(stderr,
				"HPGL: SGS (in_points) kriging failures: %lu nodes fell back to marginal mean (of %lu total, %lu skipped).\n",
				kriging_failures, static_cast<unsigned long>(points_indexes.size()),
				kriging_skipped);
		}
		{
			std::ostringstream oss;
			oss << "Done. Average speed: " << report.iterations_per_second() << " point/sec.\n";
			write(oss.str());
		}
	}
}

#endif //SEQUENTIAL_SIMULATION_H_INCLUDED_DEU0932RMNPDVCPDZFIG03249MROPWEUFSDHVIAERT0934NTFMD0GN2304UNLKWEJOFDSZF0G32R
