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

	template <typename grid_t, typename mean_provider_t, typename weight_calculator_t, typename mask_t>
	void do_sequential_gausian_simulation(
		cont_property_array_t & property,
		const grid_t & grid,
		const sgs_params_t & params,
		const mean_provider_t & mp,		
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
		node_index_t node;
		kriging_ws_t<cont_value_t, sugarbox_location_t> ws;
		unsigned long kriging_failures = 0;
		unsigned long kriging_skipped = 0;
		unsigned long kriging_ndmin_skipped = 0;
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

			// GSLIB ndmin semantics (F-14): when fewer than
			// m_min_neighbours conditioning data are available, leave the
			// node unsimulated instead of simulating from the marginal
			// distribution. Previously m_min_neighbours was stored/printed
			// but never wired into the simulation logic.
			if (params.m_min_neighbours > 0
				&& ws.indices.size() < static_cast<size_t>(params.m_min_neighbours))
			{
				++kriging_ndmin_skipped;
				continue;
			}

			if (ki_result != ki_result_t::KI_SUCCESS)
				++kriging_failures;

			double value = ki_result == ki_result_t::KI_SUCCESS
				? sample(gen, gaussian_cdf_t(mean, variance))
				: sample(gen, gaussian_cdf_t(mp[node], 1.0));		
			
			property.set_at(node, value);
			//neighbour_lookup.add_node(node);
		}
		report.stop();
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
