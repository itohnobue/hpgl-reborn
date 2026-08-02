#ifndef __KRIGING_INTERPOLATION_H__31DBE328_667D_49C4_A4CB_830C3FA89872
#define __KRIGING_INTERPOLATION_H__31DBE328_667D_49C4_A4CB_830C3FA89872

#include "select.h"
#include "combiner.h"
#include "my_kriging_weights.h"

namespace hpgl
{	
	class sk_weight_calculator_t
	{		
	public:
		template<typename covariances_t, typename coord_t>
		bool operator()(
			const coord_t & center,		
			const std::vector<coord_t> & coords,			
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights)const
		{
			double variance;
			return sk_kriging_weights_3<covariances_t, false, coord_t>(center, coords, covariances, weights, variance);
		}

		template<typename covariances_t, typename means_t, typename coord_t>
		bool operator()(
			const coord_t & center,
			mean_t /*center_mean*/,
			const std::vector<coord_t> & coords,
			const means_t &,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights)const
		{
			double variance;
			return sk_kriging_weights_3<covariances_t, false, coord_t>(center, coords, covariances, weights, variance);
		}

		template<typename covariances_t, typename coord_t>
		bool operator()(const coord_t & center, 			
			const std::vector<coord_t> & coords,
			
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights, double & variance)const
		{
			return sk_kriging_weights_3<covariances_t, true, coord_t>(center, coords, covariances, weights, variance);
		}	

		template<typename covariances_t, typename means_t, typename coord_t>
		bool operator()(const coord_t & center, 
			mean_t /*center_mean*/,
			const std::vector<coord_t> & coords,
			const means_t &,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights, double & variance)const
		{
			return sk_kriging_weights_3<covariances_t, true, coord_t>(center, coords, covariances, weights, variance);
		}	

		// Workspace-aware overloads — pass ws to sk_kriging_weights_3_ws
		template<typename covariances_t, typename means_t, typename coord_t>
		bool operator()(const coord_t & center,
			mean_t /*center_mean*/,
			const std::vector<coord_t> & coords,
			const means_t &,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights,
			weight_calc_workspace_t & ws)const
		{
			double variance;
			return sk_kriging_weights_3_ws<covariances_t, false, coord_t>(center, coords, covariances, weights, variance, ws);
		}

		template<typename covariances_t, typename means_t, typename coord_t>
		bool operator()(const coord_t & center,
			mean_t /*center_mean*/,
			const std::vector<coord_t> & coords,
			const means_t &,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights, double & variance,
			weight_calc_workspace_t & ws)const
		{
			return sk_kriging_weights_3_ws<covariances_t, true, coord_t>(center, coords, covariances, weights, variance, ws);
		}

		template<typename covariances_t, typename coord_t>
		inline bool first_stage(
						 const coord_t & center, 
						 const std::vector<coord_t> & coords, 
						 const covariances_t & covariances,
						 std::vector<kriging_weight_t> & weights)const
		{	
			double variance;
			return sk_kriging_weights_3<covariances_t, false, coord_t>(center, coords, covariances, weights, variance);
		}	

		template<typename means_t>
		inline bool second_stage(
						  std::vector<kriging_weight_t> & /*weights*/,
						  const mean_t /*center_mean*/,
						  const means_t & /*means*/)const
		{
			return true;
		}
	};
	
	class ok_weight_calculator_t
	{
	public:
		template<typename covariances_t, typename coord_t>
		bool operator()(
			const coord_t & center,		
			const std::vector<coord_t> & coords,		
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights)const
		{
			double variance;
			return ok_kriging_weights_3<covariances_t, false, coord_t>(center, coords, covariances, weights, variance);
		}

		template<typename covariances_t, typename means_t, typename coord_t>
		bool operator()(
			const coord_t & center,
			mean_t /*center_mean*/,
			const std::vector<coord_t> & coords,
			const means_t &,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights)const
		{
			double variance;
			return ok_kriging_weights_3<covariances_t, false, coord_t>(center, coords, covariances, weights, variance);
		}

		template<typename covariances_t, typename coord_t>
		bool operator()(const coord_t & center, 			
			const std::vector<coord_t> & coords,			
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights, double & variance)const
		{
			return ok_kriging_weights_3<covariances_t, true, coord_t>(center, coords, covariances, weights, variance);
		}

		template<typename covariances_t, typename means_t, typename coord_t>
		bool operator()(const coord_t & center, 
			mean_t /*center_mean*/,
			const std::vector<coord_t> & coords,
			const means_t &,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights, double & variance)const
		{
			return ok_kriging_weights_3<covariances_t, true, coord_t>(center, coords, covariances, weights, variance);
		}			

		// Workspace-aware overloads — pass ws to ok_kriging_weights_3_ws
		template<typename covariances_t, typename means_t, typename coord_t>
		bool operator()(const coord_t & center,
			mean_t /*center_mean*/,
			const std::vector<coord_t> & coords,
			const means_t &,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights,
			weight_calc_workspace_t & ws)const
		{
			double variance;
			return ok_kriging_weights_3_ws<covariances_t, false, coord_t>(center, coords, covariances, weights, variance, ws);
		}

		template<typename covariances_t, typename means_t, typename coord_t>
		bool operator()(const coord_t & center,
			mean_t /*center_mean*/,
			const std::vector<coord_t> & coords,
			const means_t &,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights, double & variance,
			weight_calc_workspace_t & ws)const
		{
			return ok_kriging_weights_3_ws<covariances_t, true, coord_t>(center, coords, covariances, weights, variance, ws);
		}
	};

	// M-3 (GSLIB sgsim OK→SK downgrade) — SGS-ONLY weight calculator (R-4/R-5).
	// GSLIB sgsim switches from OK to SK when fewer than 4 conditioning data
	// are available (`if(ktype.eq.1.and.(nclose+ncnode).lt.4)lktype=0`) to
	// avoid the "artificially inflated" OK kriging variance the Lagrange
	// multiplier introduces for very sparse neighbourhoods (a nugget-only
	// model gives 2×sill for n=1). The downgrade is implemented ONLY here —
	// the shared ok_kriging_weights_3(_ws) calculators keep the public
	// hpgl_ordinary_kriging (the kt3d analog) OK contract for n<4 (weights
	// sum to 1). R-5: the SGS-OK path combines this calculator with
	// no_mean_t() (sequential_gaussian_simulation.cpp), so the downgraded SK
	// estimate Σλᵢzᵢ has no (1−Σλᵢ)·mean term — matching GSLIB's zero-mean
	// normal-score semantics — and the n≥4 OK estimate Σλᵢzᵢ (Σλ=1) is
	// identical. This removes the n=4 discontinuity where the mean-pull
	// abruptly turned off as local data density crossed 4.
	class ok_sgs_weight_calculator_t
	{
	public:
		template<typename covariances_t, typename coord_t>
		bool operator()(
			const coord_t & center,
			const std::vector<coord_t> & coords,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights)const
		{
			double variance;
			return (*this)(center, coords, covariances, weights, variance);
		}

		template<typename covariances_t, typename means_t, typename coord_t>
		bool operator()(
			const coord_t & center,
			mean_t /*center_mean*/,
			const std::vector<coord_t> & coords,
			const means_t &,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights)const
		{
			double variance;
			return (*this)(center, coords, covariances, weights, variance);
		}

		template<typename covariances_t, typename coord_t>
		bool operator()(const coord_t & center,
			const std::vector<coord_t> & coords,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights, double & variance)const
		{
			if (coords.size() < 4)
			{
				return sk_kriging_weights_3<covariances_t, true, coord_t>(center, coords, covariances, weights, variance);
			}
			return ok_kriging_weights_3<covariances_t, true, coord_t>(center, coords, covariances, weights, variance);
		}

		template<typename covariances_t, typename means_t, typename coord_t>
		bool operator()(const coord_t & center,
			mean_t /*center_mean*/,
			const std::vector<coord_t> & coords,
			const means_t &,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights, double & variance)const
		{
			if (coords.size() < 4)
			{
				return sk_kriging_weights_3<covariances_t, true, coord_t>(center, coords, covariances, weights, variance);
			}
			return ok_kriging_weights_3<covariances_t, true, coord_t>(center, coords, covariances, weights, variance);
		}

		// Workspace-aware overloads — used by the SGS-OK path
		// (sequential_simulation.h kriging_interpolation_ws).
		template<typename covariances_t, typename means_t, typename coord_t>
		bool operator()(const coord_t & center,
			mean_t /*center_mean*/,
			const std::vector<coord_t> & coords,
			const means_t &,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights,
			weight_calc_workspace_t & ws)const
		{
			double variance;
			return (*this)(center, coords, covariances, weights, variance, ws);
		}

		template<typename covariances_t, typename means_t, typename coord_t>
		bool operator()(const coord_t & center,
			mean_t /*center_mean*/,
			const std::vector<coord_t> & coords,
			const means_t &,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights, double & variance,
			weight_calc_workspace_t & ws)const
		{
			if (coords.size() < 4)
			{
				return sk_kriging_weights_3_ws<covariances_t, true, coord_t>(center, coords, covariances, weights, variance, ws);
			}
			return ok_kriging_weights_3_ws<covariances_t, true, coord_t>(center, coords, covariances, weights, variance, ws);
		}
	};



	
	class corellogram_weight_calculator_t
	{		
	public:
		template<typename covariances_t, typename means_t, typename coord_t>
		bool operator()(const coord_t & center, 
			mean_t center_mean,
			const std::vector<coord_t> & coords,
			const means_t & means,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights)const
		{
			return corellogramed_weights_3(center, center_mean, coords, covariances, means, weights);
		}

		// Workspace-aware overload
		template<typename covariances_t, typename means_t, typename coord_t>
		bool operator()(const coord_t & center,
			mean_t center_mean,
			const std::vector<coord_t> & coords,
			const means_t & means,
			const covariances_t & covariances,
			std::vector<kriging_weight_t> & weights,
			weight_calc_workspace_t & ws)const
		{
			return corellogramed_weights_3_ws(center, center_mean, coords, covariances, means, weights, ws);
		}

		template<typename covariances_t, typename coord_t>
		inline bool first_stage(
						 const coord_t & center, 
						 const std::vector<coord_t> & coords, 
						 const covariances_t & covariances,
						 std::vector<kriging_weight_t> & weights)const
		{
		    double variance;
			return sk_kriging_weights_3<covariances_t, false, coord_t>(center, coords, covariances, weights, variance);
		}

		template<typename means_t>
		inline bool second_stage(
						  std::vector<kriging_weight_t> & weights,
						  const mean_t center_mean,
						  const means_t & means)const
		{
			// Type safety: Use size_t for size variable, validate it fits in int
			const size_t size = weights.size();

			double delta = CORRELOGRAM_DELTA;
			double meanc = center_mean;

			// NaN guard: NaN propagates through all comparisons and
			// arithmetic, silently invalidating the correlogram weights.
			// Fall back to safe default (0.0) and let the caller's NaN
			// detection handle the downstream consequences.
			if (std::isnan(meanc)) meanc = 0.0;

			// Tolerance-based comparison prevents exact-float-equality
			// failures when a mean arrives at 0.0 or 1.0 through
			// numerically fragile paths (e.g., 1.0 + 0.0 == 1.0 exact,
			// but 1.0 - 1e-16 rounds back to 1.0 which == 1.0 exact).
			// Using the same pattern as my_kriging_weights.h:833-839.
			if(meanc < delta)
			{
				meanc = delta;
			}
			if(meanc > 1.0 - delta)
			{
				meanc = 1.0 - delta;
			}

			double sigmac = sqrt(meanc * (1 - meanc));

			double meani = 0;
			// Type safety: Use size_t for loop index
			for (size_t i = 0; i < size; ++i)
			{
				meani = means[i];

				if (std::isnan(meani)) meani = 0.0;

				if(meani < delta)
				{
					meani = delta;
				}
				if(meani > 1.0 - delta)
				{
					meani = 1.0 - delta;
				}

				double sigma = sqrt(meani * (1 - meani));
				weights[i] *= sigmac / sigma;
			}
			return true;
		}
	};

	// -----------------------------------------------------------------------
	// Kriging workspace: reusable vectors for the interpolation hot path.
	// Allocated once per thread (OpenMP parallel region), reused across all
	// node iterations via resize() which is allocation-free when capacity is
	// sufficient. Eliminates 5-10 heap allocations per node in the inner loop.
	// -----------------------------------------------------------------------
	template<typename value_t, typename coord_t>
	struct kriging_ws_t {
		std::vector<node_index_t> indices;
		std::vector<kriging_weight_t> weights;
		std::vector<mean_t> means;
		std::vector<value_t> values;
		std::vector<coord_t> coords;
		coord_t node_coord;
		weight_calc_workspace_t wcalc;
	};


	enum ki_result_t
	{
		KI_SUCCESS = 0,
		KI_NO_NEIGHBOURS = 1,
		KI_SINGULARITY = 2
	};

	template<typename params_t, typename grid_t, typename result_t>
	ki_result_t kriging_interpolation(
		const params_t & params,		
		node_index_t index,		
		result_t & result)
	{
		return kriging_interpolation(
			*params.input_values, 
			*params.defineds, 
			index, 
			*params.covariances, 
			*params.means, 
			*params.neighbour_lookup, 
			*params.weight_calculator, result);
	}

	template<		
		typename values_t,
		typename defineds_t,
		typename means_t,
		typename covariances_t,
		typename neighbourhood_lookup_t,
		typename weight_calculator_t,
		typename result_t>
	ki_result_t kriging_interpolation(
			const values_t & input_values,
			const defineds_t & defineds,
			node_index_t index,			
			const covariances_t & cov,
			const means_t & mp,
			const neighbourhood_lookup_t & nl,
			const weight_calculator_t & wc,
			result_t & result
		)
	{
		typedef typename values_t::value_type value_t;
		typedef typename neighbourhood_lookup_t::coord_t coord_t;
		// Per-node heap allocations are bounded by max_neighbours (typically small),
		// not by grid size. The neighbourhood lookup limits the number of entries
		// returned to m_max_neighbours, so these vectors are O(max_neighbours) each.
		std::vector<node_index_t> indices;
		std::vector<kriging_weight_t> weights;
		std::vector<mean_t> means;
		std::vector<value_t> values;
		std::vector<coord_t> coords;
		coord_t node_coord;
		
		nl.find(index, defineds, node_coord, indices, coords);
		if (indices.size() <= 0)
			return ki_result_t::KI_NO_NEIGHBOURS;

		select(mp, indices, means);
		bool success = wc(node_coord, mp[index], coords, means, cov, weights);		
		if (success)
		{			
			select(input_values, indices, values);					
			result = combine<value_t, result_t>(values, weights, means, mp[index]);

			return ki_result_t::KI_SUCCESS;
		}
		else
		{			
			return ki_result_t::KI_SINGULARITY;
		}		
	}

	template<		
		typename values_t,
		typename defineds_t,
		typename means_t,
		typename covariances_t,
		typename neighbourhood_lookup_t,
		typename weight_calculator_t,
		typename result_t>
	ki_result_t kriging_interpolation(
			const values_t & input_values,
			const defineds_t & defineds,
			node_index_t index,			
			const covariances_t & cov,
			const means_t & mp,
			const neighbourhood_lookup_t & nl,
			const weight_calculator_t & wc,
			result_t & result,
			double & variance
		)
	{
		typedef typename values_t::value_type value_t;
		typedef typename neighbourhood_lookup_t::coord_t coord_t;
		// Per-node allocations bounded by max_neighbours, not grid size
		std::vector<node_index_t> indices;
		std::vector<kriging_weight_t> weights;
		std::vector<mean_t> means;
		std::vector<value_t> values;
		std::vector<coord_t> coords;			
		coord_t node_coord;

		nl.find(index, defineds, node_coord, indices, coords);
		if (indices.size() <= 0)
			return ki_result_t::KI_NO_NEIGHBOURS;

		select(mp, indices, means);
		bool success = wc(node_coord, mp[index], coords, means, cov, weights, variance);		

		if (success)
		{			
			select(input_values, indices, values);					
			result = combine<value_t, result_t>(values, weights, means, mp[index]);
			return ki_result_t::KI_SUCCESS;
		}
		else
		{			
			return ki_result_t::KI_SINGULARITY;
		}		
	}

	// -----------------------------------------------------------------------
	// Workspace-aware kriging_interpolation overloads.
	// Takes a kriging_ws_t<value_t, coord_t> by reference — all internal
	// vectors reused across calls, eliminating 5 heap allocations per node.
	// -----------------------------------------------------------------------

	// Params-wrapper overload (delegates to params members)
	template<typename params_t, typename result_t, typename ws_t>
	ki_result_t kriging_interpolation_ws(
		const params_t & params,
		node_index_t index,
		result_t & result,
		ws_t & ws)
	{
		return kriging_interpolation_ws(
			*params.input_values,
			*params.defineds,
			index,
			*params.covariances,
			*params.means,
			*params.neighbour_lookup,
			*params.weight_calculator,
			result, ws);
	}

	// Non-variance overload
	template<		
		typename values_t,
		typename defineds_t,
		typename means_t,
		typename covariances_t,
		typename neighbourhood_lookup_t,
		typename weight_calculator_t,
		typename result_t>
	ki_result_t kriging_interpolation_ws(
			const values_t & input_values,
			const defineds_t & defineds,
			node_index_t index,			
			const covariances_t & cov,
			const means_t & mp,
			const neighbourhood_lookup_t & nl,
			const weight_calculator_t & wc,
			result_t & result,
			kriging_ws_t<typename values_t::value_type, typename neighbourhood_lookup_t::coord_t> & ws
		)
	{
		typedef typename values_t::value_type value_t;

		nl.find(index, defineds, ws.node_coord, ws.indices, ws.coords);
		if (ws.indices.size() <= 0)
			return ki_result_t::KI_NO_NEIGHBOURS;

		select(mp, ws.indices, ws.means);
		bool success = wc(ws.node_coord, mp[index], ws.coords, ws.means, cov, ws.weights, ws.wcalc);		
		if (success)
		{			
			select(input_values, ws.indices, ws.values);					
			result = combine<value_t, result_t>(ws.values, ws.weights, ws.means, mp[index]);
			return ki_result_t::KI_SUCCESS;
		}
		else
		{			
			return ki_result_t::KI_SINGULARITY;
		}		
	}

	// Variance-enabled workspace-aware overload
	template<		
		typename values_t,
		typename defineds_t,
		typename means_t,
		typename covariances_t,
		typename neighbourhood_lookup_t,
		typename weight_calculator_t,
		typename result_t>
	ki_result_t kriging_interpolation_ws(
			const values_t & input_values,
			const defineds_t & defineds,
			node_index_t index,			
			const covariances_t & cov,
			const means_t & mp,
			const neighbourhood_lookup_t & nl,
			const weight_calculator_t & wc,
			result_t & result,
			double & variance,
			kriging_ws_t<typename values_t::value_type, typename neighbourhood_lookup_t::coord_t> & ws
		)
	{
		typedef typename values_t::value_type value_t;

		nl.find(index, defineds, ws.node_coord, ws.indices, ws.coords);
		if (ws.indices.size() <= 0)
			return ki_result_t::KI_NO_NEIGHBOURS;

		select(mp, ws.indices, ws.means);
		bool success = wc(ws.node_coord, mp[index], ws.coords, ws.means, cov, ws.weights, variance, ws.wcalc);		
		if (success)
		{			
			select(input_values, ws.indices, ws.values);					
			result = combine<value_t, result_t>(ws.values, ws.weights, ws.means, mp[index]);
			return ki_result_t::KI_SUCCESS;
		}
		else
		{			
			return ki_result_t::KI_SINGULARITY;
		}		
	}

}

#endif //__KRIGING_INTERPOLATION_H__31DBE328_667D_49C4_A4CB_830C3FA89872
