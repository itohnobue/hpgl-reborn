#include "stdafx.h"

#include "typedefs.h"
#include "property_array.h"
#include "hpgl_exception.h"
#include "sugarbox_grid.h"
#include "sugarbox_neighbour_lookup.h"
#include "covariance_param.h"
#include "is_informed_predicate.h"
#include "select.h"
#include "pretty_printer.h"
#include "progress_reporter.h"
#include "cov_model.h"
#include "gauss_solver.h"
#include "my_kriging_weights.h"
#include "kriging_interpolation.h"
#include "kriging_stats.h"
#include "api.h"

namespace hpgl
{

template<typename coord_t, typename primary_cov_model_t, typename cross_cov_model_t>
bool build_system(	
	const coord_t & center, 
	const std::vector<coord_t> & coords,
	const primary_cov_model_t & primary_cov,
	const cross_cov_model_t & cross_cov,
	double secondary_variance,
	bool secondary_present,
	std::vector<double> & A,
	std::vector<double> & b)
{
	// Int-range validation: coords.size() must fit in int
	if (coords.size() > static_cast<size_t>(std::numeric_limits<int>::max() - 1))
	{
		HPGL_LOG_STRING("Security: Coordinate count exceeds int max in build_system.");
		return false;
	}

	int neighbour_count = static_cast<int>(coords.size());
	// F-22: when the secondary is missing/undefined at the target, drop the
	// secondary equation from the system entirely (GSLIB convention) — the
	// system becomes plain primary-only kriging with matrix_size ==
	// neighbour_count. Substituting secondary_mean while keeping the
	// full-variance secondary equation produces an estimate that is not BLUP.
	int matrix_size = neighbour_count + (secondary_present ? 1 : 0);

	size_t ms = static_cast<size_t>(matrix_size);
	size_t matrix_elements = 0;
	if (!detail::safe_multiply_size_t(ms, ms, matrix_elements))
	{
		HPGL_LOG_STRING("Security: Matrix size overflow in build_system.");
		return false;
	}
	// I2-23: hard upper bound on the cokriging system size. The linear
	// solver consumes O(size^3); sizes beyond this are pathological configs.
	// PR-07: previously returned false → calc_value mapped it to
	// KI_SINGULARITY → process_node_loop silently mean-filled. Now a clear
	// error so the caller observes the failure instead of silent wrong output.
	// The C-API bound (validate_max_neighbours_or_throw, api.cpp) is aligned
	// at the same limit; this guard is defense in depth for direct C++ callers
	// and the secondary-equation +1 edge (ms = neighbour_count + 1).
	if (ms > 100000)
	{
		std::ostringstream oss;
		oss << "cokriging system size " << ms
		    << " exceeds upper bound (100000) in build_system";
		throw hpgl_exception("simple_cokriging_markI", oss.str());
	}
	A.resize(matrix_elements);

	b.resize(matrix_size);

	for (int i = 0; i < neighbour_count; ++i)
	{
		for (int j = i; j < neighbour_count; ++j)
		{
			// I2-23: size_t indexing — i*matrix_size overflows signed int
			// for >46340 neighbours (heap OOB write). Ported from
			// my_kriging_weights.h's safe indexing pattern.
			A[static_cast<size_t>(i) * ms + j] = primary_cov(coords[i], coords[j]);
			A[static_cast<size_t>(j) * ms + i] = A[static_cast<size_t>(i) * ms + j];
		}
	}
	
	if (secondary_present)
	{
		for (int i = 0; i < neighbour_count; ++i)
		{
			A[static_cast<size_t>(neighbour_count) * ms + i] = cross_cov(center, coords[i]);
			A[static_cast<size_t>(i) * ms + neighbour_count] = A[static_cast<size_t>(neighbour_count) * ms + i];
			b[i] = primary_cov(coords[i], center);
		}

		b[neighbour_count] = cross_cov(coord_t(0,0,0), coord_t(0,0,0));

		A[static_cast<size_t>(ms) * ms - 1] = secondary_variance;
	}
	else
	{
		for (int i = 0; i < neighbour_count; ++i)
		{
			b[i] = primary_cov(coords[i], center);
		}
	}

	return true;
}

bool solve_system(std::vector<double> & A, std::vector<double> & b, std::vector<kriging_weight_t> & weights)
{
	int size = static_cast<int>(b.size());
	weights.resize(static_cast<size_t>(size));

#ifdef LAPACK_SOLVER
	// Unified SPD solver: backup → dpotrf_ → (on fail) gauss_solve →
	// (on success) dpotrs_.  Replaces the HPGL-only cholesky_decomposition/
	// cholesky_solve with LAPACK-accelerated Cholesky + gauss_solve fallback.
	std::vector<double> A_backup(A);

	return detail::lapack_spd_solve_1rhs(
		&A[0], size, &weights[0], &b[0],
		&A_backup[0], "Cokriging SPD solve");
#else // HPGL_SOLVER
	// Original HPGL internal solver (fallback for non-LAPACK builds)
	size_t sz = static_cast<size_t>(size);
	std::vector<double> A_U(sz * sz, 0.0);
	std::vector<double> A_L(sz * sz, 0.0);

	bool solved = cholesky_decomposition(&A[0], &A_U[0], &A_L[0], size);
	if (!solved) {
		return gauss_solve(&A[0], &b[0], &weights[0], size);
	}
	cholesky_solve(&A_L[0], &A_U[0], &b[0], &weights[0], size);
	return true;
#endif
}

template<typename coord_t, typename primary_cov_model_t, typename cross_cov_model_t>
bool calc_weights(		
		const coord_t & center,
		std::vector<coord_t> & coords,
		const primary_cov_model_t & primary_cov,
		const cross_cov_model_t & cross_cov,
		double secondary_variance,
		bool secondary_present,
		std::vector<kriging_weight_t> & weights
		)
{
	std::vector<double> A; 
	std::vector<double> b;
	
	// build system
	if (!build_system(center, coords, primary_cov, cross_cov, secondary_variance, secondary_present, A, b))
		return false;
	// solve systemk
	return solve_system(A, b, weights);
}

bool combine(
		const std::vector<cont_value_t> & values,
		const std::vector<kriging_weight_t> & weights,
		mean_t primary_mean,
		cont_value_t secondary_value,
		mean_t secondary_mean,
		bool secondary_present,
		cont_value_t & result
	    )
{
	// F-22: when the secondary equation was dropped (secondary missing at
	// the target), weights.size() == values.size(); otherwise the secondary
	// weight is the last element.
	size_t expected_weights = values.size() + (secondary_present ? 1u : 0u);
	if (weights.size() != expected_weights)
	{
		std::ostringstream oss;
		oss << "values.size() = " << values.size() << ", weights.size() = " << weights.size()
		    << ". weights.size() != values.size()" << (secondary_present ? " + 1" : "");
		throw hpgl_exception("combine", oss.str());
	}

	int size = values.size();
	result = primary_mean;
	for (int i = 0; i < size; ++i)
	{
		result += (values[i] - primary_mean) * weights[i];
	}
	if (secondary_present)
		result += weights[size] * (secondary_value - secondary_mean);
	return true;
}

template<typename data_t, typename primary_cov_model_t, typename cross_cov_model_t, typename n_lookup_t>
ki_result_t calc_value(
		node_index_t node,
		const data_t & primary_data,
		cont_value_t secondary_value,
		bool secondary_present,
		mean_t primary_mean,
		mean_t secondary_mean,
		double secondary_variance,
		const primary_cov_model_t & primary_cov,
	       	const cross_cov_model_t & cross_cov,
		const n_lookup_t & n_lookup,
		cont_value_t & result)
{
	typedef typename n_lookup_t::coord_t coord_t;
	std::vector<coord_t> coords;
	std::vector<node_index_t> indices;
	coord_t  center;
	
	n_lookup.find(node, is_informed_predicate_t<data_t>(primary_data), center, indices, coords);

	if (indices.size() <= 0)
		return ki_result_t::KI_NO_NEIGHBOURS;

	std::vector<kriging_weight_t> weights;
	if (!calc_weights(center, coords, primary_cov, cross_cov, secondary_variance, secondary_present, weights))
		return ki_result_t::KI_SINGULARITY;

	std::vector<cont_value_t> values;
	select(primary_data, indices, values);
	
	if (!combine(values, weights, primary_mean, secondary_value, secondary_mean, secondary_present, result))
		return ki_result_t::KI_SINGULARITY;

	
	return ki_result_t::KI_SUCCESS;
}

template<typename cov_model_t>
class cross_cov_model_mark_i_t
{
	double m_coef;
	cov_model_t * m_cov_model;
public:
	cross_cov_model_mark_i_t(double p12, double d2, cov_model_t * cov_model)
		: m_cov_model(cov_model)
	{
		double cov_at_zero = (*cov_model)(coord_t(0,0,0), coord_t(0,0,0));
		// Guard: d2 must be strictly positive for sqrt().
		// cov_at_zero must also be positive.
		if (cov_at_zero <= 0 || d2 <= 0)
		{
			LOGWARNING("Mark I cross-covariance degraded: cov_at_zero or d2 non-positive, setting coef=0.\n");
			m_coef = 0;
		}
		else
		{
			m_coef = p12 * sqrt(d2) / sqrt(cov_at_zero);
		}
	}

	cross_cov_model_mark_i_t(const cross_cov_model_mark_i_t&) = delete;
	cross_cov_model_mark_i_t& operator=(const cross_cov_model_mark_i_t&) = delete;
	cross_cov_model_mark_i_t(cross_cov_model_mark_i_t&&) = delete;
	cross_cov_model_mark_i_t& operator=(cross_cov_model_mark_i_t&&) = delete;

	template<typename coord1_t, typename coord2_t>
	covariance_t operator()(const coord1_t & p1, const coord2_t & p2)const
	{
		return m_coef * (*m_cov_model)(p1, p2);
	}	
};

template<typename primary_cov_model_t, typename secondary_cov_model_t>
class cross_cov_model_mark_ii_t
{
	double m_coef;	
	secondary_cov_model_t * m_secondary_cov_model;
public:

	cross_cov_model_mark_ii_t(double p12, primary_cov_model_t * primary_cov_model, secondary_cov_model_t * secondary_cov_model)
		: m_secondary_cov_model(secondary_cov_model)
	{
		double primary_variance = (*primary_cov_model)(coord_t(0,0,0), coord_t(0,0,0));
		double secondary_variance = (*secondary_cov_model)(coord_t(0,0,0), coord_t(0,0,0));
		if (secondary_variance <= 0)
		{
			LOGWARNING("Mark II cross-covariance degraded: secondary_variance non-positive, setting coef=0.\n");
			m_coef = 0;
		}
		else
		{
			m_coef = p12 * sqrt( primary_variance / secondary_variance );
		}
	}

	cross_cov_model_mark_ii_t(const cross_cov_model_mark_ii_t&) = delete;
	cross_cov_model_mark_ii_t& operator=(const cross_cov_model_mark_ii_t&) = delete;
	cross_cov_model_mark_ii_t(cross_cov_model_mark_ii_t&&) = delete;
	cross_cov_model_mark_ii_t& operator=(cross_cov_model_mark_ii_t&&) = delete;

	template<typename coord1_t, typename coord2_t>
	covariance_t operator()(const coord1_t & p1, const coord2_t & p2)const
	{
		return m_coef * (*m_secondary_cov_model)(p1, p2);
	}
	
};

/// Deduplicated node-processing loop shared by markI and markII.
/// Previously copy-pasted identically in both functions (only the cov model
/// type differed — the template parameter absorbs the difference).
template<typename data_t, typename primary_cov_model_t, typename cross_cov_model_t, typename n_lookup_t>
static void process_node_loop(
		const data_t & input_prop,
		const data_t & secondary_data,
		mean_t primary_mean,
		mean_t secondary_mean,
		double secondary_variance,
		const primary_cov_model_t & primary_cov,
		const cross_cov_model_t & cross_cov,
		const n_lookup_t & n_lookup,
		data_t & output_prop,
		progress_reporter_t & report,
		kriging_stats_t & stats)
{
	int data_size = input_prop.size();
	unsigned long points_calculated = 0;
	unsigned long points_without_neighbours = 0;
	unsigned long points_singularity = 0;
	unsigned long points_processed = 0;
	double sum = 0;
	for (node_index_t i = 0; i < data_size; ++i)
	{
		cont_value_t result = -500;
		if (input_prop.is_informed(i))
		{
			result = input_prop[i];
		}
		else
		{
			// F-22: whether the secondary is actually defined at this node
			// decides whether the secondary equation enters the system.
			bool secondary_present = secondary_data.is_informed(i);
			cont_value_t secondary_value = secondary_present
				? secondary_data[i] : secondary_mean;
			ki_result_t ki_result = calc_value(i, input_prop, secondary_value,
				secondary_present, primary_mean, secondary_mean,
				secondary_variance, primary_cov, cross_cov, n_lookup, result);
			switch (ki_result)
			{
			case ki_result_t::KI_SUCCESS:
				++points_calculated;
				++points_processed;
				sum += result;
				break;
			case ki_result_t::KI_NO_NEIGHBOURS:
				++points_without_neighbours;
				result = primary_mean + secondary_value - secondary_mean;
				++points_processed;
				sum += result;
				break;
			case ki_result_t::KI_SINGULARITY:
				++points_singularity;
				result = primary_mean + secondary_value - secondary_mean;
				++points_processed;
				sum += result;
				break;
			}
		}
		output_prop.set_at(i, result);
		report.next_lap();
	}
	stats.m_points_calculated = points_calculated;
	stats.m_points_without_neighbours = points_without_neighbours;
	stats.m_points_singularity = points_singularity;
	stats.m_mean = points_processed > 0 ? sum / points_processed : 0;
	// m_speed_nps is set by the caller after report.stop() (the reporter
	// needs m_end, which stop() records, to compute iterations_per_second).

	// F-19: emit failure signals on all cokriging failure paths (mirrors
	// cont_kriging.h:234-241). Previously every failure silently degraded
	// to the trivial mean fallback with no observability.
	if (stats.m_points_singularity > 0 || stats.m_points_without_neighbours > 0)
	{
		fprintf(stderr,
			"HPGL: cokriging failures: %lu singularity, %lu no-neighbours (of %d total)\n",
			stats.m_points_singularity,
			stats.m_points_without_neighbours,
			data_size);
	}
}

void simple_cokriging_markI(
		const sugarbox_grid_t & grid,
		const cont_property_array_t & input_prop, 
		const cont_property_array_t & secondary_data,
		mean_t primary_mean,
		mean_t secondary_mean,
		double secondary_variance,
		double correlation_coef,
		const neighbourhood_param_t & neighbourhood_params,
		const covariance_param_t & primary_cov_params,
		cont_property_array_t & output_prop)
{
	if (input_prop.size() != output_prop.size())
	{
		std::ostringstream oss;
		oss << "Input data size: " << input_prop.size() << ". Output data size: " << output_prop.size() << ". Must be equal.";
		throw hpgl_exception("simple_cokriging", oss.str());
	}

	print_algo_name("Simple Colocated Cokriging Markov Model I");
	print_params(neighbourhood_params);
	print_params(primary_cov_params);
	print_param("Primary mean", primary_mean);
	print_param("Secondary mean", secondary_mean);
	print_param("Secondary variance", secondary_variance);
	print_param("Correllation coef", correlation_coef);

	// Range validation: correlation_coef must be in [-1, 1].
	// Python-side validation exists but the C API bypasses it.
	if (correlation_coef < -1.0 || correlation_coef > 1.0)
		throw hpgl_exception("simple_cokriging_markI",
			"correlation_coef must be in [-1, 1]");

	cov_model_t cov(primary_cov_params);	

	cross_cov_model_mark_i_t<cov_model_t> cross_cov(correlation_coef, secondary_variance, &cov);

	int data_size = input_prop.size();
	
	neighbour_lookup_t<sugarbox_grid_t, cov_model_t> n_lookup(&grid, &cov, neighbourhood_params);

	progress_reporter_t report(data_size);

	report.start(data_size);

	kriging_stats_t stats;
	process_node_loop(input_prop, secondary_data, primary_mean,
			secondary_mean, secondary_variance, cov, cross_cov,
			n_lookup, output_prop, report, stats);

	report.stop();
	stats.m_speed_nps = report.iterations_per_second();
	set_kriging_stats(stats);
}

void simple_cokriging_markII(
		const sugarbox_grid_t & grid,
		const cont_property_array_t & input_prop, 
		const cont_property_array_t & secondary_data,
		mean_t primary_mean,
		mean_t secondary_mean,		
		double correlation_coef,
		const neighbourhood_param_t & neighbourhood_params,
		const covariance_param_t & primary_cov_params,
		const covariance_param_t & secondary_cov_params,
		cont_property_array_t & output_prop)
{
	if (input_prop.size() != output_prop.size())
	{
		std::ostringstream oss;
		oss << "Input data size: " << input_prop.size() << ". Output data size: " << output_prop.size() << ". Must be equal.";
		throw hpgl_exception("simple_cokriging", oss.str());
	}

	print_algo_name("Simple Colocated Cokriging Markov Model II");
	print_params(neighbourhood_params);
	write("Primary covariation model:\n");
	print_params(primary_cov_params);
	write("Secondary covariation model:\n");
	print_params(secondary_cov_params);
	print_param("Primary mean", primary_mean);
	print_param("Secondary mean", secondary_mean);	
	print_param("Correllation coef", correlation_coef);

	// Range validation: correlation_coef must be in [-1, 1].
	// Python-side validation exists but the C API bypasses it.
	if (correlation_coef < -1.0 || correlation_coef > 1.0)
		throw hpgl_exception("simple_cokriging_markII",
			"correlation_coef must be in [-1, 1]");

	cov_model_t primary_cov(primary_cov_params);	
	cov_model_t secondary_cov(secondary_cov_params);	
	double secondary_variance = secondary_cov(sugarbox_grid_t::coord_t(0,0,0), sugarbox_grid_t::coord_t(0,0,0));

	cross_cov_model_mark_ii_t<cov_model_t, cov_model_t> cross_cov(correlation_coef, &primary_cov, &secondary_cov);

	int data_size = input_prop.size();
	
	neighbour_lookup_t<sugarbox_grid_t, cov_model_t> n_lookup(&grid, &primary_cov, neighbourhood_params);

	progress_reporter_t report(data_size);

	report.start(data_size);

	kriging_stats_t stats;
	process_node_loop(input_prop, secondary_data, primary_mean,
			secondary_mean, secondary_variance, primary_cov, cross_cov,
			n_lookup, output_prop, report, stats);

	report.stop();
	stats.m_speed_nps = report.iterations_per_second();
	set_kriging_stats(stats);
}

}
