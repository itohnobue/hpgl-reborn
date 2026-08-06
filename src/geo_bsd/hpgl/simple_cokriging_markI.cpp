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
#include "precalculated_covariance.h"
#include "api.h"

namespace hpgl
{

/// Per-node workspace for the cokriging loop (2-M-9).  Cokriging Mark I/II
/// was the only kriging family with no workspace reuse — calc_value /
/// calc_weights / solve_system / build_system allocated ~7 heap vectors per
/// node (coords, indices, weights, values, A, b, A_backup), all sequential
/// (no OpenMP).  This workspace mirrors the kriging_ws_t pattern used by the
/// other families: vectors are allocated once per loop and reused via
/// resize() (allocation-free when capacity is sufficient).  resize() never
/// shrinks capacity, so memory stays bounded by the largest neighbourhood
/// seen; correctness is unchanged (find() clears indices/coords itself).
template<typename coord_t>
struct cokriging_ws_t {
    std::vector<node_index_t> indices;
    std::vector<coord_t> coords;
    std::vector<kriging_weight_t> weights;
    std::vector<cont_value_t> values;
    std::vector<double> A;
    std::vector<double> b;
    std::vector<double> A_backup;
};

template<typename coord_t, typename primary_cov_model_t, typename cross_cov_model_t>
bool build_system(	
	const coord_t & center, 
	const std::vector<coord_t> & coords,
	const primary_cov_model_t & primary_cov,
	const cross_cov_model_t & cross_cov,
	double secondary_variance,
	bool secondary_present,
	cokriging_ws_t<coord_t> & ws)
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
	// E-M85: the bound is aligned with the lowered C-API cap
	// (MAX_NEIGHBOURS_UPPER_BOUND = 10000, api.cpp) plus the secondary-
	// equation +1 edge (ms = neighbour_count + 1) — the neighbour lookup
	// caps the effective count at min(max_neighbours, NEIGHBOUR_WORK_CAP)
	// = 10000, so ms <= 10001 on every reachable path; this guard rejects
	// direct-C++ configs beyond that (previously the bound admitted
	// 100000-neighbour systems → 80-240 GB per-node matrices).
	if (ms > 10001)
	{
		std::ostringstream oss;
		oss << "cokriging system size " << ms
		    << " exceeds upper bound (10000) in build_system";
		throw hpgl_exception("simple_cokriging_markI", oss.str());
	}
	// 2-M-9: reuse workspace buffers instead of allocating per node.
	ws.A.resize(matrix_elements);
	ws.b.resize(matrix_size);

	for (int i = 0; i < neighbour_count; ++i)
	{
		for (int j = i; j < neighbour_count; ++j)
		{
			// I2-23: size_t indexing — i*matrix_size overflows signed int
			// for >46340 neighbours (heap OOB write). Ported from
			// my_kriging_weights.h's safe indexing pattern.
			ws.A[static_cast<size_t>(i) * ms + j] = primary_cov(coords[i], coords[j]);
			ws.A[static_cast<size_t>(j) * ms + i] = ws.A[static_cast<size_t>(i) * ms + j];
		}
	}
	
	if (secondary_present)
	{
		for (int i = 0; i < neighbour_count; ++i)
		{
			ws.A[static_cast<size_t>(neighbour_count) * ms + i] = cross_cov(center, coords[i]);
			ws.A[static_cast<size_t>(i) * ms + neighbour_count] = ws.A[static_cast<size_t>(neighbour_count) * ms + i];
			ws.b[i] = primary_cov(coords[i], center);
		}

		ws.b[neighbour_count] = cross_cov(coord_t(0,0,0), coord_t(0,0,0));

		ws.A[static_cast<size_t>(ms) * ms - 1] = secondary_variance;
	}
	else
	{
		for (int i = 0; i < neighbour_count; ++i)
		{
			ws.b[i] = primary_cov(coords[i], center);
		}
	}

	return true;
}

template<typename coord_t>
bool solve_system(cokriging_ws_t<coord_t> & ws, std::vector<kriging_weight_t> & weights)
{
	int size = static_cast<int>(ws.b.size());
	weights.resize(static_cast<size_t>(size));

#ifdef LAPACK_SOLVER
	// Unified SPD solver: backup → dpotrf_ → (on fail) gauss_solve →
	// (on success) dpotrs_.  Replaces the HPGL-only cholesky_decomposition/
	// cholesky_solve with LAPACK-accelerated Cholesky + gauss_solve fallback.
	// 2-M-9: A_backup reused from the workspace instead of re-allocated.
	ws.A_backup.resize(ws.A.size());
	std::copy(ws.A.begin(), ws.A.end(), ws.A_backup.begin());

	return detail::lapack_spd_solve_1rhs(
		&ws.A[0], size, &weights[0], &ws.b[0],
		&ws.A_backup[0], "Cokriging SPD solve");
#else // HPGL_SOLVER
	// Original HPGL internal solver (fallback for non-LAPACK builds)
	size_t sz = static_cast<size_t>(size);
	std::vector<double> A_U(sz * sz, 0.0);
	std::vector<double> A_L(sz * sz, 0.0);

	bool solved = cholesky_decomposition(&ws.A[0], &A_U[0], &A_L[0], size);
	if (!solved) {
		return gauss_solve(&ws.A[0], &ws.b[0], &weights[0], size);
	}
	cholesky_solve(&A_L[0], &A_U[0], &ws.b[0], &weights[0], size);
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
		std::vector<kriging_weight_t> & weights,
		cokriging_ws_t<coord_t> & ws
		)
{
	// 2-M-9: build/solve reuse the workspace buffers.
	// build system
	if (!build_system(center, coords, primary_cov, cross_cov, secondary_variance, secondary_present, ws))
		return false;
	// solve systemk
	return solve_system(ws, weights);
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
		cont_value_t & result,
		cokriging_ws_t<typename n_lookup_t::coord_t> & ws)
{
	typedef typename n_lookup_t::coord_t coord_t;
	coord_t center;

	// 2-M-9: reuse workspace vectors — find() clears indices/coords itself,
	// and weights/values are resized by the callees below, so no per-node
	// heap allocations occur on the hot path.
	n_lookup.find(node, is_informed_predicate_t<data_t>(primary_data), center, ws.indices, ws.coords);

	if (ws.indices.size() <= 0)
		return ki_result_t::KI_NO_NEIGHBOURS;

	if (!calc_weights(center, ws.coords, primary_cov, cross_cov, secondary_variance, secondary_present, ws.weights, ws))
		return ki_result_t::KI_SINGULARITY;

	select(primary_data, ws.indices, ws.values);
	
	if (!combine(ws.values, ws.weights, primary_mean, secondary_value, secondary_mean, secondary_present, result))
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
		// F-20: `cov_at_zero <= 0 || d2 <= 0` is NaN-bypassable — NaN fails
		// both comparisons, so a NaN cov_at_zero or d2 reaches
		// p12*sqrt(d2)/sqrt(cov_at_zero) → NaN m_coef. isfinite first.
		if (!std::isfinite(cov_at_zero) || !std::isfinite(d2) || cov_at_zero <= 0 || d2 <= 0)
		{
			LOGWARNING("Mark I cross-covariance degraded: cov_at_zero or d2 non-positive/non-finite, setting coef=0.\n");
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
		// F-20: the guard previously checked secondary_variance only and used
		// a NaN-bypassable `<= 0` comparison. NaN primary_variance or
		// secondary_variance passes `<= 0` and reaches
		// p12*sqrt(pv/sv) → NaN m_coef. Also add the missing primary_variance
		// guard — sqrt of a negative/NaN primary_variance would be NaN.
		if (!std::isfinite(primary_variance) || !std::isfinite(secondary_variance)
			|| primary_variance <= 0 || secondary_variance <= 0)
		{
			LOGWARNING("Mark II cross-covariance degraded: primary/secondary variance non-positive or non-finite, setting coef=0.\n");
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
	// 2-M-9: one workspace for the whole loop — eliminates ~7 heap
	// allocations per node. Parallelism is deliberately NOT added here:
	// the plain neighbour lookup's find() reads the live informed mask and
	// the loop is shared with the stats/report bookkeeping; converting it to
	// OpenMP would need per-thread workspaces and reduction rework, which
	// is out of scope for this correctness-neutral allocation-churn fix.
	cokriging_ws_t<typename n_lookup_t::coord_t> ws;
	// II-10: the secondary equation can only enter the kriging system when
	// the secondary variance is a strictly-positive finite value. d2=0 is
	// Python-ACCEPTED (validation.py:962 rejects only < 0) but writes a raw
	// 0 to the diagonal (build_system line ~117) → singular matrix every
	// node → KI_SINGULARITY → mean-fill, silent for C-API. A NaN
	// secondary_variance bypasses the old `<= 0` ctor guard the same way.
	// When the variance is not usable, DROP the secondary equation entirely
	// (primary-only kriging — the F-22 convention for a missing secondary).
	// This single flag propagates through calc_value → calc_weights →
	// build_system → combine (all consume the same secondary_present flag);
	// no signature changes.
	bool secondary_variance_valid = std::isfinite(secondary_variance) && secondary_variance > 0;

	// E-M60: the no-neighbour/singularity fallback previously used an
	// implicit secondary weight of 1.0 (primary_mean + secondary_value -
	// secondary_mean), which overstates secondary influence for rho < 1 at
	// exactly the failure nodes the fallback is meant to rescue. The BLUP
	// secondary-only weight is Csp(0)/Css(0) = rho*sig_p/sig_s — the same
	// value the code's own m_coef machinery derives (cross_cov_model_mark_i_t
	// m_coef = rho*sig_s/sig_p at :275; markII same). Compute it once
	// (stationary models): cross_cov at zero lag is the cross-covariance
	// Csp(0); dividing by the secondary variance Css(0) yields the BLUP
	// weight. When the secondary variance is not usable (II-10 above),
	// secondary_present is forced false and secondary_value ==
	// secondary_mean, so the term is zero regardless — guard the division.
	double secondary_weight = 0.0;
	if (secondary_variance_valid)
	{
		typedef typename n_lookup_t::coord_t coord_t;
		const double cross_cov_zero = cross_cov(coord_t(0,0,0), coord_t(0,0,0));
		if (std::isfinite(cross_cov_zero))
			secondary_weight = cross_cov_zero / secondary_variance;
	}

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
			// II-10: additionally require the secondary variance to be a
			// strictly-positive finite value, else the secondary equation
			// is dropped (primary-only kriging, GSLIB convention).
			bool secondary_present = secondary_data.is_informed(i) && secondary_variance_valid;
			cont_value_t secondary_value = secondary_present
				? secondary_data[i] : secondary_mean;
			ki_result_t ki_result = calc_value(i, input_prop, secondary_value,
				secondary_present, primary_mean, secondary_mean,
				secondary_variance, primary_cov, cross_cov, n_lookup, result, ws);
			switch (ki_result)
			{
			case ki_result_t::KI_SUCCESS:
				++points_calculated;
				++points_processed;
				sum += result;
				break;
			case ki_result_t::KI_NO_NEIGHBOURS:
				++points_without_neighbours;
				// E-M60: BLUP secondary-only weight (see above) — the
				// previous implicit weight 1.0 overstated the secondary
				// influence for rho < 1.
				result = primary_mean + secondary_weight * (secondary_value - secondary_mean);
				++points_processed;
				sum += result;
				break;
			case ki_result_t::KI_SINGULARITY:
				++points_singularity;
				// E-M60: same BLUP secondary-only weight as the
				// no-neighbours branch.
				result = primary_mean + secondary_weight * (secondary_value - secondary_mean);
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

	// E-M61: the C++ entries previously validated only input==output size.
	// A shorter secondary_data silently degraded (bounds-guarded
	// is_informed → primary-only kriging with no error) and a smaller grid
	// produced unchecked sugarbox_grid arithmetic (wrong neighbourhoods /
	// mean-fill). Validate the full size contract at the C++ chokepoint —
	// the C API gates these at api.cpp:1675-1706/1818-1849, but direct-C++
	// callers bypass them.
	if (secondary_data.size() != input_prop.size())
	{
		std::ostringstream oss;
		oss << "Secondary data size: " << secondary_data.size() << ". Input data size: " << input_prop.size() << ". Must be equal.";
		throw hpgl_exception("simple_cokriging", oss.str());
	}
	if (grid.size() != input_prop.size())
	{
		std::ostringstream oss;
		oss << "Grid size: " << grid.size() << ". Input data size: " << input_prop.size() << ". Must be equal.";
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
	// F-20: comparison-only guards are NaN-bypassable — IEEE-754 guarantees
	// every comparison with NaN is false, so a NaN correlation_coef passes
	// `< -1.0 || > 1.0` unmodified and flows into cross_cov_model_mark_i_t
	// where it produces a NaN m_coef → NaN covariance matrix → silent
	// mean-fill. isfinite must be checked FIRST.
	if (!std::isfinite(correlation_coef) || correlation_coef < -1.0 || correlation_coef > 1.0)
		throw hpgl_exception("simple_cokriging_markI",
			"correlation_coef must be finite and in [-1, 1]");

	// F-20/II-06: means and variance are consumed directly by the kernel
	// (process_node_loop → combine, build_system diagonal). A NaN/Inf value
	// silently produces NaN output for direct C++ callers (the C API gates
	// these at api.cpp, but the C++ entry point is the single chokepoint for
	// direct-C++ + C callers alike). 0.0 variance stays ACCEPTED (Python
	// validation.py:962 rejects only < 0; test_zero_variance_passes pins 0.0
	// as valid — the d2=0 case is handled by the II-10 secondary-validity flag
	// as primary-only kriging, not rejected here).
	if (!std::isfinite(secondary_variance) || secondary_variance < 0.0)
		throw hpgl_exception("simple_cokriging_markI",
			"secondary_variance must be finite and >= 0");
	if (!std::isfinite(primary_mean) || !std::isfinite(secondary_mean))
		throw hpgl_exception("simple_cokriging_markI",
			"primary_mean and secondary_mean must be finite");

	// E-M86: precalculate the covariances (mirror of the
	// precalculated_covariances_t pattern used by SK/LVM/SGS/IK) —
	// build_system previously evaluated the cov model from scratch per
	// matrix element (primary block O(n^2) evals plus a second full eval
	// of the same symmetric value on the RHS; cross block 2x per element
	// via the raw model). The precalculated table turns per-element evals
	// into table lookups inside the search box with exact-model fallback
	// beyond it (F-21 semantics preserved); the markI cross model scales
	// the same table by its coefficient.
	cov_model_t cov_model(primary_cov_params);
	precalculated_covariances_t cov(cov_model, neighbourhood_params.m_radiuses);

	cross_cov_model_mark_i_t<precalculated_covariances_t> cross_cov(correlation_coef, secondary_variance, &cov);

	int data_size = input_prop.size();
	
	neighbour_lookup_t<sugarbox_grid_t, precalculated_covariances_t> n_lookup(&grid, &cov, neighbourhood_params);

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

	// E-M61: full size contract at the C++ chokepoint (sibling of the
	// markI gate above) — a shorter secondary_data or smaller grid
	// silently degraded for direct-C++ callers.
	if (secondary_data.size() != input_prop.size())
	{
		std::ostringstream oss;
		oss << "Secondary data size: " << secondary_data.size() << ". Input data size: " << input_prop.size() << ". Must be equal.";
		throw hpgl_exception("simple_cokriging", oss.str());
	}
	if (grid.size() != input_prop.size())
	{
		std::ostringstream oss;
		oss << "Grid size: " << grid.size() << ". Input data size: " << input_prop.size() << ". Must be equal.";
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
	// F-20: comparison-only guards are NaN-bypassable — NaN passes
	// `< -1.0 || > 1.0` unmodified. isfinite must be checked FIRST.
	if (!std::isfinite(correlation_coef) || correlation_coef < -1.0 || correlation_coef > 1.0)
		throw hpgl_exception("simple_cokriging_markII",
			"correlation_coef must be finite and in [-1, 1]");

	// F-20/II-06: means are consumed directly by the kernel. A NaN/Inf mean
	// silently produces NaN output for direct C++ callers. markII's
	// secondary_variance is cov-model derived at the call below (sill at
	// zero lag) and is already validated by covariance_param_t::validate
	// (sill > 0 finite); the cross_cov ctor guard covers it as
	// defense-in-depth.
	if (!std::isfinite(primary_mean) || !std::isfinite(secondary_mean))
		throw hpgl_exception("simple_cokriging_markII",
			"primary_mean and secondary_mean must be finite");

	// E-M86: precalculate both covariance fields (mirror of the
	// precalculated_covariances_t pattern used by SK/LVM/SGS/IK) — the
	// per-element model evals in build_system become table lookups; the
	// markII cross model scales the secondary table by its coefficient.
	// The zero-lag evals below (secondary variance, ctor variances) hit
	// the in-box table entries and return the same sill values as the
	// raw models.
	cov_model_t primary_cov_model(primary_cov_params);
	cov_model_t secondary_cov_model(secondary_cov_params);
	precalculated_covariances_t primary_cov(primary_cov_model, neighbourhood_params.m_radiuses);
	precalculated_covariances_t secondary_cov(secondary_cov_model, neighbourhood_params.m_radiuses);
	double secondary_variance = secondary_cov(sugarbox_grid_t::coord_t(0,0,0), sugarbox_grid_t::coord_t(0,0,0));

	cross_cov_model_mark_ii_t<precalculated_covariances_t, precalculated_covariances_t> cross_cov(correlation_coef, &primary_cov, &secondary_cov);

	int data_size = input_prop.size();
	
	neighbour_lookup_t<sugarbox_grid_t, precalculated_covariances_t> n_lookup(&grid, &primary_cov, neighbourhood_params);

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
