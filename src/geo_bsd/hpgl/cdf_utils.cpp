#include "stdafx.h"

#include "cdf_utils.h"
#include "hpgl_exception.h"
#include <cstdio>
#include <cstdlib>

namespace hpgl
{

indicator_index_t most_probable_category(const std::vector<indicator_probability_t> & cdf)
{
	size_t size = cdf.size();
	if (size == 0) { fprintf(stderr, "HPGL FATAL: most_probable_category: empty vector\n"); abort(); }
	if (size == 1) return 0;

	// Compute probability mass function (PMF) from cumulative CDF.
	// For K categories with cumulative indicators P_0, P_1, ..., P_{K-1},
	// the category probabilities are:
	//   pmf[0] = P_0, pmf[k] = P_k - P_{k-1} for k = 1..K-1
	// Select the category with the maximum probability mass.
	double max_mass = cdf[0];
	indicator_index_t max_idx = 0;
	for (indicator_index_t i = 1; i < static_cast<indicator_index_t>(size); i++)
	{
		double mass = cdf[i] - cdf[i - 1];
		// Clamp negative mass (can arise from imperfect order relations correction)
		if (mass < 0.0) mass = 0.0;
		if (mass > max_mass)
		{
			max_mass = mass;
			max_idx = i;
		}
	}
	// Also check if the first category (P_0) is still the max
	// (max_mass was initialized to cdf[0], max_idx to 0)
	return max_idx;
}


} //namespace hpgl
