#include "stdafx.h"

#include "sample.h"
#include <cmath>

namespace hpgl
{
	indicator_index_t
	sample(mt_random_generator_t & gen, const std::vector<indicator_probability_t> & probs)
	{
		double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
		// NaN guard: IEEE 754 NaN < X is always false, so NaN
		// probabilities or a NaN sum would silently bias toward the
		// last category (all comparisons fail). Return first category
		// as a safe fallback rather than an arbitrary index.
		if (std::isnan(sum))
			return 0;
		double s = gen() * sum;
		double p = 0;
		for (size_t i = 0, end_i = probs.size(); i < end_i; ++i)
		{
			p += probs[i];
			if (s < p)
				return i;
		}
		return probs.size() - 1;
	}
}