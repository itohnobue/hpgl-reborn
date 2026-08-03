#ifndef CDF_UTILS_H_INCLUDED_ASDJASLD2134791287391KAJSHDKLASHD298237912837HIUREYR8732RH3R23
#define CDF_UTILS_H_INCLUDED_ASDJASLD2134791287391KAJSHDKLASHD298237912837HIUREYR8732RH3R23

#include "typedefs.h"
#include "property_array.h"
#include <cmath>

namespace hpgl
{
	namespace detail
	{
		// NaN-safe [0,1] sanitization for kriged indicator probabilities
		// (F-37, II-11, II-12). SK weights are unconstrained — poorly
		// conditioned matrices, sparse neighbours, or extreme anisotropy can
		// push the combine() result outside [0,1], and NaN bypasses the
		// relational clamps (NaN < 0.0 and NaN > 1.0 are both false).
		//
		// Non-finite input falls back to the caller's (validated) marginal
		// probability — mirroring the KI-failure fallback used by SIS/IK
		// ("prob = marginal_probs[idx][node]") — then clamps to [0,1]. The
		// fallback itself is clamped to [0,1] as well, so a direct-C caller
		// that smuggles a non-finite marginal cannot re-introduce NaN.
		inline double sanitize_probability(double prob, double fallback)
		{
			if (!std::isfinite(prob))
				prob = fallback;
			if (!std::isfinite(prob) || prob < 0.0)
				prob = 0.0;
			else if (prob > 1.0)
				prob = 1.0;
			return prob;
		}
	}

	indicator_index_t 
	most_probable_category(
		const std::vector<indicator_probability_t> & probs);

	template <typename prop_t, typename Cdf1, typename Cdf2>
	void transform_cdf_p(prop_t & property, Cdf1 from, Cdf2 to)
	{	
		for (int idx = 0, end_idx = property.size(); idx < end_idx; ++idx)
		{
			if (property.is_informed(idx))
			{
				double P = from.prob(property.get_at(idx));
				double value = to.inverse(P);
				property.set_at(idx, value);
			}
		}	
	}

	template <typename T, typename Cdf1, typename Cdf2>
	T transform_cdf_s(T value, Cdf1 from, Cdf2 to)
	{	
		return to.inverse(from.prob(value));			
	}

	template <typename T, typename Cdf1, typename Cdf2>
	void transform_cdf_v(const std::vector<T> & in, std::vector<T> & out, Cdf1 from, Cdf2 to)
	{
		out.resize(in.size());
		for (size_t idx = 0, end_idx = in.size(); idx < end_idx; ++idx)
		{
			out[idx] = to.inverse(from.prob(in[idx]));
		}
	}	

	template <typename T, typename Cdf1, typename Cdf2>
	void transform_cdf_ptr(const T * in, std::vector<T> & out, Cdf1 from, Cdf2 to)
	{
		for (size_t idx = 0, end_idx = out.size(); idx < end_idx; ++idx)
		{
			out[idx] = to.inverse(from.prob(in[idx]));
		}
	}	


} //namespace hpgl

#endif // CDF_UTILS_H_INCLUDED_ASDJASLD2134791287391KAJSHDKLASHD298237912837HIUREYR8732RH3R23
