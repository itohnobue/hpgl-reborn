#ifndef __NON_PARAMETRIC_CDF_H__6BBB60EA_BA86_4AE3_98FF_3B287C1D1522____
#define __NON_PARAMETRIC_CDF_H__6BBB60EA_BA86_4AE3_98FF_3B287C1D1522____

#include "api.h"
#include "hpgl_exception.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace hpgl
{
	template<typename value_t, typename prob_t>
	class non_parametric_cdf_t
	{		
		typedef std::vector<std::pair<value_t, prob_t> > cdf_t;
		cdf_t m_cdf;

		struct inverse_less
		{
			template<typename pair1, typename pair2>
			bool operator()(const pair1 & p1, const pair2 & p2)
			{
				return p1.second < p2.second;
			}
		};
		
	public:
		
		template <typename T>
		non_parametric_cdf_t(const T & data)
		{
			typedef std::map<value_t, int> occ_map_t;
			typedef typename occ_map_t::iterator occ_iterator_t;

			occ_map_t occ;

			// Type safety: Use size_t for size variable
			const size_t size = data.size();
			size_t count = 0;
			for (size_t i = 0; i < size; ++i)
			{
				if (data.is_informed(i))
				{
					std::pair<occ_iterator_t, bool> result = occ.insert(std::make_pair(data[i], 1));
					if (!result.second)
					{
						result.first->second += 1;
					}
					++count;
				}
			}		

			if (count <= 0)
				return;

			if (occ.empty()) { fprintf(stderr, "HPGL FATAL: non_parametric_cdf: occ is empty but count > 0\n"); abort(); }

			std::vector<std::pair<value_t, prob_t> > pdf;
			pdf.reserve(occ.size());

			for (occ_iterator_t it = occ.begin(), it_end = occ.end(); it != it_end; ++it)
			{
				pdf.push_back(std::make_pair(it->first, double(it->second) / double(count)));
			}
			std::sort(pdf.begin(), pdf.end());

			m_cdf.reserve(pdf.size());
			m_cdf.push_back(pdf[0]);
			for (size_t i = 1; i < pdf.size(); ++i)
			{				
				m_cdf.push_back(std::make_pair(pdf[i].first, pdf[i].second + m_cdf[i-1].second));
			}			
		}

		template <typename T>
		non_parametric_cdf_t(std::vector<T> & data)
		{
			typedef std::map<value_t, int> occ_map_t;
			typedef typename occ_map_t::iterator occ_iterator_t;

			occ_map_t occ;

			// Type safety: Use size_t for size variable
			const size_t size = data.size();
			size_t count = 0;
			for (size_t i = 0; i < size; ++i)
			{
					std::pair<occ_iterator_t, bool> result = occ.insert(std::make_pair(data[i], 1));
					if (!result.second)
					{
						result.first->second += 1;
					}
					++count;
				
			}		

			if (count <= 0)
				return;

			if (occ.empty()) { fprintf(stderr, "HPGL FATAL: non_parametric_cdf: occ is empty but count > 0\n"); abort(); }

			std::vector<std::pair<value_t, prob_t> > pdf;
			pdf.reserve(occ.size());

			for (occ_iterator_t it = occ.begin(), it_end = occ.end(); it != it_end; ++it)
			{
				pdf.push_back(std::make_pair(it->first, double(it->second) / double(count)));
			}
			std::sort(pdf.begin(), pdf.end());

			m_cdf.reserve(pdf.size());
			m_cdf.push_back(pdf[0]);
			for (size_t i = 1; i < pdf.size(); ++i)
			{				
				m_cdf.push_back(std::make_pair(pdf[i].first, pdf[i].second + m_cdf[i-1].second));
			}			
		}

		void dump()
		{
			std::ostringstream oss;
			oss << "Non-parametric CDF. Size: " << m_cdf.size() << "\n";
			// Type safety: Use size_t for loop index when comparing with size()
			for (size_t i = 0; i < m_cdf.size(); ++i)
			{
				oss << "(" << m_cdf[i].first << ", " << m_cdf[i].second << ");";
			}
			oss << "\n";
			write(oss.str());
		}

		prob_t prob(value_t value)
		{
			return (*this)(value);
		}
		
		prob_t operator()(value_t value)
		{
			if (m_cdf.size() <= 0)
				return 0;

			prob_t result;
			typename cdf_t::iterator it = std::lower_bound(m_cdf.begin(), m_cdf.end(), std::make_pair(value, 1.0));
			if (it == m_cdf.end())
			{
				result = m_cdf.back().second;
			}
			else if (it == m_cdf.begin())
			{
				if (value < m_cdf.front().first)
					result = 0;
				else
					result = m_cdf.front().second;
			}
			else
			{
				typename cdf_t::iterator it2 = it - 1;
				value_t x1 = it2->first;
				value_t x2 = it->first;
				prob_t y1 = it2->second;
				prob_t y2 = it->second;
				if (x2 == x1)
					result = (y1 + y2) / 2;
				else
					result = y1 + (double)(y2 - y1) / (double)(x2 - x1) * (value - x1);
			}			
			return result;
		}

		value_t inverse(prob_t prob)
		{

			if (m_cdf.size() <= 0)
				return 0;

			value_t result;

			typename cdf_t::iterator it = std::lower_bound(m_cdf.begin(), m_cdf.end(), std::make_pair(m_cdf.back().first, prob), inverse_less());
			if (it == m_cdf.end())
				result = m_cdf.back().first;
			else if (it == m_cdf.begin())
				result = m_cdf.front().first;
			else
			{
				typename cdf_t::iterator it2 = it - 1;
				value_t x1 = it2->first;
				value_t x2 = it->first;
				prob_t y1 = it2->second;
				prob_t y2 = it->second;
				if (y2 == y1)
					result = (x1 + x2) / 2;
				else
					result = x1 + (double)(x2 - x1) / (double)(y2 - y1) * (prob - y1);				
			}			
			return result;
		}
	};

	class non_parametric_cdf_2_t
	{
		typedef float prob_t;
	public:
		cont_value_t * m_values;
		prob_t * m_probs;
		long long m_size;
	public:
		non_parametric_cdf_2_t(
				cont_value_t * values,
				prob_t * probs,
				long long size)
			: m_values(values), m_probs(probs), m_size(size)
		{	
		}

		non_parametric_cdf_2_t(
				const hpgl_non_parametric_cdf_t * cdf)
			:m_values(cdf->m_values), m_probs(cdf->m_probs), m_size(cdf->m_size)
		{
			// F-204: both SGS C++ entries (sequential_gaussian_simulation and
			// _lvm) hand this class a raw C struct and then call
			// std::lower_bound on the two arrays. The C API front door runs a
			// content scan (api.cpp validate_cdf_content_or_throw) but a
			// direct-C++ consumer bypasses it, so an unsorted range (undefined
			// lower_bound -> silently wrong back-transform) or a NaN (poisons
			// every simulated value) or a null inner array with m_size > 0
			// (SIGSEGV) was accepted here. Validate once per construction, the
			// way the sibling median_ik.cpp entry validates its marginals.
			validate("non_parametric_cdf_2_t");
		}

		bool is_empty() const { return m_size <= 0; }

		/// Content scan mirroring the C boundary's
		/// validate_cdf_content_or_throw (api.cpp:317-358): finiteness and
		/// non-decreasing order of m_values/m_probs, the [0, 1] probability
		/// range (F-397's predicate), plus the null-inner-array check. The
		/// length of the caller's buffers cannot be verified — the C struct
		/// carries no length metadata — so an m_size that overstates the
		/// allocation stays the C API's / caller's responsibility.
		void validate(const char * context)
		{
			if (m_size <= 0)
				return;
			if (m_values == 0 || m_probs == 0)
			{
				std::ostringstream oss;
				oss << context << ": cdf has m_size " << m_size
				    << " but a null m_values/m_probs array";
				throw hpgl_exception(context, oss.str());
			}
			for (long long i = 0; i < m_size; ++i)
			{
				if (!std::isfinite(static_cast<double>(m_values[i]))
					|| !std::isfinite(static_cast<double>(m_probs[i])))
				{
					std::ostringstream oss;
					oss << context << ": cdf values/probs[" << i
					    << "] is not finite";
					throw hpgl_exception(context, oss.str());
				}
				// F4 (F-397 twin): probability range — the C boundary enforces
				// it (api.cpp:330-337) and cdf.py:50-53 rejects it, but this
				// class documented the mirror and only scanned finite/sorted.
				// An out-of-range prob survives lower_bound and saturates the
				// normal-score draw, and on the SGS path the caller's buffer
				// has already been forward-transformed in place by the time
				// the harm shows.
				if (m_probs[i] < 0.0f || m_probs[i] > 1.0f)
				{
					std::ostringstream oss;
					oss << context << ": cdf m_probs[" << i << "]=" << m_probs[i]
					    << " is outside [0, 1] (probabilities must be cumulative probabilities)";
					throw hpgl_exception(context, oss.str());
				}
				if (i > 0)
				{
					if (m_values[i] < m_values[i - 1])
					{
						std::ostringstream oss;
						oss << context << ": cdf m_values must be sorted "
						    << "non-decreasing (values[" << (i - 1) << "]="
						    << m_values[i - 1] << " > values[" << i << "]="
						    << m_values[i] << ")";
						throw hpgl_exception(context, oss.str());
					}
					if (m_probs[i] < m_probs[i - 1])
					{
						std::ostringstream oss;
						oss << context << ": cdf m_probs must be sorted "
						    << "non-decreasing (probs[" << (i - 1) << "]="
						    << m_probs[i - 1] << " > probs[" << i << "]="
						    << m_probs[i] << ")";
						throw hpgl_exception(context, oss.str());
					}
				}
			}
		}

		prob_t prob(cont_value_t value)
		{
			return (*this)(value);
		}

		prob_t operator()(cont_value_t value)
		{
			prob_t result;
			long long idx1, idx2;

			if (m_size <= 0)
				return 0;
			
			cont_value_t * it = std::lower_bound(m_values, m_values + m_size, value);
			if (it == m_values + m_size)
			{
				result = m_probs[m_size-1];
			}
			else if (it == m_values)
			{
				if (value < m_values[0])
					result = 0;
				else
					result = m_probs[0];
			}
			else
			{
				idx2 = it - m_values;
				idx1 = idx2 - 1;
				cont_value_t x1, x2;
				x1 = m_values[idx1];
				x2 = m_values[idx2];
				prob_t y1, y2;
				y1 = m_probs[idx1];
				y2 = m_probs[idx2];
				if (x2 == x1)
					result = (y1 + y2) / 2;
				else
					result = y1 + (y2 - y1) / (x2 - x1) * (value - x1);
			}
			return result;
		}

		cont_value_t inverse(prob_t prob)
		{
			cont_value_t result;
			long long idx1, idx2;

			if (m_size <= 0)
				return 0;
			
			prob_t * it = std::lower_bound(m_probs, m_probs + m_size, prob);
			if (it == m_probs)
				result = m_values[0];
			else if (it == m_probs + m_size)
				result = m_values[m_size-1];
			else
			{
				idx2 = it - m_probs;
				idx1 = idx2 - 1;
				cont_value_t x1, x2;
				prob_t y1, y2;
				x1 = m_values[idx1];
				x2 = m_values[idx2];
				y1 = m_probs[idx1];
				y2 = m_probs[idx2];
				if (y2 == y1)
					result = (x1 + x2) / 2;
				else
					result = x1 + (x2 - x1) / (y2 - y1) * (prob - y1);
			}
			return result;
		}

		void dump()
		{
			std::ostringstream oss;
			oss << "Non-param cdf. Size =  " << m_size << ".\n";

			// Type safety: Use long long for loop index to match m_size type
			for (long long i = 0; i < m_size; ++i)
			{
				oss << "(" << m_values[i] << ", " << m_probs[i] << ")\n";
			}
			write(oss.str());
		}
	};
}

#endif //__NON_PARAMETRIC_CDF_H__6BBB60EA_BA86_4AE3_98FF_3B287C1D1522____
