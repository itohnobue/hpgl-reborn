#ifndef __GAUSSIAN_DISTRIBUTION_H__E30B442B_3C6E_4185_8032_DEBE4ED28813____
#define __GAUSSIAN_DISTRIBUTION_H__E30B442B_3C6E_4185_8032_DEBE4ED28813____

#include <math.h>

namespace hpgl
{
	class gaussian_cdf_t
	{
		double m_mean;
		double m_var;
	public:
		gaussian_cdf_t()
			: m_mean(0), m_var(1)
		{}

		gaussian_cdf_t(double mean, double var)
			: m_mean(mean), m_var(var)
		{}

		inline double prob(double x)
		{
			return (*this)(x);
		}

		inline double operator()(double x)const
		{
			double p = 0.2316419;
			double b[] = {0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429};
			double sx = (x - m_mean) / sqrt(m_var);
			bool neg = (sx < 0);

			if (neg)
				sx = - sx;

			double t = 1 / (1 + p * sx);

			double zx = 1 / sqrt(2 * 3.14159265358) * exp(- sx * sx / 2);
			double result = 1 - zx * ((((b[4] * t + b[3]) * t + b[2]) * t + b[1]) * t + b[0]) * t;
			if (neg)
				result = 1 - result;
			return result;
		}

		// Inverse Gaussian CDF (quantile function)
		// Computes x such that P(X <= x) = p for X ~ N(mean, variance)
		// Uses rational approximation from Abramowitz & Stegun (1964)
		//
		// Numerical stability notes:
		// - For p <= 0 or p >= 1: returns mean to avoid inf/nan
		// - For extreme p values (p < 1e-150): clamps to avoid underflow
		//   Rationale: When p2 < 1e-154, p2*p2 < 1e-308 (underflows to 0)
		//              This causes log(1/0) = inf, sqrt(inf) = inf
		//   The clamping threshold 1e-150 provides a safety margin while
		//   preserving accuracy for practical probability ranges
		inline double inverse(double p)const
		{
			// Input validation: p must be in (0, 1)
			if (p <= 0.0 || p >= 1.0)
			{
				// Return mean for extreme probabilities to avoid numerical instability
				return m_mean;
			}

			double ps[] = {-0.322232431088, -1.0, -0.342242088547,
				-0.0204231210245, -0.453642210148e-4};

			double qs[] = {0.0993484626060, 0.588581570495, 0.531103462366,
			0.103537752850, 0.38560700634e-2};

			double p2 = (p > 0.5) ? (1 - p) : p;

			// Clamp p2 to avoid underflow in p2*p2 computation
			// When p2 < ~1e-154, squaring produces subnormal/zero, causing log(inf)
			// Clamping to 1e-150 prevents this while maintaining accuracy for
			// typical geostatistical applications (probabilities > 1e-10)
			if (p2 < 1e-150)
				p2 = 1e-150;

			double y = sqrt(log(1 / (p2*p2)));
			double xp = y + ((((ps[4] * y + ps[3]) * y + ps[2]) * y + ps[1]) * y + ps[0]) /
				((((qs[4] * y + qs[3]) * y + qs[2]) * y + qs[1]) * y + qs[0]);

			if (p < 0.5)
				xp = -xp;

			return xp * sqrt(m_var) + m_mean;

		}
	};
}

#endif //__GAUSSIAN_DISTRIBUTION_H__E30B442B_3C6E_4185_8032_DEBE4ED28813____