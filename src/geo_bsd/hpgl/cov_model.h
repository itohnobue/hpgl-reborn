#ifndef COV_MODEL_H_INCLUDED_IN_HPGL_SDFKSJHKJ234SDF234JSDFIW29834KJNFHDAJK234NK32JNFDSAF234
#define COV_MODEL_H_INCLUDED_IN_HPGL_SDFKSJHKJ234SDF234JSDFIW29834KJNFHDAJK234NK32JNFDSAF234
// ====================================================================
// ROTATION CONVENTION NOTE (HPGL vs GSLIB)
//
// HPGL uses ZYX (intrinsic) Euler rotation order when building the
// anisotropy transform in create_transform():
//   rotate_z(ang1) → rotate_y(ang2) → rotate_x(ang3)
// This produces a combined matrix R = R_x * R_y * R_z (applied right-
// to-left on column vectors in the TNT 1-indexed convention).
//
// GSLIB (the industry-standard geostatistics library) uses ZXY rotation
// order (azimuth about Z, dip about X', plunge about Y''), which is
// different. Users porting GSLIB parameter files with anisotropic angles
// (m_angles) to HPGL MUST convert their angles: the same triplet of
// angles produces different anisotropy orientation under ZYX vs ZXY
// conventions.
//
// The scaling matrix applies ranges as:
//   diag(1, range_x / range_y, range_x / range_z)
// This matches both GSLIB and standard geostatistics practice.
// ====================================================================

#include "covariance_param.h"
#include "hpgl_exception.h"

namespace hpgl
{
	const double pi = 3.14159265358979323846264338327950288419;
	
	inline double grad_to_rad(double grad)
	{
		return ( grad*pi ) / 180;
	}

	template<typename ranges_t, typename angles_t>
	void create_transform(const ranges_t & ranges, const angles_t & angles, std::vector<double> & result)
	{
		using namespace TNT;
		result.resize(9);

		for (int i = 0; i < 3; ++i)
		{
			// III-09: harden the zero-check — comparison-only `ranges[i] == 0`
			// is NaN-bypassable (IEEE-754: NaN == 0 is false), and a NaN/Inf
			// range would silently produce a NaN/Inf transform.
			if (!std::isfinite(ranges[i]) || ranges[i] == 0)
				throw hpgl_exception("create_transform", "All covariance ranges should be non-zero and finite");
		}

		Matrix<double> scale(3,3,0.0);
		Matrix<double> rotate_z(3,3,0.0);
		Matrix<double> rotate_y(3,3,0.0);
		Matrix<double> rotate_x(3,3,0.0);

		// III-09: anisotropy range ratios are computed BEFORE assignment so a
		// ratio that overflows to Inf/NaN (e.g. ranges = {1e10, 1e-300, 1})
		// is rejected here instead of silently poisoning the transform with
		// Inf. An Inf scale entry flows through transfrom_and_norm → Inf norm
		// → NaN/0 covariance for every cov_model_t consumer; downstream solver
		// NaN pre-scans mitigate but any direct cov_model_t consumer gets
		// silent NaN. Close at the source.
		const double ratio_y = ranges[0] / ranges[1];
		const double ratio_z = ranges[0] / ranges[2];
		if (!std::isfinite(ratio_y) || !std::isfinite(ratio_z))
		{
			std::ostringstream oss;
			oss << "create_transform: anisotropy range ratio overflows to "
			    << "non-finite (ranges " << ranges[0] << "/" << ranges[1]
			    << " = " << ratio_y << ", " << ranges[0] << "/" << ranges[2]
			    << " = " << ratio_z << ")";
			throw hpgl_exception("create_transform", oss.str());
		}

		scale(1, 1) = 1.0;
		scale(2, 2) = ratio_y;
		scale(3, 3) = ratio_z;

		rotate_z(1, 1) = cos(grad_to_rad(angles[0]));
		rotate_z(2, 2) = cos(grad_to_rad(angles[0]));
		rotate_z(1, 2) = sin(grad_to_rad(angles[0]));
		rotate_z(2, 1) = -sin(grad_to_rad(angles[0]));
		rotate_z(3, 3) = 1;

		rotate_y(1, 1) =  cos(grad_to_rad(angles[1]));
		rotate_y(3, 3) =  cos(grad_to_rad(angles[1]));
		rotate_y(1, 3) = -sin(grad_to_rad(angles[1]));
		rotate_y(3, 1) =  sin(grad_to_rad(angles[1]));
		rotate_y(2, 2) =  1;  
  
		rotate_x(2, 2) =  cos(grad_to_rad(angles[2]));
		rotate_x(3, 3) =  cos(grad_to_rad(angles[2]));
		rotate_x(2, 3) =  sin(grad_to_rad(angles[2]));
		rotate_x(3, 2) = -sin(grad_to_rad(angles[2]));
		rotate_x(1, 1) = 1;

		Matrix<double> t = scale*((rotate_x * rotate_y) * rotate_z);
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
			{
				result[3*i+j] = t(i+1, j+1);
			}
	}

	template<typename vec_t>
	double transfrom_and_norm(const vec_t & vec, const std::vector<double> & transform)
	{
		double result = 0.0;
		for (int i = 0; i < 3; ++i)
		{
			double d = 0.0;
			for (int j = 0; j < 3; ++j)
			{
				d += vec[j] * transform[3*i + j];
			}
			result += d*d;
		}
		return sqrt(result);
	}

	class cov_model_t
	{
		covariance_param_t m_params;

		double (cov_model_t::*fun)(double)const;

		std::vector<double> m_transform;

		void init_fun()
		{
			switch (m_params.m_covariance_type)
			{
			case covariance_type_t::COV_SPHERICAL:
				fun = &cov_model_t::spherical;
				break;
			case covariance_type_t::COV_EXPONENTIAL:
				fun = &cov_model_t::exponential;
				break;
			case covariance_type_t::COV_GAUSSIAN:
				fun = &cov_model_t::gaussian;
				break;
			default:
				throw hpgl_exception("cov_model_t::init_fun", "Unknown covariance type");
			}

		}
	public:

		cov_model_t(const covariance_param_t & params)
		{
			m_params = params;
			m_params.validate();
			init_fun();
			create_transform(params.m_ranges, params.m_angles, m_transform);
		}

		template<typename ranges_t, typename angles_t>
		cov_model_t(covariance_type_t cov_type, const ranges_t & ranges,
			const angles_t & angles, double sill, double nugget)
		{
			m_params.m_covariance_type = cov_type;
			m_params.set_ranges(ranges[0], ranges[1], ranges[2]);
			m_params.set_angles(angles[0], angles[1], angles[2]);
			m_params.set_sill(sill);
			m_params.set_nugget(nugget);
			m_params.validate();
			init_fun();
			create_transform(m_params.m_ranges, m_params.m_angles, m_transform);
		}
		

		double operator()(double h)const
		{
			return ((*this).*fun)(h);
		}

		template<typename coord_t>
		double operator()(const coord_t & c1, const coord_t & c2)const
		{
			double v[3];
			v[0] = c1[0] - c2[0];
			v[1] = c1[1] - c2[1];
			v[2] = c1[2] - c2[2];
			
			//double h = sqrt(v0 * v0 + v1 * v1 + v2 * v2);
			double h = transfrom_and_norm(v, m_transform);
			return this->operator()(h);
		}
		
		double gaussian(double h)const
		{
			// Range-relative near-zero threshold: prevents unit-dependent
			// nugget-blind zone at micro-scales and overflow at macro-scales.
			double near_zero = 1e-5 * m_params.m_ranges[0];
			if(h < near_zero)
			{
				return m_params.m_sill;
			}
			else
			{
				return (m_params.m_sill - m_params.m_nugget) * exp(-3 * pow(h / m_params.m_ranges[0], 2));
			}
		}

		double exponential(double h)const
		{
			// Range-relative near-zero threshold: prevents unit-dependent
			// nugget-blind zone at micro-scales and overflow at macro-scales.
			double near_zero = 1e-5 * m_params.m_ranges[0];
			if(h < near_zero)
			{
				return m_params.m_sill;
			}
			else
			{
				return (m_params.m_sill - m_params.m_nugget) * exp(-3 * h / m_params.m_ranges[0]);
			}
		}

		double spherical(double h)const
		{
			// Range-relative near-zero threshold: prevents unit-dependent
			// nugget-blind zone at micro-scales and overflow at macro-scales.
			double near_zero = 1e-5 * m_params.m_ranges[0];
			if(h < near_zero)
			{
				return m_params.m_sill;
			}
			else
			{
				if (h > m_params.m_ranges[0])
					return 0;
				// Horner form: (1 - 1.5*x + 0.5*x³) for x = h/range.
				// Stabilized with fmax to prevent negative results from
				// cancellation when x → 1.0.
				double x = h / m_params.m_ranges[0];
				double val = (m_params.m_sill - m_params.m_nugget) * std::fmax(0.0, (1.0 - x * (1.5 - 0.5 * x * x)));
				return val;
			}
		}
	};
}

#endif
