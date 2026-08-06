#include "stdafx.h"
#include "covariance_param.h"
#include "hpgl_exception.h"
#include <cmath>


namespace hpgl
{
	covariance_param_t::covariance_param_t()
	{
		// E-M63: the default sill must be a valid positive value (the
		// set_sill invariant), not 0.  With sill=0 the natural call order
		// `covariance_param_t p; p.set_nugget(0.5);` deterministically
		// threw "nugget must be <= sill" although the eventual state
		// (sill 1.0, nugget 0.5) would have been valid.  The default is
		// overwritten by callers that configure explicitly; a bare
		// default-constructed object is now valid (validate() passes).
		m_sill = 1.0;
		m_nugget = 0;
		m_covariance_type = covariance_type_t::COV_SPHERICAL;
		// Default to unit range (1,1,1) — range=0 is rejected by
		// create_transform() (cov_model.h:44), so use a sane default
		// that will be overwritten by the caller before use.
		// Callers that omit set_ranges() will get a clear error from
		// create_transform() rather than a confusing internal failure.
		set_ranges(1, 1, 1);
		set_angles(0, 0, 0);
	}
	void covariance_param_t::set_ranges(double range1, double range2, double range3)
	{
		// Range must be strictly positive — range=0 is rejected by
		// create_transform() (cov_model.h:44), so validate consistently
		// here to provide a clear error at the point of assignment.
		if (!std::isfinite(range1) || range1 <= 0.0)
			throw hpgl_exception("covariance_param_t::set_ranges", "range1 must be > 0 and finite");
		if (!std::isfinite(range2) || range2 <= 0.0)
			throw hpgl_exception("covariance_param_t::set_ranges", "range2 must be > 0 and finite");
		if (!std::isfinite(range3) || range3 <= 0.0)
			throw hpgl_exception("covariance_param_t::set_ranges", "range3 must be > 0 and finite");
		m_ranges[0] = range1;
		m_ranges[1] = range2;
		m_ranges[2] = range3;
	}

	void covariance_param_t::set_angles(double angle1, double angle2, double angle3)
	{
		if (!std::isfinite(angle1))
			throw hpgl_exception("covariance_param_t::set_angles", "angle1 must be finite");
		if (!std::isfinite(angle2))
			throw hpgl_exception("covariance_param_t::set_angles", "angle2 must be finite");
		if (!std::isfinite(angle3))
			throw hpgl_exception("covariance_param_t::set_angles", "angle3 must be finite");
		m_angles[0] = angle1;
		m_angles[1] = angle2;
		m_angles[2] = angle3;
	}

	void covariance_param_t::set_sill(double sill)
	{
		if (!std::isfinite(sill) || sill <= 0.0)
			throw hpgl_exception("covariance_param_t::set_sill", "sill must be > 0 and finite");
		// E-M63: enforce the nugget <= sill cross-invariant here as well,
		// so `set_sill` after `set_nugget` cannot silently leave an
		// invalid state (nugget > sill).  With both setters enforcing the
		// invariant against the current state, every setter sequence is
		// order-independent: it either produces a valid state or throws
		// loudly at the offending call.
		if (m_nugget > sill)
			throw hpgl_exception("covariance_param_t::set_sill", "sill must be >= nugget");
		m_sill = sill;
	}

	void covariance_param_t::set_nugget(double nugget)
	{
		if (!std::isfinite(nugget) || nugget < 0.0)
			throw hpgl_exception("covariance_param_t::set_nugget", "nugget must be >= 0 and finite");
		if (nugget > m_sill)
			throw hpgl_exception("covariance_param_t::set_nugget", "nugget must be <= sill");
		m_nugget = nugget;
	}

	void covariance_param_t::validate() const
	{
		if (!std::isfinite(m_sill) || m_sill <= 0.0)
			throw hpgl_exception("covariance_param_t::validate", "sill must be > 0 and finite");
		if (!std::isfinite(m_nugget) || m_nugget < 0.0)
			throw hpgl_exception("covariance_param_t::validate", "nugget must be >= 0 and finite");
		if (m_nugget > m_sill)
			throw hpgl_exception("covariance_param_t::validate", "nugget must be <= sill");
		for (int i = 0; i < 3; ++i)
		{
			if (!std::isfinite(m_ranges[i]) || m_ranges[i] <= 0.0)
				throw hpgl_exception("covariance_param_t::validate", "range must be > 0 and finite");
			if (!std::isfinite(m_angles[i]))
				throw hpgl_exception("covariance_param_t::validate", "angle must be finite");
		}
		// III-09: anisotropy range ratios must not overflow to Inf/NaN. The
		// transform builder (cov_model.h create_transform) guards the same
		// ratios; this check closes the source for consumers that validate
		// without building the transform. ratios[0]/ratios[1] overflows only
		// for physically meaningless parameter sets.
		if (!std::isfinite(m_ranges[0] / m_ranges[1]) || !std::isfinite(m_ranges[0] / m_ranges[2]))
			throw hpgl_exception("covariance_param_t::validate",
				"anisotropy range ratio overflows to non-finite (ranges must be comparable in magnitude)");
	}
}