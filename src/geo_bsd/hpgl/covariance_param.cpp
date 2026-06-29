#include "stdafx.h"
#include "covariance_param.h"
#include "hpgl_exception.h"
#include <cmath>


namespace hpgl
{
	covariance_param_t::covariance_param_t()
	{
		m_sill = 0;
		m_nugget = 0;
		m_covariance_type = covariance_type_t::COV_SPHERICAL;
		set_ranges(0, 0, 0);
		set_angles(0, 0, 0);
	}
	void covariance_param_t::set_ranges(double range1, double range2, double range3)
	{
		if (!std::isfinite(range1) || range1 < 0.0)
			throw hpgl_exception("covariance_param_t::set_ranges", "range1 must be >= 0 and finite");
		if (!std::isfinite(range2) || range2 < 0.0)
			throw hpgl_exception("covariance_param_t::set_ranges", "range2 must be >= 0 and finite");
		if (!std::isfinite(range3) || range3 < 0.0)
			throw hpgl_exception("covariance_param_t::set_ranges", "range3 must be >= 0 and finite");
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
		if (!std::isfinite(sill) || sill < 0.0)
			throw hpgl_exception("covariance_param_t::set_sill", "sill must be >= 0 and finite");
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
		if (!std::isfinite(m_sill) || m_sill < 0.0)
			throw hpgl_exception("covariance_param_t::validate", "sill must be >= 0 and finite");
		if (!std::isfinite(m_nugget) || m_nugget < 0.0)
			throw hpgl_exception("covariance_param_t::validate", "nugget must be >= 0 and finite");
		if (m_nugget > m_sill)
			throw hpgl_exception("covariance_param_t::validate", "nugget must be <= sill");
	}
}