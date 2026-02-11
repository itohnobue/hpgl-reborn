/*
   Copyright 2009 HPGL Team
   This file is part of HPGL (High Perfomance Geostatistics Library).
   HPGL is free software: you can redistribute it and/or modify it under the terms of the BSD License.
   You should have received a copy of the BSD License along with HPGL.

*/


#include "stdafx.h"
#include <covariance_param.h>
#include <validation.h>


namespace hpgl
{
	covariance_param_t::covariance_param_t()
		: m_covariance_type(COV_SPHERICAL),
		m_sill(0),
		m_nugget(0)
	{
		set_ranges(0, 0, 0);
		set_angles(0, 0, 0);
	}
	void covariance_param_t::set_ranges(double range1, double range2, double range3)
	{
		// Validate range values
		double ranges[3] = {range1, range2, range3};
		for (int i = 0; i < 3; ++i)
		{
			if (std::isnan(ranges[i]) || std::isinf(ranges[i]))
			{
				throw validation_exception("covariance_param_t::set_ranges",
					"Range value at index " + std::to_string(i) + " is NaN or infinite",
					"ranges");
			}
			if (ranges[i] < validation_constants::MIN_RANGE)
			{
				throw validation_exception("covariance_param_t::set_ranges",
					"Range value at index " + std::to_string(i) + " is " +
					std::to_string(ranges[i]) + ", which is less than minimum of " +
					std::to_string(validation_constants::MIN_RANGE),
					"ranges");
			}
			if (ranges[i] > validation_constants::MAX_RANGE)
			{
				throw validation_exception("covariance_param_t::set_ranges",
					"Range value at index " + std::to_string(i) + " is " +
					std::to_string(ranges[i]) + ", which exceeds maximum of " +
					std::to_string(validation_constants::MAX_RANGE),
					"ranges");
			}
		}

		m_ranges[0] = range1;
		m_ranges[1] = range2;
		m_ranges[2] = range3;
	}

	void covariance_param_t::set_angles(double angle1, double angle2, double angle3)
	{
		double angles[3] = {angle1, angle2, angle3};

		for (int i = 0; i < 3; ++i)
		{
			if (std::isnan(angles[i]) || std::isinf(angles[i]))
			{
				throw validation_exception("covariance_param_t::set_angles",
					"Angle value at index " + std::to_string(i) + " is NaN or infinite",
					"angles");
			}
		}

		m_angles[0] = angle1;
		m_angles[1] = angle2;
		m_angles[2] = angle3;
	}
}