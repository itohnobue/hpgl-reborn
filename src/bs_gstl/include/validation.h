/*
   Copyright 2009 HPGL Team
   This file is part of HPGL (High Perfomance Geostatistics Library).
   HPGL is free software: you can redistribute it and/or modify it under the terms of the BSD License.
   You should have received a copy of the BSD License along with HPGL.

   Input Validation Framework for HPGL
   Addresses vulnerability IV-001 (CVSS 7.5) - Insufficient Input Validation
*/

#ifndef __VALIDATION_H__INCLUDED_A7B3C9D2_E4F1_4A8C_B9E7_3D2F1A8C4B6E__
#define __VALIDATION_H__INCLUDED_A7B3C9D2_E4F1_4A8C_B9E7_3D2F1A8C4B6E__

#include <stdexcept>
#include <string>
#include <type_traits>
#include <limits>
#include <cmath>
#include <sstream>
#include <iostream>
#include <typedefs.h>
#include <hpgl_exception.h>

namespace hpgl
{
	// Validation exception class for detailed error reporting
	class validation_exception : public hpgl_exception
	{
	public:
		validation_exception(const std::string& where, const std::string& what, const std::string& parameter_name = "")
			: hpgl_exception(where, what), m_parameter_name(parameter_name)
		{
			if (!parameter_name.empty())
			{
				m_message = parameter_name + ": " + what + " in " + where;
			}
			else
			{
				m_message = where + ": " + what;
			}
		}

		const char* what() const noexcept override
		{
			return m_message.c_str();
		}

	private:
		std::string m_parameter_name;
		std::string m_message;
	};

	// Validation severity levels for logging
	enum class validation_severity_t
	{
		info,
		warning,
		error,
		critical
	};

	// Validation result structure
	struct validation_result_t
	{
		bool is_valid;
		std::string message;
		validation_severity_t severity;

		validation_result_t()
			: is_valid(true), severity(validation_severity_t::info)
		{}

		validation_result_t(bool valid, const std::string& msg, validation_severity_t sev = validation_severity_t::error)
			: is_valid(valid), message(msg), severity(sev)
		{}
	};

	// Constants for validation limits
	namespace validation_constants
	{
		// Grid dimension limits
		constexpr sugarbox_grid_size_t MIN_GRID_DIMENSION = 1;
		constexpr sugarbox_grid_size_t MAX_GRID_DIMENSION = 10000000;
		constexpr size_t MAX_GRID_SIZE = 1000000000; // 1 billion cells

		// Neighbor count limits
		constexpr size_t MIN_NEIGHBORS = 1;
		constexpr size_t MAX_NEIGHBORS = 1000;
		constexpr size_t DEFAULT_MAX_NEIGHBORS = 12;

		// Radius limits
		constexpr double MIN_RADIUS = 0.0;
		constexpr double MAX_RADIUS = 1000000.0; // 1 million units

		// Covariance parameter limits
		constexpr double MIN_SILL = 0.0;
		constexpr double MAX_SILL = 1e10; // Very large but bounded
		constexpr double MIN_NUGGET = 0.0;
		constexpr double MAX_NUGGET = 1e10;
		constexpr double MIN_RANGE = 0.0;
		constexpr double MAX_RANGE = 1e10;

		// Angle limits (in degrees)
		constexpr double MIN_ANGLE = 0.0;
		constexpr double MAX_ANGLE = 360.0;

		// Probability limits
		constexpr double MIN_PROBABILITY = 0.0;
		constexpr double MAX_PROBABILITY = 1.0;
		constexpr double PROBABILITY_SUM_TOLERANCE = 0.001; // Tolerance for probability sum validation

		// Indicator limits
		constexpr size_t MAX_INDICATORS = 256;
		constexpr indicator_value_t MAX_INDICATOR_VALUE = 255;

		// Seed limits for random number generation
		constexpr long int MIN_SEED = 0;
		constexpr long int MAX_SEED = std::numeric_limits<long int>::max();
	}

	// Core validation functions
	namespace validation
	{
		// Validate grid dimensions
		inline validation_result_t validate_grid_dimensions(sugarbox_grid_size_t x, sugarbox_grid_size_t y, sugarbox_grid_size_t z)
		{
			using namespace validation_constants;

			if (x < MIN_GRID_DIMENSION || x > MAX_GRID_DIMENSION)
			{
				return validation_result_t(false,
					"Grid X dimension " + std::to_string(x) + " outside valid range [" +
					std::to_string(MIN_GRID_DIMENSION) + ", " + std::to_string(MAX_GRID_DIMENSION) + "]",
					validation_severity_t::critical);
			}

			if (y < MIN_GRID_DIMENSION || y > MAX_GRID_DIMENSION)
			{
				return validation_result_t(false,
					"Grid Y dimension " + std::to_string(y) + " outside valid range [" +
					std::to_string(MIN_GRID_DIMENSION) + ", " + std::to_string(MAX_GRID_DIMENSION) + "]",
					validation_severity_t::critical);
			}

			if (z < MIN_GRID_DIMENSION || z > MAX_GRID_DIMENSION)
			{
				return validation_result_t(false,
					"Grid Z dimension " + std::to_string(z) + " outside valid range [" +
					std::to_string(MIN_GRID_DIMENSION) + ", " + std::to_string(MAX_GRID_DIMENSION) + "]",
					validation_severity_t::critical);
			}

			// Check total grid size
			size_t total_size = static_cast<size_t>(x) * static_cast<size_t>(y) * static_cast<size_t>(z);
			if (total_size > MAX_GRID_SIZE)
			{
				return validation_result_t(false,
					"Total grid size " + std::to_string(total_size) + " exceeds maximum of " +
					std::to_string(MAX_GRID_SIZE),
					validation_severity_t::critical);
			}

			return validation_result_t(true, "Grid dimensions valid");
		}

		// Validate radius values
		inline validation_result_t validate_radius(double radius_x, double radius_y, double radius_z, const char* param_name = "radius")
		{
			using namespace validation_constants;

			if (radius_x < MIN_RADIUS || radius_x > MAX_RADIUS)
			{
				return validation_result_t(false,
					param_name + std::string("[0] (X) value ") + std::to_string(radius_x) +
					" outside valid range [" + std::to_string(MIN_RADIUS) + ", " +
					std::to_string(MAX_RADIUS) + "]",
					validation_severity_t::error);
			}

			if (radius_y < MIN_RADIUS || radius_y > MAX_RADIUS)
			{
				return validation_result_t(false,
					param_name + std::string("[1] (Y) value ") + std::to_string(radius_y) +
					" outside valid range [" + std::to_string(MIN_RADIUS) + ", " +
					std::to_string(MAX_RADIUS) + "]",
					validation_severity_t::error);
			}

			if (radius_z < MIN_RADIUS || radius_z > MAX_RADIUS)
			{
				return validation_result_t(false,
					param_name + std::string("[2] (Z) value ") + std::to_string(radius_z) +
					" outside valid range [" + std::to_string(MIN_RADIUS) + ", " +
					std::to_string(MAX_RADIUS) + "]",
					validation_severity_t::error);
			}

			return validation_result_t(true, "Radius values valid");
		}

		// Validate size_t radius values (for neighbourhood parameters)
		inline validation_result_t validate_radius(size_t radius_x, size_t radius_y, size_t radius_z, const char* param_name = "radius")
		{
			using namespace validation_constants;

			if (radius_x > static_cast<size_t>(MAX_RADIUS))
			{
				return validation_result_t(false,
					param_name + std::string("[0] (X) value ") + std::to_string(radius_x) +
					" exceeds maximum of " + std::to_string(static_cast<size_t>(MAX_RADIUS)),
					validation_severity_t::error);
			}

			if (radius_y > static_cast<size_t>(MAX_RADIUS))
			{
				return validation_result_t(false,
					param_name + std::string("[1] (Y) value ") + std::to_string(radius_y) +
					" exceeds maximum of " + std::to_string(static_cast<size_t>(MAX_RADIUS)),
					validation_severity_t::error);
			}

			if (radius_z > static_cast<size_t>(MAX_RADIUS))
			{
				return validation_result_t(false,
					param_name + std::string("[2] (Z) value ") + std::to_string(radius_z) +
					" exceeds maximum of " + std::to_string(static_cast<size_t>(MAX_RADIUS)),
					validation_severity_t::error);
			}

			return validation_result_t(true, "Radius values valid");
		}

		// Validate neighbor count
		inline validation_result_t validate_max_neighbors(size_t max_neighbors)
		{
			using namespace validation_constants;

			if (max_neighbors < MIN_NEIGHBORS)
			{
				return validation_result_t(false,
					"Max neighbors " + std::to_string(max_neighbors) +
					" is less than minimum of " + std::to_string(MIN_NEIGHBORS),
					validation_severity_t::error);
			}

			if (max_neighbors > MAX_NEIGHBORS)
			{
				return validation_result_t(false,
					"Max neighbors " + std::to_string(max_neighbors) +
					" exceeds maximum of " + std::to_string(MAX_NEIGHBORS),
					validation_severity_t::warning); // Warning for performance, not critical
			}

			return validation_result_t(true, "Max neighbors valid");
		}

		// Validate covariance parameters (sill, nugget, ranges)
		inline validation_result_t validate_covariance_parameters(double sill, double nugget, const double* ranges = nullptr)
		{
			using namespace validation_constants;

			if (std::isnan(sill) || std::isinf(sill))
			{
				return validation_result_t(false,
					"Sill value is NaN or infinite",
					validation_severity_t::error);
			}

			if (sill < MIN_SILL)
			{
				return validation_result_t(false,
					"Sill value " + std::to_string(sill) + " is less than minimum of " +
					std::to_string(MIN_SILL),
					validation_severity_t::error);
			}

			if (sill > MAX_SILL)
			{
				return validation_result_t(false,
					"Sill value " + std::to_string(sill) + " exceeds maximum of " +
					std::to_string(MAX_SILL),
					validation_severity_t::error);
			}

			if (std::isnan(nugget) || std::isinf(nugget))
			{
				return validation_result_t(false,
					"Nugget value is NaN or infinite",
					validation_severity_t::error);
			}

			if (nugget < MIN_NUGGET)
			{
				return validation_result_t(false,
					"Nugget value " + std::to_string(nugget) + " is less than minimum of " +
					std::to_string(MIN_NUGGET),
					validation_severity_t::error);
			}

			if (nugget > MAX_NUGGET)
			{
				return validation_result_t(false,
					"Nugget value " + std::to_string(nugget) + " exceeds maximum of " +
					std::to_string(MAX_NUGGET),
					validation_severity_t::error);
			}

			// Critical: Nugget should not exceed sill
			if (nugget > sill)
			{
				return validation_result_t(false,
					"Nugget value " + std::to_string(nugget) + " exceeds sill value " +
					std::to_string(sill) + " (nugget must be <= sill)",
					validation_severity_t::critical);
			}

			// Validate ranges if provided
			if (ranges != nullptr)
			{
				for (int i = 0; i < 3; ++i)
				{
					if (std::isnan(ranges[i]) || std::isinf(ranges[i]))
					{
						return validation_result_t(false,
							"Range value at index " + std::to_string(i) + " is NaN or infinite",
							validation_severity_t::error);
					}

					if (ranges[i] < MIN_RANGE)
					{
						return validation_result_t(false,
							"Range value at index " + std::to_string(i) + " is " +
							std::to_string(ranges[i]) + ", which is less than minimum of " +
							std::to_string(MIN_RANGE),
							validation_severity_t::error);
					}

					if (ranges[i] > MAX_RANGE)
					{
						return validation_result_t(false,
							"Range value at index " + std::to_string(i) + " is " +
							std::to_string(ranges[i]) + ", which exceeds maximum of " +
							std::to_string(MAX_RANGE),
							validation_severity_t::error);
					}
				}
			}

			return validation_result_t(true, "Covariance parameters valid");
		}

		// Validate angles
		inline validation_result_t validate_angles(const double* angles)
		{
			using namespace validation_constants;

			for (int i = 0; i < 3; ++i)
			{
				if (std::isnan(angles[i]) || std::isinf(angles[i]))
				{
					return validation_result_t(false,
						"Angle value at index " + std::to_string(i) + " is NaN or infinite",
						validation_severity_t::error);
				}

				// Allow angles outside [0, 360] but warn
				if (angles[i] < MIN_ANGLE || angles[i] > MAX_ANGLE)
				{
					return validation_result_t(true, // Still valid, just a warning
						"Angle value at index " + std::to_string(i) + " is " +
						std::to_string(angles[i]) + ", which is outside typical range [" +
						std::to_string(MIN_ANGLE) + ", " + std::to_string(MAX_ANGLE) + "]",
						validation_severity_t::warning);
				}
			}

			return validation_result_t(true, "Angles valid");
		}

		// Validate probability value
		inline validation_result_t validate_probability(double prob, const char* param_name = "probability")
		{
			using namespace validation_constants;

			if (std::isnan(prob) || std::isinf(prob))
			{
				return validation_result_t(false,
					std::string(param_name) + " is NaN or infinite",
					validation_severity_t::error);
			}

			if (prob < MIN_PROBABILITY || prob > MAX_PROBABILITY)
			{
				return validation_result_t(false,
					std::string(param_name) + " value " + std::to_string(prob) +
					" outside valid range [" + std::to_string(MIN_PROBABILITY) + ", " +
					std::to_string(MAX_PROBABILITY) + "]",
					validation_severity_t::error);
			}

			return validation_result_t(true, "Probability valid");
		}

		// Validate that probabilities sum to approximately 1.0
		inline validation_result_t validate_probability_sum(const double* probs, size_t count)
		{
			using namespace validation_constants;

			double sum = 0.0;
			for (size_t i = 0; i < count; ++i)
			{
				if (std::isnan(probs[i]) || std::isinf(probs[i]))
				{
					return validation_result_t(false,
						"Probability at index " + std::to_string(i) + " is NaN or infinite",
						validation_severity_t::error);
				}
				sum += probs[i];
			}

			double diff = std::abs(sum - 1.0);
			if (diff > PROBABILITY_SUM_TOLERANCE)
			{
				return validation_result_t(false,
					"Probabilities sum to " + std::to_string(sum) +
					", expected 1.0 (difference: " + std::to_string(diff) + ")",
					validation_severity_t::error);
			}

			return validation_result_t(true, "Probabilities sum to 1.0");
		}

		// Validate seed value
		inline validation_result_t validate_seed(long int seed)
		{
			using namespace validation_constants;

			if (seed < MIN_SEED)
			{
				return validation_result_t(false,
					"Seed value " + std::to_string(seed) + " is less than minimum of " +
					std::to_string(MIN_SEED),
					validation_severity_t::warning);
			}

			return validation_result_t(true, "Seed value valid");
		}

		// Validate indicator count
		inline validation_result_t validate_indicator_count(size_t count)
		{
			using namespace validation_constants;

			if (count == 0)
			{
				return validation_result_t(false,
					"Indicator count cannot be zero",
					validation_severity_t::error);
			}

			if (count > MAX_INDICATORS)
			{
				return validation_result_t(false,
					"Indicator count " + std::to_string(count) +
					" exceeds maximum of " + std::to_string(MAX_INDICATORS),
					validation_severity_t::error);
			}

			return validation_result_t(true, "Indicator count valid");
		}

		// Validate indicator value
		inline validation_result_t validate_indicator_value(indicator_value_t value, size_t indicator_count)
		{
			using namespace validation_constants;

			if (value >= indicator_count)
			{
				return validation_result_t(false,
					"Indicator value " + std::to_string(static_cast<int>(value)) +
					" exceeds indicator count " + std::to_string(indicator_count),
					validation_severity_t::error);
			}

			return validation_result_t(true, "Indicator value valid");
		}

		// Validate min_neighbors (for SGS/SIS)
		inline validation_result_t validate_min_neighbors(size_t min_neighbors, size_t max_neighbors)
		{
			if (min_neighbors > max_neighbors)
			{
				return validation_result_t(false,
					"Min neighbors " + std::to_string(min_neighbors) +
					" exceeds max neighbors " + std::to_string(max_neighbors),
					validation_severity_t::error);
			}

			return validation_result_t(true, "Min neighbors valid");
		}
	}

	// Helper macros for throwing validation exceptions
#define HPGL_VALIDATE(condition, message) \
	do { \
		if (!(condition)) { \
			throw validation_exception(__FUNCTION__, message); \
		} \
	} while(0)

#define HPGL_VALIDATE_RESULT(result) \
	do { \
		if (!(result).is_valid) { \
			if ((result).severity >= validation_severity_t::error) { \
				throw validation_exception(__FUNCTION__, (result).message); \
			} else { \
				std::cerr << "Warning: " << (result).message << std::endl; \
			} \
		} \
	} while(0)

} // namespace hpgl

#endif // __VALIDATION_H__INCLUDED_A7B3C9D2_E4F1_4A8C_B9E7_3D2F1A8C4B6E__
