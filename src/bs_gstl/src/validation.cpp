/*
   Copyright 2009 HPGL Team
   This file is part of HPGL (High Perfomance Geostatistics Library).
   HPGL is free software: you can redistribute it and/or modify it under the terms of the BSD License.
   You should have received a copy of the BSD License along with HPGL.

   Input Validation Framework Implementation for HPGL
   Addresses vulnerability IV-001 (CVSS 7.5) - Insufficient Input Validation
*/

#include "stdafx.h"
#include <validation.h>
#include <neighbourhood_param.h>
#include <ok_params.h>
#include <sk_params.h>
#include <ik_params.h>
#include <sgs_params.h>
#include <covariance_param.h>
#include <logging.h>

namespace hpgl
{
	// Namespace for parameter validation wrappers
	namespace param_validation
	{
		// Validate neighbourhood parameters
		void validate_neighbourhood_params(const neighbourhood_param_t& params)
		{
			// Validate radiuses
			validation_result_t radius_result = validation::validate_radius(
				params.m_radiuses[0],
				params.m_radiuses[1],
				params.m_radiuses[2],
				"search_radius"
			);
			HPGL_VALIDATE_RESULT(radius_result);

			// Validate max neighbors
			validation_result_t neighbors_result = validation::validate_max_neighbors(params.m_max_neighbours);
			HPGL_VALIDATE_RESULT(neighbors_result);
		}

		// Validate OK parameters
		void validate_ok_params(const ok_params_t& params)
		{
			// First validate neighbourhood params (base class)
			validate_neighbourhood_params(params);

			// Validate covariance parameters
			validation_result_t cov_result = validation::validate_covariance_parameters(
				params.m_sill,
				params.m_nugget,
				params.m_ranges
			);
			HPGL_VALIDATE_RESULT(cov_result);

			// Validate angles
			validation_result_t angles_result = validation::validate_angles(params.m_angles);
			HPGL_VALIDATE_RESULT(angles_result);
		}

		// Validate SK parameters
		void validate_sk_params(const sk_params_t& params)
		{
			// Validate OK params first (base class)
			validate_ok_params(params);

			// SK-specific validation if needed
			// For now, SK inherits all validation from OK
		}

		// Validate IK parameters
		void validate_ik_params(const ik_params_t& params)
		{
			// Validate category count
			validation_result_t count_result = validation::validate_indicator_count(params.m_category_count);
			HPGL_VALIDATE_RESULT(count_result);

			// Validate each indicator's parameters
			for (size_t i = 0; i < params.m_category_count; ++i)
			{
				// Validate covariance parameters for this indicator
				validation_result_t cov_result = validation::validate_covariance_parameters(
					params.m_sills[i],
					params.m_nuggets[i],
					params.m_ranges[i].data()
				);
				HPGL_VALIDATE_RESULT(cov_result);

				// Validate angles for this indicator
				validation_result_t angles_result = validation::validate_angles(params.m_angles[i].data());
				HPGL_VALIDATE_RESULT(angles_result);

				// Validate radiuses for this indicator
				validation_result_t radius_result = validation::validate_radius(
					params.m_radiuses[i][0],
					params.m_radiuses[i][1],
					params.m_radiuses[i][2],
					("ik_radius_" + std::to_string(i)).c_str()
				);
				HPGL_VALIDATE_RESULT(radius_result);

				// Validate neighbor limits
				validation_result_t neighbors_result = validation::validate_max_neighbors(params.m_neighbour_limits[i]);
				HPGL_VALIDATE_RESULT(neighbors_result);

				// Validate marginal probability
				validation_result_t prob_result = validation::validate_probability(
					params.m_marginal_probs[i],
					("marginal_prob_" + std::to_string(i)).c_str()
				);
				HPGL_VALIDATE_RESULT(prob_result);
			}
		}

		// Validate SGS parameters
		void validate_sgs_params(const sgs_params_t& params)
		{
			// Validate SK params first (base class)
			validate_sk_params(params);

			// Validate seed
			validation_result_t seed_result = validation::validate_seed(params.m_seed);
			HPGL_VALIDATE_RESULT(seed_result);

			// Validate min_neighbors
			validation_result_t min_neighbors_result = validation::validate_min_neighbors(
				params.m_min_neighbours,
				params.m_max_neighbours
			);
			HPGL_VALIDATE_RESULT(min_neighbors_result);
		}

		// Validate covariance parameters independently
		void validate_covariance_params(const covariance_param_t& params)
		{
			validation_result_t cov_result = validation::validate_covariance_parameters(
				params.m_sill,
				params.m_nugget,
				params.m_ranges
			);
			HPGL_VALIDATE_RESULT(cov_result);

			validation_result_t angles_result = validation::validate_angles(params.m_angles);
			HPGL_VALIDATE_RESULT(angles_result);
		}

	} // namespace param_validation

	// Extended neighbourhood_param_t with validation
	namespace validated_params
	{
		// Wrapper for neighbourhood_param_t with automatic validation
		class validated_neighbourhood_param_t : public neighbourhood_param_t
		{
		public:
			validated_neighbourhood_param_t()
				: neighbourhood_param_t()
			{}

			validated_neighbourhood_param_t(const sugarbox_search_ellipsoid_t& radiuses, int max_neighbours)
				: neighbourhood_param_t(radiuses, max_neighbours)
			{
				param_validation::validate_neighbourhood_params(*this);
			}

			void set_radiuses(size_t radius1, size_t radius2, size_t radius3)
			{
				neighbourhood_param_t::set_radiuses(radius1, radius2, radius3);
				param_validation::validate_neighbourhood_params(*this);
			}

			void set_max_neighbours(size_t max_neighbours)
			{
				m_max_neighbours = max_neighbours;
				param_validation::validate_neighbourhood_params(*this);
			}
		};

		// Wrapper for ok_params_t with automatic validation
		class validated_ok_params_t : public ok_params_t
		{
		public:
			validated_ok_params_t()
			{
				// Set default values and validate
				m_sill = 1.0;
				m_nugget = 0.0;
				set_radiuses(10, 10, 10);
				m_max_neighbours = validation_constants::DEFAULT_MAX_NEIGHBORS;
				param_validation::validate_ok_params(*this);
			}

			validated_ok_params_t(const ok_params_t& base)
				: ok_params_t(base)
			{
				param_validation::validate_ok_params(*this);
			}

			void set_ranges(double range1, double range2, double range3)
			{
				covariance_param_t::set_ranges(range1, range2, range3);
				param_validation::validate_ok_params(*this);
			}

			void set_angles(double angle1, double angle2, double angle3)
			{
				covariance_param_t::set_angles(angle1, angle2, angle3);
				param_validation::validate_ok_params(*this);
			}

			void set_sill_and_nugget(double sill, double nugget)
			{
				m_sill = sill;
				m_nugget = nugget;
				param_validation::validate_ok_params(*this);
			}
		};

		// Wrapper for sk_params_t with automatic validation
		class validated_sk_params_t : public sk_params_t
		{
		public:
			validated_sk_params_t()
			{
				// Set default values and validate
				m_sill = 1.0;
				m_nugget = 0.0;
				set_radiuses(10, 10, 10);
				m_max_neighbours = validation_constants::DEFAULT_MAX_NEIGHBORS;
				set_mean(0.0);
				m_calculate_mean = true;
				param_validation::validate_sk_params(*this);
			}

			validated_sk_params_t(const sk_params_t& base)
				: sk_params_t(base)
			{
				param_validation::validate_sk_params(*this);
			}

			void set_mean(double mean)
			{
				sk_params_t::set_mean(mean);
				// No need to validate mean value itself
			}
		};

	} // namespace validated_params

} // namespace hpgl
