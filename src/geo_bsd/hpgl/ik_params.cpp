#include "stdafx.h"
#include "ik_params.h"

namespace hpgl
{
	ik_params_t::ik_params_t()
		: m_category_count(0)
	{}

	void ik_params_t::add_indicator(
			covariance_type_t covariance_type, 
			double range1, double range2, double range3, 
			double angle1, double angle2, double angle3, 
			double sill, 
			double nugget,
			double radius1, double radius2, double radius3, 
			size_t neighbour_limit, double marginal_prob)
	{
		// E-M64: strong exception safety — build every throw-capable
		// derived parameter (covariance_param_t setters/validate,
		// neighbourhood_param_t::set_radiuses) BEFORE mutating any member
		// vector.  Previously the 8 raw pushes happened first and a
		// validation throw (invalid ranges, sill <= 0, nugget > sill)
		// left the parallel vectors with orphan elements (size drift vs
		// m_category_count).  The remaining appends can only throw
		// bad_alloc; on such a throw every vector is rolled back to
		// m_category_count so the object state is fully unchanged.
		std::vector<double> ranges;
		ranges.push_back(range1);
		ranges.push_back(range2);
		ranges.push_back(range3);

		std::vector<double> angles;
		angles.push_back(angle1);
		angles.push_back(angle2);
		angles.push_back(angle3);

		sugarbox_search_ellipsoid_t radiuses(radius1, radius2, radius3);

		covariance_param_t cov_param;
		cov_param.set_angles(angle1, angle2, angle3);
		cov_param.set_ranges(range1, range2, range3);
		cov_param.m_covariance_type = covariance_type;
		cov_param.set_sill(sill);
		cov_param.set_nugget(nugget);
		cov_param.validate();

		neighbourhood_param_t nb_param;
		nb_param.set_radiuses(radius1, radius2, radius3);
		nb_param.m_max_neighbours = neighbour_limit;

		try {
			m_covariances.push_back(covariance_type);
			m_ranges.push_back(ranges);
			m_angles.push_back(angles);
			m_sills.push_back(sill);
			m_nuggets.push_back(nugget);
			m_radiuses.push_back(radiuses);
			m_neighbour_limits.push_back(neighbour_limit);
			m_marginal_probs.push_back(marginal_prob);
			m_cov_params.push_back(cov_param);
			m_nb_params.push_back(nb_param);
			m_category_count++;
		} catch (...) {
			// Roll back partially-appended elements (bad_alloc mid-way).
			// m_category_count was not yet incremented, so resizing every
			// parallel vector to it restores the pre-call lengths.
			m_covariances.resize(m_category_count);
			m_ranges.resize(m_category_count);
			m_angles.resize(m_category_count);
			m_sills.resize(m_category_count);
			m_nuggets.resize(m_category_count);
			m_radiuses.resize(m_category_count);
			m_neighbour_limits.resize(m_category_count);
			m_marginal_probs.resize(m_category_count);
			m_cov_params.resize(m_category_count);
			m_nb_params.resize(m_category_count);
			throw;
		}
	}

	void ik_params_t::add_indicator(const indicator_param_t & indicator)
	{
		add_indicator(
			indicator.m_covariance_type,
			indicator.m_ranges[0],
			indicator.m_ranges[1],
			indicator.m_ranges[2],
			indicator.m_angles[0],
			indicator.m_angles[1],
			indicator.m_angles[2],
			indicator.m_sill,
			indicator.m_nugget,
			indicator.m_radiuses[0],
			indicator.m_radiuses[1],
			indicator.m_radiuses[2],
			indicator.m_max_neighbours,
			indicator.m_marginal_prob
			);
	}

	std::ostream & operator<<(std::ostream&s, const ik_params_t & p)
	{
		// E-M64: bound the loop by the actual sizes of the parallel
		// vectors as well as m_category_count.  The members are public,
		// so an inconsistent count > size (direct member writes, or a
		// builder interrupted mid-loop) must not cause an OOB read here.
		size_t count = p.m_category_count;
		count = std::min(count, p.m_covariances.size());
		count = std::min(count, p.m_ranges.size());
		count = std::min(count, p.m_angles.size());
		count = std::min(count, p.m_sills.size());
		count = std::min(count, p.m_nuggets.size());
		count = std::min(count, p.m_radiuses.size());
		count = std::min(count, p.m_neighbour_limits.size());
		count = std::min(count, p.m_marginal_probs.size());
		for (size_t i = 0; i < count; ++i)
		{
			s 
				<< "\t\tCovariance type: " << (int)p.m_covariances[i] << "\n"
				<< "\t\tRanges:	[" << p.m_ranges[i][0] << ", " << p.m_ranges[i][1] << ", " << p.m_ranges[i][2] << "]\n"
				<< "\t\tAngles: [" << p.m_angles[i][0] << ", " << p.m_angles[i][1] << ", " << p.m_angles[i][2] << "]\n"
				<< "\t\tSill: " << p.m_sills[i] << '\n'
				<< "\t\tNugget: " << p.m_nuggets[i] << "\n"
				<< "\t\tSearch radiuses: " << p.m_radiuses[i] << "\n"
				<< "\t\tMax number of neighbours: " << p.m_neighbour_limits[i] << "\n"
				<< "\t\tMarginal probability: " << p.m_marginal_probs[i] << "\n";
		}
		return s;
	}
}
