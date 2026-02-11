#include "stdafx.h"

#include "pretty_printer.h"
#include "ok_params.h"
#include "sk_params.h"
#include "ik_params.h"
#include "sgs_params.h"
#include "hpgl_core.h"

#include "output.h"

namespace hpgl
{
	void print_algo_name(const std::string & name)
	{
		write("---------- Starting " + name + " ----------\n");
	}



	void print_param (const std::string & param, const std::vector<double> & value)
	{
		std::ostringstream oss;
		oss << param << ": [" << value[0] << ", " << value[1] << ", " << value[2] << "]\n";
		write(oss.str());
	}

	void print_param (const std::string & param, const double * value)
	{
		std::ostringstream oss;
		oss << param << ": [" << value[0] << ", " << value[1] << ", " << value[2] << "]\n";
		write(oss.str());
	}

	void print_params(const neighbourhood_param_t & p)
	{
		print_param("Search radiuses", p.m_radiuses);
		print_param("Max number of neighbours", p.m_max_neighbours);
	}

	void print_params(const covariance_param_t & p)
	{
		print_param("Covariance type", (int)p.m_covariance_type);
		print_param("Sill", p.m_sill);
		print_param("Nugget", p.m_nugget);
		print_param("Ranges", p.m_ranges);
		print_param("Angles", p.m_angles);
	}

	void print_params(const ok_params_t & p)
	{
		print_param("Covariance type", (int)p.m_covariance_type);
		print_param("Sill", p.m_sill);
		print_param("Nugget", p.m_nugget);
		print_param("Ranges", p.m_ranges);
		print_param("Angles", p.m_angles);
		print_param("Search radiuses", p.m_radiuses);
		print_param("Max neighbours", p.m_max_neighbours);
	}

	void print_params(const sk_params_t & p)
	{
		print_param("Covariance type", (int)p.m_covariance_type);
		print_param("Sill", p.m_sill);
		print_param("Nugget", p.m_nugget);
		print_param("Ranges", p.m_ranges);
		print_param("Angles", p.m_angles);
		print_param("Search radiuses", p.m_radiuses);
		print_param("Max neighbours", p.m_max_neighbours);
		print_param("Mean", p.mean());
	}

	void print_params(const sgs_params_t & p)
	{
		print_algo_name("SGS Params");
		print_params(static_cast<const ok_params_t&>(p));
		print_param("Kriging kind", (int)p.m_kriging_kind);
		print_param("Mean kind", (int)p.m_mean_kind);
		print_param("Seed", p.m_seed);
		print_param("Min neighbours", p.m_min_neighbours);
	}

	void print_params(const ik_params_t & p)
	{
		std::string values         ("Indicators:\t");
		std::string cov_type       ("Covariance:\t");
		std::string ranges         ("Ranges    :\t");
		std::string angles         ("Angles    :\t");
		std::string sills          ("Sills     :\t");
		std::string nuggets        ("Nuggets   :\t");
		std::string radiuses       ("Radiuses  :\t");
		std::string maxneighbours  ("MaxNeighb :\t");
		std::string probs          ("MargProb  :\t");

		for (indicator_index_t idx = 0; idx < p.m_category_count; ++idx)
		{
			values += std::to_string(idx) + "\t";
			cov_type += std::to_string(p.m_covariances[idx]) + "\t";

			std::ostringstream rng_oss;
			rng_oss << p.m_ranges[idx][0] << ":" << p.m_ranges[idx][1] << ":" << p.m_ranges[idx][2] << "\t";
			ranges += rng_oss.str();

			std::ostringstream ang_oss;
			ang_oss << p.m_angles[idx][0] << ":" << p.m_angles[idx][1] << ":" << p.m_angles[idx][2] << "\t";
			angles += ang_oss.str();

			sills += std::to_string(p.m_sills[idx]) + "\t";
			nuggets += std::to_string(p.m_nuggets[idx]) + "\t";

			std::ostringstream rad_oss;
			rad_oss << p.m_radiuses[idx][0] << ":" << p.m_radiuses[idx][1] << ":" << p.m_radiuses[idx][2] << "\t";
			radiuses += rad_oss.str();

			maxneighbours += std::to_string(p.m_neighbour_limits[idx]) + "\t";
			probs += std::to_string(p.m_marginal_probs[idx]) + "\t";
		}

		write(values);
		write(cov_type);
		write(ranges);
		write(angles);
		write(sills);
		write(nuggets);
		write(radiuses);
		write(maxneighbours);
		write(probs);
	}

	void print_params(const indicator_params_t * p, int param_count, const mean_t * marginal_probs)
	{
		std::string values         ("Indicators:\t");
		std::string cov_type       ("Covariance:\t");
		std::string ranges         ("Ranges    :\t");
		std::string angles         ("Angles    :\t");
		std::string sills          ("Sills     :\t");
		std::string nuggets        ("Nuggets   :\t");
		std::string radiuses       ("Radiuses  :\t");
		std::string maxneighbours  ("MaxNeighb :\t");
		std::string probs          ("MargProb  :\t");

		for (indicator_index_t idx = 0; idx < param_count; ++idx)
		{
			values += std::to_string(idx) + "\t";
			cov_type += std::to_string(p[idx].cov_type) + "\t";

			std::ostringstream rng_oss;
			rng_oss << p[idx].ranges[0] << ":" << p[idx].ranges[1] << ":" << p[idx].ranges[2] << "\t";
			ranges += rng_oss.str();

			std::ostringstream ang_oss;
			ang_oss << p[idx].angles[0] << ":" << p[idx].angles[1] << ":" << p[idx].angles[2] << "\t";
			angles += ang_oss.str();

			sills += std::to_string(p[idx].sill) + "\t";
			nuggets += std::to_string(p[idx].nugget) + "\t";

			std::ostringstream rad_oss;
			rad_oss << p[idx].radiuses[0] << ":" << p[idx].radiuses[1] << ":" << p[idx].radiuses[2] << "\t";
			radiuses += rad_oss.str();

			maxneighbours += std::to_string(p[idx].max_neighbours) + "\t";
			if (marginal_probs != nullptr)
				probs += std::to_string(marginal_probs[idx]) + "\t";
			else
				probs += "N/A\t";
		}

		write(values);
		write(cov_type);
		write(ranges);
		write(angles);
		write(sills);
		write(nuggets);
		write(radiuses);
		write(maxneighbours);
		write(probs);
	}
}
