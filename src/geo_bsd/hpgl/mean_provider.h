#ifndef MEAN_PROVIDER_H_INCLUDED_T9TUHNORNUVKDSFJP3O4NP2CQ0CNPASDFJPO23RV238V
#define MEAN_PROVIDER_H_INCLUDED_T9TUHNORNUVKDSFJP3O4NP2CQ0CNPASDFJPO23RV238V

#include "typedefs.h"

namespace hpgl
{
	class no_mean_t
	{
	public:				
		inline double operator[](node_index_t index)const
		{
			return 0;
		}
	};

	class single_mean_t
	{
		double m_mean;
		const double* m_data = nullptr;
	public:
		single_mean_t()
			: m_mean(0.0)
		{}

		single_mean_t(double mean)
			: m_mean(mean)
		{}

		// Per-node data constructor: m_data[i] is the local marginal probability
		// at node index i. When m_data is non-null, operator[] returns the per-node
		// value; otherwise it falls back to the global constant m_mean.
		// Caller guarantees m_data is valid for all indices 0..grid_size-1.
		single_mean_t(const double* data)
			: m_mean(0.0), m_data(data)
		{}

		inline double operator[](node_index_t index)const
		{
			if (m_data != nullptr)
				return m_data[index];
			return m_mean;
		}
	};

	void create_means(const std::vector<indicator_probability_t> & marginal_probs, std::vector<single_mean_t> & means);
	
}

#endif //MEAN_PROVIDER_H_INCLUDED_T9TUHNORNUVKDSFJP3O4NP2CQ0CNPASDFJPO23RV238V
