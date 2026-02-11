#include "stdafx.h"
#include "bs_random_generator.h"
#include <random>

namespace hpgl
{
	void mt_random_generator_t::seed(long int seed)
	{
		gen.seed(static_cast<std::mt19937::result_type>(seed));
	}

	long int mt_random_generator_t::operator ()(long int N)
	{
		std::uniform_int_distribution<long int> dist(0, N - 1);
		return dist(gen);
	}

	double mt_random_generator_t::operator()()
	{
		std::uniform_real_distribution<double> dist(0.0, 1.0);
		return dist(gen);
	}
}
