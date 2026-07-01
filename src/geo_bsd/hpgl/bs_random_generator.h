#ifndef __BS_RANDOM_GENERATOR_H__E14CD484_0326_4847_A950_AEBF0B62D94D__
#define __BS_RANDOM_GENERATOR_H__E14CD484_0326_4847_A950_AEBF0B62D94D__

#include <random>
#include "typedefs.h"

namespace hpgl
{
	class mt_random_generator_t
	{
		std::mt19937 gen;
	public:
		mt_random_generator_t(){};
		mt_random_generator_t(int64_t seed)
		{ seed_from_int64(seed); }
		void seed(int64_t seed);
		long int operator() (long int N);
		double operator()();
	private:
		void seed_from_int64(int64_t seed);
	};
}

#endif //__BS_RANDOM_GENERATOR_H__E14CD484_0326_4847_A950_AEBF0B62D94D__