#include "stdafx.h"
#include "bs_random_generator.h"
#include <random>
#include <cstdio>
#include <cstdlib>

namespace hpgl
{
	void mt_random_generator_t::seed_from_int64(int64_t seed)
	{
		// Split 64-bit seed into two 32-bit halves and combine via seed_seq
		// to incorporate all entropy into the mt19937 state.
		// Direct cast to mt19937::result_type (uint32_t) silently drops
		// the upper 32 bits — seeds differing only there produce identical
		// random sequences.
		std::seed_seq seq{
			static_cast<uint32_t>(seed & 0xFFFFFFFF),
			static_cast<uint32_t>((seed >> 32) & 0xFFFFFFFF)
		};
		gen.seed(seq);
	}

	void mt_random_generator_t::seed(int64_t seed)
	{
		seed_from_int64(seed);
	}

	long int mt_random_generator_t::operator ()(long int N)
	{
		if (N <= 0) { fprintf(stderr, "HPGL FATAL: mt_random_generator_t: N must be positive, got %ld\n", N); abort(); }
		std::uniform_int_distribution<long int> dist(0, N - 1);
		return dist(gen);
	}

	double mt_random_generator_t::operator()()
	{
		std::uniform_real_distribution<double> dist(0.0, 1.0);
		return dist(gen);
	}
}
