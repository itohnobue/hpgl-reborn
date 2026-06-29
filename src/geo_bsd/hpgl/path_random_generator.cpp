#include "stdafx.h"
#include "path_random_generator.h"
#include "bs_assert.h"
#include "output.h"

using namespace hpgl;

long int  primes [] = 
{
#include "primes.txt"
, -1
};

/// Returns the largest prime in the static primes[] table
/// (the last element before the -1 sentinel).
static long int get_max_prime()
{
	static long int max_prime = [](){
		long int max_val = 0;
		for (int i = 0; primes[i] > 0; ++i)
			max_val = primes[i];
		return max_val;
	}();
	return max_prime;
}

struct rnd_state_t
{
	long int A;
	long int C;
	long int M;
	long int R;
};

void init_rnd(rnd_state_t & state, long int size, long int seed)
{
	const long int MAX_PRIME = get_max_prime();

	// Runtime bounds check: replaces BS_ASSERT (compiled out in release builds)
	if (size > MAX_PRIME)
	{
		std::ostringstream oss;
		oss << "Grid size " << size << " exceeds max supported "
		    << MAX_PRIME << ". Clamping size.";
		LOGWARNING(oss.str());
		size = MAX_PRIME;
	}

	long int idx = 0;
	state.A = 1;
	state.M = size;
	while (primes[idx] > 0)
	{
		if (primes[idx] > size)
			break;
		if (size % primes[idx] == 0)
				state.A *= primes[idx];
		++idx;
	}
	// Runtime guard: replaces BS_ASSERT (compiled out in release builds)
	if (primes[idx] <= 0)
	{
		// Prime table exhausted — all known primes are <= size.
		// This is expected for sizes equal to MAX_PRIME (the largest
		// prime), and clamped sizes above MAX_PRIME.
		// The generator quality may degrade for extremely large grids.
		LOGWARNING("Prime table exhausted for grid size. Generator may be suboptimal.");
	}

	if (state.M % 4 == 0)
		state.A *= 2;

	state.A += 1;

	idx = 0;
	while (primes[idx+1] > 0 && primes[idx+1] <= size / 3)
		++idx;
	
	for (int k = idx; k >= 0; k--)
	{
		state.C = primes[k];
		if(floor(double(size) / double(state.C)) != (double(size) / double(state.C)))
		{
			break;
		}
	}

	// Guard: ensure gcd(C, M) = 1 for full-period LCG.
	// When all primes up to size/3 divide size (happens for small even sizes),
	// the loop above falls through with C=2. For even M, gcd(2, M) ≥ 2,
	// violating the Hull-Dobell condition and producing a degenerate generator.
	// Search forward into higher primes for one coprime to M.
	if (state.C % 2 == 0 && state.M % 2 == 0)
	{
		for (int k = idx + 1; primes[k] > 0; ++k)
		{
			if (size % primes[k] != 0)
			{
				state.C = primes[k];
				break;
			}
		}
		// Ultimate fallback: 3 is the smallest odd prime; always coprime
		// to powers of 2 (the only case where all tested primes divide size).
		if (state.C % 2 == 0 && state.M % 2 == 0)
			state.C = 3;
	}

	state.R = seed % size;
}

void advance_rnd(rnd_state_t & state)
{
	int prev_r = state.R;
	state.R = ((long long) state.A * (long long) state.R + (long long) state.C) % (long long) state.M;
	BS_ASSERT(state.R >= 0 && state.R < state.M)(state.R)(prev_r);
}

namespace hpgl
{
	class path_random_generator_t::Impl
	{
	public:
		rnd_state_t m_rnd_state;
	};

	path_random_generator_t::path_random_generator_t()
		: m_impl(new Impl())
	{
	}

	path_random_generator_t::path_random_generator_t(int size, int seed)
		: m_impl(new Impl())
	{
		init_rnd(m_impl->m_rnd_state, size, seed);
	}

	void path_random_generator_t::init(int size, int seed)
	{
		init_rnd(m_impl->m_rnd_state, size, seed);
	}

	int path_random_generator_t::next()
	{
		int result = m_impl->m_rnd_state.R;
		advance_rnd(m_impl->m_rnd_state);
		return result;
	}

	path_random_generator_t::~path_random_generator_t()
	{}
}
