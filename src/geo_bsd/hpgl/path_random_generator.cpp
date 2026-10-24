#include "stdafx.h"
#include "path_random_generator.h"
#include "hpgl_exception.h"
#include <algorithm>
#include <cstdint>
#include <random>
#include <sstream>
#include <utility>

using namespace hpgl;

// F-388: stream discriminant for the path engine's seed_seq. The Gaussian
// deviates in both simulation drivers come from a mt_random_generator_t seeded
// with the same int64_t (sequential_simulation.h:223-224,
// sequential_indicator_simulation.cpp:62), so seeding THIS engine with the bare
// seed replays the same stream: key[j] drew the same value as deviate[j], and
// because the path is the sort order of those keys, each node's deviate became
// a deterministic monotone function of its own key (a 2-node grid gave node 0
// the lower deviate in 200/200 seeds — means 0.3348/0.6699 instead of 1/2).
// Mixing a fixed constant into a seed_seq gives the path engine a distinct
// state while keeping the permutation a pure, reproducible function of the
// seed. Any non-zero constant works; "PATH" is used so the intent is legible.
static const uint32_t PATH_STREAM_DISCRIMINANT = 0x50415448u;

/// Fisher-Yates shuffle (sort-by-random-key variant).
/// Assigns a uniform random double key to each index [0, size) and sorts
/// by key, producing a uniformly random permutation — GSLIB's approach.
/// Replaces the prior LCG which produced at most M orderings out of M!
/// possible permutations.
static void build_shuffled_path(std::vector<int> & path, int size, int64_t seed)
{
	// Negative seeds produce indeterminate behavior in mt19937 (unsigned cast);
	// reject early with a clear error rather than silently remapping.
	if (seed < 0) {
		std::ostringstream oss;
		oss << "Seed value " << seed << " must be non-negative.";
		throw hpgl_exception("path_random_generator_t", oss.str());
	}

	// F-388: own stream — see PATH_STREAM_DISCRIMINANT above. Splitting the
	// 64-bit seed into halves matches bs_random_generator.cpp's convention so
	// no entropy is dropped, and the constant keeps the two engines apart.
	std::seed_seq path_seq{
		static_cast<uint32_t>(seed & 0xFFFFFFFFLL),
		static_cast<uint32_t>((seed >> 32) & 0xFFFFFFFFLL),
		PATH_STREAM_DISCRIMINANT
	};
	std::mt19937 path_gen(path_seq);
	std::uniform_real_distribution<double> path_key(0.0, 1.0);
	path.resize(size);

	std::vector<std::pair<double, int> > pairs;
	pairs.reserve(size);
	for (int i = 0; i < size; ++i)
		pairs.emplace_back(path_key(path_gen), i);

	std::sort(pairs.begin(), pairs.end());

	for (int i = 0; i < size; ++i)
		path[i] = pairs[i].second;
}

namespace hpgl
{
	class path_random_generator_t::Impl
	{
	public:
		std::vector<int> m_path;
		size_t m_pos;
	};

	path_random_generator_t::path_random_generator_t()
		: m_impl(new Impl())
	{
	}

	path_random_generator_t::path_random_generator_t(int size, int64_t seed)
		: m_impl(new Impl())
	{
		build_shuffled_path(m_impl->m_path, size, seed);
		m_impl->m_pos = 0;
	}

	void path_random_generator_t::init(int size, int64_t seed)
	{
		build_shuffled_path(m_impl->m_path, size, seed);
		m_impl->m_pos = 0;
	}

	int path_random_generator_t::next()
	{
		if (m_impl->m_pos >= m_impl->m_path.size())
			m_impl->m_pos = 0;
		return m_impl->m_path[m_impl->m_pos++];
	}

	path_random_generator_t::~path_random_generator_t()
	{}
}
