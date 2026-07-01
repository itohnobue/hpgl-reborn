#include "stdafx.h"
#include "path_random_generator.h"
#include "bs_random_generator.h"
#include "hpgl_exception.h"
#include <algorithm>
#include <sstream>
#include <utility>

using namespace hpgl;

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

	mt_random_generator_t gen(seed);
	path.resize(size);

	std::vector<std::pair<double, int> > pairs;
	pairs.reserve(size);
	for (int i = 0; i < size; ++i)
		pairs.emplace_back(gen(), i);

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
