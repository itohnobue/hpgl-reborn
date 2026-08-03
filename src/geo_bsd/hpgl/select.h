#ifndef __SELECT_H__D393AF18_8E49_4668_8C3F_DCE536469498
#define __SELECT_H__D393AF18_8E49_4668_8C3F_DCE536469498

#include <cstdio>
#include <optional>
#include <sstream>
#include "typedefs.h"
#include "hpgl_exception.h"

namespace hpgl
{
	// Forward declarations for mean-provider types that have operator[]
	// but no container-level size(), and thus need specialized select overloads.
	class single_mean_t;
	class no_mean_t;

	namespace detail
	{
		// Consolidated index-validation for all select() overloads (F-01).
		//
		// The 3 prior fixes (537e2ae, 6b2d2cd, 9ab4dd0) each copy-pasted a
		// diverging validation block that ended with abort() — uncatchable
		// (SIGABRT) on a path that is reachable by a bug in index generation
		// (path generator / neighbour lookup returning a bad index). The
		// validation is now a single helper that all overloads call, so a
		// future overload cannot re-introduce a divergent copy.
		//
		// The upper-bounds check is only possible when the source exposes a
		// container-level size (generic overload). The single_mean_t /
		// no_mean_t / float* overloads have operator[] but no size() — the
		// documented contract is "caller guarantees m_data is valid for all
		// indices 0..grid_size-1" (mean_provider.h:33), so the negative-index
		// check is the only guard available there. Pass std::nullopt for
		// src_size to run only the negative check (one code path, no
		// check-divergence between overloads).
		inline void select_check_indices(
				node_index_t index,
				const std::optional<size_t> & src_size,
				const char * what)
		{
			if (index < 0)
			{
				throw hpgl_exception("select",
					std::string("Negative index in select (") + what + ")");
			}
			if (src_size.has_value() && static_cast<size_t>(index) >= *src_size)
			{
				std::ostringstream oss;
				oss << "index " << static_cast<size_t>(index)
					<< " out of bounds for source size " << *src_size
					<< " (" << what << ")";
				throw hpgl_exception("select", oss.str());
			}
		}
	}

	template<typename source_t, typename indices_t, typename dest_t>
	inline void select(const source_t & src, const indices_t & indices, dest_t & dest)
	{
		dest.resize(indices.size());
		for (size_t i = 0, end_i = indices.size(); i < end_i; ++i)
		{
			detail::select_check_indices(indices[i], src.size(), "generic source");
			dest[i] = src[indices[i]];
		}
	}

	// Specialized select for single_mean_t — mean provider has operator[] for
	// any valid grid index; no container-level size to bounds-check against.
	template<typename indices_t, typename dest_t>
	inline void select(const single_mean_t & src, const indices_t & indices, dest_t & dest)
	{
		dest.resize(indices.size());
		for (size_t i = 0, end_i = indices.size(); i < end_i; ++i)
		{
			detail::select_check_indices(indices[i], std::nullopt, "single_mean_t");
			dest[i] = src[indices[i]];
		}
	}

	// Specialized select for no_mean_t — same reasoning as single_mean_t.
	template<typename indices_t, typename dest_t>
	inline void select(const no_mean_t & src, const indices_t & indices, dest_t & dest)
	{
		dest.resize(indices.size());
		for (size_t i = 0, end_i = indices.size(); i < end_i; ++i)
		{
			detail::select_check_indices(indices[i], std::nullopt, "no_mean_t");
			dest[i] = src[indices[i]];
		}
	}

	// Specialized select for raw pointer mean arrays — no bounds check possible.
	template<typename indices_t, typename dest_t>
	inline void select(const float * const& src, const indices_t & indices, dest_t & dest)
	{
		dest.resize(indices.size());
		for (size_t i = 0, end_i = indices.size(); i < end_i; ++i)
		{
			detail::select_check_indices(indices[i], std::nullopt, "float*");
			dest[i] = src[indices[i]];
		}
	}

}

#endif //__SELECT_H__D393AF18_8E49_4668_8C3F_DCE536469498
