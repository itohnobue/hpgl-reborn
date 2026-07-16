#ifndef __SELECT_H__D393AF18_8E49_4668_8C3F_DCE536469498
#define __SELECT_H__D393AF18_8E49_4668_8C3F_DCE536469498

#include <cstdio>
#include <cstdlib>

namespace hpgl
{
	template<typename source_t, typename indices_t, typename dest_t>
	inline void select(const source_t & src, const indices_t & indices, dest_t & dest)
	{
		dest.resize(indices.size());
		for (size_t i = 0, end_i = indices.size(); i < end_i; ++i)
		{
			if (indices[i] < 0) {
				fprintf(stderr, "HPGL: select: Negative index in select\n");
				abort();
			}
			if (static_cast<size_t>(indices[i]) >= src.size()) {
				fprintf(stderr, "HPGL: select: index %zu out of bounds for source size %zu\n",
					static_cast<size_t>(indices[i]), static_cast<size_t>(src.size()));
				abort();
			}
			dest[i] = src[indices[i]];
		}
	}
}

#endif //__SELECT_H__D393AF18_8E49_4668_8C3F_DCE536469498