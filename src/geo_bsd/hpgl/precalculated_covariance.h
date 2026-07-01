#ifndef __PRECALCULATED_COVARIANCE_H__8E9E179D_F626_4EEF_8096_9F075A40B740
#define __PRECALCULATED_COVARIANCE_H__8E9E179D_F626_4EEF_8096_9F075A40B740

#include "typedefs.h"
#include "geometry.h"
#include "var_radix_utils.h"
#include <climits>
#include <cstdio>
#include <cstdlib>

namespace hpgl
{
	/*
	Implements CovarianceModel concept.
	*/	
	class precalculated_covariances_t
	{
		int dx;
		int dy;
		int dz;
		int sx;
		int sy;
		int sz;
		std::vector<covariance_t> m_covariances;
		rect_3d_t<int> m_box;
	public:		
		precalculated_covariances_t()
		{}

		template<typename covariances_t, typename radiuses_t>
		precalculated_covariances_t(const covariances_t & cov, const radiuses_t & rs)
		{
			init(cov, rs);			
		}

		template<typename covariances_t, typename radiuses_t>
		void init(const covariances_t & cov, const radiuses_t & rs)
		{			
			int rx = rs[0];
			int ry = rs[1];
			int rz = rs[2];
			dx = - rx;
			dy = - ry;
			dz = - rz;
			sx = rx * 2 + 1;
			sy = ry * 2 + 1;
			sz = rz * 2 + 1;
			// Overflow-safe multiplication: avoid silent overflow in (size_t)sx * sy * sz
			size_t size = 0;
			{
				size_t sx_s = static_cast<size_t>(sx);
				size_t sy_s = static_cast<size_t>(sy);
				size_t sz_s = static_cast<size_t>(sz);
				if (sx_s > 0 && sy_s > 0 && sz_s > 0)
				{
					if (sx_s > SIZE_MAX / sy_s)
					{
						fprintf(stderr, "HPGL FATAL: precalculated_covariance: overflow in sxsy multiplication\n");
						abort();
					}
					size_t tmp = sx_s * sy_s;
					if (tmp > SIZE_MAX / sz_s)
					{
						fprintf(stderr, "HPGL FATAL: precalculated_covariance: overflow in tmp*sz multiplication\n");
						abort();
					}
					size = tmp * sz_s;
				}
			}
			// Guard: vr_to_dec returns int; index must fit in INT_MAX
			if (size > static_cast<size_t>(INT_MAX))
			{
				fprintf(stderr, "HPGL FATAL: precalculated_covariance: volume exceeds INT_MAX\n");
				abort();
			}
			m_covariances.resize(size);
			for (int z = 0; z < sz; ++z)
				for (int y = 0; y < sy; ++y)
					for (int x = 0; x < sx; ++x)
					{
						int index = vr_to_dec(sy, sx, z, y, x);
						double c1[] = {0.0, 0.0, 0.0};
						double c2[] = {static_cast<double>(x + dx), static_cast<double>(y + dy), static_cast<double>(z + dz)};
						m_covariances[static_cast<size_t>(index)] = cov(c1, c2);
					}
			m_box = rect_3d_t<int>(-rx, -ry, -rz, rx, ry, rz);
			if (static_cast<size_t>(m_box.volume_inclusive()) != size) { fprintf(stderr, "HPGL FATAL: precalculated_covariance: box volume mismatch\n"); abort(); }
		}

		template<typename coord_t>
		covariance_t operator()(const coord_t & c1, const coord_t & c2)const
		{
			double vec[] = {static_cast<double>(c2[0] - c1[0]), static_cast<double>(c2[1] - c1[1]), static_cast<double>(c2[2] - c1[2])}; //c2 - c1;

			if (m_box.has(vec))
			{
				int index = vr_to_dec(sy, sx, vec[2] - dz, vec[1] - dy, vec[0] - dx);
				return m_covariances[static_cast<size_t>(index)];
			}
			else
			{
				return 0;
			}

		}
	};
}

#endif //__PRECALCULATED_COVARIANCE_H__8E9E179D_F626_4EEF_8096_9F075A40B740
