#ifndef __VAR_RADIX_UTILS_H__89D51E7C_755F_44CE_97B4_2D086B4EAEC3
#define __VAR_RADIX_UTILS_H__89D51E7C_755F_44CE_97B4_2D086B4EAEC3

namespace hpgl
{
	template<typename T>
	inline T vr_to_dec(int r2, int r1, T d3, T d2, T d1)
	{
		return static_cast<T>(static_cast<long long>(d3) * r2 * r1 + static_cast<long long>(d2) * r1 + d1);
	}

	template<typename T>
	inline void dec_to_vr(int r2, int r1, T dec, T & d3, T & d2, T & d1)
	{
		long long r2r1 = static_cast<long long>(r2) * r1;
		d1 = dec % r1;
		d2 = static_cast<T>((static_cast<long long>(dec) % r2r1) / r1);
		d3 = static_cast<T>(static_cast<long long>(dec) / r2r1);
	}
}

#endif //__VAR_RADIX_UTILS_H__89D51E7C_755F_44CE_97B4_2D086B4EAEC3