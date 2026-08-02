#ifndef __SGS_PARAMS_H__DD8E93BC_D7B4_4C04_A0AC_6FD6AF563A4C__
#define __SGS_PARAMS_H__DD8E93BC_D7B4_4C04_A0AC_6FD6AF563A4C__

#include <stdint.h>
#include "sk_params.h"

namespace hpgl
{
	enum kriging_kind_t 
	{
		KRIG_ORDINARY=0, KRIG_SIMPLE=1
	};

	class sgs_params_t : public sk_params_t
	{
	public:
		sgs_params_t();
		kriging_kind_t m_kriging_kind;
		int64_t m_seed;
		// 2-M-1(c): m_mean_kind is the descriptive mean-mode field. It is
		// set by the C API and READ by sequential_gaussian_simulation (the
		// auto/user stationary-mean branch selection); the LVM mode
		// (e_mean_varying) is selected by the separate LVM entry point. The
		// former m_lvm member was removed — it duplicated the mean_data
		// parameter passed to sequential_gaussian_simulation_lvm and was
		// read by no algorithm.
		mean_kind_t m_mean_kind;
		int m_min_neighbours; // GSLIB ndmin: minimum conditioning data per node; nodes with fewer are left unsimulated (F-14).
	};

	inline std::ostream & operator<<(std::ostream & o, const sgs_params_t & p)
	{
		o << (sk_params_t)p << "\tSeed: " << p.m_seed << "\n" << "Mean type: ";
		switch (p.m_mean_kind)
		{
		case mean_kind_t::e_mean_stationary_auto:
		case mean_kind_t::e_mean_stationary:
			o << "stationary";
			break;
		case mean_kind_t::e_mean_varying:
			o << "varying";
			break;
		default:
			o << "unknown";
		}
		return o << "\n";
	}
}
#endif //__SGS_PARAMS_H__DD8E93BC_D7B4_4C04_A0AC_6FD6AF563A4C__

