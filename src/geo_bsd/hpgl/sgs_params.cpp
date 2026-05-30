#include "stdafx.h"
#include "sgs_params.h"

namespace hpgl
{
	sgs_params_t::sgs_params_t()
		: m_kriging_kind(kriging_kind_t::KRIG_ORDINARY)
		, m_seed(0)
		, m_mean_kind(mean_kind_t::e_mean_stationary_auto)
		, m_lvm(nullptr)
		, m_min_neighbours(0)
	{
	}
}