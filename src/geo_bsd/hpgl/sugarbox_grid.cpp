#include "stdafx.h"
#include "sugarbox_grid.h"
#include "property_array.h"

namespace hpgl
{


	void sugarbox_grid_t::init(size_type x, size_type y, size_type z)
	{
		HPGL_CHECK(x > 0 && y > 0 && z > 0,
			"sugarbox_grid_t::init: grid dimensions must be strictly positive");
		m_x = x;
		m_y = y;
		m_z = z;
	}

	node_index_t sugarbox_grid_t::size()const
	{
		return static_cast<node_index_t>(static_cast<long long>(m_x) * m_y * m_z);
	}


	sugarbox_location_t sugarbox_grid_t::operator[](node_index_t index)const
	{
		HPGL_CHECK(m_x > 0, "sugarbox_grid_t::operator[]: m_x must be positive");
		HPGL_CHECK(m_x * m_y > 0, "sugarbox_grid_t::operator[]: m_x * m_y must be positive");
		return sugarbox_location_t(index % m_x, index % (m_x * m_y) / m_x, index / (m_x * m_y) );
	}

	sugarbox_grid_t::~sugarbox_grid_t() = default;
}//namespace hpgl
