#include "stdafx.h"
#include "sugarbox_grid.h"

namespace hpgl
{


	void sugarbox_grid_t::init(size_type x, size_type y, size_type z)
	{
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
			
		return sugarbox_location_t(index % m_x, index % (m_x * m_y) / m_x, index / (m_x * m_y) );
	}

	sugarbox_grid_t::~sugarbox_grid_t()
	{
	}
}//namespace hpgl
