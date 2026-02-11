/*
   Copyright 2009 HPGL Team
   This file is part of HPGL (High Perfomance Geostatistics Library).
   HPGL is free software: you can redistribute it and/or modify it under the terms of the BSD License.
   You should have received a copy of the BSD License along with HPGL.

*/


#ifndef __LVM_DATA_H__6C233713_4E36_4BBB_8EAA_B5F2A8297011
#define __LVM_DATA_H__6C233713_4E36_4BBB_8EAA_B5F2A8297011

#include "typedefs.h"
#include "hpgl_exception.h"
#include <memory>

namespace hpgl
{
	class cont_lvm_data_t
	{
		std::shared_ptr<std::vector<mean_t>> m_data_holder;
		std::vector<mean_t> * m_data;
	public:
		cont_lvm_data_t();

		void assign_transfer(std::shared_ptr<std::vector<mean_t>> data);
		void assign_transfer(std::vector<mean_t> * data);
		void assign_copy(const std::vector<mean_t> & data);

		inline mean_t get_at(node_index_t)const;
		inline void set_at(node_index_t, mean_t value);
	};

	class indicator_lvm_data_t
	{
	public:
		std::vector<std::shared_ptr<std::vector<mean_t>>> m_data_holder;
		std::vector<std::vector<mean_t>* > m_data;
	public:
		indicator_lvm_data_t();

		void assign(const std::vector<std::shared_ptr<std::vector<mean_t>>> & data);
		void assign_transfer(const std::vector<std::vector<mean_t>*> & data);
		void assign_copy(std::vector<std::shared_ptr<std::vector<mean_t>>> & data);
		void assign_copy(const std::vector<std::vector<mean_t>*> & data);

		inline mean_t get_at(node_index_t, indicator_index_t)const;
		inline void set_at(node_index_t, indicator_index_t, mean_t value);
		inline std::shared_ptr<std::vector<mean_t>> for_indicator(indicator_index_t)const;

		size_t size()const
		{
			return m_data_holder.size();
		}
	};

	// SECURITY FIX: Added bounds checking for node index
	mean_t cont_lvm_data_t::get_at(node_index_t node)const
	{
		if (m_data)
		{
			// SECURITY FIX: Added bounds check before vector access
			if (static_cast<size_t>(node) >= m_data->size())
			{
				throw hpgl_exception("cont_lvm_data_t::get_at", "Node index out of bounds.");
			}
			return (*m_data)[node];
		}
		else
		{
			throw hpgl_exception("cont_lvm_data_t::get_at", "No data.");
		}
	}

	// SECURITY FIX: Added bounds checking for node index
	void cont_lvm_data_t::set_at(node_index_t node, mean_t value)
	{
		if (m_data)
		{
			// SECURITY FIX: Added bounds check before vector access
			if (static_cast<size_t>(node) >= m_data->size())
			{
				throw hpgl_exception("cont_lvm_data_t::set_at", "Node index out of bounds.");
			}
			(*m_data)[node] = value;
		}
		else
		{
			throw hpgl_exception("cont_lvm_data::set_at", "No data.");
		}
	}

	// SECURITY FIX: Added bounds checking for node index and indicator index
	mean_t indicator_lvm_data_t::get_at(node_index_t node, indicator_index_t indicator) const
	{
		// SECURITY FIX: Check indicator bounds before access
		if (static_cast<size_t>(indicator) >= m_data.size())
		{
			throw hpgl_exception("indicator_lvm_data_t::get_at", "Indicator index out of bounds.");
		}

		if (m_data[indicator])
		{
			// SECURITY FIX: Added bounds check before vector access
			if (static_cast<size_t>(node) >= m_data[indicator]->size())
			{
				throw hpgl_exception("indicator_lvm_data_t::get_at", "Node index out of bounds.");
			}
			return (*(m_data[indicator]))[node];
		}
		else
		{
			throw hpgl_exception("indicator_lvm_data_t::get_at", "No data.");
		}
	}

	// SECURITY FIX: Added bounds checking for node index and indicator index
	void indicator_lvm_data_t::set_at(node_index_t node, indicator_index_t indicator, mean_t value)
	{
		// SECURITY FIX: Check indicator bounds before access
		if (static_cast<size_t>(indicator) >= m_data.size())
		{
			throw hpgl_exception("indicator_lvm_data_t::set_at", "Indicator index out of bounds.");
		}

		if (m_data[indicator])
		{
			// SECURITY FIX: Added bounds check before vector access
			if (static_cast<size_t>(node) >= m_data[indicator]->size())
			{
				throw hpgl_exception("indicator_lvm_data_t::set_at", "Node index out of bounds.");
			}
			(*m_data[indicator])[node] = value;
		}
		else
		{
			throw hpgl_exception("indicator_lvm_data_t::set_at", "No data.");
		}

	}
}

#endif //__LVM_DATA_H__6C233713_4E36_4BBB_8EAA_B5F2A8297011