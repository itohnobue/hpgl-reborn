#ifndef __MAP_CATEGORIES_H__85188541_A46B_492D_9CA8_B64EE3307DF1
#define __MAP_CATEGORIES_H__85188541_A46B_492D_9CA8_B64EE3307DF1

#include <sstream>

namespace hpgl
{
	template<typename T>
	void map_categories(std::vector<T> & data, std::vector<indicator_value_t> & indicator_values)
	{
		for (int i = 0; i < data.size(); ++i)
		{
			bool mapped = false;
			for (int j = 0; j < indicator_values.size(); ++j)
			{
				if (data[i] == indicator_values[j])
				{
					data[i] = j;
					mapped = true;
					break;
				}
			}
			if (!mapped)
			{
				std::ostringstream oss;
				oss << "Unexpected value: " << data[i] << ".";
				throw hpgl_exception("map_categories", oss.str());
			}
		}
	}

	template<typename T>
	void map_categories(indicator_property_array_t & data, std::vector<indicator_value_t> & indicator_values)
	{
		for (int i = 0; i < data.size(); ++i)
		{
			bool mapped = false;
			for (int j = 0; j < indicator_values.size(); ++j)
			{
				if (data.get_at(i) == indicator_values[j])
				{
					data.set_at(i, j);
					mapped = true;
					break;
				}
			}
			if (!mapped)
			{
				std::ostringstream oss;
				oss << "Unexpected value: " << data.get_at(i) << ".";
				throw hpgl_exception("map_categories", oss.str());
			}
		}
	}
}

#endif //__MAP_CATEGORIES_H__85188541_A46B_492D_9CA8_B64EE3307DF1