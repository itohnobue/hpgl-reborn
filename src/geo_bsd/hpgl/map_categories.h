#ifndef __MAP_CATEGORIES_H__85188541_A46B_492D_9CA8_B64EE3307DF1
#define __MAP_CATEGORIES_H__85188541_A46B_492D_9CA8_B64EE3307DF1

#include <sstream>
#include <cstdio>
#include <cstdlib>

namespace hpgl
{
	template<typename T>
	void map_categories(std::vector<T> & data, std::vector<indicator_value_t> & indicator_values)
	{
		for (size_t i = 0; i < data.size(); ++i)
		{
			bool mapped = false;
			for (size_t j = 0; j < indicator_values.size(); ++j)
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
				fprintf(stderr, "HPGL FATAL: map_categories: unexpected value at index %zu\n", i);
				abort();
			}
		}
	}

	template<typename T>
	void map_categories(indicator_property_array_t & data, std::vector<indicator_value_t> & indicator_values)
	{
		for (int i = 0; i < data.size(); ++i)
		{
			bool mapped = false;
			for (size_t j = 0; j < indicator_values.size(); ++j)
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
				fprintf(stderr, "HPGL FATAL: map_categories: unexpected value at index %d\n", i);
				abort();
			}
		}
	}
}

#endif //__MAP_CATEGORIES_H__85188541_A46B_492D_9CA8_B64EE3307DF1