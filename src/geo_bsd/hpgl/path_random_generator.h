#ifndef __PATH_RANDOM_GENERATOR_H__907328AB_FE6E_437B_9725_65FD1119284A__
#define __PATH_RANDOM_GENERATOR_H__907328AB_FE6E_437B_9725_65FD1119284A__

#include <cstdint>
#include <memory>
#include <vector>

namespace hpgl
{
	class path_random_generator_t
	{
		class Impl;
		std::shared_ptr<Impl> m_impl;
	public:
		path_random_generator_t();
		void init(int size, int64_t seed);
		path_random_generator_t(int size, int64_t seed);
		~path_random_generator_t();

		int next();
	};
}

#endif //__PATH_RANDOM_GENERATOR_H__907328AB_FE6E_437B_9725_65FD1119284A__
