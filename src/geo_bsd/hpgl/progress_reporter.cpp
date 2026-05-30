#include "stdafx.h"

#include "progress_reporter.h"
#include "output.h"
#include <chrono>

int progress_bar_length = 20;

void print_progressbar(int percent)
{	
	std::cout << "\r[";
	for (int i = 0; i < 20; ++i)
	{
		if (percent / 100.0 > i / 20.0)
			std::cout << "*";
		else
			std::cout << " ";
	}
	std::cout << "] " << percent << "%                ";
	std::cout.flush();
}


namespace hpgl
{
	void progress_reporter_t::set_iteration_count(long iteration_count)
	{
		m_iterations = iteration_count;
		m_counter = 0;
		m_delta = m_iterations / 10;
		if (m_delta == 0)
			m_delta = 1;
	}

	progress_reporter_t::progress_reporter_t(long n_iterations)
	{
		set_iteration_count(n_iterations);		
	}

	void progress_reporter_t::start()
	{
		update_progress("", 0);
		m_start = std::chrono::high_resolution_clock::now();
	}

	void progress_reporter_t::start(long n_iterations)
	{
		set_iteration_count(n_iterations);		
		start();
	}

	void progress_reporter_t::next_lap()
	{
		//BOOST_INTERLOCKED_INCREMENT(&m_counter);
		m_counter++;
		if (m_counter % m_delta == 0)
		{
		//	boost::mutex::scoped_lock lock(m_mutex);
			int perc = (int) 100.0 * m_counter / m_iterations;
			if (perc > 0)
				update_progress("", perc);
			//			write(boost::format("%1%%%...") % ((int) 100.0 * m_counter / m_iterations));
			//			std::cout << (int) 100.0 * m_counter / m_iterations << "%... ";
			//			std::cout.flush();
			//print_progressbar(m_counter * 100 / m_iterations);
		}		
	}

	void progress_reporter_t::stop()
	{
		m_end = std::chrono::high_resolution_clock::now();
		write("\n");
	}

	double progress_reporter_t::iterations_per_second()
	{
		return m_iterations / duration();
	}

	double progress_reporter_t::duration()
	{
		return std::chrono::duration<double>(m_end - m_start).count();
	}
}
