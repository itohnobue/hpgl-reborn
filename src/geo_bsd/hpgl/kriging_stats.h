#ifndef KRIGING_STATS_H_INCLUDED_93485F0934MNKSDHKFANQ29F1IOEMNFIOSDAHFRQ23G4UJASDFDOBJRH21FSD
#define KRIGING_STATS_H_INCLUDED_93485F0934MNKSDHKFANQ29F1IOEMNFIOSDAHFRQ23G4UJASDFDOBJRH21FSD

namespace hpgl
{
	struct kriging_stats_t
	{
		// E-M57: ndmin-gate skip counter (GSLIB ndmin semantics — SGS only;
		// other kernels leave it 0). An all-ndmin-skipped run previously
		// produced NO programmatic failure signal: the count was stderr-only,
		// kriging_skipped excluded ndmin skips (the no-output warning could
		// not fire), and the stats struct had no field for it. The C API
		// getter (api.cpp hpgl_get_kriging_stats) copies the C-ABI fields by
		// name and ignores this one, so adding it here is ABI-safe; the
		// SGS no-output warning and the Python wrapper consume it.
		unsigned long m_points_ndmin_skipped;
		unsigned long m_points_calculated;
		unsigned long m_points_without_neighbours;
		unsigned long m_points_singularity;
		double m_mean;
		double m_speed_nps;

		// Default member initializers: every population site overwrites the
		// fields it owns; the zero defaults guarantee the fields it does NOT
		// touch (and the new m_points_ndmin_skipped on non-SGS kernels) are
		// never uninitialized when copied into the thread-local stats
		// (api.cpp set_kriging_stats). No site aggregate-initializes this
		// struct (all use field assignment), so the initializers are safe.
		kriging_stats_t()
			: m_points_ndmin_skipped(0)
			, m_points_calculated(0)
			, m_points_without_neighbours(0)
			, m_points_singularity(0)
			, m_mean(0)
			, m_speed_nps(0)
		{}
	};

	std::ostream & operator << (std::ostream & stream, const kriging_stats_t & stats);

}

#endif //KRIGING_STATS_H_INCLUDED_93485F0934MNKSDHKFANQ29F1IOEMNFIOSDAHFRQ23G4UJASDFDOBJRH21FSD
