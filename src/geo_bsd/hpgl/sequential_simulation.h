#ifndef SEQUENTIAL_SIMULATION_H_INCLUDED_DEU0932RMNPDVCPDZFIG03249MROPWEUFSDHVIAERT0934NTFMD0GN2304UNLKWEJOFDSZF0G32R
#define SEQUENTIAL_SIMULATION_H_INCLUDED_DEU0932RMNPDVCPDZFIG03249MROPWEUFSDHVIAERT0934NTFMD0GN2304UNLKWEJOFDSZF0G32R

#include "cdf_utils.h"
#include "path_random_generator.h"
#include "bs_random_generator.h"
#include "typedefs.h"
#include "sample.h"
#include "covariance_field.h"
#include "progress_reporter.h"
#include "combiner.h"
#include "kriging_interpolation.h"
#include "precalculated_covariance.h"
#include "neighbourhood_lookup.h"
// R2-06/R2-08: the ndmin gate must scan with the SAME evaluation bound and
// fallback distance bound as neighbour_lookup_t::find() (NEIGHBOUR_SCAN_WORK_CAP
// and fallback_d2_cap are declared in sugarbox_neighbour_lookup.h). Include it
// directly so the constants/helpers are visible at this template's definition
// point (no cycle: sugarbox_neighbour_lookup.h does not include this header).
#include "sugarbox_neighbour_lookup.h"
#include "is_informed_predicate.h"
#include "cov_model.h"
#include "kriging_stats.h"
#include "api.h"
#include <sstream>

namespace hpgl
{

	struct no_mask_t
	{
		int operator[](node_index_t /*node*/)const
		{
			return 1;
		}
	};

	namespace detail
	{
		// G-07 (CONFIRMED MEDIUM): max_neighbours is a COUNT cap and, before
		// this gate, the only bound on a per-node cost that is quadratic in
		// memory (2·n² doubles of workspace) and cubic-to-quadratic in time
		// (my_kriging_weights.h:284-312). The cap's admitted maximum
		// (api.cpp:135 = 10000) is ~1.6 GB of workspace and seconds per node,
		// i.e. a multi-day run on any real grid, and the whole band from
		// n = 1000 upward is silent (the Python layer only warns ABOVE 1000,
		// validation.py:731-742). This is the simulation path's pre-flight
		// work bound, in work units, checked before the covariance table and
		// the ordering path are built (E2-139 style: refuse, do not degrade).
		//
		// Calibration (adv-05 measurements on the shipped dylib): 7.79 ms per
		// node at n = 1000, 90.9 ms at n = 3000, 405.3 ms at n = 5000. Taking
		// the cubic model time ≈ N·n³·7.8e-12 s (conservative by ~2.4x at
		// n = 5000), a budget of 1e15 corresponds to ≈ 2 h of estimated work:
		// a 1000-node run at n = 10000 sits exactly on the boundary, and the
		// multi-day band (N ≥ 1e4 at n = 10000, or N ≥ 1e5 at n = 3000) is
		// refused. Every shipped simulation config (radius (5,5,3),
		// max_neighbours 4-12, grids ≤ ~1e5 cells, tests/python) is ~7 orders
		// of magnitude below the budget.
		constexpr double SIMULATION_SOLVE_WORK_BUDGET = 1.0e15;

		// E-M58/E-M84 (GSLIB ndmin semantics): count ORIGINAL conditioning
		// data inside the FULL search volume, before/independent of the
		// max_neighbours truncation. GSLIB sgsim srchsupr counts nclose over
		// all data in the search radius first and only then truncates to
		// ndmax (`if(nclose.lt.ndmin) go to 5` precedes the solve). The
		// truncated ws.indices set is insufficient: dense simulated
		// surroundings can crowd originals out of the top-N covariance-ranked
		// set even when the full radius contains >= ndmin originals, making
		// HPGL skip MORE nodes than sgsim.
		// R-17 (CONFIRMED MEDIUM, FIX-INTRODUCED): the gate geometry must
		// MATCH the solve geometry. find() admits only the covariance-
		// threshold box offsets (calc_cov_field: cov(0, offset) > C(0)/100
		// inside the search box), while the old gate counted the whole
		// ellipsoid (dx/rx)²+(dy/ry)²+(dz/rz)² <= 1 with no threshold —
		// diverging in both directions: it counted originals the solve never
		// serves (exp range 3, distance 8: ellipsoid passes 0.64<=1 while
		// cov 3.35e-4 < 0.01 excludes it — the gate could pass with fewer
		// serveable originals than ndmin) and it missed originals the solve
		// serves (spherical range 20, corner (10,10,0): ellipsoid excludes
		// 2>1 while cov 0.116 > 0.01 admits it). The gate now counts
		// originals at the SAME covariance-threshold box offsets find()
		// scans, plus — when the primary scan finds NO informed datum at
		// all, so find() fires its radius-bounded box fallback (pure-nugget
		// / degenerate models) — originals in the full search box (the
		// fallback's admission geometry). The live `informed` predicate is
		// the same one find() receives, so the fallback condition is
		// exact.
		// R-18 (CONFIRMED MEDIUM, FIX-INTRODUCED): the old ellipsoid scan
		// iterated the full (2r+1)³ box per node (9,261 lookups/node at
		// radius 10, 274,625 at 32, 99.25M at 231), even for nodes that
		// pass the gate. The threshold-capped candidate list is typically
		// far smaller than the box (cov > C(0)/100 truncates at the model
		// range), the primary scan early-exits once min_neighbours
		// originals are found, and the box scan runs only when the primary
		// found no datum at all (mirroring find()'s own fallback) with the
		// same early exit — the per-node scan is bounded by the R-17
		// admission geometry and never exceeds what the solve itself scans.
		// R2-06 (CONFIRMED MEDIUM): the R-18 claim was FALSE for failing
		// nodes — the primary scan walked the ENTIRE uncapped candidate
		// list (up to 99.25M offsets at r=231 — the threshold list is 91.3%
		// of the box for exp range-3 at r=10: 8,457/9,261) and the box
		// branch scanned the full (2r+1)³ per failing node, while find()'s
		// main scan is capped at NEIGHBOUR_SCAN_WORK_CAP=100000 — the gate
		// could cost ~500-1000x the solve and hang (~1e14 on 1M nodes).
		// The primary scan now carries the SAME evaluation cap as find()'s
		// main scan (every candidate counted, early exit only at
		// min_neighbours originals), and the box branch evaluates only the
		// SAME distance-bounded region find()'s fallback evaluates
		// (fallback_d2_cap — see R2-07). The "never exceeds the solve"
		// claim is now true: gate primary <= find() main scan (same cap),
		// gate box <= find() fallback (same region).
		// R2-08 (CONFIRMED MEDIUM): the cap is also a CORRECTNESS alignment,
		// not just a work bound. find()'s fallback fires on count==0 AFTER
		// the capped scan; an uncapped gate saw informed data ranked beyond
		// 1e5 (e.g. a distance-29 datum ranks 101,943 > 100,000 at r=231),
		// set primary_data=true and SKIPPED nodes the solve's capped scan +
		// fallback would serve with >= min_neighbours originals — R-17(b)
		// under-admission. With the identical capped prefix, primary_data
		// is true IFF find()'s main scan finds a datum, so the gate's
		// count==0 -> box-branch condition matches find()'s exactly and the
		// gate never skips a node the solve would serve.
		template <typename grid_t, typename informed_pred_t>
		inline bool has_min_original_data_in_radius(
			const grid_t & grid,
			const sugarbox_location_t & center,
			const sugarbox_search_ellipsoid_t & radiuses,
			const std::vector<sugarbox_vector_t> & candidates,
			const std::vector<unsigned char> & originally_informed,
			const informed_pred_t & informed,
			int min_neighbours,
			// R2-07: squared-distance bound of find()'s fallback scan
			// (fallback_d2_cap over the same radiuses); -1 = full box.
			long long fallback_d2_cap)
		{
			// Primary geometry — the covariance-threshold box offset list
			// find() scans (same list, same construction), capped at the
			// SAME evaluation budget as find()'s main scan (R-15
			// NEIGHBOUR_SCAN_WORK_CAP, counting every candidate examined
			// exactly like find() does — including out-of-grid ones, so
			// the evaluated prefix is identical). Count originals at those
			// offsets, early-exiting once min_neighbours are found.
			int found = 0;
			bool primary_data = false;
			int evals = 0;
			for (const sugarbox_vector_t & vec : candidates)
			{
				if (evals >= NEIGHBOUR_SCAN_WORK_CAP)
					break;   // find()'s main scan stops at the same point
				++evals;
				int index = grid.get_index(center + vec);
				if (index < 0)
					continue;
				if (informed(index))
					primary_data = true;
				if (originally_informed[static_cast<size_t>(index)])
				{
					++found;
					if (found >= min_neighbours)
						return true;
				}
			}
			// find()'s primary scan found at least one datum (original or
			// simulated), so the fallback does NOT fire and the solve serves
			// only primary matches: the originals it can serve are exactly
			// `found`, which is below min_neighbours — skip per the ndmin
			// contract (the solve would serve fewer than ndmin originals).
			if (primary_data)
				return false;
			// No datum at any primary offset → find() fires the
			// radius-bounded box fallback. Count originals in the SAME
			// distance-bounded region the fallback evaluates (R2-07:
			// fallback_d2_cap, -1 = full box), early-exiting. The
			// monotone break/continue mirrors find()'s fallback scan so
			// the evaluated region is identical.
			const int rx = static_cast<int>(radiuses[0]);
			const int ry = static_cast<int>(radiuses[1]);
			const int rz = static_cast<int>(radiuses[2]);
			for (int dz = -rz; dz <= rz && found < min_neighbours; ++dz)
			{
				const long long dz2 = static_cast<long long>(dz) * dz;
				if (fallback_d2_cap >= 0 && dz2 > fallback_d2_cap)
				{
					if (dz >= 0)
						break;   // remaining z-layers are all beyond the fallback's eval budget
					continue;
				}
				for (int dy = -ry; dy <= ry && found < min_neighbours; ++dy)
				{
					const long long ddy = dz2 + static_cast<long long>(dy) * dy;
					if (fallback_d2_cap >= 0 && ddy > fallback_d2_cap)
					{
						if (dy >= 0)
							break;   // remaining y-rows are all beyond the fallback's eval budget
						continue;
					}
					for (int dx = -rx; dx <= rx && found < min_neighbours; ++dx)
					{
						const long long d2 = ddy + static_cast<long long>(dx) * dx;
						if (fallback_d2_cap >= 0 && d2 > fallback_d2_cap)
						{
							if (dx >= 0)
								break;   // remaining x-columns are all beyond the fallback's eval budget
							continue;
						}
						int index = grid.get_index(center + sugarbox_vector_t(dx, dy, dz));
						if (index >= 0 && originally_informed[static_cast<size_t>(index)])
						{
							++found;
							if (found >= min_neighbours)
								return true;
						}
					}
				}
			}
			return found >= min_neighbours;
		}
	}

	template <typename grid_t, typename mean_provider_t, typename fallback_mean_provider_t, typename weight_calculator_t, typename mask_t>
	void do_sequential_gausian_simulation(
		cont_property_array_t & property,
		const grid_t & grid,
		const sgs_params_t & params,
		const mean_provider_t & mp,
		// R-5: separate fallback mean provider. The kriging estimate and the
		// failure fallback draw can need DIFFERENT means: the SGS-OK path
		// passes no_mean_t() as mp (GSLIB zero-mean normal-score semantics —
		// no (1−Σλᵢ)·mean term on the n<4 SK-downgraded estimate) while the
		// fallback must still draw N(gmean, 1.0) per GSLIB
		// (sgsim.for: `cmean = gmean; cstdev = 1.0`; M-29). KRIG_SIMPLE and
		// LVM pass the same provider for both.
		const fallback_mean_provider_t & fallback_mp,
		const weight_calculator_t & weight_calculator_sgs,
		const mask_t & mask)
	{

		if (property.size() != grid.size())
		{
			std::ostringstream oss;
			oss << "Property size '" << property.size() << "' is not equal to grid size '" << grid.size() << "'";
			throw hpgl_exception("do_sequential_gausian_simulation", oss.str());
		}

		// G-07: bound the run's ESTIMATED solve work before anything is
		// allocated. max_neighbours is a count cap on a cubic per-node cost,
		// so the admitted maximum is a multi-day run (see
		// detail::SIMULATION_SOLVE_WORK_BUDGET). Throw here — the C API catch
		// converts it to a clean error — rather than allocating the
		// covariance table and the ordering path first.
		{
			const size_t work_cap = static_cast<size_t>(NEIGHBOUR_WORK_CAP);
			const size_t capped_neighbours = params.m_max_neighbours < work_cap
				? params.m_max_neighbours
				: work_cap;
			if (capped_neighbours > 0)
			{
				const double n = static_cast<double>(capped_neighbours);
				const double estimated_work = static_cast<double>(property.size()) * n * n * n;
				if (estimated_work > detail::SIMULATION_SOLVE_WORK_BUDGET)
				{
					std::ostringstream oss;
					oss << "estimated kriging work exceeds the simulation budget: "
					    << static_cast<unsigned long>(property.size()) << " nodes x "
					    << static_cast<unsigned long>(capped_neighbours)
					    << " neighbours (N*n^3 = " << estimated_work
					    << " > " << detail::SIMULATION_SOLVE_WORK_BUDGET
					    << "); lower max_neighbours or simulate fewer nodes";
					throw hpgl_exception("do_sequential_gausian_simulation", oss.str());
				}
			}
		}

		mt_random_generator_t gen;
		gen.seed(params.m_seed);
		path_random_generator_t path_gen(property.size(), params.m_seed);

		typedef precalculated_covariances_t cov_t;
		cov_t pcov(cov_model_t(params), params.m_radiuses);
		typedef /*indexed_*/ neighbour_lookup_t<grid_t, cov_t> nl_t;
		nl_t neighbour_lookup(&grid, &pcov,	params);
		
		/*for (node_index_t node = 0; node < property.size(); ++node)
		{
			if (property.is_informed(node))
				neighbour_lookup.add_node(node);
		}*/
		
		progress_reporter_t report(property.size());	
		report.start();

		// M-11 (GSLIB ndmin semantics): the ndmin gate must count ORIGINAL
		// conditioning data only (sgsim.for: "If there are fewer than ndmin
		// data points the node is not simulated" — ndmin counts original
		// data, never previously simulated nodes). The plain neighbour
		// lookup applies the live informed-mask predicate at call time, and
		// set_at marks each simulated node informed, so previously simulated
		// nodes re-enter conditioning and would inflate ws.indices.size().
		// Snapshot which cells were originally informed so the gate can
		// exclude simulated nodes from the count. This is a deliberate
		// GSLIB-compliance change: it changes covariance reproduction between
		// simulated cells only through the ndmin skip threshold — the
		// conditioning set itself still includes simulated nodes per HPGL's
		// established (non-GSLIB) design.
		std::vector<unsigned char> originally_informed(property.size(), 0);
		for (node_index_t i = 0; i < property.size(); ++i)
			originally_informed[i] = property.is_informed(i) ? 1 : 0;

		// R-17: the ndmin gate counts originals at the same
		// covariance-threshold box offsets find() scans.
		// G-09 (CONFIRMED MEDIUM): that list is NOT rebuilt here. The neighbour
		// lookup constructed above already materialised it through the same
		// calc_cov_field call (same cov_t table, same params.m_radiuses — the
		// lookup ctor reads the very neighbourhood_param_t base of `params`) and
		// already holds the same fallback_d2_cap value, so the gate references
		// the lookup's copies. The previous duplicate construction paid a second
		// full-box covariance table transient per simulation (8 V at r = 231) for
		// a byte-identical list: peak residency 28 V -> 40 V (2.78 GB -> 3.97 GB
		// at r = 231) and construction work x2.
		// R2-06/R2-07: the referenced fallback distance bound is the one the
		// solve's find() uses, so the gate's box branch still evaluates exactly
		// the region find()'s fallback would. Binding a reference costs nothing,
		// so the default (min_neighbours == 0) pays nothing either.
		const std::vector<sugarbox_vector_t> & gate_candidates = neighbour_lookup.offsets();
		const long long gate_fallback_d2_cap = neighbour_lookup.fallback_d2_cap_value();

		node_index_t node;
		kriging_ws_t<cont_value_t, sugarbox_location_t> ws;
		unsigned long kriging_failures = 0;
		unsigned long kriging_skipped = 0;
		unsigned long kriging_ndmin_skipped = 0;
		unsigned long points_calculated = 0;
		unsigned long points_without_neighbours = 0;
		unsigned long points_singularity = 0;
		// 2-M-33: count every node that received a simulated value (success
		// OR failure fallback) so m_mean uses a numerator==denominator pair
		// like the four sibling consumers (cont_kriging, indicator_kriging,
		// cokriging, SIS). Previously the denominator was points_calculated
		// (KI_SUCCESS only) while sum_simulated accumulated over ALL
		// simulated nodes — a biased mean on partial failure.
		unsigned long nodes_simulated = 0;
		double sum_simulated = 0;
		for (node_index_t counter = 0, counter_end = property.size(); counter < counter_end; ++counter, report.next_lap())		
		{
			// E-M59: honor progress-handler cancellation cooperatively —
			// every sibling kernel reads report.cancelled() (cont_kriging.h,
			// indicator_kriging.h:301, median_ik.cpp:176, SIS
			// sequential_indicator_simulation.cpp:89-96) but the SGS loop
			// previously never did, so a cancelled run (progress_reporter.cpp
			// next_lap sets m_cancelled) still ran to completion in full.
			// README:611-618 documents cancellation WITH sgs_simulation as
			// the example.
			if (report.cancelled())
				break;

			node = path_gen.next();
			if(property.is_informed(node)) 
			{
				++kriging_skipped;
				continue;
			}

			// Bounds-guard: validate node index before mask access
			if (node < 0 || node >= property.size())
			{
				++kriging_skipped;
				continue;
			}

			if (mask[node] != 1)
			{
				++kriging_skipped;
				continue;
			}

			sugarbox_location_t loc = grid[node];			

			// GSLIB ndmin semantics (F-14 + M-11 + E-M58 + E-M84): when
			// fewer than m_min_neighbours ORIGINAL conditioning data are
			// available, leave the node unsimulated instead of simulating
			// from the marginal distribution. GSLIB sgsim.for counts
			// original data only (`if(nclose.lt.ndmin) go to 5` — nclose
			// comes from srchsupr, the input-data search; previously
			// simulated nodes are searched separately by srchnd and do NOT
			// count toward ndmin). The plain lookup applies the live
			// informed-mask predicate at call time, so previously simulated
			// nodes re-enter conditioning and inflate ws.indices.size().
			// The gate therefore counts ORIGINAL data only (the
			// originally_informed snapshot above) — with NO outer
			// total-precondition (R-3: the round-1 fix nested this check
			// inside `ws.indices.size() < min`, where original_count <= size
			// always holds — making the gate byte-identical to the pre-fix
			// total gate; the outer precondition is removed here).
			// E-M84: the gate runs BEFORE the solve — GSLIB's ndmin check
			// precedes the kriging solve; previously the full solve (find +
			// O(n²) build + LAPACK) ran first and its result was discarded
			// on gate failure.
			// E-M58: the count covers ALL original data in the FULL search
			// volume, independent of the max_neighbours truncation (srchsupr
			// counts nclose before ndmax). The pre-fix gate counted only
			// ws.indices — the top-max_neighbours covariance-ranked set — so
			// dense simulated surroundings crowded originals out of the count
			// and HPGL skipped MORE nodes than sgsim.
			// R-17 (CONFIRMED MEDIUM, FIX-INTRODUCED): the search volume is
			// the SAME geometry the solve admits — the covariance-threshold
			// box offsets find() scans (cov > C(0)/100), plus the full-box
			// fallback when the primary scan finds no datum — so the gate
			// neither over-admits (passes on originals the solve never
			// serves → marginal draw) nor under-admits (skips nodes the
			// solve would serve with >= min_neighbours originals). The live
			// informed predicate is the one find() receives, making the
			// fallback condition exact.
			if (params.m_min_neighbours > 0)
			{
				if (!detail::has_min_original_data_in_radius(
						grid, loc, params.m_radiuses, gate_candidates, originally_informed,
						is_informed_predicate_t<cont_property_array_t>(property),
						params.m_min_neighbours, gate_fallback_d2_cap))
				{
					++kriging_ndmin_skipped;
					continue;
				}
			}

			double variance = 0.0;						
			cont_value_t mean = 0.0f;
			ki_result_t ki_result = kriging_interpolation_ws(property, is_informed_predicate_t<cont_property_array_t>(property), node, pcov, mp, 
				neighbour_lookup, weight_calculator_sgs, mean, variance, ws);			

			if (ki_result != ki_result_t::KI_SUCCESS)
				++kriging_failures;

			switch (ki_result)
			{
			case ki_result_t::KI_SUCCESS: ++points_calculated; break;
			case ki_result_t::KI_NO_NEIGHBOURS: ++points_without_neighbours; break;
			case ki_result_t::KI_SINGULARITY: ++points_singularity; break;
			}

			double value = ki_result == ki_result_t::KI_SUCCESS
				? sample(gen, gaussian_cdf_t(mean, variance))
				: sample(gen, gaussian_cdf_t(fallback_mp[node], 1.0));
			
			property.set_at(node, value);
			// F-018: normal-score space — the CDF back-transform runs after
			// this function returns, so this accumulator (exported as
			// stats.m_mean below) is NOT a data-space mean.
			sum_simulated += value;
			++nodes_simulated;
			//neighbour_lookup.add_node(node);
		}
		report.stop();
		// F-M6: surface kriging failures via stats (previously the counters
		// below were stderr-only; the Python wrapper could not observe SGS
		// solver failures and geo._last_kriging_stats stayed None). The
		// points_calculated semantics mirror cont_kriging: KI_SUCCESS cells.
		{
			kriging_stats_t stats;
			stats.m_points_calculated = points_calculated;
			stats.m_points_without_neighbours = points_without_neighbours;
			stats.m_points_singularity = points_singularity;
			// E-M57: surface the ndmin-gate skip count programmatically
			// (previously stderr-only — the Python wrapper could not
			// distinguish an all-ndmin-skipped run from a successful no-op).
			stats.m_points_ndmin_skipped = kriging_ndmin_skipped;
			// 2-M-33: divide by ALL simulated nodes (nodes_simulated), not
			// the success-only points_calculated — sum_simulated includes
			// failure-fallback draws.
			// F-018: the value is in NORMAL-SCORE space, not data space.
			// sum_simulated accumulates the draws made inside the sequential
			// loop (see the `sum_simulated += value;` above) and the CDF back-transform
			// (sequential_gaussian_simulation.cpp:156-167) runs only after
			// this function returns. The C API exports the field under the
			// bare key "mean", the same key the kriging entries fill with a
			// data-space estimate mean, so a transformed SGS run reports a
			// mean (≈0 for a centred normal-score transform) that cannot be
			// compared with the output property or with a kriging mean. The
			// project convention (gslib_ref.py:41-45) requires every
			// transform to state the space its output lives in; splitting
			// the key needs a new ABI field (api.h/kriging_stats.h) and is
			// recorded for that owner — this annotation is the in-kernel half.
			stats.m_mean = nodes_simulated > 0 ? sum_simulated / static_cast<double>(nodes_simulated) : 0;
			stats.m_speed_nps = report.iterations_per_second();
			set_kriging_stats(stats);
		}
		if (kriging_failures > 0)
		{
			// G-23: this aggregate guard must travel on the documented
			// output channel (write() → registered output handler, stdout
			// fallback), not raw stderr — a handler-based consumer never saw
			// it. Sibling of the informational lines in this function.
			std::ostringstream oss;
			oss << "HPGL: SGS kriging failures: " << kriging_failures
			    << " nodes fell back to marginal mean (of "
			    << static_cast<unsigned long>(property.size()) << " total, "
			    << kriging_skipped << " skipped).\n";
			write(oss.str());
		}
		if (kriging_ndmin_skipped > 0)
		{
			// G-23: handler-visible (see the kriging-failures message above).
			std::ostringstream oss;
			oss << "HPGL: SGS ndmin: " << kriging_ndmin_skipped
			    << " nodes left unsimulated (fewer than "
			    << params.m_min_neighbours << " conditioning data).\n";
			write(oss.str());
		}
		// E-M57: the no-output warning must count ndmin skips too —
		// previously only kriging_skipped (informed/masked/bounds) was
		// counted, so an all-ndmin-skipped run (every expected node left
		// unsimulated by the ndmin gate) produced NO failure signal at all.
		// Every node reaches exactly one of {kriging_skipped,
		// kriging_ndmin_skipped, nodes_simulated}, so no-output ==
		// skipped + ndmin_skipped >= size.
		if (kriging_skipped + kriging_ndmin_skipped >= static_cast<unsigned long>(property.size()))
		{
			// G-23: handler-visible (see the kriging-failures message above).
			std::ostringstream oss;
			oss << "HPGL: SGS produced no output — all "
			    << static_cast<unsigned long>(property.size())
			    << " nodes were either already informed, masked out, "
			       "or left unsimulated by the ndmin gate. Check that the output property grid contains "
			       "uninformed cells, the mask permits processing, and min_neighbours is not too high.\n";
			write(oss.str());
		}
		// F-379 (CONFIRMED MEDIUM): a cancelled run used to fall through to
		// the success path — the void C entry returned normally, the partial
		// property was indistinguishable from a completed simulation, and the
		// "Done. Average speed" line was still printed. Cancellation is a
		// control-flow outcome, so it must reach the caller: throwing here
		// makes the C API catch (api.cpp) report it via the last-exception
		// message, which the FFI layer turns into a Python RuntimeError. The
		// caller's buffer stays partially written (unavoidable — it is the
		// working buffer), but it can no longer be mistaken for a finished
		// run. Block direction: the loop still stops promptly on cancel
		// (the `if (report.cancelled()) break;` above is unchanged). Liveness:
		// the counters and warnings above still fire on non-cancelled runs —
		// the throw is conditional on report.cancelled(), never inferred from
		// a short counter.
		if (report.cancelled())
			throw hpgl_exception("sequential_gaussian_simulation",
				"cancelled by progress handler; the output property is only partially simulated");
		{
			std::ostringstream oss;
			oss << "Done. Average speed: " << report.iterations_per_second() << " point/sec.\n";
			write(oss.str());
		}
	}

	template <typename grid_t, typename mean_provider_t, typename weight_calculator_t>
	void do_sequential_gausian_simulation_in_points(
		cont_property_array_t & property,
		const grid_t & grid,
		const sgs_params_t & params,
		const mean_provider_t & mp,		
		const weight_calculator_t & weight_calculator_sgs,
		const std::vector<int> & points_indexes)
	{
		if (property.size() != grid.size())
		{
			std::ostringstream oss;
			oss << "Property size '" << property.size() << "' is not equal to grid size '" << grid.size() << "'";
			throw hpgl_exception("do_sequential_gausian_simulation_in_points", oss.str());
		}

		mt_random_generator_t gen;
		gen.seed(params.m_seed);
		path_random_generator_t path_gen(points_indexes.size(), params.m_seed);

		typedef precalculated_covariances_t cov_t;
		//typedef precalculated_covariances_t<cov_model_t, sugarbox_location_t> cov_t;
		cov_t pcov(cov_model_t(params), params.m_radiuses);
		typedef /*indexed_*/ neighbour_lookup_t<grid_t, cov_t> nl_t;	
		nl_t neighbour_lookup(&grid, &pcov,	params);
		
		/*for (node_index_t node = 0; node < property.size(); ++node)
		{
			if (property.is_informed(node))
				neighbour_lookup.add_node(node);
		}*/
		
		progress_reporter_t report(points_indexes.size());	
		report.start();
		node_index_t node;
		kriging_ws_t<cont_value_t, sugarbox_location_t> ws;
		unsigned long kriging_failures = 0;
		unsigned long kriging_skipped = 0;
		// report.next_lap() in the increment expression ensures laps are
		// counted even when is_informed() causes a 'continue' (matching
		// the regular do_sequential_gausian_simulation variant at line 68).
		for (node_index_t counter = 0, counter_end = points_indexes.size(); counter < counter_end; ++counter, report.next_lap())		
		{
			node = points_indexes[path_gen.next()];
			if(property.is_informed(node)) 
			{
				++kriging_skipped;
				continue;
			}

			double variance = 0.0;						
			sugarbox_location_t loc = grid[node];			

			cont_value_t mean = 0.0f;
			ki_result_t ki_result = kriging_interpolation_ws(property, is_informed_predicate_t<cont_property_array_t>(property), node, pcov, mp, 
				neighbour_lookup, weight_calculator_sgs, mean, variance, ws);			

			if (ki_result != KI_SUCCESS)
				++kriging_failures;

			double value = ki_result == KI_SUCCESS 
				? sample(gen, gaussian_cdf_t(mean, variance))
				: sample(gen, gaussian_cdf_t(mp[node], 1.0));		
			
			property.set_at(node, value);
			//neighbour_lookup.add_node(node);
		}
		report.stop();
		if (kriging_failures > 0)
		{
			// G-23: documented output channel, not raw stderr (see
			// do_sequential_gausian_simulation).
			std::ostringstream oss;
			oss << "HPGL: SGS (in_points) kriging failures: " << kriging_failures
			    << " nodes fell back to marginal mean (of "
			    << static_cast<unsigned long>(points_indexes.size()) << " total, "
			    << kriging_skipped << " skipped).\n";
			write(oss.str());
		}
		{
			std::ostringstream oss;
			oss << "Done. Average speed: " << report.iterations_per_second() << " point/sec.\n";
			write(oss.str());
		}
	}
}

#endif //SEQUENTIAL_SIMULATION_H_INCLUDED_DEU0932RMNPDVCPDZFIG03249MROPWEUFSDHVIAERT0934NTFMD0GN2304UNLKWEJOFDSZF0G32R
