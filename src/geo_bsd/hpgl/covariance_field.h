#ifndef COVARIANCE_FIELD_H_INCLUDED_IN_BLUE_SKY_GSTL_AND_SOME_RANDOM_SYMBOLS_AFKJSDFKSJFKSDLKASDHKLWIUEWRWIWEIURH
#define COVARIANCE_FIELD_H_INCLUDED_IN_BLUE_SKY_GSTL_AND_SOME_RANDOM_SYMBOLS_AFKJSDFKSJFKSDLKASDHKLWIUEWRWIWEIURH

#include "typedefs.h"
#include "var_radix_utils.h"
#include "cov_model.h"
#include "sugarbox_grid.h"
#include "hpgl_exception.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <optional>

namespace hpgl
{
	namespace detail
	{
		class predicate_2
		{
			const std::vector<hpgl::covariance_t> & m_data;
			int m_xradius;
			int m_yradius;
			int m_zradius;
			int m_xd;
			int m_yd;
		public:
			predicate_2(const std::vector<hpgl::covariance_t> & data, int xradius, int yradius, int zradius)
				: m_data(data), m_xradius(xradius), m_yradius(yradius), m_zradius(zradius), m_xd(xradius * 2 + 1), m_yd(yradius * 2 + 1)
			{}

			bool operator()(
				const hpgl::sugarbox_vector_t & vec1,
				const hpgl::sugarbox_vector_t & vec2)
			{
				hpgl::covariance_t value1 = m_data[hpgl::vr_to_dec(m_yd, m_xd, vec1[2] + m_zradius, vec1[1] + m_yradius, vec1[0] + m_xradius)];
				hpgl::covariance_t value2 = m_data[hpgl::vr_to_dec(m_yd, m_xd, vec2[2] + m_zradius, vec2[1] + m_yradius, vec2[0] + m_xradius)];
				if (value1 != value2)
					return value1 > value2;
				// E-M67 (CONFIRMED MEDIUM): deterministic tie-break for
				// equal covariances — ties are routine on regular grids (48
				// lattice positions at h²=14). The old covariance-only
				// comparator left the boundary cut inside a tie group
				// dependent on the input permutation (z,y,x box loop order),
				// while the indexed sibling sorts its candidates in a
				// different permutation (cluster traversal + nth_element
				// partition), so OK (indexed) and SGS (plain) could pick
				// different members of the same tie group at the
				// max_neighbours boundary — different kriging results for
				// identical input. Secondary key: squared distance (closer
				// first), then the node-index order of the indexed sibling
				// (E-M67): index = z·nx·ny + y·nx + x (sugarbox_grid.h), so
				// index ascending is the (z, y, x) lexicographic coordinate
				// order. The estimation-node center cancels in the pairwise
				// comparison of two offsets (index(C+o1) < index(C+o2)
				// ⇔ (dz1,dy1,dx1) < (dz2,dy2,dx2) lexicographic, z-major),
				// so this static comparator over plain offsets reproduces
				// the indexed path's node-index tie-break exactly —
				// R-16: the former x-major key diverged on ties like
				// (1,0,0) vs (0,0,1) (x-major admitted (0,0,1) first;
				// node index admits base+1 < base+nx·ny first). The result
				// is a strict total order (cov desc, d² asc, z, y, x asc),
				// so the plain offset list is fully determined regardless of
				// loop order and matches the indexed sibling.
				const long long dx1 = vec1[0], dy1 = vec1[1], dz1 = vec1[2];
				const long long dx2 = vec2[0], dy2 = vec2[1], dz2 = vec2[2];
				const long long d2_1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
				const long long d2_2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
				if (d2_1 != d2_2)
					return d2_1 < d2_2;
				// R-16: node-index order (z-major) — matches the indexed
				// sibling's index tie-break; the x-major key it replaces
				// diverged on (1,0,0) vs (0,0,1).
				if (vec1[2] != vec2[2])
					return vec1[2] < vec2[2];
				if (vec1[1] != vec2[1])
					return vec1[1] < vec2[1];
				return vec1[0] < vec2[0];
			}
		};
	}

// calc_distance vector case
	template <typename coord_t>
	void calc_dist_field(const search_area_t & area, std::vector<sugarbox_vector_t> & vectors)
	{
		int m_xradius = area[0];
		int m_yradius = area[1];
		int m_zradius = area[2];

		for (int z = -m_zradius; z <= m_zradius; ++z)
		{
			for (int y = -m_yradius; y <= m_yradius; ++y)
			{
				for (int x = -m_xradius; x <= m_xradius; ++x)
				{
					vectors.push_back(sugarbox_location_t(0,0,0) - sugarbox_location_t(x,y,z));										
				}
			}
		}

		// Sort by squared distance from origin (ascending) so that
		// neighbour lookup iterates nearest cells first.  Without
		// sorting, find() picks the first N vector positions in
		// z,y,x iteration order, which may skip closer neighbours
		// that happen to appear later in the unsorted list.
		std::sort(vectors.begin(), vectors.end(),
			[](const sugarbox_vector_t & a, const sugarbox_vector_t & b) {
				return (a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
				     < (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
			});
	}

	// Shared radius-magnitude guard for the (2r+1)³ covariance box.
	// calc_cov_field, covariance_field_t::init, and
	// precalculated_covariances_t::init each build a box of
	// (2rx+1)*(2ry+1)*(2rz+1) doubles. E2-139: the previous INT_MAX bound
	// (2,147,483,647 cells) admitted radii up to 644, materializing a
	// 17.1 GB covariance table plus up to 25.7 GB of offset vectors at
	// construction — a minutes-long hang/OOM on legal configs (the old
	// "463" comment arithmetic was wrong: (2·463+1)³ = 7.96e8 < INT_MAX,
	// i.e. radius 463 did NOT trip the guard). The bound is now
	// memory-sane: 1e8 cells ≈ 0.8 GB of doubles (covariance table) plus
	// ≤ 1.2 GB of offset vectors ≈ 2 GB worst case (cubic radius ≤ 231) —
	// far above every in-tree use (largest test radius 30, book radius
	// 160) while bounding the construction hang. Throws hpgl_exception
	// (catchable by Python via error_guard) — never abort()/exit().
	//
	// Entry points that build the box indirectly (ordinary-kriging default,
	// median_ik, cokriging markI/markII) call the ellipsoid overload before
	// allocating.  The int-triple overload is the core; the ellipsoid
	// overload truncates the double radii to int exactly like the box
	// construction paths do, so the guard rejects the same inputs the
	// builder would reject.
	inline size_t validate_covariance_radiuses_or_throw(
			int xradius, int yradius, int zradius, const char * context)
	{
		// E2-139: memory-sane (2r+1)³ volume cap — ~0.8 GB doubles + ≤1.2 GB
		// offset vectors worst case. Also keeps the box index (vr_to_dec
		// returns int) well inside INT_MAX.
		constexpr size_t COVARIANCE_FIELD_VOLUME_CAP = 100000000ULL;
		// Overflow-safe size_t arithmetic: no signed-int multiplication
		// (rx*2+1) happens before the magnitude checks below, so a radius
		// near INT_MAX cannot wrap UB-first.  Negative radii wrap to huge
		// size_t values and are rejected by the overflow checks.
		const size_t sx = static_cast<size_t>(xradius) * 2 + 1;
		const size_t sy = static_cast<size_t>(yradius) * 2 + 1;
		const size_t sz = static_cast<size_t>(zradius) * 2 + 1;
		size_t volume = 0;
		if (sx > 0 && sy > 0 && sz > 0)
		{
			if (sx > SIZE_MAX / sy)
				throw hpgl_exception(context, "overflow in covariance box sx*sy");
			const size_t tmp = sx * sy;
			if (tmp > SIZE_MAX / sz)
				throw hpgl_exception(context, "overflow in covariance box volume");
			volume = tmp * sz;
		}
		// vr_to_dec returns int; the box index must fit in INT_MAX
		// (same bound as precalculated_covariances_t::init, I2-24).
		if (volume > COVARIANCE_FIELD_VOLUME_CAP)
		{
			throw hpgl_exception(context,
				"covariance volume exceeds the memory-safe limit of 100000000 cells (search radii too large)");
		}
		return volume;
	}

	inline size_t validate_covariance_radiuses_or_throw(
			const sugarbox_search_ellipsoid_t & radiuses, const char * context)
	{
		return validate_covariance_radiuses_or_throw(
			static_cast<int>(radiuses[0]),
			static_cast<int>(radiuses[1]),
			static_cast<int>(radiuses[2]),
			context);
	}

	// type covariance_model_t:
	//     covariance_t operator(coord_t, coord_t)
	template <typename covariance_model_t, typename coord_t>
	void calc_cov_field(const search_area_t & area, const covariance_model_t & cov, std::vector<sugarbox_vector_t> & vectors)
	{
		// F-M2: radius-magnitude guard — the (2r+1)³ box is allocated
		// below; reject radii whose box exceeds INT_MAX with a catchable
		// hpgl_exception instead of exhausting memory.
		validate_covariance_radiuses_or_throw(area, "calc_cov_field");
		std::vector<covariance_t> data;	
		int m_xradius = area[0];
		int m_yradius = area[1];
		int m_zradius = area[2];
		//int m_xdiameter = (m_xradius * 2 + 1);
		//int m_ydiameter = (m_yradius * 2 + 1);
		//int m_zdiameter = (m_zradius * 2 + 1);

		double threshold = cov(coord_t(0, 0, 0), coord_t(0,0,0)) / 100;
		for (int z = -m_zradius; z <= m_zradius; ++z)
		{
			for (int y = -m_yradius; y <= m_yradius; ++y)
			{
				for (int x = -m_xradius; x <= m_xradius; ++x)
				{
					double value = cov(coord_t(0,0,0), coord_t(x,y,z));

					data.push_back(value);
					if (value > threshold)
					{
						vectors.push_back(sugarbox_location_t(0,0,0) - sugarbox_location_t(x,y,z));										
					}
				}
			}
		}

		std::sort(vectors.begin(), vectors.end(), detail::predicate_2(data, m_xradius, m_yradius, m_zradius));

		// threshold was already computed above; reuse it for the second pass

		for (size_t idx = 0, end_idx = vectors.size(); idx < end_idx; ++idx)
		{
			sugarbox_vector_t vec = vectors[idx];
			if (cov(coord_t(0,0,0), coord_t(vec[0], vec[1], vec[2])) < threshold)
			{
				vectors.resize(idx);
				break;
			}		
		}

		//It seems there is duplicate threshold cutt-off....
	}

class covariance_field_t
{
	std::vector<covariance_t> m_data;	
	int m_xradius;
	int m_yradius;
	int m_zradius;
	int m_xdiameter;
	int m_ydiameter;
	int m_zdiameter;
	std::vector<sugarbox_vector_t> m_vectors;
	// PR-06 (F-21 sibling): exact model fallback for pairs beyond the
	// precomputed box. The box table truncates data-to-data covariance to 0
	// for neighbour pairs farther apart than the search radius, while the RHS
	// (data-to-target) stays exact — an internally inconsistent kriging
	// system on the median_ik path. Storing the source model lets operator()
	// return the exact covariance for out-of-box pairs (GSLIB cova3
	// behavior), mirroring precalculated_covariances_t (precalculated_covariance.h).
	// cov_model_t is not default-constructible, hence std::optional.
	std::optional<cov_model_t> m_exact_model;
	void init(int xradius, 
			int yradius, 
			int zradius, 
			const cov_model_t &);
public:
	covariance_field_t(
			int xradius, 
			int yradius, 
			int zradius, 
			const cov_model_t &);

	covariance_field_t(
			const sugarbox_search_ellipsoid_t & ellipsoid,
			const cov_model_t &);
		
	inline double value(int x, int y, int z)const;
	const std::vector<sugarbox_vector_t> & vectors()const
	{
		return m_vectors;
	};
	
	size_t size() const
	{
		return m_data.size();
	}

	inline double operator()(const sugarbox_location_t  & loc1 , const sugarbox_location_t & loc2)const
	{
		sugarbox_vector_t vec = loc1 - loc2;
		if (vec[0] < - m_xradius || vec[0] > m_xradius 
			|| vec[1] < - m_yradius || vec[1] > m_yradius 
			|| vec[2] < - m_zradius || vec[2] > m_zradius )
		{
			// PR-06: exact covariance beyond the search box — do not
			// truncate to 0, which makes the data-to-data LHS inconsistent
			// with the exact RHS in median_ik kriging systems (F-21 sibling).
			if (m_exact_model.has_value())
				return (*m_exact_model)(loc1, loc2);
			return 0;
		}
		else
			return value(vec[0], vec[1], vec[2]);
	}
};

	double
covariance_field_t::value(
	int x,
	int y,
	int z)const
{
	return m_data[static_cast<size_t>(vr_to_dec(m_ydiameter, m_xdiameter,
		static_cast<long long>(z) + m_zradius,
		static_cast<long long>(y) + m_yradius,
		static_cast<long long>(x) + m_xradius))];
	/*int xx = x + m_xradius; 
	int yy = y + m_yradius;
	int zz = z + m_zradius;

	int idx = zz * m_xdiameter * m_ydiameter + yy * m_xdiameter + xx;
	return m_data[idx];*/
}

} //namespace hpgl;

#endif //COVARIANCE_FIELD_H_INCLUDED_IN_BLUE_SKY_GSTL_AND_SOME_RANDOM_SYMBOLS_AFKJSDFKSJFKSDLKASDHKLWIUEWRWIWEIURH
