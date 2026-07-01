"""
Math verification tests for HPGL covariance models.

Tests validate covariance model numerical correctness against analytically
computed reference values from the C++ implementation (cov_model.h).

All tests use simple_kriging_weights() with a single neighbor at a known
distance: w = C(d)/C(0) = C(d)/sill (for nugget=0), so C(d) = w * sill.

Reference values come from Stage 6 research:
  - Spherical: C(h) = (sill - nugget) * max(0, 1 - 1.5(h/a) + 0.5(h/a)^3)
  - Exponential: C(h) = (sill - nugget) * exp(-3*h/a)
  - Gaussian: C(h) = (sill - nugget) * exp(-3*(h/a)^2)

Tolerances: rtol=1e-5, atol=1e-7 (float32-compatible)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import covariance, simple_kriging_weights
except (ImportError, OSError):
    pass


# =============================================================================
# Tolerance configuration
# =============================================================================
COV_RTOL = 1e-5
COV_ATOL = 1e-7


def compute_covariance_from_sk_weight(sill, nugget, cov_type, ranges, h):
    """Compute covariance C(h) using simple_kriging_weights with 1 neighbor.

    For a single neighbor at distance h with nugget=0:
      w = C(h) / C(0) = C(h) / sill   =>   C(h) = w * sill

    For nugget > 0:
      C(h) = w * C(0) = w * sill  (since C(0)=sill)

    Returns the computed covariance value (float).
    """
    weights = simple_kriging_weights(
        center_point=(0.0, 0.0, 0.0),
        n_x=np.array([float(h)], dtype='float32'),
        n_y=np.array([0.0], dtype='float32'),
        n_z=np.array([0.0], dtype='float32'),
        ranges=ranges,
        sill=sill,
        cov_type=cov_type,
        nugget=nugget,
    )
    return float(weights[0]) * sill


# =============================================================================
# Spherical Model Tests
# =============================================================================

@pytest.mark.hpgl
class TestSphericalCovariance:
    """Spherical covariance model: C(h) = (sill-nugget)*max(0, 1-1.5x+0.5x^3)"""

    def test_sph_h0(self):
        """SPH-1: sill=1.0, nugget=0.0, range=10.0, h=0 → C=1.0"""
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.spherical, (10.0, 10.0, 10.0), 0.0)
        np.testing.assert_allclose(C, 1.0, rtol=COV_RTOL, atol=COV_ATOL)

    def test_sph_half_range(self):
        """SPH-2: sill=1.0, nugget=0.0, range=10.0, h=5.0 → C=0.3125"""
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.spherical, (10.0, 10.0, 10.0), 5.0)
        np.testing.assert_allclose(C, 0.3125, rtol=COV_RTOL, atol=COV_ATOL)

    def test_sph_h_equals_range(self):
        """SPH-3: sill=1.0, nugget=0.0, range=10.0, h=10.0 → C=0.0"""
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.spherical, (10.0, 10.0, 10.0), 10.0)
        np.testing.assert_allclose(C, 0.0, rtol=COV_RTOL, atol=COV_ATOL)

    def test_sph_h_gt_range(self):
        """SPH-4: sill=1.0, nugget=0.0, range=10.0, h=20.0 → C=0.0"""
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.spherical, (10.0, 10.0, 10.0), 20.0)
        np.testing.assert_allclose(C, 0.0, rtol=COV_RTOL, atol=COV_ATOL)

    def test_sph_with_nugget_h0(self):
        """SPH-5: sill=1.0, nugget=0.1, range=10.0, h=0 → C=1.0 (h<0.0001 guard)"""
        C = compute_covariance_from_sk_weight(1.0, 0.1, covariance.spherical, (10.0, 10.0, 10.0), 0.0)
        np.testing.assert_allclose(C, 1.0, rtol=COV_RTOL, atol=COV_ATOL)

    def test_sph_with_nugget_half_range(self):
        """SPH-6: sill=1.0, nugget=0.1, range=10.0, h=5.0 → C=0.28125"""
        C = compute_covariance_from_sk_weight(1.0, 0.1, covariance.spherical, (10.0, 10.0, 10.0), 5.0)
        np.testing.assert_allclose(C, 0.28125, rtol=COV_RTOL, atol=COV_ATOL)

    def test_sph_double_sill(self):
        """SPH-7: sill=2.0, nugget=0.0, range=5.0, h=2.5 → C=0.625"""
        C = compute_covariance_from_sk_weight(2.0, 0.0, covariance.spherical, (5.0, 5.0, 5.0), 2.5)
        np.testing.assert_allclose(C, 0.625, rtol=COV_RTOL, atol=COV_ATOL)

    def test_sph_large_nugget_half_range(self):
        """SPH-8: sill=1.0, nugget=0.5, range=10.0, h=5.0 → C=0.15625"""
        C = compute_covariance_from_sk_weight(1.0, 0.5, covariance.spherical, (10.0, 10.0, 10.0), 5.0)
        expected = (1.0 - 0.5) * 0.3125  # = 0.15625
        np.testing.assert_allclose(C, expected, rtol=COV_RTOL, atol=COV_ATOL)

    def test_sph_small_range(self):
        """SPH-9: sill=1.0, nugget=0.0, range=1.0, h=0.5 → C=0.3125"""
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.spherical, (1.0, 1.0, 1.0), 0.5)
        np.testing.assert_allclose(C, 0.3125, rtol=COV_RTOL, atol=COV_ATOL)

    def test_sph_tiny_distance(self):
        """SPH-10: sill=1.0, nugget=0.0, range=100.0, h=5e-5 → C=1.0 (h<0.0001 guard)"""
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.spherical, (100.0, 100.0, 100.0), 5e-5)
        np.testing.assert_allclose(C, 1.0, rtol=COV_RTOL, atol=COV_ATOL)


# =============================================================================
# Exponential Model Tests
# =============================================================================

@pytest.mark.hpgl
class TestExponentialCovariance:
    """Exponential covariance model: C(h) = (sill-nugget) * exp(-3*h/a)"""

    def test_exp_h0(self):
        """EXP-1: sill=1.0, nugget=0.0, range=5.0, h=0 → C=1.0"""
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.exponential, (5.0, 5.0, 5.0), 0.0)
        np.testing.assert_allclose(C, 1.0, rtol=COV_RTOL, atol=COV_ATOL)

    def test_exp_half_range(self):
        """EXP-2: sill=1.0, nugget=0.0, range=5.0, h=2.5 → C=exp(-1.5)~0.223130"""
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.exponential, (5.0, 5.0, 5.0), 2.5)
        expected = np.exp(-1.5)
        np.testing.assert_allclose(C, expected, rtol=COV_RTOL, atol=COV_ATOL)

    def test_exp_equals_range(self):
        """EXP-3: sill=1.0, nugget=0.0, range=5.0, h=5.0 → C=exp(-3.0)~0.049787"""
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.exponential, (5.0, 5.0, 5.0), 5.0)
        expected = np.exp(-3.0)
        np.testing.assert_allclose(C, expected, rtol=COV_RTOL, atol=COV_ATOL)

    def test_exp_double_range(self):
        """EXP-4: sill=1.0, nugget=0.0, range=5.0, h=10.0 → C=exp(-6.0)~0.002479"""
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.exponential, (5.0, 5.0, 5.0), 10.0)
        expected = np.exp(-6.0)
        np.testing.assert_allclose(C, expected, rtol=COV_RTOL, atol=COV_ATOL)

    def test_exp_with_nugget(self):
        """EXP-5: sill=1.0, nugget=0.1, range=5.0, h=2.5 → C=0.9*exp(-1.5)~0.200817"""
        C = compute_covariance_from_sk_weight(1.0, 0.1, covariance.exponential, (5.0, 5.0, 5.0), 2.5)
        expected = (1.0 - 0.1) * np.exp(-1.5)
        np.testing.assert_allclose(C, expected, rtol=COV_RTOL, atol=COV_ATOL)

    def test_exp_double_sill(self):
        """EXP-6: sill=2.0, nugget=0.0, range=3.0, h=3.0 → C=2*exp(-3.0)~0.099574"""
        C = compute_covariance_from_sk_weight(2.0, 0.0, covariance.exponential, (3.0, 3.0, 3.0), 3.0)
        expected = 2.0 * np.exp(-3.0)
        np.testing.assert_allclose(C, expected, rtol=COV_RTOL, atol=COV_ATOL)

    def test_exp_range10_h10(self):
        """EXP-7: sill=1.0, nugget=0.0, range=10.0, h=10.0 → C=exp(-3.0)"""
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.exponential, (10.0, 10.0, 10.0), 10.0)
        expected = np.exp(-3.0)
        np.testing.assert_allclose(C, expected, rtol=COV_RTOL, atol=COV_ATOL)

    def test_exp_range_div_3(self):
        """EXP-8: sill=1.0, nugget=0.0, range=10.0, h=range/3=3.333... → C=exp(-1.0)"""
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.exponential, (10.0, 10.0, 10.0), 10.0 / 3.0)
        expected = np.exp(-1.0)
        np.testing.assert_allclose(C, expected, rtol=COV_RTOL, atol=COV_ATOL)


# =============================================================================
# Gaussian Model Tests
# =============================================================================

@pytest.mark.hpgl
class TestGaussianCovariance:
    """Gaussian covariance model: C(h) = (sill-nugget) * exp(-3*(h/a)^2)"""

    def test_gau_h0(self):
        """GAU-1: sill=1.0, nugget=0.0, range=5.0, h=0 → C=1.0"""
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.gaussian, (5.0, 5.0, 5.0), 0.0)
        np.testing.assert_allclose(C, 1.0, rtol=COV_RTOL, atol=COV_ATOL)

    def test_gau_half_range(self):
        """GAU-2: sill=1.0, nugget=0.0, range=5.0, h=2.5 → C=exp(-0.75)~0.472367"""
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.gaussian, (5.0, 5.0, 5.0), 2.5)
        expected = np.exp(-0.75)
        np.testing.assert_allclose(C, expected, rtol=COV_RTOL, atol=COV_ATOL)

    def test_gau_equals_range(self):
        """GAU-3: sill=1.0, nugget=0.0, range=5.0, h=5.0 → C=exp(-3.0)~0.049787"""
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.gaussian, (5.0, 5.0, 5.0), 5.0)
        expected = np.exp(-3.0)
        np.testing.assert_allclose(C, expected, rtol=COV_RTOL, atol=COV_ATOL)

    def test_gau_double_range(self):
        """GAU-4: sill=1.0, nugget=0.0, range=5.0, h=10.0 → C=exp(-12.0)~6.14e-6"""
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.gaussian, (5.0, 5.0, 5.0), 10.0)
        expected = np.exp(-12.0)
        np.testing.assert_allclose(C, expected, rtol=COV_RTOL, atol=COV_ATOL)

    def test_gau_with_nugget(self):
        """GAU-5: sill=1.0, nugget=0.1, range=5.0, h=2.5 → C=0.9*exp(-0.75)~0.425130"""
        C = compute_covariance_from_sk_weight(1.0, 0.1, covariance.gaussian, (5.0, 5.0, 5.0), 2.5)
        expected = (1.0 - 0.1) * np.exp(-0.75)
        np.testing.assert_allclose(C, expected, rtol=COV_RTOL, atol=COV_ATOL)

    def test_gau_range10_h5(self):
        """GAU-6: sill=1.0, nugget=0.0, range=10.0, h=5.0 → C=exp(-3*(0.5)^2)=exp(-0.75)"""
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.gaussian, (10.0, 10.0, 10.0), 5.0)
        expected = np.exp(-0.75)
        np.testing.assert_allclose(C, expected, rtol=COV_RTOL, atol=COV_ATOL)

    def test_gau_range_div_sqrt3(self):
        """GAU-7: sill=1.0, nugget=0.0, range=10.0, h=10/sqrt(3)~5.774 → C=exp(-1.0)"""
        h = 10.0 / np.sqrt(3.0)
        C = compute_covariance_from_sk_weight(1.0, 0.0, covariance.gaussian, (10.0, 10.0, 10.0), h)
        expected = np.exp(-1.0)
        np.testing.assert_allclose(C, expected, rtol=COV_RTOL, atol=COV_ATOL)


# =============================================================================
# Anisotropic Range Tests
# =============================================================================

@pytest.mark.hpgl
class TestAnisotropicCovariance:
    """Anisotropic range tests — different ranges per axis."""

    def test_ani_x_along_range(self):
        """ANI-1: ranges=(10,5,5), neighbor at (5,0,0) → h_eff=5.0, C(spherical)=0.3125"""
        C = compute_covariance_from_sk_weight(
            1.0, 0.0, covariance.spherical, (10.0, 5.0, 5.0), 5.0
        )
        # h_eff along X: sqrt((5/10*10)^2 + 0 + 0) = 5, x=5/10=0.5 → C=0.3125
        np.testing.assert_allclose(C, 0.3125, rtol=COV_RTOL, atol=COV_ATOL)

    def test_anisotropy_x_vs_y_weight(self):
        """ANI-2/3: With ranges=(10,5,5), neighbor along Y gets lower weight than along X."""
        # Neighbor along X at distance 5: h_eff_x = 5.0 (range_X=10)
        wx = simple_kriging_weights(
            center_point=(0.0, 0.0, 0.0),
            n_x=np.array([5.0], dtype='float32'),
            n_y=np.array([0.0], dtype='float32'),
            n_z=np.array([0.0], dtype='float32'),
            ranges=(10.0, 5.0, 5.0),
            sill=1.0,
            cov_type=covariance.spherical,
            nugget=0.0,
        )
        # Neighbor along Y at distance 5: h_eff_y = sqrt(0 + (10*5/5)^2 + 0) = 10.0
        wy = simple_kriging_weights(
            center_point=(0.0, 0.0, 0.0),
            n_x=np.array([0.0], dtype='float32'),
            n_y=np.array([5.0], dtype='float32'),
            n_z=np.array([0.0], dtype='float32'),
            ranges=(10.0, 5.0, 5.0),
            sill=1.0,
            cov_type=covariance.spherical,
            nugget=0.0,
        )
        # X axis (range 10): h_eff=5 → C=0.3125, w=0.3125
        # Y axis (range 5): h_eff=10 → C=0.0, w=0.0
        assert wx[0] > wy[0], f"w_x={wx[0]} should be greater than w_y={wy[0]}"

    def test_isotropic_equal_weights(self):
        """With equal ranges and zero angles, same-distance neighbors get same weight
        regardless of direction (isotropy)."""
        for dist in [3.0, 5.0, 7.0]:
            wx = simple_kriging_weights(
                center_point=(0.0, 0.0, 0.0),
                n_x=np.array([dist], dtype='float32'),
                n_y=np.array([0.0], dtype='float32'),
                n_z=np.array([0.0], dtype='float32'),
                ranges=(10.0, 10.0, 10.0),
                sill=1.0,
                cov_type=covariance.spherical,
                nugget=0.0,
            )
            wy = simple_kriging_weights(
                center_point=(0.0, 0.0, 0.0),
                n_x=np.array([0.0], dtype='float32'),
                n_y=np.array([dist], dtype='float32'),
                n_z=np.array([0.0], dtype='float32'),
                ranges=(10.0, 10.0, 10.0),
                sill=1.0,
                cov_type=covariance.spherical,
                nugget=0.0,
            )
            wz = simple_kriging_weights(
                center_point=(0.0, 0.0, 0.0),
                n_x=np.array([0.0], dtype='float32'),
                n_y=np.array([0.0], dtype='float32'),
                n_z=np.array([dist], dtype='float32'),
                ranges=(10.0, 10.0, 10.0),
                sill=1.0,
                cov_type=covariance.spherical,
                nugget=0.0,
            )
            np.testing.assert_allclose(wx[0], wy[0], rtol=COV_RTOL, atol=COV_ATOL)
            np.testing.assert_allclose(wx[0], wz[0], rtol=COV_RTOL, atol=COV_ATOL)


# =============================================================================
# Cross-model Comparison Tests
# =============================================================================

@pytest.mark.hpgl
class TestCrossModelProperties:
    """Verify known relationships between covariance models."""

    def test_gaussian_higher_near_origin(self):
        """At h=a/2, Gaussian C > Exponential C (Gaussian decays slower near origin).

        Gaussian has a flat top (behaves like exp(-kh^2)) and stays closer to 1.0
        for small h. Exponential drops immediately.

        At range=5, h=2.5: Gaussian C = exp(-0.75) ≈ 0.472,
                         Exponential C = exp(-1.5) ≈ 0.223.
        This is the complement of test_gaussian_approaches_zero_fast (line 332)
        which tests Gaussian < Exponential at h=2a (far-field decay).
        """
        h = 2.5  # half of range=5
        C_exp = compute_covariance_from_sk_weight(1.0, 0.0, covariance.exponential, (5.0, 5.0, 5.0), h)
        C_gau = compute_covariance_from_sk_weight(1.0, 0.0, covariance.gaussian, (5.0, 5.0, 5.0), h)
        assert 0 < C_exp < C_gau <= 1.0, (
            f"At h=a/2 Gaussian ({C_gau}) should be higher than Exponential ({C_exp})"
        )

    def test_gaussian_approaches_zero_fast(self):
        """At h=2a, Gaussian C < Exponential C (Gaussian drops faster at large distances)."""
        h = 10.0  # double range=5
        C_exp = compute_covariance_from_sk_weight(1.0, 0.0, covariance.exponential, (5.0, 5.0, 5.0), h)
        C_gau = compute_covariance_from_sk_weight(1.0, 0.0, covariance.gaussian, (5.0, 5.0, 5.0), h)
        # exp(-6) ~ 0.00248, exp(-12) ~ 6.14e-6
        assert C_gau < C_exp, (
            f"Gaussian should decay faster at 2a: C_gau={C_gau}, C_exp={C_exp}"
        )

    def test_spherical_has_finite_range(self):
        """Spherical reaches exactly 0 at h=range (not asymptotic)."""
        # At h=range, spherical C = 0 exactly, exponential/gaussian > 0
        C_sph = compute_covariance_from_sk_weight(1.0, 0.0, covariance.spherical, (10.0, 10.0, 10.0), 10.0)
        C_exp = compute_covariance_from_sk_weight(1.0, 0.0, covariance.exponential, (10.0, 10.0, 10.0), 10.0)
        C_gau = compute_covariance_from_sk_weight(1.0, 0.0, covariance.gaussian, (10.0, 10.0, 10.0), 10.0)
        np.testing.assert_allclose(C_sph, 0.0, atol=COV_ATOL)
        assert C_exp > 0, "Exponential should be >0 at h=range"
        assert C_gau > 0, "Gaussian should be >0 at h=range"


# =============================================================================
# Covariance Matrix Positive-Definiteness Tests (M-T12)
# =============================================================================

def _spherical_cov_py(h, sill, nugget, range_val):
    """Analytical spherical covariance: C(h) in Python."""
    if h < 1e-4:
        return sill
    if h > range_val:
        return 0.0
    x = h / range_val
    return max(0.0, (sill - nugget) * (1.0 - 1.5 * x + 0.5 * x ** 3))


def _exponential_cov_py(h, sill, nugget, range_val):
    """Analytical exponential covariance: C(h) in Python."""
    if h < 1e-4:
        return sill
    return (sill - nugget) * np.exp(-3.0 * h / range_val)


def _gaussian_cov_py(h, sill, nugget, range_val):
    """Analytical Gaussian covariance: C(h) in Python."""
    if h < 1e-4:
        return sill
    return (sill - nugget) * np.exp(-3.0 * (h / range_val) ** 2)


def _build_covariance_matrix(points, cov_func, sill, nugget, ranges):
    """Build a covariance matrix K where K_ij = C(h_ij).

    Parameters
    ----------
    points : numpy.ndarray, shape (n, 3)
        Coordinates of n points.
    cov_func : callable
        Covariance function C(h, sill, nugget, range_val).
    sill : float
    nugget : float
    ranges : tuple (rx, ry, rz)
        Anisotropic ranges. For isotropic case use equal values.

    Returns
    -------
    K : numpy.ndarray, shape (n, n)
        Covariance matrix.
    """
    rx, ry, rz = ranges
    n = len(points)
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dz = points[i, 2] - points[j, 2]
            # Anisotropic effective distance: h_eff for zero rotation
            h_eff = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2 + (dz / rz) ** 2) * rx
            K[i, j] = cov_func(h_eff, sill, nugget, rx)
    return K


def _is_positive_definite(K):
    """Check if matrix K is positive-definite via Cholesky decomposition."""
    try:
        np.linalg.cholesky(K)
        return True
    except np.linalg.LinAlgError:
        return False


@pytest.mark.hpgl
class TestCovarianceMatrixPositiveDefinite:
    """Verify covariance matrices are positive-definite for various models."""

    @staticmethod
    def _generate_random_points(n, seed=42):
        rng = np.random.default_rng(seed)
        return rng.uniform(0, 100, size=(n, 3))

    def test_spherical_isotropic_pd(self):
        """PD-1: Spherical isotropic covariance matrix is positive-definite."""
        points = self._generate_random_points(10)
        K = _build_covariance_matrix(points, _spherical_cov_py,
                                      sill=1.0, nugget=0.1, ranges=(20.0, 20.0, 20.0))
        assert _is_positive_definite(K), "Spherical isotropic matrix should be PD"

    def test_exponential_isotropic_pd(self):
        """PD-2: Exponential isotropic covariance matrix is positive-definite."""
        points = self._generate_random_points(10)
        K = _build_covariance_matrix(points, _exponential_cov_py,
                                      sill=1.0, nugget=0.1, ranges=(20.0, 20.0, 20.0))
        assert _is_positive_definite(K), "Exponential isotropic matrix should be PD"

    def test_gaussian_isotropic_pd(self):
        """PD-3: Gaussian isotropic covariance matrix is positive-definite."""
        points = self._generate_random_points(10)
        K = _build_covariance_matrix(points, _gaussian_cov_py,
                                      sill=1.0, nugget=0.1, ranges=(20.0, 20.0, 20.0))
        assert _is_positive_definite(K), "Gaussian isotropic matrix should be PD"

    def test_spherical_anisotropic_pd(self):
        """PD-4: Spherical anisotropic covariance matrix is positive-definite."""
        points = self._generate_random_points(10)
        K = _build_covariance_matrix(points, _spherical_cov_py,
                                      sill=1.0, nugget=0.1, ranges=(30.0, 15.0, 10.0))
        assert _is_positive_definite(K), "Spherical anisotropic matrix should be PD"

    def test_all_models_zero_nugget_pd(self):
        """PD-5: Covariance matrices with zero nugget are still PD for
        distinct points (nugget=0 may be singular for coincident points but
        with random distinct points it should be PD)."""
        points = self._generate_random_points(8, seed=99)
        for cov_func, name in [
            (_spherical_cov_py, "spherical"),
            (_exponential_cov_py, "exponential"),
            (_gaussian_cov_py, "gaussian"),
        ]:
            K = _build_covariance_matrix(points, cov_func,
                                          sill=1.0, nugget=0.0, ranges=(20.0, 20.0, 20.0))
            assert _is_positive_definite(K), f"{name} zero-nugget matrix should be PD"

    def test_all_models_large_nugget_pd(self):
        """PD-6: Covariance matrices with large nugget are strongly PD."""
        points = self._generate_random_points(5, seed=7)
        for cov_func, name in [
            (_spherical_cov_py, "spherical"),
            (_exponential_cov_py, "exponential"),
            (_gaussian_cov_py, "gaussian"),
        ]:
            K = _build_covariance_matrix(points, cov_func,
                                          sill=2.0, nugget=0.9, ranges=(10.0, 10.0, 10.0))
            assert _is_positive_definite(K), f"{name} large-nugget matrix should be PD"

    def test_eigenvalues_all_positive(self):
        """PD-7: All eigenvalues of covariance matrices are positive."""
        points = self._generate_random_points(12, seed=123)
        for cov_func, name, ranges in [
            (_spherical_cov_py, "spherical", (25.0, 25.0, 25.0)),
            (_exponential_cov_py, "exponential", (25.0, 25.0, 25.0)),
            (_gaussian_cov_py, "gaussian", (25.0, 25.0, 25.0)),
            (_spherical_cov_py, "spherical-ani", (40.0, 20.0, 10.0)),
        ]:
            K = _build_covariance_matrix(points, cov_func,
                                          sill=1.0, nugget=0.05, ranges=ranges)
            eigenvalues = np.linalg.eigvalsh(K)
            assert np.all(eigenvalues > 0), (
                f"{name}: all eigenvalues must be positive, min={eigenvalues.min():.2e}"
            )


# =============================================================================
# HPGL C++ Covariance Matrix PD Tests (F7-06)
# =============================================================================

def _build_covariance_matrix_hpgl(points, sill, nugget, cov_type, ranges):
    """Build a covariance matrix K_ij = C(h_ij) using HPGL's C++ covariance code.

    Calls simple_kriging_weights() (C++ via HPGL) for each pair of points
    to compute C(h_ij). This exercises HPGL's actual covariance implementation,
    not pure Python formulas.

    Parameters
    ----------
    points : numpy.ndarray, shape (n, 3)
        Coordinates of n points.
    sill : float
    nugget : float
    cov_type : int
        HPGL covariance type constant (covariance.spherical, etc.).
    ranges : tuple (rx, ry, rz)

    Returns
    -------
    K : numpy.ndarray, shape (n, n)
        Covariance matrix built from HPGL C++ covariance values.
    """
    n = len(points)
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dz = points[i, 2] - points[j, 2]
            h = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            C = compute_covariance_from_sk_weight(sill, nugget, cov_type, ranges, h)
            K[i, j] = C
            K[j, i] = C
    return K


@pytest.mark.hpgl
class TestCovarianceMatrixPositiveDefiniteHPGL:
    """Verify covariance matrices from HPGL's C++ implementation are PD."""

    @staticmethod
    def _generate_random_points(n, seed=42):
        rng = np.random.default_rng(seed)
        return rng.uniform(0, 100, size=(n, 3))

    def test_spherical_hpgl_pd(self):
        """PD-H1: Spherical covariance matrix via HPGL C++ is positive-definite."""
        points = self._generate_random_points(10)
        K = _build_covariance_matrix_hpgl(points, sill=1.0, nugget=0.1,
                                           cov_type=covariance.spherical,
                                           ranges=(20.0, 20.0, 20.0))
        assert _is_positive_definite(K), "Spherical HPGL matrix should be PD"

    def test_exponential_hpgl_pd(self):
        """PD-H2: Exponential covariance matrix via HPGL C++ is positive-definite."""
        points = self._generate_random_points(10)
        K = _build_covariance_matrix_hpgl(points, sill=1.0, nugget=0.1,
                                           cov_type=covariance.exponential,
                                           ranges=(20.0, 20.0, 20.0))
        assert _is_positive_definite(K), "Exponential HPGL matrix should be PD"

    def test_gaussian_hpgl_pd(self):
        """PD-H3: Gaussian covariance matrix via HPGL C++ is positive-definite."""
        points = self._generate_random_points(10)
        K = _build_covariance_matrix_hpgl(points, sill=1.0, nugget=0.1,
                                           cov_type=covariance.gaussian,
                                           ranges=(20.0, 20.0, 20.0))
        assert _is_positive_definite(K), "Gaussian HPGL matrix should be PD"

    def test_all_models_hpgl_zero_nugget_pd(self):
        """PD-H4: HPGL covariance matrices with zero nugget are still PD."""
        points = self._generate_random_points(8, seed=99)
        for cov_type, name in [
            (covariance.spherical, "spherical"),
            (covariance.exponential, "exponential"),
            (covariance.gaussian, "gaussian"),
        ]:
            K = _build_covariance_matrix_hpgl(points, sill=1.0, nugget=0.0,
                                               cov_type=cov_type,
                                               ranges=(20.0, 20.0, 20.0))
            assert _is_positive_definite(K), f"{name} HPGL zero-nugget matrix should be PD"

    def test_all_models_hpgl_large_nugget_pd(self):
        """PD-H5: HPGL covariance matrices with large nugget are strongly PD."""
        points = self._generate_random_points(5, seed=7)
        for cov_type, name in [
            (covariance.spherical, "spherical"),
            (covariance.exponential, "exponential"),
            (covariance.gaussian, "gaussian"),
        ]:
            K = _build_covariance_matrix_hpgl(points, sill=2.0, nugget=0.9,
                                               cov_type=cov_type,
                                               ranges=(10.0, 10.0, 10.0))
            assert _is_positive_definite(K), f"{name} HPGL large-nugget matrix should be PD"

    def test_hpgl_eigenvalues_all_positive(self):
        """PD-H6: All eigenvalues of HPGL covariance matrices are positive."""
        points = self._generate_random_points(12, seed=123)
        for cov_type, name in [
            (covariance.spherical, "spherical"),
            (covariance.exponential, "exponential"),
            (covariance.gaussian, "gaussian"),
        ]:
            K = _build_covariance_matrix_hpgl(points, sill=1.0, nugget=0.05,
                                               cov_type=cov_type,
                                               ranges=(25.0, 25.0, 25.0))
            eigenvalues = np.linalg.eigvalsh(K)
            assert np.all(eigenvalues > 0), (
                f"{name} HPGL: all eigenvalues must be positive, min={eigenvalues.min():.2e}"
            )
