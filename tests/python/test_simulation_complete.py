"""
Comprehensive tests for HPGL simulation algorithms:
- Sequential Gaussian Simulation (SGS)
- Sequential Indicator Simulation (SIS)

Tests cover:
- Reproducibility with same seed (non-vacuous 10%-uninformed pins)
- Golden-value determinism pins for a fixed seed (G-57) and seeds above 2**32 (G-59)
- Kriging type variations (SK vs OK for SGS)
- Statistical properties validation (CDF range, variogram structure,
  indicator proportions over the original uninformed cells)
- LVM support (both SGS and SIS)
- use_harddata parameter
- mask parameter (selective simulation with the mask overlapping
  originally-uninformed cells)
- Result validation (shape, indicator count, statistics)
- F-133 kriging-failure bounded-output contract (invoked-kriging arm only; the
  variance ≤ 0 fallback branch is NOT constructible from a grid fixture and
  remains unpinned — F-679)

(T-15: use_correlogram tests deleted — the parameter is inert in the
non-LVM path, sis.py:370-376. T-33: vacuous small-grid tests deleted.
D-23/D-24: multiple-realizations pair deleted as redundant with the kept
different-seed tests.)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "geo_bsd"))

try:
    from geo_bsd.cdf import CdfData
    from geo_bsd.geo import ContProperty, CovarianceModel, IndProperty, SugarboxGrid, covariance
    from geo_bsd.sgs import sgs_simulation
    from geo_bsd.sis import sis_simulation
except (ImportError, OSError):
    # Dummy covariance for module-level parametrize decorators
    # Tests are skipped by @pytest.mark.hpgl when HPGL is unavailable
    class _DummyCovarianceTypes:
        spherical = 0
        exponential = 1
        gaussian = 2

    covariance = _DummyCovarianceTypes()


# =============================================================================
# Fixtures for Simulation Tests
# =============================================================================


@pytest.fixture
def sgs_cdf_data_2threshold():
    """CDF data with 2 thresholds (median IK case)"""
    values = np.array([25.0, 75.0], dtype="float32")
    probs = np.array([0.5, 1.0], dtype="float32")
    return CdfData(values, probs)


@pytest.fixture
def sgs_cdf_data_multi():
    """CDF data with multiple thresholds"""
    values = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0], dtype="float32")
    probs = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype="float32")
    return CdfData(values, probs)


@pytest.fixture
def sgs_lvm_mean(sample_grid):
    """LVM mean array for SGS (spatially varying mean)"""
    # Create a gradient mean field
    x, y, z = sample_grid.x, sample_grid.y, sample_grid.z
    mean = np.zeros((x, y, z), dtype="float32", order="F")
    for i in range(x):
        for j in range(y):
            for k in range(z):
                mean[i, j, k] = 30.0 + 10.0 * (i + j + k) / (x + y + z)
    return mean


@pytest.fixture
def sis_data_3indicator():
    """SIS data for 3-indicator case"""
    data = []
    for _i in range(3):
        data.append(
            {
                "cov_model": CovarianceModel(
                    type=covariance.spherical,
                    ranges=(5.0, 5.0, 3.0),
                    angles=(0.0, 0.0, 0.0),
                    sill=1.0,
                    nugget=0.1,
                ),
                "radiuses": (5, 5, 3),
                "max_neighbours": 12,
            }
        )
    return data


@pytest.fixture
def sis_data_5indicator():
    """SIS data for 5-indicator case"""
    data = []
    for _i in range(5):
        data.append(
            {
                "cov_model": CovarianceModel(
                    type=covariance.exponential,
                    ranges=(6.0, 6.0, 4.0),
                    angles=(0.0, 0.0, 0.0),
                    sill=1.0,
                    nugget=0.05,
                ),
                "radiuses": (6, 6, 4),
                "max_neighbours": 15,
            }
        )
    return data


@pytest.fixture
def sis_lvm_marginal_probs(sample_grid):
    """LVM marginal probabilities for SIS (spatially varying)

    Each cell's probabilities across all categories must sum to ~1.0
    (within PROBABILITY_SUM_TOLERANCE). We compute unnormalized values
    then normalize per-cell to ensure this constraint.
    """
    x, y, z = sample_grid.x, sample_grid.y, sample_grid.z
    # Create 3 spatially varying probability fields, then normalize per cell
    marginal_probs = []
    probs_sum = np.zeros((x, y, z), dtype="float32", order="F")
    for cat in range(3):
        probs = np.zeros((x, y, z), dtype="float32", order="F")
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    probs[i, j, k] = 0.2 + 0.1 * cat + 0.05 * (i / x)
        marginal_probs.append(probs)
        probs_sum += probs
    # Normalize per-cell so each cell sums to 1.0
    for cat in range(3):
        marginal_probs[cat] /= probs_sum
    return marginal_probs


@pytest.fixture
def simulation_mask(sample_grid):
    """Mask for selective simulation.

    Covers the central block (2 <= i < x-2, 2 <= j < y-2) AND the x=0 plane
    where the sample_property/sample_indicator_property uninformed cells live
    (mask[::10] = 0 -> flat indices 0,10,..,490 -> x=0 in F-order reshape).
    The x=0-plane overlap is needed so the mask intersects the originally
    uninformed cells (T-11/F-132): with only the central block, the mask is
    disjoint from every uninformed cell and no simulation runs. Half the
    x=0 plane (j < 5) is left outside the mask so tests can also assert that
    out-of-mask uninformed cells are preserved (mask honored).
    """
    x, y, z = sample_grid.x, sample_grid.y, sample_grid.z
    mask = np.zeros((x, y, z), dtype="uint8", order="F")
    # Simulate central region
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if 2 <= i < x - 2 and 2 <= j < y - 2:
                    mask[i, j, k] = 1
    # Overlap with the uninformed x=0 plane (j < 5 half)
    for j in range(min(5, y)):
        for k in range(z):
            mask[0, j, k] = 1
    return mask


# =============================================================================
# SGS - Basic Execution Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationBasic:
    """Test basic SGS execution and parameter handling"""

    def test_sgs_with_2threshold_cdf(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_2threshold
    ):
        """Test SGS with 2-threshold CDF (median case).

        F-668: the original pin sampled ``result.data[result.mask > 0]``, but the
        C++ kernel sets ``m_mask[index] = 1`` for every simulated node
        (property_array.h set_at), so the output mask is ALL-ONES and the pin
        sampled the hard data (0..100) — it could not fail. It now samples a
        pre-simulation snapshot of the ORIGINAL uninformed cells, which the
        kernel fills from the 2-threshold CDF ([25, 75]). Measured: every
        simulated value is in [25.0, 75.0] while the untouched input over the
        same 50 cells spans [1.2, 99.1], so a returned input clone fails.
        """
        orig_uninformed = sample_property.mask == 0  # snapshot BEFORE the call
        input_values = sample_property.data[orig_uninformed].astype("float64").copy()

        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_2threshold,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="sk",
        )

        assert isinstance(result, ContProperty)
        # Results at the originally-uninformed cells must lie within the CDF's
        # value range (flattened: fix_shape() makes both arrays 3D).
        min_cdf = float(sgs_cdf_data_2threshold.values.min())
        max_cdf = float(sgs_cdf_data_2threshold.values.max())
        simulated_values = result.data.astype("float64").reshape(-1, order="F")[orig_uninformed]
        assert np.all(simulated_values >= min_cdf - 1.0)
        assert np.all(simulated_values <= max_cdf + 1.0)
        assert not np.allclose(simulated_values, input_values), (
            "SGS must simulate the uninformed cells, not return the input copy"
        )

    def test_sgs_accepts_tuple_prop(self, sample_grid, sample_covariance_model, sgs_cdf_data_multi):
        """Test SGS accepts tuple input for prop parameter"""
        np.random.seed(42)
        data = np.random.rand(500).astype("float32") * 100
        mask = np.ones(500, dtype="uint8")

        result = sgs_simulation(
            prop=(data, mask),  # Tuple input
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))


# =============================================================================
# SGS - Reproducibility Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationReproducibility:
    """Test SGS reproducibility"""

    def test_sgs_same_seed_same_result(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS produces identical results with same seed.

        F-668: the mask-equality pin was dropped — both masks are all-ones after
        the C++ kernel mutates the mask buffer, so it could not fail. The
        data-array equality below is the surviving (strictly stronger)
        determinism pin.
        """
        result1 = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=12345,
            kriging_type="sk",
        )

        result2 = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=12345,
            kriging_type="sk",
        )

        np.testing.assert_array_equal(result1.data, result2.data)

    def test_sgs_different_seed_different_result(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS produces different results with different seeds"""
        result1 = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="sk",
        )

        result2 = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=12345,
            kriging_type="sk",
        )

        # Results should be different
        assert not np.array_equal(result1.data, result2.data)

    def test_sgs_golden_values_fixed_seed(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """G-57: golden-value determinism pin (literal expected values).

        Every other determinism pin in the suite is DIFFERENTIAL — two runs of
        the same seed compared with each other — so a defect that seeds one RNG
        with a constant (or drops the seed from a substream) leaves them all
        green. This pin fixes the actual stream instead: seed=12345 on this
        fixture must reproduce these 50 values at the originally-uninformed
        cells. ``rtol=1e-5`` absorbs cross-platform LAPACK/libm rounding: the
        values are float32 output, so one ULP is 2.4e-7 absolute at the smallest
        pinned value (2.199) and 7.6e-6 at 93.2 (8.2e-8 relative) — the
        relative tolerance allows 86-163 ULP across the array, i.e. drift from a
        different BLAS/libm rather than bit equality. It still discriminates a
        wrong stream or a wrong transform by orders of magnitude: seed=12346
        gives sum 2836.1048 vs 2501.6024 here, a 13.4% shift (measured
        post-F-388; the pre-F-388 stream summed differently, and these literals
        are that run's post-fix output, reproduced twice bit-identically).
        """
        uninformed = sample_property.mask == 0  # snapshot BEFORE the call
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=12345,
            kriging_type="sk",
            mean=None,
        )

        simulated = result.data.astype("float64").reshape(-1, order="F")[uninformed]
        expected = np.array(
            [
                74.43055725,
                55.44454956,
                32.97294617,
                2.19907880,
                45.37827301,
                21.71049881,
                42.06908798,
                11.15379143,
                37.22657776,
                58.17209244,
                56.26425552,
                33.32684708,
                93.21020508,
                34.57640457,
                26.74790573,
                29.47037506,
                5.08821154,
                29.55109406,
                8.90661812,
                16.82874107,
                4.37538815,
                37.87688828,
                30.01182938,
                60.82464981,
                68.50180817,
                15.02322865,
                44.64905930,
                82.92551422,
                78.31241608,
                44.16361618,
                14.62460136,
                12.00122070,
                89.12643433,
                85.29293060,
                36.42172623,
                53.97834015,
                60.14435196,
                50.33384323,
                93.59438324,
                77.91049957,
                38.29335022,
                83.73644257,
                88.89095306,
                93.71423340,
                91.87658691,
                85.38634491,
                72.10525513,
                62.11160278,
                80.64768219,
                50.01913834,
            ],
            dtype="float64",
        )
        assert simulated.size == expected.size, (
            f"the fixture must leave {expected.size} uninformed cells, got {simulated.size}"
        )
        np.testing.assert_allclose(
            simulated,
            expected,
            rtol=1e-5,
            atol=0,
            err_msg=(
                "SGS seed=12345 no longer reproduces the pinned random stream — "
                "the seed/RNG wiring changed, not just the kriging math"
            ),
        )

    def test_sgs_seed_above_2_32_is_a_distinct_stream(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """G-59: a seed that differs only ABOVE the 32-bit boundary must change
        the stream.

        The whole default suite uses seeds 42/999/12345/54321 (< 2**16), so a
        width collapse — the historical ``static_cast<uint32_t>`` that turned
        seed 2**32+5 into 5 — was invisible. The C++ generator now splits the
        int64 seed into two 32-bit words fed to ``seed_seq``
        (``bs_random_generator.cpp:9-21``); this pin is the regression gate.
        Measured: seed 5 and seed 2**32+5 differ at 50/500 cells, and each seed
        reproduces itself exactly.
        """

        def run(seed):
            return sgs_simulation(
                prop=sample_property,
                grid=sample_grid,
                cdf_data=sgs_cdf_data_multi,
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=sample_covariance_model,
                seed=seed,
                kriging_type="sk",
                mean=None,
            )

        low = run(5)
        low_again = run(5)
        high = run(2**32 + 5)
        high_again = run(2**32 + 5)

        np.testing.assert_array_equal(low.data, low_again.data)
        np.testing.assert_array_equal(high.data, high_again.data)
        assert not np.array_equal(low.data, high.data), (
            "seed 2**32+5 reproduced seed 5's field — the 64-bit seed is being "
            "truncated to 32 bits before the generator is seeded"
        )


# =============================================================================
# SGS - Kriging Type Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationKrigingType:
    """Test SGS kriging type parameter (SK vs OK)"""

    def test_sgs_sk_vs_ok_produce_different_results(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SK and OK produce different results"""
        result_sk = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="sk",
        )

        result_ok = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            kriging_type="ok",
        )

        # SK and OK should produce different results
        assert not np.array_equal(result_sk.data, result_ok.data)

    def test_sgs_invalid_kriging_type_raises_error(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test invalid kriging type raises appropriate error"""
        with pytest.raises(ValueError, match="invalid kriging_type"):
            sgs_simulation(
                prop=sample_property,
                grid=sample_grid,
                cdf_data=sgs_cdf_data_multi,
                radiuses=(5, 5, 3),
                max_neighbours=12,
                cov_model=sample_covariance_model,
                seed=42,
                kriging_type="invalid",
            )


# =============================================================================
# SGS - LVM (Locally Varying Mean) Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationLVM:
    """Test SGS with Locally Varying Mean"""

    def test_sgs_with_lvm_array(
        self,
        sample_property,
        sample_grid,
        sample_covariance_model,
        sgs_lvm_mean,
        sgs_cdf_data_multi,
    ):
        """Test SGS with LVM mean array.

        G-51: the pre-fix oracle here was ``not np.allclose(lvm, auto)``, which
        a mean-handling error still satisfies (measured: it passes with
        ``mean=1e6``). The LVM mean is in DATA space and both kernels forward-
        transform it through the CDF before kriging
        (``sequential_gaussian_simulation.cpp:75-83`` for the scalar mean,
        ``:254`` ``transform_cdf_ptr`` for the LVM buffer), so two oracles that
        ARE discriminating on this fixture are pinned:

        (a) Scalar-vs-constant-LVM parity. ``mean=50.0`` (scalar kernel) and
            ``mean=np.full(..., 50.0)`` (LVM kernel) must produce byte-identical
            output — both route the same value through the same
            ``transform_cdf_s``/``transform_cdf_ptr`` arithmetic
            (``cdf_utils.h:54,70``). Measured: identical, max|diff| = 0.0.
            A LVM mean left in the WRONG space (the ``transform_cdf_ptr`` call
            dropped, so 50 is used as a normal score) is ~100x the CDF's whole
            normal-score range; the largest in-space move measurable here
            (data-space mean 50 -> 100, the CDF top) already shifts every
            simulated cell by up to 52.2 units, so parity fails.
        (b) Direction. The gradient mean (30-40, below the data mean ~50) must
            pull the simulated cells below the auto-mean run: measured
            65.03 vs 66.13 over the 50 originally-uninformed cells. A wrong-space
            mean saturates upwards (measured: an effective mean at the CDF top
            gives 75.86, above the auto run), so it flips this assertion.

        The weak ``not allclose(lvm, auto)`` assertion is kept only as the
        "mean is used at all" smoke pin — it is NOT the oracle (G-51/F-671).
        """
        uninformed = sample_property.mask == 0  # snapshot BEFORE the calls
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mean=sgs_lvm_mean,
        )

        assert isinstance(result, ContProperty)
        assert result.data.shape == sample_property.data.shape
        assert not np.any(np.isnan(result.data.astype("float64")))
        gradient_values = result.data.astype("float64").reshape(-1, order="F")[uninformed]

        result_no_lvm = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mean=None,
        )
        auto_values = result_no_lvm.data.astype("float64").reshape(-1, order="F")[uninformed]
        # Smoke pin: the LVM mean is consulted at all.
        assert not np.allclose(
            result.data.astype("float64"),
            result_no_lvm.data.astype("float64"),
            rtol=1e-4,
            atol=1e-4,
        ), "LVM mean should produce different result than auto-computed mean"

        # (b) G-51 direction oracle: a mean below the data mean pulls the
        # simulated cells down (measured 65.03 vs 66.13). Skipping the mean's
        # forward transform leaves 30-40 in normal-score space and saturates
        # the run upwards, which flips this comparison.
        assert np.mean(gradient_values) < np.mean(auto_values), (
            f"gradient LVM mean (30-40, data mean ~50) must pull the simulated "
            f"cells below the auto-mean run (got {np.mean(gradient_values):.2f} "
            f"vs {np.mean(auto_values):.2f})"
        )

        # (a) G-51 parity oracle: a constant LVM mean is the same quantity as
        # a scalar stationary mean, and both are forward-transformed through
        # this CDF before kriging — the outputs must be byte-identical.
        constant_mean = np.full(
            (sample_grid.x, sample_grid.y, sample_grid.z), 50.0, dtype="float32", order="F"
        )
        result_scalar_mean = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mean=50.0,
        )
        result_constant_lvm = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mean=constant_mean,
        )
        np.testing.assert_array_equal(
            result_scalar_mean.data,
            result_constant_lvm.data,
            err_msg=(
                "a constant LVM mean (50.0) and the scalar mean (50.0) must be "
                "the same quantity in the same space — a difference means the "
                "LVM mean is not forward-transformed through the CDF like the "
                "scalar mean is"
            ),
        )

    def test_sgs_with_scalar_mean(
        self, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS with scalar mean value.

        T-10: the original fixture (sample_property, mean ~50) made this test
        vacuous — the auto-computed mean also lands near 50, so the assertion
        passed whether or not the scalar mean was honored. Use a property
        whose data mean (~5) is far below the scalar mean (50.0) so the
        scalar-mean honoring is observable: the simulated-only mean with
        mean=50.0 must exceed the auto-mean run's simulated-only mean.
        """
        rng = np.random.RandomState(42)
        data = rng.rand(500).astype("float32") * 10  # mean ~ 5
        mask = np.ones(500, dtype="uint8")
        mask[::10] = 0
        prop = ContProperty(data, mask)
        orig_uninformed = (mask.reshape((sample_grid.x, sample_grid.y, sample_grid.z), order="F") == 0)

        result = sgs_simulation(
            prop=prop,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mean=50.0,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        # F-131/T-10: scalar mean must influence the simulation. With data
        # mean ~5, mean=50.0 must pull the SIMULATED cells' mean above the
        # auto-mean run's (empirically ~16 vs ~13 on the 50 uninformed cells).
        sim_values = result.data.astype("float64")[orig_uninformed]
        auto_result = sgs_simulation(
            prop=prop,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mean=None,
        )
        auto_values = auto_result.data.astype("float64")[orig_uninformed]
        assert np.mean(sim_values) > np.mean(auto_values), (
            f"SGS mean=50.0 must pull simulated cells above the auto-mean "
            f"run (got {np.mean(sim_values):.2f} vs {np.mean(auto_values):.2f})"
        )
        assert not np.all(result.data == 0), "SGS scalar mean: result should not be all-zeros"

    def test_sgs_with_mean_none(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS with mean=None (auto-computed).

        F-668: the original pin sampled ``result.data[result.mask > 0]``, which is
        ALL-ONES after the C++ kernel mutates the mask buffer, so the "simulated"
        mean was ~90 % hard data (mean ~50) and the 30 % bound only failed for a
        simulated-cell shift above ~150. It now samples a pre-simulation snapshot
        of the 50 ORIGINAL uninformed cells. Measured on that sample (seed 42):
        simulated mean 66.13 against a data mean of 50.20 — a 31.7 % deviation,
        which is why the bound is 40 % here (the 30 % figure was calibrated on the
        vacuous 500-cell sample and is not reachable on the corrected one). The
        not-allclose pin is what fails if the run returns an input copy.
        """
        orig_uninformed = sample_property.mask == 0  # snapshot BEFORE the call
        input_values = sample_property.data[orig_uninformed].astype("float64").copy()
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mean=None,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        # Smoke claim: auto-computed mean should produce simulated values whose
        # mean is within 40% of the data mean (measured deviation 31.7%).
        data_mean = np.mean(sample_property.data[sample_property.mask == 1].astype("float64"))
        simulated_values = result.data.astype("float64").reshape(-1, order="F")[orig_uninformed]
        simulated_mean = np.mean(simulated_values)
        assert abs(simulated_mean - data_mean) < max(abs(data_mean) * 0.4, 15.0), (
            f"mean=None: simulated mean {simulated_mean:.2f} deviates from "
            f"data mean {data_mean:.2f} by more than 40%"
        )
        assert not np.allclose(simulated_values, input_values), (
            "mean=None: SGS must simulate the uninformed cells, not return the input copy"
        )


# =============================================================================
# SGS - use_harddata Parameter Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationUseHarddata:
    """Test SGS use_harddata parameter"""

    def test_sgs_use_harddata_true(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS with use_harddata=True preserves input data"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            use_harddata=True,
        )

        assert isinstance(result, ContProperty)
        # use_harddata=True must preserve hard data values at informed positions
        informed_mask = sample_property.mask == 1
        np.testing.assert_allclose(
            result.data[informed_mask].astype("float64"),
            sample_property.data[informed_mask].astype("float64"),
            rtol=1e-5,
            err_msg="Hard data values should be preserved with use_harddata=True",
        )

    def test_sgs_use_harddata_false(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS with use_harddata=False ignores input data"""
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            use_harddata=False,
        )

        assert isinstance(result, ContProperty)
        # With use_harddata=False, output should not match input where informed
        # (output is simulated fresh)
        informed_mask = sample_property.mask == 1
        assert not np.allclose(
            result.data[informed_mask].astype("float64"),
            sample_property.data[informed_mask].astype("float64"),
            rtol=1e-5,
        ), "Result should differ from input with use_harddata=False"


# =============================================================================
# SGS - Mask Parameter Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationMask:
    """Test SGS mask parameter for selective simulation"""

    def test_sgs_with_mask(
        self,
        sample_property,
        sample_grid,
        sample_covariance_model,
        sgs_cdf_data_multi,
        simulation_mask,
    ):
        """Test SGS with mask for selective simulation.

        T-11/F-132: the original simulation_mask covered only the central
        block (x=2..7), disjoint from every uninformed cell (all at x=0) — so
        nothing was simulated and the std>0 assertion was armed by hard data.
        The fixture now also permits the x=0 plane (j<5 half), which overlaps
        the originally-uninformed cells. This test asserts:
          (a) in-mask originally-uninformed cells ARE simulated (finite,
              non-trivial spread, and different from their input values — F-667;
              the spread pin alone also passes on the untouched input);
          (b) out-of-mask originally-uninformed cells are byte-identical to
              the input (the mask is honored — a dropped mask fails this).
        """
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
            mask=simulation_mask,
        )

        assert isinstance(result, ContProperty)
        # Check that result shape matches input
        assert result.data.shape == sample_property.data.shape
        mask_3d = simulation_mask.reshape((sample_grid.x, sample_grid.y, sample_grid.z), order="F")
        orig_mask_3d = sample_property.mask.reshape(
            (sample_grid.x, sample_grid.y, sample_grid.z), order="F"
        )
        orig_data_3d = sample_property.data.reshape(
            (sample_grid.x, sample_grid.y, sample_grid.z), order="F"
        )
        sim_data = result.data.astype("float64")

        # (a) in-mask originally-uninformed cells must be simulated
        in_mask_uninformed = (mask_3d == 1) & (orig_mask_3d == 0)
        assert np.sum(in_mask_uninformed) > 0, (
            "simulation_mask must overlap the originally-uninformed cells"
        )
        sim_cells = sim_data[in_mask_uninformed]
        orig_cells = orig_data_3d[in_mask_uninformed].astype("float64")
        assert np.all(np.isfinite(sim_cells)), "Simulated cells must be finite"
        # F-667(b): the std pin below also passes on the untouched input
        # (measured input std 28.64 over these 25 cells), so pin that the cells
        # were actually re-simulated: measured, all 25 differ from their input
        # values in a real run and a returned input clone is byte-identical.
        assert not np.allclose(sim_cells, orig_cells), (
            "in-mask uninformed cells must be re-simulated, not copied from the input"
        )
        assert np.std(sim_cells) > 0, "Simulated cells should not be uniform"

        # (b) out-of-mask originally-uninformed cells are preserved (mask honored)
        out_mask_uninformed = (mask_3d == 0) & (orig_mask_3d == 0)
        assert np.sum(out_mask_uninformed) > 0, "mask should leave some uninformed cells outside"
        np.testing.assert_array_equal(
            sim_data[out_mask_uninformed],
            orig_data_3d[out_mask_uninformed],
            err_msg="out-of-mask uninformed cells must be preserved (mask honored)",
        )


# =============================================================================
# SGS - Statistical Validation Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationStatistics:
    """Test SGS statistical properties"""

    def test_sgs_result_within_cdf_range(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """Test SGS results are within CDF value range.

        F-668: the original pin sampled ``result.data[result.mask > 0]``, which is
        ALL-ONES after the C++ kernel mutates the mask buffer, so it sampled the
        hard data. It now samples a pre-simulation snapshot of the 50 ORIGINAL
        uninformed cells. The range bounds alone cannot separate a no-op here (the
        input values are also inside [0, 100]), so the not-allclose pin against
        the input snapshot is the discriminating assertion (measured: the
        simulated values span [16.8, 99.6] and differ from the input).
        """
        orig_uninformed = sample_property.mask == 0  # snapshot BEFORE the call
        input_values = sample_property.data[orig_uninformed].astype("float64").copy()
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
        )

        min_cdf = sgs_cdf_data_multi.values.min()
        max_cdf = sgs_cdf_data_multi.values.max()

        # Check that simulated values are within CDF range (with tolerance)
        simulated_values = result.data.astype("float64").reshape(-1, order="F")[orig_uninformed]
        assert np.all(simulated_values >= min_cdf - 1.0)
        assert np.all(simulated_values <= max_cdf + 1.0)
        assert not np.allclose(simulated_values, input_values), (
            "SGS must simulate the uninformed cells, not return the input copy"
        )

    def test_sgs_result_variogram_structure(
        self, sample_property, sample_grid, sample_covariance_model, sgs_cdf_data_multi
    ):
        """C24: SGS output must have positive spatial correlation at short lags.

        T-13/N2-02: result.mask is ALL-ONES after the C++ kernel mutates the
        mask buffer (property_array.h set_at), so the original x-direction
        lag loop sampled the whole grid (mostly hard data) and the γ(1)<γ(5)
        gap was sampling noise. This hardened version samples the ORIGINAL
        uninformed cells (sample_property.mask==0 — the x=0 plane, 50 cells)
        and uses y-direction lags (the uninformed cells all lie in x=0, so
        x-lags have no pairs). Empirically: 45 near (y-lag 1) / 25 far
        (y-lag 5) pairs, γ(1)≈300 < γ(5)≈335.

        F-667(a): the γ(1)<γ(5) ordering is NOT sufficient — measured on the RAW
        INPUT the same selection gives γ(1)=921.5 < γ(5)=1061.8, so a no-op run
        satisfies it. The added pin requires the simulated γ(1) to move away from
        the input's own γ(1) by more than 25 % of it (measured: 300.3 vs 921.5,
        a 67 % drop; a returned input clone gives exactly 0).
        """
        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=sample_covariance_model,
            seed=42,
        )

        sim = result.data.astype("float64")
        grid = sample_grid
        # Select the ORIGINAL uninformed cells (all at x=0 for this fixture).
        orig_uninformed = (sample_property.mask.reshape((grid.x, grid.y, grid.z), order="F") == 0)
        input_data = sample_property.data.astype("float64").reshape(
            (grid.x, grid.y, grid.z), order="F"
        )

        # y-direction lags: near = distance 1, far = distance 5 (≥ range).
        near_pairs = []
        far_pairs = []
        for x in range(grid.x):
            for y in range(grid.y):
                for z in range(grid.z):
                    if not orig_uninformed[x, y, z]:
                        continue
                    if y + 1 < grid.y and orig_uninformed[x, y + 1, z]:
                        near_pairs.append(((x, y, z), (x, y + 1, z)))
                    if y + 5 < grid.y and orig_uninformed[x, y + 5, z]:
                        far_pairs.append(((x, y, z), (x, y + 5, z)))

        assert near_pairs and far_pairs, (
            "T-13: expected non-empty lag pairs over the original-uninformed cells"
        )

        def gamma(values, pairs):
            diffs = [(values[a] - values[b]) ** 2 for a, b in pairs]
            return 0.5 * float(np.mean(diffs))

        gamma_near = gamma(sim, near_pairs)  # Experimental γ(lag 1)
        gamma_far = gamma(sim, far_pairs)  # Experimental γ(lag 5)
        gamma_near_input = gamma(input_data, near_pairs)  # same pairs on the input

        # The spherical model with sill=1.0, nugget=0.1, range=5.0 predicts:
        #   γ(h=1) ≈ 0.1 + 0.9 * (1.5*(1/5) - 0.5*(1/5)^3) ≈ 0.1 + 0.267 = 0.367
        #   γ(h=5) ≈ 0.1 + 0.9 * 1.0 = 1.0
        # Short-lag variance must be LESS than long-lag variance
        assert gamma_near < gamma_far, (
            f"γ(lag 1)={gamma_near:.3f} should be < γ(lag 5)={gamma_far:.3f} "
            f"for a positively correlated field"
        )
        # F-667(a): the ordering above also holds on the raw input (921.5 < 1061.8),
        # so it cannot separate a no-op from a real run. The simulated short-lag
        # semivariance must differ materially from the input's.
        assert abs(gamma_near - gamma_near_input) > 0.25 * max(gamma_near_input, 1.0), (
            f"γ(lag 1) of the output ({gamma_near:.3f}) is indistinguishable from "
            f"the input's ({gamma_near_input:.3f}) — the uninformed cells were not simulated"
        )


# =============================================================================
# SGS - Covariance Model Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialGaussianSimulationCovariance:
    """Test SGS with different covariance models"""

    @pytest.mark.parametrize(
        "cov_type", [covariance.spherical, covariance.exponential, covariance.gaussian]
    )
    def test_sgs_different_covariance_types(
        self, sample_property, sample_grid, sgs_cdf_data_multi, cov_type
    ):
        """Test SGS with different covariance types.

        F-681: the parametrization implied three-model coverage but asserted only
        ``isinstance`` + no-NaN/Inf, and the NaN/Inf pins are unreachable-false
        (sgs.py raises on non-finite output first). Each case now also runs the
        same seed with a DIFFERENT covariance type and asserts the output
        differs — a dispatch regression that ignores ``cov_model.type`` collapses
        the two runs to identical output. Measured: 50/500 cells differ for every
        model pair (spherical/exponential/gaussian).
        """
        cov_model = CovarianceModel(
            type=cov_type, ranges=(5.0, 5.0, 3.0), angles=(0.0, 0.0, 0.0), sill=1.0, nugget=0.1
        )

        result = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
            seed=42,
        )

        assert isinstance(result, ContProperty)
        assert not np.any(np.isnan(result.data.astype("float64")))
        assert not np.any(np.isinf(result.data.astype("float64")))

        # F-681: model-dispatch pin — a different cov_model.type, same seed, must
        # produce a different field (the model is actually consulted).
        other_type = (
            covariance.gaussian if cov_type != covariance.gaussian else covariance.spherical
        )
        other_model = CovarianceModel(
            type=other_type, ranges=(5.0, 5.0, 3.0), angles=(0.0, 0.0, 0.0), sill=1.0, nugget=0.1
        )
        other = sgs_simulation(
            prop=sample_property,
            grid=sample_grid,
            cdf_data=sgs_cdf_data_multi,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=other_model,
            seed=42,
        )
        assert not np.array_equal(result.data, other.data), (
            f"cov_type={cov_type} and cov_type={other_type} produced identical "
            f"output — the covariance model type is not reaching the kernel"
        )


# =============================================================================
# SIS - Basic Execution Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialIndicatorSimulationBasic:
    """Test basic SIS execution and parameter handling"""

    def test_sis_basic_execution_5indicator(
        self, sample_grid, sis_data_5indicator
    ):
        """Test SIS with 5 indicators.

        Uses a 5-category property matching the 5-indicator data config
        (F-24: indicator_count must match len(data) — a mismatch now raises).

        N2-06: kept as the 5-indicator SIS execution test (2/3-indicator execution
        is pinned by test_sis_fixes.py F1 arithmetic; the parametrized twin is
        fully-informed). F-680: ``isinstance`` + ``indicator_count`` both hold on
        the input clone that ``sis.py`` returns when ``use_harddata=True``, so they
        proved only that the call did not raise. The 50 originally-uninformed cells
        must now differ from their input categories (measured: 41/50 differ after a
        real run).
        """
        data = np.random.RandomState(42).randint(0, 5, 500, dtype="uint8")
        mask = np.ones(500, dtype="uint8")
        mask[::10] = 0
        orig_uninformed = mask == 0  # snapshot BEFORE the call
        input_categories = data[orig_uninformed].copy()
        prop = IndProperty(data, mask, 5)
        result = sis_simulation(
            prop=prop,
            grid=sample_grid,
            data=sis_data_5indicator,
            seed=42,
            marginal_probs=[0.2, 0.2, 0.2, 0.2, 0.2],
        )

        assert isinstance(result, IndProperty)
        assert result.indicator_count == 5
        # F-680: the kernel must actually simulate the uninformed cells.
        simulated = result.data.reshape(-1, order="F")[orig_uninformed]
        assert np.all(simulated < result.indicator_count), (
            "SIS simulated cells must be valid categories"
        )
        assert not np.array_equal(simulated, input_categories), (
            "5-indicator SIS must simulate the uninformed cells, not return the input clone"
        )

    def test_sis_accepts_tuple_prop(self, sample_grid, sis_data_3indicator):
        """Test SIS accepts tuple input for prop parameter"""
        np.random.seed(42)
        data = np.random.randint(0, 3, 500, dtype="uint8")
        mask = np.ones(500, dtype="uint8")

        result = sis_simulation(
            prop=(data, mask, 3),  # Tuple input
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        assert isinstance(result, IndProperty)
        assert result.data.shape[0] == 500
        assert np.all(result.data < result.indicator_count)


# =============================================================================
# SIS - Reproducibility Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialIndicatorSimulationReproducibility:
    """Test SIS reproducibility"""

    def test_sis_same_seed_same_result(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS produces identical results with same seed.

        F-668: the mask-equality pin was dropped — both masks are all-ones after
        the C++ kernel mutates the mask buffer, so it could not fail. The
        data-array equality below is the surviving (strictly stronger)
        determinism pin.
        """
        result1 = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=54321,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        result2 = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=54321,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        np.testing.assert_array_equal(result1.data, result2.data)

    def test_sis_different_seed_different_result(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS produces different results with different seeds"""
        result1 = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        result2 = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=999,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        # Results should be different
        assert not np.array_equal(result1.data, result2.data)

    def test_sis_golden_values_fixed_seed(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """G-57: golden-value determinism pin for SIS (literal expected values).

        The reproducibility pins in this class are differential only, so a
        constant-seeded RNG stays green there. This pin fixes the stream for
        seed=54321 on this fixture: the category counts over the 50
        originally-uninformed cells must be [20, 27, 3] and the first 12 drawn
        categories must match the literal sequence below. A different seed
        changes the counts (seed 54421 gives [10, 25, 15]; 2**32+54321 gives
        [24, 21, 5], so the pin reads the stream, not just the marginal
        probabilities). All four counts are the measured post-F-388 stream,
        reproduced bit-identically on a re-run.
        """
        uninformed = sample_indicator_property.mask == 0  # snapshot BEFORE the call
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=54321,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        drawn = result.data.reshape(-1, order="F")[uninformed].astype("int64")
        assert drawn.size == 50, f"the fixture must leave 50 uninformed cells, got {drawn.size}"
        np.testing.assert_array_equal(
            np.bincount(drawn, minlength=3),
            np.array([20, 27, 3]),
            err_msg=(
                "SIS seed=54321 no longer reproduces the pinned category counts — "
                "the seed/RNG wiring changed, not just the kriging math"
            ),
        )
        np.testing.assert_array_equal(
            drawn[:12],
            np.array([0, 2, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]),
            err_msg="SIS seed=54321 no longer reproduces the pinned draw sequence",
        )

    def test_sis_seed_above_2_32_is_a_distinct_stream(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """G-59: a SIS seed that differs only above the 32-bit boundary must
        change the stream.

        The default suite uses seeds 42/999/12345/54321 (< 2**16), so a width
        collapse was invisible. Measured: seed 5 and seed 2**32+5 differ at
        33/500 cells, and each seed reproduces itself exactly.
        """

        def run(seed):
            return sis_simulation(
                prop=sample_indicator_property,
                grid=sample_grid,
                data=sis_data_3indicator,
                seed=seed,
                marginal_probs=[0.3, 0.4, 0.3],
            )

        low = run(5)
        low_again = run(5)
        high = run(2**32 + 5)
        high_again = run(2**32 + 5)

        np.testing.assert_array_equal(low.data, low_again.data)
        np.testing.assert_array_equal(high.data, high_again.data)
        assert not np.array_equal(low.data, high.data), (
            "seed 2**32+5 reproduced seed 5's field — the 64-bit seed is being "
            "truncated to 32 bits before the generator is seeded"
        )


# =============================================================================
# SIS - LVM (Locally Varying Marginal Probabilities) Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialIndicatorSimulationLVM:
    """Test SIS with Locally Varying Marginal Probabilities"""

    def test_sis_with_lvm_marginal_probs(
        self, sample_indicator_property, sample_grid, sis_data_3indicator, sis_lvm_marginal_probs
    ):
        """Test SIS with LVM marginal probabilities.

        F-671: the original test asserted only ``isinstance`` +
        ``indicator_count``, so a mutation that IGNORES or TRANSPOSES
        ``marginal_probs`` stayed green — all 50 ``sis_simulation`` call sites
        were enumerated and no LVM site asserted a value, a distribution or an
        LVM-vs-scalar difference. The behavioural oracle below runs two LVM cases
        with the per-cell marginals REVERSED (category 2 requested at 0.7 then
        0.1) at the same seed. Measured (seed 42, the 50 originally-uninformed
        cells): shares [0.36, 0.38, 0.26] vs [0.36, 0.20, 0.44] and 9/50 cells
        differ. An implementation that ignores the arrays returns the same field
        for both runs; a transposed implementation swaps them — both fail. Note
        that category 0 does NOT track a 0.7 request on this fixture (both runs
        give 0.36), so the pin is the contrast, not an absolute share.

        This pin stands alone: the SGS-LVM mean test is NOT cited as coverage for
        it (F-671/G-51 oracle family; G-51 measures that test as
        non-discriminating).
        """
        orig_uninformed = sample_indicator_property.mask == 0  # snapshot BEFORE the calls
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=sis_lvm_marginal_probs,
        )

        assert isinstance(result, IndProperty)
        assert result.indicator_count == 3

        def lvm_constant(weights):
            return [
                np.full(
                    (sample_grid.x, sample_grid.y, sample_grid.z), w, dtype="float32", order="F"
                )
                for w in weights
            ]

        high_cat2 = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=lvm_constant([0.7, 0.2, 0.1]),
        )
        low_cat2 = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=lvm_constant([0.1, 0.2, 0.7]),
        )

        assert not np.array_equal(high_cat2.data, low_cat2.data), (
            "reversed LVM marginal_probs produced identical output — the arrays "
            "are being ignored (or transposed)"
        )
        share_cat2_high = float(
            np.mean(high_cat2.data.reshape(-1, order="F")[orig_uninformed] == 2)
        )
        share_cat2_low = float(
            np.mean(low_cat2.data.reshape(-1, order="F")[orig_uninformed] == 2)
        )
        assert share_cat2_high < share_cat2_low, (
            f"category 2 share {share_cat2_high:.3f} (requested 0.1) should be "
            f"< {share_cat2_low:.3f} (requested 0.7) — the marginal probabilities "
            f"are not driving the draw"
        )

    def test_sis_lvm_use_correlogram_false(
        self, sample_indicator_property, sample_grid, sis_data_3indicator, sis_lvm_marginal_probs
    ):
        """F-677: the LVM ``use_correlogram=False`` dispatch must execute.

        ``sis.py:405-416`` does not forward ``use_correlogram`` at all in the
        non-LVM path, and every LVM test in the suite passes ``True``, so the C++
        ``sk_weight_calculator_t`` branch
        (sequential_indicator_simulation.cpp:343-356) had no executing test.

        Measured: on this fixture the ``True`` and ``False`` runs are
        BYTE-IDENTICAL (0/50 uninformed cells differ) — both weight calculators
        agree for these indicator sills — so no True-vs-False behavioural
        difference is asserted (doing so would be a permanently-red test). The pin
        is execution coverage plus the kriging-path counters: the call completes,
        yields valid categories, and the C++ SIS path reports kriging work
        (measured points_calculated = 150 = 50 uninformed nodes x 3 indicators).
        The uninformed cells must also differ from the input clone.
        """
        from geo_bsd import geo

        orig_uninformed = sample_indicator_property.mask == 0  # snapshot BEFORE the call
        input_categories = sample_indicator_property.data[orig_uninformed].copy()
        geo._last_kriging_stats = {"points_calculated": 0}  # stale value must fail > 0
        try:
            result = sis_simulation(
                prop=sample_indicator_property,
                grid=sample_grid,
                data=sis_data_3indicator,
                seed=42,
                marginal_probs=sis_lvm_marginal_probs,
                use_correlogram=False,
            )
            stats = geo._last_kriging_stats
        finally:
            geo._last_kriging_stats = None

        assert isinstance(result, IndProperty)
        assert result.indicator_count == 3
        assert np.all(result.data < result.indicator_count), (
            "SIS LVM (use_correlogram=False) simulated cells must be valid categories"
        )
        assert stats is not None, (
            "geo._last_kriging_stats should be populated after an LVM SIS run"
        )
        assert stats["points_calculated"] > 0, (
            f"the LVM use_correlogram=False branch did not krige any node: {stats}"
        )
        simulated = result.data.reshape(-1, order="F")[orig_uninformed]
        assert not np.array_equal(simulated, input_categories), (
            "SIS LVM (use_correlogram=False) must simulate the uninformed cells, "
            "not return the input clone"
        )

    def test_sis_with_scalar_marginal_probs(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS with scalar marginal probabilities"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        assert isinstance(result, IndProperty)
        assert np.all(result.data < result.indicator_count)


# =============================================================================
# SIS - use_harddata Parameter Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialIndicatorSimulationUseHarddata:
    """Test SIS use_harddata parameter"""

    def test_sis_use_harddata_true(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS with use_harddata=True preserves input data"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
            use_harddata=True,
        )

        assert isinstance(result, IndProperty)
        # use_harddata=True must preserve hard data values at informed positions
        informed_mask = sample_indicator_property.mask == 1
        assert np.array_equal(
            result.data[informed_mask], sample_indicator_property.data[informed_mask]
        ), "Hard data should be preserved with use_harddata=True"

    def test_sis_use_harddata_false(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS with use_harddata=False ignores input data"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
            use_harddata=False,
        )

        assert isinstance(result, IndProperty)
        # With use_harddata=False, output should generally differ from input
        informed_mask = sample_indicator_property.mask == 1
        assert not np.array_equal(
            result.data[informed_mask], sample_indicator_property.data[informed_mask]
        ), "Result should differ from input with use_harddata=False"


# =============================================================================
# SIS - Mask Parameter Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialIndicatorSimulationMask:
    """Test SIS mask parameter for selective simulation"""

    def test_sis_with_mask(
        self, sample_indicator_property, sample_grid, sis_data_3indicator, simulation_mask
    ):
        """Test SIS with mask for selective simulation.

        T-11/F-132: the original test was pure smoke (isinstance + shape).
        The simulation_mask fixture now overlaps the originally-uninformed
        cells (x=0 plane half), so this test asserts:
          (a) in-mask originally-uninformed cells ARE simulated (valid
              category values, non-trivial spread, and different from their
              input categories — F-667; spread alone passes on the input);
          (b) out-of-mask originally-uninformed cells are preserved
              (mask honored).
        """
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
            mask=simulation_mask,
        )

        assert isinstance(result, IndProperty)
        assert result.data.shape == sample_indicator_property.data.shape
        mask_3d = simulation_mask.reshape((sample_grid.x, sample_grid.y, sample_grid.z), order="F")
        orig_mask_3d = sample_indicator_property.mask.reshape(
            (sample_grid.x, sample_grid.y, sample_grid.z), order="F"
        )
        orig_data_3d = sample_indicator_property.data.reshape(
            (sample_grid.x, sample_grid.y, sample_grid.z), order="F"
        )

        # (a) in-mask originally-uninformed cells must be simulated (valid category)
        in_mask_uninformed = ((mask_3d == 1) & (orig_mask_3d == 0)).ravel(order="F")
        assert np.sum(in_mask_uninformed) > 0, (
            "simulation_mask must overlap the originally-uninformed cells"
        )
        sim_cells = result.data[in_mask_uninformed]
        orig_cells = orig_data_3d.ravel(order="F")[in_mask_uninformed]
        assert np.all(sim_cells < result.indicator_count), (
            "SIS simulated cells must be valid categories"
        )
        # F-667(c): the two pins below also hold on the untouched input (measured:
        # the input carries all three categories over these 25 cells, so
        # len(unique) >= 2), so pin that the cells were actually re-simulated:
        # measured, 16/25 differ from their input categories in a real run.
        assert not np.array_equal(sim_cells, orig_cells), (
            "in-mask uninformed cells must be re-simulated, not copied from the input"
        )
        assert len(np.unique(sim_cells)) >= 2, "SIS simulated cells should not be uniform"

        # (b) out-of-mask originally-uninformed cells are preserved (mask honored)
        out_mask_uninformed = ((mask_3d == 0) & (orig_mask_3d == 0)).ravel(order="F")
        assert np.sum(out_mask_uninformed) > 0, "mask should leave some uninformed cells outside"
        np.testing.assert_array_equal(
            result.data[out_mask_uninformed],
            orig_data_3d.ravel(order="F")[out_mask_uninformed],
            err_msg="out-of-mask uninformed cells must be preserved (mask honored)",
        )


# =============================================================================
# SIS - Statistical Validation Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialIndicatorSimulationStatistics:
    """Test SIS statistical properties"""

    def test_sis_result_within_indicator_range(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """Test SIS results are within valid indicator range"""
        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        # All values should be less than indicator_count
        assert np.all(result.data < result.indicator_count)

    def test_sis_indicator_proportions(
        self, sample_indicator_property, sample_grid, sis_data_3indicator
    ):
        """C24: SIS indicator proportions should approximately match marginal_probs.

        T-12/N2-03: result.mask is ALL-ONES after the C++ kernel mutates the
        mask buffer, so the old `result.data[result.mask > 0]` sampled ALL 500
        cells (450 hard + 50 simulated) and passed by hard-data dominance.
        This version samples the ORIGINAL uninformed cells (50 cells) and asserts
        the simulated proportions follow marginal_probs.

        F-666: sampling the right cells was not enough — the fixture's OWN
        uninformed cells are a uniform draw (measured prior [0.34, 0.34, 0.32]),
        so the asserted target [0.3, 0.4, 0.3] was already satisfied by the
        untouched input (maxdev 0.06 < 0.15) and a no-op `sis_simulation` passed.
        The pre-simulation prior is therefore forced to a single category
        ([1, 0, 0]) on a local property: a no-op now fails with maxdev 0.70,
        while the real kernel still produces [0.36, 0.36, 0.28] (maxdev 0.06)
        because it draws from the REQUESTED marginals, not from the input.

        E-05 (post-fix TEST-UPDATE): seed 42 → simulated-only proportions
        [0.36, 0.36, 0.28], max deviation 0.06 vs the 0.15 threshold (0.09
        headroom) — measured again with the forced [1,0,0] prior.
        """
        data = sample_indicator_property.data.copy("F")
        mask = sample_indicator_property.mask.copy("F")
        uninformed = mask == 0  # snapshot BEFORE the call
        data[uninformed] = 0  # F-666: pre-simulation prior becomes [1, 0, 0]
        prop = IndProperty(data, mask, 3)

        result = sis_simulation(
            prop=prop,
            grid=sample_grid,
            data=sis_data_3indicator,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        simulated = result.data.reshape(-1, order="F")[uninformed]
        n = len(simulated)
        assert n > 0, "original fixture must have uninformed cells to sample"
        target_probs = [0.3, 0.4, 0.3]

        for i, target in enumerate(target_probs):
            actual_prop = np.sum(simulated == i) / n
            assert abs(actual_prop - target) < 0.15, (
                f"Indicator {i}: proportion {actual_prop:.3f} deviates from "
                f"target {target:.3f} by more than 0.15"
            )


# =============================================================================
# SIS - Covariance Model Tests
# =============================================================================


@pytest.mark.hpgl
class TestSequentialIndicatorSimulationCovariance:
    """Test SIS with different covariance models"""

    @pytest.mark.parametrize(
        "cov_type", [covariance.spherical, covariance.exponential, covariance.gaussian]
    )
    def test_sis_different_covariance_types(self, sample_indicator_property, sample_grid, cov_type):
        """Test SIS with different covariance types.

        F-681: the parametrization implied three-model coverage but the only
        assertion (``np.all(result.data < indicator_count)``) is true for any
        model, including one silently mapped to spherical. Each case now also runs
        the same seed with a DIFFERENT covariance type and asserts the output
        differs (measured: 4-10 of the 500 cells differ for every model pair).
        """
        sis_data = []
        for _i in range(3):
            sis_data.append(
                {
                    "cov_model": CovarianceModel(
                        type=cov_type,
                        ranges=(5.0, 5.0, 3.0),
                        angles=(0.0, 0.0, 0.0),
                        sill=1.0,
                        nugget=0.1,
                    ),
                    "radiuses": (5, 5, 3),
                    "max_neighbours": 12,
                }
            )

        result = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=sis_data,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        other_type = (
            covariance.gaussian if cov_type != covariance.gaussian else covariance.spherical
        )
        other_data = []
        for _i in range(3):
            other_data.append(
                {
                    "cov_model": CovarianceModel(
                        type=other_type,
                        ranges=(5.0, 5.0, 3.0),
                        angles=(0.0, 0.0, 0.0),
                        sill=1.0,
                        nugget=0.1,
                    ),
                    "radiuses": (5, 5, 3),
                    "max_neighbours": 12,
                }
            )

        other = sis_simulation(
            prop=sample_indicator_property,
            grid=sample_grid,
            data=other_data,
            seed=42,
            marginal_probs=[0.3, 0.4, 0.3],
        )

        assert isinstance(result, IndProperty)
        assert np.all(result.data < result.indicator_count)
        # F-681: model-dispatch pin — a different cov_model.type, same seed, must
        # produce a different field (measured: 4-10 of the 500 cells differ for
        # every model pair); a kernel that ignores cov_model.type makes the two
        # runs identical.
        assert not np.array_equal(result.data, other.data), (
            f"cov_type={cov_type} and cov_type={other_type} produced identical "
            f"output — the covariance model type is not reaching the kernel"
        )


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.hpgl
class TestSimulationEdgeCases:
    """Test edge cases and error handling"""

    # F-133: SGS kriging-failure test (variance ≤ 0 fallback)
    def test_sgs_kriging_invoked_bounded_output(self):
        """F-133/F-679: SGS with kriging INVOKED produces bounded finite output.

        T-14: the original fixture was fully-informed (mask all ones), so the
        C++ kernel skipped every node before the kriging call and the assertions
        ran on the identity clone. This fixture leaves ONE cell uninformed
        (mask [1,1,1,0]) so the kriging path is genuinely invoked.

        F-679: this test does NOT cover the F-133 variance ≤ 0 fallback, and the
        old name/docstring claimed it did. Measured on this fixture:
        points_calculated=1 and points_singularity=0, i.e. kriging SUCCEEDED and
        the ``ki_result != KI_SUCCESS`` fallback branch
        (sequential_simulation.h:393-395) is not taken — the degenerate case is
        not constructible here because the four data sit at distinct grid nodes
        (non-singular covariance matrix). The assertions now pin what the fixture
        can actually distinguish: kriging ran, the single simulated cell was
        inverted (4.696 ≠ its 5.0 input value, so a returned input clone fails),
        and the output is finite and bounded. The F-133 fallback branch remains
        UNPINNED — that gap is recorded, not papered over.
        """
        from geo_bsd import geo

        data = np.array([5.0, 5.0, 5.0, 5.0], dtype="float32")
        mask = np.array([1, 1, 1, 0], dtype="uint8")  # one uninformed → kriging invoked
        grid = SugarboxGrid(x=2, y=2, z=1)

        prop = ContProperty(data, mask)

        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(2.0, 2.0, 2.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.0,
        )

        cdf_data = CdfData(
            values=np.array([0.0, 10.0], dtype="float32"),
            probs=np.array([0.0, 1.0], dtype="float32"),
        )

        geo._last_kriging_stats = {"points_calculated": 0}  # stale value must fail > 0
        try:
            result = sgs_simulation(
                prop=prop,
                grid=grid,
                cdf_data=cdf_data,
                radiuses=(3, 3, 3),
                max_neighbours=4,
                cov_model=cov_model,
                seed=42,
            )
            stats = geo._last_kriging_stats
        finally:
            geo._last_kriging_stats = None

        assert isinstance(result, ContProperty)
        assert stats is not None, (
            "geo._last_kriging_stats should be populated after sgs_simulation"
        )
        # F-679: kriging was invoked for the single uninformed cell and succeeded
        # (the F-133 failure branch is NOT taken by this fixture).
        assert stats["points_calculated"] == 1, (
            f"expected exactly one kriged node, got {stats}"
        )
        assert stats["points_singularity"] == 0, (
            f"this fixture is not singular — it cannot pin the F-133 fallback: {stats}"
        )
        # F-667(d)/F-679: the input clone holds 5.0 at the uninformed cell; a real
        # run inverts the CDF there (measured 4.6961), so an input copy fails.
        simulated_value = float(result.data.reshape(-1, order="F")[3])
        assert abs(simulated_value - 5.0) > 1e-6, (
            f"the uninformed cell ({simulated_value}) was not simulated — it still "
            f"holds its input value"
        )
        assert not np.any(np.isnan(result.data.astype("float64"))), (
            "SGS kriging failure: result should not contain NaN"
        )
        assert not np.any(np.isinf(result.data.astype("float64"))), (
            "SGS kriging failure: result should not contain Inf"
        )
        # Bounded output: the simulated value comes from the [0, 10] CDF (measured
        # max 5.0), so no extreme value can escape.
        result_flat = result.data.astype("float64").flatten()
        assert np.all(np.abs(result_flat) < 10.0), (
            f"result values {result_flat} should be bounded by the CDF value range"
        )
        # F-679: the former ``not np.all(result.data == 0)`` pin was unfailable —
        # three of the four cells are hard data at 5.0 — and is replaced by the
        # simulated-value pin above, which fails on an input copy.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
