import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import ContProperty, SugarboxGrid
    from geo_bsd.gtsim import (
        _norm_cdf,
        _norm_ppf,
        pseudo_gaussian_transform,
        tk_calculation,
    )
except (ImportError, OSError):
    pass  # HPGL_AVAILABLE from conftest handles availability


@pytest.mark.hpgl
class TestCdfDataInverse:
    """Unit tests for CdfData.inverse — the F-02 data-space threshold mapper.

    Mirrors the C++ non_parametric_cdf_2_t::inverse semantics
    (non_parametric_cdf.h:263-292): probabilities below the first cumulative
    probability map to the smallest value, above the last map to the largest
    value, and interior probabilities are linearly interpolated between the
    bracketing (value, prob) pairs (std::lower_bound over the probs array).
    """

    def _cdf(self):
        from geo_bsd.cdf import CdfData

        return CdfData(
            values=np.array([0.0, 10.0, 20.0], dtype="float32"),
            probs=np.array([0.2, 0.5, 1.0], dtype="float32"),
        )

    def test_below_first_prob_maps_to_smallest_value(self):
        c = self._cdf()
        assert c.inverse(0.1) == 0.0
        assert c.inverse(0.2) == 0.0  # first prob inclusive (lower_bound)

    def test_above_last_prob_maps_to_largest_value(self):
        c = self._cdf()
        assert c.inverse(1.0) == 20.0
        assert c.inverse(1.5) == 20.0

    def test_interior_linear_interpolation(self):
        c = self._cdf()
        # 0.35 between (0.0, 0.2) and (10.0, 0.5): x1 + (x2-x1)/(y2-y1)*(p-y1)
        # = 0 + 10/0.3 * 0.15 = 5.0
        assert abs(c.inverse(0.35) - 5.0) < 1e-6
        # 0.9 between (10.0, 0.5) and (20.0, 1.0): 10 + 10/0.5*0.4 = 18.0
        assert abs(c.inverse(0.9) - 18.0) < 1e-6

    def test_vectorized(self):
        c = self._cdf()
        out = c.inverse(np.array([0.1, 0.35, 0.9, 1.5]))
        np.testing.assert_allclose(out, [0.0, 5.0, 18.0, 20.0], atol=1e-6)

    def test_scalar_returns_scalar(self):
        c = self._cdf()
        assert np.isscalar(c.inverse(0.5))

    def test_empty_cdf_raises(self):
        from geo_bsd.cdf import CdfData

        c = CdfData(values=np.array([], dtype="float32"),
                    probs=np.array([], dtype="float32"))
        with pytest.raises(ValueError, match="empty CDF"):
            c.inverse(0.5)

    def test_roundtrip_with_calc_cdf(self):
        """F-02 end-to-end building block: inverse(prob(value)) == value for
        informed data (CDF built by calc_cdf from a property)."""
        from geo_bsd.cdf import calc_cdf

        rng = np.random.RandomState(7)
        values = (rng.rand(200) * 10).astype("float32")
        prop = ContProperty(values, np.ones(200, dtype="uint8"))
        c = calc_cdf(prop)
        # For each unique value the cumulative prob inverts back to ~value.
        for v in np.unique(values):
            p = float(np.mean(values <= v))
            recovered = c.inverse(p)
            assert abs(recovered - v) < 1.0


def _make_cont_prop(size, values=None, mask=None):
    if values is None:
        np.random.seed(42)
        values = np.random.rand(size).astype("float32")
    else:
        values = np.array(values, dtype="float32")
    if mask is None:
        mask = np.ones(size, dtype="uint8")
    else:
        mask = np.array(mask, dtype="uint8")
    return ContProperty(values, mask)


@pytest.mark.hpgl
class TestTkCalculation:
    def test_basic_calculation(self):
        """Execution smoke + mutation contract.

        V-29: the finite-only tk smokes (default params / custom mean / custom
        std_dev) are merged here — they are non-discriminating (finite input
        stays finite even if the threshold computation is skipped), so they
        add only crash detection, which this execution smoke already provides.
        """
        prop = _make_cont_prop(10, values=[0.5] * 10)
        result = tk_calculation(prop)
        assert result is prop
        assert result.data.size == 10
        assert np.all(np.isfinite(result.data))

        # Merged smoke: default (mean=0, std_dev=1) inverse-CDF thresholds finite.
        prop2 = _make_cont_prop(100)
        result2 = tk_calculation(prop2)
        assert np.all(np.isfinite(result2.data.flat[:]))

        # Merged smoke: custom mean / std_dev produce finite output.
        prop3 = _make_cont_prop(10, values=[0.5] * 10)
        assert np.all(np.isfinite(tk_calculation(prop3, mean=0.5).data))
        assert np.all(np.isfinite(tk_calculation(prop3, std_dev=2.0).data))

    def test_zero_std_dev_raises(self):
        prop = _make_cont_prop(10, values=[0.5] * 10)
        with pytest.raises(ValueError, match="std_dev must be positive"):
            tk_calculation(prop, std_dev=0)

    def test_negative_std_dev_raises(self):
        prop = _make_cont_prop(10, values=[0.5] * 10)
        with pytest.raises(ValueError, match="std_dev must be positive"):
            tk_calculation(prop, std_dev=-1.0)

    def test_inverse_cdf_monotonic(self):
        """Inverse CDF thresholds decrease as input probabilities increase.

        Since t = mean - std_dev * Φ⁻¹(p), and Φ⁻¹ is strictly increasing,
        larger p → larger Φ⁻¹(p) → smaller t. So pk=0.1 gives a higher
        threshold than pk=0.9."""
        values_low_prob = np.array([0.1] * 5, dtype="float32")
        values_high_prob = np.array([0.9] * 5, dtype="float32")
        prop_low = _make_cont_prop(5, values=list(values_low_prob))
        prop_high = _make_cont_prop(5, values=list(values_high_prob))
        result_low = tk_calculation(prop_low, mean=0.0, std_dev=1.0)
        result_high = tk_calculation(prop_high, mean=0.0, std_dev=1.0)
        # Lower probability → higher threshold (makes indicator=1 less likely)
        assert result_low.data.flat[0] > result_high.data.flat[0]

    def test_larger_std_dev_spreads_thresholds(self):
        """Larger std_dev should produce wider spread in inverse CDF thresholds.

        Since t = mean - std_dev * Φ⁻¹(p), scaling std_dev scales the spread."""
        values = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype="float32")
        prop_narrow = _make_cont_prop(5, values=list(values))
        prop_wide = _make_cont_prop(5, values=list(values.copy()))
        result_narrow = tk_calculation(prop_narrow, mean=0.0, std_dev=0.5)
        result_wide = tk_calculation(prop_wide, mean=0.0, std_dev=5.0)
        range_narrow = float(np.max(result_narrow.data) - np.min(result_narrow.data))
        range_wide = float(np.max(result_wide.data) - np.min(result_wide.data))
        assert range_wide > range_narrow

    def test_mutates_input_property(self):
        prop = _make_cont_prop(5, values=[0.1, 0.3, 0.5, 0.7, 0.9])
        original_data = prop.data.copy()
        tk_calculation(prop)
        assert not np.array_equal(prop.data, original_data)


@pytest.mark.hpgl
class TestNormPpfAccuracy:
    """G-61: pin the normal-quantile approximation's MEASURED accuracy.

    ``_norm_ppf`` (``gtsim.py:45-79``) is the Abramowitz & Stegun 26.2.23
    rational approximation. It used to be documented as
    "Beasley-Springer-Moro … ~4.5e-5", which is false by 10x: re-measured
    against a 50-digit mpmath reference over p in [1e-15, 1-1e-15], the max
    absolute error is **4.442946e-04 at p=0.358**, and the Φ(Φ⁻¹(p)) round-trip
    error is **1.668618e-04 at p=0.37**. The docstring now states those figures;
    these tests keep them true, so an accuracy claim cannot drift away from the
    arithmetic again. A 4.5e-5 bound — the old claim — fails on every case below
    by an order of magnitude.
    """

    # (p, Φ⁻¹(p)) — reference quantiles evaluated at 50-digit precision
    # (mpmath, dps=50), rounded to double. Φ⁻¹(0.05) = -1.6448536269514722 and
    # Φ⁻¹(0.99) = 2.3263478740408408 are the textbook constants, which is an
    # independent check that the literals are the TRUE quantiles and not this
    # implementation's own output.
    CASES = [
        (1e-06, -4.753424308822899),
        (0.01, -2.326347874040841),
        (0.05, -1.6448536269514726),
        (0.25, -0.6744897501960817),
        (0.358, -0.36380985902969576),  # worst measured point
        (0.5, 0.0),
        (0.642, 0.36380985902969576),
        (0.75, 0.6744897501960817),
        (0.95, 1.6448536269514722),
        (0.99, 2.3263478740408408),
        (1.0 - 1e-06, 4.753424308817087),
    ]

    def test_norm_ppf_absolute_accuracy(self):
        """Max abs error over the pinned grid is 4.4429e-4 — bound 1e-3.

        The bound keeps ~2.2x headroom over the measured error (so it is not
        brittle to a platform's ``log``/``sqrt`` last-bit differences) while
        still failing on the documented-but-false 4.5e-5 claim and on any
        regression to a cruder approximation.
        """
        for p, expected in self.CASES:
            got = float(_norm_ppf(np.array([p]))[0])
            assert abs(got - expected) < 1e-3, (
                f"_norm_ppf({p!r}) = {got!r}, true quantile {expected!r}: "
                f"abs error {abs(got - expected):.3e} exceeds the measured "
                f"accuracy 4.4429e-4 (bound 1e-3)"
            )

    def test_norm_cdf_roundtrip_accuracy(self):
        """Φ(Φ⁻¹(p)) reproduces p to 1.6686e-4 — bound 1e-3.

        The round-trip is the property ``tk_calculation``/the F-02 threshold
        mapping rely on (``gtsim.py:421-422``): a quantile error that the
        inverse-CDF mapping does not cancel would move the data-space threshold.
        """
        ps = np.array([p for p, _ in self.CASES])
        roundtrip = np.asarray(_norm_cdf(_norm_ppf(ps)), dtype="float64")
        err = np.abs(roundtrip - ps)
        worst = int(np.argmax(err))
        assert err[worst] < 1e-3, (
            f"Φ(Φ⁻¹(p)) round-trip error {err[worst]:.3e} at p={ps[worst]!r} "
            f"exceeds the measured 1.6686e-4 (bound 1e-3)"
        )


@pytest.mark.hpgl
class TestPseudoGaussianTransform:
    """The transform draws from ITS OWN engine, never the process-global RNG.

    G-58: this class used to open five tests with ``np.random.seed(42)``. Those
    calls were inert — ``pseudo_gaussian_transform`` takes an optional ``rng``
    and, when none is passed, constructs ``np.random.RandomState(0)``
    (``gtsim.py:22-27``), so the global seed could not influence the result and
    the assertions could not distinguish the default engine from any other.
    The seeds are removed; ``test_default_engine_golden_output`` pins the
    default engine's actual values instead.
    """

    def test_binary_zeros_transformed(self):
        prop = _make_cont_prop(10, values=[0.0] * 10)
        pk_prop = _make_cont_prop(10, values=[0.5] * 10)
        result = pseudo_gaussian_transform(prop, pk_prop)
        assert result is prop
        for v in result.data.flat:
            assert 0.0 <= v < 0.5

    def test_binary_ones_transformed(self):
        prop = _make_cont_prop(10, values=[1.0] * 10)
        pk_prop = _make_cont_prop(10, values=[0.5] * 10)
        result = pseudo_gaussian_transform(prop, pk_prop)
        assert result is prop
        for v in result.data.flat:
            assert 0.5 <= v <= 1.0

    def test_mixed_binary(self):
        values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        prop = _make_cont_prop(8, values=values)
        pk_prop = _make_cont_prop(8, values=[0.5] * 8)
        result = pseudo_gaussian_transform(prop, pk_prop)
        zeros_mask = np.array(values) == 0.0
        ones_mask = np.array(values) == 1.0
        assert np.all(result.data.flat[zeros_mask] < 0.5)
        assert np.all(result.data.flat[ones_mask] >= 0.5)

    def test_non_binary_values_unchanged(self):
        values = [0.3, 0.7, 0.4, 0.6]
        prop = _make_cont_prop(4, values=values)
        pk_prop = _make_cont_prop(4, values=[0.5] * 4)
        original = prop.data.copy()
        result = pseudo_gaussian_transform(prop, pk_prop)
        np.testing.assert_array_equal(result.data, original)

    def test_different_pk_thresholds(self):
        prop = _make_cont_prop(10, values=[0.0] * 10)
        pk_low = _make_cont_prop(10, values=[0.1] * 10)
        result = pseudo_gaussian_transform(prop, pk_low)
        for v in result.data.flat:
            assert 0.0 <= v < 0.1

    def test_default_engine_golden_output(self):
        """G-58: the DEFAULT engine's exact values are pinned.

        ``pseudo_gaussian_transform(prop, pk)`` with no ``rng`` constructs
        ``np.random.RandomState(0)`` (``gtsim.py:22-27``), so its output is a
        fixed function of the fixture: ``uniform(0, pk)`` for every zero cell
        and then ``uniform(pk, 1)`` for every one cell, in flat index order,
        stored into ``prop.data`` (float32). Before this pin the class asserted
        only range bounds and one input echo, so the default engine could be
        re-seeded, switched to the process-global RNG, or bypassed entirely and
        every assertion still passed.

        The expected values are the actual float32 output of the default
        engine, measured by probe on this fixture. They are checked under two
        different process-global seeds (0 and 12345): the transform must not
        read the global RNG, and a global-RNG fallback would change the values.
        Mutation caught: any change to the default engine (seed removed →
        unseeded global draws; seed changed → different stream; the transform
        hard-coded to a constant) breaks the exact comparison.
        """
        values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        golden = np.array(
            [
                0.274406760931015,
                0.7118273973464966,
                0.35759466886520386,
                0.822947084903717,
                0.3013816773891449,
                0.7187936305999756,
                0.2724415957927704,
                0.945886492729187,
            ],
            dtype="float32",
        )

        saved_state = np.random.get_state()
        for global_seed in (0, 12345):
            # The global seed must be inert: the engine is private to the call.
            np.random.seed(global_seed)
            prop = _make_cont_prop(8, values=values)
            pk_prop = _make_cont_prop(8, values=[0.5] * 8)
            result = pseudo_gaussian_transform(prop, pk_prop)
            np.testing.assert_array_equal(
                result.data.ravel(order="K"),
                golden,
                err_msg=(
                    "G-58: pseudo_gaussian_transform's default engine "
                    f"(RandomState(0)) changed its output under global seed "
                    f"{global_seed}"
                ),
            )

        # Leave the process-global RNG as this test found it: the seeds above
        # are this test's input, not a state change for later tests.
        np.random.set_state(saved_state)


@pytest.mark.hpgl
class TestGtsimNoFileWrites:
    def test_no_debug_files_in_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        files_before = set(os.listdir(tmp_path))
        prop = _make_cont_prop(5, values=[0.5] * 5)
        tk_calculation(prop)
        files_after = set(os.listdir(tmp_path))
        new_files = files_after - files_before
        debug_files = [f for f in new_files if f.endswith((".txt", ".dat", ".csv", ".log"))]
        assert len(debug_files) == 0


# =============================================================================
# gtsim_2ind Tests (Q3 fix — previously zero coverage)
# =============================================================================

# gtsim_2ind depends on geo.py (through `from .geo import *`).
# The geo.py file has a pre-existing syntax error at line 403 (indentation).
# When geo.py is fixed, gtsim_2ind has an additional known bug (A5):
# sgs_simulation called without required cdf_data parameter.
# These tests are written to work once geo.py and gtsim.py are fixed.
try:
    from geo_bsd.geo import ContProperty, CovarianceModel, SugarboxGrid, covariance
    from geo_bsd.gtsim import gtsim_2ind

    _GTSIM_2IND_AVAILABLE = True
except (ImportError, SyntaxError, IndentationError, RuntimeError, OSError):
    _GTSIM_2IND_AVAILABLE = False


@pytest.mark.skipif(
    not _GTSIM_2IND_AVAILABLE, reason="gtsim_2ind not available (requires working geo.py)"
)
class TestGtsim2Ind:
    """Tests for the gtsim_2ind Gaussian Truncated Simulation workflow."""

    def _make_grid_prop(self, x=5, y=5, z=2):
        """Create a small grid and continuous property for testing."""
        np.random.seed(42)
        grid = SugarboxGrid(x=x, y=y, z=z)
        size = x * y * z
        # Binary 0/1 data with ~20% uninformed
        data = np.where(np.random.rand(size) < 0.6, 0.0, 1.0).astype("float32")
        mask = np.ones(size, dtype="uint8")
        prop = ContProperty(data, mask)
        return grid, prop

    def _make_sk_params(self):
        """Create simple kriging parameters."""
        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )
        return {
            "radiuses": (3, 3, 2),
            "max_neighbours": 8,
            "cov_model": cov_model,
        }

    def test_gtsim_2ind_basic_execution(self):
        """gtsim_2ind with default parameters."""
        grid, prop = self._make_grid_prop()
        sk_params = self._make_sk_params()

        result = gtsim_2ind(grid, prop, sk_params, do_sk=True, seed=42)
        assert isinstance(result, ContProperty)
        assert np.all(np.isfinite(result.data))

    def test_gtsim_2ind_reproducibility_same_seed(self):
        """gtsim_2ind with same seed produces identical output.

        N-04 docs correction: gtsim_2ind creates its own
        RandomState(seed) (gtsim.py:290), so it does NOT consume the global
        np.random — the old "same global random state" claim and the
        np.random.seed(42) resets below are harmless no-ops, kept for
        stability only.

        Partial masks (D-09/B-02): with fully-informed props the SGS step is a
        no-op and the comparison is trivially identical; uninformed cells make
        the reproducibility assertion meaningful.
        """
        grid, prop1 = self._make_grid_prop()
        _, prop2 = self._make_grid_prop()
        sk_params = self._make_sk_params()

        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop1.mask.size) < 0.7).astype("uint8")
        prop1.mask[:] = partial_mask
        prop2.mask[:] = partial_mask

        # Harmless no-op (gtsim_2ind uses its own RandomState(seed)); kept
        # for stability.
        np.random.seed(42)
        result1 = gtsim_2ind(grid, prop1, sk_params, do_sk=True, seed=42)
        np.random.seed(42)
        result2 = gtsim_2ind(grid, prop2, sk_params, do_sk=True, seed=42)
        np.testing.assert_array_equal(result1.data, result2.data)

    def test_gtsim_2ind_different_seeds_produce_different(self):
        """gtsim_2ind with different seeds produces different output.

        Uses a partially-informed property (mask with uninformed cells) so
        SGS actually simulates values. With a fully-informed property the SGS
        step is a no-op (nothing to simulate), the pseudo-Gaussian transform
        degenerates (kriging probabilities are exactly 0/1), and no randomness
        is consumed — different seeds then produce identical output. That is
        a fixture artifact, not a seed bug: with uninformed cells present,
        different seeds genuinely produce different simulated fields.
        """
        grid, prop1 = self._make_grid_prop()
        _, prop2 = self._make_grid_prop()
        sk_params = self._make_sk_params()

        # Leave ~30% of cells uninformed in both props so SGS actually
        # simulates them (seed-dependent). Identical masks for a fair
        # seed comparison. (2-M-12: gtsim_2ind no longer mutates the
        # caller's prop, but fresh props keep the comparison clean.)
        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop1.mask.size) < 0.7).astype("uint8")
        prop1.mask[:] = partial_mask
        prop2.mask[:] = partial_mask

        result1 = gtsim_2ind(grid, prop1, sk_params, do_sk=True, seed=42)
        result2 = gtsim_2ind(grid, prop2, sk_params, do_sk=True, seed=12345)
        assert not np.array_equal(result1.data, result2.data)

    def test_gtsim_2ind_golden_seeded_chain(self):
        """G-57: literal golden values for gtsim's seeded transform/threshold chain.

        Every determinism block in the suite compares two runs made with the
        SAME seed, so none of them can fail when the seed stops driving the
        stream at all — both runs then agree on the wrong field. This pin fixes
        literal values instead, for the stages ``gtsim_2ind`` owns in Python, in
        the order it executes them:

          1. seed -> engine: ``RandomState((seed ^ (seed >> 32)) & 0xFFFFFFFF)``
             (``gtsim.py:361``, the width-preserving 64-bit fold). This test
             performs that derivation itself, so what it pins is the transform's
             values FOR that engine — not the fold: seed 42 folds to 42, which
             is exactly what a 32-bit truncation yields too, so a changed fold
             leaves this pin green. The fold's width preservation is pinned by
             ``test_gtsim_2ind_seed_fold_is_width_preserving``, which observes
             the engine ``gtsim_2ind`` builds for ``2**32 + 42`` (fold -> 43,
             truncation -> 42),
          2. ``pseudo_gaussian_transform`` over the internal working copies
             (``gtsim.py:320-321``, ``:367-368``; the fixtures here are 1-D, so
             the ``order="F"`` copies cannot change the draw order),
          3. the threshold chain: ``tk_calculation`` (``gtsim.py:334``) then the
             F-02/II-39 mapping into data space
             ``cdf.inverse(Φ((tk - tk_mean)/tk_std_dev))`` (``gtsim.py:421-422``).

        The expected values are the current output for this fixture, measured by
        probe; a different seed changes all 50 transformed cells and the
        threshold, so the pin discriminates any stream change.

        Deliberately stops before the C++ SGS stage: F-388 (in flight in
        ``path_random_generator.cpp``) gives the path engine its own seed_seq,
        which re-permutes the node visiting order for a fixed seed. Any field
        value measured before that rebuild is stale, so a pinned field would be
        a permanently red oracle rather than a regression pin; everything above
        is numpy/CDF-only and therefore rebuild-independent. The simulated
        field stays covered by the differential and behavioural pins in this
        class.
        """
        from geo_bsd.cdf import calc_cdf

        _, prop = self._make_grid_prop()
        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop.mask.size) < 0.7).astype("uint8")
        prop.mask[:] = partial_mask

        pk_prop = ContProperty(
            np.full(prop.data.size, 0.5, dtype="float32"),
            np.ones(prop.data.size, dtype="uint8"),
        )

        # 1. The seed->engine derivation gtsim_2ind performs (gtsim.py:361).
        seed = 42
        engine = np.random.RandomState((int(seed) ^ (int(seed) >> 32)) & 0xFFFFFFFF)
        # 2. The internal working copies (gtsim.py:320-321, :367). tk_calculation
        #    overwrites the probability data in place, and gtsim restores it
        #    before the transform (gtsim.py:333-338).
        prop_work = ContProperty(prop.data.copy(order="F"), prop.mask.copy(order="F"))
        pk_work = ContProperty(pk_prop.data.copy(order="F"), pk_prop.mask.copy(order="F"))
        original_pk = pk_work.data.copy(order="F")
        threshold_normal = tk_calculation(pk_work, mean=0.0, std_dev=1.0).data.copy(
            order="F"
        )
        pk_work.data = original_pk
        transformed = pseudo_gaussian_transform(prop_work, pk_work, engine)

        np.testing.assert_array_equal(
            transformed.data.ravel(order="K")[:8],
            np.array(
                [
                    0.18727006018161774,
                    0.9828160405158997,
                    0.9041986465454102,
                    0.47535714507102966,
                    0.36599695682525635,
                    0.2993292510509491,
                    0.07800932228565216,
                    0.6523069143295288,
                ],
                dtype="float32",
            ),
            err_msg=(
                "G-57: gtsim's seeded pseudo-Gaussian transform no longer "
                "reproduces its golden values (seed 42) — the seed->engine "
                "derivation or the transform's stream changed"
            ),
        )

        # 3. F-02/II-39: the pk=0.5 normal-score threshold mapped into data space
        #    through the same empirical CDF the SGS back-transform uses.
        cdf_data = calc_cdf(transformed)
        tk_std = (threshold_normal.ravel(order="K") - 0.0) / 1.0
        tk_data = np.asarray(cdf_data.inverse(_norm_cdf(tk_std)), dtype="float64")
        assert abs(float(tk_data.ravel()[0]) - 0.3733120624821078) < 1e-7, (
            "G-57: the data-space threshold for pk=0.5 changed, got "
            f"{float(tk_data.ravel()[0])!r}, golden 0.3733120624821078"
        )

    def test_gtsim_2ind_seed_fold_is_width_preserving(self, monkeypatch):
        """G-27/G-57: the seed -> engine fold at ``gtsim.py:361`` keeps the
        full 64-bit seed width instead of truncating it to the low 32 bits.

        The golden-chain pin above derives ``RandomState(42)`` itself, and seed
        42 cannot discriminate the fold: every mapping that sends 42 to 42 —
        the fold, and equally the historical ``static_cast<uint32_t>``
        truncation that made seeds 5 and 2**32+5 replay one stream — yields the
        same engine. ``seed = 2**32 + 42`` does discriminate: the fold gives
        ``(2**32 + 42) ^ 1 == 43``, a truncation gives 42.

        The spy below observes the engine ``gtsim_2ind`` itself builds (the
        third argument of its ``pseudo_gaussian_transform`` call at
        ``gtsim.py:368``) instead of rebuilding it, so a changed fold — or a
        dropped ``rng``, which would fall back to the default engine — fails
        here. The ``folded != truncated`` guard keeps the pin from going
        vacuous if the seed choice stops discriminating the two engines.
        """
        import geo_bsd.gtsim as gtsim_mod

        seed = 2**32 + 42
        folded_seed = (seed ^ (seed >> 32)) & 0xFFFFFFFF

        grid, prop = self._make_grid_prop()
        sk_params = self._make_sk_params()

        captured = {}
        real_transform = gtsim_mod.pseudo_gaussian_transform

        def _spy(prop_arg, pk_arg, rng=None):
            if "draws" not in captured and rng is not None:
                captured["draws"] = rng.random_sample(2)
            return real_transform(prop_arg, pk_arg, rng)

        monkeypatch.setattr(gtsim_mod, "pseudo_gaussian_transform", _spy)
        gtsim_2ind(grid, prop, sk_params, do_sk=True, seed=seed)

        folded = np.random.RandomState(folded_seed).random_sample(2)
        truncated = np.random.RandomState(seed & 0xFFFFFFFF).random_sample(2)
        assert not np.array_equal(folded, truncated), (
            f"seed {seed} no longer discriminates the fold from a truncation"
        )
        assert captured, "gtsim_2ind never called pseudo_gaussian_transform with an rng"
        np.testing.assert_array_equal(
            captured["draws"],
            folded,
            err_msg=(
                "G-27: gtsim_2ind's seed -> engine fold changed — the engine it "
                f"built for seed {seed} is not RandomState({folded_seed})"
            ),
        )

    def test_gtsim_2ind_ignores_global_numpy_seed(self):
        """G-57/G-58: ``seed`` drives the run, the process-global RNG does not.

        The class used to bracket same-seed runs with ``np.random.seed(42)``,
        which is inert: ``gtsim_2ind`` builds its own engines. This pins the
        property behaviourally on the full pipeline — same ``seed``, two
        DIFFERENT process-global numpy seeds, bit-identical facies fields. A
        regression that reaches for ``np.random`` (or drops the seeded engine
        for an unseeded default) breaks it.
        """
        grid, prop1 = self._make_grid_prop()
        _, prop2 = self._make_grid_prop()
        sk_params = self._make_sk_params()

        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop1.mask.size) < 0.7).astype("uint8")
        prop1.mask[:] = partial_mask
        prop2.mask[:] = partial_mask

        saved_state = np.random.get_state()
        np.random.seed(1)
        result1 = gtsim_2ind(grid, prop1, sk_params, do_sk=True, seed=42)
        np.random.seed(987654321)
        result2 = gtsim_2ind(grid, prop2, sk_params, do_sk=True, seed=42)
        np.random.set_state(saved_state)
        np.testing.assert_array_equal(
            result1.data,
            result2.data,
            err_msg=(
                "G-57: gtsim_2ind output changed with the process-global numpy "
                "seed — the run must be driven by its seed argument only"
            ),
        )

    # =========================================================================
    # F-02 (HIGH): truncation must compare in the SAME space as the SGS output
    # =========================================================================

    def test_gtsim_2ind_pk_05_proportion_roughly_half(self):
        """F-02: with pk=0.5 everywhere, the facies-1 proportion must be ~0.5.

        The C++ SGS kernel back-transforms the simulated standard-normal
        field into DATA space through the in-scope empirical CDF
        (sequential_gaussian_simulation.cpp back_transform →
        transform_cdf_p(output, gaussian_cdf_t(), ncdf)), so prop1 is a
        data-space value. Pre-fix the truncation compared that data-space
        output against the NORMAL-SCORE threshold tk = -Φ⁻¹(0.5) = 0, so
        every non-negative data-space cell classified as facies 1 →
        proportion 1.0 (live repro). Post-fix the threshold is mapped to
        data space via the same CDF (tk_data = F⁻¹(Φ(tk))), restoring
        monotone equivalence with the Gaussian-space comparison → ~0.5.
        """
        grid, prop = self._make_grid_prop(x=10, y=10, z=5)
        sk_params = self._make_sk_params()

        # Partial masks (D-09/B-02): uninformed cells are simulated through
        # the pk threshold so the ~0.5 proportion is a real simulation
        # outcome, not a hard-data copy artifact.
        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop.mask.size) < 0.7).astype("uint8")
        prop.mask[:] = partial_mask

        pk_data = np.full(prop.data.size, 0.5, dtype="float32")
        pk_prop = ContProperty(pk_data, np.ones(prop.data.size, dtype="uint8"))

        result = gtsim_2ind(grid, prop, sk_params, do_sk=False, pk_prop=pk_prop, seed=42)
        # N2-08: compute frac1 over SIMULATED cells only (partial_mask==0).
        # The E2-32 hard-data restore re-copies original facies at informed
        # cells, so an all-cell frac1 has a hard-data floor of ~0.31 (156/500
        # hard facies-1 cells) and dilutes both F-02 and II-39 regressions
        # into the [0.2, 0.8] window. Simulated-only: correct=0.539,
        # F-02-regression=1.0, II-39-regression=0.0 (empirically verified).
        sim_mask_3d = (partial_mask == 0).reshape((grid.x, grid.y, grid.z), order="F")
        frac1 = float(np.mean(result.data[sim_mask_3d] == 1.0))
        assert 0.2 < frac1 < 0.8, (
            f"F-02: pk=0.5 must give ~0.5 facies-1 proportion over simulated "
            f"cells, got {frac1} "
            f"(pre-fix data-space vs normal-score comparison gave 1.0)"
        )

    def test_gtsim_2ind_non_default_tk_params_do_not_distort(self):
        """II-39: non-default tk_mean/tk_std_dev must not distort proportions.

        tk_calculation returns tk = tk_mean - tk_std_dev·Φ⁻¹(p); the engine
        simulates a STANDARD-normal field, so the effective threshold is
        (tk - tk_mean)/tk_std_dev = -Φ⁻¹(p) — the tk params must normalize
        away. Pre-fix, tk_mean=5.0/tk_std_dev=2.0 with pk=0.5 gave threshold
        tk=5.0 → every data-space cell < 5 → proportion 0.0. Post-fix the
        normalized threshold gives ~0.5, identical to the default params.
        """
        grid, prop = self._make_grid_prop(x=10, y=10, z=5)
        sk_params = self._make_sk_params()

        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop.mask.size) < 0.7).astype("uint8")
        prop.mask[:] = partial_mask

        pk_data = np.full(prop.data.size, 0.5, dtype="float32")
        pk_prop = ContProperty(pk_data, np.ones(prop.data.size, dtype="uint8"))

        result = gtsim_2ind(
            grid, prop, sk_params, do_sk=False, pk_prop=pk_prop,
            seed=42, tk_mean=5.0, tk_std_dev=2.0,
        )
        # N2-08: simulated-only frac1 (see G-34 rationale — E2-32 hard-data
        # restore dilutes all-cell proportions).
        sim_mask_3d = (partial_mask == 0).reshape((grid.x, grid.y, grid.z), order="F")
        frac1 = float(np.mean(result.data[sim_mask_3d] == 1.0))
        assert 0.2 < frac1 < 0.8, (
            f"II-39: non-default tk params must not distort the ~0.5 "
            f"proportion over simulated cells, got {frac1} "
            f"(pre-fix tk_mean=5 gave 0.0)"
        )

    def test_gtsim_2ind_default_and_non_default_tk_agree(self):
        """II-39 control: same seed, default vs non-default tk params must
        produce the same facies field (params normalized away)."""
        grid, prop1 = self._make_grid_prop(x=8, y=8, z=4)
        _, prop2 = self._make_grid_prop(x=8, y=8, z=4)
        sk_params = self._make_sk_params()

        # Partial masks (D-09/B-02): with fully-informed props the SGS step
        # is a no-op and the two runs trivially agree.
        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop1.mask.size) < 0.7).astype("uint8")
        prop1.mask[:] = partial_mask
        prop2.mask[:] = partial_mask

        pk_data = np.full(prop1.data.size, 0.5, dtype="float32")
        pk_prop1 = ContProperty(pk_data, np.ones(prop1.data.size, dtype="uint8"))
        pk_prop2 = ContProperty(pk_data.copy(), np.ones(prop2.data.size, dtype="uint8"))

        np.random.seed(42)
        r1 = gtsim_2ind(grid, prop1, sk_params, do_sk=False, pk_prop=pk_prop1, seed=42)
        np.random.seed(42)
        r2 = gtsim_2ind(
            grid, prop2, sk_params, do_sk=False, pk_prop=pk_prop2,
            seed=42, tk_mean=5.0, tk_std_dev=2.0,
        )
        np.testing.assert_array_equal(r1.data, r2.data)

    def test_gtsim_2ind_low_pk_gives_few_facies1(self):
        """F-02 monotonicity: pk=0.1 everywhere must give far fewer facies-1
        cells than pk=0.9 everywhere (the data-space threshold mapping is
        monotone in pk).

        E2-32: the fixtures use PARTIAL masks so SGS actually simulates the
        uninformed cells — a fully-informed property degenerates (every cell
        is hard data, the pseudo-Gaussian transform is deterministic, and pk
        no longer drives the classification: 0.40234375 == 0.40234375 above).
        Uninformed cells get simulated from the pk threshold, so the
        monotonicity is measurable again.
        """
        grid, prop1 = self._make_grid_prop(x=8, y=8, z=4)
        _, prop2 = self._make_grid_prop(x=8, y=8, z=4)
        sk_params = self._make_sk_params()

        # Leave ~70% of cells uninformed so SGS simulates them (identical
        # masks for a fair pk comparison — 2-M-12 pattern, :389-392).
        rng = np.random.RandomState(123)
        partial_mask = (rng.rand(prop1.mask.size) < 0.7).astype("uint8")
        prop1.mask[:] = partial_mask
        prop2.mask[:] = partial_mask

        pk_low = ContProperty(np.full(prop1.data.size, 0.1, dtype="float32"),
                              np.ones(prop1.data.size, dtype="uint8"))
        pk_high = ContProperty(np.full(prop2.data.size, 0.9, dtype="float32"),
                               np.ones(prop2.data.size, dtype="uint8"))

        r_low = gtsim_2ind(grid, prop1, sk_params, do_sk=False, pk_prop=pk_low, seed=42)
        r_high = gtsim_2ind(grid, prop2, sk_params, do_sk=False, pk_prop=pk_high, seed=42)
        frac_low = float(np.mean(r_low.data == 1.0))
        frac_high = float(np.mean(r_high.data == 1.0))
        # Measured gap with this partial-mask fixture is ~0.23; assert a
        # margin safely below it so the monotonicity claim is meaningful but
        # not fragile to seed/rounding shifts.
        assert frac_high > frac_low + 0.15, (
            f"F-02: pk=0.9 must give more facies-1 than pk=0.1, "
            f"got {frac_low} vs {frac_high}"
        )
