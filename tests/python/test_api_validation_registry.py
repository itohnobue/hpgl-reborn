"""C-API validation registry completeness gate (pat-20260802223236).

What this file enforces (and only this): every ``hpgl_*`` entry point declared
in api.h / exported by the library has exactly one registry row in api.cpp's
HPGL_VALIDATION_REGISTRY, and every row carries a summary that (a) is not
empty and (b) names at least one recognised validation facet.

What it does NOT enforce: whether a summary is *true*. The summary is free
text in api.cpp, so a row that claims a validation the entry point does not
perform cannot be detected from the text alone (F-444; two such rows existed
at api.cpp:433 and :440 — see the finding, owned by the C++ source, not by
this test). The registry therefore guarantees bookkeeping, not behavioural
mirror-validation; behaviour is covered by the per-entry tests.

These tests are mechanical — they do not judge validation quality. They
verify the pairing (exported symbol ↔ registry row) that a new entry point
must maintain.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.hpgl_wrap import _hpgl_so

    HPGL_AVAILABLE = True
except (ImportError, OSError):
    HPGL_AVAILABLE = False


_API_H_PATH = (
    Path(__file__).parent.parent.parent / "src" / "geo_bsd" / "hpgl" / "api.h"
)


# F-444: recognised validation facets. A registry summary is free text, so the
# gate cannot verify that a claim is true; it CAN refuse a row whose summary
# names no validation facet at all — the ``{"hpgl_new_entry", "x"}`` row that
# satisfied the previous non-empty-string check. Extend this map when a new
# entry point introduces a genuinely new facet.
_VALIDATION_FACETS = {
    "pointers": r"\bpointers?\b",
    "nullability": r"\bnon-null\b|\bnull\b",
    "size": r"\bsize\b",
    "shape": r"\bshape\b|\bdims\b",
    "volume": r"\bvolume\b",
    "count": r"\bcount\b",
    "finiteness": r"\bfinite\b",
    "radius": r"\bradius\b",
    "bounds": r"\bbounds?\b",
    "accessor": r"\baccessor\b",
    "params": r"\bparams\b",
    "mask": r"\bmask\b",
    "remap": r"\bremap\b",
    "values": r"\bundefined_value\b|\bvalues\b",
    "mean": r"\bmeans?\b",
    "probability": r"\bmarginal_probs\b|\bprobs\b",
    "neighbours": r"\bmax_neighbours\b|\bneighbours_count\b",
    "indicator": r"\bindicator_count\b",
    "kriging-kind": r"\bkriging_kind\b",
    "cdf": r"\bcdf\b",
    "equality": r"\bequal\b|\bequality\b|\bmatch\b",
    "thread": r"\bthread\b",
    "handler": r"\bhandler\b",
    "variance": r"\bvariance\b",
    "weights": r"\bweights2?\b",
    "coverage": r"\bcoverage\b",
}


def _named_facets(summary: str) -> list[str]:
    """Validation facets a registry summary names (F-444 gate unit)."""
    return [
        name
        for name, pattern in _VALIDATION_FACETS.items()
        if re.search(pattern, summary, re.IGNORECASE)
    ]


def _api_h_hpgl_symbols() -> set[str]:
    """Extract every declared ``hpgl_*`` entry-point name from api.h."""
    text = _API_H_PATH.read_text(encoding="utf-8")
    # HPGL_API ... hpgl_name(   — the C declarations. Registry accessors and
    # the C++ helpers (set/reset_kriging_stats) are inside the extern "C"
    # block; the C++ helpers are declared AFTER the #ifdef __cplusplus close
    # and are excluded by requiring `HPGL_API` or bare name + `(` that we
    # only collect from the extern "C" region.
    symbols = set()
    for m in re.finditer(r"\bhpgl_[a-zA-Z0-9_]+\s*\(", text):
        name = m.group(0).strip()[:-1].strip()
        if name.startswith("hpgl_") and name != "hpgl_set_kriging_stats" and name != "hpgl_reset_kriging_stats":
            symbols.add(name)
    return symbols


def _registry_symbols() -> dict[str, str]:
    """Read the C registry: {entry_point_name: validation_summary}."""
    count = _hpgl_so.hpgl_get_api_validation_registry_count()
    result = {}
    for i in range(count):
        name = _hpgl_so.hpgl_get_api_validation_registry_name(i)
        val = _hpgl_so.hpgl_get_api_validation_registry_validation(i)
        if name:
            result[name.decode("utf-8", errors="replace")] = (
                val.decode("utf-8", errors="replace") if val else ""
            )
    return result


def _exported_hpgl_symbols() -> set[str]:
    """Walk the loaded library's exported ``hpgl_*`` function symbols.

    Uses the platform symbol table (nm on macOS/Linux) so this is a true
    exported-symbol walk, not just ``dir()`` on the CDLL handle (which only
    lists symbols that have been referenced so far). Falls back to
    ``dir(_hpgl_so)`` when nm is unavailable (e.g. Windows CI without
    dumpbin), which still catches registry rows missing from the loaded
    handle.
    """
    lib_path = getattr(_hpgl_so, "_name", None)
    if lib_path is None:
        return set()
    symbols = set()
    if sys.platform.startswith("darwin"):
        import subprocess

        out = subprocess.run(
            ["nm", "-gU", lib_path], capture_output=True, text=True
        )
        for line in out.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] in ("T", "t"):
                name = parts[2]
                # macOS C symbols are prefixed with a leading underscore
                # (_hpgl_foo). Strip it before matching.
                if name.startswith("_hpgl_") or name.startswith("hpgl_"):
                    name = name.lstrip("_")
                if name.startswith("hpgl_"):
                    symbols.add(name)
    elif sys.platform.startswith("linux"):
        import subprocess

        out = subprocess.run(
            ["nm", "-D", "--defined-only", lib_path], capture_output=True, text=True
        )
        for line in out.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] in ("T", "t", "W"):
                name = parts[2]
                if name.startswith("hpgl_"):
                    symbols.add(name)
    else:
        symbols = {
            s
            for s in dir(_hpgl_so)
            if s.startswith("hpgl_")
            and not s.startswith("__")
        }
    return symbols


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL library not available")
class TestApiValidationRegistry:
    """Every exported hpgl_* entry point must have a registry row."""

    def test_every_declared_symbol_has_registry_row(self):
        """api.h declares N entry points; the registry must cover all of them.

        A new entry point added to api.h WITHOUT a registry row fails here —
        the class can no longer silently recur.
        """
        declared = _api_h_hpgl_symbols()
        assert declared, "api.h parse found no hpgl_* symbols (test broken)"
        registry = _registry_symbols()
        missing = sorted(declared - set(registry))
        assert not missing, (
            "Entry points declared in api.h but missing from the C-API "
            f"validation registry: {missing}. Add a row to "
            "HPGL_VALIDATION_REGISTRY in api.cpp with its validation summary."
        )

    def test_every_registry_row_maps_to_exported_symbol(self):
        """The registry must not contain rows for symbols that don't exist."""
        exported = _exported_hpgl_symbols()
        registry = _registry_symbols()
        if not exported:
            pytest.skip("could not enumerate exported symbols on this platform")
        stale = sorted(set(registry) - exported)
        assert not stale, (
            "Registry rows with no exported symbol (stale registry entries): "
            f"{stale}. Remove them from HPGL_VALIDATION_REGISTRY."
        )

    def test_every_exported_symbol_has_registry_row(self):
        """The nm walk of the actual library must match the registry.

        This is the true generated completeness check: whatever the library
        actually exports under hpgl_* must be in the registry, so a symbol
        added at build time (but not in api.h's grep-able text, e.g. via a
        macro or generator) is still caught.
        """
        exported = _exported_hpgl_symbols()
        if not exported:
            pytest.skip("could not enumerate exported symbols on this platform")
        registry = _registry_symbols()
        missing = sorted(exported - set(registry))
        assert not missing, (
            "Exported hpgl_* symbols missing from the C-API validation "
            f"registry: {missing}. Add each to HPGL_VALIDATION_REGISTRY in "
            "api.cpp."
        )

    def test_registry_summaries_are_nonempty(self):
        """Every row must carry a summary that names a validation facet.

        F-444: a non-empty-string check alone accepts any placeholder (the row
        ``{"hpgl_new_entry", "x"}`` passed all four tests), so a new entry
        point could claim validation it does not perform and nothing turned
        red. Each summary must now name at least one facet from
        ``_VALIDATION_FACETS`` — still bookkeeping, not a truth check (the
        text cannot be verified against the code mechanically), but a row with
        no validation content fails.
        """
        registry = _registry_symbols()
        empty = [name for name, summary in registry.items() if not summary.strip()]
        assert not empty, (
            f"Registry rows with empty validation summaries: {empty}. "
            "Every entry point must document what it validates."
        )
        faceless = sorted(
            name for name, summary in registry.items() if not _named_facets(summary)
        )
        assert not faceless, (
            f"Registry rows whose summary names no recognised validation "
            f"facet: {faceless}. Say what the entry point validates — one of "
            f"{sorted(_VALIDATION_FACETS)} — instead of a placeholder string."
        )
