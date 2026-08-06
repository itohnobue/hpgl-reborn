# Knowledge Base
Last updated: 2026-08-06T18:27:33.080683

## [got-20260616055758-f1d951]
Category: gotcha
Tags: hpgl-reborn, cvariogram, error-handling
Changed: 2026-06-16T05:57:58.641306

hpgl-reborn cvariogram module: no error retrieval API exists in _cvariogram (all functions return void). Python cvariogram.py has zero error detection at 4 call sites (calc_variograms, calc_variograms_from_point_set, cvar_stack_layers, fill_ellipsoid_directions). Fix requires C-level API addition. HIGH severity. (f0761e6)

## [con-20260616055802-b8b840]
Category: config
Tags: hpgl-reborn, build, cmake
Changed: 2026-06-16T05:58:02.886997

hpgl-reborn build/test commands: build via 'cmake --build build' (ninja), tests via 'uv run pytest tests/python/ -v' (620 pass, 2 skip OpenMP on macOS). CMake hardening flags were incomplete for hpgl_variogram target — fixed in f0761e6.

## [dis-20260616202352-0ad059]
Category: discovery
Tags: build, security, windows, msvc
Changed: 2026-06-16T20:23:52.732019

HPGL Reborn build audit: CMake security hardening is applied to hpgl_core and hpgl_variogram targets for non-Debug builds on Linux (fstack-protector, FORTIFY_SOURCE, relro, noexecstack, GLIBCXX_ASSERTIONS) and MSVC (/sdl, /guard:cf). However, the primary Windows build path (MSVC vcxproj via build.bat) has ZERO security hardening flags. Both vcxproj files also leak debug symbols (GenerateDebugInformation=true) in Release configurations.

## [dec-20260622193540-b18f4a]
Category: decision
Tags: hpgl-reborn, pylasdev-reborn, policy, ci-cd, forbidden, owner-decision
Changed: 2026-06-22T19:35:40.292103

hpgl-reborn and pylasdev-reborn are deliberately minimal projects. FORBIDDEN to add: CI/CD infrastructure of any kind (.github/, GitHub Actions workflows ci.yml/release.yml, Dependabot config, .pre-commit-config.yaml, .gitlab-ci.yml, .circleci/, .travis.yml — any CI provider), GitHub-specific metadata (FUNDING.yml, ISSUE_TEMPLATE/, PULL_REQUEST_TEMPLATE.md, CODEOWNERS, dependabot.yml), unnecessary community/governance docs (CODE_OF_CONDUCT.md, CONTRIBUTING.md, SECURITY.md), CI-status README badges ([![CI](...actions/workflows/...)]). KEEP intact: README, LICENSE, CHANGELOG, RELEASING.md (pylasdev only), pyproject.toml, source/test/build files. GitHub repository URLs in pyproject.toml and GitHub Issues/Discussions links in README are FINE (legitimate project references, not CI/CD). When any agent proposes adding these forbidden artifacts (often as part of 'production hardening', 'release readiness', or 'best practices' passes), refuse. They were added once before and owner had to manually remove them — do not repeat. Policy documented in AGENTS.md → 'Project-specific policies' section.

## [dec-20260622194447-206ed2]
Category: decision
Tags: hpgl-reborn, pylasdev-reborn, policy, documentation, forbidden, operator-gated, strict
Changed: 2026-06-22T19:44:47.268665

hpgl-reborn and pylasdev-reborn DOCUMENTATION POLICY (strict, operator-gated): NO new documentation may be added — not new files, not new sections inside existing files (README, CHANGELOG, etc.). Documentation work is restricted to FIXING ERRORS in existing documentation only (typos, broken links, factually incorrect statements, rendering/formatting bugs). Adding new documentation includes: creating new doc files (guides, tutorials, ADRs, architecture docs), adding new sections/subsections/appendices to README or any existing doc, 'filling documentation gaps', 'documenting an existing-but-undocumented feature', 'improving the README', 'best-practice docs', 'production readiness docs', 'release notes drafts', re-adding sections the operator previously removed (Changelog, Migration Guide, Validation Limits, Getting Help). THE ONLY EXCEPTION: explicit direct instruction from the operator (the human user, in the current session) to add specific documentation. A spawned agent or workflow proposing documentation additions — even as part of 'production hardening', 'release readiness', or 'best practices' pass — does NOT count as operator approval. When in doubt, ask the operator; do not add. Policy documented in AGENTS.md → 'Project-specific policies' → 'Documentation policy (strict — operator-gated)' subsection.

## [dis-20260627091641-b4b0b4]
Category: discovery
Tags: hpgl, dead-code, cpp
Changed: 2026-06-27T09:16:41.135566

HPGL variograms.cpp: Removed 4 dead functions (get_mask_flat_idx, get_mask_flat_idx_nocheck, is_informed, is_informed_nocheck) and 1 duplicate resize() in precalculated_covariance.h as part of I2F-039/040/041/026 fixes.

## [got-20260628021934-561bbd]
Category: gotcha
Tags: validation, dead-code, hpgl
Changed: 2026-06-28T02:19:34.808956

hpgl-reborn: 4 validation decorators in validation.py (validate_grid_params, validate_kriging_params, validate_simulation_params, validate_file_params) are defined and exported but ZERO usages in codebase. All validation happens via explicit Validator class method calls.

## [got-20260628021935-5b32fe]
Category: gotcha
Tags: cvariogram, crash, hpgl
Changed: 2026-06-28T02:19:35.651030

hpgl-reborn cvariogram.py line 14: module-level C library load crashes entire geo_bsd package import if native library missing. No graceful degradation or fallback to pure-Python variogram. __init__.py imports cvariogram unconditionally.

## [got-20260628021936-9f3b4f]
Category: gotcha
Tags: type-safety, ctypes, cvariogram, hpgl
Changed: 2026-06-28T02:19:36.804647

hpgl-reborn: CalcVariogramsFromPointSet (cvariogram.py:352-355) and CStackLayers/_create_float_data (cvariogram.py:383) take C pointers from numpy arrays without dtype validation. Contrast: CalcVariograms validates float32/uint8 explicitly. Passing wrong dtypes causes silent C-level memory corruption.

## [pat-20260628021940-3b034d]
Category: pattern
Tags: strides, dead-code, hpgl
Changed: 2026-06-28T02:19:40.404789

hpgl-reborn: both geo.py (__get_strides) and cvariogram.py (__strides) have 1D/2D stride branches that are unreachable in current code. Callers always pass 3D arrays. The 2D branch produces strides compatible with 2D arrays but _c_array(C.c_int, 3, array.shape) requires 3-element shape — would crash with cryptic error.

## [con-20260701025315-cd45d6]
Category: context
Tags: indicator-kriging, gslib, geostatistics, research, hpgl-reborn
Changed: 2026-07-01T02:53:15.320501

RESEARCH COMPLETE: s1-research-indicator report written (401 lines). Researched IK, order relations correction (GSLIB ORDREL.FOR verbatim), Median IK, SIS, Markov Model I/II for hpgl-reborn discovery agents. All 6 questions at CONFIRMED tier. Key finding: GSLIB uses two-pass monotonic envelope averaging for order relations, NOT iterative pairwise averaging as task description expected — flagged for discovery agents.

## [pat-20260701081644-bcddb1]
Category: pattern
Tags: hpgl-reborn, cpp, optimization
Changed: 2026-07-01T08:16:44.989674

HPGL Reborn uses workspace kriging (kriging_interpolation_ws) for hot loops. SGS/SIS switched from non-workspace variant to reduce heap allocations per simulated node. Pattern: declare kriging_ws_t before loop, pass to each call.

## [dec-20260701081645-dab5e5]
Category: decision
Tags: hpgl-reborn, build, blas
Changed: 2026-07-01T08:16:45.065189

HPGL Reborn ILP64 ABI risk documented rather than auto-detected. The lapack_compat.h has dead ILP64 code paths that never activate because build system never defines ILP64 macros. Added compile-time #pragma message on LP64 path to alert users.

## [got-20260701081645-24b413]
Category: gotcha
Tags: hpgl-reborn, tests, quality
Changed: 2026-07-01T08:16:45.140318

HPGL Reborn test suite had pervasive bare except Exception: pass anti-pattern that silently masks regressions. Fixed all instances to use specific exception types (CriticalValidationError, RuntimeError).

## [got-20260724074703-b42be8]
Category: gotcha
Tags: numerical, ieee754, gotcha, python, cpp, nan, comparison
Changed: 2026-07-24T07:47:03.983605

IEEE 754 NaN-ineffective clamping and comparison in numerical pipelines: NaN < X, NaN > X, NaN == X, and NaN != X all evaluate to False per IEEE 754. This means standard clamping guards (if x < 0: x = 0; if x > 1: x = 1) and range checks are NaN-ineffective — NaN passes through all comparison-based validation silently. This is distinct from 'NaN wasn'\''t validated' — the guards EXIST but are structurally incapable of detecting NaN because IEEE 754 comparison semantics defeat them. The same pattern applies to categorical dispatch chains where if x < threshold selects categories — NaN falls through ALL branches because every comparison is False, always returning the last/default category (systematic bias). In hpgl-reborn: 5 CONFIRMED MEDIUM findings across both discovery iterations — F-08 (no NaN check before dpotrf_), F-39 (Gaussian sample() NaN passes through 2-category clamp), F-51 (correllogram second_stage NaN-unaware + exact float equality), I2-F09 (categorical sample() NaN→last-category bias), I2-F15 (_norm_ppf NaN through np.clip). Fix: every comparison-based guard in a numerical pipeline needs explicit math.isnan()/std::isnan() checks BEFORE the comparison — clamping that checks x < 0 is NOT a NaN guard. Checklist: (a) for every comparison-based guard (if x < min, if x > max), verify NaN is explicitly handled before the comparison, (b) for every categorical dispatch chain using thresholds, verify NaN input produces explicit error, not default/last category, (c) np.clip is NaN-ineffective — test with np.nan input explicitly.

## [got-20260724074713-2117a3]
Category: gotcha
Tags: numerical, gotcha, cpp, threshold, unit-dependent
Changed: 2026-07-24T07:47:13.906555

Hardcoded unit-dependent numerical threshold creates blind zone: A hardcoded near-zero threshold (e.g., 0.0001) used to treat small covariance values as zero is unit-dependent — at micro-scales (millimeter, micron), legitimate values fall below the threshold and are incorrectly zeroed. At macro-scales, noise above the threshold escapes detection. The threshold is absolute but the data scale is relative to unit choice. In hpgl-reborn: F-46 (cov_model.h:170,183,195) — hardcoded 0.0001 nugget-blind zone confirmed MEDIUM. Fix: (a) parameterize the threshold — derive from data range (e.g., epsilon * max_abs_value) or make it configurable, (b) use relative comparison: abs(x) < epsilon * scale where scale is data-dependent, (c) document the threshold unit assumption explicitly if hardcoding is unavoidable. Checklist: for every hardcoded fabs(x) < CONSTANT comparison, verify the constant is appropriate for all supported units.

## [got-20260724074718-5ea5da]
Category: gotcha
Tags: validation, gotcha, python, ffi, type-conversion, numpy
Changed: 2026-07-24T07:47:18.896421

Type-conversion variable-reference mismatch in validation: When input data is converted to a new type (e.g., np.require(data, '\''uint8'\'') which wraps -1→255) and then validated, the validation code checks the ORIGINAL unconverted variable instead of the converted one. The type conversion silently transforms invalid values into apparently-valid ones, and the validation uses the pre-conversion value — producing a false pass for invalid data. In hpgl-reborn: I2-F12 (geo.py:186-208) — IndProperty used np.require(data, '\''uint8'\'', '\''F'\'') which wraps negative ints to large uint8 values, then the range check at line 204 used the ORIGINAL data array (not the converted self.data). PRIOR_FIX_ATTEMPT at 4 separate commits never detected the variable-reference mismatch. Fix: after any type conversion, update the variable reference passed to validation — or validate BEFORE conversion. Checklist: (a) for every type conversion (np.require, astype, ctypes cast), verify the POST-conversion value is validated, not the pre-conversion one, (b) the validation must run on the same memory the downstream code will read.

## [got-20260724074724-8a6528]
Category: gotcha
Tags: ffi, gotcha, python, assert, optimization, safety
Changed: 2026-07-24T07:47:24.267114

Bare assert at FFI or safety-critical boundary removed by python -O: Python'\''s assert statement is compiled away when running with -O (optimized) flag. At FFI boundaries, array stride validation, or any safety-critical code path, a bare assert that guards against memory corruption provides zero protection in optimized/production mode — the check is silently removed. The code appears guarded but has no guard at runtime. Use explicit if condition: raise ValueError instead of assert. In hpgl-reborn: I2-F35 (ffi_adapter.py:356) — bare assert stride check for 3D arrays is removed by python -O; the contiguous-array creation path also has ZERO stride check (asymmetry confirmed). BOTH-FOUND by two specialists. Fix: replace ALL bare asserts at module/FFI boundaries with explicit if/raise. Keep asserts only for internal invariants that are logically impossible to violate (and even then, document why -O removal is acceptable). Checklist: (a) grep for '\''assert '\'' in FFI adapter files, (b) every assert guarding input data (array shape, strides, dtype) must be if/raise, (c) every assert at module public API boundaries must be if/raise.

## [got-20260724074732-e63532]
Category: gotcha
Tags: ffi, gotcha, python, cpp, error-handling, boundary
Changed: 2026-07-24T07:47:32.514124

FFI error suppression via identical-message collision: When an error suppression protocol uses string equality on error messages to distinguish new errors from stale ones (current_error == saved_snapshot → suppress), two genuinely different calls producing the same error message are indistinguishable from a stale error. The suppression protocol incorrectly swallows the second call'\''s error. This is a contract-semantic gap: the protocol was designed to suppress re-reads of a sticky error, but string-equality collision makes it suppress distinct errors with coincidentally identical messages. In hpgl-reborn: I2-F32 (ffi_adapter.py:137-141) — snapshot protocol uses bytes equality on error messages; C++ never clears errors (overwrite-only); retry with same bad inputs → identical message → err == snapshot → suppression. Affects ALL 11 void-returning FFI wrappers. BOUNDARY-FOUND (intersection agent). Fix: use monotonic counters or call IDs instead of message-text equality for stale-error detection. Minimum: add a generation counter incremented on every FFI call; snapshot = (counter, message); only suppress when counter matches AND message matches. Checklist: for every error-suppression or dedup protocol using string/message equality, verify it cannot collide on distinct errors.

## [got-20260724074739-8b85d4]
Category: gotcha
Tags: ffi, gotcha, python, cpp, concurrency, ctypes, callback, race
Changed: 2026-07-24T07:47:39.061634

FFI callback trampoline use-after-free race (Python ctypes CFUNCTYPE + C++ function pointer): When Python passes a ctypes CFUNCTYPE callback to C++ (stored as a function pointer) and the Python-side CFUNCTYPE object is later freed (del old_h, reassignment, or GC), the C++ function pointer becomes dangling. CPython frees the ffi_closure/trampoline immediately on deallocation. If C++ calls the pointer after Python frees it → use-after-free / segfault. The race window is between C++ copying the handler pointer (protected by one mutex) and C++ invoking the handler (protected by a different mutex) — Python can free the CFUNCTYPE between these two operations. In hpgl-reborn: I2-F33 (geo.py:1692-1711, 1745-1764) — callback handler replacement can race with kriging function calls because kriging functions do NOT acquire _hpgl_call_lock. PRIOR_FIX_ATTEMPT at 6b2d2cd3 added lock for handler mutations but not for kriging serialization. BOUNDARY-FOUND. Fix: (a) hold the Python-side reference (keep a strong ref to the CFUNCTYPE object) for the entire lifetime C++ may call it, (b) serialize kriging calls with handler mutations under the same lock, (c) set C++ handler to nullptr BEFORE freeing the Python callback. Checklist: every ctypes callback passed to C++ must have documented lifetime guarantees — the Python reference must outlive ALL C++ call sites.

## [got-20260724074743-3331f1]
Category: gotcha
Tags: gotcha, python, dead-code, comments, monitoring
Changed: 2026-07-24T07:47:43.668987

Dead monitoring infrastructure with misleading comments: Infrastructure fields (counters, call IDs) that are written to but NEVER read create false confidence — maintainers reading the comments and field names assume the detection logic works, but the code only tracks state without acting on it. The tracking fields accumulate data that is never consumed by any decision logic, making the infrastructure dead code disguised as working instrumentation. In hpgl-reborn: F7-01 (ffi_adapter.py:106,122,151,158) — _hpgl_call_counter, _hpgl_call_id, _hpgl_suppressed_call_id written in every FFI call but read in ZERO decision paths. Added for I2-F32 cross-call error detection but detection logic was never implemented. BOTH-FOUND. Fix: either (a) remove the dead code and update comments to note the detection is unimplemented, or (b) implement the missing detection logic. Checklist: for every 'tracking'/'monitoring' field, grep for reads — if zero reads, it'\''s dead infrastructure. Comments that describe unimplemented behavior are worse than no comments.

## [got-20260724074748-03bde0]
Category: gotcha
Tags: testing, gotcha, numerical, cpp, python, regression
Changed: 2026-07-24T07:47:48.526945

Numerical library bug fixes without direct regression tests: When numerical bugs are fixed, relying solely on indirect integration tests for verification creates silent regression risk — the specific edge case (NaN detection, kriging failure tracking, correlation coefficient range validation) can silently regress because no test directly exercises the fix. Integration tests exercise the fix through multiple abstraction layers where the specific behavior can be masked by upstream guards or downstream fallbacks. In hpgl-reborn: F7-06 (post-fix review) — 14 C++ fixes across 8 source files had zero dedicated C++ unit tests. All verification was indirect via Python integration tests. Behaviors like categorical sample() NaN detection, correllogram NaN handling, and kriging failure stat tracking could silently regress. Fix: for every numerical bug fix, add at least one direct unit test that (a) exercises the exact failure input, (b) verifies the exact expected output/error, (c) runs at the same language level as the fix (C++ fix → C++ test). Checklist: after fixing a numerical bug, verify a test exists that would FAIL if the fix were reverted.

## [pat-20260724134312-6a29cc]
Category: pattern
Tags: python, numpy, dataclass, type-validation, consistency, isinstance
Changed: 2026-07-24T13:43:12.395034

Type gate consistency across related fields in dataclass validation: When one field in a dataclass __post_init__ accepts a broader isinstance type (e.g., seed accepts (int, numpy.integer)), ALL related integer/neighbour fields in the same dataclass and sibling dataclasses must accept the same type. An inconsistency where seed accepts numpy.integer but min_neighbours/max_neighbours rejects it creates a confusing API: np.int64(5) succeeds for seed but raises TypeError for neighbours. Checklist: (a) for every isinstance gate that was broadened to accept numpy types, grep for all sibling isinstance gates on related numeric fields in the same dataclass and its sibling classes — every one must have the same type tuple, (b) the bool exclusion guard (or isinstance(self.field, bool)) must also be present on every gate to prevent Python bool-as-int bypass, (c) after adding type flexibility, mechanically audit all dataclass __post_init__ methods for isinstance consistency. In hpgl-reborn: F-08 (MEDIUM, CONFIRMED, both-found) — SGSConfig.seed accepted (int, numpy.integer) but min_neighbours/max_neighbours only accepted int; same gap in SISConfig.max_neighbours. Fix: added numpy.integer to all 3 neighbour isinstance gates.

## [pat-20260724134318-f0500c]
Category: pattern
Tags: python, ffi, validation, numerical, nan, cpp
Changed: 2026-07-24T13:43:18.305420

Post-FFI output validation — always validate outputs from C/C++ calls before returning to callers: When Python wrapper functions call C/C++ through FFI (ctypes, cffi, numpy ctypes), the C++ backend can produce NaN/Inf through internal solver paths (ill-conditioned matrices, dpotrs_ on near-singular SPD systems, NaN weights from failed solves). Without post-call output validation, corrupted data silently propagates to downstream consumers. Every FFI wrapper function that receives a result array from C++ should validate isfinite() on the output before returning. Checklist: (a) for every Python wrapper function that calls C++ and receives a mutable output array, add numpy.isfinite() check on the output after the C++ call returns, (b) raise RuntimeError (not ValueError) for output corruption — this is a C++ computation failure, not a user input error, (c) use numpy.all(numpy.isfinite(out_array)) for float arrays; uint8/indicator arrays are naturally finite and can skip the check, (d) the check should run AFTER the C++ call but BEFORE the return statement — at the point where the output is fully populated. In hpgl-reborn: F-12 (MEDIUM, CONFIRMED, single-found) — all 7 kriging wrapper functions lacked post-C++ isfinite() screening on out_prop.data. Fix: added numpy.isfinite() checks with RuntimeError in ordinary_kriging, simple_kriging, lvm_kriging, median_ik, indicator_kriging, simple_cokriging_markI, simple_cokriging_markII.

## [got-20260724134324-46e739]
Category: gotcha
Tags: numerical, c++, validation, lapack, solver, nan
Changed: 2026-07-24T13:43:24.749980

Partial input validation in numerical solver functions — validate ALL inputs, not just the primary one: When a solver function validates one input (e.g., covariance matrix A for NaN/Inf) but not a secondary input (e.g., RHS vectors B, B0, B1), the partial validation creates a false sense of security — NaN in the unchecked input slips through to the solver core (dpotrf_, dpotrs_, gauss_solve), producing NaN solutions that propagate to kriging output. This is distinct from sibling-function asymmetry (pat-20260719191502-d9b8db) — here the asymmetry is WITHIN a single function between primary and secondary inputs. When the primary input gets an isfinite() scan because it was the one that historically caused bugs, mechanically verify that ALL other inputs to the same solver also receive the same scan. Checklist: (a) for every solver dispatch function, enumerate ALL vector/matrix inputs (primary matrix, RHS vectors, work arrays), (b) verify every input that can carry NaN/Inf has an isfinite() scan before solver dispatch — including secondary RHS vectors, (c) the error path for secondary-input failure should match the primary-input failure path (same error message format, same return value) so callers handle both paths identically, (d) performance cost is O(n) per vector — negligible compared to O(n³) solver time. In hpgl-reborn: F-24 (MEDIUM, CONFIRMED, both-found by 4 agents) — lapack_spd_solve_1rhs and lapack_spd_solve_2rhs pre-screened matrix A for NaN/Inf but not the RHS vectors B/B0/B1. Fix: added isfinite() scans for all RHS vectors following the exact pattern of the existing matrix A check.

## [got-20260724221100-1e1435]
Category: gotcha
Tags: python, ffi, numerical, validation, hpgl-reborn, cpp, solver
Changed: 2026-07-24T22:11:00.355910

FFI fallback finite values defeat post-call isfinite() validation — solver failure detection infrastructure must be wired: When a C++ numerical solver produces fallback finite values on failure (mean_on_failure fills cells with global mean, gaussian_cdf_t fills with [0,1], marginal_probs fills with valid probabilities), Python-side numpy.isfinite() post-validation CANNOT detect the failure — the fallback values are mathematically finite by design. This defeats the standard post-FFI validation pattern (pat-20260724134318-f0500c): isfinite() passes, no RuntimeError is raised, and corrupted/synthetic data silently propagates to downstream consumers. The three failure modes found in hpgl-reborn are: (1) mean_on_failure in SK/LVM kriging fills failed cells with finite global mean, (2) gaussian_cdf_t fallback in SGS simulation always produces finite [0,1] values, (3) marginal_probs fallback in SIS simulation always produces valid probabilities. In all three, isfinite() passed and the C++ error_guard protocol saw no exception (solver failure ≠ C++ exception). Fix: the actual detection mechanism must come from solver-level counters (kriging_stats_t) that count partial failures, not from output-array analysis. Wire get_kriging_stats() (or equivalent) into EVERY wrapper that calls a solver with fallback behavior — if the C++ side populates failure counters, the Python side must read them. Do NOT rely on output-array inspection alone when the solver has ANY fallback path that produces finite values. Checklist: (a) for every FFI wrapper, identify ALL solver failure fallback paths at the C++ level, (b) verify whether each fallback produces finite values that would pass isfinite(), (c) if yes, the isfinite() guard is structurally insufficient — add stats/counter-based failure detection, (d) NEVER assume a fallback value that 'looks wrong' will be detectable by range checks — mean_on_failure uses the GLOBAL mean which is a legitimate kriging output. In hpgl-reborn: 3 confirmed HIGH findings (H-1 SK/LVM, IT2-H-2 SGS, IT2-H-3 SIS) across 2 discovery iterations — 6+ agents confirmed the pattern across 3 different solver types. Fix: wired get_kriging_stats() via module-level _last_kriging_stats attribute in all affected wrappers.

## [got-20260724221110-b876cb]
Category: gotcha
Tags: python, validation, type-conversion, regression, hpgl-reborn, uint8
Changed: 2026-07-24T22:11:10.948368

Type-conversion validation fix regression — switching pre-conversion to post-conversion data reference silently reintroduces the bypass: When a production check fix addresses a type-conversion validation gap (got-20260724074718-5ea5da: validation must check POST-conversion values), a fix that mechanically replaces the pre-conversion variable reference (data parameter) with the post-conversion variable reference (self.data property) can silently REINTRODUCE the bypass if the property getter returns the already-converted/truncated value. In hpgl-reborn: commit e19aa84 attempted to fix 'IndProperty uint8 wrap' by changing the range check from 'data >= indicator_count' to 'self.data >= indicator_count'. But self.data is numpy.require(data, 'uint8', 'F') — already truncated: input 256.0 → uint8 0 → 0 >= 3 is False → passes silently. The ORIGINAL code (using the pre-conversion data parameter) correctly caught 256.0 >= 3 = True. The fix claimed to address something that was already working and broke it. Five subsequent production checks (8107ae2, dca7038, 9faa039, 9ab4dd0, 95ae241) did not catch the regression. Root cause: the fix author applied a valid general pattern (validate post-conversion values) to a case where the validation MUST happen pre-conversion to detect values that will be corrupted by conversion. Checklist: (a) when a production check fix changes which variable is referenced in a validation check, manually verify that the NEW variable reference holds the value BEFORE any lossy transformation, (b) for uint8/int conversion: the pre-conversion float/int array CAN detect overflow (>255 or <0); the post-conversion uint8 array CANNOT (wrapping is silent), (c) a fix that changes 'data' to 'self.data' without understanding why the original used the parameter is a regression risk — the original choice is rarely arbitrary, (d) after every fix near a type-conversion site, test with edge-case input that would wrap/overflow in conversion.

## [got-20260724221118-8aecf5]
Category: gotcha
Tags: python, warnings, exception, validation, hpgl-reborn, inheritance
Changed: 2026-07-24T22:11:18.940587

Python warnings.warn(category=X) requires X to be a Warning subclass — Exception subclasses can never serve as warning categories: Python's warnings.warn() function internally checks isinstance(category, Warning). If a validation library defines a custom warning class (e.g., ValidationWarning) that inherits from Exception instead of Warning, it can NEVER be used as a warning category — attempting warnings.warn('message', ValidationWarning) will raise TypeError: 'category must be a Warning subclass, not ValidationWarning'. The class is silently dead for its intended purpose. In hpgl-reborn: validation.py defines ValidationWarning(Exception) at line 95-99 but its sole documented use site (validate_max_neighbors at line 556-561) correctly uses the default UserWarning instead — the docstring falsely claims ValidationWarning would be raised. ValidationWarning has zero instantiations, zero raise sites, and zero warnings.warn() calls anywhere in the codebase. The inheritance chain makes it structurally incapable of ever being used as a warning category — it must be refactored to class ValidationWarning(Warning) before it can appear in any warnings.warn() call. Checklist: (a) for every custom warning/exception class in a library, verify its inheritance chain matches its intended use — Warning for warnings.warn(), Exception for raise, (b) a class that inherits Exception but is referenced in docstrings as 'raises X as a warning' is architecturally broken — either change the inheritance or change the docs, (c) grep for warnings.warn(category=YourClass) and verify isinstance(YourClass, Warning) before claiming it works.

## [got-20260724221129-c5992d]
Category: gotcha
Tags: python, cpp, ffi, dead-code, diagnostics, hpgl-reborn, integration
Changed: 2026-07-24T22:11:29.529628

Cross-language dead infrastructure — C++→C API→Python binding exists but zero callers at wrapper layer: When a diagnostic infrastructure (stats counters, failure tracking, runtime telemetry) exists at all three layers (C++ struct + setter, C API export, Python ctypes binding) but is never called by any production wrapper, the entire chain is dead code disguised as working instrumentation. Comments claiming 'Python wrapper reads stats' are provably false — grep for the function name in the wrapper layer returns zero matches. This creates a false confidence problem doubly worse than single-language dead code (got-20260724074743-3331f1): maintainers see the binding re-export in ffi_adapter.py and assume it's wired, and the C++ comment says it IS wired, but the call sites don't exist. The infrastructure was built but the integration was never completed. In hpgl-reborn: kriging_stats_t (kriging_stats.h), set_kriging_stats() calls in SK/LVM (simple_kriging.cpp:76, lvm_kriging.cpp:37), C API hpgl_get_kriging_stats() (api.h:192-193, api.cpp:159-166), Python binding hpgl_wrap.py:589-617, re-exported at ffi_adapter.py:50 — ALL EXIST. Zero callers in geo.py, sgs.py, sis.py. Comment at cont_kriging.h:231 'The Python wrapper reads stats for error propagation' is factually incorrect. This was the systemic root cause behind 3 CONFIRMED HIGH silent failure findings. Checklist: (a) for every C++ diagnostic struct with a C API export and Python binding, grep for the Python function name in ALL wrapper modules — not just the binding file, (b) zero matches = dead integration — the stats are being tracked but never consumed, (c) verify every comment claiming cross-language usage against grep output — comments lie, grep is truth. In hpgl-reborn: IT2-M-4 (CONFIRMED MEDIUM, both-found by 2 agents) identified get_kriging_stats() as the root systemic cause of all 3 HIGH findings.

## [pat-20260724231046-bbc95d]
Category: pattern
Tags: python, ffi, state-management, hpgl-reborn
Changed: 2026-07-24T23:10:46.765785

FFI module-level shared state — every wrapper MUST write or clear shared state variables: When multiple FFI wrapper functions in the same Python module share a module-level state variable (e.g., _last_kriging_stats), every wrapper that calls into C++ MUST either (a) populate the variable with fresh data from C++ counters/stats, or (b) explicitly reset it to a None sentinel. A wrapper that neither populates nor resets leaves stale data from the PREVIOUS call — possibly from a completely different algorithm — visible to callers who inspect the variable. The stale data is structurally indistinguishable from legitimate fresh data. This creates a false diagnostic signal: the caller sees plausible-looking stats that are actually from a different computation context. Fix: audit every FFI wrapper function for shared module-level state variables; every wrapper MUST write to the variable on every code path (populate OR reset); for wrappers where C++ produces no stats, set _state = None as an honest sentinel; for wrappers where C++ produces stats, call get_stats() and populate. In hpgl-reborn: median_ik, indicator_kriging, simple_cokriging_markI, and simple_cokriging_markII share geo._last_kriging_stats but NEVER touch it — stale OK/SK/LVM stats leak to callers after these functions. 4 CONFIRMED MEDIUM findings (UF-03, UF-04, UF-05, UF-07).

## [got-20260724231051-953a54]
Category: gotcha
Tags: python, ffi, exception-safety, hpgl-reborn
Changed: 2026-07-24T23:10:51.126096

Defensive module-level state reset before FFI call for exception safety: When an FFI wrapper populates a module-level state variable AFTER a C++ call (on the success path only), a C++ exception (hpgl_exception → Python RuntimeError) will skip the population line — the module-level variable retains stale data from the PREVIOUS call on the same thread. Callers who catch the exception and inspect the state variable see plausible-looking data from a completely different computation context, with no indication it's stale. Fix: set _last_state = None BEFORE the C++ call (defensive nullification), then populate with fresh data on the success path. The pre-call None ensures exception-exit paths produce None (honest 'no data') rather than stale data. Checklist: (a) for every FFI wrapper that populates module-level state AFTER a C++ call, add a _state = None BEFORE the call, (b) verify the None-set line precedes any C++ call that could raise, (c) after C++ returns successfully, populate with fresh data. In hpgl-reborn: SK/OK/LVM wrappers populate _last_kriging_stats AFTER C++ calls (geo.py:1152, 1256, 1376) — a C++ exception before these lines leaves stale stats from prior call. SGS/SIS demonstrate the correct defensive pattern: _last_kriging_stats = None before the C++ call (sgs.py:248, sis.py:277). CONFIRMED MEDIUM (UF-07).

## [got-20260724231056-65acd0]
Category: gotcha
Tags: testing, ffi, python, hpgl-reborn
Changed: 2026-07-24T23:10:56.575763

FFI integration test structurally bypasses wrapper layer — tests must call the public Python API not the raw C binding: When a test verifies FFI integration by calling the raw C API binding directly (e.g., ctypes wrapped function) instead of the Python public API wrapper function, the test structurally bypasses the layer of code it claims to verify. The wrapper's integration code — populating module-level state variables, applying validation, transforming data structures — is never exercised. If the wrapper's state wiring line were deleted, the test would still pass, producing false confidence that the integration is verified. Fix: (a) tests for FFI wrapper integration MUST call the public Python API function (e.g., ordinary_kriging(), not hpgl_get_kriging_stats() directly), (b) assert on the wrapper-managed state variable (e.g., geo._last_kriging_stats["points_calculated"] > 0) to verify the wrapper's state management code actually ran, (c) structurally verify the test would FAIL if the wrapper's state wiring line were deleted — delete the line temporarily and run the test to confirm. Checklist: for every integration test near an FFI boundary, grep for the test's function call — if it calls a ctypes binding function directly instead of the public Python API, the test is structurally incomplete. In hpgl-reborn: test_kriging_stats_reported_after_ordinary_kriging at test_cpp_fixes.py:516-533 calls get_kriging_stats() from hpgl_wrap directly (C API binding), bypassing geo._last_kriging_stats — would pass even if geo.py:1152 stats wiring was deleted. PRIOR_FIX_ATTEMPT at e19aa84 created this test. CONFIRMED MEDIUM (UF-06).

## [got-20260801220227-b91b90]
Category: gotcha
Tags: geo, security, f28
Changed: 2026-08-01T22:02:27.333290

geo.py F-28 basedir: DEFAULT_BASE_DIR=cwd at import; tmp_path callers MUST pass explicit basedir or CriticalValidationError fires

## [got-20260801220227-3d10e9]
Category: gotcha
Tags: geo, kriging, f33
Changed: 2026-08-01T22:02:27.403101

F-33 SK/LVM: mean_on_failure fills finite means passing isfinite; points_singularity>0 is the genuine failure signal (RuntimeError); no-neighbour mean-fill is documented contract (pure-nugget covariance, sparse data) -> warning only. All-masked degenerate input must warn not raise

## [got-20260801220227-ff0d76]
Category: gotcha
Tags: geo, f55, shape
Changed: 2026-08-01T22:02:27.472001

F-55 shape contract: load_cont_property/load_ind_property with 3-tuple size now return 3D Fortran on BOTH fast and fallback paths; read_inc_file_float/byte stay 1D (low-level contract). Downstream flat arithmetic must ravel.

## [got-20260802015631-1bc183]
Category: gotcha
Tags: ffi, ctypes, numerical, python
Changed: 2026-08-02T01:56:31.147576

ctypes small-int scalar types do NOT range-check: assigning an out-of-range Python value to c_ubyte/c_uint8 silently wraps mod 256 (300->44) with no error, unlike numpy 2.x which raises OverflowError for Python ints. At FFI boundaries, ctypes casts must be preceded by explicit Python-side range validation, because the C layer cannot detect the corruption. In hpgl-reborn: F-44 (CONFIRMED MEDIUM, boundary-found) - indicator_values passed to _c_array(c_ubyte,...) on the GSLIB byte write path silently wrapped 300->44 (silent data corruption in written files) while the INC path raised a confusing numpy OverflowError; fractional floats silently truncated (1.5->1); duplicate indicator_values not rejected on write path. Fix: [0,255] validation on the Python write path before ctypes conversion. Checklist: (a) grep _c_array/c_ubyte/c_int at FFI boundaries, (b) validate every value in Python before ctypes conversion - the conversion itself provides zero protection, (c) behavior differs by path: numpy 2.x raises on ints, ctypes always wraps.

## [got-20260802015636-a24aca]
Category: gotcha
Tags: ffi, build, macos, hpgl-reborn
Changed: 2026-08-02T01:56:36.327255

macOS dylib shadowing via lib{name} search-order preference: on macOS, ctypes/loader search order prefers lib{name}.dylib over {name}.dylib. A stale lib{name}.dylib left in the runtime directory silently shadows a freshly built {name}.dylib, so the test suite and production runs exercise a month-old binary. build.sh copies only hpgl.dylib and never removes the stale libhpgl.dylib; hash-guard verification with an empty expected-hash dict is a permanent no-op; smoke tests that only check 'library loaded' cannot catch API incompleteness. In hpgl-reborn: F-03 (CONFIRMED HIGH) - stale libhpgl.dylib (Jun 27, missing hpgl_get_kriging_stats) shadowed fresh hpgl.dylib (Jul 24), 4 test skips; same for lib_cvariogram.dylib (I2-43, I2-47); freshness checks passed silently because only 2 of the needed symbols were checked (PR-08). Fix: build.sh must delete stale lib{name}.dylib before copying; symbol-freshness check must assert the NEW API symbols exist (e.g. _EXPECTED_LIBRARY_SYMBOLS includes every new export); smoke test must assert API completeness, not just loadability. Checklist: (a) after any rebuild verify which file CDLL actually loads (lib._name), (b) grep the runtime dir for stale lib-prefixed dylibs and delete them, (c) extend symbol-freshness checks whenever a new C API function is added.

## [got-20260802015641-a15dae]
Category: gotcha
Tags: numerical, python, numpy, hpgl-reborn
Changed: 2026-08-02T01:56:41.416884

numpy int/uint8 arrays are NOT boolean masks: numpy treats an int/uint8 array passed to indexing as an integer (fancy) index array, not a boolean mask. A uint8 mask with values 0/1 used where a bool mask was intended causes integer fancy indexing - selecting rows instead of masking, inflating counts (5x pair-count inflation: uint8 5x5x5 gave [625,1940,2223] vs bool [125,420,507]) or crashing with IndexError when the index exceeds the dimension. In hpgl-reborn: F-01 (CONFIRMED HIGH, both-found) - variogram.py CubeScan passed non-boolean (int/uint8) masks to numpy indexing; codebase standard mask dtype is uint8 so the docstring's bool-or-int promise was broken; empirically verified. Fix: convert masks explicitly: arr.astype(bool) or arr > 0 before using as a mask; validate mask dtype is bool at API boundaries. Checklist: (a) grep for indexing like data[mask] where mask may be int/uint8, (b) verify mask.astype(bool) or comparison-to-zero before fancy indexing, (c) test masks with dtype uint8, not only bool - the corruption is silent in count-based results.

## [got-20260802015647-ee8c09]
Category: gotcha
Tags: numerical, cpp, python, ffi, hpgl-reborn
Changed: 2026-08-02T01:56:47.423769

float32 CDF last-value rounding to exactly 1.0f corrupts the max datum: when a CDF is stored in float32, the final plotting position can round UP to exactly 1.0f. If the inverse-CDF path treats p>=1.0 by returning the mean (or median), the MAXIMUM datum maps to normal score 0.0 (median) instead of the correct high tail score - the max datum silently collapses to the median on back-transform. GSLIB uses midpoint plotting positions to avoid this. In hpgl-reborn: F-04 (CONFIRMED HIGH, both-found) - gaussian_distribution.h:73-77 / non_parametric_cdf.h:245-259 / cdf.py:126-131 / sequential_gaussian_simulation.cpp:44 - last float32 CDF prob rounds to exactly 1.0f, inverse(p>=1.0) returns mean, max datum -> normal score 0.0, round-trip corrupts max datum to median; verified by exact float32 arithmetic; PFA b305411 introduced the guard (incomplete fix). I2-27 adds the low-side mirror: LVM means below data-min map to p=0.0 -> inverse(0.0) -> m_mean=0.0 -> local means collapse to median on back-transform. Fix: saturate tails (mirror non_parametric_cdf.h:271-291), not return mean for p>=1.0; handle both high-side p rounding to 1.0f and low-side p=0. Checklist: (a) for any float32 CDF, verify the max plotting position cannot round to exactly 1.0f, (b) test inverse(p=1.0f) explicitly with a float32-built CDF, (c) verify the extreme data values survive a round-trip through the transform.

## [got-20260802015652-a891f8]
Category: gotcha
Tags: numerical, cpp, ffi, hpgl-reborn
Changed: 2026-08-02T01:56:52.868521

FP-to-int cast is UB on NaN/huge values in C++: converting a NaN, Inf, or out-of-range float/double to int is undefined behavior in C++ (on x86-64 typically produces INT_MIN/garbage), and numpy has similar silent-wrap traps in ctypes paths. A NaN thickness/value cast to int can become INT_MIN, then be clamped to 0, then write blank_value to an entire column or poison cumulative state - whole-column silent corruption. In hpgl-reborn: I2-25 (CONFIRMED MEDIUM, both-found) - stack_layers.h:43,52,66 casts NaN/huge thickness (reachable via F-32: Python writers write NaN verbatim) to int: INT_MIN -> clamped 0 -> negative path writes blank_value to entire column [0,nz), cumulative_k becomes NaN poisoning subsequent layers; no NaN/range validation before the cast. Fix: guard every FP->int cast with an explicit std::isfinite + range check before converting, or reject non-finite input at the API boundary. Checklist: (a) grep for C-style or static_cast<int> of float/double expressions in C++, (b) verify isfinite check precedes every such cast, (c) trace whether NaN can reach the cast from any writer/loader path (Python writers must validate isfinite before writing).

## [got-20260802015657-f8eb69]
Category: gotcha
Tags: concurrency, cpp, build, hpgl-reborn
Changed: 2026-08-02T01:56:57.884636

OpenMP break inside worksharing loop is non-conforming: per OpenMP spec (2.11.2), break inside a worksharing loop (for/parallel for) has undefined behavior when OMP_CANCELLATION is not enabled (the default). The breaking thread's remaining iterations are dropped, leaving NaN cells and incomplete counters while other threads continue. Cancellation requires #pragma omp cancel for + OMP_CANCELLATION=true, or a flag-check restructure. In hpgl-reborn: F-23 (CONFIRMED MEDIUM, conditional on OpenMP builds) - cont_kriging.h:193-198 and median_ik.cpp:117-122 use #pragma omp cancel for + unconditional break; sibling indicator_kriging.h:216-228 is CORRECT (break inside #else non-OpenMP branch); shipped artifact had OpenMP disabled (build/CMakeCache NOTFOUND, 0 fopenmp flags) but OpenMP IS the documented default config - a legitimate production config where it fires. Fix: move break into the #else non-OpenMP branch, or use conforming cancellation flags. Checklist: (a) grep for 'break' inside omp for/parallel regions, (b) verify cancellation patterns conform to OpenMP spec, (c) test both OpenMP-enabled and serial builds - the bug only manifests with OpenMP on.

## [got-20260802015702-0830cc]
Category: gotcha
Tags: build, packaging, python, hpgl-reborn
Changed: 2026-08-02T01:57:02.885123

scikit-build-core sdist: include wins over exclude - exclude patterns in [tool.scikit-build] sdist config do NOT reliably drop files that a broader include pattern pulls in. Stale compiled binaries (.so/.dylib) and __pycache__ left in the source tree get shipped in the sdist despite exclude patterns; consumers building from the sdist get stale platform binaries that shadow the fresh build. In hpgl-reborn: I2-44 (CONFIRMED MEDIUM) - pyproject.toml sdist exclude patterns present, yet sdist shipped 17 stale artifacts (6 binaries: hpgl.so/dylib, libhpgl.dylib, _cvariogram.so/dylib, lib_cvariogram.dylib; 11 .pyc), reproduced FRESH from the extracted sdist's own pyproject. Fix: verify the actual sdist contents after build (extract and list binaries), remove stale binaries from the source tree, and test that building from the sdist reproduces the same package as building from the repo. Checklist: (a) after any sdist build, extract and grep for binaries/.pyc, (b) do not trust exclude patterns - verify the shipped artifact, (c) clean stale build artifacts before packaging.

## [got-20260802015708-3e64f9]
Category: gotcha
Tags: build, cmake, macos, hpgl-reborn
Changed: 2026-08-02T01:57:08.244590

find_package(OpenMP) does not search Homebrew prefix on macOS: CMake's FindOpenMP does not look in /opt/homebrew/opt/libomp (or /usr/local/opt/libomp) by default. When libomp is installed via Homebrew and OpenMP_ROOT is not set, find_package(OpenMP) silently returns NOTFOUND and HPGL-style builds compile WITHOUT OpenMP despite HPGL_USE_OPENMP=ON - the parallel algorithms are silently disabled and shipped artifacts contain zero fopenmp flags. In hpgl-reborn: confirmed (CONFIRMED, s3-adv-M2 evidence: build/CMakeCache OpenMP_CXX_FLAGS=NOTFOUND, 0 fopenmp in compile_commands; fixed in s7 by adding Homebrew libomp prefix detection setting OpenMP_ROOT=/opt/homebrew/opt/libomp before find_package(OpenMP)). Fix: detect Homebrew libomp prefix and set OpenMP_ROOT before find_package(OpenMP); consider failing loudly when OpenMP is requested but not found. Checklist: (a) on macOS with Homebrew, check whether OpenMP was silently NOT found (grep CMakeCache for NOTFOUND), (b) set OpenMP_ROOT to the libomp prefix, (c) verify compile_commands contains fopenmp when OpenMP is requested.

## [pat-20260802015713-aaa15f]
Category: pattern
Tags: build, cmake, hpgl-reborn
Changed: 2026-08-02T01:57:13.987588

CMake PARENT_SCOPE is dynamic-scoping, NOT static: set(VAR ... PARENT_SCOPE) inside a function modifies the CALLER's variable at the time the call executes. Two consecutive PARENT_SCOPE sets in the same function from different collection calls ACCUMULATE into the parent (each reads the parent's then-current value), they do NOT replace. A discovery misreading this as 'the second call overwrites the first, so BLAS dir is dropped' is a FALSE POSITIVE - empirically, the variables union (BLAS;/LAPACK both present). In hpgl-reborn: F-60 (REJECTED by adversarial, empirical CMake reproduction of the exact function: PARENT_SCOPE accumulates - BLAS;/LAPACK union; discovery misread CMake dynamic scoping; macOS uses Accelerate.framework anyway). Lesson: before filing a CMake variable-scoping finding, reproduce the exact function in a minimal CMake script - PARENT_SCOPE semantics are dynamic and accumulation is the norm. Checklist: (a) when a finding claims a CMake variable is 'dropped' or 'overwritten' by a PARENT_SCOPE set, reproduce the function in isolation first, (b) remember PARENT_SCOPE reads the parent's value at call time and accumulates across calls, (c) check the final parent value empirically before concluding data loss.

## [got-20260802015719-44782b]
Category: gotcha
Tags: numerical, cpp, geostatistics, hpgl-reborn
Changed: 2026-08-02T01:57:19.252421

box-limited covariance functor truncating data-to-data pairs to 0 while RHS exact creates an inconsistent kriging system: when the LHS (data-to-data) covariance functor truncates pairs farther than the search box to 0, but the RHS (data-to-unknown) uses the exact model, the kriging system becomes inconsistent - the covariance model says data-to-data covariance is non-zero at distances where the functor reports 0, silently biasing all production kriging paths (SK/OK/SGS/SIS/IK/LVM/median-IK) at ~0.3 sill error when range a > 2r. In hpgl-reborn: F-21 (CONFIRMED MEDIUM) - precalculated_covariances_t/covariance_field_t box-limited functor truncates out-of-box data-to-data pairs to 0 while RHS exact; F-21 sibling PR-06 persists on covariance_field_t::operator() (median_ik path, LHS spans 2xradius vs RHS in-box). Fix: data-to-data pairs beyond the box must use the exact covariance (or the truncation must apply symmetrically to LHS and RHS). Checklist: (a) for any box-limited covariance functor, verify LHS and RHS use the SAME truncation rule, (b) test kriging with range > box radius and compare against exact-model reference, (c) audit both precalculated_covariances_t and covariance_field_t paths.

## [got-20260802015741-bf72dc]
Category: gotcha
Tags: ffi, memory, python, cpp, hpgl-reborn
Changed: 2026-08-02T01:57:41.396416

numpy sliced/non-contiguous arrays passed to C++ flat-indexed loops cause heap OOB writes: C++ code that indexes a result array flat (cube_index = ix + nx*(iy + ny*iz)) assumes the Python wrapper validated shape AND passed C-contiguous arrays. Sliced numpy views (legitimate numpy slicing) have non-contiguous strides; C++ flat indexing then writes past the buffer or misaddresses data. Result-array shape must be validated against the write pattern (x/y dims, not just nz) before the FFI call, and arrays should be made contiguous (np.ascontiguousarray) before passing. In hpgl-reborn: F-06 (CONFIRMED HIGH) - CStackLayers heap OOB WRITE: result x/y-shape not validated vs layer shape, layer (5,5,1)+result (4,4,10) -> cube_index 209>160, empirical SIGSEGV; F-08 (CONFIRMED HIGH) - non-contiguous sliced layers -> OOB WRITE into cumulative_k + wrong data mapping, no contiguity check anywhere. Fix: validate every shape dimension (not just volume product) and enforce contiguity (np.ascontiguousarray or flags check) before FFI. Checklist: (a) for every C++ flat-indexed result write, verify the Python wrapper validates x/y/z dims individually, (b) verify slices/views are made contiguous before passing to C, (c) test with sliced/transposed inputs, not just fresh arrays.

## [got-20260802015746-f503a5]
Category: gotcha
Tags: numerical, cpp, memory, hpgl-reborn
Changed: 2026-08-02T01:57:46.675575

int overflow in matrix index arithmetic (i*matrix_size+j) causes heap OOB: computing A[i*matrix_size+j] in signed int overflows once matrix_size*matrix_size exceeds INT_MAX (~46340^2). Allocation may use a safe size_t path while indexing uses signed int - the mismatch (safe alloc + unsafe index) produces a heap OOB write for large neighbourhood counts. Same class: size*size pre-scan overflow lets NaN matrices reach the solver. In hpgl-reborn: I2-23 (CONFIRMED MEDIUM) - simple_cokriging_markI.cpp:55-69 A[i*matrix_size+j] computed in signed int -> heap OOB when neighbourhood count > 46340; solver NaN pre-scan size*size int overflow (solver_entry_point.h:111); Python validate_max_neighbors only warns >1000, C API has no upper bound. Fix: use size_t/checked arithmetic for index computation (port static_cast<size_t> from my_kriging_weights.h:167), pre-scan size*size in 64-bit before dispatch. Checklist: (a) grep for i*size+j index patterns computed in int, (b) verify allocation and indexing use the SAME integer width, (c) add an upper bound on matrix dimensions so overflow is unreachable.

## [got-20260802015752-d96c7e]
Category: gotcha
Tags: ffi, cpp, error-handling, hpgl-reborn
Changed: 2026-08-02T01:57:52.209522

abort()/SIGABRT in a C++ library is uncatchable by Python/ctypes: when a C++ library call aborts the process (abort(), std::terminate, assert), the Python caller gets a process death, not a catchable exception - try/except RuntimeError cannot intercept SIGABRT. Libraries called via FFI must throw catchable exceptions (hpgl_exception) instead of aborting, and validate inputs so the abort path is unreachable. In hpgl-reborn: I2-24 (CONFIRMED MEDIUM) - precalculated_covariance.h:71-75 abort() for radius >= 645 in SK/LVM paths ((2r+1)^3 > INT_MAX); Python MAX_RADIUS=1e6 allows r >= 645; contrast: clusterizer.cpp:106-108 throws catchable hpgl_exception for the analogous case. Fix: replace abort() with throw hpgl_exception (catchable -> Python RuntimeError). Checklist: (a) grep C++ for abort()/exit()/terminate in library paths reachable via FFI, (b) verify every defensive guard throws, not aborts, (c) test the boundary condition directly from Python - a SIGABRT is a crash, not an error.

## [got-20260802015758-6e1cf2]
Category: gotcha
Tags: concurrency, ffi, python, cpp, hpgl-reborn
Changed: 2026-08-02T01:57:58.280795

reentrant FFI handler deadlock: holding a non-recursive lock (threading.Lock / non-recursive std::mutex) across a C++ call that can invoke a Python callback which itself calls back into the library (set_output_handler / set_progress_handler / another kriging) causes self-deadlock on the same thread. The fix-stage regression in hpgl-reborn (PR-02 CONFIRMED MEDIUM, both-found+boundary) was adding _hpgl_call_lock wraps (8 new) around kriging FFI calls: a reentrant handler call from within a locked kriging call deadlocked (empirical 10s TIMEOUT-DEADLOCK, handler fired once). Fix: use threading.RLock (recursive) instead of Lock, or drop the lock before invoking callbacks; on the C++ side, do not hold the handler mutex while calling the handler (F-53). Checklist: (a) any lock acquired around an FFI call that can invoke a Python callback must be recursive (RLock), (b) if a C++ mutex guards handler invocation, check for reentrant callbacks back into the library, (c) after adding locking to FFI wrappers, test the reentrant path (handler that calls back).

## [got-20260802015803-470e78]
Category: gotcha
Tags: io, security, ffi, hpgl-reborn
Changed: 2026-08-02T01:58:03.494338

temp-file writers must use O_NOFOLLOW/O_EXCL + unique names + cleanup on failure: a fixed-name temp file (<target>.tmp) shared by concurrent writers corrupts (last rename wins, interleaved writes) and leaks on every failure path (write_value throw, fflush/rename failure) with zero cleanup. Worse, opening <target>.tmp with plain fopen('w+') follows symlinks - an attacker with write access to the target directory can plant a symlink at target.tmp and redirect the C++ write through it (TOCTOU between Python validation and C++ fopen). Python's safe_open_write uses O_NOFOLLOW but C++ re-opens the path itself. In hpgl-reborn: F-52 (CONFIRMED MEDIUM) - property_writer.cpp .tmp never removed on write failure, concurrent writers share one .tmp path; I2-58 (CONFIRMED MEDIUM, boundary-found) - C++ writers bypass O_NOFOLLOW, symlink-follow TOCTOU on <filename>.tmp (STRIDE Tampering). Fix: open temp files with O_NOFOLLOW|O_EXCL, use per-writer unique temp names (pid/thread), and remove the temp file on every failure path. Checklist: (a) grep fopen of derived temp paths - add O_NOFOLLOW/O_EXCL, (b) verify cleanup on ALL failure paths (throw, fflush, rename), (c) verify concurrent writers do not share one temp path.

## [got-20260802015809-929f31]
Category: gotcha
Tags: numerical, testing, ffi, hpgl-reborn
Changed: 2026-08-02T01:58:09.474395

LAPACK reads matrices COLUMN-major (Fortran order): test data written in row-major order and read by LAPACK with uplo='U' silently tests a DIFFERENT matrix than intended. A row-major matrix that should be non-SPD can appear SPD to dpotrf (info=0) because LAPACK reads the upper triangle of the transposed layout, and the fallback-path assertion then fails unexpectedly (or worse, passes on the wrong matrix). In hpgl-reborn: PR-04 (CONFIRMED MEDIUM, both-found) - test_solver_entry_point.cpp:146,172,299 defective test data: row-major matrices read by LAPACK column-major with uplo='U' -> dpotrf succeeded (SPD) on a different matrix -> fallback assertions permanently RED; fixed with genuinely non-SPD matrices [1,3,3,1]/[1,2,2,4] read column-major. Fix: when constructing LAPACK test matrices, either write them in column-major layout or verify the matrix is genuinely non-SPD under the storage order LAPACK will read. Checklist: (a) for any LAPACK test, compute the matrix LAPACK will actually see (column-major read), (b) verify non-SPD matrices are non-SPD in that layout (dpotrf info>0), (c) run the test and confirm the expected branch actually executes.

## [got-20260802015814-729334]
Category: gotcha
Tags: build, packaging, python, hpgl-reborn
Changed: 2026-08-02T01:58:14.559709

wheel install list must include every module imported unconditionally at import time: scikit-build-core/CMake packages install Python files via an explicit install(FILES) list; a module added to the source tree but NOT added to the install list produces a wheel that fails import (ModuleNotFoundError) for users - even when the source tree works. The failure is only detectable by installing the built wheel and importing, not by running tests in-tree. In hpgl-reborn: F-10 (CONFIRMED HIGH, PFA 24d834f repeat regression) - wheel had no importable top-level package and CMake install list omitted config.py + ffi_adapter.py (both imported unconditionally by __init__.py/sgs.py/sis.py/gtsim.py) -> import hpgl fails; I2-07 (CONFIRMED MEDIUM) - wheel exposes 'hpgl' but docs/tests use 'geo_bsd' (import-name divergence), cibuildwheel test imports a nonexistent name. Fix: keep the install list in sync with every unconditionally-imported module and add a wheel-install-import smoke test. Checklist: (a) when adding a Python module, grep the CMake install(FILES) list and add it, (b) verify wheel package layout matches the documented import name, (c) test: pip install the built wheel in a clean venv and import.

## [got-20260802015819-d37416]
Category: gotcha
Tags: io, security, ffi, hpgl-reborn
Changed: 2026-08-02T01:58:19.532194

self-referential basedir defeats path-containment validation: when a validator derives the 'base' directory from the filename's own directory (basedir = dirname(filename)), any file trivially passes relative_to containment - symlink containment is defeated because the base moves with the file. Symlink escapes (e.g. a symlinked temp dir pointing into another tree) become undetectable. In hpgl-reborn: F-28 (CONFIRMED MEDIUM, boundary-found) - self-referential basedir at ALL 10 production call sites (geo.py:446,510,672,714,819,846,877,952; routines.py:241,305,487) defeats PathValidator symlink containment; DEFAULT_BASE_DIR is dead code (0 uses); live symlink escape reproduced, control with a real base rejects. Fix: use a trusted independent base (DEFAULT_BASE_DIR or an explicitly passed root), never the file's own directory; canonicalize paths with realpath before containment checks. Checklist: (a) for any path-containment validator, verify basedir is NOT derived from the checked file's own directory, (b) verify realpath/canonicalization happens before relative_to checks (symlinks must be resolved first), (c) test with a symlinked directory to confirm rejection.

## [got-20260802015825-2228af]
Category: gotcha
Tags: memory, validation, build, hpgl-reborn
Changed: 2026-08-02T01:58:25.844107

unbounded user-controlled sizes/counts cause memory exhaustion and infinite hangs: parameters that size allocations or drive loop counts (max_neighbours, grid volume, radius, lag counts, file sizes) need magnitude caps at BOTH the Python boundary AND the C++ side. Python-side caps that only warn (validate_max_neighbors warns >1000 but continues) or are absent entirely let C++ allocate 16-80GB or run 7.7e11 iterations. C++ internal limits must match API-boundary caps - a mismatch (API allows 1e7, internal limit 1e5) silently routes to fallback (mean-fill) instead of erroring. In hpgl-reborn: F-25 (no hard upper bound on max_neighbours -> ~32GB reserve), F-49 (1e9 volume threshold -> 40-80GB), F-38 (lag_sep=1e6 x num_lags=10000 -> 7.7e11 iterations, effectively infinite hang), I2-04 (mgrid no volume cap -> 1152 TB), I2-05 (per-token loop no size cap), PR-07 (1e7 API bound vs 1e5 internal limit -> silent mean-fill) - all CONFIRMED MEDIUM. Fix: enforce hard caps in Python validation, re-enforce in C++, and align boundary caps with internal algorithm limits. Checklist: (a) every size/count parameter has a hard cap, (b) Python warning is not a cap - C++ must reject, (c) API-boundary caps match internal limits, (d) exceeding a cap raises, not mean-fills silently.

## [ref-20260802033523-af9675]
Category: reference
Tags: gslib, file-format, geostatistics, interop
Changed: 2026-08-02T03:35:23.239174

GSLIB file format (Deutsch & Journel): GEO-EAS family ASCII. Data files: title/nvar/nvar-names/rows. Grid property files: line1=title, line2=min max, then values one per line. Ordering: X fastest then Y then Z, loc=(iz-1)*nx*ny+(iy-1)*nx+ix (1-based). Missing = outside ±1.0e21 trimming window (no NaN/Inf text). No byte/float marker in file; categorical = integer codes, same 2-line header. Sources: gslib.com format.html (official), sgsim.fpp, PyGSLIB issue #24. Full report: tmp/s1-research-gslib-report.md

## [pat-20260802092553-868986]
Category: pattern
Tags: regression, fix, verification, stats-wiring, hpgl-reborn
Changed: 2026-08-02T09:25:53.396614

fix-introduced regression escapes on fix-modified lines: fix agents routinely ship new defects on the exact lines they ADD or MODIFY, and their own verification misses them because it checks the fix intent, not the new code. In hpgl-reborn v2.0.1: F1 (SIS expected-count formula added by F-M6-py produced spurious 'could not be kriged' warning on every successful 2-category run - empirically 65/128), F2 (geo.py doc block modified by F-N1 left a false median_ik/IK claim - F-N1 doc-error class recurring on the same lines), F8/F9/F10 (m_mean numerator/denominator scale mismatches in NEW F-M5/F-M6 stats code), F11 (sgs expected overcounts ndmin-skips - line added by F-M6-py), F12 (F-M16 rejection over-applied to ellipse-mask path). The F-33 stats-wiring class was repeatedly half-applied across wrappers (geo.py hotspot: 6 PRIOR_FIX_ATTEMPT findings; cokriging/OK/SGS/SIS each missed the same wiring SK/LVM got). Fix: post-fix review MUST treat fix-added lines with the same adversarial rigor as pre-existing code - diff-verify every added/modified line, test the exact branch the fix touches (2-category not just 3-category SIS), and when a wiring pattern is applied to one wrapper, mechanically audit ALL sibling wrappers for the same wiring. Checklist: (a) after any fix, re-audit git diff added lines as new defects, (b) test the fix's edge-case branch (not just the nominal path), (c) for cross-layer counters (Python expected formula vs C++ per-branch eval counts), verify the counting semantics match branch-by-branch.

## [pat-20260802092600-9a144b]
Category: pattern
Tags: caps, validation, radius, sibling, hpgl-reborn
Changed: 2026-08-02T09:26:00.050441

radius magnitude caps must be applied to ALL sibling paths at the SAME threshold: a guard added to ONE path (precalculated_covariances_t INT_MAX guard) is repeatedly missed on sibling paths that allocate the same (2r+1)^3 volume (covariance_field_t::init, calc_cov_field, OK-default, median_ik, cokriging markI/markII). In hpgl-reborn v2.0.1: F-M2/F-M3/F-N7 (covariance_field.h:84-99 and covariance_field.cpp:54-107 lack the INT_MAX/radius-magnitude guard that precalculated_covariances_t has - radius 1e6 -> (2r+1)^3 = 64 EB / tens of GB; 3ad77ee added the guard to only one of two sibling classes), F-N11 (clusterizer total_volume cap permits exactly 1e9 -> 8GB vector + 1e9 heap objects hang), F-M16 (mask allocated BEFORE volume cap fires). Fix: when adding a cap/guard, grep ALL sibling classes and entry points that construct the same quantity and apply the identical threshold; verify by enumerating every entry point (OK-default api.cpp:634, median_ik.cpp:43, cokriging m1/m2) not just the class where the bug was found. Checklist: (a) for any (2r+1)^n-style allocation, enumerate every constructor/call site, (b) same threshold at every site - a guard in one sibling is not a guard, (c) caps must fire BEFORE allocation, not after.

## [pat-20260802092606-5ce233]
Category: pattern
Tags: caps, performance, work-estimate, complexity, hpgl-reborn
Changed: 2026-08-02T09:26:06.445447

work-based caps vs count caps: a count cap (MAX_POINT_SET_SIZE, MAX_NEIGHBORS, grid cell count) does NOT bound the WORK a loop performs when the loop is O(n^2 x lag) or grid x window. Count caps bound allocations, not iteration complexity. In hpgl-reborn v2.0.1: F-H2 (calc_variograms_from_point_set has NO work cap - O(size^2 x lag_count); 1e6 points x 1e4 lags = 1e16 ops; empty tunnel still O(size^2)=1e12 iterations ~ 3+ hours; grid path got MAX_WINDOW_VOLUME but point-set path only count caps), F-M12 (MAX_WINDOW_VOLUME=1e8 caps window OFFSETS only; inner loop iterates ALL grid cells per offset - total = window x grid up to 1e17). Fix: for every nested loop, compute the effective product (outer x inner complexity) and cap THAT - a work-estimate cap (pairs x lags, window x grid) - not just each count individually. Checklist: (a) identify loops that are O(n^2)/O(n^3) or product-of-two-counts, (b) derive the worst-case total work as a single number, (c) cap the total work estimate, not the individual counts, (d) apply the same work cap to ALL entry points of the algorithm (grid AND point-set paths).

## [got-20260802092612-329455]
Category: gotcha
Tags: documentation, gotcha, docstring, cross-language, hpgl-reborn
Changed: 2026-08-02T09:26:12.403923

module docstrings that factually misdescribe C++ wiring are a repeat-regression source: a false reason in a docstring misled 3+ fix attempts in hpgl-reborn v2.0.1. F-N1/F2: geo.py:102-105 module sentinel doc falsely claimed cokriging/median_ik/IK 'do not call set_kriging_stats' when they DO (simple_cokriging_markI.cpp:439/:498, median_ik.cpp:201, indicator_kriging.h:323) - the doc-error class recurred on the same lines across fix attempts (F-N1 corrected the cokriging claim, F2 found the median_ik/IK claim still false). F5: a fix-added header comment (covariance_field.h:82-87) falsely claimed entry-point guards existed (grep: zero matches). F-M14: '1D or 3D' comment false for all three sibling functions. Fix: when a docstring states a fact about wiring ('X does/does not call Y'), verify against grep of the actual call sites BEFORE trusting it, and after ANY fix that changes wiring, re-check the docstring it references. Checklist: (a) docstrings describing cross-language wiring must be verified by grep, not trusted, (b) fix agents must re-read the docstring after changing the code it describes, (c) a false reason in a doc is as dangerous as a false assertion in code - it steers subsequent fix attempts to the wrong conclusion.

## [got-20260802092618-6b077c]
Category: gotcha
Tags: testing, regression-mask, test-contract, hpgl-reborn
Changed: 2026-08-02T09:26:18.600661

tests that assert the BROKEN behavior become regression masks: when a test asserts the current buggy behavior (or a documented limitation), the suite stays green while the defect ships - the test actively prevents the fix from landing and hides the regression. In hpgl-reborn v2.0.1: F-M6/F-N4 - test_geo_state_fixes.py:119/:133 assert 'geo._last_kriging_stats is None' after SUCCESSFUL sgs/sis, and test_cpp_fixes.py:555-590 asserts stats UNCHANGED after SGS - both codify the missing stats wiring as the expected contract; the suite was green while SGS/SIS failure detection was broken. F-M15: gtsim.py:216-223 3ad77ee commit message claims a regression test exists (grep: 0 tests exercise the clamp path) - a claimed-but-missing test. Fix: when fixing a behavior, find and flip every test asserting the old behavior IN THE SAME CHANGE; treat a commit message claiming a test as unverified until grep confirms it; add a test that would FAIL on the old behavior. Checklist: (a) grep for tests asserting None/stale/unchanged state that a fix should populate, (b) behavior-surface changes must update BOTH test contracts in the same change (F-M6 wiring flips test_cpp_fixes.py:555-590 AND test_geo_state_fixes.py:106-133), (c) a green suite is not evidence of correctness when tests codify the bug.

## [got-20260802092624-d7ef9a]
Category: gotcha
Tags: concurrency, openmp, deadlock, handler, hpgl-reborn
Changed: 2026-08-02T09:26:24.111358

RLock fixes SAME-thread re-entry only - cross-thread handler deadlock: a progress/output handler invoked from an OpenMP WORKER thread that re-enters a lock-guarded API deadlocks even with threading.RLock, because the worker thread does not hold the lock (the main thread does, and waits at the OpenMP barrier). In hpgl-reborn v2.0.1: F-M22 - geo.py:2168-2265 progress handler fired 7/11 times on OpenMP worker threads; worker calling any _hpgl_call_lock-guarded geo function (re-entering kriging/set_output_handler) blocks forever; main waits at barrier; empirically reproduced (acquire(blocking=False) returned False on workers); thread-local t_in_handler only guards same-thread recursion. This extends got-20260802015758-6e1cf2 (same-thread reentrancy): switching Lock->RLock fixes the same-thread case but NOT the cross-thread case. Fix: serialize handler invocation so callbacks never run on worker threads, or make the handler path lock-free (defer/queue the re-entrant call), or ensure the lock is never held across the C++ call that fires the callback. Checklist: (a) after RLock conversion, test the callback path under OpenMP (multi-threaded), not just same-thread, (b) verify which thread the C++ callback actually fires on (OpenMP workers are real threads), (c) a thread-local guard is same-thread-only - it cannot protect against worker-thread re-entry.

## [got-20260802092630-2eec8a]
Category: gotcha
Tags: gslib, interop, io, security, hpgl-reborn
Changed: 2026-08-02T09:26:30.576629

GSLIB interop: missing-value sentinels (+/-1.0e21) must be trimmed on EVERY read path and property names must be validated before header writes: third-party GSLIB files use the standard sentinel convention; loading sentinels as real data silently corrupts statistics, and writing untrusted property names into a 2-line header is header injection (leading whitespace, '--', newlines break the format / can inject rows). In hpgl-reborn v2.0.1: F-M18 (no +/-1.0e21 trimming on ANY read path - grep 1e21 -> 0 hits src+tests; LoadGslibFile keeps sentinels as data; get_gslib_property masks exact equality only), F3 (get_gslib_property NaN-undefined branch skips the +/-1.0e21 window while C++ read_inc_file.cpp:303 applies it unconditionally - empirically [1,1,1,1] vs [1,0,0,1]), F-N16/F-N17 (raw property-name/keys/Caption writes into GSLIB headers - no validate_property_name at any sink; 'A\nB' key and 'cap\ninjected\n3' caption execution-reproduced RuntimeError/ValueError). Fix: apply the sentinel window (isnan | < -1.0e21 | > 1.0e21) consistently across Python AND C++ read paths including the NaN-undefined branch; validate property names (reject \n, --, leading ws, empty) at EVERY write sink (Python wrappers AND C++ writers). Checklist: (a) grep 1.0e21/1e21 in read paths - sentinel trimming must be unconditional, not exact-equality, (b) every GSLIB header write must validate property names first, (c) Python and C++ read paths must apply the SAME sentinel window.

## [got-20260802092636-bfc185]
Category: gotcha
Tags: concurrency, data-race, ffi, state, hpgl-reborn
Changed: 2026-08-02T09:26:36.976729

module-global FFI state written OUTSIDE the serialization lock is a data race: when a Python wrapper family shares a module-global (_last_kriging_stats) and the design explicitly supports concurrent kriging via RLock, reset+populate of that global OUTSIDE the lock is a race - two threads kriging concurrently cross-contaminate stats, producing spurious RuntimeError or missed failure warnings. In hpgl-reborn v2.0.1: F-M23/F-N12 - all 12 write sites (geo.py:1364/1369,1471/1478,1600/1615,1727,1813,1913,2013; sis.py:133; sgs.py:155) reset AND populate OUTSIDE 'with _hpgl_call_lock' (grep 20 uses, none cover stats); reads via _check_kriging_failure_stats are lock-free; RLock design explicitly supports concurrent kriging (ffi_adapter.py:106). Fix: move shared-state reset+populate INSIDE the same lock that serializes the C++ calls; every write and read of the module-global must be under the lock. Checklist: (a) for every module-level shared variable written by FFI wrappers, verify reset AND populate are both inside the serialization lock, (b) concurrent-call design (RLock) makes unsynchronized shared state a real race even if stress tests don't trigger it, (c) reads of the shared state must also take the lock.

## [got-20260802092643-d2d982]
Category: gotcha
Tags: openmp, exception-safety, ffi, terminate, hpgl-reborn
Changed: 2026-08-02T09:26:43.882248

C++ exception escaping an OpenMP worksharing region is uncatchable (std::terminate/UB): a bad_alloc or any exception thrown inside an omp for/parallel-for region and NOT caught inside the region propagates out of the worksharing construct - the OpenMP runtime calls std::terminate (uncatchable by Python/ctypes) and the region is UB. Guards must either validate BEFORE the parallel region (catchable) or catch inside it. In hpgl-reborn v2.0.1: F-H1 (missing validate_max_neighbours_or_throw on hpgl_indicator_kriging -> 2e9 c_int -> reserve(2e9)x2 = 32GB/thread inside OpenMP -> bad_alloc escaping worksharing = std::terminate, 10/11 siblings DO guard), F-M8 (BLAS thread guard not RAII: bad_alloc from ws.A.resize(1e10) inside worksharing -> restore never runs -> BLAS pinned to 1 thread for process lifetime + exception escapes = UB, documented property_array.h:9-14), F-M9 (legal-cap 100000 -> 80GB matrix -> bad_alloc inside OpenMP -> std::terminate). Fix: validate/allocate on the caller thread BEFORE the parallel region (exceptions there are catchable -> clean -1/error return), or wrap the parallel region body in try/catch and record an exception_ptr to rethrow after the region; never let an exception cross the worksharing boundary. Checklist: (a) grep for allocations/throws inside omp for/parallel regions, (b) any allocation sized by user input must be hoisted out of the region or validated before entry, (c) C++-side guard exceptions must reach Python as catchable RuntimeError, not SIGABRT/terminate.

## [got-20260802092650-41b872]
Category: gotcha
Tags: regression, sibling, propagation, arithmetic, hpgl-reborn
Changed: 2026-08-02T09:26:50.038509

sibling fix not propagated - a fix applied to ONE fallback/sibling path silently leaves the same defect in parallel paths: when an arithmetic/guard fix lands in one location (e.g. solver_entry_point.h size_t), the identical pattern in sibling paths (gauss_solve fallback :158/:273/:276) is missed because the fixer only touched the discovered instance. In hpgl-reborn v2.0.1: F-M10 (I2-23 size_t fix went to solver_entry_point.h only, not propagated to gauss_solve fallback - signed-int size*size overflow for size>46340 -> heap OOB read UB; same-commit sibling miss), F-M3 (3ad77ee INT_MAX guard added to precalculated_covariances_t only, missed covariance_field_t), F-M14 (F-02 tuple fix applied to CalcCovarianceFunction/CalcIndCorrelationFunction but NOT CalcVariogramFunction), F-N15 (write-side O_NOFOLLOW hardened property_writer.cpp:74 but read-side read_inc_file.cpp:218/:274 still plain fopen). Fix: when fixing a class of bug (overflow, guard, hardening), grep the codebase for ALL instances of the same pattern (sibling functions, fallback paths, symmetric read/write sides) and fix every one in the same change. Checklist: (a) after any fix, grep for the same arithmetic/guard pattern across sibling and fallback code, (b) symmetric operations (read/write, entry/fallback, LHS/RHS) must both be hardened, (c) verify each call site reaches the fixed path, not an unfixed sibling.

## [pat-20260802223223-084a44]
Category: pattern
Tags: caps, performance, python, cpp, verification
Changed: 2026-08-02T22:32:23.500179

work caps copied across languages must be calibrated per-language throughput - parity of the constant is NOT parity of the guarantee: a C++ work cap (MAX_TOTAL_PAIR_LAG_WORK=1e12 in variograms.cpp:40) mirrored verbatim into pure Python (variogram.py:42) is still effectively-infinite because Python runs ~4-4.6e7 work-units/s vs C++ ~1e12/s (~4-5 orders slower): 1e12 units takes ~6h-108 days in Python vs ~17 min in C++. In hpgl-reborn v2.0.2: PY-F1 (CONFIRMED MEDIUM) - the round-1 fix copied the C++ constant into Python, but adversarial reproduction showed 1e6-pt x 1-lag = exactly 1e12 passes the cap and hangs (600s+ timeouts); the R-1 rationale (constant parity) was wrong, the corrected rationale is guarantee parity (bounded pure-Python runtime ~2.5s => Python cap 1e8). Fix: when mirroring a work cap across languages, derive the cap from the language throughput (target bounded worst-case runtime), document the deliberate divergence in a comment, and do NOT copy the constant blindly. Checklist: (a) any work cap copied C++->Python must be re-derived from Python throughput, (b) verify the worst-case pure-Python runtime is bounded (seconds), not just that a constant exists, (c) state the divergence rationale in the code comment.

## [pat-20260802223230-f97aae]
Category: pattern
Tags: build, packaging, verification, macos
Changed: 2026-08-02T22:32:30.467186

a verification gate only catches what it actually inspects - a gate that checks LC_LOAD_DYLIB deps (otool -L) but never LC_RPATH/LC_BUILD_VERSION (otool -l) cannot catch absolute-path leaks or min-OS mismatches; a gate that checks only symbol presence (smoke test checks _HAS_KRIGING_STATS, never loads _cvariogram) cannot catch the defect class it was built to prevent; a build gate that exits 0 on smoke failure is not a gate. In hpgl-reborn v2.0.2: R-23 (wheel tagged macosx_11_0 embeds libomp.a objects compiled with LC_BUILD_VERSION minos 15.0 - gate checks neither dylib min-OS nor archive min-OS), R-24 (gate runs otool -L only, never otool -l - LC_RPATH/LC_BUILD_VERSION unchecked, N4 fold-in regressible silently), R-26 (non-wheel smoke gate has no relocatability check - HPGL_STATIC_LIBOMP=OFF stale cache deploys with absolute libomp dep and the gate passes), 2-M-22 (build.sh implicit exit 0 / build.bat exit /b 0 on smoke failure), 2-M-23/R-22 (INSTALL_RPATH and exported hpglTargets.cmake bake absolute build-machine BLAS/include paths - gate does not scan .cmake files). Fix: the gate must assert the exact artifact property that defines the defect class (LC_RPATH, LC_BUILD_VERSION, absolute paths in dylibs AND exported .cmake files), and gate failures must be non-zero exits. Checklist: (a) list the artifact properties the defect class depends on and verify the gate inspects each one, (b) a smoke gate must load the actual module/symbol it claims to verify, (c) build gates must exit non-zero on any smoke failure, (d) scan exported .cmake files for absolute build-machine paths.

## [pat-20260802223236-e889f3]
Category: pattern
Tags: ffi, cpp, validation, boundary
Changed: 2026-08-02T22:32:36.646439

C API entry points must validate what their Python wrappers validate - direct C callers bypass all Python-side validation, so unvalidated enum values, zero/negative bounds, and out-of-range marginal probabilities silently produce wrong output (mean-fill, algorithm substitution, all-0/all-1) instead of errors. In hpgl-reborn v2.0.2: M-28 (m_min_neighbours unvalidated at C API - min>max => every node skipped => fully-unsimulated SGS output, no error), M-31 (max_neighbours=0 accepted by C API - empty neighbourhood => SK all-mean-fill / OK all-undefined), 2-M-4 (m_kriging_kind unvalidated - every value != KRIG_SIMPLE silently routes to ordinary kriging), 2-M-35 (hpgl_median_ik copies marginal_probs raw, no [0,1]/sum validation - marginal[1]=5.0 => silent all-1/all-0), 2-M-2 (hpgl_simple_kriging_weights unbounded neighbours_count - 100k pts = 160GB alloc, no validate_max_neighbours among 11 siblings). Fix: every C API entry point that has a Python wrapper must replicate the wrapper's validation gates (validate_max_neighbours_or_throw, value-range checks, enum validation) before acting; do not rely on Python-only validation. Checklist: (a) for each C API function, grep its Python wrapper for validation and mirror it at the C boundary, (b) test direct-C callers with invalid values, not just Python callers, (c) enum values, zero, negative, and out-of-range inputs must be rejected at the C boundary.

## [pat-20260802223242-44d3e0]
Category: pattern
Tags: testing, regression, verification, cpp
Changed: 2026-08-02T22:32:42.509108

a fix can be behaviorally inert - landed and reviewed yet byte-identical to pre-fix behavior - when the new guard is nested inside a condition that always holds, so the guard code never alters any branch decision; the paired regression test that passes pre- and post-fix is vacuous. In hpgl-reborn v2.0.2: R-3 (M-11 ndmin fix - gate 'if (min>0 && ws.indices.size() < min) { count originals; if (original_count < min) skip; }' is mathematically unreachable in intent because original_data_count <= ws.indices.size() always: whenever the outer condition holds the inner is always true; when previously-simulated nodes inflate the total the gate is never entered. The regression test test_sgs_ndmin_counts_original_data_only passes pre- and post-fix identically - vacuous, single datum, inflation path never exercised). Fix: prove the fix changes observable behavior (a test that FAILS on pre-fix code and PASSES on post-fix code), and verify the test exercises the exact branch the fix targets - a test passing both before and after the fix is not a regression test. Checklist: (a) for any fix, run the new test against pre-fix code - if it passes, the test is vacuous, (b) verify the guard algebra can actually alter the branch decision (no subset/superset implication), (c) regression tests must exercise the inflation/pathological path, not the nominal path.

## [pat-20260802223248-880835]
Category: pattern
Tags: coordination, ffi, python, cpp, verification
Changed: 2026-08-02T22:32:48.334757

parallel fix agents editing the same finding against different assumed states produce a cross-agent contract break: when a finding spans two layers (Python wrapper + C++ core), each agent fixes its layer assuming a different C++ state, and the resulting docstring documents the PRE-fix behavior while the code implements the POST-fix behavior - a silent contract contradiction. In hpgl-reborn v2.0.2: R-2 (2-M-1 fix - Python agent rewrote the sgs.py 'mean' docstring to say the user mean is NOT applied on the OK branch, but the parallel C++ agent's M-29 fix made the OK branch honor the mean (single_mean_t); empirically mean=100 vs mean=-100 produced output means 94.17 vs -92.98, contradicting the docstring). Fix: when a finding spans layers, fix documentation LAST from the landed code, or establish a shared post-fix contract snapshot before agents edit; docstrings must be verified against the actual landed C++ behavior, not assumed. Checklist: (a) any cross-layer finding fixed by multiple agents needs a shared contract statement of the POST-fix behavior, (b) after all agents land, verify docstrings against the actual landed code (grep + empirical output), (c) docstrings documenting 'user input ignored' must be checked against the current implementation, not historical behavior.

## [pat-20260802223254-2e7feb]
Category: pattern
Tags: verification, workflow, fix
Changed: 2026-08-02T22:32:54.282828

findings can be silently dropped during fix assignment: a finding present in the verified FIX LIST can be absent from every fix-agent task file, and the omission is undetected until post-fix review finds the code unchanged. In hpgl-reborn v2.0.2: R-1 (H-1 HIGH - present in s5-synth fix list line 168 but absent from the edit-density split AND all 5 s6 fix task files, grep=0; remained unfixed through a full fix cycle), R-11 (2-M-35 - edit-density assigned to C++ Agent A but the actual task file carried 10 findings not 11, dropping 2-M-35). Fix: mechanically verify that every finding in the fix list appears in at least one fix-agent task file before spawning (grep the finding IDs across task files); the edit-density split table is planning, the task files are execution - they can diverge. Checklist: (a) grep each fix-list finding ID across all fix task files, (b) count findings per task file against the edit-density plan, (c) post-fix, grep the source for the expected fix markers (new validation, new guards) to confirm the change actually landed.

## [pat-20260802223300-1fe0bc]
Category: pattern
Tags: gslib, geostatistics, verification, research
Changed: 2026-08-02T22:33:00.439142

GSLIB reference-scope verification: an algorithm change taken from a GSLIB reference program must be scoped to the exact routine that implements it, not applied at a shared code path used by other algorithms - applying a sgsim-scoped behavior at a shared OK calculator silently changes the public contract of unrelated entry points. In hpgl-reborn v2.0.2: M-3/R-4 (OK->SK downgrade when <4 data exists only in sgsim.for lktype=0; applied at the shared OK calculator my_kriging_weights.h:383-387 it changed the public hpgl_ordinary_kriging n<4 contract (SigmaLambda != 1) beyond the cited scope, broke the test pin test_sk_weights_sum_to_one_with_ok_equivalent (70.7329 vs pinned 75.0) and required a product decision to restrict to SGS), M-29 (SGS OK fallback drew N(0,1) instead of GSLIB N(gmean,1.0) - user mean ignored on fallback). Fix: before porting a reference behavior, verify which reference routine(s) implement it (grep the .for/.fpp source); implement it only on the code path serving that routine, or get an explicit product decision to change the shared contract. Checklist: (a) for any GSLIB-cited behavior, grep the reference program to identify the exact routine, (b) verify the shared code path is used ONLY by that routine before applying, (c) changing a public contract requires updating the test pin AND a product decision.

## [pat-20260802223306-eb50d0]
Category: pattern
Tags: python, numpy, ffi, mutation
Changed: 2026-08-02T22:33:06.748138

numpy ravel()/ravel('K') return a VIEW not a copy when the array is contiguous - in-place writes through the view mutate the CALLER's buffer even when the wrapper later restores its own attribute; restoring prop.data = copy hides the corruption from the wrapper but the caller's aliased array is already destroyed. Same class: rebinding prop.data = prop.data.copy() re-points the caller's attribute, and injecting keys into a caller-owned dict persists across reuse. In hpgl-reborn v2.0.2: M-24 (gtsim_2ind tk_calculation writes thresholds in-place through ravel('K') view - pk_prop.data buffer destroyed, :253 restore only hides from prop.data, external buffer corrupted), 2-M-12 (gtsim_2ind mutates caller prop via prop.data copy-rebind + pseudo_gaussian_transform in-place; comment claiming 'avoid mutating caller data (I2F-11)' false - partial repeat of 6b2d2cd fix), M-25 (sis_simulation injects radiuses/max_neighbours into caller data dicts in place - stale injected values silently used on dict reuse). Fix: copy input arrays before any in-place write when the contract promises non-mutation (use .copy() BEFORE the view, not after), and test with aliased caller buffers. Checklist: (a) any wrapper that writes into an input array must operate on a private copy, not a view of caller memory, (b) restoring an attribute after corrupting a shared view is not a fix - the caller's buffer is already damaged, (c) dict inputs must be copied before injection, (d) a 'never mutated' comment must be verified with an aliasing test.

## [pat-20260802223312-d44c2d]
Category: pattern
Tags: numerical, reproducibility, cpp, verification
Changed: 2026-08-02T22:33:12.335815

every randomized path must be seed-controlled for reproducibility: a thread_local RNG seeded from random_device()^time(nullptr) makes identical inputs produce different outputs on every run - published experiments non-reproducible; a seed_rand_once() that is a no-op silently defeats the intent. In hpgl-reborn v2.0.2: 2-M-3 (grid-path variogram percent sampling variograms.cpp:642 - thread_local mt19937 seeded random_device()^time, seed_rand_once() is a NO-OP at :210-214, no seed parameter anywhere in C API or Python; identical inputs => different variograms; every OTHER randomized path in the codebase is seed-controlled - this one path was missed). Fix: every randomized path must accept an explicit seed and be deterministic for a given seed; entropy seeding belongs only at public entry points that document non-reproducibility. Checklist: (a) grep for thread_local RNG + random_device/time seeding, (b) verify every randomized path exposes a seed parameter and is reproducible for fixed seed, (c) a seed_rand_once()-style initialization function must actually set the seed (no no-op stubs), (d) tests for randomized paths must use a fixed seed and assert determinism.

## [pat-20260802223318-fbd59f]
Category: pattern
Tags: ffi, validation, python, cpp, memory
Changed: 2026-08-02T22:33:18.143534

invariants enforced only at construction are broken by public property setters: when a class validates data/mask shape invariants in validate() at construction only, but public setters (prop.mask = ..., prop.data = ...) reassign via numpy.require+contiguity checks with NO shape-match validation, and the FFI layer checks data.size but NEVER mask.size, C++ code that indexes mask by data m_size performs a heap OOB write (0x01 up to 1e9 cells) + OOB read. In hpgl-reborn v2.0.2: 2-M-15 (geo.py setters break data/mask shape invariant; ffi_adapter checks data.size only; property_array.h set_at/is_informed bound mask by data m_size -> heap OOB; reachable via public setter with mask-replace/reshape usage). Fix: property setters must re-validate cross-array invariants (shape match between data and mask), and FFI constructors must size-check EVERY buffer passed to C++ (mask.size as well as data.size). Checklist: (a) for any paired-buffer API (data+mask), verify every setter preserves the shape invariant, (b) FFI must validate the size of every array C++ will index, not just the primary one, (c) test setter paths (mask reassignment) not just construction paths, (d) C++ bounds must be defensive even when Python is expected to validate.

## [pat-20260802223421-6e2873]
Category: pattern
Tags: numerical, cpp, python, verification
Changed: 2026-08-02T22:34:21.063029

parallel language implementations of the same algorithm must agree on the underlying METRIC/semantics, not just the API signature - when a Python fallback reimplements a C++ kernel, the binning metric (Euclidean distance vs projection onto anisotropy axis), self-pair handling, and search semantics must match the C++ reference; API parity without metric parity silently produces different results for the same inputs. In hpgl-reborn v2.0.2: 2-M-13 (Python pure-Python variogram scans bin by raw Euclidean distance while BOTH C++ kernels bin by projection onto principal anisotropy axis - same template+data => different variogram curves, empirically lag-3 4.5000 vs 3.6937; tunnel filters identical, divergence isolated to lag-binning metric; product decision: align Python to C++ projection metric), M-21 (GridStyle didn't skip self-pairs while ContStyle did - the two Python scans disagreed on same data; product decision needed on reference kernel). Fix: when a Python path mirrors a C++ kernel, document and match the metric/semantic contract explicitly (projection vs Euclidean, self-pair policy), and add a cross-language equivalence test asserting the same output for the same input. Checklist: (a) identify the metric each kernel uses (grep the binning code), (b) Python fallback must match the C++ reference metric, (c) sibling paths within the same language must agree (self-pair skip, etc.), (d) cross-language equivalence tests guard the parity.

## [got-20260803001248-1d7a62]
Category: gotcha
Tags: gslib, sgsim, geostatistics
Changed: 2026-08-03T00:12:48.411382

GSLIB sgsim reference: ndmin counts ONLY original hard data (nclose from srchsupr); previously-simulated nodes (ncnode) never count. nclose<ndmin => node SKIPPED (written as UNEST=-99.0), NOT N(mean,1) fallback.

## [got-20260803001248-8cfb28]
Category: gotcha
Tags: gslib, sgsim, geostatistics
Changed: 2026-08-03T00:12:48.481536

GSLIB sgsim reference: OK->SK downgrade at (nclose+ncnode).lt.4 (fewer than 4 total data). N(mean,1.0) fallback (cmean=gmean,cstdev=1.0) only when zero data or singular kriging matrix.

## [got-20260803001248-7468e0]
Category: gotcha
Tags: gslib, file-format, geostatistics
Changed: 2026-08-03T00:12:48.553434

GSLIB format: GEO-EAS style ASCII, space-delimited, Fortran list-directed read (NaN/Inf text = read error). Missing = trimming window +/-1.0e21 (user-facing); internal UNEST version-dependent (-99.0 in GSLIB 2.907, 1.0e21 classic). Grid output header: title / 'nvar nx ny nz' / names; x fastest, then y, then z.

## [pat-20260803001248-62e195]
Category: pattern
Tags: gslib, sisim, ordrel
Changed: 2026-08-03T00:12:48.619961

ORDREL order-relation corrections (GSLIB ordrel.for): clamp indicator probs to [0,1]; continuous: upward correction then downward correction then average ccdfo(i)=0.5*(ccdf1(i)+ccdf2(i)) => monotonic non-decreasing; categorical: renormalize to sum 1.

## [ref-20260803001248-026bbf]
Category: reference
Tags: gslib, reference
Changed: 2026-08-03T00:12:48.689707

GSLIB 2.0/2.907 original Fortran sources mirrored at github.com/exepulveda/gslib2.0 (src/original/{sgsim,sisim,gtsim,gslib}/*.for) — authoritative reference for sgsim/sisim/gtsim semantics.

## [got-20260803180147-394516]
Category: gotcha
Tags: ffi, memory, python, cpp, validation
Changed: 2026-08-03T18:01:47.795269

FFI output-buffer contract: contiguity, size, AND writeability must all be enforced before a C call writes into a caller buffer. Three independent facets fail separately in hpgl-reborn: (a) CONTIGUITY - numpy sliced/non-contiguous views passed to C++ flat-indexed loops cause heap OOB writes (F-28 cvariogram.py:618-634, live negative-stride probe); (b) SIZE - caller-provided output buffers are not validated against the C++ write volume, so a too-small buffer silently truncates results (II-40 num_lags=4/buffer 2 -> mismatched tuple; F-24/F-25 data.size==mask.size missing -> heap OOB read), and an FFI constructor can stamp grid dims into a mask buffer defeating downstream per-dim guards (III-14 create_ubyte_array); (c) WRITEABILITY - a writeable=False buffer is silently mutated and a true read-only mmap SIGBUSes uncatchably on CalcVariogramsFromPointSet and CStackLayers (III-39). C++ must also fully initialize result buffers before return (III-40 CStackLayers top-tail cells unwritten -> stale-cell corruption on buffer reuse). Fix: validate contiguity (np.ascontiguousarray), size (every dim, not just volume), and writeability (flags WRITEABLE) before EVERY FFI call, and have C++ initialize outputs. Checklist: (a) for every FFI call receiving an output buffer, verify contiguity, exact size, and writeability, (b) verify C++ writes every element or explicitly initializes, (c) test with sliced views, undersized buffers, and read-only buffers.

## [pat-20260803180149-88787b]
Category: pattern
Tags: caps, performance, verification, complexity
Changed: 2026-08-03T18:01:49.666814

a work cap's CONSTANT must be validated against the real worst-case cost of the loop it bounds - an existing cap can under-estimate the true work by orders of magnitude and is then not a cap at all. In hpgl-reborn v2.0.3: F-03 (Python CubeScan had NO total-work cap - 2.6e12-8.4e13 measured ops, 24GB mgrid at 1000^3), F-27 (GridStyle cap under-estimates n^2 x NumLags cost by factor 5.0e5-7.2e5 - the cap value was derived from the wrong complexity), F-30 (MovingAverage cap bounds MEMORY not WORK - N x V product loops still multi-hour). Extends pat-20260802092606-5ce233 (count caps do not bound work) and pat-20260802223223-084a44 (per-language calibration): even a correctly-shaped work cap fails when its VALUE is wrong. Fix: after adding a work cap, measure or derive the actual worst-case operation count for the exact loop structure and set the cap to bound the target worst-case runtime; verify with a pathological input that the cap fires. Checklist: (a) compute the real loop complexity (product terms), (b) sanity-check the cap constant against a measured worst-case runtime, (c) a cap that passes a multi-hour input is not a cap.

## [got-20260803180153-63845d]
Category: gotcha
Tags: python, cpp, ffi, geostatistics, validation
Changed: 2026-08-03T18:01:53.332378

mask semantics - the meaning of 'informed' (mask != 0 vs mask == 1) must be consistent across Python and C++ implementations: Python variogram routines treat any non-zero mask cell as informed while the C++ kriging kernel requires exactly 1, so mask=2 cells are counted by Python diagnostics but never simulated (III-13, false '2 of 3 cells could not be kriged'); three variogram implementations disagree on what a non-zero mask means (II-41 CalcVPC mask=2 halves; Cubes2PointSet 8+8 trailing zeros; MovingAverage3D every cell undefined) despite a docstring contract permitting non-zero masks. Fix: validate that masks are binary (0/1) at the Python boundary and align the C++ semantic to mask != 0 (or vice versa) with a cross-language test. Checklist: (a) grep every mask consumer for its 'informed' predicate (mask==1 vs mask!=0), (b) Python and C++ must use the same predicate, (c) validate mask values are binary at API boundaries.

## [got-20260803180154-b873f2]
Category: gotcha
Tags: numerical, reproducibility, cpp, ffi
Changed: 2026-08-03T18:01:54.564330

64-bit seeds are silently truncated mod 2^32 when handed to a 32-bit RNG state: static_cast to mt19937::result_type (uint32) makes distinct seeds (e.g. 5 and 2^32+5) produce bit-identical variograms (II-15 variograms.cpp:707, live probe; the prior 'fix' 05bcec5 introduced the defect); the ctypes side wraps mod 2^64 (III-02, 2^63 -> -2^63). Seed handling must preserve the full 64-bit value (split into two 32-bit words for mt19937 seeding) and validate range BEFORE conversion. Checklist: (a) for every RNG seed path, verify the full-width seed survives to the RNG state, (b) test seeds that differ only above 2^32 produce different streams, (c) validate seed range at the API boundary before truncating conversion.

## [got-20260803180158-e9ddad]
Category: gotcha
Tags: numerical, geostatistics, python, cpp
Changed: 2026-08-03T18:01:58.635479

data-space vs normal-score-space mixing in transforms silently corrupts results: when a pipeline mixes CDF/normal-score transforms, every comparison must happen in ONE space. In hpgl-reborn: F-02 (GTSIM truncation compares data-space output against normal-score thresholds - category proportions grossly wrong, 0.50->1.000), II-39 (SGS output is DATA space, not standard-normal - the back-transform exists at :142-146; non-default tk params silently distort proportions and the F-02 fix alone does NOT repair it), III-37 (scalar stationary mean=50 NOT CDF-transformed while the LVM mean IS - sparse cells pinned to CDF max datum). Fix: document each output's space (data vs standard-normal), transform every input to the same space before comparison/truncation, and add a round-trip test. Checklist: (a) for any truncation/back-transform, verify the compared quantities are in the same space, (b) sibling branches (scalar vs LVM mean) must apply the same transform, (c) round-trip tests must cover the max datum.

## [got-20260803180159-7eedaa]
Category: gotcha
Tags: build, packaging, python, version
Changed: 2026-08-03T18:01:59.836276

__version__ derived from the INSTALLED distribution metadata can be stale relative to the source tree: src/geo_bsd/__init__.py reported '1.6.0' because a stale hpgl-1.6.0.dist-info remained installed, while the source and wheel were 2.0.2 (II-28, live-reproduced); the guard test was tautological (checked the installed dist, not the source). Fix: derive __version__ from the source (a version file, or importlib.metadata guarded by a source-vs-installed consistency check) and make the version gate compare source version against installed metadata rather than asserting a constant. Checklist: (a) verify __version__ equals the source package version in a clean venv, (b) a stale dist-info in the environment must not change the reported version, (c) version-gate tests must fail when source and installed versions diverge.

## [got-20260803180203-57a007]
Category: gotcha
Tags: python, packaging, compatibility
Changed: 2026-08-03T18:02:03.988895

PEP 604 union annotations (X | Y) executed at module scope break Python < 3.10 unless 'from __future__ import annotations' is present: geo.py:163 ('dict | None') raised TypeError on import on the declared-supported Python 3.9 (III-24, proven on 3.9.25 with installed wheel; the line was introduced by a production-check commit cbe636c and survived 4 passes because build/tests run on 3.13). Declared version support (requires-python, classifiers, cp39 matrix, py3-none tag) with no old-version CI means nothing catches it. Fix: either add 'from __future__ import annotations' at the top of every module with PEP 604 annotations, or avoid PEP 604 at module scope; add a lowest-supported-version import smoke test (e.g. a 3.9 wheel install + import). Checklist: (a) grep for union annotations and 'from __future__ import annotations', (b) verify every module with PEP 604 syntax at module scope has the future import, (c) run an import smoke test on the oldest declared-supported Python.

## [got-20260803180205-fa2e8c]
Category: gotcha
Tags: io, parsing, cpp, ffi
Changed: 2026-08-03T18:02:05.435084

fixed-buffer tokenizers silently corrupt a token that straddles a buffer boundary: read_inc_file.cpp token_stream_t accumulates across refills but the III-10 fix memcpy'd only the final chunk - a '99' token split at the 511-char boundary became 9.0 with rc=0 and NO fallback (silent wrong data, strictly worse than the pre-fix loud error); the token-count check (I2-56) is defeated because the split token counts as two values. Python-side line.split() materializes tokens before any length check (II-42, 3.4MB line -> 500k strings/28MB RSS). Fix: buffer reassembly that preserves straddling-token leading bytes and excludes delimiter/trailing bytes in BOTH text and byte paths; bound token length before materialization. Checklist: (a) for every fixed-buffer tokenizer, test a token exactly spanning a refill boundary, (b) verify token-count checks survive boundary splits, (c) add a regression test for the straddle path, not just the overlong-token throw path.

## [got-20260803180209-08702a]
Category: gotcha
Tags: io, windows, cpp, file-writing
Changed: 2026-08-03T18:02:09.626210

Windows file-replacement contract: a file with an open handle CANNOT be renamed/replaced (MoveFileExA -> ERROR_SHARING_VIOLATION; CRT opens with _SH_DENYNO, no FILE_SHARE_DELETE) - the R-05 regression broke all 4 property_writer.cpp writers because the II-14 fix renames the temp while the FILE* is still open (pre-fix self-rename was a silent no-op); and the II-14 guard removes the TARGET on write error (guard armed on target, rename no-op) destroying the pre-existing file. Fix: close/flush the handle BEFORE rename/replace in every writer, and arm the guard on the TEMP file, never the target. Checklist: (a) every write path must close the FILE* before rename/MoveFileExA, (b) failure cleanup must remove the temp, not the target, (c) verify the error path preserves the pre-existing target.

## [got-20260803180210-03bda0]
Category: gotcha
Tags: concurrency, io, ffi, locale
Changed: 2026-08-03T18:02:10.793826

non-glibc setlocale race: on macOS (de_DE) concurrent I/O wrappers that call setlocale while another thread parses with sscanf corrupt the parse (fractional parts dropped 1.5->1.0) AND permanently corrupt the process locale; glibc's setlocale is thread-safe but other platforms are not. The C++ locale_keeper mutex only guards the setlocale calls, not the parse window (II-35, live race reproduced). Fix: serialize the full parse window (setlocale + sscanf + restore) under one lock, or avoid setlocale entirely (use a locale-independent parse). Checklist: (a) grep setlocale and sscanf/%f parse sites, (b) verify the lock covers parse AND restore, not just the setlocale call, (c) test under a non-glibc locale (de_DE) with concurrent calls.

## [got-20260803180214-392ca5]
Category: gotcha
Tags: numerical, lapack, cpp, validation
Changed: 2026-08-03T18:02:14.517186

LAPACK INFO=0 does not guarantee numerically meaningful results: a near-singular SPD matrix can pass dpotrf_/dpotrs_ with INFO=0 and return wild weights (1.0e12, -1.0e12) reported as KI_SUCCESS (II-09 solver_entry_point.h:147-172, compiled probe; exactly-singular is caught, near-singular is not). Add weight-magnitude validation on the success path (or a condition-number check) so wild estimates are rejected. Checklist: (a) after every successful LAPACK solve, validate weight magnitude / condition, (b) test near-singular (not just exactly-singular) matrices, (c) treat INFO=0 as 'no factorization error', not 'numerically trustworthy'.

## [got-20260803180215-d64b36]
Category: gotcha
Tags: numerical, validation, cpp, python
Changed: 2026-08-03T18:02:15.831913

zero/empty boundary values silently switch an algorithm to a DIFFERENT branch: max_neighbours=0 (legal 'unconditional simulation') silently becomes 1-neighbour conditioned kriging (II-13, pure-nugget fallback fires on count==0, live probe mean 4.7493 vs unconditional 0); thickness==0.0 routes to the erosion branch and blanks an ENTIRE column (II-57, zero-layer-first -> blank_value, C++ probe; PRIOR_FIX_ATTEMPT 3ad77ee missed the zero boundary). Fix: explicitly handle zero/empty boundary values (raise or implement the documented semantics), never let them fall through to an unrelated branch. Checklist: (a) for every branch selected by a magnitude test (>=0 vs >0), test the exact zero input, (b) document and validate zero-value semantics at the API boundary, (c) regression-test the zero case, not just positive values.

## [got-20260803180219-d413b7]
Category: gotcha
Tags: build, cmake, windows, msvc
Changed: 2026-08-03T18:02:19.562243

multi-config CMake generators (Visual Studio, Xcode) IGNORE CMAKE_BUILD_TYPE - a preset that sets CMAKE_BUILD_TYPE=Release with no configuration member and no --config silently builds Debug (II-22 windows-msvc preset: no /sdl, no /guard:cf in Debug; the prior 'fix' e791f6b REMOVED --config and CAUSED the regression). Fix: pass --config Release explicitly in the build command or add a configuration member to the preset; never rely on CMAKE_BUILD_TYPE with multi-config generators. Checklist: (a) for every multi-config generator build, verify the actual configuration via the build command, (b) a 'fix' that removes --config while using a multi-config generator is a regression, (c) test the built binary's flags (e.g. /guard:cf presence) to confirm the intended config.

## [got-20260803180220-67e8a0]
Category: gotcha
Tags: python, numpy, compatibility
Changed: 2026-08-03T18:02:20.636725

numpy >= 2.4 raises ValueError when a 1-element array is assigned to a scalar slot (previously a deprecation): scripts that do scalar = arr[0:1] (or shape-(1,) arrays into scalar positions) crash on modern numpy - F-09 (2_corr_npv.py crashes first iteration), F-17 (shape-(1,) array -> scalar slot in 8.3/cdf_pdf.py, HIGH->MEDIUM, numpy-version-dependent). Fix: index with arr[0] (scalar) instead of arr[0:1] (array), and run book/sample scripts on the newest supported numpy. Checklist: (a) grep for [0:1] or shape-(1,) slices assigned to scalar slots, (b) test scripts on numpy >= 2.4, (c) prefer explicit scalar extraction.

## [pat-20260803180224-d58501]
Category: pattern
Tags: testing, verification, sample-scripts
Changed: 2026-08-03T18:02:24.800155

sample-script tests must EXECUTE the scripts, not assert on their source strings: TestSampleScriptImports-style tests that only check importability or string contents let scripts with stale file references ship broken - F-06 (test_gtsimk.py loads a filename no script produces - documented workflow always fails), F-53 (mean_calc_hist.py loads a nonexistent file), F-54 (>=9 sample scripts with stale data-path references), R-06 (8 scripts crash post-fix with CriticalValidationError on '..' in TEST_DATA_DIR because the test asserted source strings only). Fix: sample-script verification must run the scripts end-to-end (execute with representative data), asserting exit/expected output, not just import or string presence. Checklist: (a) any sample-script test that doesn't execute the script is incomplete, (b) assert runtime behavior (output values, error-free run), not source content, (c) run scripts with the actual data files they reference.

## [got-20260803180225-9e2f10]
Category: gotcha
Tags: numerical, geostatistics, python, cpp
Changed: 2026-08-03T18:02:25.747985

exact-equality lag binning drops legitimate pairs when coordinates are fractional: PointSetScanGridStyle matches lag distances with exact integer equality, so 0.5-m point spacing yields ZERO pairs while integer spacing works (III-15, live repro; fractional coordinates are legal - no integrality check). Use tolerance-based lag matching for point-set scans. Checklist: (a) for any distance-based binning, verify matching uses a tolerance window, (b) test with fractional coordinate spacing, (c) exact float equality in binning is a latent silent-zero-output bug.

## [got-20260803180229-53975d]
Category: gotcha
Tags: python, packaging, api
Changed: 2026-08-03T18:02:29.600780

documented public API must be smoke-tested at the TOP-LEVEL import: gtsim_2ind (II-29) and SGSConfig/SISConfig/GTSIMConfig (II-30) are documented as public but missing from geo_bsd.__init__ - every top-level access AttributeErrors while tests pass because they import via submodule paths (geo_bsd.gtsim, geo_bsd.config); the sample scripts gtsim.py/gtsimk.py similarly ImportError on private helpers (F-04). Fix: smoke-test the documented public surface at the top level (from geo_bsd import X for every documented name) and re-export everything documented as public. Checklist: (a) for every documented public name, verify a top-level import works, (b) tests must exercise the documented import path, not submodule shortcuts, (c) __init__ re-exports must match the docs.

## [got-20260806182524-427544]
Category: gotcha
Tags: numerical, cpp, lapack, solver
Changed: 2026-08-06T18:25:24.042395

solver magnitude gates must be scale-invariant and path-aware: a raw solution-magnitude bound on |X| is scale-dependent because X = A^-1*b scales with the data/sill (in hpgl-reborn the gauss/OK gate block regressed 5 times: E-M87 absolute gate -> R-01 AND-form -> R3-01/R3-02 dynamic-range -> R4-01 path-aware target_variance; each variant rejected a legal input class: small-sill OK systems with sill ~1e-4..1e-3 sanctioned by MIN_SILL=1e-6, cokriging sigma ratios >1e3, correlogram sigma_c>sigma_0). Fix: measure scale-invariant quantities (final weight combination w = X0 - mu*X1, or normalize the bound by matrix scale) and pass the true target variance as a parameter instead of inferring A_orig[0]. A gate that fires on API-legal inputs is a regression, not a guard. Refines got-20260803180214-392ca5 (LAPACK INFO=0): weight validation must exist AND be scale-invariant.

## [got-20260806182527-fb458a]
Category: gotcha
Tags: caps, numerical, cpp, python, regression
Changed: 2026-08-06T18:25:27.383126

work-cap and limit changes regress legal input classes: 10 of 18 post-fix CODE-FIX findings were FIX-INTRODUCED from cap/normalization changes (hpgl-reborn Stage-6): SCAN_LIMIT=4913+FALLBACK_WINDOW=8 collapsed effective search radius to ~8-10.6 cells while a distance-12 datum ranks 7,154 (E-H2); max_neighbours in (4913,100000] accepted but never honored (E2-145); E2-139 10x radius rule broke shipped book workflow 7.3/2_var.py (16x legal) and sk_test.py; 100000->10000 cap broke 3 pinned tests (R-07); lag_min clamp made lag 0 unbinable (R-02); hemisphere cut dropped zero-distance distinct-coincident pairs (R-08); fallback full-box scan unbounded 99.25M cells/node (R-13/R-15/R-18). Fix: before landing any cap/limit change, enumerate the real worst-case lattice/count arithmetic, test every legal boundary (radius 10x, zero/edge values, coincident points), and grep shipped workflows that exercise the class. Extends pat-20260803180149-88787b and pat-20260802092553-868986.

## [got-20260806182534-96ff3b]
Category: gotcha
Tags: numerical, cpp, simulation
Changed: 2026-08-06T18:25:34.040812

gate geometry must match the solve's admission predicate: the ndmin gate counted originals with a box+threshold geometry (covariance_field.h) while the solve admits by ellipsoid covariance, diverging in BOTH directions - (a) gate counts a datum the solve excludes -> marginal draw despite ndmin contract, (b) gate excludes a datum the solve's fallback would serve -> node under-admitted (hpgl-reborn R-17, R2-08, R2-06, E-M58, E-M84). Same class: R-13/R-15/R-18 unbounded full-box scans in the gate path and E2-01/R-09 ndim==3-only gates vs equal-volume 2D admission. Fix: a gate that decides admission must use the SAME predicate as the solve it guards (count the solve's own candidate set, or replicate its exact box/threshold/ellipsoid test), and the gate cost must stay bounded to solve cost. Checklist: for any pre-solve gate, verify it admits exactly the nodes the solve would serve - test the far-field fallback case both directions.

## [got-20260806182536-168e5f]
Category: gotcha
Tags: io, numerical, python, cpp
Changed: 2026-08-06T18:25:36.347073

%E-6 (7 significant digits) does NOT round-trip float32 mask sentinels: a fractional undefined_value (e.g. -99.00005) written with %E-6 reloads as a slightly different float, so exact-equality re-masking silently marks 17-30% of masked cells INFORMED (hpgl-reborn E2-06, measured 18.1%/30.7%, adversarial re-probe 17.5%/29.5%; R-06 tolerance mismatch: Python 1e-6-relative masks near-sentinel real data while C++ uses exact float32 ==). Fix: write >=9 significant digits (%.9E) for sentinels and/or use tolerance-aware re-mask, and keep the re-mask window identical across Python slow-path, C++ fast reader, and writer. Extends got-20260802092630-2eec8a (sentinel trimming) with the write-side precision facet.

## [got-20260806182541-e31c23]
Category: gotcha
Tags: numerical, gslib, cpp
Changed: 2026-08-06T18:25:41.838200

ordrel S>1 renormalization: when the per-category PMF sums to more than 1 (S>1, heterogeneous per-category covariances), clamping each probability to [0,1] and then scaling the top to 1.0 is a NO-OP (clamp pins probs.back()==1.0) so excess mass is silently TRUNCATED onto earlier categories instead of GSLIB's divide-by-total (sumcdf) - causing a silent category-selection FLIP (hpgl-reborn E-M56: p=[0.6,0.7,0.5] S=1.8 -> HPGL argmax 0 vs GSLIB argmax 1; reachable with per-category covariance models and >=3 categories). Fix: when sum of corrected PMF > 1, divide every category by the total (GSLIB sumcdf renormalization) rather than clamp+scale-top. Extends pat-20260803001248-62e195 (ORDREL): 'renormalize to sum 1' must be implemented as divide-by-total, and clamp-first defeats it whenever S>1.

## [got-20260806182544-6d7b41]
Category: gotcha
Tags: numerical, cpp, lapack
Changed: 2026-08-06T18:25:44.550404

consistency-check reference must be path-aware, not assumed from the matrix: a Schur-consistency/variance check whose reference is inferred from A_orig[0] is only correct when A[0][0] equals the actual kriging target variance - true for SK/cokriging but FALSE on the sigma-scaled correlogram path where A[i][j]=cov*(sigmas[i]*sigmas[j]) so A[0][0]=sill*sigma0^2 != target sill*sigmac^2, spuriously REJECTING valid SIS-LVM estimators whenever sigmac > sigma0 (hpgl-reborn R4-01: |X|=17688, PSD, kriging variance 0.0223>=0, yet rejected against A[0][0]*(1+2e-8); accepted with correct reference sigma_c^2=0.25). Fix: pass the true target variance (sigma_c^2) as an explicit parameter to the solver/check, or parameterize the check per path; never infer the reference from a matrix diagonal that scaling may have transformed.

## [pat-20260806182549-f8b7ea]
Category: pattern
Tags: testing, python, regression
Changed: 2026-08-06T18:25:49.848723

vacuous-test family: fully-informed fixtures defeat truncation/restore assertions - a test whose fixture mask is all-ones with binary data makes hard_cells == every cell, so output == input and proportion assertions become tautologies (hpgl-reborn R-04: test_gtsim_2ind_low_pk_gives_few_facies1 + 4 sibling tests vacuous; B-02: 7 gtsim_2ind siblings vacuous; B-06: cannot-fail tests; B-07: test_property_cleanup cannot fail by design). Extends pat-20260802223242-44d3e0 (behaviorally-inert fixes): a regression test must FAIL on pre-fix code - use partially-masked fixtures (binary hard + -99 masked cells) so restore pins only informed cells and truncation semantics stay exercised. Checklist: (a) for any proportion/truncation assertion, verify the fixture has masked cells, (b) delete or rewrite tests that cannot fail, (c) a test passing pre- and post-fix is not a regression test.

## [got-20260806182551-a330e5]
Category: gotcha
Tags: testing, python, performance
Changed: 2026-08-06T18:25:51.766531

stale .benchmarks baselines cause deterministic-looking benchmark failures: committed/checked-in performance baselines (June-16) vs post-fix kriging perf triggered a 3x regression check on ok_medium_grid baseline 0.0131s (hpgl-reborn B-04 CONFIRMED) - the failure looks real but is pure calibration staleness. Fix: delete .benchmarks/ before any test run so fresh calibration happens (first run only checks the hard floor), and never treat benchmark-check failures against stale baselines as code regressions. Checklist: (a) before perf test runs, verify .benchmarks/ baselines are fresh, (b) treat benchmark failures as stale-baseline artifacts until baselines are regenerated, (c) consider a hard floor only for first-run calibration.

## [got-20260806182611-e67fb7]
Category: gotcha
Tags: cpp, ffi, validation
Changed: 2026-08-06T18:26:11.799280

direct-C++ entry points must validate what the C API gates: a class of defects recurs where the C API fully validates input (sizes, finiteness, bounds) but the direct C++ public entry (hpgl_core.h) skips it - unchecked secondary/grid sizes -> abort()/SIGABRT or silent degradation (hpgl-reborn E-M61 cokriging, E-M62 calc_mean pointer overload NaN, E2-109 SGS-LVM mean_data OOB read, E2-118 SIS-LVM rows, E-M63/E2-126 default-ctor-trap leaving invariant uninitialized). Fix: validate at the C++ entry point itself, not only in the Python/C wrapper; every public C++ function that accepts raw arrays must check sizes/finiteness exactly like its C API sibling. Checklist: (a) for every C API that validates, check whether the underlying C++ function is separately callable and gated, (b) test the direct-C++ path with the same adversarial inputs as the C API, (c) default constructors must initialize invariants (or the setter order trap repeats).

## [got-20260806182614-31a5fd]
Category: gotcha
Tags: ffi, cpp, concurrency
Changed: 2026-08-06T18:26:14.130429

kriging/FFI C entries must reject input/output buffer aliasing: all 7 kriging C entries in hpgl-reborn lacked in/out buffer-identity guards (grep: 22 m_data comparisons, all nullptr, zero identity checks), so an aliased buffer (same array for input and output) causes progressive overwrite and an OpenMP data race on IK (E2-53 CONFIRMED both-found; Python immune only because it clones). Fix: add buffer-aliasing guards (in != out) to every FFI/numerical entry that reads and writes arrays; document the aliasing prohibition. Checklist: (a) for every C entry with in/out arrays, grep for identity checks between them, (b) test with deliberately aliased buffers (should raise loudly, not corrupt), (c) verify OpenMP builds are race-free under aliasing.

## [got-20260806182619-6ddcd8]
Category: gotcha
Tags: ffi, cpp, validation
Changed: 2026-08-06T18:26:19.932714

validate BEFORE destructive in-place mutation of caller data: SGS-LVM performed the destructive forward-CDF transform in-place on the caller's conditioning data BEFORE the radius-guard throw, so a failed run silently destroyed the direct-C caller's input (hpgl-reborn E2-112; Python protected only because it clones). Same class: E2-53 aliasing, pat-20260802223306-eb50d0 (numpy view mutation). Fix: order operations so all throwing validation/allocation happens BEFORE any in-place write to caller-owned buffers; when a function mutates its input, either document it loudly or operate on a private copy. Checklist: (a) for every function that transforms its input in place, verify every throw path precedes the first write, (b) test the failing-input path and confirm caller data is intact after the error, (c) wrappers that clone inputs mask the hazard for direct-C/C++ consumers.

## [got-20260806182623-e94b73]
Category: gotcha
Tags: numerical, python, numpy
Changed: 2026-08-06T18:26:23.033018

finiteness gates must test the SOURCE data, not a derived boolean: isfinite(derived_boolean) is ALWAYS True because numpy treats bool as a number - a gate that computes a boolean (e.g. np.isfinite(cube)) and then tests finiteness of that boolean (or of an index array) is structurally defeated, letting NaN-informed cells silently through (hpgl-reborn E2-27: the 828443e NaN gate on derived-boolean arg deflated marginals, sum 0.667 vs 1.0). Fix: call the finiteness predicate directly on the source cube (np.isfinite(cube)) and gate on the result, never on a derived boolean/index value; test with NaN present to prove the gate fires. Checklist: (a) grep gates whose argument is a comparison/index expression rather than the source array, (b) verify the gate fails loudly with NaN input, (c) sibling gates must all test the source, not derived values.

## [got-20260806182632-c70ef4]
Category: gotcha
Tags: ffi, numerical, python, cpp
Changed: 2026-08-06T18:26:32.717049

float32-range boundary: numpy.isfinite(float64) is True for values beyond float32 max, but converting to C.c_float / float32 silently produces inf - and inf <= 0.0 is FALSE, so a guard that checks thickness/scalez <= 0 fails to catch it, producing silent all-blank output instead of an error (hpgl-reborn E-M80 boundary-found: scalez=1e39 -> inf -> deposit loop never runs -> blank CStackLayers with no exception; PFA e5d8f73 added isfinite checks but missed the float32-range hole). Fix: when crossing a float32 boundary (ctypes C.c_float, numpy float32 conversion), range-check against float32 max BEFORE the conversion, not just isfinite. Checklist: (a) grep every float64->float32/ctypes c_float conversion for an explicit range check, (b) test values in (3.4e38, inf) - finite in float64, inf in float32, (c) verify the failure is loud, not silent blank output.

## [got-20260806182643-5ae68d]
Category: gotcha
Tags: io, cpp, parsing
Changed: 2026-08-06T18:26:43.916361

sscanf/atof numeric-prefix acceptance silently accepts garbage tokens: sscanf('5/', '%f') and atof('1.5abc') return success with partial parsing (no %n/endptr check), so corrupt tokens load as data instead of raising like the slow parser does (hpgl-reborn E-M73: read_inc_file.cpp:239,288 - fast C++ reader accepted '5/', '1.5abc', '255.0' while the Python slow parser raised; contradicts the documented 'fast reader THROWS on unparseable junk' contract). Fix: use strtod/sscanf with %n or endptr and verify the ENTIRE token was consumed (no trailing non-whitespace), or strict regex validation before parse. Checklist: (a) grep sscanf/atof/strtod sites for %n/endptr full-consumption checks, (b) test tokens with trailing garbage ('1.5abc', '5/', leading '+'), (c) the fast and slow parsers must reject the same token set.

## [got-20260806182646-af8fa9]
Category: gotcha
Tags: numerical, python, numpy, cpp
Changed: 2026-08-06T18:26:46.517097

integer-dtype accumulation truncates fractional means: computing a mean into an int/uint array (numpy) silently truncates fractional results to 0 (hpgl-reborn E2-26: MovingAverage3D int cube mean 0.63 -> 0 on both paths; also the E-M10 float32-before-squaring parity divergence where per-pair variance cast to float32 before squaring vs C++ double accumulation gives ~0.2% data-dependent divergence). Fix: upcast accumulators (float64) before arithmetic and cast only at the output, and match the reference implementation's accumulation precision exactly; check for dtype on every reduction target. Checklist: (a) grep reductions assigned into int/uint arrays, (b) verify float accumulators are float64 (or the same as the reference), (c) test fractional inputs, (d) cross-language parity requires identical accumulation precision, not just identical formulas.

## [got-20260806182659-b8afcc]
Category: gotcha
Tags: numerical, cpp, python, verification
Changed: 2026-08-06T18:26:59.933319

variogram twins diverge on binning conventions, not just metric: beyond the projection-vs-Euclidean metric class (pat-20260802223421-6e2873), Python/C++/shared twins diverge on (a) self-pair handling - grid kernel counted zero-distance self-pairs into lag 0 while point-set skips idx1==idx2 (hpgl-reborn E-M74, nugget diluted on grid path), (b) interval convention - grid closed intervals at lag-band ends double-count exact band-end distances while point-set/Python are half-open (E-M75), (c) rotated-ellipsoid hemisphere cut double-counts central-overlap pairs (E2-30: 6/89 pairs at azimuth=45), (d) accumulation precision float32 vs double (E-M10). Fix: when two implementations compute the same metric, verify they agree on self-pair policy, bin interval convention, hemisphere/overlap handling, and accumulation precision - each is a silent divergence surface. Checklist: (a) compare self-pair filters and interval boundaries across twins, (b) test band-end and zero-offset distances explicitly, (c) add a cross-language equivalence test.

## [got-20260806182713-f5d42e]
Category: gotcha
Tags: cpp, numerical
Changed: 2026-08-06T18:27:13.391053

loop-counter type must cover the count's range: an unsigned char (or other undersized) loop index wraps at 256 and can produce an infinite loop or wrap-around when the iteration count can exceed it - hpgl-reborn E-M54 pretty_printer.cpp:103 loops an indicator_index_t (unsigned char, typedefs.h:122) against size_t m_category_count: count >= 256 makes idx wrap 0-255-0 forever (hang); add_indicator has no cap and the C API gates block >255 but the direct-C++ path reaches print_params before kernel guards. Fix: use the same integer type as the count variable (size_t) for loop counters; validate/cap the count at the API boundary; when a type is deliberately narrow (uint8 indicator_index_t), enforce the cap at EVERY entry point, not only the C API. Checklist: (a) grep loop counters whose declared type is narrower than the compared count, (b) verify caps exist at all entry points reaching the loop, (c) test the boundary count (256, 65536) not just nominal values.

