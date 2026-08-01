# Knowledge Base
Last updated: 2026-08-01T15:33:42.549396
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
