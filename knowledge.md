# Knowledge Base
Last updated: 2026-10-04T20:18:40.531737

## [got-20260616055758-f1d951]
Category: gotcha
Tags: cvariogram, error-handling
Changed: 2026-06-16T05:57:58.641306

hpgl-reborn cvariogram module: no error retrieval API exists in _cvariogram (all functions return void). Python cvariogram.py has zero error detection at 4 call sites (calc_variograms, calc_variograms_from_point_set, cvar_stack_layers, fill_ellipsoid_directions). Fix requires C-level API addition. HIGH severity. (f0761e6)

## [dec-20260622193540-b18f4a]
Category: decision
Tags: pylasdev-reborn, policy, ci-cd, forbidden, owner-decision
Changed: 2026-06-22T19:35:40.292103

hpgl-reborn and pylasdev-reborn are deliberately minimal projects. FORBIDDEN to add: CI/CD infrastructure of any kind (.github/, GitHub Actions workflows ci.yml/release.yml, Dependabot config, .pre-commit-config.yaml, .gitlab-ci.yml, .circleci/, .travis.yml — any CI provider), GitHub-specific metadata (FUNDING.yml, ISSUE_TEMPLATE/, PULL_REQUEST_TEMPLATE.md, CODEOWNERS, dependabot.yml), unnecessary community/governance docs (CODE_OF_CONDUCT.md, CONTRIBUTING.md, SECURITY.md), CI-status README badges ([![CI](...actions/workflows/...)]). KEEP intact: README, LICENSE, CHANGELOG, RELEASING.md (pylasdev only), pyproject.toml, source/test/build files. GitHub repository URLs in pyproject.toml and GitHub Issues/Discussions links in README are FINE (legitimate project references, not CI/CD). When any agent proposes adding these forbidden artifacts (often as part of 'production hardening', 'release readiness', or 'best practices' passes), refuse. They were added once before and owner had to manually remove them — do not repeat. Policy documented in AGENTS.md → 'Project-specific policies' section.

## [dec-20260622194447-206ed2]
Category: decision
Tags: pylasdev-reborn, policy, documentation, forbidden, operator-gated, strict
Changed: 2026-06-22T19:44:47.268665

hpgl-reborn and pylasdev-reborn DOCUMENTATION POLICY (strict, operator-gated): NO new documentation may be added — not new files, not new sections inside existing files (README, CHANGELOG, etc.). Documentation work is restricted to FIXING ERRORS in existing documentation only (typos, broken links, factually incorrect statements, rendering/formatting bugs). Adding new documentation includes: creating new doc files (guides, tutorials, ADRs, architecture docs), adding new sections/subsections/appendices to README or any existing doc, 'filling documentation gaps', 'documenting an existing-but-undocumented feature', 'improving the README', 'best-practice docs', 'production readiness docs', 'release notes drafts', re-adding sections the operator previously removed (Changelog, Migration Guide, Validation Limits, Getting Help). THE ONLY EXCEPTION: explicit direct instruction from the operator (the human user, in the current session) to add specific documentation. A spawned agent or workflow proposing documentation additions — even as part of 'production hardening', 'release readiness', or 'best practices' pass — does NOT count as operator approval. When in doubt, ask the operator; do not add. Policy documented in AGENTS.md → 'Project-specific policies' → 'Documentation policy (strict — operator-gated)' subsection.

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

## [pat-20260701081644-bcddb1]
Category: pattern
Tags: cpp, optimization
Changed: 2026-07-01T08:16:44.989674

HPGL Reborn uses workspace kriging (kriging_interpolation_ws) for hot loops. SGS/SIS switched from non-workspace variant to reduce heap allocations per simulated node. Pattern: declare kriging_ws_t before loop, pass to each call.

## [dec-20260701081645-dab5e5]
Category: decision
Tags: build, blas
Changed: 2026-07-01T08:16:45.065189

HPGL Reborn ILP64 ABI risk documented rather than auto-detected. The lapack_compat.h has dead ILP64 code paths that never activate because build system never defines ILP64 macros. Added compile-time #pragma message on LP64 path to alert users.

## [got-20260724074703-b42be8]
Category: gotcha
Tags: numerical, nan, validation, python, cpp
Changed: 2026-07-24T07:47:03.983605

NaN defeats comparison-based validation, and finiteness must be tested on the SOURCE data. (a) Per IEEE 754, NaN<X, NaN>X, NaN==X, NaN!=X are ALL False, so standard clamping guards (if x<0: x=0; if x>1: x=1) and range checks are NaN-ineffective — NaN passes through silently. Categorical dispatch chains using thresholds send NaN to the last/default category (systematic bias). Every comparison-based guard in a numerical pipeline needs an explicit math.isnan()/std::isnan() check BEFORE the comparison; np.clip is NaN-ineffective. hpgl-reborn examples: no NaN check before dpotrf_; Gaussian sample() clamp; correllogram second_stage; categorical sample() NaN->last-category; _norm_ppf. (b) Finiteness gates must test the SOURCE data, not a derived boolean: isfinite(derived_boolean) is ALWAYS True (numpy treats bool as a number), so a gate computing np.isfinite(cube) and then testing finiteness of that boolean (or of an index array) is structurally defeated, letting NaN-informed cells through (E2-27: sum 0.667 vs 1.0). Call np.isfinite(cube) directly and gate on the result; test with NaN present to prove the gate fires. Checklist: (a) for every comparison-based guard, verify NaN is handled before the comparison; (b) for every categorical dispatch, NaN input produces an explicit error, not a default category; (c) grep gates whose argument is a comparison/index expression rather than the source array; (d) test with np.nan input explicitly.

## [got-20260724074713-2117a3]
Category: gotcha
Tags: numerical, cpp, threshold, unit-dependent
Changed: 2026-07-24T07:47:13.906555

Hardcoded unit-dependent numerical threshold creates blind zone: A hardcoded near-zero threshold (e.g., 0.0001) used to treat small covariance values as zero is unit-dependent — at micro-scales (millimeter, micron), legitimate values fall below the threshold and are incorrectly zeroed. At macro-scales, noise above the threshold escapes detection. The threshold is absolute but the data scale is relative to unit choice. In hpgl-reborn: F-46 (cov_model.h:170,183,195) — hardcoded 0.0001 nugget-blind zone confirmed MEDIUM. Fix: (a) parameterize the threshold — derive from data range (e.g., epsilon * max_abs_value) or make it configurable, (b) use relative comparison: abs(x) < epsilon * scale where scale is data-dependent, (c) document the threshold unit assumption explicitly if hardcoding is unavoidable. Checklist: for every hardcoded fabs(x) < CONSTANT comparison, verify the constant is appropriate for all supported units.

## [got-20260724074718-5ea5da]
Category: gotcha
Tags: type-conversion, ffi, numpy, ctypes, validation
Changed: 2026-07-24T07:47:18.896421

Type conversion silently turns invalid values into apparently-valid ones, and validation must run on the right value and path. (a) Variable-reference mismatch: when data is converted (np.require(data,'uint8') wraps -1->255) and then validated, checking the ORIGINAL unconverted variable produces a false pass (I2-F12 geo.py:186-208: the range check used the pre-conversion array). The conversion and the validation must target the same memory. (b) The 'obvious' fix can reintroduce the bypass: replacing the pre-conversion reference with the post-conversion property (self.data) is wrong when the property getter returns the already-truncated value — input 256.0 -> uint8 0 -> '0 >= 3' is False and passes silently, whereas the original pre-conversion check correctly caught 256.0 >= 3; five subsequent production checks missed the regression (e19aa84). Rule: validate BEFORE a lossy transformation when the goal is to detect overflow, or after conversion when the goal is to detect an already-corrupted value — and test the edge that wraps. (c) ctypes small-int scalar types do NOT range-check: assigning an out-of-range Python value to c_ubyte/c_uint8 wraps mod 256 (300->44) with no error (unlike numpy 2.x, which raises for Python ints); at FFI boundaries validate ranges in Python before ctypes conversion (F-44: indicator_values 300->44 silently corrupted written files). Checklist: (a) after any type conversion, verify which variable the validation sees and whether it is pre- or post-conversion; (b) for uint8/int conversion, only the pre-conversion array detects overflow (>255, <0) — post-conversion wrapping is silent; (c) grep _c_array/c_ubyte/c_int and validate in Python first; (d) after any fix near a conversion site, test an edge input that would wrap/overflow.

## [got-20260724074724-8a6528]
Category: gotcha
Tags: ffi, python, assert, optimization, safety
Changed: 2026-07-24T07:47:24.267114

Bare assert at FFI or safety-critical boundary removed by python -O: Python'\''s assert statement is compiled away when running with -O (optimized) flag. At FFI boundaries, array stride validation, or any safety-critical code path, a bare assert that guards against memory corruption provides zero protection in optimized/production mode — the check is silently removed. The code appears guarded but has no guard at runtime. Use explicit if condition: raise ValueError instead of assert. In hpgl-reborn: I2-F35 (ffi_adapter.py:356) — bare assert stride check for 3D arrays is removed by python -O; the contiguous-array creation path also has ZERO stride check (asymmetry confirmed). BOTH-FOUND by two specialists. Fix: replace ALL bare asserts at module/FFI boundaries with explicit if/raise. Keep asserts only for internal invariants that are logically impossible to violate (and even then, document why -O removal is acceptable). Checklist: (a) grep for '\''assert '\'' in FFI adapter files, (b) every assert guarding input data (array shape, strides, dtype) must be if/raise, (c) every assert at module public API boundaries must be if/raise.

## [got-20260724074732-e63532]
Category: gotcha
Tags: ffi, python, cpp, error-handling, boundary
Changed: 2026-07-24T07:47:32.514124

FFI error suppression via identical-message collision: When an error suppression protocol uses string equality on error messages to distinguish new errors from stale ones (current_error == saved_snapshot → suppress), two genuinely different calls producing the same error message are indistinguishable from a stale error. The suppression protocol incorrectly swallows the second call'\''s error. This is a contract-semantic gap: the protocol was designed to suppress re-reads of a sticky error, but string-equality collision makes it suppress distinct errors with coincidentally identical messages. In hpgl-reborn: I2-F32 (ffi_adapter.py:137-141) — snapshot protocol uses bytes equality on error messages; C++ never clears errors (overwrite-only); retry with same bad inputs → identical message → err == snapshot → suppression. Affects ALL 11 void-returning FFI wrappers. BOUNDARY-FOUND (intersection agent). Fix: use monotonic counters or call IDs instead of message-text equality for stale-error detection. Minimum: add a generation counter incremented on every FFI call; snapshot = (counter, message); only suppress when counter matches AND message matches. Checklist: for every error-suppression or dedup protocol using string/message equality, verify it cannot collide on distinct errors.

## [got-20260724074743-3331f1]
Category: gotcha
Tags: dead-code, diagnostics, cross-language, validation
Changed: 2026-07-24T07:47:43.668987

Dead/unwired infrastructure disguised as working instrumentation creates false confidence. (a) Tracking fields (counters, call IDs) written but NEVER read accumulate data no decision logic consumes (ffi_adapter _hpgl_call_counter/_hpgl_call_id/_hpgl_suppressed_call_id written every FFI call, read in zero decision paths; added for cross-call error detection but the detection was never implemented). (b) Cross-language dead integration: diagnostic infrastructure can exist at ALL layers (C++ struct + setter, C API export, Python ctypes binding, re-export) yet have ZERO callers in the wrapper layer — comments claiming 'the Python wrapper reads stats' are provably false (kriging_stats_t / get_kriging_stats() fully plumbed but zero callers in geo.py/sgs.py/sis.py; this was the systemic root cause behind 3 CONFIRMED HIGH silent-failure findings). (c) Unused public helpers: 4 validation decorators defined and exported with zero usages; all validation goes through explicit Validator calls. Fix: either remove the dead code and correct the comments, or implement the missing wiring. Checklist: (a) for every 'tracking'/'monitoring' field, grep for READS — zero reads = dead infrastructure; (b) for every C++ diagnostic struct with a C API export and Python binding, grep the Python function name in ALL wrapper modules (not just the binding file) — zero matches = dead integration; (c) verify every comment claiming cross-language usage against grep; (d) a comment describing unimplemented behavior is worse than no comment.

## [pat-20260724134312-6a29cc]
Category: pattern
Tags: python, numpy, dataclass, type-validation, consistency, isinstance
Changed: 2026-07-24T13:43:12.395034

Type gate consistency across related fields in dataclass validation: When one field in a dataclass __post_init__ accepts a broader isinstance type (e.g., seed accepts (int, numpy.integer)), ALL related integer/neighbour fields in the same dataclass and sibling dataclasses must accept the same type. An inconsistency where seed accepts numpy.integer but min_neighbours/max_neighbours rejects it creates a confusing API: np.int64(5) succeeds for seed but raises TypeError for neighbours. Checklist: (a) for every isinstance gate that was broadened to accept numpy types, grep for all sibling isinstance gates on related numeric fields in the same dataclass and its sibling classes — every one must have the same type tuple, (b) the bool exclusion guard (or isinstance(self.field, bool)) must also be present on every gate to prevent Python bool-as-int bypass, (c) after adding type flexibility, mechanically audit all dataclass __post_init__ methods for isinstance consistency. In hpgl-reborn: F-08 (MEDIUM, CONFIRMED, both-found) — SGSConfig.seed accepted (int, numpy.integer) but min_neighbours/max_neighbours only accepted int; same gap in SISConfig.max_neighbours. Fix: added numpy.integer to all 3 neighbour isinstance gates.

## [pat-20260724134318-f0500c]
Category: pattern
Tags: ffi, validation, numerical, python, cpp
Changed: 2026-07-24T13:43:18.305420

Post-FFI output validation — validate outputs from C/C++ calls before returning to callers. C++ can produce NaN/Inf through internal solver paths (ill-conditioned matrices, dpotrs_ on near-singular SPD systems); without post-call validation corrupted data propagates silently. Every FFI wrapper that receives a result array should validate numpy.isfinite() on the output after the call, raising RuntimeError (a C++ computation failure, not a user input error); uint8/indicator arrays are naturally finite and skip the check. BUT isfinite() is structurally insufficient whenever the solver has a FALLBACK that produces finite values by design: mean_on_failure fills failed cells with the global mean, gaussian_cdf_t fills [0,1], marginal_probs fills valid probabilities — all pass isfinite(), no exception is raised, and synthetic data propagates. When any fallback produces finite values, the real detection must come from solver-level counters (kriging_stats_t) wired into every wrapper; never rely on output-array inspection alone. Checklist: (a) every FFI wrapper validates isfinite() on float output before returning; (b) for every solver failure fallback, determine whether it produces finite values that pass isfinite(); (c) if yes, add stats/counter-based failure detection and wire get_kriging_stats() (or equivalent) into the wrapper; (d) never assume a fallback value that 'looks wrong' will be caught by a range check — mean_on_failure uses the legitimate global mean.

## [got-20260724134324-46e739]
Category: gotcha
Tags: numerical, cpp, validation, solver, simulation
Changed: 2026-07-24T13:43:24.749980

Validate ALL inputs a solver/gate uses, and make admission gates use the SAME predicate as the solve they guard. (1) Partial input validation within one function: validating the primary matrix A for NaN/Inf but not the RHS vectors B/B0/B1 lets NaN slip into dpotrf_/dpotrs_/gauss_solve and propagate NaN solutions (F-24, both-found by 4 agents); enumerate ALL vector/matrix inputs (primary matrix, RHS vectors, work arrays) and scan each, and make the error path for secondary inputs match the primary path. (2) Gate geometry must match the solve's admission predicate: an ndmin gate counting originals with a box+threshold geometry while the solve admits by ellipsoid covariance diverges in BOTH directions — it counts a datum the solve excludes (marginal draw despite the ndmin contract) or excludes a datum the solve's fallback would serve (under-admission) (R-17, R2-08, R2-06, E-M58, E-M84). A gate that decides admission must count the solve's own candidate set / replicate its exact box-threshold-ellipsoid test, and the gate cost must stay bounded to solve cost. Checklist: (a) enumerate every input to a solver and scan each for non-finite; (b) for any pre-solve gate, verify it admits exactly the nodes the solve would serve — test the far-field fallback case in both directions.

## [got-20260724221118-8aecf5]
Category: gotcha
Tags: python, warnings, exception, validation, inheritance
Changed: 2026-07-24T22:11:18.940587

Python warnings.warn(category=X) requires X to be a Warning subclass — Exception subclasses can never serve as warning categories: Python's warnings.warn() function internally checks isinstance(category, Warning). If a validation library defines a custom warning class (e.g., ValidationWarning) that inherits from Exception instead of Warning, it can NEVER be used as a warning category — attempting warnings.warn('message', ValidationWarning) will raise TypeError: 'category must be a Warning subclass, not ValidationWarning'. The class is silently dead for its intended purpose. In hpgl-reborn: validation.py defines ValidationWarning(Exception) at line 95-99 but its sole documented use site (validate_max_neighbors at line 556-561) correctly uses the default UserWarning instead — the docstring falsely claims ValidationWarning would be raised. ValidationWarning has zero instantiations, zero raise sites, and zero warnings.warn() calls anywhere in the codebase. The inheritance chain makes it structurally incapable of ever being used as a warning category — it must be refactored to class ValidationWarning(Warning) before it can appear in any warnings.warn() call. Checklist: (a) for every custom warning/exception class in a library, verify its inheritance chain matches its intended use — Warning for warnings.warn(), Exception for raise, (b) a class that inherits Exception but is referenced in docstrings as 'raises X as a warning' is architecturally broken — either change the inheritance or change the docs, (c) grep for warnings.warn(category=YourClass) and verify isinstance(YourClass, Warning) before claiming it works.

## [pat-20260724231046-bbc95d]
Category: pattern
Tags: ffi, state-management, concurrency, exception-safety
Changed: 2026-07-24T23:10:46.765785

FFI module-level shared state (e.g. _last_kriging_stats) must have a correct lifecycle on every path. Every wrapper that calls into C++ MUST either populate the module-level variable with fresh C++ stats or explicitly reset it to a None sentinel — a wrapper that neither populates nor resets leaks stale data from a PREVIOUS (possibly different-algorithm) call that is indistinguishable from fresh data (median_ik, indicator_kriging, simple_cokriging_markI/markII never touched the shared stats). Set the sentinel to None BEFORE the C++ call (defensive nullification) so an exception path yields honest None rather than stale data; populate on the success path. When the design supports concurrent calls (RLock), reset+populate OUTSIDE the serialization lock is a data race — two concurrent krigings cross-contaminate stats, producing spurious RuntimeError or missed failure warnings; every write AND read of the module-global must be under the same lock that serializes the C++ calls. Checklist: (a) audit every wrapper for shared module-level state — every path writes (populate) or clears (None); (b) the None-set precedes any C++ call that can raise; (c) concurrent-call designs make unsynchronized shared state a real race even if stress tests do not trigger it — reset, populate, and reads all inside the lock.

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

## [got-20260802015636-a24aca]
Category: gotcha
Tags: ffi, build, macos
Changed: 2026-08-02T01:56:36.327255

macOS dylib shadowing via lib{name} search-order preference: on macOS, ctypes/loader search order prefers lib{name}.dylib over {name}.dylib. A stale lib{name}.dylib left in the runtime directory silently shadows a freshly built {name}.dylib, so the test suite and production runs exercise a month-old binary. build.sh copies only hpgl.dylib and never removes the stale libhpgl.dylib; hash-guard verification with an empty expected-hash dict is a permanent no-op; smoke tests that only check 'library loaded' cannot catch API incompleteness. In hpgl-reborn: F-03 (CONFIRMED HIGH) - stale libhpgl.dylib (Jun 27, missing hpgl_get_kriging_stats) shadowed fresh hpgl.dylib (Jul 24), 4 test skips; same for lib_cvariogram.dylib (I2-43, I2-47); freshness checks passed silently because only 2 of the needed symbols were checked (PR-08). Fix: build.sh must delete stale lib{name}.dylib before copying; symbol-freshness check must assert the NEW API symbols exist (e.g. _EXPECTED_LIBRARY_SYMBOLS includes every new export); smoke test must assert API completeness, not just loadability. Checklist: (a) after any rebuild verify which file CDLL actually loads (lib._name), (b) grep the runtime dir for stale lib-prefixed dylibs and delete them, (c) extend symbol-freshness checks whenever a new C API function is added.

## [got-20260802015641-a15dae]
Category: gotcha
Tags: numerical, python, numpy, cpp, masks
Changed: 2026-08-02T01:56:41.416884

Mask semantics must be consistent and dtype-correct across Python and C++. (1) numpy int/uint8 arrays are NOT boolean masks: passing an int/uint8 array to indexing performs integer fancy indexing (selects rows) instead of masking, inflating counts ~5x (F-01: uint8 5x5x5 -> [625,1940,2223] vs bool [125,420,507]) or crashing when the index exceeds a dimension. Convert explicitly (arr.astype(bool) or arr>0) and validate mask dtype is bool at API boundaries. (2) The 'informed' predicate must be identical: Python variogram routines treat any non-zero mask cell as informed while the C++ kriging kernel requires exactly 1, so mask=2 cells are counted by diagnostics but never simulated (III-13, false '2 of 3 cells could not be kriged'); three variogram impls disagree on what a non-zero mask means despite a docstring permitting non-zero. Align on one predicate (mask != 0, or validate binary) with a cross-language test. Checklist: (a) grep data[mask] where mask may be int/uint8 — convert to bool; test with uint8, not just bool; (b) grep every mask consumer for its informed predicate and unify Python/C++; (c) validate masks are binary at API boundaries.

## [got-20260802015647-ee8c09]
Category: gotcha
Tags: numerical, cpp, python, ffi
Changed: 2026-08-02T01:56:47.423769

float32 CDF last-value rounding to exactly 1.0f corrupts the max datum: when a CDF is stored in float32, the final plotting position can round UP to exactly 1.0f. If the inverse-CDF path treats p>=1.0 by returning the mean (or median), the MAXIMUM datum maps to normal score 0.0 (median) instead of the correct high tail score - the max datum silently collapses to the median on back-transform. GSLIB uses midpoint plotting positions to avoid this. In hpgl-reborn: F-04 (CONFIRMED HIGH, both-found) - gaussian_distribution.h:73-77 / non_parametric_cdf.h:245-259 / cdf.py:126-131 / sequential_gaussian_simulation.cpp:44 - last float32 CDF prob rounds to exactly 1.0f, inverse(p>=1.0) returns mean, max datum -> normal score 0.0, round-trip corrupts max datum to median; verified by exact float32 arithmetic; PFA b305411 introduced the guard (incomplete fix). I2-27 adds the low-side mirror: LVM means below data-min map to p=0.0 -> inverse(0.0) -> m_mean=0.0 -> local means collapse to median on back-transform. Fix: saturate tails (mirror non_parametric_cdf.h:271-291), not return mean for p>=1.0; handle both high-side p rounding to 1.0f and low-side p=0. Checklist: (a) for any float32 CDF, verify the max plotting position cannot round to exactly 1.0f, (b) test inverse(p=1.0f) explicitly with a float32-built CDF, (c) verify the extreme data values survive a round-trip through the transform.

## [got-20260802015652-a891f8]
Category: gotcha
Tags: numerical, cpp, ffi, conversion, nan
Changed: 2026-08-02T01:56:52.868521

Float conversions have two silent-corruption boundaries. (a) FP->int cast is UB on NaN/Inf/out-of-range in C++ (x86-64 typically INT_MIN/garbage) and has silent-wrap traps in ctypes paths: a NaN thickness cast to int becomes INT_MIN, clamps to 0, then writes blank_value to an entire column and poisons cumulative state (I2-25 stack_layers.h:43,52,66; reachable via NaN written verbatim by Python). Guard every FP->int cast with explicit std::isfinite + range checks, or reject non-finite input at the API boundary. (b) numpy.isfinite(float64) is True for values beyond float32 max, but converting to C.c_float/float32 silently produces inf — and inf <= 0.0 is False, so a guard checking thickness/scalez <= 0 fails to catch it, producing silent all-blank output (E-M80: scalez=1e39 -> inf -> deposit loop never runs -> blank CStackLayers with no exception). When crossing a float32 boundary, range-check against float32 max BEFORE the conversion, not just isfinite. Checklist: (a) grep every FP->int cast and every float64->float32/c_float conversion for an explicit isfinite + range check preceding it; (b) test NaN/huge values in (3.4e38, inf) — finite in float64, inf in float32 — and confirm a loud failure, not silent blank output.

## [got-20260802015657-f8eb69]
Category: gotcha
Tags: concurrency, openmp, cpp, exception-safety
Changed: 2026-08-02T01:56:57.884636

OpenMP region hazards. (1) 'break' inside a worksharing loop (for/parallel for) is non-conforming (undefined behavior without OMP_CANCELLATION=true, the default): the breaking thread's remaining iterations are dropped, leaving NaN cells and incomplete counters while other threads continue (F-23 cont_kriging.h:193-198, median_ik.cpp:117-122); move the break into the #else non-OpenMP branch or use conforming cancellation flags (sibling indicator_kriging.h:216-228 is correct). (2) A C++ exception (e.g. bad_alloc) escaping an omp for/parallel-for region calls std::terminate (uncatchable by Python/ctypes) and is UB (F-H1, F-M8, F-M9): guards must validate BEFORE the parallel region (catchable -> clean error return) or catch inside and rethrow after; any allocation sized by user input must be hoisted out or pre-validated, and RAII guards must not depend on a restore line that an exception can skip. Checklist: (a) grep for 'break' inside omp for/parallel regions and for allocations/throws inside them; (b) C++ guard exceptions must reach Python as catchable RuntimeError, not SIGABRT/terminate; (c) test both OpenMP-enabled and serial builds.

## [got-20260802015708-3e64f9]
Category: gotcha
Tags: build, cmake, macos
Changed: 2026-08-02T01:57:08.244590

find_package(OpenMP) does not search Homebrew prefix on macOS: CMake's FindOpenMP does not look in /opt/homebrew/opt/libomp (or /usr/local/opt/libomp) by default. When libomp is installed via Homebrew and OpenMP_ROOT is not set, find_package(OpenMP) silently returns NOTFOUND and HPGL-style builds compile WITHOUT OpenMP despite HPGL_USE_OPENMP=ON - the parallel algorithms are silently disabled and shipped artifacts contain zero fopenmp flags. In hpgl-reborn: confirmed (CONFIRMED, s3-adv-M2 evidence: build/CMakeCache OpenMP_CXX_FLAGS=NOTFOUND, 0 fopenmp in compile_commands; fixed in s7 by adding Homebrew libomp prefix detection setting OpenMP_ROOT=/opt/homebrew/opt/libomp before find_package(OpenMP)). Fix: detect Homebrew libomp prefix and set OpenMP_ROOT before find_package(OpenMP); consider failing loudly when OpenMP is requested but not found. Checklist: (a) on macOS with Homebrew, check whether OpenMP was silently NOT found (grep CMakeCache for NOTFOUND), (b) set OpenMP_ROOT to the libomp prefix, (c) verify compile_commands contains fopenmp when OpenMP is requested.

## [pat-20260802015713-aaa15f]
Category: pattern
Tags: build, cmake
Changed: 2026-08-02T01:57:13.987588

CMake PARENT_SCOPE is dynamic-scoping, NOT static: set(VAR ... PARENT_SCOPE) inside a function modifies the CALLER's variable at the time the call executes. Two consecutive PARENT_SCOPE sets in the same function from different collection calls ACCUMULATE into the parent (each reads the parent's then-current value), they do NOT replace. A discovery misreading this as 'the second call overwrites the first, so BLAS dir is dropped' is a FALSE POSITIVE - empirically, the variables union (BLAS;/LAPACK both present). In hpgl-reborn: F-60 (REJECTED by adversarial, empirical CMake reproduction of the exact function: PARENT_SCOPE accumulates - BLAS;/LAPACK union; discovery misread CMake dynamic scoping; macOS uses Accelerate.framework anyway). Lesson: before filing a CMake variable-scoping finding, reproduce the exact function in a minimal CMake script - PARENT_SCOPE semantics are dynamic and accumulation is the norm. Checklist: (a) when a finding claims a CMake variable is 'dropped' or 'overwritten' by a PARENT_SCOPE set, reproduce the function in isolation first, (b) remember PARENT_SCOPE reads the parent's value at call time and accumulates across calls, (c) check the final parent value empirically before concluding data loss.

## [got-20260802015719-44782b]
Category: gotcha
Tags: numerical, cpp, geostatistics
Changed: 2026-08-02T01:57:19.252421

box-limited covariance functor truncating data-to-data pairs to 0 while RHS exact creates an inconsistent kriging system: when the LHS (data-to-data) covariance functor truncates pairs farther than the search box to 0, but the RHS (data-to-unknown) uses the exact model, the kriging system becomes inconsistent - the covariance model says data-to-data covariance is non-zero at distances where the functor reports 0, silently biasing all production kriging paths (SK/OK/SGS/SIS/IK/LVM/median-IK) at ~0.3 sill error when range a > 2r. In hpgl-reborn: F-21 (CONFIRMED MEDIUM) - precalculated_covariances_t/covariance_field_t box-limited functor truncates out-of-box data-to-data pairs to 0 while RHS exact; F-21 sibling PR-06 persists on covariance_field_t::operator() (median_ik path, LHS spans 2xradius vs RHS in-box). Fix: data-to-data pairs beyond the box must use the exact covariance (or the truncation must apply symmetrically to LHS and RHS). Checklist: (a) for any box-limited covariance functor, verify LHS and RHS use the SAME truncation rule, (b) test kriging with range > box radius and compare against exact-model reference, (c) audit both precalculated_covariances_t and covariance_field_t paths.

## [got-20260802015752-d96c7e]
Category: gotcha
Tags: ffi, cpp, error-handling
Changed: 2026-08-02T01:57:52.209522

abort()/SIGABRT in a C++ library is uncatchable by Python/ctypes: when a C++ library call aborts the process (abort(), std::terminate, assert), the Python caller gets a process death, not a catchable exception - try/except RuntimeError cannot intercept SIGABRT. Libraries called via FFI must throw catchable exceptions (hpgl_exception) instead of aborting, and validate inputs so the abort path is unreachable. In hpgl-reborn: I2-24 (CONFIRMED MEDIUM) - precalculated_covariance.h:71-75 abort() for radius >= 645 in SK/LVM paths ((2r+1)^3 > INT_MAX); Python MAX_RADIUS=1e6 allows r >= 645; contrast: clusterizer.cpp:106-108 throws catchable hpgl_exception for the analogous case. Fix: replace abort() with throw hpgl_exception (catchable -> Python RuntimeError). Checklist: (a) grep C++ for abort()/exit()/terminate in library paths reachable via FFI, (b) verify every defensive guard throws, not aborts, (c) test the boundary condition directly from Python - a SIGABRT is a crash, not an error.

## [got-20260802015758-6e1cf2]
Category: gotcha
Tags: ffi, ctypes, callback, concurrency, race, lifetime
Changed: 2026-08-02T01:57:58.280795

FFI callbacks (Python ctypes CFUNCTYPE handed to C++ as a function pointer) have two independent hazards: DEADLOCK and USE-AFTER-FREE. (1) Reentrancy/deadlock: holding a non-recursive lock (threading.Lock / non-recursive std::mutex) across a C++ call that can invoke a Python callback which re-enters the library self-deadlocks on the same thread; use threading.RLock or drop the lock before invoking callbacks, and on the C++ side never hold the handler mutex while calling the handler. RLock fixes SAME-thread re-entry only: a handler invoked from an OpenMP WORKER thread that re-enters a lock-guarded API deadlocks even with RLock (the worker does not hold the lock; the main thread waits at the OpenMP barrier). A thread-local re-entry guard is same-thread-only. Fix: serialize handler invocation so callbacks never run on worker threads, make the handler path lock-free (defer/queue), or never hold the lock across the C++ call that fires the callback. (2) Callback lifetime: when the Python CFUNCTYPE object is freed (del, reassignment, GC) CPython frees the ffi_closure/trampoline immediately; if C++ calls the stored pointer afterwards it is use-after-free/segfault. The race is between C++ copying the handler pointer (one mutex) and invoking it (another mutex). Fix: keep a strong Python reference to the callback for C++'s entire lifetime, set the C++ handler to nullptr BEFORE freeing the Python callback, and serialize handler mutation with the calls that use it. Checklist: (a) any lock around an FFI call that can invoke a Python callback must be recursive AND must not be held across the callback — test the callback path under OpenMP (multi-threaded) and verify which thread it fires on; (b) every ctypes callback passed to C++ must have documented lifetime guarantees — the Python reference must outlive ALL C++ call sites.

## [got-20260802015803-470e78]
Category: gotcha
Tags: io, security, file-writing, windows
Changed: 2026-08-02T01:58:03.494338

Safe file write/replace. (a) Temp-file writers must use O_NOFOLLOW/O_EXCL + per-writer unique names + cleanup on every failure path: a fixed-name <target>.tmp shared by concurrent writers corrupts (last rename wins) and leaks on every failure path, and fopen('w+') follows symlinks — an attacker with write access can plant <target>.tmp and redirect the C++ write through it (TOCTOU between Python validation and C++ fopen) (F-52, I2-58). Open temp files with O_NOFOLLOW|O_EXCL and unique per-writer names, and remove the temp on every failure path. (b) Windows: a file with an open handle CANNOT be renamed/replaced (MoveFileExA -> ERROR_SHARING_VIOLATION; CRT opens _SH_DENYNO, no FILE_SHARE_DELETE), so a fix that renames the temp while the FILE* is still open breaks all 4 property_writer writers (R-05); and a failure guard armed on the TARGET removes the pre-existing file (II-14). Close/flush the handle BEFORE rename in every writer, and arm the failure guard on the TEMP file, never the target. Checklist: (a) temp files opened O_NOFOLLOW|O_EXCL with unique names and removed on ALL failure paths (throw, fflush, rename); (b) confirm concurrent writers do not share one temp path; (c) close the handle before rename; error cleanup removes the temp, not the target.

## [got-20260802015809-929f31]
Category: gotcha
Tags: numerical, testing, ffi
Changed: 2026-08-02T01:58:09.474395

LAPACK reads matrices COLUMN-major (Fortran order): test data written in row-major order and read by LAPACK with uplo='U' silently tests a DIFFERENT matrix than intended. A row-major matrix that should be non-SPD can appear SPD to dpotrf (info=0) because LAPACK reads the upper triangle of the transposed layout, and the fallback-path assertion then fails unexpectedly (or worse, passes on the wrong matrix). In hpgl-reborn: PR-04 (CONFIRMED MEDIUM, both-found) - test_solver_entry_point.cpp:146,172,299 defective test data: row-major matrices read by LAPACK column-major with uplo='U' -> dpotrf succeeded (SPD) on a different matrix -> fallback assertions permanently RED; fixed with genuinely non-SPD matrices [1,3,3,1]/[1,2,2,4] read column-major. Fix: when constructing LAPACK test matrices, either write them in column-major layout or verify the matrix is genuinely non-SPD under the storage order LAPACK will read. Checklist: (a) for any LAPACK test, compute the matrix LAPACK will actually see (column-major read), (b) verify non-SPD matrices are non-SPD in that layout (dpotrf info>0), (c) run the test and confirm the expected branch actually executes.

## [got-20260802015814-729334]
Category: gotcha
Tags: build, packaging, python, api
Changed: 2026-08-02T01:58:14.559709

Packaging must ship every importable module and match the documented public API. (a) The install list (scikit-build-core/CMake install(FILES)) must include every module imported unconditionally at import time — a module added to the tree but not the install list produces a wheel that ModuleNotFoundError on import even though in-tree tests pass (F-10, PFA repeat regression). (b) scikit-build sdist include wins over exclude: exclude patterns do NOT reliably drop files a broader include pattern pulls in, so stale binaries/.pyc ship and shadow the fresh build (I2-44: sdist shipped 17 stale artifacts) — verify the actual sdist contents after build (extract and list binaries/.pyc). (c) Documented public API must be smoke-tested at the TOP-LEVEL import: names documented as public but missing from __init__ raise AttributeError while tests pass via submodule imports (gtsim_2ind, SGSConfig/SISConfig/GTSIMConfig); re-export everything documented and test 'from pkg import X' for every documented name. Checklist: (a) when adding a Python module, add it to the install list; (b) test: pip install the built wheel in a clean venv and import; (c) after any sdist build, extract and grep for binaries/.pyc; (d) smoke-test the documented public surface at the top level.

## [got-20260802015819-d37416]
Category: gotcha
Tags: security, path, validation
Changed: 2026-08-02T01:58:19.532194

Path-containment validation is defeated by a self-referential basedir: when the validator derives the base from the filename's own directory (basedir=dirname(filename)), any file trivially passes relative_to, symlink containment is defeated (the base moves with the file), and symlink escapes become undetectable — at ALL 10 production call sites (F-28); DEFAULT_BASE_DIR was dead code. Use a trusted independent base (DEFAULT_BASE_DIR or an explicitly passed root), never the file's own directory, and canonicalize with realpath BEFORE containment checks. Note: DEFAULT_BASE_DIR=cwd at import means tmp_path callers must pass an explicit basedir or CriticalValidationError fires. Checklist: (a) basedir is never derived from the checked file's directory; (b) realpath before relative_to; (c) test with a symlinked directory to confirm rejection.

## [ref-20260802033523-af9675]
Category: reference
Tags: gslib, file-format, geostatistics, interop
Changed: 2026-08-02T03:35:23.239174

GSLIB file format (Deutsch & Journel; 'GEO-EAS' family ASCII, space-delimited, Fortran list-directed read — NaN/Inf text = read error). Data files: title / nvar / nvar names / rows. Grid property files: line1=title, line2=min max, then values one per line. Ordering: X fastest, then Y, then Z; loc=(iz-1)*nx*ny+(iy-1)*nx+ix (1-based). Missing = outside the +/-1.0e21 trimming window; internal UNEST is version-dependent (-99.0 in GSLIB 2.907, 1.0e21 classic). No byte/float marker in the file; categorical = integer codes, same 2-line header. Grid output header: title / 'nvar nx ny nz' / names. Source: gslib.com format.html; references sgsim.fpp, PyGSLIB issue #24.

## [pat-20260802092553-868986]
Category: pattern
Tags: regression, fix, verification, stats-wiring
Changed: 2026-08-02T09:25:53.396614

fix-introduced regression escapes on fix-modified lines: fix agents routinely ship new defects on the exact lines they ADD or MODIFY, and their own verification misses them because it checks the fix intent, not the new code. In hpgl-reborn v2.0.1: F1 (SIS expected-count formula added by F-M6-py produced spurious 'could not be kriged' warning on every successful 2-category run - empirically 65/128), F2 (geo.py doc block modified by F-N1 left a false median_ik/IK claim - F-N1 doc-error class recurring on the same lines), F8/F9/F10 (m_mean numerator/denominator scale mismatches in NEW F-M5/F-M6 stats code), F11 (sgs expected overcounts ndmin-skips - line added by F-M6-py), F12 (F-M16 rejection over-applied to ellipse-mask path). The F-33 stats-wiring class was repeatedly half-applied across wrappers (geo.py hotspot: 6 PRIOR_FIX_ATTEMPT findings; cokriging/OK/SGS/SIS each missed the same wiring SK/LVM got). Fix: post-fix review MUST treat fix-added lines with the same adversarial rigor as pre-existing code - diff-verify every added/modified line, test the exact branch the fix touches (2-category not just 3-category SIS), and when a wiring pattern is applied to one wrapper, mechanically audit ALL sibling wrappers for the same wiring. Checklist: (a) after any fix, re-audit git diff added lines as new defects, (b) test the fix's edge-case branch (not just the nominal path), (c) for cross-layer counters (Python expected formula vs C++ per-branch eval counts), verify the counting semantics match branch-by-branch.

## [pat-20260802092606-5ce233]
Category: pattern
Tags: caps, performance, validation, complexity
Changed: 2026-08-02T09:26:06.445447

Size/work caps must bound the right quantity, at the right value, in the right language, at every entry point. (1) COUNT caps do NOT bound WORK: a count cap (max points, max neighbours, grid cells) bounds allocations, not iteration complexity — O(n^2 x lag) or window x grid loops remain effectively infinite (calc_variograms_from_point_set is O(size^2 x lag); MAX_WINDOW_VOLUME caps offsets while the inner loop iterates all grid cells). Derive the worst-case total work as a single number (pairs x lags, window x grid) and cap THAT, for EVERY entry point (grid AND point-set). (2) A work cap's CONSTANT must be validated against the real loop cost — an existing cap can under-estimate the true work by orders of magnitude and is then not a cap at all (GridStyle cap under-estimated n^2 x NumLags by 5e5-7e5x; MovingAverage cap bounded MEMORY not WORK; a cap that passes a multi-hour input is not a cap). (3) Caps mirrored across LANGUAGES must be calibrated per-language throughput, not copied: a C++ cap of 1e12 mirrored verbatim into pure Python (4-4.6e7 units/s) hangs for ~6h-108 days; derive the Python cap from a bounded worst-case runtime (guarantee parity, not constant parity) and comment the divergence. (4) Every user-controlled size/count parameter needs a HARD cap at the Python boundary AND re-enforced in C++ (a warning is not a cap); API-boundary caps must match internal algorithm limits (1e7 API vs 1e5 internal silently routes to mean-fill); caps must fire BEFORE allocation. (5) Landing a cap/limit change is itself regression-prone: enumerate the real worst-case arithmetic and test every LEGAL boundary (radius 10x, zero/edge values, coincident points, shipped book workflows) — SCAN_LIMIT/FALLBACK_WINDOW collapsed the effective search radius to ~8-10.6 cells, a clamp made lag 0 unbinable, a 100000->10000 cap broke pinned tests; 10 of 18 post-fix findings were fix-introduced from cap/normalization changes. Checklist: (a) cap the work-estimate, not each count; (b) sanity-check the cap constant against a measured worst-case runtime; (c) per-language calibration with the divergence rationale in a comment; (d) hard caps both sides, firing before allocation; (e) grep shipped workflows that exercise the class before landing a cap change.

## [got-20260802092612-329455]
Category: gotcha
Tags: documentation, docstring, cross-language, coordination
Changed: 2026-08-02T09:26:12.403923

Docstrings that misdescribe code are a repeat-regression source, especially when a fix spans layers. (a) When a docstring states a fact about wiring ('X does/does not call Y'), verify it against grep of the actual call sites before trusting it — a false reason in a doc misled 3+ fix attempts (geo.py:102-105 sentinel doc falsely claimed cokriging/median_ik/IK do not call set_kriging_stats when they DO; a fix-added covariance_field.h header comment falsely claimed entry-point guards existed; a '1D or 3D' comment false for all three siblings). After ANY fix that changes wiring, re-check the docstring it references. (b) Cross-layer findings fixed by multiple agents can end with the docstring documenting PRE-fix behavior while the code implements POST-fix behavior (R-2: the Python agent's docstring said the user mean is not applied while the parallel C++ agent made it honored). Establish a shared post-fix contract statement before agents edit, fix documentation LAST from the landed code, and verify docstrings against the actual landed behavior (grep + empirical output). Checklist: (a) docstrings describing cross-language wiring verified by grep; (b) after changing code, re-read the docstring that describes it; (c) after parallel cross-layer fixes, verify docs against landed code.

## [got-20260802092630-2eec8a]
Category: gotcha
Tags: gslib, interop, io, numerical, sentinel
Changed: 2026-08-02T09:26:30.576629

GSLIB missing-value sentinel handling must be correct on EVERY path and in the SAME float space. (a) TRIM ON ALL READ PATHS: third-party GSLIB files use the +/-1.0e21 sentinel; loading sentinels as real data silently corrupts statistics. Trimming must be applied unconditionally (isnan | < -1.0e21 | > 1.0e21) on BOTH Python and C++ read paths, including the NaN-undefined branch (F-M18/F3: LoadGslibFile kept sentinels as data; get_gslib_property masked exact equality only, while C++ read_inc_file.cpp applied the window unconditionally -> [1,1,1,1] vs [1,0,0,1]). (b) WRITE PRECISION: %E-6 (7 significant digits) does NOT round-trip float32 sentinels — a fractional undefined_value such as -99.00005 reloads slightly different, so exact-equality re-masking marks 17-30% of masked cells INFORMED (E2-06 measured 18.1%/30.7%). Write >=9 significant digits (%.9E) and/or use tolerance-aware re-masking shared across the Python slow path, the C++ fast reader, and the writer. (c) COMPUTE WINDOWS IN THE COMPARISON'S FLOAT SPACE: float32(1.0e21)=1.000000020E+21 is DATA under a float32-computed window but SENTINEL under a float64-computed one, because the boundary constant itself does not round-trip exactly (P-01, a 4-PFA repeat-regression hotspot in read_inc_file.cpp:374-380; fixed by computing the window in float64 and comparing promoted doubles). Also validate GSLIB property names before ANY header write (reject newline, '--', leading whitespace, empty) — raw names are header/row injection. Checklist: (a) grep 1.0e21/1e21 read paths — sentinel trim must be unconditional, not exact-equality; (b) every header write validates property names first; (c) Python and C++ apply the SAME sentinel window; (d) write >=9 sig digits and pin exact-edge behavior with a cross-stack parity test; (e) compute min/max windows in the same float type as the compared value.

## [got-20260802092650-41b872]
Category: gotcha
Tags: regression, sibling, propagation, caps
Changed: 2026-08-02T09:26:50.038509

A guard/fix applied to ONE path silently leaves the same defect in sibling paths. Fix agents touch only the discovered instance: an INT_MAX/radius guard added to precalculated_covariances_t but not covariance_field_t/calc_cov_field/OK-default/median_ik/cokriging; a size_t index fix landed in solver_entry_point.h but not the gauss_solve fallback; an O_NOFOLLOW hardening on the write side but not the read side; an isinstance tuple fix on one variogram function but not its sibling. Fix: when fixing a CLASS of bug (overflow, guard, cap, hardening), grep the codebase for ALL instances of the same pattern — sibling functions, fallback paths, symmetric read/write sides — and fix every one in the same change, verifying each call site reaches the fixed path. Checklist: (a) after any fix, grep for the same arithmetic/guard pattern across siblings and fallbacks; (b) symmetric operations (read/write, entry/fallback, LHS/RHS) all hardened; (c) for any (2r+1)^n-style allocation, enumerate every constructor/call site and apply the identical threshold — a guard in one sibling is not a guard.

## [pat-20260802223230-f97aae]
Category: pattern
Tags: build, packaging, verification, macos
Changed: 2026-08-02T22:32:30.467186

a verification gate only catches what it actually inspects - a gate that checks LC_LOAD_DYLIB deps (otool -L) but never LC_RPATH/LC_BUILD_VERSION (otool -l) cannot catch absolute-path leaks or min-OS mismatches; a gate that checks only symbol presence (smoke test checks _HAS_KRIGING_STATS, never loads _cvariogram) cannot catch the defect class it was built to prevent; a build gate that exits 0 on smoke failure is not a gate. In hpgl-reborn v2.0.2: R-23 (wheel tagged macosx_11_0 embeds libomp.a objects compiled with LC_BUILD_VERSION minos 15.0 - gate checks neither dylib min-OS nor archive min-OS), R-24 (gate runs otool -L only, never otool -l - LC_RPATH/LC_BUILD_VERSION unchecked, N4 fold-in regressible silently), R-26 (non-wheel smoke gate has no relocatability check - HPGL_STATIC_LIBOMP=OFF stale cache deploys with absolute libomp dep and the gate passes), 2-M-22 (build.sh implicit exit 0 / build.bat exit /b 0 on smoke failure), 2-M-23/R-22 (INSTALL_RPATH and exported hpglTargets.cmake bake absolute build-machine BLAS/include paths - gate does not scan .cmake files). Fix: the gate must assert the exact artifact property that defines the defect class (LC_RPATH, LC_BUILD_VERSION, absolute paths in dylibs AND exported .cmake files), and gate failures must be non-zero exits. Checklist: (a) list the artifact properties the defect class depends on and verify the gate inspects each one, (b) a smoke gate must load the actual module/symbol it claims to verify, (c) build gates must exit non-zero on any smoke failure, (d) scan exported .cmake files for absolute build-machine paths.

## [pat-20260802223236-e889f3]
Category: pattern
Tags: ffi, cpp, validation, boundary
Changed: 2026-08-02T22:32:36.646439

Every entry point that can be reached directly must replicate the validation of the layer above. (a) C API entry points must validate what their Python wrappers validate — direct C callers bypass Python validation, so unvalidated enum values, zero/negative bounds, and out-of-range marginal probabilities silently produce wrong output instead of errors (M-28 min>max -> every node skipped, fully-unsimulated output; M-31 max_neighbours=0 -> all mean-fill/undefined; 2-M-4 kriging_kind != simple -> routes to ordinary; 2-M-35 marginal[1]=5.0 -> all-1/all-0; 2-M-2 unbounded neighbours_count). (b) Direct C++ public entry points (hpgl_core.h) must validate what the C API gates — unchecked secondary/grid sizes -> abort()/SIGABRT or silent degradation (E-M61 cokriging, E-M62 mean NaN, E2-109 SGS-LVM mean_data OOB read, E2-118 SIS-LVM rows, default-ctor invariant traps). Validate at the C++ entry itself, not only in the Python/C wrapper. Checklist: (a) for each C API function, mirror its Python wrapper's validation at the C boundary; (b) for every validating C API, check whether the underlying C++ function is separately callable and gate it too; (c) test direct-C/C++ callers with the same adversarial inputs as Python; (d) default constructors initialize invariants.

## [pat-20260802223242-44d3e0]
Category: pattern
Tags: testing, regression, verification
Changed: 2026-08-02T22:32:42.509108

A regression test must FAIL on pre-fix code — otherwise it is not a regression test. (1) Behaviorally-inert fixes: a guard nested inside a condition that always holds never alters a branch, and the paired test passes pre- and post-fix (R-3 ndmin gate mathematically unreachable because original_data_count <= ws.indices.size() always). Prove the fix changes observable behavior before writing the test. (2) Tests asserting the BROKEN behavior become regression masks: a test that pins the current buggy behavior (e.g. asserts geo._last_kriging_stats is None after a SUCCESSFUL run) keeps the suite green while the defect ships and actively blocks the fix; flip every such test in the SAME change and grep for tests asserting None/stale/unchanged state a fix should populate. (3) Vacuous fixtures/assertions: an all-ones mask with binary data makes output==input and proportion assertions tautological — use partially-masked fixtures. (4) Numerical fixes without DIRECT tests: integration tests can mask the specific behavior through upstream guards/downstream fallbacks — add at least one unit test at the same language level that exercises the exact failure input and expected output and would FAIL if the fix were reverted. (5) Proposed additions can be bool-invariant OR permanently-red: an addition whose asserted gate fires identically with and without the code under test (a magnitude gate !isfinite catching NaN-in-X both with and without the scan; a no-op test-build macro such as HPGL_LOG_STRING) can never fail — it is not a regression test; a construction whose key conjunct is structurally dead (indefinite-matrix wild solutions always have B'X<0, so a Schur-consistency conjunct is always false) can never pass. Before writing a test, prove it discriminates — enumerate the code states it must distinguish and confirm the assertion flips between them; check for no-op macros and structurally-dead conjuncts. Checklist: (a) run any new test against pre-fix code — passing means vacuous; (b) assertions must flip between code states (bool-invariant = useless); (c) verify the guard algebra can actually alter the branch decision (no subset/superset implication); (d) regression tests exercise the pathological path, not the nominal one; (e) prove the discriminating conjunct is satisfiable in both directions (can pass AND can fail) and grep for no-op test-build macros before relying on log/status output.

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
Tags: ffi, python, numpy, mutation, validation
Changed: 2026-08-02T22:33:06.748138

Do not corrupt caller-owned data — and never mutate before validation. (a) numpy ravel()/ravel('K') return a VIEW when contiguous; in-place writes through the view mutate the CALLER's buffer even if the wrapper later restores its own attribute (prop.data = copy hides the corruption but the caller's aliased array is already destroyed) (M-24/M-25). Copy inputs BEFORE any in-place write when the contract promises non-mutation. (b) Validate BEFORE destructive in-place mutation: SGS-LVM performed the forward-CDF transform in-place on the caller's conditioning data BEFORE the radius-guard throw, so a failed run destroyed the direct-C caller's input (E2-112; Python protected only because it clones). Order all throwing validation/allocation BEFORE the first in-place write; when a function mutates its input, document it or operate on a private copy. (c) Invariants enforced only at construction are broken by public property setters: setters (prop.mask=..., prop.data=...) reassign via numpy.require+contiguity checks with NO shape-match validation while the FFI layer checks data.size but never mask.size, so C++ indexing mask by data size is a heap OOB write (2-M-15); setters must re-validate cross-array invariants and FFI constructors must size-check EVERY buffer. Checklist: (a) any wrapper writing into an input array operates on a private copy, not a view — restoring an attribute after corrupting a shared view is not a fix; (b) all throws precede the first in-place write — test the failing path leaves caller data intact; (c) paired-buffer APIs re-validate the shape invariant on setters, and FFI validates every array C++ indexes.

## [pat-20260802223312-d44c2d]
Category: pattern
Tags: numerical, reproducibility, cpp, python
Changed: 2026-08-02T22:33:12.335815

Every randomized path must be seed-controlled for reproducibility, and the full seed width must survive. (1) A thread_local RNG seeded from random_device()^time makes identical inputs produce different outputs every run; a seed_rand_once() that is a no-op silently defeats the intent; every randomized path must accept an explicit seed and be deterministic for it — entropy seeding belongs only at documented non-reproducible public entry points (2-M-3: grid-path variogram sampling missed while every other randomized path was seed-controlled). (2) 64-bit seeds are silently truncated mod 2^32 when static_cast to a 32-bit mt19937 state — distinct seeds (5 and 2^32+5) give bit-identical output (II-15; the prior 'fix' 05bcec5 introduced it); ctypes additionally wraps mod 2^64 (III-02). Preserve the full width (split into two 32-bit words for mt19937 seeding) and validate range before conversion. Checklist: (a) grep thread_local RNG + random_device/time seeding; (b) every randomized path exposes a seed and is reproducible for a fixed seed, with a determinism test; (c) verify seeds differing only above 2^32 produce different streams.

## [pat-20260802223421-6e2873]
Category: pattern
Tags: numerical, geostatistics, python, cpp, verification
Changed: 2026-08-02T22:34:21.063029

Parallel implementations of the same algorithm must agree on the METRIC and all binning conventions, not just the API signature. (1) Metric: a Python fallback that bins by raw Euclidean distance while the C++ kernels bin by projection onto the anisotropy axis produces different curves for identical inputs (2-M-13: lag-3 4.5000 vs 3.6937). (2) Self-pair handling: the grid kernel counted zero-distance self-pairs into lag 0 while the point-set skips idx1==idx2 (E-M74, nugget diluted on the grid path); the Python CubeScan still bins the (0,0,0) offset into lag 0 (5*last-cell-count zero-var self-pairs), value-equivalent so a value-based pin passes, but count-sensitive comparisons and FirstLagDistance>0 templates diverge. (3) Interval convention: grid closed intervals double-count exact band-end distances while point-set/Python are half-open (E-M75). (4) Hemisphere/overlap handling and accumulation precision (float32 vs double; E-M10) diverge too. (5) Exact-equality lag binning drops legitimate pairs at fractional coordinate spacing (0.5-m spacing -> zero pairs; III-15) — use tolerance-based matching. Fix: document and match the metric + self-pair policy + interval convention + hemisphere handling + accumulation precision, and add cross-language equivalence tests. Checklist: (a) grep each kernel's binning metric; the Python fallback must match the C++ reference; (b) compare self-pair filters and interval boundaries across twins and test band-end and zero-offset distances; (c) use tolerance, never exact float equality, in distance binning.

## [got-20260803001248-1d7a62]
Category: gotcha
Tags: gslib, sgsim, geostatistics
Changed: 2026-08-03T00:12:48.411382

GSLIB sgsim reference semantics: ndmin counts ONLY original hard data (nclose from srchsupr); previously-simulated nodes (ncnode) NEVER count. nclose<ndmin => the node is SKIPPED (written as UNEST=-99.0), NOT an N(mean,1) fallback. The OK->SK downgrade applies at (nclose+ncnode).lt.4 (fewer than 4 total data); the N(mean,1.0) fallback (cmean=gmean, cstdev=1.0) fires ONLY when there is zero data or a singular kriging matrix. Checklist: (a) count only original hard data toward ndmin, never previously-simulated nodes; (b) a below-ndmin node is skipped (UNEST), not mean-filled; (c) the mean fallback is reserved for zero-data/singular cases.

## [pat-20260803001248-62e195]
Category: pattern
Tags: gslib, sisim, ordrel, numerical
Changed: 2026-08-03T00:12:48.619961

ORDREL order-relation corrections (GSLIB ordrel.for): clamp indicator probabilities to [0,1]; continuous case: upward correction then downward correction then average ccdfo(i)=0.5*(ccdf1(i)+ccdf2(i)) -> monotonic non-decreasing; categorical: renormalize to sum 1. CRITICAL: when the per-category PMF sums to MORE than 1 (S>1, heterogeneous per-category covariances), clamping each probability to [0,1] and then scaling the top to 1.0 is a NO-OP (clamp pins probs.back()==1.0), so excess mass is silently TRUNCATED onto earlier categories instead of GSLIB's divide-by-total (sumcdf) — causing a silent category-selection FLIP (E-M56: p=[0.6,0.7,0.5] S=1.8 -> HPGL argmax 0 vs GSLIB argmax 1; reachable with per-category covariance models and >=3 categories). 'Renormalize to sum 1' must be implemented as divide-by-total, never clamp+scale-top. Checklist: (a) implement S>1 as dividing every category by the total; (b) clamp-first defeats renormalization whenever S>1; (c) test PMFs whose sum exceeds 1.

## [ref-20260803001248-026bbf]
Category: reference
Tags: gslib
Changed: 2026-08-03T00:12:48.689707

GSLIB 2.0/2.907 original Fortran sources mirrored at github.com/exepulveda/gslib2.0 (src/original/{sgsim,sisim,gtsim,gslib}/*.for) — authoritative reference for sgsim/sisim/gtsim semantics.

## [got-20260803180147-394516]
Category: gotcha
Tags: ffi, memory, python, cpp, validation
Changed: 2026-08-03T18:01:47.795269

FFI output-buffer contract: contiguity, size, AND writeability (and non-aliasing) must all be enforced before a C call writes into a caller buffer. (a) CONTIGUITY: numpy sliced/non-contiguous views passed to C++ flat-indexed loops cause heap OOB writes (F-28 negative-stride probe; F-06/F-08 shape+contiguity) — enforce np.ascontiguousarray. (b) SIZE: caller output buffers are not validated against the write volume (too-small buffers silently truncate; missing data.size==mask.size causes heap OOB read; an FFI constructor can stamp grid dims into a mask defeating downstream guards); validate EVERY dimension, not just the volume. C++ must also initialize result buffers — top/tail cells left unwritten cause stale-cell corruption on buffer reuse. (c) WRITEABILITY: a writeable=False buffer is silently mutated and a read-only mmap SIGBUSes uncatchably; check flags WRITEABLE. (d) ALIASING: the 7 kriging C entries lacked in/out buffer-identity guards (zero identity checks), so an aliased buffer (same array input and output) causes progressive overwrite and an OpenMP data race on IK (E2-53); add in != out guards. Checklist: (a) before every FFI call verify contiguity, exact per-dim size, writeability, and non-aliasing; (b) C++ writes every element or explicitly initializes; (c) test sliced views, undersized buffers, read-only buffers, and deliberately aliased buffers.

## [got-20260803180158-e9ddad]
Category: gotcha
Tags: numerical, geostatistics, python, cpp
Changed: 2026-08-03T18:01:58.635479

data-space vs normal-score-space mixing in transforms silently corrupts results: when a pipeline mixes CDF/normal-score transforms, every comparison must happen in ONE space. In hpgl-reborn: F-02 (GTSIM truncation compares data-space output against normal-score thresholds - category proportions grossly wrong, 0.50->1.000), II-39 (SGS output is DATA space, not standard-normal - the back-transform exists at :142-146; non-default tk params silently distort proportions and the F-02 fix alone does NOT repair it), III-37 (scalar stationary mean=50 NOT CDF-transformed while the LVM mean IS - sparse cells pinned to CDF max datum). Fix: document each output's space (data vs standard-normal), transform every input to the same space before comparison/truncation, and add a round-trip test. Checklist: (a) for any truncation/back-transform, verify the compared quantities are in the same space, (b) sibling branches (scalar vs LVM mean) must apply the same transform, (c) round-trip tests must cover the max datum.

## [got-20260803180159-7eedaa]
Category: gotcha
Tags: python, packaging, compatibility, version
Changed: 2026-08-03T18:01:59.836276

Version/compatibility gates. (a) __version__ derived from INSTALLED distribution metadata can be stale relative to the source tree: a stale hpgl-1.6.0.dist-info made __init__ report 1.6.0 while source/wheel were 2.0.2 (II-28), and the guard test was tautological (checked the installed dist, not the source). Derive __version__ from the source and make the gate compare source vs installed metadata so a stale dist-info cannot change the reported version. (b) PEP 604 union annotations (X | Y) executed at module scope break Python <3.10 unless 'from __future__ import annotations' is present (III-24: geo.py:163 raised TypeError on import on 3.9; survived 4 passes because build/tests run on 3.13). Declared support with no old-version test catches nothing — add the future import or avoid PEP 604 at module scope, and add a lowest-supported-version import smoke test. (c) numpy >=2.4 raises ValueError when a 1-element array is assigned to a scalar slot (arr[0:1] into a scalar) where it previously deprecated (F-09/F-17); index with arr[0] and test on the newest supported numpy. Checklist: (a) verify __version__ equals the source version in a clean venv; (b) every module with module-scope PEP 604 has the future import, and a 3.9 import smoke test exists; (c) grep [0:1]/shape-(1,) slices assigned to scalar slots.

## [got-20260803180210-03bda0]
Category: gotcha
Tags: concurrency, io, ffi, locale
Changed: 2026-08-03T18:02:10.793826

non-glibc setlocale race: on macOS (de_DE) concurrent I/O wrappers that call setlocale while another thread parses with sscanf corrupt the parse (fractional parts dropped 1.5->1.0) AND permanently corrupt the process locale; glibc's setlocale is thread-safe but other platforms are not. The C++ locale_keeper mutex only guards the setlocale calls, not the parse window (II-35, live race reproduced). Fix: serialize the full parse window (setlocale + sscanf + restore) under one lock, or avoid setlocale entirely (use a locale-independent parse). Checklist: (a) grep setlocale and sscanf/%f parse sites, (b) verify the lock covers parse AND restore, not just the setlocale call, (c) test under a non-glibc locale (de_DE) with concurrent calls.

## [got-20260803180214-392ca5]
Category: gotcha
Tags: numerical, lapack, solver, validation, cpp
Changed: 2026-08-03T18:02:14.517186

Numerical solver success/validation gates must be meaningful, scale-invariant, and path-aware. (1) LAPACK INFO=0 does NOT mean trustworthy: a near-singular SPD matrix can pass dpotrf_/dpotrs_ with INFO=0 and return wild weights (1.0e12, -1.0e12) reported as success — exactly-singular is caught, near-singular is not (II-09); add weight-magnitude/condition validation on the success path. (2) Magnitude gates must be SCALE-INVARIANT and path-aware: a raw |X| bound is scale-dependent (X=A^-1*b scales with data/sill) and rejected legal inputs across 5 regression attempts (E-M87 absolute gate -> R-01 AND-form -> R3-01/R3-02 dynamic-range -> R4-01 path-aware) — small-sill OK systems (sill ~1e-4 sanctioned by MIN_SILL=1e-6), cokriging sigma ratios >1e3, correlogram sigma_c>sigma_0. Measure a scale-invariant quantity (final weight combination w=X0-mu*X1, or normalize by matrix scale) and pass the TRUE target variance as a parameter instead of inferring A_orig[0]. (3) A consistency/variance check's reference must be PATH-AWARE: inferring it from A_orig[0] is correct only when A[0][0] equals the actual target variance (true for SK/cokriging), FALSE on the sigma-scaled correlogram path where A[i][j]=cov*(sigmas[i]*sigmas[j]) — spurious rejection whenever sigmac>sigma0 (R4-01). A gate that fires on API-legal inputs is a regression, not a guard. Checklist: (a) after every successful LAPACK solve, validate magnitude/condition, and test near-singular (not just exactly-singular) matrices; (b) measure scale-invariant quantities and pass the target variance explicitly; (c) never infer a reference from a matrix diagonal that scaling may have transformed.

## [got-20260803180215-d64b36]
Category: gotcha
Tags: numerical, validation, cpp, python
Changed: 2026-08-03T18:02:15.831913

zero/empty boundary values silently switch an algorithm to a DIFFERENT branch: max_neighbours=0 (legal 'unconditional simulation') silently becomes 1-neighbour conditioned kriging (II-13, pure-nugget fallback fires on count==0, live probe mean 4.7493 vs unconditional 0); thickness==0.0 routes to the erosion branch and blanks an ENTIRE column (II-57, zero-layer-first -> blank_value, C++ probe; PRIOR_FIX_ATTEMPT 3ad77ee missed the zero boundary). Fix: explicitly handle zero/empty boundary values (raise or implement the documented semantics), never let them fall through to an unrelated branch. Checklist: (a) for every branch selected by a magnitude test (>=0 vs >0), test the exact zero input, (b) document and validate zero-value semantics at the API boundary, (c) regression-test the zero case, not just positive values.

## [got-20260803180219-d413b7]
Category: gotcha
Tags: build, cmake, windows, msvc, security
Changed: 2026-08-03T18:02:19.562243

Windows/MSVC build configuration. (1) multi-config CMake generators (Visual Studio, Xcode) IGNORE CMAKE_BUILD_TYPE — a preset setting CMAKE_BUILD_TYPE=Release with no configuration member and no --config silently builds Debug (no /sdl, no /guard:cf; the prior 'fix' e791f6b that removed --config caused the regression); pass --config Release explicitly or add a configuration member, and verify the built binary's flags. (2) The primary Windows build path (MSVC vcxproj via build.bat) historically had ZERO security hardening flags while the CMake paths applied /sdl, /guard:cf for non-Debug; both vcxproj files also leaked debug symbols (GenerateDebugInformation=true) in Release. Checklist: (a) verify the actual generator configuration from the build command, not CMAKE_BUILD_TYPE — removing --config under a multi-config generator is a regression; (b) confirm hardening/optimization flags and no debug-symbol leak in the shipped Release binary.

## [pat-20260803180224-d58501]
Category: pattern
Tags: testing, verification, harness
Changed: 2026-08-03T18:02:24.800155

Test-harness false-passes. (1) Sample-script tests must EXECUTE the scripts, not assert on source strings or importability — string/import checks let scripts with stale file references ship broken (F-53/F-54: >=9 sample scripts stale; 8 crash post-fix on '..'). (2) C-ABI status/message buffers are not cleared on success, so adjacent sub-tests that check for a substring pass against the PREVIOUS failure's residue (T-23/T-24) — message-gate tests need DISTINCT substrings per gate or snapshot-before/assert-changed, and the C ABI should clear the buffer on success. (3) Mutual-deletion race: parallel audit agents each deleting a coverage class citing the OTHER's file as surviving -> both deletions land and coverage is lost unpinned (E-02); re-verify sibling-survival citations against the FINAL merged tree and serialize deletions that cite cross-agent survivors. (4) FFI integration tests that call the raw C binding instead of the public Python API structurally bypass the wrapper layer (state wiring, validation, transforms) and pass even if the wiring is deleted (UF-06); integration tests must call the public API and assert on wrapper-managed state. Checklist: (a) any sample-script test that does not execute the script is incomplete; (b) grep sibling gates for substring overlap and clear/snapshot message buffers; (c) verify cross-agent survival citations post-merge; (d) tests of wrapper integration must exercise the wrapper, not the binding.

## [pat-20260806182549-f8b7ea]
Category: pattern
Tags: testing, fixtures, vacuity
Changed: 2026-08-06T18:25:49.848723

Test fixtures and helpers can silently defeat themselves. (1) Partially-masked fixtures: an all-ones mask makes hard_cells==every cell, so restore/truncation assertions become tautologies (R-04, B-02, B-06, B-07) — use binary hard + sentinel-masked cells so restore pins only informed cells. (2) A fixture helper's parameter can LIE about arming: a parameter named informed_frac=1.0 may still mask cells because helper math clamps independently (T-08: masked index 0; values <0.5 impossible) — verify the parameter-to-fixture mapping with a discriminator (assert exactly 0 masked cells for informed_frac=1.0) and audit helper ranges for clamping. (3) Simulation OUTPUT masks are all-ones: C++ set_at writes m_mask=1 for every simulated node, so after SGS/SIS the result mask is ALL-ONES regardless of input arming — tests sampling result.mask==0 select nothing (N2-02/N2-03); snapshot the ORIGINAL input mask before simulation and sample uninformed cells from it. (4) Bare except Exception: pass in test/production code silently masks regressions — use specific exception types. Checklist: (a) proportion/truncation assertions must use fixtures with masked cells; (b) test fixture-helper extreme parameter values; (c) sample uninformed cells from a pre-simulation input-mask snapshot; (d) no bare except in test or production paths.

## [got-20260806182551-a330e5]
Category: gotcha
Tags: testing, python, performance
Changed: 2026-08-06T18:25:51.766531

stale .benchmarks baselines cause deterministic-looking benchmark failures: committed/checked-in performance baselines (June-16) vs post-fix kriging perf triggered a 3x regression check on ok_medium_grid baseline 0.0131s (hpgl-reborn B-04 CONFIRMED) - the failure looks real but is pure calibration staleness. Fix: delete .benchmarks/ before any test run so fresh calibration happens (first run only checks the hard floor), and never treat benchmark-check failures against stale baselines as code regressions. Checklist: (a) before perf test runs, verify .benchmarks/ baselines are fresh, (b) treat benchmark failures as stale-baseline artifacts until baselines are regenerated, (c) consider a hard floor only for first-run calibration.

## [got-20260806182643-5ae68d]
Category: gotcha
Tags: io, parsing, cpp
Changed: 2026-08-06T18:26:43.916361

Numeric parsing must consume the ENTIRE token and handle buffer boundaries. (a) sscanf/atof accept numeric prefixes: sscanf('5/','%f') and atof('1.5abc') succeed with partial parsing (no %n/endptr check), so corrupt tokens load as data while the Python slow parser raises (E-M73) — use strtod/sscanf with %n/endptr and verify no trailing non-whitespace, and make fast/slow parsers reject the same token set. (b) Fixed-buffer tokenizers must preserve straddling tokens: the read_inc_file token_stream_t 511-char boundary split '99' into 9.0 with rc=0 and no fallback (silent wrong data, worse than the pre-fix loud error; the token-count check is defeated because the split counts as two). Reassemble across refills preserving leading bytes and excluding delimiter/trailing bytes in both text and byte paths; bound token length before materialization. Checklist: (a) grep sscanf/atof/strtod for %n/endptr full-consumption checks and test trailing garbage; (b) test a token exactly spanning a refill boundary; (c) token-count checks must survive boundary splits.

## [got-20260806182646-af8fa9]
Category: gotcha
Tags: numerical, python, numpy, cpp
Changed: 2026-08-06T18:26:46.517097

integer-dtype accumulation truncates fractional means: computing a mean into an int/uint array (numpy) silently truncates fractional results to 0 (hpgl-reborn E2-26: MovingAverage3D int cube mean 0.63 -> 0 on both paths; also the E-M10 float32-before-squaring parity divergence where per-pair variance cast to float32 before squaring vs C++ double accumulation gives ~0.2% data-dependent divergence). Fix: upcast accumulators (float64) before arithmetic and cast only at the output, and match the reference implementation's accumulation precision exactly; check for dtype on every reduction target. Checklist: (a) grep reductions assigned into int/uint arrays, (b) verify float accumulators are float64 (or the same as the reference), (c) test fractional inputs, (d) cross-language parity requires identical accumulation precision, not just identical formulas.

## [got-20260806182713-f5d42e]
Category: gotcha
Tags: cpp, numerical, integer-width, overflow
Changed: 2026-08-06T18:27:13.391053

Integer width errors in indexing/counting cause OOB writes and hangs. (a) Loop-counter type must cover the count's range: an unsigned-char loop index wraps at 256 (E-M54 pretty_printer.cpp:103 loops indicator_index_t against size_t m_category_count -> infinite 0-255-0 hang); use the same integer type as the count, and when a type is deliberately narrow enforce the cap at EVERY entry point, not only the C API. (b) Index arithmetic i*matrix_size+j in signed int overflows once matrix_size^2 > INT_MAX (~46340^2); allocation may use a safe size_t path while indexing uses signed int — the mismatch (safe alloc + unsafe index) produces a heap OOB write (I2-23); use size_t/checked arithmetic for index computation, verify allocation and indexing use the SAME integer width, pre-scan size*size in 64-bit, and add an upper bound so overflow is unreachable. Checklist: (a) grep loop counters narrower than the compared count; (b) grep i*size+j index patterns computed in int and confirm the same width as allocation; (c) test boundary counts (256, 46340).

## [dis-20260810004209-b6adb2]
Category: discovery
Tags: testing, fixtures, slow, python
Changed: 2026-08-10T00:42:09.085669

BIG_SOFT_DATA big fixtures are 468,120 cells = 166×141×20, NOT 160×141×20 as the filenames imply: BIG_SOFT_DATA_160_141_20.INC and BIG_SOFT_DATA_CON_160_141_20.INC (both ~2MB) contain a 166×141×20 property (166*141*20 = 468,120; 160*141*20 = 451,200); any analysis computing geometry from the filename mis-sizes the grid. These fixtures are the top machine-freeze-risk class in the test suite — H-02 (AGENTS.md MANDATORY) requires @pytest.mark.slow on every test loading them; all legacy big-fixture tests and R-06 load cases now carry @slow (17 slow markers across 6 files: test_legacy_migrated 11, test_performance 2, test_cdf_edge_cases 1, test_edge_cases 1, test_memory_leaks 1, test_production_fixes_geo 1), excluding 468K-cell loads from the default suite. Checklist: (a) always verify fixture geometry from the data, not the filename, (b) any test loading a big fixture must be @slow-marked and run last/alone, (c) the default suite must exclude 468K-cell loads entirely.

