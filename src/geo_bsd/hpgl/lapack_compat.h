#ifndef LAPACK_COMPAT_H
#define LAPACK_COMPAT_H

// LAPACK Compatibility Header for HPGL
// Bridges CLAPACK interface to Intel MKL
// Supports Intel MKL, OpenBLAS, and other LAPACK implementations

// F-414: the integer width is a property of the LINKED library, and no compile
// time macro in this repository observes it — CMakeLists.txt never defines
// OPENBLAS_USE64BITINT / BLAS64 / LAPACK_ILP64 / MKL_ILP64 (grep: 0 hits), and
// neither FindBLAS nor FindLAPACK propagates the width CMake resolved
// (BLA_SIZEOF_INTEGER / BLA_VENDOR=Intel10_64ilp are user inputs). The selection
// below therefore reads macros the BUILD must set; a build that links an ILP64
// library without defining one of them declares 4-byte prototypes against
// 8-byte entries and corrupts the stack at the first solve. The build-side
// propagation is the owning fix (CMakeLists.txt) — this header only refuses to
// guess where it cannot express the convention.
//
// Detect ILP64 vs LP64 interface for LAPACK
// ILP64 uses 64-bit integers (long long), LP64 uses 32-bit integers (int)
#if (defined(_WIN32) || defined(_WIN64)) && (defined(USE_INTEL_MKL) || defined(HPGL_USE_MKL))
    // Windows: Intel MKL uses MKL_INT which is typically int
    #include <mkl_lapack.h>
    #include <mkl_types.h>

    // Check for ILP64 variant of MKL
    #ifdef MKL_ILP64
        // MKL ILP64 uses 64-bit integers
        typedef MKL_INT integer;
        // MKL_INT is fixed by the MKL headers themselves when MKL_ILP64 is
        // defined, so this is a self-consistency check on the header contract
        // (it does NOT observe the linked library's interface).
        static_assert(sizeof(MKL_INT) == 8, "MKL ILP64 requires 64-bit integer type");
    #else
        // Standard MKL uses 32-bit integers (LP64)
        typedef MKL_INT integer;
        static_assert(sizeof(MKL_INT) == 4, "MKL LP64 expects 32-bit integer type");
    #endif
#else
    // Linux/macOS: Standard LAPACK
    // Try to detect ILP64 at compile time via common macros
    #if defined(OPENBLAS_USE64BITINT) || defined(ACCELERATE_LAPACK_ILP64) || \
        defined(BLAS64) || defined(LAPACK_ILP64) || defined(MKL_ILP64)
        // ILP64 interface selected - use 64-bit integers
        #include <cstdint>
        typedef int64_t lapack_int;
        typedef int64_t integer;

        // F-414: the entry-point NAMES differ per vendor under ILP64, so the
        // macro that selected this branch also selects the symbols:
        //   - OpenBLAS (USE_64BITINT=1) exports SUFFIXED names: dpotrf_64_.
        //   - MKL selects a separate interface library (libmkl_intel_ilp64)
        //     under the SAME unsuffixed names: dpotrf_. BLAS64 / LAPACK_ILP64 /
        //     MKL_ILP64 below therefore keep the unsuffixed prototypes.
        //   - Accelerate's ILP64 entries carry an asm-level suffix
        //     (dpotrf$NEWLAPACK$ILP64) that C prototypes cannot express.
        #if defined(ACCELERATE_LAPACK_ILP64)
            #error "HPGL: the Accelerate ILP64 LAPACK interface uses asm-suffixed entry points (dpotrf$NEWLAPACK$ILP64) that lapack_compat.h cannot bind. Link an LP64 Accelerate, or add explicit __asm__ labels before enabling ACCELERATE_LAPACK_ILP64."
        #elif defined(OPENBLAS_USE64BITINT)
            extern "C" {
                // LAPACK Cholesky decomposition - double precision (OpenBLAS ILP64)
                void dpotrf_64_(const char* uplo, const int64_t* n, double* a,
                                const int64_t* lda, int64_t* info);

                // LAPACK Cholesky solver - double precision (OpenBLAS ILP64)
                void dpotrs_64_(const char* uplo, const int64_t* n, const int64_t* nrhs,
                                const double* a, const int64_t* lda, double* b,
                                const int64_t* ldb, int64_t* info);
            }
        #else
            extern "C" {
                // LAPACK Cholesky decomposition - double precision (ILP64)
                void dpotrf_(const char* uplo, const int64_t* n, double* a,
                             const int64_t* lda, int64_t* info);

                // LAPACK Cholesky solver - double precision (ILP64)
                void dpotrs_(const char* uplo, const int64_t* n, const int64_t* nrhs,
                             const double* a, const int64_t* lda, double* b,
                             const int64_t* ldb, int64_t* info);
            }
        #endif

        // Self-consistency check on the typedef this header just set — NOT a
        // check of the linked library's ABI (F-414: the previous comment
        // implied detection).
        static_assert(sizeof(integer) == 8, "ILP64 LAPACK requires 64-bit integer type");
    #else
        // Default LP64 interface - use 32-bit integers
        // No ILP64 macro was defined. This header cannot see the linked
        // library's integer width, so it assumes LP64. If you are linking an
        // ILP64 library you must define the macro matching its SYMBOL
        // convention: OPENBLAS_USE64BITINT (OpenBLAS, suffixed dpotrf_64_),
        // or BLAS64 / LAPACK_ILP64 / MKL_ILP64 (unsuffixed ILP64 interface,
        // e.g. MKL's libmkl_intel_ilp64). Defining OPENBLAS_USE64BITINT
        // against a library that exports the unsuffixed names (or vice versa)
        // produces a link error, not a silent width mismatch.
        #pragma message("HPGL: using LP64 BLAS (32-bit Fortran integers). If linking an ILP64 library, define the macro for ITS symbol convention: OPENBLAS_USE64BITINT for OpenBLAS (dpotrf_64_), or BLAS64/LAPACK_ILP64/MKL_ILP64 for an unsuffixed ILP64 interface.")
        typedef int lapack_int;
        typedef int integer;

        extern "C" {
            // LAPACK Cholesky decomposition - double precision (LP64)
            void dpotrf_(const char* uplo, const int* n, double* a,
                         const int* lda, int* info);

            // LAPACK Cholesky solver - double precision (LP64)
            void dpotrs_(const char* uplo, const int* n, const int* nrhs,
                         const double* a, const int* lda, double* b,
                         const int* ldb, int* info);
        }

        // Self-consistency check on the typedef above — not ABI detection
        // (F-414; see the ILP64 branch's identical note).
        static_assert(sizeof(integer) == 4, "LP64 LAPACK uses 32-bit integer type");
    #endif
#endif

#endif // LAPACK_COMPAT_H
