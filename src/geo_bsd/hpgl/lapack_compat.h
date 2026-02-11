#ifndef LAPACK_COMPAT_H
#define LAPACK_COMPAT_H

// LAPACK Compatibility Header for HPGL
// Bridges CLAPACK interface to Intel MKL
// Supports Intel MKL, OpenBLAS, and other LAPACK implementations

// Detect ILP64 vs LP64 interface for LAPACK
// ILP64 uses 64-bit integers (long long), LP64 uses 32-bit integers (int)
#if defined(_WIN32) || defined(_WIN64)
    // Windows: Intel MKL uses MKL_INT which is typically int
    #include <mkl_lapack.h>
    #include <mkl_types.h>

    // Check for ILP64 variant of MKL
    #ifdef MKL_ILP64
        // MKL ILP64 uses 64-bit integers
        typedef MKL_INT integer;
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
        defined(BLAS64) || defined(LAPACK_ILP64)
        // ILP64 interface detected - use 64-bit integers
        #include <cstdint>
        typedef int64_t lapack_int;
        typedef int64_t integer;

        extern "C" {
            // LAPACK Cholesky decomposition - double precision (ILP64)
            void dpotrf_(const char* uplo, const int64_t* n, double* a,
                         const int64_t* lda, int64_t* info);

            // LAPACK Cholesky solver - double precision (ILP64)
            void dpotrs_(const char* uplo, const int64_t* n, const int64_t* nrhs,
                         const double* a, const int64_t* lda, double* b,
                         const int64_t* ldb, int64_t* info);
        }

        static_assert(sizeof(integer) == 8, "ILP64 LAPACK requires 64-bit integer type");
    #else
        // Default LP64 interface - use 32-bit integers
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

        static_assert(sizeof(integer) == 4, "LP64 LAPACK uses 32-bit integer type");
    #endif
#endif

#endif // LAPACK_COMPAT_H
