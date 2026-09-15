#pragma once
#include <cublas_v2.h>
#include <type_traits>

namespace cublas 
{

// ========= scal =========
template <typename T>
inline cublasStatus_t scal(cublasHandle_t handle, int n, const T* alpha, T* x, int incx) {
    if constexpr (std::is_same<T, float>::value) {
        return cublasSscal(handle, n, alpha, x, incx);
    } else {
        return cublasDscal(handle, n, alpha, x, incx);
    }
}

// ========= axpy =========
template <typename T>
inline cublasStatus_t axpy(cublasHandle_t handle, int n, const T* alpha, const T* x, int incx, T* y, int incy) {
    if constexpr (std::is_same<T, float>::value) {
        return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
    } else{
        return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
    }
}

// ========= dot =========
template <typename T>
inline cublasStatus_t dot(cublasHandle_t handle, int n, const T* x, int incx, const T* y, int incy, T* result) {
    if constexpr (std::is_same<T, float>::value) {
        return cublasSdot(handle, n, x, incx, y, incy, result);
    } else {
        return cublasDdot(handle, n, x, incx, y, incy, result);
    }
}

// ========= norm =========
template <typename T>
inline cublasStatus_t nrm2(cublasHandle_t handle, int n, const T* x, int incx, T* y) {
    if constexpr (std::is_same<T, float>::value) {
        return cublasSnrm2(handle, n, x, incx, y);
    } else {
        return cublasDnrm2(handle, n, x, incx, y);
    }
}

// ========= getrfBatched =========
template <typename T>
inline cublasStatus_t getrfBatched(cublasHandle_t handle,
                                   int n,
                                   T** Aarray,
                                   int lda,
                                   int* PivotArray,
                                   int* infoArray,
                                   int batchSize) {
    if constexpr (std::is_same<T, float>::value) {
        return cublasSgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
    } else {
        return cublasDgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
    }
}


// ========= getriBatched =========
template <typename T>
inline cublasStatus_t getriBatched(cublasHandle_t handle,
                                   int n,
                                   const T** Aarray,
                                   int lda,
                                   int* PivotArray,
                                   T** Carray,
                                   int ldc,
                                   int* infoArray,
                                   int batchSize) {
    if constexpr (std::is_same<T, float>::value) {
        return cublasSgetriBatched(handle, n, Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize);
    } else {
        return cublasDgetriBatched(handle, n, Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize);
    }
}


} // namespace cublas
