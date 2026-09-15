#pragma once
#include <cusparse_v2.h>
#include <type_traits>

namespace cusparse
{

// ===== csric02_analysis =====
template <typename T>
inline cusparseStatus_t csric02_analysis(cusparseHandle_t handle,
                                         int m,
                                         int nnz,
                                         const cusparseMatDescr_t descrA,
                                         const T* csrSortedValA,
                                         const int* csrSortedRowPtrA,
                                         const int* csrSortedColIndA,
                                         csric02Info_t info,
                                         cusparseSolvePolicy_t policy,
                                         void* pBuffer)
{
    if constexpr (std::is_same<T, float>::value) {
        return cusparseScsric02_analysis(handle, m, nnz, descrA,
                                         csrSortedValA, csrSortedRowPtrA,
                                         csrSortedColIndA, info, policy, pBuffer);
    } else {
        return cusparseDcsric02_analysis(handle, m, nnz, descrA,
                                         csrSortedValA, csrSortedRowPtrA,
                                         csrSortedColIndA, info, policy, pBuffer);
    }
}

// ===== csric02 =====
template <typename T>
inline cusparseStatus_t csric02(cusparseHandle_t handle,
                                int m,
                                int nnz,
                                const cusparseMatDescr_t descrA,
                                T* csrSortedValA,
                                const int* csrSortedRowPtrA,
                                const int* csrSortedColIndA,
                                csric02Info_t info,
                                cusparseSolvePolicy_t policy,
                                void* pBuffer)
{
    if constexpr (std::is_same<T, float>::value) {
        return cusparseScsric02(handle, m, nnz, descrA,
                                csrSortedValA, csrSortedRowPtrA,
                                csrSortedColIndA, info, policy, pBuffer);
    } else {
        return cusparseDcsric02(handle, m, nnz, descrA,
                                csrSortedValA, csrSortedRowPtrA,
                                csrSortedColIndA, info, policy, pBuffer);
    }
}

// ===== csrsv2_analysis =====
template <typename T>
inline cusparseStatus_t csrsv2_analysis(cusparseHandle_t handle,
                                        cusparseOperation_t trans,
                                        int m,
                                        int nnz,
                                        const cusparseMatDescr_t descrA,
                                        const T* csrValA,
                                        const int* csrRowPtrA,
                                        const int* csrColIndA,
                                        csrsv2Info_t info,
                                        cusparseSolvePolicy_t policy,
                                        void* pBuffer)
{
    if constexpr (std::is_same<T, float>::value) {
        return cusparseScsrsv2_analysis(handle, trans, m, nnz, descrA,
                                        csrValA, csrRowPtrA, csrColIndA,
                                        info, policy, pBuffer);
    } else {
        return cusparseDcsrsv2_analysis(handle, trans, m, nnz, descrA,
                                        csrValA, csrRowPtrA, csrColIndA,
                                        info, policy, pBuffer);
    }
}

// ===== csrsv2_solve =====
template <typename T>
inline cusparseStatus_t csrsv2_solve(cusparseHandle_t handle,
                                     cusparseOperation_t trans,
                                     int m,
                                     int nnz,
                                     const T* alpha,  // <-- keep const here
                                     const cusparseMatDescr_t descrA,
                                     const T* csrValA,
                                     const int* csrRowPtrA,
                                     const int* csrColIndA,
                                     csrsv2Info_t info,
                                     const T* x,  // RHS
                                     T* y,        // solution
                                     cusparseSolvePolicy_t policy,
                                     void* pBuffer)
{
    if constexpr (std::is_same<T, float>::value) {
        return cusparseScsrsv2_solve(handle, trans, m, nnz,
                                     alpha, descrA, csrValA,
                                     csrRowPtrA, csrColIndA,
                                     info, x, y, policy, pBuffer);
    } else {
        return cusparseDcsrsv2_solve(handle, trans, m, nnz,
                                     alpha, descrA, csrValA,
                                     csrRowPtrA, csrColIndA,
                                     info, x, y, policy, pBuffer);
    }
}

// ===== csrsv2_bufferSize =====
template <typename T>
inline cusparseStatus_t csrsv2_bufferSize(cusparseHandle_t handle,
                                          cusparseOperation_t trans,
                                          int m,
                                          int nnz,
                                          const cusparseMatDescr_t descrA,
                                          T* csrValA,
                                          const int* csrRowPtrA,
                                          const int* csrColIndA,
                                          csrsv2Info_t info,
                                          int* pBufferSizeInBytes)
{
    if constexpr (std::is_same<T, float>::value) {
        return cusparseScsrsv2_bufferSize(handle, trans, m, nnz,
                                          descrA, csrValA, csrRowPtrA,
                                          csrColIndA, info, pBufferSizeInBytes);
    } else {
        return cusparseDcsrsv2_bufferSize(handle, trans, m, nnz,
                                          descrA, csrValA, csrRowPtrA,
                                          csrColIndA, info, pBufferSizeInBytes);
    }
}


// ========= csric02_bufferSize =========
template <typename T>
inline cusparseStatus_t csric02_bufferSize(cusparseHandle_t handle,
                                           int m,
                                           int nnz,
                                           const cusparseMatDescr_t descrA,
                                           T* csrValA,
                                           const int* csrRowPtrA,
                                           const int* csrColIndA,
                                           csric02Info_t info,
                                           int* bufferSize)
{
    if constexpr (std::is_same<T, float>::value) {
        return cusparseScsric02_bufferSize(handle,
                                           m,
                                           nnz,
                                           descrA,
                                           csrValA,
                                           csrRowPtrA,
                                           csrColIndA,
                                           info,
                                           bufferSize);
    } else {
        return cusparseDcsric02_bufferSize(handle,
                                           m,
                                           nnz,
                                           descrA,
                                           csrValA,
                                           csrRowPtrA,
                                           csrColIndA,
                                           info,
                                           bufferSize);
    }
}

} // namespace cusparse
