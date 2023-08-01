#pragma once
#include "ldlt_serial.hpp"
#include <CL/sycl.hpp>
#include "PackedSymmetricMatrix.hpp"
#include <stddef.h>


//blocks per row/column
#define BLOCKS 64

#define BLOCK_I BLOCK_SIZE*BLOCK_SIZE*BLOCKS
#define BLOCK_J BLOCK_SIZE*BLOCK_SIZE

//formerly BLOCK_J*blocks*bi+BLOCK_J*bi
inline size_t pick_block(size_t col, size_t row, size_t blocks){
    return blocks*(blocks+1)/2 - (blocks - col) * ((blocks - col) + 1) / 2 + (row - col);
}
void ldlt_block(REAL_DATATYPE *matrix, size_t n);
void ldlt_block_packed(PackedSymmetricMatrix<REAL_DATATYPE> &matrix, size_t n);
void ldlt_coarse(PackedSymmetricMatrix<REAL_DATATYPE> &matrix, size_t n, sycl::queue &q);
void mm_kernel(sycl::queue &q, float *matrix_a, float *matrix_b, float *matrix_c, float alpha, size_t N);
sycl::event mm_kernel_abt(sycl::queue &q, float *matrix_a, float *matrix_b, float *matrix_c, float alpha, size_t N);
sycl::event mm_kernel_abt_local(sycl::queue &q, float *matrix_a, float *matrix_b, float *matrix_c, float alpha, size_t N, size_t M);
sycl::event mm_kernel_abt(sycl::queue &q, float *matrix_a, float *matrix_b, float *matrix_c, float alpha, size_t N, std::vector<sycl::event> dependencies);
/* Perform the in-place LDL^T factorization of a packed symmetric matrix of size n x n i
 * Will occupy n * (n+1) / 2 elements in memory
 * Stored like Fortran, by columns!
 * We know that all elements of L are 1, so we store D there instead
 * Practically, the new matrix is really L - I + D
 */
void ldlt(REAL_DATATYPE *matrix, size_t n);
void ldlt_cpu(REAL_DATATYPE *matrix, size_t n);