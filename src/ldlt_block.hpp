#pragma once
#include "ldlt_serial.hpp"
#include <CL/sycl.hpp>
#include "PackedSymmetricMatrix.hpp"
#include <stddef.h>


//blocks per row/column
#define BLOCKS 64

#define BLOCK_I BLOCK_SIZE*BLOCK_SIZE*BLOCKS
#define BLOCK_J BLOCK_SIZE*BLOCK_SIZE

#define REAL_DATATYPE float

void ldlt_block(float *matrix);
void ldlt_parallel(PackedSymmetricMatrix<REAL_DATATYPE>* matrix, sycl::queue &q);
void mm_kernel(sycl::queue &q, float *matrix_a, float *matrix_b, float *matrix_c, float alpha, size_t N);
void mm_kernel_abt(sycl::queue &q, float *matrix_a, float *matrix_b, float *matrix_c, float alpha, size_t N);
/* Perform the in-place LDL^T factorization of a packed symmetric matrix of size n x n i
 * Will occupy n * (n+1) / 2 elements in memory
 * Stored like Fortran, by columns!
 * We know that all elements of L are 1, so we store D there instead
 * Practically, the new matrix is really L - I + D
 */
void ldlt(REAL_DATATYPE *matrix, size_t n);