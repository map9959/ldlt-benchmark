#pragma once
#include "ldlt_serial.hpp"
#include <CL/sycl.hpp>
#include "PackedSymmetricMatrix.hpp"

//blocks per row/column
#define BLOCKS 64

#define BLOCK_I BLOCK_SIZE*BLOCK_SIZE*BLOCKS
#define BLOCK_J BLOCK_SIZE*BLOCK_SIZE

void ldlt_block(float *matrix);
void ldlt_parallel(PackedSymmetricMatrix *matrix, sycl::queue *q);
void mm_kernel(sycl::queue &q, float *matrix_a, float *matrix_b, float *matrix_c, size_t N);