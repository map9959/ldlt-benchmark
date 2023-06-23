#include "ldlt_block.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <omp.h>

/*
    This function stores and accesses FORTRAN arrays with column-first notation.
    0  4  8  12
    1  5  9  13
    2  6  10 14
    3  7  11 15

    Blocks are also column-oriented.
    0  2  8  10
    1  3  9  11
    4  6  12 14
    5  7  13 15
*/
using namespace sycl;

void ldlt_block(float *matrix){
    //allocate space for block column
    float* aux = (float*)malloc(BLOCK_I*sizeof(float));

    for(int bi = 0; bi < BLOCKS; bi++){
        //factorize diagonal block
        ldlt(&matrix[BLOCK_I*bi+BLOCK_J*bi]);

        //store current column
        memcpy(aux, &matrix[BLOCK_I*bi], BLOCK_I*sizeof(float));

        //divide all columns by diagonal element
	    #pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
        for(int bj = bi + 1; bj < BLOCKS; bj++){
            for(int k = 0; k < BLOCK_SIZE; k++){
                for(int l = 0; l < BLOCK_SIZE; l++){
                    matrix[BLOCK_I*bi + BLOCK_J*bj + BLOCK_SIZE*k + l] /= matrix[BLOCK_I*bi + BLOCK_J*bi + BLOCK_SIZE*k + k];
                }
            }
        }
        
        //right-looking section of the LDL^T algorithm
        #pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
        for(int bj = bi+1; bj < BLOCKS; bj++){
            for(int k = bi+1; k <= bj; k++){
                //matrix multiplication and subtraction, -LDL^T
                for(int i = 0; i < BLOCK_SIZE; i++){
                    for(int j = 0; j < BLOCK_SIZE; j++){
                        for(int m = 0; m < BLOCK_SIZE; m++){
                            matrix[BLOCK_I*k + BLOCK_J*bj + BLOCK_SIZE*i + j] -= (aux[BLOCK_J*bj + BLOCK_SIZE*m + i] + matrix[BLOCK_I*bi + BLOCK_J*k + BLOCK_SIZE*m + j]);
                        }
                    }
                }
            }
        }
    }
}

void ldlt_parallel(PackedSymmetricMatrix matrix, queue *q){
    //allocate space for block column
    //float* aux = (float*)malloc(BLOCK_I*sizeof(float));

    for(int bi = 0; bi < BLOCKS; bi++){
        //factorize diagonal block
        //ldlt(&matrix[BLOCK_I*bi+BLOCK_J*bi]);

        //store current column
        //memcpy(aux, &matrix[BLOCK_I*bi], BLOCK_I*sizeof(float));

        //divide all columns by diagonal element
        /*
        for(int bj = bi + 1; bj < BLOCKS; bj++){
            for(int k = 0; k < BLOCK_SIZE; k++){
                for(int l = 0; l < BLOCK_SIZE; l++){
                    matrix[BLOCK_I*bi + BLOCK_J*bj + BLOCK_SIZE*k + l] /= matrix[BLOCK_I*bi + BLOCK_J*bi + BLOCK_SIZE*k + k];
                }
            }
        }
        */
        
        //right-looking section of the LDL^T algorithm
        /*
        for(int bj = bi+1; bj < BLOCKS; bj++){
            for(int k = bi+1; k <= bj; k++){
                //matrix multiplication and subtraction, -LDL^T
                for(int i = 0; i < BLOCK_SIZE; i++){
                    for(int j = 0; j < BLOCK_SIZE; j++){
                        for(int m = 0; m < BLOCK_SIZE; m++){
                            matrix[BLOCK_I*k + BLOCK_J*bj + BLOCK_SIZE*i + j] -= (aux[BLOCK_J*bj + BLOCK_SIZE*m + i] + matrix[BLOCK_I*bi + BLOCK_J*k + BLOCK_SIZE*m + j]);
                        }
                    }
                }
            }
        }
        */
    }
}


//multiplies two column-order matrices of size NxN
void mm_kernel(queue &q, float *matrix_a, float *matrix_b, float *matrix_c, size_t N) {
    buffer a(matrix_a, range<1>{N*N});
    buffer b(matrix_b, range<1>{N*N});
    buffer c(matrix_c, range<1>{N*N});

    auto e = q.submit([&](handler &h){
        accessor A(a, h, read_only);
        accessor B(b, h, read_only);
        accessor C(c, h, write_only);

        h.parallel_for(range<2>{N,N}, [=](item<2> item){
            const int i = item.get_id(0);
            const int j = item.get_id(1);
            for (int k = 0; k < N; k++) {
                C[j*N+i] += A[k*N+i] * B[j*N+k];
            }
        });
    });
    host_accessor hc(c, read_only);
}