#include "ldlt_block.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <omp.h>

/*
    This function stores and accesses FORTRAN arrays with column-first notation.
    Example: a 4x4 matrix should be indexed as such:
    0  4  8  12
    1  5  9  13
    2  6  10 14
    3  7  11 15

    Blocks are also column-oriented.
    Example: a 4x4 matrix with 2x2 blocks of 2x2 elements should be indexed as such:
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
        //std::cout << "Working on block column " << bi << std::endl << std::flush;
        //factorize diagonal block
        ldlt_serial(&matrix[BLOCK_I*bi+BLOCK_J*bi]);

        //resolve in-block dependencies
        for(int bj = bi + 1; bj < BLOCKS; bj++){
            for(int k = 0; k < BLOCK_SIZE; k++){
                //store current column
                memcpy(&aux[BLOCK_J*bj+BLOCK_SIZE*k], &matrix[BLOCK_I*bi+BLOCK_J*bj+BLOCK_SIZE*k], BLOCK_SIZE*sizeof(float));
                //divide column by diagonal element
                for(int l = 0; l < BLOCK_SIZE; l++){
                    matrix[BLOCK_I*bi + BLOCK_J*bj + BLOCK_SIZE*k + l] /= matrix[BLOCK_I*bi + BLOCK_J*bi + BLOCK_SIZE*k + k];
                }
                //subtract from the right
                for(int l = k+1; l < BLOCK_SIZE; l++){
                    for(int m = 0; m < BLOCK_SIZE; m++){
                        matrix[BLOCK_I*bi + BLOCK_J*bj + BLOCK_SIZE*m + l] -= aux[BLOCK_J*bj+BLOCK_SIZE*m+k] * matrix[BLOCK_I*bi+BLOCK_J*bi+BLOCK_SIZE*k+l];
                    }
                }
            }
        }
        
        
        //right-looking section of the LDL^T algorithm
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

void ldlt_parallel(PackedSymmetricMatrix<float>* matrix, queue &q){
    //allocate space for block column
    float* aux = malloc_shared<float>((size_t)BLOCK_I, q);
    //allocate workspace
    
    float* workspace = malloc_shared<float>((size_t)(BLOCK_I*BLOCKS), q);

    for(int bi = 0; bi < BLOCKS; bi++){
        //std::cout << "Working on block column " << bi << std::endl << std::flush;
        //factorize diagonal block
        matrix->transferDiagonalBlock(&workspace[BLOCK_I*bi+BLOCK_J*bi], bi, BLOCK_SIZE);
        ldlt_serial(&workspace[BLOCK_I*bi+BLOCK_J*bi]);
        matrix->changeDiagonalBlock(&workspace[BLOCK_I*bi+BLOCK_J*bi], bi, BLOCK_SIZE);

        //std::cout << "Resolving block column " << bi << std::endl << std::flush;
        //resolve in-block dependencies
        for(int bj = bi+1; bj < BLOCKS; bj++){
            matrix->addBlockToNegative(&workspace[BLOCK_I*bi+BLOCK_J*bj], bi, bj, BLOCK_SIZE);
            auto e = q.submit([&](handler &h){
                h.parallel_for(range<1>{BLOCK_SIZE}, [=](item<1> item){
                    const int k = item.get_id(0);
                    //store current column
                    memcpy(&aux[BLOCK_J*bj+BLOCK_SIZE*k], &workspace[BLOCK_I*bi+BLOCK_J*bj+BLOCK_SIZE*k], BLOCK_SIZE*sizeof(float));
                    //divide column by diagonal element
                    for(int l = 0; l < BLOCK_SIZE; l++){
                        workspace[BLOCK_I*bi + BLOCK_J*bj + BLOCK_SIZE*k + l] /= workspace[BLOCK_I*bi + BLOCK_J*bi + BLOCK_SIZE*k + k];
                    }
                    //subtract from the right
                    for(int l = k+1; l < BLOCK_SIZE; l++){
                        for(int m = 0; m < BLOCK_SIZE; m++){
                            workspace[BLOCK_I*bi + BLOCK_J*bj + BLOCK_SIZE*m + l] -= aux[BLOCK_J*bj+BLOCK_SIZE*m+k] * workspace[BLOCK_I*bi+BLOCK_J*bi+BLOCK_SIZE*k+l];
                        }
                    }
                });
            });
            matrix->changeBlock(&workspace[BLOCK_I*bi+BLOCK_J*bj], bi, bj, BLOCK_SIZE, e);
        }
        q.wait();
        
        //right-looking section of the LDL^T algorithm
        for(int bj = bi+1; bj < BLOCKS; bj++){
            for(int k = bi+1; k <= bj; k++){
                //matrix multiplication and subtraction, -LDL^T
                mm_kernel_abt(q, &aux[BLOCK_J*bj], &workspace[BLOCK_I*bi + BLOCK_J*k], &workspace[BLOCK_I*k + BLOCK_J*bj], -1, BLOCK_SIZE);
            }
        }
        q.wait();
    }
    
}


//multiplies two column-order matrices A, B of size NxN and adds them to C
void mm_kernel(queue &q, float *matrix_a, float *matrix_b, float *matrix_c, float alpha, size_t N) {
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
                C[j*N+i] += alpha * A[k*N+i] * B[j*N+k];
            }
        });
    });
    host_accessor hc(c, read_only);
}

//multiplies two column-order matrices A, B^T of size NxN and adds them to C
void mm_kernel_abt(queue &q, float *matrix_a, float *matrix_b, float *matrix_c, float alpha, size_t N) {
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
                C[j*N+i] += alpha * A[k*N+i] * B[k*N+j];
            }
        });
    });
    host_accessor hc(c, read_only);
}

/* Perform the in-place LDL^T factorization of a packed symmetric matrix of size n x n i
 * Will occupy n * (n+1) / 2 elements in memory
 * Stored like Fortran, by columns!
 * We know that all elements of L are 1, so we store D there instead
 * Practically, the new matrix is really L - I + D
 */
void ldlt(REAL_DATATYPE *matrix, size_t n){
    queue q;
    PackedSymmetricMatrix packed_matrix = PackedSymmetricMatrix(matrix, n);
    ldlt_parallel(&packed_matrix, q);
}