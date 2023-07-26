#include "ldlt_block.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <oneapi/mkl/blas.hpp>
#include <fstream>
#include <iomanip>

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

void ldlt_block(float *matrix, size_t n){
    size_t blocks = n/BLOCK_SIZE;
    //allocate space for block column
    float* aux = (float*)malloc(BLOCK_J*blocks*sizeof(float));

    for(int bi = 0; bi < blocks; bi++){
        //std::cout << "Working on block column " << bi << std::endl << std::flush;
        //factorize diagonal block
        ldlt_serial(&matrix[BLOCK_J*blocks*bi+BLOCK_J*bi]);

        //resolve in-block dependencies
        #pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
        for(int bj = bi + 1; bj < blocks; bj++){
            //for each row in each block
            for(int k = 0; k < BLOCK_SIZE; k++){
                //for each element in each row
                for(int l = 0; l < BLOCK_SIZE; l++){
                    //store element
                    aux[BLOCK_J*bj + BLOCK_SIZE*l + k] = matrix[BLOCK_J*blocks*bi + BLOCK_J*bj + BLOCK_SIZE*l + k];
                    //divide by diagonal
                    matrix[BLOCK_J*blocks*bi + BLOCK_J*bj + BLOCK_SIZE*l + k] /= matrix[BLOCK_J*blocks*bi + BLOCK_J*bi + BLOCK_SIZE*l + l];
                    //subtract from the right
                    for(int m = l+1; m < BLOCK_SIZE; m++){
                        matrix[BLOCK_J*blocks*bi + BLOCK_J*bj + BLOCK_SIZE*m + k] -= aux[BLOCK_J*bj+BLOCK_SIZE*l+k] * matrix[BLOCK_J*blocks*bi+BLOCK_J*bi+BLOCK_SIZE*l+m];
                    }
                }
            }
        }
        
        //right-looking section of the LDL^T algorithm
        //#pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
        for(int bj = bi+1; bj < blocks; bj++){
            for(int k = bi+1; k <= bj; k++){
                //matrix multiplication and subtraction, -LDL^T
                #pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
                for(int i = 0; i < BLOCK_SIZE; i++){
                    for(int j = 0; j < BLOCK_SIZE; j++){
                        float temp = 0.f;
                        for(int m = 0; m < BLOCK_SIZE; m++){
                            temp += aux[BLOCK_J*bj + BLOCK_SIZE*m + i] * matrix[BLOCK_J*blocks*bi + BLOCK_J*k + BLOCK_SIZE*m + j];
                        }
                        matrix[BLOCK_J*blocks*k + BLOCK_J*bj + BLOCK_SIZE*i + j] -= temp;
                    }
                }
            }
        }
    }
}

void save_matrix_temp(float* matrix, std::string fname, int blocks){
    std::ofstream f(fname);
    f << '[';
    for(int bj = 0; bj < blocks; bj++){
        for(int j = 0; j < BLOCK_SIZE; j++){
            for(int bi = 0; bi < blocks; bi++){
                for(int i = 0; i < BLOCK_SIZE; i++){
                    f << std::fixed << std::setprecision(5) << matrix[BLOCK_J*blocks*bi + BLOCK_J*bj + BLOCK_SIZE*i + j] << ",";
                }
            }
            f << ";\n";
        }
    }
    f << ']';
}

void ldlt_coarse(PackedSymmetricMatrix<float> &matrix, size_t n, queue &q){
    size_t blocks = n/BLOCK_SIZE;
    //allocate space for block column
    float* aux = malloc_shared<float>((size_t)(BLOCK_J*blocks), q);
    //allocate workspace
    float* workspace = malloc_shared<float>((size_t)(BLOCK_J*blocks*(blocks+1)/2), q);
    memset(workspace, 0, BLOCK_J*blocks*blocks);

    for(int bi = 0; bi < blocks; bi++){
        //std::cout << "Working on block column " << bi << std::endl << std::flush;
        //transfer and factorize diagonal block, transfer back
        matrix.addDiagToNegative(&workspace[BLOCK_J*pick_block(bi, bi, blocks)], bi, BLOCK_SIZE);
        ldlt_serial(&workspace[BLOCK_J*pick_block(bi, bi, blocks)]);
        matrix.changeDiagonalBlock(&workspace[BLOCK_J*pick_block(bi, bi, blocks)], bi, BLOCK_SIZE);
        
        if(bi != blocks-1){
            //std::cout << "Resolving block column " << bi << std::endl << std::flush;
            //resolve in-block dependencies
            for(int bj = bi+1; bj < blocks; bj++){
                matrix.addBlockToNegative(&workspace[BLOCK_J*pick_block(bi, bj, blocks)], bi, bj, BLOCK_SIZE);
            }
            const size_t bcol = blocks-(bi+1);
            q.submit([=](handler &h){
                h.parallel_for(range<2>{bcol, BLOCK_SIZE}, [=](id<2> idx){
                    const int bj = idx[0]+bi+1;
                    const int k = idx[1];
                    const size_t diag_block = BLOCK_J*pick_block(bi, bi, blocks);
                    const size_t update_block = BLOCK_J*pick_block(bi, bj, blocks);
                    for(int l = 0; l < BLOCK_SIZE; l++){
                        aux[BLOCK_J*bj + BLOCK_SIZE*l + k] = workspace[update_block + BLOCK_SIZE*l + k];
                        workspace[update_block + BLOCK_SIZE*l + k] /= workspace[diag_block + BLOCK_SIZE*l + l];
                        float temp = aux[BLOCK_J*bj+BLOCK_SIZE*l+k];
                        for(int m = l+1; m < BLOCK_SIZE; m++){
                            workspace[update_block + BLOCK_SIZE*m + k] -= temp * workspace[diag_block + BLOCK_SIZE*l + m];
                        }
                    }
                });
            }).wait();
            for(int bj = bi+1; bj < blocks; bj++){
                matrix.changeBlock(&workspace[BLOCK_J*pick_block(bi, bj, blocks)], bi, bj, BLOCK_SIZE);
            }
            //right-looking section of the LDL^T algorithm
            //matrix multiplication and addition, -LDL^T
            //for(){
                q.submit([&](handler &h){
                    h.parallel_for(range<3>{bcol*bcol, BLOCK_SIZE,BLOCK_SIZE}, [=](id<3> idx){
                        const int b = idx[0];
                        const int i = idx[1];
                        const int j = idx[2];
                        const int bj = b/bcol+bi+1;
                        const int k = b%bcol+bi+1;

                        if(k <= bj){
                            const size_t l_block = BLOCK_J*pick_block(bi, k, blocks);
                            const size_t update_block = BLOCK_J*pick_block(k, bj, blocks);
                            float temp = workspace[update_block + i*BLOCK_SIZE + j];
                            for (int m = 0; m < BLOCK_SIZE; m++) {
                                temp += aux[BLOCK_J*bj + m*BLOCK_SIZE + i] * workspace[l_block + m*BLOCK_SIZE + j];
                            }
                            workspace[update_block + i*BLOCK_SIZE + j] = temp;
                        }
                    });
                });
                q.wait();
            //}
        }
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

//multiplies two column-order matrices A, B^T of size NxN and adds them to C. requires dependencies
event mm_kernel_abt(queue &q, float *matrix_a, float *matrix_b, float *matrix_c, float alpha, size_t N) {
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
            float temp = 0.f;
            for (int k = 0; k < N; k++) {
                temp += alpha * A[k*N+i] * B[k*N+j];
            }
            C[i*N+j] = temp;
        });
    });
    host_accessor hc(c, read_only);
    return e;
}

//multiplies two column-order matrices A, B^T of size NxN and adds them to C
event mm_kernel_abt_local(queue &q, float *matrix_a, float *matrix_b, float *matrix_c, float alpha, size_t N, size_t M) {
    buffer a(matrix_a, range<1>{N*N});
    buffer b(matrix_b, range<1>{N*N});
    buffer c(matrix_c, range<1>{N*N});

    auto e = q.submit([&](handler &h){
        accessor A(a, h, read_only);
        accessor B(b, h, read_only);
        accessor C(c, h, write_only);

        range<2> global_size(M,M);
        range<2> work_group_size(M,M);

        local_accessor<float, 2> A_tile(range<2>(M, M), h);
        local_accessor<float, 2> B_tile(range<2>(M, M), h);

        h.parallel_for(nd_range<2>{global_size, work_group_size}, [=](nd_item<2> item){
            const int i = item.get_global_id(0);
            const int j = item.get_global_id(1);
            const int x = item.get_local_id(0);
            const int y = item.get_local_id(1);

            float temp = 0.f;
            int k;
            for (int t = 0; t < N; t += M) {
                A_tile[x][y] = A[i * N + (t + y)];
                B_tile[x][y] = B[(t + x) * N + j];
                item.barrier(access::fence_space::local_space);
                for (k = 0; k < M; k++) {
                    temp += alpha * A_tile[k][x] * B_tile[k][y];
                }
            }
            C[i*N+j] = temp;
            
        });
    });
    host_accessor hc(c, read_only);
    return e;
}

//multiplies two column-order matrices A, B^T of size NxN and adds them to C. requires dependencies
event mm_kernel_abt(queue &q, float *matrix_a, float *matrix_b, float *matrix_c, float alpha, size_t N, std::vector<event> dependencies) {
    buffer a(matrix_a, range<1>{N*N});
    buffer b(matrix_b, range<1>{N*N});
    buffer c(matrix_c, range<1>{N*N});

    auto e = q.submit([&](handler &h){
        h.depends_on(dependencies);
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
    return e;
}

/* Perform the in-place LDL^T factorization of a packed symmetric matrix of size n x n i
 * Will occupy n * (n+1) / 2 elements in memory
 * Stored like Fortran, by columns!
 * We know that all elements of L are 1, so we store D there instead
 * Practically, the new matrix is really L - I + D
 */
void ldlt(REAL_DATATYPE *matrix, size_t n){
    if(n % BLOCK_SIZE != 0){
        std::cout << "ERROR: Block size must be divisible by " << BLOCK_SIZE << '\n';
        return;
    }
    queue q;
    PackedSymmetricMatrix packed_matrix = PackedSymmetricMatrix(matrix, n);
    ldlt_coarse(packed_matrix, n, q);
}