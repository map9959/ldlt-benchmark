#include "ldlt_block.hpp"
#include "PackedSymmetricMatrix.hpp"
#include <iomanip>
#include <random>
#include <iostream>
#include <chrono>
#include <CL/sycl.hpp>

/*
    These functions store and access FORTRAN arrays with column-first notation.
    0  4  8  12
    1  5  9  13
    2  6  10 14
    3  7  11 15

    Blocks are also column-oriented.
    0  2  8  10
    1  3  9  11
    4  6  12 14
    5  7  13 15

    Indexing: matrix[BLOCK_I*bi + BLOCK_J*bj + BLOCK_SIZE*i + j]
*/

void random_sym_block(float* matptr){
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<float> random_float(-5,5);

    #pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
    for(int i = 0; i < BLOCK_SIZE; i++){
        for(int j = i; j < BLOCK_SIZE; j++){
            float rand = random_float(e);
            matptr[i*BLOCK_SIZE+j] = rand;
            matptr[j*BLOCK_SIZE+i] = rand;
        }
    }
}

float* random_sym_matrix(){
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<float> random_float(-5,5);

    float* matptr = (float*)malloc((BLOCK_SIZE*BLOCKS)*(BLOCK_SIZE*BLOCKS)*sizeof(float));

    for(int bi = 0; bi < BLOCKS; bi++){
        for(int bj = bi; bj < BLOCKS; bj++){
            if(bi == bj){
                random_sym_block(&matptr[BLOCK_I*bi + BLOCK_J*bj]);
                continue;
            }
            #pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
            for(int i = 0; i < BLOCK_SIZE; i++){
                for(int j = 0; j < BLOCK_SIZE; j++){
                    float rand = random_float(e);
                    matptr[BLOCK_I*bi + BLOCK_J*bj + BLOCK_SIZE*i + j] = rand;
                    matptr[BLOCK_I*bj + BLOCK_J*bi + BLOCK_SIZE*j + i] = rand;
                }
            }
        }
    }
    
    return matptr;
}

//Print a block matrix, given a pointer to a block matrix with column-first notation
void print_matrix(float* matrix){
    for(int bj = 0; bj < BLOCKS; bj++){
        for(int j = 0; j < BLOCK_SIZE; j++){
            for(int bi = 0; bi < BLOCKS; bi++){
                for(int i = 0; i < BLOCK_SIZE; i++){
                    std::cout << std::fixed << std::setprecision(5) << matrix[BLOCK_I*bi + BLOCK_J*bj + BLOCK_SIZE*i + j] << " ";
                }
            }
            std::cout << "\n";
        }
    }
}

int main(int argc, char *argv[]){
    std::cout << "using " << NUM_THREADS << " threads\n";

    auto packed_matrix = PackedSymmetricMatrix(BLOCK_SIZE*BLOCKS);
    auto packed_matrix_start = std::chrono::high_resolution_clock::now();
    packed_matrix.fill();
    auto packed_matrix_end = std::chrono::high_resolution_clock::now();
    auto packed_matrix_diff = std::chrono::duration_cast<std::chrono::milliseconds>(packed_matrix_end-packed_matrix_start).count();
    std::cout << "generated " << BLOCK_SIZE*BLOCKS << "x" << BLOCK_SIZE*BLOCKS << " packed matrix with block size " << BLOCK_SIZE  << " in " << packed_matrix_diff << " ms\n";
    //print_matrix(matrix);
    std::cout << "\n";

    auto matrix_start = std::chrono::high_resolution_clock::now();
    float* matrix = random_sym_matrix();
    auto matrix_end = std::chrono::high_resolution_clock::now();
    auto matrix_diff = std::chrono::duration_cast<std::chrono::milliseconds>(matrix_end-matrix_start).count();
    std::cout << "generated " << BLOCK_SIZE*BLOCKS << "x" << BLOCK_SIZE*BLOCKS << " matrix with block size " << BLOCK_SIZE  << " in " << matrix_diff << " ms\n";
    //print_matrix(matrix);
    std::cout << "\n";

    auto ldlt_start = std::chrono::high_resolution_clock::now();
    ldlt_block(matrix);
    auto ldlt_end = std::chrono::high_resolution_clock::now();
    auto ldlt_diff = std::chrono::duration_cast<std::chrono::milliseconds>(ldlt_end-ldlt_start).count();
    std::cout << "factorized " << BLOCK_SIZE*BLOCKS << "x" << BLOCK_SIZE*BLOCKS << " matrix with block size " << BLOCK_SIZE << " in " << ldlt_diff << " ms\n";
    //print_matrix(matrix);

    free(matrix);


    return 0;
}
