#include "ldlt_block.hpp"
#include <iomanip>
#include <random>
#include <iostream>
#include <chrono>

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

    Indexing: matrix[BLOCK_I*bi + BLOCK_J*bj + NB*i + j]
*/

void random_sym_block(float* matptr){
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<float> random_float(-5,5);

    //#pragma omp parallel for num_threads(NUM_THREADS)
    for(int i = 0; i < NB; i++){
        for(int j = i; j < NB; j++){
            float rand = random_float(e);
            matptr[i*NB+j] = rand;
            matptr[j*NB+i] = rand;
        }
    }
}

float* random_sym_matrix(){
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<float> random_float(-5,5);

    float* matptr = (float*)malloc((NB*B)*(NB*B)*sizeof(float));

    for(int bi = 0; bi < B; bi++){
        for(int bj = bi; bj < B; bj++){
            if(bi == bj){
                random_sym_block(&matptr[BLOCK_I*bi + BLOCK_J*bj]);
                continue;
            }
            //#pragma omp parallel for num_threads(NUM_THREADS)
            for(int i = 0; i < NB; i++){
                for(int j = 0; j < NB; j++){
                    float rand = random_float(e);
                    matptr[BLOCK_I*bi + BLOCK_J*bj + NB*i + j] = rand;
                    matptr[BLOCK_I*bj + BLOCK_J*bi + NB*j + i] = rand;
                }
            }
        }
    }
    
    return matptr;
}

//Print a block matrix, given a pointer to a block matrix with column-first notation
void print_matrix(float* matrix){
    for(int bj = 0; bj < B; bj++){
        for(int j = 0; j < NB; j++){
            for(int bi = 0; bi < B; bi++){
                for(int i = 0; i < NB; i++){
                    std::cout << std::fixed << std::setprecision(5) << matrix[BLOCK_I*bi + BLOCK_J*bj + NB*i + j] << " ";
                }
            }
            std::cout << "\n";
        }
    }
}

int main(int argc, char *argv[]){
    std::cout << "using " << NUM_THREADS << " threads\n";

    auto matrix_start = std::chrono::high_resolution_clock::now();
    float* matrix = random_sym_matrix();
    auto matrix_end = std::chrono::high_resolution_clock::now();
    auto matrix_diff = std::chrono::duration_cast<std::chrono::milliseconds>(matrix_end-matrix_start).count();
    std::cout << "generated " << NB*B << "x" << NB*B << " matrix with block size " << NB  << " in " << matrix_diff << " ms\n";
    //print_matrix(matrix);
    std::cout << "\n";

    auto ldlt_start = std::chrono::high_resolution_clock::now();
    ldlt_block(matrix);
    auto ldlt_end = std::chrono::high_resolution_clock::now();
    auto ldlt_diff = std::chrono::duration_cast<std::chrono::milliseconds>(ldlt_end-ldlt_start).count();
    std::cout << "factorized " << NB*B << "x" << NB*B << " matrix with block size " << NB << " in " << ldlt_diff << " ms\n";
    //print_matrix(matrix);

    free(matrix);
    return 0;
}