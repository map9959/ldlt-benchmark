#include "ldlt_block.hpp"
#include <iomanip>
#include <random>
#include <iostream>
#include <chrono>
#include <sys/time.h>
#include <fstream>

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
    std::uniform_real_distribution<float> random_float(1,5);

    //#pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
    for(int i = 0; i < BLOCK_SIZE; i++){
        for(int j = i; j < BLOCK_SIZE; j++){
            float rand = random_float(e);
            matptr[i*BLOCK_SIZE+j] = rand;
            matptr[j*BLOCK_SIZE+i] = rand;
        }
    }
}

void random_sym_block_var(float* matptr, int size){
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<float> random_float(1,5);

    //#pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
    for(int i = 0; i < size; i++){
        for(int j = i; j < size; j++){
            float rand = random_float(e);
            matptr[i*size+j] = rand;
            matptr[j*size+i] = rand;
        }
    }
}

float* random_sym_matrix(size_t b){
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<float> random_float(1,5);

    float* matptr = (float*)malloc((BLOCK_SIZE)*(BLOCK_SIZE)*b*b*sizeof(float));

    for(int bi = 0; bi < b; bi++){
        for(int bj = bi; bj < b; bj++){
            if(bi == bj){
                random_sym_block(&matptr[BLOCK_J*b*bi + BLOCK_J*bj]);
                continue;
            }
            //#pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
            for(int i = 0; i < BLOCK_SIZE; i++){
                for(int j = 0; j < BLOCK_SIZE; j++){
                    float rand = random_float(e);
                    matptr[BLOCK_J*b*bi + BLOCK_J*bj + BLOCK_SIZE*i + j] = rand;
                    matptr[BLOCK_J*b*bj + BLOCK_J*bi + BLOCK_SIZE*j + i] = rand;
                }
            }
        }
    }
    
    return matptr;
}

//Print a block matrix, given a pointer to a block matrix with column-first notation
void print_matrix(float* matrix){
    std::cout << '[';
    for(int bj = 0; bj < BLOCKS; bj++){
        for(int j = 0; j < BLOCK_SIZE; j++){
            for(int bi = 0; bi < BLOCKS; bi++){
                for(int i = 0; i < BLOCK_SIZE; i++){
                    std::cout << std::fixed << std::setprecision(5) << matrix[BLOCK_I*bi + BLOCK_J*bj + BLOCK_SIZE*i + j] << ",";
                }
            }
            std::cout << ";\n";
        }
    }
    std::cout << ']';
}

//Save a block matrix to a txt file, given a pointer to a block matrix with column-first notation
void save_matrix(float* matrix, std::string fname, int blocks){
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

int main(int argc, char *argv[]){
    using namespace sycl;
    sycl::property_list q_prop{};
    queue q(sycl::gpu_selector_v, q_prop);

    //std::cout << omp_get_thread_num() << "\n";
    std::cout << "using device: " << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "using " << NUM_THREADS << " threads\n";

    size_t mat_size = BLOCK_SIZE*BLOCKS*16-64;
    double flops = (double)mat_size*mat_size*mat_size/3.0;

    auto packed_matrix = PackedSymmetricMatrix<REAL_DATATYPE>(mat_size, q);
    auto packed_matrix_start = std::chrono::high_resolution_clock::now();
    packed_matrix.fill();
    auto packed_matrix_end = std::chrono::high_resolution_clock::now();
    auto packed_matrix_diff = std::chrono::duration_cast<std::chrono::milliseconds>(packed_matrix_end-packed_matrix_start).count();
    std::cout << "generated " << mat_size << "x" << mat_size << " packed matrix with block size " << BLOCK_SIZE << " in " << packed_matrix_diff << " ms\n";
    //packed_matrix.print();
    //std::cout << "\n";

    auto matrix_start = std::chrono::high_resolution_clock::now();
    //float* matrix = random_sym_matrix(mat_size/BLOCK_SIZE);
    auto matrix_end = std::chrono::high_resolution_clock::now();
    auto matrix_diff = std::chrono::duration_cast<std::chrono::milliseconds>(matrix_end-matrix_start).count();
    std::cout << "generated " << mat_size << "x" << mat_size << " matrix with block size " << BLOCK_SIZE << " in " << matrix_diff << " ms\n"; 
    //save_matrix(matrix, "example-matrix-128.txt", mat_size/BLOCK_SIZE);
    //std::cout << "\n";
    
    struct timeval tp0, tp1;
    gettimeofday(&tp0, nullptr);
    //ldlt_block(matrix, mat_size);
    gettimeofday(&tp1, nullptr);
    double t0 = tp0.tv_sec + (double)tp0.tv_usec / 1e6;
    double t1 = tp1.tv_sec + (double)tp1.tv_usec / 1e6;
    double time_seconds = t1-t0;
    double gigaflops_per_s = flops / 1e9 / time_seconds;
    std::cout << "size: " << mat_size << "\nblock size: " << BLOCK_SIZE << "\nelapsed time: " << time_seconds << " s\nperformance: " << gigaflops_per_s << " gigaflops/s\n";
    //save_matrix(matrix, "example-matrix-128-ldlt.txt", mat_size/BLOCK_SIZE);
    
    //packed_matrix.save("example-packed-128.txt");
    gettimeofday(&tp0, nullptr);
    ldlt(packed_matrix.get_data(), mat_size);
    gettimeofday(&tp1, nullptr);
    t0 = tp0.tv_sec + (double)tp0.tv_usec / 1e6;
    t1 = tp1.tv_sec + (double)tp1.tv_usec / 1e6;
    double time_seconds_par = t1-t0;
    double gigaflops_per_s_par = flops / 1e9 / time_seconds_par;
    std::cout << "size: " << mat_size << "\nblock size: " << BLOCK_SIZE << "\nelapsed time: " << time_seconds_par << " s\nperformance: " << gigaflops_per_s_par << " gigaflops/s\n";
    std::cout << "speedup rate: " << time_seconds/(double)time_seconds_par << '\n';
    std::cout << "hypothetical speedup rate: " << gigaflops_per_s_par/(double)23 << '\n';
    //packed_matrix.save("example-packed-128-ldlt.txt");
    //save_matrix(packed_matrix, "example-matrix-128-ldlt-parallel.txt", mat_size/BLOCK_SIZE);

    return 0;
}