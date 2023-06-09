#include "ldlt_serial.hpp"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <random>

/*
    This function stores and accesses arrays with FORTRAN (column-first) notation.
    0  4  8  12
    1  5  9  13
    2  6  10 14
    3  7  11 15
*/

float* random_sym_matrix(int size){
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<float> random_float(1,10);

    float* matptr = (float*)malloc(size*size*sizeof(float));

    for(int i = 0; i < size; i++){
        for(int j = i; j < size; j++){
            float rand = random_float(e);
            matptr[i*size+j] = rand;
            matptr[j*size+i] = rand;
        }
    }
    return matptr;
}

//Print matrix given a matrix with column-first notation
void print_matrix(int size, float* matrix){
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            std::cout << matrix[j*size+i] << " ";
        }
        std::cout << "\n";
    }
}

int main(int argc, char *argv[]){
    int matrix_size = 4;
    float* matrix = random_sym_matrix(matrix_size);
    print_matrix(matrix_size, matrix);
    std::cout << "\n";
    ldlt(matrix_size, matrix);
    print_matrix(matrix_size, matrix);
    free(matrix);
    return 0;
}