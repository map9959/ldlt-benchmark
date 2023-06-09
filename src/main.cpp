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

float* random_sym_matrix(){
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<float> random_float(1,10);

    float* matptr = (float*)malloc(NB*NB*sizeof(float));

    for(int i = 0; i < NB; i++){
        for(int j = i; j < NB; j++){
            float rand = random_float(e);
            matptr[i*NB+j] = rand;
            matptr[j*NB+i] = rand;
        }
    }
    return matptr;
}

//Print matrix given a matrix with column-first notation
void print_matrix(float* matrix){
    for(int i = 0; i < NB; i++){
        for(int j = 0; j < NB; j++){
            std::cout << matrix[j*NB+i] << " ";
        }
        std::cout << "\n";
    }
}

int main(int argc, char *argv[]){
    float* matrix = random_sym_matrix();
    //print_matrix(matrix);
    //std::cout << "\n";
    ldlt(matrix);
    //print_matrix(matrix);
    free(matrix);
    return 0;
}