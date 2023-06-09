#pragma once
#include "ldlt_serial.hpp"

/*
    This function stores and accesses arrays with FORTRAN (column-first) notation.
    0  4  8  12
    1  5  9  13
    2  6  10 14
    3  7  11 15
*/

void ldlt(int size, float *matrix){
    for(int i = 0; i < size; i++){
        //divide column
        for(int j = i+1; j < size; j++){
            matrix[i*size+j] /= matrix[i*size+i];
        }
        for(int j = i+1; j < size; j++){
            for(int k = i+1; k <= j; k++){
                matrix[k*size+j] -= matrix[i*size+j]*matrix[i*size+i]*matrix[i*size+k];
            }
        }
    }
}