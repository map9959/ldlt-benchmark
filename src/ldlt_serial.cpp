#include "ldlt_serial.hpp"

/*
    This function stores and accesses FORTRAN arrays with column-first notation.
    0  4  8  12
    1  5  9  13
    2  6  10 14
    3  7  11 15
*/

void ldlt(float *matrix){
    for(int i = 0; i < NB; i++){
        //divide column
        for(int j = i+1; j < NB; j++){
            matrix[i*NB+j] /= matrix[i*NB+i];
        }
        for(int j = i+1; j < NB; j++){
            for(int k = i+1; k <= j; k++){
                matrix[k*NB+j] -= matrix[i*NB+j]*matrix[i*NB+i]*matrix[i*NB+k];
            }
        }
    }
}