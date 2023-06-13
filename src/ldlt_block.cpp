#include "ldlt_block.hpp"

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

void ldlt_block(float *matrix){
    for(int i = 0; i < B; i++){
        //factorize diagonal block
        /*
        for(int j = i+1; j < NB; j++){
            matrix[i*NB+j] /= matrix[i*NB+i];
        }
        */
        //store current column
        /*
        */
        for(int j = i + 1; j < B; j++){
            for(int k = 0; k < NB; k++){
                //divide all columns by diagonal element
            }
        }
        for(int j = i+1; j < B; j++){
            for(int k = i+1; k <= j; k++){
                //multiplication and addition
            }
        }
    }
}