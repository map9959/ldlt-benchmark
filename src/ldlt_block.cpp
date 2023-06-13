#include "ldlt_block.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>

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
    //allocate space for block column
    float* aux = (float*)malloc(BLOCK_I*sizeof(float));

    for(int bi = 0; bi < B; bi++){
        //factorize diagonal block
        ldlt(&matrix[BLOCK_I*bi+BLOCK_J*bi]);

        //store current column
        memcpy(aux, &matrix[BLOCK_I*bi], BLOCK_I*sizeof(float));

        //divide all columns by diagonal element
        for(int bj = bi + 1; bj < B; bj++){
            for(int k = 0; k < NB; k++){
                for(int l = 0; l < NB; l++){
                    matrix[BLOCK_I*bi + BLOCK_J*bj + NB*k + l] /= matrix[BLOCK_I*bi + BLOCK_J*bi + NB*k + k];
                }
            }
        }
        
        //right-looking section of the LDL^T algorithm
        for(int bj = bi+1; bj < B; bj++){
            for(int k = bi+1; k <= bj; k++){
                //matrix multiplication and subtraction, -LDL^T
                for(int i = 0; i < NB; i++){
                    for(int j = 0; j < NB; j++){
                        for(int m = 0; m < NB; m++){
                            matrix[BLOCK_I*k + BLOCK_J*bj + NB*i + j] -= (aux[BLOCK_J*bj + NB*m + i] + matrix[BLOCK_I*bi + BLOCK_J*k + NB*m + j]);
                        }
                    }
                }
            }
        }
    }
}