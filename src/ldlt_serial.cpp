#include "ldlt_serial.hpp"

/*
    This function stores and accesses FORTRAN arrays with column-first notation.
    0  4  8  12
    1  5  9  13
    2  6  10 14
    3  7  11 15
*/

void ldlt_serial(REAL_DATATYPE *matrix){
    for(int i = 0; i < BLOCK_SIZE; i++){
        //divide column
	    #pragma omp parallel for num_threads(NUM_THREADS)
        for(int j = i+1; j < BLOCK_SIZE; j++){
            matrix[i*BLOCK_SIZE+j] /= matrix[i*BLOCK_SIZE+i];
        }
	    #pragma omp parallel for num_threads(NUM_THREADS)
        for(int j = i+1; j < BLOCK_SIZE; j++){
            for(int k = i+1; k <= j; k++){
                matrix[k*BLOCK_SIZE+j] -= matrix[i*BLOCK_SIZE+j]*matrix[i*BLOCK_SIZE+i]*matrix[i*BLOCK_SIZE+k];
            }
        }
    }
}
void ldlt_serial_var(REAL_DATATYPE *matrix, int blocksize){
    for(int i = 0; i < blocksize; i++){
        //divide column
	    //#pragma omp parallel for num_threads(NUM_THREADS)
        for(int j = i+1; j < blocksize; j++){
            matrix[i*blocksize+j] /= matrix[i*blocksize+i];
        }
	    //#pragma omp parallel for num_threads(NUM_THREADS)
        for(int j = i+1; j < blocksize; j++){
            for(int k = i+1; k <= j; k++){
                matrix[k*blocksize+j] -= matrix[i*blocksize+j]*matrix[i*blocksize+i]*matrix[i*blocksize+k];
            }
        }
    }
}