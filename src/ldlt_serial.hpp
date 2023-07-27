//elements in a block per row/column
#define BLOCK_SIZE 64

//number of threads to use
#define NUM_THREADS 32

#define REAL_DATATYPE float

void ldlt_serial(REAL_DATATYPE *matrix);
void ldlt_serial_var(REAL_DATATYPE *matrix, int blocksize);