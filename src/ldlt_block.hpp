#pragma once
#include "ldlt_serial.hpp"

//blocks per row/column
#define BLOCKS 64

#define BLOCK_I BLOCK_SIZE*BLOCK_SIZE*BLOCKS
#define BLOCK_J BLOCK_SIZE*BLOCK_SIZE

void ldlt_block(float *matrix);
