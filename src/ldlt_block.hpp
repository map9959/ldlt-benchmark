#pragma once
#include "ldlt_serial.hpp"

//blocks per row/column
#define B 64

#define BLOCK_I NB*NB*B
#define BLOCK_J NB*NB

void ldlt_block(float *matrix);
