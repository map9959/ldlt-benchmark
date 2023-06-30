#include "PackedSymmetricMatrix.hpp"
#include "ldlt_block.hpp"
#include <iostream>
#include <iomanip>
#include <random>

PackedSymmetricMatrix::PackedSymmetricMatrix(int c){
    cols = c;
    size = cols * (cols + 1)/2;
    data = new float[size];
}
PackedSymmetricMatrix::~PackedSymmetricMatrix(){
    delete data;
}
//row must be >= col
inline int PackedSymmetricMatrix::index(int col, int row){
    return size - (cols - col) * ((cols - col) + 1) / 2 + (row - col);
}
inline float PackedSymmetricMatrix::element(int col, int row){
    return data[index(col, row)];
}
inline float* PackedSymmetricMatrix::colPointer(int col){
    return &data[index(col, col)];
}
inline float* PackedSymmetricMatrix::elementPointer(int col, int row){
    return &data[index(col, row)];
}
void PackedSymmetricMatrix::transferDiagonalBlock(float *dest, int block, int blocksize){
    for(int i = 0; i < blocksize; i++){
        memcpy(dest+i*blocksize+i, this->colPointer(blocksize*block+i), (blocksize-i)*sizeof(float));
    }
}
//row must be >= col
void PackedSymmetricMatrix::transferBlock(float *dest, int col, int row, int blocksize){
    for(int i = 0; i < blocksize; i++){
        memcpy(dest+i*blocksize, this->elementPointer(blocksize*col+i, blocksize*row), blocksize*sizeof(float));
    }
}
//row must be > col
void PackedSymmetricMatrix::addBlockToNegative(float *dest, int col, int row, int blocksize)
{
    for(int i = 0; i < blocksize; i++){
        for(int j = 0; j < blocksize; j++){
            dest[i*blocksize+j] = -dest[i*blocksize+j]+this->element(col*blocksize+j, row*blocksize+i);
        }
    }
}
void PackedSymmetricMatrix::changeDiagonalBlock(float *src, int block, int blocksize)
{
    for(int i = 0; i < blocksize; i++){
        memcpy(this->colPointer(blocksize*block+i), src+i*blocksize+i, (blocksize-i)*sizeof(float));
    }
}
void PackedSymmetricMatrix::changeBlock(float *src, int col ,int row, int blocksize)
{
    for(int i = 0; i < blocksize; i++){
        memcpy(this->elementPointer(blocksize*col+i, blocksize*row), src+i*blocksize, blocksize*sizeof(float));
    }
}
void PackedSymmetricMatrix::print()
{
    for(int i = 0; i < cols; i++){
        for(int j = 0; j < i; j++){
            std::cout << std::fixed << std::setprecision(5) << element(j, i) << " ";
        }
        for(int j = i; j < cols; j++){
            std::cout << std::fixed << std::setprecision(5) << element(i, j) << " ";
        }
        std::cout << '\n';
    }
}
void PackedSymmetricMatrix::fill(){
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<float> random_float(1,5);
    for(int i = 0; i < size; i++){
        //data[i] = random_float(e);
        data[i] = i;
    }
}