#include "PackedSymmetricMatrix.hpp"
#include "ldlt_block.hpp"
#include <iostream>
#include <iomanip>
#include <random>
template <typename T> PackedSymmetricMatrix<T>::PackedSymmetricMatrix(size_t c, sycl::queue &qu){
    cols = c;
    size = cols * (cols + 1)/2;
    q = qu;
    data = sycl::malloc_shared<T>((size_t)size, q);
    parallel = true;
}
template <typename T> PackedSymmetricMatrix<T>::PackedSymmetricMatrix(size_t c){
    cols = c;
    size = cols * (cols + 1)/2;
    data = new T[size];
    parallel = false;
}
template <typename T> PackedSymmetricMatrix<T>::PackedSymmetricMatrix(T* d, size_t c){
    data = d;
    cols = c;
    size = cols * (cols + 1)/2;
    parallel = false;
}
template <typename T> PackedSymmetricMatrix<T>::~PackedSymmetricMatrix(){
    if(parallel){
        free(data, q);
    }else{
        
    }
}
//row must be >= col
template <typename T> const inline size_t PackedSymmetricMatrix<T>::index(int col, int row) const{
    return size - (cols - col) * ((cols - col) + 1) / 2 + (row - col);
}
/*
template <typename T> inline T PackedSymmetricMatrix<T>::element(int col, int row){
    return data[index(col, row)];
}
*/
template <typename T> inline T* const PackedSymmetricMatrix<T>::colPointer(int col) const{
    return &data[index(col, col)];
}
/*
template <typename T> inline T* PackedSymmetricMatrix<T>::elementPointer(int col, int row){
    return &data[index(col, row)];
}
*/
template <typename T> void const PackedSymmetricMatrix<T>::transferDiagonalBlock(T *dest, int block, int blocksize) const{
    for(int i = 0; i < blocksize; i++){
        for(int j = i; j <= blocksize; j++){
            dest[i*blocksize+j] = this->element(block*blocksize+i, block*blocksize+j);
        }
    }
}
//row must be >= col
template <typename T> void PackedSymmetricMatrix<T>::transferBlock(T *dest, int col, int row, int blocksize){
    for(int i = 0; i < blocksize; i++){
        memcpy(dest+i*blocksize, this->elementPointer(blocksize*col+i, blocksize*row), blocksize*sizeof(float));
    }
}
//row must be > col
/*template <typename T> void PackedSymmetricMatrix<T>::addBlockToNegative(T *dest, int col, int row, int blocksize)
{
    for(int i = 0; i < blocksize; i++){
        for(int j = 0; j < blocksize; j++){
            dest[i*blocksize+j] = -dest[i*blocksize+j]+this->element(col*blocksize+i, row*blocksize+j);
        }
    }
}*/
template <typename T> void const PackedSymmetricMatrix<T>::changeDiagonalBlock(T *src, int block, int blocksize) const
{
    for(int i = 0; i < blocksize; i++){
        memcpy(this->colPointer(blocksize*block+i), src+i*blocksize+i, (blocksize-i)*sizeof(float));
    }
}
/*
template <typename T> void PackedSymmetricMatrix<T>::changeBlock(float *src, int col ,int row, int blocksize)
{
    for(int i = 0; i < blocksize; i++){
        memcpy(this->elementPointer(blocksize*col+i, blocksize*row), src+i*blocksize, blocksize*sizeof(float));
    }
}
*/
//prints in MATLAB format
template <typename T> void PackedSymmetricMatrix<T>::print()
{
    std::cout << '[';
    for(int i = 0; i < cols; i++){
        for(int j = 0; j < i; j++){
            std::cout << std::fixed << std::setprecision(5) << element(j, i) << ",";
        }
        for(int j = i; j < cols; j++){
            std::cout << std::fixed << std::setprecision(5) << element(i, j) << ",";
        }
        std::cout << ";\n";
    }
    std::cout << ']';
}
template <typename T> void PackedSymmetricMatrix<T>::fill(){
    std::srand(31415);
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<T> random(1,5);
    for(size_t i = 0; i < size; i++){
        data[i] = random(e);
        //data[i] = i%100000+1;
    }
}
//saves to file in MATLAB format
template <typename T> void PackedSymmetricMatrix<T>::save(std::string fname){
    std::ofstream f(fname);
    f << '[';
    for(int i = 0; i < cols; i++){
        for(int j = 0; j < i; j++){
            f << std::fixed << std::setprecision(5) << element(j, i) << ",";
        }
        for(int j = i; j < cols; j++){
            f << std::fixed << std::setprecision(5) << element(i, j) << ",";
        }
        f << ";\n";
    }
    f << ']';
}
template <typename T> T* PackedSymmetricMatrix<T>::get_data(){
    return data;
}

template class PackedSymmetricMatrix<float>;
template class PackedSymmetricMatrix<double>;