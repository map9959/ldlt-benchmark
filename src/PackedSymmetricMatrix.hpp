/*
This class implements a packed symmetric matrix data structure, storing the lower half in column-order.
Example: [a_11, a_21, ..., a_n1, a_22, a_32, ..., a_n2, ..., a_nn]
*/
#pragma once
#include <CL/sycl.hpp>
template <typename T> class PackedSymmetricMatrix{
    public:
        PackedSymmetricMatrix(int c, sycl::queue &qu);
        PackedSymmetricMatrix(int c);
        PackedSymmetricMatrix(T* d, int c);
        ~PackedSymmetricMatrix();
        inline int index(int col, int row);
        inline T element(int col, int row){
            return data[size - (cols - col) * ((cols - col) + 1) / 2 + (row - col)];
        }
        inline T* colPointer(int col);
        inline T* elementPointer(int col, int row){
            return &data[size - (cols - col) * ((cols - col) + 1) / 2 + (row - col)];
        }
        void transferDiagonalBlock(T* dest, int block, int blocksize);
        void transferBlock(T* dest, int col, int row, int blocksize);
        inline void addBlockToNegative(T* dest, int col, int row, int blocksize){
            for(int i = 0; i < blocksize; i++){
                for(int j = 0; j < blocksize; j++){
                    dest[i*blocksize+j] = -dest[i*blocksize+j]+this->element(col*blocksize+j, row*blocksize+i);
                }
            }
        }
        void changeDiagonalBlock(T* src, int block, int blocksize);
        inline void changeBlock(T* src, int col, int row, int blocksize){
            for(int i = 0; i < blocksize; i++){
                memcpy(this->elementPointer(blocksize*col+i, blocksize*row), src+i*blocksize, blocksize*sizeof(T));
            }
        }
        inline void changeBlock(T* src, int col, int row, int blocksize, sycl::event e){
            for(int i = 0; i < blocksize; i++){
                //memcpy(this->elementPointer(blocksize*col+i, blocksize*row), src+i*blocksize, blocksize*sizeof(T));
                q.submit([&](sycl::handler &h){
                    h.depends_on(e);
                    h.memcpy(this->elementPointer(blocksize*col+i, blocksize*row), src+i*blocksize, blocksize*sizeof(T));
                });
            }
        }
        inline void changeBlock(T* src, int col, int row, int blocksize, sycl::handler &h){
            for(int i = 0; i < blocksize; i++){
                //memcpy(this->elementPointer(blocksize*col+i, blocksize*row), src+i*blocksize, blocksize*sizeof(T));
                h.memcpy(this->elementPointer(blocksize*col+i, blocksize*row), src+i*blocksize, blocksize*sizeof(T));
            }
        }
        void fill();
        void print();
    private:
        T* data;
        int size;
        int cols;
        sycl::queue q;
};