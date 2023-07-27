/*
This class implements a packed symmetric matrix data structure, storing the lower half in column-order.
Example: [a_11, a_21, ..., a_n1, a_22, a_32, ..., a_n2, ..., a_nn]
*/
#pragma once
#include <CL/sycl.hpp>
template <typename T> class PackedSymmetricMatrix{
    public:
        PackedSymmetricMatrix(size_t c, sycl::queue &qu);
        PackedSymmetricMatrix(size_t c);
        PackedSymmetricMatrix(T* d, size_t c);
        ~PackedSymmetricMatrix();
        inline int const index(int col, int row) const;
        inline T const element(int col, int row) const{
            return data[size - (cols - col) * ((cols - col) + 1) / 2 + (row - col)];
        }
        inline T* const colPointer(int col) const;
        inline T* const elementPointer(int col, int row) const{
            return &data[size - (cols - col) * ((cols - col) + 1) / 2 + (row - col)];
        }
        void const transferDiagonalBlock(T* dest, int block, int blocksize) const;
        inline void const addDiagToNegative(T* dest, int block, int blocksize) const{
            for(int i = 0; i < blocksize; i++){
                for(int j = i; j < blocksize; j++){
                    dest[i*blocksize+j] = -dest[i*blocksize+j]+this->element(block*blocksize+i, block*blocksize+j);
                }
            }
        }
        void transferBlock(T* dest, int col, int row, int blocksize);
        inline void const addBlockToNegative(T* dest, int col, int row, int blocksize) const{
            for(int i = 0; i < blocksize; i++){
                for(int j = 0; j < blocksize; j++){
                    dest[i*blocksize+j] = -dest[i*blocksize+j]+this->element(col*blocksize+i, row*blocksize+j);
                }
            }
        }
        void const changeDiagonalBlock(T* src, int block, int blocksize) const;
        inline void changeBlock(T* src, int col, int row, int blocksize) const{
            for(int i = 0; i < blocksize; i++){
                memcpy(this->elementPointer(blocksize*col+i, blocksize*row), src+i*blocksize, blocksize*sizeof(T));
            }
        }
        void fill();
        void print();
        void save(std::string fname);
        T* get_data();
    private:
        T* data;
        size_t size;
        int cols;
        sycl::queue q;
};