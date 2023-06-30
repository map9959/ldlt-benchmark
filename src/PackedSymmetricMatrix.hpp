/*
This class implements a packed symmetric matrix data structure, storing the lower half in column-order.
Example: [a_11, a_21, ..., a_n1, a_22, a_32, ..., a_n2, ..., a_nn]
*/
#pragma once
class PackedSymmetricMatrix{
    public:
        PackedSymmetricMatrix(int c);
        ~PackedSymmetricMatrix();
        inline int index(int col, int row);
        inline float element(int col, int row);
        inline float* colPointer(int col);
        inline float* elementPointer(int col, int row);
        void transferDiagonalBlock(float* dest, int block, int blocksize);
        void transferBlock(float* dest, int col, int row, int blocksize);
        void addBlockToNegative(float* dest, int col, int row, int blocksize);
        void changeDiagonalBlock(float* src, int block, int blocksize);
        void changeBlock(float* src, int col, int row, int blocksize);
        void fill();
        void print();
    private:
        float* data;
        int size;
        int cols;
};