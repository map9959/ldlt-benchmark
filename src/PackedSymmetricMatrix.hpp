/*
This class implements a packed symmetric matrix data structure, storing the lower half in column-order.
Example: [a_11, a_21, ..., a_n1, a_22, a_32, ..., a_n2, ..., a_nn]
*/
class PackedSymmetricMatrix{
    public:
        PackedSymmetricMatrix(int c);
        ~PackedSymmetricMatrix();
        inline int index(int row, int col);
        inline float element(int row, int col);
        inline float* colPointer(int col);
        inline float* elementPointer(int col, int row);
        void fill();
        void print();
    private:
        float* data;
        int size;
        int cols;
};