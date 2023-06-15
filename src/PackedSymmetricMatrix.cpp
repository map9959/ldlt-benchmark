#include "PackedSymmetricMatrix.hpp"
#include "ldlt_block.hpp"
#include <iostream>
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
    return size - (cols - col) * ((cols - col) + 1) / 2 + row;
}
inline float PackedSymmetricMatrix::element(int col, int row){
    return data[index(col, row)];
}
void PackedSymmetricMatrix::print(){
    for(int i = 0; i < cols; i++){
        for(int j = 0; j < i; j++){
            std::cout << element(j, i) << " ";
        }
        for(int j = i; j < cols; j++){
            std::cout << element(i, j) << " ";
        }
        std::cout << '\n';
    }
}
void PackedSymmetricMatrix::fill(){
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<float> random_float(-5,5);
    #pragma omp parallel for num_threads(NUM_THREADS)
    for(int i = 0; i < size; i++){
        data[i] = random_float(e);
    }
}