default: ldlt.out

CXX = clang++

#CXX_OPTS = -I/usr/include/linux/ -I/usr/lib/gcc/x86_64-redhat-linux/8/include/

CXX_OPTS = -I/mnt/home/mpaciullo/intel/oneapi/compiler/2023.0.0/linux/compiler/include/ -fopenmp -fsycl -O3 -fsycl-targets=nvptx64-nvidia-cuda

PackedSymmetricMatrix.o: PackedSymmetricMatrix.cpp PackedSymmetricMatrix.hpp
	$(CXX) $(CXX_OPTS) -c PackedSymmetricMatrix.cpp

ldlt_serial.o: ldlt_serial.cpp ldlt_serial.hpp
	$(CXX) $(CXX_OPTS) -c ldlt_serial.cpp

ldlt_block.o: ldlt_block.cpp ldlt_block.hpp
	$(CXX) $(CXX_OPTS) -c ldlt_block.cpp

main.o: main.cpp ldlt_serial.hpp ldlt_block.hpp
	$(CXX) $(CXX_OPTS) -c main.cpp

ldlt.out: PackedSymmetricMatrix.o ldlt_serial.o ldlt_block.o main.o
	$(CXX) $(CXX_OPTS) PackedSymmetricMatrix.o ldlt_block.o ldlt_serial.o main.o -o ldlt.out

run: ldlt.out
	./ldlt.out

clean:
	rm -f *.o ldlt ldlt.out
