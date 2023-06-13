default: ldlt

CXX = clang++

#CXX_OPTS = -I/usr/include/linux/ -I/usr/lib/gcc/x86_64-redhat-linux/8/include/

CXX_OPTS = -fopenmp -I/usr/lib/gcc/x86_64-redhat-linux/8/include/ -fsycl -fsycl-targets=nvptx64-nvidia-cuda

ldlt_serial.o: ldlt_serial.cpp ldlt_serial.hpp
	$(CXX) $(CXX_OPTS) -c ldlt_serial.cpp

ldlt_block.o: ldlt_block.cpp ldlt_block.hpp
	$(CXX) $(CXX_OPTS) -c ldlt_block.cpp

main.o: main.cpp ldlt_serial.hpp ldlt_block.hpp
	$(CXX) $(CXX_OPTS) -c main.cpp

ldlt: ldlt_serial.o ldlt_block.o main.o
	$(CXX) $(CXX_OPTS) ldlt_block.o ldlt_serial.o main.o -o ldlt

clean:
	rm -f *.o ldlt
