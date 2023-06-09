ADD_LIBRARY(ldlt
	SHARED
	ldlt_serial.hpp
	ldlt_serial.cpp
	)

ADD_EXECUTABLE(benchmark main.cpp)

TARGET_LINK_LIBRARIES(benchmark
	ldlt
	)
