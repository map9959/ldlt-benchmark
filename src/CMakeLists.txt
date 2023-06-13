add_library(ldlt
	SHARED
	ldlt_serial.hpp
	ldlt_serial.cpp
	)

add_executable(benchmark main.cpp)

target_link_libraries(benchmark
	ldlt
	)
