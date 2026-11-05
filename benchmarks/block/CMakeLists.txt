cmake_minimum_required (VERSION 3.22)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)



if (SWARMIO_BENCH_SOURCE)
    make_benchmark(block-benchmark-module block-bench "${SWARMIO_BENCH_SOURCE}")
    target_link_libraries(block-benchmark-module ${CUDA_CUDA_LIBRARY})
else ()
    make_benchmark (block-benchmark-module block-bench "main.cu;")
endif ()

make_benchmark_choice (block-benchmark block-benchmark-smartio block-benchmark-module)
