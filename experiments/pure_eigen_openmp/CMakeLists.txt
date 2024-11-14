set(BENCHMARK_NAME pure_eigen_openmp)

add_executable(${BENCHMARK_NAME} src/pure_eigen_openmp.cpp)
target_include_directories(${BENCHMARK_NAME} PRIVATE "include")
target_link_libraries(${BENCHMARK_NAME} crocoddyl::crocoddyl)
