set(BENCHMARK_NAME multithreading)

add_executable(${BENCHMARK_NAME} src/multithreading.cpp src/ddp.cpp)
target_include_directories(${BENCHMARK_NAME} PRIVATE "include")
target_link_libraries(${BENCHMARK_NAME} example-robot-data::example-robot-data
                      crocoddyl::crocoddyl)
