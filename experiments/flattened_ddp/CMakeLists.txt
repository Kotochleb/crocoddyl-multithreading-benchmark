set(BENCHMARK_NAME flattened_ddp_test)

add_executable(${BENCHMARK_NAME} src/flattened_ddp_test.cpp src/ddp.cpp)
target_include_directories(${BENCHMARK_NAME} PRIVATE "include")
target_link_libraries(${BENCHMARK_NAME} example-robot-data::example-robot-data
                      crocoddyl::crocoddyl)
