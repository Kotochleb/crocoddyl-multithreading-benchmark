set(BENCHMARK_NAME tracing)

include(FindLTTngUST REQUIRED)
add_library(tracepoint-provider SHARED src/tracepoint-provider.cpp)
target_link_libraries(tracepoint-provider LTTng::UST)

add_executable($w{BENCHMARK_NAME} src/tracing.cpp)
target_include_directories(${BENCHMARK_NAME} PRIVATE "include")
target_link_libraries(${BENCHMARK_NAME} example-robot-data::example-robot-data
                      crocoddyl::crocoddyl tracepoint-provider)
add_custom_target("benchmarks-tracing" ${BENCHMARK_NAME})
