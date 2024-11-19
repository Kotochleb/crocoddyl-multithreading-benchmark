set(${PROJECT_NAME}_EXPERIMENT pure_eigen_openmp pure_eigen_openmp_pointers
                               pure_openmp)

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Eigen3 REQUIRED)

if(NOT OpenMP_CXX_FOUND)
  message(FATAL_ERROR "OpenMP not found")
endif()

foreach(EXPERIMENT_NAME ${${PROJECT_NAME}_EXPERIMENT})
  add_executable(${EXPERIMENT_NAME} src/${EXPERIMENT_NAME}.cpp)
  target_include_directories(${EXPERIMENT_NAME} PRIVATE "include")
  target_link_libraries(${EXPERIMENT_NAME} Eigen3::Eigen OpenMP::OpenMP_CXX
                        stdc++)
endforeach(EXPERIMENT_NAME ${${PROJECT_NAME}_EXPERIMENT})
