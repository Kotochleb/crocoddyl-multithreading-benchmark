set(${PROJECT_NAME}_EXPERIMENT pure_eigen_openmp pure_eigen_openmp_pointers)

find_package(OpenMP REQUIRED COMPONENTS CXX)

foreach(EXPERIMENT_NAME ${${PROJECT_NAME}_EXPERIMENT})
  add_executable(${EXPERIMENT_NAME} src/${EXPERIMENT_NAME}.cpp)
  target_include_directories(${EXPERIMENT_NAME} PRIVATE "include")
  target_link_libraries(${EXPERIMENT_NAME} Eigen3::Eigen OpenMP::OpenMP_CXX)
endforeach(EXPERIMENT_NAME ${${PROJECT_NAME}_EXPERIMENT})
