set(${PROJECT_NAME}_EXPERIMENT pure_eigen_openmp pure_eigen_openmp_pointers)

foreach(EXPERIMENT_NAME ${${PROJECT_NAME}_EXPERIMENT})
  add_executable(${EXPERIMENT_NAME} src/${EXPERIMENT_NAME}.cpp)
  target_include_directories(${EXPERIMENT_NAME} PRIVATE "include")
  target_link_libraries(${EXPERIMENT_NAME} crocoddyl::crocoddyl)
endforeach(EXPERIMENT_NAME ${${PROJECT_NAME}_EXPERIMENT})
