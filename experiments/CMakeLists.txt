set(${PROJECT_NAME}_EXPERIMENT ddp_separated_memory flattened_ddp
                               pure_eigen_openmp)

foreach(EXPERIMENT_NAME ${${PROJECT_NAME}_EXPERIMENT})
  add_subdirectory(${EXPERIMENT_NAME})
endforeach(EXPERIMENT_NAME ${${PROJECT_NAME}_EXPERIMENT})
