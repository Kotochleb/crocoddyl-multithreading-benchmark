cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Project properties
set(PROJECT_NAME crocoddyl-multithreading-benchmark)
set(PROJECT_DESCRIPTION "Performance benchmark for Crocoddyl")
set(PROJECT_URL https://github.com/kotochleb/${PROJECT_NAME})

# Project configuration
option(DISABLE_EIGEN_PARALELLIZATION
       "Disable Eigen parallelization with OpenMP" ON)
set(DISABLE_EIGEN_PARALELLIZATION TRUE)

# Project dependencies
if(DISABLE_EIGEN_PARALELLIZATION)
  add_definitions(-DEIGEN_DONT_PARALLELIZE)
endif()

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(EXISTS "${JRL_CMAKE_MODULES}/base.cmake")
  message(STATUS "JRL cmakemodules found in 'cmake/' git submodule")
else()
  find_package(jrl-cmakemodules QUIET CONFIG)
  if(jrl-cmakemodules_FOUND)
    get_property(
      JRL_CMAKE_MODULES
      TARGET jrl-cmakemodules::jrl-cmakemodules
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "JRL cmakemodules found on system at ${JRL_CMAKE_MODULES}")
  elseif(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    message(
      FATAL_ERROR
        "\nCan't find jrl-cmakemodules. Please either:\n"
        "  - use git submodule: 'git submodule update --init'\n"
        "  - or install https://github.com/jrl-umi3218/jrl-cmakemodules\n"
        "  - or upgrade your CMake version to >= 3.14 to allow automatic fetching\n"
    )
  else()
    message(STATUS "JRL cmakemodules not found. Let's fetch it.")
    include(FetchContent)
    FetchContent_Declare(
      "jrl-cmakemodules"
      GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  endif()
endif()

# JRL-cmakemodule setup
include("${JRL_CMAKE_MODULES}/base.cmake")
include("${JRL_CMAKE_MODULES}/boost.cmake")
include("${JRL_CMAKE_MODULES}/apple.cmake")

# Project definition
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

add_compile_options(-Wno-error)

apply_default_apple_configuration()

add_project_dependency(OpenMP REQUIRED)
add_project_dependency(crocoddyl REQUIRED)
add_project_dependency(pinocchio REQUIRED)
add_project_dependency(hpp-fcl REQUIRED)
add_project_dependency(example-robot-data REQUIRED)

include_directories(common/include)

add_subdirectory(experiments)
