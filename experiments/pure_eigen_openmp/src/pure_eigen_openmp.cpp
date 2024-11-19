#define EIGEN_DONT_PARALLELIZE

#include <omp.h>

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

template <int CompileMatrixSize, int N, bool sync_mem>
void benchmark_base(
    const unsigned int nthreads, const std::string &csv_filename,
    const std::string &alignment, const std::size_t parallel_size,
    const std::size_t single_core_size, const unsigned int number_of_trials,
    const std::size_t buffer_size, const bool enable_opm = true) {
  typedef Eigen::Matrix<double, CompileMatrixSize, CompileMatrixSize>
      MatrixType;

  // Data buffers for timings
  std::vector<long int> timings(number_of_trials);
  std::vector<long int> timing_buffer(buffer_size);

  // Prepare data for parallel computation
  std::vector<MatrixType, Eigen::aligned_allocator<MatrixType>> parallel_data(
      parallel_size);
  std::vector<MatrixType, Eigen::aligned_allocator<MatrixType>> sym_tmp(
      parallel_size);
  std::vector<Eigen::LLT<MatrixType>,
              Eigen::aligned_allocator<Eigen::LLT<MatrixType>>>
      parallel_llt(parallel_size);

  // Prepare data for single core computation
  std::vector<MatrixType, Eigen::aligned_allocator<MatrixType>>
      single_core_data(single_core_size);
  std::vector<Eigen::LLT<MatrixType>,
              Eigen::aligned_allocator<Eigen::LLT<MatrixType>>>
      single_core_llt(single_core_size);

  // Create constants
  const MatrixType eps = 1e-6 * MatrixType::Identity(N, N);

  // Allocate data for benchmark
  for (std::size_t i = 0; i < parallel_size; i++) {
    parallel_data[i] = MatrixType::Random(N, N);
    sym_tmp[i] = MatrixType::Zero(N, N);
    parallel_llt[i] = Eigen::LLT<MatrixType>(N);
  }

  for (std::size_t i = 0; i < single_core_size; i++) {
    single_core_data[i] = MatrixType::Random(N, N);
    single_core_llt[i] = Eigen::LLT<MatrixType>(N);
  }

  // Prepare data for single core computation
  MatrixType single_core_sym_tmp = MatrixType::Zero(N, N);
  Eigen::LLT<MatrixType> llt;

  // omp_set_dynamic(0);

  // Construct problems on single core
  for (std::size_t i = 0; i < number_of_trials; i += buffer_size) {
    for (std::size_t j = 0; j < buffer_size; j++) {
// Solve problem in parallel
#pragma omp parallel for num_threads(nthreads) if (enable_opm) \
    shared(parallel_data) firstprivate(eps) schedule(static)
      for (std::size_t k = 0; k < parallel_size; k++) {
        sym_tmp[k].noalias() = parallel_data[k] * parallel_data[k].transpose();
        sym_tmp[k].normalize();
        parallel_data[k] = sym_tmp[k] + eps;
        parallel_llt[k].compute(parallel_data[k]);
      }

      // Make the matrix symmetric positive definite
      for (std::size_t k = 0; k < single_core_size; k++) {
        single_core_sym_tmp.noalias() =
            single_core_data[k] * single_core_data[k].transpose();
        single_core_sym_tmp.normalize();
        single_core_data[k] = single_core_sym_tmp + eps;
      }

      // Start measuring time
      std::chrono::high_resolution_clock::time_point begin =
          std::chrono::high_resolution_clock::now();
      // Enforce memory synchronization between single threaded and
      // multithreaded
      if (sync_mem) {
        for (std::size_t k = 0; k < single_core_size; k++) {
          for (std::size_t l = 0; l < parallel_size; l++) {
            single_core_data[k] +=
                parallel_data[l] * 1.0 / static_cast<double>(parallel_size);
          }
        }
      }
      for (std ::size_t k = 0; k < single_core_size; k++) {
        single_core_llt[k].compute(single_core_data[k]);
      }
      // Stop measuring time
      std::chrono::high_resolution_clock::time_point end =
          std::chrono::high_resolution_clock::now();
      timing_buffer[j] =
          std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
              .count();
    }

    std::copy(timing_buffer.begin(), timing_buffer.end(), timings.begin() + i);
  }

  const std::string sync_mem_str = sync_mem ? "true" : "false";

  // Save data to CSV
  std::ofstream csv;
  csv.open(csv_filename);
  csv << "nthreads,sync_mem,alignment,time," << std::endl;
  for (const long int time : timings) {
    csv << nthreads << "," << sync_mem_str << "," << alignment << "," << time
        << "," << std::endl;
  }
  csv.close();
}

template <int CompileMatrixSize, int N>
inline void benchmark_sync_mem(
    const unsigned int nthreads, const std::string &csv_filename,
    const bool sync_mem, const std::string &alignment,
    const std::size_t parallel_size, const std::size_t single_core_size,
    const unsigned int number_of_trials, const std::size_t buffer_size,
    const bool enable_opm = true) {
  // Usage of templates ensures that the if statement will be ignored by
  // compiler making the code more efficient and preventing any chance of
  // incorrect branch predictions.
  if (sync_mem) {
    benchmark_base<CompileMatrixSize, N, true>(
        nthreads, csv_filename, alignment, parallel_size, single_core_size,
        number_of_trials, buffer_size, enable_opm);
  } else {
    benchmark_base<CompileMatrixSize, N, false>(
        nthreads, csv_filename, alignment, parallel_size, single_core_size,
        number_of_trials, buffer_size, enable_opm);
  }
}

int main(int argc, char *argv[]) {
  // Ensure all arguments are passed
  if (argc != 6) {
    std::cerr << "Incorrect number of arguments! Example usage 'multithreading "
                 "<number of cores>' '<csv output file path>' '<sync memory "
                 "true/false>' '<memory alignment dynamic/static>' '<Enable "
                 "OpenMP true/false>'.";
    return 1;
  }
  // Parse arguments
  const unsigned int nthreads = static_cast<unsigned int>(std::stoi(argv[1]));
  const std::string csv_filename = std::string(argv[2]);
  const bool sync_mem = std::string(argv[3]) == "true";
  const std::string alignment = std::string(argv[4]);
  const bool enable_opm = std::string(argv[5]) == "true";

  // Define constant parameters
  const std::size_t parallel_size = 150;
  const std::size_t single_core_size = 30;
  const std::size_t buffer_size = 20;
  const unsigned int number_of_trials = 100;
  // Size of matrices
  constexpr int K = 20;

  // Test dynamically allocated arrays vs statically allocated
  if (alignment == "dynamic") {
    benchmark_sync_mem<Eigen::Dynamic, K>(
        nthreads, csv_filename, sync_mem, alignment, parallel_size,
        single_core_size, number_of_trials, buffer_size, enable_opm);
  } else if (alignment == "static") {
    benchmark_sync_mem<K, K>(nthreads, csv_filename, sync_mem, alignment,
                             parallel_size, single_core_size, number_of_trials,
                             buffer_size, enable_opm);
  }

  return 0;
}
