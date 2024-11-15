#include <omp.h>

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "crocoddyl/core/utils/file-io.hpp"
#include "crocoddyl/core/utils/timer.hpp"

template <int CompileMatrixSize, int N, bool sync_mem>
void benchmark_base(const unsigned int nthreads,
                    const std::string &csv_filename,
                    const std::string &alignment,
                    const std::size_t parallel_size,
                    const std::size_t single_core_size,
                    const std::size_t number_of_trials) {
  typedef Eigen::Matrix<double, CompileMatrixSize, CompileMatrixSize>
      MatrixType;
  const std::string sync_mem_str = sync_mem ? "true" : "false";

  CsvStream csv(csv_filename);
  csv << "nthreads"
      << "sync_mem"
      << "alignment"
      << "time" << csv.endl;

  // Prepare data for parallel computation
  std::vector<MatrixType, Eigen::aligned_allocator<MatrixType>> parallel_data(
      parallel_size);
  std::vector<MatrixType, Eigen::aligned_allocator<MatrixType>> sym_tmp(
      parallel_size);
  std::vector<Eigen::LLT<MatrixType>,
              Eigen::aligned_allocator<Eigen::LLT<MatrixType>>>
      parallel_ltt(parallel_size);

  // Prepare data for single core computation
  std::vector<MatrixType, Eigen::aligned_allocator<MatrixType>>
      single_core_data(single_core_size);
  std::vector<Eigen::LLT<MatrixType>,
              Eigen::aligned_allocator<Eigen::LLT<MatrixType>>>
      single_core_ltt(single_core_size);

  // Create constants
  const MatrixType eps = 1e-6 * MatrixType::Identity(N, N);

  // Allocate data for benchmark
  for (std::size_t i = 0; i < parallel_size; i++) {
    parallel_data[i] = MatrixType::Zero(N, N);
    sym_tmp[i] = MatrixType::Zero(N, N);
    parallel_ltt[i] = Eigen::LLT<MatrixType>(N);
  }

  for (std::size_t i = 0; i < single_core_size; i++) {
    single_core_data[i] = MatrixType::Zero(N, N);
    single_core_ltt[i] = Eigen::LLT<MatrixType>(N);
  }

  // Prepare data for single core computation
  MatrixType single_core_sym_tmp = MatrixType::Zero(N, N);
  Eigen::LLT<MatrixType> llt;

  // Create timer
  crocoddyl::Timer timer;

  // Construct problems on single core
  for (std::size_t i = 0; i < number_of_trials; i++) {
// Solve problem in parallel
#pragma omp parallel for num_threads(nthreads) firstprivate(eps)
    for (std::size_t j = 0; j < N; j++) {
      parallel_data[j] = MatrixType::Random(N, N);
      sym_tmp[j].noalias() = parallel_data[j] * parallel_data[j].transpose();
      parallel_data[j] = sym_tmp[j] + eps;
      parallel_ltt[j].compute(parallel_data[j]);
    }

    // Make the matrix symmetric positive definite
    for (std::size_t j = 0; j < single_core_size; j++) {
      single_core_data[j] = MatrixType::Random(N, N);
      single_core_sym_tmp.noalias() =
          single_core_data[j] * single_core_data[j].transpose();
      single_core_data[j] = single_core_sym_tmp + eps;
    }

    // Measure single core time
    timer.reset();
    // Enforce memory synchronization between single threaded and multithreaded
    if (sync_mem) {
      for (std ::size_t j = 0; j < single_core_size; j++) {
        for (std::size_t k = 0; k < parallel_size; k++) {
          single_core_data[j] +=
              parallel_data[k] * 1.0 / static_cast<double>(parallel_size);
        }
      }
    }
    for (std ::size_t j = 0; j < single_core_size; j++) {
      single_core_ltt[j].compute(single_core_data[j]);
    }

    const double comp_time = timer.get_us_duration();
    csv << nthreads << sync_mem_str << alignment << comp_time << csv.endl;
  }
}

template <int CompileMatrixSize, int N>
inline void benchmark_sync_mem(const unsigned int nthreads,
                               const std::string &csv_filename,
                               const bool sync_mem,
                               const std::string &alignment,
                               const std::size_t parallel_size,
                               const std::size_t single_core_size,
                               const std::size_t number_of_trials) {
  // Usage of templates ensures that the if statement will be ignored by
  // compiler making the code more efficient and preventing any chance of
  // incorrect branch predictions.
  if (sync_mem) {
    benchmark_base<CompileMatrixSize, N, true>(
        nthreads, csv_filename, alignment, parallel_size, single_core_size,
        number_of_trials);
  } else {
    benchmark_base<CompileMatrixSize, N, false>(
        nthreads, csv_filename, alignment, parallel_size, single_core_size,
        number_of_trials);
  }
}

int main(int argc, char *argv[]) {
  // Ensure all arguments are passed
  if (argc != 5) {
    std::cerr << "Incorrect number of arguments! Example usage 'multithreading "
                 "<number of cores> <csv output file path>' '<sync memory "
                 "true/false>' <memory alignment dynamic/static>.";
    return 1;
  }
  // Parse arguments
  const unsigned int nthreads = std::stoi(argv[1]);
  const std::string csv_filename = std::string(argv[2]);
  const bool sync_mem = std::string(argv[3]) == "true";
  const std::string alignment = std::string(argv[4]);

  // Define constant parameters
  const std::size_t parallel_size = 150;
  const std::size_t single_core_size = 30;
  const unsigned int number_of_trials = 1e2;
  // Size of matrices
  constexpr int K = 20;

  // Test dynamically allocated arrays vs statically allocated
  if (alignment == "dynamic") {
    benchmark_sync_mem<Eigen::Dynamic, K>(nthreads, csv_filename, sync_mem,
                                          alignment, parallel_size,
                                          single_core_size, number_of_trials);
  } else if (alignment == "static") {
    benchmark_sync_mem<K, K>(nthreads, csv_filename, sync_mem, alignment,
                             parallel_size, single_core_size, number_of_trials);
  }

  return 0;
}
