
#include <omp.h>

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "crocoddyl/core/utils/file-io.hpp"
#include "crocoddyl/core/utils/timer.hpp"

template <typename MatrixType, int N, bool sync_mem>
void benchmark_base(const unsigned int nthreads,
                    const std::string &csv_filename,
                    const std::string &alignment, const std::size_t vector_size,
                    const std::size_t number_of_trials) {
  const std::string sync_mem_str = sync_mem ? "true" : "false";

  CsvStream csv(csv_filename);
  csv << "nthreads" << "sync_mem" << "alignment" << "time" << csv.endl;

  // Prepare data for parallel computation
  std::vector<MatrixType, Eigen::aligned_allocator<MatrixType>> data_tensor(N);

  const MatrixType eps = 1e-6 * MatrixType::Identity(N, N);
  MatrixType sym_tmp = MatrixType::Zero(N, N);

  for (std::size_t i = 0; i < N; i++) {
    data_tensor[i] = MatrixType::Zero(N, N);
  }

  // Prepare data for single core computation
  MatrixType A = MatrixType::Zero(N, N);
  MatrixType A_sym_tmp = MatrixType::Zero(N, N);
  Eigen::LLT<MatrixType> llt;

  // Create timer
  crocoddyl::Timer timer;

  for (std::size_t i = 0; i < number_of_trials; i++) {
    for (std::size_t i = 0; i < N; i++) {
      data_tensor[i] = MatrixType::Random(N, N);
      sym_tmp.noalias() = data_tensor[i] * data_tensor[i].transpose();
      data_tensor[i] = sym_tmp + eps;
    }

#pragma omp parallel for num_threads(nthreads)
    for (std::size_t j = 0; j < N; j++) {
      Eigen::LLT<MatrixType> llt;
      llt.compute(data_tensor[j]);
    }

    // Make the matrix symmetric positive definite
    A = MatrixType::Random(N, N);
    A_sym_tmp = A * A.transpose();
    A = A_sym_tmp + eps;

    // Measure single core time
    timer.reset();
    // Enforce memory synchronization between single threaded and multithreaded
    if (sync_mem) {
      for (std::size_t j = 0; j < N; j++) {
        A += data_tensor[j] * 1.0 / static_cast<double>(N);
      }
    }
    llt.compute(A);
    csv << nthreads << sync_mem_str << alignment << timer.get_us_duration()
        << csv.endl;
  }
}

template <typename MatrixType, int N>
inline void benchmark_sync_mem(const unsigned int nthreads,
                               const std::string &csv_filename,
                               const bool sync_mem,
                               const std::string &alignment,
                               const std::size_t vector_size,
                               const std::size_t number_of_trials) {
  if (sync_mem) {
    benchmark_base<MatrixType, N, true>(nthreads, csv_filename, alignment,
                                        vector_size, number_of_trials);
  } else {
    benchmark_base<MatrixType, N, false>(nthreads, csv_filename, alignment,
                                         vector_size, number_of_trials);
  }
}

int main(int argc, char *argv[]) {
  // Obtain number of cores as argument
  if (argc != 5) {
    std::cerr << "Incorrect number of arguments! Example usage 'multithreading "
                 "<number of cores> <csv output file path>' '<sync memory "
                 "true/false>' <memory alignment dynamic/static>.";
    return 1;
  }
  const unsigned int nthreads = std::stoi(argv[1]);
  const std::string csv_filename = std::string(argv[2]);
  const bool sync_mem = std::string(argv[3]) == "true";
  const std::string alignment = std::string(argv[4]);

  const std::size_t vector_size = 150;
  const unsigned int number_of_trials = 1e2;
  // Size of matrices
  constexpr int K = 50;

  // Test dynamically allocated arrays
  if (alignment == "dynamic") {
    benchmark_sync_mem<Eigen::MatrixXd, K>(nthreads, csv_filename, sync_mem,
                                           alignment, vector_size,
                                           number_of_trials);
  } else if (alignment == "static") {
    benchmark_sync_mem<Eigen::Matrix<double, K, K>, K>(
        nthreads, csv_filename, sync_mem, alignment, vector_size,
        number_of_trials);
  }

  return 0;
}
