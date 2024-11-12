#define CROCODDYL_WITH_MULTITHREADING

#define EIGEN_DONT_PARALLELIZE

#ifdef CROCODDYL_WITH_MULTITHREADING
#include <omp.h>
#endif

#define CROCODDYL_WITH_NTHREADS 19

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "crocoddyl/core/utils/file-io.hpp"
#include "crocoddyl/core/utils/timer.hpp"

int main() {
  const size_t N = 200;  // Number of parallely computer matrices
  const size_t K = 50;   // Size of the matrix
  unsigned int T = 1e3;  // number of trials
  const std::string csv_filename = "/tmp/dummy_test.csv";
  CsvStream csv(csv_filename);
  csv << "fn_name" << "nthreads" << "time" << csv.endl;

  std::cout << "Starting the loop..." << std::endl;

  auto ithreads = std::vector<int>(CROCODDYL_WITH_NTHREADS);
  std::generate(ithreads.begin(), ithreads.end(), [] {
    static int i = 1;
    return i++;
  });
  std::random_device rd;
  std::mt19937 g(rd());

  for (unsigned int i = 0; i < T; ++i) {
    std::shuffle(ithreads.begin(), ithreads.end(), g);
    for (const auto ithread : ithreads) {
      std::vector<std::shared_ptr<Eigen::MatrixXd>> my_data(N);

      for (size_t j = 0; j < N; j++) {
        my_data[j] =
            std::make_shared<Eigen::MatrixXd>(Eigen::MatrixXd::Random(K, K));
        *my_data[j] =
            (*my_data[j]) *
            (my_data[j]->transpose());  // Make it symmetric positive definite
        *my_data[j] += 1e-6 * Eigen::MatrixXd::Identity(K, K);
      }

#pragma omp parallel for num_threads(ithread)
      for (size_t j = 0; j < N; j++) {
        Eigen::LLT<Eigen::MatrixXd> llt;
        llt.compute(*my_data[j]);
      }

      // ddp.solve(xs, us, 1, true);
      Eigen::MatrixXd A = Eigen::MatrixXd::Random(K, K);
      A = A * A.transpose();  // Make it symmetric positive definite
      A += 1e-6 * Eigen::MatrixXd::Identity(K, K);
      Eigen::LLT<Eigen::MatrixXd> llt;

      // Benchmark Eigen
      crocoddyl::Timer timer;
      for (size_t j = 0; j < N; j++) {
        A += *my_data[j] * 1.0 / static_cast<double>(N);
      }
      llt.compute(A);
      csv << "single_core" << ithread << timer.get_us_duration() << csv.endl;
    }
    std::cout << "Iteration: " << i << " of " << T << std::endl;
  }

  std::cout << "done" << std::endl;

  return 0;
}
