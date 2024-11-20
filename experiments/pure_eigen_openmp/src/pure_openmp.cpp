#define EIGEN_DONT_PARALLELIZE

#include <omp.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

template <int N, bool sync_mem>
void benchmark_base(const unsigned int nthreads,
                    const std::string &csv_filename,
                    const std::size_t parallel_size,
                    const std::size_t single_core_size,
                    const unsigned int number_of_trials,
                    const std::size_t buffer_size,
                    const bool enable_opm = true) {
  // Data buffers for timings
  std::vector<long int> timings(number_of_trials);
  std::vector<long int> timing_buffer(buffer_size);

  // Prepare data for parallel computation
  std::vector<std::vector<double>> parallel_data(parallel_size,
                                                 std::vector<double>(N));
  std::vector<double> parallel_results(parallel_size);

  // Prepare data for single core computation
  std::vector<std::vector<double>> single_core_data(single_core_size,
                                                    std::vector<double>(N));
  std::vector<double> single_core_results(single_core_size);

  std::random_device dev;
  std::mt19937 rng(dev());
  std::normal_distribution<double> normal_dist(0.0, 0.5);

  // Allocate data for benchmark
  for (std::size_t i = 0; i < parallel_size; i++) {
    for (std::size_t j = 0; j < N; j++) {
      parallel_data[i][j] = normal_dist(rng);
    }
    parallel_results[i] = 0.0;
  }

  for (std::size_t i = 0; i < single_core_size; i++) {
    for (std::size_t j = 0; j < N; j++) {
      single_core_data[i][j] = normal_dist(rng);
    }
    single_core_results[i] = 0.0;
  }

  omp_set_dynamic(0);

  // Construct problems on single core
  for (std::size_t i = 0; i < number_of_trials; i += buffer_size) {
    for (std::size_t j = 0; j < buffer_size; j++) {
// Solve problem in parallel
#pragma omp parallel default(none) shared(parallel_data, parallel_results) \
    num_threads(nthreads) if (enable_opm) firstprivate(parallel_size)
#pragma omp for nowait schedule(static)
      for (std::size_t k = 0; k < parallel_size; k++) {
        // Clear counter
        parallel_results[k] = 0.0;
        // Sum up data to the result
        for (std::size_t l = 0; l < N; l++) {
          parallel_results[k] += parallel_data[k][l];
        }
        // Modify data to compute different values each time
        for (std::size_t l = 0; l < N; l++) {
          parallel_data[k][l] += parallel_results[k];
          parallel_data[k][l] /= static_cast<double>(N);
        }
      }

      // Start measuring time
      std::chrono::high_resolution_clock::time_point begin =
          std::chrono::high_resolution_clock::now();
      for (std::size_t k = 0; k < single_core_size; k++) {
        for (std::size_t l = 0; l < parallel_size; l++) {
          // Alternate sign to oscillate about 0.0
          for (std::size_t m = 0; m < N; m++) {
            const double sign = (l % 2 ? -1.0 : 1.0);
            if (sync_mem) {
              // Enforce memory synchronization between single threaded and
              // multithreaded
              single_core_data[k][m] += parallel_data[l][m] * sign /
                                        static_cast<double>(parallel_size);
            } else {
              // Roughly equivalent computation but without synchronization
              single_core_data[k][m] +=
                  sign / static_cast<double>(parallel_size);
            }
          }
        }
      }

      for (std::size_t k = 0; k < single_core_size; k++) {
        single_core_results[k] = 0.0;
        for (std::size_t l = 0; l < N; l++) {
          single_core_results[k] += single_core_data[k][l];
        }
        for (std::size_t l = 0; l < N; l++) {
          single_core_data[k][l] += single_core_results[k];
          single_core_data[k][l] /= static_cast<double>(N);
        }
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
  csv << "nthreads,sync_mem,time" << std::endl;
  for (const long int time : timings) {
    csv << nthreads << "," << sync_mem_str << "," << time << std::endl;
  }
  csv.close();
}

template <int N>
inline void benchmark_sync_mem(
    const unsigned int nthreads, const std::string &csv_filename,
    const bool sync_mem, const std::size_t parallel_size,
    const std::size_t single_core_size, const unsigned int number_of_trials,
    const std::size_t buffer_size, const bool enable_opm = true) {
  // Usage of templates ensures that the if statement will be ignored by
  // compiler making the code more efficient and preventing any chance of
  // incorrect branch predictions.
  if (sync_mem) {
    benchmark_base<N, true>(nthreads, csv_filename, parallel_size,
                            single_core_size, number_of_trials, buffer_size,
                            enable_opm);
  } else {
    benchmark_base<N, false>(nthreads, csv_filename, parallel_size,
                             single_core_size, number_of_trials, buffer_size,
                             enable_opm);
  }
}

int main(int argc, char *argv[]) {
  // Ensure all arguments are passed
  if (argc != 5) {
    std::cerr << "Incorrect number of arguments! Example usage 'multithreading "
                 "<number of cores>' '<csv output file path>' '<sync memory "
                 "true/false>' '<Enable "
                 "OpenMP true/false>'.";
    return 1;
  }
  // Parse arguments
  const unsigned int nthreads = static_cast<unsigned int>(std::stoi(argv[1]));
  const std::string csv_filename = std::string(argv[2]);
  const bool sync_mem = std::string(argv[3]) == "true";
  const bool enable_opm = std::string(argv[4]) == "true";

  // Define constant parameters
  const std::size_t parallel_size = 100;
  const std::size_t single_core_size = 30;
  const std::size_t buffer_size = 20;
  const unsigned int number_of_trials = 100;
  // Size of matrices
  constexpr int N = 20 * 20;

  benchmark_sync_mem<N>(nthreads, csv_filename, sync_mem, parallel_size,
                        single_core_size, number_of_trials, buffer_size,
                        enable_opm);

  return 0;
}
