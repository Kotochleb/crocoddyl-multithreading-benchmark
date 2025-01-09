## Purpose

This experiment is attempting to reproduce performance loss in singe core parts of Crocoddyl code with only OpenMp and Eigen. It checks how different ways of allocating memory and its flow in the code affect the performance.

The test checks how placement of matrices on heap vs stack can affect performance as well as how synchronization of memory between multi core and single core affects performance.

Variable `mem_sync` enables/disables synchronization of the memory between multi-core and single-core parts.
Variable `mem_type` changes internally used matrices between `Eigen::MatrixXd` and `Eigen::Matrix<double, K, K>`, where `K` is the size of the matrix.

![alt text](figures/test.svg)