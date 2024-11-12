///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#define CROCODDYL_WITH_MULTITHREADING

#ifdef CROCODDYL_WITH_MULTITHREADING
#include <omp.h>
#endif
#undef CROCODDYL_WITH_NTHREADS
#ifndef CROCODDYL_WITH_NTHREADS
#define CROCODDYL_WITH_NTHREADS 19
#endif

#ifdef CROCODDYL_WITH_CODEGEN
#include "crocoddyl/core/codegen/action-base.hpp"
#endif

#include "arm-kinova.hpp"
#include "arm.hpp"
// #include "crocoddyl/core/solvers/ddp.hpp"
#include <unistd.h>

#include <random>

#include "crocoddyl/core/utils/file-io.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include "ddp.hpp"
#include "legged-robots.hpp"

#define STDDEV(vec)                               \
  std::sqrt(((vec - vec.mean())).square().sum() / \
            (static_cast<double>(vec.size()) - 1))
#define AVG(vec) (vec.mean())

void print_benchmark(RobotEENames robot) {
  unsigned int N = 100;  // number of nodes
  unsigned int T = 3e3;  // number of trials

  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > runningModels;
  boost::shared_ptr<crocoddyl::ActionModelAbstract> terminalModel;
  boost::shared_ptr<crocoddyl::StateMultibody> state;
  Eigen::VectorXd x0(15);
  x0.setZero();
  x0.head(6) << 1.5707, 2.618, -1.5707, 3.1415, 2.618, 0.0;

  // Building the running and terminal models
  for (unsigned int i = 0; i < N; ++i) {
    boost::shared_ptr<crocoddyl::ActionModelAbstract> runningModel;
    if (robot.robot_name == "Talos_arm") {
      crocoddyl::benchmark::build_arm_action_models(runningModel,
                                                    terminalModel);
    } else if (robot.robot_name == "Kinova_arm") {
      crocoddyl::benchmark::build_arm_kinova_action_models(runningModel,
                                                           terminalModel);
    } else {
      crocoddyl::benchmark::build_contact_action_models(robot, runningModel,
                                                        terminalModel);
    }
    state = boost::static_pointer_cast<crocoddyl::StateMultibody>(
        runningModel->get_state());

    Eigen::VectorXd default_state(state->get_nq() + state->get_nv());

    boost::shared_ptr<crocoddyl::IntegratedActionModelEulerTpl<double> > rm =
        boost::static_pointer_cast<
            crocoddyl::IntegratedActionModelEulerTpl<double> >(runningModel);

    boost::shared_ptr<
        crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<double> >
        dm = boost::static_pointer_cast<
            crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<double> >(
            rm->get_differential());
    // default_state
    //     << dm->get_pinocchio().referenceConfigurations[robot.reference_conf],
    //     Eigen::VectorXd::Zero(state->get_nv());

    runningModels.push_back(runningModel);
  }

  std::cout << "NQ: " << state->get_nq() << std::endl;
  std::cout << "Number of nodes: " << N << std::endl << std::endl;

  boost::shared_ptr<crocoddyl::ShootingProblem> problem =
      boost::make_shared<crocoddyl::ShootingProblem>(x0, runningModels,
                                                     terminalModel);
  // Computing the warm-start
  std::vector<Eigen::VectorXd> xs(N + 1, x0);
  std::vector<Eigen::VectorXd> us(
      N, Eigen::VectorXd::Zero(runningModels[0]->get_nu()));

  // crocoddyl::SolverDDP ddp(problem);
  colmpc_test::SolverDDP ddp(problem);
  ddp.setCandidate(xs, us, false);

  /******************************* create csv file
   * *******************************/
  const std::string csv_filename = "/tmp/" + robot.robot_name + "_" +
                                   std::to_string(state->get_nq()) +
                                   "DoF.bench";
  CsvStream csv(csv_filename);
  csv << "fn_name" << "nthreads" << "time" << csv.endl;

  /*******************************************************************************/
  /*********************************** TIMINGS
   * ***********************************/
  Eigen::ArrayXd duration(T);
  Eigen::ArrayXd avg(CROCODDYL_WITH_NTHREADS);
  Eigen::ArrayXd stddev(CROCODDYL_WITH_NTHREADS);

  std::cout << "ddp.calcDiff" << std::endl;

  auto ithreads = std::vector<int>(CROCODDYL_WITH_NTHREADS);
  std::generate(ithreads.begin(), ithreads.end(), [] {
    static int i = 1;
    return i++;
  });
  std::random_device rd;
  std::mt19937 g(rd());

  const int ms = 10;

  // ddp.calcDiff timings
  crocoddyl::Timer timer;
  for (unsigned int i = 0; i < T; ++i) {
    // std::shuffle(ithreads.begin(), ithreads.end(), g);
    for (const auto ithread : ithreads) {
      for (int k = 0; k < 20; k++) {
        ddp.get_problem()->set_nthreads(ithread);

        // usleep(ms * 1000);

        timer.reset();
        ddp.solve(xs, us, 1, true);
        csv << "ddp.solve()" << ddp.get_problem()->get_nthreads()
            << timer.get_us_duration() << csv.endl;
        // usleep(ms * 1000);
        timer.reset();
        ddp.calcDiff();
        csv << "ddp.calcDiff()" << (ddp.get_problem()->get_nthreads())
            << timer.get_us_duration() << csv.endl;
      }
    }
  }

  std::cout << "done" << std::endl;
}

int main() {
  // crocoddyl::Stopwatch& swatch = crocoddyl::getProfiler();
  // swatch.enable_profiler();

  std::vector<std::string> contact_names;
  std::vector<crocoddyl::ContactType> contact_types;

  std::cout << "********************  Kinova Arm  ******************"
            << std::endl;
  RobotEENames kinovaArm(
      "Kinova_arm", contact_names, contact_types,
      EXAMPLE_ROBOT_DATA_MODEL_DIR "/kinova_description/robots/kinova.urdf",
      EXAMPLE_ROBOT_DATA_MODEL_DIR "/kinova_description/srdf/kinova.srdf",
      "gripper_left_joint", "arm_up");

  print_benchmark(kinovaArm);

  // swatch.report_all(3);

  // // Quadruped Solo Benchmarks
  // std::cout << "********************Quadruped Solo******************"
  //           << std::endl;
  // contact_names.clear();
  // contact_types.clear();
  // contact_names.push_back("FR_KFE");
  // contact_names.push_back("HL_KFE");
  // contact_types.push_back(crocoddyl::Contact3D);
  // contact_types.push_back(crocoddyl::Contact3D);
  // RobotEENames quadrupedSolo(
  //     "Solo", contact_names, contact_types,
  //     EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/robots/solo.urdf",
  //     EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/srdf/solo.srdf",
  //     "HL_KFE", "standing");

  // print_benchmark(quadrupedSolo);

  // // Quadruped Anymal Benchmarks
  // std::cout << "********************Quadruped Anymal******************"
  //           << std::endl;
  // contact_names.clear();
  // contact_types.clear();
  // contact_names.push_back("RF_KFE");
  // contact_names.push_back("LF_KFE");
  // contact_names.push_back("LH_KFE");
  // contact_types.push_back(crocoddyl::Contact3D);
  // contact_types.push_back(crocoddyl::Contact3D);
  // contact_types.push_back(crocoddyl::Contact3D);
  // RobotEENames quadrupedAnymal(
  //     "Anymal", contact_names, contact_types,
  //     EXAMPLE_ROBOT_DATA_MODEL_DIR
  //     "/anymal_b_simple_description/robots/anymal.urdf",
  //     EXAMPLE_ROBOT_DATA_MODEL_DIR
  //     "/anymal_b_simple_description/srdf/anymal.srdf",
  //     "RH_KFE", "standing");

  // print_benchmark(quadrupedAnymal);

  // // Quadruped HyQ Benchmarks
  // std::cout << "******************** Quadruped HyQ ******************"
  //           << std::endl;
  // contact_names.clear();
  // contact_types.clear();
  // contact_names.push_back("rf_kfe_joint");
  // contact_names.push_back("lf_kfe_joint");
  // contact_names.push_back("lh_kfe_joint");
  // contact_types.push_back(crocoddyl::Contact3D);
  // contact_types.push_back(crocoddyl::Contact3D);
  // contact_types.push_back(crocoddyl::Contact3D);
  // RobotEENames quadrupedHyQ("HyQ", contact_names, contact_types,
  //                           EXAMPLE_ROBOT_DATA_MODEL_DIR
  //                           "/hyq_description/robots/hyq_no_sensors.urdf",
  //                           EXAMPLE_ROBOT_DATA_MODEL_DIR
  //                           "/hyq_description/srdf/hyq.srdf",
  //                           "rh_kfe_joint", "standing");

  // print_benchmark(quadrupedHyQ);

  // // Biped icub Benchmarks
  // std::cout << "********************Biped iCub ***********************"
  //           << std::endl;
  // contact_names.clear();
  // contact_types.clear();
  // contact_names.push_back("r_ankle_roll");
  // contact_names.push_back("l_ankle_roll");
  // contact_types.push_back(crocoddyl::Contact6D);
  // contact_types.push_back(crocoddyl::Contact6D);

  // RobotEENames bipedIcub(
  //     "iCub", contact_names, contact_types,
  //     EXAMPLE_ROBOT_DATA_MODEL_DIR
  //     "/icub_description/robots/icub_reduced.urdf",
  //     EXAMPLE_ROBOT_DATA_MODEL_DIR "/icub_description/srdf/icub.srdf",
  //     "r_wrist_yaw", "half_sitting");
  // print_benchmark(bipedIcub);

  // // Biped icub Benchmarks
  // std::cout << "********************Biped Talos***********************"
  //           << std::endl;
  // contact_names.clear();
  // contact_types.clear();
  // contact_names.push_back("leg_right_6_joint");
  // contact_names.push_back("leg_left_6_joint");
  // contact_types.push_back(crocoddyl::Contact6D);
  // contact_types.push_back(crocoddyl::Contact6D);

  // RobotEENames bipedTalos(
  //     "Talos", contact_names, contact_types,
  //     EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_reduced.urdf",
  //     EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf",
  //     "arm_right_7_joint", "half_sitting");
  // print_benchmark(bipedTalos);

  // Panda with colmpc Benchmarks
  // std::cout << "********************Panda With Colmpc***********************"
  //           << std::endl;
  // contact_names.clear();
  // contact_types.clear();

  // RobotEENames armPanda(
  //     "Panda_arm", contact_names, contact_types,
  //     "/home/gepetto/ros_ws/src/colmpc/examples/models/urdf/franka2.urdf",
  //     "/home/gepetto/ros_ws/src/colmpc/examples/models/srdf/demo.srdf",
  //     "panda2_hand_tcp", "arm_up");
  // print_benchmark(armPanda);

  return 0;
}
