
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "arm-kinova.hpp"
#include "crocoddyl/core/utils/file-io.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include "ddp.hpp"

boost::shared_ptr<crocoddyl::ShootingProblem> get_problem(
    const unsigned int N) {
  boost::shared_ptr<crocoddyl::ActionModelAbstract> runningModel, terminalModel;
  crocoddyl::benchmark::build_arm_kinova_action_models(runningModel,
                                                       terminalModel);

  // Get the initial state
  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(
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
  default_state << dm->get_pinocchio().referenceConfigurations["arm_up"],
      Eigen::VectorXd::Zero(state->get_nv());

  Eigen::VectorXd x0(default_state);
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > runningModels(
      N, runningModel);
  boost::shared_ptr<crocoddyl::ShootingProblem> problem =
      boost::make_shared<crocoddyl::ShootingProblem>(x0, runningModels,
                                                     terminalModel);
  return problem;
}

int main(int argc, char* argv[]) {
  // Obtain number of cores as argument
  if (argc != 3) {
    std::cerr << "Incorrect number of arguments! Example usage 'multithreading "
                 "<number of cores> <csv output file path>'.";
    return 1;
  }

  const unsigned int nthreads = std::stoi(argv[1]);
  const std::string csv_filename = std::string(argv[2]);

  unsigned int N = 40;  // number of nodes
  unsigned int number_of_trials = 1e2;

  auto problem = get_problem(N);

  // Computing the warm-start
  std::vector<Eigen::VectorXd> xs(N + 1, problem->get_x0());
  std::vector<Eigen::VectorXd> us(
      N, Eigen::VectorXd::Zero(problem->get_runningModels()[0]->get_nu()));
  for (unsigned int i = 0; i < N; ++i) {
    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model =
        problem->get_runningModels()[i];
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data =
        problem->get_runningDatas()[i];
    model->quasiStatic(data, us[i], problem->get_x0());
  }

  flattened_ddp::SolverDDP ddp(problem);
  ddp.setCandidate(xs, us, false);
  ddp.get_problem()->set_nthreads(nthreads);

  CsvStream csv(csv_filename);
  csv << "nthreads" << "time" << csv.endl;

  crocoddyl::Timer timer;

  for (unsigned int i = 0; i < number_of_trials; i++) {
    timer.reset();
    ddp.solve(xs, us, 1, true);
    csv << ddp.get_problem()->get_nthreads() << timer.get_us_duration()
        << csv.endl;
  }

  return 0;
}
