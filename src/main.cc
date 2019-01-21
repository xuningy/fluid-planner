// Copyright 2018 Toyota Research Institute.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <chrono>
#include <ctime>
#include <iostream>
#include <vector>

#include <fluid_planner.h>

#include <cost_function/cost_function.h>
#include <cost_function/flow_field.h>
#include <cost_function/goal.h>
#include <trajectory/state.h>
#include <trajectory/trajectory.h>
#include <utils/conversion_utils.h>
#include <utils/data_handling.h>
#include <utils/stats_utils.h>
#include <utils/print_utils.h>

#include <libgp_interface.h>

namespace tr = trajectory;
namespace cf = cost_function;
namespace fp = fluid_planner;
namespace cu = conversion_utils;
namespace su = stats_utils;
namespace pu = print_utils;

// Given a matrix of data, `TransformTrajectoryIntoLocalFrame` turns the
// segment of waypoints into a vector of trajectory::State.
std::vector<tr::State<float>>
TransformOdomIntoLocalFrame(const Eigen::MatrixXf &odom) {
  std::vector<tr::State<float>> traj;

  // Transformations from the start point to the world frame such that pose at
  // time 0 is (0, 0, 0).

  // Rotation.
  Eigen::Matrix3f R_wb;
  R_wb = Eigen::AngleAxisf(odom(3, 0), Eigen::Vector3f::UnitZ());

  // Translation.
  Eigen::Vector3f T_wb = Eigen::Vector3f(odom(0, 0), odom(1, 0), odom(3, 0));

  // Transform all odometry points.
  for (size_t i = 0; i < odom.cols(); i++) {
    float yaw = odom(3, i);
    Eigen::Vector3f pos =
        R_wb.transpose() *
        (Eigen::Vector3f(odom(0, i), odom(1, i), odom(3, i)) - T_wb);
    Eigen::Vector3f vel = R_wb.transpose() *
                          Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
                          Eigen::Vector3f(odom(5, i), 0, 0);
    traj.push_back(tr::State<float>(pos, vel, yaw));
  }

  return traj;
}

// An example usage of the FLUID planner.
int main() {
  // Load data from csv.
  Eigen::MatrixXf data = data_handling::LoadCsv("../data/data1.csv", 8, 14014);

  float dt = 0.008;
  std::deque<tr::State<float>> training_examples;

  // Parameters
  fp::TrajProperties tp;
  tp.num_seg = 3;
  tp.traj_duration = 5;
  tp.v_ub = 60;
  tp.v_lb = 0;
  tp.omega_lb = -1;
  tp.omega_ub = 1;
  tp.num_action_space_discretizations = 60;

  fp::PlannerProperties pp;
  pp.sample_dev = 0.2;
  pp.num_iter = 15;
  pp.rho = 0.1;
  pp.k = 1;
  pp.num_traj = 10000;

  std::string cp_type = "FLOW";

  cf::GP gp;
  gp.dim = 2;
  gp.cov_kernel = "CovSum( CovSEiso, CovNoise)";
  gp.hyp_params_x = std::vector<float>{1.2, 4.1, 4.0};
  gp.hyp_params_y = std::vector<float>{1.0, 1.5, 4.0};
  size_t window_size = 6;

  // Initialize replan window.
  float replan_every_s = 0.5;
  int replan_dt = std::round(replan_every_s / dt);

  // Flow Field segment duration length.
  float max_seg_duration = 5;
  int dS = std::round(max_seg_duration / dt);

  // Segment goal, sampled from some time in the future.
  float goal_length = tp.traj_duration;
  int dN = std::round(goal_length / dt);

  /* ====================================================================== */

  fp::FluidPlanner flp;
  flp.Initialize(pp, tp);

  cf::FlowField ff_cf;
  cf::Goal goal_cf;

  if (cp_type == "FLOW") {
    ff_cf.Initialize(gp, window_size);
  }

  // Open files to write to.
  std::ofstream file_traj, file_traj_world, file_traj_local, file_goal,
      file_field, file_meta, file_seg;

  // save to output folder
  std::string output_folder = "/home/xuning/Dropbox/TRI2018/output-data";

  time_t now = time(0);
  tm *ltm = localtime(&now);
  int year = 1970 + ltm->tm_year;
  int month = 1 + ltm->tm_mon;
  int day = ltm ->tm_mday;
  int hour = 1 + ltm->tm_hour;
  int minute = 1 + ltm->tm_min;

  file_traj_world.open("traj_world.csv");
  file_traj_local.open("traj_local.csv");
  file_traj.open("traj.csv");
  file_field.open("field.csv");
  file_meta.open("meta.csv");
  file_seg.open("traj_segments.csv");
  if (cp_type == "GOAL")
    file_goal.open("goal.csv");

  /* ====================================================================== */

  std::cout << "Starting planner..." << std::endl;
  int N = data.cols();

  for (int i = dS; i <= dS + replan_dt*100; i += replan_dt) {
    std::cout << "Planner time stamp: " << i * dt << "s" << std::endl;

    // Update the GOAL cost function if needed. goal = [X, Y, heading]
    if (cp_type == "GOAL") {
      Eigen::Vector3f goal_i =
          Eigen::Vector3f(data(0, i + dN), data(1, i + dN), data(3, i + dN));
      goal_cf.SetGoal(goal_i);
      data_handling::WritePointToFile(goal_i, file_goal);
    }

    float yaw = data(3, i);
    Eigen::Vector3f start_pos(
        Eigen::Vector3f(data(0, i), data(1, i), data(3, i)));
    Eigen::Vector3f start_vel(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
                              Eigen::Vector3f(data(5, i), 0, 0));
    tr::State<float> start(start_pos, start_vel, yaw);

    // Plan!
    std::vector<tr::Trajectory<float>> traj = flp.Plan(start, ff_cf);

    // Update FLOW cost function.
    if (cp_type == "FLOW") {
      printf("Updating Flow Field cost function.\n");
      auto t1 = std::chrono::high_resolution_clock::now();

      // Clear the GPs.
      ff_cf.Clear();

      // Extract data segment.
      int seg_len = std::min(i, dS);
      if (seg_len == 0)
        continue;
      else
        std::cout << "Adding " << seg_len << " odometry points to the flow field" << std::endl;
      Eigen::MatrixXf data_segment = data.block(0, i - seg_len, 6, seg_len);

      // Transform segment.
      std::vector<tr::State<float>> odom_segment =
          TransformOdomIntoLocalFrame(data_segment);
      ff_cf.AddSegment(odom_segment);

      auto t2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> dur1 = t2 - t1;
      printf("Adding segment took: %.4fs\n", dur1.count());

      // Learn.
      ff_cf.Learn();

      auto t3 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> dur2 = t3 - t2;
      printf("Learning cf took: %.4fs\n", dur2.count());

      data_handling::WriteOdomToFile(odom_segment, file_seg);
    }

    // Write to file.
    std::cout << "Writing meta data,";
    std::cout << std::flush;
    file_meta << i * dt << "," << traj.size() << std::endl;

    std::cout << " trajectories, ";
    std::cout << std::flush;

    data_handling::WriteWptsToFile(traj, file_traj_world, "WORLD");
    data_handling::WriteWptsToFile(traj, file_traj_local, "LOCAL");
    data_handling::WriteTrajectoriesToFile(traj, file_traj);

    std::cout << "and local Flow Field to file." << std::endl;
    Eigen::MatrixXf Vx, Vy;
    cf::GridSize gs;
    ff_cf.GenerateLocalFlowField(gs, &Vx, &Vy);
    data_handling::WriteFlowFieldToFile(Vx, Vy, file_field);

    std::cout << std::endl;
  } // end iterator

  file_traj_world.close();
  file_traj_local.close();
  file_meta.close();
  if (cp_type == "GOAL")
    file_goal.close();

  return 0;
}
