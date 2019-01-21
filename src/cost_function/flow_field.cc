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

#include <cost_function/flow_field.h>

namespace tr = trajectory;

namespace cost_function {

static constexpr float kZeroSpeed = 0.01;

FlowField::FlowField() {}
FlowField::~FlowField() {}

void FlowField::Initialize(const GP &gp, const size_t window_size) {
  gp_x_.Initialize(gp.dim, gp.cov_kernel, gp.hyp_params_x);
  gp_y_.Initialize(gp.dim, gp.cov_kernel, gp.hyp_params_y);
  window_size_ = window_size;
  return;
}

void FlowField::Learn() {
  for (auto &s : training_examples_) {
    Eigen::Vector2f pos_2d = Eigen::Vector2f(s.Pos(0), s.Pos(1));
    gp_x_.Train(pos_2d, s.Vel(0));
    gp_y_.Train(pos_2d, s.Vel(1));
  }
  return;
}

void FlowField::Clear() {
  gp_x_.Clear();
  gp_y_.Clear();
  return;
}

void FlowField::AddSegment(const std::vector<tr::State<float>> &seg) {
  // If there are too many elements, erase them.
  if (num_examples_.size() >= window_size_) {
    for (size_t i = 0; i < num_examples_.size() - window_size_ + 1; i++) {
      size_t num_examples_to_erase = num_examples_[0];
      num_examples_.pop_front();
      training_examples_.erase(training_examples_.begin(),
                               training_examples_.begin() +
                                   num_examples_to_erase);
    }
  }

  // Add data.
  num_examples_.push_back(seg.size());
  training_examples_.insert(training_examples_.end(), seg.begin(), seg.end());

  // Check that we have the correct number of training sets.
  assert(num_examples_.size() <= window_size_);
  return;
}

std::vector<float>
FlowField::ComputeCost(std::vector<tr::Trajectory<float>> &trajs) {
  std::vector<float> cost;
  for (auto &tj : trajs) {
    Eigen::Matrix<float, 3, Eigen::Dynamic> pos, vel;
    // TODO(xuning@cmu.edu): Make the dt=0.3 a parameter.
    tj.GetWaypointsLocal(0.3, &pos, &vel);
    cost.push_back(ComputeCost(tj.Z(), pos, vel));
  }
  return cost;
}

float FlowField::ComputeCost( const Eigen::VectorXf &Z,
    const Eigen::Matrix<float, 3, Eigen::Dynamic> &pos,
    const Eigen::Matrix<float, 3, Eigen::Dynamic> &vel) {

  int num_samples = pos.cols();
  Eigen::VectorXf vel_x, vel_y;
  Eigen::MatrixXf pos_2d = pos.block(0, 0, 2, num_samples);
  gp_x_.Predict(num_samples, pos_2d, &vel_x);
  gp_y_.Predict(num_samples, pos_2d, &vel_y);

  // Reformat matrices.
  Eigen::MatrixXf vel_pred(2, num_samples);
  Eigen::VectorXf vel_x_f = vel_x.cast<float>();
  Eigen::VectorXf vel_y_f = vel_y.cast<float>();
  vel_pred << vel_x_f.transpose(), vel_y_f.transpose();

  Eigen::MatrixXf vel_actual(2, num_samples);
  vel_actual = vel.block(0, 0, 2, num_samples).cast<float>();

  Eigen::VectorXf vel_pred_norm = vel_pred.colwise().norm();
  Eigen::VectorXf vel_actual_norm = vel_actual.colwise().norm();

  // Normalize the vectors.
  vel_pred.colwise().normalize();
  vel_actual.colwise().normalize();

  // The reward is the dot product between the predicted velocity field and the
  // actual velocity vectors.
  Eigen::VectorXf reward = (vel_pred.transpose() * vel_actual).diagonal();

  // Any value that's not predicted, make that value zero.
  for (size_t i = 0; i < reward.size(); i++) {
    if (vel_pred_norm(i) < kZeroSpeed) {
      // TODO(xuning@cmu.edu): Make the 'arbitrarily large value for the
      // unpopulated flow field regions' a parameter. Currently set to 10.
      // (put a negative sign in front of it, it will be flipped down below.)
      reward(i) = -80;
    }
    // If we encounter an instance with zero velocity trajectories, make the cost as big as the norm of the predicted cost.
    if (vel_actual_norm(i) < kZeroSpeed) {
      reward(i) = 0;
    }
  }

  // Compute the magnitude of the velocity.
  Eigen::VectorXf speed_cost = (vel_pred_norm - vel_actual_norm).array().abs();

  // penalize continuity of the velocities
  size_t num_params = 0.5 * Z.size();
  float cont_cost = std::abs(Z.block(0, 0, num_params, 1).sum());

  // Cost is the sum of the -reward and speed cost.
  Eigen::VectorXf cost = -reward + speed_cost;
  // if (Z(0) < 0.01 || Z(1) < 0.01 || Z(2) < 0.01) {
  //   std::cout << std::endl;
  //   pu::print(Z, "Z");
  //   pu::print(-reward, "direction cost");
  //   pu::print(speed_cost, "speed cost");
  //   std::cout << "cont cost: " << cont_cost << " direction cost: " << -reward.sum() << " speed cost: " << speed_cost.sum() << " TOTAL: " << cost.sum() + cont_cost <<  std::endl;
  //   pu::print(vel_actual_norm, "vel_actual_norm");
  // }


  // Debug: just in case if we run into a NAN in the sum, this will output all of the velocities.
  if (std::isnan(cost.sum())) {
    std::cout << std::endl;
    std::cout << std::endl;
    pu::print(vel_x_f, "vel_pred_x");
    pu::print(vel_y_f, "vel_pred_y");
    pu::print(vel_pred, "vel_pred_normalized");
    pu::print(vel_pred_norm, "vel_pred_norm");
    std::cout << std::endl;
    pu::print(vel, "vel_actual_actual");
    pu::print(vel_actual, "vel_actual_normalized");
    pu::print(vel_actual_norm, "vel_actual_norm");
    pu::print(-reward, "cost contains NAN");
    std::cout << std::endl;
    std::cout << std::endl;
  }
  return cost.sum() + cont_cost;
}

void FlowField::GenerateLocalFlowField(const GridSize &gs, Eigen::MatrixXf *Vx,
                                       Eigen::MatrixXf *Vy) {
  Eigen::VectorXf pos_x = Eigen::VectorXf::LinSpaced(gs.Nx, gs.min_x, gs.max_x);
  Eigen::VectorXf pos_y = Eigen::VectorXf::LinSpaced(gs.Ny, gs.min_y, gs.max_y);

  Vx->resize(gs.Nx, gs.Ny);
  Vy->resize(gs.Nx, gs.Ny);

  std::cout << "Num Samples for gp_x and gp_y:" << gp_x_.GetNumSamples() << ", " << gp_y_.GetNumSamples() << std::endl;
  for (size_t i = 0; i < gs.Nx; i++) {
    for (size_t j = 0; j < gs.Ny; j++) {
      Eigen::Vector2f xy_coord(pos_x(i), pos_y(j));
      Vx->coeffRef(i, j) = gp_x_.Predict(xy_coord);
      Vy->coeffRef(i, j) = gp_y_.Predict(xy_coord);
    }
  }
  return;
}

} // namespace cost_function
