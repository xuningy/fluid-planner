
#include <iostream>
#include <vector>

#include <cost_function/cost_function.h>
#include <cost_function/flow_field.h>
#include <utils/print_utils.h>

#include <libgp_interface.h>

namespace cf = cost_function;
namespace pu = print_utils;

// An example usage of the FLUID planner.
int main() {
  // Load data from csv.

  cf::GP gp;
  gp.dim = 4;
  gp.cov_kernel = "CovSum( CovSEiso, CovNoise)";
  gp.hyp_params_x = std::vector<float>{1.2, 4.1, -2.0};
  gp.hyp_params_y = std::vector<float>{1.0, 1.5, 4.0};
  size_t window_size = 6;

  LibgpInterface<float> libgp;
  libgp.Initialize(gp.dim, gp.cov_kernel, gp.hyp_params_x);

  // Eigen single instance
  Eigen::VectorXf datapt(4);
  datapt << 1, 2, 3, 4;
  libgp.Train(datapt, 10.0);
  Eigen::VectorXf datapt2(4);
  datapt2 << 4, 5, 6, 7;
  libgp.Train(datapt2, 22.0);

  pu::print(datapt, "X");
  std::cout << libgp.Predict(datapt) << std::endl;
  pu::print(datapt2, "X");
  std::cout << libgp.Predict(datapt2) << std::endl;



  return 0;
}
