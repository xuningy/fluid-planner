#pragma once

#include <chrono>
#include <deque>
#include <iostream>
#include <vector>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <trajectory/state.h>

namespace print_utils {

  inline void print(const std::vector<size_t> &vec, const std::string name = "") {
    std::cout << name << ": ";
    for (auto &v : vec) std::cout << v << " ";
    std::cout << std::endl;
  }

  inline void print(const std::vector<int> &vec, const std::string name = "") {
    std::cout << name << ": ";
    for (auto &v : vec) std::cout << v << " ";
    std::cout << std::endl;
  }

  inline void print(const std::vector<float> &vec, const std::string name = "") {
    std::cout << name << ": ";
    for (auto &v : vec) std::cout << v << " ";
    std::cout << std::endl;
  }

  inline void print(const std::vector<double> &vec, const std::string name = "") {
    std::cout << name << ": ";
    for (auto &v : vec) std::cout << v << " ";
    std::cout << std::endl;
  }

  template <typename Derived>
  inline void print(const Eigen::DenseBase<Derived> &mat, const std::string name = "") {
    Eigen::IOFormat OctaveFmt(4, 0, ", ", ";\n", "", "", "[", "]");
    Eigen::IOFormat OctaveVecFmt(4, 0, ", ", " ", "", "", "[", "]");

    std::cout << name << "(" << mat.rows() << "x" << mat.cols() << "): ";
    if (mat.rows() == 1 || mat.cols() == 1) std::cout << mat.format(OctaveVecFmt) << std::endl;
    else {
      std::cout << std::endl;
      std::cout << mat.format(OctaveFmt) << std::endl;
    }
  }

  inline void print(std::chrono::high_resolution_clock::time_point t1, std::chrono::high_resolution_clock::time_point t2) {
    std::chrono::duration<float> dur = t1 - t2;
    printf("(%.4fs)", dur.count());
  }

  inline void print(const std::vector<trajectory::State<float>> &traj,
    const std::string name = "") {
    std::cout << name << "[ X, Y, Z | Vx, Vy, Vz ]" << std::endl;
    for (auto &tj : traj) {
      std::cout << tj.Pos(0) << ", "
                << tj.Pos(1) << ", "
                << tj.Pos(2) << " | "
                << tj.Vel(0) << ", "
                << tj.Vel(1) << ", "
                << tj.Vel(2) << std::endl;
    }
  }

  inline void print(const std::deque<trajectory::State<float>> &traj,
    const std::string name = "") {
    std::cout << name << "[ X, Y, Z | Vx, Vy, Vz ]" << std::endl;
    for (auto &tj : traj) {
      std::cout << tj.Pos(0) << ", "
                << tj.Pos(1) << ", "
                << tj.Pos(2) << " | "
                << tj.Vel(0) << ", "
                << tj.Vel(1) << ", "
                << tj.Vel(2) << std::endl;
    }
  }

}
