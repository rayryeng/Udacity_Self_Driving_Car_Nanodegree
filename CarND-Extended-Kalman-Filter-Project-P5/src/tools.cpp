#include "tools.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */

  // Sanity checks - check if either inputs are empty
  if (estimations.empty() || ground_truth.empty()) {
    throw std::invalid_argument("Estimations or Ground Truth vector is empty");
  }

  // Check if the inputs are the same size
  if (estimations.size() != ground_truth.size()) {
    throw std::invalid_argument(
        "Estimations does not match Ground Truth vector size");
  }

  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
    std::cout << "Invalid estimation or ground_truth data"
              << "\n";
    return rmse;
  }

  // accumulate squared residuals
  for (unsigned int i = 0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];

    // coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  // calculate the mean
  rmse = rmse / estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();

  // return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */
  // Note - assuming 3 x 4 matrix where each row is a quantity
  // i.e. rho, phi, rhodot
  // Each column is a state variable we already know
  // px, py, vx, vy
  MatrixXd Hj(3, 4);

  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // check division by zero
  if (std::abs(px) <= 1e-10 && std::abs(py) <= 1e-10) { return Hj; }

  // Compute Jacobian now
  const float dv1 = std::sqrt(px * px + py * py);
  const float dv2 = px * px + py * py;
  const float dv3 = dv2 * dv1;

  Hj << px / dv1, py / dv1, 0, 0,
       -py / dv2, px / dv2, 0, 0,
      py * (vx * py - vy * px) / dv3, px * (vy * px - vx * py) / dv3, px / dv1,
      py / dv1;

  return Hj;
}

VectorXd Tools::CartesianToPolar(const VectorXd &x) {
  const float px = x(0);
  const float py = x(1);
  const float vx = x(2);
  const float vy = x(3);

  // Calculate distance to pedestrian
  const float rho = std::sqrt(px * px + py * py);

  // Calculate bearing angle an normalise it
  float phi = std::atan2(py, px);
  while (phi < -M_PI) {
    phi += 2.0f * M_PI;
  }
  while (phi > M_PI) {
    phi -= 2.0f * M_PI;
  }

  // Calculate distance rate
  const float rhodot = (px * vx + py * vy) / rho;

  VectorXd z(3);
  z << rho, phi, rhodot;
  return z;
}

VectorXd Tools::PolarToCartesian(const VectorXd &z) {
  const float rho = z(0);
  const float phi = z(1);
  const float rhodot = z(2);

  const float px = rho * std::cos(phi);
  const float py = rho * std::sin(phi);
  const float vx = rhodot * std::cos(phi);
  const float vy = rhodot * std::sin(phi);

  VectorXd x(4);
  x << px, py, vx, vy;
  return x;
}
