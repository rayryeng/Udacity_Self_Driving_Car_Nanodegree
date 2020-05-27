#define _USE_MATH_DEFINES
#include "kalman_filter.h"
#include "tools.h"
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;


/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in,
                        MatrixXd &P_in,
                        MatrixXd &F_in,
                        MatrixXd &H_in,
                        MatrixXd &R_in,
                        MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::UpdateXandP(const VectorXd& z, const VectorXd& z_pred,
                               const bool normaliseAngle) {
  VectorXd y = z - z_pred;
  if (normaliseAngle) {
    while (y(1) < -M_PI) {
      y(1) += 2.0f * M_PI;
    }
    while (y(1) > M_PI) {
      y(1) -= 2.0f * M_PI;
    }
  }
  const MatrixXd Ht = H_.transpose();
  const MatrixXd S = H_ * P_ * Ht + R_;
  const MatrixXd Si = S.inverse();
  const MatrixXd PHt = P_ * Ht;
  const MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  const MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_; 
}

void KalmanFilter::Predict() {
  /**
   * TODO: predict the state
   */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * TODO: update the state by using Kalman Filter equations
   */
  const VectorXd z_pred = H_ * x_;
  UpdateXandP(z, z_pred);
 
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */
  Tools tools;

  // Don't update the Jacobian if the position is too small
  const float px = x_(0);
  const float py = x_(1);
  if (std::abs(px) <= 1e-10 || std::abs(py) <= 1e-10) {
    return;
  }

  const VectorXd z_pred = tools.CartesianToPolar(x_);
  H_ = tools.CalculateJacobian(x_);
  UpdateXandP(z, z_pred, true);
}
