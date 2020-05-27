#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
   * TODO: Finish initializing the FusionEKF.
   * TODO: Set the process and measurement noises
   */
  // Select out the positions only in LiDAR
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.P_ = MatrixXd(4, 4);
  // We are fairly confident about the position but
  // not the velocity
  ekf_.P_ << 10, 0, 0, 0,
             0, 10, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;  
  ekf_.Q_ = MatrixXd(4, 4);
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  const VectorXd z = measurement_pack.raw_measurements_;  
  if (!is_initialized_) {
    /**
     * TODO: Initialize the state ekf_.x_ with the first measurement.
     * TODO: Create the covariance matrix.
     * You'll need to convert radar from polar to cartesian coordinates.
     */

    // Also make sure to set previous timestamp
    previous_timestamp_ = measurement_pack.timestamp_;

    // first measurement
    ekf_.x_ = VectorXd(4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // TODO: Convert radar from polar to cartesian coordinates 
      //         and initialize state.
      const VectorXd x = tools.PolarToCartesian(z);
      ekf_.x_ << x(0), x(1), 0, 0; // Cannot use the speed to initiate the pedestrian
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // TODO: Initialize state.
      // Missing velocity information so assume 0
      ekf_.x_ << z(0), z(1), 0, 0;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  /**
   * TODO: Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * TODO: Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  // Calculate time difference
  const float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;

  // Set previous timestamp
  previous_timestamp_ = measurement_pack.timestamp_;

  // Calculate F to help find predicted states
  const float noise_ax = 9;
  const float noise_ay = 9;

  if (dt > 0.0f) {
    ekf_.F_ << 1, 0, dt, 0,
              0, 1, 0, dt,
              0, 0, 1, 0,
              0, 0, 0, 1;
    
    // Calculate process covariance matrix
    const float dt2 = dt * dt;
    const float dt3 = dt2 * dt / 2;
    const float dt4 = dt2 * dt2 / 4;
    
    ekf_.Q_ << dt4 * noise_ax, 0, dt3 * noise_ax, 0,
              0, dt4 * noise_ay, 0, dt3 * noise_ay,
              dt3 * noise_ax, 0, dt2 * noise_ax, 0,
              0, dt3 * noise_ay, 0, dt2 * noise_ay;  

    ekf_.Predict();
  }

  /**
   * Update
   */

  /**
   * TODO:
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // TODO: Radar updates
    // Set the measurement noise matrix
    // Note that when calculating the prediction and measurement
    // residual y, we are doing it in polar coordinates directly
    ekf_.R_ = R_radar_;

    // For completeness - calculate Jacobian from previous state
    Hj_ = tools.CalculateJacobian(ekf_.x_);

    // Now update
    ekf_.UpdateEKF(z);
    
  } else {
    // TODO: Laser updates

    // Set the measurement matrix
    ekf_.H_ = H_laser_;

    // Set the measurement noise matrix
    ekf_.R_ = R_laser_;

    // Now update
    ekf_.Update(z);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
