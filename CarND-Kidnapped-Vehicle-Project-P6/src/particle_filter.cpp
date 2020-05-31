/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// Added by Ray
#include <limits>

#include "helper_functions.h"

using std::string;
using std::vector;

// Added by Ray - for convenience
using std::default_random_engine;
using std::discrete_distribution;
using std::normal_distribution;

// Added by Ray - implement multivariate Gaussian
// Taken from the lectures
namespace {
// mu_x, mu_y - Actual landmark positinn
// x_obs, y_obs - Observed matched landmark position warped from vehicle
// frame to map frame
// sig_x, sig_y - landmark uncertainty
double multiv_prob(double sig_x,
                   double sig_y,
                   double x_obs,
                   double y_obs,
                   double mu_x,
                   double mu_y) {
  // calculate normalization term
  const double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  const double exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2))) +
                          (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

  // calculate weight using normalization terms and exponent
  const double weight = gauss_norm * exp(-exponent);
  return weight;
}
}  // namespace

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 50;  // TODO: Set the number of particles

  // Step #1 - Create normal distribution generator for x, y and heading
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // Step #2 - Add random Gaussian noise to each particle and set the
  // weight to 1, then push back
  particles.clear();
  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    particle.id = i;
    particles.push_back(particle);
  }
}

void ParticleFilter::prediction(double delta_t,
                                double std_pos[],
                                double velocity,
                                double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  default_random_engine gen;

  for (auto& p : particles) {
    // First create noise profile centered at the current coordinates
    normal_distribution<double> dist_x(p.x, std_pos[0]);
    normal_distribution<double> dist_y(p.y, std_pos[1]);
    normal_distribution<double> dist_theta(p.theta, std_pos[2]);

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);

    // Now add in the estimated trajectory
    // If yaw rate is close to 0, this means we're not turning and are
    // going in a straight line, so this is just your standard constant
    // velocity model
    if (abs(yaw_rate) <= 1e-10) {
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    } else {
      p.x += (velocity / yaw_rate) *
             (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      p.y += (velocity / yaw_rate) *
             (cos(p.theta) - cos(p.theta + yaw_rate + delta_t));
      p.theta += yaw_rate * delta_t;
    }
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  // For each observation...
  for (auto& o : observations) {
    double min_dist = std::numeric_limits<double>::max();
    const double x1 = o.x;
    const double y1 = o.y;
    int index = -1;
    // Go through all of the predicted landmark positions and find the one
    // that is the closest to the observation
    for (size_t i = 0; i < predicted.size(); i++) {
      double obs_dist = dist(x1, y1, predicted[i].x, predicted[i].y);
      if (obs_dist < min_dist) {
        min_dist = obs_dist;
        index = i;
      }
    }
    // For this observation, this is closest to predicted[index]
    o.id = index;
  }
}

void ParticleFilter::updateWeights(double sensor_range,
                                   double std_landmark[],
                                   const vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a multi-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // For each particle...
  for (auto& part : particles) {
    // Get particle coords (wrt map frame)
    const double x_part = part.x;
    const double y_part = part.y;
    const double theta = part.theta;

    // Note - Observations are the positions of the map landmarks with respect
    // to the car.  We need to transform these so that they're with respect
    // to the particle, which is with respect to the map
    vector<LandmarkObs> transformed_observations;
    for (const auto& o : observations) {
      const double x_obs = o.x;
      const double y_obs = o.y;

      // Transform observation's x coordinate to map frame
      const double x_map = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);

      // Do the same for the y coordinate
      const double y_map = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);

      // Place in vector - ID is set to -1 as we don't care what this is right
      // now.  This will be changed in the data association step.
      transformed_observations.push_back(LandmarkObs{-1, x_map, y_map});
    }

    // Using the sensor range, we also need to isolate out which landmarks
    // are closest to the car
    vector<LandmarkObs> predicted;
    for (const auto& m : map_landmarks.landmark_list) {
      if (dist(x_part, y_part, m.x_f, m.y_f) <= sensor_range) {
        predicted.push_back(LandmarkObs{m.id_i, m.x_f, m.y_f});
      }
    }

    // Perform data association so that given the landmarks that are
    // visible to the car, match each observed landmark to the actual map
    // landmark
    dataAssociation(predicted, transformed_observations);

    // For each landmark in sensor range, use this to calculate the particle's
    // final weight
    // Take note that multiplying a series of small numbers can lead to
    // instability.  Therefore, if we took the log of each number, summed up
    // the result, then took the exp, this is the same thing but more stable.
    double log_sum = 0.0;
    for (const auto& o : transformed_observations) {
      const int id = o.id;
      const double x_obs = o.x;
      const double y_obs = o.y;
      const double mu_x = predicted[id].x;
      const double mu_y = predicted[id].y;

      log_sum += log(multiv_prob(std_landmark[0], std_landmark[1], x_obs, y_obs,
                                 mu_x, mu_y));
    }
    part.weight = exp(log_sum);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // Separate out weights into a vector
  vector<double> weights;
  for (const auto& p : particles) {
    weights.push_back(p.weight);
  }

  // Create discrete distribution random object that
  // will generate indices from 0 to N - 1 with N being
  // the total number of weights or number of particles
  default_random_engine gen;
  discrete_distribution<int> dist_weights(weights.begin(), weights.end());

  // Create a new vector by iterating over as many particles
  // as we have and randomly sampling out the particles given the above
  // probability distribution
  std::vector<Particle> new_particles;
  for (int i = 0; i < particles.size(); i++) {
    const int index = dist_weights(gen);
    new_particles.push_back(particles[index]);
  }
  // Replace
  particles = std::move(new_particles);
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}