#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h"  // Added by Ray

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  // Added by Ray
  int lane = 1;
  double ref_vel = 0.0;

  // Changed by Ray - to include lane variable and reference velocity
  h.onMessage([&ref_vel, &lane, &map_waypoints_x, &map_waypoints_y,
               &map_waypoints_s, &map_waypoints_dx,
               &map_waypoints_dy](uWS::WebSocket<uWS::SERVER> ws, char *data,
                                  size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object

          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side
          //   of the road.
          // Format: [id,x,y,vx,vy,s,d]
          auto sensor_fusion = j[1]["sensor_fusion"];

          json msgJson;

          vector<double> next_x_vals;
          vector<double> next_y_vals;

          /**
           * TODO: define a path made up of (x,y) points that the car will visit
           *   sequentially every .02 seconds
           */

          // This logic was mostly borrowed from Aaron Brown from Udacity
          // with some slight changes

          // Get the number of waypoints from the previous iteration
          int prev_size = previous_path_x.size();
          // If there are points available, then make the current longitudinal
          // distance the end of the previous path to discourage the jerking
          // from the cold start
          if (prev_size > 0) { car_s = end_path_s; }

          // Variables that help indicate whether we're too close or if
          // there is a car to the left or right of us
          bool too_close = false;
          bool car_left = false;
          bool car_right = false;

          // Loop through each car visible on our side of the road...
          for (int i = 0; i < sensor_fusion.size(); i++) {
            // Get distance away from double yellow lane marker on the left
            float d = sensor_fusion[i][6];

            // Get velocity components in both x and y for the car
            double vx = sensor_fusion[i][3];
            double vy = sensor_fusion[i][4];

            // Get the speed of the car
            double check_speed = sqrt(vx * vx + vy * vy);

            // Get the longitudinal distance along the path since the start
            double check_car_s = sensor_fusion[i][5];

            // "Project" the end waypoint from the previous iteration out
            check_car_s += (double)prev_size * 0.02 * check_speed;

            // If we see that the observed car is in our lane...
            if (d < (2 + 4 * lane + 2) && (d > (2 + 4 * lane - 2))) {
              // If we see that the projected waypoint from the previous iter.
              // is farther out than where we currently are located at
              // and if the distance between these points is less than 30 m
              // we're getting too close, so flag it
              if ((check_car_s > car_s) && ((check_car_s - car_s) < 30)) {
                too_close = true;
              }
            }

            // If there is a car on the left lane, and if we see that
            // the longitudinal distance between the ego vehicle and it
            // is less than 30m, then it's not safe to change to the left
            if ((d > 2 + 4 * lane - 6) && (d < 2 + 4 * lane - 2) &&
                (abs(check_car_s - car_s) < 30)) {
              car_left = true;
            }

            // Same logic but applied to the right lane
            if ((d > 2 + 4 * lane + 2) && (d < 2 + 4 * lane + 6) &&
                (abs(check_car_s - car_s) < 30)) {
              car_right = true;
            }
          }

          // If we are approaching a car in our lane too closely...
          if (too_close) {
            // If we're not in the left lane and there isn't a car
            // in the left lane, move to the left
            if (lane > 0 && !car_left) {
              lane--;
            } else if (!car_right && lane < 2) {
              // If we're not in the right lane and there isn't a car
              // in the right lane, move to the right
              lane++;
            }
          } else {
            // If we're not too close and if we're not in the middle lane
            if (lane != 1) {
              // If we're on the left lane and there isn't a car on the
              // right lane, or if we're in the right lane and there isn't
              // a car on the left lane, move back to the middle
              if ((lane == 0 && !car_right) || (lane == 2 && !car_left)) {
                lane = 1;
              }
            }
          }

          // Storing the first two way points
          // First access the current car's map x and y coordinates
          // and orientation / heading
          vector<double> ptsx;
          vector<double> ptsy;

          double ref_x = car_x;
          double ref_y = car_y;
          double ref_yaw = deg2rad(car_yaw);

          // If there was only one waypoint from the previous iteration
          // then we create a second waypoint that is tangent along
          // the trajectory of the previous waypoint
          // Note that we'll just keep the yaw for the current iteration
          if (prev_size < 2) {
            double prev_car_x = car_x - cos(car_yaw);
            double prev_car_y = car_y - sin(car_yaw);

            ptsx.push_back(prev_car_x);
            ptsx.push_back(car_x);
            ptsy.push_back(prev_car_y);
            ptsy.push_back(car_y);
          } else {
            // If there were more, then access the previous two points
            ref_x = previous_path_x[prev_size - 1];
            ref_y = previous_path_y[prev_size - 1];

            double ref_x_prev = previous_path_x[prev_size - 2];
            double ref_y_prev = previous_path_y[prev_size - 2];

            // We have a more reliable way to calculate the heading with the
            // two previous points
            ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

            ptsx.push_back(ref_x_prev);
            ptsx.push_back(ref_x);
            ptsy.push_back(ref_y_prev);
            ptsy.push_back(ref_y);
          }

          // In Frenet space, we will add three waypoints at 30, 60
          // and 90 m out
          // What's nice here is that the lane variable changes depending on
          // whether we need to pass a car or not, so the waypoints get
          // adjusted to the left or right lane depending on whether we need
          // to change
          // Convert from Frenet to XY
          vector<double> next_wp0 =
              getXY(car_s + 30, (2 + 4 * lane), map_waypoints_s,
                    map_waypoints_x, map_waypoints_y);
          vector<double> next_wp1 =
              getXY(car_s + 60, (2 + 4 * lane), map_waypoints_s,
                    map_waypoints_x, map_waypoints_y);
          vector<double> next_wp2 =
              getXY(car_s + 90, (2 + 4 * lane), map_waypoints_s,
                    map_waypoints_x, map_waypoints_y);

          ptsx.push_back(next_wp0[0]);
          ptsx.push_back(next_wp1[0]);
          ptsx.push_back(next_wp2[0]);

          ptsy.push_back(next_wp0[1]);
          ptsy.push_back(next_wp1[1]);
          ptsy.push_back(next_wp2[1]);

          // Given all of the waypoints we've defined so far, we need to
          // define the points with respect to the car's local frame
          // The waypoints above are wrt to the map frame
          // Perform a rotation and translation to do that
          for (int i = 0; i < ptsx.size(); i++) {
            double shift_x = ptsx[i] - ref_x;
            double shift_y = ptsy[i] - ref_y;
            ptsx[i] = shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw);
            ptsy[i] = shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw);
          }

          // Now define a spline to help us interpolate between these waypoints
          // for smooth transition to a different lane, or if we just need to
          // keep our lane, that's fine too.  The spline will just define
          // waypoints along the same lane
          tk::spline s;
          s.set_points(ptsx, ptsy);

          // Make sure we follow along the path of the previous points first
          // so that we discourage jerking
          for (int i = 0; i < previous_path_x.size(); i++) {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          // This logic allows for the gradual transition of a lane change
          // The target distance (s value) is 30 metres out
          // and we need to predict the map coordinates y that allow us
          // to smoothly move to the desired lane
          double target_x = 30;
          double target_y = s(target_x);
          double target_dist = sqrt(target_x * target_x + target_y * target_y);

          // We want to define 50 waypoints, and sampling each waypoint at
          // 0.02 seconds results in 25 m/s which is ~ 50 MPH our desired
          // speed

          // This accumulates the x value so that we are properly
          // able to define them along the path from where we started
          // to where we need to go
          double x_add_on = 0;

          // Create waypoints until we reach our 50 mark, starting from
          // the points from the previous path
          for (int i = 1; i <= 50 - previous_path_x.size(); i++) {
            // If we are too close to a vehicle in front of us, slow
            // down at ~ 5 m/s^2
            if (too_close) {
              ref_vel -= .224;
            } else if (ref_vel < 49.5) {
              // If we're not close and we're slow, speed up back to ~50 MPH
              // at 5 m/s^2
              ref_vel += .224;
            }

            // As we speed up, the waypoints will progressively get farther
            // out or closer if we slow down (like on a curve)
            // 2.24 is ~10 m/s^2 - our prescribed limit
            // This will properly condense or expand out the waypoints
            double N = target_dist / (0.02 * ref_vel / 2.24);
            // Calculate the right x waypoint, then interpolate with splines
            double x_point = x_add_on + target_x / N;
            double y_point = s(x_point);
            x_add_on = x_point;

            // Remember that the current coordinate frame is wrt the ego
            // vehicle - must transform back to map coordinates
            double x_ref = x_point;
            double y_ref = y_point;

            x_point = (x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw));
            y_point = (x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw));
            x_point += ref_x;
            y_point += ref_y;

            // Save these next waypoints
            next_x_vals.push_back(x_point);
            next_y_vals.push_back(y_point);
          }

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\"," + msgJson.dump() + "]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  });  // end h.onMessage

  h.onConnection([](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }

  h.run();
}