#!/usr/bin/env python3

from asl_tb3_lib.navigation import BaseNavigator
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.navigation import TrajectoryPlan
from numpy import linalg
import scipy.interpolate
from P1_astar import DetOccupancyGrid2D, AStar
from utils import generate_planning_problem
from utils import plot_line_segments


import rclpy                    # ROS2 client library
from rclpy.node import Node     # ROS2 node baseclass
from scipy.interpolate import splev
import numpy as np


class TurtleNavigator(BaseNavigator):
    V_PREV_THRES = 0.0001 # moved here
    
    def __init__(self) -> None:
        # give it a default node name
        super().__init__("turtle_navigator")
        self.get_logger().info("TurtleNavigator node started!")
        self.kpx = 1.0
        self.kpy = 1.0
        self.kdx = 0.5
        self.kdy = 0.5
        self.kp = 0.2
        self.V_max = 0.5  # maximum linear velocity
        self.om_max = 1.0 # maximum angular velocity
        self.reset()
        
        # Define start and goal
        self.start = (0.0, 0.0)   # example starting position
        self.goal = (2.0, 3.0)    # example goal position


        
    def reset(self):
        self.t_prev = 0.0
        self.V_prev = 0.0
        self.om_prev = 0.0
    
    def compute_control_with_goal(self, curr_state: TurtleBotState, des_state: TurtleBotState)->TurtleBotControl:
        heading_error = des_state.theta - curr_state.theta
        wrapped_error = wrap_angle(heading_error)
        angular_velocity = wrapped_error * self.kp
        control = TurtleBotControl()
        control.omega = angular_velocity
        return control
    
    def compute_trajectory_tracking_control(self,
        state: TurtleBotState, # has v and omega
        plan: TrajectoryPlan, # has path (waypoints), path_x_spline, path_y_spline (both are time series), and duration (total time in float)
        t: float,
    ) -> TurtleBotControl:
    
        # get time step
        dt = t - self.t_prev
        
        # get desired pos, vel, acc from spline at specific time
        x_d = splev(t, plan.path_x_spline) # x pos
        xd_d = splev(t, plan.path_x_spline, der=1) # x vel
        xdd_d = splev(t, plan.path_x_spline, der=2) # x acc
        y_d = splev(t, plan.path_y_spline)
        yd_d = splev(t, plan.path_y_spline, der=1)
        ydd_d = splev(t, plan.path_y_spline, der=2)
        
        # unpack turtlebotstate
        x = state.x
        y = state.y
        th = state.theta

        
        # avoid singularity
        if abs(self.V_prev) < self.V_PREV_THRES:
            self.V_prev = self.V_PREV_THRES

        
        xd = self.V_prev*np.cos(th)
        yd = self.V_prev*np.sin(th)

        # compute virtual controls
        u = np.array([xdd_d + self.kpx*(x_d-x) + self.kdx*(xd_d-xd), ydd_d + self.kpy*(y_d-y) + self.kdy*(yd_d-yd)])

        # compute real controls
        J = np.array([[np.cos(th), -self.V_prev*np.sin(th)],[np.sin(th), self.V_prev*np.cos(th)]])
        a, om = linalg.solve(J, u)
        V = self.V_prev + a*dt
        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        return TurtleBotControl(v=V, omega=om)
    
    def compute_smooth_plan(self, path, v_desired=0.15, spline_alpha=0.05) -> TrajectoryPlan:
        # Ensure path is a numpy array
        path = np.asarray(path)

        # Compute and set the following variables:
        #   1. ts: 
        #      Compute an array of time stamps for each planned waypoint assuming some constant 
        #      velocity between waypoints. 
        #
        #   2. path_x_spline, path_y_spline:
        #      Fit cubic splines to the x and y coordinates of the path separately
        #      with respect to the computed time stamp array.
        #      Hint: Use scipy.interpolate.splrep
        
        ##### YOUR CODE STARTS HERE #####
        # timestamp is cumulative euclidean distance between each point divided by v_desired
        
        deltas = np.diff(path, axis=0) # get delta between each point on path
        segment_lengths = np.linalg.norm(deltas, axis=1) # use euclid dist
        total_length = np.sum(segment_lengths) # sum them up
        cumulative_distances = np.insert(np.cumsum(segment_lengths), 0, 0) 
        ts = cumulative_distances / v_desired 
        
        # using the timestamps we can interpolate x and y using the splrep function
        path_x_spline = scipy.interpolate.splrep(ts, path[:, 0], s=spline_alpha)
        path_y_spline = scipy.interpolate.splrep(ts, path[:, 1], s=spline_alpha)
        ###### YOUR CODE END HERE ######
        
        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )
    
    def compute_trajectory_plan(self, start: tuple, goal: tuple) -> TrajectoryPlan:
        # 1. Solve A* problem
        width = self.occupancy_grid.width
        height = self.occupancy_grid.height
        astar = AStar(
            statespace_lo=(0, 0),
            statespace_hi=(width, height),
            x_init=start,
            x_goal=goal,
            occupancy=self.occupancy_grid,
            resolution=self.resolution
        )
        
        if not astar.solve():
            self.get_logger().warn("No path found by A*")
            return None
        
        # 2. Get raw path
        path = astar.path  # list of (x, y)
        
        # 3. Smooth path using existing function
        plan = self.compute_smooth_plan(path)
        
        # 4. Reset previous control states
        self.reset()
        
        # 5. Return TrajectoryPlan
        return plan




if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = TurtleNavigator()    # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exits

