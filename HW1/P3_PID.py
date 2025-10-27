#!/usr/bin/env python3

import numpy as np
import rclpy

from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):

    def __init__(self):
        super().__init__()
        self.kp = 5
    def compute_control_with_goal(self, curr_state: TurtleBotState, des_state: TurtleBotState)->TurtleBotControl:
        heading_error = des_state.theta - curr_state.theta
        wrapped_error = wrap_angle(heading_error)
        angular_velocity = wrapped_error * self.kp
        control = TurtleBotControl()
        control.omega = angular_velocity
        return control
    
if __name__ == "__main__":
    rclpy.init()
    node = HeadingController()
    rclpy.spin(node)
    rclpy.shutdown()