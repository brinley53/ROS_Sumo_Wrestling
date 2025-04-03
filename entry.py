#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import time

# Constants
TURNING_SPEED = 0.5 / 100
MOVING_SPEED = 0.1
WALL_DISTANCE = 0.25  # Minimum distance from wall (meters)
ESCAPE_SPEED = 0.2  # Speed when escaping a wall
TURN_SPEED = 0.5  # Speed for turning away from walls
CENTER_TOLERANCE = 20  # Error margin for "centered" opponent
ATTACK_SPEED = 0.6  # Increased attack speed
MIN_CONTOUR_SIZE = 200
START_SPEED = 0.3  # New start speed for forward movement and drifting
DRIFT_TURN_SPEED = -1.0  # Angular speed for drifting right (negative for right turn)
DRIFT_DURATION = 1.5  # Time to complete the drift
FORWARD_DISTANCE = 0.3  # Distance to move forward (meters)

class ColorTracking(Node):
    def __init__(self, name):
        super().__init__(name)

        # ROS2 publishers and subscribers
        self.cmd_vel = self.create_publisher(Twist, 'controller/cmd_vel', 1)
        
        self.start = True
        
        self.subscription = self.create_subscription(
            Image,
            'ascamera/camera_publisher/rgb0/image',
            self.listener_callback,
            1)
        
        self.move_forward_and_drift_right()

            
        self.charge = False
        self.bridge = CvBridge()

    def move_forward_and_drift_right(self):
        """ Move forward 0.2 meters and then drift 180 degrees to the right. """
        twist = Twist()
        input("Enter...")
        
        twist.linear.y = START_SPEED  # Keep moving forward at the start speed
        twist.angular.z = DRIFT_TURN_SPEED  # Smooth rightward arc
        self.cmd_vel.publish(twist)
        time.sleep(DRIFT_DURATION)

        # Stop drifting
        twist.linear.y = 0.0
        twist.angular.z = 0.0
        self.cmd_vel.publish(twist)
        self.start = False

    def listener_callback(self, data):
        """ Process camera input and decide movement. """
        if self.start:
        	return
        current_frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

        # Convert BGR to LAB color space
        lab_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2LAB)
        blurred_frame = cv2.GaussianBlur(lab_frame, (5, 5), 0)

        # Define LAB color range for detecting neon orange
        lower_bound = np.array([80, 140, 0])
        upper_bound = np.array([180, 220, 230])

        # Create a binary mask for detected color
        mask = cv2.inRange(blurred_frame, lower_bound, upper_bound)

        # Get centroid of detected object
        centroid_x, centroid_y = self.get_color_centroid(mask)

        twist = Twist()

        # Check if we need to move away from a wall

        if centroid_x is not None and centroid_y is not None:
            # Target detected: Align and move forward
            image_center_x = current_frame.shape[1] // 2
            error_x = centroid_x - image_center_x

            
            if abs(error_x) < CENTER_TOLERANCE:
                self.charge = True
           
            twist.angular.z = -error_x * TURNING_SPEED  # Proportional turn
            	
        else:
            # No target detected: Rotate to scan for opponent
            twist.angular.z = 1.25
            self.charge = False
            
        twist.linear.x = ATTACK_SPEED if self.charge else 0.0

        self.cmd_vel.publish(twist)
        
        

    def get_color_centroid(self, mask):
        """ Compute the centroid of the largest detected contour. """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < MIN_CONTOUR_SIZE:
            return None, None

        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None, None

        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])

        return centroid_x, centroid_y

def main(args=None):
    rclpy.init(args=args)
    color_tracking_node = ColorTracking('color_tracking_node')
    try:
        rclpy.spin(color_tracking_node)
    except KeyboardInterrupt:
        color_tracking_node.get_logger().info("Keyboard Interrupt (Ctrl+C): Stopping node.")
    finally:
        color_tracking_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
