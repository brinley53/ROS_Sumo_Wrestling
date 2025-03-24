#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist

# Constants
TURNING_SPEED = 0.5 / 100
MOVING_SPEED = 0.1
WALL_DISTANCE = 0.25  # Minimum distance from wall (meters)
ESCAPE_SPEED = 0.2  # Speed when escaping a wall
TURN_SPEED = 0.5  # Speed for turning away from walls
CENTER_TOLERANCE = 20  # Error margin for "centered" opponent
ATTACK_SPEED = 0.3
MIN_CONTOUR_SIZE = 200

class ColorTracking(Node):
    def __init__(self, name):
        super().__init__(name)
        self.escape_attempts = 0  # Track consecutive escapes
        self.last_escape_time = 0

        # ROS2 publishers and subscribers
        self.cmd_vel = self.create_publisher(Twist, 'controller/cmd_vel', 1)
        self.subscription = self.create_subscription(
            Image,
            'ascamera/camera_publisher/rgb0/image',
            self.listener_callback,
            1)
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            1)

        self.bridge = CvBridge()
        self.lidar_data = []

    def lidar_callback(self, data):
        """ Update LiDAR readings and store distance measurements. """
        self.lidar_data = data.ranges  # Store full 360° scan

    def get_wall_direction(self):
        """ Determine if a wall is too close and return the direction to move away. """
        if not self.lidar_data:
            return None

        # Read distances in different directions (excluding front)
        back = min(self.lidar_data[150:210])  # 150° - 210°
        left = min(self.lidar_data[60:120])  # 60° - 120°
        right = min(self.lidar_data[240:300])  # 240° - 300°

        # Determine closest wall and move away
        if back < WALL_DISTANCE:
            return "forward"
        elif left < WALL_DISTANCE:
            return "right"
        elif right < WALL_DISTANCE:
            return "left"

        return None

    def listener_callback(self, data):
        """ Process camera input and decide movement. """
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
        escape_direction = self.get_wall_direction()

        if escape_direction == "forward":
            twist.linear.x = ESCAPE_SPEED  # Move forward
        elif escape_direction == "left":
            twist.angular.z = TURN_SPEED  # Turn left
        elif escape_direction == "right":
            twist.angular.z = -TURN_SPEED  # Turn right
        elif centroid_x is not None and centroid_y is not None:
            # Target detected: Align and move forward
            image_center_x = current_frame.shape[1] // 2
            error_x = centroid_x - image_center_x

            twist.angular.z = -error_x * TURNING_SPEED  # Proportional turn

            if abs(error_x) < CENTER_TOLERANCE:
                twist.linear.x = ATTACK_SPEED  # Charge forward aggressively
            else:
                twist.linear.x = MOVING_SPEED  # Normal approach speed
        else:
            # No target detected: Rotate to scan for opponent
            twist.linear.x = 0.0
            twist.angular.z = 0.5

        self.cmd_vel.publish(twist)

        #*********** DEBUG - Display mask and centroid **************
        cv2.circle(current_frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)  #draws a centroid
        cv2.imshow('Mask', mask)
        cv2.imshow('Camera', current_frame)
        cv2.waitKey(1)  #refresh display
        #This might show a blank if the bounds are wrong. Better to test 
        #*********** END DEBUG ****************

        #Check escape attempts
        if escape_direction is not None:
            self.escape_attempts += 1
            self.last_escape_time = time.time()
            if self.escape_attempts > 3:  # Stuck after 3 attempts
                twist.angular.z = 1.0 * (-1 if escape_direction == "left" else 1)  # Hard turn
            else:
                self.escape_attempts = 0  # Reset counter
                

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
