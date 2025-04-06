#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import time

from collections import deque   # BFS uses queue

# Constants
TURNING_SPEED = 0.5 / 100
MOVING_SPEED = 0.1
WALL_DISTANCE = 0.1  # Minimum distance from wall (meters) (4 in ~ 0.1 m)
#ESCAPE_SPEED = 0.2  # Speed when escaping a wall
TURN_SPEED = 0.5  # Speed for turning away from walls
#CENTER_TOLERANCE = 20  # Error margin for "centered" opponent
#ATTACK_SPEED = 0.6  # Increased attack speed
#MIN_CONTOUR_SIZE = 200
START_SPEED = 0.3  # New start speed for forward movement and drifting
#DRIFT_TURN_SPEED = -1.0  # Angular speed for drifting right (negative for right turn)
#DRIFT_DURATION = 1.5  # Time to complete the drift
FORWARD_DISTANCE = 0.3  # Distance to move forward (meters)

END_DISTANCE = 0.1 # Minimum distance from green wall (meters) (4 in ~ 0.1 m)
FORWARD_SPEED = 0.2
COLOR_TOLERANCE = 20  # For color detection centering

#Green color range      *** CHANGE TO GREEN ROOM COLOR ****
GREEN_LOWER = np.array([0, 0, 0])
GREEN_UPPER = np.array([0, 0, 0])

class MazeNavigator(Node):
    def __init__(self, name):
        super().__init__(name)
        
        # ROS2 publishers and subscribers
        self.cmd_vel = self.create_publisher(Twist, 'controller/cmd_vel', 1)

        #self.start = True

        self.camera_sub = self.create_subscription(
            Image, 'ascamera/camera_publisher/rgb0/image', 
            self.camera_callback, 1)
        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback, 1)
            
        self.bridge = CvBridge()
        self.lidar_data = []
        self.at_end = False
        
        # Maze navigation variables
        self.maze_graph = {}  # Graph representation of maze
        self.visited = set()  # Visited nodes
        self.current_node = (0, 0)  # Current position in grid
        self.heading = 0  # 0=N, 1=E, 2=S, 3=W
        self.path = []  # BFS path to follow
        self.path_index = 0  # Current path position
        
        # State management
        self.following_wall = False
        self.wall_side = 'right'    # start at right for now
        self.first_attempt = True
        self.start_time = time.time()
        self.mapping_complete = False
        self.position_history = []
        
        
        # Timing for attempts
        self.start_time = time.time()
        self.first_attempt_duration = 240  # 4 minutes
        self.second_attempt_duration = 120  # 2 minutes


    def lidar_callback(self, data):
        """Process Lidar data for wall detection and navigation"""
        self.lidar_data = data.ranges
        
        if self.at_end:
            return
            
        twist = Twist()
        
        if self.first_attempt and not self.mapping_complete:
            # Mapping attempt
            self.map_maze(twist)
        else:
            if self.first_attempt and not self.mapping_complete:
                # Transition to execution phase
                self.mapping_complete = True
                self.plan_path()
                self.first_attempt = False
                
            # Actual attempt
            self.follow_path(twist)
            
        self.cmd_vel.publish(twist)
       
    def map_maze(self, twist):
        """First attempt: Explore maze and build graph"""
        if not self.following_wall:
            self.find_wall()
        else:
            self.explore_with_wall_following(twist)
            self.record_position()
        
    def explore_with_wall_following(self, twist):
        """Use right hand wall method"""
        front_dist = min(min(self.lidar_data[:30]), min(self.lidar_data[330:]))
        right_dist = min(self.lidar_data[240:300])
        left_dist = min(self.lidar_data[60:120])
        
        # Wall following logic
        if front_dist < WALL_DISTANCE:
            twist.angular.z = TURN_SPEED  # Turn left
            twist.linear.x = 0.0
            self.record_turn('left')
        elif right_dist > WALL_DISTANCE * 1.5:
            twist.angular.z = -TURN_SPEED * 0.5  # Turn right slightly
            twist.linear.x = FORWARD_SPEED * 0.5
        elif right_dist < WALL_DISTANCE * 0.7:
            twist.angular.z = TURN_SPEED * 0.3  # Turn left slightly
            twist.linear.x = FORWARD_SPEED * 0.7
        else:
            twist.linear.x = FORWARD_SPEED  # Move forward
            twist.angular.z = 0.0
            self.record_straight()

    def record_position(self):
        """Record current node and connections in graph"""
        front_clear = min(min(self.lidar_data[:30]), min(self.lidar_data[330:])) > WALL_DISTANCE * 1.5
        right_clear = min(self.lidar_data[240:300]) > WALL_DISTANCE * 1.5
        left_clear = min(self.lidar_data[60:120]) > WALL_DISTANCE * 1.5
        
        connections = []
        if front_clear:
            connections.append('front')
        if right_clear:
            connections.append('right')
        if left_clear:
            connections.append('left')
        
        # Add reverse if we're not at start
        if self.current_node != (0, 0):
            connections.append('back')
        
        self.maze_graph[self.current_node] = connections
        self.visited.add(self.current_node)

    def record_turn(self, direction):
        """Update position after a turn"""
        if direction == 'left':
            self.heading = (self.heading - 1) % 4
        elif direction == 'right':
            self.heading = (self.heading + 1) % 4

    def record_straight(self):
        """Update position after moving straight"""
        x, y = self.current_node
        # Put nswe directions for now to get a rough idea of the map
        # North is the front of wherever Mox is facing after placing
        if self.heading == 0:  #north
            self.current_node = (x, y + 1)
        elif self.heading == 1:  #east
            self.current_node = (x + 1, y)
        elif self.heading == 2:  #south
            self.current_node = (x, y - 1)
        elif self.heading == 3:  #west
            self.current_node = (x - 1, y)

    def plan_path(self):
        """2nd ATTEMPT: after mapping using BFS"""
        start = (0, 0)
        end = self.find_farthest_point()
        
        if end is None:
            self.get_logger().warn("Could not determine end point, using farthest point")
            end = max(self.visited, key=lambda p: abs(p[0]) + abs(p[1]))
        
        self.path = self.bfs(start, end)
        
        if self.path:
            self.get_logger().info(f"Path planned: {self.path}")
            self.path_index = 0
        else:
            self.get_logger().error("No path found to green room")

    def bfs(self, start, end):
        """BFS algorithm"""
        queue = deque()
        queue.append([start])
        visited = set([start])
        
        while queue:
            path = queue.popleft()
            node = path[-1]
            
            if node == end:
                return path
                
            for neighbor in self.get_neighbors(node):
                if neighbor not in visited and neighbor in self.maze_graph:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
                    
        return None

    def get_neighbors(self, node):
        """Get accessible neighbors from current node"""
        x, y = node
        neighbors = []
        directions = self.maze_graph.get(node, [])
        
        if 'front' in directions:
            if self.heading == 0:  # Facing North
                neighbors.append((x, y + 1))
            elif self.heading == 1:  # Facing East
                neighbors.append((x + 1, y))
            elif self.heading == 2:  # Facing South
                neighbors.append((x, y - 1))
            elif self.heading == 3:  # Facing West
                neighbors.append((x - 1, y))
                
        if 'right' in directions:
            if self.heading == 0:  # Facing North
                neighbors.append((x + 1, y))
            elif self.heading == 1:  # Facing East
                neighbors.append((x, y - 1))
            elif self.heading == 2:  # Facing South
                neighbors.append((x - 1, y))
            elif self.heading == 3:  # Facing West
                neighbors.append((x, y + 1))
                
        if 'left' in directions:
            if self.heading == 0:  # Facing North
                neighbors.append((x - 1, y))
            elif self.heading == 1:  # Facing East
                neighbors.append((x, y + 1))
            elif self.heading == 2:  # Facing South
                neighbors.append((x + 1, y))
            elif self.heading == 3:  # Facing West
                neighbors.append((x, y - 1))
                
        if 'back' in directions:
            if self.heading == 0:  # Facing North
                neighbors.append((x, y - 1))
            elif self.heading == 1:  # Facing East
                neighbors.append((x - 1, y))
            elif self.heading == 2:  # Facing South
                neighbors.append((x, y + 1))
            elif self.heading == 3:  # Facing West
                neighbors.append((x + 1, y))
                
        return neighbors

    def follow_path(self, twist):
        """Follow the planned BFS path"""
        if not self.path or self.path_index >= len(self.path):
            return
            
        target_node = self.path[self.path_index]
        
        if self.current_node == target_node:
            self.path_index += 1
            if self.path_index >= len(self.path):
                return
            target_node = self.path[self.path_index]
            
        self.move_to_node(twist, target_node)

    def move_to_node(self, twist, target_node):
        """Move toward the target node"""
        current_x, current_y = self.current_node
        target_x, target_y = target_node

        '''
        It should go something like this
        Turn Decisions:
        Current | Target | Turn
        -----------------------------
        North   | East   | Right 
        East    | South  | Right   
        South   | West   | Right 
        West    | North  | Right 
        North   | West   | Left 

            0
        3 <-  -> 1   NSWE directions look like this
            2
        '''
        
        # Determine required movement
        # Need to move North
        if target_y > current_y:
            if self.heading == 0:  # Already facing North
                twist.linear.x = FORWARD_SPEED
            elif self.heading == 1:  # Facing East
                twist.angular.z = TURN_SPEED  # Turn left
            elif self.heading == 2:  # Facing South
                twist.angular.z = TURN_SPEED  # Turn left (or right, could do 180)
            elif self.heading == 3:  # Facing West
                twist.angular.z = -TURN_SPEED  # Turn right
         # Need to move East
        elif target_x > current_x:
            if self.heading == 0:  # Facing North
                twist.angular.z = -TURN_SPEED  # Turn right
            elif self.heading == 1:  # Already facing East
                twist.linear.x = FORWARD_SPEED
            elif self.heading == 2:  # Facing South
                twist.angular.z = TURN_SPEED  # Turn left
            elif self.heading == 3:  # Facing West
                twist.angular.z = TURN_SPEED  # Turn left (or right, could do 180)
        # Need to move South
        elif target_y < current_y:
            if self.heading == 0:  # Facing North
                twist.linear.x = 0.0
                twist.angular.z = TURN_SPEED  # Turn left (180° turn)
            elif self.heading == 1:  # Facing East
                twist.linear.x = 0.0
                twist.angular.z = -TURN_SPEED  # Turn right
            elif self.heading == 2:  # Already facing South
                twist.linear.x = FORWARD_SPEED
                twist.angular.z = 0.0
            elif self.heading == 3:  # Facing West
                twist.linear.x = 0.0
                twist.angular.z = TURN_SPEED  # Turn left 
        # Need to move West
        elif target_x < current_x:
            if self.heading == 0:  # Facing North
                twist.linear.x = 0.0
                twist.angular.z = TURN_SPEED  # Turn left
            elif self.heading == 1:  # Facing East
                twist.linear.x = 0.0
                twist.angular.z = TURN_SPEED  # Turn left (180° turn)
            elif self.heading == 2:  # Facing South
                twist.linear.x = 0.0
                twist.angular.z = -TURN_SPEED  # Turn right
            elif self.heading == 3:  # Already facing West
                twist.linear.x = FORWARD_SPEED
                twist.angular.z = 0.0
                
        else:  # Already at target position
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.path_index += 1  # Move to next node in path


    def find_farthest_point(self):
        """Find the farthest explored point from start"""
        if not self.visited:
            return None
        return max(self.visited, key=lambda p: p[0]**2 + p[1]**2)

    def camera_callback(self, data):
        """Detect green room"""
        if self.at_end:
            return
            
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        blurred = cv2.GaussianBlur(lab_frame, (5, 5), 0)
        
        green_mask = cv2.inRange(blurred, GREEN_LOWER, GREEN_UPPER)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                if self.lidar_data and min(self.lidar_data[:30] + self.lidar_data[330:]) < END_DISTANCE:
                    self.at_end = True
                    twist = Twist()
                    self.cmd_vel.publish(twist)
                    self.get_logger().info("GREEN ROOM REACHED")

def main(args=None):
    rclpy.init(args=args)
    navigator = MazeNavigator('maze_navigator')
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        navigator.get_logger().info("Keyboard Interrupt (Ctrl+C): Stopping node.")
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
