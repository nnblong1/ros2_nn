#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry
from std_srvs.srv import Trigger
import math
import sys
import time

class HoverChecker(Node):
    def __init__(self):
        super().__init__('hover_checker')
        self.sub = self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.odom_callback, rclpy.qos.qos_profile_sensor_data)
        self.cli = self.create_client(Trigger, '/uam/arm_takeoff')
        self.start_time = None
        self.takeoff_triggered = False
        self.takeoff_time = 0
        self.get_logger().info("Hover checker started. Waiting for odometry...")

    def odom_callback(self, msg):
        now = time.time()

        if self.start_time is None:
            self.start_time = now
            self.get_logger().info("Receiving Odom! Waiting 5s before takeoff...")
        
        # Trigger takeoff after 5s of getting data
        if not self.takeoff_triggered and (now - self.start_time > 5.0):
            if not self.cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn("Service not ready yet...")
                return
            req = Trigger.Request()
            self.cli.call_async(req)
            self.takeoff_triggered = True
            self.takeoff_time = now
            self.get_logger().info("Takeoff triggered.")
            return

        if not self.takeoff_triggered:
            return

        # Check for crash or success
        elapsed = now - self.takeoff_time
        
        q0, q1, q2, q3 = msg.q[0], msg.q[1], msg.q[2], msg.q[3]
        sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        sinp = 2.0 * (q0 * q2 - q3 * q1)
        if abs(sinp) >= 1: pitch = math.copysign(math.pi / 2.0, sinp)
        else: pitch = math.asin(sinp)

        enu_z = -msg.position[2]

        if elapsed > 1.0:
            if abs(roll) > math.radians(45) or abs(pitch) > math.radians(45):
                self.get_logger().error(f"FLIPPED! Roll: {math.degrees(roll):.1f}, Pitch: {math.degrees(pitch):.1f}")
                sys.exit(1)
            if enu_z < 0.2 and elapsed > 8.0:
                self.get_logger().error("FAILED TO CLIMB")
                sys.exit(1)
        
        if elapsed > 15.0:
            if enu_z > 1.0:
                self.get_logger().info("SUCCESS: Hovered.")
                sys.exit(0)
            else:
                self.get_logger().error("FAILED TO CLIMB")
                sys.exit(1)

def main(args=None):
    rclpy.init(args=args)
    checker = HoverChecker()
    try:
        rclpy.spin(checker)
    except SystemExit as e:
        sys.exit(e.code)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
