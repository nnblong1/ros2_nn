#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry
from std_msgs.msg import Float64MultiArray
import csv
import time
import math
import os
import sys

class RBFNNDataLogger(Node):
    def __init__(self):
        super().__init__('rbfnn_data_logger')
        
        self.filename = 'rbfnn_flight_data.csv'
        if os.path.exists(self.filename):
            os.remove(self.filename)
            
        self.csv_file = open(self.filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['time', 'alt_z', 'roll', 'pitch', 'm_hat', 'n0_x', 'n0_y', 'n0_z'])
        
        self.start_time = time.time()
        
        self.enu_z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.m_hat = 0.0
        self.n0 = [0.0, 0.0, 0.0]
        
        self.sub_odom = self.create_subscription(
            VehicleOdometry,
            "/fmu/out/vehicle_odometry",
            self.odom_cb,
            rclpy.qos.qos_profile_sensor_data
        )
        self.sub_debug = self.create_subscription(
            Float64MultiArray,
            "/uam/debug_state",
            self.debug_cb,
            10
        )
        
        self.timer = self.create_timer(0.05, self.timer_cb) # 20Hz logging
        self.get_logger().info("Data Logger started. Logging at 20Hz.")

    def odom_cb(self, msg: VehicleOdometry):
        self.enu_z = -msg.position[2]
        
        q0, q1, q2, q3 = msg.q[0], msg.q[1], msg.q[2], msg.q[3]
        sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
        self.roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
        
        sinp = 2.0 * (q0 * q2 - q3 * q1)
        if abs(sinp) >= 1:
            self.pitch = math.degrees(math.copysign(math.pi / 2.0, sinp))
        else:
            self.pitch = math.degrees(math.asin(sinp))

    def debug_cb(self, msg: Float64MultiArray):
        if len(msg.data) >= 17:
            self.m_hat = msg.data[13]
            self.n0 = [msg.data[14], msg.data[15], msg.data[16]]

    def timer_cb(self):
        t = time.time() - self.start_time
        
        # Giám sát Rớt / Bay quá cao / Nghiêng quá mức (Nới lỏng theo yêu cầu người dùng)
        if t > 20.0:
            # Cho phép drone nghiêng tới 80 độ mới coi là lật (rất lỏng)
            if abs(self.roll) > 80.0 or abs(self.pitch) > 80.0:
                self.get_logger().error(f"💥 PHÁT HIỆN LẬT: Roll={self.roll:.1f}, Pitch={self.pitch:.1f}")
                self.csv_file.close()
                os._exit(1)
            
            # Cho phép độ cao từ -0.5m đến 10m (tránh lỗi trôi EKF trên mặt đất)
            if self.enu_z < -0.5 or self.enu_z > 10.0:
                self.get_logger().error(f"💥 PHÁT HIỆN RỚT/VỌT: Độ cao={self.enu_z:.2f}m")
                self.csv_file.close()
                os._exit(1)

                
        self.csv_writer.writerow([
            f"{t:.3f}", f"{self.enu_z:.3f}", f"{self.roll:.3f}", f"{self.pitch:.3f}",
            f"{self.m_hat:.3f}", f"{self.n0[0]:.4f}", f"{self.n0[1]:.4f}", f"{self.n0[2]:.4f}"
        ])
        
        if int(t * 20) % 20 == 0:
            self.csv_file.flush()

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RBFNNDataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
