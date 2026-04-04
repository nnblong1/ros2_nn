#!/usr/bin/env python3
"""
rbfnn_training_supervisor.py
----------------------------
Mission orchestrator cho quá trình huấn luyện RBFNN tự động.
- Phase 1: Gửi lệnh takeoff, đợi 15s cho RBFNN hội tụ các trọng số ban đầu.
- Phase 2: Gọi arm_trajectory_generator.py chạy pattern combined để huấn luyện.
- Crash Detection: Giám sát Odom, nếu rớt độ cao hoặc lật góc quá mức sẽ exit(1).
"""

import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry
from std_srvs.srv import Trigger
import math
import sys
import time
import subprocess
import os
import signal

class RBFNNTrainingSupervisor(Node):
    def __init__(self):
        super().__init__('rbfnn_training_supervisor')
        
        # Cấu hình
        self.phase1_duration = 40.0 # giây
        self.phase2_duration = 180.0 # giây (thời gian chạy pattern combined)
        self.crash_z_threshold = 0.2 # độ cao ENU (m) coi là rớt
        self.crash_angle_threshold = math.radians(60.0) # 60 độ
        
        # State
        self.start_time = time.time()
        self.init_wait_duration = 14.0
        self.has_taken_off = False
        self.takeoff_time = 0.0
        self.has_reached_safe_altitude = False
        self.phase2_started = False
        self.arm_process = None
        
        # Launch logger
        cmd = ["ros2", "run", "uam_controller", "rbfnn_data_logger.py"]
        self.logger_process = subprocess.Popen(cmd, env=os.environ.copy())
        
        # Subscriber để giám sát crash
        self.sub_odom = self.create_subscription(
            VehicleOdometry,
            "/fmu/out/vehicle_odometry",
            self.odom_callback,
            rclpy.qos.qos_profile_sensor_data
        )
        
        # Service client để gửi lệnh takeoff
        self.cli_arm_takeoff = self.create_client(Trigger, '/uam/arm_takeoff')
        
        # Timer chính
        self.timer = self.create_timer(1.0, self.main_loop)
        
        self.get_logger().info("🔥 Khởi động RBFNN Training Supervisor")
        
    def odom_callback(self, msg: VehicleOdometry):
        # Tính toán độ cao (ENU Z) từ (NED Z)
        enu_z = -msg.position[2]
        
        # Tính roll/pitch từ quaternion (w, x, y, z)
        q0 = msg.q[0]
        q1 = msg.q[1]
        q2 = msg.q[2]
        q3 = msg.q[3]
        
        # Roll
        sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch
        sinp = 2.0 * (q0 * q2 - q3 * q1)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)
            
        # Kiểm tra điều kiện Crash (chỉ check khi đã takeoff đủ lâu, ví dụ sau 5s)
        if self.has_taken_off and (time.time() - self.takeoff_time > 5.0):
            if abs(roll) > self.crash_angle_threshold or abs(pitch) > self.crash_angle_threshold:
                self.get_logger().error(f"💥 Phát hiện rớt: Lật góc (Roll={math.degrees(roll):.1f}°, Pitch={math.degrees(pitch):.1f}°)")
                self.trigger_crash()
                
            if not self.has_reached_safe_altitude:
                if enu_z > 1.0:
                    self.has_reached_safe_altitude = True
            else:
                if enu_z < self.crash_z_threshold:
                    self.get_logger().error(f"💥 Phát hiện rớt: Độ cao quá thấp ({enu_z:.2f}m < {self.crash_z_threshold}m)")
                    self.trigger_crash()
                
    def trigger_crash(self):
        # Đóng tiến trình arm (nếu có) và thoát với code 1
        if self.arm_process:
            self.arm_process.terminate()
        if self.logger_process:
            self.logger_process.terminate()
        sys.exit(1)
        
    def send_takeoff_cmd(self):
        while not self.cli_arm_takeoff.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Đợi service /uam/arm_takeoff ...')
        
        req = Trigger.Request()
        future = self.cli_arm_takeoff.call_async(req)
        # Không cần block chờ future ở đây, bridge node tự xử lý
        return future

    def main_loop(self):
        current_time = time.time()
        
        # 0. Chờ khởi tạo hệ thống
        if current_time - self.start_time < self.init_wait_duration:
            self.get_logger().info(f"⏳ Khởi động: Đang chờ hệ thống thiết lập ({(current_time - self.start_time):.1f}/{self.init_wait_duration}s)...")
            return
            
        # 1. Nếu chưa cất cánh
        if not self.has_taken_off:
            self.get_logger().info("🛫 Phase 1: Gửi lệnh ARM & TAKEOFF...")
            self.send_takeoff_cmd()
            self.has_taken_off = True
            self.takeoff_time = current_time
            return
            
        # 2. Nếu đang Phase 1 (đợi Hover ổn định 15s)
        elapsed_since_takeoff = current_time - self.takeoff_time
        if not self.phase2_started:
            if elapsed_since_takeoff < self.phase1_duration:
                self.get_logger().info(f"⏳ Phase 1: Đợi Hover ổn định ({elapsed_since_takeoff:.1f}/{self.phase1_duration}s)...")
            else:
                # Chuyển sang Phase 2
                self.get_logger().info("🚀 Phase 2: Bắt đầu chạy trajectory cánh tay để train...")
                self.phase2_started = True
                self.phase2_start_time = current_time
                
                # Gọi arm_trajectory_generator.py (Pattern: combined)
                cmd = [
                    "ros2", "run", "uam_controller", "arm_trajectory_generator.py",
                    "--pattern", "combined",
                    "--duration", str(int(self.phase2_duration))
                ]
                # Lưu ý: cần thiết lập biến môi trường ROS nếu chạy qua subprocess
                env = os.environ.copy()
                self.arm_process = subprocess.Popen(cmd, env=env)
            return

        # 3. Nếu đang ở Phase 2
        elapsed_phase2 = current_time - self.phase2_start_time
        if elapsed_phase2 < self.phase2_duration:
            self.get_logger().info(f"🦾 Phase 2: Đang chạy pattern ({elapsed_phase2:.1f}/{self.phase2_duration}s)...")
            # Kiểm tra xem arm_process có bị lỗi không
            if self.arm_process.poll() is not None and self.arm_process.returncode != 0:
                self.get_logger().error("💥 arm_trajectory_generator dừng đột ngột!")
                self.trigger_crash()
        else:
            # Thành công!
            self.get_logger().info("✅ Hoàn thành quy trình huấn luyện an toàn. Kết thúc bằng 0.")
            if self.arm_process:
                self.arm_process.terminate()
            if self.logger_process:
                self.logger_process.terminate()
            sys.exit(0)


def main(args=None):
    rclpy.init(args=args)
    node = RBFNNTrainingSupervisor()
    
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"Ngoại lệ: {e}")
        node.trigger_crash()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
