#!/usr/bin/env python3
"""
uam_mission_bridge.py
---------------------
Quản lý trạng thái nhiệm vụ cho UAM.
Trong kiến trúc RBFNN Rate Controller (Hướng 2), node này chịu trách nhiệm 
cấp phát điểm đến Vị trí dưới dạng TrajectorySetpoint cho Firmware PX4.
Px4 sẽ chạy Position & Attitude Controller và sinh ra VehicleRatesSetpoint.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    VehicleStatus, 
    VehicleOdometry, 
    VehicleCommand,
    OffboardControlMode,
    TrajectorySetpoint
)
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped
import time

import numpy as np
import math
import json


class UAMMissionBridge(Node):
    STATE_IDLE      = "IDLE"
    STATE_ARMED     = "ARMED"
    STATE_TAKEOFF   = "TAKEOFF"
    STATE_HOLD      = "HOLD"
    STATE_GOTO      = "GOTO"
    STATE_LAND      = "LAND"
    STATE_EMERGENCY = "EMERGENCY"

    def __init__(self):
        super().__init__("uam_mission_bridge")

        self.declare_parameter("takeoff_height", -2.0)       # NED âm là bay lên
        self.declare_parameter("cruise_speed", 2.0)
        self.declare_parameter("position_threshold", 0.25)
        self.declare_parameter("loop_rate_hz", 20.0)

        self.takeoff_height = self.get_parameter("takeoff_height").value
        self.cruise_speed   = self.get_parameter("cruise_speed").value
        self.pos_threshold  = self.get_parameter("position_threshold").value
        self.rate_hz        = self.get_parameter("loop_rate_hz").value

        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Publishers ──
        self.pub_enable        = self.create_publisher(Bool, "/uam/controller_enable", qos_reliable)
        self.pub_offboard_mode = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos_sensor)
        self.pub_trajectory    = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_sensor)
        self.pub_vehicle_cmd   = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos_sensor)
        self.pub_state_str     = self.create_publisher(String, "/uam/state", qos_reliable)

        # ── Subscribers ──
        self.sub_status = self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v1", self._cb_status, qos_sensor)
        self.sub_odom   = self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self._cb_odom, qos_sensor)
        self.sub_goto   = self.create_subscription(PoseStamped, "/uam/cmd/goto_pose", self._cb_goto_cmd, qos_reliable)

        # ── Services ──
        self.srv_arm   = self.create_service(Trigger, "/uam/arm_takeoff", self._srv_arm_takeoff)
        self.srv_land  = self.create_service(Trigger, "/uam/land",        self._srv_land)
        self.srv_hold  = self.create_service(Trigger, "/uam/hold",        self._srv_hold)
        self.srv_rtl   = self.create_service(Trigger, "/uam/rtl",         self._srv_rtl)
        self.srv_emg   = self.create_service(Trigger, "/uam/emergency",   self._srv_emergency)

        self.vehicle_status = VehicleStatus()
        self.current_pos    = np.zeros(3)  # NED
        self.current_yaw    = 0.0
        
        self.setpoint       = np.zeros(3)
        self.setpoint_yaw   = 0.0
        
        self.state          = self.STATE_IDLE
        self.initialized    = False
        self.px4_timestamp  = 0

        self.timer = self.create_timer(1.0 / self.rate_hz, self._control_loop)
        self.get_logger().info("✅ UAM Mission Bridge (PX4 Position Commander) sẵn sàng!")

    def _cb_status(self, msg: VehicleStatus):
        if self.vehicle_status.nav_state != msg.nav_state or self.vehicle_status.arming_state != msg.arming_state:
            self.get_logger().info(f"🔔 PX4 Status Change: NavState={msg.nav_state}, ArmingState={msg.arming_state}")
        self.vehicle_status = msg
        self.px4_timestamp = msg.timestamp

    def _cb_odom(self, msg: VehicleOdometry):
        # Trực tiếp dùng NED từ Odometry để điều khiển hệ PX4
        self.current_pos = np.array([msg.position[0], msg.position[1], msg.position[2]])
        q = msg.q
        siny = 2.0 * (q[0] * q[3] + q[1] * q[2])
        cosy = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
        self.current_yaw = math.atan2(siny, cosy)
        self.initialized = True

    def _cb_goto_cmd(self, msg: PoseStamped):
        # Đổi toạ độ ROS ENU sang NED
        x = msg.pose.position.x
        y = -msg.pose.position.y
        z = -msg.pose.position.z 
        self.get_logger().info(f"🗺️  Nhận lệnh goto: NED({x:.2f}, {y:.2f}, {z:.2f})")
        if self.state in [self.STATE_HOLD, self.STATE_GOTO]:
            self.setpoint = np.array([x, y, z])
            self.state = self.STATE_GOTO

    def _srv_arm_takeoff(self, request, response):
        if not self.initialized:
            response.success = False
            response.message = "Chưa có odometry để cất cánh!"
            return response
        if self.state != self.STATE_IDLE:
            response.success = False
            response.message = f"Đang ở trạng thái {self.state}"
            return response
            
        if getattr(self, "offboard_counter", 0) < 10:
            response.success = False
            response.message = "Chưa đủ 10 nhịp gửi tín hiệu Offboard, thử lại sau 1 giây!"
            return response

        self.get_logger().info(f"🚀 Lệnh cất cánh nhận được. Đang chuẩn bị (NED Z={self.takeoff_height}m)...")
        self._publish_enable(True)
        
        # Lưu toạ độ hiện tại làm điểm bắt đầu
        self.setpoint     = np.array([self.current_pos[0], self.current_pos[1], self.current_pos[2]])
        self.setpoint_yaw = self.current_yaw
        
        # Chuyển trạng thái sang TAKEOFF ngay để bắt đầu luồng lệnh
        self.state = self.STATE_TAKEOFF
        self._takeoff_step = 0
        self._takeoff_timer = 0
            
        response.success = True
        response.message = "Đã nhận lệnh cất cánh, đang thực hiện trình tự Arm + Offboard..."
        return response

    def _srv_land(self, request, response):
        if self.state in [self.STATE_IDLE, self.STATE_EMERGENCY]:
            response.success = False
            return response
        self.get_logger().info("🛬 Bắt đầu hạ cánh")
        self.state = self.STATE_LAND
        response.success = True
        return response

    def _srv_hold(self, request, response):
        if self.state in [self.STATE_IDLE, self.STATE_EMERGENCY]:
            response.success = False
            return response
        self.setpoint = self.current_pos.copy()
        self.state = self.STATE_HOLD
        response.success = True
        return response

    def _srv_rtl(self, request, response):
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
        self.state = self.STATE_EMERGENCY
        self._publish_enable(False)
        response.success = True
        return response

    def _srv_emergency(self, request, response):
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
        self.state = self.STATE_EMERGENCY
        self._publish_enable(False)
        response.success = True
        return response

    def _control_loop(self):
        self._publish_state_str()
        if not self.initialized:
            return

        # Luôn bắn tín hiệu OffboardControlMode lớn hơn 2Hz để nuôi PX4 (yêu cầu bắt buộc)
        self._publish_offboard_mode()

        if self.state == self.STATE_IDLE:
            if not hasattr(self, "offboard_counter"):
                self.offboard_counter = 0
            self.offboard_counter += 1
            # Bắn toạ độ giữ nguyên vị trí đất để PX4 làm quen
            self._publish_setpoint(self.current_pos, self.current_yaw)
            self._publish_enable(False)
            
        elif self.state == self.STATE_TAKEOFF:
            if not hasattr(self, '_takeoff_step'): self._takeoff_step = 0
            if not hasattr(self, '_retry_timer'): self._retry_timer = 0
            
            # Step 0: Request OFFBOARD Mode (NavState 14)
            if self._takeoff_step == 0:
                if self.vehicle_status.nav_state != 14:
                    self._retry_timer += 1
                    if self._retry_timer % 20 == 1: # Once per second
                        self.get_logger().info("⏳ 1. Đang yêu cầu OFFBOARD (nav_state=14)...")
                        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                else:
                    self.get_logger().info("✅ Đã vào OFFBOARD mode.")
                    self._takeoff_step = 1
                    self._retry_timer = 0
                self._publish_setpoint(self.setpoint, self.setpoint_yaw)
                return

            # Step 1: Request ARM (ArmingState 2)
            if self._takeoff_step == 1:
                if self.vehicle_status.arming_state != 2: # ARMED=2
                    self._retry_timer += 1
                    if self._retry_timer % 20 == 1:
                        self.get_logger().info("⏳ 2. Đang yêu cầu ARM (arming_state=2)...")
                        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                else:
                    self.get_logger().info("✅ Đã ARM động cơ.")
                    self._takeoff_step = 2
                    self._retry_timer = 0
                    self._takeoff_timer = 0
                self._publish_setpoint(self.setpoint, self.setpoint_yaw)
                return

            if self._takeoff_step == 2:
                # Chờ 2s cho động cơ khởi động mượt và áp lực đẩy
                self._takeoff_timer += 1
                if self._takeoff_timer > 40: 
                    self.get_logger().info("🚀 3. Bắt đầu cất cánh (NED Z ramping)...")
                    self._takeoff_step = 3
                self._publish_setpoint(self.setpoint, self.setpoint_yaw)
                return

            # Step 3: Z-Ramping
            if self.setpoint[2] > self.takeoff_height: 
                self.setpoint[2] -= 0.2 / self.rate_hz
                if self.setpoint[2] < self.takeoff_height:
                    self.setpoint[2] = self.takeoff_height
            
            self._publish_setpoint(self.setpoint, self.setpoint_yaw)
            
            # Log chi tiết mỗi giây
            if self._takeoff_timer % int(self.rate_hz) == 0:
                self.get_logger().info(f"📏 ALT: setpoint_z={self.setpoint[2]:.2f} | actual_z={self.current_pos[2]:.2f} | target={self.takeoff_height:.2f}")
            self._takeoff_timer += 1
            
            if abs(self.current_pos[2] - self.takeoff_height) < self.pos_threshold:
                self.get_logger().info("✈️  Đạt độ cao mục tiêu. Chuyển sang HOLD.")
                self.state = self.STATE_HOLD
                self._takeoff_step = 0
                self._retry_timer = 0

        elif self.state == self.STATE_HOLD:
            self._publish_setpoint(self.setpoint, self.setpoint_yaw)

        elif self.state == self.STATE_GOTO:
            self._publish_setpoint(self.setpoint, self.setpoint_yaw)
            dist = np.linalg.norm(self.current_pos - self.setpoint)
            if dist < self.pos_threshold:
                self.state = self.STATE_HOLD

        elif self.state == self.STATE_LAND:
            # Hạ cánh ở tốc độ 0.2m/s
            self.setpoint[2] += 0.2 / self.rate_hz
            self._publish_setpoint(self.setpoint, self.setpoint_yaw)
            # Chạm đất (ví dụ Z ~ 0) tuỳ thuộc map của bạn
            if self.current_pos[2] > -0.15:
                self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
                self._publish_enable(False)
                self.state = self.STATE_IDLE

    def _arm_and_offboard(self):
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

    def _publish_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0):
        msg = VehicleCommand()
        msg.timestamp          = self.get_clock().now().nanoseconds // 1000
        msg.command            = command
        msg.param1             = float(param1)
        msg.param2             = float(param2)
        msg.target_system      = 1
        msg.target_component   = 1
        msg.source_system      = 1
        msg.source_component   = 1
        msg.from_external      = True
        self.pub_vehicle_cmd.publish(msg)

    def _publish_offboard_mode(self):
        # ★ FIX #2 (Safety Doc): Trong Approach B, PHẢI giữ position=True.
        # PX4 cần chạy Position → Attitude Controller để sinh VehicleRatesSetpoint.
        # Node C++ (RBFNN) sẽ lấy RatesSetpoint đó và tính Torque trực tiếp.
        # PX4 Rate Controller gains = 0 nên không xung đột.
        # ⚠️ KHÔNG ĐƯỢC dùng direct_actuator=True, nó sẽ tắt pipeline rate setpoint!
        msg = OffboardControlMode()
        msg.timestamp    = self.get_clock().now().nanoseconds // 1000
        msg.position     = True
        msg.velocity     = False
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False
        self.pub_offboard_mode.publish(msg)

    def _publish_setpoint(self, pos: np.ndarray, yaw: float):
        msg = TrajectorySetpoint()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.position  = pos.astype(float).tolist()
        msg.yaw       = float(yaw)
        msg.velocity  = [float('nan')] * 3
        msg.acceleration = [float('nan')] * 3
        self.pub_trajectory.publish(msg)

    def _publish_enable(self, enable: bool):
        msg = Bool()
        msg.data = enable
        self.pub_enable.publish(msg)

    def _publish_state_str(self):
        state_info = {
            "mission_state": self.state,
            "target": {
                "x": round(float(self.setpoint[0]), 2), "y": round(float(self.setpoint[1]), 2),
                "z": round(float(self.setpoint[2]), 2), "yaw": round(float(self.setpoint_yaw), 2)
            }
        }
        msg = String()
        msg.data = json.dumps(state_info)
        self.pub_state_str.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = UAMMissionBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
