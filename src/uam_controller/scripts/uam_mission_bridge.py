#!/usr/bin/env python3
"""
uam_mission_bridge.py
---------------------
Quản lý trạng thái nhiệm vụ cho UAM.
Kết hợp state machine và ROS2 services, ra lệnh bật/tắt bộ điều khiển C++ (uam_adaptive_controller)
đồng thời điều khiển vòng ngoài (position setpoint / yaw setpoint).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import VehicleStatus, VehicleOdometry, VehicleCommand
from std_msgs.msg import String, Bool, Float64
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, Vector3

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

        # Khai báo tham số
        self.declare_parameter("takeoff_height", 2.0)       # Mặc định cất cánh cao 2.0m (ENU)
        self.declare_parameter("cruise_speed", 2.0)         # m/s
        self.declare_parameter("position_threshold", 0.25)  # m (ngưỡng xác định đến nơi)
        self.declare_parameter("loop_rate_hz", 20.0)

        self.takeoff_height = self.get_parameter("takeoff_height").value
        self.cruise_speed   = self.get_parameter("cruise_speed").value
        self.pos_threshold  = self.get_parameter("position_threshold").value
        self.rate_hz        = self.get_parameter("loop_rate_hz").value

        # QoS profiles
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
        self.pub_enable      = self.create_publisher(Bool,           "/uam/controller_enable", qos_reliable)
        self.pub_setpoint    = self.create_publisher(Vector3,        "/uam/position_setpoint", qos_reliable)
        self.pub_yaw         = self.create_publisher(Float64,        "/uam/yaw_setpoint",      qos_reliable)
        self.pub_vehicle_cmd = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos_sensor)
        self.pub_state_str   = self.create_publisher(String,         "/uam/state",             qos_reliable)

        # ── Subscribers ──
        self.sub_status = self.create_subscription(VehicleStatus,   "/fmu/out/vehicle_status",   self._cb_status, qos_sensor)
        self.sub_odom   = self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self._cb_odom,   qos_sensor)
        self.sub_goto   = self.create_subscription(PoseStamped,     "/uam/cmd/goto_pose",        self._cb_goto_cmd, qos_reliable)

        # ── Services ──
        self.srv_arm   = self.create_service(Trigger, "/uam/arm_takeoff", self._srv_arm_takeoff)
        self.srv_land  = self.create_service(Trigger, "/uam/land",        self._srv_land)
        self.srv_hold  = self.create_service(Trigger, "/uam/hold",        self._srv_hold)
        self.srv_rtl   = self.create_service(Trigger, "/uam/rtl",         self._srv_rtl)
        self.srv_emg   = self.create_service(Trigger, "/uam/emergency",   self._srv_emergency)

        # ── Trạng thái nội bộ ──
        self.vehicle_status = VehicleStatus()
        self.current_pos    = np.zeros(3)  # [x, y, z] ENU (Z dương là lên)
        self.current_yaw    = 0.0
        
        self.setpoint       = np.zeros(3)
        self.setpoint_yaw   = 0.0
        
        self.state          = self.STATE_IDLE
        self.initialized    = False
        
        self.waypoints      = []
        self.wp_index       = 0
        self.mission_active = False

        # ── Timer chính ──
        self.timer = self.create_timer(1.0 / self.rate_hz, self._control_loop)
        
        self.get_logger().info("✅ UAM Mission Bridge node khởi động")

    # ═══════════════════════════════════════════════════════════════════════
    # Callbacks
    # ═══════════════════════════════════════════════════════════════════════

    def _cb_status(self, msg: VehicleStatus):
        self.vehicle_status = msg

    def _cb_odom(self, msg: VehicleOdometry):
        # PX4 Odom: NED. Convert sang ENU
        self.current_pos = np.array([msg.position[0], -msg.position[1], -msg.position[2]])
        
        # Lấy yaw từ quaternion
        q = msg.q
        siny = 2.0 * (q[0] * q[3] + q[1] * q[2])
        cosy = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
        self.current_yaw = math.atan2(siny, cosy)
        
        self.initialized = True

    def _cb_goto_cmd(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        self.get_logger().info(f"🗺️  Nhận lệnh goto qua topic: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        if self.state in [self.STATE_HOLD, self.STATE_GOTO]:
            self.setpoint = np.array([x, y, z])
            self.state = self.STATE_GOTO

    # ═══════════════════════════════════════════════════════════════════════
    # Service Handlers
    # ═══════════════════════════════════════════════════════════════════════

    def _srv_arm_takeoff(self, request, response):
        if not self.initialized:
            response.success = False
            response.message = "Chưa có odometry để cất cánh!"
            return response
            
        if self.state != self.STATE_IDLE:
            response.success = False
            response.message = f"Đang ở trạng thái {self.state}, không thể takeoff"
            return response

        self.get_logger().info(f"🚀 Bắt đầu trình tự Takeoff -> {self.takeoff_height}m")
        
        # 1. Enable Adaptive Controller C++
        self._publish_enable(True)
        
        # 2. Set điểm đến là vị trí hiện tại (bắt đầu từ pz thực tế để ramp lên)
        self.setpoint     = np.array([self.current_pos[0], self.current_pos[1], self.current_pos[2]])
        self.setpoint_yaw = self.current_yaw
        
        # Mở khoá và chuyển chế độ
        self._arm_and_offboard()
        
        self.state = self.STATE_TAKEOFF
        # Reset takeoff timer cho warm-up delay
        if hasattr(self, '_takeoff_start_time'):
            del self._takeoff_start_time
        response.success = True
        response.message = "Takeoff bắt đầu"
        return response

    def _srv_land(self, request, response):
        if self.state in [self.STATE_IDLE, self.STATE_EMERGENCY]:
            response.success = False
            response.message = "Đã tiếp đất hoặc emergency"
            return response
            
        self.get_logger().info("🛬 Bắt đầu hạ cánh")
        self.state = self.STATE_LAND
        response.success = True
        response.message = "Hạ cánh"
        return response

    def _srv_hold(self, request, response):
        if self.state in [self.STATE_IDLE, self.STATE_LAND, self.STATE_EMERGENCY]:
            response.success = False
            response.message = "Không thể hold từ trạng thái này"
            return response
            
        self.get_logger().info("🔒 Giữ vị trí hiện tại")
        self.setpoint = self.current_pos.copy()
        self.state = self.STATE_HOLD
        response.success = True
        response.message = "Giữ vị trí"
        return response

    def _srv_rtl(self, request, response):
        self.get_logger().info("🏠 Kích hoạt Return-to-Launch qua PX4")
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
        self.state = self.STATE_EMERGENCY
        self._publish_enable(False)
        response.success = True
        response.message = "RTL kích hoạt"
        return response

    def _srv_emergency(self, request, response):
        self.get_logger().warn("🛑 DỪNG KHẨN CẤP")
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
        self.state = self.STATE_EMERGENCY
        self._publish_enable(False)
        response.success = True
        response.message = "Disarm khẩn cấp"
        return response

    # ═══════════════════════════════════════════════════════════════════════
    # Control Loop chính
    # ═══════════════════════════════════════════════════════════════════════

    def _control_loop(self):
        # Publish trạng thái hiện tại (string/JSON)
        self._publish_state_str()
        
        if not self.initialized:
            return

        if self.state == self.STATE_IDLE:
            # Tắt controller
            self._publish_enable(False)
            
        elif self.state == self.STATE_TAKEOFF:
            # Initial warm-up: chờ motor spin-up + controller settle trước khi ramp
            if not hasattr(self, '_takeoff_start_time'):
                self._takeoff_start_time = self.get_clock().now().nanoseconds / 1e9

            elapsed = self.get_clock().now().nanoseconds / 1e9 - self._takeoff_start_time

            # Chờ 4s warm-up (motor spin-up + controller settle)
            if elapsed < 4.0:
                self._publish_setpoint(self.setpoint, self.setpoint_yaw)
                return

            # Ramping altitude setpoint (0.3 m/s — mềm hơn để tránh thrust spike)
            if self.setpoint[2] < self.takeoff_height:
                self.setpoint[2] += 0.3 / self.rate_hz
                if self.setpoint[2] > self.takeoff_height:
                    self.setpoint[2] = self.takeoff_height
            
            self._publish_setpoint(self.setpoint, self.setpoint_yaw)
            
            # Kiểm tra độ cao (ENU Z) đạt ngưỡng thực tế
            if abs(self.current_pos[2] - self.takeoff_height) < self.pos_threshold:
                self.get_logger().info("✈️  Đã đạt độ cao takeoff → HOLD")
                self.state = self.STATE_HOLD

        elif self.state == self.STATE_HOLD:
            self._publish_setpoint(self.setpoint, self.setpoint_yaw)

        elif self.state == self.STATE_GOTO:
            self._publish_setpoint(self.setpoint, self.setpoint_yaw)
            
            dist = np.linalg.norm(self.current_pos - self.setpoint)
            if dist < self.pos_threshold:
                self.get_logger().info(
                    f"📍 Đã đến đích ({self.setpoint[0]:.1f}, "
                    f"{self.setpoint[1]:.1f}, {self.setpoint[2]:.1f}) → HOLD")
                self.state = self.STATE_HOLD

        elif self.state == self.STATE_LAND:
            # Đặt z setpoint xuống dần (hoặc âm để PX4 chạm đất tự tắt)
            # Vì Adaptive Controller cố gắng bám setpoint, ta đặt setpoint Z = 0.0
            land_target = np.array([self.current_pos[0], self.current_pos[1], 0.0])
            self._publish_setpoint(land_target, self.setpoint_yaw)
            
            # Nếu chạm đất (Z < 0.15m trong hệ ENU hoặc NED tuỳ)
            if self.current_pos[2] < 0.15:
                self.get_logger().info("🛬 Đã hạ cánh → DISARM")
                self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
                self._publish_enable(False)
                self.state = self.STATE_IDLE

        elif self.state == self.STATE_EMERGENCY:
            self._publish_enable(False)

    # ═══════════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _arm_and_offboard(self):
        # Arm
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("🔓 Cấp lệnh ARM")
        
        # Chuyển sang Offboard Mode (mode 6)
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.get_logger().info("📡 Cấp lệnh OFFBOARD MODE")

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

    def _publish_enable(self, enable: bool):
        msg = Bool()
        msg.data = enable
        self.pub_enable.publish(msg)
        
    def _publish_setpoint(self, pos: np.ndarray, yaw: float):
        msg_pos = Vector3()
        msg_pos.x = float(pos[0])
        msg_pos.y = float(pos[1])
        msg_pos.z = float(pos[2])
        self.pub_setpoint.publish(msg_pos)
        
        msg_yaw = Float64()
        msg_yaw.data = float(yaw)
        self.pub_yaw.publish(msg_yaw)

    def _publish_state_str(self):
        state_info = {
            "mission_state": self.state,
            "target": {
                "x": round(float(self.setpoint[0]), 2),
                "y": round(float(self.setpoint[1]), 2),
                "z": round(float(self.setpoint[2]), 2),
                "yaw": round(float(self.setpoint_yaw), 2)
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
