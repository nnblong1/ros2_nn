#!/usr/bin/env python3
"""
UAV Offboard Control Node - PX4 + ROS2 + Raspberry Pi 4
Giao tiếp với PX4 qua micro-XRCE-DDS (uXRCE-DDS)
Hỗ trợ: Arm/Disarm, Takeoff, Goto, Land, Position Hold
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    VehicleOdometry,
    VehicleLocalPosition,
)

import numpy as np
import math


class OffboardControl(Node):
    """Node điều khiển Offboard cho UAV Quadrotor."""

    # Trạng thái máy bay
    STATE_IDLE      = "IDLE"
    STATE_ARMED     = "ARMED"
    STATE_TAKEOFF   = "TAKEOFF"
    STATE_HOLD      = "HOLD"
    STATE_GOTO      = "GOTO"
    STATE_LAND      = "LAND"

    def __init__(self):
        super().__init__("offboard_control")

        # ── Khai báo tham số ──────────────────────────────────────────────
        self.declare_parameter("takeoff_height",    -2.0)   # NED: âm = lên
        self.declare_parameter("cruise_speed",       2.0)   # m/s
        self.declare_parameter("position_threshold", 0.25)  # m
        self.declare_parameter("loop_rate_hz",      20.0)

        self.takeoff_height    = self.get_parameter("takeoff_height").value
        self.cruise_speed      = self.get_parameter("cruise_speed").value
        self.pos_threshold     = self.get_parameter("position_threshold").value
        rate_hz                = self.get_parameter("loop_rate_hz").value

        # ── QoS profile chuẩn PX4 ────────────────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_offboard_mode = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos)
        self.pub_trajectory    = self.create_publisher(
            TrajectorySetpoint,  "/fmu/in/trajectory_setpoint",   qos)
        self.pub_vehicle_cmd   = self.create_publisher(
            VehicleCommand,      "/fmu/in/vehicle_command",        qos)

        # ── Subscribers ───────────────────────────────────────────────────
        self.sub_status   = self.create_subscription(
            VehicleStatus,        "/fmu/out/vehicle_status",
            self._cb_status, qos)
        self.sub_odom     = self.create_subscription(
            VehicleOdometry,      "/fmu/out/vehicle_odometry",
            self._cb_odometry, qos)
        self.sub_local_pos = self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position",
            self._cb_local_pos, qos)

        # ── Biến nội bộ ───────────────────────────────────────────────────
        self.vehicle_status   = VehicleStatus()
        self.local_pos        = VehicleLocalPosition()
        self.current_pos      = np.zeros(3)   # [x, y, z] NED
        self.current_yaw      = 0.0
        self.setpoint         = np.zeros(3)
        self.setpoint_yaw     = 0.0
        self.state            = self.STATE_IDLE
        self.offboard_counter = 0             # đếm để kích hoạt offboard

        # ── Timer điều khiển chính ────────────────────────────────────────
        period = 1.0 / rate_hz
        self.timer = self.create_timer(period, self._control_loop)

        self.get_logger().info("✅ OffboardControl node khởi động thành công")
        self.get_logger().info(f"   Takeoff height : {self.takeoff_height} m (NED)")
        self.get_logger().info(f"   Cruise speed   : {self.cruise_speed} m/s")

    # ═══════════════════════════════════════════════════════════════════════
    # Callbacks
    # ═══════════════════════════════════════════════════════════════════════

    def _cb_status(self, msg: VehicleStatus):
        self.vehicle_status = msg

    def _cb_odometry(self, msg: VehicleOdometry):
        self.current_pos = np.array([msg.position[0], msg.position[1], msg.position[2]])

    def _cb_local_pos(self, msg: VehicleLocalPosition):
        self.local_pos    = msg
        self.current_pos  = np.array([msg.x, msg.y, msg.z])
        self.current_yaw  = msg.heading

    # ═══════════════════════════════════════════════════════════════════════
    # Control Loop (chạy ở tần số loop_rate_hz)
    # ═══════════════════════════════════════════════════════════════════════

    def _control_loop(self):
        # Luôn publish offboard mode để giữ kết nối
        self._publish_offboard_mode()

        if self.state == self.STATE_IDLE:
            # Cần publish ít nhất 10 lần trước khi chuyển sang offboard
            self.offboard_counter += 1
            self._publish_setpoint(self.current_pos, self.current_yaw)

        elif self.state == self.STATE_ARMED:
            self._publish_setpoint(self.current_pos, self.current_yaw)
            # Sẵn sàng để nhận lệnh tiếp theo

        elif self.state == self.STATE_TAKEOFF:
            target = np.array([self.current_pos[0],
                                self.current_pos[1],
                                self.takeoff_height])
            self._publish_setpoint(target, self.setpoint_yaw)
            if abs(self.current_pos[2] - self.takeoff_height) < self.pos_threshold:
                self.get_logger().info("✈️  Đã đạt độ cao takeoff → HOLD")
                self.state = self.STATE_HOLD

        elif self.state == self.STATE_HOLD:
            self._publish_setpoint(self.setpoint, self.setpoint_yaw)

        elif self.state == self.STATE_GOTO:
            self._publish_setpoint(self.setpoint, self.setpoint_yaw)
            dist = np.linalg.norm(self.current_pos[:2] - self.setpoint[:2])
            if dist < self.pos_threshold:
                self.get_logger().info(
                    f"📍 Đã đến điểm đích ({self.setpoint[0]:.1f}, "
                    f"{self.setpoint[1]:.1f}, {self.setpoint[2]:.1f}) → HOLD")
                self.state = self.STATE_HOLD

        elif self.state == self.STATE_LAND:
            land_target = np.array([self.current_pos[0],
                                    self.current_pos[1],
                                    0.0])
            self._publish_setpoint(land_target, self.setpoint_yaw)
            if abs(self.current_pos[2]) < 0.1:
                self.get_logger().info("🛬 Đã hạ cánh → IDLE")
                self._disarm()
                self.state = self.STATE_IDLE

    # ═══════════════════════════════════════════════════════════════════════
    # Public API (gọi từ bên ngoài hoặc từ mission node)
    # ═══════════════════════════════════════════════════════════════════════

    def arm_and_takeoff(self):
        """Arm và cất cánh lên độ cao mặc định."""
        if self.offboard_counter < 10:
            self.get_logger().warn("⚠️  Chưa đủ heartbeat để vào Offboard mode")
            return
        self._engage_offboard_mode()
        self._arm()
        self.setpoint     = np.array([self.current_pos[0],
                                       self.current_pos[1],
                                       self.takeoff_height])
        self.setpoint_yaw = self.current_yaw
        self.state        = self.STATE_TAKEOFF
        self.get_logger().info(f"🚀 Takeoff → z={self.takeoff_height} m")

    def goto_position(self, x: float, y: float, z: float,
                      yaw: float = None):
        """Bay đến tọa độ NED (x, y, z) mét."""
        if self.state not in [self.STATE_HOLD, self.STATE_GOTO]:
            self.get_logger().warn(f"⚠️  Không thể goto từ trạng thái {self.state}")
            return
        self.setpoint     = np.array([x, y, z])
        self.setpoint_yaw = yaw if yaw is not None else self.current_yaw
        self.state        = self.STATE_GOTO
        self.get_logger().info(f"➡️  Goto ({x:.1f}, {y:.1f}, {z:.1f})")

    def hold_position(self):
        """Dừng và giữ vị trí hiện tại."""
        self.setpoint     = self.current_pos.copy()
        self.setpoint_yaw = self.current_yaw
        self.state        = self.STATE_HOLD
        self.get_logger().info("🔒 Giữ vị trí")

    def land(self):
        """Hạ cánh tại vị trí hiện tại."""
        if self.state == self.STATE_IDLE:
            self.get_logger().warn("⚠️  UAV chưa được cất cánh")
            return
        self.state = self.STATE_LAND
        self.get_logger().info("🛬 Bắt đầu hạ cánh")

    def return_to_home(self):
        """Kích hoạt Return-to-Launch qua VehicleCommand."""
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
        self.get_logger().info("🏠 RTL kích hoạt")

    def emergency_stop(self):
        """Kill motors khẩn cấp."""
        self._disarm()
        self.state = self.STATE_IDLE
        self.get_logger().warn("🛑 EMERGENCY STOP!")

    # ═══════════════════════════════════════════════════════════════════════
    # Publish helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp  = self._ts()
        msg.position   = True
        msg.velocity   = False
        msg.acceleration = False
        msg.attitude   = False
        msg.body_rate  = False
        self.pub_offboard_mode.publish(msg)

    def _publish_setpoint(self, pos: np.ndarray, yaw: float):
        msg = TrajectorySetpoint()
        msg.timestamp  = self._ts()
        msg.position   = pos.astype(float).tolist()
        msg.yaw        = float(yaw)
        msg.velocity   = [float("nan")] * 3
        msg.acceleration = [float("nan")] * 3
        self.pub_trajectory.publish(msg)

    def _publish_vehicle_command(self, command: int,
                                  param1: float = 0.0,
                                  param2: float = 0.0):
        msg = VehicleCommand()
        msg.timestamp          = self._ts()
        msg.command            = command
        msg.param1             = param1
        msg.param2             = param2
        msg.target_system      = 1
        msg.target_component   = 1
        msg.source_system      = 1
        msg.source_component   = 1
        msg.from_external      = True
        self.pub_vehicle_cmd.publish(msg)

    def _engage_offboard_mode(self):
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.get_logger().info("📡 Chuyển sang Offboard mode")

    def _arm(self):
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.state = self.STATE_ARMED
        self.get_logger().info("🔓 ARM")

    def _disarm(self):
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
        self.get_logger().info("🔐 DISARM")

    def _ts(self) -> int:
        """Trả về timestamp microseconds."""
        return self.get_clock().now().nanoseconds // 1000


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = OffboardControl()

    # Demo mission đơn giản: Arm → Takeoff → Hold
    # Trong thực tế, gọi qua ROS2 Services hoặc từ mission_manager.py
    import threading, time

    def run_demo():
        time.sleep(3.0)                   # Chờ node ổn định
        node.arm_and_takeoff()
        time.sleep(8.0)                   # Bay lên và chờ
        node.goto_position(5.0, 0.0, -2.5)
        time.sleep(6.0)
        node.goto_position(5.0, 5.0, -2.5)
        time.sleep(6.0)
        node.goto_position(0.0, 0.0, -2.5)
        time.sleep(5.0)
        node.land()

    t = threading.Thread(target=run_demo, daemon=True)
    t.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("⛔ Dừng node...")
        node.emergency_stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
