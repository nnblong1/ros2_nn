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
    VehicleLocalPosition,
    VehicleCommand,
    VehicleCommandAck,
    OffboardControlMode,
    TimesyncStatus,
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
    STATE_PRIME_OFFBOARD = "PRIME_OFFBOARD"
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
        self.declare_parameter("offboard_prime_duration_s", 3.0)
        self.declare_parameter("require_local_position_ready", True)
        self.declare_parameter("require_heading_good_for_control", False)
        self.declare_parameter("auto_enable_external_controller", False)
        self.declare_parameter("allow_external_torque_handoff", True)
        self.declare_parameter("external_handoff_stable_time_s", 5.0)
        self.declare_parameter("external_handoff_max_position_error_m", 0.30)
        self.declare_parameter("external_handoff_max_horizontal_speed_ms", 0.20)
        self.declare_parameter("external_handoff_max_vertical_speed_ms", 0.12)

        self.takeoff_height = self.get_parameter("takeoff_height").value
        self.cruise_speed   = self.get_parameter("cruise_speed").value
        self.pos_threshold  = self.get_parameter("position_threshold").value
        self.rate_hz        = self.get_parameter("loop_rate_hz").value
        self.offboard_prime_duration_s = self.get_parameter("offboard_prime_duration_s").value
        self.offboard_prime_ticks = max(1, int(math.ceil(self.offboard_prime_duration_s * self.rate_hz)))
        self.require_local_position_ready = bool(self.get_parameter("require_local_position_ready").value)
        self.require_heading_good_for_control = bool(
            self.get_parameter("require_heading_good_for_control").value
        )
        self.auto_enable_external_controller = bool(self.get_parameter("auto_enable_external_controller").value)
        self.allow_external_torque_handoff = bool(
            self.get_parameter("allow_external_torque_handoff").value
        )
        self.external_handoff_stable_time_s = float(
            self.get_parameter("external_handoff_stable_time_s").value
        )
        self.external_handoff_max_position_error_m = float(
            self.get_parameter("external_handoff_max_position_error_m").value
        )
        self.external_handoff_max_horizontal_speed_ms = float(
            self.get_parameter("external_handoff_max_horizontal_speed_ms").value
        )
        self.external_handoff_max_vertical_speed_ms = float(
            self.get_parameter("external_handoff_max_vertical_speed_ms").value
        )

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
        self.sub_local_pos = self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self._cb_local_pos, qos_sensor)
        self.sub_cmd_ack = self.create_subscription(VehicleCommandAck, "/fmu/out/vehicle_command_ack", self._cb_vehicle_command_ack, qos_sensor)
        self.sub_timesync = self.create_subscription(TimesyncStatus, "/fmu/out/timesync_status", self._cb_timesync, qos_sensor)
        self.sub_goto   = self.create_subscription(PoseStamped, "/uam/cmd/goto_pose", self._cb_goto_cmd, qos_reliable)

        # ── Services ──
        self.srv_arm   = self.create_service(Trigger, "/uam/arm_takeoff", self._srv_arm_takeoff)
        self.srv_land  = self.create_service(Trigger, "/uam/land",        self._srv_land)
        self.srv_hold  = self.create_service(Trigger, "/uam/hold",        self._srv_hold)
        self.srv_rtl   = self.create_service(Trigger, "/uam/rtl",         self._srv_rtl)
        self.srv_emg   = self.create_service(Trigger, "/uam/emergency",   self._srv_emergency)

        self.vehicle_status = VehicleStatus()
        self.current_pos    = np.zeros(3)  # NED
        self.current_vel    = np.zeros(3)  # NED
        self.current_yaw    = 0.0
        
        self.setpoint       = np.zeros(3)
        self.setpoint_yaw   = 0.0
        
        self.state          = self.STATE_IDLE
        self.initialized    = False
        self.has_local_pos  = False
        self.local_position_ready = False
        self.local_position_flags = {
            "xy_valid": False,
            "z_valid": False,
            "v_xy_valid": False,
            "v_z_valid": False,
            "heading_good_for_control": False,
        }
        self.px4_timestamp  = 0
        self.px4_timestamp_ros_us = 0
        self.offboard_counter = 0
        self._prime_counter = 0
        self._takeoff_step = 0
        self._takeoff_timer = 0
        self._retry_timer = 0
        self._external_handoff_block_warned = False
        self._external_handoff_stable_since = 0.0
        self._external_handoff_wait_logged = False

        self.timer = self.create_timer(1.0 / self.rate_hz, self._control_loop)
        self.get_logger().info("✅ UAM Mission Bridge (PX4 Position Commander) sẵn sàng!")
        self.get_logger().info(
            f"   Offboard prime: {self.offboard_prime_duration_s:.2f}s "
            f"({self.offboard_prime_ticks} ticks @ {self.rate_hz:.1f} Hz)"
        )
        if self.auto_enable_external_controller and not self.allow_external_torque_handoff:
            self.get_logger().warn(
                "auto_enable_external_controller=true nhưng allow_external_torque_handoff=false. "
                "Mission bridge sẽ giữ /uam/controller_enable=false."
            )

    def _cb_status(self, msg: VehicleStatus):
        if self.vehicle_status.nav_state != msg.nav_state or self.vehicle_status.arming_state != msg.arming_state:
            self.get_logger().info(f"🔔 PX4 Status Change: NavState={msg.nav_state}, ArmingState={msg.arming_state}")
        self.vehicle_status = msg
        self._update_px4_timestamp(msg.timestamp)

    def _cb_odom(self, msg: VehicleOdometry):
        if not self.has_local_pos:
            # Fallback NED từ Odometry nếu VehicleLocalPosition chưa sẵn sàng.
            self.current_pos = np.array([msg.position[0], msg.position[1], msg.position[2]])
            self.current_vel = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]])
            q = msg.q
            siny = 2.0 * (q[0] * q[3] + q[1] * q[2])
            cosy = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
            self.current_yaw = math.atan2(siny, cosy)
            self.initialized = True
        self._update_px4_timestamp(msg.timestamp)

    def _cb_local_pos(self, msg: VehicleLocalPosition):
        self.current_pos = np.array([msg.x, msg.y, msg.z])
        self.current_vel = np.array([msg.vx, msg.vy, msg.vz])
        self.current_yaw = msg.heading
        self.has_local_pos = True
        self.local_position_flags = {
            "xy_valid": bool(msg.xy_valid),
            "z_valid": bool(msg.z_valid),
            "v_xy_valid": bool(msg.v_xy_valid),
            "v_z_valid": bool(msg.v_z_valid),
            "heading_good_for_control": bool(msg.heading_good_for_control),
        }
        self.local_position_ready = (
            self.local_position_flags["xy_valid"]
            and self.local_position_flags["z_valid"]
            and self.local_position_flags["v_xy_valid"]
            and self.local_position_flags["v_z_valid"]
            and (
                self.local_position_flags["heading_good_for_control"]
                or not self.require_heading_good_for_control
            )
        )
        self.initialized = True
        self._update_px4_timestamp(msg.timestamp)

    def _cb_timesync(self, msg: TimesyncStatus):
        self._update_px4_timestamp(msg.timestamp)

    def _cb_vehicle_command_ack(self, msg: VehicleCommandAck):
        watched_commands = (
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
        )
        if msg.command not in watched_commands:
            return

        command_name = self._vehicle_command_name(msg.command)
        result_name = self._vehicle_command_ack_result_name(msg.result)
        text = (
            f"PX4 ACK {command_name}: {result_name} "
            f"(result={msg.result}, result_param1={msg.result_param1}, "
            f"result_param2={msg.result_param2})"
        )
        if msg.result == VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED:
            self.get_logger().info(text)
        else:
            self.get_logger().warn(text)

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
            response.message = "Chưa nhận /fmu/out/vehicle_odometry, không gửi lệnh Offboard/ARM sang PX4."
            return response
        if self.require_local_position_ready and not self.local_position_ready:
            response.success = False
            flags = ", ".join(
                f"{name}={str(value).lower()}"
                for name, value in self.local_position_flags.items()
            )
            response.message = (
                "Local position/heading chưa sẵn sàng "
                "(cần xy_valid, z_valid, v_xy_valid, v_z_valid"
                + (", heading_good_for_control" if self.require_heading_good_for_control else "")
                + f"). Trạng thái hiện tại: {flags}. Không vào Offboard position takeoff."
            )
            return response
        if self.state != self.STATE_IDLE:
            response.success = False
            response.message = f"Đang ở trạng thái {self.state}"
            return response

        self.get_logger().info(f"🚀 Lệnh cất cánh nhận được. Đang chuẩn bị (NED Z={self.takeoff_height}m)...")
        self._log_px4_input_links()
        
        # Lưu toạ độ hiện tại làm điểm bắt đầu
        self.setpoint     = np.array([self.current_pos[0], self.current_pos[1], self.current_pos[2]])
        self.setpoint_yaw = self.current_yaw
        
        # Prime Offboard heartbeat/setpoint trước khi PX4 cho phép switch mode.
        self.state = self.STATE_PRIME_OFFBOARD
        self._prime_counter = 0
        self._takeoff_step = 0
        self._takeoff_timer = 0
        self._retry_timer = 0
            
        response.success = True
        response.message = (
            f"Đã nhận lệnh cất cánh, đang prime Offboard {self.offboard_prime_duration_s:.2f}s "
            "trước khi gửi lệnh mode/ARM..."
        )
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
            self.offboard_counter += 1
            # Bắn toạ độ giữ nguyên vị trí đất để PX4 làm quen
            self._publish_setpoint(self.current_pos, self.current_yaw)
            self._publish_enable(False)

        elif self.state == self.STATE_PRIME_OFFBOARD:
            self._publish_enable(False)
            self._track_current_ground_setpoint()
            self._publish_setpoint(self.setpoint, self.setpoint_yaw)
            self._prime_counter += 1

            if self._prime_counter == 1 or self._prime_counter % max(1, int(self.rate_hz)) == 0:
                self.get_logger().info(
                    f"⏳ Priming Offboard signal {self._prime_counter}/{self.offboard_prime_ticks} ticks..."
                )

            if self._prime_counter >= self.offboard_prime_ticks:
                self.get_logger().info("✅ Đã prime đủ Offboard signal. Bắt đầu trình tự OFFBOARD -> ARM.")
                self.state = self.STATE_TAKEOFF
                self._takeoff_step = 0
                self._takeoff_timer = 0
                self._retry_timer = 0
            
        elif self.state == self.STATE_TAKEOFF:
            self._publish_enable(False)
            
            # Step 0: Request OFFBOARD Mode
            if self._takeoff_step == 0:
                self._track_current_ground_setpoint()
                if self.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                    self._retry_timer += 1
                    if self._retry_timer % 20 == 1: # Once per second
                        self.get_logger().info(
                            f"⏳ 1. Đang yêu cầu OFFBOARD "
                            f"(nav_state={VehicleStatus.NAVIGATION_STATE_OFFBOARD})..."
                        )
                        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                else:
                    self.get_logger().info("✅ Đã vào OFFBOARD mode.")
                    self._takeoff_step = 1
                    self._retry_timer = 0
                self._publish_setpoint(self.setpoint, self.setpoint_yaw)
                return

            # Step 1: Request ARM
            if self._takeoff_step == 1:
                self._track_current_ground_setpoint()
                if self.vehicle_status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                    self._retry_timer += 1
                    if self._retry_timer % 20 == 1:
                        self.get_logger().info(
                            f"⏳ 2. Đang yêu cầu ARM "
                            f"(arming_state={VehicleStatus.ARMING_STATE_ARMED})..."
                        )
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
                self._track_current_ground_setpoint()
                self._takeoff_timer += 1
                if self._takeoff_timer > 40: 
                    self.get_logger().info("🚀 3. Bắt đầu cất cánh (NED Z ramping)...")
                    self.setpoint = self.current_pos.copy()
                    self.setpoint_yaw = self.current_yaw
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
                self._external_handoff_stable_since = 0.0
                self._external_handoff_wait_logged = False

        elif self.state == self.STATE_HOLD:
            self._publish_enable(self._external_enable_allowed())
            self._publish_setpoint(self.setpoint, self.setpoint_yaw)

        elif self.state == self.STATE_GOTO:
            self._publish_enable(self._external_enable_allowed())
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
        msg.timestamp          = self._timestamp_us()
        msg.command            = command
        msg.param1             = float(param1)
        msg.param2             = float(param2)
        msg.target_system      = 1
        msg.target_component   = 1
        msg.source_system      = 1
        msg.source_component   = 1
        msg.from_external      = True
        self.pub_vehicle_cmd.publish(msg)
        self.get_logger().info(
            f"📤 Sent {self._vehicle_command_name(command)} "
            f"(param1={float(param1):.1f}, param2={float(param2):.1f}, ts={msg.timestamp})"
        )

    def _publish_offboard_mode(self):
        # Approach B: keep position=True so PX4 position/attitude loops continue
        # generating VehicleRatesSetpoint. The custom PX4 v1.16.2-rbfnn firmware
        # lets ROS replace only mc_rate_control torque/thrust after handoff.
        msg = OffboardControlMode()
        msg.timestamp    = self._timestamp_us()
        msg.position     = True
        msg.velocity     = False
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False
        self.pub_offboard_mode.publish(msg)

    def _publish_setpoint(self, pos: np.ndarray, yaw: float):
        msg = TrajectorySetpoint()
        msg.timestamp = self._timestamp_us()
        msg.position  = pos.astype(float).tolist()
        msg.yaw       = float(yaw)
        msg.velocity  = [float('nan')] * 3
        msg.acceleration = [float('nan')] * 3
        self.pub_trajectory.publish(msg)

    def _publish_enable(self, enable: bool):
        msg = Bool()
        msg.data = enable
        self.pub_enable.publish(msg)

    def _external_enable_allowed(self) -> bool:
        if not self.auto_enable_external_controller:
            return False
        if self.allow_external_torque_handoff:
            now = self.get_clock().now().nanoseconds / 1e9
            position_error = float(np.linalg.norm(self.current_pos - self.setpoint))
            horizontal_speed = float(np.linalg.norm(self.current_vel[:2]))
            vertical_speed = abs(float(self.current_vel[2]))
            stable = (
                position_error <= self.external_handoff_max_position_error_m
                and horizontal_speed <= self.external_handoff_max_horizontal_speed_ms
                and vertical_speed <= self.external_handoff_max_vertical_speed_ms
            )

            if stable:
                if self._external_handoff_stable_since <= 0.0:
                    self._external_handoff_stable_since = now
                    self._external_handoff_wait_logged = False

                stable_duration = now - self._external_handoff_stable_since

                if stable_duration >= self.external_handoff_stable_time_s:
                    if not self._external_handoff_wait_logged:
                        self.get_logger().info(
                            "External torque handoff gate ready: "
                            f"stable {stable_duration:.1f}s, pos_err={position_error:.2f}m, "
                            f"v_xy={horizontal_speed:.2f}m/s, v_z={vertical_speed:.2f}m/s."
                        )
                        self._external_handoff_wait_logged = True
                    return True

                if not self._external_handoff_wait_logged:
                    self.get_logger().info(
                        "Đang chờ hover ổn định trước external handoff: "
                        f"{stable_duration:.1f}/{self.external_handoff_stable_time_s:.1f}s."
                    )
                    self._external_handoff_wait_logged = True

            else:
                self._external_handoff_stable_since = 0.0
                self._external_handoff_wait_logged = False

            return False

        if not self._external_handoff_block_warned:
            self._external_handoff_block_warned = True
            self.get_logger().warn(
                "Chặn auto external torque handoff vì allow_external_torque_handoff=false. "
                "Đặt true khi chạy firmware custom PX4 v1.16.2-rbfnn."
            )
        return False

    def _track_current_ground_setpoint(self):
        self.setpoint = self.current_pos.copy()
        self.setpoint_yaw = self.current_yaw

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

    def _update_px4_timestamp(self, timestamp: int):
        if timestamp:
            self.px4_timestamp = int(timestamp)
            self.px4_timestamp_ros_us = self.get_clock().now().nanoseconds // 1000

    def _timestamp_us(self) -> int:
        # PX4 commander checks OffboardControlMode freshness in PX4 boot-time.
        # Extrapolate from the last PX4 timestamp so outgoing setpoints do not
        # look stale between odometry/status/timesync callbacks.
        now_us = self.get_clock().now().nanoseconds // 1000
        if self.px4_timestamp and self.px4_timestamp_ros_us:
            elapsed_us = now_us - self.px4_timestamp_ros_us
            if 0 <= elapsed_us < 5_000_000:
                return int(self.px4_timestamp + elapsed_us)
            return int(self.px4_timestamp)
        return now_us

    def _log_px4_input_links(self):
        self.get_logger().info(
            "PX4 input DDS matches: "
            f"offboard={self.pub_offboard_mode.get_subscription_count()}, "
            f"trajectory={self.pub_trajectory.get_subscription_count()}, "
            f"vehicle_cmd={self.pub_vehicle_cmd.get_subscription_count()}"
        )
        if not self.px4_timestamp:
            self.get_logger().warn("Chưa có PX4 timestamp từ status/odometry/timesync; Offboard có thể bị từ chối.")

    def _vehicle_command_name(self, command: int) -> str:
        names = {
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE: "VEHICLE_CMD_DO_SET_MODE",
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM: "VEHICLE_CMD_COMPONENT_ARM_DISARM",
        }
        return names.get(command, f"VehicleCommand({command})")

    def _vehicle_command_ack_result_name(self, result: int) -> str:
        names = {
            VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED: "ACCEPTED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED: "TEMPORARILY_REJECTED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_DENIED: "DENIED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_UNSUPPORTED: "UNSUPPORTED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_FAILED: "FAILED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_IN_PROGRESS: "IN_PROGRESS",
            VehicleCommandAck.VEHICLE_CMD_RESULT_CANCELLED: "CANCELLED",
        }
        return names.get(result, f"UNKNOWN({result})")

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
