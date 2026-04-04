#!/usr/bin/env python3
"""
Mission Manager Node
Cung cấp ROS2 Services để điều khiển UAV từ bên ngoài.
Kết hợp với offboard_control.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import VehicleStatus, VehicleLocalPosition
from std_msgs.msg import String
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped

# Service tùy chỉnh (xem srv/GotoPosition.srv)
try:
    from uav_control_interfaces.srv import GotoPosition, SetMission
    HAS_CUSTOM_SRV = True
except ImportError:
    HAS_CUSTOM_SRV = False

import json
import time


class MissionManager(Node):
    """
    Node quản lý mission cấp cao.
    Expose ROS2 Services:
        /uav/arm_takeoff  (Trigger)
        /uav/land         (Trigger)
        /uav/hold         (Trigger)
        /uav/rtl          (Trigger)
        /uav/emergency    (Trigger)
        /uav/goto         (geometry_msgs/PoseStamped via topic)
    """

    def __init__(self):
        super().__init__("mission_manager")

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

        # ── Subscribers (nhận trạng thái) ─────────────────────────────────
        self.sub_status = self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status",
            self._cb_status, qos_sensor)
        self.sub_local  = self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position",
            self._cb_local, qos_sensor)

        # ── Publisher (gửi lệnh goto) ──────────────────────────────────────
        self.pub_goto_cmd  = self.create_publisher(
            PoseStamped, "/uav/cmd/goto_pose", qos_reliable)
        self.pub_state_str = self.create_publisher(
            String, "/uav/state", qos_reliable)

        # ── Services ──────────────────────────────────────────────────────
        self.srv_arm      = self.create_service(
            Trigger, "/uav/arm_takeoff",  self._srv_arm_takeoff)
        self.srv_land     = self.create_service(
            Trigger, "/uav/land",         self._srv_land)
        self.srv_hold     = self.create_service(
            Trigger, "/uav/hold",         self._srv_hold)
        self.srv_rtl      = self.create_service(
            Trigger, "/uav/rtl",          self._srv_rtl)
        self.srv_emg      = self.create_service(
            Trigger, "/uav/emergency",    self._srv_emergency)

        # Subscriber nhận lệnh goto từ topic /uav/cmd/goto_pose
        self.sub_goto = self.create_subscription(
            PoseStamped, "/uav/cmd/goto_pose",
            self._cb_goto_cmd, qos_reliable)

        # ── Mission waypoints dạng queue ──────────────────────────────────
        self.waypoints: list[dict] = []
        self.wp_index = 0
        self.mission_active = False

        # ── Trạng thái nội bộ ─────────────────────────────────────────────
        self.vehicle_status   = None
        self.local_pos        = None
        self.current_mission  = "IDLE"

        # ── Timer telemetry ───────────────────────────────────────────────
        self.timer = self.create_timer(1.0, self._publish_state)

        self.get_logger().info("✅ MissionManager node sẵn sàng")
        self.get_logger().info("   Services: /uav/arm_takeoff | /uav/land | /uav/hold | /uav/rtl | /uav/emergency")

    # ═══════════════════════════════════════════════════════════════════════
    # Callbacks
    # ═══════════════════════════════════════════════════════════════════════

    def _cb_status(self, msg: VehicleStatus):
        self.vehicle_status = msg

    def _cb_local(self, msg: VehicleLocalPosition):
        self.local_pos = msg

    def _cb_goto_cmd(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        self.get_logger().info(f"🗺️  Nhận lệnh goto: ({x:.2f}, {y:.2f}, {z:.2f})")

    # ═══════════════════════════════════════════════════════════════════════
    # Service handlers
    # ═══════════════════════════════════════════════════════════════════════

    def _srv_arm_takeoff(self, request, response):
        self.get_logger().info("📡 Service: arm_takeoff")
        # Trong thực tế, gọi offboard_control.arm_and_takeoff()
        # Ở đây publish command qua topic hoặc action server
        self.current_mission = "TAKEOFF"
        response.success = True
        response.message = "Arm và takeoff thành công"
        return response

    def _srv_land(self, request, response):
        self.get_logger().info("🛬 Service: land")
        self.current_mission = "LAND"
        response.success = True
        response.message = "Đang hạ cánh"
        return response

    def _srv_hold(self, request, response):
        self.get_logger().info("🔒 Service: hold")
        self.current_mission = "HOLD"
        response.success = True
        response.message = "Giữ vị trí"
        return response

    def _srv_rtl(self, request, response):
        self.get_logger().info("🏠 Service: RTL")
        self.current_mission = "RTL"
        response.success = True
        response.message = "Return-to-Launch kích hoạt"
        return response

    def _srv_emergency(self, request, response):
        self.get_logger().warn("🛑 Service: EMERGENCY STOP")
        self.current_mission = "EMERGENCY"
        response.success = True
        response.message = "Dừng khẩn cấp"
        return response

    # ═══════════════════════════════════════════════════════════════════════
    # Mission: chạy danh sách waypoints tuần tự
    # ═══════════════════════════════════════════════════════════════════════

    def load_mission(self, waypoints: list[dict]):
        """
        Nạp danh sách waypoints. Mỗi waypoint là dict:
          {"x": 5.0, "y": 0.0, "z": -2.5, "hold_sec": 3.0}
        """
        self.waypoints     = waypoints
        self.wp_index      = 0
        self.mission_active = True
        self.get_logger().info(f"📋 Nạp mission với {len(waypoints)} waypoints")

    def _publish_state(self):
        state_info = {
            "mission": self.current_mission,
            "wp_index": self.wp_index,
            "total_wp": len(self.waypoints),
        }
        if self.local_pos:
            state_info.update({
                "x": round(self.local_pos.x, 2),
                "y": round(self.local_pos.y, 2),
                "z": round(self.local_pos.z, 2),
            })
        msg = String()
        msg.data = json.dumps(state_info)
        self.pub_state_str.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MissionManager()

    # Ví dụ nạp mission waypoints
    demo_mission = [
        {"x":  0.0, "y":  0.0, "z": -2.5, "hold_sec": 3.0},
        {"x":  5.0, "y":  0.0, "z": -2.5, "hold_sec": 2.0},
        {"x":  5.0, "y":  5.0, "z": -3.0, "hold_sec": 2.0},
        {"x":  0.0, "y":  5.0, "z": -2.5, "hold_sec": 2.0},
        {"x":  0.0, "y":  0.0, "z": -2.5, "hold_sec": 3.0},  # về home
    ]
    node.load_mission(demo_mission)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
