#!/usr/bin/env python3
"""
Telemetry Monitor Node
Theo dõi trạng thái UAV, ghi log, cảnh báo an toàn.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    VehicleStatus,
    VehicleLocalPosition,
    BatteryStatus,
    SensorCombined,
    VehicleGlobalPosition,
)
from std_msgs.msg import String

import json
import math


class TelemetryMonitor(Node):
    """Giám sát và ghi log telemetry từ PX4."""

    # Ngưỡng cảnh báo
    BATT_WARN_PCT  = 25.0   # %
    BATT_CRIT_PCT  = 15.0   # %
    MAX_ALTITUDE_M =  50.0  # m
    MAX_SPEED_MS   =  10.0  # m/s

    def __init__(self):
        super().__init__("telemetry_monitor")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            VehicleStatus,        "/fmu/out/vehicle_status",
            self._cb_status, qos)
        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position",
            self._cb_local, qos)
        self.create_subscription(
            BatteryStatus,        "/fmu/out/battery_status",
            self._cb_battery, qos)
        self.create_subscription(
            VehicleGlobalPosition, "/fmu/out/vehicle_global_position",
            self._cb_global, qos)
        self.create_subscription(
            SensorCombined,       "/fmu/out/sensor_combined",
            self._cb_sensor, qos)

        # ── Publisher telemetry tổng hợp ──────────────────────────────────
        self.pub_telem = self.create_publisher(
            String, "/uav/telemetry", 10)

        # ── Dữ liệu nội bộ ───────────────────────────────────────────────
        self.status   = VehicleStatus()
        self.local    = VehicleLocalPosition()
        self.battery  = BatteryStatus()
        self.gps      = VehicleGlobalPosition()
        self.sensors  = SensorCombined()

        # ── Timer publish telemetry ───────────────────────────────────────
        self.create_timer(0.5, self._publish_telemetry)
        self.create_timer(1.0, self._safety_check)

        self.get_logger().info("✅ TelemetryMonitor khởi động")

    # ═══════════════════════════════════════════════════════════════════════
    # Callbacks
    # ═══════════════════════════════════════════════════════════════════════

    def _cb_status(self,  msg): self.status  = msg
    def _cb_local(self,   msg): self.local   = msg
    def _cb_battery(self, msg): self.battery = msg
    def _cb_global(self,  msg): self.gps     = msg
    def _cb_sensor(self,  msg): self.sensors = msg

    # ═══════════════════════════════════════════════════════════════════════
    # Publish telemetry JSON
    # ═══════════════════════════════════════════════════════════════════════

    def _publish_telemetry(self):
        speed = math.sqrt(
            self.local.vx**2 + self.local.vy**2 + self.local.vz**2)

        arm_state_map = {
            VehicleStatus.ARMING_STATE_INIT:     "INIT",
            VehicleStatus.ARMING_STATE_STANDBY:  "STANDBY",
            VehicleStatus.ARMING_STATE_ARMED:    "ARMED",
            VehicleStatus.ARMING_STATE_STANDBY_ERROR: "STANDBY_ERROR",
            VehicleStatus.ARMING_STATE_SHUTDOWN: "SHUTDOWN",
        }
        arm_str = arm_state_map.get(self.status.arming_state, "UNKNOWN")

        nav_state_map = {
            VehicleStatus.NAVIGATION_STATE_MANUAL:   "MANUAL",
            VehicleStatus.NAVIGATION_STATE_OFFBOARD: "OFFBOARD",
            VehicleStatus.NAVIGATION_STATE_AUTO_MISSION: "MISSION",
            VehicleStatus.NAVIGATION_STATE_AUTO_LOITER: "LOITER",
            VehicleStatus.NAVIGATION_STATE_AUTO_RTL:    "RTL",
            VehicleStatus.NAVIGATION_STATE_AUTO_LAND:   "LAND",
        }
        nav_str = nav_state_map.get(self.status.nav_state, "OTHER")

        data = {
            "arm_state":   arm_str,
            "nav_state":   nav_str,
            "pos": {
                "x":       round(self.local.x, 2),
                "y":       round(self.local.y, 2),
                "z":       round(self.local.z, 2),
                "lat":     round(self.gps.lat, 7),
                "lon":     round(self.gps.lon, 7),
                "alt":     round(self.gps.alt, 2),
            },
            "vel": {
                "vx":      round(self.local.vx, 2),
                "vy":      round(self.local.vy, 2),
                "vz":      round(self.local.vz, 2),
                "speed":   round(speed, 2),
            },
            "heading_deg": round(math.degrees(self.local.heading), 1),
            "battery": {
                "pct":     round(self.battery.remaining * 100, 1),
                "voltage": round(self.battery.voltage_v, 2),
                "current": round(self.battery.current_a, 2),
            },
            "failsafe":    self.status.failsafe,
        }

        msg = String()
        msg.data = json.dumps(data)
        self.pub_telem.publish(msg)

    # ═══════════════════════════════════════════════════════════════════════
    # Safety checks
    # ═══════════════════════════════════════════════════════════════════════

    def _safety_check(self):
        batt_pct = self.battery.remaining * 100.0
        if batt_pct < self.BATT_CRIT_PCT:
            self.get_logger().error(
                f"🔴 PIN NGUY HIỂM: {batt_pct:.1f}% — Cần hạ cánh NGAY!")
        elif batt_pct < self.BATT_WARN_PCT:
            self.get_logger().warn(
                f"🟡 Pin yếu: {batt_pct:.1f}% — Xem xét hạ cánh")

        alt = abs(self.local.z)
        if alt > self.MAX_ALTITUDE_M:
            self.get_logger().warn(
                f"⚠️  Độ cao cao: {alt:.1f} m (giới hạn {self.MAX_ALTITUDE_M} m)")

        speed = math.sqrt(
            self.local.vx**2 + self.local.vy**2 + self.local.vz**2)
        if speed > self.MAX_SPEED_MS:
            self.get_logger().warn(
                f"⚠️  Tốc độ cao: {speed:.1f} m/s (giới hạn {self.MAX_SPEED_MS} m/s)")

        if self.status.failsafe:
            self.get_logger().error("🛑 FAILSAFE đã kích hoạt!")


def main(args=None):
    rclpy.init(args=args)
    node = TelemetryMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
