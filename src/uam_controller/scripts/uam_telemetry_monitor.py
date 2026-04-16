#!/usr/bin/env python3
"""
uam_telemetry_monitor.py
------------------------
Monitor mở rộng cho UAM.
Kết hợp telemetry PX4 cơ bản (pin, gps, status) và telemetry của
adaptive controller (RBFNN, khối lượng ước lượng, CoM).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    VehicleStatus, VehicleLocalPosition, BatteryStatus, 
    SensorCombined, VehicleGlobalPosition
)
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState

import json
import math


class UAMTelemetryMonitor(Node):
    
    def __init__(self):
        super().__init__("uam_telemetry_monitor")

        self.declare_parameter("battery_warn_pct", 25.0)
        self.declare_parameter("battery_crit_pct", 15.0)
        self.declare_parameter("max_altitude_m", 50.0)
        self.declare_parameter("max_speed_ms", 10.0)
        self.declare_parameter("m_hat_warn_deviation", 1.0)
        
        self.BATT_WARN_PCT   = self.get_parameter("battery_warn_pct").value
        self.BATT_CRIT_PCT   = self.get_parameter("battery_crit_pct").value
        self.MAX_ALTITUDE_M  = self.get_parameter("max_altitude_m").value
        self.MAX_SPEED_MS    = self.get_parameter("max_speed_ms").value
        self.M_HAT_DEV       = self.get_parameter("m_hat_warn_deviation").value
        
        self.m_nominal       = 2.35 # kg — khối lượng danh định (UAV + arm)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ──
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v1", self._cb_status, qos)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self._cb_local, qos)
        self.create_subscription(BatteryStatus, "/fmu/out/battery_status", self._cb_battery, qos)
        self.create_subscription(VehicleGlobalPosition, "/fmu/out/vehicle_global_position", self._cb_global, qos)
        
        self.create_subscription(String, "/uam/state", self._cb_mission_state, 10)
        self.create_subscription(Float64MultiArray, "/uam/debug_state", self._cb_debug_state, 10)
        self.create_subscription(JointState, "/joint_states", self._cb_joint_states, 10)

        # ── Publisher ──
        self.pub_telem = self.create_publisher(String, "/uam/telemetry", 10)

        # ── Data nội bộ ──
        self.status        = VehicleStatus()
        self.local         = VehicleLocalPosition()
        self.battery       = BatteryStatus()
        self.gps           = VehicleGlobalPosition()
        
        self.mission_state = "UNKNOWN"
        self.m_hat         = 0.0
        self.rbfnn_n0      = [0.0, 0.0, 0.0]
        self.g_cog         = [0.0, 0.0, 0.0]
        self.joints        = []

        self.create_timer(0.5, self._publish_telemetry)
        self.create_timer(1.0, self._safety_check)

        self.get_logger().info("✅ UAM Telemetry Monitor khởi động")

    def _cb_status(self,  msg): self.status  = msg
    def _cb_local(self,   msg): self.local   = msg
    def _cb_battery(self, msg): self.battery = msg
    def _cb_global(self,  msg): self.gps     = msg
    def _cb_mission_state(self, msg): 
        try:
            self.mission_state = json.loads(msg.data).get("mission_state", "UNKNOWN")
        except:
            pass

    def _cb_debug_state(self, msg):
        if len(msg.data) >= 20:
            self.m_hat    = msg.data[13]
            self.rbfnn_n0 = [msg.data[14], msg.data[15], msg.data[16]]
            self.g_cog    = [msg.data[17], msg.data[18], msg.data[19]]
            
    def _cb_joint_states(self, msg):
        self.joints = list(msg.position)

    def _publish_telemetry(self):
        speed = math.sqrt(self.local.vx**2 + self.local.vy**2 + self.local.vz**2)

        nav_state_map = {
            VehicleStatus.NAVIGATION_STATE_MANUAL:   "MANUAL",
            VehicleStatus.NAVIGATION_STATE_OFFBOARD: "OFFBOARD",
            VehicleStatus.NAVIGATION_STATE_AUTO_LOITER: "LOITER",
            VehicleStatus.NAVIGATION_STATE_AUTO_RTL:    "RTL",
            VehicleStatus.NAVIGATION_STATE_AUTO_LAND:   "LAND",
        }
        nav_str = nav_state_map.get(self.status.nav_state, "OTHER")

        data = {
            "mission_state": self.mission_state,
            "nav_state":   nav_str,
            "pos": {
                "x":       round(self.local.x, 2),
                "y":       round(-self.local.y, 2),
                "z":       round(-self.local.z, 2),
                "lat":     round(self.gps.lat, 7),
                "lon":     round(self.gps.lon, 7),
                "alt":     round(self.gps.alt, 2),
            },
            "vel_speed":   round(speed, 2),
            "heading_deg": round(math.degrees(self.local.heading), 1),
            "battery": {
                "pct":     round(self.battery.remaining * 100, 1),
                "voltage": round(self.battery.voltage_v, 2),
            },
            "failsafe":    self.status.failsafe,
            "uam_metrics": {
                "m_hat": round(self.m_hat, 2),
                "rbfnn_n0": [round(v, 2) for v in self.rbfnn_n0],
                "g_cog": [round(v, 2) for v in self.g_cog]
            }
        }

        msg = String()
        msg.data = json.dumps(data)
        self.pub_telem.publish(msg)

    def _safety_check(self):
        # batt_pct = self.battery.remaining * 100.0
        # if batt_pct < self.BATT_CRIT_PCT:
        #     self.get_logger().error(f"🔴 PIN NGUY HIỂM: {batt_pct:.1f}% — Cần hạ cánh NGAY!")
        # elif batt_pct < self.BATT_WARN_PCT and batt_pct > 0.1:
        #     self.get_logger().warn(f"🟡 Pin yếu: {batt_pct:.1f}% — Xem xét hạ cánh")

        alt = abs(self.local.z)
        if alt > self.MAX_ALTITUDE_M:
            self.get_logger().warn(f"⚠️  Độ cao cao: {alt:.1f} m (giới hạn {self.MAX_ALTITUDE_M} m)")

        if self.m_hat > 0.0 and abs(self.m_hat - self.m_nominal) > self.M_HAT_DEV:
            self.get_logger().warn(f"⚖️  Khối lượng ước lượng m̂ ({self.m_hat:.2f} kg) sai lệch lớn so với {self.m_nominal} kg!")

        if self.status.failsafe:
            self.get_logger().error("🛑 FAILSAFE đã kích hoạt!")

def main(args=None):
    rclpy.init(args=args)
    node = UAMTelemetryMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
