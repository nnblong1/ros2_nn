#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool
from px4_msgs.msg import VehicleLandDetected, VehicleOdometry, VehicleStatus
from std_srvs.srv import Trigger

class QGCRBFNNTrigger(Node):
    def __init__(self):
        super().__init__('qgc_rbfnn_trigger')

        self.declare_parameter('enable_height_m', 1.8)
        self.declare_parameter('vertical_speed_max_ms', 0.15)
        self.declare_parameter('horizontal_speed_max_ms', 0.30)
        self.declare_parameter('stable_hover_time_s', 3.0)
        self.declare_parameter('min_arm_time_s', 2.0)
        self.declare_parameter('require_manual_confirmation', True)

        self.enable_height_m = float(self.get_parameter('enable_height_m').value)
        self.vertical_speed_max_ms = float(self.get_parameter('vertical_speed_max_ms').value)
        self.horizontal_speed_max_ms = float(self.get_parameter('horizontal_speed_max_ms').value)
        self.stable_hover_time_s = float(self.get_parameter('stable_hover_time_s').value)
        self.min_arm_time_s = float(self.get_parameter('min_arm_time_s').value)
        self.require_manual_confirmation = bool(self.get_parameter('require_manual_confirmation').value)

        self.sub_odom = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_cb,
            qos_profile_sensor_data,
        )
        self.sub_status = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v1',
            self.status_cb,
            qos_profile_sensor_data,
        )
        self.sub_land = self.create_subscription(
            VehicleLandDetected,
            '/fmu/out/vehicle_land_detected',
            self.land_cb,
            qos_profile_sensor_data,
        )
        self.pub_enable = self.create_publisher(Bool, '/uam/controller_enable', 10)
        self.srv_enable = self.create_service(
            Trigger, '/uam/enable_external_controller', self.enable_external_cb
        )
        self.srv_disable = self.create_service(
            Trigger, '/uam/disable_external_controller', self.disable_external_cb
        )

        self.timer = self.create_timer(0.1, self.loop)

        self.has_odom = False
        self.armed = False
        self.landed = True
        self.nav_state = 0
        self.z = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.arm_time = 0.0
        self.controller_enabled = False
        self.stable_counter = 0
        self.hover_ready = False
        self.enable_requested = False

        if self.require_manual_confirmation:
            self.get_logger().info(
                'QGC trigger started in manual handoff mode. '
                'Use /uam/enable_external_controller after stable hover is detected.'
            )
        else:
            self.get_logger().info(
                'QGC trigger started in auto handoff mode. '
                'External controller will be enabled automatically after stable hover.'
            )

    def odom_cb(self, msg):
        self.has_odom = True
        self.z = msg.position[2]
        self.vx = msg.velocity[0]
        self.vy = msg.velocity[1]
        self.vz = msg.velocity[2]

    def status_cb(self, msg):
        armed = (msg.arming_state == 2)
        if armed and not self.armed:
            self.arm_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info('Vehicle armed. Waiting for stable hover before external-controller handoff.')
        if not armed and self.armed:
            self.get_logger().info('Vehicle disarmed. External rate controller gate reset.')
            self.hover_ready = False
            self.enable_requested = False
        self.armed = armed
        self.nav_state = msg.nav_state

    def land_cb(self, msg):
        self.landed = bool(msg.landed)

    def enable_external_cb(self, request, response):
        del request
        if not self.armed:
            response.success = False
            response.message = 'Vehicle is not armed.'
            return response

        if self.controller_enabled:
            response.success = True
            response.message = 'External controller is already enabled.'
            return response

        if not self.hover_ready:
            response.success = False
            response.message = (
                'Stable hover gate is not ready yet. Wait until hover has been stable '
                f'for {self.stable_hover_time_s:.1f}s.'
            )
            return response

        self.enable_requested = True
        self.controller_enabled = True
        response.success = True
        response.message = 'External-controller handoff request accepted.'
        self.get_logger().info('Manual request received. External rate controller will be enabled now.')
        return response

    def disable_external_cb(self, request, response):
        del request
        self.controller_enabled = False
        self.enable_requested = False
        response.success = True
        response.message = 'External controller disabled. PX4 internal fallback remains active.'
        self.get_logger().info('Manual request received. External rate controller disabled.')
        return response

    def loop(self):
        msg = Bool()

        if not self.armed or not self.has_odom:
            self.controller_enabled = False
            self.stable_counter = 0
            self.hover_ready = False
            self.enable_requested = False
            msg.data = False
            self.pub_enable.publish(msg)
            return

        now = self.get_clock().now().nanoseconds / 1e9
        armed_long_enough = (now - self.arm_time) >= self.min_arm_time_s
        hover_alt_ok = self.z <= -self.enable_height_m
        vertical_speed_ok = abs(self.vz) <= self.vertical_speed_max_ms
        horizontal_speed_ok = (self.vx * self.vx + self.vy * self.vy) ** 0.5 <= self.horizontal_speed_max_ms
        airborne = not self.landed

        if not self.controller_enabled:
            if armed_long_enough and airborne and hover_alt_ok and vertical_speed_ok and horizontal_speed_ok:
                self.stable_counter += 1
                if not self.hover_ready and self.stable_counter * 0.1 >= self.stable_hover_time_s:
                    self.hover_ready = True
                    if self.require_manual_confirmation:
                        self.get_logger().info(
                            f"Stable hover detected at {abs(self.z):.2f} m. "
                            "Call /uam/enable_external_controller to switch to external mode."
                        )
                    else:
                        self.enable_requested = True
                        self.controller_enabled = True
                        self.get_logger().info(
                            f"Stable hover detected at {abs(self.z):.2f} m. Enabling external rate controller."
                        )
            else:
                if self.hover_ready:
                    self.get_logger().warn(
                        'Stable-hover gate lost before handoff. External switch request has been reset.'
                    )
                self.stable_counter = 0
                self.hover_ready = False
                self.enable_requested = False
        msg.data = self.controller_enabled
        self.pub_enable.publish(msg)


def main():
    rclpy.init()
    node = QGCRBFNNTrigger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
