#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool
from px4_msgs.msg import VehicleLandDetected, VehicleOdometry, VehicleStatus
from std_srvs.srv import Trigger

class QGCRBFNNTrigger(Node):
    NAV_STATE_NAMES = {
        getattr(VehicleStatus, 'NAVIGATION_STATE_MANUAL', 0): 'MANUAL',
        getattr(VehicleStatus, 'NAVIGATION_STATE_ALTCTL', 1): 'ALTCTL',
        getattr(VehicleStatus, 'NAVIGATION_STATE_POSCTL', 2): 'POSCTL',
        getattr(VehicleStatus, 'NAVIGATION_STATE_AUTO_MISSION', 3): 'AUTO_MISSION',
        getattr(VehicleStatus, 'NAVIGATION_STATE_AUTO_LOITER', 4): 'AUTO_LOITER',
        getattr(VehicleStatus, 'NAVIGATION_STATE_AUTO_RTL', 5): 'AUTO_RTL',
        getattr(VehicleStatus, 'NAVIGATION_STATE_OFFBOARD', 14): 'OFFBOARD',
        getattr(VehicleStatus, 'NAVIGATION_STATE_STAB', 15): 'STAB',
        getattr(VehicleStatus, 'NAVIGATION_STATE_AUTO_TAKEOFF', 17): 'AUTO_TAKEOFF',
        getattr(VehicleStatus, 'NAVIGATION_STATE_AUTO_LAND', 18): 'AUTO_LAND',
        getattr(VehicleStatus, 'NAVIGATION_STATE_EXTERNAL1', 23): 'EXTERNAL1',
        getattr(VehicleStatus, 'NAVIGATION_STATE_EXTERNAL2', 24): 'EXTERNAL2',
        getattr(VehicleStatus, 'NAVIGATION_STATE_EXTERNAL3', 25): 'EXTERNAL3',
        getattr(VehicleStatus, 'NAVIGATION_STATE_EXTERNAL4', 26): 'EXTERNAL4',
        getattr(VehicleStatus, 'NAVIGATION_STATE_EXTERNAL5', 27): 'EXTERNAL5',
        getattr(VehicleStatus, 'NAVIGATION_STATE_EXTERNAL6', 28): 'EXTERNAL6',
        getattr(VehicleStatus, 'NAVIGATION_STATE_EXTERNAL7', 29): 'EXTERNAL7',
        getattr(VehicleStatus, 'NAVIGATION_STATE_EXTERNAL8', 30): 'EXTERNAL8',
    }

    def __init__(self):
        super().__init__('qgc_rbfnn_trigger')

        self.declare_parameter('enable_height_m', 1.8)
        self.declare_parameter('vertical_speed_max_ms', 0.15)
        self.declare_parameter('horizontal_speed_max_ms', 0.30)
        self.declare_parameter('stable_hover_time_s', 3.0)
        self.declare_parameter('min_arm_time_s', 2.0)
        self.declare_parameter('require_manual_confirmation', True)
        self.declare_parameter('allow_external_torque_handoff', True)
        self.declare_parameter('required_nav_states', 'OFFBOARD,POSCTL,AUTO_LOITER,AUTO_TAKEOFF')

        self.enable_height_m = float(self.get_parameter('enable_height_m').value)
        self.vertical_speed_max_ms = float(self.get_parameter('vertical_speed_max_ms').value)
        self.horizontal_speed_max_ms = float(self.get_parameter('horizontal_speed_max_ms').value)
        self.stable_hover_time_s = float(self.get_parameter('stable_hover_time_s').value)
        self.min_arm_time_s = float(self.get_parameter('min_arm_time_s').value)
        self.require_manual_confirmation = bool(self.get_parameter('require_manual_confirmation').value)
        self.allow_external_torque_handoff = bool(
            self.get_parameter('allow_external_torque_handoff').value
        )
        self.required_nav_states_text = str(self.get_parameter('required_nav_states').value)
        self.required_nav_states = self._parse_required_nav_states(self.required_nav_states_text)

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
        self.handoff_block_warned = False
        self.mode_gate_block_warned = False
        self.last_mode_gate_warn_s = 0.0

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
        if not self.allow_external_torque_handoff:
            self.get_logger().warn(
                'External torque handoff is disabled by launch/config. '
                'The trigger will monitor stable hover but will not publish /uam/controller_enable=true.'
            )
        self.get_logger().info(
            'External handoff nav-state gate: %s'
            % ','.join(self._nav_state_name(s) for s in sorted(self.required_nav_states))
        )

    def odom_cb(self, msg):
        self.has_odom = True
        self.z = msg.position[2]
        self.vx = msg.velocity[0]
        self.vy = msg.velocity[1]
        self.vz = msg.velocity[2]

    def status_cb(self, msg):
        armed_state = getattr(VehicleStatus, 'ARMING_STATE_ARMED', 2)
        armed = (msg.arming_state == armed_state)
        if armed and not self.armed:
            self.arm_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info('Vehicle armed. Waiting for stable hover before optional external torque handoff.')
        if not armed and self.armed:
            self.get_logger().info('Vehicle disarmed. External torque handoff gate reset.')
            self.hover_ready = False
            self.enable_requested = False
            self.handoff_block_warned = False
            self.mode_gate_block_warned = False
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

        if not self.allow_external_torque_handoff:
            response.success = False
            response.message = (
                'External torque handoff is disabled by allow_external_torque_handoff=false. '
                'Use allow_external_torque_handoff:=true with the custom PX4 v1.16.2-rbfnn firmware.'
            )
            return response

        if not self._nav_state_allowed():
            response.success = False
            response.message = (
                'PX4 nav_state is %s, but external torque handoff requires one of: %s. '
                'Do not enable external torque while QGC/PX4 is still in STAB.'
                % (
                    self._nav_state_name(self.nav_state),
                    ','.join(self._nav_state_name(s) for s in sorted(self.required_nav_states)),
                )
            )
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
        response.message = 'External controller disabled. PX4 internal control remains active.'
        self.get_logger().info('Manual request received. External torque handoff disabled.')
        return response

    def loop(self):
        msg = Bool()

        if not self.armed or not self.has_odom:
            self.controller_enabled = False
            self.stable_counter = 0
            self.hover_ready = False
            self.enable_requested = False
            self.mode_gate_block_warned = False
            msg.data = False
            self.pub_enable.publish(msg)
            return

        now = self.get_clock().now().nanoseconds / 1e9
        armed_long_enough = (now - self.arm_time) >= self.min_arm_time_s
        hover_alt_ok = self.z <= -self.enable_height_m
        vertical_speed_ok = abs(self.vz) <= self.vertical_speed_max_ms
        horizontal_speed_ok = (self.vx * self.vx + self.vy * self.vy) ** 0.5 <= self.horizontal_speed_max_ms
        airborne = not self.landed
        nav_state_ok = self._nav_state_allowed()

        if self.controller_enabled and not nav_state_ok:
            self.controller_enabled = False
            self.enable_requested = False
            self.hover_ready = False
            self.stable_counter = 0
            self.get_logger().error(
                'PX4 nav_state changed to %s while external torque was enabled. '
                'Disabling /uam/controller_enable; switch PX4 to %s before retrying.'
                % (
                    self._nav_state_name(self.nav_state),
                    ','.join(self._nav_state_name(s) for s in sorted(self.required_nav_states)),
                )
            )

        if not self.controller_enabled:
            if armed_long_enough and airborne and hover_alt_ok and vertical_speed_ok and horizontal_speed_ok and nav_state_ok:
                self.stable_counter += 1
                if not self.hover_ready and self.stable_counter * 0.1 >= self.stable_hover_time_s:
                    self.hover_ready = True
                    self.mode_gate_block_warned = False
                    if self.require_manual_confirmation:
                        if self.allow_external_torque_handoff:
                            self.get_logger().info(
                                f"Stable hover detected at {abs(self.z):.2f} m. "
                                "Call /uam/enable_external_controller to enable external torque."
                            )
                        elif not self.handoff_block_warned:
                            self.handoff_block_warned = True
                            self.get_logger().warn(
                                f"Stable hover detected at {abs(self.z):.2f} m, "
                                "but external torque handoff is disabled by launch/config."
                            )
                    else:
                        if self.allow_external_torque_handoff:
                            self.enable_requested = True
                            self.controller_enabled = True
                            self.get_logger().info(
                                f"Stable hover detected at {abs(self.z):.2f} m. Enabling external torque controller."
                            )
                        elif not self.handoff_block_warned:
                            self.handoff_block_warned = True
                            self.get_logger().warn(
                                f"Stable hover detected at {abs(self.z):.2f} m, "
                                "but external torque handoff is disabled by launch/config."
                            )
            else:
                if self.hover_ready:
                    self.get_logger().warn(
                        'Stable-hover gate lost before handoff. External switch request has been reset.'
                    )
                if not nav_state_ok:
                    self._warn_mode_gate(now)
                self.stable_counter = 0
                self.hover_ready = False
                self.enable_requested = False
        msg.data = self.controller_enabled
        self.pub_enable.publish(msg)

    def _parse_required_nav_states(self, text):
        states = set()
        for raw_token in text.split(','):
            token = raw_token.strip().upper()
            if not token:
                continue
            if token.isdigit():
                states.add(int(token))
                continue
            attr = token if token.startswith('NAVIGATION_STATE_') else f'NAVIGATION_STATE_{token}'
            if hasattr(VehicleStatus, attr):
                states.add(int(getattr(VehicleStatus, attr)))
            else:
                self.get_logger().warn(
                    "Ignoring unknown required_nav_states token '%s'" % raw_token.strip()
                )

        if not states:
            states = {getattr(VehicleStatus, 'NAVIGATION_STATE_OFFBOARD', 14)}
            self.get_logger().warn(
                'required_nav_states resolved to empty; falling back to OFFBOARD only.'
            )
        return states

    def _nav_state_allowed(self):
        return int(self.nav_state) in self.required_nav_states

    def _nav_state_name(self, state):
        return self.NAV_STATE_NAMES.get(int(state), f'NAV_STATE_{int(state)}')

    def _warn_mode_gate(self, now):
        if self.mode_gate_block_warned and now - self.last_mode_gate_warn_s < 2.0:
            return

        self.mode_gate_block_warned = True
        self.last_mode_gate_warn_s = now
        self.get_logger().warn(
            'External torque handoff blocked: PX4 nav_state=%s. Required: %s. '
            'QGC Stabilized mode is not accepted for handoff.'
            % (
                self._nav_state_name(self.nav_state),
                ','.join(self._nav_state_name(s) for s in sorted(self.required_nav_states)),
            )
        )


def main():
    rclpy.init()
    node = QGCRBFNNTrigger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
