#!/usr/bin/env python3
"""
arm_virtual_state_node.py
-------------------------
Publish a kinematic /joint_states stream from commanded arm positions.

This node deliberately decouples the arm dynamics calculation from the Gazebo
arm physics. It is useful when the Gazebo CAD/joint chain is only visual or is
not reliable enough, while the controller still needs q, qdot and qddot-derived
wrench from arm_dynamics_node.py.
"""

import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class ArmVirtualStateNode(Node):
    N_JOINTS = 6
    JOINT_NAMES = [f'Joint_{i}' for i in range(1, N_JOINTS + 1)]

    def __init__(self):
        super().__init__('arm_virtual_state_node')

        self.declare_parameter('command_topic', '/arm_controller/joint_trajectory_plan')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('max_velocity_rad_s', 1.0)
        self.declare_parameter('max_acceleration_rad_s2', 4.0)
        self.declare_parameter('publish_before_command', True)

        self.command_topic = self.get_parameter('command_topic').value
        self.joint_state_topic = self.get_parameter('joint_state_topic').value
        self.rate_hz = max(1.0, float(self.get_parameter('publish_rate_hz').value))
        self.max_velocity = max(0.01, float(self.get_parameter('max_velocity_rad_s').value))
        self.max_acceleration = max(0.01, float(self.get_parameter('max_acceleration_rad_s2').value))
        self.publish_before_command = bool(self.get_parameter('publish_before_command').value)

        self.q = np.zeros(self.N_JOINTS)
        self.dq = np.zeros(self.N_JOINTS)
        self.target_q = np.zeros(self.N_JOINTS)
        self._have_command = False
        self._last_tick_time: Optional[float] = None

        self._name_to_index = {}
        for i, name in enumerate(self.JOINT_NAMES):
            joint_num = i + 1
            for alias in (
                name,
                name.lower(),
                name.replace('_', ' '),
                name.replace('_', ' ').lower(),
                f'joint{joint_num}',
                f'joint_{joint_num}',
                f'joint {joint_num}',
                f'j{joint_num}',
                str(joint_num),
            ):
                self._name_to_index[alias] = i

        self.pub = self.create_publisher(JointState, self.joint_state_topic, 10)
        self.sub = self.create_subscription(
            JointState,
            self.command_topic,
            self._on_command,
            10,
        )
        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)

        self.get_logger().info(
            'Arm virtual state started | command_topic=%s | joint_state_topic=%s | '
            'rate=%.1fHz | vmax=%.2f rad/s | amax=%.2f rad/s^2'
            % (
                self.command_topic,
                self.joint_state_topic,
                self.rate_hz,
                self.max_velocity,
                self.max_acceleration,
            )
        )

    def _positions_from_command(self, msg: JointState) -> Optional[np.ndarray]:
        if msg.name:
            positions = self.target_q.copy()
            updated = 0
            for name, pos in zip(msg.name, msg.position):
                idx = self._name_to_index.get(name)
                if idx is None:
                    idx = self._name_to_index.get(name.lower())
                if idx is None:
                    self.get_logger().warn(f'Bỏ qua joint name không biết: {name}')
                    continue
                try:
                    value = float(pos)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    positions[idx] = value
                    updated += 1
            return positions if updated > 0 else None

        if len(msg.position) >= self.N_JOINTS:
            values = np.array([float(p) for p in msg.position[:self.N_JOINTS]], dtype=float)
            values[~np.isfinite(values)] = 0.0
            return values

        self.get_logger().warn(
            'Joint command không có name và chỉ có %d position, cần 6 phần tử.'
            % len(msg.position)
        )
        return None

    def _on_command(self, msg: JointState):
        positions = self._positions_from_command(msg)
        if positions is None:
            return
        self.target_q = positions
        self._have_command = True

    def _tick(self):
        now = self.get_clock().now().nanoseconds / 1e9
        if self._last_tick_time is None:
            self._last_tick_time = now
            if self.publish_before_command:
                self._publish()
            return

        dt = max(1e-4, min(now - self._last_tick_time, 0.1))
        self._last_tick_time = now

        if self._have_command:
            error = self.target_q - self.q
            desired_dq = np.clip(error / dt, -self.max_velocity, self.max_velocity)
            max_delta_dq = self.max_acceleration * dt
            self.dq += np.clip(desired_dq - self.dq, -max_delta_dq, max_delta_dq)
            self.q += self.dq * dt

            # Snap tiny residuals to prevent slow numerical drift around target.
            close = np.abs(self.target_q - self.q) < 1e-4
            self.q[close] = self.target_q[close]
            self.dq[close] = 0.0

        if self._have_command or self.publish_before_command:
            self._publish()

    def _publish(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.JOINT_NAMES)
        msg.position = [float(v) for v in self.q]
        msg.velocity = [float(v) for v in self.dq]
        msg.effort = [0.0] * self.N_JOINTS
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArmVirtualStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
