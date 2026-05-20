#!/usr/bin/env python3
"""
arm_gazebo_command_node.py
--------------------------
Nhận lệnh JointState từ /arm_controller/joint_trajectory_plan
và publish trực tiếp tới Gazebo transport (KHÔNG cần ros_gz_bridge).

Cách tiếp cận:
  1. Thử dùng gz-transport Python bindings (nhanh, không overhead)
  2. Nếu không có, dùng subprocess gọi `gz topic -p` (chậm hơn nhưng luôn hoạt động)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import subprocess
import shutil
import time

# ── Thử import gz-transport Python bindings ──
GZ_TRANSPORT_OK = False
GzNode = None
GzDouble = None

for _transport, _msgs in [
    ('gz.transport13', 'gz.msgs10'),
    ('gz.transport12', 'gz.msgs9'),
]:
    try:
        _t = __import__(_transport, fromlist=['Node'])
        _m = __import__(f'{_msgs}.double_pb2', fromlist=['Double'])
        GzNode = _t.Node
        GzDouble = _m.Double
        GZ_TRANSPORT_OK = True
        break
    except ImportError:
        continue


class ArmGazeboCommandNode(Node):
    """Chuyển đổi JointState → publish trực tiếp tới Gazebo transport."""

    JOINT_NAMES = [
        'Revolute_20', 'Revolute_22', 'Revolute_23',
        'Revolute_26', 'Revolute_28', 'Revolute_30'
    ]

    def __init__(self):
        super().__init__('arm_gazebo_command_node')

        # Match x500_hop/model.sdf JointPositionController topics.
        self.declare_parameter('model_prefix', 'model/x500_hop')
        self.declare_parameter('command_topic', '/arm_controller/joint_trajectory_plan')
        self.declare_parameter('hold_publish_rate_hz', 20.0)
        self.declare_parameter('subprocess_pulse_period_s', 0.5)
        prefix = self.get_parameter('model_prefix') \
                     .get_parameter_value().string_value
        self.command_topic = self.get_parameter('command_topic') \
                                 .get_parameter_value().string_value
        self.hold_publish_rate_hz = max(
            1.0,
            self.get_parameter('hold_publish_rate_hz').get_parameter_value().double_value,
        )
        self.subprocess_pulse_period_s = max(
            0.0,
            self.get_parameter('subprocess_pulse_period_s').get_parameter_value().double_value,
        )

        # Build danh sách topic Gazebo matching SDF:
        # /model/x500_hop/joint/Revolute_20/cmd_pos, ...
        self.gz_topics = [
            f'/{prefix}/joint/{joint_name}/cmd_pos'
            for joint_name in self.JOINT_NAMES
        ]

        self._gz_bin = shutil.which('gz') or shutil.which('ign')

        # ── Chọn phương thức publish ──
        if GZ_TRANSPORT_OK:
            self._init_gz_transport()
        else:
            self._init_subprocess_fallback()

        self._last_positions = [0.0] * len(self.JOINT_NAMES)
        self._have_command = False
        self._command_count = 0
        self._last_info_log = 0.0
        self._last_subprocess_pulse = 0.0

        # Subscribe lệnh trajectory plan (ROS 2)
        self.sub_joint_plan = self.create_subscription(
            JointState,
            self.command_topic,
            self._on_joint_plan,
            10,
        )
        self.hold_timer = self.create_timer(
            1.0 / self.hold_publish_rate_hz,
            self._publish_hold_command,
        )

        self.get_logger().info(
            f'Arm Gazebo Command Node started | '
            f'method={"gz-transport" if GZ_TRANSPORT_OK else "subprocess"} | '
            f'command_topic={self.command_topic} | '
            f'hold_rate={self.hold_publish_rate_hz:.1f}Hz | '
            f'subprocess_pulse={self.subprocess_pulse_period_s:.2f}s'
        )
        for t in self.gz_topics:
            self.get_logger().info(f'  → {t}')

    # ──────────────── gz-transport Python ────────────────
    def _init_gz_transport(self):
        self._gz_node = GzNode()
        self._gz_pubs = []
        for topic in self.gz_topics:
            pub = self._gz_node.advertise(topic, GzDouble)
            self._gz_pubs.append(pub)
        self.get_logger().info('Sử dụng gz-transport Python bindings')

    def _publish_gz_transport(self, positions):
        for i, pub in enumerate(self._gz_pubs):
            msg = GzDouble()
            msg.data = float(positions[i])
            pub.publish(msg)

    # ──────────────── subprocess fallback ────────────────
    def _init_subprocess_fallback(self):
        if not self._gz_bin:
            self.get_logger().error(
                'Không tìm thấy lệnh gz hoặc ign! '
                'Cánh tay sẽ KHÔNG DI CHUYỂN được.'
            )
        else:
            self.get_logger().warn(
                f'gz-transport Python không có, dùng subprocess: {self._gz_bin}'
            )

    def _publish_subprocess(self, positions):
        if not self._gz_bin:
            return
        for i, topic in enumerate(self.gz_topics):
            try:
                subprocess.Popen(
                    [self._gz_bin, 'topic', '-t', topic,
                     '-m', 'gz.msgs.Double',
                     '-p', f'data: {positions[i]}'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                self.get_logger().error(f'gz topic -p failed: {e}')

    def _publish_to_gazebo(self, positions, *, force_subprocess=False):
        if GZ_TRANSPORT_OK:
            self._publish_gz_transport(positions)
        else:
            self._publish_subprocess(positions)

        now = time.time()
        pulse_due = (
            self.subprocess_pulse_period_s > 0.0
            and now - self._last_subprocess_pulse >= self.subprocess_pulse_period_s
        )
        if force_subprocess or pulse_due:
            self._last_subprocess_pulse = now
            self._publish_subprocess(positions)

    def _publish_hold_command(self):
        if not self._have_command:
            return

        # Gazebo transport can occasionally miss single-shot commands during
        # discovery or under load. Re-publishing the latest command turns the
        # ROS trajectory stream into a held position setpoint for the Gazebo
        # JointPositionController.
        self._publish_to_gazebo(self._last_positions)

    # ──────────────── Callback chính ────────────────
    def _on_joint_plan(self, msg: JointState):
        num_expected = len(self.JOINT_NAMES)
        if len(msg.position) < num_expected:
            self.get_logger().warn(
                f'JointState chỉ có {len(msg.position)} phần tử, cần {num_expected}'
            )
            return

        self._last_positions = [float(p) for p in msg.position[:num_expected]]
        self._have_command = True
        self._command_count += 1
        self._publish_to_gazebo(
            self._last_positions,
            force_subprocess=(self._command_count == 1),
        )

        now = time.time()
        if self._command_count == 1 or now - self._last_info_log > 2.0:
            self._last_info_log = now
            self.get_logger().info(
                'Arm command #%d → [%s]' % (
                    self._command_count,
                    ', '.join(f'{p:.3f}' for p in self._last_positions),
                )
            )


def main(args=None):
    rclpy.init(args=args)
    node = ArmGazeboCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
