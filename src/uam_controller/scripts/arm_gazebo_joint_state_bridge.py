#!/usr/bin/env python3
"""
Bridge Gazebo transport joint states to ROS 2 /joint_states.

This bridge is only for the custom arm-equipped x500_hop model. Keep this node
off when running a non-arm PX4 model such as gz_x500.
"""

import time

import rclpy
from rclpy.node import Node as RosNode
from sensor_msgs.msg import JointState


GZ_TRANSPORT_OK = False
GzNode = None
GzModel = None

for _transport, _msgs in [
    ('gz.transport13', 'gz.msgs10'),
    ('gz.transport12', 'gz.msgs9'),
]:
    try:
        _t = __import__(_transport, fromlist=['Node'])
        _m = __import__(f'{_msgs}.model_pb2', fromlist=['Model'])
        GzNode = _t.Node
        GzModel = _m.Model
        GZ_TRANSPORT_OK = True
        break
    except ImportError:
        continue


class ArmGazeboJointStateBridge(RosNode):
    """Convert configured Gazebo arm joints into ROS JointState order."""

    SDF_JOINT_NAMES = [
        'Revolute_20', 'Revolute_22', 'Revolute_23',
        'Revolute_26', 'Revolute_28', 'Revolute_30',
    ]
    ROS_JOINT_NAMES = [f'Joint_{i}' for i in range(1, 7)]
    DEFAULT_GZ_TOPICS = [
        '/model/x500_hop/joint_state',
        '/model/x500_hop_0/joint_state',
        '/world/default/model/x500_hop/joint_state',
        '/world/default/model/x500_hop_0/joint_state',
    ]

    def __init__(self):
        super().__init__('arm_gazebo_joint_state_bridge')

        self.declare_parameter('gazebo_joint_state_topics', self.DEFAULT_GZ_TOPICS)
        self.declare_parameter('publish_topic', '/joint_states')

        self.gz_topics = list(
            self.get_parameter('gazebo_joint_state_topics').get_parameter_value().string_array_value
        )
        if not self.gz_topics:
            self.gz_topics = self.DEFAULT_GZ_TOPICS

        publish_topic = self.get_parameter('publish_topic').get_parameter_value().string_value
        self.pub = self.create_publisher(JointState, publish_topic, 10)

        self._last_rx_log = 0.0
        self._active_topic = None

        if not GZ_TRANSPORT_OK:
            self.get_logger().error(
                'Không import được gz transport/model protobuf; /joint_states sẽ không có dữ liệu arm.'
            )
            return

        self._gz_node = GzNode()
        for topic in self.gz_topics:
            self._gz_node.subscribe(GzModel, topic, self._make_callback(topic))
            self.get_logger().info(f'Subscribed Gazebo joint state: {topic}')

        self.get_logger().info(f'Publishing ROS joint states: {publish_topic}')

    def _make_callback(self, topic):
        def _callback(msg):
            self._on_gz_joint_state(topic, msg)
        return _callback

    def _on_gz_joint_state(self, topic, msg):
        joints = {joint.name: joint for joint in msg.joint}
        if not all(name in joints for name in self.SDF_JOINT_NAMES):
            return

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(self.ROS_JOINT_NAMES)
        out.position = []
        out.velocity = []
        out.effort = []

        for name in self.SDF_JOINT_NAMES:
            axis = joints[name].axis1
            out.position.append(float(axis.position))
            out.velocity.append(float(axis.velocity))
            out.effort.append(float(axis.force))

        self.pub.publish(out)

        now = time.time()
        if self._active_topic != topic or now - self._last_rx_log > 5.0:
            self._active_topic = topic
            self._last_rx_log = now
            self.get_logger().info(
                'Joint state bridge active from %s -> [%s]' % (
                    topic,
                    ', '.join(f'{p:.3f}' for p in out.position),
                )
            )


def main(args=None):
    rclpy.init(args=args)
    node = ArmGazeboJointStateBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
