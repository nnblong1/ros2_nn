#!/usr/bin/env python3
"""
Launch file chính: khởi động toàn bộ stack điều khiển UAV
  - offboard_control   : điều khiển cấp thấp PX4
  - mission_manager    : quản lý mission cấp cao
  - telemetry_monitor  : giám sát và ghi log
  - micro_ros_agent    : cầu nối DDS giữa ROS2 và PX4
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Arguments ─────────────────────────────────────────────────────────
    serial_port_arg = DeclareLaunchArgument(
        "serial_port",
        default_value="/dev/ttyAMA0",
        description="Serial port kết nối RPi4 với Pixhawk 6C",
    )
    baud_arg = DeclareLaunchArgument(
        "baud",
        default_value="921600",
        description="Baud rate UART",
    )
    takeoff_height_arg = DeclareLaunchArgument(
        "takeoff_height",
        default_value="-2.0",
        description="Độ cao cất cánh (m, NED âm = lên)",
    )

    serial_port   = LaunchConfiguration("serial_port")
    baud          = LaunchConfiguration("baud")
    takeoff_height = LaunchConfiguration("takeoff_height")

    # ── micro-ROS Agent (cầu nối DDS) ─────────────────────────────────────
    # Kết nối Pixhawk 6C → RPi4 qua UART (serial)
    micro_ros_agent = ExecuteProcess(
        cmd=[
            "ros2", "run", "micro_ros_agent", "micro_ros_agent",
            "serial",
            "--dev", serial_port,
            "-b", baud,
            "-v", "4",           # verbose level
        ],
        output="screen",
        name="micro_ros_agent",
    )

    # ── Node: offboard_control ─────────────────────────────────────────────
    offboard_node = Node(
        package="uav_control",
        executable="offboard_control",
        name="offboard_control",
        output="screen",
        parameters=[{
            "takeoff_height":    takeoff_height,
            "cruise_speed":      2.0,
            "position_threshold": 0.25,
            "loop_rate_hz":      20.0,
        }],
        remappings=[],
    )

    # ── Node: mission_manager ─────────────────────────────────────────────
    mission_node = Node(
        package="uav_control",
        executable="mission_manager",
        name="mission_manager",
        output="screen",
    )

    # ── Node: telemetry_monitor ───────────────────────────────────────────
    telem_node = Node(
        package="uav_control",
        executable="telemetry_monitor",
        name="telemetry_monitor",
        output="screen",
    )

    # ── Khởi động theo thứ tự ─────────────────────────────────────────────
    # micro-ROS agent trước → đợi 2s → offboard và telemetry → đợi 1s → mission
    return LaunchDescription([
        serial_port_arg,
        baud_arg,
        takeoff_height_arg,

        LogInfo(msg="🚁 Khởi động UAV control stack..."),

        micro_ros_agent,

        TimerAction(period=2.0, actions=[
            LogInfo(msg="📡 micro-ROS agent đã khởi động"),
            offboard_node,
            telem_node,
        ]),

        TimerAction(period=3.0, actions=[
            mission_node,
            LogInfo(msg="✅ Toàn bộ stack đã sẵn sàng"),
        ]),
    ])