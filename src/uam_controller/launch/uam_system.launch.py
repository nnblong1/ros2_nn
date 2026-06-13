"""
uam_system.launch.py
--------------------
Khởi động toàn bộ hệ thống UAM.

NGUYÊN TẮC THIẾT KẾ:
  - Dùng firmware custom PX4 v1.16.2-rbfnn để giữ PX4 position/attitude loop
    và cho ROS backstepping+RBFNN thay mc_rate_control sau handoff.
  - Mọi tham số điều khiển nằm trong:
      config/uam_controller_params.yaml   (mặc định)
      config/uam_controller_params_sim.yaml   (khi sim:=true)
  - File này chỉ quyết định: node nào chạy, chạy với config nào,
    kết nối topic ra sao, và thứ tự khởi động.

Cách dùng:
  # Chạy phần cứng thật
  ros2 launch uam_controller uam_system.launch.py

  # Chạy Gazebo SITL
  ros2 launch uam_controller uam_system.launch.py sim:=true

  # Tắt bypass để bay so sánh bằng PX4 internal rate PID
  ros2 launch uam_controller uam_system.launch.py sim:=true allow_external_torque_handoff:=false

  # Tự bật backstepping+RBFNN sau khi takeoff xong và chuyển sang HOLD/GOTO
  ros2 launch uam_controller uam_system.launch.py sim:=true auto_enable_external_controller:=true

  # Chỉ định file config khác
  ros2 launch uam_controller uam_system.launch.py \\
      config_file:=/path/to/my_custom_params.yaml

  # Chỉ định file trọng số LSTM khác
  ros2 launch uam_controller uam_system.launch.py \\
      model_path:=/path/to/retrained_weights.pth
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg_share = FindPackageShare('uam_controller')

    # ═══════════════════════════════════════════════════════════
    #  KHAI BÁO ARGUMENT
    #  Chỉ có 5 loại argument hợp lệ ở đây:
    #    1. Chọn môi trường (sim / hardware)
    #    2. Chọn file config
    #    3. Chọn cho phép torque handoff sang ROS backstepping+RBFNN
    #    4. Chọn tự bật controller ngoài sau HOLD/GOTO
    #    5. Chọn file model (phụ thuộc đường dẫn máy)
    # ═══════════════════════════════════════════════════════════

    arg_sim = DeclareLaunchArgument(
        'sim',
        default_value='false',
        description='true = Gazebo SITL | false = phần cứng thật'
    )

    arg_config = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution(
            [pkg_share, 'config', 'uam_controller_params.yaml']
        ),
        description='Đường dẫn file YAML chứa toàn bộ tham số hệ thống'
    )

    arg_allow_external_torque_handoff = DeclareLaunchArgument(
        'allow_external_torque_handoff',
        default_value='true',
        description='true = cho phép ROS publish torque/thrust khi mission bridge enable; false = chỉ PX4 internal rate PID.'
    )

    arg_auto_enable_external_controller = DeclareLaunchArgument(
        'auto_enable_external_controller',
        default_value='false',
        description='true = mission bridge publish /uam/controller_enable=true sau khi vào HOLD/GOTO.'
    )

    sim         = LaunchConfiguration('sim')
    config_file = LaunchConfiguration('config_file')
    allow_external_torque_handoff = LaunchConfiguration('allow_external_torque_handoff')
    auto_enable_external_controller = LaunchConfiguration('auto_enable_external_controller')

    # ═══════════════════════════════════════════════════════════
    #  NODE 0 – Micro XRCE-DDS Agent
    #  Cầu nối PX4 Firmware ↔ ROS2 qua Micro XRCE-DDS
    #  Hardware: UART /dev/ttyAMA0 @ 921600 bps
    #  Sim    : UDP port 8888
    # ═══════════════════════════════════════════════════════════

    xrce_hardware = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'serial', '--dev', '/dev/ttyAMA0', '-b', '921600'],
        name='micro_xrce_dds_agent',
        output='screen',
        condition=UnlessCondition(sim)
    )

    xrce_sim = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        name='micro_xrce_dds_agent_sim',
        output='screen',
        condition=IfCondition(sim)
    )

    # ═══════════════════════════════════════════════════════════
    #  NODE 1 – Adaptive Backstepping + RBFNN  (C++, 100 Hz)
    #  Tham số: đọc từ config_file, sau đó launch override cổng torque handoff
    # ═══════════════════════════════════════════════════════════

    backstepping_node = Node(
        package='uam_controller',
        executable='uam_backstepping_rbfnn_node',
        name='uam_adaptive_controller',
        output='screen',
        parameters=[
            config_file,
            {
                'allow_external_torque_output': ParameterValue(
                    allow_external_torque_handoff,
                    value_type=bool,
                ),
            }
        ],
        remappings=[
            ('/fmu/in/offboard_control_mode',   '/fmu/in/offboard_control_mode'),
            ('/fmu/in/vehicle_torque_setpoint',  '/fmu/in/vehicle_torque_setpoint'),
            ('/fmu/in/vehicle_thrust_setpoint',  '/fmu/in/vehicle_thrust_setpoint'),
            ('/fmu/out/vehicle_odometry',        '/fmu/out/vehicle_odometry'),
            ('/fmu/out/vehicle_status',          '/fmu/out/vehicle_status_v1'),
        ],
        additional_env={'ROS_DOMAIN_ID': '0'}
    )


    # ═══════════════════════════════════════════════════════════
    #  NODE 3 – Arm Dynamics Newton-Euler  (Python, 50 Hz)
    #  Tham số: ĐỌC HOÀN TOÀN từ config_file
    # ═══════════════════════════════════════════════════════════

    arm_dynamics_node = Node(
        package='uam_controller',
        executable='arm_dynamics_node.py',
        name='arm_dynamics_node',
        output='screen',
        parameters=[config_file]    # link_masses, link_lengths, dh_*
    )

    # ═══════════════════════════════════════════════════════════
    #  THỨ TỰ KHỞI ĐỘNG
    #  DDS Agent cần sẵn sàng trước khi các node ROS2 kết nối PX4
    #
    #  t=0s   DDS Agent khởi động
    #  t=2s   Backstepping node bắt đầu (chờ DDS ổn định)
    #  t=2.5s Arm Dynamics node bắt đầu
    # ═══════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════
    #  NODE 4 – Arm Command Bridge (ROS 2 JointState -> Gazebo Cmd)
    #  Publish trực tiếp tới Gazebo transport (không cần ros_gz_bridge)
    # ═══════════════════════════════════════════════════════════

    arm_cmd_node = Node(
        package='uam_controller',
        executable='arm_gazebo_command_node.py',
        name='arm_gazebo_command_node',
        output='screen',
        condition=IfCondition(sim)
    )

    arm_joint_state_bridge_node = Node(
        package='uam_controller',
        executable='arm_gazebo_joint_state_bridge.py',
        name='arm_gazebo_joint_state_bridge',
        output='screen',
        condition=IfCondition(sim)
    )

    # ═══════════════════════════════════════════════════════════
    #  NODE 4.5 – Arm Initial Pose (run once then exit)
    #  Gửi góc ban đầu cho cánh tay để co vào trước khi takeoff
    # ═══════════════════════════════════════════════════════════

    arm_initial_pose_proc = Node(
        package='uam_controller',
        executable='arm_initial_pose.py',
        name='arm_initial_pose',
        output='screen',
        condition=IfCondition(sim)
    )

    # ═══════════════════════════════════════════════════════════
    #  NODE 5 & 6 – Mission Bridge & Telemetry Monitor
    # ═══════════════════════════════════════════════════════════

    mission_bridge_node = Node(
        package='uam_controller',
        executable='uam_mission_bridge.py',
        name='uam_mission_bridge',
        output='screen',
        parameters=[
            config_file,
            {
                'allow_external_torque_handoff': ParameterValue(
                    allow_external_torque_handoff,
                    value_type=bool,
                ),
                'auto_enable_external_controller': ParameterValue(
                    auto_enable_external_controller,
                    value_type=bool,
                ),
            }
        ]
    )

    telemetry_node = Node(
        package='uam_controller',
        executable='uam_telemetry_monitor.py',
        name='uam_telemetry_monitor',
        output='screen',
        parameters=[config_file]
    )

    data_logger_node = Node(
        package='uam_controller',
        executable='rbfnn_data_logger.py',
        name='rbfnn_data_logger',
        output='screen'
    )

    delayed_arm_cmd      = TimerAction(period=1.0,   actions=[arm_cmd_node])
    delayed_arm_js_bridge = TimerAction(period=1.0,  actions=[arm_joint_state_bridge_node])
    delayed_arm_pose     = TimerAction(period=2.0,   actions=[arm_initial_pose_proc])  # t=2s: script tự chờ 3s nữa → joint đầu tiên lúc t≈5s
    delayed_backstepping = TimerAction(period=2.0,   actions=[backstepping_node])
    delayed_arm_dynamics = TimerAction(period=2.5,   actions=[arm_dynamics_node])
    delayed_telemetry    = TimerAction(period=3.5,   actions=[telemetry_node])
    delayed_logger       = TimerAction(period=4.0,   actions=[data_logger_node])

    # 🚨 QUAN TRỌNG: Cất cánh SAU KHI tay đã gập xong
    # arm_initial_pose: t=2s start + 3s startup + 6×2s joints = t≈17s done
    delayed_mission      = TimerAction(period=20.0,   actions=[mission_bridge_node])

    return LaunchDescription([
        # Arguments
        arg_sim,
        arg_config,
        arg_allow_external_torque_handoff,
        arg_auto_enable_external_controller,
        # DDS bridge (khởi động ngay)
        xrce_hardware,
        xrce_sim,
        # Các node điều khiển (khởi động có trễ)
        delayed_mission,
        delayed_backstepping,
        delayed_arm_dynamics,
        delayed_arm_cmd,
        delayed_arm_js_bridge,
        delayed_arm_pose,    # Gửi tư thế co tay sau 5s
        delayed_telemetry,
        delayed_logger,
    ])
