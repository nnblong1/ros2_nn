"""
uam_qgc_mode.launch.py
----------------------
Chạy UAM system với QGroundControl làm giao diện điều khiển chính.

KHÁC BIỆT VỚI uam_system.launch.py:
  - KHÔNG chạy uam_mission_bridge  → QGC đảm nhận arm/takeoff/waypoints
  - Có thể chạy external rate controller qua MC_RATE_EXT_EN của PX4
  - Vẫn chạy arm nodes             → Gazebo arm control vẫn hoạt động

LUỒNG ĐIỀU KHIỂN VỚI QGC:
  1. Chạy lệnh này để khởi động ROS 2 nodes + DDS Agent
  2. Mở terminal riêng: cd ~/PX4-Autopilot && make px4_sitl gz_x500_hop
  3. QGC tự kết nối qua UDP:14550
  4. Arm bằng QGC → Takeoff → giữ hover 2m ổn định
  5. Nếu MC_RATE_EXT_EN=1 và enable_rbfnn=true, trigger sẽ tự bật external rate controller sau hover ổn định

CHẾ ĐỘ BAY:
  - enable_rbfnn=false                → PX4 internal controller hoàn toàn
  - enable_rbfnn=true + MC_RATE_EXT_EN=0
                                     → ROS node chạy nhưng PX4 vẫn dùng internal rate controller
  - enable_rbfnn=true + MC_RATE_EXT_EN=1
                                     → sau hover ổn định, ROS node thay rate controller bằng torque/thrust external

Cách dùng:
  # Gazebo SITL
  ros2 launch uam_controller uam_qgc_mode.launch.py sim:=true

  # Phần cứng thật
  ros2 launch uam_controller uam_qgc_mode.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
    LogInfo,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg_share = FindPackageShare('uam_controller')

    # ═══════════════════════════════════════════════════════════
    #  ARGUMENTS
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
        description='Đường dẫn file YAML chứa tham số hệ thống'
    )

    arg_rbfnn = DeclareLaunchArgument(
        'enable_rbfnn',
        default_value='true',
        description='true = chạy RBFNN Backstepping node (cần cho OFFBOARD mode)'
    )

    arg_handoff_mode = DeclareLaunchArgument(
        'external_handoff_mode',
        default_value='manual',
        description='manual = chờ gọi service để chuyển sang external | auto = tự bật sau hover ổn định'
    )

    arg_rbfnn_output_enable = DeclareLaunchArgument(
        'rbfnn_output_enable',
        default_value='true',
        description='true = backstepping + RBFNN, false = backstepping only'
    )

    arg_start_data_logger = DeclareLaunchArgument(
        'start_data_logger',
        default_value='true',
        description='true = ghi dữ liệu thí nghiệm ra CSV/JSON/Markdown'
    )

    arg_experiment_case = DeclareLaunchArgument(
        'experiment_case',
        default_value='manual_qgc',
        description='Tên ca thử nghiệm, dùng để đặt tên thư mục kết quả'
    )

    arg_experiment_output_root = DeclareLaunchArgument(
        'experiment_output_root',
        default_value='/home/wicom/PX4-Autopilot/Tools/simulation/gz/pid_search_results/uam_verification',
        description='Thư mục gốc để lưu kết quả kiểm chứng'
    )

    arg_log_rate_hz = DeclareLaunchArgument(
        'experiment_log_rate_hz',
        default_value='20.0',
        description='Tần số ghi dữ liệu thí nghiệm'
    )

    sim          = LaunchConfiguration('sim')
    config_file  = LaunchConfiguration('config_file')
    enable_rbfnn = LaunchConfiguration('enable_rbfnn')
    handoff_mode = LaunchConfiguration('external_handoff_mode')
    rbfnn_output_enable = LaunchConfiguration('rbfnn_output_enable')
    start_data_logger = LaunchConfiguration('start_data_logger')
    experiment_case = LaunchConfiguration('experiment_case')
    experiment_output_root = LaunchConfiguration('experiment_output_root')
    experiment_log_rate_hz = LaunchConfiguration('experiment_log_rate_hz')

    # ═══════════════════════════════════════════════════════════
    #  NODE 0 – Micro XRCE-DDS Agent
    #  Cầu nối PX4 ↔ ROS 2 (telemetry + OFFBOARD setpoints)
    #  Hardware : UART /dev/ttyAMA0 @ 921600 bps
    #  Sim      : UDP port 8888
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
    #  NODE 1 – RBFNN Backstepping Controller (C++, 100 Hz)
    #  Chỉ publish Torque/Thrust khi nhận cờ enable từ qgc_rbfnn_trigger.
    # ═══════════════════════════════════════════════════════════

    backstepping_node = Node(
        package='uam_controller',
        executable='uam_backstepping_rbfnn_node',
        name='uam_adaptive_controller',
        output='screen',
        parameters=[
            config_file,
            {
                'rbfnn_enable': ParameterValue(rbfnn_output_enable, value_type=bool),
            }
        ],
        remappings=[
            ('/fmu/in/offboard_control_mode',   '/fmu/in/offboard_control_mode'),
            ('/fmu/in/vehicle_torque_setpoint',  '/fmu/in/vehicle_torque_setpoint'),
            ('/fmu/in/vehicle_thrust_setpoint',  '/fmu/in/vehicle_thrust_setpoint'),
            ('/fmu/out/vehicle_odometry',        '/fmu/out/vehicle_odometry'),
            ('/fmu/out/vehicle_status',          '/fmu/out/vehicle_status_v1'),
        ],
        additional_env={'ROS_DOMAIN_ID': '0'},
        condition=IfCondition(enable_rbfnn)
    )

    # ═══════════════════════════════════════════════════════════
    #  NODE 2 – Arm Dynamics Newton-Euler (Python, 50 Hz)
    #  Tính lực/momen bù từ động học cánh tay
    # ═══════════════════════════════════════════════════════════

    arm_dynamics_node = Node(
        package='uam_controller',
        executable='arm_dynamics_node.py',
        name='arm_dynamics_node',
        output='screen',
        parameters=[config_file]
    )

    # ═══════════════════════════════════════════════════════════
    #  NODE 3 – Arm Gazebo Command Bridge (Sim only)
    #  Chuyển JointState → lệnh Gazebo transport
    # ═══════════════════════════════════════════════════════════

    arm_cmd_node = Node(
        package='uam_controller',
        executable='arm_gazebo_command_node.py',
        name='arm_gazebo_command_node',
        output='screen',
        condition=IfCondition(sim)
    )

    # ═══════════════════════════════════════════════════════════
    #  NODE 4 – Arm Initial Pose (chạy 1 lần rồi thoát)
    #  Gập cánh tay về vị trí an toàn trước khi takeoff
    # ═══════════════════════════════════════════════════════════

    arm_initial_pose_node = Node(
        package='uam_controller',
        executable='arm_initial_pose.py',
        name='arm_initial_pose',
        output='screen',
        condition=IfCondition(sim)
    )

    # ═══════════════════════════════════════════════════════════
    #  NODE 5 – Telemetry Monitor
    #  In thông tin bay ra terminal (altitude, attitude, v.v.)
    # ═══════════════════════════════════════════════════════════

    telemetry_node = Node(
        package='uam_controller',
        executable='uam_telemetry_monitor.py',
        name='uam_telemetry_monitor',
        output='screen',
        parameters=[config_file]
    )

    # ═══════════════════════════════════════════════════════════
    #  NODE 6 – RBFNN Data Logger
    # ═══════════════════════════════════════════════════════════

    data_logger_node = Node(
        package='uam_controller',
        executable='rbfnn_data_logger.py',
        name='rbfnn_data_logger',
        output='screen',
        parameters=[{
            'case_name': experiment_case,
            'output_root': experiment_output_root,
            'log_rate_hz': ParameterValue(experiment_log_rate_hz, value_type=float),
            'target_alt_m': 2.0,
        }],
        condition=IfCondition(start_data_logger)
    )

    # ═══════════════════════════════════════════════════════════
    #  NODE 7 – QGC Auto Trigger
    #  Tự động kích hoạt RBFNN khi hover ổn định
    # ═══════════════════════════════════════════════════════════

    qgc_trigger_node = Node(
        package='uam_controller',
        executable='qgc_rbfnn_trigger.py',
        name='qgc_rbfnn_trigger',
        output='screen',
        parameters=[{
            'require_manual_confirmation': ParameterValue(
                PythonExpression(["'", handoff_mode, "' == 'manual'"]),
                value_type=bool,
            )
        }],
        condition=IfCondition(enable_rbfnn)
    )

    # ═══════════════════════════════════════════════════════════
    #  THÔNG BÁO HƯỚNG DẪN KHI KHỞI ĐỘNG
    # ═══════════════════════════════════════════════════════════

    startup_info = LogInfo(
        msg=(
            '\n'
            '╔══════════════════════════════════════════════════════════╗\n'
            '║           UAM QGroundControl Mode - Ready                ║\n'
            '╠══════════════════════════════════════════════════════════╣\n'
            '║  QGC kết nối tự động qua UDP:14550                       ║\n'
            '║                                                          ║\n'
            '║  Bước 1: Khởi động PX4 SITL (terminal riêng):            ║\n'
            '║    cd ~/PX4-Autopilot && make px4_sitl gz_x500_hop       ║\n'
            '║                                                          ║\n'
            '║  Bước 2: Mở QGroundControl → vehicle tự xuất hiện        ║\n'
            '║                                                          ║\n'
            '║  Bước 3: Trong QGC:                                      ║\n'
            '║    • Kéo Arm để khởi động động cơ                        ║\n'
            '║    • Đẩy ga Takeoff (Position Mode / Altitude Mode)      ║\n'
            '║                                                          ║\n'
            '║  External rate controller chỉ hoạt động khi:             ║\n'
            '║    • enable_rbfnn=true                                    ║\n'
            '║    • MC_RATE_EXT_EN=1 trong PX4                           ║\n'
            '║    • hover ổn định đủ thời gian trigger                   ║\n'
            '║    • manual mode: gọi /uam/enable_external_controller     ║\n'
            '╚══════════════════════════════════════════════════════════╝'
        )
    )

    # ═══════════════════════════════════════════════════════════
    #  THỨ TỰ KHỞI ĐỘNG
    #  t=0s   DDS Agent + Startup info
    #  t=2s   RBFNN Controller (chờ DDS ổn định)
    #  t=2.5s Arm Dynamics
    #  t=3s   Arm Cmd Bridge (sim)
    #  t=2s   Arm Initial Pose (sim) → tự chờ thêm 3s bên trong
    #  t=3.5s Telemetry Monitor
    #  t=4s   Data Logger
    # ═══════════════════════════════════════════════════════════

    delayed_backstepping  = TimerAction(period=2.0,  actions=[backstepping_node])
    delayed_arm_dynamics  = TimerAction(period=2.5,  actions=[arm_dynamics_node])
    delayed_arm_cmd       = TimerAction(period=3.0,  actions=[arm_cmd_node])
    delayed_arm_pose      = TimerAction(period=2.0,  actions=[arm_initial_pose_node])
    delayed_telemetry     = TimerAction(period=3.5,  actions=[telemetry_node])
    delayed_logger        = TimerAction(period=4.0,  actions=[data_logger_node])

    return LaunchDescription([
        # ── Arguments ──
        arg_sim,
        arg_config,
        arg_rbfnn,
        arg_handoff_mode,
        arg_rbfnn_output_enable,
        arg_start_data_logger,
        arg_experiment_case,
        arg_experiment_output_root,
        arg_log_rate_hz,
        # ── Hướng dẫn ──
        startup_info,
        # ── DDS Agent (khởi động ngay) ──
        xrce_hardware,
        xrce_sim,
        # ── Các controller node (khởi động có trễ) ──
        delayed_backstepping,
        delayed_arm_dynamics,
        delayed_arm_cmd,
        delayed_arm_pose,
        delayed_telemetry,
        delayed_logger,
        qgc_trigger_node,
    ])
