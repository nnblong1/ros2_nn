"""
uam_qgc_mode.launch.py
----------------------
Chạy UAM system với QGroundControl làm giao diện điều khiển chính.

KHÁC BIỆT VỚI uam_system.launch.py:
  - KHÔNG chạy uam_mission_bridge  → QGC đảm nhận arm/takeoff/waypoints
  - Dùng firmware custom PX4 v1.16.2-rbfnn: PX4 giữ position/attitude loop,
    ROS backstepping+RBFNN thay mc_rate_control sau khi handoff
  - Vẫn chạy arm nodes             → Gazebo arm control hoạt động với model x500_hop

LUỒNG ĐIỀU KHIỂN VỚI QGC:
  1. Chạy lệnh này để khởi động ROS 2 nodes + DDS Agent
  2. Mở terminal riêng: cd ~/PX4-Autopilot && make px4_sitl gz_x500_hop
  3. QGC tự kết nối qua UDP:14550
  4. Arm bằng QGC → Takeoff → giữ hover 2m ổn định
  5. Trigger bật external controller thủ công hoặc tự động sau hover ổn định

CHẾ ĐỘ BAY:
  - MC_RATE_EXT_EN=1 trong airframe gz_x500_hop:
      PX4 fallback về rate PID nội bộ khi ROS chưa publish torque mới.
  - external_handoff_mode:=manual:
      sau hover ổn định, gọi /uam/enable_external_controller để bắt đầu bypass.
  - external_handoff_mode:=auto:
      tự bật bypass sau hover ổn định.
  - qgc_rbfnn_trigger chỉ cho bật handoff khi PX4 đang ở mode được phép
    (mặc định OFFBOARD/POSCTL/AUTO_LOITER/AUTO_TAKEOFF). STAB bị chặn.

Cách dùng:
  # Gazebo SITL
  ros2 launch uam_controller uam_qgc_mode.launch.py sim:=true

  # Gazebo SITL với file YAML tùy chọn
  ros2 launch uam_controller uam_qgc_mode.launch.py \
    sim:=true \
    config_file:=/home/wicom/ros2_ws/src/uam_controller/config/uam_controller_params.yaml

  # Chạy một YAML đã tune xong từ best-param-search
  ros2 launch uam_controller uam_qgc_mode.launch.py \
    sim:=true \
    config_file:=/home/wicom/uam_results/rbfnn_best_param_search_<timestamp>/final_best_uam_controller_params.yaml \
    arm_ff_enable:=true

  # Backstepping-only với YAML tùy chọn, không dùng output RBFNN
  ros2 launch uam_controller uam_qgc_mode.launch.py \
    sim:=true \
    config_file:=/path/to/custom_uam_controller_params.yaml \
    rbfnn_output_enable:=false

  Lưu ý:
  - config_file phải là đường dẫn tuyệt đối hoặc đường dẫn mà shell hiện tại resolve được.
  - YAML cần giữ đúng node keys: arm_dynamics_node.ros__parameters và
    uam_adaptive_controller.ros__parameters.
  - Nếu YAML nằm trong src sau khi sửa code/package, build lại rồi source workspace:
    colcon build --packages-select uam_controller
    source /home/wicom/ros2_ws/install/setup.bash
  - Dùng nhánh PX4 px4-v1.16.2-rbfnn và target gz_x500_hop để có model arm custom.

  # Phần cứng thật
  ros2 launch uam_controller uam_qgc_mode.launch.py sim:=false xrce_serial_dev:=/dev/ttyACM0
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
        description='manual = chờ service sau hover | auto = tự bật bypass sau hover ổn định'
    )

    arg_allow_external_torque_handoff = DeclareLaunchArgument(
        'allow_external_torque_handoff',
        default_value='true',
        description='true = cho phép ROS publish torque/thrust khi trigger enable; false = chỉ monitor/log.'
    )

    arg_required_nav_states = DeclareLaunchArgument(
        'required_nav_states',
        default_value='OFFBOARD,POSCTL,AUTO_LOITER,AUTO_TAKEOFF',
        description='Danh sách nav_state PX4 được phép handoff external torque. Không nên thêm STAB.'
    )

    arg_rbfnn_output_enable = DeclareLaunchArgument(
        'rbfnn_output_enable',
        default_value='true',
        description='true = backstepping + RBFNN, false = backstepping only'
    )

    arg_arm_ff_enable = DeclareLaunchArgument(
        'arm_ff_enable',
        default_value='true',
        description='true = dùng /arm/interaction_wrench làm feedforward torque trong external controller'
    )

    arg_arm_virtual_disturbance_enable = DeclareLaunchArgument(
        'arm_virtual_disturbance_enable',
        default_value='false',
        description='true = tiêm torque nhiễu từ cánh tay ảo vào torque setpoint để mô phỏng plant bị cánh tay tác động'
    )

    arg_start_data_logger = DeclareLaunchArgument(
        'start_data_logger',
        default_value='true',
        description='true = ghi dữ liệu thí nghiệm ra CSV/JSON/Markdown'
    )

    arg_arm_state_source = DeclareLaunchArgument(
        'arm_state_source',
        default_value='commanded',
        description='commanded = /joint_states lấy từ lệnh khớp, gazebo = lấy từ joint thật trong Gazebo'
    )

    arg_use_gazebo_arm_visual = DeclareLaunchArgument(
        'use_gazebo_arm_visual',
        default_value='false',
        description='true = vẫn gửi lệnh sang Gazebo arm để nhìn chuyển động; false = chỉ tính động lực học logic'
    )

    arg_experiment_case = DeclareLaunchArgument(
        'experiment_case',
        default_value='manual_qgc',
        description='Tên ca thử nghiệm, dùng để đặt tên thư mục kết quả'
    )

    arg_experiment_output_root = DeclareLaunchArgument(
        'experiment_output_root',
        default_value='/home/wicom/uam_results/uam_verification',
        description='Thư mục gốc để lưu kết quả kiểm chứng'
    )

    arg_log_rate_hz = DeclareLaunchArgument(
        'experiment_log_rate_hz',
        default_value='20.0',
        description='Tần số ghi dữ liệu thí nghiệm'
    )

    arg_start_xrce_agent = DeclareLaunchArgument(
        'start_xrce_agent',
        default_value='true',
        description='true = launch tự chạy MicroXRCEAgent | false = agent đã chạy ở terminal/service khác'
    )

    arg_xrce_serial_dev = DeclareLaunchArgument(
        'xrce_serial_dev',
        default_value='/dev/ttyAMA0',
        description='Thiết bị serial cho MicroXRCEAgent khi sim=false, ví dụ /dev/ttyAMA0, /dev/ttyUSB0, /dev/ttyACM0'
    )

    arg_xrce_baud = DeclareLaunchArgument(
        'xrce_baud',
        default_value='921600',
        description='Baudrate uXRCE-DDS serial khi sim=false'
    )

    sim          = LaunchConfiguration('sim')
    config_file  = LaunchConfiguration('config_file')
    enable_rbfnn = LaunchConfiguration('enable_rbfnn')
    handoff_mode = LaunchConfiguration('external_handoff_mode')
    allow_external_torque_handoff = LaunchConfiguration('allow_external_torque_handoff')
    required_nav_states = LaunchConfiguration('required_nav_states')
    rbfnn_output_enable = LaunchConfiguration('rbfnn_output_enable')
    arm_ff_enable = LaunchConfiguration('arm_ff_enable')
    arm_virtual_disturbance_enable = LaunchConfiguration('arm_virtual_disturbance_enable')
    start_data_logger = LaunchConfiguration('start_data_logger')
    arm_state_source = LaunchConfiguration('arm_state_source')
    use_gazebo_arm_visual = LaunchConfiguration('use_gazebo_arm_visual')
    experiment_case = LaunchConfiguration('experiment_case')
    experiment_output_root = LaunchConfiguration('experiment_output_root')
    experiment_log_rate_hz = LaunchConfiguration('experiment_log_rate_hz')
    start_xrce_agent = LaunchConfiguration('start_xrce_agent')
    xrce_serial_dev = LaunchConfiguration('xrce_serial_dev')
    xrce_baud = LaunchConfiguration('xrce_baud')

    # ═══════════════════════════════════════════════════════════
    #  NODE 0 – Micro XRCE-DDS Agent
    #  Cầu nối PX4 ↔ ROS 2 (telemetry + OFFBOARD setpoints)
    #  Hardware : configurable serial device @ configurable baud
    #  Sim      : UDP port 8888
    # ═══════════════════════════════════════════════════════════

    xrce_hardware = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'serial', '--dev', xrce_serial_dev, '-b', xrce_baud],
        name='micro_xrce_dds_agent',
        output='screen',
        condition=IfCondition(PythonExpression(["'", sim, "' == 'false' and '", start_xrce_agent, "' == 'true'"]))
    )

    xrce_sim = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        name='micro_xrce_dds_agent_sim',
        output='screen',
        condition=IfCondition(PythonExpression(["'", sim, "' == 'true' and '", start_xrce_agent, "' == 'true'"]))
    )

    # ═══════════════════════════════════════════════════════════
    #  NODE 1 – RBFNN Backstepping Controller (C++, 200 Hz)
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
                'allow_external_torque_output': ParameterValue(
                    allow_external_torque_handoff,
                    value_type=bool,
                ),
                'arm_ff_enable': ParameterValue(arm_ff_enable, value_type=bool),
                'arm_virtual_disturbance_enable': ParameterValue(
                    arm_virtual_disturbance_enable,
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
    #  NODE 2.5 – Virtual Arm State
    #  Dùng lệnh khớp làm nguồn /joint_states để tính nội lực arm,
    #  không phụ thuộc joint vật lý/CAD trong Gazebo.
    # ═══════════════════════════════════════════════════════════

    arm_virtual_state_node = Node(
        package='uam_controller',
        executable='arm_virtual_state_node.py',
        name='arm_virtual_state_node',
        output='screen',
        condition=IfCondition(PythonExpression(["'", arm_state_source, "' == 'commanded'"]))
    )

    # ═══════════════════════════════════════════════════════════
    #  NODE 3 – Arm Gazebo Command Bridge (Sim only, optional visual)
    #  Chuyển JointState → lệnh Gazebo transport
    # ═══════════════════════════════════════════════════════════

    arm_cmd_node = Node(
        package='uam_controller',
        executable='arm_gazebo_command_node.py',
        name='arm_gazebo_command_node',
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", sim, "' == 'true' and '", use_gazebo_arm_visual, "' == 'true'"
        ]))
    )

    # ═══════════════════════════════════════════════════════════
    #  NODE 3.5 – Arm Joint State Bridge (Sim only)
    #  Chuyển Gazebo JointStatePublisher → ROS /joint_states cho RNE/logger.
    # ═══════════════════════════════════════════════════════════

    arm_joint_state_bridge_node = Node(
        package='uam_controller',
        executable='arm_gazebo_joint_state_bridge.py',
        name='arm_gazebo_joint_state_bridge',
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", sim, "' == 'true' and '", arm_state_source, "' == 'gazebo'"
        ]))
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
        condition=IfCondition(PythonExpression([
            "'", sim, "' == 'true' and '", use_gazebo_arm_visual, "' == 'true'"
        ]))
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
            ),
            'allow_external_torque_handoff': ParameterValue(
                allow_external_torque_handoff,
                value_type=bool,
            ),
            'required_nav_states': required_nav_states,
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
            '║  SITL: QGC kết nối tự động qua UDP:14550                  ║\n'
            '║  Real: kiểm tra XRCE serial + QGC trước khi arm           ║\n'
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
            '║  Firmware: px4-v1.16.2-rbfnn, MC_RATE_EXT_EN=1           ║\n'
            '║  ROS chỉ publish torque sau khi trigger hover bật enable. ║\n'
            '║  Manual: gọi /uam/enable_external_controller sau hover.   ║\n'
            '║  STAB bị chặn; dùng POSCTL/HOLD/OFFBOARD trước handoff.   ║\n'
            '╚══════════════════════════════════════════════════════════╝'
        )
    )

    # ═══════════════════════════════════════════════════════════
    #  THỨ TỰ KHỞI ĐỘNG
    #  t=0s   DDS Agent + Startup info
    #  t=2s   RBFNN Controller (chờ DDS ổn định)
    #  t=2.5s Arm Dynamics
    #  t=3s   Arm Cmd Bridge + Joint State Bridge (sim)
    #  t=2s   Arm Initial Pose (sim) → tự chờ thêm 3s bên trong
    #  t=3.5s Telemetry Monitor
    #  t=4s   Data Logger
    # ═══════════════════════════════════════════════════════════

    delayed_backstepping  = TimerAction(period=2.0,  actions=[backstepping_node])
    delayed_arm_dynamics  = TimerAction(period=2.5,  actions=[arm_dynamics_node])
    delayed_arm_virtual   = TimerAction(period=2.7,  actions=[arm_virtual_state_node])
    delayed_arm_cmd       = TimerAction(period=3.0,  actions=[arm_cmd_node])
    delayed_arm_js_bridge = TimerAction(period=3.0,  actions=[arm_joint_state_bridge_node])
    delayed_arm_pose      = TimerAction(period=2.0,  actions=[arm_initial_pose_node])
    delayed_telemetry     = TimerAction(period=3.5,  actions=[telemetry_node])
    delayed_logger        = TimerAction(period=4.0,  actions=[data_logger_node])

    return LaunchDescription([
        # ── Arguments ──
        arg_sim,
        arg_config,
        arg_rbfnn,
        arg_handoff_mode,
        arg_allow_external_torque_handoff,
        arg_required_nav_states,
        arg_rbfnn_output_enable,
        arg_arm_ff_enable,
        arg_arm_virtual_disturbance_enable,
        arg_start_data_logger,
        arg_arm_state_source,
        arg_use_gazebo_arm_visual,
        arg_experiment_case,
        arg_experiment_output_root,
        arg_log_rate_hz,
        arg_start_xrce_agent,
        arg_xrce_serial_dev,
        arg_xrce_baud,
        # ── Hướng dẫn ──
        startup_info,
        # ── DDS Agent (khởi động ngay) ──
        xrce_hardware,
        xrce_sim,
        # ── Các controller node (khởi động có trễ) ──
        delayed_backstepping,
        delayed_arm_dynamics,
        delayed_arm_virtual,
        delayed_arm_cmd,
        delayed_arm_js_bridge,
        delayed_arm_pose,
        delayed_telemetry,
        delayed_logger,
        qgc_trigger_node,
    ])
