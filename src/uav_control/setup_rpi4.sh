#!/bin/bash
# ============================================================
#  SETUP SCRIPT: UAV ROS2 Stack trên Raspberry Pi 4
#  PX4 Pixhawk 6C + RPi4 + Ubuntu 22.04 + ROS2 Humble
# ============================================================
set -e
echo "🚀 Bắt đầu cài đặt UAV ROS2 Stack..."

# ── Bước 1: Cài ROS2 Humble ─────────────────────────────────
echo "📦 Bước 1: Cài đặt ROS2 Humble..."
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

sudo apt install -y software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop ros-dev-tools
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# ── Bước 2: Cài px4_msgs ────────────────────────────────────
echo "📦 Bước 2: Cài đặt px4_msgs..."
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/PX4/px4_msgs.git
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select px4_msgs
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc

# ── Bước 3: Cài micro-ROS ───────────────────────────────────
echo "📦 Bước 3: Cài đặt micro-ROS agent..."
cd ~/ros2_ws/src
git clone https://github.com/micro-ROS/micro_ros_setup.git
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select micro_ros_setup
source install/setup.bash

ros2 run micro_ros_setup create_agent_ws.sh
ros2 run micro_ros_setup build_agent.sh
source install/setup.bash

# Cài qua pip (nếu có binary):
# pip3 install micro-ros-agent  (tuỳ distro)

# ── Bước 4: Cài package điều khiển UAV ─────────────────────
echo "📦 Bước 4: Build px4_uav_control package..."
cd ~/ros2_ws/src
# Sao chép source code vào đây
cp -r /path/to/px4_uav_control .   # Thay bằng đường dẫn thực tế

cd ~/ros2_ws
colcon build --packages-select px4_uav_control
source install/setup.bash

# ── Bước 5: Cấu hình UART RPi4 ──────────────────────────────
echo "⚙️  Bước 5: Cấu hình UART..."
# Tắt serial console, bật UART hardware
sudo raspi-config nonint do_serial_cons 1   # Tắt login shell
sudo raspi-config nonint do_serial_hw 0     # Bật hardware UART

# Thêm user vào dialout group
sudo usermod -a -G dialout $USER

# ── Bước 6: Cấu hình PX4 Pixhawk 6C ────────────────────────
echo ""
echo "📋 BƯỚC CẦU HÌNH PX4 (thực hiện thủ công trong QGroundControl):"
echo "  1. MAV_0_CONFIG    = TELEM2"
echo "  2. SER_TEL2_BAUD   = 921600 8N1"
echo "  3. XRCE_DDS_CFG    = TELEM2"
echo "  4. UXRCE_DDS_DOM   = 0"
echo "  5. COM_RCL_EXCEPT  = 4    (tắt failsafe khi mất RC trong offboard)"
echo "  6. EKF2_HGT_REF    = Vision/GPS tuỳ setup"
echo ""

# ── Bước 7: Systemd service (tự khởi động) ──────────────────
echo "⚙️  Bước 7: Tạo systemd service..."
sudo tee /etc/systemd/system/uav_control.service > /dev/null <<EOF
[Unit]
Description=UAV ROS2 Control Stack
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER
Environment="HOME=/home/$USER"
ExecStart=/bin/bash -c 'source /opt/ros/humble/setup.bash && \
  source /home/$USER/ros2_ws/install/setup.bash && \
  ros2 launch px4_uav_control uav_bringup.launch.py'
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable uav_control.service
echo "✅ Service đã đăng ký. Khởi động: sudo systemctl start uav_control"

echo ""
echo "✅ Cài đặt hoàn tất!"
echo ""
echo "🚀 Chạy thủ công:"
echo "   source ~/ros2_ws/install/setup.bash"
echo "   ros2 launch px4_uav_control uav_bringup.launch.py"
echo ""
echo "🔧 Lệnh điều khiển UAV:"
echo "   ros2 service call /uav/arm_takeoff std_srvs/srv/Trigger"
echo "   ros2 service call /uav/land       std_srvs/srv/Trigger"
echo "   ros2 service call /uav/hold       std_srvs/srv/Trigger"
echo "   ros2 service call /uav/rtl        std_srvs/srv/Trigger"
echo "   ros2 topic echo  /uav/telemetry"
