#!/bin/bash
# auto_train_loop.sh
# ------------------
# Vòng lặp huẩn luyện: Tự động chạy cả PX4 và ROS2. Dọn dẹp sạch sẽ khi lỗi và lặp lại.

echo "==========================================="
echo "   AUTONOMOUS RBFNN TRAINING SUPERVISOR    "
echo "   (Chế độ Gộp Hệ Thống Toàn Diện)         "
echo "==========================================="

TRAIN_LOG="training_auto_restart.log"

while true; do
    echo "[$(date)] Bước 0: Quét dọn các tàn dư PX4/Gazebo cũ (nếu có)..." | tee -a $TRAIN_LOG
    killall -9 px4 2>/dev/null
    killall -9 ruby 2>/dev/null
    killall -9 MicroXRCEAgent 2>/dev/null
    killall -9 gzserver 2>/dev/null
    killall -9 gzclient 2>/dev/null
    sleep 2

    echo "[$(date)] Bước 1: Khởi động PX4 SITL & Gazebo..." | tee -a $TRAIN_LOG
    cd ~/PX4-Autopilot
    make px4_sitl gz_x500_hop &
    PX4_PID=$!
    
    echo "Đang đợi 20 giây để Gazebo nạp môi trường..."
    sleep 20

    echo "[$(date)] Bước 2: Khởi động ROS2 System Launch..." | tee -a $TRAIN_LOG
    cd ~/ros2_ws
    source install/setup.bash
    
    ros2 launch uam_controller uam_system.launch.py sim:=true &
    ROS_LAUNCH_PID=$!

    echo "Đang đợi kết nối ROS2 - PX4 (10s)..."
    sleep 10

    echo "========================================================"
    echo " BẮT ĐẦU CHƯƠNG TRÌNH SUPERVISOR ĐIỀU KHIỂN & ĐÁNH GIÁ "
    echo "========================================================"
    
    ros2 run uam_controller rbfnn_training_supervisor.py
    EXIT_CODE=$?

    echo "Đang dọn dẹp các tiến trình ROS2..."
    kill -SIGINT $ROS_LAUNCH_PID 2>/dev/null
    
    sleep 3
    
    # Force kill ROS 2 nodes
    kill -9 $ROS_LAUNCH_PID 2>/dev/null
    pkill -f "uam_backstepping_rbfnn_node" 2>/dev/null
    pkill -f "uam_mission_bridge" 2>/dev/null
    pkill -f "uam_telemetry_monitor" 2>/dev/null
    pkill -f "arm_dynamics_node" 2>/dev/null
    # pkill -f "lstm_predictive_node" 2>/dev/null
    pkill -f "arm_trajectory_generator" 2>/dev/null
    wait $ROS_LAUNCH_PID 2>/dev/null

    if [ $EXIT_CODE -eq 0 ]; then
        echo "==========================================="
        echo "🎉 HOÀN THÀNH TRAINING. KHÔNG PHÁT HIỆN LỖI. "
        echo "==========================================="
        break
    else
        echo "==========================================="
        echo "💥 PHÁT HIỆN CRASH MÔ PHỎNG (Exit Code: $EXIT_CODE). "
        echo "Tiến hành tắt PX4/Gazebo và chạy lại bài tập..."
        echo "==========================================="
        
        # Kill PX4 background process
        kill -SIGINT $PX4_PID 2>/dev/null
        sleep 2
        killall -9 px4 2>/dev/null
        killall -9 ruby 2>/dev/null
        killall -9 MicroXRCEAgent 2>/dev/null
        killall -9 gzserver 2>/dev/null
        killall -9 gzclient 2>/dev/null
        wait $PX4_PID 2>/dev/null
        
        echo "Đã dọn dẹp xong. Khởi động lại sau 3 giây..."
        sleep 3
    fi
done
