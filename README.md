Nếu `/fmu/out/...` đã có dữ liệu ổn định, đi theo thứ tự này. Đừng bật external controller ngay.
# Terminal 1: QGC/MAVLink qua USB
mavlink-routerd /dev/ttyACM0:2000000 -e IP_LAPTOP_QGC:14550
(ip a để lấy IP_LAPTOP_QGC)

# Terminal 2: XRCE-DDS qua TELEM2
sudo MicroXRCEAgent serial --dev /dev/ttyS0 -b 921600

(MTF 01P - TELEM3 - SER_TEL3_BAUD:115200)
**1. Chạy baseline PX4 internal**
Terminal Pi:
# Terminal 3: ROS 2 nodes, không start lại Agent

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_nn/install/setup.bash

ros2 launch uam_controller uam_qgc_mode.launch.py \
  sim:=false \
  start_xrce_agent:=false \
  enable_rbfnn:=false \
  start_data_logger:=false
```


**2. Kiểm tra telemetry trước khi arm**
Terminal khác:

```bash
  source ~/ros2_nn/install/setup.bash

  ros2 topic echo /fmu/out/vehicle_status_v1 --once
  ros2 topic echo /fmu/out/vehicle_odometry --once
  ros2 topic echo /fmu/out/vehicle_land_detected --once
```

Kiểm tra thêm QGC:

- QGC nhận vehicle.
- GPS/estimator/local position OK nếu bay ngoài trời hoặc flow/VIO/mocap nếu dùng trong nhà.ssh 
- Battery OK.
- RC/manual mode hoạt động.
- Kill switch/failsafe đã test.
- Propeller đúng chiều, frame đúng, motor test đúng thứ tự.

**3. Arm và hover bằng PX4 internal**
Trên QGC:

1. Chọn mode ổn định như `Position` hoặc `Altitude`, tùy sensor bạn có.
2. Arm.
3. Takeoff thấp trước, khoảng `0.5 m`.
4. Nếu ổn, hover `1.5-2 m`.
5. Không bật `/uam/enable_external_controller` ở bước này.

Mục tiêu: xác nhận firmware + frame + sensor + motor + QGC bay ổn với controller PX4 gốc.

**4. Theo dõi trong lúc hover**
Terminal Pi:

```bash
ros2 topic echo /uam/telemetry
```

Hoặc kiểm tra input/output PX4:

```bash
ros2 topic hz /fmu/out/vehicle_odometry
ros2 topic hz /fmu/out/vehicle_rates_setpoint
```

Nếu `/fmu/out/vehicle_rates_setpoint` có dữ liệu khi bay, ROS đã nhận được setpoint mà external rate controller sẽ bám sau này.

**5. Hạ cánh và disarm**
Dùng QGC Land hoặc điều khiển tay hạ xuống. Sau đó kiểm tra không có node crash.

**6. Chạy external ở mức an toàn nhất**
Chỉ sau khi baseline hover ổn:

```bash
ros2 launch uam_controller uam_qgc_mode.launch.py \
  sim:=false \
  start_xrce_agent:=true \
  xrce_serial_dev:=/dev/ttyAMA0 \
  xrce_baud:=921600 \
  enable_rbfnn:=true \
  rbfnn_output_enable:=false \
  external_handoff_mode:=manual \
  start_data_logger:=true \
  experiment_output_root:=/home/piros2/uam_verification_logs
```

Trên PX4 cần đảm bảo param custom đã bật cơ chế bypass, ví dụ:

```text
MC_RATE_EXT_EN = 1
```

Nếu param này chưa bật, ROS node có chạy nhưng PX4 vẫn dùng internal rate controller.

**7. Handoff thủ công**
Sau khi UAV đã hover ổn định `1.8-2 m`, gọi:

```bash
ros2 service call /uam/enable_external_controller std_srvs/srv/Trigger
```

Ban đầu dùng:

```text
rbfnn_output_enable:=false
```

tức là **Backstepping-only**, chưa cho RBFNN tác động. Khi Backstepping-only ổn mới thử:

```text
rbfnn_output_enable:=true
```

**8. Trình tự tăng rủi ro**
Đi theo cấp này:

1. PX4 internal hover, arm fixed.
2. External Backstepping-only, arm fixed.
3. External Backstepping + RBFNN, arm fixed.
4. External Backstepping + RBFNN, arm chuyển động nhẹ.
5. Tăng biên độ/chuyển động tay máy.

Không nhảy thẳng tới RBFNN + arm chuyển động.