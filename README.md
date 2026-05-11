# UAM PX4 + Gazebo Simulation Runbook

Tài liệu này tổng hợp luồng chạy end-to-end cho đề tài UAM quadrotor + robot arm 6 DoF:

- PX4 SITL + Gazebo
- QGroundControl takeoff / hover
- chuyển sang external backstepping + RBFNN
- kích hoạt cánh tay để tạo nhiễu
- ghi log, phân tích kết quả, và tuning tham số

## 1. Cấu trúc liên quan

- `models/x500_hop/model.sdf`: mô hình UAV + arm + plugin Gazebo
- `tools/`: các script chạy mô phỏng, bridge topic, logger, analyzer
- `~/ros2_ws/src/uam_controller`: launch file, controller node, arm node, logger node

## 2. Chuẩn bị môi trường

Mỗi terminal ROS2 nên source đúng môi trường:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

Build lại khi có thay đổi code:

```bash
cd ~/ros2_ws
colcon build --packages-select uam_controller

cd ~/PX4-Autopilot
make px4_sitl gz_x500_hop
```

Khi cần bắt đầu một phiên mô phỏng sạch:

```bash
cd ~/PX4-Autopilot/Tools/simulation/gz
./tools/stop_uam_sim.sh
```

## 3. Luồng chạy nhanh nhất

### 3.1 Baseline hover với PX4 internal controller

Đây là bước kiểm tra UAV có thể takeoff và hold ở khoảng 2 m trước khi bật external control.

```bash
cd ~/PX4-Autopilot/Tools/simulation/gz
./tools/run_qgc_baseline.py --controller-mode baseline
```

Hoặc dùng wrapper shell:

```bash
./tools/run_uam_qgc.sh baseline
```

Luồng này sẽ:

- cleanup các tiến trình cũ
- chạy `make px4_sitl gz_x500_hop`
- chạy `ros2 launch uam_controller uam_qgc_mode.launch.py sim:=true enable_rbfnn:=false`
- tự khởi động bridge `gz_joint_state_bridge.py` để nối `/model/x500_hop_0/joint_state` sang `/joint_states`

Thao tác bay:

1. Mở QGroundControl.
2. Arm.
3. Takeoff lên khoảng 2 m.
4. Giữ hover ổn định với PX4 internal controller.

## 4. Chuyển sang external backstepping + RBFNN

Sau khi baseline hover ổn định, bật external mode bằng helper:

```bash
cd ~/PX4-Autopilot/Tools/simulation/gz
./tools/run_qgc_baseline.py --controller-mode external --rbfnn-output-enable true
```

Hoặc:

```bash
./tools/run_uam_qgc.sh auto
```

Các chế độ quan trọng:

- `--controller-mode external`: bật `MC_RATE_EXT_EN=1`
- `--rbfnn-output-enable false`: chỉ backstepping, chưa bật output RBFNN
- `--rbfnn-output-enable true`: backstepping + RBFNN
- `--external-handoff-mode manual|auto`: chọn chuyển tay hay tự chuyển sau hover ổn định
- `--profile pitch_damped`: profile PX4 an toàn hơn khi takeoff còn rung hoặc pitch chưa ổn

Nếu chạy manual, khi UAV đã hover ổn định, có thể gọi:

```bash
source ~/ros2_ws/install/setup.bash
ros2 service call /uam/enable_external_controller std_srvs/srv/Trigger
```

Ghi nhớ:

- external controller chỉ nên bật sau khi hover ổn định
- nếu node ROS chết hoặc topic stale, PX4 sẽ fallback về internal control
- file launch `uam_qgc_mode.launch.py` là nhánh QGC, không dùng `uam_mission_bridge`

## 5. Kích hoạt cánh tay và tuning RBFNN

Để test disturbance rejection của backstepping + RBFNN:

### 5.1 Chạy case kiểm chứng có log

```bash
cd ~/PX4-Autopilot/Tools/simulation/gz
./tools/run_uam_verification.sh internal_hover
./tools/run_uam_verification.sh external_bs_hover
./tools/run_uam_verification.sh external_rbfnn_hover
./tools/run_uam_verification.sh external_rbfnn_arm_sin02
./tools/run_uam_verification.sh external_rbfnn_arm_combined03
```

Các case này lần lượt dùng để:

- `internal_hover`: baseline PX4
- `external_bs_hover`: external backstepping בלבד, chưa bật RBFNN
- `external_rbfnn_hover`: external backstepping + RBFNN, arm fixed
- `external_rbfnn_arm_sin02`: arm sin nhẹ với biên độ 0.2 rad
- `external_rbfnn_arm_combined03`: arm excitation tổng hợp, biên độ 0.3 rad

### 5.2 Ghi chú về handoff

`run_uam_verification.sh` mặc định chờ tay. Sau khi hover ổn định 3 giây, bạn có thể:

```bash
source ~/ros2_ws/install/setup.bash
ros2 service call /uam/enable_external_controller std_srvs/srv/Trigger
```

Các biến môi trường hữu ích:

- `HANDOFF_MODE=manual|auto`
- `HANDOFF_SETTLE_S=3`
- `AUTO_ARM_AFTER_S=<seconds>`
- `RESULTS_ROOT=/path/to/output`
- `LOG_RATE_HZ=20.0`

### 5.3 Kích arm để tạo nhiễu

Trong giai đoạn tuning RBFNN, nên bắt đầu với arm fixed, rồi mới cho arm chuyển động nhẹ.

Ví dụ:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run uam_controller arm_trajectory_generator.py --pattern sin --duration 120 --amplitude 0.2 --rate 10
```

Hoặc:

```bash
ros2 run uam_controller arm_trajectory_generator.py --pattern combined --duration 300 --amplitude 0.3 --rate 10
```

Lưu ý:

- `arm_initial_pose.py` sẽ gập arm về tư thế an toàn khi sim khởi động
- `arm_gazebo_command_node.py` đẩy lệnh khớp sang Gazebo
- `rbfnn_data_logger.py` ghi log đủ để dùng cho báo cáo và phân tích

## 6. Tuning PID baseline trước khi bật external

Nếu cần chỉnh lại PX4 baseline trước khi qua external control:

```bash
cd ~/PX4-Autopilot/Tools/simulation/gz
python3 tools/px4_pid_slider_tuner.py --autostart
```

Tuner này:

- mở UI slider cho các tham số PX4
- cho phép arm / disarm trực tiếp
- lưu preset JSON trong `Tools/simulation/gz/pid_search_results/pid_slider_presets/`
- copy được các dòng `param set-default ...`

Tuning tự động có thể dùng:

```bash
python3 tools/px4_slider_autotune.py --trials 20 --stop-when-good
```

Khuyến nghị cho baseline:

- tune trước khi bật external
- ưu tiên ổn định hover 2 m
- nếu đã có preset tốt, chỉ cần nạp lại preset đó trước khi chạy external experiments
- nếu takeoff còn rung, thử thêm `--profile pitch_damped` ở `run_qgc_baseline.py` hoặc wrapper `run_uam_qgc.sh`

## 7. Ghi log và phân tích kết quả

Logger của thí nghiệm lưu dữ liệu dưới:

```text
Tools/simulation/gz/pid_search_results/uam_verification/<timestamp>_<case>/
```

Trong thư mục đó thường có:

- `flight_timeseries.csv`
- `metadata.json`
- `summary.json`
- `summary.md`

Để so sánh log PX4 native và log RBFNN:

```bash
python3 tools/px4_uam_log_analyzer.py \
  <px4_native_log.ulg> \
  <rbfnn_custom_log.ulg> \
  --output-dir px4_uam_log_analysis \
  --nn-topic uam_debug \
  --nn-fields n_hat[0],n_hat[1],n_hat[2]
```

Script analyzer chấp nhận cả `.ulg/.ulog` lẫn `.csv`.

Kết quả sinh ra:

- đồ thị position response
- đồ thị attitude response
- đồ thị output RBFNN
- bảng RMSE / max error trong `tracking_metrics.md`

## 8. Kiểm tra topic quan trọng

Các topic nên có trước khi tin rằng luồng external đã đúng:

```bash
ros2 topic echo /joint_states --once
ros2 topic info /fmu/in/vehicle_torque_setpoint -v
ros2 topic info /fmu/in/vehicle_thrust_setpoint -v
ros2 topic echo /fmu/out/vehicle_status_v1 --once
ros2 topic echo /fmu/out/vehicle_odometry --once
```

Nếu `/joint_states` không thay đổi khi arm cử động, hãy kiểm tra bridge `gz_joint_state_bridge.py`.

## 9. Cleanup và chạy lại

Khi muốn dừng sạch toàn bộ PX4 / Gazebo / ROS2:

```bash
cd ~/PX4-Autopilot/Tools/simulation/gz
./tools/stop_uam_sim.sh
```

Script này sẽ:

- gửi `SIGTERM` trước
- chờ một khoảng thời gian
- chỉ dùng `SIGKILL` nếu tiến trình vẫn còn sống

## 10. Gợi ý thứ tự cho đồ án

1. Tune PX4 baseline bằng `px4_pid_slider_tuner.py`.
2. Chạy `internal_hover` để xác nhận UAV giữ được 2 m.
3. Chạy `external_bs_hover` để xác nhận backstepping thay rate controller.
4. Chạy `external_rbfnn_hover` với arm fixed.
5. Chạy `external_rbfnn_arm_sin02` để bắt đầu tune RBFNN với disturbance nhẹ.
6. Dùng `px4_uam_log_analyzer.py` để lấy số liệu và hình cho báo cáo.
