# UAM PX4 + ROS 2 Runbook

Tài liệu này tách rõ hai chế độ vận hành của hệ UAM quadrotor + tay máy 6 DoF:

- **Phần A: mô phỏng với QGC/Gazebo**: PX4 SITL, Gazebo, QGroundControl, virtual arm disturbance, logger và bảng kết quả.
- **Phần B: chạy thực tế với QGC/PX4**: PX4 thật, Raspberry Pi 4B, Micro XRCE-DDS serial, QGC điều khiển, external rate controller chỉ bật sau khi hover ổn định.

Kiến trúc điều khiển hiện tại:

```text
QGC / PX4 position controller
        -> PX4 attitude controller
        -> /fmu/out/vehicle_rates_setpoint
        -> ROS 2 uam_backstepping_rbfnn_node
        -> /fmu/in/vehicle_torque_setpoint
        -> /fmu/in/vehicle_thrust_setpoint
        -> PX4 external mc_rate_control branch
        -> PX4 control allocation
        -> motors
```

Khi `MC_RATE_EXT_EN=0`, PX4 dùng rate PID nội bộ. Khi `MC_RATE_EXT_EN=1`, PX4 nhận torque/thrust từ ROS 2 nếu dữ liệu còn mới; nếu ROS 2 ngừng publish/stale, PX4 fallback về rate PID nội bộ.

## Chuẩn Bị Chung

Mỗi terminal ROS 2:

```bash
source /opt/ros/humble/setup.bash
source /home/wicom/ros2_ws/install/setup.bash
```

Build lại sau khi sửa code:

```bash
cd /home/wicom/ros2_ws
colcon build --packages-select uam_controller
source /home/wicom/ros2_ws/install/setup.bash
```

File launch chính:

```text
/home/wicom/ros2_ws/src/uam_controller/launch/uam_qgc_mode.launch.py
```

Các tham số launch quan trọng:

| Tham số | Ý nghĩa |
| --- | --- |
| `sim:=true/false` | `true`: Gazebo SITL, `false`: phần cứng thật |
| `config_file:=...yaml` | YAML tham số controller/arm |
| `external_handoff_mode:=manual/auto` | `manual`: gọi service để bật external, `auto`: trigger tự bật sau hover ổn định |
| `rbfnn_output_enable:=true/false` | bật/tắt output RBFNN trong controller |
| `arm_ff_enable:=true/false` | bật/tắt feedforward torque từ mô hình cánh tay |
| `arm_virtual_disturbance_enable:=true/false` | chỉ dùng mô phỏng để tiêm nhiễu cánh tay ảo vào plant |
| `arm_state_source:=commanded/gazebo/external` | nguồn `/joint_states`: lệnh ảo, Gazebo joint, hoặc driver tay máy thật |
| `use_gazebo_arm_visual:=true/false` | có gửi lệnh để nhìn arm Gazebo chuyển động hay không |
| `start_xrce_agent:=true/false` | launch tự chạy MicroXRCEAgent hay agent đã chạy ngoài |
| `xrce_serial_dev:=...` | cổng serial Micro XRCE-DDS khi `sim:=false` |

YAML tốt hiện tại cho báo cáo Case B combined:

```bash
CASE_B_CONFIG=/home/wicom/uam_results/case_b_combined_ff_narrow_sweep_20260607_150103/configs/case_b_ff85_x030.yaml
```

Kết quả đẹp đang dùng cho báo cáo:

```text
Case B, combined_amp080, x030, trial 4
Verdict: GOOD
XY mean drift: 0.0395 m
XY max drift : 0.0810 m
Altitude RMSE: 0.0343 m
Rate err RMS : 0.1020 rad/s
```

---

# Phần A: Mô Phỏng Với QGC/Gazebo

## A1. Dọn Mô Phỏng Cũ

```bash
cd /home/wicom/PX4-Autopilot/Tools/simulation/gz
./tools/stop_uam_sim.sh
```

## A2. Terminal 1: Chạy PX4 SITL + Gazebo

```bash
cd /home/wicom/PX4-Autopilot
make px4_sitl gz_x500_hop
```

Nếu target `gz_x500_hop` không tồn tại sau khi clean build, kiểm tra lại model/airframe `x500_hop` trong PX4 tree trước khi chạy tiếp.

Trong PX4 shell, chọn chế độ rate controller:

```bash
# Baseline PX4 internal rate PID
param set MC_RATE_EXT_EN 0

# External ROS 2 backstepping/RBFNN rate controller
param set MC_RATE_EXT_EN 1
```

Kiểm tra:

```bash
param show MC_RATE_EXT_EN
```

## A3. Terminal 2: Chạy ROS 2 QGC Mode

### Baseline, chưa thay rate controller

```bash
source /opt/ros/humble/setup.bash
source /home/wicom/ros2_ws/install/setup.bash

ros2 launch uam_controller uam_qgc_mode.launch.py \
  sim:=true \
  enable_rbfnn:=true \
  external_handoff_mode:=manual \
  rbfnn_output_enable:=false \
  arm_ff_enable:=false \
  arm_virtual_disturbance_enable:=false \
  arm_state_source:=commanded \
  use_gazebo_arm_visual:=false
```

Với baseline thật sự, đặt `MC_RATE_EXT_EN=0` trong PX4 shell. ROS node có thể chạy để monitor/logger nhưng PX4 vẫn dùng PID nội bộ.

### Case B báo cáo: Backstepping + RBFNN + FF + combined arm

```bash
source /opt/ros/humble/setup.bash
source /home/wicom/ros2_ws/install/setup.bash

CASE_B_CONFIG=/home/wicom/uam_results/case_b_combined_ff_narrow_sweep_20260607_150103/configs/case_b_ff85_x030.yaml
REPORT_ROOT=/home/wicom/uam_results/manual_case_b_report_$(date +%Y%m%d_%H%M%S)

ros2 launch uam_controller uam_qgc_mode.launch.py \
  sim:=true \
  config_file:=$CASE_B_CONFIG \
  enable_rbfnn:=true \
  external_handoff_mode:=manual \
  rbfnn_output_enable:=true \
  arm_ff_enable:=true \
  arm_virtual_disturbance_enable:=true \
  arm_state_source:=commanded \
  use_gazebo_arm_visual:=false \
  experiment_case:=case_b_rbfnn_ff_combined_amp080 \
  experiment_output_root:=$REPORT_ROOT
```

Ý nghĩa cấu hình trên:

- `rbfnn_output_enable:=true`: dùng Backstepping + RBFNN.
- `arm_ff_enable:=true`: bật feedforward từ mô hình cánh tay.
- `arm_virtual_disturbance_enable:=true`: mô phỏng plant bị nhiễu bởi cánh tay ảo. Chỉ dùng trong mô phỏng.
- `arm_state_source:=commanded`: `/joint_states` được sinh từ lệnh khớp, không phụ thuộc arm Gazebo vật lý.
- `use_gazebo_arm_visual:=false`: không gửi lệnh sang arm Gazebo, tránh lỗi frame/joint vật lý.

## A4. Điều Khiển Trong QGroundControl

1. Mở QGroundControl.
2. Chờ PX4 kết nối.
3. Arm.
4. Takeoff lên khoảng `2 m`.
5. Giữ hover ổn định vài giây.
6. Nếu dùng `external_handoff_mode:=manual`, gọi service để bật external controller.

```bash
source /opt/ros/humble/setup.bash
source /home/wicom/ros2_ws/install/setup.bash

ros2 service call /uam/enable_external_controller std_srvs/srv/Trigger
```

Sau khi external đã bật, chạy trajectory cánh tay combined:

```bash
ros2 run uam_controller arm_trajectory_generator.py \
  --pattern combined \
  --duration 180 \
  --amplitude 0.08 \
  --rate 10
```

## A5. Chạy Kiểm Chứng Tự Động Case B Combined

Lệnh này tự chạy 7 repeat cho Case B combined:

```bash
source /opt/ros/humble/setup.bash
source /home/wicom/ros2_ws/install/setup.bash

cd /home/wicom/PX4-Autopilot/Tools/simulation/gz

CASE_B_CONFIG=/home/wicom/uam_results/case_b_combined_ff_narrow_sweep_20260607_150103/configs/case_b_ff85_x030.yaml

ros2 run uam_controller run_rbfnn_verification_suite.py \
  --config "$CASE_B_CONFIG" \
  --output-root /home/wicom/uam_results/case_b_combined_report_$(date +%Y%m%d_%H%M%S) \
  --cases B \
  --pattern combined \
  --duration-s 180 \
  --amplitude 0.08 \
  --rate-hz 10 \
  --repeats 7
```

Sau khi chạy xong, xem bảng median:

```bash
cat /home/wicom/uam_results/<run>/combined_amp080/rbfnn_ab_comparison_combined_amp080_median.md
```

Tiêu chí chọn run đẹp cho báo cáo:

- Verdict `GOOD` hoặc `OK`.
- `xy_max_m` càng nhỏ càng tốt.
- `angle_rms_deg` không vượt khoảng `1 deg` nhiều.
- `rate_err_rms_radps` thấp.
- `n_hat_norm_rms` khác 0 nhưng không drift tăng vô hạn.
- `arm_ff_max_nm`, `tau_residual_rms_nm`, `ff_disturbance_dot_mean` có giá trị hợp lý.

## A6. Tạo Bảng/Đồ Thị Kiểu ReviewPX4 Cho Báo Cáo

Script bảng chi tiết:

```bash
/home/wicom/uam_results/ve_bang_tu_csv.py
```

Chạy với trial mặc định đang dùng trong báo cáo:

```bash
python3 /home/wicom/uam_results/ve_bang_tu_csv.py
```

Chạy với CSV khác:

```bash
python3 /home/wicom/uam_results/ve_bang_tu_csv.py \
  --csv /path/to/flight_timeseries.csv
```

Output nằm cạnh log:

```text
review_tables/
  detailed_review_report.md
  tables/
    01_overview.png/csv
    02_flight_performance.png/csv
    03_controller_arm_rbfnn.png/csv
    04_rbfnn_detail.png/csv
    05_feedforward_detail.png/csv
    06_axis_statistics.png/csv
    07_joint_statistics.png/csv
    08_data_health.png/csv
  plots/
    rbfnn_vs_rate_error_by_axis.png
    ff_disturbance_residual_by_axis.png
    ...
```

Các bảng quan trọng cho báo cáo:

- `02_flight_performance`: lỗi vị trí, độ cao, attitude, rate.
- `04_rbfnn_detail`: tham số và output `n_hat` của RBFNN.
- `05_feedforward_detail`: FF, disturbance, residual, correlation, tỉ lệ FF/disturbance.
- `06_axis_statistics`: thống kê từng trục.
- `07_joint_statistics`: cử động từng khớp.

## A7. So Sánh A/B/C/D Khi Cần Chứng Minh Đóng Góp RBFNN

Các case:

```text
A: RBFNN off, FF on
B: RBFNN on,  FF on
C: RBFNN on,  FF off
D: RBFNN off, FF off
```

Chạy suite đầy đủ:

```bash
ros2 run uam_controller run_rbfnn_verification_suite.py \
  --config "$CASE_B_CONFIG" \
  --output-root /home/wicom/uam_results/rbfnn_abcd_$(date +%Y%m%d_%H%M%S) \
  --cases A,B,C,D \
  --pattern combined \
  --duration-s 180 \
  --amplitude 0.08 \
  --rate-hz 10 \
  --repeats 7
```

Đọc kết quả:

```bash
cat /home/wicom/uam_results/<suite>/combined_amp080/rbfnn_ab_comparison_combined_amp080_median.md
```

Ghi nhớ: mục tiêu báo cáo hiện tại là Case B combined đẹp và ổn định. A/B/C/D dùng để phân tích đóng góp, không nhất thiết là cấu hình bay cuối cùng.

## A8. Kiểm Tra Topic Trong Mô Phỏng

```bash
ros2 topic echo /joint_states --once
ros2 topic echo /arm/interaction_wrench --once
ros2 topic echo /fmu/out/vehicle_odometry --once
ros2 topic info /fmu/in/vehicle_torque_setpoint -v
ros2 topic info /fmu/in/vehicle_thrust_setpoint -v
ros2 topic echo /uam/debug_state --once
```

Nếu dùng `arm_state_source:=commanded`, `/joint_states` phải đổi theo lệnh từ `arm_trajectory_generator.py`.

---

# Phần B: Chạy Thực Tế Với QGC/PX4

Phần này dành cho Raspberry Pi 4B trên UAV thật kết nối PX4 thật qua Micro XRCE-DDS. QGroundControl vẫn là giao diện arm/takeoff/điều khiển chính.

## B1. Nguyên Tắc An Toàn

Trước khi bật external trên UAV thật:

1. Test không gắn cánh quạt.
2. Test QGC/PX4 baseline với `MC_RATE_EXT_EN=0`.
3. Test ROS 2 topic và Micro XRCE-DDS ổn định.
4. Test service `/uam/enable_external_controller` khi motor chưa chạy.
5. Chỉ bật external sau khi hover ổn định.
6. Luôn có RC/QGC kill switch và người giám sát.

Không dùng trên UAV thật:

```bash
arm_virtual_disturbance_enable:=true
```

`arm_virtual_disturbance_enable` là nhiễu ảo để mô phỏng plant bị cánh tay tác động. Trên UAV thật, nhiễu đã tồn tại vật lý, không được tiêm thêm vào torque setpoint.

## B2. Firmware/PX4 Cần Có

PX4 thật cần:

- uXRCE-DDS client bật và nối được với Raspberry Pi.
- Topic `/fmu/out/vehicle_odometry`, `/fmu/out/vehicle_rates_setpoint` publish sang ROS 2.
- Topic `/fmu/in/vehicle_torque_setpoint`, `/fmu/in/vehicle_thrust_setpoint` nhận từ ROS 2.
- Nhánh external rate-control `MC_RATE_EXT_EN`.

Kiểm tra/thay đổi tham số bằng QGC hoặc MAVLink shell:

```bash
param show MC_RATE_EXT_EN
param set MC_RATE_EXT_EN 0   # bay baseline PX4 internal
param set MC_RATE_EXT_EN 1   # cho phép external rate torque/thrust
```

Nếu PX4 không nhận ROS 2 setpoint, giữ `MC_RATE_EXT_EN=0` và kiểm tra XRCE trước.

## B3. Raspberry Pi: Source Và Kiểm Tra Serial

Trên Raspberry Pi:

```bash
source /opt/ros/humble/setup.bash
source /home/wicom/ros2_ws/install/setup.bash
```

Kiểm tra thiết bị serial:

```bash
ls -l /dev/ttyAMA0 /dev/ttyUSB0 /dev/ttyACM0
```

Thiết bị thường dùng:

```text
/dev/ttyAMA0  hoặc  /dev/ttyUSB0  hoặc  /dev/ttyACM0
```

Baudrate đang dùng mặc định:

```text
921600
```

## B4. Chạy ROS 2 Stack Cho UAV Thật

Trường hợp Pi tự chạy Micro XRCE-DDS Agent:

```bash
source /opt/ros/humble/setup.bash
source /home/wicom/ros2_ws/install/setup.bash

REAL_CONFIG=/home/wicom/ros2_ws/src/uam_controller/config/uam_controller_params.yaml
REAL_REPORT_ROOT=/home/wicom/uam_results/real_qgc_$(date +%Y%m%d_%H%M%S)

ros2 launch uam_controller uam_qgc_mode.launch.py \
  sim:=false \
  config_file:=$REAL_CONFIG \
  start_xrce_agent:=true \
  xrce_serial_dev:=/dev/ttyAMA0 \
  xrce_baud:=921600 \
  enable_rbfnn:=true \
  external_handoff_mode:=manual \
  rbfnn_output_enable:=true \
  arm_ff_enable:=true \
  arm_virtual_disturbance_enable:=false \
  arm_state_source:=external \
  use_gazebo_arm_visual:=false \
  experiment_case:=real_qgc_case_b \
  experiment_output_root:=$REAL_REPORT_ROOT
```

Nếu MicroXRCEAgent đã chạy bằng service/terminal khác:

```bash
ros2 launch uam_controller uam_qgc_mode.launch.py \
  sim:=false \
  config_file:=$REAL_CONFIG \
  start_xrce_agent:=false \
  enable_rbfnn:=true \
  external_handoff_mode:=manual \
  rbfnn_output_enable:=true \
  arm_ff_enable:=true \
  arm_virtual_disturbance_enable:=false \
  arm_state_source:=external \
  use_gazebo_arm_visual:=false \
  experiment_case:=real_qgc_case_b \
  experiment_output_root:=$REAL_REPORT_ROOT
```

Ý nghĩa riêng cho chạy thật:

- `sim:=false`: dùng serial Micro XRCE-DDS thay vì UDP SITL.
- `arm_virtual_disturbance_enable:=false`: không tiêm nhiễu ảo.
- `arm_state_source:=external`: không chạy virtual/gazebo joint-state node; driver tay máy thật phải publish `/joint_states`.
- `arm_ff_enable:=true`: dùng mô hình động lực học cánh tay để bù torque nếu `/joint_states` và `/arm/interaction_wrench` đúng.

Nếu chưa có driver tay máy thật publish `/joint_states`, có thể dùng tạm:

```bash
arm_state_source:=commanded
```

Khi đó hệ tính lực/moment theo lệnh khớp, không phải trạng thái đo thật. Chỉ dùng để thử logic, không nên dùng làm kết luận bay thật cuối cùng.

## B5. Kiểm Tra Topic Trước Khi Bay

Trước khi arm:

```bash
ros2 topic echo /fmu/out/vehicle_status_v1 --once
ros2 topic echo /fmu/out/vehicle_odometry --once
ros2 topic echo /fmu/out/vehicle_rates_setpoint --once
ros2 topic echo /joint_states --once
ros2 topic echo /arm/interaction_wrench --once
ros2 topic info /fmu/in/vehicle_torque_setpoint -v
ros2 topic info /fmu/in/vehicle_thrust_setpoint -v
```

Kỳ vọng:

- `/fmu/out/...` có dữ liệu mới.
- `/joint_states` đúng 6 khớp và không bị stale.
- `/arm/interaction_wrench` khác 0 khi tay máy chuyển động.
- `/fmu/in/vehicle_torque_setpoint` và `/fmu/in/vehicle_thrust_setpoint` có publisher từ controller.

## B6. Quy Trình Bay Thật Qua QGC

1. Đặt `MC_RATE_EXT_EN=0`.
2. Arm/takeoff bằng QGC, kiểm tra PX4 internal hover trước.
3. Land, kiểm tra log/topic nếu baseline chưa ổn.
4. Khi baseline ổn, đặt `MC_RATE_EXT_EN=1`.
5. Takeoff bằng QGC, hover khoảng `2 m`.
6. Chỉ sau khi hover ổn định, bật external:

```bash
ros2 service call /uam/enable_external_controller std_srvs/srv/Trigger
```

7. Cho tay máy chạy trajectory đơn giản, biên độ nhỏ trước.
8. Nếu attitude/XY/altitude xấu, tắt external hoặc land ngay.

## B7. Chạy Tay Máy Thật

Nếu tay máy thật đã có driver ROS 2, driver cần:

- nhận lệnh joint trajectory hoặc lệnh riêng của arm controller,
- publish `/joint_states` với 6 khớp,
- đồng bộ tên/thứ tự khớp với mô hình động lực học,
- đảm bảo timestamp/frequency đủ mới cho `arm_dynamics_node.py`.

Kiểm tra `/joint_states`:

```bash
ros2 topic hz /joint_states
ros2 topic echo /joint_states --once
```

Nếu cần chạy lệnh cánh tay dạng combined trong ROS 2:

```bash
ros2 run uam_controller arm_trajectory_generator.py \
  --pattern combined \
  --duration 180 \
  --amplitude 0.03 \
  --rate 10
```

Với UAV thật, bắt đầu từ biên độ nhỏ hơn mô phỏng, ví dụ `0.02-0.03 rad`, rồi tăng dần sau khi log ổn.

## B8. Log Và Bảng Kết Quả Chạy Thật

Logger tạo:

```text
$REAL_REPORT_ROOT/<timestamp>_<case>/
  flight_timeseries.csv
  metadata.json
  summary.json
  summary.md
```

Tạo bảng chi tiết:

```bash
python3 /home/wicom/uam_results/ve_bang_tu_csv.py \
  --csv /path/to/real_flight/flight_timeseries.csv
```

Các bảng cần xem:

- `02_flight_performance`: UAV có giữ altitude/XY/attitude không.
- `04_rbfnn_detail`: output `n_hat` có hợp lý không.
- `05_feedforward_detail`: FF có cùng chiều disturbance không, residual có giảm không.
- `08_data_health`: topic có stale hay mất dữ liệu không.

## B9. Debug Nhanh Khi Chạy Thật

Nếu QGC/PX4 không thấy ROS 2:

```bash
pgrep -af MicroXRCEAgent
ros2 topic list | sort
```

Nếu external không tác dụng:

```bash
param show MC_RATE_EXT_EN
ros2 topic echo /uam/debug_state --once
ros2 topic info /fmu/in/vehicle_torque_setpoint -v
```

Nếu tay máy không tạo lực/moment:

```bash
ros2 topic echo /joint_states --once
ros2 topic echo /arm/interaction_wrench --once
```

Nếu dữ liệu bị stale:

```bash
ros2 topic hz /fmu/out/vehicle_odometry
ros2 topic hz /fmu/out/vehicle_rates_setpoint
ros2 topic hz /joint_states
```

## B10. Cấu Hình Khuyến Nghị Khi Bảo Vệ

Mô phỏng báo cáo:

```bash
config_file:=/home/wicom/uam_results/case_b_combined_ff_narrow_sweep_20260607_150103/configs/case_b_ff85_x030.yaml
rbfnn_output_enable:=true
arm_ff_enable:=true
arm_virtual_disturbance_enable:=true
arm_state_source:=commanded
use_gazebo_arm_visual:=false
```

Chạy thật ban đầu:

```bash
config_file:=/home/wicom/ros2_ws/src/uam_controller/config/uam_controller_params.yaml
rbfnn_output_enable:=true
arm_ff_enable:=true
arm_virtual_disturbance_enable:=false
arm_state_source:=external
use_gazebo_arm_visual:=false
external_handoff_mode:=manual
```

Không dùng kết quả mô phỏng virtual disturbance để khẳng định trực tiếp bay thật. Với bay thật, phải dựa trên log thật: `/joint_states`, `/arm/interaction_wrench`, attitude/rate/XY/altitude và fallback behavior của PX4.
