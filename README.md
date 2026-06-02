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

Mặc định launch hiện dùng `arm_state_source:=commanded`: cánh tay được mô phỏng
ở mức động học để tính nội lực, không phụ thuộc joint vật lý trong Gazebo.
Luồng dùng cho tính toán lực/moment:

```text
/arm_controller/joint_trajectory_plan
        ↓
arm_virtual_state_node.py
        ↓
/joint_states
        ↓
arm_dynamics_node.py
        ↓
/arm/interaction_wrench
```

Chế độ này phù hợp khi chỉ cần lực/moment do chuyển động cánh tay tác dụng lên
UAV, chưa cần va chạm/contact hay mô phỏng joint vật lý thật trong Gazebo.

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

- `arm_virtual_state_node.py` tạo `/joint_states` từ lệnh khớp và có giới hạn vận tốc/gia tốc.
- `arm_dynamics_node.py` dùng `/joint_states` để tính `/arm/interaction_wrench`.
- `rbfnn_data_logger.py` ghi log đủ để dùng cho báo cáo và phân tích.
- Nếu vẫn muốn thử điều khiển joint vật lý trong Gazebo, chạy launch với
  `arm_state_source:=gazebo use_gazebo_arm_visual:=true`.

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

### 7.1 So sánh A/B/C để chứng minh đóng góp RBFNN

Controller hiện dùng input RBFNN 21 chiều:

```text
[omega(3), e_omega(3), q_arm(6), dq_arm(6), tau_residual(3)]
```

Trong đó `tau_residual` xấp xỉ phần nhiễu cánh tay còn lại sau nhánh
`arm_ff`. Vì vậy các YAML cũ vẫn chạy được, nhưng nên tune lại vì
`rbfnn_gaussian_width` cũ quá nhỏ cho input nhiều chiều.

Sau khi có YAML tham số tốt, chạy 3 ca có cùng trajectory cánh tay và cùng
điều kiện bay. Điểm khác biệt duy nhất là nhánh RBFNN và feedforward cánh tay.

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

REPORT_ROOT=/home/wicom/uam_results/rbfnn_ab_report_$(date +%Y%m%d_%H%M%S)
BEST_CONFIG=/home/wicom/uam_results/<run>/final_best_uam_controller_params.yaml
```

Case A: tắt output RBFNN, giữ `arm_ff`.

```bash
ros2 launch uam_controller uam_qgc_mode.launch.py sim:=true \
  config_file:=$BEST_CONFIG \
  rbfnn_output_enable:=false \
  arm_ff_enable:=true \
  arm_virtual_disturbance_enable:=true \
  arm_state_source:=commanded \
  use_gazebo_arm_visual:=false \
  experiment_case:=case_a_no_rbfnn_ff \
  experiment_output_root:=$REPORT_ROOT
```

Case B: bật RBFNN và giữ `arm_ff`.

```bash
ros2 launch uam_controller uam_qgc_mode.launch.py sim:=true \
  config_file:=$BEST_CONFIG \
  rbfnn_output_enable:=true \
  arm_ff_enable:=true \
  arm_virtual_disturbance_enable:=true \
  arm_state_source:=commanded \
  use_gazebo_arm_visual:=false \
  experiment_case:=case_b_rbfnn_ff \
  experiment_output_root:=$REPORT_ROOT
```

Case C: bật RBFNN, tắt `arm_ff` để xem RBFNN có tự học residual hay không.

```bash
ros2 launch uam_controller uam_qgc_mode.launch.py sim:=true \
  config_file:=$BEST_CONFIG \
  rbfnn_output_enable:=true \
  arm_ff_enable:=false \
  arm_virtual_disturbance_enable:=true \
  arm_state_source:=commanded \
  use_gazebo_arm_visual:=false \
  experiment_case:=case_c_rbfnn_no_ff \
  experiment_output_root:=$REPORT_ROOT
```

Trong mỗi case, sau khi UAV hover và external controller đã bật, chạy cùng một
trajectory cánh tay:

```bash
ros2 run uam_controller arm_trajectory_generator.py \
  --pattern slow_step --duration 120 --amplitude 0.05 --rate 5

ros2 run uam_controller arm_trajectory_generator.py \
  --pattern combined --duration 180 --amplitude 0.08 --rate 10
```

Sau khi chạy đủ 3 case, tạo bảng so sánh:

```bash
ros2 run uam_controller rbfnn_compare_ab_results.py --root "$REPORT_ROOT"
```

Hoặc chạy tự động cả 3 case A/B/C bằng YAML best:

```bash
source /opt/ros/humble/setup.bash
source /home/wicom/ros2_ws/install/setup.bash

ros2 run uam_controller run_rbfnn_verification_suite.py \
  --config /home/wicom/uam_results/<run>/final_best_uam_controller_params.yaml \
  --output-root /home/wicom/uam_results/rbfnn_verification_suite_$(date +%Y%m%d_%H%M%S) \
  --pattern slow_step \
  --duration-s 120 \
  --amplitude 0.05 \
  --rate-hz 5 \
  --repeats 3 \
  --include-strong
```

Script này dùng đúng YAML đã tìm được ở chế độ `--fixed-config`, tự chạy:

```text
Case A: RBFNN off, arm_ff on,  virtual disturbance on
Case B: RBFNN on,  arm_ff on,  virtual disturbance on
Case C: RBFNN on,  arm_ff off, virtual disturbance on
```

Mặc định nên dùng `slow_step amp=0.05` và `combined amp=0.08`. Không dùng
`slow_step amp=0.25` làm case chính nếu log báo `joint_pos_span` lớn bất
thường, vì khi đó dữ liệu joint state đã trôi khỏi lệnh điều khiển.

Với `--repeats 3`, script tạo thêm bảng median trong từng thư mục trajectory:

```text
rbfnn_ab_comparison_<pattern>_ampXXX_median.csv
rbfnn_ab_comparison_<pattern>_ampXXX_median.md
```

Kết quả cần xem chính:

- `rate_err_rms_radps`: Case B phải giảm so với Case A.
- `n_hat_norm_rms/max`: không được gần 0 toàn bộ thời gian và không tăng vô hạn.
- `angle_rms_deg`, `xy_mean_m`: Case B không được xấu hơn Case A đáng kể.
- `tau_residual_rms_nm`: Case B không nên lớn hơn Case A.
- `ff_disturbance_dot_mean`: nếu âm thì `arm_ff` nhiều khả năng sai dấu và cần
  sửa trước khi kết luận về RBFNN.
- Case C giúp kiểm tra `arm_ff` có đang bù quá mạnh hoặc sai scale/sign không.

### 7.2 Quy trình hiện tại để lấy kết quả tốt nhất

Kết quả kiểm chứng gần nhất cho thấy YAML conservative đã tốt hơn YAML tuning
ban đầu. Với `slow_step_amp080`, Case B cải thiện rõ so với Case A; với
`combined_amp080`, Case B cải thiện XY nhưng rate/attitude chưa đủ vượt gate.
Vì vậy chưa nên chốt báo cáo chỉ bằng một run `--repeats 3`. Quy trình nên làm
tiếp là chạy lại nhiều lần hơn, sau đó thêm bài test sai lệch mô hình
`arm_ff` để làm rõ đóng góp của RBFNN.

YAML candidate hiện tại:

```bash
BEST_CONFIG=/home/wicom/uam_results/rbfnn_best_param_search_fffix_20260529_181119/yaml_conservation.yaml
```

Không dùng YAML cũ trước khi sửa dấu `arm_ff`, ví dụ các run có
`arm_ff_scale_pitch` âm trong khi `arm_virtual_disturbance_scale_pitch` dương.
Những YAML đó làm feedforward khuếch đại nhiễu và làm kết luận về RBFNN sai.

Chạy lại kiểm chứng chính với nhiều repeat hơn:

```bash
cd /home/wicom/PX4-Autopilot/Tools/simulation/gz
./tools/stop_uam_sim.sh

source /opt/ros/humble/setup.bash
source /home/wicom/ros2_ws/install/setup.bash

ros2 run uam_controller run_rbfnn_verification_suite.py \
  --config "$BEST_CONFIG" \
  --output-root /home/wicom/uam_results/rbfnn_verification_suite_conservative_$(date +%Y%m%d_%H%M%S) \
  --pattern slow_step \
  --duration-s 300 \
  --amplitude 0.08 \
  --rate-hz 5 \
  --repeats 5 \
  --include-strong
```

Nếu máy đủ ổn định, dùng `--repeats 7` để lấy median đáng tin hơn. Không nên
đổi YAML giữa các case A/B/C trong cùng một suite.

Sau khi chạy xong, đọc hai file median:

```bash
cat /home/wicom/uam_results/<suite>/slow_step_amp080/rbfnn_ab_comparison_slow_step_amp080_median.md
cat /home/wicom/uam_results/<suite>/combined_amp080/rbfnn_ab_comparison_combined_amp080_median.md
```

Tiêu chí chọn kết quả tốt để đưa vào báo cáo:

- Case B có số `OK/GOOD` nhiều hơn hoặc ít nhất không kém Case A.
- `median_rate_err_rms_radps` của B nhỏ hơn A, đặc biệt ở `slow_step_amp080`.
- `median_xy_mean_m` và `median_xy_max_m` của B không xấu hơn A; nếu tốt hơn
  rõ thì ghi nhận là RBFNN giúp giảm drift.
- `median_angle_rms_deg` của B không tăng đáng kể so với A.
- `median_n_hat_norm_rms` khác 0 nhưng nhỏ, không có dấu hiệu tăng vô hạn.
- `median_ff_disturbance_dot_mean` dương; nếu âm thì không dùng run đó.

Nếu B vẫn tệ hơn A, giảm tác động RBFNN thêm một mức rồi chạy lại:

```text
rbfnn_output_gain: 0.20 -> 0.30
rbfnn_lr:          0.0015 -> 0.0025
rbfnn_e_modification: 0.05 -> 0.08
```

Không tăng `rbfnn_lr` khi B đang gây drift XY, vì lúc đó mạng đang học nhiễu
hoặc tạo bias chứ không bù residual sạch.

### 7.3 Bài test sai lệch mô hình để chứng minh vai trò RBFNN

Khi `arm_ff` quá khớp với nhiễu ảo, residual còn rất nhỏ nên RBFNN không có
nhiều việc để làm. Để chứng minh RBFNN có đóng góp trong báo cáo, nên chạy thêm
case model-mismatch: giữ nhiễu ảo thật như cũ nhưng cố tình giảm mô hình
feedforward còn khoảng 70%.

Tạo YAML mismatch từ YAML conservative:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml

src = Path("/home/wicom/uam_results/rbfnn_best_param_search_fffix_20260529_181119/yaml_conservation.yaml")
dst = src.with_name("yaml_conservation_ff70.yaml")

data = yaml.safe_load(src.read_text())
params = data["uam_backstepping_rbfnn_node"]["ros__parameters"]

for key in ("arm_ff_scale_roll", "arm_ff_scale_pitch", "arm_ff_scale_yaw"):
    params[key] = float(params[key]) * 0.70

dst.write_text(yaml.safe_dump(data, sort_keys=False))
print(dst)
PY
```

Chạy verification lại bằng YAML mismatch:

```bash
MISMATCH_CONFIG=/home/wicom/uam_results/rbfnn_best_param_search_fffix_20260529_181119/yaml_conservation_ff70.yaml

ros2 run uam_controller run_rbfnn_verification_suite.py \
  --config "$MISMATCH_CONFIG" \
  --output-root /home/wicom/uam_results/rbfnn_verification_suite_ff70_$(date +%Y%m%d_%H%M%S) \
  --pattern slow_step \
  --duration-s 300 \
  --amplitude 0.08 \
  --rate-hz 5 \
  --repeats 5 \
  --include-strong
```

Kết quả mong muốn trong mismatch:

- Case A xấu hơn do `arm_ff` không còn bù đủ.
- Case B tốt hơn A về `rate_err_rms_radps`, `angle_rms_deg` hoặc XY.
- `n_hat_norm_rms` của B lớn hơn run conservative gốc nhưng vẫn không drift.
- Case C có thể tốt ở vài chỉ số, nhưng nếu thiếu `arm_ff` mà vẫn fail nhiều
  thì kết luận đúng là RBFNN nên học residual chứ không thay toàn bộ mô hình.

Đây là bài test hữu ích cho báo cáo vì nó trả lời câu hỏi: khi mô hình cánh tay
không hoàn hảo, RBFNN có học phần sai lệch còn lại hay không.

## 8. Kiểm tra topic quan trọng

Các topic nên có trước khi tin rằng luồng external đã đúng:

```bash
ros2 topic echo /joint_states --once
ros2 topic info /fmu/in/vehicle_torque_setpoint -v
ros2 topic info /fmu/in/vehicle_thrust_setpoint -v
ros2 topic echo /fmu/out/vehicle_status_v1 --once
ros2 topic echo /fmu/out/vehicle_odometry --once
```

Nếu chạy chế độ mặc định `arm_state_source:=commanded`, `/joint_states` phải đổi
theo lệnh `/arm_controller/joint_trajectory_plan` dù Gazebo arm không cử động.
Nếu chạy `arm_state_source:=gazebo`, hãy kiểm tra bridge `arm_gazebo_joint_state_bridge.py`.

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

Cách chạy thủ công từng bước:

**1. Dọn tiến trình cũ**

```bash
cd /home/wicom/PX4-Autopilot/Tools/simulation/gz
./tools/stop_uam_sim.sh
```

**2. Build/source ROS2 nếu vừa sửa code**

```bash
cd /home/wicom/ros2_ws
colcon build --packages-select uam_controller
source /opt/ros/humble/setup.bash
source /home/wicom/ros2_ws/install/setup.bash
```

**3. Terminal 1: chạy PX4 + Gazebo**

```bash
cd /home/wicom/PX4-Autopilot
make px4_sitl gz_x500_hop
```

Chờ Gazebo mở và PX4 boot xong. Trong PX4 shell, nếu muốn chạy external RBFNN rate controller:

```bash
param set MC_RATE_EXT_EN 1
param set COM_RC_IN_MODE 4
```

Kiểm tra:

```bash
param show MC_RATE_EXT_EN
```

Nếu chỉ muốn baseline PX4 PID nội bộ thì để:

```bash
param set MC_RATE_EXT_EN 0
```

**4. Terminal 2: chạy ROS2 controller stack**

```bash
source /opt/ros/humble/setup.bash
source /home/wicom/ros2_ws/install/setup.bash

ros2 launch uam_controller uam_qgc_mode.launch.py \
  sim:=true \
  enable_rbfnn:=true \
  external_handoff_mode:=manual \
  rbfnn_output_enable:=true \
  arm_ff_enable:=true \
  arm_virtual_disturbance_enable:=false \
  arm_state_source:=commanded \
  use_gazebo_arm_visual:=false
```

Ý nghĩa:
- `enable_rbfnn:=true`: chạy node `uam_backstepping_rbfnn_node`.
- `rbfnn_output_enable:=true`: dùng Backstepping + RBFNN.
- `arm_ff_enable:=true`: dùng `/arm/interaction_wrench` làm torque bù trong external controller.
- `arm_virtual_disturbance_enable:=false`: không tiêm nhiễu cánh tay ảo vào UAV; dùng để kiểm tra controller với cấu hình bình thường.
- `external_handoff_mode:=manual`: chưa tự bật external, phải gọi service sau khi UAV hover ổn định.
- `arm_state_source:=commanded`: dùng cánh tay động học logic để tính nội lực, không dùng joint vật lý Gazebo.
- `use_gazebo_arm_visual:=false`: không gửi lệnh sang mô hình arm Gazebo đang lỗi frame.

Để kiểm tra UAV có thật sự “cảm” lực cánh tay ảo trong mô phỏng, bật nhiễu ảo:

```bash
arm_virtual_disturbance_enable:=true
```

Nếu muốn nhìn rõ nhiễu trước khi bù, chạy một lượt với:

```bash
arm_virtual_disturbance_enable:=true \
arm_ff_enable:=false
```

Sau đó chạy lại với:

```bash
arm_virtual_disturbance_enable:=true \
arm_ff_enable:=true
```

Lưu ý: khi không dùng joint vật lý Gazebo, nhiễu cánh tay không tự tác dụng lên
plant. `arm_virtual_disturbance_enable:=true` là nhánh mô phỏng nhiễu đó bằng
torque ảo; `arm_ff_enable:=true` là nhánh bù lại torque này.

Nếu muốn chạy Backstepping không RBFNN:

```bash
rbfnn_output_enable:=false
```

**5. QGroundControl**

Trong QGC:

1. Chờ vehicle connected.
2. Arm.
3. Takeoff lên khoảng `2 m`.
4. Giữ hover ổn định vài giây.

**6. Terminal 3: bật external controller**

Sau khi UAV đã hover ổn định:

```bash
source /opt/ros/humble/setup.bash
source /home/wicom/ros2_ws/install/setup.bash

ros2 service call /uam/enable_external_controller std_srvs/srv/Trigger
```

Sau lệnh này, ROS2 bắt đầu publish:

```text
/fmu/in/vehicle_torque_setpoint
/fmu/in/vehicle_thrust_setpoint
```

PX4 `mc_rate_control` sẽ dùng external torque/thrust nếu `MC_RATE_EXT_EN=1` và dữ liệu còn mới.

**7. Chạy chuyển động cánh tay nếu cần**

```bash
source /opt/ros/humble/setup.bash
source /home/wicom/ros2_ws/install/setup.bash

ros2 run uam_controller arm_trajectory_generator.py \
  --pattern slow_step \
  --duration 120 \
  --amplitude 0.02 \
  --rate 5 \
  --step-hold-time 15 \
  --transition-time 5
```

**8. Theo dõi nhanh**

```bash
ros2 topic echo /uam/debug_state
```

hoặc:

```bash
ros2 topic echo /joint_states
ros2 topic echo /arm/interaction_wrench
ros2 topic echo /fmu/in/vehicle_torque_setpoint
```

**Luồng chuẩn đề xuất**

```text
Terminal 1: make px4_sitl gz_x500_hop
PX4 shell : param set MC_RATE_EXT_EN 1
Terminal 2: ros2 launch uam_controller uam_qgc_mode.launch.py sim:=true ...
QGC       : arm + takeoff + hover 2 m
Terminal 3: ros2 service call /uam/enable_external_controller ...
Terminal 4: ros2 run uam_controller arm_trajectory_generator.py ...
```

Nếu UAV rung/mất ổn định, dừng external bằng cách kill ROS launch hoặc set:

```bash
param set MC_RATE_EXT_EN 0
```

PX4 sẽ quay về rate PID nội bộ.Mặc dù đã thành công gửi lệnh rồi, nhưng có lỗi logic ở cánh tay, gửi Joint 1, 3, 4, 5, 6 thì không phản ứng gì, còn Joint 2 thì làm cả cánh tay (từ joint 2) bị tách ra khỏi UAV


Draw the result:
cd uam_results/

export CSV=/home/wicom/uam_results/report_manual_tests/20260526_xxxxxx_report_manual_slow_step_amp002/flight_timeseries.csv

python3 /home/wicom/uam_results/ve_bang_tu_csv.py
