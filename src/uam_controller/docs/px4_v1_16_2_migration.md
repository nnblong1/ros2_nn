# PX4 v1.16.2-rbfnn Migration Notes

## Mục tiêu

Firmware custom vẫn dùng cơ chế cũ của đề tài:

- PX4 chạy position controller và attitude controller để sinh `vehicle_rates_setpoint`.
- ROS `uam_backstepping_rbfnn_node` đọc `vehicle_rates_setpoint`.
- ROS publish `vehicle_torque_setpoint` và `vehicle_thrust_setpoint`.
- PX4 `mc_rate_control` trên nhánh `px4-v1.16.2-rbfnn` bypass PID rate controller khi setpoint ngoài còn mới.

Điểm thay đổi là nền firmware được đưa về PX4 v1.16.2 để ổn định hơn. Đây không phải workflow vanilla-only.

## Firmware PX4

Nhánh cần dùng:

```bash
cd /home/wicom/PX4-Autopilot
git checkout px4-v1.16.2-rbfnn
git submodule update --init --recursive Tools/simulation/gz
git -C Tools/simulation/gz checkout e7675e23189040aceb62b7a4af48d84e06557166
make px4_sitl gz_x500_hop
```

Các thay đổi chính trong PX4:

- `MC_RATE_EXT_EN`: bật cổng bypass external rate controller.
- `MC_RATE_EXT_TMO`: timeout setpoint ngoài, mặc định `0.10 s`.
- `vehicle_rates_setpoint` được bridge ra ROS qua `/fmu/out/vehicle_rates_setpoint`.
- `/fmu/in/vehicle_torque_setpoint` và `/fmu/in/vehicle_thrust_setpoint` dùng `PublicationMulti`, để ROS publish vào uORB instance 1.
- `mc_rate_control` giữ instance 0 cho output nội bộ của PX4 và đọc external setpoint ở instance 1.

Airframe `4000_gz_x500_hop` mặc định đặt `MC_RATE_EXT_EN=1`. Điều này vẫn an toàn cho takeoff/hover nền vì firmware chỉ bypass khi torque external còn mới; nếu ROS chưa enable hoặc topic bị stale, PX4 tự fallback về PID rate controller nội bộ.

Lưu ý phần cứng thật: `4000_gz_x500_hop` hiện nằm trong
`ROMFS/px4fmu_common/init.d-posix/airframes`, chỉ dùng cho SITL/POSIX.
Log trên board `PX4_FMU_V6C` sẽ không chạy airframe POSIX này. Nếu log phần cứng
có `SYS_AUTOSTART=4001`, board đang dùng airframe hardware `4001_quad_x`
generic, nên `MC_RATE_EXT_EN` giữ default `0` và các tham số riêng của airframe
4000 không được nạp. Muốn dùng `SYS_AUTOSTART=4000` trên phần cứng, phải thêm
airframe tương ứng vào `ROMFS/px4fmu_common/init.d/airframes`, thêm vào
`CMakeLists.txt`, rebuild/flash firmware, rồi chọn airframe 4000 hoặc set
`SYS_AUTOSTART=4000` và reboot. Nếu chỉ dùng QGC/import params, phải đảm bảo
firmware đã có airframe 4000; nếu không, sau reboot PX4 sẽ không nạp đúng cấu
hình airframe.

### Đảm bảo firmware hardware có airframe 4000 trước khi QGC/import params

QGC/import params chỉ ghi giá trị tham số vào flash của flight controller. Nó
không thể thêm file airframe mới vào ROMFS của firmware. Vì vậy trước khi import
file params có `SYS_AUTOSTART=4000`, firmware flash lên board phải chứa một file
airframe hardware có tiền tố `4000_...`.

Quy trình đúng cho board `PX4_FMU_V6C`:

1. Trong source PX4, tạo airframe hardware mới trong:

   ```bash
   /home/wicom/PX4-Autopilot/ROMFS/px4fmu_common/init.d/airframes/4000_uam_hop
   ```

   File này phải bắt đầu bằng:

   ```sh
   #!/bin/sh
   #
   # @name UAM HOP Quad X
   # @type Quadrotor x
   # @class Copter
   #

   . ${R}etc/init.d/rc.mc_defaults

   param set-default SYS_AUTOSTART 4000
   ```

   Sau đó copy các nhóm param cần cho phần cứng từ airframe SITL
   `init.d-posix/airframes/4000_gz_x500_hop`: `CA_METHOD`, toàn bộ
   `CA_ROTOR*`, `MC_RATE_EXT_EN`, `MC_RATE_EXT_TMO`, các gain `MC_*`,
   `MPC_*`, `BAT1_N_CELLS`, và các tham số vehicle thật đã kiểm chứng.
   Không copy các dòng chỉ dành cho SITL như `PX4_SIMULATOR`, `PX4_GZ_*`,
   `SIM_GZ_*`, hoặc các circuit breaker `CBRK_*` nếu chưa có quyết định an
   toàn rõ ràng cho bay thật.

2. Thêm file mới vào:

   ```bash
   /home/wicom/PX4-Autopilot/ROMFS/px4fmu_common/init.d/airframes/CMakeLists.txt
   ```

   Trong nhóm `[4000, 4999] Quadrotor x`, thêm dòng:

   ```cmake
       4000_uam_hop
   ```

3. Rebuild và flash đúng target hardware:

   ```bash
   cd /home/wicom/PX4-Autopilot
   make px4_fmu-v6c_default
   ```

   Sau đó flash firmware `.px4` sinh ra qua QGC, hoặc dùng workflow upload
   phần cứng nếu board đang ở bootloader.

4. Sau khi flash và reboot, kiểm tra trong QGC:

   - Vào `Analyze Tools` -> `MAVLink Console`.
   - Chạy `ls /etc/init.d/airframes` và xác nhận có file `4000_uam_hop`.
   - Chạy `param show MC_RATE_EXT_TMO`. Nếu báo không có param, firmware đang
     chạy chưa phải nhánh custom mới.
   - Set hoặc import params với `SYS_AUTOSTART=4000`, `MC_RATE_EXT_EN=1`,
     `MC_RATE_EXT_TMO=0.10`, rồi reboot.
   - Sau reboot, kiểm tra lại:

     ```sh
     param show SYS_AUTOSTART
     param show MC_RATE_EXT_EN
     param show MC_RATE_EXT_TMO
     param show CA_ROTOR0_PX
     param show CA_ROTOR0_PY
     ```

     Kết quả đạt yêu cầu là `SYS_AUTOSTART=4000`, `MC_RATE_EXT_EN=1`,
     `MC_RATE_EXT_TMO` tồn tại, và `CA_ROTOR*` không còn là geometry generic
     của `4001_quad_x` như `PX=1`, `PY=1`.

Nếu QGC import params thành công nhưng sau reboot `SYS_AUTOSTART=4001`, hoặc
`MC_RATE_EXT_TMO` không tồn tại, thì lỗi nằm ở firmware/airframe chưa đúng chứ
không phải ở file params. Phải sửa airframe trong PX4, build lại và flash lại
trước khi bay.

### Cập nhật firmware custom sau khi thêm `4000_uam_hop`

Quy trình này dùng cho board thật `PX4_FMU_V6C` sau khi source PX4 đã có:

- `ROMFS/px4fmu_common/init.d/airframes/4000_uam_hop`
- entry `4000_uam_hop` trong
  `ROMFS/px4fmu_common/init.d/airframes/CMakeLists.txt`

1. Kiểm tra đang ở đúng nhánh custom:

   ```bash
   cd /home/wicom/PX4-Autopilot
   git branch --show-current
   git status --short ROMFS/px4fmu_common/init.d/airframes/4000_uam_hop \
       ROMFS/px4fmu_common/init.d/airframes/CMakeLists.txt \
       src/modules/mc_rate_control
   ```

   Nhánh cần là `px4-v1.16.2-rbfnn`. Nếu file `4000_uam_hop` đang là
   untracked thì vẫn build được ở máy local, nhưng cần đưa file này vào git
   trước khi chuyển source sang máy khác.

2. Build firmware hardware:

   ```bash
   cd /home/wicom/PX4-Autopilot
   make px4_fmu-v6c_default
   ```

   File cần flash sau khi build xong:

   ```bash
   /home/wicom/PX4-Autopilot/build/px4_fmu-v6c_default/px4_fmu-v6c_default.px4
   ```

3. Kiểm tra artifact build đã đóng gói airframe 4000:

   ```bash
   test -f build/px4_fmu-v6c_default/etc/init.d/airframes/4000_uam_hop
   rg -n "SYS_AUTOSTART|MC_RATE_EXT_EN|MC_RATE_EXT_TMO" \
       build/px4_fmu-v6c_default/etc/init.d/airframes/4000_uam_hop
   ```

   Kết quả phải có `SYS_AUTOSTART 4000`, `MC_RATE_EXT_EN 1`, và
   `MC_RATE_EXT_TMO 0.10`.

4. Flash bằng QGC:

   - Tháo cánh.
   - Cắm USB flight controller.
   - Mở QGC -> `Vehicle Setup` -> `Firmware`.
   - Chọn `Advanced settings` hoặc `Custom firmware file`.
   - Chọn file:

     ```text
     /home/wicom/PX4-Autopilot/build/px4_fmu-v6c_default/px4_fmu-v6c_default.px4
     ```

   - Chờ flash xong và để board reboot.

   Nếu dùng terminal thay QGC và board đang vào bootloader, có thể dùng:

   ```bash
   cd /home/wicom/PX4-Autopilot
   make px4_fmu-v6c_default upload
   ```

5. Chọn airframe mới và bắt PX4 load default của airframe:

   Trong QGC -> `Analyze Tools` -> `MAVLink Console`, chạy:

   ```sh
   param set SYS_AUTOSTART 4000
   param set SYS_AUTOCONFIG 1
   param save
   reboot
   ```

   `SYS_AUTOCONFIG=1` là bước quan trọng khi đổi từ `4001_quad_x` sang
   `4000_uam_hop`: PX4 sẽ reset các param không thuộc nhóm calibration/RC
   được giữ lại, rồi nạp default từ file airframe mới ở lần boot kế tiếp.
   Trước khi làm bước này nên export params hiện tại từ QGC để có bản backup.

6. Sau reboot, kiểm tra airframe và param:

   Trong QGC `MAVLink Console`:

   ```sh
   ls /etc/init.d/airframes
   param show SYS_AUTOSTART
   param show SYS_AUTOCONFIG
   param show MC_RATE_EXT_EN
   param show MC_RATE_EXT_TMO
   param show CA_ROTOR0_PX
   param show CA_ROTOR0_PY
   param show CA_ROTOR1_PX
   param show CA_ROTOR1_PY
   ```

   Kết quả đạt yêu cầu:

   - Danh sách airframe có `4000_uam_hop`.
   - `SYS_AUTOSTART=4000`.
   - `SYS_AUTOCONFIG=0` sau khi quá trình reset/reboot đã hoàn tất.
   - `MC_RATE_EXT_EN=1`.
   - `MC_RATE_EXT_TMO=0.10`.
   - `CA_ROTOR0_PX=-0.159`, `CA_ROTOR0_PY=-0.159`,
     `CA_ROTOR1_PX=0.159`, `CA_ROTOR1_PY=0.159`.

7. Kiểm tra bắt buộc trước khi lắp cánh:

   - QGC `Actuator Test`: xác nhận motor 0-3 đúng vị trí và đúng chiều quay
     như comment trong `4000_uam_hop`.
   - QGC `Sensors`: calibration vẫn hợp lệ sau flash.
   - QGC `Parameters`: các serial/RC/failsafe/power/battery params thực tế
     vẫn đúng với wiring của vehicle.
   - MAVLink Console: `param show MC_RATE_EXT_TMO` không báo missing.

Chỉ bay test sau khi các kiểm tra trên đạt. Nếu sau flash vẫn thấy
`SYS_AUTOSTART=4001`, hoặc `MC_RATE_EXT_TMO` missing, thì board chưa chạy đúng
firmware custom mới.

Nếu `MC_RATE_EXT_TMO` không xuất hiện trong ULog, binary đang bay chưa chứa
định nghĩa param này. Source hiện tại định nghĩa `MC_RATE_EXT_TMO` trong
`src/modules/mc_rate_control/mc_rate_control_params.c`; vì vậy cần build/flash
lại đúng nhánh PX4 trước khi đánh giá log mới.

Lưu ý quan trọng về submodule Gazebo: nếu `git -C Tools/simulation/gz rev-parse
--short HEAD` trả về `e05f4312d3`, đó là commit official và sẽ không có
`models/x500_hop`. Phải checkout lại commit custom `e7675e2318` trước khi chạy
`make px4_sitl gz_x500_hop`.

## ROS Launch

Chạy QGC workflow:

```bash
source /home/wicom/ros2_ws/install/setup.bash
ros2 launch uam_controller uam_qgc_mode.launch.py sim:=true
```

Mặc định:

- `allow_external_torque_handoff:=true`
- `external_handoff_mode:=manual`
- ROS chỉ publish torque/thrust sau khi stable hover gate sẵn sàng và có lệnh enable.
- `/uam/enable_external_controller` chỉ bật cờ ROS `/uam/controller_enable`.
  Service này không set PX4 param `MC_RATE_EXT_EN`, không set `SYS_AUTOSTART`,
  và không tự chuyển PX4 sang Offboard/External nav mode.
- `qgc_rbfnn_trigger.py` chặn handoff nếu PX4 còn ở `STAB`. Mặc định chỉ cho
  phép `OFFBOARD`, `POSCTL`, `AUTO_LOITER`, hoặc `AUTO_TAKEOFF`; có thể đổi
  bằng launch arg `required_nav_states`.

Bật handoff thủ công sau khi QGC đã takeoff và hover ổn định:

```bash
ros2 service call /uam/enable_external_controller std_srvs/srv/Trigger {}
```

Trước khi gọi service này, kiểm tra QGC/PX4 đang ở `POSCTL`, `AUTO_LOITER`
hoặc `OFFBOARD`. Nếu còn `STAB`, service phải bị từ chối và không được publish
external torque.

Tắt handoff để quay về fallback PX4 internal rate PID:

```bash
ros2 service call /uam/disable_external_controller std_srvs/srv/Trigger {}
```

Muốn so sánh bay chỉ bằng PX4 internal rate controller:

```bash
ros2 launch uam_controller uam_qgc_mode.launch.py sim:=true allow_external_torque_handoff:=false
```

## Safety gate cho external backstepping + RBFNN

Nếu UAV lật ngay sau khi chuyển sang external, không được xem đó là lỗi nhỏ
của mô phỏng. Với pipeline bypass `mc_rate_control`, ROS đang gửi moment trực
tiếp vào PX4 control allocation, nên một bước moment sai dấu/sai biên độ có thể
lật máy rất nhanh.

Các nguyên nhân chính đã được khóa lại trong ROS controller:

- Không bật external ngay khi vừa vào `HOLD/GOTO`. `uam_mission_bridge.py` phải
  thấy UAV hover ổn định liên tục trước khi publish `/uam/controller_enable=true`.
- Moment external được ramp, clamp và rate-limit trước khi publish
  `/fmu/in/vehicle_torque_setpoint`.
- `base_pitch_offset` và `base_roll_offset` không còn được cộng trực tiếp mặc
  định. Nếu cần dùng offset tĩnh, chỉ bật lại bằng `base_offset_enable: true`
  sau khi đã xác định đúng dấu và biên độ trong SITL.
- Nếu roll/pitch vượt `external_safety_tilt_deg` hoặc body-rate vượt
  `external_safety_rate_rad_s`, node dừng publish torque để PX4 fallback về PID
  rate controller nội bộ sau `MC_RATE_EXT_TMO`.
- Với QGC workflow, trigger không cho bật external torque trong `STAB`; phải
  chuyển sang mode có position/hold/offboard trước khi handoff.

Các tham số an toàn mặc định:

```yaml
uam_adaptive_controller:
  ros__parameters:
    base_offset_enable: false
    external_torque_ramp_s: 5.0
    external_torque_limit_initial: 0.03
    external_torque_limit_final: 0.35
    external_torque_rate_limit_norm_s: 0.40
    external_safety_tilt_deg: 30.0
    external_safety_rate_rad_s: 2.5

uam_mission_bridge:
  ros__parameters:
    external_handoff_stable_time_s: 5.0
    external_handoff_max_position_error_m: 0.30
    external_handoff_max_horizontal_speed_ms: 0.20
    external_handoff_max_vertical_speed_ms: 0.12
```

Quy trình kiểm chứng bắt buộc:

1. Chạy SITL với `allow_external_torque_handoff:=false` và xác nhận Offboard
   takeoff/HOLD ổn định bằng PX4 internal rate PID.
2. Chỉ sau đó chạy lại với `allow_external_torque_handoff:=true` và
   `auto_enable_external_controller:=true`.
3. Quan sát log phải có dòng chờ hover ổn định và sau đó mới có dòng
   `External torque handoff gate ready`.
4. Nếu có `External torque safety fault`, dừng thử nghiệm, không bay thật, và
   kiểm tra dấu moment, scaling `tau_axis_max_*`, trọng tâm/cánh tay, và
   `vehicle_rates_setpoint`.

## Bay Offboard indoor bằng local setpoint

Indoor không dùng GPS mission/waypoint. Toàn bộ lệnh bay phải đi theo local
frame của PX4:

- ROS 2 publish `px4_msgs/msg/TrajectorySetpoint` vào `/fmu/in/trajectory_setpoint`.
- ROS 2 publish `px4_msgs/msg/OffboardControlMode` vào `/fmu/in/offboard_control_mode`.
- `OffboardControlMode.position = true`.
- Không gửi global latitude/longitude.
- Không dùng QGC mission/waypoint GPS.
- Không dùng `VehicleCommand` dạng `NAV_WAYPOINT`/global mission để điều hướng.

Trong package này, node đúng để bay indoor Offboard là
`scripts/uam_mission_bridge.py`, chạy qua:

```bash
source /home/wicom/ros2_ws/install/setup.bash
ros2 launch uam_controller uam_system.launch.py sim:=true
```

Với phần cứng thật:

```bash
source /home/wicom/ros2_ws/install/setup.bash
ros2 launch uam_controller uam_system.launch.py sim:=false
```

`uam_mission_bridge.py` đã dùng đúng pipeline indoor:

- Gửi heartbeat Offboard liên tục lớn hơn 2 Hz bằng `OffboardControlMode`.
- Đặt `msg.position = True`.
- Gửi `TrajectorySetpoint.position = [x, y, z]` theo local NED của PX4.
- Giữ `velocity` và `acceleration` là `NaN` khi chỉ điều khiển vị trí.
- Không tạo global mission item.

Quy ước frame:

- PX4 `TrajectorySetpoint.position` là local NED.
- `x`: hướng North/local forward theo estimator.
- `y`: hướng East/local right theo estimator.
- `z`: Down, nên bay lên 2 m là `z = -2.0`.
- Nếu lệnh đến từ ROS ENU, phải đổi sang NED trước khi publish cho PX4.
  `uam_mission_bridge.py` đang đổi `/uam/cmd/goto_pose` từ ENU sang NED:
  `x_ned = x_enu`, `y_ned = -y_enu`, `z_ned = -z_enu`.

Điều kiện trước khi chuyển Offboard indoor:

- PX4 phải có local position estimate hợp lệ từ indoor localization
  như VIO, motion capture, optical flow/range, hoặc SITL ground truth.
- `/fmu/out/vehicle_local_position` phải có `xy_valid`, `z_valid`,
  `v_xy_valid`, `v_z_valid`.
- `heading_good_for_control` không bắt buộc trong SITL vì PX4 đôi khi để flag
  này `false` dù local position/velocity đã hợp lệ. Nếu muốn bắt buộc heading
  tốt trên phần cứng thật, đặt `require_heading_good_for_control: true` trong
  YAML.
- ROS phải stream `OffboardControlMode` và `TrajectorySetpoint` trước khi gửi
  lệnh vào Offboard/ARM. `uam_mission_bridge.py` đã prime Offboard trước khi arm.

Quy trình bay local indoor:

```bash
ros2 service call /uam/arm_takeoff std_srvs/srv/Trigger {}
```

Node sẽ:

- Prime Offboard heartbeat/setpoint.
- Gửi PX4 vào Offboard mode.
- Arm.
- Ramp local setpoint lên `takeoff_height`, mặc định `-2.0 m` NED.
- Chuyển sang HOLD tại local position.

Gửi điểm đến local mới bằng ROS ENU `PoseStamped`:

```bash
ros2 topic pub --once /uam/cmd/goto_pose geometry_msgs/msg/PoseStamped \
"{
  header: {frame_id: 'map'},
  pose: {
    position: {x: 1.0, y: 0.0, z: 2.0},
    orientation: {w: 1.0}
  }
}"
```

Ví dụ trên nghĩa là bay tới vị trí local ROS ENU `x=1.0 m`, `y=0.0 m`,
cao `2.0 m`. Node sẽ đổi sang PX4 NED trước khi publish
`TrajectorySetpoint`.

Các lệnh không dùng trong indoor:

- Không upload mission GPS từ QGroundControl.
- Không gửi waypoint latitude/longitude.
- Không dùng `VehicleGlobalPosition` làm setpoint điều khiển.
- Không phụ thuộc `COM_ARM_WO_GPS` để thay thế local estimator; tham số này chỉ
  bỏ yêu cầu GPS, không tạo được local position estimate.

## Quy trình thực tế khi bay indoor Offboard

Nguyên tắc thao tác:

- QGroundControl chỉ dùng để quan sát trạng thái, kiểm tra mode, pin, estimator,
  emergency/kill/disarm nếu cần.
- Không tạo mission trong tab Plan của QGC.
- Không upload waypoint GPS.
- Mọi lệnh bay indoor đi từ ROS 2 local setpoint qua `uam_mission_bridge.py`.
- Lần chạy đầu nên để `auto_enable_external_controller:=false` để xác nhận
  local Offboard bằng PX4 internal rate PID đã ổn định trước.
- Chỉ bật `auto_enable_external_controller:=true` khi đã kiểm chứng takeoff/HOLD
  ổn định.

### Quy trình SITL trước khi bay thật

Mở QGC trước hoặc sau PX4 đều được. Nếu dùng AppImage:

```bash
cd /home/wicom
./QGroundControl.AppImage
```

Nếu QGC đã cài trong menu ứng dụng thì mở bằng giao diện desktop. Sau khi mở,
để QGC ở tab Fly, không upload mission.

Terminal 1 - chạy PX4 SITL custom:

```bash
cd /home/wicom/PX4-Autopilot
git checkout px4-v1.16.2-rbfnn
git submodule update --init --recursive Tools/simulation/gz
git -C Tools/simulation/gz checkout e7675e23189040aceb62b7a4af48d84e06557166
git -C Tools/simulation/gz rev-parse --short HEAD
make px4_sitl gz_x500_hop
```

Kết quả cần thấy:

- Gazebo mở model `x500_hop`.
- QGC tự nhận vehicle qua UDP.
- PX4 shell không báo thiếu model `x500_hop`.
- Commit submodule Gazebo nên là commit custom có model `x500_hop`
  như `e7675e2318`.

Terminal 2 - chạy ROS 2 local Offboard, chưa bật RBFNN handoff:

```bash
source /home/wicom/ros2_ws/install/setup.bash
ros2 launch uam_controller uam_system.launch.py \
  sim:=true \
  allow_external_torque_handoff:=false \
  auto_enable_external_controller:=false
```

Chế độ này dùng ROS để gửi local `TrajectorySetpoint`, nhưng PX4 vẫn dùng
internal rate PID. Dùng nó để kiểm tra indoor Offboard local trước.

Terminal 3 - kiểm tra topic PX4/ROS:

```bash
source /home/wicom/ros2_ws/install/setup.bash
ros2 topic list --no-daemon
```

Các topic tối thiểu cần có:

- `/fmu/in/offboard_control_mode`
- `/fmu/in/trajectory_setpoint`
- `/fmu/out/vehicle_local_position`
- `/fmu/out/vehicle_rates_setpoint`
- `/fmu/in/vehicle_torque_setpoint`
- `/fmu/in/vehicle_thrust_setpoint`

Kiểm tra local position:

```bash
source /home/wicom/ros2_ws/install/setup.bash
ros2 topic echo /fmu/out/vehicle_local_position --once --no-daemon
```

Trước khi arm, cần local position hợp lệ. Trong SITL thường đã có sẵn. Với
phần cứng indoor phải có VIO/mocap/optical-flow/range hoặc nguồn local
position tương đương.

Terminal 4 - ra lệnh cất cánh local Offboard:

```bash
source /home/wicom/ros2_ws/install/setup.bash
ros2 service call /uam/arm_takeoff std_srvs/srv/Trigger {}
```

Theo dõi trong QGC:

- Mode chuyển sang Offboard.
- Vehicle arm.
- UAV lên khoảng 2 m.
- Sau khi đạt độ cao, `/uam/state` chuyển sang `HOLD`.

Kiểm tra state:

```bash
source /home/wicom/ros2_ws/install/setup.bash
ros2 topic echo /uam/state --once
```

Terminal 5 - gửi điểm đến local indoor:

```bash
source /home/wicom/ros2_ws/install/setup.bash
ros2 topic pub --once /uam/cmd/goto_pose geometry_msgs/msg/PoseStamped \
"{
  header: {frame_id: 'map'},
  pose: {
    position: {x: 1.0, y: 0.0, z: 2.0},
    orientation: {w: 1.0}
  }
}"
```

Lệnh trên là ROS ENU: đi tới `x=1 m`, `y=0 m`, cao `2 m`. Node sẽ tự đổi sang
PX4 NED trước khi publish `TrajectorySetpoint`.

Nếu local Offboard đã ổn định, dừng ROS bằng `Ctrl+C`, dừng PX4/Gazebo bằng
`Ctrl+C`, rồi chạy lại với RBFNN handoff.

Terminal 2 - chạy lại ROS 2 và cho phép tự bật RBFNN sau HOLD/GOTO:

```bash
source /home/wicom/ros2_ws/install/setup.bash
ros2 launch uam_controller uam_system.launch.py \
  sim:=true \
  allow_external_torque_handoff:=true \
  auto_enable_external_controller:=true
```

Sau đó lặp lại:

```bash
source /home/wicom/ros2_ws/install/setup.bash
ros2 service call /uam/arm_takeoff std_srvs/srv/Trigger {}
```

Khi mission bridge vào `HOLD` hoặc `GOTO`, nó sẽ publish
`/uam/controller_enable=true`. Lúc đó ROS backstepping+RBFNN bắt đầu publish
external torque/thrust. Nếu ROS mất setpoint hoặc dừng node, PX4 fallback về
internal rate PID sau `MC_RATE_EXT_TMO`.

Theo dõi controller enable:

```bash
source /home/wicom/ros2_ws/install/setup.bash
ros2 topic echo /uam/controller_enable --once
```

Dừng handoff trong tình huống không ổn định:

- Cách nhanh nhất trong SITL: `Ctrl+C` terminal ROS launch. PX4 sẽ fallback sau
  timeout.
- Nếu đang bay thật: ưu tiên chuyển mode/land/disarm từ QGC hoặc RC theo quy
  trình an toàn của bạn. Không tiếp tục gửi setpoint mới khi local estimator lỗi.

### Quy trình phần cứng indoor

Trước khi cấp pin:

- Propeller tháo ra khi test lần đầu.
- Firmware đã nạp từ branch `px4-v1.16.2-rbfnn`.
- QGC thấy đúng airframe/custom params.
- Indoor localization đã chạy và publish vào PX4.
- Micro XRCE-DDS serial đúng cổng, thường là `/dev/ttyAMA0`, `/dev/ttyUSB0`
  hoặc `/dev/ttyACM0`.

Terminal 1 - mở QGC:

```bash
cd /home/wicom
./QGroundControl.AppImage
```

Trong QGC:

- Không mở Plan để upload mission.
- Không tạo waypoint GPS.
- Kiểm tra vehicle connected.
- Kiểm tra estimator/local position không báo lỗi.

Terminal 2 - chạy ROS 2 với phần cứng, kiểm tra local Offboard trước:

```bash
source /home/wicom/ros2_ws/install/setup.bash
ros2 launch uam_controller uam_system.launch.py \
  sim:=false \
  allow_external_torque_handoff:=false \
  auto_enable_external_controller:=false
```

Nếu Micro XRCE-DDS Agent trong launch không đúng serial device, chạy agent
riêng ở terminal khác rồi chỉnh launch sau. Ví dụ:

```bash
MicroXRCEAgent serial --dev /dev/ttyACM0 -b 921600
```

Terminal 3 - kiểm tra local position:

```bash
source /home/wicom/ros2_ws/install/setup.bash
ros2 topic echo /fmu/out/vehicle_local_position --once --no-daemon
```

Chỉ arm khi local estimate hợp lệ. Nếu `xy_valid` hoặc `z_valid` false, không
bay Offboard indoor.

Terminal 4 - cất cánh local Offboard:

```bash
source /home/wicom/ros2_ws/install/setup.bash
ros2 service call /uam/arm_takeoff std_srvs/srv/Trigger {}
```

Sau khi bay ổn bằng PX4 internal PID, mới thử RBFNN handoff. Dừng ROS hiện tại,
rồi chạy lại:

```bash
source /home/wicom/ros2_ws/install/setup.bash
ros2 launch uam_controller uam_system.launch.py \
  sim:=false \
  allow_external_torque_handoff:=true \
  auto_enable_external_controller:=true
```

Sau đó gọi lại:

```bash
source /home/wicom/ros2_ws/install/setup.bash
ros2 service call /uam/arm_takeoff std_srvs/srv/Trigger {}
```

Khuyến nghị cho lần bay thật đầu tiên:

- Không dùng `auto_enable_external_controller:=true` ngay từ đầu.
- Bay bằng `allow_external_torque_handoff:=false` trước để xác nhận local
  Offboard, estimator, hướng frame, và chiều `z`.
- Khi đã chắc chắn hover ổn, mới bật handoff trong lần bay sau.
- Giữ QGC/RC sẵn sàng chuyển mode hoặc disarm.

## Điều kiện hoạt động

External torque path chỉ thật sự điều khiển motor khi đồng thời thỏa:

- PX4 đang chạy nhánh `px4-v1.16.2-rbfnn`.
- Airframe đúng đã được nạp. SITL dùng `SYS_AUTOSTART=4000`
  (`4000_gz_x500_hop`). Phần cứng thật phải có airframe hardware 4000 tương
  ứng hoặc set tay toàn bộ param cần thiết.
- PX4 param `MC_RATE_EXT_EN=1`.
- PX4 firmware/log có `MC_RATE_EXT_TMO`.
- PX4 nav mode không phải `STAB`; với QGC trigger mặc định cần `OFFBOARD`,
  `POSCTL`, `AUTO_LOITER`, hoặc `AUTO_TAKEOFF`.
- ROS nhận được `/fmu/out/vehicle_rates_setpoint`.
- `/uam/controller_enable=true`.
- ROS publish torque/thrust mới hơn `MC_RATE_EXT_TMO`.

Nếu một điều kiện mất đi, PX4 dùng lại internal rate PID.
