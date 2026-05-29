# Quy tac ket noi Pi 4, PX4, QGC

Neu `/fmu/out/...` da co du lieu on dinh, di theo thu tu nay. Dung bat external controller ngay.

## MAVLink RX loss / loop checklist

Loi `rx loss` rat cao tren PX4 USB CDC, dac biet khi `mavlink status` hien `Forwarding: On` va `Received Messages` tu `sysid: 1, compid: 1`, thuong la dau hieu PX4 dang nhan lai goi MAVLink cua chinh no. Day la loi topology/routing, khong nen xu ly dau tien bang cach doi USB baudrate.

`sysid:255, compid:190, msgid:69` thuong la QGC/GCS gui `MANUAL_CONTROL`. Neu sequence loss rat lon, hay nghi toi viec cung mot nguon GCS dang di qua nhieu duong hoac bi forward vong lai.

Quy tac bat buoc:

- `/dev/ttyACM0`: chi danh cho MAVLink qua `mavlink-routerd`.
- TELEM2/UART (`/dev/serial0`, hien dang tro toi `/dev/ttyS0` tren Pi nay): chi danh cho XRCE-DDS qua `MicroXRCEAgent`.
- QGC laptop: chi mot UDP link tren `14550`; tat MAVLink forwarding va xoa cac target host/endpoint thua.
- Khong chay `MicroXRCEAgent` tren `/dev/ttyACM0`.
- Khong chay `MicroXRCEAgent` tren `/dev/ttyAMA0` trong setup nay, vi `/dev/ttyAMA0` la `/dev/serial1`, khong phai UART TELEM2 dang noi vao GPIO14/15.
- Khong chay dong thoi MAVROS/MAVSDK/mavlink-router endpoint khac neu chua xac dinh ro luong routing.

Tren PX4 NSH, tat MAVLink forwarding va dam bao TELEM2 khong bi MAVLink chiem:

```sh
param show MAV_*_FORWARD
param set MAV_0_FORWARD 0
param set MAV_1_FORWARD 0
param set MAV_2_FORWARD 0
param set MAV_3_FORWARD 0
param set MAV_HB_FORW_EN 0

param show MAV_*_CONFIG
param show XRCE_DDS_CFG
```

Neu thay instance nao co `MAV_N_CONFIG = 102` hoac `TELEM 2`, disable instance do vi TELEM2 chi dung cho XRCE-DDS:

```sh
# Vi du: neu MAV_1_CONFIG dang la TELEM2/102 thi disable instance do.
param set MAV_1_CONFIG 0
param set XRCE_DDS_CFG 102
param save
reboot
```

Neu `MAV_3_FORWARD` hoac mot param nao do khong ton tai tren firmware cua ban thi bo qua dong loi do. Sau reboot, `mavlink status` tren USB CDC phai hien `Forwarding: Off`.

Kiem tra runtime tren Pi truoc khi arm:

```bash
tools/check_mavlink_topology.sh
lsof /dev/ttyACM0 /dev/serial0 /dev/ttyAMA0 /dev/ttyS0
ss -lunp | rg '14540|14550|14555|14557|14580'
ps -ef | rg 'mavlink|mavros|MAVSDK|router|QGroundControl|MicroXRCEAgent'
```

Neu `rx loss` tang lai, giu nguyen trang thai va chup bang chung endpoint:

```bash
ss -lunp | rg '14540|14550|14555|14557|14580'
sudo tcpdump -ni any 'udp port 14540 or udp port 14550 or udp port 14555 or udp port 14557 or udp port 14580'
```

**Nguon tham chieu**

- PX4 `MAV_N_FORWARD`: https://docs.px4.io/v1.13/en/advanced_config/parameter_reference#MAV_0_FORWARD
- PX4 `MAV_N_CONFIG` va gia tri TELEM2: https://docs.px4.io/v1.13/en/advanced_config/parameter_reference#MAV_0_CONFIG
- MAVLink `MANUAL_CONTROL (69)`: https://mavlink.io/en/messages/common.html#MANUAL_CONTROL
- MAVLink `MAV_COMP_ID_MISSIONPLANNER (190)`: https://mavlink.io/en/messages/common.html#MAV_COMP_ID_MISSIONPLANNER

## Ket noi sach cho baseline

Terminal 1: QGC/MAVLink qua USB CDC:

```bash
mavlink-routerd /dev/ttyACM0:2000000 -e IP_LAPTOP_QGC:14550
```

Dung `ip a` de lay `IP_LAPTOP_QGC`. Tren QGC, disable MAVLink forwarding truoc khi test `rx loss`.

Terminal 2: XRCE-DDS qua TELEM2/UART rieng:
Tren Pi, dung agent dang giu `/dev/ttyAMA0` neu con chay nham:

```bash
sudo pkill MicroXRCEAgent
sudo pkill mavlink-routerd
sudo lsof -l /dev/ttyAMA0 /dev/serial0 /dev/ttyS0
```

Neu no tu chay lai, tuc la co service. Tim va stop service do:

```bash
systemctl list-units --type=service | grep -Ei 'xrce|micro|ros|uav'
```

Roi stop service tuong ung, vi du:

```bash
sudo systemctl stop <ten-service>
```

Ket noi XRCE dung UART TELEM2:

```bash
sudo MicroXRCEAgent serial --dev /dev/serial0 -b 921600
```

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
  xrce_serial_dev:=/dev/serial0 \
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

**9. Acceptance test cho MAVLink loop**

Sau khi reboot PX4:

```sh
mavlink status
```

Ket qua mong muon tren instance USB CDC:

- `Forwarding: Off`
- `rx loss` khong tiep tuc tang nhanh
- `sysid: 1, compid: 1` khong tang, hoac bien mat khoi `Received Messages`

Sau khi bat XRCE Agent tren `/dev/serial0`, kiem tra lai:

```bash
ros2 topic echo /fmu/out/vehicle_status_v1 --once
ros2 topic echo /fmu/out/vehicle_odometry --once
```

`mavlink status` van phai giu `Forwarding: Off` va `rx loss` on dinh. Neu van con loss, capture endpoint dang loop:

```bash
ss -lunp | rg '14540|14550|14555|14557|14580'
sudo tcpdump -ni any 'udp port 14540 or udp port 14550 or udp port 14555 or udp port 14557 or udp port 14580'
```
