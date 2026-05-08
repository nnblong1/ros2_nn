Việc sử dụng kết hợp **PX4 Autopilot (Pixhawk)** và **Raspberry Pi 4 (RPI 4)** làm máy tính nhúng đồng hành (Companion Computer) thay đổi hoàn toàn chiến lược triển khai so với việc tự viết code từ đầu. PX4 sở hữu một bộ mã nguồn đóng gói rất chặt chẽ với cấu trúc PID xếp tầng (như đã phân tích ở hình ảnh PID 2 bậc tự do và lý thuyết trước đó), do đó, việc ép nó chạy trực tiếp Mạng Nơ-ron trên chip vi điều khiển ARM Cortex-M của Pixhawk là rất khó khăn.

Dưới đây là kiến trúc hệ thống và hướng dẫn cấu hình lại Phương pháp **Điều khiển Backstepping Thích nghi Mạng nơ-ron (Neuro-Adaptive Backstepping)** được thiết kế riêng cho hệ sinh thái PX4 + Raspberry Pi 4.

### 1. Kiến trúc Hệ thống Phân tán (Offboard Control Architecture)

Bạn không nên thay thế hoàn toàn bộ trộn động cơ (motor mixer) của PX4, mà hãy áp dụng mô hình phân chia điện toán:

- **Raspberry Pi 4 (Vòng ngoài - Mạng Nơ-ron & Backstepping):** Chạy hệ điều hành Ubuntu + ROS 2. Node ROS 2 sẽ nhận dữ liệu trạng thái (odometry) của UAV và góc xoay của tay máy 6-DoF. Tại đây, RBFNN sẽ ước lượng nhiễu trực tuyến. Khối Backstepping sẽ tính toán và xuất ra lệnh điều khiển dưới dạng **Vận tốc góc mong muốn (Target Body Rates: $p, q, r$)** và **Lực đẩy tổng (Normalized Thrust)**.
    
- **Pixhawk/PX4 (Vòng trong - Bám sát Vận tốc góc):** Đặt ở chế độ **Offboard Mode**. Bộ điều khiển PID siêu tốc (vòng nội) của PX4 sẽ nhận các tín hiệu Rate/Thrust từ RPI 4 và xuất xung PWM trực tiếp cho 4 động cơ. Điều này tận dụng được bộ lọc vi phân (Derivative filter) và các bộ lọc nhiễu IMU xuất sắc đã có sẵn trong lõi PX4.
    

### 2. Hướng dẫn Triển khai Từng bước (Cập nhật cho PX4 + RPI 4)

**Bước 1: Thiết lập Giao tiếp Phần cứng và Phần mềm**

- **Phần cứng:** Kết nối cổng TELEM2 trên Pixhawk với chân TX/RX (GPIO 14, 15) của Raspberry Pi 4 qua giao thức UART.
    
- **Phần mềm:** Cài đặt ROS 2 (Humble hoặc Foxy) trên RPI 4. Sử dụng **Micro-XRCE-DDS Agent** (chuẩn mới của PX4 từ bản v1.13 trở đi, thay thế cho MAVROS) để đồng bộ hóa các topic uORB nội bộ của PX4 thành các topic ROS 2 tự nhiên.
    

**Bước 2: Thu thập Dữ liệu (State Extraction)**

Trên RPI 4, viết một Node C++ hoặc Python (khuyến nghị C++ để đạt tần số 100Hz) đăng ký theo dõi (subscribe) các topic:

- `/fmu/out/vehicle_odometry`: Lấy vị trí, vận tốc và góc Euler hiện tại của UAV.
    
- `Joint_states` (từ driver tay máy ROS 2): Lấy góc và vận tốc góc của 6 khớp cánh tay.
    

**Bước 3: Chạy RBFNN và Luật Cập nhật Trọng số trên Raspberry Pi 4**

Mạng RBFNN (Radial Basis Function Neural Network) có cấu trúc chỉ 1 lớp ẩn nên RPI 4 hoàn toàn có thể tính toán mượt mà bằng các ma trận Eigen (C++) hoặc NumPy (Python) mà không cần GPU.

- **Vector đầu vào $x$ của RBFNN:** Chứa sai số vị trí, vận tốc của UAV và tọa độ trọng tâm (CoG) ước tính của cánh tay 6-DoF.
    
- **Luật cập nhật Lyapunov:** Code hàm vi phân cập nhật ma trận trọng số $\hat{W}$ của RBFNN trực tuyến (online) theo mỗi vòng lặp `dt`:
    
    $$\dot{\hat{W}} = \Gamma ( \Phi(x) z^T - \sigma \hat{W} )$$
    
    _(Trong đó: $\Gamma$ là ma trận tốc độ học, $\Phi(x)$ là hàm kích hoạt Gaussian, $z$ là biến trạng thái lọc của Backstepping, $\sigma$ là hệ số rò rỉ rào cản để giữ an toàn trọng số)._
    

**Bước 4: Tính toán Backstepping và Đẩy tín hiệu (Publishing)**

- Thay vì tự tính ma trận trộn động cơ, thuật toán Backstepping trên RPI 4 sẽ trừ đi lực nhiễu do RBFNN xấp xỉ được $\hat{f}(x)$, từ đó nội suy ra một vector véc-tơ véc-tơ Vận tốc Góc (Roll rate, Pitch rate, Yaw rate) và Lực đẩy cần thiết.
    
- Publish vector này vào topic `/fmu/in/vehicle_rates_setpoint` của PX4.
    
- Đồng thời publish tín hiệu nhịp tim vào topic `/fmu/in/offboard_control_mode` ở tần số tối thiểu 50Hz để PX4 giữ nguyên chế độ Offboard.
    

---

### 3. Tài liệu Nghiên cứu Thực tế với PX4 (Cập nhật 2024 - 2025)

Vì bạn dùng hệ sinh thái PX4, bạn nên tham khảo cách các tác giả khác nhúng mô hình AI/Động lực học phi tuyến vào nền tảng này:

**1. Về Triển khai Cấu trúc Xếp tầng Mạng Nơ-ron trên PX4:**

- **"A Neural Network Mode for PX4 on Embedded Flight Controllers"** (NTNU-ARL, _arXiv 2025_): Bài báo này cực kỳ quan trọng cho bạn. Dù họ đào tạo mô hình bằng Học tăng cường (RL), nhưng họ đã công bố mã nguồn mở chỉ cách đưa khuôn khổ Mạng Nơ-ron (TensorFlow Lite) thẳng vào cấu trúc xếp tầng của mã nguồn PX4, cung cấp giải pháp cho bài toán từ-mô-phỏng-ra-thực-tế (sim-to-real).
    

**2. Về Bù trừ Trọng tâm (CoM) cho UAM dùng PX4:**

- **"AI-based online center of mass estimation and compensation for aerial manipulators"** (_arXiv 2024_): Nghiên cứu này dùng PX4 làm flight controller cấp thấp và Intel NUC (có thể thay bằng RPI 4 của bạn) làm máy tính đồng hành. Họ hướng dẫn cách bù trừ trực tuyến sự thay đổi trọng tâm của một máy bay gắn tay máy thao tác.
    

**3. Về Điều khiển Backstepping cho UAM trên Máy tính đồng hành:**

- **"Adaptive Backstepping Control of an Unmanned Aerial Manipulator"** (_MDPI Drones_, Năm 2025): Phân tích thiết kế bộ điều khiển phản hồi backstepping thích nghi cho một UAV đa rotor gắn cánh tay. Hiệu ứng của cánh tay máy được mô hình hóa dưới dạng _lực nhiễu, mô-men xoắn, và sự không chắc chắn tham số_ trong không gian làm việc. Đây là lý thuyết nền tảng lý tưởng để bạn code vòng lặp trên RPI 4.
  
### 4. Các hướng dẫn
#### 0. Các cơ sở về toán
1.  [[Mô hình động lực học UAV (dynamic model)]]

