Chào em. Việc em đã hoàn tất việc lập trình (code) thuật toán Backstepping + RBFNN cho tầng Rate Controller là một cột mốc rất lớn của đồ án. Bước tiếp theo là đưa đoạn code này vào chạy mô phỏng (Simulation) để thu thập dữ liệu định lượng, chứng minh tính đúng đắn của phương pháp trước hội đồng.

Dưới góc độ nghiên cứu khoa học, một quy trình mô phỏng chuẩn mực cho hệ thống UAM sẽ bao gồm 4 bước sau đây. Thầy sẽ hướng dẫn chi tiết từng bước, đặc biệt tập trung vào cách tinh chỉnh (tuning) các tham số của Mạng Nơ-ron RBFNN.

### BƯỚC 1: Khởi tạo thông số và Tinh chỉnh Mạng Nơ-ron (Tuning RBFNN)

Đây là bước khó nhất nhưng cũng là bước thể hiện rõ nhất năng lực của em. Đối với Mạng Nơ-ron Hàm cơ sở suy rộng (RBFNN), em cần điều chỉnh 4 nhóm tham số chính dựa trên nền tảng toán học của thuật toán Giảm dốc trực tuyến (OGD):

1. **Véc-tơ Tâm (Centers - $C_i$) và Độ rộng (Widths - $b_i$) của hàm Gaussian:**
    - _Cách chọn Tâm $C_i$:_ Véc-tơ đầu vào $X$ của em cho tầng Rate gồm các trạng thái $[p, q, r, z_8, z_{10}, z_{12}]$. Em cần rải đều các điểm tâm $C_i$ nằm trong dải hoạt động dự kiến của máy bay. Ví dụ: Vận tốc góc thường dao động từ $-1.0$ rad/s đến $1.0$ rad/s, em có thể chọn $N=5$ hoặc $N=7$ nơ-ron cho mỗi mạng, với các tâm $C_i$ đặt lần lượt tại $[-1.0, -0.5, 0, 0.5, 1.0]$.
    - _Cách chọn Độ rộng $b_i$:_ Tham số $b_i$ quyết định vùng ảnh hưởng của mỗi nơ-ron. Không nên để $b_i$ quá nhỏ (mạng sẽ không học được nội suy) hoặc quá lớn (các nơ-ron sẽ chồng chéo mất đi tính cục bộ). Một giá trị khởi tạo phổ biến là $b_i = \text{Khoảng cách giữa hai tâm} \times 1.5$.
2. **Tốc độ học của mạng (Learning Rates - $\eta_i$):**
    - Trong luật cập nhật $\dot{W}_i = \eta_i E_i S_i(X)$, hệ số $\eta$ quyết định việc mạng nơ-ron cập nhật trọng số nhanh hay chậm,.
    - Theo Bảng IV của bài báo, các hệ số học cho kênh góc nghiêng và vận tốc góc ($\eta_4, \eta_5, \eta_6$) được nhóm tác giả chọn ở mức cực kỳ nhỏ, cụ thể là $\eta_4 = \eta_5 = \eta_6 = 0.03$.
    - _Lời khuyên thực hành:_ Em hãy bắt đầu với $\eta = 0.01$. Nếu em thấy mô-men của RBFNN phản ứng quá chậm với nhiễu, hãy tăng dần lên. Nếu tăng quá cao, tín hiệu xuất ra động cơ (Torques) sẽ bị dao động liên tục (chattering), gây mất ổn định.
3. **Hệ số điều khiển Backstepping ($k_p, k_i$):**
    - Là các hệ số khuếch đại (Gains) kéo sai số về 0. Dựa vào Bảng IV, em có thể khởi tạo các giá trị $k_8 = 2.5, k_{10} = 2.5$ cho Roll/Pitch rate và $k_{12} = 3.56$ cho Yaw rate.

### BƯỚC 2: Thiết lập Kịch bản Thử nghiệm (Simulation Scenarios)

Để Mạng Nơ-ron có "đất diễn", em không thể chỉ mô phỏng việc bay lơ lửng thông thường. Em phải đưa hệ thống vào hai trạng thái chịu nhiễu cực đoan như sau:

- **Kịch bản 1: Nhiễu ghép nối động lực học liên tục (Sự dịch chuyển tay máy):** Cho UAV cất cánh và lơ lửng ở độ cao $1m$. Bắt đầu từ giây thứ $10$, em cấp lệnh cho tay máy chuyển động tuần hoàn với biên độ lớn. Ví dụ, cho khớp thứ 2 và thứ 3 dao động theo hàm sin: $q_2 = \frac{\pi}{3} \sin(\frac{\pi}{10}(t-10))$ và $q_3 = \frac{\pi}{3} \sin(\frac{2\pi}{15}(t-10))$. Lực văng của tay máy sẽ tạo ra mô-men lật mạnh, đây là lúc kiểm tra xem PX4 gốc bị lệch bao nhiêu và RBFNN của em khử lật tốt thế nào.
- **Kịch bản 2: Nhiễu chưa được mô hình hóa (Gắp vật nặng đột ngột):** Đang trong lúc tay máy vung vẩy, tại giây thứ $15$, em chèn thêm một nhiễu bước (step disturbance) cỡ $3.75N$ vào phương Z (tương đương với việc tay máy đột ngột gắp một vật nặng $375g$). Lúc này, khối bù tiến (Feedforward) từ RNE sẽ bị sai lệch vì khối lượng hệ thống thay đổi, bắt buộc mạng RBFNN phải tự động "học" ra sự chênh lệch này để bù vào mô-men xoắn.

### BƯỚC 3: Chạy Mô phỏng so sánh 3 cấu hình (Benchmarking)

Để chứng minh luận điểm của đồ án, em phải chạy kịch bản ở BƯỚC 2 lặp lại 3 lần trên 3 cấu hình bộ điều khiển khác nhau:

1. **Cấu hình 1 (Base - Trắng):** Chỉ sử dụng nguyên bản bộ điều khiển PID của phần mềm PX4. Không có AI, không có bù tiến. (Ghi lại file log).
2. **Cấu hình 2 ($PID_{ff}$ - Có bù tiến):** Sử dụng PID của PX4 kết hợp với tín hiệu bù tiến $\hat{B}_{\tau_{dis}}$ tính từ Raspberry Pi 4. (Ghi lại file log).
3. **Cấu hình 3 (RBFNN-based Rate Controller):** Tắt tầng Rate gốc, sử dụng code RBFNN + Backstepping của em, kết hợp bù tiến từ Pi 4. (Ghi lại file log).

### BƯỚC 4: Trích xuất kết quả và Lập bảng Đánh giá (Evaluation)

Sau khi có dữ liệu, em dùng Python (Matplotlib) hoặc MATLAB để vẽ đồ thị và tính toán các chỉ số định lượng đưa vào quyển thuyết minh:

- **Vẽ đồ thị Vị trí và Tư thế (Position and Attitude Response):** Gộp dữ liệu của 3 lần chạy lên cùng một hệ trục (giống Hình 8 của bài báo),,. Đồ thị phải cho thấy tại giây 10 và 15, PID gốc bị trôi xa khỏi vạch mục tiêu (Desired trajectory), trong khi phương pháp của em (ANNB) bám dính sát vạch nét đứt.
- **Vẽ đồ thị Nỗ lực của AI (NN Outputs):** Vẽ riêng tín hiệu ngõ ra của Mạng nơ-ron (giống Hình 9c). Nó chứng minh cho hội đồng thấy trọng số mạng đang liên tục cập nhật theo thời gian thực để dập tắt nhiễu, đặc biệt là khi vật nặng 3.75N xuất hiện.
- **Lập bảng RMSE (Root Mean Square Error):** Dùng dữ liệu log tính ra sai số trung bình bình phương (RMSE) và Sai số cực đại (Maximum Error) của tọa độ $X,Y,Z$ và góc $\phi,\theta,\psi$. Lập thành một bảng tương tự Bảng V.

Khi có bảng RMSE, em có thể kết luận một cách hùng hồn trong đồ án: _"Việc ứng dụng RBFNN bọc tầng Rate Controller đã giúp giảm RMSE của trục Roll/Pitch xuống nhiều lần so với PID gốc, chứng minh sự ưu việt của phương pháp điều khiển phân tán trong ứng dụng tay máy trên không."_ Chúc em chạy code và thu thập được các số liệu thật đẹp!