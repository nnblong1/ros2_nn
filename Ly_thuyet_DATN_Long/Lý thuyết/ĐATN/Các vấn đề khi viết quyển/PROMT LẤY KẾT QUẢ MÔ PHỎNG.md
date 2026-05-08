Chào em. Việc em đã hoàn thiện phần mã nguồn (code) cho thuật toán RBFNN nhúng vào tầng Rate Controller là một tiến độ rất đáng khen ngợi. Chuyển sang giai đoạn Đánh giá kết quả (Evaluation), em cần nhớ một nguyên tắc trong khoa học điều khiển: **Thuật toán dù hay đến đâu bằng toán học cũng vô nghĩa nếu không được chứng minh bằng các con số thực nghiệm định lượng.**

Dưới góc độ hướng dẫn, thầy sẽ chỉ ra chính xác những dữ liệu em cần lấy, cách thiết kế kịch bản thử nghiệm trên PX4, và cung cấp một "Siêu Prompt" để em nhờ AI (ChatGPT/Claude) viết script phân tích file log (`.ulog`) xuất ra từ PX4.

---

### 1. Vấn đề gì cần lấy kết quả (Đại lượng cần đo lường)?

Để chứng minh "RBFNN-based Rate Controller" của em ưu việt hơn bộ điều khiển gốc của PX4, em cần trích xuất các dữ liệu sau từ bộ ghi log của PX4 (`logger`):

- **Nhóm 1: Sai số bám quỹ đạo và Tư thế (So sánh với PX4 gốc):**
    - _Tọa độ (Position):_ $x, y, z$ thực tế so với $x_d, y_d, z_d$ mục tiêu.
    - _Góc nghiêng (Attitude):_ Roll ($\phi$), Pitch ($\theta$), Yaw ($\psi$) thực tế so với góc mục tiêu do vòng lặp ngoài cấp xuống.
    - _Mục đích:_ Chứng minh rằng khi Rate Controller bên trong cứng cáp nhờ AI, máy bay không bị trôi vị trí (drift) ở bên ngoài.
- **Nhóm 2: Tín hiệu nội bộ của Mạng Nơ-ron (Bắt buộc phải có để minh chứng AI hoạt động):**
    - _Đầu ra RBFNN:_ Tín hiệu mô-men bù trừ $\hat{f}_{add_\phi}, \hat{f}_{add_\theta}, \hat{f}_{add_\psi}$ xuất ra từ mạng Nơ-ron theo thời gian.
    - _Mục đích:_ Chứng minh AI thực sự đang "học" và xuất ra tín hiệu phản ứng lại với nhiễu động lực học, thay vì chỉ là một cục code chạy ngầm vô tác dụng.
- **Nhóm 3: Phân tích Lỗi Định lượng (Quantitative Metrics):**
    - Tính toán **RMSE (Root Mean Square Error)** và **Maximum Error** cho 6 kênh $(x, y, z, \phi, \theta, \psi)$ để lập thành bảng.

### 2. Thực hiện kịch bản thí nghiệm (Test Scenarios) như thế nào?

Em không thể cho máy bay bay lơ lửng bình thường rồi lấy kết quả được. Em phải "hành hạ" hệ thống để xem AI chống chịu ra sao. Trên môi trường PX4 SITL (hoặc bay thực tế), hãy thực hiện 2 kịch bản giống như bài báo tham khảo:

- **Kịch bản 1: Nhiễu ghép nối động lực học liên tục (Coupling Disturbance).**
    - _Thực hiện:_ Cho UAV cất cánh và giữ lơ lửng (Hovering) ở độ cao $1m$. Bắt đầu từ giây thứ 10, gửi lệnh cho cánh tay robot múa các khớp theo dạng sóng Sin (ví dụ: $q_2 = \frac{\pi}{3}\sin(\frac{\pi}{10}t)$).
    - _Ý nghĩa:_ Tay máy vung vẩy sẽ sinh ra lực lật liên tục. Ta sẽ xem PX4 gốc bị rung lắc thế nào và RBFNN của em khóa chặt góc nghiêng ra sao.
- **Kịch bản 2: Nhiễu ngoại lực đột ngột (Step Disturbance / Payload Change).**
    - _Thực hiện:_ Đang bay lơ lửng, tại giây thứ 15, mô phỏng việc tay máy đột ngột gắp một vật nặng (ví dụ 375g, tương đương nạp thêm một xung lực $3.75N$ vào trục Z và tạo ra mô-men lệch tâm).
    - _Ý nghĩa:_ Chứng minh mạng Nơ-ron có khả năng cập nhật trọng số tức thời (Online Learning) để gánh thêm phần khối lượng chưa được khai báo trong mô hình toán học ban đầu.

_Lưu ý thao tác:_ Chạy kịch bản trên với firmware PX4 mặc định $\rightarrow$ Tải file `.ulog` về. Sau đó bật Custom Flight Mode chứa RBFNN của em $\rightarrow$ Chạy lại đúng kịch bản đó $\rightarrow$ Tải file `.ulog` thứ hai về.

---

### 3. Prompt cho AI viết Code phân tích kết quả

Vì log của PX4 là định dạng `.ulog`, việc bóc tách dữ liệu khá thủ công. Thầy đã thiết kế sẵn cho em một Prompt chuẩn kỹ thuật. Em hãy copy nguyên văn đoạn tiếng Anh dưới đây gửi cho ChatGPT/Claude để nó viết một Script Python tự động đọc file log, tính RMSE và vẽ đồ thị nét liền/nét đứt chuẩn bài báo IEEE.

**[BẮT ĐẦU COPY TỪ ĐÂY]**

**Role & Task:** You are a Robotics Data Scientist specializing in PX4 Autopilot log analysis. My graduation thesis involves comparing the native PX4 PID Rate Controller against my custom "RBFNN-based Rate Controller" for an Unmanned Aerial Manipulator (UAM). I have conducted SITL simulations and extracted two `.ulog` files (or `.csv` converted from ulog):

1. `px4_native_log.ulog`
2. `rbfnn_custom_log.ulog`

Write a comprehensive Python script using `pyulog`, `pandas`, `numpy`, and `matplotlib` to parse these logs, compute quantitative metrics, and plot publication-quality graphs replicating the style of IEEE papers.

**1. Data Parsing Requirements:**

- Extract Attitude data: Roll, Pitch, Yaw (actual vs. setpoint) from `vehicle_attitude` and `vehicle_attitude_setpoint`.
- Extract Local Position data: X, Y, Z (actual vs. setpoint) from `vehicle_local_position` and `vehicle_local_position_setpoint`.
- _(Only from rbfnn_custom_log)_: Extract the custom Neural Network outputs (e.g., logged via a custom uORB topic or debug arrays) which represent the learned compensating torques for Roll, Pitch, Yaw.

**2. Plotting Requirements (Publication Quality):** Create the following subplots with strict formatting (Times New Roman font, grid, legends, X-axis as Time in seconds):

- **Plot 1 (Position Response - like Fig 8a,b,c):** A 3x1 subplot (X, Y, Z). Plot the `Setpoint` (solid black line), `PX4 Native Response` (dashed gray line, thick), and `RBFNN Response` (solid blue line).
- **Plot 2 (Attitude Response - like Fig 8d,e,f):** A 3x1 subplot (Roll, Pitch, Yaw). Same line styles as Plot 1.
- **Plot 3 (NN Outputs - like Fig 9c):** A 1x1 plot showing the 3 outputs of the RBFNN torques over time to prove the network is reacting to disturbances.

**3. Quantitative Metrics (RMSE Table):** Calculate the Root Mean Square Error (RMSE) and Maximum Error for Position (X,Y,Z) and Attitude (Roll, Pitch, Yaw) for both the `px4_native_log` and `rbfnn_custom_log`.

- Time-align the setpoint and actual measurements before computing the error.
- Print the results in a formatted markdown table in the console (like Table V: Mean, Max, RMSE).

**Code Structure:** Use Object-Oriented Programming (OOP) with a class `PX4LogAnalyzer`. Handle time synchronization gracefully since uORB topics log at different frequencies. Include extensive comments.

**[KẾT THÚC COPY]**

---

**Lời khuyên cuối của thầy:** Khi AI trả về code Python, em chỉ cần cài đặt thư viện `pyulog` (`pip install pyulog`) và đưa 2 file log của em vào chung thư mục chứa script. Script này sẽ tự động sinh ra đồ thị đè lên nhau (Native vs Custom) và bảng RMSE. Em chỉ cần chụp màn hình đồ thị và dán thẳng bảng RMSE vào Chương 5: Đánh giá Kết quả của đồ án, lập luận rằng: _"Nhờ sự can thiệp của RBFNN ở tầng Rate, RMSE của góc Roll/Pitch đã giảm đi X%, giúp vị trí X,Y ổn định khi tay máy cử động, chứng minh tính đúng đắn của phương pháp"_. Chúc em ra được kết quả đẹp!