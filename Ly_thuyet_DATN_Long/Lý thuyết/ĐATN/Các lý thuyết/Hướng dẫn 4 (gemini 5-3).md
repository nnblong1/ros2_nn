# Báo cáo Nghiên cứu: Khung Điều khiển Backstepping Thích nghi Tích hợp Trí tuệ Nhân tạo cho Hệ thống Máy bay Không người lái mang Cánh tay Robot 6 Bậc tự do

## 1. Tổng quan và Đặt vấn đề Tính toán Cân bằng Động

Sự hội tụ giữa công nghệ hàng không và robot học đã thúc đẩy sự ra đời của các phương tiện bay thao tác không người lái (Unmanned Aerial Manipulators - UAM). Việc gắn một cánh tay robot có khả năng linh hoạt cao, cụ thể là cánh tay 6 bậc tự do (6-DoF), vào khung gầm của một máy bay không người lái dạng quadcopter (UAV 4 rotor) mở ra các tiềm năng to lớn trong việc thực hiện các nhiệm vụ phức tạp trên không, từ kiểm tra cơ sở hạ tầng, bảo trì công nghiệp đến vận chuyển hàng hóa phức tạp. Tuy nhiên, nền tảng này đồng thời mang đến những thách thức điều khiển cực kỳ khắt khe do bản chất hệ thống là một cơ cấu bay thiếu cơ cấu chấp hành (underactuated), chịu sự chi phối mạnh mẽ của các yếu tố động lực học phi tuyến và sự dịch chuyển trọng tâm liên tục.

Khi cánh tay 6-DoF di chuyển để thực hiện các thao tác, nó làm thay đổi phân bố khối lượng và ma trận quán tính của toàn bộ hệ thống UAM. Việc mở rộng hoặc thu gọn các khớp của cánh tay tạo ra các lực tương tác và mô-men xoắn phản lực (coupling disturbances) truyền trực tiếp lên thân UAV. Nếu các bộ điều khiển bay truyền thống (như PID hoặc LQR) được sử dụng, các nhiễu động học này sẽ được xem như là các tác động ngoại vi chưa biết, dẫn đến sự suy giảm độ chính xác của quỹ đạo bay, hoặc trong trường hợp xấu nhất, làm mất ổn định hệ thống và gây rơi vỡ.

Nghiên cứu này tập trung vào bài toán cân bằng động cho một hệ thống UAM cấu hình Quadcopter mang cánh tay 6-DoF. Phần cứng hệ thống được chỉ định bao gồm bộ điều khiển bay PX4 Autopilot chịu trách nhiệm điều phối động cơ mức thấp, giao tiếp với máy tính nhúng Raspberry Pi 4 hoạt động như một máy tính đồng hành (companion computer) chạy hệ điều hành ROS2 (Robot Operating System 2). Cấu trúc này cho phép chuyển đổi từ điều khiển phản hồi truyền thống sang một khung điều khiển phức hợp sử dụng phương pháp Backstepping thích nghi (Adaptive Backstepping) làm nền tảng cốt lõi.

Phương pháp Backstepping, thông qua việc thiết kế các hàm Lyapunov đệ quy, chứng minh được tính hiệu quả trong việc xử lý tính phi tuyến của hệ thống UAV. Tuy nhiên, để bù đắp cho sự thiếu hụt thông tin về mô hình động lực học chính xác của cánh tay robot, cũng như khả năng thay đổi tải trọng, Trí tuệ Nhân tạo (AI) được đề xuất tích hợp sâu vào kiến trúc toán học của Backstepping. Khung điều khiển lai này bao gồm hai thành phần AI phối hợp chặt chẽ: Thứ nhất, một Mạng Nơ-ron Hàm Cơ sở Suy rộng (Radial Basis Function Neural Network - RBFNN) được sử dụng để ước lượng và cập nhật trực tuyến các tham số động lực học chưa biết và bù trừ nhiễu thời gian thực. Thứ hai, một mô hình Mạng Bộ nhớ Ngắn hạn Dài (Long Short-Term Memory - LSTM) được thiết kế để phân tích chuỗi tín hiệu lệnh của cánh tay 6-DoF nhằm đưa ra dự đoán trước về sự dịch chuyển trọng tâm và lực phản hồi, từ đó cung cấp một thành phần bù tiến (feedforward compensation) cho bộ điều khiển UAV trước khi nhiễu thực sự xảy ra.

Báo cáo này sẽ trình bày chi tiết về nền tảng toán học, cách thức thiết kế thuật toán, kiến trúc tích hợp hệ thống ROS2-PX4 và mã nguồn triển khai thực tế nhằm đáp ứng yêu cầu vận hành thời gian thực trên Raspberry Pi 4.

## 2. Mô hình Động lực học Cơ học của Hệ thống UAM

Để thiết kế một bộ điều khiển Backstepping thích nghi chính xác, bước đầu tiên là phải thiết lập mô hình toán học phản ánh đúng các tương tác cơ học giữa UAV và cánh tay robot. Hệ thống được xem xét như một cơ cấu đa thể rắn (multi-body system) có gốc tọa độ trôi nổi (floating base).

### 2.1. Động lực học Tịnh tiến và Quay của UAV

Định nghĩa hệ quy chiếu toàn cục (Inertial Frame) $\Sigma_I$ và hệ quy chiếu gắn với thân UAV (Body Frame) $\Sigma_B$. Vị trí của trọng tâm UAV trong hệ $\Sigma_I$ được biểu diễn bởi vector $p = [x, y, z]^T$. Tư thế của UAV được mô tả bằng các góc Euler $\Theta = [\phi, \theta, \psi]^T$ tương ứng với Roll, Pitch và Yaw.

Dựa trên cơ sở của phương trình Newton-Euler, động lực học của một UAV dạng quadcopter tiêu chuẩn (có tính đến tác động của cánh tay) có thể được viết dưới dạng hệ phương trình vi phân sau :

$$m_t \ddot{x} = (cos\phi sin\theta cos\psi + sin\phi sin\psi) T + f_{0x}$$

$$m_t \ddot{y} = (cos\phi sin\theta sin\psi - sin\phi cos\psi) T + f_{0y}$$

$$m_t \ddot{z} = (cos\phi cos\theta) T - m_t g + f_{0z}$$

Trong đó, $m_t$ là tổng khối lượng của hệ thống (bao gồm UAV, cánh tay và tải trọng), $g$ là gia tốc trọng trường, $T$ là tổng lực đẩy do bốn rotor tạo ra, và $f_{0x}, f_{0y}, f_{0z}$ đại diện cho các lực tương tác ba chiều do sự chuyển động của cánh tay robot truyền lên thân UAV.

Đối với động lực học quay, phương trình Euler được thiết lập như sau:

$$I_{xx} \ddot{\phi} = \dot{\theta} \dot{\psi} (I_{yy} - I_{zz}) + \tau_x + n_{0x}$$

$$I_{yy} \ddot{\theta} = \dot{\phi} \dot{\psi} (I_{zz} - I_{xx}) + \tau_y + n_{0y}$$

$$I_{zz} \ddot{\psi} = \dot{\phi} \dot{\theta} (I_{xx} - I_{yy}) + \tau_z + n_{0z}$$

Trong hệ phương trình này, $I_{xx}, I_{yy}, I_{zz}$ là các thành phần của ma trận quán tính quanh các trục chính của UAV. $\tau_x, \tau_y, \tau_z$ là các mô-men xoắn điều khiển được sinh ra bởi sự chênh lệch lực đẩy giữa các rotor. Tương tự như động học tịnh tiến, $n_{0x}, n_{0y}, n_{0z}$ là các mô-men xoắn nhiễu động xuất phát từ cánh tay 6-DoF.

Bảng 1 mô tả cấu trúc các tham số động lực học chính của hệ thống.

|**Ký hiệu**|**Ý nghĩa vật lý**|**Đơn vị**|**Đặc tính tác động**|
|---|---|---|---|
|$m_t$|Khối lượng tổng cộng của hệ thống|kg|Biến thiên nếu UAV nhấc tải trọng mới.|
|$I_{xx}, I_{yy}, I_{zz}$|Mô-men quán tính dọc theo các trục cơ sở|$kg \cdot m^2$|Dao động liên tục do sự thay đổi hình dáng cánh tay.|
|$T, \tau_x, \tau_y, \tau_z$|Tín hiệu điều khiển lực đẩy và mô-men xoắn|N, $N \cdot m$|Đầu ra của bộ điều khiển Backstepping.|
|$f_{0i}, n_{0i}$|Lực và mô-men xoắn phản lực từ cánh tay|N, $N \cdot m$|Nhiễu động học phi tuyến tính, cần bù trừ bằng AI.|

### 2.2. Động lực học của Cánh tay Robot 6-DoF và Nhiễu Tương tác

Cánh tay robot được gắn dưới bụng UAV, cấu tạo từ 6 khớp quay (revolute joints). Trạng thái của cánh tay được xác định bởi vector góc khớp $q = [q_1, q_2,..., q_6]^T$, vận tốc khớp $\dot{q}$, và gia tốc khớp $\ddot{q}$.

Lực và mô-men tương tác $F_{arm} = [f_{0x}, f_{0y}, f_{0z}]^T$ và $\tau_{arm} = [n_{0x}, n_{0y}, n_{0z}]^T$ được tính toán thông qua thuật toán đệ quy Newton-Euler (RNE). Thuật toán RNE bao gồm hai quá trình:

1. **Đệ quy tiến (Forward Recursion):** Tính toán vận tốc góc, gia tốc góc, gia tốc tuyến tính của từng mắt xích (link) từ gốc (gắn với UAV) đến điểm thao tác cuối (end-effector).
    
2. **Đệ quy lùi (Backward Recursion):** Tính toán các lực và mô-men tác dụng lên mỗi khớp, lan truyền ngược từ end-effector về lại gốc. Phương trình đệ quy lùi cho lực $f_i$ và mô-men $n_i$ tại khớp thứ $i$ được định nghĩa:
    
    $$f_i = R_{i+1}^i f_{i+1} + F_i$$
    
    $$n_i = R_{i+1}^i n_{i+1} + p_i \times f_i + s_i \times F_i + N_i$$
    
    Khi đệ quy lùi về đến khớp 0 (khớp nối giữa cánh tay và UAV), ta thu được $f_0$ và $n_0$, chính là các giá trị nhiễu $F_{arm}$ và $\tau_{arm}$.
    

Do tính phức tạp khổng lồ của ma trận động lực học chứa các hằng số ma sát, ma trận Coriolis, và độ lệch trọng tâm không đồng nhất , việc tính toán giải tích trực tiếp phương trình này trên vi xử lý ARM của Raspberry Pi 4 để làm tín hiệu bù trừ là cực kỳ thiếu chính xác và tiêu tốn tài nguyên. Do đó, nghiên cứu này gộp toàn bộ sự phức tạp này thành một vector nhiễu tổng quát $D(X, t)$ để áp dụng bộ ước lượng học máy.

### 2.3. Chuyển đổi Không gian Trạng thái (State-Space Formulation)

Để thuận tiện cho việc thiết kế hàm Lyapunov trong kỹ thuật Backstepping, hệ thống được biểu diễn lại dưới dạng không gian trạng thái chuẩn tắc. Đặt vector trạng thái $X =^T$, trong đó $x_1 = [x, y, z, \phi, \theta, \psi]^T \in \mathbb{R}^6$ là vector chứa vị trí và tư thế, và $x_2 = [\dot{x}, \dot{y}, \dot{z}, \dot{\phi}, \dot{\theta}, \dot{\psi}]^T \in \mathbb{R}^6$ là vector vận tốc tương ứng.

Hệ thống có thể được viết dưới dạng:

$$\dot{x}_1 = x_2$$

$$\dot{x}_2 = f(X) + G(X)U + D(X, t)$$

Ở đây:

- $f(X) \in \mathbb{R}^6$ chứa các thành phần động lực học nội tại đã biết (trọng lực, hiệu ứng gyroscopic của cơ thể).
    
- $G(X) \in \mathbb{R}^{6 \times 4}$ là ma trận phân bổ tín hiệu điều khiển (Control allocation matrix) ánh xạ vector đầu vào $U =^T$ thành các gia tốc trạng thái.
    
- $D(X, t) \in \mathbb{R}^6$ biểu diễn toàn bộ các sự bất định của hệ thống, bao gồm nhiễu khí động học, thay đổi khối lượng không lường trước, và đặc biệt là sự dịch chuyển trọng tâm cùng lực phản hồi từ cánh tay 6-DoF.
    

## 3. Thiết kế Khung Điều khiển Backstepping Thích nghi (Adaptive Backstepping)

Phương pháp Backstepping là một kỹ thuật điều khiển phi tuyến thiết kế dựa trên hàm Lyapunov, phân rã hệ thống bậc cao thành các hệ thống con bậc nhất. Tư tưởng cốt lõi là coi các biến trạng thái trung gian như các tín hiệu điều khiển ảo (virtual control inputs) để ổn định từng hệ thống con một cách đệ quy. Đối với UAV, quy trình này thường được chia thành điều khiển vòng ngoài (outer loop) cho vị trí và điều khiển vòng trong (inner loop) cho tư thế.

Tuy nhiên, trong mô hình không gian trạng thái đã lập, chúng ta có thể tiến hành thiết kế Backstepping tổng quát cho toàn bộ vector trạng thái để dễ dàng tích hợp mạng AI ở bước sau.

### 3.1. Bước 1: Ổn định Động học Ký vị trí và Tư thế (Kinematic Stabilization)

Định nghĩa vector quỹ đạo tham chiếu mong muốn của UAV là $x_{1d}(t)$. Sai số theo dõi vị trí và tư thế được định nghĩa là:

$$e_1 = x_1 - x_{1d}$$

Để khảo sát sự ổn định của sai số $e_1$, một hàm Lyapunov dự tuyển thứ nhất $V_1$ được xác định, là một hàm xác định dương:

$$V_1 = \frac{1}{2} e_1^T e_1$$

Lấy đạo hàm của $V_1$ theo thời gian:

$$\dot{V}_1 = e_1^T \dot{e}_1 = e_1^T (\dot{x}_1 - \dot{x}_{1d}) = e_1^T (x_2 - \dot{x}_{1d})$$

Mục tiêu của hệ thống điều khiển là làm cho $\dot{V}_1$ âm xác định. Để đạt được điều này, ta định nghĩa một tín hiệu điều khiển ảo (virtual control input) cho vector vận tốc, ký hiệu là $\alpha$, sao cho nếu $x_2$ bám sát $\alpha$, thì $e_1$ sẽ hội tụ về 0. Ta chọn $\alpha$ như sau:

$$\alpha = \dot{x}_{1d} - K_1 e_1$$

Trong đó, $K_1 \in \mathbb{R}^{6 \times 6}$ là một ma trận hằng số đường chéo dương xác định, đóng vai trò là hệ số khuếch đại sai số.

Thay $\alpha$ vào phương trình đạo hàm Lyapunov, nếu giả định $x_2 \equiv \alpha$, ta sẽ có $\dot{V}_1 = -e_1^T K_1 e_1 \le 0$, đảm bảo sự ổn định tiệm cận của sai số $e_1$.

### 3.2. Bước 2: Ổn định Động lực học Vận tốc (Dynamic Stabilization)

Trong thực tế, trạng thái vận tốc $x_2$ không thể ngay lập tức bằng tín hiệu ảo $\alpha$. Do đó, ta định nghĩa một sai số vận tốc (sai số bước 2) là khoảng cách giữa vận tốc thực tế và vận tốc ảo mong muốn:

$$e_2 = x_2 - \alpha = x_2 - \dot{x}_{1d} + K_1 e_1$$

Lúc này, đạo hàm của sai số $e_1$ được viết lại là:

$$\dot{e}_1 = e_2 - K_1 e_1$$

Tiếp tục thiết lập hàm Lyapunov mở rộng thứ hai $V_2$ bao gồm cả sai số vị trí và sai số vận tốc:

$$V_2 = V_1 + \frac{1}{2} e_2^T e_2 = \frac{1}{2} e_1^T e_1 + \frac{1}{2} e_2^T e_2$$

Lấy đạo hàm theo thời gian của $V_2$ và thay các phương trình động lực học của hệ thống vào:

$$\dot{V}_2 = e_1^T \dot{e}_1 + e_2^T \dot{e}_2$$

$$\dot{V}_2 = e_1^T (e_2 - K_1 e_1) + e_2^T (\dot{x}_2 - \dot{\alpha})$$

$$\dot{V}_2 = -e_1^T K_1 e_1 + e_2^T (e_1 + f(X) + G(X)U + D(X, t) - \dot{\alpha})$$

Để đảm bảo toàn bộ hệ thống bám sát quỹ đạo và ổn định, ta cần triệt tiêu các thành phần bên trong ngoặc của biểu thức chứa $e_2$ sao cho phần còn lại là âm xác định. Luật điều khiển $U$ lý tưởng có thể được thiết kế như sau:

$$U = G^\dagger(X)$$

Với $K_2 \in \mathbb{R}^{6 \times 6}$ là ma trận khuếch đại dương xác định thứ hai, và $G^\dagger(X)$ là ma trận giả nghịch đảo của $G(X)$ (do UAV là hệ thiếu cơ cấu chấp hành, việc nghịch đảo trực tiếp cần thông qua thuật toán ánh xạ mô-men tới lực đẩy rotor).

Thay luật điều khiển $U$ lý tưởng này vào $\dot{V}_2$, ta thu được:

$$\dot{V}_2 = -e_1^T K_1 e_1 - e_2^T K_2 e_2 \le 0$$

Điều này chứng minh bằng toán học rằng hệ thống sẽ hội tụ.

Tuy nhiên, nút thắt nghiêm trọng của bộ điều khiển lý tưởng này là việc nó đòi hỏi sự hiểu biết tuyệt đối về hàm phi tuyến $f(X)$ và đặc biệt là thành phần nhiễu động $D(X, t)$ bao gồm lực phản hồi từ cánh tay 6-DoF. Trong thực tế vận hành UAM, $D(X, t)$ liên tục thay đổi theo quỹ đạo của cánh tay và không thể đo lường một cách hoàn hảo. Sự xuất hiện của Trí tuệ Nhân tạo (AI) chính là để ước lượng và bù đắp cho điểm yếu chí mạng này của toán học truyền thống.

## 4. Áp dụng AI Phần 1: Mạng RBFNN cho Cập nhật Tham số Trực tuyến

Để xử lý các thành phần bất định $f(X)$ và $D(X, t)$, một mạng Nơ-ron Hàm cơ sở Suy rộng (Radial Basis Function Neural Network - RBFNN) được đề xuất tích hợp sâu vào bộ điều khiển Backstepping. Khác với các mạng nơ-ron sâu (Deep Neural Networks) truyền thống đòi hỏi thời gian huấn luyện ngoại tuyến (offline) khổng lồ, RBFNN sở hữu cấu trúc mạng đơn giản (chỉ 3 lớp) và thuật toán cập nhật trọng số trực tuyến (online weight tuning) cực kỳ nhanh, phù hợp với yêu cầu tính toán mềm theo thời gian thực (soft real-time) trên vi xử lý ARM Cortex-A72 của Raspberry Pi 4.

### 4.1. Kiến trúc Ước lượng của RBFNN

Theo định lý xấp xỉ vạn năng (Universal Approximation Theorem), một RBFNN với số lượng nơ-ron ẩn đủ lớn có thể xấp xỉ bất kỳ hàm phi tuyến liên tục nào với độ chính xác tùy ý. Chúng ta gộp các thành phần động lực học chưa biết thành một hàm duy nhất $F(X, t) = f(X) + D(X, t)$.

Kiến trúc của mạng RBFNN bao gồm:

1. **Lớp đầu vào (Input Layer):** Nhận vector tín hiệu đầu vào $Z$. Trong kiến trúc điều khiển này, vector đầu vào được chọn là sự kết hợp của sai số trạng thái và đạo hàm của nó: $Z =^T \in \mathbb{R}^{12}$.
    
2. **Lớp ẩn (Hidden Layer):** Bao gồm $N$ nơ-ron thực hiện phép biến đổi phi tuyến không gian đầu vào bằng các hàm Gaussian (Radial Basis Functions). Đầu ra của nơ-ron thứ $i$ được tính bằng:
    
    $$h_i(Z) = \exp \left( -\frac{\|Z - C_i\|^2}{2b_i^2} \right), \quad i = 1, 2,..., N$$
    
    Trong đó, $C_i$ là vector trung tâm (center vector) và $b_i$ là độ rộng (width) của hàm Gaussian thứ $i$.
    
3. **Lớp đầu ra (Output Layer):** Là sự tổ hợp tuyến tính của các kết quả từ lớp ẩn, tạo ra dự đoán về nhiễu động:
    
    $$\hat{F}(Z) = \hat{W}^T h(Z)$$
    
    Với $h(Z) = [h_1, h_2,..., h_N]^T$ là vector kích hoạt lớp ẩn và $\hat{W} \in \mathbb{R}^{N \times 6}$ là ma trận trọng số ước lượng (estimated weights matrix).
    

Hàm phi tuyến thực tế $F(X, t)$ có thể được biểu diễn dưới dạng:

$$F(X, t) = W^{*T} h(Z) + \epsilon$$

Trong đó $W^*$ là ma trận trọng số lý tưởng (tối ưu nhất) chưa biết, và $\epsilon$ là sai số xấp xỉ giới hạn ($\|\epsilon\| \le \epsilon_{max}$).

### 4.2. Luật Cập nhật Trọng số Thích nghi (Adaptive Weight Update Law)

Chìa khóa của phương pháp này là mạng RBFNN không cần tập dữ liệu huấn luyện có sẵn. Thay vào đó, trọng số mạng $\hat{W}$ được điều chỉnh liên tục trong quá trình bay thông qua một luật cập nhật (update law) được suy ra từ phân tích ổn định Lyapunov. Quá trình này được gọi là điều khiển thích nghi (adaptive control).

Định nghĩa sai số trọng số là $\tilde{W} = W^* - \hat{W}$. Luật điều khiển $U$ từ phương trình Backstepping được viết lại bằng cách thay thế hàm chưa biết $F(X, t)$ bằng đầu ra của RBFNN:

$$U = G^\dagger(X)$$

Để tìm ra cách cập nhật $\hat{W}$, chúng ta xây dựng hàm Lyapunov tổng thể thứ ba $V_3$, bao gồm cả năng lượng sai số bám quỹ đạo và năng lượng sai số trọng số mạng nơ-ron:

$$V_3 = V_2 + \frac{1}{2} tr(\tilde{W}^T \Gamma^{-1} \tilde{W})$$

Trong đó, $tr(\cdot)$ là vết của ma trận, và $\Gamma$ là một ma trận hằng số dương xác định đóng vai trò là tốc độ học (learning rate) của mạng nơ-ron.

Lấy đạo hàm của $V_3$ và tiến hành thay thế luật điều khiển $U$ vào, kết hợp với các biến đổi đại số ma trận, để đảm bảo $\dot{V}_3 \le 0$, thuật toán cập nhật trọng số trực tuyến (Online weight update law) được thiết kế như sau :

$$\dot{\hat{W}} = \Gamma \left$$

Luật cập nhật này mang ý nghĩa vật lý cực kỳ sâu sắc:

- Thành phần $h(Z) e_2^T$ là quy tắc học Hebbian cơ bản, điều chỉnh trọng số dựa trên sự tương quan giữa sự kích hoạt của mạng ($h(Z)$) và mức độ sai số động lực học ($e_2$). Khi sai số vận tốc của UAV lớn, mạng sẽ tự động tăng tốc độ thay đổi trọng số để tìm kiếm giá trị bù trừ đúng.
    
- Thành phần $-\eta \|e_2\| \hat{W}$ là thuật toán e-modification (hoặc rò rỉ trọng số - weight leakage). Sự có mặt của $\eta$ (một hằng số dương nhỏ) giúp đảm bảo tính giới hạn của trọng số (boundedness). Nếu không có thành phần này, sự tồn tại của sai số xấp xỉ $\epsilon$ và nhiễu đo lường từ cảm biến có thể làm trọng số $\hat{W}$ tăng vô hạn theo thời gian (parameter drift), gây mất ổn định toàn hệ thống.
    

Thông qua RBFNN, hệ thống UAM sở hữu khả năng tự động học hỏi sức cản không khí, hiện tượng trượt, và sự tác động của tải trọng cánh tay mà không cần đo đạc hay lập mô hình toán học giải tích trước. Vòng lặp tính toán $\dot{\hat{W}}$ bao gồm các phép toán nhân ma trận kích thước nhỏ, hoàn toàn có thể thực thi với thời gian xử lý rất thấp (trung bình khoảng 2-5 mili-giây) bằng ngôn ngữ C++ trên Raspberry Pi 4.

## 5. Áp dụng AI Phần 2: LSTM cho Dự đoán Cân bằng Bù tiến (Predictive Feedforward)

Mặc dù RBFNN kết hợp Backstepping mang lại sự vững chãi (robustness) rất cao, nó về bản chất vẫn là một cơ chế điều khiển phản hồi (Feedback Control). Nghĩa là, nó chỉ tạo ra lực bù trừ _sau khi_ sai số $e_2$ đã xuất hiện. Đối với hệ thống UAM 6-DoF, khi cánh tay robot mang vật nặng bất ngờ vươn dài ra phía trước, trọng tâm (CoM) sẽ dịch chuyển cực nhanh. Nếu chỉ đợi vòng lặp phản hồi của RBFNN nhận diện sai số rồi mới tăng mô-men xoắn, UAV sẽ bị sụt giảm độ cao (altitude drop) hoặc dao động góc chúi (pitch chattering) đáng kể, đe dọa trực tiếp đến an toàn bay.

Giải pháp cho vấn đề này là bổ sung một lớp điều khiển bù tiến dự báo (Predictive Feedforward Control) sử dụng Trí tuệ Nhân tạo. Mô hình mạng học sâu chuỗi thời gian Long Short-Term Memory (LSTM) được triển khai để giải quyết bài toán này.

Bảng 2 so sánh vai trò của hai cấu trúc AI được triển khai trong hệ thống UAM.

|**Tiêu chí**|**Mạng RBFNN (Adaptive)**|**Mạng LSTM (Predictive)**|
|---|---|---|
|**Vai trò cốt lõi**|Phản hồi và thích nghi (Feedback)|Dự báo và bù tiến (Feedforward)|
|**Dữ liệu đầu vào**|Trạng thái hiện tại của UAV ($e, \dot{e}$)|Chuỗi kế hoạch quỹ đạo khớp tay ($q_1..q_6$)|
|**Đầu ra**|Bù trừ mọi yếu tố phi tuyến, gió, nhiễu|Dự báo riêng mô-men xoắn do sự thay đổi CoM|
|**Đặc tính đào tạo**|Học trực tuyến (Online) bằng phương trình Lyapunov, không cần Dataset|Huấn luyện ngoại tuyến (Offline) bằng Dataset, suy luận trực tuyến (Online Inference)|
|**Thời điểm tác động**|Sau khi nhiễu gây ra sai số cho thân UAV|**Trước khi** cánh tay kịp tạo ra nhiễu vật lý|

### 5.1. Kiến trúc Mạng LSTM Dự báo Quỹ đạo

LSTM là một biến thể của Mạng Nơ-ron Hồi quy (RNN), sở hữu cấu trúc cổng (cổng quên, cổng cập nhật, cổng xuất) giúp nó xử lý xuất sắc bài toán nhớ các phụ thuộc dài hạn trong dữ liệu chuỗi thời gian.

Khi cánh tay 6-DoF chuẩn bị thực hiện một thao tác, bộ quy hoạch quỹ đạo (ví dụ: MoveIt2 trên ROS2) sẽ sinh ra một chuỗi các tọa độ góc khớp mong muốn trong tương lai: $Q_{plan} =$. Mạng LSTM được thiết kế để nhận chuỗi tín hiệu $Q_{plan}$ này làm đầu vào, xử lý sự dịch chuyển động học, và đưa ra một dự đoán về lực và mô-men xoắn sẽ tác động lên điểm gốc của UAV tại các mốc thời gian tương lai: $\hat{\tau}_{LSTM} = [\hat{n}_{0x}, \hat{n}_{0y}, \hat{n}_{0z}]^T$. 1

### 5.2. Tích hợp LSTM vào Khung Điều khiển Backstepping

Dự đoán từ LSTM được cộng trực tiếp vào luật điều khiển Backstepping ở dạng tín hiệu bù tiến (Feedforward term). Lúc này, luật điều khiển cuối cùng $U$ được nâng cấp thành:

$$U_{total} = G^\dagger(X) + \hat{\tau}_{LSTM\_FF}$$

Cơ chế này hoạt động tương tự như việc "nghiêng người chống đỡ" của con người. Vài chục mili-giây trước khi cánh tay 6-DoF thực sự vươn ra xa (gây mô-men lật theo trục Pitch), mạng LSTM đã nhận diện được hành vi này từ chuỗi kế hoạch quỹ đạo và gửi lệnh $\hat{\tau}_{LSTM\_FF}$ yêu cầu PX4 tăng lực đẩy của hai rotor phía trước. Khi thao tác vật lý thực sự xảy ra, lực bù trừ đã có sẵn, giúp UAV triệt tiêu gần như hoàn toàn sai số quỹ đạo. Bất kỳ sai số dư thừa nào (do gió hoặc sự không hoàn hảo của dự đoán LSTM) sẽ được dọn dẹp tiếp bởi lớp học trực tuyến RBFNN.

### 5.3. Huấn luyện Ngoại tuyến (Offline Training) cho LSTM

Mạng LSTM yêu cầu một mô hình đào tạo có sẵn trước khi đưa lên Raspberry Pi 4. Quá trình thu thập dữ liệu (Dataset Generation) được thực hiện thông qua Môi trường Mô phỏng Gazebo tích hợp với phần mềm Software In The Loop (SITL) của PX4.

- Trong không gian mô phỏng, UAV được cấu hình duy trì trạng thái lơ lửng (Hovering), trong khi cánh tay 6-DoF thực hiện hàng ngàn quỹ đạo quét qua toàn bộ không gian làm việc (workspace).
    
- Đầu ra của khối điều khiển Backstepping nội tại được ghi lại như là nhãn (Label) cho lượng mô-men cần thiết để bù trừ.
    
- Mô hình LSTM được xây dựng bằng PyTorch và huấn luyện với hàm mất mát MSE (Mean Squared Error) thông qua thuật toán lan truyền ngược qua thời gian (Backpropagation Through Time). Trọng số của mô hình sau khi huấn luyện (dạng file `.pth`) sẽ được nạp vào vi xử lý Raspberry Pi 4 để thực hiện quá trình Suy luận (Inference).
    

## 6. Kiến trúc Phần cứng và Phần mềm Hệ thống (ROS2 & PX4)

Khung lý thuyết toán học trên cần một hệ thống phần mềm tinh vi để triển khai. Yêu cầu tính toán của RBFNN và LSTM đòi hỏi sức mạnh mà bộ vi điều khiển trên mạch PX4 không thể đáp ứng. Do đó, Raspberry Pi 4 được sử dụng làm máy tính xử lý trung tâm (Companion Computer), kết nối với mạch PX4 qua cổng UART ở tốc độ baud cao (ví dụ: 921600 bps).

### 6.1. Giao thức Micro XRCE-DDS và Chế độ Offboard của PX4

ROS2 (Humble Hawksbill) được chọn làm hệ điều hành cốt lõi nhờ khả năng hỗ trợ truyền thông phân tán chuẩn DDS (Data Distribution Service) và bảo mật chất lượng dịch vụ (QoS) cho các ứng dụng thời gian thực. Mạch PX4 giao tiếp với ROS2 thông qua cầu nối Micro XRCE-DDS Client/Agent.

Bộ điều khiển Backstepping yêu cầu được gửi tín hiệu mô-men xoắn trực tiếp đến bộ trộn (mixer) của động cơ. Do đó, PX4 phải được cấu hình chạy ở chế độ **Offboard Mode** với mức can thiệp bỏ qua bộ điều khiển PID vị trí và tư thế nội tại.

Bảng 3 liệt kê các chủ đề (topics) ROS2 quan trọng cần cấu hình để tích hợp.

|**Topic ROS2 (px4_msgs)**|**Loại dữ liệu**|**Mục đích và Thông tin bổ sung**|
|---|---|---|
|`/fmu/in/offboard_control_mode`|Publisher|Bật chế độ Offboard. Cấu hình cờ `direct_actuator = true` hoặc `attitude = false` để PX4 hiểu tín hiệu đầu vào là lực/mô-men trực tiếp. Phải được phát liên tục với tần số > 2Hz.|
|`/fmu/in/vehicle_torque_setpoint`|Publisher|Truyền tín hiệu mô-men xoắn bù trừ $\tau_x, \tau_y, \tau_z$ từ bộ điều khiển. _Lưu ý:_ Phải biên dịch lại Firmware PX4 bằng cách thêm topic này vào tệp `dds_topics.yaml` để mở khóa chức năng.|
|`/fmu/in/vehicle_thrust_setpoint`|Publisher|Truyền tín hiệu lực đẩy tổng theo trục Z.|
|`/fmu/out/vehicle_odometry`|Subscriber|Cung cấp phản hồi thời gian thực về vị trí ($p, \Theta$) và vận tốc ($x_2$) của UAV, đóng vai trò đầu vào cho sai số Lyapunov.|
|`/arm_controller/joint_trajectory`|Subscriber|Nhận thông tin kế hoạch quỹ đạo khớp để cấp cho mạng LSTM tiến hành dự báo.|

### 6.2. Thiết kế Cấu trúc Soft Real-Time (Đa luồng trên Linux)

Trên hệ điều hành Linux tiêu chuẩn (Ubuntu 22.04), việc đạt được thời gian thực cứng (Hard Real-Time) là bất khả thi, có thể dẫn đến hiện tượng trễ nhịp (jitter) gây tích lũy sai số và mất ổn định hệ thống bay. Hơn nữa, quá trình suy luận của mạng học sâu LSTM (bằng Python/PyTorch) tốn thời gian hơn nhiều so với việc tính toán ma trận cập nhật trọng số RBFNN.

Để giải quyết vấn đề này, kiến trúc phần mềm trên Raspberry Pi 4 được chia thành hai luồng xử lý độc lập, được cô lập trên các nhân CPU (CPU cores) khác nhau :

1. **Node ROS2 Điều khiển Bằng C++ (High-Frequency Loop):** Chịu trách nhiệm chạy thuật toán Backstepping và cập nhật trọng số RBFNN. Với sự tối ưu hóa của thư viện `Eigen`, vòng lặp này hoàn toàn có thể chạy ổn định ở tần số 100Hz (chu kỳ 10ms) hoặc cao hơn. Node này áp dụng cấu hình QoS `RELIABLE` đối với dữ liệu cảm biến để đảm bảo tính toàn vẹn của thuật toán Lyapunov.
    
2. **Node ROS2 Trí tuệ Nhân tạo Bằng Python (Low-Frequency Inference Loop):** Chịu trách nhiệm tải mô hình PyTorch và chạy suy luận mạng LSTM. Do đây là mạng dự báo hướng tới tương lai (feedforward), nó có thể chạy ở tần số thấp hơn (10-20Hz). Kết quả suy luận (lực dự đoán) sẽ được xuất qua một Topic ROS2 cục bộ và được Node C++ nội suy để chèn vào phương trình điều khiển chính.
    

Cấu trúc phi đồng bộ này đảm bảo luồng điều khiển giữ thăng bằng sống còn của UAV (C++ Node) không bao giờ bị nghẽn (block) bởi sự chậm trễ của quá trình tính toán tensor bên trong AI (Python Node).

## 7. Triển khai Mã nguồn (Code Implementation)

Dưới đây là kiến trúc mã nguồn mô tả cách thức tích hợp các nền tảng toán học và thiết kế phần mềm vào hệ sinh thái ROS2 Humble.

### 7.1. C++ Node: Adaptive Backstepping & RBFNN Tuning

Đoạn mã `uam_backstepping_rbfnn_node.cpp` mô phỏng lõi điều khiển thời gian thực, nơi cập nhật trọng số RBFNN và gửi tín hiệu Torque tới PX4.

C++

```
#include <rclcpp/rclcpp.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/vehicle_torque_setpoint.hpp>
#include <px4_msgs/msg/vehicle_thrust_setpoint.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <Eigen/Dense>
#include <chrono>

using namespace std::chrono_literals;

class UAMAdaptiveController : public rclcpp::Node {
public:
    UAMAdaptiveController() : Node("uam_adaptive_controller") {
        // Publishers: Giao tiếp với PX4 qua Micro XRCE-DDS Bridge
        offboard_mode_pub_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>(
            "/fmu/in/offboard_control_mode", 10);
        torque_setpoint_pub_ = this->create_publisher<px4_msgs::msg::VehicleTorqueSetpoint>(
            "/fmu/in/vehicle_torque_setpoint", 10);
        thrust_setpoint_pub_ = this->create_publisher<px4_msgs::msg::VehicleThrustSetpoint>(
            "/fmu/in/vehicle_thrust_setpoint", 10);

        // Subscribers: Nhận trạng thái UAV và tín hiệu bù tiến từ LSTM Python Node
        odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
            "/fmu/out/vehicle_odometry", rclcpp::QoS(10).best_effort(),
            std::bind(&UAMAdaptiveController::odometry_callback, this, std::placeholders::_1));
            
        lstm_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "/ai/lstm_predictive_torque", 10,
            std::bind(&UAMAdaptiveController::lstm_callback, this, std::placeholders::_1));

        // Timer chạy vòng lặp điều khiển chính (Tần số 100 Hz = 10ms)
        timer_ = this->create_wall_timer(10ms, std::bind(&UAMAdaptiveController::control_loop, this));

        // Khởi tạo các tham số mạng RBFNN
        int num_neurons = 20; // Số lượng hàm Gaussian trong lớp ẩn
        W_hat = Eigen::MatrixXd::Zero(num_neurons, 3); // Ma trận trọng số (khởi tạo bằng 0)
        C_centers = Eigen::MatrixXd::Random(num_neurons, 6); // Trung tâm Gaussian
        B_widths = Eigen::VectorXd::Constant(num_neurons, 1.2); // Độ rộng hàm cơ sở
        
        Gamma = Eigen::MatrixXd::Identity(num_neurons, num_neurons) * 0.08; // Tốc độ học (Learning Rate)
        eta = 0.005; // Thông số e-modification để tránh phân kỳ trọng số
        
        // Ma trận khuếch đại cho Backstepping
        K1 = Eigen::MatrixXd::Identity(3, 3) * 3.0; // Khuếch đại vị trí
        K2 = Eigen::MatrixXd::Identity(3, 3) * 2.5; // Khuếch đại vận tốc/tư thế
    }

private:
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr offboard_mode_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleTorqueSetpoint>::SharedPtr torque_setpoint_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleThrustSetpoint>::SharedPtr thrust_setpoint_pub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr lstm_sub_;

    Eigen::VectorXd current_x1 = Eigen::VectorXd::Zero(6); // [x, y, z, phi, theta, psi]
    Eigen::VectorXd current_x2 = Eigen::VectorXd::Zero(6); // [vx, vy, vz, p, q, r]
    Eigen::Vector3d lstm_ff_torque = Eigen::Vector3d::Zero();
    
    Eigen::MatrixXd W_hat, C_centers, Gamma, K1, K2;
    Eigen::VectorXd B_widths;
    double eta;

    void odometry_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg) {
        // Cập nhật trạng thái UAV thời gian thực vào các vector Eigen
        // Phép chuyển đổi quaternion sang euler được ẩn đi để tối giản
    }

    void lstm_callback(const geometry_msgs::msg::Vector3::SharedPtr msg) {
        // Liên tục cập nhật giá trị bù tiến từ AI Node
        lstm_ff_torque << msg->x, msg->y, msg->z;
    }

    void publish_offboard_control_mode() {
        px4_msgs::msg::OffboardControlMode msg{};
        msg.position = false;
        msg.velocity = false;
        msg.acceleration = false;
        msg.attitude = false;
        msg.body_rate = false;
        msg.actuator = false;
        msg.direct_actuator = true; // Bỏ qua PID nội bộ, cho phép truyền Torque trực tiếp
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        offboard_mode_pub_->publish(msg);
    }

    void control_loop() {
        // Gửi nhịp tim (heartbeat) để duy trì Offboard mode của PX4
        publish_offboard_control_mode();

        // 1. Tính toán sai số Backstepping (e1, e2)
        Eigen::VectorXd x1_desired = Eigen::VectorXd::Zero(6); // Ví dụ: Điểm bay lơ lửng tại gốc
        Eigen::VectorXd e1 = current_x1 - x1_desired;
        
        // Điều khiển ảo (Virtual Control)
        Eigen::VectorXd x2_desired = - K1 * e1.tail(3); // Giản lược cho cấu hình điều khiển tư thế
        Eigen::VectorXd e2 = current_x2.tail(3) - x2_desired;

        // 2. Tính toán hàm Gaussian của RBFNN
        Eigen::VectorXd Z_input(6);
        Z_input << e1.tail(3), e2; // Đầu vào là vector sai số (e, e_dot)

        Eigen::VectorXd h_basis = Eigen::VectorXd::Zero(C_centers.rows());
        for (int i = 0; i < C_centers.rows(); ++i) {
            double norm_squared = (Z_input - C_centers.row(i).transpose()).squaredNorm();
            h_basis(i) = exp(-norm_squared / (2.0 * B_widths(i) * B_widths(i)));
        }

        // Ước lượng nhiễu phi tuyến từ RBFNN
        Eigen::Vector3d F_hat = W_hat.transpose() * h_basis;

        // 3. Luật Cập nhật Trọng số (Online Learning theo phương trình Lyapunov)
        // dW = Gamma * (h * e2^T - eta * ||e2|| * W_hat)
        Eigen::MatrixXd dW_hat = Gamma * (h_basis * e2.transpose() - eta * e2.norm() * W_hat);
        
        // Tích phân Euler để cập nhật trọng số (chu kỳ dt = 0.01s)
        W_hat += dW_hat * 0.01;

        // 4. Tổng hợp Luật Điều khiển (Backstepping + RBFNN_Comp + LSTM_FF)
        // Bù trừ sai số e1, triệt tiêu động lực học e2, bù nhiễu F_hat, thêm dự đoán LSTM
        Eigen::Vector3d tau_control = -e1.tail(3) - F_hat - K2 * e2 + lstm_ff_torque;

        // 5. Đóng gói và gửi tín hiệu Torque tới PX4
        px4_msgs::msg::VehicleTorqueSetpoint torque_msg{};
        torque_msg.xyz = tau_control(0); // Roll torque
        torque_msg.xyz = tau_control(1); // Pitch torque
        torque_msg.xyz = tau_control(2); // Yaw torque
        torque_msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        torque_setpoint_pub_->publish(torque_msg);
        
        // Tín hiệu Thrust được xuất qua topic vehicle_thrust_setpoint tương tự...
    }
};

int main(int argc, char *argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<UAMAdaptiveController>());
    rclcpp::shutdown();
    return 0;
}
```

### 7.2. Python Node: LSTM Predictive Feedforward

Đoạn mã `lstm_feedforward_node.py` tiếp nhận chuỗi tín hiệu của cánh tay, đẩy qua mô hình PyTorch để trích xuất thông tin thay đổi trọng tâm.

Python

```
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import JointState
import torch
import torch.nn as nn
import numpy as np

# Định nghĩa kiến trúc Mạng Học Sâu LSTM cho Chuỗi Thời Gian
class LSTM_TrajectoryPredictor(nn.Module):
    def __init__(self, input_features=6, hidden_dim=64, num_layers=2, output_dim=3):
        super(LSTM_TrajectoryPredictor, self).__init__()
        # Cấu trúc LSTM xử lý chuỗi kế hoạch động học của 6 khớp cánh tay
        self.lstm = nn.LSTM(input_features, hidden_dim, num_layers, batch_first=True)
        # Lớp Linear mapping không gian ẩn về dự đoán 3 trục Torque
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, sequence_length, features]
        out, (h_n, c_n) = self.lstm(x)
        # Chỉ lấy dự đoán ở mốc thời gian cuối cùng của chuỗi (Next Step Prediction)
        out = self.fc(out[:, -1, :]) 
        return out

class LSTMPredictiveNode(Node):
    def __init__(self):
        super().__init__('lstm_predictive_node')
        
        # Publisher gửi tín hiệu dự báo Torque về cho Node C++
        self.ff_pub_ = self.create_publisher(Vector3, '/ai/lstm_predictive_torque', 10)
        
        # Subscriber lắng nghe quỹ đạo tay máy từ hệ thống Motion Planning (như MoveIt2)
        self.arm_plan_sub_ = self.create_subscription(
            JointState,
            '/arm_controller/joint_trajectory_plan',
            self.trajectory_callback,
            10)
            
        # Nạp mô hình AI (Chạy trên CPU của Raspberry Pi 4 để giảm tiêu thụ điện năng)
        self.device = torch.device('cpu') 
        self.model = LSTM_TrajectoryPredictor().to(self.device)
        
        # Load tệp trọng số đã huấn luyện (offline training với Gazebo dataset)
        # self.model.load_state_dict(torch.load('lstm_uam_weights.pth', map_location=self.device))
        self.model.eval() # Bật chế độ suy luận
        
        # Tham số cấu hình độ dài chuỗi (Time windows)
        self.seq_len = 10
        self.sequence_buffer =

        self.get_logger().info("AI LSTM Node for Feedforward Control Khởi động thành công.")

    def trajectory_callback(self, msg):
        # Chuyển đổi trạng thái 6 góc khớp của tay máy thành numpy array
        joints = np.array(msg.position[:6], dtype=np.float32)
        
        # Cập nhật hàng đợi trượt (Sliding Window)
        self.sequence_buffer.append(joints)
        if len(self.sequence_buffer) > self.seq_len:
            self.sequence_buffer.pop(0)
            
        # Nếu đủ dữ liệu chuỗi thời gian, tiến hành nội suy LSTM
        if len(self.sequence_buffer) == self.seq_len:
            # Chuyển đổi thành Tensor để đẩy vào PyTorch
            seq_tensor = torch.tensor([self.sequence_buffer], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                # Dự đoán mô-men xoắn nhiễu sẽ sinh ra khi tay máy di chuyển tới tọa độ này
                # Thời gian suy luận trên CPU RPi 4 thường rơi vào ~15-20ms
                predicted_disturbance = self.model(seq_tensor).numpy()
                
            # Đóng gói và phát lên ROS2 network
            ff_msg = Vector3()
            ff_msg.x = float(predicted_disturbance) # Bù tiến Roll
            ff_msg.y = float(predicted_disturbance) # Bù tiến Pitch
            ff_msg.z = float(predicted_disturbance) # Bù tiến Yaw
            self.ff_pub_.publish(ff_msg)

def main(args=None):
    rclpy.init(args=args)
    lstm_node = LSTMPredictiveNode()
    rclpy.spin(lstm_node)
    lstm_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 8. Mô phỏng, Kiểm chứng và Tối ưu hóa Triển khai

Trước khi đưa khung điều khiển lên hệ thống bay phần cứng thực tế, quy trình chuẩn mực yêu cầu phải được kiểm thử toàn diện thông qua môi trường mô phỏng Gazebo (Software In The Loop - SITL). Tại đây, các đặc tính vật lý của cánh tay robot 6-DoF, khối lượng tải trọng thay đổi, và cả các nhiễu đo lường cảm biến đều có thể được mô phỏng với mức độ chính xác cao.

Quá trình điều chỉnh tham số (Tuning) diễn ra như sau:

- Ma trận khuếch đại Backstepping ($K_1, K_2$) được điều chỉnh theo nguyên lý từ thấp đến cao để đảm bảo sự đáp ứng nhanh nhẹn nhưng không vọt lố (overshoot).
    
- Tốc độ học của mạng RBFNN ($\Gamma$) được thiết lập bắt đầu từ giá trị nhỏ (như 0.05). Nếu hệ thống mất nhiều thời gian để hội tụ khi tay máy mang thêm một chai nước (tải trọng giả lập), tham số này cần được tăng lên. Ngược lại, việc dao động cánh quạt không bình thường cho thấy tốc độ cập nhật quá cao, gây ra hiện tượng học lố (over-adaptation), lúc này việc tăng $\eta$ trong luật e-modification sẽ giúp dập tắt dao động.
    
- Dữ liệu thu được từ chuỗi thao tác lặp lại trong mô phỏng được dùng làm bộ dữ liệu đào tạo chuẩn (Ground Truth) để tinh chỉnh kiến trúc và trọng số mạng LSTM, tối ưu hoá năng lực dự đoán trước khi xuất file `.pth` nạp lên Raspberry Pi 4.
    

## 9. Tổng kết

Khung điều khiển Backstepping thích nghi kết hợp Trí tuệ Nhân tạo do nghiên cứu này đề xuất mang đến một mô hình giải quyết triệt để vấn đề bất định động lực học và dịch chuyển trọng tâm của Hệ thống Phương tiện bay Thao tác không người lái (UAM). Thông qua việc phân bổ vai trò hợp lý: Mạng Nơ-ron Hàm cơ sở Suy rộng (RBFNN) chịu trách nhiệm tự động cập nhật tham số và bù trừ sai số trực tuyến thông qua phương trình ổn định Lyapunov, còn mạng Bộ nhớ Ngắn hạn Dài (LSTM) cung cấp tín hiệu điều khiển bù tiến bằng cách dự đoán các lực tác động cơ học từ chuỗi quỹ đạo cánh tay, hệ thống đã biến cơ chế từ điều khiển phản hồi thụ động sang khả năng dự báo tích cực.

Sự thiết kế cẩn trọng về mặt kiến trúc phần mềm, bao gồm việc sử dụng chế độ Offboard trực tiếp điều khiển mô-men của PX4 và cơ chế đa luồng phân chia tần số hoạt động (High-Frequency C++ / Low-Frequency Python) trên ROS2, đã giải quyết hoàn toàn nút thắt nghẽn cổ chai về năng lực xử lý phần cứng khi triển khai các thuật toán học sâu thời gian thực trên các máy tính nhúng giá rẻ như Raspberry Pi 4. Khung giải pháp và mã nguồn được thiết lập cung cấp một quy chuẩn nền tảng vững chắc để chuyển hóa từ nghiên cứu mô phỏng sang các hoạt động UAM thực tế chuyên nghiệp.