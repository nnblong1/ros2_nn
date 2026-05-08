Khi nghiên cứu và phát triển các hệ thống bay phức tạp như UAV kết hợp với cơ cấu chấp hành (ví dụ như cánh tay robot), việc nắm vững mô hình động lực học cơ bản của Quadrotor là nền tảng cốt lõi để xây dựng bộ điều khiển ổn định. Dưới đây là chứng minh và giải thích chi tiết cho phương trình chuyển động tịnh tiến ($x$, $y$, $z$) của Quadrotor dưới tác động của các góc quay Pitch và Roll.

## 1. Định nghĩa Hệ quy chiếu và Ma trận xoay

### 1.1. Để mô tả chuyển động, chúng ta cần hai hệ tọa độ:

- **Hệ quy chiếu Trái Đất (Earth Frame - $E$):** Cố định với mặt đất, trục $Z$ hướng lên trên.
    
- **Hệ quy chiếu Thân máy bay (Body Frame - $B$):** Gắn chặt với tâm khối lượng của Quadrotor, trục $Z_B$ hướng vuông góc với mặt phẳng chứa các rotor.
    

### 1.2. Trạng thái góc (Euler angles) của UAV được xác định bởi:

- $\phi$ (Roll): Góc cuộn quanh trục $X$.
    
- $\theta$ (Pitch): Góc chúi quanh trục $Y$.
    
- $\psi$ (Yaw): Góc xoay quanh trục $Z$.
    ![[Pasted image 20260304103127.png]]

![Image](https://www.mathworks.com/help/phased/ref/rotx.png)
### 1.3. Các ma trận phép quay 
$$
R_x(\theta)=  
\begin{bmatrix}  
1 & 0 & 0 \\  
0 & \cos\theta & -\sin\theta \\  
0 & \sin\theta & \cos\theta  
\end{bmatrix}  $$

🔹 Trục **x** giữ nguyên  
🔹 Tọa độ **y, z** thay đổi theo quy tắc lượng giác

Phép quay góc ( $\theta$ ) quanh trục **y**:

$$
R_y(\theta)=  
\begin{bmatrix}  
\cos\theta & 0 & \sin\theta \\  
0 & 1 & 0 \\  
-\sin\theta & 0 & \cos\theta  
\end{bmatrix}  
$$

🔹 Trục **y** giữ nguyên  
🔹 Tọa độ **x, z** thay đổi

Phép quay góc ( $\theta$ ) quanh trục **z**:
$$R_z(\theta)=  
\begin{bmatrix}  
\cos\theta & -\sin\theta & 0 \\  
\sin\theta & \cos\theta & 0 \\  
0 & 0 & 1  
\end{bmatrix}  
$$

🔹 Trục **z** giữ nguyên  
🔹 Tọa độ **x, y** thay đổi
### 1.4. Tính chất quan trọng

- $(R^T = R^{-1})$ (ma trận trực giao)
- $\det(R) = 1$
- Có thể ghép nhiều phép quay: $R = R_z R_y R_x$
(Lưu ý: **thứ tự nhân ma trận rất quan trọng**)

Ma trận xoay $R$ chuyển đổi một vector từ hệ quy chiếu Thân ($B$) sang hệ quy chiếu Trái Đất ($E$) được tính bằng cách nhân 3 ma trận xoay cơ bản $R = R_z(\psi)R_y(\theta)R_x(\phi)$:

$$R = \begin{bmatrix} \cos\psi\cos\theta & \cos\psi\sin\theta\sin\phi - \sin\psi\cos\phi & \cos\psi\sin\theta\cos\phi + \sin\psi\sin\phi \\ \sin\psi\cos\theta & \sin\psi\sin\theta\sin\phi + \cos\psi\cos\phi & \sin\psi\sin\theta\cos\phi - \cos\psi\sin\phi \\ -\sin\theta & \cos\theta\sin\phi & \cos\theta\cos\phi \end{bmatrix}$$

## 2. Các lực tác dụng lên Quadrotor

Bỏ qua lực cản không khí trong mô hình lý tưởng, có hai lực chính tác dụng lên tâm khối lượng của Quadrotor:

1. **Trọng lực ($F_g$):** Luôn hướng xuống dọc theo trục $Z$ của hệ Trái Đất.
    
    $$F_g = \begin{bmatrix} 0 \\ 0 \\ -mg \end{bmatrix}$$
    
2. **Lực đẩy tổng cộng (Thrust - $T$):** Được tạo ra bởi 4 động cơ, luôn đẩy dọc theo trục $Z$ của hệ Thân ($B$).
    
    $$F_{thrust, B} = \begin{bmatrix} 0 \\ 0 \\ T \end{bmatrix}$$
    

Để tính tổng lực trong hệ quy chiếu Trái Đất, ta dùng ma trận xoay $R$ để chiếu lực đẩy từ hệ Thân sang hệ Trái Đất:

$$F_{thrust, E} = R \begin{bmatrix} 0 \\ 0 \\ T \end{bmatrix}$$

Cột thứ 3 của ma trận xoay $R$ chính là kết quả của phép nhân này:

$$F_{thrust, E} = \begin{bmatrix} T(\cos\psi\sin\theta\cos\phi + \sin\psi\sin\phi) \\ T(\sin\psi\sin\theta\cos\phi - \cos\psi\sin\phi) \\ T(\cos\theta\cos\phi) \end{bmatrix}$$

## 3. Chứng minh phương trình chuyển động ($x, y, z$) [[bouabdallah2004.pdf]]

Áp dụng định luật II Newton cho chuyển động tịnh tiến:

$$m\ddot{\mathbf{r}} = F_g + F_{thrust, E}$$

Trong đó, $\mathbf{r} = [x, y, z]^T$ là vector vị trí của Quadrotor trong hệ quy chiếu Trái Đất, $m$ là khối lượng, và $\ddot{\mathbf{r}}$ là gia tốc. Thay các thành phần lực vào, ta có hệ phương trình:

$$m\ddot{x} = T(\cos\psi\sin\theta\cos\phi + \sin\psi\sin\phi)$$

$$m\ddot{y} = T(\sin\psi\sin\theta\cos\phi - \cos\psi\sin\phi)$$

$$m\ddot{z} = T(\cos\theta\cos\phi) - mg$$

## 4. Giải thích ý nghĩa vật lý (Tác động của Pitch và Roll)

Các phương trình trên giải thích chính xác lý do tại sao một thiết bị bay không người lái dạng Quadrotor lại di chuyển được trong không gian 3D dù lực đẩy của nó chỉ sinh ra trên một trục duy nhất (trục $Z$ của thân):

- **Chuyển động dọc trục Z (Độ cao):** Phương trình $m\ddot{z} = T(\cos\theta\cos\phi) - mg$ cho thấy lực nâng thực tế chống lại trọng lực không phải là toàn bộ lực $T$, mà chỉ là hình chiếu của nó lên trục $Z$ của Trái Đất. Khi UAV nghiêng (tức là $\theta$ hoặc $\phi$ khác **0**), giá trị $\cos\theta\cos\phi$ sẽ nhỏ hơn **1**. Điều này có nghĩa là khi Pitch hoặc Roll, lực nâng thẳng đứng sẽ bị giảm đi. Để giữ nguyên độ cao khi bay tới/lui/trái/phải, bộ điều khiển phải tăng tổng lực đẩy $T$.
    
- **Chuyển động theo trục X (Pitching):** Giả sử UAV không Yaw ($\psi = 0$) và không Roll ($\phi = 0$). Phương trình trục $X$ trở thành $m\ddot{x} = T(\sin\theta)$. Điều này giải thích rằng: Bằng cách tạo ra góc Pitch ($\theta$), Quadrotor đã làm nghiêng vector lực đẩy $T$, tạo ra một thành phần lực nằm ngang hướng dọc theo trục $X$. Thành phần lực này chính là nguyên nhân gia tốc UAV tiến về phía trước hoặc lùi về phía sau.
    
- **Chuyển động theo trục Y (Rolling):** Tương tự, nếu không Yaw ($\psi = 0$) và không Pitch ($\theta = 0$), phương trình trục $Y$ trở thành $m\ddot{y} = T(-\sin\phi)$. Khi Quadrotor thực hiện góc Roll ($\phi$), vector lực đẩy sẽ nghiêng sang hai bên, sinh ra thành phần lực ngang đẩy thiết bị bay sang trái hoặc phải dọc theo trục $Y$.
    

Mô hình này là lý do vì sao Quadrotor thuộc loại hệ thống "underactuated" (thiếu cơ cấu chấp hành) – bạn không thể điều khiển trực tiếp độc lập gia tốc $x$ hoặc $y$ mà bắt buộc phải thay đổi góc Pitch ($\theta$) và Roll ($\phi$) để hướng vector lực đẩy vào mặt phẳng ngang.

Để phân tích chuyển động xoay (Roll, Pitch, Yaw) của Quadrotor, chúng ta sử dụng **ma trận mô-men quán tính (Inertia Matrix/Tensor)**. Mô-men quán tính đại diện cho mức độ chống lại sự thay đổi vận tốc góc của phương tiện khi có một mô-men xoắn (torque) tác dụng lên nó.

## 1. Ma trận Mô-men quán tính ($I$ hoặc $J$)

Đối với một vật thể rắn trong không gian 3D, mô-men quán tính được biểu diễn bằng một ma trận $3 \times 3$ đối xứng:

$$I = \begin{bmatrix} I_{xx} & -I_{xy} & -I_{xz} \\ -I_{xy} & I_{yy} & -I_{yz} \\ -I_{xz} & -I_{yz} & I_{zz} \end{bmatrix}$$

Trong đó:

- **$I_{xx}, I_{yy}, I_{zz}$ (Mô-men quán tính chính):** Đặc trưng cho quán tính quanh các trục $X$ (Roll), $Y$ (Pitch), và $Z$ (Yaw) của hệ quy chiếu Thân ($B$).
    
- **$I_{xy}, I_{xz}, I_{yz}$ (Tích quán tính):** Đặc trưng cho sự mất cân bằng khối lượng giữa các mặt phẳng tọa độ.
    

## 2. Mô hình chuẩn của Quadrotor (Giả định đối xứng)

Trong các mô hình điều khiển Quadrotor cơ bản, người ta thường thiết kế phương tiện có tính đối xứng cao (cấu hình chữ X hoặc dấu +) với tâm khối lượng nằm ngay gốc tọa độ hệ Thân. Do đó, các tích quán tính bị triệt tiêu ($I_{xy} = I_{xz} = I_{yz} = 0$).

Ma trận quán tính được đơn giản hóa thành một ma trận đường chéo:

$$I = \begin{bmatrix} I_{xx} & 0 & 0 \\ 0 & I_{yy} & 0 \\ 0 & 0 & I_{zz} \end{bmatrix}$$

Đồng thời, do tính đối xứng hình học giữa trục $X$ và trục $Y$, ta thường có $I_{xx} \approx I_{yy}$. Tuy nhiên, $I_{zz}$ thường lớn hơn đáng kể so với hai giá trị kia do khối lượng của 4 cụm động cơ/cánh quạt nằm xa trục $Z$.

## 3. Phương trình Động lực học quay (Rotational Dynamics)

Áp dụng phương trình Euler cho chuyển động quay của vật rắn, tổng mô-men xoắn $\tau$ tác dụng lên tâm khối lượng (trong hệ Thân) được tính bằng:

$$\tau = I\dot{\omega} + \omega \times (I\omega)$$

Trong đó:

- $\tau = [\tau_x, \tau_y, \tau_z]^T$ là các vector mô-men điều khiển (Roll, Pitch, Yaw torques).
    
- $\omega = [p, q, r]^T$ là vận tốc góc trong hệ Thân (tương đương với $\dot{\phi}, \dot{\theta}, \dot{\psi}$ khi các góc Euler nhỏ).
    

Khai triển phương trình này với ma trận $I$ đường chéo, ta được hệ phương trình mô tả gia tốc góc:

- **Trục X (Roll):** $I_{xx}\dot{p} = \tau_x - qr(I_{zz} - I_{yy})$
    
- **Trục Y (Pitch):** $I_{yy}\dot{q} = \tau_y - pr(I_{xx} - I_{zz})$
    
- **Trục Z (Yaw):** $I_{zz}\dot{r} = \tau_z - pq(I_{yy} - I_{xx})$
    

Các thành phần như $qr(I_{zz} - I_{yy})$ chính là **hiệu ứng con quay hồi chuyển (gyroscopic effect)** do chuyển động quay quanh các trục khác tạo ra.

### 1. Đồng bộ ký hiệu giữa hai tài liệu

Trước tiên, ta cần quy đổi các ký hiệu từ hệ phương trình (5) (ảnh 2) sang dạng ký hiệu của phương trình (13, 14, 15) (ảnh 1):

- **Mô-men quán tính:** $I_x, I_y, I_z$ trong ảnh 2 tương đương với $I_{xx}, I_{yy}, I_{zz}$ trong ảnh 1.
    
- **Tín hiệu điều khiển (Mô-men xoắn):** Các ngõ vào $U_2, U_3, U_4$ tương đương với các mô-men $\tau_x, \tau_y, \tau_z$.
    

### 2. Chứng minh từng phương trình [[Adaptive Control Approaches for an_Unmanned Aerial Manipulation System.pdf]]

#### Chứng minh phương trình Roll (Trục X)

Bắt đầu với phương trình đầu tiên trong nhóm động học quay của hệ (5):

$$\ddot{\phi} = \dot{\theta}\dot{\psi}\left(\frac{I_y - I_z}{I_x}\right) - \frac{J_r}{I_x}\dot{\theta}\Omega + \frac{1}{I_x}U_2$$

**Bước 1:** Nhân cả 2 vế với $I_x$ (tức là $I_{xx}$) để khử mẫu số:

$$I_{xx}\ddot{\phi} = \dot{\theta}\dot{\psi}(I_{yy} - I_{zz}) - J_r\dot{\theta}\Omega + \tau_x$$

**Bước 2:** Áp dụng các giả định của hệ thống mới (ảnh 1):

- Tài liệu ở ảnh 1 **bỏ qua hiệu ứng con quay hồi chuyển của cụm động cơ/cánh quạt** tác dụng lên khung máy bay. Do đó, thành phần chứa $J_r\Omega$ được coi là xấp xỉ 0 và bị loại bỏ.
    
- Hệ thống ở ảnh 1 có gắn thêm tay máy robot, sinh ra các mô-men nhiễu ngoại lai. Ta cộng thêm mô-men nhiễu trục X là $n_{0x}$ vào vế phải.
    

**Kết quả:**

$$I_{xx}\ddot{\phi} = \mathbf{\dot{\theta}\dot{\psi}}(I_{yy} - I_{zz}) + \tau_x + n_{0x}$$

_(Lưu ý: Qua bước chứng minh này, hệ phương trình (5) ở ảnh 2 đã xác nhận lại điều chúng ta đã phân tích trước đó: **Phương trình (13) ở ảnh 1 gõ sai** $\dot{\phi}\dot{\psi}$, đáp án đúng về mặt vật lý phải là $\dot{\theta}\dot{\psi}$)._

#### Chứng minh phương trình Pitch (Trục Y)

Bắt đầu với phương trình thứ hai của hệ (5):

$$\ddot{\theta} = \dot{\phi}\dot{\psi}\left(\frac{I_z - I_x}{I_y}\right) + \frac{J_r}{I_y}\dot{\phi}\Omega + \frac{1}{I_y}U_3$$

**Bước 1:** Nhân 2 vế với $I_y$ (tức là $I_{yy}$) và thay đổi ký hiệu tương ứng:

$$I_{yy}\ddot{\theta} = \dot{\phi}\dot{\psi}(I_{zz} - I_{xx}) + J_r\dot{\phi}\Omega + \tau_y$$

**Bước 2:** Tương tự như trục X, ta bỏ qua hiệu ứng con quay của cánh quạt ($J_r\dot{\phi}\Omega \approx 0$) và cộng thêm mô-men nhiễu từ tay máy robot $n_{0y}$.

**Kết quả:** (Hoàn toàn khớp với phương trình 14)

$$I_{yy}\ddot{\theta} = \dot{\phi}\dot{\psi}(I_{zz} - I_{xx}) + \tau_y + n_{0y}$$

#### Chứng minh phương trình Yaw (Trục Z)

Bắt đầu với phương trình thứ ba của hệ (5):

$$\ddot{\psi} = \dot{\phi}\dot{\theta}\left(\frac{I_x - I_y}{I_z}\right) + \frac{1}{I_z}U_4$$

**Bước 1:** Nhân 2 vế với $I_z$ (tức là $I_{zz}$) và quy đổi ký hiệu:

$$I_{zz}\ddot{\psi} = \dot{\phi}\dot{\theta}(I_{xx} - I_{yy}) + \tau_z$$

**Bước 2:** Phương trình trên trục Z ban đầu không có thành phần hiệu ứng con quay của cánh quạt, ta chỉ cần cộng thêm mô-men nhiễu trục Z của tay máy robot $n_{0z}$.

**Kết quả:** (Hoàn toàn khớp với phương trình 15)

$$I_{zz}\ddot{\psi} = \dot{\phi}\dot{\theta}(I_{xx} - I_{yy}) + \tau_z + n_{0z}$$
$$Q_{dd} = \begin{bmatrix} \ddot{q}_1 \ \ddot{q}_{11} - K_{d1}e_{12} \\ -K_{p2}(e_{22} - K_{p2}e_{21}) - e_{21} - K_{d2}e_{22} \\ -K_{p3}(e_{32} - K_{p3}e_{31}) - e_{31} - K_{d3}e_{32} \\ -K_{p4}(e_{42} - K_{p4}e_{41}) - e_{41} - K_{d4}e_{42} \\ -K_{p5}(e_{52} - K_{p5}e_{51}) - e_{51} - K_{d5}e_{52} \\ -K_{p6}(e_{62} - K_{p6}e_{61}) - e_{61} - K_{d6}e_{62} \end{bmatrix}$$$