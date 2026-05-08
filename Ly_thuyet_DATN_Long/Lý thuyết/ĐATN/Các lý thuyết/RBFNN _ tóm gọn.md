Dưới đây là tổng hợp các điểm chính và những công thức toán học cốt lõi của Mạng nơ-ron hàm cơ sở bán kính (RBFNN):

### I. Các điểm chính của RBFNN

1. **Kiến trúc mạng:** RBFNN là mạng truyền thẳng gồm đúng 3 lớp: Lớp đầu vào, Lớp ẩn và Lớp đầu ra (lớp tính tổng). Mạng hoạt động như một cầu nối thống nhất cho các bài toán xấp xỉ hàm, nội suy, phân loại và ước lượng mật độ, đồng thời có **tốc độ huấn luyện nhanh hơn** mạng perceptron đa lớp (MLP).
2. **Cơ chế hoạt động:** Các nơ-ron ở lớp ẩn đóng vai trò như các "thụ thể cục bộ", sử dụng hàm cơ sở bán kính để tính toán trọng số ảnh hưởng dựa trên **khoảng cách Euclidean** từ véc-tơ đầu vào đến tâm của nơ-ron. Khoảng cách càng gần 0, mức độ đóng góp càng gần 1. Lớp đầu ra chỉ thực hiện phép cộng tuyến tính.
3. **Bốn tham số cần huấn luyện:** Để xây dựng mạng hiệu quả và tránh quá khớp, quá trình học phải xác định: (1) Số lượng nơ-ron lớp ẩn, (2) Tọa độ tâm, (3) Độ rộng/độ rải của hàm RBF, và (4) Trọng số kết nối.
4. **Chiến lược học 2 giai đoạn:** Quá trình huấn luyện thường được chia thành **Học không giám sát** (để thiết lập tâm và độ rộng cho lớp ẩn) và **Học có giám sát tuyến tính** (để tính toán trọng số kết nối tới lớp đầu ra).

### II. Các công thức quan trọng

**1. Công thức Kiến trúc Cơ bản**

- **Đầu ra của lớp ẩn (Hàm Gaussian):** Tính mức độ kích hoạt dựa trên khoảng cách Euclidean từ đầu vào $x$ đến tâm $c_j$ với độ rộng $\sigma_j$: $$h_j(x) = \exp \left( - \frac{||x - c_j||^2}{\sigma_j^2} \right)$$
- **Đầu ra của mạng (Lớp tính tổng):** Phép biến đổi tuyến tính kết hợp đầu ra lớp ẩn với trọng số $w_{kj}$ và độ lệch $b_k$: $$y_k(x) = \sum_{j=1}^{n_h} w_{kj} \cdot h_j(x) + b_k$$

**2. Các Hàm Hạt nhân (RBF Kernels) phổ biến** Ngoài Gaussian, hệ thống có thể dùng các hàm cơ sở khác (với $r$ là khoảng cách bán kính):

- **Generalized multi-quadric:** $\Phi(r) = (r^2 + c^2)^\beta$ (với $c > 0, 0 < \beta < 1$).
- **Generalized inverse multi-quadric:** $\Phi(r) = (r^2 + c^2)^{-\alpha}$ (với $0 < \alpha < c$).
- **Thin plate spline:** $\Phi(r) = r^2\ln(r)$.

**3. Công thức Xác định Tâm (Clustering)**

- **K-means:** Tính toán tâm cụm $c_j$ bằng trung bình cộng của các điểm dữ liệu $X_p$ thuộc cụm $S_j$: $$c_j = \frac{1}{N} \sum_{p \in S_j} X_p$$
- **Tối ưu hóa bầy đàn (PSO):** Cập nhật vận tốc ($v$) và vị trí ($x$) của hạt để tìm tâm tối ưu: $$v_{id}^{k+1} = \omega v_{id}^k + c_1 r_1 (pbest_{id}^k - x_{id}^k) + c_2 r_2 (gbest_d^k - x_{id}^k)$$ $$x_{id}^{k+1} = x_{id}^k + v_{id}^{k+1}$$

**4. Công thức Huấn luyện Trọng số (Weights)**

- **Bình phương tối thiểu trực giao (OLS):** Phát triển mạng bằng cách thêm tâm tuần tự trong không gian trực giao $Q$, giúp tìm trọng số $G$ mà không cần tính ma trận giả nghịch đảo: $$G = (Q'Q)^{-1}Q'T$$
- **Độ dốc lớn nhất tối ưu (OSD - Optimum Steepest Descent):** Tính véc-tơ delta trọng số tối ưu $\Delta W_{opt}$ dựa trên ma trận lỗi $E$ và ma trận RBF $\phi$, sau đó cập nhật trọng số mới $W_{new}$: $$\Delta W_{opt} = \lambda_{opt}\Delta W = \frac{(E\phi)(E\phi)^T(E\phi)}{(E\phi\phi^T)(E\phi\phi^T)^T}$$ $$W_{new} = W_{old} + \frac{(E\phi)(E\phi)^T(E\phi)}{(E\phi\phi^T)(E\phi\phi^T)^T}$$