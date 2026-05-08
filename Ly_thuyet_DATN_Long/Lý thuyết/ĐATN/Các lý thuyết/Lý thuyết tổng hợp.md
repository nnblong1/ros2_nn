# 1. Phương pháp Backstepping
# 2. Mạng Nơ-ron Hàm Cơ sở Suy rộng (Radial Basis Functional Neural Network - RBFNN)  [[RBFNN.pdf]] , [[RBFNN _ tóm gọn]]
## 2.1. Kiến trúc
Ý tưởng về RBFNN bắt nguồn từ lý thuyết xấp xỉ hàm. Khoảng cách Euclid được tính từ điểm được đánh giá đến tâm của mỗi tế bào thần kinh, và hàm cơ sở xuyên tâm (RBF) (còn được gọi là hàm hạt nhân hoặc hàm Gaussian) được áp dụng cho khoảng cách để tính trọng lượng (ảnh hưởng) cho mỗi tế bào thần kinh. Hàm cơ sở xuyên tâm được đặt tên như vậy vì khoảng cách bán kính là đối số của hàm. Nói cách khác, RBF đại diện cho các thụ thể cục bộ; Đầu ra của nó phụ thuộc vào khoảng cách của đầu vào từ một vectơ được lưu trữ nhất định. Nếu khoảng cách từ vecto đầu vào $\overrightarrow{x}$   tới điểm chính giữa $\overrightarrow{c_i}$ của mỗi Hàm Cơ sở Xuyên tâm (RBF) $\varphi_i$ . Ví dụ: khoảng cách Euclidean $||\overrightarrow{x}-\overrightarrow{c_i}||$  bằng 0 thì mức độ đóng góp (trọng số ảnh hưởng) của điểm này là 1; ngược lại, khoảng cách càng tăng thì mức độ đóng góp càng tiến dần về 0

Mạng nơ-ron chức năng cơ sở xuyên tâm là mạng ba lớp. Bao gồm: Input Layer, Hidden Layer, Output Layer.
![[Pasted image 20260309192143.png]]

### 2.1.1. Lớp đầu vào (Input Layer)
Lớp này có thể bao gồm nhiều biến dự báo, trong đó mỗi biến được gắn với một nơ-ron độc lập. Chức năng duy nhất của lớp đầu vào là truyền trực tiếp các véc-tơ đầu vào đến lớp ẩn mà không qua bất kỳ xử lý tính toán nào
### 2.1.2. Lớp ẩn (Hidden Layer)
Đây là lớp thực hiện phép biến đổi phi tuyến. Lớp này bao gồm một số lượng nơ-ron nh​ sử dụng hàm cơ sở bán kính (phổ biến nhất là hàm Gaussian). Mỗi hàm Gaussian thứ j được đặc trưng bởi hai tham số quan trọng:
-  **Tâm** ($c_j$​): Tọa độ trung tâm của hàm.
- **Độ rộng** ($\sigma_j$): Quyết định phạm vi ảnh hưởng của hàm.
Hệ thống phân loại RBFNN bắt đầu bằng việc tính toán khoảng cách Euclidean giữa véc-tơ đầu vào $x$ và tâm $c_j$​ của hàm. Sau đó, đầu ra của nơ-ron ẩn thứ j, ký hiệu là $h_j​(x)$, được tính bằng công thức:
$$h_j(x) = \exp(\frac{-||x-c_j||^2}{\sigma_j^2}) \tag{1}$$
### 2.1.3. Lớp đầu ra (Output Layer)
Trái ngược với lớp ẩn, lớp đầu ra chỉ thực hiện các phép toán tuyến tính. Giá trị đầu ra từ các nơ-ron ở lớp ẩn sẽ được nhân với các trọng số kết nối (weights) tương ứng và truyền đến bộ tính tổng,. Đầu ra thứ $k$ của mạng đối với véc-tơ đầu vào $x$ được tính bằng phương trình: $$ y_k(x) = \sum_{j=1}^{n_h} {w_{kj}.h_j(x) + b_k} \tag{2}$$Trong đó:

- $y_k​(x)$ là kết quả tại đơn vị đầu ra thứ $k$.
- $w_{kj}​$ là trọng số kết nối giữa đơn vị đầu ra thứ $k$ và nơ-ron ẩn thứ $j$.
- $b_k​$là giá trị độ lệch (bias) được nhân với một trọng số và cấp vào lớp đầu ra để điều chỉnh kết quả cuối cùng.
## 2.2. Thuật toán huấn luyện
Học hoặc đào tạo mạng nơ-ron là một quá trình mà mạng tự thích ứng với kích thích bằng cách điều chỉnh tham số thích hợp, dẫn đến việc tạo ra phản ứng mong muốn. Do đó, để có được độ chính xác xấp xỉ / phân loại tương tự và trong số lượng đơn vị RBF cần thiết, các thông số sau được xác định bởi quá trình đào tạo của RBFNN
- Số lượng nơ-ron trong lớp ẩn (lý tưởng nhất là nhỏ hơn rất nhiều so với số lượng điểm dữ liệu để tránh quá khớp - overfitting).
- Tọa độ tâm (centers) của mỗi nơ-ron ẩn.
- Trọng số (weights) kết nối từ lớp ẩn tới tới đầu ra.
### 2.2.1. Lựa chọn Hàm Hạt nhân (RBF Kernels)
Gaussian kernel thường được lựa chọn để sử dụng cho hàm kernel. Một số RBFs phổ biến được sử dụng

| Kernel                                     | Công thức tổng quát                         |
| ------------------------------------------ | ------------------------------------------- |
| Generalized multi-quadric function         | $\phi(r )= (r^2+c^2)^\beta, c>0, 0<\beta<1$ |
| Generalized inverse multi-quadric function | $\phi(r )= (r^2+c^2)^{-\alpha}, 0<\alpha<c$ |
| Thin plate spline basis function           | $\phi(r )= r^2ln(r)$                        |
| Cubic function                             | $\sigma(r)=r^3$                             |
| Linear function                            | $\sigma(r)=r$                               |
Trong Generalized multi-quadric function, biểu diễn ma trận của hàm cơ sở có một tính chất quang phổ quan trọng: nó gần như là xác định âm. Franke đã phát hiện ra rằng RBF này cung cấp bề mặt nội suy chính xác nhất trong hai chiều. Ngoài ra, ông phát hiện ra rằng Generalized inverse multi-quadric function có thể cung cấp các xấp xỉ tuyệt vời, ngay cả khi số lượng tâm nhỏ. Tuy nhiên, tác giả trình bày rằng đôi khi một giá trị lớn của $\sigma$ có thể hữu ích. Ngược lại, không có sự lựa chọn tốt về $\sigma$ được biết đến hiện nay trong trường hợp hàm multi-quadric basis.

Thin plate spline basis function có bản chất toàn cục hơn hàm Gaussian, tức là một nhiễu loạn nhỏ của một trong các điểm điều khiển luôn ảnh hưởng đến các hệ số tương ứng với tất cả các điểm khác. Tương tự, các hàm cơ sở đa thức như Cubic và Linear có một số độ ảnh hưởng trong một số ứng dụng nhất định.
### 2.2.2. Xác định Tâm và Độ rộng (Học không giám sát - Unsupervised Learning)
Lợi thế lớn của RBFNN là có thể chọn các tham số này mà không cần tối ưu hóa phi tuyến toàn bộ mạng. Có 3 hướng tiếp cận chính để tìm tâm mạng:
**2.2.2.1. Chọn tâm cố định ngẫu nhiên (Fixed centers selected at random)**
Đây là cách đơn giản và nhanh nhất, trong đó các tâm được chọn ngẫu nhiên từ tập dữ liệu đầu vào. Tuy nhiên, hiệu suất thường không cao.

Điểm tâm được chọn cố định ở điểm M, và M được lựa chọn ngẫu nhiên từ tập dữ liệu các điểm N. Đặc biệt, có thể sử dụng các RBF chuẩn hóa tập trung tại {$c_i$}: $$\phi_j(x)=\exp(-\frac{||x-c_j||^2}{2\sigma^2_j}) \tag{3}$$  trong đó {$c_j$} $\subseteq$ {$X^P$} và $\sigma_j$ là chiều rộng của các hàm Gaussian.
**2.2.2.2. Các kỹ thuật phân cụm (clustering)**
Các kỹ thuật phân cụm có thể được sử dụng để tìm một tập hợp các trung tâm phản ánh chính xác hơn sự phân bố của các điểm dữ liệu.
1. **Thuật toán phân cụm K-means**
    Chọn trước số K của các trung tâm, và sau đó làm theo một quy trình tính toán lại đơn giản để chia các điểm dữ liệu {$X^P$} thành K tập hợp rời rạc $S_j$và $N_j$ điểm dữ liệu nhằm tối thiểu hóa các hàm phân cụm bình phương $$J = \sum_{j=1}^K{\sum_{peS_j}{||X^P-c_j||^2}}\tag{4}$$
    với $c_j$ là giá trị trung bình của tập điểm dữ liệu trong tập $S_j$ trong công thức [5]
    $$c_j = \frac{1}{N}\sum_{peS_j}{X^P}\tag{5}$$
    Tuy nhiên, có hai nhược điểm nội tại liên quan đến việc sử dụng K-means. Đầu tiên là do tính chất lặp đi lặp lại của nó, có thể dẫn đến thời gian hội tụ lâu và thứ hai bắt nguồn từ việc nó không có khả năng tự động xác định số lượng trung tâm RBF, do đó dẫn đến quy trình thử và sai tốn thời gian để thiết lập kích thước của lớp ẩn.
    Vô số kỹ thuật thay thế đã được đề xuất để giải quyết những nhược điểm này. Một cách là sử dụng một số phương pháp không giám sát được cải tiến như: phân cụm mờ (fuzzy clustering), bản đồ tự tổ chức (self organizing map - SOM) , phân cụm trừ dựa trên tối ưu hóa bầy hạt (particle swarm optimization - PSO), thuật toán phân cụm K-means động (dynamic K-means clustering algorithm), thuật toán K-means cải tiến (improved K-means algorithm), phân cụm sóng hài K (K-harmonic clustering) và phân cụm tự cộng (self-additive clustering) đã được sử dụng để lựa chọn trung tâm trong RBFNN. 
    Vì vậy Fathi và Montazer nghiên cứu sử dụng kỹ thuật tối ưu hóa bầy đàn (PSO) để điều chỉnh tâm của các hàm Gaussian trong khi thuật toán lân cận gần nhất p (p-nearest) được sử dụng để điều chỉnh độ rộng. Vận tốc và tọa độ được cập nhật theo quy luật:$$
v_{id}^{k+1}=\omega v_{id}^{k}
+ c_1 r_1 (pbest_{id}^{k}-x_{id}^{k})
+ c_2 r_2 (gbest_{d}^{k}-x_{id}^{k})
\tag{6}
$$$$
x_{id}^{k+1}=x_{id}^{k}+v_{id}^{k+1}, \quad i=1,2,\ldots,n
\tag{7}
$$$$
\omega=\omega_{\max}-\frac{k}{k_{\max}}(\omega_{\max}-\omega_{\min})
\tag{8}
$$
    Với vị trí hiện tại của hạt i trong lần lặp thứ k là $x^k_{id}$ và  $v^k_{id}$ là vận tốc hiện tại của hạt được sử dụng để xác định vận tốc mới $v^{k+1}_{id}$. $c_1$ và $c_2$ là hệ số gia tốc. $r_1$ và $r_2$ là 2 số ngẫu nhiên độc lập được phân bố đều trong khoảng $[0.1]$ . Ngoài ra $v_i \in [-v_{max}, v_{max}]$ với $v_{max}$ là hằng số phụ thuộc vàoo bài toán được sử dụng để hạn chế sự dịch chuyển quá mức của các hạt. $pbest_{id}^{k}$ là vị trí tốt nhất trước đó dọc theo chiều thứ d của hạt i trong lần lặp k (được ghi nhớ của mọi hạt). $\omega_{max}$ và $\omega_{min}$ là max và min của $\omega$. $k_{max}$ là số lần lặp tối đa.
   
    Mặc dù có khả năng tối ưu hóa cao, PSO có thể bị mắc kẹt trong điểm tối ưu cục bộ, làm chậm tốc độ hội tụ. Ngoài ra, việc sử dụng p-nearest neighbor algorithm để tính toán độ rộng của các đơn vị RBF dẫn đến mất thông tin về phân bố không gian của tập dữ liệu huấn luyện; và do đó độ rộng được tính toán không đóng góp đáng kể vào hiệu suất phân loại của dữ liệu rất phức tạp như hình ảnh. Để khắc phục nhược điểm này, Montazer và Giveki đã đề xuất một phiên bản thích ứng mới của PSO, có khả năng hoạt động với dữ liệu chiều cao. Kết quả thu được cho thấy tốc độ hội tụ nhanh, phản hồi mạng tốt hơn và tương tự với ít dữ liệu huấn luyện hơn, điều này cho thấy khả năng khái quát hóa của mạng nơ-ron được cải tiến được đề xuất trong. Chiến lược thích ứng được đề xuất hoạt động như sau:  $$v_{id}^{k+1}=\omega_i^kv_{id}^k+c_1r_1(pbest_{id}^k-x_{id}^k)+c_2r_2(gbest_d^k - x_{id}^k) \tag{9}
   $$ $$x_{id}^{k+1}=x_{id}^k+\mu_i^kv_{id}^{k+1}, i = 1,2,\dots,n \tag{10}$$
	Chiến lược thích ứng là một phương pháp điều chỉnh động hệ số trọng số quán tính $\omega$ và vận tốc mới $v_{id}^{k+1}$ bằng cách đưa vào hệ số $\mu$
	Trọng số quán tính $\omega$ có ảnh hưởng lớn đến hiệu suất tối ưu. Các nghiên cứu thực nghiệm về PSO với trọng số quán tính $\omega$ đã chỉ ra rằng nếu $\omega$ lớn có khả năng tìm kiếm toàn cục tốt hơn so với $\omega$ nhỏ làm nhanh hội tụ hơn. Phương trình [6] được cập nhật bằng cách sử dụng chiến lược cập nhật tuyến tính của phương trình [8] . Do đó $\omega$ chỉ phụ thuộc vào lần lặp hiện tại và số lần lặp tối đa ($k$ và $k_{max}$) và không thể thích ứng với các đặc điểm phức tạp và có tính phi tuyến cao. Nếu vần đề cực kỳ phức tạp, khả năng tìm kiếm toàn cục sẽ không đủ trong lần lặp sau. Vì vậy để khắc phục những nhược điểm trên, một phương pháp cải tiến để cập nhật $\omega$ như sau: 
   $$\omega_i^{k}= k_1 h_i^{k} + k_2 b_i^{k} + \omega_0 \tag{11}$$$$h_i^{k}= \left[\max\left\{F_{id}^{k}, F_{id}^{k-1}\right\} - \min\left\{F_{id}^{k}, F_{id}^{k-1}\right\}\right] / f_1 \tag{12}$$$$b_i^{k}= \frac{1}{n} \times \sum_{i=1}^{n} (F_i^{k} - F_{avg}) / f_2 \tag{13}$$
   $$\begin{equation}  
\mu_i^{k} =  
\begin{cases}  
\left(\dfrac{v_{\max}}{v_i^{k}}\right)e^{-\left(\frac{k}{k_{\max}}\right)^2}, & \text{if } v_i^{k} > v_{\max} \\
1, & \text{if } v_{\min} < v_i^{k} < v_{\max}\\ 
\left(\dfrac{v_{\min}}{v_i^{k}}\right)e^{-\left(\frac{k}{k_{\max}}\right)^2}, & \text{if } v_i^{k} < v_{\min}  
\end{cases}  
\tag{14}  
\end{equation}$$
    Với $\omega_0 \in (0,1]$ là hệ số quán tính điều chỉnh tác động của vận tốc trước đó lên vận tốc hiện tại (hầu hết trong các trường hợp bằng 1). Trong phương trình [11], các hệ sôs $k_1$ và $k_2$ thường được chọn thực nghiệm trong phạm vi $[0,1]$. Trong phương trình [14], tham số $\mu$ điều chỉnh giá trị một cách thích ứng của $v^{k+1}$ bằng cách xem xét giá trị của $v^k$
	- $h_i^k$ là **tốc độ tiến hóa**.  
  
	- $b_i^k$ là **phương sai trung bình của giá trị fitness của bầy hạt**.  
  
	- $F_{id}^k$ là **giá trị fitness của** $pbest_{id}^k$, tức là  $F(pbest_{id}^k)$.  
  
	- $F_{id}^{k-1}$ là **giá trị fitness của** $pbest_{id}^{k-1}$, tức là $F(pbest_{id}^{k-1})$.  
  
	- $f_1$ là **hàm chuẩn hóa**: $f_1 = \max \{\Delta F_1, \Delta F_2, \dots, \Delta F_n\},\Delta F_i = |F_{id}^k - F_{id}^{k-1}|$  
	- n là **kích thước của bầy hạt (particle swarm)**.  
	- $F_i^k$ là **giá trị fitness hiện tại của hạt thứ \(i\)**.  
	- $F_{avg}là$ **giá trị fitness trung bình của tất cả các hạt trong bầy tại vòng lặp thứ $k$.  
  
	- $f_2$ là **hàm chuẩn hóa**: $$f_2 =max\{ |F_1^k - F_{avg}|,  
	|F_2^k - F_{avg}|, \dots,|F_n^k - F_{avg}|\}$$
	 Việc điều chỉnh động giúp PSO không chỉ tránh được các điểm tối ưu cục bộ mà còn tăng cường sự đa dạng của quần thể, từ đó cải thiện chất lượng của các giải pháp. Để tính toán các tâm RBFNN bằng cách sử dụng các thuật toán PSO, giả sử rằng một hạt đơn lẻ đại diện cho một tập hợp k vectơ tâm cụm $X=(M_1,M_2,\dots,M_k)$ với $M_j=(s_{j1},\dots,s_{jl},\dots,s_{jf})$ đề cập đến vecto tâm cụm thứ j của một hạt. $M_j$ có $f$ cột đại diện cho số lượng đặc trưng cho mỗi mẫu của tập dữ liệu. Mỗi bầy chứa một số giải pháp phân cụm dữ liệu. Khoảng cách Euclidean của mẫu đầu vào các các tâm cụm được đo bằng công thức:	$$
d(M_{ij}, P_{rl}) =
\sqrt{\sum_{i=1}^{f} (S_{ji} - t_{rl})^2}
\quad \text{for } 1 \le j \le k,\; 1 \le r \le n,\; 1 \le l \le f \tag{15}
$$
   Sau khi tính toán tất cả các khoảng cách của mỗi hạt, đặc tính $l$ của từng mẫu r được so sánh với đặc tính tương ứng của cụm $j$, sau đó gán 1 cho $Z_{jrl} = 1$ khi khoảng cách Euclidean cho mỗi đặc tính $l$ của mẫu $r$ là nhỏ nhất:$$
Z_{jrl} =
\begin{cases}
1 & \text{nếu } d(M_{ij}, P_{rl}) \text{ là nhỏ nhất} \\
0 & \text{ngược lại} \tag{16}
\end{cases}
$$

	Trong bước tiếp theo, **giá trị trung bình của dữ liệu** $N_{jl}$được tính cho mỗi hạt theo công thức:

$$
N_{jl} =
\frac{\sum_{r=1}^{n} t_{rl} \times Z_{jrl}}
{\sum_{r=1}^{n} Z_{jrl}}
\qquad
1 \le j \le k,\; 1 \le l \le f \tag{17}
$$

	Ngoài ra, với mỗi thuộc tính $l$ của cụm $j$, khoảng cách Euclid giữa trung bình dữ liệu $N_{jl}$ và tâm cụm $S_{jl}$ được tính như sau:

$$
d(N_{jl}, S_{jl}) =
\sqrt{(N_{jl} - S_{jl})^2}
\qquad
1 \le j \le k,\; 1 \le l \le f \tag{18}
$$

	Tiếp theo, hàm fitness của mỗi cụm được tính bằng cách cộng các khoảng cách đã tính:

$$
F(M_j) =
\sum_{l=1}^{f} d(N_{jl}, S_{jl})
\qquad
1 \le j \le k \tag{19}
$$
	==**Thuật toán 1**.  Phân cụm bằng PSO để tìm tâm của nơ-ron RBF (The pseudocode of the proposed PSO clustering for RBF unit center)==
	Sử dụng thuật toán này, ta có thể **điều chỉnh các tâm của RBF** bằng cách cập nhật **gbest** thông qua quá trình lặp với $k_{max}$ vòng lặp.

	**Khởi tạo (Initialization)**

- **Với mỗi Hạt `Particle[i]` Thực hiện:**
    - Khởi tạo véc-tơ Vị trí `X[i]` nằm trong phạm vi từ giá trị nhỏ nhất (min) đến lớn nhất (max) của các mẫu dữ liệu.
    - Khởi tạo véc-tơ Vận tốc `V[i]` nằm trong phạm vi `[-a, a]` (trong đó `a = max(data) - min(data)`).
    - Gán giá trị khởi tạo của `Particle[i]` vào vị trí tốt nhất của cá nhân nó, ký hiệu là `pbest_i`.

	**Vòng lặp tiến hóa (Evolutionary Loop)**

	- **Trong khi (While) chưa đạt đến số vòng lặp tối đa Thực hiện:**
    - _(1) Tính toán hàm thích nghi:_
        - **Với mỗi** `Particle[i]` **Thực hiện:**
            - **Với mỗi** Cụm `Cluster[j]` **Thực hiện:**
                - Tính toán Hàm thích nghi (Fitness Function) sử dụng các Phương trình (17), (18) và (19).
    - _(2) Cập nhật Vị trí tốt nhất cá nhân (`pbest`):_
        - **Nếu (If)** số lần chạy (run number) > 1 **Thì (Then):**
            - **Với mỗi** `Particle[i]` **Thực hiện:**
                - **Với mỗi** `Cluster[j]` **Thực hiện:**
                    - **Nếu** Hàm thích nghi của `Cluster[j]` thuộc `Particle[i]` tốt hơn Hàm thích nghi của `Cluster[j]` thuộc `pbest_i` **Thì:**
                        - Lưu `Cluster[j]` của `Particle[i]` vào thành `Cluster[j]` của `pbest_i`.
    - _(3) Cập nhật Vị trí tốt nhất toàn cục (`gbest`):_
        - **Với mỗi** `Cluster[j]` **Thực hiện:**
            - **Với mỗi** `Particle[i]` **Thực hiện:**
                - Lưu giá trị `pbest` tốt nhất (dựa trên Hàm thích nghi) vào `gbest`.
    - _(4) Điều chỉnh tham số thích nghi:_
        - Tính toán trọng số quán tính $\omega$ bằng Phương trình (11).
        - Tính toán hệ số $\mu$ bằng Phương trình (14).
    - _(5) Cập nhật Vận tốc và Vị trí hạt:_
        - **Với mỗi** `Particle[i]` **Thực hiện:**
            - Cập nhật véc-tơ Vận tốc `V[i]` bằng Phương trình (9).
            - Cập nhật véc-tơ Vị trí `X[i]` bằng Phương trình (10).

		**Kết thúc Thuật toán**

	- Sau khi toàn bộ quá trình lặp kết thúc, các tâm của mạng RBF (RBF unit centers) sẽ được thiết lập bằng giá trị tốt nhất toàn cục `gbest` tìm được.

	**Mục đích của thuật toán:** Thuật toán này thay thế các phương pháp phân cụm truyền thống như K-means để tự động tìm ra tọa độ không gian tối ưu cho các tâm của hàm cơ sở bán kính (RBF centers). Việc kết hợp đánh giá hàm thích nghi liên tục và tự điều chỉnh linh hoạt các hệ số $\omega, \mu$ giúp bầy đàn tránh bị kẹt ở cực tiểu cục bộ và tìm ra cụm dữ liệu chính xác hơn cho mạng nơ-ron.
   
	 Hơn nữa, Montazer và Giveki thảo luận về ảnh hưởng của việc điều chỉnh độ rộng đến hiệu suất của mạng RBFNN trong cả bài toán phân loại và xấp xỉ hàm. Do đó, họ đề xuất một phương pháp điều chỉnh độ rộng mới. Nhận thức được tầm quan trọng cao của sự phân bố không gian của tập dữ liệu huấn luyện và tính phi tuyến tính của hàm số mà người ta mong muốn xấp xỉ, họ đã tính đến chúng trong bài toán phân loại.
 
	 Do đó, khoảng cách Euclidean giữa các nút trung tâm và đạo hàm bậc hai của hàm số được xấp xỉ được sử dụng để đo hai yếu tố này. Vì chiều rộng của các nút trung tâm trong các khu vực phi tuyến tính cao phải nhỏ hơn chiều rộng của các nút trung tâm trong các khu vực bằng phẳng, Montazer và Giveki đã đề xuất tính toán chiều rộng của RBFNN theo Thuật toán 2.
	==**Thuật toán 2** điều chỉnh độ rộng  cho các hàm cơ sở bán kính trong mạng RBFNN.==

	Mục đích của thuật toán này là khắc phục hạn chế của các phương pháp tính độ rộng truyền thống, bằng cách kết hợp thông tin về sự phân bố không gian của tập dữ liệu huấn luyện để mạng có thể xử lý tốt các dữ liệu phức tạp 

	- **Bước 1:** Tính toán tọa độ các tâm của hàm cơ sở bán kính (RBF centers) bằng cách sử dụng thuật toán phân cụm PSO cải tiến (đã được mô tả ở Thuật toán 1).
	- **Bước 2:** Tính toán giá trị trung bình của bình phương khoảng cách (mean of squared distances) giữa tâm của cụm thứ $j$ và $p$-nút lân cận gần nhất (p-nearest neighbors) của nó.
	- **Bước 3:** Tính toán hệ số nhân (coefficient factor) theo công thức: $$coeff = \frac{d_{max}}{\sqrt{N}}$$ Trong đó, $N$ là tổng số lượng các nơ-ron ở lớp ẩn (hidden units) và $d_{max}$ là khoảng cách lớn nhất giữa các tâm này.
	- **Bước 4:** Tìm ra khoảng cách lớn nhất tính từ mỗi tâm, sau đó thực hiện chuẩn hóa véc-tơ khoảng cách (normalize the distance vector).
	- **Bước 5:** Nhân véc-tơ khoảng cách vừa thu được ở Bước 4 với hệ số nhân ($coeff$) đã tính ở Bước 3.
	- **Bước 6:** Cộng véc-tơ kết quả ở Bước 5 với véc-tơ đã tính được ở Bước 2. Tổng của hai véc-tơ này sẽ tạo ra các giá trị **độ rộng (widths)** cuối cùng cho mạng RBFNN theo mô hình PSO-OSD cải tiến.

	**Ý nghĩa của thuật toán:** Thông qua quy trình này, thuật toán đảm bảo rằng độ rộng của các nơ-ron trung tâm nằm trong những khu vực dữ liệu có độ phi tuyến cao (phức tạp) sẽ tự động được điều chỉnh nhỏ hơn so với độ rộng của các nơ-ron nằm ở những khu vực dữ liệu phẳng (flat areas). Điều này giúp mạng RBFNN tăng cường sức mạnh tổng quát hóa và khả năng phản hồi chính xác đối với tập dữ liệu mới. Đối với bài toán xấp xỉ hàm (function approximation), tài liệu còn đề xuất tính toán độ rộng kết hợp thêm với đạo hàm bậc hai để đo độ cong của hàm.

	Trong trường hợp gặp vấn đề xấp xỉ hàm, chiều rộng có thể được tính bằng cách sử dụng Phương trình [20] như sau:$$
\sigma_i =
\frac{d_{\max}}{\sqrt{N}}
\frac{r_i}{\bar{r}}
\left[
\frac{1}{1 + |f''(c_i)|}
\right]^{\frac{1}{4}} \tag{20}
$$

	Đối với **nút trung tâm** $c_i$, khoảng cách trung bình giữa nút này và $p$ nút lân cận gần nhất của nó được dùng để đo đặc trưng phân bố không gian tại nút trung tâm đó, được định nghĩa như sau:

$$
r_i =
\frac{1}{p}
\left(
\sum_{j=1}^{p}
\|c_j - c_i\|
\right)^{\frac{1}{2}} \tag{21}
$$

	Trong đó $c_j$ là **các nút lân cận gần nhất** của $c_i$.  
	$r_i$ là **khoảng cách mật độ tham chiếu** của $c_i$ và $\bar{r}$ là **giá trị trung bình của các khoảng cách mật độ tham chiếu của tất cả các nút trung tâm**, được tính như sau:

$$
\bar{r} =
\frac{1}{N}
\left(
\sum_{i=1}^{N} r_i
\right) \tag{22}
$$

	$f''(c_i)$ là đạo hàm bậc hai của hàm $f$ tại điểm $c_i$ và có thể được tính bằng **phương pháp sai phân hữu hạn trung tâm** (central finite difference method).  
	Do đạo hàm bậc hai được dùng để đo **độ cong của hàm**, nên **giá trị tuyệt đối của đạo hàm bậc hai** được sử dụng để so sánh mức độ phi tuyến của các vùng khác nhau trong tập dữ liệu.

	**Hàm cơ sở bán kính (RBF)** được đề xuất ban đầu để giải các bài toán **truy hồi ảnh (image retrieval)** trên các tập dữ liệu lớn. Một phiên bản mạnh hơn của **thuật toán tối ưu bầy đàn (PSO)**, Thuật toán PSO này được sử dụng để **điều chỉnh các tâm của các hàm Gaussian trong lớp ẩn của mạng RBFNN**. Mạng **RBFNN cải tiến** sau đó được sử dụng để giải bài toán **phân loại ảnh**.
 **2.2.2.3. Phát triển mạng RBFNN bằng Bình phương tối thiểu trực giao Growing RBFNN using orthogonal least squares (OLS)**
 1. Cơ chế hồi quy từng bước tiến (Forward Stepwise Regression)
    Thuật toán OLS hoạt động theo quy trình hồi quy từng bước tiến. Hệ thống không chọn tất cả các điểm dữ liệu làm tâm cùng một lúc mà tiến hành lựa chọn một cách tuần tự. Tiêu chí cốt lõi để chọn một tâm đưa vào mạng là nó phải tạo ra sự sụt giảm lớn nhất đối với tổng bình phương sai số ở đầu ra. Nhờ vậy, cấu trúc mạng RBFNN phát triển dần một cách tinh gọn nhưng đem lại hiệu quả xấp xỉ cao nhất.
 2. Biểu diễn tóan học trong không gian trực giao (Orthogonnal Space)
    - Thuật toán OLS xây dựng một tập hợp các vecto trực giao (ký hiệu là Q) cho không gian được tạo ra bởi các vecto cơ sở $\phi_k$.
    - Ma trận tổng quát $\Phi$ sẽ được phân tích thành $\Phi =QA$, trong đó A là một ma trận tam giác trên.
    - Bằng cách sử dụng biểu diễn trực giao này, nghiệm bài toán mạng RBF được thể hiện bằng công thức: $$T=\Phi W=QG \tag{23}$$
    - Nghiệm bình phương tối thiểu để tìm vecto trọng số G trong không gian trực giao được tính toán bằng công thức:$$G=(Q'Q)^{-1}Q'T \tag{24}$$
    3. **Ưu điểm cốt lõi: Loại bỏ ma trận giả nghịch đảo** Việc tính toán các trọng số kết nối bằng phương pháp Giả nghịch đảo (Pseudoinverse) truyền thống đòi hỏi phải nghịch đảo ma trận kích thước cực lớn, gây tốn kém tính toán và bộ nhớ. OLS khắc phục hoàn toàn vấn đề này. Khi đưa vào không gian không gian trực giao, phép nhân ma trận Q′Q sẽ tạo ra một **ma trận đường chéo (diagonal matrix)**. Nhờ tính chất đặc biệt của ma trận đường chéo, hệ thống **tránh được hoàn toàn việc tính toán giả nghịch đảo** phức tạp, giúp thuật toán thực thi nhanh và tiết kiệm tài nguyên hệ thống đáng kể.
# 3. Mạng Bộ nhớ Ngắn hạn (Long Short-Term Memory - LSTM) ([https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://phamdinhkhanh.github.io/2019/04/22/Ly_thuyet_ve_mang_LSTM.html))
## 3.1. Mạng nơ ron truy hồi (RNN - Recurrent Neural Network)

Trong lý thuyết về ngôn ngữ, ngữ nghĩa của một câu được tạo thành từ mối liên kết của những từ trong câu theo một cấu trúc ngữ pháp. Nếu xét từng từ một đứng riêng lẻ ta không thể hiểu được nội dụng của toàn bộ câu, nhưng dựa trên những từ xung quanh ta có thể hiểu được trọn vẹn một câu nói. Như vậy cần phải có một kiến trúc đặc biệt hơn cho các mạng nơ ron biểu diễn ngôn ngữ nhằm mục đích liên kết các từ liền trước với các từ ở hiện tại để tạo ra mối liên hệ xâu chuỗi. Mạng nơ ron truy hồi đã được thiết kế đặc biệt để giải quyết yêu cầu này:

![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-rolled.png)

> **Hình 1: Mạng nơ ron truy hồi với vòng lặp**

Hình trên biểu diễn kiến trúc của một mạng nơ ron truy hồi. Trong kiến trúc này mạng nơ ron sử dụng một đầu vào là một véc tơ  và trả ra đầu ra là một giá trị ẩn . Đầu vào được đấu với một thân mạng nơ ron  có tính chất truy hồi và thân này được đấu tới đầu ra .

Vòng lặp  ở thân mạng nơ ron là điểm mấu chốt trong nguyên lý hoạt động của mạng nơ ron truy hồi. Đây là chuỗi sao chép nhiều lần của cùng một kiến trúc nhằm cho phép các thành phần có thể kết nối liền mạch với nhau theo mô hình chuỗi. Đầu ra của vòng lặp trước chính là đầu vào của vòng lặp sau. Nếu trải phẳng thân mạng nơ ron  ta sẽ thu được một mô hình dạng:

![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

> **Hình 2: Cấu trúc trải phẳng của mạng nơ ron truy hồi**

Kiến trúc mạng nơ ron truy hồi này tỏ ra khá thành công trong các tác vụ của deep learning như: Nhận diện dọng nói (_speech recognition_), các mô hình ngôn ngữ, mô hình dịch, chú thích hình ảnh (_image captioning_),….

## 3.2. Hạn chế của mạng nơ ron truy hồi

Một trong những điểm đặc biệt của RNN đó là nó có khả năng kết nối các thông tin liền trước với nhiệm vụ hiện tại, chẳng hạn như trong câu văn: ‘học sinh đang tới _trường học_’. Dường như trong một ngữ cảnh ngắn hạn, từ _trường học_ có thể được dự báo ngay tức thì mà không cần thêm các thông tin từ những câu văn khác gần đó. Tuy nhiên có những tình huống đòi hỏi phải có nhiều thông tin hơn chẳng hạn như: ‘hôm qua Bống đi học nhưng không mang áo mưa. Trên đường đi học trời mưa. Cặp sách của Bống bị _ướt_’. Chúng ta cần phải học để tìm ra từ _ướt_ ở một ngữ cảnh dài hơn so với chỉ 1 câu. Tức là cần phải biết các sự kiện trước đó như _trời mưa_, _không mang áo mưa_ để suy ra sự kiện bị _ướt_. Những sự liên kết ngữ nghĩa dài như vậy được gọi là `phụ thuộc dài hạn` (_long-term dependencies_). Về mặt lý thuyết mạng RNN có thể giải quyết được những sự phụ thuộc trong dài hạn. Tuy nhiên trên thực tế RNN lại cho thấy khả năng học trong dài hạn kém hơn. Một trong những nguyên nhân chính được giải thích đó là sự triệt tiêu đạo hàm của hàm cost function sẽ diễn ra khi trải quả chuỗi dài các tính toán truy hồi. Một phiên bản mới của mạng RNN là mạng LSTM ra đời nhằm khắc phục hiện tường này nhờ một cơ chế đặc biệt.

## 3.3. Mạng trí nhớ ngắn hạn định hướng dài hạn (LSTM - Long short term memory)

Mạng _trí nhớ ngắn hạn định hướng dài hạn_ còn được viết tắt là LSTM làm một kiến trúc đặc biệt của RNN có khả năng học được sự phục thuộc trong dài hạn (_long-term dependencies_) được giới thiệu bởi [Hochreiter & Schmidhuber (1997)](http://www.bioinf.jku.at/publications/older/2604.pdf). Kiến trúc này đã được phổ biến và sử dụng rộng rãi cho tới ngày nay. LSTM đã tỏ ra khắc phục được rất nhiều những hạn chế của RNN trước đây về triệt tiêu đạo hàm. Tuy nhiên cấu trúc của chúng có phần phức tạp hơn mặc dù vẫn dữ được tư tưởng chính của RNN là sự sao chép các kiến trúc theo dạng chuỗi.

Một mạng RNN tiêu chuẩn sẽ có kiến trúc rất đơn giản chẳng hạn như đối với kiến trúc gồm một tầng ẩn là hàm tanh như bên dưới.

![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png)

> **Hình 3: Sự lặp lại kiến trúc module trong mạng RNN chứa một tầng ẩn**

LSTM cũng có một chuỗi dạng như thế nhưng phần kiến trúc lặp lại có cấu trúc khác biệt hơn. Thay vì chỉ có một tầng đơn, chúng có tới 4 tầng ẩn (3 sigmoid và 1 tanh) tương tác với nhau theo một cấu trúc đặc biệt.

![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

> **Hình 4: Sự lặp lại kiến trúc module trong mạng LSTM chứa 4 tầng ẩn (3 sigmoid và 1 tanh) tương tác**

Các kí hiệu có thể diễn giải như sau:

![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM2-notation.png)

> **Hình 5: Diễn giải các kí hiệu trong đồ thị mạng nơ ron (áp dụng chung cho toàn bộ bài)**

Trong sở đồ tính toán trên, mỗi một phép tính sẽ triển khai trên một véc tơ. Trong đó hình tròn màu hồng biểu diễn một toán tử đối với véc tơ như phép cộng véc tơ, phép nhân vô hướng các véc tơ. Màu vàng thể hiện hàm activation mà mạng nơ ron sử dụng để học trong tầng ẩn, thông thường là các hàm phi tuyến sigmoid và tanh. Kí hiệu 2 đường thẳng nhập vào thể hiện phép chập kết quả trong khi kí hiệu 2 đường thẳng rẽ nhánh thể hiện cho nội dung véc tơ trước đó được sao chép để đi tới một phần khác của mạng nơ ron.

## 3.4. Ý tưởng đằng sau LSTM

Ý tưởng chính của LSTM là thành phần ô trạng thái (cell state) được thể hiện qua đường chạy ngang qua đỉnh đồ thị như hình vẽ bên dưới:

![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-C-line.png)

> **Hình 6: Đường đi của ô trạng thái (cell state) trong mạng LSTM**

Ô trạng thái là một dạng băng chuyền chạy thẳng xuyên suốt toàn bộ chuỗi với chỉ một vài tương tác tuyến tính nhỏ giúp cho thông tin có thể truyền dọc theo đồ thị mạng nơ ron ổn định.

LSTM có khả năng xóa và thêm thông tin vào ô trạng thái và điều chỉnh các luồng thông tin này thông qua các cấu trúc gọi là cổng.

Cổng là cơ chế đặc biệt để điều chỉnh luồng thông tin đi qua. Chúng được tổng hợp bởi một tầng ẩn của hàm activation sigmoid và với một toán tử nhân như đồ thị.

![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-gate.png)

> **Hình 7: Một cổng của hàm sigmoid trong LSTM**

Hàm sigmoid sẽ cho đầu ra là một giá trị xác xuất nằm trong khoảng từ 0 đến 1, thể hiện rằng có bao nhiêu phần thông tin sẽ đi qua cổng. Giá trị bằng 0 ngụ ý rằng không cho phép thông tin nào đi qua, giá trị bằng 1 sẽ cho toàn bộ thông tin đi qua.

Một mạng LSTM sẽ có 3 cổng có kiến trúc dạng này để bảo vệ và kiểm soát các ô trạng thái.

## 3.5. Thứ tự các bước của LSTM

Bước đầu tiên trong LSTM sẽ quyết định xem thông tin nào chúng ta sẽ cho phép đi qua ô trạng thái (cell state). Nó được kiểm soát bởi hàm sigmoid trong một tầng gọi là tầng quên (_forget gate layer_). Đầu tiên nó nhận đầu vào là 2 giá trị  và  và trả về một giá trị nằm trong khoảng 0 và 1 cho mỗi giá trị của ô trạng thái . Nếu giá trị bằng 1 thể hiện ‘giữ toàn bộ thông tin’ và bằng 0 thể hiện ‘bỏ qua toàn bộ chúng’.

Trở lại ví dụ về ngôn ngữ, chúng ta đang cố gắng dự báo từ tiếp theo dựa trên toàn bộ những từ trước đó. Trong những bài toán như vậy, ô trạng thái có thể bao gồm loại của chủ ngữ hiện tại, để cho đại từ ở câu tiếp theo được sử dụng chính xác. Chẳng hạn như chúng ta đang mô tả về một người bạn là con trai thì các đại từ nhân xưng ở tiếp theo phải là anh, thằng, hắn thay vì cô, con ấy (xin lỗi vì lấy ví dụ hơi thô). Tuy nhiên chủ ngữ không phải khi nào cũng cố định. Khi chúng ta nhìn thấy một chủ ngữ mới, chúng ta muốn quên đi loại của một chủ ngữ cũ. Do đó tầng quên cho phép cập nhật thông tin mới và lưu giữ giá trị của nó khi có thay đổi theo thời gian.

![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png)

> **Hình 8: Tầng cổng quên (_forget gate layer_)**

Bước tiếp theo chúng ta sẽ quyết định loại thông tin nào sẽ được lưu trữ trong ô trạng thái. Bước này bao gồm 2 phần. Phần đầu tiên là một tầng ẩn của hàm sigmoid được gọi là tầng cổng vào (_input gate layer_) quyết định giá trị bao nhiêu sẽ được cập nhật. Tiếp theo, tầng ẩn hàm tanh sẽ tạo ra một véc tơ của một giá trị trạng thái mới  mà có thể được thêm vào trạng thái. Tiếp theo kết hợp kết quả của 2 tầng này để tạo thành một cập nhật cho trạng thái.

Trong ví dụ của mô hình ngôn ngữ, chúng ta muốn thêm loại của một chủ ngữ mới vào ô trạng thái để thay thế phần trạng thái cũ muốn quên đi.

![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png)

> **Hình 9: Cập nhật giá trị cho ô trạng thái bằng cách kết hợp 2 kết quả từ tầng cổng vào và tẩng ẩn hàm tanh**

Đây là thời điểm để cập nhật một ô trạng thái cũ,  sang một trạng thái mới . Những bước trước đó đã quyết định làm cái gì, và tại bước này chỉ cần thực hiện nó.

Chúng ta nhân trạng thái cũ với  tương ứng với việc quên những thứ quyết định được phép quên sớm. Phần tử đề cử  là một giá trị mới được tính toán tương ứng với bao nhiêu được cập nhật vào mỗi giá trị trạng thái.

![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png)

> **Hình 10: Ô trạng thái mới**

Cuối cùng cần quyết định xem đầu ra sẽ trả về bao nhiêu. Kết quả ở đầu ra sẽ dựa trên ô trạng thái, nhưng sẽ là một phiên bản được lọc. Đầu tiên, chúng ta chạy qua một tầng sigmoid nơi quyết định phần nào của ô trạng thái sẽ ở đầu ra. Sau đó, ô trạng thái được đưa qua hàm tanh (để chuyển giá trị về khoảng -1 và 1) và nhân nó với đầu ra của một cổng sigmoid, do đó chỉ trả ra phần mà chúng ta quyết định.

![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png)

> **Hình 11: Điều chỉnh thông tin ở đầu ra thông qua hàm tanh**


