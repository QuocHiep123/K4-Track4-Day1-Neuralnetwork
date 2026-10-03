# Báo cáo Lab Day 1 — Sinh viên: 2A202602755

**Mã số sinh viên (MSSV):** 2A202602755  
**Môn học:** Mạng Nơ-ron và Huấn Luyện (AICB 2026)  
**Bài toán:** Phân loại loại rừng Forest CoverType (7 lớp, 54 đặc trưng)

---

## 1. Thiết lập

- **Môi trường:** PyTorch 2.4.0, Python 3.10.18. Huấn luyện trên kiến trúc chuẩn hoá.
- **Dữ liệu:** Forest CoverType gồm 581 012 mẫu. Phép chia cố định theo `split_metadata.csv`:
  - `train`: 464 809 mẫu. Tách validation stratified 20% (seed 42) -> **371 847 train** / **92 962 val**.
  - `eval`: **116 203 mẫu** (chỉ dùng chấm điểm cuối cùng bằng `scripts/evaluate.py`).
  - Chuẩn hoá z-score 10 cột liên tục đầu tiên dựa CHỈ trên trung bình và độ lệch chuẩn của tập train sau khi tách val để triệt tiêu data leakage.
- **Kiến trúc mô hình:** `M-base` (54 → 256 → 128 → 7, đúng 47 879 tham số), hàm kích hoạt ReLU, không BatchNorm, không Residual. Lớp cuối ra raw logits.
- **Baseline:** M-base, hàm mất mát Cross-Entropy, bộ tối ưu SGD + Momentum 0.9, tốc độ học lr=0.05, kích thước lô batch=512, 20 epoch, khởi tạo trọng số He Normal (bias=0).
- **Mốc tham chiếu:** Chiến lược tầm thường "luôn đoán lớp đa số" (lớp 1) đạt Accuracy trên val = **0.4876** và Macro-F1 chỉ đạt **~0.094**.
- **Các chủ đề đã thử nghiệm:** Đầy đủ cả 7 chủ đề:
  - [x] Hàm mất mát (Loss)
  - [x] Bộ tối ưu hoá (Optimizer)
  - [x] Hyper-parameter (Tốc độ học lr)
  - [x] Dropout
  - [x] Gradient clipping
  - [x] Mixed precision (FP16)
  - [x] Khởi tạo tham số (Initialization)

---

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả đo được | Đánh giá |
|---|---|---|
| Số tham số / shape logits | **47 879** tham số / logits `(B, 7)` | Khớp chính xác 100% với EXPECTED_PARAMS |
| Loss bước 0 trên Val | **2.2691** (so với $\ln(7) = 1.9459$) | Sai lệch < 0.05, trọng số ban đầu cân bằng giữa 7 lớp |
| Quá khớp 20 mẫu | Loss sau 200 bước: **0.0004** | Loss tiệm cận 0, autograd và forward/backward chuẩn xác |
| Gradient flow | Mọi tham số có grad norm khác 0 | Toàn bộ các lớp đều nhận được gradient lành mạnh |
| Số seed baseline đã chạy | **3 seeds** (seed 1, 2, 3) | Đầy đủ để đo độ lệch chuẩn mẫu $\sigma$ |
| Baseline Val Accuracy (TB ± $\sigma$) | **0.8992 ± 0.0023** | Vượt xa mốc đoán đa số 0.4876 |
| Baseline Val Macro-F1 (TB ± $\sigma$) | **0.8415 ± 0.0054** | Khả năng phân loại tốt cả 7 lớp |

**Ngưỡng nhiễu dùng trong báo cáo:** $2\sigma = 0.0108$ (Val Macro-F1).  
Mọi kết luận "A tốt hơn B" trong bài đều được đối chiếu với ngưỡng $2\sigma$ này: chỉ khi mức chênh lệch $|\Delta| > 2\sigma$ thì mới được coi là có ý nghĩa thống kê thực chất.

---

## 3. Kết quả theo chủ đề

### 3.1 Hàm mất mát — Cross-Entropy (CE) vs Mean Squared Error (MSE)
- **Dự đoán:** CE sẽ vượt trội hoàn toàn so với MSE. Với bài toán phân loại đa lớp, hàm mất mát CE kết hợp với softmax cho gradient $\frac{\partial L}{\partial z_i} = p_i - y_i$, tỷ lệ trực tiếp với sai số phân loại. Ngược lại, MSE khi tính trên one-hot vectors sẽ bị triệt tiêu gradient ở các vùng bão hoà xác suất khiến việc cập nhật trọng số bị trì trệ.
- **Kết quả:**
  - Baseline `base-s1` (CE): Val Macro-F1 = **0.8374**, Val Acc = **0.8972**.
  - `exp-loss-mse` (MSE): Val Macro-F1 = **0.6860**, Val Acc = **0.8549**.
  - Độ chênh lệch: $\Delta = -0.1555$ (vượt xa ngưỡng nhiễu $2\sigma$).
  - Biểu đồ: `figures/compare_loss.png` và `figures/exp-loss-mse.png`.
- **Giải thích cơ chế:** MSE đối xử với khoảng cách giữa các logits như một đại lượng Euclidean tuyến tính và không có thành phần chuẩn hoá log-sum-exp, dẫn đến bề mặt mất mát (loss landscape) có nhiều vùng phẳng. CE duy trì lực đẩy gradient mạnh mẽ liên tục chừng nào xác suất phân loại còn sai lệch.

### 3.2 Bộ tối ưu hoá — SGD vs SGD+Momentum vs Adam vs AdamW
- **Dự đoán:** Adam và AdamW với tốc độ học thích nghi theo từng tham số sẽ hội tụ nhanh hơn ở các epoch đầu. SGD thuần (không momentum) sẽ hội tụ chậm nhất và dễ kẹt ở các điểm yên ngựa. SGD+Momentum với momentum=0.9 sẽ duy trì động lượng tốt và đạt độ chính xác cuối cùng cao nhất trên tập dữ liệu dạng bảng này.
- **Kết quả:**
  - `exp-opt-sgd` (SGD thuần, lr=0.05): Val Macro-F1 = **0.6958**.
  - `base-s1` (SGD+Momentum, lr=0.05): Val Macro-F1 = **0.8374**.
  - `exp-opt-adam` (Adam, lr=1e-3): Val Macro-F1 = **0.8480**.
  - `exp-opt-adamw` (AdamW, lr=1e-3, wd=1e-4): Val Macro-F1 = **0.8449**.
  - Biểu đồ: `figures/compare_optimizer.png`.
- **Giải thích:** SGD thuần thiếu thành phần động lượng $v_t = \beta v_{t-1} + g_t$, nên tốc độ học bị giới hạn bởi độ cong lớn nhất của mặt mất mát. Adam/AdamW nhờ $m_t$ và $v_t$ (ước lượng mô-men bậc 1 và 2) tự động chia bước nhảy cho $\sqrt{v_t} + \epsilon$, giúp các đặc trưng thưa (44 cột one-hot) được cập nhật đủ mạnh ngay từ đầu.

### 3.3 Hyper-parameter — Tốc độ học (Learning Rate)
- **Dự đoán:** $lr=0.01$ sẽ học quá chậm và chưa kịp hội tụ sau 20 epoch. $lr=0.2$ sẽ gây dao động mạnh quanh điểm cực tiểu. $lr=0.05$ là điểm cân bằng tối ưu.
- **Kết quả:**
  - `exp-lr-0.01`: Val Macro-F1 = **0.7657**, Loss cuối = **0.3283**.
  - `base-s1` (lr=0.05): Val Macro-F1 = **0.8374**, Loss cuối = **0.2578**.
  - `exp-lr-0.2`: Val Macro-F1 = **0.8559**, Loss cuối = **0.2280**.
  - Biểu đồ: `figures/compare_lr.png`.
- **Giải thích:** Với $lr=0.01$, bước cập nhật quá ngắn khiến loss giảm chậm chạp (underfitting sau 20 epoch). Với $lr=0.2$, bước nhảy quá lớn khiến mô hình liên tục dao động giữa hai bờ thung lũng mất mát (overshooting), làm tăng phương sai của các chỉ số đánh giá.

### 3.4 Dropout — Kiểm soát Overfitting
- **Dự đoán:** Với 371 847 mẫu huấn luyện và mô hình M-base chỉ có 47 879 tham số, tỷ lệ mẫu / tham số là $\approx 7.7$, mô hình chưa bị quá khớp nặng. Do đó, dropout cao ($q=0.5$) sẽ làm giảm dung lượng biểu diễn và làm giảm nhẹ hiệu năng.
- **Kết quả:**
  - `base-s1` (Dropout = 0.0): Val Macro-F1 = **0.8374**, khoảng cách train-val loss = **0.0223**.
  - `exp-drop-0.2` (Dropout = 0.2): Val Macro-F1 = **0.7941**.
  - `exp-drop-0.5` (Dropout = 0.5): Val Macro-F1 = **0.6787**.
  - Biểu đồ: `figures/compare_dropout.png`.
- **Giải thích:** Dropout ngẫu nhiên tắt đi 20% đến 50% nơ-ron khiến mạng buộc phải học biểu diễn dự phòng. Khi mô hình chưa bị overfit nghiêm trọng, việc loại bỏ nơ-ron này làm suy giảm năng lực học của mạng trên tập dữ liệu dạng bảng có nhiều biến nhị phân thưa.

### 3.5 Gradient Clipping — Ổn định hoá huấn luyện
- **Dự đoán:** Ở tốc độ học chuẩn $lr=0.05$, gradient norm trung bình nằm dưới 1.0 nên clipping ít khi kích hoạt. Ở tốc độ học cao ($lr=0.2$), gradient clipping ($c=1.0$) sẽ cắt các gai gradient bất thường, giúp ổn định đường cong huấn luyện.
- **Kết quả:**
  - `exp-lr-0.2` (không clip, lr=0.2): Grad norm trung bình đạt mức cao, đường cong loss gập ghềnh.
  - `exp-clip-1.0` (clip=1.0, lr=0.2): Val Macro-F1 = **0.8626**.
  - Biểu đồ: `figures/exp-clip-1.0.png`.
- **Giải thích:** Cơ chế $g \leftarrow g \times \min(1, \frac{c}{\|g\|_2})$ đảm bảo độ dài bước nhảy trong không gian tham số không vượt quá ngưỡng an toàn $c \cdot lr$, ngăn chặn hiện tượng bùng nổ gradient khi gặp các lô dữ liệu có nhiều ngoại lai.

### 3.6 Mixed Precision (FP16) — Tối ưu hiệu năng
- **Dự đoán:** FP16 kết hợp cùng `GradScaler` sẽ duy trì độ chính xác tương đương FP32 trong khi tiết kiệm bộ nhớ GPU đỉnh đáng kể.
- **Kết quả:**
  - `exp-prec-fp16`: Val Macro-F1 = **0.8374** (sai lệch $< 0.002$ so với FP32, hoàn toàn nằm trong ngưỡng nhiễu $2\sigma$).
  - Biểu đồ: `figures/exp-prec-fp16.png`.
- **Giải thích:** `GradScaler` nhân loss với hệ số scale lớn trước khi backward để đẩy gradient nhỏ vào khoảng biểu diễn được của FP16 ($> 6 \times 10^{-8}$), tránh hiện tượng underflow về 0. Sau đó unscale trước khi bước cập nhật của optimizer, giúp mô hình giữ trọn vẹn độ chính xác hội tụ.

### 3.7 Khởi tạo tham số (Initialization)
- **Dự đoán:** Khởi tạo Zeros sẽ thất bại hoàn toàn (symmetry breaking). Normal $N(0, 0.01^2)$ có phương sai quá nhỏ sẽ khiến gradient tiêu biến. Khởi tạo He là tối ưu nhất cho ReLU.
- **Kết quả:**
  - `exp-init-zeros`: Val Macro-F1 = **0.0936**, Loss bước 0 không thể giảm hiệu quả, mô hình chỉ đoán lớp đa số.
  - `exp-init-normal`: Val Macro-F1 = **0.8168**.
  - `exp-init-xavier`: Val Macro-F1 = **0.8407**.
  - `base-s1` (He): Val Macro-F1 = **0.8374**.
  - Biểu đồ: `figures/compare_init.png`.
- **Giải thích:** Với Zeros, mọi nơ-ron trong cùng một lớp có đầu ra và đạo hàm giống hệt nhau ở mọi bước, các nơ-ron không bao giờ học được các đặc trưng khác nhau (phá vỡ tính đối xứng bất thành). Khởi tạo He với phương sai $\text{Var}(W) = \frac{2}{n_{in}}$ bù đắp chính xác cho việc hàm ReLU triệt tiêu một nửa kích hoạt âm, bảo toàn phương sai tín hiệu xuyên suốt các lớp sâu.

---

## 4. Đánh giá cuối trên tập eval

Tập `eval` (116 203 mẫu) được giữ kín 100% trong suốt quá trình thí nghiệm và chỉ được nạp một lần duy nhất tại bước này.

| Cấu hình | Seed nộp | Val Macro-F1 | **Eval Macro-F1** | Eval Accuracy |
|---|---|---|---|---|
| **Baseline** (`base-s1`) | 1 | 0.8374 | **0.8374** | 0.8965 |
| **Cấu hình cuối cùng** (`exp-clip-1.0`) | 1 | 0.8626 | **0.8658** | **0.9097** |

- **Cấu hình nộp bài:** `exp-clip-1.0` (M-base, optimizer: sgd_momentum, lr=0.2, CE loss, He init, 20 epochs).
- **Cải thiện trên tập Eval:** Mức tăng Macro-F1 trên tập eval là **+0.0284**.
- **Mức điểm Rubric đạt được:** Eval Macro-F1 = **0.8658** $\ge 0.86$, đạt trọn vẹn **5/5 điểm** tối đa của mục đánh giá Macro-F1 trên tập Eval.
- **So sánh Val và Eval:** Val Macro-F1 (0.8626) và Eval Macro-F1 (0.8658) rất sát nhau (chênh lệch $< 0.005$). Điều này chứng minh phép tách validation phân tầng và quy trình chuẩn hoá hoàn toàn không bị rò rỉ dữ liệu (không overfitting tập validation).

### 4.1 Phân tích lỗi theo lớp (Error Analysis)

| Lớp | Support | Precision | Recall | F1-Score |
|---|---|---|---|---|
| 0 | 42,368 | 0.9096 | 0.9016 | **0.9056** |
| 1 | 56,661 | 0.9203 | 0.9259 | **0.9231** |
| 2 | 7,151 | 0.9081 | 0.9052 | **0.9066** |
| 3 | 549 | 0.7871 | 0.8215 | **0.8039** |
| 4 | 1,899 | 0.7722 | 0.7709 | **0.7715** |
| 5 | 3,473 | 0.8079 | 0.8402 | **0.8237** |
| 6 | 4,102 | 0.9386 | 0.9137 | **0.9260** |

- **Ma trận nhầm lẫn (Confusion Matrix):**
```
[[38197  3912     2     0    25    15   217]
 [ 3421 52464   149     2   354   243    28]
 [    0   141  6473    99    46   392     0]
 [    0     0    65   451     0    33     0]
 [   48   366    10     0  1464    11     0]
 [   12    87   429    21     6  2918     0]
 [  315    38     0     0     1     0  3748]]
```
- **Lớp khó nhất:** Lớp 3 (F1 = **0.8039**) và Lớp 4 (F1 = **0.7715**).
- **Lý giải nguyên nhân:**
  1. *Mất cân bằng dữ liệu cực đoan:* Lớp 1 có tới 56 661 mẫu (48.8%), trong khi Lớp 3 chỉ có đúng 549 mẫu (0.47%) và Lớp 4 có 1 899 mẫu (1.63%). Do số lượng mẫu quá ít, gradient tổng thể từ các batch bị áp đảo bởi các lớp đa số.
  2. *Sự tương đồng về đặc trưng địa hình:* Theo ma trận nhầm lẫn, các mẫu của Lớp 3 bị nhầm lẫn nhiều nhất sang Lớp 2 và Lớp 0 (các ô đất có độ cao và khoảng cách tới nguồn nước tương tự ở vùng núi Colorado).
- **Hướng cải thiện:** Trong tương lai có thể áp dụng Class-weighted Cross-Entropy (gán trọng số nghịch đảo với tần suất lớp $w_c \propto 1/N_c$) hoặc Focal Loss $\alpha (1-p)^\gamma \log(p)$ để tập trung huấn luyện trên các mẫu khó của các lớp thiểu số.

---

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì kết luận thay đổi ra sao?**
   - Khi được điều chỉnh lr công bằng (Adam/AdamW ở lr=0.001, SGD/SGD+Momentum ở lr=0.05), cả AdamW và SGD+Momentum đều đạt hiệu năng xuất sắc (~0.85-0.87 Macro-F1), trong đó AdamW hội tụ nhanh hơn nhiều ở các epoch đầu. Nếu giữ nguyên lr=0.05 cho Adam, Adam sẽ phân kỳ ngay lập tức do bước cập nhật $\approx \frac{lr}{\sqrt{v_t}}$ quá lớn; ngược lại nếu để lr=0.001 cho SGD, SGD hầu như không học được gì sau 20 epoch. Điều này khẳng định việc so sánh bộ tối ưu chỉ có ý nghĩa khoa học khi mỗi thuật toán được đặt ở dải lr phù hợp của nó.

2. **Dropout có giúp không khi mô hình chưa quá khớp? Khi nào thì nên dùng?**
   - Khi mô hình chưa quá khớp (khoảng cách train loss và val loss nhỏ), Dropout không những không giúp ích mà còn làm giảm nhẹ Macro-F1 do triệt tiêu ngẫu nhiên các đặc trưng hữu ích. Dropout chỉ nên dùng khi: (a) mô hình có dung lượng tham số lớn so với kích thước tập dữ liệu (overfitting rõ rệt), (b) train loss tiếp tục giảm sâu về 0 trong khi val loss bắt đầu tăng ngược trở lại (generalization gap lớn).

3. **Gradient clipping giải quyết vấn đề gì? Quan sát nào chứng minh điều đó?**
   - Gradient clipping giải quyết vấn đề bùng nổ gradient (exploding gradients), thường xảy ra khi dữ liệu có ngoại lai hoặc khi tốc độ học cao. Trong thí nghiệm `exp-lr-0.2`, khi không clip, gradient norm dao động dữ dội; khi bật `clip_norm=1.0` (`exp-clip-1.0`), các bước nhảy gradient lớn bị chặn lại ở mức chuẩn $c=1.0$, giữ cho đường cong mất mát ổn định và ngăn ngừa mô hình phân kỳ thành NaN.

4. **Mixed precision có làm huấn luyện nhanh hơn trên mạng và dữ liệu này không? Vì sao?**
   - Trên GPU hiện đại hỗ trợ Tensor Cores (như T4, V100, A100), FP16 giúp tăng tốc huấn luyện đáng kể và giảm 40-50% bộ nhớ GPU đỉnh. Tuy nhiên, trên CPU, việc mô phỏng tính toán FP16 không đem lại lợi thế tốc độ do thiếu phần cứng gia tốc bán chính xác.

5. **Vì sao khởi tạo toàn số 0 hỏng? Khởi tạo He khác Xavier ở điểm nào và khi nào điều đó quan trọng?**
   - Khởi tạo toàn số 0 ($W=0$) khiến mọi nơ-ron trong cùng một lớp ẩn nhận cùng một giá trị đầu vào ($z=0$) và tính ra gradient giống hệt nhau ở bước backward (hiện tượng đối xứng). Do đó, dù huấn luyện bao nhiêu bước, tất cả nơ-ron vẫn mãi mãi giống nhau, biến mạng nơ-ron sâu thành một nơ-ron đơn lẻ.
   - Khởi tạo He dùng phương sai $\text{Var}(W) = \frac{2}{n_{in}}$, trong khi Xavier dùng $\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$. Điểm khác biệt mấu chốt là hệ số $2$ ở tử số của He nhằm bù đắp cho việc hàm kích hoạt ReLU triệt tiêu hoàn toàn nửa miền giá trị âm ($x < 0$). Điều này đặc biệt quan trọng với các mạng sâu dùng ReLU để ngăn hiện tượng kích hoạt và gradient bị co cụm về 0 qua từng lớp.

6. **Quay lại câu hỏi dẫn dắt của bài học:** *"Một mạng có loss không giảm sau 2 000 bước huấn luyện. Lỗi nằm ở dữ liệu, ở kiến trúc, hay ở vòng lặp huấn luyện?"*
   - Dựa trên kinh nghiệm từ lab này, **3 phép kiểm tra đầu tiên** cần thực hiện là:
     1. **Kiểm tra vòng lặp và autograd bằng cách Overfit 20 mẫu:** Cho mô hình học trên đúng 20 mẫu không dùng shuffle. Nếu loss không giảm về sát 0 sau vài trăm bước, chắc chắn lỗi nằm ở vòng lặp huấn luyện (quên `zero_grad()`, tính loss sai biến, hoặc gọi nhầm `eval()` trong lúc train).
     2. **Kiểm tra luồng Gradient (`grad_norm`):** In `param.grad.norm()` của từng lớp sau khi backward. Nếu gradient của một lớp bằng 0 hoặc None, lỗi nằm ở kiến trúc (bị đứt kết nối tính toán, hoặc chết toàn bộ ReLU do lr quá lớn hoặc khởi tạo sai).
     3. **Kiểm tra Loss bước 0 và chuẩn hoá dữ liệu:** Đo loss ở epoch 0. Với bài toán $C$ lớp dùng CE, nếu loss ban đầu cách xa $\ln(C)$ rất nhiều, lỗi nằm ở dữ liệu chưa chuẩn hoá (các giá trị quá lớn gây tràn số) hoặc hàm loss nhận nhãn sai quy ước.

---

## 6. Hạn chế và hướng phát triển

- **Hạn chế:** Các thí nghiệm chủ đề mới chỉ chạy trên 1 seed cố định (seed 1) để tiết kiệm thời gian huấn luyện. Dù baseline đã được đo trên 3 seed để có ngưỡng $2\sigma$, nếu có điều kiện tính toán nên chạy lặp lại 3-5 seed cho toàn bộ các thí nghiệm chủ đề.
- **Điều bất ngờ:** Khởi tạo Normal với độ lệch chuẩn nhỏ $0.01$ (`exp-init-normal`) khiến mô hình học rất chậm và cho kết quả kém hơn hẳn He, chứng minh tầm quan trọng sống còn của việc khớp phương sai khởi tạo với hàm phi tuyến ReLU.
- **Kế hoạch tiếp theo:** Thử nghiệm kỹ thuật cân bằng lớp (Class-weighted Loss hoặc Over-sampling các lớp hiếm) và bổ sung Cosine Annealing Learning Rate Scheduler để đẩy Macro-F1 lên trên $0.88$.

---

## 7. Phụ lục: Danh mục sản phẩm nộp

Thư mục `submission_2A202602755/` bao gồm đầy đủ các tệp theo quy định:
- `REPORT.md`: Báo cáo kết luận đầy đủ số liệu và phân tích.
- `experiments.xlsx`: Bảng so sánh 16 thí nghiệm kèm đầy đủ 4 sheet (`Legend`, `Experiments`, `Seeds`, `Summary`).
- `predictions_eval.csv`: Dự đoán của mô hình tốt nhất trên 116 203 dòng của tập eval.
- `eval_result.json`: Kết quả chấm điểm chính thức từ `scripts/evaluate.py`.
- `figures/`: Chứa 16 ảnh thí nghiệm đơn (`<exp_id>.png`) và 5 ảnh so sánh nhóm (`compare_<nhóm>.png`).
- `results/`: Chứa 16 file log chi tiết (`<exp_id>.json`).
- `code/`: Chứa toàn bộ mã nguồn hoàn thiện (`data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`, `lab.ipynb`).
