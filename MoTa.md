# MoTa.md

# Mô tả hướng paper PRICAI 2026: ReCIL — Causal Regime-Conditioned Interaction Learning for Multi-Asset Forecasting

> Tài liệu này mô tả lại hướng tiếp cận paper đang làm theo cách dễ hiểu, đủ chi tiết để domain expert review. Mục tiêu không phải viết một bản paper hoàn chỉnh ngay, mà là tạo một bản “bản đồ nghiên cứu” giúp nhóm kiểm tra: ý tưởng có hợp lý không, contribution có đủ mạnh không, thí nghiệm nên chạy thế nào, nếu kết quả đẹp thì kể câu chuyện paper ra sao.

---

## 0. Nguồn và bối cảnh đã đối chiếu

Các nguồn chính cần hiểu trước khi review tài liệu này:

1. **StockMixer gốc**  
   - Paper: *StockMixer: A Simple Yet Strong MLP-Based Architecture for Stock Price Forecasting*, AAAI 2024.  
   - Repo: `https://github.com/SJTU-DMTai/StockMixer`  
   - Ý chính: dùng kiến trúc MLP nhẹ, gồm indicator mixing, temporal mixing và stock mixing để dự báo cổ phiếu.

2. **Repo cải tiến đang có**  
   - Repo: `https://github.com/teri1712/Context-Aware-gMLP-for-Multi-Scale-Stock-Forecasting`  
   - Ý chính: đưa `market context` vào gating/gMLP để mô hình học trạng thái thị trường mà không cần attention hoặc graph operation nặng.

3. **PRICAI 2026**  
   - Website: `https://2026.pricai.org/`  
   - Call for Papers: `https://2026.pricai.org/calls/call-for-papers`  
   - PRICAI nhận các bài về AI nói chung, gồm Machine Learning, Neural Networks & Deep Learning, Data Mining & Knowledge Discovery, Explainable/Trustworthy AI và các ứng dụng AI.  
   - Long paper theo Springer LNAI: 12–16 trang, tính cả references.

---

# 1. Tóm tắt cực ngắn cho domain expert

## 1.1. Paper này muốn nói gì?

Paper không nên được kể như:

> “Chúng tôi cải tiến StockMixer bằng cách thêm gMLP.”

Cách kể đó dễ bị reviewer đánh là incremental, tức chỉ sửa nhẹ một model đã có.

Paper nên được kể như:

> “Trong bài toán dự báo nhiều tài sản tài chính, quan hệ giữa các tài sản không cố định. Khi thị trường yên bình, cổ phiếu có thể đi theo tín hiệu riêng. Khi thị trường khủng hoảng hoặc biến động mạnh, nhiều cổ phiếu trở nên đồng pha hơn. Vì vậy, mô hình không nên dùng một interaction operator cố định cho mọi ngày. Chúng tôi đề xuất một framework học tương tác giữa tài sản có điều kiện theo regime thị trường, được tính nhân quả từ dữ liệu quá khứ.”

Tên gợi ý:

**ReCIL — Regime-Conditioned Interaction Learning**

Tên đầy đủ hơn:

**Causal Regime-Conditioned Low-Rank Interaction Learning for Multi-Asset Forecasting**

## 1.2. Nói dễ hiểu bằng ví dụ

Hãy tưởng tượng dự báo cổ phiếu giống như dự báo hành vi của một lớp học.

- Ngày bình thường: mỗi học sinh hành xử khá riêng. Người chăm vẫn chăm, người lười vẫn lười.
- Ngày có kiểm tra lớn: cả lớp bị ảnh hưởng bởi một yếu tố chung. Nhiều học sinh cùng căng thẳng, cùng học, cùng thay đổi hành vi.
- Ngày có sự kiện đặc biệt: một nhóm học sinh bị ảnh hưởng mạnh hơn nhóm khác.

Nếu ta dùng một “bản đồ quan hệ” cố định giữa học sinh cho mọi ngày, nó sẽ sai trong các ngày đặc biệt. Tương tự, nếu ta dùng một ma trận quan hệ cố định giữa cổ phiếu, nó sẽ không thích nghi tốt với market regime.

ReCIL giải quyết bằng cách:

1. Nhìn lại quá khứ gần đây để xác định trạng thái thị trường.
2. Từ trạng thái đó, chọn cách trộn thông tin giữa các cổ phiếu.
3. Khi thị trường đổi trạng thái, cách trộn cũng đổi.

---

# 2. Vấn đề nghiên cứu

## 2.1. Input và output của bài toán

Mỗi ngày, ta có dữ liệu của nhiều cổ phiếu.

Ký hiệu:

```text
N = số cổ phiếu / tài sản
T = số ngày nhìn lại, ví dụ 16 ngày
F = số đặc trưng mỗi ngày, ví dụ open, high, low, close, volume
```

Input tại ngày `t`:

```text
X_t ∈ R^{N × T × F}
```

Nghĩa là:

- Có `N` cổ phiếu.
- Mỗi cổ phiếu nhìn lại `T` ngày.
- Mỗi ngày có `F` đặc trưng.

Output:

```text
ŷ_t ∈ R^N
```

Mỗi phần tử là điểm dự báo return hoặc score cho một cổ phiếu trong ngày kế tiếp.

Mục tiêu thực tế thường không chỉ là dự báo đúng giá tuyệt đối, mà là **xếp hạng cổ phiếu**: cổ phiếu nào đáng mua hơn, cổ phiếu nào nên tránh hơn.

## 2.2. Hạn chế của cách làm tĩnh

StockMixer gốc có một ý tưởng rất tốt: dùng stock mixing để học quan hệ giữa các cổ phiếu mà không cần graph ngành nghề hay graph external.

Tuy nhiên, stock mixing đó về bản chất là **static interaction**:

```text
Một bộ tham số học chung cho mọi ngày.
```

Vấn đề là thị trường tài chính phi tĩnh:

- Khi thị trường bình thường, cổ phiếu có nhiều chuyển động riêng.
- Khi thị trường biến động mạnh, tương quan giữa cổ phiếu thường tăng.
- Khi một factor chung chi phối thị trường, việc stock selection khó hơn.
- Khi dispersion cao, cơ hội chọn cổ phiếu tốt/xấu rõ hơn.

Do đó, một mixer cố định không đủ linh hoạt.

## 2.3. Hạn chế của graph/attention-heavy models

Một hướng khác là dùng GNN, Transformer hoặc attention đầy đủ để học quan hệ giữa cổ phiếu.

Nhưng các hướng này có nhược điểm:

1. Tốn tài nguyên hơn.
2. Dễ overfit vì dữ liệu tài chính theo ngày không quá nhiều.
3. GNN có thể cần graph prior như sector, industry, supply chain, ownership, news relation.
4. Quan hệ tài chính thay đổi theo thời gian, nên graph cố định cũng có thể sai.

Vì vậy, paper này chọn hướng trung gian:

```text
Không dùng graph prior.
Không dùng full attention nặng.
Dùng low-rank interaction nhẹ.
Nhưng interaction được điều kiện hóa bởi market regime.
```

---

# 3. Ý tưởng chính của paper

## 3.1. Thesis chính

Thesis của paper:

> Cross-sectional interactions among assets are regime-dependent. A model should condition its interaction operator on causal market states rather than using a single static interaction operator for all market regimes.

Dịch dễ hiểu:

> Quan hệ giữa các cổ phiếu thay đổi theo trạng thái thị trường. Vì vậy, mô hình nên đổi cách trộn thông tin giữa các cổ phiếu tùy theo trạng thái thị trường, thay vì dùng một cách trộn cố định.

## 3.2. Market context là gì?

Market context là một vector nhỏ mô tả trạng thái thị trường tại ngày dự báo, chỉ dùng dữ liệu quá khứ.

Ví dụ vector 5 chiều:

```text
c_t = [mean_return, trend, volatility, dispersion, pca_ratio]
```

Ý nghĩa:

| Thành phần | Ý nghĩa trực giác | Ví dụ diễn giải |
|---|---|---|
| `mean_return` | Thị trường chung đang tăng hay giảm | Nếu đa số cổ phiếu giảm, market return âm |
| `trend` | Xu hướng gần đây mạnh hay yếu | 16 ngày gần nhất đang dốc lên hay dốc xuống |
| `volatility` | Mức độ biến động | Thị trường đang yên bình hay hỗn loạn |
| `dispersion` | Độ phân tán giữa cổ phiếu | Cổ phiếu tốt/xấu tách biệt hay đi cùng nhau |
| `pca_ratio` | Mức chi phối của factor chung | Nếu PC1 giải thích nhiều phương sai, thị trường đang bị factor chung kéo |

Ví dụ:

```text
Ngày A:
mean_return = 0.001
trend = 0.002
volatility = 0.006
dispersion = 0.010
pca_ratio = 0.20
```

Diễn giải:

```text
Thị trường tăng nhẹ, biến động thấp, factor chung không quá mạnh.
Mô hình có thể tin hơn vào tín hiệu riêng từng cổ phiếu.
```

Ngày B:

```text
mean_return = -0.015
trend = -0.010
volatility = 0.035
dispersion = 0.006
pca_ratio = 0.65
```

Diễn giải:

```text
Thị trường giảm mạnh, biến động cao, factor chung chi phối lớn.
Mô hình nên tăng vai trò của market-level interaction, vì cổ phiếu có xu hướng đồng pha.
```

---

# 4. Kiến trúc đề xuất: ReCIL

## 4.1. Pipeline tổng quát

ReCIL gồm 5 khối:

```text
Input X_t
  ↓
Indicator/Temporal Encoder
  ↓
Multi-scale temporal representations
  ↓
Causal Market Context Encoder c_t
  ↓
Regime-Conditioned Interaction Mixer
  ↓
Prediction head
  ↓
Predicted return scores ŷ_t
```

Nói đơn giản:

1. Đầu tiên mô hình đọc lịch sử từng cổ phiếu.
2. Sau đó mô hình tóm tắt trạng thái thị trường.
3. Từ trạng thái thị trường, mô hình chọn cách trộn thông tin giữa các cổ phiếu.
4. Cuối cùng mô hình dự báo score/return cho từng cổ phiếu.

## 4.2. Phần reuse từ StockMixer/repo cải tiến

Nên tận dụng lại:

1. Data loader và dataset đã preprocess.
2. Indicator mixing / temporal mixing làm backbone nhẹ.
3. Ranking-aware loss.
4. Market context code từ repo cải tiến.
5. Train loop cơ bản.

Nhưng không nên giữ nguyên framing là “enhance StockMixer”.

Paper mới nên nói:

```text
We instantiate ReCIL using a lightweight MLP-based encoder for fair and efficient evaluation.
```

Không nên nói:

```text
We improve StockMixer by adding gMLP.
```

## 4.3. Khối 1 — Causal Market State Encoder

### Mục tiêu

Tạo vector `c_t` mô tả trạng thái thị trường trước khi dự báo.

### Điều kiện bắt buộc

Không được dùng thông tin tương lai.

Ví dụ nếu dự báo ngày `t+1`, context `c_t` chỉ được tính từ các ngày `t-T+1` đến `t`.

### Cách tính happy path

Giả sử ta có close price của `N` cổ phiếu trong 16 ngày gần nhất.

1. Tính return từng cổ phiếu.
2. Tính mean return toàn thị trường.
3. Fit một đường xu hướng đơn giản để lấy slope.
4. Tính realized volatility.
5. Tính dispersion giữa cổ phiếu.
6. Tính PCA ratio của component đầu tiên.

Pseudo-code:

```python
window_close = close[t-T+1:t+1, :]        # only historical data
returns = window_close[1:] / window_close[:-1] - 1

mean_return = returns.mean()
trend = linear_slope(returns.mean(axis=1))
volatility = returns.mean(axis=1).std()
dispersion = returns[-1].std()
pca_ratio = first_pca_explained_variance_ratio(returns)

c_t = [mean_return, trend, volatility, dispersion, pca_ratio]
```

### Sửa leakage bắt buộc

Không normalize context bằng toàn bộ dữ liệu.

Sai:

```python
scaler.fit(context_all_days)  # includes validation/test days
```

Đúng:

```python
scaler.fit(context_train_days)
context_train = scaler.transform(context_train_days)
context_val   = scaler.transform(context_val_days)
context_test  = scaler.transform(context_test_days)
```

Happy path mong muốn:

```text
Sau khi sửa, kết quả có thể thấp hơn một chút so với repo cũ, nhưng đáng tin hơn.
Reviewer sẽ khó bắt lỗi leakage.
```

## 4.4. Khối 2 — Regime-Conditioned Low-Rank Interaction Mixer

### Ý tưởng

Thay vì một stock mixer cố định, ta dùng nhiều expert nhỏ. Mỗi expert học một kiểu tương tác giữa cổ phiếu.

Ví dụ:

| Expert | Trực giác |
|---|---|
| Expert 1 | Thị trường bình thường, cổ phiếu đi theo tín hiệu riêng |
| Expert 2 | Thị trường biến động mạnh, cổ phiếu đồng pha hơn |
| Expert 3 | Thị trường phân hóa mạnh, stock selection quan trọng |
| Expert 4 | Factor chung chi phối, market-level signal quan trọng |

Market context `c_t` quyết định nên dùng expert nào nhiều hơn.

### Công thức dễ hiểu

Giả sử có 4 experts.

```text
π_t = softmax(MLP(c_t))
```

Ví dụ:

```text
π_t = [0.10, 0.70, 0.15, 0.05]
```

Diễn giải:

```text
Ngày này mô hình dùng Expert 2 nhiều nhất, có thể vì volatility cao.
```

Output:

```text
H'_t = H_t + π_1 E_1(H_t) + π_2 E_2(H_t) + π_3 E_3(H_t) + π_4 E_4(H_t)
```

### Vì sao dùng low-rank?

Nếu có `N = 1026` cổ phiếu, full interaction matrix `N × N` có hơn 1 triệu phần tử. Dễ overfit và tốn tài nguyên.

Low-rank mixer làm như sau:

```text
N stocks → m latent market states → N stocks
```

với `m << N`, ví dụ `m = 16` hoặc `m = 32`.

Nói dễ hiểu:

```text
Thay vì học trực tiếp quan hệ từng cặp cổ phiếu, mô hình học vài trạng thái thị trường ẩn, rồi dùng các trạng thái đó để truyền thông tin trở lại từng cổ phiếu.
```

### Pseudo-code

```python
class RegimeConditionedLowRankMixer(nn.Module):
    def __init__(self, num_assets, hidden_dim, num_experts=4, market_dim=16):
        self.experts_down = nn.ModuleList([
            nn.Linear(num_assets, market_dim) for _ in range(num_experts)
        ])
        self.experts_up = nn.ModuleList([
            nn.Linear(market_dim, num_assets) for _ in range(num_experts)
        ])
        self.context_router = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, H, context):
        # H: [batch, assets, dim]
        # context: [batch, 5]
        weights = torch.softmax(self.context_router(context), dim=-1)
        outputs = []
        for k in range(self.num_experts):
            z = H.transpose(1, 2)              # [batch, dim, assets]
            z = self.experts_down[k](z)        # [batch, dim, market_dim]
            z = torch.gelu(z)
            z = self.experts_up[k](z)          # [batch, dim, assets]
            z = z.transpose(1, 2)              # [batch, assets, dim]
            outputs.append(z)
        mixed = sum(weights[:, k].view(-1, 1, 1) * outputs[k]
                    for k in range(self.num_experts))
        return H + mixed
```

## 4.5. Khối 3 — Context-Gated Multi-Scale Temporal Fusion

### Ý tưởng

Không phải lúc nào scale thời gian cũng quan trọng như nhau.

Ví dụ:

- Thị trường biến động mạnh: tín hiệu ngắn hạn có thể quan trọng hơn.
- Thị trường có trend ổn định: tín hiệu dài hơn có thể hữu ích.
- Thị trường phân hóa: cần giữ tín hiệu riêng của từng cổ phiếu.

Vì vậy, market context cũng nên quyết định trọng số của các scale thời gian.

### Cách làm

Tạo 3 representation:

```text
Z_short  = representation từ window ngắn
Z_medium = representation từ window trung bình
Z_long   = representation từ window dài
```

Context sinh trọng số:

```text
ρ_t = softmax(MLP_scale(c_t))
```

Fusion:

```text
Z_t = ρ_short Z_short + ρ_medium Z_medium + ρ_long Z_long
```

Ví dụ:

```text
Ngày volatility thấp:
ρ = [0.25, 0.35, 0.40]
→ mô hình tin hơn vào medium/long trend.

Ngày volatility cao:
ρ = [0.65, 0.25, 0.10]
→ mô hình ưu tiên tín hiệu ngắn hạn.
```

### Vì sao contribution này đáng giữ?

Nó làm paper nhất quán hơn:

```text
Regime không chỉ ảnh hưởng quan hệ giữa cổ phiếu.
Regime cũng ảnh hưởng horizon thời gian nên được ưu tiên.
```

---

# 5. Contribution đề xuất

## Contribution 1 — Problem framing mới

Bài không chỉ nói “dự báo cổ phiếu”. Bài nói về:

```text
Regime-conditioned interaction learning for non-stationary multivariate time series.
```

Đóng góp này giúp paper có tính AI tổng quát hơn, phù hợp PRICAI hơn.

## Contribution 2 — Causal market context

Bài đưa market context vào mô hình nhưng kiểm soát chặt nhân quả:

```text
Context chỉ dùng dữ liệu quá khứ.
Normalization chỉ fit trên train.
Evaluation không leakage.
```

Đây là điểm rất quan trọng vì financial ML dễ bị leakage.

## Contribution 3 — Regime-conditioned low-rank interaction mixer

Đây là đóng góp kiến trúc chính.

Thay vì static stock mixer:

```text
same interaction for all days
```

ReCIL dùng:

```text
market regime → routing weights → mixture of interaction experts
```

## Contribution 4 — Context-gated multi-scale fusion

Bài cho phép market context quyết định scale thời gian nào quan trọng.

Đây là đóng góp phụ nhưng giúp paper có câu chuyện hoàn chỉnh:

```text
Regime controls both cross-sectional interaction and temporal scale.
```

## Contribution 5 — Regime-wise evidence and interpretability

Không chỉ báo bảng metric tổng.

Bài phải chứng minh:

```text
ReCIL cải thiện mạnh hơn ở high-volatility / high-PCA-ratio regimes.
Gate activation có tương quan hợp lý với market context.
```

Đây là phần làm paper đáng tin hơn, không chỉ là thêm module.

---

# 6. Vì sao có khả năng được PRICAI accept nếu làm đúng?

## 6.1. Phù hợp scope PRICAI

PRICAI nhận các bài về AI, Machine Learning, Neural Networks/Deep Learning, Data Mining/Knowledge Discovery và AI applications. Bài này nằm ở giao điểm:

```text
Machine Learning + Deep Learning + Data Mining + Financial AI Application
```

Nếu viết tốt, bài không bị giới hạn là finance engineering đơn thuần, vì method là dynamic/conditional interaction learning cho multivariate time series.

## 6.2. Không bị quá incremental nếu framing đúng

Nếu nói “thêm gMLP vào StockMixer”, bài yếu.

Nếu nói “regime-conditioned interaction learning”, bài mạnh hơn vì:

1. Nêu vấn đề tổng quát: static interaction không đủ cho non-stationary systems.
2. Đề xuất module có tính nguyên lý: causal context → dynamic low-rank operator.
3. Có phân tích regime-wise để chứng minh đúng thesis.

## 6.3. Tính thực dụng cao

PRICAI không nhất thiết đòi mô hình cực lớn. Một bài nhẹ nhưng thông minh có thể có lợi thế.

Điểm mạnh:

```text
Không cần graph prior.
Không dùng full attention nặng.
Chạy được trong vài ngày trên RTX 3090.
Dễ tái lập từ repo có sẵn.
Có interpretability thông qua context gate.
```

## 6.4. Nếu thí nghiệm đẹp, câu chuyện rất rõ

Happy path kết quả:

```text
ReCIL thắng static baseline trên RankIC/Precision.
ReCIL cải thiện mạnh trong high-volatility và high-PCA-ratio regimes.
Expert weights thay đổi hợp lý theo context.
Overhead tính toán nhỏ.
```

Nếu đạt được 4 điểm này, paper có câu chuyện thuyết phục.

## 6.5. Không thể bảo đảm strong accept

Không nên tự tin quá mức. Strong accept phụ thuộc reviewer, độ mới so với related work, chất lượng writing, reproducibility và kết quả cuối.

Nhưng với protocol sạch và story đúng, hướng này có cơ hội tốt hơn nhiều so với một bài “StockMixer + gMLP”.

---

# 7. Những lỗi phải tránh tuyệt đối

## 7.1. Tránh leakage context

Sai lầm nguy hiểm nhất:

```text
Tính min/max hoặc mean/std của context bằng toàn bộ dataset, gồm test set.
```

Reviewer tài chính hoặc time-series sẽ bắt lỗi ngay.

Bắt buộc:

```text
Fit scaler trên train only.
Apply scaler đó cho validation/test.
```

Tốt hơn nhưng có thể không kịp:

```text
Rolling normalization chỉ dùng quá khứ.
```

## 7.2. Sửa metric RIC

Trong repo cũ/cải tiến có dấu hiệu dùng:

```text
RIC = mean(IC) / std(IC)
```

Đây không phải Rank IC/Spearman IC. Đây gần với ICIR.

Phải tách rõ:

```text
IC     = average Pearson correlation
RankIC = average Spearman rank correlation
ICIR   = mean(IC_t) / std(IC_t)
```

Nếu vẫn gọi sai, reviewer có thể đánh rất nặng.

## 7.3. Không zero-fill invalid assets khi tính correlation

Sai:

```text
Invalid stocks được đưa vào correlation với giá trị 0.
```

Đúng:

```text
Mỗi ngày chỉ tính correlation trên các stocks có label hợp lệ.
```

## 7.4. Không chạy quá nhiều baseline yếu

Với thời gian 2–3 ngày, không nên cố chạy LSTM, GRU, Transformer, GNN, PatchTST, iTransformer, v.v.

Thay vào đó:

```text
Chạy ít model nhưng fair, cùng codebase, cùng seed, cùng protocol.
```

Reviewer sẽ tin ablation hơn một bảng baseline hỗn loạn.

## 7.5. Không gọi bài là StockMixer++

Không nên dùng các tên:

```text
StockMixer-gMLP
Enhanced StockMixer
StockMixer++
```

Nên dùng:

```text
ReCIL
Causal Regime-Conditioned Interaction Learning
Regime-Conditioned Low-Rank Interaction Mixer
```

---

# 8. Thiết kế thí nghiệm — Happy Path cực chi tiết

Phần này mô tả quá trình nếu mọi thứ thành công. Đây là workflow lý tưởng để chạy trong 2–3 ngày trên RTX 3090.

---

## 8.1. Mục tiêu thí nghiệm

Thí nghiệm cần trả lời 5 câu hỏi:

1. ReCIL có tốt hơn static interaction backbone không?
2. Market context có ích thật không, hay chỉ thêm noise?
3. Dynamic interaction mixer có tốt hơn context chỉ concat ở predictor không?
4. Multi-scale gate có đóng góp thêm không?
5. Lợi ích có rõ hơn trong các regime khó như high-volatility/high-PCA-ratio không?

---

## 8.2. Dataset nên dùng

### Chính thức

```text
NASDAQ
S&P500
```

Lý do:

- Đây là stock datasets phù hợp nhất với câu chuyện multi-asset forecasting.
- Có số lượng cổ phiếu đủ lớn để interaction learning có ý nghĩa.
- Có thể so sánh bối cảnh với StockMixer gốc.

### Nếu kịp

```text
Crypto
```

Lý do:

- Dùng như robustness check.
- Nhưng không nên để Crypto là điểm chính vì cơ chế thị trường khác equity.

---

## 8.3. Các model variants cần chạy

Chỉ chạy 5 variants. Không chạy quá nhiều.

| Ký hiệu | Tên | Mô tả | Mục đích |
|---|---|---|---|
| M0 | Static-LRI | Backbone với static low-rank interaction | Baseline trực tiếp |
| M1 | Context-Predictor | Thêm context vào prediction head, interaction vẫn tĩnh | Kiểm tra context đơn giản có ích không |
| M2 | Context-Gate | Dùng context để gate interaction, chưa MoE | Kiểm tra dynamic gating |
| M3 | MoE-LRI | Mixture of low-rank interaction experts | Đóng góp chính |
| M4 | ReCIL-Full | M3 + context-gated multi-scale fusion | Model đầy đủ |

### Vì sao không cần nhiều hơn?

Vì mục tiêu là chứng minh mechanism:

```text
static → context available → dynamic gate → mixture experts → multi-scale full
```

Đây là chuỗi ablation rất sạch.

---

## 8.4. Cấu hình seeds

Happy path:

```text
3 seeds × 2 datasets × 5 models = 30 runs
```

Ví dụ seeds:

```text
seed = 0, 1, 2
```

Nếu chạy nhanh:

```text
Thêm Crypto với 1–2 seeds.
```

Nếu quá chậm:

```text
NASDAQ: 3 seeds
S&P500: 2 seeds
Crypto: bỏ
```

Không nên chỉ chạy 1 seed cho toàn bộ paper nếu có thể tránh.

---

## 8.5. Metrics chính

Bảng chính nên có:

| Metric | Ý nghĩa | Vì sao cần |
|---|---|---|
| IC | Pearson correlation giữa score dự báo và return thật | Đo linear association |
| RankIC | Spearman correlation | Đo chất lượng ranking |
| ICIR | mean(IC) / std(IC) | Đo độ ổn định tín hiệu |
| Precision@10 | Top 10 dự báo có đúng là nhóm tốt không | Gần ứng dụng chọn cổ phiếu |
| Sharpe Ratio | Hiệu quả portfolio đơn giản | Có ý nghĩa tài chính |

Nếu kịp thêm:

| Metric | Ý nghĩa |
|---|---|
| Turnover | Danh mục thay đổi nhiều hay ít |
| Cost-adjusted Sharpe | Sharpe sau transaction cost |
| Max drawdown | Rủi ro giảm sâu |

---

# 9. Happy path triển khai code từng bước

## Step 0 — Tạo branch nghiên cứu

### Mục tiêu

Tách nhánh paper khỏi repo gốc để không làm hỏng code đang có.

### Lệnh minh họa

```bash
git checkout -b recil-pricai2026
```

### Kết quả mong đợi

Có một branch riêng chứa:

```text
src/recil/
configs/
scripts/
results/
figures/
```

Cấu trúc gợi ý:

```text
project/
  src/
    recil/
      model_recil.py
      context.py
      evaluator_clean.py
      train_recil.py
      regime_analysis.py
  configs/
    nasdaq_m0.yaml
    nasdaq_m1.yaml
    nasdaq_m2.yaml
    nasdaq_m3.yaml
    nasdaq_m4.yaml
    sp500_m0.yaml
    ...
  scripts/
    run_main.sh
    collect_results.py
  results/
  figures/
```

---

## Step 1 — Đóng băng baseline code trước khi sửa

### Mục tiêu

Lưu lại trạng thái code hiện tại để nếu kết quả sau khi sửa metric/leakage giảm, nhóm biết lý do.

### Việc làm

1. Chạy lại một run NASDAQ từ repo hiện tại.
2. Lưu log cũ.
3. Không dùng kết quả này làm kết quả chính nếu còn leakage/metric lỗi.

### Output

```text
results_raw_old/nasdaq_old_seed0.log
```

### Diễn giải

Kết quả cũ chỉ dùng để debug, không đưa vào paper chính.

---

## Step 2 — Sửa evaluator trước

### Mục tiêu

Metric đúng thì mọi kết luận sau mới đáng tin.

### Cần implement

```python
def pearson_ic(pred, label, mask):
    # pred, label: [num_assets]
    valid = mask.astype(bool)
    x = pred[valid]
    y = label[valid]
    if len(x) < 2:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]
```

```python
def rank_ic(pred, label, mask):
    valid = mask.astype(bool)
    x = pred[valid]
    y = label[valid]
    if len(x) < 2:
        return np.nan
    return spearmanr(x, y).correlation
```

### Output mong đợi

Một file:

```text
src/recil/evaluator_clean.py
```

có các metric:

```text
IC
RankIC
ICIR
Precision@10
Sharpe
```

### Happy path

Chạy evaluator trên dữ liệu giả:

```python
pred  = [0.4, 0.1, 0.3, 0.2]
label = [0.5, 0.0, 0.2, 0.1]
mask  = [1, 1, 1, 1]
```

Vì thứ tự gần giống nhau, `RankIC` phải dương cao.

Nếu đổi:

```python
pred  = [0.1, 0.4, 0.2, 0.3]
```

thì `RankIC` giảm hoặc âm.

---

## Step 3 — Sửa context scaler để không leakage

### Mục tiêu

Đảm bảo context ở test không bị normalize bằng thống kê của test.

### Cách làm

Tách context theo split:

```python
context_all = compute_context_from_close(eod_data)

context_train = context_all[train_indices]
context_val   = context_all[val_indices]
context_test  = context_all[test_indices]

scaler.fit(context_train)
context_train = scaler.transform(context_train)
context_val   = scaler.transform(context_val)
context_test  = scaler.transform(context_test)
```

### Output mong đợi

Một file:

```text
src/recil/context.py
```

có class:

```python
class TrainOnlyContextScaler:
    def fit(self, context_train): ...
    def transform(self, context): ...
```

### Happy path kiểm tra

In ra:

```text
Train context min/max: expected within [0,1]
Val/Test context: can be slightly below 0 or above 1
```

Nếu val/test có giá trị ngoài `[0,1]`, điều đó không sai. Nó chứng minh scaler không nhìn thấy test.

---

## Step 4 — Implement M0: Static-LRI baseline

### Mục tiêu

Tạo baseline trực tiếp và công bằng.

M0 nên gần với backbone hiện tại nhưng gọi bằng tên mới:

```text
Static Low-Rank Interaction Backbone
```

### Mô tả

M0 dùng:

```text
Temporal encoder
Static low-rank interaction mixer
Prediction head
```

Không dùng market context.

### Output mong đợi

```text
model_name = static_lri
```

Log ví dụ:

```text
Dataset: NASDAQ
Model: Static-LRI
Seed: 0
IC: 0.033
RankIC: 0.045
ICIR: 0.39
Precision@10: 0.531
Sharpe: 1.44
```

Các số trên chỉ minh họa, không phải kết quả cam kết.

---

## Step 5 — Implement M1: Context-Predictor

### Mục tiêu

Kiểm tra market context có ích không nếu chỉ đưa vào cuối.

### Cách làm

Lấy representation cuối `h_i` của từng cổ phiếu, concat với context `c_t`:

```text
score_i = Head([h_i, c_t])
```

Interaction vẫn tĩnh.

### Ý nghĩa ablation

Nếu M1 > M0:

```text
Market context có thông tin hữu ích.
```

Nếu M1 không hơn M0 nhưng M3/M4 hơn:

```text
Context không nên chỉ concat ở cuối; nó cần điều khiển interaction.
```

---

## Step 6 — Implement M2: Context-Gate

### Mục tiêu

Kiểm tra dynamic gating có ích không.

### Cách làm

Dùng context để sinh gate:

```text
g_t = sigmoid(MLP(c_t))
H'_t = H_t + g_t ⊙ E(H_t)
```

Chỉ có một expert `E`, nhưng mức độ truyền thông tin được context điều chỉnh.

### Happy path diễn giải

Nếu volatility cao:

```text
g_t tăng ở một số dimensions liên quan market-level interaction.
```

Nếu volatility thấp:

```text
g_t giảm, mô hình giữ nhiều tín hiệu riêng hơn.
```

---

## Step 7 — Implement M3: MoE-LRI

### Mục tiêu

Đây là đóng góp chính.

### Cách làm

Có `K=4` experts.

```text
π_t = softmax(MLP(c_t))
H'_t = H_t + Σ_k π_{t,k} E_k(H_t)
```

### Hyperparameter gợi ý

```text
K = 4
market_dim = 16 hoặc 32
router_hidden_dim = 32
```

### Happy path

Sau khi train, expert weights có phân bố không collapse:

```text
Expert 1 mean weight: 0.31
Expert 2 mean weight: 0.24
Expert 3 mean weight: 0.18
Expert 4 mean weight: 0.27
```

Nếu một expert luôn 0.99, cần thêm entropy regularization nhẹ hoặc giảm router capacity.

### Optional entropy regularization

```text
L_total = L_pred + λ * L_rank - η * entropy(π_t)
```

Nhưng nếu không cần, bỏ để đơn giản.

---

## Step 8 — Implement M4: ReCIL-Full

### Mục tiêu

Full model gồm:

```text
M3 + context-gated multi-scale fusion
```

### Cách làm

Có 3 scale:

```text
short, medium, long
```

Context sinh scale weights:

```text
ρ_t = softmax(MLP_scale(c_t))
Z_t = Σ_s ρ_{t,s} Z_{t,s}
```

### Happy path

Ở high-volatility days:

```text
ρ_short tăng.
```

Ở low-volatility trend days:

```text
ρ_medium hoặc ρ_long tăng.
```

Nếu mô hình full chỉ cải thiện nhẹ so với M3, vẫn ổn. M3 là contribution chính, M4 là full framework.

---

# 10. Happy path chạy thí nghiệm

## 10.1. Chạy sanity trên NASDAQ seed 0

### Lệnh minh họa

```bash
python src/recil/train_recil.py \
  --dataset NASDAQ \
  --model static_lri \
  --seed 0 \
  --epochs 5 \
  --debug
```

### Mục tiêu

Không cần kết quả tốt ngay. Chỉ cần:

```text
Loss giảm.
Không NaN.
Validation metric có giá trị hợp lệ.
GPU memory ổn.
```

### Output mong đợi

```text
Epoch 1 | train_loss 0.021 | val_IC 0.010
Epoch 2 | train_loss 0.018 | val_IC 0.017
Epoch 3 | train_loss 0.016 | val_IC 0.021
...
```

Nếu metric âm ở debug run, chưa sao. Debug chỉ kiểm tra pipeline.

---

## 10.2. Chạy đủ một model trên NASDAQ

### Lệnh minh họa

```bash
python src/recil/train_recil.py \
  --dataset NASDAQ \
  --model recil_full \
  --seed 0 \
  --epochs 100 \
  --save_dir results/nasdaq/recil_full/seed0
```

### Output cần lưu

```text
results/nasdaq/recil_full/seed0/metrics.json
results/nasdaq/recil_full/seed0/predictions.npy
results/nasdaq/recil_full/seed0/labels.npy
results/nasdaq/recil_full/seed0/masks.npy
results/nasdaq/recil_full/seed0/context.npy
results/nasdaq/recil_full/seed0/expert_weights.npy
results/nasdaq/recil_full/seed0/scale_weights.npy
results/nasdaq/recil_full/seed0/train.log
```

### Vì sao cần lưu nhiều thứ?

Vì sau đó cần làm:

```text
Regime-wise analysis
Gate interpretability
Figures
Error analysis
```

Nếu chỉ lưu metric tổng, sẽ không phân tích sâu được.

---

## 10.3. Chạy batch cho tất cả variants

### Script gợi ý

```bash
#!/bin/bash
DATASETS=(NASDAQ SP500)
MODELS=(static_lri context_predictor context_gate moe_lri recil_full)
SEEDS=(0 1 2)

for DATA in ${DATASETS[@]}; do
  for MODEL in ${MODELS[@]}; do
    for SEED in ${SEEDS[@]}; do
      python src/recil/train_recil.py \
        --dataset $DATA \
        --model $MODEL \
        --seed $SEED \
        --epochs 100 \
        --save_dir results/$DATA/$MODEL/seed$SEED
    done
  done
done
```

### Happy path thời gian

Trên RTX 3090, nếu mỗi run không quá dài, có thể xong trong 2–3 ngày.

Nếu thấy quá chậm, giảm:

```text
epochs từ 100 xuống 60–80
hoặc S&P500 chỉ 2 seeds
hoặc bỏ Crypto
```

Không nên bỏ ablation chính.

---

# 11. Happy path kết quả chính

## 11.1. Kết quả mong muốn ở bảng chính

Ví dụ layout bảng:

| Model | NASDAQ IC | NASDAQ RankIC | NASDAQ P@10 | NASDAQ SR | S&P500 IC | S&P500 RankIC | S&P500 P@10 | S&P500 SR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Static-LRI | 0.xxx | 0.xxx | 0.xxx | x.xxx | 0.xxx | 0.xxx | 0.xxx | x.xxx |
| Context-Predictor | 0.xxx | 0.xxx | 0.xxx | x.xxx | 0.xxx | 0.xxx | 0.xxx | x.xxx |
| Context-Gate | 0.xxx | 0.xxx | 0.xxx | x.xxx | 0.xxx | 0.xxx | 0.xxx | x.xxx |
| MoE-LRI | 0.xxx | 0.xxx | 0.xxx | x.xxx | 0.xxx | 0.xxx | 0.xxx | x.xxx |
| ReCIL-Full | **0.xxx** | **0.xxx** | **0.xxx** | **x.xxx** | **0.xxx** | **0.xxx** | **0.xxx** | x.xxx |

### Happy path lý tưởng

```text
ReCIL-Full hoặc MoE-LRI đứng đầu ở IC/RankIC/Precision trên cả NASDAQ và S&P500.
Sharpe có thể không đứng đầu tuyệt đối, nhưng không được quá tệ.
```

### Nếu Sharpe không tốt nhất

Không nhất thiết chết paper. Có thể giải thích:

```text
Ranking quality and portfolio Sharpe do not always move together because Sharpe depends on turnover, concentration and transaction cost.
```

Nhưng nếu Sharpe giảm mạnh, cần kiểm tra portfolio construction.

---

## 11.2. Ablation mong muốn

Bảng ablation nên cho thấy từng module có ích.

Ví dụ happy path:

| Variant | Context | Dynamic interaction | MoE | Scale gate | RankIC |
|---|---:|---:|---:|---:|---:|
| M0 | ✗ | ✗ | ✗ | ✗ | 0.040 |
| M1 | ✓ | ✗ | ✗ | ✗ | 0.043 |
| M2 | ✓ | ✓ | ✗ | ✗ | 0.047 |
| M3 | ✓ | ✓ | ✓ | ✗ | 0.052 |
| M4 | ✓ | ✓ | ✓ | ✓ | 0.055 |

Diễn giải:

```text
Context alone helps slightly.
Dynamic gating helps more.
Mixture of experts gives the largest jump.
Scale gating gives additional improvement.
```

Nếu M4 không hơn M3:

```text
Có thể chọn M3 làm main model và đưa scale gate vào appendix hoặc nói là optional extension.
```

---

# 12. Regime-wise analysis — phần làm paper nổi bật

## 12.1. Vì sao cần regime analysis?

Nếu bài chỉ có bảng tổng, reviewer có thể nói:

```text
Model phức tạp hơn nên tốt hơn một chút, không rõ vì sao.
```

Regime analysis trả lời:

```text
Model tốt hơn đúng ở nơi thesis dự đoán: những regime mà static interaction khó xử lý.
```

## 12.2. Chia regime như thế nào?

Từ context `c_t`, chia test days thành các nhóm.

Ví dụ với volatility:

```python
vol = context_test[:, volatility_index]
threshold = np.median(vol)
low_vol_days = vol <= threshold
high_vol_days = vol > threshold
```

Tương tự:

```text
low/high dispersion
low/high PCA ratio
uptrend/downtrend
```

## 12.3. Bảng happy path

| Regime | Static-LRI RankIC | ReCIL RankIC | Gain |
|---|---:|---:|---:|
| Low volatility | 0.045 | 0.048 | +0.003 |
| High volatility | 0.020 | 0.035 | +0.015 |
| Low PCA ratio | 0.047 | 0.050 | +0.003 |
| High PCA ratio | 0.018 | 0.034 | +0.016 |
| Low dispersion | 0.030 | 0.037 | +0.007 |
| High dispersion | 0.040 | 0.053 | +0.013 |

Diễn giải mong muốn:

```text
ReCIL cải thiện đặc biệt rõ trong high-volatility và high-PCA-ratio regimes.
Điều này ủng hộ giả thuyết rằng interaction giữa cổ phiếu nên được điều kiện hóa bởi market state.
```

## 12.4. Nếu kết quả không như mong muốn

Nếu ReCIL chỉ tốt ở low-volatility:

```text
Có thể model học ổn định tốt hơn nhưng chưa xử lý crisis regime.
Cần giảm capacity hoặc thêm regularization.
```

Nếu ReCIL tốt tổng thể nhưng regime gain không rõ:

```text
Paper vẫn có thể nộp, nhưng câu chuyện regime-conditioned yếu đi.
Cần dùng gate interpretability để hỗ trợ.
```

---

# 13. Gate interpretability — phần dễ tạo ấn tượng

## 13.1. Cần lưu gì?

Trong mỗi test day, lưu:

```text
context vector c_t
expert weights π_t
scale weights ρ_t
prediction scores
true returns
```

## 13.2. Phân tích expert weights

Tính correlation:

```python
corr_expert_vol = corr(expert_weight[:, k], volatility)
corr_expert_pca = corr(expert_weight[:, k], pca_ratio)
corr_scale_vol = corr(scale_weight[:, short], volatility)
```

## 13.3. Happy path kết quả

Ví dụ:

| Relation | Correlation | Diễn giải |
|---|---:|---|
| Expert 2 weight vs volatility | +0.45 | Expert 2 được dùng nhiều khi thị trường biến động |
| Expert 3 weight vs dispersion | +0.38 | Expert 3 có thể là stock-picking expert |
| Short-scale weight vs volatility | +0.52 | Biến động cao thì ưu tiên horizon ngắn |
| Long-scale weight vs trend stability | +0.41 | Trend ổn định thì ưu tiên horizon dài hơn |

## 13.4. Figure nên vẽ

### Figure 1 — Model architecture

Nội dung:

```text
Input → Temporal encoder → Context encoder → Regime-conditioned interaction mixer → Prediction
```

### Figure 2 — Expert weights over time

Trục x:

```text
Test days
```

Trục y:

```text
Volatility và expert weight
```

Thông điệp:

```text
Expert weights thay đổi theo market state, không cố định.
```

### Figure 3 — Regime-wise gain

Bar chart:

```text
Gain của ReCIL so với Static-LRI trong low/high regimes.
```

Thông điệp:

```text
Gain lớn hơn ở regime khó.
```

---

# 14. Efficiency analysis

## 14.1. Vì sao cần?

Paper muốn tránh bị so với Transformer/GNN nặng. Vì vậy cần chứng minh:

```text
Model adaptive nhưng vẫn lightweight.
```

## 14.2. Cần đo gì?

| Chỉ số | Cách đo |
|---|---|
| Params | Tổng số parameters |
| Training time/epoch | Trung bình thời gian mỗi epoch |
| GPU memory | Peak memory |
| Inference time | Thời gian dự báo test |

## 14.3. Happy path bảng

| Model | Params | Time/epoch | GPU memory | RankIC |
|---|---:|---:|---:|---:|
| Static-LRI | 1.00× | 1.00× | 1.00× | 0.xxx |
| MoE-LRI | 1.15× | 1.12× | 1.10× | 0.xxx |
| ReCIL-Full | 1.22× | 1.18× | 1.15× | 0.xxx |

Thông điệp:

```text
ReCIL cải thiện chất lượng dự báo với overhead nhỏ.
```

---

# 15. Cách viết kết quả nếu happy path thành công

## 15.1. Main result narrative

Có thể viết:

```text
Across NASDAQ and S&P500, ReCIL consistently improves ranking-oriented metrics over the static interaction backbone. The improvement is more pronounced in RankIC and Precision@10, suggesting that regime conditioning mainly benefits cross-sectional stock selection rather than merely reducing point-wise prediction error.
```

Dịch ý:

```text
ReCIL cải thiện nhất ở các metric ranking, phù hợp với mục tiêu chọn cổ phiếu.
```

## 15.2. Ablation narrative

```text
Adding market context only at the prediction head yields limited improvement, indicating that market states are not most useful as additional features. Instead, the major gain appears when context modulates the interaction operator, supporting our central claim that cross-sectional dependencies are regime-dependent.
```

Dịch ý:

```text
Context không chỉ là feature phụ. Nó có ích nhất khi dùng để điều khiển cách cổ phiếu tương tác.
```

## 15.3. Regime narrative

```text
The performance gain is largest during high-volatility and high-common-factor regimes. This confirms that static interaction operators are particularly insufficient when market-wide factors dominate individual stock movements.
```

Dịch ý:

```text
Mô hình thắng mạnh nhất lúc thị trường khó, đúng với giả thuyết ban đầu.
```

## 15.4. Interpretability narrative

```text
The learned routing weights exhibit meaningful associations with market-state variables. For example, one expert is activated more strongly under high volatility, while the short-term temporal scale receives larger weights during turbulent periods.
```

Dịch ý:

```text
Gate không phải hộp đen ngẫu nhiên. Nó phản ứng có cấu trúc với trạng thái thị trường.
```

---

# 16. Section outline cho paper PRICAI

## Title gợi ý

```text
ReCIL: Causal Regime-Conditioned Interaction Learning for Multi-Asset Forecasting
```

Hoặc:

```text
Causal Regime-Conditioned Low-Rank Interaction Learning for Multi-Asset Forecasting
```

## Abstract

Nên gồm 5 câu:

1. Multivariate financial forecasting cần học temporal patterns và cross-sectional interactions.
2. Existing static mixers giả định quan hệ tài sản ổn định, điều này sai trong thị trường phi tĩnh.
3. Đề xuất ReCIL, dùng causal market context để điều khiển low-rank interaction experts và multi-scale temporal fusion.
4. Mô hình nhẹ, không cần graph prior/full attention.
5. Experiments cho thấy cải thiện ranking metrics, đặc biệt trong high-volatility/high-common-factor regimes.

## 1. Introduction

Cấu trúc:

1. Nêu non-stationary cross-sectional dependency.
2. Phê bình static interaction và heavy graph/attention.
3. Nêu idea regime-conditioned interaction.
4. Liệt kê contributions.

## 2. Related Work

Nên chia:

```text
2.1 Stock and multi-asset forecasting
2.2 MLP-based time-series models
2.3 Dynamic/conditional computation
2.4 Regime-aware financial modeling
```

## 3. Method

```text
3.1 Problem formulation
3.2 Causal market-state construction
3.3 Regime-conditioned low-rank interaction mixer
3.4 Context-gated multi-scale fusion
3.5 Training objective
3.6 Complexity analysis
```

## 4. Experiments

```text
4.1 Datasets and protocol
4.2 Metrics
4.3 Main results
4.4 Ablation study
4.5 Regime-wise analysis
4.6 Interpretability and efficiency
```

## 5. Conclusion

Nhấn lại:

```text
Static interaction is insufficient under non-stationary regimes.
Causal regime conditioning provides a lightweight and interpretable alternative.
```

---

# 17. Checklist kết quả tối thiểu trước khi submit

## Code/evaluation checklist

- [ ] Context không leakage.
- [ ] Scaler fit train only.
- [ ] IC tính Pearson đúng.
- [ ] RankIC tính Spearman đúng.
- [ ] ICIR tách riêng, không gọi là RIC.
- [ ] Correlation tính trên valid assets, không zero-fill invalid assets.
- [ ] Có ít nhất 2 datasets chính: NASDAQ và S&P500.
- [ ] Có ít nhất 2–3 seeds cho model chính và baseline chính.
- [ ] Có main ablation M0–M4.
- [ ] Có regime-wise analysis.
- [ ] Có gate interpretability.
- [ ] Có efficiency table.

## Paper checklist

- [ ] Title không nhắc StockMixer.
- [ ] Abstract không bán như incremental improvement.
- [ ] Introduction nêu rõ dynamic interaction problem.
- [ ] Contribution rõ và không phóng đại.
- [ ] Related work có StockMixer nhưng không xem nó là trung tâm duy nhất.
- [ ] Method có causal context và no-leakage protocol.
- [ ] Experiment có fair ablation.
- [ ] Limitations trung thực.

---

# 18. Kịch bản nếu kết quả không hoàn hảo

## Trường hợp 1: M4 không hơn M3

Giải pháp:

```text
Chọn M3 làm main model.
Đưa scale gate thành optional extension hoặc appendix.
Tên model vẫn là ReCIL nhưng full version có thể là ReCIL-MoE.
```

## Trường hợp 2: Sharpe thấp hơn baseline

Giải pháp:

```text
Tập trung vào RankIC/Precision nếu mục tiêu là ranking.
Thêm turnover analysis để giải thích.
Không claim trading system hoàn chỉnh.
```

## Trường hợp 3: ReCIL chỉ thắng NASDAQ, không thắng S&P500

Giải pháp:

```text
Kiểm tra market_dim, K, seed.
Nếu vẫn vậy, viết trung thực: gain depends on cross-sectional richness.
NASDAQ có nhiều assets hơn nên interaction learning có đất diễn hơn.
```

## Trường hợp 4: Context-Predictor cũng tốt ngang MoE

Giải pháp:

```text
Phải xem lại MoE implementation.
Nếu context chỉ concat đã đủ, contribution yếu hơn.
Có thể chuyển paper thành causal market context integration, nhưng khả năng accept thấp hơn.
```

## Trường hợp 5: Gate weights không interpretable

Giải pháp:

```text
Không đưa correlation table nếu xấu.
Chỉ đưa regime-wise performance.
Hoặc thêm entropy/orthogonality regularization để experts khác nhau hơn.
```

---

# 19. Kịch bản happy path đầy đủ từ chạy đến paper

## Ngày 1

### Buổi sáng

- Tạo branch `recil-pricai2026`.
- Sửa evaluator.
- Sửa context scaler.
- Test no-leakage.

### Buổi chiều

- Implement M0, M1, M2.
- Chạy sanity NASDAQ seed 0.
- Kiểm tra log, loss, metric.

### Buổi tối

- Implement M3 và M4.
- Chạy debug 5 epochs cho tất cả variants trên NASDAQ.
- Nếu có NaN, sửa ngay.

## Ngày 2

### Buổi sáng

- Chạy NASDAQ full: M0–M4, seeds 0–2.

### Buổi chiều

- Chạy S&P500 full: M0–M4, seeds 0–2.

### Buổi tối

- Collect results.
- Tạo bảng main results tạm.
- Nếu M4 xấu, kiểm tra M3.

## Ngày 3

### Buổi sáng

- Regime-wise analysis.
- Gate interpretability.
- Efficiency table.

### Buổi chiều

- Vẽ figures.
- Viết Method và Experiments.

### Buổi tối

- Viết Introduction/Abstract.
- Kiểm tra claims có quá đà không.
- Đóng gói paper theo 12–16 trang.

---

# 20. Ví dụ minh họa toàn bộ logic bằng một ngày thị trường

Giả sử ngày `t`, ta dự báo return ngày `t+1` cho 5 cổ phiếu A, B, C, D, E.

## Input

Mô hình nhìn lại 16 ngày:

```text
A: close/open/high/low/volume trong 16 ngày
B: close/open/high/low/volume trong 16 ngày
C: close/open/high/low/volume trong 16 ngày
D: close/open/high/low/volume trong 16 ngày
E: close/open/high/low/volume trong 16 ngày
```

## Context

Từ 16 ngày này, tính được:

```text
mean_return = -0.012
trend = -0.008
volatility = 0.031
dispersion = 0.009
pca_ratio = 0.62
```

Diễn giải:

```text
Thị trường đang giảm, biến động cao, cổ phiếu bị factor chung chi phối mạnh.
```

## Router chọn expert

```text
π = [0.05, 0.72, 0.18, 0.05]
```

Diễn giải:

```text
Expert 2 được dùng chính, có thể là high-volatility/common-factor expert.
```

## Scale gate

```text
ρ_short = 0.68
ρ_medium = 0.23
ρ_long = 0.09
```

Diễn giải:

```text
Mô hình ưu tiên tín hiệu ngắn hạn vì thị trường đang hỗn loạn.
```

## Prediction

Mô hình xuất score:

```text
A: 0.012
B: -0.004
C: 0.020
D: -0.010
E: 0.006
```

Ranking:

```text
C > A > E > B > D
```

Nếu return thật ngày `t+1` cũng có C/A/E tốt hơn B/D, RankIC và Precision@K sẽ tốt.

## Ý nghĩa paper

Ở ngày thị trường thường, mô hình có thể dùng expert khác và scale khác. Do đó, ReCIL không cố áp một ma trận tương tác duy nhất cho mọi ngày.

---

# 21. Bản tóm tắt để đưa vào nhóm review

Paper nên được hiểu như sau:

> Chúng ta không làm một bản StockMixer++. Chúng ta tận dụng backbone MLP nhẹ và market-context branch đã có để xây dựng một framework mới: ReCIL. ReCIL giải quyết vấn đề static interaction trong multivariate financial forecasting bằng cách dùng market context nhân quả để điều khiển low-rank interaction experts và temporal scale fusion. Điểm mạnh của paper nằm ở framing, no-leakage protocol, ablation sạch, regime-wise analysis và gate interpretability.

Kết quả đẹp nhất cần đạt:

```text
ReCIL > Static-LRI trên RankIC/Precision.
Gain lớn hơn ở high-volatility và high-PCA-ratio regimes.
Expert/scale weights thay đổi hợp lý theo context.
Overhead nhỏ.
```

Nếu đạt được, câu chuyện paper đủ mạnh để nộp PRICAI với khả năng cạnh tranh tốt.

---

# 22. Final recommendation

Tôi khuyên chọn hướng chính thức:

```text
ReCIL: Causal Regime-Conditioned Interaction Learning for Multi-Asset Forecasting
```

Model chính:

```text
Causal market context encoder
+ mixture of low-rank interaction experts
+ context-gated multi-scale temporal fusion
+ ranking-aware objective
```

Thí nghiệm chính:

```text
NASDAQ + S&P500
5 variants M0–M4
2–3 seeds
IC, RankIC, ICIR, Precision@10, Sharpe
Regime-wise analysis
Gate interpretability
Efficiency table
```

Đây là cấu hình tối ưu giữa:

```text
độ mới
khả năng chạy trong 2–3 ngày
khả năng phòng thủ trước reviewer
khả năng viết thành một paper AI chứ không chỉ finance engineering
```
