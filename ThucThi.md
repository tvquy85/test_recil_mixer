# ThucThi.md

# Kế hoạch thực thi mô hình ReCIL-Mixer cho bài PRICAI 2026

**Tên hướng nghiên cứu đề xuất:** ReCIL-Mixer — *Regime-Conditioned Interaction Learning Mixer for Multi-Asset Forecasting*  
**Mục tiêu thực thi:** biến repo cải tiến hiện tại thành một mô hình/pipeline đủ sạch, đủ mới, đủ dễ kiểm chứng để viết bài nộp PRICAI 2026.  
**Nguyên tắc triển khai:** không chạy một task quá dài; mỗi bước phải có mục tiêu, file cần sửa/tạo, test nhỏ, tiêu chí pass/fail và expected output để Codex dễ thực thi.

---

## 0. Tóm tắt cực ngắn cho người review

Paper không nên được kể là:

> “Chúng tôi cải tiến StockMixer bằng cách thêm gMLP.”

Cách kể đó dễ bị đánh là incremental.

Paper nên được kể là:

> “Chúng tôi đề xuất một framework học tương tác giữa nhiều tài sản tài chính có điều kiện theo regime thị trường. Thay vì dùng một stock-interaction operator cố định cho mọi ngày, mô hình dùng market context nhân quả để điều khiển cách trộn thông tin giữa các tài sản và cách chọn temporal scale.”

Nói dễ hiểu:

- Ngày thị trường yên bình: cổ phiếu có tín hiệu riêng, không nên trộn quá mạnh.
- Ngày thị trường stress: nhiều cổ phiếu bị kéo bởi factor chung, interaction giữa tài sản quan trọng hơn.
- Ngày dispersion cao: cơ hội stock picking rõ hơn, model cần phân biệt tốt hơn giữa tài sản mạnh/yếu.

Do đó mô hình cần một **regime-conditioned interaction operator** thay vì static mixer.

---

## 1. Nguồn uy tín làm nền cho thiết kế

Các nguồn này dùng để định vị contribution, tránh thiết kế cảm tính.

| Nguồn | Vai trò trong thiết kế |
|---|---|
| StockMixer, AAAI 2024 — “StockMixer: A Simple Yet Strong MLP-Based Architecture for Stock Price Forecasting” | Gợi ý backbone nhẹ: indicator mixing, temporal mixing, stock mixing; nhưng hạn chế là interaction tương đối tĩnh. Link: https://ojs.aaai.org/index.php/AAAI/article/view/28681 |
| Official StockMixer repo | Dùng lại data loader, training style, loss ranking/regression nếu thuận tiện. Link: https://github.com/SJTU-DMTai/StockMixer |
| Repo cải tiến Context-Aware-gMLP | Dùng lại ý tưởng market context và gated MLP; nhưng cần sửa leakage, metric và framing. Link: https://github.com/teri1712/Context-Aware-gMLP-for-Multi-Scale-Stock-Forecasting |
| MLP-Mixer, NeurIPS 2021 | Củng cố triết lý MLP-based token/channel mixing thay vì Transformer/GNN nặng. Link: https://arxiv.org/abs/2105.01601 |
| gMLP — “Pay Attention to MLPs” | Củng cố ý tưởng gated MLP attention-free; không nên claim ReCIL là gMLP chuẩn nếu không có spatial gating đầy đủ. Link: https://arxiv.org/abs/2105.08050 |
| FiLM — Feature-wise Linear Modulation | Cơ sở cho conditioning bằng context: context sinh scale/shift để điều biến representation. Link: https://arxiv.org/abs/1709.07871 |
| Temporal Fusion Transformer | Cơ sở cho gating/selection/interpretable time-series modules; không cần dùng Transformer nặng, chỉ học nguyên tắc selection theo context. Link: https://arxiv.org/abs/1912.09363 |
| PatchTST, ICLR 2023 | Cơ sở cho patch/multi-scale temporal representation; patching giữ local semantics và giảm chi phí. Link: https://arxiv.org/abs/2211.14730 |
| RankNet / Learning to Rank | Cơ sở cho pairwise ranking loss trong bài toán xếp hạng cổ phiếu. Link: https://icml.cc/Conferences/2015/wp-content/uploads/2015/06/icml_ranking.pdf |
| PRICAI 2026 CFP | Long paper 12–16 trang, Springer LNAI; phù hợp hướng AI/ML/Data Mining/Application. Link: https://2026.pricai.org/calls/call-for-papers |

---

## 2. Kết quả cuối cùng cần đạt

Sau khi hoàn thành các bước, repo phải có:

1. **Mô hình ReCIL-Mixer hoàn chỉnh**
   - Causal Market State Encoder.
   - Indicator Encoder.
   - Multi-Scale Temporal Mixer.
   - Regime-Gated Scale Fusion.
   - Regime-Conditioned Low-Rank Interaction Experts.
   - Context-Gated Residual Head.

2. **Pipeline dữ liệu không leakage**
   - Market context chỉ tính từ quá khứ.
   - Context scaler fit trên train only.
   - Validation/test không dùng thống kê tương lai.

3. **Evaluator đúng**
   - IC = Pearson correlation theo từng ngày, trung bình qua ngày.
   - RankIC = Spearman rank correlation theo từng ngày, trung bình qua ngày.
   - ICIR = mean(IC_t) / std(IC_t).
   - Precision@K.
   - Sharpe Ratio.
   - Tất cả metric phải loại invalid assets bằng mask, không zero-fill để tính correlation.

4. **Ablation gọn nhưng đủ mạnh**
   - M0 Static-LRI.
   - M1 Context-Only.
   - M2 Context-Gated Single Expert.
   - M3 Regime-MoE Interaction.
   - M4 Full ReCIL-Mixer.

5. **Phân tích giúp paper mạnh**
   - Regime-wise performance.
   - Router/scale weight interpretability.
   - Efficiency table.
   - Mean ± std qua seed nếu đủ thời gian.

---

## 3. Kiến trúc mục tiêu

### 3.1. Input/output

Input tại ngày `t`:

```text
X_t ∈ R^{N × T × F}
```

Trong đó:

```text
N = số tài sản / cổ phiếu
T = lookback window, mặc định 16
F = số feature mỗi ngày, ví dụ open, high, low, close, volume
```

Output:

```text
ŷ_t ∈ R^N
```

Mỗi phần tử là predicted return score cho tài sản `i` tại ngày `t+1`.

### 3.2. Tensor flow

```text
X ∈ R^{B × N × T × F}

1. Indicator Encoder
   Z = MLP_ind(X)
   Z ∈ R^{B × N × T × D}

2. Causal Market State Encoder
   c = market_context(X_close, mask)
   c ∈ R^{B × C}
   e = MLP_ctx(c)
   e ∈ R^{B × D}

3. Multi-Scale Temporal Mixer
   for s in {1, 2, 4}:
       H_s = TemporalMixer_s(Patch_s(Z))
       H_s ∈ R^{B × N × D}

4. Regime-Gated Scale Fusion
   ρ = softmax(W_scale e)
   H = Σ_s ρ_s H_s
   H ∈ R^{B × N × D}

5. FiLM Modulation
   γ, β = MLP_film(e)
   H = H * (1 + γ) + β

6. Regime-Conditioned Low-Rank Interaction Experts
   π = softmax(W_router e)
   ΔH = Σ_k π_k E_k(H)
   H_inter = LayerNorm(H + ΔH)

7. Context-Gated Residual
   g = sigmoid(MLP_gate(e))
   H_out = H + g ⊙ (H_inter - H)

8. Prediction Head
   ŷ = MLP_pred(H_out)
   ŷ ∈ R^{B × N}
```

### 3.3. Default hyperparameters

```text
T = 16
D = 64
context_dim C = 7
scales = [1, 2, 4]
num_experts K = 4
market_dim m = 32
Dropout = 0.1
Optimizer = Adam hoặc AdamW
Learning rate = 1e-4 hoặc 5e-4
Ranking loss alpha = 0.1 hoặc 1.0, chọn theo validation
```

Nếu thiếu thời gian/GPU:

```text
K = 3
market_dim m = 16
Datasets chính = NASDAQ + S&P500
Seeds = 2 thay vì 3
```

---

# 4. Quy ước làm việc cho Codex

Mỗi task dưới đây nên được giao cho Codex như một đơn vị độc lập. Không yêu cầu Codex “làm tất cả”. Mỗi task phải kết thúc bằng:

```text
1. Files changed
2. Unit tests / smoke tests passed
3. Short explanation of what changed
4. Any known limitation
```

Nếu một task fail, không chuyển sang task sau. Sửa fail trước.

---

# 5. Phase A — Chuẩn hóa repo và kiểm kê hiện trạng

## Task A01 — Tạo branch và snapshot hiện trạng

**Mục tiêu:** tránh làm hỏng repo gốc; có baseline để so sánh.

**Codex prompt gợi ý:**

```text
Create a new branch named recil-mixer. Do not change model code yet. Inspect the repository structure and write a short file docs/repo_audit.md listing current data files, model files, training scripts, evaluator scripts, and any hard-coded dataset paths.
```

**Files cần tạo:**

```text
docs/repo_audit.md
```

**Test:**

```bash
git status
python -m compileall src
```

**Pass criteria:**

```text
- Repo compile được.
- docs/repo_audit.md có danh sách file chính.
- Chưa sửa logic mô hình.
```

---

## Task A02 — Tạo cấu trúc module mới không phá code cũ

**Mục tiêu:** thêm namespace riêng cho ReCIL để tránh lẫn với StockMixer/gMLP hiện tại.

**Codex prompt gợi ý:**

```text
Create a new package src/recil with __init__.py and placeholder files: context.py, metrics.py, losses.py, modules.py, model.py, data.py, train_recil.py, run_experiments.py, analysis.py. Do not change old files except imports if necessary. Add minimal docstrings.
```

**Files tạo:**

```text
src/recil/__init__.py
src/recil/context.py
src/recil/metrics.py
src/recil/losses.py
src/recil/modules.py
src/recil/model.py
src/recil/data.py
src/recil/train_recil.py
src/recil/run_experiments.py
src/recil/analysis.py
```

**Test:**

```bash
python -m compileall src/recil
```

**Pass criteria:**

```text
- Import không lỗi.
- Chưa ảnh hưởng script cũ.
```

---

# 6. Phase B — Dữ liệu, leakage và market context

## Task B01 — Implement causal market context raw features

**Mục tiêu:** tính context theo window quá khứ, không normalize trong hàm raw.

**File:**

```text
src/recil/context.py
```

**Context features:**

```text
1. market_return
2. market_trend
3. market_volatility
4. cross_sectional_dispersion
5. pca_ratio
6. market_breadth
7. downside_volatility
```

**Hàm cần có:**

```python
def compute_market_context_raw(close_window, valid_mask=None, eps=1e-8):
    """
    close_window: np.ndarray or torch.Tensor, shape [N, T]
    valid_mask: optional, shape [N, T] or [N]
    returns: context vector shape [7]
    """
```

**Quy tắc:**

- Chỉ dùng `close_window` trong quá khứ.
- Không dùng ngày target.
- Không normalize trong hàm này.
- Nếu asset invalid, loại khỏi mean/std/PCA.
- Nếu quá ít valid assets, return vector zeros hoặc safe fallback có warning.

**Ví dụ toy:**

```text
N = 3, T = 5
Asset A tăng đều
Asset B giảm đều
Asset C flat
Context phải finite, không NaN.
```

**Unit test:**

Tạo file:

```text
tests/test_context.py
```

Test cases:

```text
- output shape = (7,)
- không NaN/Inf
- volatility không âm
- breadth nằm trong [0, 1]
- pca_ratio nằm trong [0, 1]
- nếu tất cả close constant thì volatility gần 0
```

**Command:**

```bash
pytest tests/test_context.py -q
```

**Pass criteria:** tất cả tests pass.

---

## Task B02 — Implement train-only context scaler

**Mục tiêu:** sửa rủi ro leakage lớn nhất: scaler chỉ fit trên train.

**File:**

```text
src/recil/context.py
```

**Class cần có:**

```python
class TrainOnlyStandardizer:
    def fit(self, x): ...
    def transform(self, x): ...
    def fit_transform(self, x): ...
```

**Yêu cầu:**

```text
- Fit mean/std chỉ trên train contexts.
- transform val/test bằng statistics của train.
- Lưu được mean/std để reproducibility.
- Không dùng min/max toàn bộ train+val+test.
```

**Unit test:**

```text
- train mean sau transform gần 0.
- train std sau transform gần 1.
- val transform không refit.
- nếu gọi transform trước fit thì raise RuntimeError.
```

**Command:**

```bash
pytest tests/test_context.py -q
```

**Pass criteria:** pass.

---

## Task B03 — Build context cache theo split

**Mục tiêu:** tránh tính context lặp lại khi train, đồng thời đảm bảo không leakage.

**File:**

```text
src/recil/data.py
```

**Hàm cần có:**

```python
def build_context_cache(eod_data, masks, train_range, val_range, test_range, lookback):
    """
    Compute raw contexts for all usable prediction days.
    Fit scaler only on train_range.
    Return normalized context array and metadata.
    """
```

**Quy tắc:**

- Với ngày dự báo `t`, context dùng window `[t-lookback, ..., t-1]` hoặc đúng convention repo hiện tại nhưng không bao gồm future target.
- Fit scaler chỉ trên train days.
- Lưu metadata:

```text
context_mean
context_std
lookback
train_range
val_range
test_range
feature_names
```

**Smoke test:**

Tạo synthetic data:

```text
D = 40 days, N = 5 assets, lookback = 4
train = days 4..20
val = 21..30
test = 31..39
```

Check:

```text
- cache shape = [D, 7] hoặc [num_samples, 7]
- train contexts normalized mean gần 0
- val/test finite
```

**Pass criteria:** không leakage theo index.

---

## Task B04 — Integrate dataset item trả thêm context

**Mục tiêu:** mỗi sample train trả `x`, `target`, `mask`, `context`.

**File:**

```text
src/recil/data.py
```

**Dataset item:**

```python
{
    "x": Tensor[N, T, F],
    "y": Tensor[N],
    "mask": Tensor[N],
    "context": Tensor[7],
    "date_index": int
}
```

**Test:**

```text
- __getitem__(0) có đủ keys.
- x shape đúng.
- context shape = [7].
- y/mask shape = [N].
- dtype torch.float32.
```

**Command:**

```bash
pytest tests/test_data.py -q
```

---

# 7. Phase C — Metrics đúng và không bị reviewer bắt lỗi

## Task C01 — Implement Pearson IC, RankIC, ICIR với mask

**File:**

```text
src/recil/metrics.py
```

**Hàm cần có:**

```python
def pearson_corr_masked(pred, target, mask, eps=1e-8): ...
def spearman_corr_masked(pred, target, mask, eps=1e-8): ...
def compute_ic_series(preds, targets, masks): ...
def summarize_ic(ic_series, rankic_series): ...
```

**Quy tắc:**

- Với mỗi ngày, chỉ dùng assets có `mask=1`.
- Nếu số valid assets < 3, return NaN cho ngày đó và bỏ khỏi mean.
- IC là Pearson.
- RankIC là Spearman = Pearson của ranks.
- ICIR = mean(IC_t) / std(IC_t).
- Không đặt tên RIC cho ICIR.

**Toy test:**

```text
pred = [1,2,3], target=[1,2,3], mask=[1,1,1] => IC=1, RankIC=1
pred = [3,2,1], target=[1,2,3] => IC=-1, RankIC=-1
mask=[1,1,0] => tính trên 2 asset; nếu rule min_valid=3 thì NaN
```

**Command:**

```bash
pytest tests/test_metrics.py -q
```

**Pass criteria:** tất cả toy cases đúng.

---

## Task C02 — Implement Precision@K và Sharpe

**File:**

```text
src/recil/metrics.py
```

**Hàm:**

```python
def precision_at_k(pred, target, mask, k=10): ...
def long_only_daily_return(pred, target_return, mask, k=10): ...
def sharpe_ratio(daily_returns, annualization=252, eps=1e-8): ...
```

**Precision@K definition:**

- Chọn top-K predicted assets trong valid universe.
- Ground truth top-K là top-K realized return trong valid universe.
- Precision@K = overlap / K.
- Nếu valid assets < K, dùng `K_eff = valid_count`.

**Sharpe:**

```text
SR = mean(daily_returns) / std(daily_returns) * sqrt(252)
```

**Test:**

```text
- pred ranking trùng target ranking => P@K = 1.
- pred ranking ngược => P@K thấp.
- mask loại một asset thì asset đó không được chọn.
- daily_returns constant positive => std gần 0, phải xử lý eps không NaN.
```

---

## Task C03 — Replace old evaluator usage bằng evaluator mới

**Mục tiêu:** training/test script dùng evaluator sạch.

**Files:**

```text
src/recil/train_recil.py
src/recil/run_experiments.py
```

**Yêu cầu:**

- In ra đủ:

```text
IC
RankIC
ICIR
Precision@10
Sharpe
num_valid_days
```

- Không in `RIC` nếu không phải Spearman.
- Save JSON metrics:

```text
outputs/{dataset}/{model_name}/seed_{seed}/metrics.json
```

**Smoke test:**

```bash
python -m src.recil.metrics
```

Hoặc tạo toy script chạy evaluator trên synthetic predictions.

---

# 8. Phase D — Implement từng module mô hình

## Task D01 — IndicatorEncoder

**File:**

```text
src/recil/modules.py
```

**Class:**

```python
class IndicatorEncoder(nn.Module):
    def __init__(self, num_features, d_model, dropout=0.1): ...
    def forward(self, x):
        # x: [B, N, T, F]
        # return: [B, N, T, D]
```

**Architecture:**

```text
Linear(F, D)
GELU
Dropout
Linear(D, D)
LayerNorm(D)
```

**Test:**

```text
Input [2, 5, 16, 5] => output [2, 5, 16, 64]
No NaN.
Backward works.
```

**Command:**

```bash
pytest tests/test_modules.py::test_indicator_encoder -q
```

---

## Task D02 — MarketContextEncoder

**File:**

```text
src/recil/modules.py
```

**Class:**

```python
class MarketContextEncoder(nn.Module):
    def __init__(self, context_dim=7, d_model=64, dropout=0.1): ...
    def forward(self, context):
        # context: [B, 7]
        # return: [B, D]
```

**Architecture:**

```text
Linear(7, 32)
GELU
Dropout
Linear(32, D)
LayerNorm(D)
```

**Test:**

```text
Input [4, 7] => output [4, 64]
No NaN.
```

---

## Task D03 — Patchify temporal scale

**File:**

```text
src/recil/modules.py
```

**Function/class:**

```python
def patchify_time(z, scale):
    # z: [B, N, T, D]
    # scale=1 => [B,N,T,D]
    # scale=2 => [B,N,T/2,D]
    # scale=4 => [B,N,T/4,D]
```

**Implementation:**

- Dùng average pooling theo time hoặc reshape + mean.
- Nếu `T % scale != 0`, crop phần đầu hoặc raise ValueError. Khuyến nghị raise để test rõ.

**Test:**

```text
T=16, scale=1 => P=16
T=16, scale=2 => P=8
T=16, scale=4 => P=4
T=15, scale=4 => raise ValueError
```

---

## Task D04 — CausalTemporalMixer đơn giản

**Mục tiêu:** tạo temporal encoder nhẹ. Không cần implement causal triangular weight phức tạp nếu deadline gấp; có thể dùng Conv1D causal hoặc masked MLP.

**File:**

```text
src/recil/modules.py
```

**Class:**

```python
class CausalTemporalMixer(nn.Module):
    def __init__(self, d_model, patch_len, dropout=0.1): ...
    def forward(self, z):
        # z: [B, N, P, D]
        # return: [B, N, D]
```

**Khuyến nghị implementation nhanh và an toàn:**

```text
LayerNorm(D)
Depthwise/pointwise Conv1D theo time với padding causal
GELU
Dropout
Take last token
```

Nếu muốn đúng MLP hơn:

```text
TokenMixing MLP with lower-triangular mask
Channel MLP
Take last token
```

**Pass criteria quan trọng:** không được dùng future token. Với input toy chỉ thay đổi token tương lai, output tại last token không có future ngoài window. Vì mình lấy last token trong lookback, không có token tương lai so với ngày dự báo.

**Test:**

```text
Input [B=2,N=5,P=8,D=64] => output [2,5,64]
Backward works.
```

---

## Task D05 — RegimeGatedScaleFusion

**File:**

```text
src/recil/modules.py
```

**Class:**

```python
class RegimeGatedScaleFusion(nn.Module):
    def __init__(self, d_model, num_scales): ...
    def forward(self, h_scales, context_emb):
        # h_scales: list of [B,N,D]
        # context_emb: [B,D]
        # return: fused [B,N,D], scale_weights [B,S]
```

**Công thức:**

```text
rho = softmax(Linear(context_emb))
H = Σ_s rho_s * H_s
```

**Test:**

```text
- scale_weights shape [B,S]
- sum scale_weights mỗi sample = 1
- output shape [B,N,D]
- nếu h_scales giống nhau thì output bằng h_scale đó
```

---

## Task D06 — FiLMModulation

**File:**

```text
src/recil/modules.py
```

**Class:**

```python
class FiLMModulation(nn.Module):
    def __init__(self, d_model): ...
    def forward(self, h, context_emb):
        # h: [B,N,D]
        # context_emb: [B,D]
        # return: [B,N,D]
```

**Công thức:**

```text
gamma, beta = Linear(context_emb).chunk(2)
H = H * (1 + gamma) + beta
```

**Test:**

```text
- output shape đúng
- no NaN
- nếu linear weights zero init thì output gần input
```

**Implementation note:**

Khởi tạo linear cuối bằng zeros để FiLM bắt đầu gần identity. Điều này ổn định training.

---

## Task D07 — RegimeConditionedLowRankExperts

**File:**

```text
src/recil/modules.py
```

**Class:**

```python
class RegimeConditionedLowRankExperts(nn.Module):
    def __init__(self, num_assets, d_model, market_dim=32, num_experts=4, dropout=0.1): ...
    def forward(self, h, context_emb):
        # h: [B,N,D]
        # context_emb: [B,D]
        # return: h_inter [B,N,D], router_weights [B,K]
```

**Cách implement dễ đúng:**

Mỗi expert có hai linear theo asset axis:

```python
A_k = nn.Linear(num_assets, market_dim, bias=False)
B_k = nn.Linear(market_dim, num_assets, bias=False)
```

Trong forward:

```python
# h: [B,N,D]
h_t = h.transpose(1, 2)           # [B,D,N]
z = A_k(h_t)                      # [B,D,m]
z = GELU(z)
z = B_k(z)                        # [B,D,N]
delta_k = z.transpose(1, 2)       # [B,N,D]
```

Router:

```python
pi = softmax(router(context_emb), dim=-1)  # [B,K]
delta = sum_k pi[:,k,None,None] * delta_k
h_inter = LayerNorm(h + dropout(delta))
```

**Test:**

```text
Input h [2,10,64], context [2,64] => output [2,10,64], router [2,4]
Router sum = 1.
Backward works.
```

**Important:** không dùng full `N×N` interaction.

---

## Task D08 — ContextGatedResidual

**File:**

```text
src/recil/modules.py
```

**Class:**

```python
class ContextGatedResidual(nn.Module):
    def __init__(self, d_model): ...
    def forward(self, h_base, h_inter, context_emb):
        # return h_out, gate
```

**Công thức:**

```text
g = sigmoid(MLP(context_emb))  # [B,D]
h_out = h_base + g[:,None,:] * (h_inter - h_base)
```

**Test:**

```text
- gate in [0,1]
- output shape [B,N,D]
- nếu gate gần 0 => output gần h_base
- nếu gate gần 1 => output gần h_inter
```

---

## Task D09 — PredictionHead

**File:**

```text
src/recil/modules.py
```

**Class:**

```python
class PredictionHead(nn.Module):
    def __init__(self, d_model, dropout=0.1): ...
    def forward(self, h):
        # h: [B,N,D]
        # return pred: [B,N]
```

**Architecture:**

```text
LayerNorm(D)
Linear(D, D/2)
GELU
Dropout
Linear(D/2, 1)
```

**Test:**

```text
Input [2,10,64] => pred [2,10]
```

---

# 9. Phase E — Assemble full ReCIL-Mixer

## Task E01 — Implement ReCILMixer model

**File:**

```text
src/recil/model.py
```

**Class:**

```python
class ReCILMixer(nn.Module):
    def __init__(self, num_assets, num_features, d_model=64, context_dim=7, market_dim=32, num_experts=4, scales=(1,2,4), dropout=0.1): ...
    def forward(self, x, context, mask=None):
        # returns pred, aux
```

**aux cần trả:**

```python
{
    "scale_weights": scale_weights,
    "router_weights": router_weights,
    "context_gate": gate
}
```

**Forward shape test:**

```text
x: [B=2,N=20,T=16,F=5]
context: [2,7]
pred: [2,20]
scale_weights: [2,3]
router_weights: [2,4]
context_gate: [2,64]
```

**Command:**

```bash
pytest tests/test_model.py -q
```

---

## Task E02 — Implement model variants for ablation

**Mục tiêu:** không viết 5 model riêng lộn xộn; dùng config flags.

**File:**

```text
src/recil/model.py
```

**Variants:**

```text
M0_STATIC_LRI:
- no context scale fusion
- no FiLM
- static single low-rank interaction

M1_CONTEXT_ONLY:
- context embedding concat/add vào prediction head
- no dynamic interaction

M2_CONTEXT_GATED_SINGLE:
- single low-rank expert
- context gate residual

M3_REGIME_MOE:
- multiple low-rank experts with context router
- no scale gate

M4_FULL_RECIL:
- scale gate + FiLM + MoE interaction + context gate
```

**Implementation options:**

```python
variant="full" | "static" | "context_only" | "single_gate" | "moe"
```

**Test:**

```text
Mỗi variant forward pass được trên toy input.
Output shape giống nhau.
Aux có keys phù hợp.
```

---

# 10. Phase F — Loss function

## Task F01 — Masked MSE loss

**File:**

```text
src/recil/losses.py
```

**Function:**

```python
def masked_mse_loss(pred, target, mask, eps=1e-8): ...
```

**Test:**

```text
pred=[1,2,100], target=[1,3,0], mask=[1,1,0]
loss = ((1-1)^2 + (2-3)^2)/2 = 0.5
```

---

## Task F02 — Pairwise ranking loss

**File:**

```text
src/recil/losses.py
```

**Function:**

```python
def pairwise_rank_loss(pred, target, mask, margin_type="logistic"):
    ...
```

**Recommended formula:**

```text
L_rank = mean log(1 + exp(-sign(y_i - y_j) * (pred_i - pred_j)))
```

**Constraints:**

- Chỉ dùng valid assets.
- Nếu valid_count < 2, return zero loss.
- Có thể sample pairs nếu N lớn để tiết kiệm GPU.

**Pair sampling option:**

```text
max_pairs_per_day = 4096
```

**Test:**

```text
- Nếu pred ranking đúng target, loss thấp hơn pred ranking ngược.
- Không NaN nếu chỉ có 1 valid asset.
```

---

## Task F03 — Total loss wrapper

**File:**

```text
src/recil/losses.py
```

**Function:**

```python
def recil_loss(pred, target, mask, aux=None, alpha_rank=0.1, lambda_entropy=0.0): ...
```

**Optional router entropy:**

Nếu `lambda_entropy > 0` và `aux["router_weights"]` tồn tại:

```text
L = MSE + alpha_rank * RankLoss - lambda_entropy * Entropy(router)
```

Mặc định:

```text
lambda_entropy = 0
```

Tránh thêm regularizer nếu không cần.

---

# 11. Phase G — Training pipeline gọn, có checkpoint và logging

## Task G01 — Config system tối giản

**File:**

```text
src/recil/train_recil.py
```

**Không cần YAML phức tạp nếu gấp. Dùng argparse.**

Arguments:

```text
--dataset nasdaq|sp500|crypto
--data-root dataset
--variant full|static|context_only|single_gate|moe
--seed 0
--epochs 50
--batch-size 1 hoặc theo repo hiện tại
--lr 1e-4
--d-model 64
--market-dim 32
--num-experts 4
--lookback 16
--alpha-rank 0.1
--device cuda
--output-dir outputs
--quick-test flag
```

**Test:**

```bash
python -m src.recil.train_recil --help
```

Pass nếu help hiển thị đủ arguments.

---

## Task G02 — Training loop 1 epoch trên synthetic data

**Mục tiêu:** kiểm tra model/loss/optimizer hoạt động trước khi chạm dataset thật.

**File:**

```text
src/recil/train_recil.py
```

**Flag:**

```text
--synthetic
```

**Synthetic data:**

```text
B samples = 20
N = 20
T = 16
F = 5
context_dim = 7
target = noisy linear function of last-day close + market context
```

**Expected:**

```text
- Loss finite.
- Loss giảm nhẹ sau vài epoch.
- Metrics finite.
```

**Command:**

```bash
python -m src.recil.train_recil --synthetic --variant full --epochs 3 --quick-test
```

**Pass criteria:** không NaN, output metrics JSON tạo được.

---

## Task G03 — Integrate real dataset loader

**Mục tiêu:** dùng dữ liệu repo hiện tại.

**File:**

```text
src/recil/data.py
src/recil/train_recil.py
```

**Yêu cầu:**

- Load NASDAQ trước.
- Tôn trọng split train/val/test hiện tại trong repo nếu có.
- Build context cache theo Task B03.
- DataLoader trả batch sample.

**Smoke command:**

```bash
python -m src.recil.train_recil --dataset nasdaq --variant static --epochs 1 --quick-test
```

**Pass criteria:**

```text
- Train 1 epoch chạy được.
- Val metrics in ra finite.
- outputs/.../metrics.json có file.
```

---

## Task G04 — Checkpoint best validation

**File:**

```text
src/recil/train_recil.py
```

**Logic:**

- Evaluate validation mỗi epoch.
- Chọn best theo `RankIC` trước, nếu NaN thì theo `IC`.
- Save:

```text
best_model.pt
last_model.pt
metrics.json
train_log.csv
config.json
```

**Pass criteria:**

```text
- Sau run 2 epochs có đủ files.
- Có best_epoch trong metrics.json.
```

---

# 12. Phase H — Experiment runner không quá dài

## Task H01 — Create small experiment runner

**File:**

```text
src/recil/run_experiments.py
```

**Mục tiêu:** chạy từng model/dataset/seed có kiểm soát. Không launch hàng loạt quá lớn.

**Arguments:**

```text
--datasets nasdaq sp500
--variants static context_only single_gate moe full
--seeds 0 1 2
--max-parallel 1
--epochs 50
--dry-run
```

**Dry run output:**

In danh sách command sẽ chạy, không chạy thật.

**Command:**

```bash
python -m src.recil.run_experiments --datasets nasdaq --variants static full --seeds 0 --dry-run
```

**Pass criteria:** dry-run commands đúng.

---

## Task H02 — Happy path experiment schedule

**Không nhất thiết code, nhưng cần ghi trong docs.**

**File:**

```text
docs/experiment_plan.md
```

**Kịch bản nếu mọi thứ chạy tốt:**

### Stage 1 — Sanity, 1–2 giờ

```bash
python -m src.recil.train_recil --synthetic --variant full --epochs 3 --quick-test
python -m src.recil.train_recil --dataset nasdaq --variant static --epochs 1 --quick-test
python -m src.recil.train_recil --dataset nasdaq --variant full --epochs 1 --quick-test
```

Expected:

```text
- Không NaN.
- Metrics finite.
- Output files đủ.
```

### Stage 2 — NASDAQ ablation, nửa ngày

```bash
python -m src.recil.run_experiments \
  --datasets nasdaq \
  --variants static context_only single_gate moe full \
  --seeds 0 1 2 \
  --epochs 50
```

Expected:

```text
Full hoặc MoE thắng static về RankIC/ICIR/P@10.
```

### Stage 3 — S&P500 confirmation, nửa ngày đến một ngày

```bash
python -m src.recil.run_experiments \
  --datasets sp500 \
  --variants static context_only single_gate moe full \
  --seeds 0 1 2 \
  --epochs 50
```

Expected:

```text
Trend giống NASDAQ, không nhất thiết thắng mọi metric.
```

### Stage 4 — Crypto optional

```bash
python -m src.recil.run_experiments \
  --datasets crypto \
  --variants static moe full \
  --seeds 0 1 \
  --epochs 50
```

Nếu thiếu thời gian, bỏ Crypto. Regime analysis quan trọng hơn.

---

# 13. Phase I — Analysis cho paper

## Task I01 — Aggregate metrics

**File:**

```text
src/recil/analysis.py
```

**Function:**

```python
def aggregate_results(output_dir): ...
```

**Output:**

```text
results_summary.csv
```

Columns:

```text
dataset, variant, seed, IC, RankIC, ICIR, Precision@10, Sharpe, best_epoch
```

Aggregate table:

```text
mean ± std by dataset × variant
```

**Command:**

```bash
python -m src.recil.analysis --output-dir outputs --aggregate
```

---

## Task I02 — Regime-wise analysis

**Mục tiêu:** chứng minh contribution chính: model cải thiện khi regime khó.

**File:**

```text
src/recil/analysis.py
```

**Inputs:**

```text
- test predictions saved per day
- realized returns
- masks
- raw/normalized context per day
```

**Regime splits:**

```text
high_vol vs low_vol: split theo median market_volatility
high_pca vs low_pca: split theo median pca_ratio
high_dispersion vs low_dispersion: split theo median dispersion
uptrend vs downtrend: market_trend >= 0 hoặc < 0
```

**Output:**

```text
regime_results.csv
```

Columns:

```text
dataset, variant, regime_name, regime_side, IC, RankIC, Precision@10, Sharpe, num_days
```

**Expected paper claim nếu thành công:**

```text
ReCIL-Mixer improves most over static interaction in high-volatility and high-common-factor regimes, supporting the hypothesis that cross-asset interactions should be regime-conditioned.
```

---

## Task I03 — Router and scale interpretability

**Mục tiêu:** tạo figure/bảng reviewer thích.

**File:**

```text
src/recil/analysis.py
```

**Need saved aux:**

```text
scale_weights per test day: [num_days, S]
router_weights per test day: [num_days, K]
context_gate mean per test day
context raw values per test day
```

**Analysis:**

```text
corr(scale_short_weight, market_volatility)
corr(scale_long_weight, trend_abs hoặc low_vol indicator)
corr(router_expert_k, pca_ratio)
corr(router_expert_k, dispersion)
```

**Output:**

```text
interpretability_correlations.csv
scale_weights_timeseries.png
router_weights_timeseries.png
```

**Happy path interpretation:**

```text
- Short scale weight tăng khi volatility cao.
- Một expert có weight cao khi pca_ratio cao, đại diện common-factor regime.
- Một expert khác tăng khi dispersion cao, đại diện stock-picking regime.
```

Nếu correlation không đẹp, không claim quá mạnh; chỉ báo qualitative figure.

---

## Task I04 — Efficiency table

**Mục tiêu:** chứng minh mô hình nhẹ, phù hợp thesis tránh attention/graph nặng.

**File:**

```text
src/recil/analysis.py
```

**Measure:**

```text
num_params
training_time_per_epoch
gpu_memory_peak nếu có torch.cuda.max_memory_allocated
inference_time_per_test_epoch
```

**Output:**

```text
efficiency_table.csv
```

**Expected claim:**

```text
ReCIL-Mixer adds adaptive regime conditioning with modest parameter/time overhead compared with static low-rank interaction.
```

---

# 14. Phase J — Paper artifacts

## Task J01 — Generate paper tables

**File:**

```text
src/recil/analysis.py
```

**Output:**

```text
paper_tables/main_results_latex.tex
paper_tables/ablation_latex.tex
paper_tables/regime_latex.tex
paper_tables/efficiency_latex.tex
```

**Format:**

- Bold best.
- Underline second-best nếu có.
- Dùng mean ± std nếu có seeds.

**Note:**

Không cần auto-perfect LaTeX. Chỉ cần table dễ copy vào paper.

---

## Task J02 — Generate method figure spec

**Không cần vẽ đẹp ngay.**

**File:**

```text
docs/figure_spec.md
```

**Nội dung:**

```text
Figure 1: Overall architecture
- Input X: N assets × T lookback × F indicators
- Causal market context encoder
- Multi-scale temporal mixers
- Regime-gated scale fusion
- Regime-conditioned low-rank interaction experts
- Prediction head

Figure 2: Regime-conditioned interaction
- Context c_t routes among K low-rank experts
- Each expert maps assets -> latent market factors -> assets

Figure 3: Interpretability
- Market volatility vs short-scale weight
- PCA ratio vs expert weight
```

---

# 15. Full happy path nếu mọi thứ thành công

## Day 0 / trước khi chạy chính

Hoàn thành:

```text
A01–A02
B01–B04
C01–C03
D01–D09
E01–E02
F01–F03
G01–G04
```

Commands:

```bash
python -m compileall src/recil
pytest tests -q
python -m src.recil.train_recil --synthetic --variant full --epochs 3 --quick-test
python -m src.recil.train_recil --dataset nasdaq --variant full --epochs 1 --quick-test
```

Expected:

```text
- All unit tests pass.
- Synthetic training finite.
- Real data one-epoch smoke pass.
```

## Day 1 — NASDAQ chính

Run:

```bash
python -m src.recil.run_experiments \
  --datasets nasdaq \
  --variants static context_only single_gate moe full \
  --seeds 0 1 2 \
  --epochs 50
```

Expected happy path:

```text
M0 Static-LRI: baseline ổn.
M1 Context-Only: tăng nhẹ hoặc không tăng.
M2 Single-Gate: tăng rõ hơn M1.
M3 MoE: tăng RankIC/ICIR.
M4 Full: best hoặc near-best, nhất là RankIC/P@10.
```

Nếu M4 không best nhưng M3 best:

```text
Paper có thể dùng M3 làm main model, scale gate là optional/ablation.
```

## Day 2 — S&P500 confirm

Run:

```bash
python -m src.recil.run_experiments \
  --datasets sp500 \
  --variants static context_only single_gate moe full \
  --seeds 0 1 2 \
  --epochs 50
```

Expected happy path:

```text
Có cùng xu hướng với NASDAQ trên IC/RankIC.
Sharpe có thể không thắng mọi lúc; không sao nếu ranking metrics tốt.
```

## Day 3 — Analysis + optional Crypto

First run analysis:

```bash
python -m src.recil.analysis --output-dir outputs --aggregate
python -m src.recil.analysis --output-dir outputs --regime
python -m src.recil.analysis --output-dir outputs --interpretability
python -m src.recil.analysis --output-dir outputs --efficiency
```

Nếu còn GPU/time:

```bash
python -m src.recil.run_experiments \
  --datasets crypto \
  --variants static moe full \
  --seeds 0 1 \
  --epochs 50
```

Expected paper story:

```text
1. Full/MoE ReCIL improves ranking quality over static low-rank interaction.
2. Gains are larger under high-volatility and high-PCA-ratio regimes.
3. Router/scale weights correlate with market context, giving interpretability.
4. Computational overhead is moderate.
```

---

# 16. Nếu kết quả không như mong muốn

## Case 1 — Full ReCIL không thắng M3 MoE

Action:

```text
- Dùng M3 Regime-MoE Interaction làm main model.
- Đưa scale gate vào ablation như optional module.
- Claim chính chuyển thành regime-conditioned interaction, không claim multi-scale gating là core.
```

## Case 2 — Context-only thắng gần bằng MoE

Action:

```text
- Kiểm tra leakage lần nữa.
- Kiểm tra context có bị target leak không.
- Nếu sạch, viết claim rằng market context là strong causal signal; dynamic interaction vẫn giúp trong regime-wise analysis.
```

## Case 3 — Metric đẹp nhưng Sharpe xấu

Action:

```text
- Nhấn paper là forecasting/ranking model, không phải complete trading system.
- Thêm Precision@K và RankIC làm main metrics.
- Báo Sharpe phụ, tránh overclaim.
```

## Case 4 — RankIC không tăng nhưng IC tăng

Action:

```text
- Kiểm tra pairwise rank loss alpha.
- Thử alpha_rank = 1.0 thay vì 0.1.
- Chạy lại only NASDAQ seed 0 để sanity.
```

## Case 5 — Overfit nhanh

Action:

```text
- Tăng dropout 0.1 -> 0.2.
- Giảm D 64 -> 32.
- Giảm K 4 -> 3.
- Early stopping theo validation RankIC.
```

---

# 17. Checklist trước khi viết paper

## Method checklist

```text
[ ] Có mô tả causal market context.
[ ] Có nói rõ scaler fit train-only.
[ ] Có công thức Regime-Conditioned Low-Rank Interaction Experts.
[ ] Có công thức Regime-Gated Scale Fusion nếu dùng full model.
[ ] Có complexity O(K * B * D * N * m), so với full O(B * D * N^2).
[ ] Có giải thích không dùng graph prior/attention nặng.
```

## Experiment checklist

```text
[ ] NASDAQ và S&P500 có kết quả.
[ ] Static-LRI baseline cùng protocol.
[ ] Ít nhất 3–5 variants ablation.
[ ] Metrics đúng: IC, RankIC, ICIR, P@10, Sharpe.
[ ] Không còn RIC sai nghĩa.
[ ] Mask invalid assets đúng.
[ ] Có mean ± std nếu chạy multi-seed.
[ ] Có regime-wise table.
[ ] Có interpretability table/figure.
[ ] Có efficiency table.
```

## Reproducibility checklist

```text
[ ] config.json lưu đủ hyperparameters.
[ ] seed set cho random/numpy/torch.
[ ] best_model.pt lưu được.
[ ] metrics.json lưu đầy đủ.
[ ] train_log.csv lưu từng epoch.
[ ] output path có dataset/variant/seed.
```

---

# 18. Định dạng câu chuyện paper sau khi có kết quả

## Abstract skeleton

```text
Multivariate financial forecasting requires learning both temporal patterns and cross-sectional asset interactions. Existing lightweight interaction models typically use static mixing operators, implicitly assuming that asset dependencies remain stable across market regimes. This assumption is restrictive because correlations, volatility, dispersion, and common-factor dominance vary substantially over time. We propose ReCIL-Mixer, a causal regime-conditioned interaction learning framework for multi-asset forecasting. ReCIL-Mixer constructs market-state context from historical observations only, uses it to route a mixture of low-rank interaction experts, and optionally selects temporal scales through regime-gated fusion. The resulting model adapts cross-sectional information exchange without graph priors or expensive attention. Experiments on equity benchmarks show improved ranking-oriented forecasting metrics over static interaction backbones, with larger gains under high-volatility and high-common-factor regimes, while maintaining lightweight computational cost.
```

## Contribution skeleton

```text
1. We formulate multi-asset forecasting as causal regime-conditioned interaction learning, addressing the limitation of static cross-sectional mixing under non-stationary market regimes.

2. We propose ReCIL-Mixer, a lightweight architecture combining causal market-state encoding, regime-gated temporal scale fusion, and context-conditioned low-rank interaction experts without graph priors or expensive attention.

3. We provide regime-wise and interpretability analyses showing that the model adaptively changes interaction experts and temporal scale preferences under different market conditions.
```

---

# 19. Điều cần tránh tuyệt đối

```text
[ ] Không gọi mô hình là StockMixer++.
[ ] Không claim “gMLP” là contribution chính nếu implementation không đúng gMLP chuẩn.
[ ] Không normalize context bằng toàn bộ dataset.
[ ] Không gọi ICIR là RIC.
[ ] Không tính correlation trên zero-filled invalid assets.
[ ] Không chạy quá nhiều baseline mà thiếu ablation sạch.
[ ] Không claim trading profitability nếu chưa có transaction cost/turnover đầy đủ.
[ ] Không overclaim strong accept; chỉ nói thiết kế tối ưu xác suất accept.
```

---

# 20. Lệnh kiểm tra tổng hợp cuối cùng

Sau khi hoàn thành implementation:

```bash
python -m compileall src/recil
pytest tests -q
python -m src.recil.train_recil --synthetic --variant full --epochs 3 --quick-test
python -m src.recil.train_recil --dataset nasdaq --variant static --epochs 1 --quick-test
python -m src.recil.train_recil --dataset nasdaq --variant full --epochs 1 --quick-test
python -m src.recil.run_experiments --datasets nasdaq --variants static full --seeds 0 --dry-run
```

Nếu tất cả pass, mới chạy experiments dài.

---

# 21. Minimal deliverables cho Codex sau mỗi milestone

## Sau Phase B

```text
- context.py hoàn chỉnh.
- data.py build được context cache.
- tests/test_context.py pass.
```

## Sau Phase C

```text
- metrics.py hoàn chỉnh.
- tests/test_metrics.py pass.
- evaluator mới không dùng RIC sai.
```

## Sau Phase D/E

```text
- modules.py và model.py hoàn chỉnh.
- tests/test_modules.py pass.
- tests/test_model.py pass.
```

## Sau Phase F/G

```text
- train_recil.py chạy synthetic.
- train_recil.py chạy real dataset 1 epoch.
- metrics/log/checkpoint tạo đúng.
```

## Sau Phase H/I

```text
- run_experiments.py dry-run đúng.
- analysis.py aggregate/regime/interpretability/efficiency chạy được.
- paper tables tạo được.
```

---

# 22. Kết luận thực thi

Đường đi có xác suất cao nhất không phải là làm mô hình cực phức tạp. Đường đi tốt nhất là:

```text
1. Sửa sạch leakage và metric.
2. Implement dynamic regime-conditioned interaction nhẹ.
3. Chạy ablation gọn, cùng protocol.
4. Chứng minh bằng regime-wise analysis và interpretability.
5. Viết paper xoay quanh causal regime-conditioned interaction learning, không xoay quanh StockMixer.
```

Nếu làm đúng, contribution sẽ đủ rõ:

```text
static stock mixing
→ causal regime-conditioned low-rank interaction learning
```

và đủ thực dụng:

```text
no graph prior
no heavy attention
low-rank computation
interpretable routing
```

Đây là hướng thực thi hợp lý nhất để tạo một bài nộp PRICAI 2026 có khả năng được đánh giá mạnh trong điều kiện tài nguyên 2–3 ngày GPU RTX 3090.
