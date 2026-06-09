# patch_code.md — Hướng dẫn bước tiếp theo sau khi áp dụng patch ReCIL

## 0. Mục tiêu của giai đoạn tiếp theo

Sau khi đã có bản patch tối ưu code, mục tiêu tiếp theo không phải là thêm nhiều module mới ngay lập tức. Mục tiêu đúng là kiểm chứng xem patch có làm pipeline sạch hơn, ổn định hơn và đủ mạnh để tạo bằng chứng cho paper PRICAI 2026 hay không.

Giai đoạn này cần trả lời 5 câu hỏi:

1. Patch có chạy ổn trên repo gốc không?
2. Patch có giữ được signal tốt của Stage 2 cũ không?
3. Model nào nên là main model cho paper: `single_gate`, `full`, `moe`, hay `static`?
4. Kết quả có đủ mạnh trên cả NASDAQ và SP500 không?
5. Bảng kết quả, ablation, regime analysis và efficiency đã đủ paper-final chưa?

Nguyên tắc quan trọng:

- Không claim quá kết quả thật.
- Không ép `full` hoặc `moe` làm main model nếu RankIC không tốt.
- Ưu tiên `single_gate` nếu nó tiếp tục thắng hoặc ổn định hơn `static`.
- Phải có evidence trên ít nhất NASDAQ + SP500 trước khi viết paper nghiêm túc.
- Sau khi cải tiến code, việc quan trọng nhất là validation, không phải tiếp tục thêm kiến trúc mới.

---

## 1. Chuẩn bị môi trường

### 1.1. Vào đúng thư mục repo

```bash
cd /path/to/test_recil_mixer
```

Nếu đang có bản code cũ, nên tạo branch riêng để tránh làm hỏng main branch:

```bash
git checkout -b recil-patch-validation
```

### 1.2. Giải nén patch

Đặt file `recil_optimization_patch.zip` ở root repo, sau đó chạy:

```bash
unzip recil_optimization_patch.zip -d .
```

Sau khi giải nén, cấu trúc quan trọng cần có:

```text
StockMixer/src/recil/modules.py
StockMixer/src/recil/model.py
StockMixer/src/recil/metrics.py
StockMixer/src/recil/losses.py
StockMixer/src/recil/train_recil.py
StockMixer/src/recil/run_experiments.py
StockMixer/tests/recil/test_model_pipeline_optimizations.py
```

### 1.3. Vào thư mục StockMixer

```bash
cd StockMixer
```

Thiết lập `PYTHONPATH` khi chạy lệnh:

```bash
export PYTHONPATH=.
```

Nếu dùng Windows PowerShell:

```powershell
$env:PYTHONPATH="."
```

---

## 2. Kiểm tra patch có apply đúng không

### 2.1. Kiểm tra các file đã được thay thế

```bash
git status
```

Cần thấy các file sau đã thay đổi hoặc được thêm mới:

```text
modified: src/recil/model.py
modified: src/recil/metrics.py
modified: src/recil/losses.py
modified: src/recil/train_recil.py
modified: src/recil/run_experiments.py
modified/new: src/recil/modules.py
new: tests/recil/test_model_pipeline_optimizations.py
```

Nếu không thấy thay đổi, khả năng cao bạn đã giải nén sai thư mục.

### 2.2. Kiểm tra cú pháp Python

```bash
PYTHONPATH=. python -m py_compile \
  src/recil/modules.py \
  src/recil/model.py \
  src/recil/metrics.py \
  src/recil/losses.py \
  src/recil/train_recil.py \
  src/recil/run_experiments.py \
  tests/recil/test_model_pipeline_optimizations.py
```

Tiêu chí pass:

```text
Không có SyntaxError
Không có ImportError
Không có traceback
```

Nếu fail ở bước này, chưa được chạy experiment lớn.

---

## 3. Chạy unit test mới

```bash
PYTHONPATH=. python -m pytest -q tests/recil/test_model_pipeline_optimizations.py
```

Test này kiểm tra các điểm quan trọng:

1. Các variant có parameter count khác nhau hợp lý.
2. Invalid assets không làm đổi prediction của valid assets.
3. Metrics mới có các chỉ số cần thiết như RankICIR, Turnover, CostSharpe.
4. Forward pass chạy được với các variant chính.

Tiêu chí pass:

```text
All tests passed
```

Nếu fail, cần xử lý theo nhóm lỗi:

| Lỗi | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| ImportError | Sai `PYTHONPATH` hoặc thiếu file | Chạy lại từ thư mục `StockMixer`, dùng `PYTHONPATH=.` |
| Shape mismatch | Dataset/model config khác giả định | In shape của `x`, `y`, `mask`, `context` để sửa forward |
| CUDA error | GPU/driver/environment | Chạy CPU trước để isolate lỗi |
| Test mask fail | Mask chưa được truyền xuyên suốt | Kiểm tra `apply_asset_mask()` trong `model.py` và `modules.py` |

---

## 4. Chạy smoke test synthetic

Mục tiêu là kiểm tra training loop mới có hoạt động không trước khi dùng dữ liệu thật.

```bash
PYTHONPATH=. python -m src.recil.train_recil \
  --synthetic \
  --quick-test \
  --variant single_gate \
  --epochs 2 \
  --output-dir outputs_patch_smoke \
  --alpha-rank 0.3 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0
```

Sau khi chạy, kiểm tra output:

```bash
ls outputs_patch_smoke
```

Cần có các file log hoặc metrics tùy pipeline hiện tại.

Tiêu chí pass:

```text
Training chạy hết 2 epochs
Loss finite
Không có NaN/Inf
Có output metrics
Có parameter_report hoặc active_params log
```

Nếu smoke test fail, không chạy Stage 2/Stage 3.

---

## 5. Reproduce Stage 2 cũ bằng code mới

Đây là bước rất quan trọng. Patch đã sửa mask-safe interaction và variant-specific modules nên kết quả có thể thay đổi. Ta cần biết thay đổi đó là tốt, xấu hay chỉ khác nhẹ.

### 5.1. Chạy NASDAQ với setup gần Stage 2 cũ

```bash
PYTHONPATH=. python -m src.recil.run_experiments \
  --datasets nasdaq \
  --variants static context_only single_gate moe full \
  --seeds 0 1 \
  --epochs 30 \
  --alpha-rank 0.3 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0 \
  --transaction-cost-bps 10 \
  --output-dir outputs_stage2_patch_reproduce
```

### 5.2. So sánh với Stage 2 cũ

Kết quả cũ cần nhớ:

```text
single_gate RankIC ≈ 0.032052
static RankIC ≈ 0.031013
context_only RankIC ≈ 0.025968
moe RankIC ≈ 0.019773
full RankIC ≈ 0.013265
```

Không bắt buộc kết quả mới phải y hệt. Vì patch đã sửa logic mask và module count nên kết quả có thể khác. Điều cần kiểm tra là:

```text
[ ] Metrics không collapse toàn bộ.
[ ] single_gate vẫn cạnh tranh mạnh với static.
[ ] static không bị thay đổi bất thường quá lớn.
[ ] full/moe không tạo NaN hoặc unstable output.
[ ] active_params khác nhau giữa variants.
[ ] runtime/epoch_sec được log.
[ ] transaction-cost metrics được log.
```

### 5.3. Cách đánh giá reproduce

| Tình huống | Kết luận |
|---|---|
| `single_gate` vẫn thắng hoặc ngang `static` | Có thể chuyển sang Stage 3 core |
| `static` thắng nhẹ nhưng `single_gate` có regime/cost lợi hơn | Vẫn có thể chuyển Stage 3 |
| Tất cả variant RankIC gần 0 hoặc âm | Có khả năng patch làm hỏng training/data flow |
| `full/moe` NaN | Cần giảm LR, tăng clipping, kiểm tra entropy/router |
| Kết quả biến động rất mạnh giữa seed 0 và 1 | Cần tăng seeds hoặc thêm regularization |

---

## 6. Stage 3 core — bước quyết định main model

Sau khi reproduce ổn, chạy Stage 3 core trên NASDAQ và SP500.

### 6.1. Lệnh chạy đề xuất

```bash
PYTHONPATH=. python -m src.recil.run_experiments \
  --datasets nasdaq sp500 \
  --variants static context_only single_gate \
  --seeds 0 1 2 \
  --epochs 50 \
  --alpha-rank 0.3 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0 \
  --transaction-cost-bps 10 \
  --output-dir outputs_stage3_patch_core
```

### 6.2. Vì sao chỉ chạy 3 variant trước?

Ba variant này đủ để trả lời câu hỏi chính của paper:

```text
static: interaction không có context
context_only: context chỉ thêm như feature
single_gate: context điều khiển interaction residual
```

Nếu `single_gate` thắng `context_only`, ta có bằng chứng rằng context không chỉ là extra feature, mà hữu ích khi dùng để gate interaction.

Nếu `single_gate` thắng hoặc ngang `static`, ta có cơ sở chọn `single_gate` làm main model.

### 6.3. Tiêu chí chọn main model

| Kết quả Stage 3 core | Quyết định |
|---|---|
| `single_gate` thắng `static` RankIC trên cả NASDAQ và SP500 | Chọn `single_gate` làm main model |
| `single_gate` thắng một dataset, thua nhẹ một dataset | Xem regime-wise gains và P@10/CostSharpe |
| `single_gate` thua `static` rõ trên cả hai | Chưa đủ để claim ReCIL-SG mạnh hơn |
| `context_only` gần bằng hoặc hơn `single_gate` | Novelty của interaction gate yếu, cần phân tích lại |
| `static` thắng mọi thứ | Paper phải pivot hoặc cần tuning thêm |

### 6.4. Acceptance gate tối thiểu cho Stage 3 core

```text
[ ] NASDAQ chạy xong đủ 3 seeds.
[ ] SP500 chạy xong đủ 3 seeds.
[ ] Không có NaN/Inf.
[ ] Có RankIC, IC, ICIR, RankICIR, P@10.
[ ] Có Return@10, Turnover@10, CostSharpe@10.
[ ] Có active_params, epoch_sec, GPU memory nếu dùng GPU.
[ ] single_gate có ít nhất một lợi thế rõ: RankIC, P@10, CostSharpe, hoặc regime-wise gain.
```

---

## 7. Stage 3 extended — kiểm tra full và moe

Chỉ chạy extended sau khi Stage 3 core đã xong.

```bash
PYTHONPATH=. python -m src.recil.run_experiments \
  --datasets nasdaq sp500 \
  --variants moe full \
  --seeds 0 1 2 \
  --epochs 50 \
  --alpha-rank 0.3 \
  --lambda-entropy 1e-3 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0 \
  --transaction-cost-bps 10 \
  --output-dir outputs_stage3_patch_extended
```

### 7.1. Cách đọc kết quả full/moe

Không nên mặc định nghĩ `full` là model chính. `full` chỉ nên làm main nếu nó thắng rõ RankIC.

| Kết quả | Cách viết trong paper |
|---|---|
| `full` thắng RankIC, P@10 và ổn định qua seeds | Có thể chọn `full` làm main |
| `single_gate` thắng RankIC, `full` thắng Sharpe | Main là `single_gate`, `full` là top-k diagnostic |
| `moe/full` thua RankIC nhưng có CostSharpe tốt | Viết như expressivity vs ranking stability trade-off |
| `moe/full` unstable | Đưa vào ablation/limitation, không overclaim |

### 7.2. Kiểm tra router collapse

Nếu có log router entropy, cần xem:

```text
entropy quá thấp: một expert thống trị, MoE gần như không còn mixture
entropy quá cao: router gần uniform, context không có tác dụng rõ
entropy biến động mạnh giữa seeds: expert semantics không ổn định
```

Nếu collapse, thử:

```bash
--lambda-entropy 1e-2
```

hoặc giảm số expert nếu code có flag tương ứng:

```bash
--num-experts 3
```

---

## 8. Tuning hẹp nếu kết quả chưa đủ mạnh

Không tuning quá rộng vì dễ mất thời gian và làm story rối. Tuning nên hẹp và có mục đích.

### 8.1. Tuning `alpha_rank`

Nếu RankIC yếu nhưng loss/IC ổn:

```bash
for a in 0.1 0.3 1.0; do
  PYTHONPATH=. python -m src.recil.run_experiments \
    --datasets nasdaq \
    --variants single_gate \
    --seeds 0 \
    --epochs 50 \
    --alpha-rank $a \
    --weight-decay 1e-4 \
    --grad-clip-norm 1.0 \
    --output-dir outputs_tune_alpha_$a
done
```

Cách chọn:

```text
Chọn alpha_rank có validation RankIC tốt nhất, không chọn theo test.
```

### 8.2. Tuning regularization

Nếu overfitting hoặc seed variance lớn:

```bash
for wd in 1e-5 1e-4 5e-4; do
  PYTHONPATH=. python -m src.recil.run_experiments \
    --datasets nasdaq \
    --variants single_gate \
    --seeds 0 \
    --epochs 50 \
    --alpha-rank 0.3 \
    --weight-decay $wd \
    --grad-clip-norm 1.0 \
    --output-dir outputs_tune_wd_$wd
done
```

### 8.3. Tuning dropout

Nếu code hỗ trợ `--dropout`:

```bash
for d in 0.05 0.1 0.2; do
  PYTHONPATH=. python -m src.recil.run_experiments \
    --datasets nasdaq \
    --variants single_gate \
    --seeds 0 \
    --epochs 50 \
    --alpha-rank 0.3 \
    --dropout $d \
    --weight-decay 1e-4 \
    --grad-clip-norm 1.0 \
    --output-dir outputs_tune_dropout_$d
done
```

### 8.4. Quy tắc chống over-tuning

```text
[ ] Tune trên validation, không tune trên test.
[ ] Không thử quá nhiều config nếu không có lý do.
[ ] Ghi lại tất cả config đã thử.
[ ] Không chỉ report config thắng nếu đã thử nhiều lần trên test.
[ ] Config chọn trên NASDAQ phải được xác nhận lại trên SP500.
```

---

## 9. Baseline ngoài ReCIL

Để paper mạnh hơn, cần ít nhất một baseline ngoài các ablation nội bộ.

Thứ tự ưu tiên:

```text
1. Original StockMixer hoặc StockMixer-compatible baseline.
2. Static-LRI.
3. Context-only.
4. ReCIL-SG.
5. MoE/full ablation.
```

Nếu repo có sẵn StockMixer baseline, chạy cùng split/dataset/seed. Nếu không có, tạo một baseline tương thích:

```text
temporal mixer + stock/static mixer + same prediction head
no market context
same input window
same target
same train/val/test split
same metrics
```

Không dùng baseline với protocol khác vì reviewer sẽ bắt lỗi.

---

## 10. Regime-wise analysis

Regime-wise analysis là phần giúp paper khác biệt. Không chỉ cần main metric, mà phải chứng minh model giúp đúng lúc market regime thay đổi.

### 10.1. Regime cần phân tích

Ít nhất gồm:

```text
high volatility vs low volatility
high dispersion vs low dispersion
high PCA/common-factor dominance vs low PCA/common-factor dominance
bull/trend-up vs bear/trend-down nếu có
```

### 10.2. Cách trình bày

Không chỉ trình bày absolute RankIC. Nên trình bày gain so với static:

```text
Gain = RankIC_ReCIL - RankIC_Static
```

Bảng paper-final:

| Dataset | Regime | Static RankIC | ReCIL-SG RankIC | Gain |
|---|---|---:|---:|---:|
| NASDAQ | Low volatility | ... | ... | ... |
| NASDAQ | High volatility | ... | ... | ... |
| SP500 | Low volatility | ... | ... | ... |
| SP500 | High volatility | ... | ... | ... |

### 10.3. Diễn giải đúng

Nếu ReCIL tốt hơn trong high-volatility/high-dispersion regime, claim rất mạnh:

```text
The gain is concentrated in regimes where static cross-asset interaction is least reliable.
```

Nếu gain không tập trung theo regime:

```text
Không claim regime-specific benefit. Chỉ claim general lightweight context-gated interaction nếu main results đủ tốt.
```

---

## 11. Efficiency analysis

Patch đã thêm nền để tính active parameters. Cần tạo bảng efficiency thật.

### 11.1. Metrics cần log

```text
active_params
all_params
epoch_sec
inference_ms_per_day
peak_gpu_memory_mb
```

### 11.2. Cách đọc

| Trường hợp | Diễn giải |
|---|---|
| `single_gate` active_params chỉ tăng nhẹ so với `static` | Claim lightweight hợp lý |
| `full` active_params cao nhưng RankIC không hơn | Dùng làm evidence cho over-parameterization |
| `moe/full` chậm hơn nhiều | Không nên chọn làm main nếu metric không thắng rõ |
| Runtime gần nhau nhưng params khác | Cần kiểm tra bottleneck data loading hoặc metric computation |

### 11.3. Bảng paper-final

| Model | Active Params | Epoch Sec | Inference ms/day | Peak GPU MB |
|---|---:|---:|---:|---:|
| Static-LRI | ... | ... | ... | ... |
| Context-only | ... | ... | ... | ... |
| ReCIL-SG | ... | ... | ... | ... |
| ReCIL-MoE | ... | ... | ... | ... |
| ReCIL-Full | ... | ... | ... | ... |

---

## 12. Transaction-cost diagnostics

Sharpe không nên report đơn giản nếu không tính cost. Vì stock selection có turnover, cần ít nhất cost-adjusted metric.

### 12.1. Metrics cần có

```text
Return@10
Turnover@10
CostReturn@10
CostSharpe@10
```

### 12.2. Cost levels nên thử

```text
0 bps
5 bps
10 bps
20 bps
```

Nếu chỉ có thời gian cho một mức, dùng:

```text
10 bps
```

### 12.3. Cách diễn giải

| Kết quả | Cách viết |
|---|---|
| ReCIL Sharpe cao nhưng turnover quá cao | Không claim portfolio superiority mạnh |
| ReCIL CostSharpe vẫn cao hơn static | Có thể dùng như diagnostic support |
| Static CostSharpe tốt hơn ReCIL | Giữ Sharpe ở phần phụ, tập trung RankIC |
| Cost-adjusted metrics không ổn định | Không để làm main claim |

---

## 13. Statistical validation

Để kết quả đáng tin hơn, thêm bootstrap confidence interval cho daily RankIC difference.

### 13.1. Metric chính

```text
DeltaRankIC_t = RankIC_ReCIL_t - RankIC_Static_t
```

### 13.2. Bootstrap đề xuất

```text
block bootstrap theo thời gian
block size: 5 hoặc 10 trading days
number of bootstrap samples: 1000
```

### 13.3. Report

```text
mean DeltaRankIC
95% confidence interval
win rate across days
win rate across seeds
```

### 13.4. Acceptance signal mạnh

```text
95% CI của DeltaRankIC nằm trên 0
hoặc CI hơi chạm 0 nhưng regime-wise gain và P@10 cùng ủng hộ
```

Không nên chỉ dựa vào p-value.

---

## 14. File output cần tổng hợp

Sau toàn bộ experiment, nên có thư mục:

```text
outputs_stage3_patch_core/
outputs_stage3_patch_extended/
outputs_analysis_final/
```

Trong `outputs_analysis_final/`, nên có:

```text
main_results.csv
main_results_latex.tex
ablation_results.csv
ablation_latex.tex
regime_results.csv
regime_latex.tex
efficiency_table.csv
efficiency_latex.tex
transaction_cost_results.csv
bootstrap_ci_results.csv
interpretability_correlations.csv
experiment_manifest.json
paper_ready_summary.md
```

`paper_ready_summary.md` nên trả lời:

```text
[ ] Main model là gì?
[ ] Nó thắng baseline nào?
[ ] Thắng trên dataset nào?
[ ] Metric chính là gì?
[ ] Có regime-wise support không?
[ ] Có cost-adjusted support không?
[ ] Có efficiency support không?
[ ] Claim nào được phép viết?
[ ] Claim nào không được phép viết?
```

---

## 15. Quyết định viết paper theo kết quả

### 15.1. Nếu `single_gate` thắng rõ

Paper story:

```text
Static cross-asset interaction is too rigid under non-stationary markets.
A simple temporally-causal market context gate improves ranking stability.
More expressive MoE/full variants are not always better.
Controlled adaptivity is the key.
```

Main model:

```text
ReCIL-SG
```

### 15.2. Nếu `full` thắng rõ

Paper story:

```text
Multi-scale market context and dynamic expert routing improve cross-sectional interaction learning.
```

Main model:

```text
ReCIL-Full
```

Nhưng chỉ chọn hướng này nếu `full` thắng RankIC, không chỉ thắng Sharpe.

### 15.3. Nếu `static` vẫn thắng

Không nên nộp claim “ReCIL beats static”. Có thể pivot thành:

```text
A rigorous study of context-conditioned interaction in financial forecasting shows that simple gates are safer than high-capacity routers, but static interaction remains a strong baseline.
```

Hướng này khoa học nhưng yếu hơn cho PRICAI. Cần thêm cải tiến hoặc tìm dataset/regime nơi ReCIL thật sự thắng.

---

## 16. Checklist trước khi viết paper

```text
[ ] Patch apply thành công.
[ ] Unit tests pass.
[ ] Synthetic smoke test pass.
[ ] Stage 2 reproduce ổn.
[ ] Stage 3 core chạy đủ NASDAQ + SP500.
[ ] Có ít nhất 3 seeds.
[ ] Có baseline ngoài ReCIL nếu kịp.
[ ] Có main_results table.
[ ] Có ablation table.
[ ] Có regime-wise table.
[ ] Có efficiency table với active_params.
[ ] Có cost-adjusted diagnostics.
[ ] Có bootstrap hoặc confidence interval.
[ ] Không có NaN/Inf.
[ ] Không có leakage trong context.
[ ] Mask invalid assets được áp dụng.
[ ] Không claim “causal inference”.
[ ] Không claim `full` là best nếu RankIC không ủng hộ.
[ ] Paper story khớp đúng với kết quả thật.
```

---

## 17. Lệnh chạy khuyến nghị theo thứ tự

### Bước 1: test

```bash
PYTHONPATH=. python -m pytest -q tests/recil/test_model_pipeline_optimizations.py
```

### Bước 2: smoke synthetic

```bash
PYTHONPATH=. python -m src.recil.train_recil \
  --synthetic \
  --quick-test \
  --variant single_gate \
  --epochs 2 \
  --output-dir outputs_patch_smoke \
  --alpha-rank 0.3 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0
```

### Bước 3: reproduce Stage 2

```bash
PYTHONPATH=. python -m src.recil.run_experiments \
  --datasets nasdaq \
  --variants static context_only single_gate moe full \
  --seeds 0 1 \
  --epochs 30 \
  --alpha-rank 0.3 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0 \
  --transaction-cost-bps 10 \
  --output-dir outputs_stage2_patch_reproduce
```

### Bước 4: Stage 3 core

```bash
PYTHONPATH=. python -m src.recil.run_experiments \
  --datasets nasdaq sp500 \
  --variants static context_only single_gate \
  --seeds 0 1 2 \
  --epochs 50 \
  --alpha-rank 0.3 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0 \
  --transaction-cost-bps 10 \
  --output-dir outputs_stage3_patch_core
```

### Bước 5: Stage 3 extended

```bash
PYTHONPATH=. python -m src.recil.run_experiments \
  --datasets nasdaq sp500 \
  --variants moe full \
  --seeds 0 1 2 \
  --epochs 50 \
  --alpha-rank 0.3 \
  --lambda-entropy 1e-3 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0 \
  --transaction-cost-bps 10 \
  --output-dir outputs_stage3_patch_extended
```

---

## 18. Kết luận vận hành

Thứ tự ưu tiên đúng sau patch là:

```text
1. Chứng minh code mới không lỗi.
2. Chứng minh patch không làm mất signal cũ.
3. Chứng minh main model trên NASDAQ + SP500.
4. Chứng minh bằng ablation rằng context-gated interaction có ý nghĩa.
5. Chứng minh bằng regime analysis rằng model giúp trong non-stationary regimes.
6. Chứng minh bằng efficiency rằng model đủ lightweight.
7. Chỉ sau đó mới viết paper.
```

Điểm mấu chốt cho PRICAI:

```text
Không phải model càng phức tạp càng tốt.
Model thắng là model có evidence sạch nhất, ổn định nhất, và giải thích được rõ nhất.
```
