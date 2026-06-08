# Stage 0 Sanity Check Audit

Date: 2026-06-08

Decision: pass

No block remains for Stage 0. Compile, unit tests, synthetic quick training, and NASDAQ static/full quick smoke all passed.

## Environment

- Python: `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe`
- CUDA: available
- GPU used by `--device auto`: `cuda`
- GPU name: NVIDIA GeForce RTX 3090
- Scope: quick plumbing smoke only; metrics below are not model-quality evidence.

## Commands

| Step | Command | Status | Rough runtime |
|---|---|---:|---:|
| compileall | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m compileall src/recil` | pass | 0.07s |
| pytest | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest tests/recil -q` | pass, 65 tests | 7.15s |
| synthetic | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.train_recil --synthetic --variant full --epochs 3 --quick-test --output-dir outputs_stage0` | pass | 3.74s |
| nasdaq static | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.train_recil --dataset nasdaq --variant static --epochs 1 --quick-test --output-dir outputs_stage0` | pass | 3.78s |
| nasdaq full | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.train_recil --dataset nasdaq --variant full --epochs 1 --quick-test --output-dir outputs_stage0` | pass | 3.71s |

## Output Files

Each smoke run created:

```text
aux_outputs.npz
best_model.pt
config.json
contexts.npy
labels.npy
last_model.pt
masks.npy
metrics.json
predictions.npy
train_log.csv
```

Run directories:

```text
outputs_stage0/synthetic/full/seed_0
outputs_stage0/nasdaq/static/seed_0
outputs_stage0/nasdaq/full/seed_0
```

## NaN And Inf Checks

No NaN/Inf was found in saved prediction arrays, labels, masks, or metrics for the three smoke runs.

| Run | Prediction shape | Context shape | Aux shapes | Metrics finite | Clean metric names |
|---|---:|---:|---|---:|---:|
| synthetic/full | `(4, 20)` | `(4, 7)` | scale `(4, 3)`, router `(4, 4)`, gate `(4, 64)` | pass | pass |
| nasdaq/static | `(4, 1026)` | `(4, 7)` | scale `(0, 0)`, router `(0, 0)`, gate `(0, 0)` | pass | pass |
| nasdaq/full | `(4, 1026)` | `(4, 7)` | scale `(4, 3)`, router `(4, 4)`, gate `(4, 64)` | pass | pass |

Clean metric keys observed:

```text
mse
IC
RankIC
ICIR
Precision@10
Sharpe
num_valid_days
num_days
```

No legacy `RIC` metric key was present.

## Smoke Metrics

These values verify plumbing only.

| Run | mse | IC | RankIC | ICIR | Precision@10 | Sharpe | num_days |
|---|---:|---:|---:|---:|---:|---:|---:|
| synthetic/full | 0.001804 | 0.021488 | 0.088722 | 0.127658 | 0.600000 | 184.654579 | 4 |
| nasdaq/static | 0.008631 | 0.175172 | 0.160714 | 2.587275 | 0.025000 | 9.002450 | 4 |
| nasdaq/full | 0.007717 | 0.173498 | 0.163305 | 2.510893 | 0.050000 | 7.124601 | 4 |

## Notes

- NASDAQ data loaded successfully with shape `(1026, 1245, 5)`.
- The quick-test cap produced four test days per run.
- Stage 0 is safe to proceed to dry-run ablation scheduling.
