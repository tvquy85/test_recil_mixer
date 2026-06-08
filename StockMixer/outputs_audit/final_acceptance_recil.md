# Final Acceptance Audit For ReCIL

Date: 2026-06-08

Decision: pass

Recommended next stage: `stage_1_small_scale`

The ReCIL implementation is ready for controlled small-scale experiments. This
acceptance gate verifies compile/tests, quick synthetic and NASDAQ smoke runs,
metric naming, no-leakage context construction, reproducible artifacts, and
experiment-runner dry-run behavior. These checks do not establish paper-quality
model performance.

## Environment

- Python: `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe`
- Reason for interpreter choice: prior environment audit found system `python3`
  lacks required ML/test dependencies.
- CUDA available: yes
- GPU: NVIDIA GeForce RTX 3090
- Scope: final readiness gate before long experiments and PRICAI paper writing.

## Sources

- Local project instructions: `AGENTS.md`
- Local design and execution docs: `MoTa.md`, `ThucThi.md`
- Local StockMixer paper source: `StockMixer/paper.md`
- Prior task chain: `tasks/01_KiemKeRepoVaNguon.md` through
  `tasks/15_RegimeInterpretabilityEfficiency.md`
- Stage 0 audit: `StockMixer/outputs_audit/stage_0_sanity_check.md`
- PRICAI 2026 CFP: https://2026.pricai.org/calls/call-for-papers
- PyTorch checkpoint guidance:
  https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
- PyTorch reproducibility guidance:
  https://docs.pytorch.org/docs/stable/notes/randomness.html

## Commands

| Step | Command | Status | Rough runtime |
|---|---|---:|---:|
| compileall | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m compileall src/recil` | pass | <0.1s |
| pytest | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m pytest tests/recil -q` | pass, 74 tests, 1 known warning | 13.3s |
| synthetic smoke | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.train_recil --synthetic --variant full --epochs 3 --quick-test --output-dir outputs_final_smoke` | pass | 3.2s |
| NASDAQ static smoke | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.train_recil --dataset nasdaq --variant static --epochs 1 --quick-test --output-dir outputs_final_smoke` | pass | 3.6s |
| NASDAQ full smoke | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.train_recil --dataset nasdaq --variant full --epochs 1 --quick-test --output-dir outputs_final_smoke` | pass | 3.6s |
| runner dry-run | `/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments --datasets nasdaq --variants static full --seeds 0 --epochs 1 --dry-run` | pass | 1.6s |

The pytest warning is expected: `analysis.py` records a warning that
`train_log.csv` does not yet contain wall-clock epoch time, so
`time_per_epoch_sec` is left blank in efficiency analysis.

## Smoke Outputs

Run directories created:

```text
outputs_final_smoke/synthetic/full/seed_0
outputs_final_smoke/nasdaq/static/seed_0
outputs_final_smoke/nasdaq/full/seed_0
```

Each run created the required reproducibility artifacts:

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

Checkpoint loadability was verified with
`torch.load(..., map_location="cpu", weights_only=False)`. Each best checkpoint
contains `model_state_dict`.

## Metric Checks

Command:

```bash
rg -n "\"RIC\"|\bRIC\b" src/recil tests/recil outputs_final_smoke || true
```

Result: no matches. No new ReCIL evaluator or smoke output reports legacy
`RIC`.

Command:

```bash
rg -n "RankIC|ICIR|Precision@10|Sharpe" src/recil tests/recil outputs_final_smoke
```

Result: pass. Clean metric names are present in code, tests, and smoke outputs.

Observed smoke metrics:

| Run | mse | IC | RankIC | ICIR | Precision@10 | Sharpe | num_days |
|---|---:|---:|---:|---:|---:|---:|---:|
| synthetic/full | 0.001804 | 0.021488 | 0.088722 | 0.127658 | 0.600000 | 184.654579 | 4 |
| nasdaq/static | 0.008631 | 0.175172 | 0.160714 | 2.587275 | 0.025000 | 9.002450 | 4 |
| nasdaq/full | 0.007717 | 0.173498 | 0.163305 | 2.510893 | 0.050000 | 7.124601 | 4 |

These values are smoke-test plumbing evidence only.

## no-leakage Context Check

The context protocol passes the final no-leakage review:

- `compute_market_context_raw(close_window, valid_mask=None, eps=1e-8)` computes
  raw context from a provided historical close-price window and does not perform
  normalization.
- `build_context_cache` constructs each raw context from
  `eod[:, offset : offset + lookback, close_col]` and
  `mask[:, offset : offset + lookback]`; the target day is excluded.
- `TrainOnlyStandardizer().fit(raw[valid_train])` fits scaling state only on
  train offsets.
- `contexts = scaler.transform(raw)` applies the train-fitted scaler to all
  offsets, including validation and test.
- Cache metadata records
  `same_input_window_offset_to_offset_plus_lookback_exclusive`.

## Artifact And Finite-Value Checks

Read-only inspection over `outputs_final_smoke/*/*/seed_0` verified:

- Required files are present for all three smoke runs.
- `config.json` stores dataset, resolved dataset, variant, seed, lookback,
  steps, `d_model`, `market_dim`, `num_experts`, `device_resolved`,
  `synthetic`, and `quick_test`.
- `metrics.json` contains exactly the clean test metric keys:
  `mse`, `IC`, `RankIC`, `ICIR`, `Precision@10`, `Sharpe`,
  `num_valid_days`, and `num_days`.
- `predictions.npy`, `labels.npy`, `masks.npy`, and `contexts.npy` are finite.
- Synthetic/full output shapes: predictions `(4, 20)`, contexts `(4, 7)`,
  scale weights `(4, 3)`, router weights `(4, 4)`, context gate `(4, 64)`.
- NASDAQ/static output shapes: predictions `(4, 1026)`, contexts `(4, 7)`,
  static aux arrays `(0, 0)`.
- NASDAQ/full output shapes: predictions `(4, 1026)`, contexts `(4, 7)`,
  scale weights `(4, 3)`, router weights `(4, 4)`, context gate `(4, 64)`.

## Runner Dry-Run

The dry-run command printed two deterministic training commands:

```text
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.recil.train_recil --dataset nasdaq --variant static --seed 0 --epochs 1 --output-dir outputs
D:\LOBProj\LOBExp\.venv\Scripts\python.exe -m src.recil.train_recil --dataset nasdaq --variant full --seed 0 --epochs 1 --output-dir outputs
```

No Stage 1 training was launched in this task.

## Known Limitations

- Passing this checklist does not guarantee paper-quality results.
- Quick-test smoke runs use only four test days and should not be interpreted
  as model-quality evidence.
- The current `train_log.csv` lacks wall-clock timing, so efficiency analysis
  leaves `time_per_epoch_sec` blank.
- Sharpe is a simple diagnostic without transaction costs, turnover, slippage,
  or capacity modeling.
- NYSE remains unsupported in this workspace because the audited local NYSE
  files are zero bytes.
- Stage escalation must still follow `StockMixer/docs/experiment_plan_recil.md`;
  do not jump to medium or full scale until `stage_1_small_scale` has its own
  pass audit.

## Final Gate

Pass criteria status:

- Compile and tests pass: pass.
- Synthetic and NASDAQ quick smoke runs pass: pass.
- No new ReCIL evaluator reports `RIC`: pass.
- Clean `RankIC`, `ICIR`, `Precision@10`, and `Sharpe` metrics are present:
  pass.
- no-leakage context construction is enforced by code and tests: pass.
- Reproducible output artifacts are present and loadable: pass.
- Concrete next stage is defined: `stage_1_small_scale`.

Final recommendation: proceed to `stage_1_small_scale` using the staged command
documented in `StockMixer/docs/experiment_plan_recil.md`, then write
`StockMixer/outputs_audit/stage_1_small_scale.md` before any Stage 2 run.
