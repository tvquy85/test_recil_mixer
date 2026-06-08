# ReCIL Experiment Plan

This plan follows the project rule to start small, write an audit per stage, and only escalate after the prior gate passes.

## Stage 0 Sanity Check

Status: pass

Audit: `StockMixer/outputs_audit/stage_0_sanity_check.md`

Completed checks:

- `compileall` passed for `src/recil`.
- ReCIL unit tests passed.
- Synthetic `full` quick training passed.
- NASDAQ `static` and `full` one-epoch quick smoke passed.
- RTX 3090 was available and used through `--device auto`.
- No NaN/Inf was found in saved predictions or metrics.

Gate: do not run later stages if Stage 0 regresses.

## Stage 1 Small Scale

Goal: confirm short ablation plumbing on real equity data before broader runs.

Command:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments \
  --datasets nasdaq \
  --variants static full \
  --seeds 0 \
  --epochs 5 \
  --output-dir outputs_stage1
```

Pass gate:

- Both runs complete with checkpoints and metrics.
- Metrics JSON uses clean keys: `IC`, `RankIC`, `ICIR`, `Precision@10`, `Sharpe`.
- Predictions, labels, masks, contexts, and aux outputs are finite where applicable.
- Write `outputs_audit/stage_1_small_scale.md` before moving on.

## Stage 2 Medium Scale

Goal: main NASDAQ ablation across all five variants.

Command:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments \
  --datasets nasdaq \
  --variants static context_only single_gate moe full \
  --seeds 0 1 \
  --epochs 30 \
  --output-dir outputs_stage2
```

Recommended extension: increase to 50 epochs only if Stage 1 is stable and runtime is acceptable.

Pass gate:

- All commands in `run_manifest.jsonl` have `status=pass`.
- No NaN/Inf in saved predictions or metrics.
- Rank-oriented metrics are available for every variant and seed.
- Write `outputs_audit/stage_2_medium_scale.md`.

## Stage 3 Full Scale

Goal: final equity ablation for paper tables.

Command:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments \
  --datasets nasdaq sp500 \
  --variants static context_only single_gate moe full \
  --seeds 0 1 2 \
  --epochs 50 \
  --output-dir outputs_stage3
```

Escalate beyond 50 epochs only if the Stage 2 audit supports it and GPU time allows.

Pass gate:

- NASDAQ and SP500 both complete for all variants and seeds.
- Results can be aggregated as mean and standard deviation by dataset and variant.
- Efficiency and interpretability artifacts are available for `moe` and `full`.
- Write `outputs_audit/stage_3_full_scale.md`.

## Optional Crypto

Crypto is optional and should run only after equity results are stable.

Suggested command:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments \
  --datasets crypto \
  --variants static moe full \
  --seeds 0 1 \
  --epochs 30 \
  --output-dir outputs_crypto
```

## Safety Rules

- Do not jump stages when the previous stage fails.
- Keep `--max-parallel 1` until resource-aware scheduling is implemented.
- Treat one-epoch or quick-test results as plumbing checks, not quality evidence.
- Use `--dry-run` before launching any stage to inspect exact commands.
