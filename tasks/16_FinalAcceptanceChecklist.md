# 16 - Final Acceptance Checklist

## Goal

Run the final gate before long experiments or paper writing. This task verifies implementation quality, no-leakage protocol, metrics correctness, output reproducibility, and experiment readiness.

## Sources

- `MoTa.md`
- `ThucThi.md`
- `AGENTS.md`
- `tasks/01_KiemKeRepoVaNguon.md` through `tasks/15_RegimeInterpretabilityEfficiency.md`
- PRICAI 2026 CFP: https://2026.pricai.org/calls/call-for-papers

## Files

Create:

```text
StockMixer/outputs_audit/final_acceptance_recil.md
```

Update only if checks reveal issues:

```text
StockMixer/src/recil/*
StockMixer/tests/recil/*
StockMixer/docs/*
```

## Steps

1. Run static verification:

   ```bash
   cd StockMixer
   python3 -m compileall src/recil
   python3 -m pytest tests/recil -q
   ```

2. Run smoke commands:

   ```bash
   cd StockMixer
   python3 -m src.recil.train_recil --synthetic --variant full --epochs 3 --quick-test --output-dir outputs_final_smoke
   python3 -m src.recil.train_recil --dataset nasdaq --variant static --epochs 1 --quick-test --output-dir outputs_final_smoke
   python3 -m src.recil.train_recil --dataset nasdaq --variant full --epochs 1 --quick-test --output-dir outputs_final_smoke
   python3 -m src.recil.run_experiments --datasets nasdaq --variants static full --seeds 0 --epochs 1 --dry-run
   ```

3. Verify metric names:

   ```bash
   rg -n "\"RIC\"|\\bRIC\\b" src/recil tests/recil outputs_final_smoke || true
   rg -n "RankIC|ICIR|Precision@10|Sharpe" src/recil tests/recil outputs_final_smoke
   ```

   Any `RIC` occurrence in new ReCIL code must be removed unless it is in a comment explaining old evaluator behavior.

4. Verify no-leakage context:
   - `compute_market_context_raw` has no normalization.
   - `TrainOnlyStandardizer.fit` is called only with train contexts.
   - Val/test contexts are transformed using train state.

5. Verify output reproducibility:
   - `config.json` stores dataset, variant, seed, lookback, steps, model hyperparameters.
   - `metrics.json` stores best epoch, validation metrics, test metrics.
   - Predictions, labels, masks, contexts, and aux outputs are saved for analysis.

6. Write final acceptance audit:
   - Pass/fail per check.
   - Exact commands run.
   - Known limitations.
   - Recommended next stage (`stage_1_small_scale`, `stage_2_medium_scale`, or blocked).

## Test

```bash
test -f StockMixer/outputs_audit/final_acceptance_recil.md
rg -n "compileall|pytest|no-leakage|RankIC|ICIR|stage_1_small_scale|blocked" StockMixer/outputs_audit/final_acceptance_recil.md
```

## Pass criteria

- Compile and tests pass.
- Synthetic and NASDAQ one-epoch smoke runs pass or any blocker is precise and reproducible.
- No new ReCIL evaluator reports `RIC`.
- Final audit recommends a concrete next stage.

## Expected output

A final readiness report for running controlled ReCIL experiments and later writing the PRICAI paper.

## Limitations

Passing this checklist does not guarantee paper-quality results. It only verifies that the pipeline is credible enough to run experiments.
