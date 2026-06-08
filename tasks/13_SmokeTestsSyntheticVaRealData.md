# 13 - Smoke Tests Synthetic Va Real Data

## Goal

Verify the implemented pipeline end to end before running long experiments: compile, unit tests, synthetic training, and one-epoch NASDAQ smoke tests.

## Sources

- `tasks/03_ContextRawKhongLeakage.md` through `tasks/12_TrainingCLICheckpointLogging.md`
- `AGENTS.md`
- `StockMixer/src/Original/train.py`

## Files

Update only if failures require fixes:

```text
StockMixer/src/recil/*
StockMixer/tests/recil/*
```

Create audit output:

```text
StockMixer/outputs_audit/stage_0_sanity_check.md
```

## Steps

1. Run compile:

   ```bash
   cd StockMixer
   python3 -m compileall src/recil
   ```

2. Run unit tests:

   ```bash
   cd StockMixer
   python3 -m pytest tests/recil -q
   ```

3. Run synthetic debug:

   ```bash
   cd StockMixer
   python3 -m src.recil.train_recil --synthetic --variant full --epochs 3 --quick-test --output-dir outputs_stage0
   ```

4. Run real NASDAQ one-epoch smoke if dependencies/data load correctly:

   ```bash
   cd StockMixer
   python3 -m src.recil.train_recil --dataset nasdaq --variant static --epochs 1 --quick-test --output-dir outputs_stage0
   python3 -m src.recil.train_recil --dataset nasdaq --variant full --epochs 1 --quick-test --output-dir outputs_stage0
   ```

5. Write audit:
   - Command run
   - Runtime rough estimate
   - Whether CUDA/RTX 3090 was used
   - Any NaN/Inf
   - Output files created
   - Decision: pass or block

## Test

```bash
test -f StockMixer/outputs_audit/stage_0_sanity_check.md
rg -n "compileall|pytest|synthetic|nasdaq|pass|block|NaN|CUDA" StockMixer/outputs_audit/stage_0_sanity_check.md
```

## Pass criteria

- Compile passes.
- Unit tests pass.
- Synthetic full model run finishes and writes metrics.
- NASDAQ static/full one-epoch smoke either passes or has a documented environment/data blocker.

## Expected output

Stage 0 audit proving the pipeline is safe enough for ablation dry-runs.

## Limitations

One-epoch smoke metrics do not indicate model quality. They only validate plumbing.
