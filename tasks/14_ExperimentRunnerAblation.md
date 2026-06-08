# 14 - Experiment Runner Ablation

## Goal

Implement controlled experiment runner and define staged ablation schedule for NASDAQ and SP500.

## Sources

- `ThucThi.md`
- `AGENTS.md`
- `tasks/12_TrainingCLICheckpointLogging.md`
- `tasks/13_SmokeTestsSyntheticVaRealData.md`

## Files

Update:

```text
StockMixer/src/recil/run_experiments.py
```

Create:

```text
StockMixer/docs/experiment_plan_recil.md
StockMixer/tests/recil/test_run_experiments.py
```

## Steps

1. Implement CLI:

   ```text
   --datasets nasdaq sp500
   --variants static context_only single_gate moe full
   --seeds 0 1 2
   --epochs 50
   --output-dir outputs
   --max-parallel 1
   --dry-run
   --quick-test
   ```

2. Dry-run must print exact `python3 -m src.recil.train_recil ...` commands without executing them.

3. Real-run rules:
   - Default `--max-parallel 1` for safety.
   - Stop on first failed command unless `--keep-going` is explicitly implemented and documented.
   - Write `run_manifest.jsonl` with dataset, variant, seed, command, start/end time, status.

4. Write staged experiment plan:
   - `stage_0_sanity_check`: already covered by Task 13.
   - `stage_1_small_scale`: NASDAQ, variants `static full`, seed `0`, 5 epochs.
   - `stage_2_medium_scale`: NASDAQ, all variants, seeds `0 1`, 30-50 epochs.
   - `stage_3_full_scale`: NASDAQ + SP500, all variants, seeds `0 1 2`, 50+ epochs if time allows.
   - Crypto optional only after main equity results.

5. Tests:
   - Dry-run includes all requested dataset/variant/seed combinations.
   - Dry-run does not create training outputs.
   - Invalid variant fails fast.

## Test

```bash
cd StockMixer
python3 -m src.recil.run_experiments --datasets nasdaq --variants static full --seeds 0 --epochs 1 --dry-run
python3 -m pytest tests/recil/test_run_experiments.py -q
test -f docs/experiment_plan_recil.md
```

## Pass criteria

- Dry-run commands are correct and reproducible.
- Experiment plan uses staged escalation and pass gates.
- Runner does not launch long experiments by default during tests.

## Expected output

A safe mechanism to launch ablations after smoke tests pass.

## Limitations

This task does not require actually running stage 2 or stage 3 experiments.
