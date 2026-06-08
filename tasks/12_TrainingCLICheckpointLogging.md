# 12 - Training CLI Checkpoint Logging

## Goal

Implement a clean training CLI for synthetic and real ReCIL runs with checkpointing, logging, and JSON outputs.

## Sources

- `StockMixer/src/Original/train.py`
- `StockMixer/src/Enhanced/train_ablation.py`
- `tasks/05_DatasetItemVaAlignment.md`
- `tasks/07_PrecisionSharpeVaEvaluator.md`
- `tasks/08_LossMaskedMSEPairwiseRank.md`
- `tasks/11_ReCILMixerVaVariants.md`

## Files

Update:

```text
StockMixer/src/recil/train_recil.py
StockMixer/tests/recil/test_train_cli.py
```

## Steps

1. Implement argparse CLI:

   ```text
   --dataset nasdaq|sp500|crypto
   --data-root StockMixer/dataset
   --variant static|context_only|single_gate|moe|full
   --seed 0
   --epochs 50
   --batch-size 1
   --lr 1e-4
   --d-model 64
   --market-dim 32
   --num-experts 4
   --lookback 16
   --steps 1
   --alpha-rank 0.1
   --device auto|cpu|cuda
   --output-dir outputs
   --synthetic
   --quick-test
   ```

2. Set deterministic seeds for `random`, `numpy`, and `torch`.

3. Implement synthetic path:
   - Small dataset with `N=20`, `T=16`, `F=5`, `context_dim=7`.
   - Target is a noisy but learnable function of last-day input and context.
   - Useful for debugging without dependency on real pickles.

4. Implement real data path using Task 05 dataset.

5. Training loop:
   - Train chronologically shuffled train offsets only.
   - Evaluate validation and test each epoch.
   - Select best by validation `RankIC`; if NaN, fall back to validation `IC`; if both NaN, fall back to negative validation loss.
   - Use new ReCIL loss and evaluator.

6. Output layout:

   ```text
   outputs/{dataset}/{variant}/seed_{seed}/config.json
   outputs/{dataset}/{variant}/seed_{seed}/train_log.csv
   outputs/{dataset}/{variant}/seed_{seed}/metrics.json
   outputs/{dataset}/{variant}/seed_{seed}/best_model.pt
   outputs/{dataset}/{variant}/seed_{seed}/last_model.pt
   outputs/{dataset}/{variant}/seed_{seed}/predictions.npy
   outputs/{dataset}/{variant}/seed_{seed}/labels.npy
   outputs/{dataset}/{variant}/seed_{seed}/masks.npy
   outputs/{dataset}/{variant}/seed_{seed}/contexts.npy
   outputs/{dataset}/{variant}/seed_{seed}/aux_outputs.npz
   ```

7. Tests:
   - `--help` works.
   - Synthetic `--epochs 1 --quick-test` creates output files.
   - Metrics JSON contains required keys and no `RIC`.

## Test

```bash
cd StockMixer
python3 -m src.recil.train_recil --help
python3 -m src.recil.train_recil --synthetic --variant full --epochs 1 --quick-test --output-dir outputs_test
python3 -m pytest tests/recil/test_train_cli.py -q
```

## Pass criteria

- CLI help displays arguments.
- Synthetic run finishes without NaN.
- Output directory contains config, log, metrics, and checkpoints.
- Metrics use clean names.

## Expected output

A reproducible entrypoint for smoke tests and real experiments.

## Limitations

Do not start long GPU experiments in this task. That belongs to Task 14.
