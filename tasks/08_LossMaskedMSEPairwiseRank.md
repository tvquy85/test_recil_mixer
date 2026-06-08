# 08 - Loss Masked MSE Pairwise Rank

## Goal

Implement loss functions for ReCIL training: masked regression, pairwise ranking, and total loss wrapper.

## Sources

- `StockMixer/src/Original/model.py`
- `StockMixer/src/Enhanced/model_original_refactor.py`
- RankNet / pairwise ranking: https://doi.org/10.1145/1102351.1102363

## Files

Update:

```text
StockMixer/src/recil/losses.py
StockMixer/tests/recil/test_losses.py
```

## Steps

1. Implement:

   ```python
   def masked_mse_loss(pred, target, mask, eps=1e-8): ...
   def pairwise_rank_loss(pred, target, mask, max_pairs_per_day=4096, eps=1e-8): ...
   def recil_loss(pred, target, mask, aux=None, alpha_rank=0.1, lambda_entropy=0.0): ...
   ```

2. Shape rules:
   - Accept `[N]` or `[B, N]`.
   - Return scalar tensor.
   - Mask uses `mask > 0.5`.

3. Masked MSE:
   - Average only over valid assets.
   - If no valid assets, return zero tensor connected to graph.

4. Pairwise ranking:
   - Only compare valid pairs within the same day.
   - Use logistic pairwise loss:

     ```text
     log(1 + exp(-sign(y_i - y_j) * (pred_i - pred_j)))
     ```

   - Skip pairs where target tie is near zero.
   - If valid pairs exceed `max_pairs_per_day`, sample deterministically from current random state or document the sampling method.

5. Optional entropy:
   - If `lambda_entropy > 0` and `aux["router_weights"]` exists, subtract `lambda_entropy * entropy`.
   - Default `lambda_entropy=0.0`.

6. Tests:
   - Masked MSE toy case equals expected value.
   - Correct ranking has lower pairwise loss than reversed ranking.
   - One valid asset returns zero rank loss, no NaN.
   - Total loss combines terms correctly.

## Test

```bash
cd StockMixer
python3 -m pytest tests/recil/test_losses.py -q
python3 -m compileall src/recil
```

## Pass criteria

- Loss tests pass.
- Losses handle all-invalid or one-valid cases without NaN.
- Pairwise loss does not use masked assets.

## Expected output

Stable losses for synthetic and real ReCIL training.

## Limitations

Pair sampling can introduce variance for large universes. Keep seeds fixed and log `max_pairs_per_day`.
