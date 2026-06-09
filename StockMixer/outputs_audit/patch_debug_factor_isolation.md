# Patch Debug Factor Isolation Audit

Date: 2026-06-08 22:33:42 +07

Run ID: `patch_debug_factor_isolation`

Decision: `pass_for_stage2_reproduce_after_fix`, not Stage 3 yet.

The RankIC regression was investigated by isolating training config, model-side masking, and architecture refactor effects. The main root cause is not transaction-cost metrics, not AdamW alone, not gradient clipping alone, and not model-side asset masking alone. The largest recoverable factor is the patch refactor of the encoder and low-rank expert initialization.

## Sources

- StockMixer AAAI 2024: the baseline motivation is a simple MLP-style stock/time mixer, so architecture changes should be isolated carefully before scaling experiments: https://mlanthology.org/aaai/2024/fan2024aaai-stockmixer/
- RankNet: pairwise ranking loss is valid for ranking, but its weight is a tunable objective balance, not a fixed universal constant: https://icml.cc/Conferences/2005/proceedings/papers/012_LearningToRank_BurgesEtAl.pdf
- PyTorch AdamW: AdamW decouples weight decay from adaptive moments, so `weight_decay` must be treated as a separate experimental intervention: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- PyTorch gradient clipping: `clip_grad_norm_` modifies gradients in-place, so it must be isolated as an intervention: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- FiLM: feature-wise conditioning is technically sound, but conditioning/gates must be diagnosed rather than assumed beneficial: https://arxiv.org/abs/1709.07871
- TFT interpretability motivation: learned gates/weights are useful diagnostics for temporal/regime behavior, not automatic evidence of better ranking: https://arxiv.org/abs/1912.09363

## Preflight And Validation

| Check | Result |
|---|---|
| `compileall src/recil` before experiments | pass |
| `pytest tests/recil -q` before experiments | pass, `77 passed` |
| Added `--no-mask-invalid-assets` CLI isolation flag | pass |
| `py_compile train_recil.py run_experiments.py` after CLI change | pass |
| `pytest test_train_cli.py test_run_experiments.py` after CLI change | pass, `6 passed` |
| Module/model tests after refactor fix | pass, `31 passed` |
| Full regression after refactor fix | pass, `77 passed` |
| Legacy `RIC` scan across debug outputs | pass, no matches |
| Artifact checks for refactor-fix outputs | pass, arrays finite and checkpoints load |

## Code Fixes Applied

Two non-ad-hoc corrections were made after comparing the patched code against the pre-patch implementation and Task 09 contract:

1. `IndicatorEncoder` was restored to the original/spec architecture:

```text
Linear(F,D) -> GELU -> Dropout -> Linear(D,D) -> LayerNorm(D)
```

The patch had moved normalization to the raw feature input and removed the final representation LayerNorm. This affected all variants, including `static`.

2. `RegimeConditionedLowRankExperts` now uses standard trainable output initialization and bias by default.

The patch zero-initialized the final expert projection and removed bias. For `static`, this made the stock-interaction path start almost as no interaction plus normalization, which weakened ranking signal in 30-epoch runs.

The patch's useful improvements remain: mask-aware utilities, variant-specific modules, parameter reports, RankICIR/cost diagnostics, AdamW support, grad norm logging, and runner forwarding.

## Phase A: Old Training Regime On Patched Code

Command:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments \
  --datasets nasdaq \
  --variants static single_gate moe \
  --seeds 0 1 \
  --epochs 30 \
  --alpha-rank 0.1 \
  --weight-decay 0 \
  --grad-clip-norm 0 \
  --transaction-cost-bps 10 \
  --output-dir outputs_patch_debug_old_regime
```

Result: fail. Reverting `alpha_rank`, `weight_decay`, and gradient clipping did not recover RankIC.

| Output | Variant | RankIC | IC | Precision@10 | Sharpe |
|---|---|---:|---:|---:|---:|
| `outputs_stage2` baseline | static | 0.031013 | 0.034343 | 0.046203 | 0.027600 |
| `outputs_stage2` baseline | single_gate | 0.032052 | 0.027211 | 0.053165 | 0.878273 |
| `outputs_stage2` baseline | moe | 0.019773 | 0.022368 | 0.055485 | 1.180955 |
| `outputs_stage2_patch_reproduce` | static | 0.014719 | 0.011003 | 0.016667 | 1.674755 |
| `outputs_stage2_patch_reproduce` | single_gate | 0.013176 | 0.009810 | 0.015823 | 2.323219 |
| `outputs_stage2_patch_reproduce` | moe | 0.016358 | 0.013358 | 0.013291 | 1.467217 |
| `outputs_patch_debug_old_regime` | static | 0.009001 | 0.005842 | 0.014557 | 0.771768 |
| `outputs_patch_debug_old_regime` | single_gate | 0.007160 | 0.006715 | 0.017300 | 0.879531 |
| `outputs_patch_debug_old_regime` | moe | 0.009029 | 0.007524 | 0.016667 | 1.351632 |

Interpretation: the regression is not explained by the changed training hyperparameters alone. The patched architecture/refactor path must be inspected.

## Phase C: Model-Side Mask Isolation

Implemented flag:

```bash
--no-mask-invalid-assets
```

This disables model-side hidden/prediction masking while keeping loss/evaluation masks intact.

Command:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments \
  --datasets nasdaq \
  --variants static \
  --seeds 0 1 \
  --epochs 30 \
  --alpha-rank 0.1 \
  --weight-decay 0 \
  --grad-clip-norm 0 \
  --transaction-cost-bps 10 \
  --output-dir outputs_patch_debug_mask_off \
  --no-mask-invalid-assets
```

Result: fail.

| Output | Variant | RankIC | IC | Precision@10 | Sharpe |
|---|---|---:|---:|---:|---:|
| `outputs_patch_debug_mask_off` | static | 0.008256 | 0.006437 | 0.014768 | 1.696831 |

Interpretation: model-side asset masking alone is not the root cause.

## Refactor Fix Validation

After restoring encoder normalization placement and non-zero expert output initialization, static and single_gate were rerun with old regime.

Static command:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments \
  --datasets nasdaq \
  --variants static \
  --seeds 0 1 \
  --epochs 30 \
  --alpha-rank 0.1 \
  --weight-decay 0 \
  --grad-clip-norm 0 \
  --transaction-cost-bps 10 \
  --output-dir outputs_patch_debug_static_refactor_fix
```

Single-gate command:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments \
  --datasets nasdaq \
  --variants single_gate \
  --seeds 0 1 \
  --epochs 30 \
  --alpha-rank 0.1 \
  --weight-decay 0 \
  --grad-clip-norm 0 \
  --transaction-cost-bps 10 \
  --output-dir outputs_patch_debug_single_gate_refactor_fix
```

Result: pass for debug stage.

| Output | Variant | RankIC | IC | Precision@10 | Sharpe |
|---|---|---:|---:|---:|---:|
| `outputs_patch_debug_static_refactor_fix` | static | 0.023790 | 0.024266 | 0.049789 | 0.886853 |
| `outputs_patch_debug_single_gate_refactor_fix` | single_gate | 0.027645 | 0.024671 | 0.048101 | 0.447940 |

Single-gate seed-level results:

| Seed | RankIC | IC | Precision@10 | Sharpe | CostSharpe@10 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.026392 | 0.023850 | 0.048101 | 0.797194 | 0.775104 |
| 1 | 0.028898 | 0.025493 | 0.048101 | 0.098685 | -0.062288 |

Interpretation:

- `single_gate` recovered above the acceptance threshold `RankIC_mean >= 0.025`.
- It is still below the original Stage 2 `single_gate` baseline `0.032052`, so this is a partial but meaningful recovery.
- `static` recovered strongly from `0.009001` to `0.023790`, close to but still below the `0.025` target.
- Precision@10 recovered to roughly the original range, which supports the same conclusion as RankIC.

## Gate Diagnostics

`single_gate` context gate on test days:

| Output | Seed | Gate Mean | Gate Std | Min | Max |
|---|---:|---:|---:|---:|---:|
| old-regime before fix | 0 | 0.617406 | 0.441362 | 0.000008 | 0.999994 |
| old-regime before fix | 1 | 0.543639 | 0.454179 | 0.000081 | 0.999761 |
| refactor fix | 0 | 0.076698 | 0.021953 | 0.039571 | 0.142149 |
| refactor fix | 1 | 0.086940 | 0.026064 | 0.042838 | 0.172825 |

Interpretation:

- Before the fix, the gate was not simply suppressed; it was highly saturated/unstable.
- After the fix, the gate is conservative and stable while RankIC improves.
- Gate collapse is not the primary failure mode; unstable upstream representation/expert initialization was more important.

## Conclusion

Root cause:

1. Primary: encoder normalization placement was changed away from the original Task 09 design and pre-patch implementation.
2. Primary: low-rank expert output was zero-initialized and bias-free by default, weakening stock-interaction learning in short/medium runs.
3. Secondary: training hyperparameters can still matter, but reverting them alone did not recover performance.
4. Not root cause: model-side asset masking alone.
5. Not sufficient evidence: higher Sharpe/CostSharpe without RankIC recovery.

Scientific status:

- Debug stage passes after the refactor fix because `single_gate RankIC_mean = 0.027645 >= 0.025`.
- Stage 3 is still premature because the full 5-variant Stage 2 reproduce has not yet been rerun after this fix.

## Recommended Next Step

Run a new Stage 2 reproduce after the refactor fix:

```bash
/mnt/d/LOBProj/LOBExp/.venv/Scripts/python.exe -m src.recil.run_experiments \
  --datasets nasdaq \
  --variants static context_only single_gate moe full \
  --seeds 0 1 \
  --epochs 30 \
  --alpha-rank 0.1 \
  --weight-decay 0 \
  --grad-clip-norm 0 \
  --transaction-cost-bps 10 \
  --output-dir outputs_stage2_refactor_fix_reproduce
```

Acceptance for proceeding to Stage 3:

- `single_gate` remains `RankIC_mean >= 0.025`.
- `static` remains competitive and preferably `RankIC_mean >= 0.025`.
- `moe/full` do not produce NaN/Inf and are not selected as main model unless RankIC improves, not merely Sharpe.
- Full tests remain pass.
- Analysis and audit are written for `outputs_stage2_refactor_fix_reproduce`.
