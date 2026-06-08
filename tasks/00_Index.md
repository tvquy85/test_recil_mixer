# 00 - Index: Thu Tu Thuc Thi ReCIL-Mixer

## Goal

Tao ban do task tuan tu de trien khai ReCIL-Mixer theo tung buoc nho, co gate kiem chung ro rang, thay vi giao mot task lon dai. Khong duoc nhay sang task tiep theo neu task hien tai chua pass.

## Sources

- `MoTa.md`
- `ThucThi.md`
- `StockMixer/src/Original/train.py`
- `StockMixer/src/Original/model.py`
- `StockMixer/src/Original/evaluator.py`
- `StockMixer/src/Enhanced/preprocess_context.py`
- StockMixer AAAI 2024: https://mlanthology.org/aaai/2024/fan2024aaai-stockmixer/
- PRICAI 2026 CFP: https://2026.pricai.org/calls/call-for-papers

## Files

Task files to execute in order:

```text
tasks/01_KiemKeRepoVaNguon.md
tasks/02_MoiTruongVaDataContract.md
tasks/03_ContextRawKhongLeakage.md
tasks/04_ContextScalerTrainOnlyVaCache.md
tasks/05_DatasetItemVaAlignment.md
tasks/06_MetricsMaskedICRankIC.md
tasks/07_PrecisionSharpeVaEvaluator.md
tasks/08_LossMaskedMSEPairwiseRank.md
tasks/09_EncoderTemporalModules.md
tasks/10_RegimeLowRankExperts.md
tasks/11_ReCILMixerVaVariants.md
tasks/12_TrainingCLICheckpointLogging.md
tasks/13_SmokeTestsSyntheticVaRealData.md
tasks/14_ExperimentRunnerAblation.md
tasks/15_RegimeInterpretabilityEfficiency.md
tasks/16_FinalAcceptanceChecklist.md
```

## Steps

1. Run tasks strictly from `01` to `16`.
2. At the end of each task, report:
   - Files changed
   - Tests / smoke tests passed
   - Short explanation
   - Known limitations
3. If a task fails, stop and fix that task before continuing.
4. Prefer new code under `StockMixer/src/recil`; do not modify `StockMixer/src/Original` except for read-only audit or explicit compatibility notes.
5. Use `python3 -m ...` in commands because this workspace has `python3` available and may not have `python`.

## Test

```bash
ls tasks
rg -n "Goal|Files|Test|Pass criteria|Sources" tasks
```

## Pass criteria

- All files `00` to `16` exist.
- Every task has `Goal`, `Sources`, `Files`, `Test`, and `Pass criteria`.
- The execution order and stop-on-fail rule are explicit.

## Expected output

A readable task roadmap that can be handed to another engineer or agent.

## Limitations

This file is an index only. It does not implement ReCIL-Mixer.
