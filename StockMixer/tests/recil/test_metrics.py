import numpy as np
import pytest

from src.recil.metrics import (
    compute_ic_series,
    evaluate_predictions,
    long_only_daily_return,
    pearson_corr_masked,
    precision_at_k,
    sharpe_ratio,
    spearman_corr_masked,
    summarize_ic,
)


def test_perfect_ranking_has_unit_ic_and_rankic():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    target = np.array([10.0, 20.0, 30.0, 40.0])
    mask = np.ones(4)
    assert pearson_corr_masked(pred, target, mask) == pytest.approx(1.0)
    assert spearman_corr_masked(pred, target, mask) == pytest.approx(1.0)


def test_reversed_ranking_is_negative():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    target = np.array([4.0, 3.0, 2.0, 1.0])
    mask = np.ones(4)
    assert pearson_corr_masked(pred, target, mask) == pytest.approx(-1.0)
    assert spearman_corr_masked(pred, target, mask) == pytest.approx(-1.0)


def test_masked_outlier_does_not_affect_result():
    pred = np.array([1.0, 2.0, 3.0, 1000.0])
    target = np.array([1.0, 2.0, 3.0, -1000.0])
    mask = np.array([1.0, 1.0, 1.0, 0.0])
    assert pearson_corr_masked(pred, target, mask) == pytest.approx(1.0)
    assert spearman_corr_masked(pred, target, mask) == pytest.approx(1.0)


def test_too_few_valid_assets_returns_nan():
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([3.0, 2.0, 1.0])
    mask = np.array([1.0, 1.0, 0.0])
    assert np.isnan(pearson_corr_masked(pred, target, mask))
    assert np.isnan(spearman_corr_masked(pred, target, mask))


def test_constant_vectors_return_nan():
    mask = np.ones(4)
    assert np.isnan(pearson_corr_masked([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0], mask))
    assert np.isnan(spearman_corr_masked([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0], mask))
    assert np.isnan(pearson_corr_masked([1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0], mask))
    assert np.isnan(spearman_corr_masked([1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0], mask))


def test_spearman_ties_use_average_ranks():
    pred = np.array([1.0, 1.0, 2.0, 3.0])
    target = np.array([1.0, 2.0, 2.0, 3.0])
    mask = np.ones(4)
    expected = np.corrcoef(np.array([0.5, 0.5, 2.0, 3.0]), np.array([0.0, 1.5, 1.5, 3.0]))[0, 1]
    assert spearman_corr_masked(pred, target, mask) == pytest.approx(expected)


def test_compute_ic_series_day_major_and_asset_major():
    preds = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    targets = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    masks = np.ones_like(preds)
    day_major = compute_ic_series(preds, targets, masks)
    asset_major = compute_ic_series(preds.T, targets.T, masks.T, asset_major=True)
    assert np.allclose(day_major["IC"], np.array([1.0, -1.0], dtype=np.float32))
    assert np.allclose(day_major["RankIC"], np.array([1.0, -1.0], dtype=np.float32))
    assert np.allclose(asset_major["IC"], day_major["IC"])
    assert np.allclose(asset_major["RankIC"], day_major["RankIC"])


def test_summarize_ic_ignores_nan_days_and_uses_clean_keys():
    ic = np.array([1.0, np.nan, -0.5, 0.5])
    rankic = np.array([0.5, np.nan, 0.5, 0.5])
    summary = summarize_ic(ic, rankic)
    assert summary["IC"] == pytest.approx((1.0 - 0.5 + 0.5) / 3.0)
    assert summary["RankIC"] == pytest.approx(0.5)
    assert summary["ICIR"] == pytest.approx(np.array([1.0, -0.5, 0.5]).mean() / np.array([1.0, -0.5, 0.5]).std())
    assert np.isnan(summary["RankICIR"])
    assert summary["num_valid_days"] == 3
    assert summary["num_rankic_valid_days"] == 3
    forbidden_key = "R" + "IC"
    assert forbidden_key not in summary


def test_precision_at_k_perfect_overlap():
    pred = np.array([0.1, 0.4, 0.3, 0.2])
    target = np.array([-0.2, 0.5, 0.3, 0.0])
    mask = np.ones(4)
    assert precision_at_k(pred, target, mask, k=2) == pytest.approx(1.0)


def test_masked_asset_cannot_be_selected_for_ranking_or_return():
    pred = np.array([100.0, 0.4, 0.3, 0.2])
    target = np.array([-100.0, 0.5, 0.3, 0.0])
    mask = np.array([0.0, 1.0, 1.0, 1.0])
    assert precision_at_k(pred, target, mask, k=1) == pytest.approx(1.0)
    assert long_only_daily_return(pred, target, mask, k=1) == pytest.approx(0.5)


def test_precision_at_k_uses_effective_k():
    pred = np.array([0.1, 0.4, 0.3, 0.2])
    target = np.array([0.4, 0.3, 0.2, 0.1])
    mask = np.array([1.0, 1.0, 0.0, 0.0])
    assert precision_at_k(pred, target, mask, k=10) == pytest.approx(1.0)
    assert long_only_daily_return(pred, target, mask, k=10) == pytest.approx(0.35)


def test_no_valid_assets_returns_nan_for_ranking_metrics():
    pred = np.array([1.0, 2.0])
    target = np.array([2.0, 1.0])
    mask = np.zeros(2)
    assert np.isnan(precision_at_k(pred, target, mask, k=1))
    assert np.isnan(long_only_daily_return(pred, target, mask, k=1))


def test_constant_positive_returns_sharpe_is_finite():
    value = sharpe_ratio(np.array([0.01, 0.01, 0.01]))
    assert np.isfinite(value)
    assert value > 0.0


def test_evaluate_predictions_clean_keys_and_masked_mse():
    preds = np.array(
        [
            [100.0, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    targets = np.array(
        [
            [-100.0, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    masks = np.array(
        [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    result = evaluate_predictions(preds, targets, masks, k=2)
    assert set(result) == {"mse", "IC", "RankIC", "ICIR", "Precision@2", "Sharpe", "num_valid_days", "num_days"}
    assert result["mse"] == pytest.approx(0.0)
    assert result["Precision@2"] == pytest.approx(1.0)
    assert result["num_valid_days"] == 2
    assert result["num_days"] == 2
    forbidden_key = "R" + "IC"
    assert forbidden_key not in result


def test_evaluate_predictions_asset_major_matches_day_major():
    preds = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    targets = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    masks = np.ones_like(preds)
    day_major = evaluate_predictions(preds, targets, masks, k=1)
    asset_major = evaluate_predictions(preds.T, targets.T, masks.T, k=1, asset_major=True)
    assert day_major == asset_major
