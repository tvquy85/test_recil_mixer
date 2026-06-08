# train_ablation.py
"""
train_ablation.py
==================

Unified training script for StockMixer ablations:
- original:   the baseline StockMixer (as in model_original.py)
- gated_nocontext: gMLP stock-mixing without market context
- gated_context:   gMLP stock-mixing conditioned on 5-D market context

Highlights
----------
- Keeps original dataset format/targets/loss/metrics.
- Switch architectures via --arch without touching data code.
- Supports two data sources:
    * --data_source pickle  (default): dataset/{MARKET}/[eod,mask,gt,price]_data.pkl
    * --data_source csv     : via load_EOD_data(data_path, market_name, tickers, steps)
- Computes market context (5-D) with correct alignment: uses window ending at offset-1.
- Gate saturation logging for gated variants (ratio of gates near 0 and 1).

Example
-------
python train_ablation.py --arch original --epochs 1
python train_ablation.py --arch gated_nocontext --depth 3 --m 20 --dropout 0.1 --epochs 1
python train_ablation.py --arch gated_context --depth 3 --m 20 --dropout 0.1 --epochs 1 \
    --data_source csv --data_path /data/csv --tickers_file tickers.txt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import random
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

CUR_DIR = Path(__file__).resolve().parent
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))

from evaluator import evaluate  # shared evaluator (IC, RIC, Sharpe@5, prec@10)
try:
    from model_original_refactor import get_loss  # same loss as original (MSE + pairwise-rank)
except Exception:
    # Fallback: define local get_loss identical to refactor implementation
    import torch.nn.functional as F  # type: ignore
    import torch as _torch

    def get_loss(prediction: _torch.Tensor, ground_truth: _torch.Tensor, base_price: _torch.Tensor,
                 mask: _torch.Tensor, batch_size: int, alpha: float):
        device = prediction.device
        all_one = _torch.ones(batch_size, 1, dtype=_torch.float32, device=device)
        return_ratio = (prediction - base_price) / (base_price + 1e-8)
        reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
        pre_pw_dif = (return_ratio @ all_one.T) - (all_one @ return_ratio.T)
        gt_pw_dif = (all_one @ ground_truth.T) - (ground_truth @ all_one.T)
        mask_pw = mask @ mask.T
        rank_loss = _torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))
        loss = reg_loss + alpha * rank_loss
        return loss, reg_loss, rank_loss, return_ratio


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Make cuDNN deterministic where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset_pickle(base_dir: str, market_name: str, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset from pickle files (default path layout).
    Returns: eod_data, mask_data, gt_data, price_data
    """
    dataset_dir = os.path.join(base_dir, 'dataset', market_name)

    if market_name.upper() == 'SP500':
        # Special-case SP500 (as in original code)
        data = np.load(os.path.join(base_dir, 'dataset', 'SP500', 'SP500.npy'))
        # Use only the last 915 days as in original implementation
        data = data[:, 915:, :]  # shape (N, T, F)
        price_data = data[:, :, -1]  # closing price
        mask_data = np.ones((data.shape[0], data.shape[1]))
        eod_data = data
        gt_data = np.zeros((data.shape[0], data.shape[1]))
        for ticket in range(data.shape[0]):
            for row in range(1, data.shape[1]):
                # steps-ahead return ratio
                prev = data[ticket][row - steps][-1]
                cur = data[ticket][row][-1]
                gt_data[ticket][row] = (cur - prev) / (prev + 1e-8)
        return eod_data, mask_data, gt_data, price_data

    # Default pickle layout (e.g., NASDAQ)
    with open(os.path.join(dataset_dir, 'eod_data.pkl'), 'rb') as f:
        eod_data = pickle.load(f)
    with open(os.path.join(dataset_dir, 'mask_data.pkl'), 'rb') as f:
        mask_data = pickle.load(f)
    with open(os.path.join(dataset_dir, 'gt_data.pkl'), 'rb') as f:
        gt_data = pickle.load(f)
    with open(os.path.join(dataset_dir, 'price_data.pkl'), 'rb') as f:
        price_data = pickle.load(f)
    return eod_data, mask_data, gt_data, price_data


def load_dataset_csv(data_path: str, market_name: str, tickers_file: str, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset from CSVs via shared loader.
    Returns: eod_data, mask_data, gt_data, price_data (price_data := base_price)
    """
    from load_data import load_EOD_data  # shared loader
    with open(tickers_file, 'r') as f:
        tickers = [ln.strip() for ln in f if ln.strip()]
    eod_data, mask_data, gt_data, base_price = load_EOD_data(
        data_path=data_path,
        market_name=market_name,
        tickers=tickers,
        steps=steps
    )
    price_data = base_price
    return eod_data, mask_data, gt_data, price_data


def compute_market_context_if_needed(arch: str, eod_data, lookback_length: int) -> np.ndarray | torch.Tensor | None:
    """Compute 5-D market context per window if architecture requires it."""
    if arch != 'gated_context':
        return None
    # Prefer NumPy version; works for np.ndarray inputs (pickle/csv branches)
    from preprocess_context import market_state_from_closes
    return market_state_from_closes(eod_data, close_col=-1, window=lookback_length)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified training script for StockMixer ablations")

    # Architecture & model hyper-params
    parser.add_argument('--arch', type=str, default='original',
                        choices=['original', 'gated_nocontext', 'gated_context'],
                        help='Model architecture to use')
    parser.add_argument('--depth', type=int, default=2, help='Number of gMLP blocks for gated models')
    parser.add_argument('--m', type=int, default=20, help='Hidden dimension (market states) for stock mixing')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability in gMLP blocks')

    # Reproducibility & device
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cuda', type=int, default=-1, help='CUDA device index (-1 for CPU)')

    # Training schedule & loss
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--alpha', type=float, default=0.1, help='Weight for ranking loss term')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')

    # Data & windowing
    parser.add_argument('--market_name', type=str, default='NASDAQ', help='Market name (NASDAQ, SP500, ...)')
    parser.add_argument('--lookback_length', type=int, default=16, help='Lookback window length T')
    parser.add_argument('--steps', type=int, default=1, help='Prediction horizon (steps ahead)')
    parser.add_argument('--valid_index', type=int, default=756, help='Train/Val split index')
    parser.add_argument('--test_index', type=int, default=1008, help='Val/Test split index')

    # Data source selection
    parser.add_argument('--data_source', type=str, default='pickle', choices=['pickle', 'csv'],
                        help='Data source: "pickle" uses pkl files; "csv" uses load_EOD_data()')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Root folder containing CSV files when --data_source=csv')
    parser.add_argument('--tickers_file', type=str, default=None,
                        help='Text file listing tickers (one per line) when --data_source=csv')

    args = parser.parse_args()

    # Determinism
    set_deterministic(args.seed)

    # Device
    if args.cuda >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.cuda}')
    else:
        device = torch.device('cpu')

    # Base dir (repo root)
    base_dir = str(Path(__file__).resolve().parents[2])

    # Load dataset
    if args.data_source == 'pickle':
        eod_data, mask_data, gt_data, price_data = load_dataset_pickle(base_dir, args.market_name, args.steps)
    else:
        if args.data_path is None or args.tickers_file is None:
            raise ValueError('When --data_source=csv you must provide --data_path and --tickers_file')
        eod_data, mask_data, gt_data, price_data = load_dataset_csv(
            data_path=args.data_path,
            market_name=args.market_name,
            tickers_file=args.tickers_file,
            steps=args.steps
        )

    # Infer stats
    stock_num = eod_data.shape[0]
    fea_num = eod_data.shape[2]
    trade_dates = mask_data.shape[1]

    # Market context (when needed)
    market_ctx = compute_market_context_if_needed(args.arch, eod_data, args.lookback_length)
    # market_ctx shape: (T - lookback_length, 5)  → we'll index by (offset-1)

    # Instantiate model
    if args.arch == 'original':
        try:
            from model_original import StockMixer as OriginalStockMixer
        except Exception:
            # Fallback: load baseline StockMixer from src/Original/model.py
            import importlib.util as _ilu
            baseline_path = Path(__file__).resolve().parents[2] / 'src' / 'Original' / 'model.py'
            spec = _ilu.spec_from_file_location('baseline_model', str(baseline_path))
            if spec is None or spec.loader is None:
                raise
            baseline_mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(baseline_mod)  # type: ignore[attr-defined]
            OriginalStockMixer = getattr(baseline_mod, 'StockMixer')

        model = OriginalStockMixer(
            stocks=stock_num,
            time_steps=args.lookback_length,
            channels=fea_num,
            market=args.m,
            scale=3,  # kept for compatibility with original
        )
    elif args.arch == 'gated_nocontext':
        from model_gated_nocontext import GatedMLPNoContext
        model = GatedMLPNoContext(
            stocks=stock_num,
            time_steps=args.lookback_length,
            channels=fea_num,
            market_hidden=args.m,
            depth=args.depth,
            dropout=args.dropout
        )
    elif args.arch == 'gated_context':
        from model_gated_withcontext import GatedMLPWithContext
        model = GatedMLPWithContext(
            stocks=stock_num,
            time_steps=args.lookback_length,
            channels=fea_num,
            market_hidden=args.m,
            depth=args.depth,
            ctx_dim=5,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Precompute shuffled offsets per epoch
    all_offsets = np.arange(start=0, stop=args.valid_index, dtype=int)

    def get_batch(offset: int | None = None):
        """Fetch one training batch aligned to `offset` (window start)."""
        if offset is None:
            offset = random.randrange(0, args.valid_index)

        T = args.lookback_length
        S = args.steps

        # Mask over [offset, offset+T+S) → min across time (valid if all valid)
        mask_batch = mask_data[:, offset: offset + T + S]
        mask_batch = np.min(mask_batch, axis=1)

        data = eod_data[:, offset: offset + T, :]               # (N, T, F)
        price = price_data[:, offset + T - 1]                    # (N,)
        gt = gt_data[:, offset + T + S - 1]                      # (N,)

        if args.arch == 'gated_context':
            if offset == 0:
                ctx_vec = np.zeros(5, dtype=np.float32)
            else:
                # market_ctx aligned to end at offset-1
                ctx_vec = market_ctx[offset - 1]
            return data, mask_batch[:, None], price[:, None], gt[:, None], ctx_vec
        else:
            return data, mask_batch[:, None], price[:, None], gt[:, None]

    def to_device_tuple(batch, arch: str):
        """Convert numpy batch to torch tensors on correct device."""
        if arch == 'gated_context':
            data_batch, mask_batch, price_batch, gt_batch, ctx = batch
        else:
            data_batch, mask_batch, price_batch, gt_batch = batch
            ctx = None

        data_t = torch.as_tensor(data_batch, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(mask_batch, dtype=torch.float32, device=device)
        price_t = torch.as_tensor(price_batch, dtype=torch.float32, device=device)
        gt_t = torch.as_tensor(gt_batch, dtype=torch.float32, device=device)
        ctx_t = None
        if ctx is not None:
            ctx_t = torch.as_tensor(ctx, dtype=torch.float32, device=device)
        return data_t, mask_t, price_t, gt_t, ctx_t

    def validate(start_index: int, end_index: int):
        """Roll forward over [start_index, end_index) and collect performance."""
        with torch.no_grad():
            length = end_index - start_index
            cur_pred = np.zeros((stock_num, length), dtype=float)
            cur_gt = np.zeros((stock_num, length), dtype=float)
            cur_mask = np.zeros((stock_num, length), dtype=float)

            loss_acc = reg_acc = rank_acc = 0.0

            begin = start_index - args.lookback_length - args.steps + 1
            finish = end_index - args.lookback_length - args.steps + 1
            for cur_offset in range(begin, finish):
                batch = get_batch(cur_offset)
                data_t, mask_t, price_t, gt_t, ctx_t = to_device_tuple(batch, args.arch)

                if args.arch == 'gated_context':
                    pred = model(data_t, ctx_t)
                else:
                    pred = model(data_t)

                cur_loss, cur_reg, cur_rank, cur_rr = get_loss(
                    pred, gt_t, price_t, mask_t, stock_num, args.alpha
                )
                loss_acc += cur_loss.item()
                reg_acc += cur_reg.item()
                rank_acc += cur_rank.item()

                idx = cur_offset - begin
                cur_pred[:, idx] = cur_rr[:, 0].detach().cpu().numpy()
                cur_gt[:, idx] = gt_t[:, 0].detach().cpu().numpy()
                cur_mask[:, idx] = mask_t[:, 0].detach().cpu().numpy()

            length = finish - begin
            loss_acc /= max(1, length)
            reg_acc /= max(1, length)
            rank_acc /= max(1, length)
            perf = evaluate(cur_pred, cur_gt, cur_mask)

        return loss_acc, reg_acc, rank_acc, perf

    best_val_loss = float('inf')
    best_val_perf = None
    best_test_perf = None

    for epoch in range(args.epochs):
        np.random.shuffle(all_offsets)
        train_loss = train_reg = train_rank = 0.0

        # Number of training batches per epoch
        num_batches = args.valid_index - args.lookback_length - args.steps + 1
        if num_batches <= 0:
            raise ValueError("valid_index too small relative to lookback_length and steps")

        for j in range(num_batches):
            offset = all_offsets[j]

            # For context model, skip offset==0 (no prior window to compute context)
            if args.arch == 'gated_context' and offset == 0:
                continue

            batch = get_batch(offset)
            data_t, mask_t, price_t, gt_t, ctx_t = to_device_tuple(batch, args.arch)

            optimizer.zero_grad()

            if args.arch == 'gated_context':
                pred = model(data_t, ctx_t)
            else:
                pred = model(data_t)

            loss, reg_loss, rank_loss, _ = get_loss(
                pred, gt_t, price_t, mask_t, stock_num, args.alpha
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_reg += reg_loss.item()
            train_rank += rank_loss.item()

        train_loss /= num_batches
        train_reg /= num_batches
        train_rank /= num_batches

        # Validation & Test
        val_loss, val_reg, val_rank, val_perf = validate(args.valid_index, args.test_index)
        test_loss, test_reg, test_rank, test_perf = validate(args.test_index, trade_dates)

        # Gate saturation for gated variants
        if args.arch in ('gated_nocontext', 'gated_context'):
            low_ratio, high_ratio = model.stock_mixer.gate_saturation()
        else:
            low_ratio = high_ratio = 0.0

        # Epoch log
        print(f"Epoch {epoch + 1}/{args.epochs}  Train: total={train_loss:.4e} (mse={train_reg:.4e}, alpha*rank={train_rank:.4e})")
        print(f"  Val: total={val_loss:.4e} (mse={val_reg:.4e}, alpha*rank={val_rank:.4e})  "
              f"perf: mse={val_perf['mse']:.4e}, IC={val_perf['IC']:.4e}, RIC={val_perf['RIC']:.4e}, "
              f"prec@10={val_perf['prec_10']:.4e}, SR={val_perf['sharpe5']:.4e}")
        print(f"  Test: total={test_loss:.4e} (mse={test_reg:.4e}, alpha*rank={test_rank:.4e}) "
              f"perf: mse={test_perf['mse']:.4e}, IC={test_perf['IC']:.4e}, RIC={test_perf['RIC']:.4e}, "
              f"prec@10={test_perf['prec_10']:.4e}, SR={test_perf['sharpe5']:.4e}")
        if args.arch in ('gated_nocontext', 'gated_context'):
            print(f"  Gate saturation: low<0.05={low_ratio:.4f}, high>0.95={high_ratio:.4f}")

        # Track best by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_perf = val_perf
            best_test_perf = test_perf

        # Show current best
        if best_val_perf is not None and best_test_perf is not None:
            print(f"Best Val  : mse={best_val_perf['mse']:.4e}, IC={best_val_perf['IC']:.4e}, "
                  f"RIC={best_val_perf['RIC']:.4e}, prec@10={best_val_perf['prec_10']:.4e}, "
                  f"SR={best_val_perf['sharpe5']:.4e}")
            print(f"Best Test : mse={best_test_perf['mse']:.4e}, IC={best_test_perf['IC']:.4e}, "
                  f"RIC={best_test_perf['RIC']:.4e}, prec@10={best_test_perf['prec_10']:.4e}, "
                  f"SR={best_test_perf['sharpe5']:.4e}")


if __name__ == '__main__':
    main()
