"""
Training script for the improved stock forecasting models.

This script mirrors the structure of the original ``train.py`` but allows the
user to select between several model architectures, including the original
``StockMixer`` (imported from ``model.py``), the proposed ``GatedMLPStockModel``
and ``CrossAttentionStockModel`` (imported from ``improved_models.py``).

To keep the example self‑contained, this script assumes that the preprocessed
dataset (EOD data, masks, ground truth and price data) has already been
generated and pickled in ``dataset/{market_name}``.  See ``load_data.py`` for
the preprocessing pipeline.  When running on a new dataset, adjust the
``market_name``, ``stock_num``, ``lookback_length`` and related hyperparameters
accordingly.

Example usage::

    python train_improved.py --model gated_mlp --market NASDAQ

"""
import argparse
import os
import random
import pickle

import numpy as np
import torch

from evaluator import evaluate
from Original.improved_models_old import get_loss, GatedMLPStockModel, CrossAttentionStockModel
from model import StockMixer  # reuse the original model for comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train improved stock forecasting models")
    parser.add_argument("--market", type=str, default="NASDAQ",
                        help="Name of the market (e.g. NASDAQ, NYSE, SP500)")
    parser.add_argument("--model", type=str, choices=["stock_mixer", "gated_mlp", "cross_attn"],
                        default="gated_mlp",
                        help="Model architecture to use")
    parser.add_argument("--lookback", type=int, default=16,
                        help="Length of the lookback window")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Weight for the ranking loss")
    parser.add_argument("--market_dim", type=int, default=20,
                        help="Hidden dimension for market (StockMixer only)")
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="Hidden dimension for improved models")
    parser.add_argument("--heads", type=int, default=4,
                        help="Number of attention heads for cross attention model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # fix random seeds for reproducibility
    np.random.seed(123456789)
    torch.random.manual_seed(12345678)

    # determine computing device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load preprocessed data
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(base_dir, "dataset", args.market)

    with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
        eod_data = pickle.load(f)
    with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
        mask_data = pickle.load(f)
    with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
        gt_data = pickle.load(f)
    with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
        price_data = pickle.load(f)

    stock_num, trade_dates, fea_num = eod_data.shape
    lookback_length = args.lookback
    steps = 1  # one day horizon
    valid_index = int(trade_dates * 0.5)
    test_index = int(trade_dates * 0.75)

    valid_index = 756
    test_index = 1008
    fea_num = 5
    market_num = 20

    # build model
    if args.model == "stock_mixer":
        model = StockMixer(stocks=stock_num,
                           time_steps=lookback_length,
                           channels=fea_num,
                           market=args.market_dim,
                           scale=3)
    elif args.model == "gated_mlp":
        model = GatedMLPStockModel(stocks=stock_num,
                                   time_steps=lookback_length,
                                   channels=fea_num,
                                   hidden_dim=args.hidden_dim,
                                   use_cross_attn=False)
    elif args.model == "cross_attn":
        model = GatedMLPStockModel(stocks=stock_num,
                                   time_steps=lookback_length,
                                   channels=fea_num,
                                   hidden_dim=args.hidden_dim,
                                   use_cross_attn=True,
                                   num_heads=args.heads)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss = float("inf")
    best_val_perf = None
    best_test_perf = None

    # precompute batch offsets for shuffling
    batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)

    def get_batch(offset: int):
        # gather a slice of the EOD tensor from offset to offset+lookback_length
        mask_batch = mask_data[:, offset: offset + lookback_length + steps]
        # use min mask across the lookback to determine valid stocks
        mask_batch = np.min(mask_batch, axis=1)
        return (
            eod_data[:, offset:offset + lookback_length, :],
            np.expand_dims(mask_batch, axis=1),
            np.expand_dims(price_data[:, offset + lookback_length - 1], axis=1),
            np.expand_dims(gt_data[:, offset + lookback_length + steps - 1], axis=1)
        )

    def validate(start_index: int, end_index: int):
        with torch.no_grad():
            cur_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
            cur_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
            cur_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
            total_loss = reg_loss_total = rank_loss_total = 0.0
            for cur_offset in range(start_index - lookback_length - steps + 1,
                                    end_index - lookback_length - steps + 1):
                data_batch, mask_batch, price_batch, gt_batch = map(
                    lambda x: torch.Tensor(x).to(device), get_batch(cur_offset)
                )
                preds = model(data_batch)
                loss, reg_loss, rank_loss, rr = get_loss(preds, gt_batch, price_batch,
                                                         mask_batch, stock_num, args.alpha)
                total_loss += loss.item()
                reg_loss_total += reg_loss.item()
                rank_loss_total += rank_loss.item()
                idx = cur_offset - (start_index - lookback_length - steps + 1)
                cur_pred[:, idx] = rr[:, 0].cpu().numpy()
                cur_gt[:, idx] = gt_batch[:, 0].cpu().numpy()
                cur_mask[:, idx] = mask_batch[:, 0].cpu().numpy()
            total_loss /= (end_index - start_index)
            reg_loss_total /= (end_index - start_index)
            rank_loss_total /= (end_index - start_index)
            perf = evaluate(cur_pred, cur_gt, cur_mask)
        return total_loss, reg_loss_total, rank_loss_total, perf

    for epoch in range(args.epochs):
        np.random.shuffle(batch_offsets)
        train_loss = train_reg_loss = train_rank_loss = 0.0
        for j in range(valid_index - lookback_length - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device), get_batch(batch_offsets[j])
            )
            optimizer.zero_grad()
            preds = model(data_batch)
            loss, reg_loss, rank_loss, _ = get_loss(preds, gt_batch, price_batch,
                                                     mask_batch, stock_num, args.alpha)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_reg_loss += reg_loss.item()
            train_rank_loss += rank_loss.item()
        train_loss /= (valid_index - lookback_length - steps + 1)
        train_reg_loss /= (valid_index - lookback_length - steps + 1)
        train_rank_loss /= (valid_index - lookback_length - steps + 1)
        print(f"Epoch {epoch+1:03d}: Train loss {train_loss:.4e} = {train_reg_loss:.4e} + alpha*{train_rank_loss:.4e}")
        # validate on held-out set
        val_loss, val_reg, val_rank, val_perf = validate(valid_index, test_index)
        print(f"          Val   loss {val_loss:.4e} = {val_reg:.4e} + alpha*{val_rank:.4e}")
        test_loss, test_reg, test_rank, test_perf = validate(test_index, trade_dates)
        print(f"          Test  loss {test_loss:.4e} = {test_reg:.4e} + alpha*{test_rank:.4e}")
        # track the best performing epoch on validation set
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_perf = val_perf
            best_test_perf = test_perf
        # report
        print("          Val   performance:")
        print('            mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
            val_perf['mse'], val_perf['IC'], val_perf['RIC'], val_perf['prec_10'], val_perf['sharpe5']))
        print("          Test  performance:")
        print('            mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
            test_perf['mse'], test_perf['IC'], test_perf['RIC'], test_perf['prec_10'], test_perf['sharpe5']))
        
        
        print("Best val performance:")
        print('mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
        best_val_perf['mse'],
        best_val_perf['IC'],
        best_val_perf['RIC'],
        best_val_perf['prec_10'],
        best_val_perf['sharpe5']))

        print("Best test performance (at best val):")
        print('mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
        best_test_perf['mse'],
        best_test_perf['IC'],
        best_test_perf['RIC'],
        best_test_perf['prec_10'],
        best_test_perf['sharpe5']))
        
    # final summary
    print("Best validation performance:")
    print('mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
        best_val_perf['mse'], best_val_perf['IC'], best_val_perf['RIC'], best_val_perf['prec_10'], best_val_perf['sharpe5']))
    print("Corresponding test performance:")
    print('mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
        best_test_perf['mse'], best_test_perf['IC'], best_test_perf['RIC'], best_test_perf['prec_10'], best_test_perf['sharpe5']))


if __name__ == '__main__':
    main()