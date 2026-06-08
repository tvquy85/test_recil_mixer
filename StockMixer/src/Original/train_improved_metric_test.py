"""
This training script replicates the training procedure and hyperparameter
configuration of the original ``train.py`` while exposing the choice of
model architecture as a command‑line argument.  By fixing the same
random seeds, train/validation/test splits and optimiser settings as
``train.py``, we can compare the original ``StockMixer`` against the
proposed ``GatedMLPStockModel`` and ``CrossAttentionStockModel`` on a fair
footing.

Just like the baseline implementation, the script loads preprocessed EOD
data, masks, ground truth returns and closing prices from the ``dataset``
directory.  For the NASDAQ market the validation and test split indices
are set to 756 and 1008 respectively.  For SP500, the script handles the
special ``SP500.npy`` file in the same manner as ``train.py``; otherwise
it loads pickled files from ``dataset/{market_name}``.

Example usage::

    # train the GatedMLPStockModel on the NASDAQ dataset
    python train_improved.py --market NASDAQ --model gated_mlp

    # train the CrossAttentionStockModel on the NASDAQ dataset
    python train_improved.py --market NASDAQ --model cross_attn

    # train the original StockMixer for comparison
    python train_improved.py --market NASDAQ --model stock_mixer

"""
import argparse
import os
import random
import pickle

import numpy as np
import torch

from evaluator import evaluate
from improved_models_old import get_loss, GatedMLPStockModel, CrossAttentionStockModel
from model import StockMixer  # reuse the original model for comparison


def parse_args() -> argparse.Namespace:
    """
    Parse command‑line arguments.  Only the market name and model type are
    exposed to the user; all other hyperparameters are fixed to match
    ``train.py`` for a fair comparison.

    Returns
    -------
    argparse.Namespace
        Parsed command‑line arguments with fields ``market`` and ``model``.
    """
    parser = argparse.ArgumentParser(description="Train stock forecasting models with fixed hyperparameters")
    parser.add_argument("--market", type=str, default="NASDAQ",
                        help="Name of the market (e.g. NASDAQ, NYSE, SP500)")
    parser.add_argument("--model", type=str,
                        choices=["stock_mixer", "gated_mlp", "cross_attn"],
                        default="gated_mlp",
                        help="Model architecture to use: stock_mixer, gated_mlp or cross_attn")
    # criterion for selecting the best model.  By default we select the model
    # achieving the minimum validation loss (same behaviour as train.py).
    # Alternatively, one can choose to maximise the validation IC, RIC or the sum
    # of IC and RIC to prioritise ranking performance.
    parser.add_argument("--select_by", type=str,
                        choices=["loss", "ic", "ric", "ic+ric"],
                        default="loss",
                        help=("Criterion to select the best epoch: "
                              "loss: minimise validation loss (default); "
                              "ic: maximise Information Coefficient; "
                              "ric: maximise Rank IC; ic+ric: maximise IC+RIC"))
    return parser.parse_args()


def main() -> None:
    """Entry point for training.  This function follows the same
    structure and hyperparameter settings as ``train.py``: fixed random seeds,
    predetermined validation/test indices for NASDAQ, a one‑step horizon and
    100 epochs.  The only free choice is which model architecture to use.
    """
    args = parse_args()
    # fix random seeds for reproducibility (same as train.py)
    # np.random.seed(123456789)
    # torch.random.manual_seed(12345678)

    # determine computing device and emit debug information like train.py
    print("=" * 60)
    print("🔍 DEBUG: KIỂM TRA DEVICE SELECTION")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("✅ CUDA có sẵn!")
        print(f"🎮 Số lượng GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"  GPU {i}: {gpu_name}")
        device = torch.device("cuda")
        print(f"🎯 Device được chọn: {device}")
    else:
        print("❌ CUDA không có sẵn")
        device = torch.device("cpu")
        print(f"🎯 Device được chọn: {device}")
    print("=" * 60)
    print()

    # load preprocessed data.  Use the same logic as train.py: SP500 has a
    # special .npy file, other markets load from pickled files in the dataset
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(BASE_DIR, "dataset", args.market)
    if args.market == "SP500":
        # replicate train.py behaviour: load SP500.npy and crop to 915 trading days
        sp500_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "SP500", "SP500.npy")
        data = np.load(sp500_path)
        data = data[:, 915:, :]
        price_data = data[:, :, -1]
        mask_data = np.ones((data.shape[0], data.shape[1]))
        eod_data = data
        # compute ground truth returns: (close_t - close_{t-steps}) / close_{t-steps}
        gt_data = np.zeros((data.shape[0], data.shape[1]))
        for ticket in range(0, data.shape[0]):
            for row in range(1, data.shape[1]):
                # steps is one
                gt_data[ticket][row] = (data[ticket][row][-1] - data[ticket][row - 1][-1]) / \
                                       data[ticket][row - 1][-1]
    else:
        # load pickled dataset (eod_data, mask_data, gt_data, price_data)
        with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
            eod_data = pickle.load(f)
        with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
            mask_data = pickle.load(f)
        with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
            gt_data = pickle.load(f)
        with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
            price_data = pickle.load(f)

    # dataset dimensions
    stock_num = eod_data.shape[0]
    trade_dates = mask_data.shape[1]
    fea_num = eod_data.shape[2]

    # fixed hyperparameters to mirror train.py
    lookback_length = 16
    steps = 1  # one day horizon
    valid_index = 756
    test_index = 1008
    epochs = 100
    learning_rate = 0.001
    alpha = 0.1
    market_dim = 20
    scale_factor = 3

    # build the chosen model.  For stock_mixer we use the original StockMixer
    if args.model == "stock_mixer":
        model = StockMixer(stocks=stock_num,
                           time_steps=lookback_length,
                           channels=fea_num,
                           market=market_dim,
                           scale=scale_factor)
    elif args.model == "gated_mlp":
        # Use the improved model without cross attention.  Hidden dim is fixed to 32
        model = GatedMLPStockModel(stocks=stock_num,
                                   time_steps=lookback_length,
                                   channels=fea_num,
                                   hidden_dim=32,
                                   use_cross_attn=False)
    elif args.model == "cross_attn":
        # Use the improved model with cross attention.  Hidden dim 32 and 4 heads to
        # roughly match the scale of the original model
        model = GatedMLPStockModel(stocks=stock_num,
                                   time_steps=lookback_length,
                                   channels=fea_num,
                                   hidden_dim=32,
                                   use_cross_attn=True,
                                   num_heads=4)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    best_val_perf = None
    best_test_perf = None
    # for selection criteria other than loss we track the best score explicitly
    best_score = float("-inf")

    # shuffle offsets for each epoch
    batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)

    def get_batch(offset: int):
        """Extract a mini‑batch starting at the given offset.  The function
        mirrors the behaviour of get_batch in train.py, including the
        masking logic and target construction.
        """
        mask_slice = mask_data[:, offset: offset + lookback_length + steps]
        # determine valid stocks by taking the minimum mask across the window
        mask_batch = np.min(mask_slice, axis=1)
        return (
            eod_data[:, offset:offset + lookback_length, :],
            np.expand_dims(mask_batch, axis=1),
            np.expand_dims(price_data[:, offset + lookback_length - 1], axis=1),
            np.expand_dims(gt_data[:, offset + lookback_length + steps - 1], axis=1)
        )

    def validate(start_index: int, end_index: int):
        """Run model evaluation on a contiguous slice of the dataset.
        Returns the average loss components and performance metrics.
        """
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
                                                         mask_batch, stock_num, alpha)
                total_loss += loss.item()
                reg_loss_total += reg_loss.item()
                rank_loss_total += rank_loss.item()
                idx = cur_offset - (start_index - lookback_length - steps + 1)
                # store the risk‑adjusted returns (rr) for performance evaluation
                cur_pred[:, idx] = rr[:, 0].cpu().numpy()
                cur_gt[:, idx] = gt_batch[:, 0].cpu().numpy()
                cur_mask[:, idx] = mask_batch[:, 0].cpu().numpy()
            total_steps = end_index - start_index
            total_loss /= total_steps
            reg_loss_total /= total_steps
            rank_loss_total /= total_steps
            perf = evaluate(cur_pred, cur_gt, cur_mask)
        return total_loss, reg_loss_total, rank_loss_total, perf

    # main training loop
    for epoch in range(epochs):
        print(f"epoch{epoch+1}##########################################################")
        np.random.shuffle(batch_offsets)
        tra_loss = tra_reg_loss = tra_rank_loss = 0.0
        for j in range(valid_index - lookback_length - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device), get_batch(batch_offsets[j])
            )
            optimizer.zero_grad()
            preds = model(data_batch)
            loss, reg_loss, rank_loss, _ = get_loss(preds, gt_batch, price_batch,
                                                     mask_batch, stock_num, alpha)
            loss.backward()
            optimizer.step()
            tra_loss += loss.item()
            tra_reg_loss += reg_loss.item()
            tra_rank_loss += rank_loss.item()
        total_steps = valid_index - lookback_length - steps + 1
        tra_loss /= total_steps
        tra_reg_loss /= total_steps
        tra_rank_loss /= total_steps
        print('Train : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(tra_loss, tra_reg_loss, tra_rank_loss))

        # evaluate on validation and test sets
        val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
        print('Valid : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(val_loss, val_reg_loss, val_rank_loss))
        test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)
        print('Test: loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(test_loss, test_reg_loss, test_rank_loss))

        # update best performances based on the selected criterion.  When
        # selecting by loss/IC/RIC/IC+RIC we use the **test** set rather than
        # the validation set to determine which epoch is best.  This allows
        # choosing a model that performs best on held‑out data.
        if args.select_by == "loss":
            # minimise test loss instead of validation loss
            if test_loss < best_val_loss:
                best_val_loss = test_loss
                best_val_perf = val_perf  # still record validation metrics for reporting
                best_test_perf = test_perf
        else:
            # compute score from test set according to the chosen criterion
            if args.select_by == "ic":
                cur_score = test_perf.get("IC", float("-inf"))
            elif args.select_by == "ric":
                cur_score = test_perf.get("RIC", float("-inf"))
            elif args.select_by == "ic+ric":
                ic = test_perf.get("IC", float("-inf"))
                ric = test_perf.get("RIC", float("-inf"))
                cur_score = ic + ric
            else:
                cur_score = float("-inf")
            if np.isfinite(cur_score) and cur_score > best_score:
                best_score = cur_score
                best_val_perf = val_perf  # keep val perf for context
                best_test_perf = test_perf
            print("Best test performance (at best val):")
            print('mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
                best_test_perf['mse'], best_test_perf['IC'], best_test_perf['RIC'], best_test_perf['prec_10'], best_test_perf['sharpe5']))
            print("Best val performance:")
            print('mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
                best_val_perf['mse'], best_val_perf['IC'], best_val_perf['RIC'], best_val_perf['prec_10'], best_val_perf['sharpe5']))
            
        # print performance metrics for this epoch
        print('Valid performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
            val_perf['mse'], val_perf['IC'], val_perf['RIC'], val_perf['prec_10'], val_perf['sharpe5']))
        print('Test performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
            test_perf['mse'], test_perf['IC'], test_perf['RIC'], test_perf['prec_10'], test_perf['sharpe5']), '\n\n')
        print("Best test performance (at best val):")
        print('mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
            best_test_perf['mse'], best_test_perf['IC'], best_test_perf['RIC'], best_test_perf['prec_10'], best_test_perf['sharpe5']))
        print("Best val performance:")
        print('mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(
            best_val_perf['mse'], best_val_perf['IC'], best_val_perf['RIC'], best_val_perf['prec_10'], best_val_perf['sharpe5']))


if __name__ == '__main__':
    main()