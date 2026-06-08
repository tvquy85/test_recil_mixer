import os
import pickle
import random
import sys

import numpy as np
import torch as torch

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Original.evaluator import evaluate
from Original.model import get_loss  # reuse the original loss function
from model_gatedMLP import StockMixerGatedNoContext  # our new model

"""
train_gatedMLP.py
===================

This training script provides a drop‑in replacement for the original
``train.py`` that trains the ``StockMixerGatedNoContext`` model.  It
mirrors the original data loading, batching, loss computation and
evaluation logic to ensure a fair comparison.  The only difference is
that the stock mixing stage now uses a gated MLP (gMLP) without
external context.  You can run this script in the same way as the
original training script; for example:

```
python train_gatedMLP.py
```

Optional command‑line arguments allow you to customise the gMLP depth
and dropout rate:

```
python train_gatedMLP.py --depth 3 --dropout 0.1
```

All other hyperparameters (lookback length, market dimension, etc.)
remain as in ``train.py``.  The dataset must reside in the same
location (``../dataset/{market}``) with the same pickle files.
"""


def parse_args(args):
    """Parse command‑line arguments to override gMLP hyperparameters.

    Only ``--depth`` and ``--dropout`` are supported.  All other
    parameters are inherited from the original training script.
    """
    depth = 1
    dropout = 0.0
    i = 0
    while i < len(args):
        if args[i] == '--depth' and i + 1 < len(args):
            depth = int(args[i + 1])
            i += 2
        elif args[i] == '--dropout' and i + 1 < len(args):
            dropout = float(args[i + 1])
            i += 2
        else:
            i += 1
    return depth, dropout


def main():
    # Parse optional depth and dropout arguments
    depth, dropout = parse_args(sys.argv[1:])

    # Set seeds for reproducibility
    np.random.seed(123456789)
    torch.random.manual_seed(12345678)

    # Select device (GPU if available)
    # Debug: Kiểm tra CUDA availability và chọn device
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

    # Hyperparameters (match those in train.py)
    data_path = '../dataset'
    market_name = 'NASDAQ'
    relation_name = 'wikidata'
    stock_num = 1026
    lookback_length = 16
    epochs = 100
    valid_index = 756
    test_index = 1008
    fea_num = 5
    market_num = 20  # hidden dimension for gMLP (latent market factors)
    steps = 1
    learning_rate = 0.001
    alpha = 0.1
    scale_factor = 3  # unused here but kept for consistency

    # Derive dataset directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(BASE_DIR, "dataset", market_name)

    # Load dataset from pickle files (as in train.py)
    if market_name == "SP500":
        data = np.load(os.path.join(data_path, 'SP500', 'SP500.npy'))
        data = data[:, 915:, :]
        price_data = data[:, :, -1]
        mask_data = np.ones((data.shape[0], data.shape[1]))
        eod_data = data
        gt_data = np.zeros((data.shape[0], data.shape[1]))
        for ticket in range(data.shape[0]):
            for row in range(1, data.shape[1]):
                prev = data[ticket][row - steps][-1]
                cur = data[ticket][row][-1]
                gt_data[ticket][row] = (cur - prev) / (prev + 1e-8)
    else:
        with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
            eod_data = pickle.load(f)
        with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
            mask_data = pickle.load(f)
        with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
            gt_data = pickle.load(f)
        with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
            price_data = pickle.load(f)

    # Infer dimensions
    fea_num = eod_data.shape[2]
    trade_dates = mask_data.shape[1]

    # Instantiate the gated StockMixer model
    model = StockMixerGatedNoContext(
        stocks=stock_num,
        time_steps=lookback_length,
        channels=fea_num,
        market=market_num,
        scale=scale_factor,
        depth=depth,
        dropout=dropout
    ).to(device)

    # Optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Track best results
    best_valid_loss = np.inf
    best_valid_perf = None
    best_test_perf = None

    # Precompute batch offsets for shuffling
    batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)

    def get_batch(offset=None):
        """Generate a training batch aligned to ``offset``.

        Returns a tuple of (data, mask, price, ground_truth), each
        shaped appropriately for input to the model and loss
        function.  The mask is computed over the lookback window and
        prediction horizon and then reduced to a single indicator per
        stock.
        """
        if offset is None:
            offset = random.randrange(0, valid_index)
        seq_len = lookback_length
        # Compute mask over lookback + prediction horizon
        mask_batch = mask_data[:, offset: offset + seq_len + steps]
        mask_batch = np.min(mask_batch, axis=1)
        return (
            eod_data[:, offset: offset + seq_len, :],
            np.expand_dims(mask_batch, axis=1),
            np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
            np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1),
        )

    def validate(start_index, end_index):
        """Evaluate the model on a contiguous slice of the dataset."""
        with torch.no_grad():
            length = end_index - start_index
            cur_pred = np.zeros((stock_num, length), dtype=float)
            cur_gt = np.zeros((stock_num, length), dtype=float)
            cur_mask = np.zeros((stock_num, length), dtype=float)
            loss_acc = reg_acc = rank_acc = 0.0
            for cur_offset in range(start_index - lookback_length - steps + 1,
                                   end_index - lookback_length - steps + 1):
                data_batch, mask_batch, price_batch, gt_batch = map(
                    lambda x: torch.tensor(x, dtype=torch.float32, device=device),
                    get_batch(cur_offset)
                )
                pred = model(data_batch)
                cur_loss, cur_reg, cur_rank, cur_rr = get_loss(
                    pred, gt_batch, price_batch, mask_batch, stock_num, alpha
                )
                loss_acc += cur_loss.item()
                reg_acc += cur_reg.item()
                rank_acc += cur_rank.item()
                idx = cur_offset - (start_index - lookback_length - steps + 1)
                cur_pred[:, idx] = cur_rr[:, 0].detach().cpu().numpy()
                cur_gt[:, idx] = gt_batch[:, 0].detach().cpu().numpy()
                cur_mask[:, idx] = mask_batch[:, 0].detach().cpu().numpy()
            length = end_index - start_index
            loss_acc /= length
            reg_acc /= length
            rank_acc /= length
            perf = evaluate(cur_pred, cur_gt, cur_mask)
        return loss_acc, reg_acc, rank_acc, perf

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        np.random.shuffle(batch_offsets)
        train_loss = train_reg = train_rank = 0.0
        # Number of training batches per epoch
        num_batches = valid_index - lookback_length - steps + 1
        for j in range(num_batches):
            offset = batch_offsets[j]
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.tensor(x, dtype=torch.float32, device=device),
                get_batch(offset)
            )
            optimizer.zero_grad()
            pred = model(data_batch)
            loss, reg_loss, rank_loss, _ = get_loss(
                pred, gt_batch, price_batch, mask_batch, stock_num, alpha
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_reg += reg_loss.item()
            train_rank += rank_loss.item()
        train_loss /= num_batches
        train_reg /= num_batches
        train_rank /= num_batches
        # Validation and test evaluation
        val_loss, val_reg, val_rank, val_perf = validate(valid_index, test_index)
        test_loss, test_reg, test_rank, test_perf = validate(test_index, trade_dates)
        # Print training and validation summary
        print(f"Train : loss:{train_loss:.2e}  =  {train_reg:.2e} + alpha*{train_rank:.2e}")
        print(f"Valid : loss:{val_loss:.2e}  =  {val_reg:.2e} + alpha*{val_rank:.2e}")
        print(f"Test  : loss:{test_loss:.2e}  =  {test_reg:.2e} + alpha*{test_rank:.2e}")
        print(f"Valid performance:\n"
              f"  mse:{val_perf['mse']:.2e}, IC:{val_perf['IC']:.2e}, RIC:{val_perf['RIC']:.2e}, "
              f"prec@10:{val_perf['prec_10']:.2e}, SR:{val_perf['sharpe5']:.2e}")
        print(f"Test performance:\n"
              f"  mse:{test_perf['mse']:.2e}, IC:{test_perf['IC']:.2e}, RIC:{test_perf['RIC']:.2e}, "
              f"prec@10:{test_perf['prec_10']:.2e}, SR:{test_perf['sharpe5']:.2e}\n")
        # Update best validation loss and record performances
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_valid_perf = val_perf
            best_test_perf = test_perf
        # Print current best metrics
        if best_valid_perf is not None and best_test_perf is not None:
            print("Best validation performance so far:")
            print(
                f"  mse:{best_valid_perf['mse']:.2e}, IC:{best_valid_perf['IC']:.2e}, "
                f"RIC:{best_valid_perf['RIC']:.2e}, prec@10:{best_valid_perf['prec_10']:.2e}, "
                f"SR:{best_valid_perf['sharpe5']:.2e}"
            )
            print("Best test performance at this point:")
            print(
                f"  mse:{best_test_perf['mse']:.2e}, IC:{best_test_perf['IC']:.2e}, "
                f"RIC:{best_test_perf['RIC']:.2e}, prec@10:{best_test_perf['prec_10']:.2e}, "
                f"SR:{best_test_perf['sharpe5']:.2e}\n"
            )


if __name__ == '__main__':
    main()