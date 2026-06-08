import torch

from src.recil.losses import masked_mse_loss, pairwise_rank_loss, recil_loss


def test_masked_mse_loss_toy_expected_value():
    pred = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
    target = torch.tensor([1.5, 1.0, 0.0])
    mask = torch.tensor([1.0, 1.0, 0.0])
    loss = masked_mse_loss(pred, target, mask)
    assert loss == torch.tensor((0.25 + 1.0) / 2.0)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad[2] == 0.0


def test_masked_mse_all_invalid_is_graph_connected_zero():
    pred = torch.tensor([1.0, 2.0], requires_grad=True)
    target = torch.tensor([3.0, 4.0])
    mask = torch.zeros(2)
    loss = masked_mse_loss(pred, target, mask)
    assert loss.item() == 0.0
    loss.backward()
    assert pred.grad is not None
    assert torch.all(pred.grad == 0.0)


def test_losses_accept_vector_and_batch_shapes():
    pred_vec = torch.tensor([1.0, 2.0, 3.0])
    target_vec = torch.tensor([1.0, 1.5, 2.5])
    mask_vec = torch.ones(3)
    pred_batch = pred_vec.unsqueeze(0)
    target_batch = target_vec.unsqueeze(0)
    mask_batch = mask_vec.unsqueeze(0)
    assert torch.allclose(masked_mse_loss(pred_vec, target_vec, mask_vec), masked_mse_loss(pred_batch, target_batch, mask_batch))
    assert torch.allclose(pairwise_rank_loss(pred_vec, target_vec, mask_vec), pairwise_rank_loss(pred_batch, target_batch, mask_batch))


def test_correct_ranking_has_lower_pairwise_loss_than_reversed():
    target = torch.tensor([0.3, 0.2, 0.1])
    mask = torch.ones(3)
    correct = torch.tensor([3.0, 2.0, 1.0])
    reversed_pred = torch.tensor([1.0, 2.0, 3.0])
    assert pairwise_rank_loss(correct, target, mask) < pairwise_rank_loss(reversed_pred, target, mask)


def test_one_valid_asset_and_all_ties_return_zero_rank_loss():
    one_valid = pairwise_rank_loss(torch.tensor([1.0, 2.0]), torch.tensor([0.1, 0.2]), torch.tensor([1.0, 0.0]))
    all_ties = pairwise_rank_loss(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.1, 0.1, 0.1]), torch.ones(3))
    assert one_valid.item() == 0.0
    assert all_ties.item() == 0.0
    assert torch.isfinite(one_valid)
    assert torch.isfinite(all_ties)


def test_masked_outlier_does_not_affect_rank_loss():
    pred = torch.tensor([1000.0, 3.0, 2.0, 1.0])
    target = torch.tensor([-1000.0, 0.3, 0.2, 0.1])
    mask = torch.tensor([0.0, 1.0, 1.0, 1.0])
    expected = pairwise_rank_loss(pred[1:], target[1:], mask[1:])
    assert torch.allclose(pairwise_rank_loss(pred, target, mask), expected)


def test_pairwise_rank_loss_deterministic_subsampling_is_stable():
    pred = torch.linspace(-1.0, 1.0, steps=12)
    target = torch.linspace(1.0, -1.0, steps=12)
    mask = torch.ones(12)
    first = pairwise_rank_loss(pred, target, mask, max_pairs_per_day=7)
    second = pairwise_rank_loss(pred, target, mask, max_pairs_per_day=7)
    assert torch.isfinite(first)
    assert torch.allclose(first, second)


def test_recil_loss_combines_mse_and_rank_terms():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 1.0, 2.0])
    mask = torch.ones(3)
    expected = masked_mse_loss(pred, target, mask) + 0.25 * pairwise_rank_loss(pred, target, mask)
    assert torch.allclose(recil_loss(pred, target, mask, alpha_rank=0.25), expected)


def test_recil_loss_entropy_term_reduces_total():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 1.0, 2.0])
    mask = torch.ones(3)
    aux = {"router_weights": torch.tensor([[0.5, 0.5], [0.8, 0.2]])}
    base = recil_loss(pred, target, mask, aux=aux, alpha_rank=0.1, lambda_entropy=0.0)
    with_entropy = recil_loss(pred, target, mask, aux=aux, alpha_rank=0.1, lambda_entropy=0.2)
    assert with_entropy < base
