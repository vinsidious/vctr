import torch


def positional_accuracy(y_pred, y_true):
    # Check inputs
    assert isinstance(y_pred, torch.Tensor), 'y_pred must be a PyTorch tensor'
    assert isinstance(y_true, torch.Tensor), 'y_true must be a PyTorch tensor'
    assert y_pred.shape == y_true.shape, 'y_pred and y_true must have the same shape'

    # Compute number of true positives and false negatives
    tp = (y_true == y_pred).sum().item()
    fn = y_true.ne(y_pred).sum().item()

    # Compute the positional accuracy metric
    pa = tp / (tp + fn)

    return pa
