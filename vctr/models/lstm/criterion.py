import numpy as np
import torch
import torch.nn as nn
import torchmetrics as tm
from vctr.models.lstm.metrics import positional_accuracy


import numpy as np


class WeightedObjective(nn.Module):
    def __init__(self, class_weights, weight_loss=1.0, weight_acc=1.0, weight_f1=1.0):
        super(WeightedObjective, self).__init__()
        assert isinstance(class_weights, torch.Tensor), 'class_weights must be a PyTorch tensor'

        self.class_weights = class_weights.to('mps')
        self.weight_loss = torch.tensor(weight_loss, dtype=torch.float32).to('mps')
        self.weight_acc = torch.tensor(weight_acc, dtype=torch.float32).to('mps')
        self.weight_f1 = torch.tensor(weight_f1, dtype=torch.float32).to('mps')

    def forward(self, y_pred, y_true):
        y_labels = y_pred.argmax(axis=1)

        ypd = y_pred.clone().cpu().detach().numpy()
        ytd = y_true.clone().cpu().detach().numpy()
        cwd = self.class_weights.clone().cpu().detach().numpy()

        # Calculate the cross-entropy loss
        cross_entropy_loss = cross_entropy(ypd, ytd, cwd)

        accuracy = tm.Accuracy(task='multiclass', num_classes=3, average='macro').to('mps')(y_labels, y_true)
        f1 = tm.F1Score(task='multiclass', num_classes=3, average='macro').to('mps')(y_labels, y_true)

        # Calculate the weighted sum of the three scores
        return self.weight_loss * cross_entropy_loss - self.weight_acc * accuracy - self.weight_f1 * (1 - f1)


def cross_entropy(logits, targets, weights=None, label_smoothing=0.1):
    epsilon = 1e-8  # Small epsilon value to avoid taking the logarithm of 0 or 1
    num_classes = logits.shape[1]
    probs = softmax(logits)
    probs = np.clip(probs, epsilon, 1 - epsilon)
    targets = np.eye(num_classes)[targets]  # Convert integer labels to one-hot vectors
    smooth_targets = (1 - label_smoothing) * targets + label_smoothing / num_classes
    class_weights = weights[np.newaxis, :].T  # Transpose the weights array
    ce = -np.mean(class_weights * np.sum(smooth_targets * np.log(probs), axis=1))
    return ce


def softmax(logits):
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    exp_logits[np.isnan(exp_logits) | (exp_logits == 0)] = 1e-9
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
