import contextlib

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, roc_curve
from sklearn.preprocessing import StandardScaler


def to_numpy(array):
    return np.array(array) if isinstance(array, (list, np.ndarray)) else array.cpu().detach().numpy()


# Calculate the softmax function for a numpy array.
def softmax(arr):
    """
    Compute softmax values for each row of a 2D array.
    """
    # Subtract the maximum value from each value in the array to prevent
    # overflow errors
    arr -= np.max(arr, axis=1)[:, np.newaxis]

    # Apply the exponentiation element-wise to every element of the array
    exp_arr = np.exp(arr)

    # Calculate the denominator for the softmax function
    denominator = np.sum(exp_arr, axis=1)[:, np.newaxis]

    # Divide each element of the exponentiated array by the denominator to
    # obtain the softmax output
    softmax_output = exp_arr / denominator

    return softmax_output


class RunningAverage:
    def __init__(self, initial_value=0.3, decay_factor=0.9):
        self.alpha = decay_factor
        self.ema = initial_value

    def add(self, value):
        if isinstance(value, (list, tuple)):
            for x in value:
                self.ema = self.alpha * self.ema + (1 - self.alpha) * x
        else:
            self.ema = self.alpha * self.ema + (1 - self.alpha) * value

    def get_average(self):
        return self.ema


class ThresholdManager:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.classes = list(range(num_classes))
        self.recent_predictions = []
        self.thresholds = [RunningAverage() for _ in range(num_classes)]
        self.scaler = StandardScaler()
        self.platt_scaler = SGDClassifier(loss='log_loss', max_iter=1000)
        self.has_fit_platt_scaler = False

    def logits_to_proba(self, logits):
        logits = to_numpy(logits)

        if not self.has_fit_platt_scaler:
            return softmax(logits)

        logits_scaled = self.scaler.transform(logits)
        return self.platt_scaler.predict_proba(logits_scaled)

    def platt_fit(self, logits, y_true):
        logits = to_numpy(logits)
        y_true = to_numpy(y_true)

        logits_scaled = self.scaler.fit_transform(logits)
        self.platt_scaler.partial_fit(logits_scaled, y_true, classes=self.classes)

        self.has_fit_platt_scaler = True

    def compute_optimal_thresholds(self, logits, y_true, num_thresholds=100):
        logits = to_numpy(logits)
        y_true = to_numpy(y_true)

        self.platt_fit(logits, y_true)

        y_probs = self.logits_to_proba(logits)
        optimal_thresholds = []

        for c in range(self.num_classes):
            best_threshold, best_f1 = 0, 0

            for threshold in np.linspace(0, 1, num_thresholds):
                y_preds = (y_probs[:, c] >= threshold).astype(int)
                y_true_c = (y_true == c).astype(int)
                f1 = f1_score(y_true_c, y_preds, average='macro', zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            optimal_thresholds.append(best_threshold)

        for i, thresh in enumerate(self.thresholds):
            thresh.add(optimal_thresholds[i])

        return optimal_thresholds

    def apply_thresholds(self, y_logits, thresholds=None):
        y_logits = to_numpy(y_logits)
        y_preds = np.zeros_like(y_logits, dtype=int)

        if thresholds is None:
            thresholds = [thresh.get_average() for thresh in self.thresholds]

        y_probs = self.logits_to_proba(y_logits)

        for c in range(len(thresholds)):
            y_preds[:, c] = (y_probs[:, c] >= thresholds[c]).astype(int)

        # New logic to handle multiple classes above their thresholds
        multiple_classes = y_preds.sum(axis=1) > 1
        for i in range(y_preds.shape[0]):
            if multiple_classes[i]:
                if y_preds[i, 1] and y_preds[i, 2]:
                    # If both classes 1 and 2 are above their thresholds, select the least recently predicted one
                    y_preds[i, :] = 0
                    least_recent_class = self.get_least_recent_class()
                    y_preds[i, least_recent_class] = 1
                    self.recent_predictions.append(least_recent_class)
                elif y_preds[i, 1]:
                    self.select_only_class_above_thresh(y_preds, i, 1)
                elif y_preds[i, 2]:
                    self.select_only_class_above_thresh(y_preds, i, 2)

        # Convert one-hot encoded predictions to class labels
        y_preds = np.argmax(y_preds, axis=1)

        return torch.tensor(y_preds, dtype=torch.int32).to('mps')

    def select_only_class_above_thresh(self, y_preds, i, arg2):
        # If only class 1 is above its threshold, select class 1
        y_preds[i, :] = 0
        y_preds[i, arg2] = 1
        self.recent_predictions.append(arg2)

    def get_least_recent_class(self):
        # Find the least recently predicted class from the recent_predictions list
        if not self.recent_predictions:
            return 1
        least_recent_class = None
        for cls in [1, 2]:
            if cls not in self.recent_predictions:
                least_recent_class = cls
                break
            if least_recent_class is None or self.recent_predictions[::-1].index(cls) > self.recent_predictions[
                ::-1
            ].index(least_recent_class):
                least_recent_class = cls
        return least_recent_class
