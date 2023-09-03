import numpy as np
import torch
import vectorbtpro as vbt


def backtest_portfolio(pred, close_values, freq, pf_params={}):
    pf = vbt.Portfolio.from_signals(
        close_values,
        pred == 1,
        pred == 2,
        freq=freq,
        init_cash=10000,
        fees=0.0006,
        slippage=0.001,
        log=True,
        **pf_params,
    )
    return pf.total_return


def backtest_portfolio_sortino(pred, close_values, freq, pf_params={}):
    pf = vbt.Portfolio.from_signals(
        close_values,
        pred == 1,
        pred == 2,
        freq=freq,
        init_cash=10000,
        fees=0.0006,
        slippage=0.001,
        log=True,
        **pf_params,
    )
    return pf.sortino_ratio


def find_optimal_thresholds(logits, df, pf_params={}):
    # Compute the value of `freq`
    time_diff = df.index[1] - df.index[0]
    freq = f'{time_diff.seconds // 60}T'

    # Ensure that df and logits have the same length
    df = df[-len(logits) :].copy()

    def calculate_metrics(logits, thresholds):
        # Apply the thresholds
        preds = torch.zeros_like(logits)
        for i, threshold in enumerate(thresholds):
            preds[:, i] = (logits[:, i] > threshold).float()

        # Convert predictions to class labels
        pred_labels = torch.argmax(preds, dim=1)

        # Calculate the backtest return percentage
        return backtest_portfolio(pred_labels.cpu().numpy(), df['close'], freq, pf_params)

    # Move tensors to the same device as logits
    logits.device

    # Initialize thresholds
    thresholds = [0.0] * logits.shape[1]

    # We won't compute the threshold for the majority class (class index 0)
    for i in range(1, logits.shape[1]):
        best_threshold = 0.0
        best_metric_value = 0.0
        for threshold in np.arange(0.0, 1.0, 0.001):
            thresholds[i] = threshold
            current_metrics = calculate_metrics(logits, thresholds)
            if current_metrics > best_metric_value:
                best_metric_value = current_metrics
                best_threshold = threshold
        thresholds[i] = best_threshold

    return thresholds
