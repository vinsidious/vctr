import numpy as np
import pandas as pd
import vectorbtpro as vbt


def bolster_extrema_old(prices: pd.Series, extrema: pd.Series, n_pct: float) -> pd.Series:
    extrema_copy = extrema.copy()

    price_values = prices.values
    price_index = prices.index.values

    for i, value in extrema.iteritems():
        if value == 0:
            continue

        timestamp = extrema.index[i]
        price = price_values[i]
        lower_bound = price * (1 - n_pct)
        upper_bound = price * (1 + n_pct)

        left_index = np.searchsorted(price_index, timestamp, side='left') - 1
        while left_index >= 0:
            if price_values[left_index] < lower_bound:
                break
            if price_values[left_index] <= upper_bound:
                extrema_copy[price_index[left_index]] = value
            left_index -= 1

        right_index = np.searchsorted(price_index, timestamp, side='right')
        while right_index < len(price_values):
            if price_values[right_index] > upper_bound:
                break
            if price_values[right_index] >= lower_bound:
                extrema_copy[price_index[right_index]] = value
            right_index += 1

    return extrema_copy


import numpy as np


def bolster_extrema(prices, extrema, wiggle_pct):
    if len(prices) != len(extrema):
        raise ValueError('Input arrays must have the same length')

    prices = np.array(prices)
    extrema = np.array(extrema)

    local_min, local_max = 1, 2
    tolerance = prices * wiggle_pct

    lower_bounds = prices - tolerance
    upper_bounds = prices + tolerance

    local_min_indices = np.where(extrema == local_min)[0]
    local_max_indices = np.where(extrema == local_max)[0]

    for i in local_min_indices:
        left = max(i - 1, 0)
        right = min(i + 1, len(prices) - 1)
        if lower_bounds[i] <= prices[left] <= upper_bounds[i]:
            extrema[left] = local_min
        if lower_bounds[i] <= prices[right] <= upper_bounds[i]:
            extrema[right] = local_min

    for i in local_max_indices:
        left = max(i - 1, 0)
        right = min(i + 1, len(prices) - 1)
        if lower_bounds[i] <= prices[left] <= upper_bounds[i]:
            extrema[left] = local_max
        if lower_bounds[i] <= prices[right] <= upper_bounds[i]:
            extrema[right] = local_max

    return extrema


def label_extrema(close: pd.Series, up_th: float, down_th: float) -> pd.Series:
    """
    Calculate labels for the given close prices using the LEXLB method, based on provided up and down thresholds.

    Args:
        close (pd.Series): Close price series.
        up_th (float): Up threshold.
        down_th (float): Down threshold.

    Returns:
        pd.Series: Labeled close price series.
    """
    return vbt.LEXLB.run(close, up_th, down_th)


def label_trends(close: pd.Series, up_th: float, down_th: float) -> pd.Series:
    """
    Calculate labels for the given close prices using the TRENDLB method in Binary mode,
    based on provided up and down thresholds.

    Args:
        close (pd.Series): Close price series.
        up_th (float): Up threshold.
        down_th (float): Down threshold.

    Returns:
        pd.Series: Labeled close price series.
    """
    return vbt.TRENDLB.run(close, up_th, down_th, vbt.labels.enums.TrendLabelMode.Binary)


def label_data_extrema_multi(df: pd.DataFrame, th=0.05, wiggle=0.015) -> pd.DataFrame:
    """
    Add labels to the input DataFrame using the label_extrema function with the specified up and down thresholds.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'close' column with close price series.

    Returns:
        pd.DataFrame: DataFrame with an added 'labels' column.
    """
    labels = label_extrema(df.close, th, th).labels.replace({-1: 1, 1: 2})
    labels = bolster_extrema(df.close, labels, wiggle)
    df['label'] = labels
    df['label'] = df['label'].shift(-1).fillna(0).astype(int)
    df = df.dropna()
    return df


def label_data_trends_binary(df: pd.DataFrame, th=0.01) -> pd.DataFrame:
    """
    Add labels to the input DataFrame using the label_trends function with the specified up and down thresholds.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'close' column with close price series.

    Returns:
        pd.DataFrame: DataFrame with an added 'labels' column.
    """
    df['label'] = label_trends(df.close, th, th).labels
    df.dropna(inplace=True)
    return df
