from typing import List, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from keras.utils import to_categorical
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


def extract_garch_features(data_x_2d: pd.DataFrame, time_series, garch_columns):
    n_features = time_series.shape[1]
    volatilities = np.zeros((time_series.shape[0], len(garch_columns)))
    standardized_residuals = np.zeros((time_series.shape[0], len(garch_columns)))

    for idx, column_name in enumerate(garch_columns):
        column_index = data_x_2d.columns.get_loc(column_name)
        single_feature_series = time_series[:, column_index]
        model = arch_model(single_feature_series, vol='Garch', p=1, q=1)
        fitted_model = model.fit(disp='off')
        volatility = fitted_model.conditional_volatility
        std_resid = fitted_model.resid / fitted_model.conditional_volatility

        volatilities[:, idx] = volatility
        standardized_residuals[:, idx] = std_resid

    return volatilities, standardized_residuals


def preprocess_data(
    data_x_2d: pd.DataFrame, data_y_2d: pd.DataFrame, lookback: int, test_pct: float = 0.0, torch=False
) -> List[Tuple[np.ndarray, np.ndarray]]:
    sets = []

    if test_pct > 0:
        # # Split the data using time series information
        # tscv = TimeSeriesSplit(n_splits=int(1 / test_pct))
        # for train_index, test_index in tscv.split(data_x_2d.index):
        #     X_train, X_test = data_x_2d.iloc[train_index], data_x_2d.iloc[test_index]
        #     y_train, y_test = data_y_2d.iloc[train_index], data_y_2d.iloc[test_index]
        #     sets.append((X_train, y_train))

        X_train, y_train, X_test, y_test = stratified_split(data_x_2d, data_y_2d, test_pct)

        sets.extend(((X_train, y_train), (X_test, y_test)))
    else:
        sets.append((data_x_2d, data_y_2d))

    output_sets = []

    # Scale the data. Each set needs to be fit/transformed separately. Don't scale the target (obviously).
    for data_x_2d, data_y_2d in sets:
        # Scale X
        data_x_2d_scaled = scale_data(data_x_2d)

        # Extract GARCH features
        # volatility, standardized_residuals = extract_garch_features(
        #     data_x_2d, data_x_2d_scaled, ['open', 'high', 'low', 'close', 'volume']
        # )
        # garch_features = np.stack((volatility, standardized_residuals), axis=-1)
        # garch_features = garch_features.reshape(data_x_2d_scaled.shape[0], -1)

        # Concatenate GARCH features to the scaled data
        # data_x_2d_scaled_garch = np.concatenate((data_x_2d_scaled, garch_features), axis=1)

        # Reshape X
        data_x_3d = _2d_to_3d(data_x_2d_scaled, lookback)

        # Squeeze y to 1 dimension if necessary
        data_y_1d = np.squeeze(data_y_2d)

        if torch:
            output_sets.append((data_x_3d, data_y_1d[lookback - 1 :]))
        else:
            output_sets.append((data_x_3d, one_hot_encode(data_y_1d)[lookback - 1 :]))

    # Return the sets
    if len(output_sets) == 1:
        return output_sets[0]
    else:
        return (*output_sets[0], *output_sets[1])


def preprocess_data_2d(X: np.ndarray, y: np.ndarray, test_pct: float = 0.2):
    X_train, y_train, X_test, y_test = stratified_split(X, y, test_pct)
    X_train_scaled = scale_data(X_train)
    X_test_scaled = scale_data(X_test)
    return X_train_scaled, y_train, X_test_scaled, y_test


def stratified_split(X, y, test_pct):
    num_test = int(test_pct * len(y))
    return X[:num_test], y[:num_test], X[num_test:], y[num_test:]


def scale_data(data_x_2d: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.nan_to_num(data_x_2d, nan=0, posinf=0, neginf=0))


def one_hot_encode(data_y_2d: np.ndarray) -> np.ndarray:
    return to_categorical(data_y_2d)


def _2d_to_3d(data_2d: np.ndarray, lookback: int) -> np.ndarray:
    n_samples, n_features = data_2d.shape
    n_windows = n_samples - lookback + 1
    data_3d = np.empty((n_windows, lookback, n_features))
    for i in range(n_windows):
        data_3d[i] = data_2d[i : i + lookback]
    return data_3d
