import warnings
from enum import Enum
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
import sys

warnings.filterwarnings('ignore', module='scipy')


class CorrelationMeasure(Enum):
    PEARSON = 'pearson'
    SPEARMAN = 'spearman'
    KENDALL = 'kendall'


def parameter_search(
    data, indicator_func, param_dict, max_lag=5, decay=0.9, correlation_measure=CorrelationMeasure.PEARSON
):
    # Initialize the best score and best parameters
    best_score = 0
    best_params = None

    # Replace all instances of '2' with '1' in data['label']
    data['label'] = np.where(data['label'] == 2, 1, data['label'])

    correlation_funcs = {
        CorrelationMeasure.PEARSON: pearsonr,
        CorrelationMeasure.SPEARMAN: spearmanr,
        CorrelationMeasure.KENDALL: kendalltau,
    }

    correlation_func = correlation_funcs[correlation_measure]

    # Convert param_dict into GridSearch space
    param_space = {}
    for k, v in param_dict.items():
        if isinstance(v[0], int):
            param_space[k] = list(range(v[0], v[1] + 1))  # inclusive on both ends
        elif isinstance(v[0], float):
            param_space[k] = np.arange(v[0], v[1], 0.1).tolist()  # step size of 0.1
        elif isinstance(v, list):
            param_space[k] = v
        else:
            raise ValueError('Invalid parameter type. Supported types are list, int, and float.')

    grid = ParameterGrid(param_space)

    # Iterate through each combination of parameters
    for params in grid:
        # Keep track of the original columns
        original_columns = set(data.columns)

        # Apply the indicator function with the current parameters
        data_indicators = indicator_func(data.copy(), **params)

        # Determine which columns were added by the indicator function
        new_columns = list(set(data_indicators.columns) - original_columns)

        # Initialize the score to zero
        score = 0

        for column in new_columns:
            for lag in range(max_lag + 1):
                # Create a binary prediction array where a 'match' is when both the column and label are non-zero
                predictions = np.where(
                    (data_indicators[column].shift(lag) != 0) & (data_indicators['label'] != 0), 1, 0
                )
                # Calculate the correlation between the predictions and the label using the selected correlation measure
                temp_score = correlation_func(data_indicators['label'], predictions)[0]
                # Apply the decay factor
                temp_score *= decay ** lag

                if temp_score > score:
                    score = temp_score

        # In-place update of the console output
        sys.stdout.write(f'\rTrying params: {params}, score: {score}')
        sys.stdout.flush()

        if score > best_score:
            best_score = score
            best_params = params
            print(f'\nNew best params: {best_params}, score: {best_score}')

    return best_params, best_score
