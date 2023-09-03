import torch
from vctr.data.data_loader import get_data_with_features_and_labels
from vctr.data.lstm_preprocessor import preprocess_data
from vctr.models.lstm.defaults import SEQUENCE_LENGTH, TEST_PCT


import torch
from torch.utils.data import Dataset


def get_loader(X, y, close_values=None, batch_size=128):
    if close_values is None:
        train_data_tensor = torch.tensor(X, dtype=torch.float32)
        train_target_tensor = torch.tensor(y, dtype=torch.long)
        train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_target_tensor)
        return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    else:
        train_dataset = OHLCVDataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), close_values
        )
        return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def get_inference_loader(*args, batch_size=128, **kwargs):
    X, y = get_data_with_features_and_labels(*args, **kwargs)
    X, y = preprocess_data(X, y, lookback=SEQUENCE_LENGTH, torch=True)
    return get_loader(X, y, batch_size=batch_size)


def get_train_and_val_loaders(
    *args, batch_size=128, lookback=SEQUENCE_LENGTH, test_pct=TEST_PCT, return_data=False, **kwargs
):
    data = get_data_with_features_and_labels(*args, separate=False, **kwargs)
    X, y = data.drop(['label'], axis=1), data['label']

    if test_pct == 0:
        X_train, y_train = preprocess_data(X, y, lookback=lookback, torch=True)
    else:
        X_train, y_train, X_val, y_val = preprocess_data(X, y, lookback=lookback, torch=True, test_pct=test_pct)

    train_loader = get_loader(X_train, y_train, batch_size=batch_size)
    val_loader = get_loader(X_val, y_val, batch_size=batch_size) if test_pct != 0 else None

    return (train_loader, val_loader, data) if return_data else (train_loader, val_loader)


def get_train_and_val_loaders_with_close(
    *args, batch_size=128, lookback=SEQUENCE_LENGTH, test_pct=TEST_PCT, return_data=False, **kwargs
):
    data = get_data_with_features_and_labels(*args, separate=False, **kwargs)
    X, y = data.drop(['label'], axis=1), data['label']
    close_values = data['close']
    X_train, y_train, X_val, y_val = preprocess_data(X, y, lookback=lookback, torch=True, test_pct=test_pct)

    # Get the close values for the train and val sets
    close_values_train = close_values.iloc[lookback - 1 : len(X_train) + lookback - 1]
    close_values_val = close_values.iloc[len(X_train) + lookback - 1 : len(X_train) + len(X_val) + lookback - 1]

    train_loader = get_loader(X_train, y_train, close_values=close_values_train, batch_size=batch_size)
    val_loader = get_loader(X_val, y_val, close_values=close_values_val, batch_size=batch_size)

    return (train_loader, val_loader, data) if return_data else (train_loader, val_loader)


class OHLCVDataset(Dataset):
    def __init__(self, input_data, targets, close_values):
        self.input_data = input_data
        self.targets = targets
        self.close_values = close_values

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.targets[idx], self.close_values[idx]
