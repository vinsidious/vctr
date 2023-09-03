from datetime import timedelta

import dateutil
import numpy as np
import pandas as pd
import pytz
import torch
import torchmetrics as tm
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from torch_lr_finder import LRFinder
from tqdm import tqdm
from vctr.data.data_loader import get_data_with_features_and_labels
from vctr.data.lstm_preprocessor import preprocess_data
from vctr.models.lstm.data import get_loader, get_train_and_val_loaders_with_close
from vctr.models.lstm.defaults import DROPOUT, HIDDEN_SIZE, INPUT_SIZE, NUM_CLASSES, NUM_LAYERS, SEQUENCE_LENGTH
from vctr.models.lstm.main import VCNet
from vctr.models.lstm.thresholds import backtest_portfolio
from vctr.models.lstm.utils import EarlyStopping, trn_value, val_value


MODELS_DIR = '/Users/vince/vctr/data/models'


def train_model(
    batch_size=128,
    crypto=True,
    end=None,
    epochs=15,
    label_args=None,
    learning_rate=0.001,
    model=None,
    no_cache=False,
    start=None,
    symbol='ETH',
    timeframes=['1h'],
    sequence_length=24,
    test_pct=0.3,
    find_optimal_lr=False,
):
    train_loader, val_loader = get_train_and_val_loaders_with_close(
        end=end,
        start=start,
        symbol=symbol,
        timeframes=timeframes,
        label_args=label_args,
        batch_size=batch_size,
        no_cache=no_cache,
        crypto=crypto,
        test_pct=test_pct,
        lookback=sequence_length,
    )

    # Compute class weights
    class_weights = torch.zeros(NUM_CLASSES)
    for _, labels, _ in train_loader:
        for label in labels:
            class_weights[label] += 1
    class_weights = class_weights / class_weights.sum()
    class_weights = 1 / class_weights

    criterion = CrossEntropyLoss(weight=class_weights).to('mps')
    # criterion = WeightedObjective(class_weights=class_weights, weight_loss=1.0, weight_acc=1.5, weight_f1=1.5).to('mps')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=10)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        verbose=True,
        min_lr=0.00000001,
        patience=1,
    )

    if find_optimal_lr:
        # Create an instance of LRFinder and perform the learning rate range test
        lr_finder = LRFinder(model, optimizer, criterion, device='mps')
        lr_finder.range_test(
            train_loader, val_loader=val_loader, start_lr=1e-8, end_lr=1e-2, num_iter=40, step_mode='exp'
        )

        # Plot the learning rate vs. loss
        lr_finder.plot()

        # Reset the model and optimizer to their initial state
        lr_finder.reset()

    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        val_loss = 0.0

        train_f1r = 0.0
        val_f1r = 0.0

        num_train_batches = len(train_loader)
        num_val_batches = len(val_loader)

        all_logits = []
        all_targets = []
        largest_val_profit = -np.inf

        # Train
        with tqdm(
            total=num_train_batches, desc=f'Epoch {epoch}/{epochs}, Train Loss: ', unit='batch', colour='#90EE90'
        ) as train_pbar:
            desc = ''

            model.train()
            for batch_idx, (inputs, target, close_values) in enumerate(train_loader):
                inputs = inputs.to('mps')
                target = target.to('mps')

                # _, logits = run_inference(model, inputs, target)
                # labels = logits.argmax(axis=1)
                logits = model(inputs)
                loss = criterion(logits, target)

                # Add L2 regularization to the loss
                l2_reg = 0.0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        l2_reg += torch.norm(param, p=2)
                loss += 0.001 * l2_reg

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

                preds = logits.argmax(axis=1).cpu().numpy()
                profit = backtest_portfolio(preds, close_values, timeframes[0].replace('m', 'T'))

                f1r = tm.F1Score(task='multiclass', num_classes=NUM_CLASSES, average='macro').to('mps')(
                    logits, target
                )
                train_f1r += f1r

                desc = f'Epoch {epoch} / {epochs}, Loss: {trn_value(train_loss / (batch_idx+1))}, F1R: {trn_value(train_f1r / (batch_idx+1))}, Profit: {profit * 100:.2f}%'
                train_pbar.set_description(desc)
                train_pbar.update()

            # Validation
            model.eval()
            train_pbar.reset(total=num_val_batches)
            train_pbar.colour = '#ADD8E6'
            with torch.no_grad():
                for batch_idx, (inputs, target, close_values) in enumerate(val_loader):
                    inputs, target = inputs.to('mps'), target.to('mps')

                    logits = model(inputs)

                    all_logits.append(logits)
                    all_targets.append(target)

                    loss = criterion(logits, target)
                    val_loss += loss.item()

                    preds = logits.argmax(axis=1).cpu().numpy()
                    profit = backtest_portfolio(preds, close_values, timeframes[0].replace('m', 'T'))

                    f1r = tm.F1Score(task='multiclass', num_classes=NUM_CLASSES, average='macro').to('mps')(
                        logits, target
                    )
                    val_f1r += f1r

                    train_pbar.set_description(
                        ', '.join(
                            [
                                f'Epoch {epoch} / {epochs}',
                                f'Loss: [{trn_value(train_loss / num_train_batches)} {val_value(val_loss / (batch_idx+1))}]',
                                f'F1R: [{trn_value(train_f1r / num_train_batches)} {val_value(val_f1r / (batch_idx+1))}]',
                                f'Profit: {profit * 100:.2f}%',
                            ]
                        )
                    )
                    train_pbar.update()

                    if profit > largest_val_profit:
                        largest_val_profit = profit
                        save_model(model, f'{model.name}-highest-profit')

            train_pbar.colour = '#ADD8E6'
            train_pbar.update()

        # if early_stopping(val_loss):
        #     break

        scheduler.step(val_loss)

        # if val_loss == early_stopping.best_loss and epoch % 5 == 0:
        # all_logits = torch.cat(all_logits, dim=0)
        # all_targets = torch.cat(all_targets, dim=0)

        # # Compute the decision thresholds
        # model.thresholds = find_optimal_thresholds(all_logits, df)

        # Save the model
        save_model(model, 'latest')
        save_model(model)

    save_model(model, 'latest')
    save_model(model)

    return model


def dynamic_threshold_adjustment(logits, target_distribution):
    """
    Adjusts per-class decision threshold to match the target distribution.

    :param logits: 2D array-like, shape (n_samples, n_classes)
        Logits from a multiclass classifier.
    :param target_distribution: 1D array-like, shape (n_classes,)
        Array of target class distributions.
    :return: 1D array, shape (n_samples,)
        Array of predicted class labels.
    """

    logits = logits.cpu().numpy()

    # Normalize the target distribution
    target_distribution = np.array(target_distribution) / np.sum(target_distribution)

    # Compute the softmax of logits
    softmax = np.exp(logits) / np.sum(np.exp(logits), axis=1)[:, np.newaxis]

    n_samples, n_classes = softmax.shape
    labels = np.full(n_samples, -1, dtype=int)

    # Calculate the number of samples needed for each class based on the desired distribution
    target_counts = (n_samples * target_distribution).astype(int)

    # Sort the predictions for each class by probability, only including the predictions
    # where the probability for that class is the greatest probability of the 3 classes
    max_probs_indices = np.argmax(softmax, axis=1)
    sorted_indices = np.argsort(-softmax, axis=0)

    for c in range(n_classes):
        count = 0
        for idx in sorted_indices[:, c]:
            if max_probs_indices[idx] == c:
                labels[idx] = c
                count += 1
                if count >= target_counts[c]:
                    break

    # Default to the majority class (index 0) in the event there are insufficient predictions
    # for any of the other classes
    labels[labels == -1] = 0

    return labels


def predict_with_thresholds(logits, thresholds):
    """
    Function that accepts logits, decision thresholds and returns the predictions.

    Args:
    logits (torch.Tensor): Logits output from the classifier (shape: batch_size x num_classes).
    thresholds (list): List of decision thresholds for each class (length: num_classes).

    Returns:
    predictions (torch.Tensor): Predicted class labels (shape: batch_size).
    """
    # Convert logits to probabilities using softmax
    probabilities = torch.softmax(logits, dim=-1)

    # Apply decision thresholds
    binary_preds = (
        probabilities >= torch.tensor(thresholds, dtype=torch.float32).to(probabilities.device)
    ).float()

    # Assign the class with the highest probability that meets the threshold criteria
    predictions = (
        torch.where(
            binary_preds == torch.max(binary_preds, dim=-1, keepdim=True).values,
            torch.arange(binary_preds.size(1), device=probabilities.device).expand_as(binary_preds).float(),
            torch.zeros_like(binary_preds) - 1,
        )
        .max(dim=-1)
        .values.long()
    )

    return predictions


def predict(
    model,
    symbol,
    timeframe=None,
    timeframes=None,
    start=None,
    end=None,
    batch_size=128,
    crypto=True,
    label_args=None,
    sequence_length=24,
):
    # If `model` is a string, load the model from disk.
    if isinstance(model, str):
        model = load_model(model)

    if start is not None:
        try:
            original_start = dateutil.parser.parse(start)
        except TypeError:
            original_start = start
        # Move the start date back 30 days to account for the lookback period
        # and the time it takes to get most indicators calculated
        start = original_start - timedelta(days=30)
        if not original_start.tzinfo:
            # If original_start is not timezone aware, convert it here
            original_start = pytz.utc.localize(original_start)

    if timeframe is not None:
        timeframes = np.asarray(timeframe)

    data = get_data_with_features_and_labels(
        symbol,
        timeframes=timeframes,
        start=start,
        end=end,
        separate=False,
        crypto=crypto,
        label_args=label_args,
    )

    if end is not None:
        data = data[:end].copy()

    X, y = preprocess_data(data.drop(['label'], axis=1), data['label'], lookback=sequence_length, torch=True)
    loader = get_loader(X, y, batch_size=batch_size)

    model.eval()

    # Inference on the inputs
    all_logits = []
    for batch, _ in loader:
        with torch.no_grad():
            logits = model(batch.to('mps'))
            all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)
    predictions = dynamic_threshold_adjustment(all_logits, [0.98, 0.01, 0.01])

    data = data[len(data) - len(predictions) :]
    data['pred'] = pd.Series(predictions, index=data.index)

    if start is not None:
        data = data[data.index >= original_start]

    return data


def load_model(name):
    return torch.load(f'{MODELS_DIR}/{name}.pt')


def save_model(model, name=None):
    if name is None:
        name = model.name
    torch.save(model, f'{MODELS_DIR}/{name}.pt')


def new_model(
    input_size: int = INPUT_SIZE,
    hidden_size: int = HIDDEN_SIZE,
    num_layers: int = NUM_LAYERS,
    num_classes: int = NUM_CLASSES,
    dropout: float = DROPOUT,
    batch_norm: bool = True,
):
    return VCNet(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        # dropout_rate=dropout,
        # kernel_size=3,
    ).to('mps')
