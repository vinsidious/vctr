import os
import re

from colorama import Fore, Style

MODEL_FOLDER_PATH = '/Users/vince/vctr/data/models'


def get_latest_model_number():
    """
    Given the path to a folder, find the most recent LSTM model
    and return its version number as an integer.
    """
    model_regex = r'^lstm-mk-(\d+)'
    latest_model_num = -1
    for filename in os.listdir(MODEL_FOLDER_PATH):
        if match := re.match(model_regex, filename):
            model_num = int(match[1])
            if model_num > latest_model_num:
                latest_model_num = model_num
    return latest_model_num


def get_new_model_name():
    """
    Given the path to a folder containing LSTM models, return a new
    model name with an incremented version number.
    """
    latest_model_num = get_latest_model_number()
    new_model_num = latest_model_num + 1
    new_model_name = f'lstm-mk-{new_model_num}'
    return new_model_name


trn_color = Fore.LIGHTGREEN_EX
val_color = Fore.LIGHTBLUE_EX


def trn_value(val):
    return f'{trn_color}{val:.3f}{Style.RESET_ALL}'


def val_value(val):
    return f'{val_color}{val:.3f}{Style.RESET_ALL}'


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0.001):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f'INFO: Early stopping counter {self.counter} of {self.patience}')
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
        return self.early_stop
