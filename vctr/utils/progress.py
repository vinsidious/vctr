from colorama import Fore, Back, Style
from tqdm import tqdm
import tensorflow as tf


def t(val):
    if val < 1:
        return round(val * 100)
    else:
        return f'{val:.2f}'


import tensorflow as tf
from tqdm import tqdm


class TqdmProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.epoch_bar = None
        self.batch_bar = None

    def on_train_begin(self, logs=None):
        self.metrics = ['loss', 'rec', 'prec', 'acc']

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.batch_bar = tqdm(
            total=self.params['steps'],
            desc=f'Epoch {self.epoch + 1}',
            unit='batch',
            ncols=135,
            dynamic_ncols=True,
            position=0,
            colour='blue',
        )

    batch_logs = {}

    def on_batch_end(self, batch, logs=None):
        self.batch_logs = logs
        self.batch_bar.set_description(self.get_tuples(logs))
        self.batch_bar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        self.batch_bar.set_description(self.get_tuples(self.batch_logs, logs))
        self.batch_bar.close()

    def get_tuples(self, train_logs, val_logs=None):
        pairs = []

        def lc(val):
            return Fore.LIGHTCYAN_EX + f'{val:.3f}' + Style.RESET_ALL

        def lg(val):
            return Fore.LIGHTGREEN_EX + f'{val:.3f}' + Style.RESET_ALL

        for metric in self.metrics:
            train_value = train_logs.get(metric, 0)
            if val_logs:
                val_value = val_logs.get(f'val_{metric}', 0)
                pairs.append(f'{metric} [{lc(train_value)} {lg(val_value)}]')
            else:
                pairs.append(f'{metric} [{lc(train_value)}]')
        return ', '.join(pairs)
