import os

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
from keras.layers import LSTM, Dense
from keras.models import Sequential, load_model
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.utils import compute_class_weight
from vctr.data.data_loader import get_data_with_features_and_labels
from vctr.data.lstm_preprocessor import preprocess_data
from vctr.utils.progress import TqdmProgressCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class LSTM_Model:
    def __init__(
        self,
        name='lstm',
        n_timesteps=30,
        batch_size=64,
        n_features=None,
        targets=[('label', [0, 1, 2])],
        label_args=(0.02, 0.005),
        n_classes=3,
    ):
        self.name = name
        self.n_timesteps = n_timesteps
        self.batch_size = batch_size
        self.n_features = n_features
        self.targets = targets
        self.label_args = label_args
        self.epochs = 100
        self.n_classes = n_classes
        self.model_id = f'{self.name}_{self.n_timesteps}_{self.n_features}_{self.n_classes}_{self.batch_size}'
        self.model_filepath = f'models/{self.model_id}.h5'

        self.load_model()

    def train(self, symbols, timeframe):
        # Cast symbols to list if it's a string
        if isinstance(symbols, str):
            symbols = [symbols]

        for symbol in symbols:
            print(f'Training {symbol} ({timeframe})...')
            self._train(symbol, timeframe)

    def _train(self, symbol, timeframe):
        # Load the dataset
        X, y = get_data_with_features_and_labels(symbol, timeframe, label_args=self.label_args)
        X_train, y_train, X_val, y_val = preprocess_data(X, y, lookback=self.n_timesteps, test_pct=0.3)

        cw = compute_class_weight('balanced', classes=self.targets[0][1], y=y.values)
        class_weights = dict(enumerate(cw))

        self.model.fit(
            X_train,
            y_train,
            verbose=0,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=[
                ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=0, min_lr=0.00001),
                TqdmProgressCallback(),
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=6, restore_best_weights=True, verbose=1, mode='min'
                ),
            ],
        )

    def predict(self, symbol, timeframe):
        X, y = get_data_with_features_and_labels(symbol, timeframe, label_args=self.label_args)
        data = X.copy().join(y)
        X, y = preprocess_data(X, y, lookback=self.n_timesteps)

        return self._predict(X, y, data)

    def predict_custom(self, symbol, timeframe):
        X, y = get_data_with_features_and_labels(symbol, timeframe, label_args=self.label_args)
        data = X.copy().join(y)
        X, y = preprocess_data(X, y, lookback=self.n_timesteps)

        y_proba = self.model.predict(X)
        return X, y, y_proba, data

    def _predict(self, X_test, y_test, data):
        y_pred = self.model.predict(X_test)
        data = data[len(data) - len(y_pred) :].copy()

        y_pred = pd.Series(np.argmax(y_pred, axis=1), name='pred')
        y_true = pd.Series(np.argmax(y_test, axis=1), name='label')[-len(y_pred) :]

        # set the index of each series to be the index of the dataframe
        y_pred.index = data.index
        y_true.index = data.index

        data['pred'] = y_pred
        data['label'] = y_true

        # Calculate and print metrics
        precision_val = precision_score(y_true.values, y_pred.values, average='macro')
        f1_macro_val = f1_score(y_true.values, y_pred.values, average='macro')
        recall_val = recall_score(y_true.values, y_pred.values, average='macro')
        accuracy_val = accuracy_score(y_true.values, y_pred.values)

        print(f'Precision Macro: {precision_val * 100:.2f}%')
        print(f'F1 Macro: {f1_macro_val * 100:.2f}%')
        print(f'Recall Macro: {recall_val * 100:.2f}%')
        print(f'Accuracy: {accuracy_val * 100:.2f}%')

        # Print classification report
        print(classification_report(y_true.values, y_pred.values))

        return data

    def create_model(self):
        self.model = Sequential(
            [
                LSTM(128, input_shape=(self.n_timesteps, self.n_features)),
                Dense(self.n_classes, activation='softmax'),
            ]
        )

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=[
                tf.metrics.Precision(name='prec'),
                tf.metrics.Recall(name='rec'),
                tf.metrics.CategoricalAccuracy(name='acc'),
            ],
        )

        return self.model

    def save_model(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save(self.model_filepath)

    def load_model(self):
        if not os.path.exists(self.model_filepath):
            self.create_model()
            return
        else:
            with tf.keras.utils.custom_object_scope():
                self.model = load_model(self.model_filepath)
