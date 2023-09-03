import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax


class EchoStateNetwork:
    def __init__(self, input_dim, reservoir_size, output_dim, spectral_radius=0.95, seed=None):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.seed = seed

        self.init_weights()

    def init_weights(self):
        np.random.seed(self.seed)
        self.Win = np.random.rand(self.reservoir_size, 1 + self.input_dim) - 0.5
        self.W = np.random.rand(self.reservoir_size, self.reservoir_size) - 0.5

        # Set spectral radius
        radius = np.max(np.abs(np.linalg.eigvals(self.W)))
        self.W *= self.spectral_radius / radius

    def train(self, X, y, ridge_alpha=1e-6):
        # Shape: (batch_size, sequence_length, num_features)
        batch_size, sequence_length, num_features = X.shape
        X = X.reshape(-1, num_features)

        X_augmented = np.hstack((np.ones((X.shape[0], 1)), X))
        X_reservoir = np.tanh(np.dot(self.Win, X_augmented.T))

        # Reshape back to batch_size, sequence_length, reservoir_size
        X_reservoir = X_reservoir.T.reshape(batch_size, sequence_length, self.reservoir_size)

        # Get last hidden state
        X_last = X_reservoir[:, -1, :]

        self.clf = LogisticRegression(class_weight='balanced')
        self.clf.fit(X_last, y)

    def predict(self, X):
        batch_size, sequence_length, num_features = X.shape
        X = X.reshape(-1, num_features)

        X_augmented = np.hstack((np.ones((X.shape[0], 1)), X))
        X_reservoir = np.tanh(np.dot(self.Win, X_augmented.T))

        # Reshape back to batch_size, sequence_length, reservoir_size
        X_reservoir = X_reservoir.T.reshape(batch_size, sequence_length, self.reservoir_size)

        # Get last hidden state
        X_last = X_reservoir[:, -1, :]

        # Predict
        y_pred = self.clf.predict(X_last)
        return y_pred

    def predict_proba(self, X):
        if not hasattr(self.clf, 'decision_function'):
            raise AttributeError('LogisticRegression classifier has no decision_function method.')

        batch_size, sequence_length, num_features = X.shape
        X = X.reshape(-1, num_features)

        X_augmented = np.hstack((np.ones((X.shape[0], 1)), X))
        X_reservoir = np.tanh(np.dot(self.Win, X_augmented.T))

        # Reshape back to batch_size, sequence_length, reservoir_size
        X_reservoir = X_reservoir.T.reshape(batch_size, sequence_length, self.reservoir_size)

        # Get last hidden state
        X_last = X_reservoir[:, -1, :]

        # Predict probabilities
        decision_function = self.clf.decision_function(X_last)
        y_pred_proba = softmax(decision_function, axis=1)
        return y_pred_proba
