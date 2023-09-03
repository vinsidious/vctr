import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load data
data = pd.read_csv("time_series_data.csv")

# Prepare data for LSTM
time_series_data = data[["time_series_feature_1", "time_series_feature_2"]].values
scaler = MinMaxScaler(feature_range=(0, 1))
time_series_data = scaler.fit_transform(time_series_data)

# LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=64, input_shape=(time_series_data.shape[1], 1), return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=64, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=32, activation="relu"))
lstm_model.add(Dense(units=1, activation="sigmoid"))

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Reshape data for LSTM training
time_series_data = np.reshape(time_series_data, (time_series_data.shape[0], time_series_data.shape[1], 1))

# Train LSTM model
X_train, X_test, y_train, y_test = train_test_split(time_series_data, data["movement"], test_size=0.2, random_state=42)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# Extract temporal features from LSTM
temporal_features_train = lstm_model.predict(X_train)
temporal_features_test = lstm_model.predict(X_test)

# Combine temporal features with other features
other_features_train = data.drop(columns=["time_series_feature_1", "time_series_feature_2", "movement"]).iloc[X_train.index]
other_features_test = data.drop(columns=["time_series_feature_1", "time_series_feature_2", "movement"]).iloc[X_test.index]

combined_features_train = np.concatenate((temporal_features_train, other_features_train.values), axis=1)
combined_features_test = np.concatenate((temporal_features_test, other_features_test.values), axis=1)

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(combined_features_train, y_train)

# Predict and evaluate
y_pred = rfc.predict(combined_features_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
