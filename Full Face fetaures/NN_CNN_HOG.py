import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

# Load the data
features_df = pd.read_csv('sheet/cnn_features_last50.csv')  # Update with your actual file path
target_df = pd.read_csv('sheet/valence.csv')          # Update with your actual file path

# Preprocessing
X = features_df.values
y = target_df.values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the neural network model
# model = Sequential([
#     Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
#     Dense(64, activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(1)  # Output layer with 1 neuron for regression
# ])

model = Sequential([
    Dense(100, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(1)
])
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Time per epoch
start_time = time.time()

# Train the model on 70% of the data
history = model.fit(X_scaled, y, epochs=100, batch_size=32, verbose=1)

end_time = time.time()
time_per_epoch = (end_time - start_time) / len(history.history['loss'])
overall_time = end_time - start_time

print(f'Time per Epoch: {time_per_epoch} seconds')
print(f'Overall Time Taken: {overall_time} seconds')

# Evaluate the model on the entire dataset
predictions_all = model.predict(X_scaled)

# Calculate MSE and RMSE for the entire dataset
mse_all = mean_squared_error(y, predictions_all)
print(f'Mean Squared Error (MSE) for Entire Dataset: {mse_all}')

rmse_all = np.sqrt(mse_all)
print(f'Root Mean Squared Error (RMSE) for Entire Dataset: {rmse_all}')

# Optionally, you can save the trained model
model.save('valence_prediction_model_128_64_32.h5')  # Update with your desired file path
