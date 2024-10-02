# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
#
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
#
# # Load the data
# features_df = pd.read_csv('cnn_features.csv')  # Update with your actual file path
# target_df = pd.read_csv('valence.csv')     # Update with your actual file path
#
# # Preprocessing
# X = features_df.values
# y = target_df.values
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # Standardize features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#
#
# # Define the neural network model
# model = Sequential([
#     Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
#     Dense(64, activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(1)  # Output layer with 1 neuron for regression
# ])
# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')
#
# # Train the model
# model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
#
# # Evaluate the model
# loss = model.evaluate(X_test, y_test, verbose=0)
# print(f'Test Loss: {loss}')
#
# # Make predictions
# predictions = model.predict(X_test)
#
# # Calculate MSE
# mse = mean_squared_error(y_test, predictions)
# print(f'Mean Squared Error (MSE): {mse}')
#
# # Calculate RMSE
# rmse = np.sqrt(mse)
# print(f'Root Mean Squared Error (RMSE): {rmse}')
#
# # Optionally, you can save the trained model
# model.save('valence_prediction_model_cnn.h5')  # Update with your desired fil
#
#
# # Make predictions on the testing data
# predictions_testing = model.predict(X_testing)
#
# # Load the true target values for testing data
# true_values_testing = pd.read_csv('cnn_features.csv')  # Update with the path to your new testing target file
#
# # Calculate MSE and RMSE for testing data
# mse_testing = mean_squared_error(true_values_testing, predictions_testing)
# print(f'Mean Squared Error (MSE) for Testing Data: {mse_testing}')
#
# rmse_testing = np.sqrt(mse_testing)
# print(f'Root Mean Squared Error (RMSE) for Testing Data: {rmse_testing}')

#++++++++++++++++++++++#test the model on on entire dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Load the data
features_df = pd.read_csv('cnn_features.csv')  # Update with your actual file path
target_df = pd.read_csv('valence.csv')          # Update with your actual file path

# Preprocessing
X = features_df.values
y = target_df.values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer with 1 neuron for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on 70% of the data
model.fit(X_scaled, y, epochs=100, batch_size=32, verbose=1)

# Evaluate the model on the entire dataset
predictions_all = model.predict(X_scaled)

# Calculate MSE and RMSE for the entire dataset
mse_all = mean_squared_error(y, predictions_all)
print(f'Mean Squared Error (MSE) for Entire Dataset: {mse_all}')

rmse_all = np.sqrt(mse_all)
print(f'Root Mean Squared Error (RMSE) for Entire Dataset: {rmse_all}')

# Optionally, you can save the trained model
model.save('valence_prediction_model_128_64_32.h5')  # Update with your desired file path
