import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import time

# Load the data
features_df = pd.read_csv('sheet/cnn_features_last50.csv')  # Update with your actual file path
target_df = pd.read_csv('sheet/valence.csv')          # Update with your actual file path

# Preprocessing
X = features_df.values
y = target_df.values.ravel()  # Flatten the target array

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the Gradient Boosting Regressor model
gradient_boosting_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=1.0,
    min_samples_split=2,
    random_state=42  # For reproducibility
)

# Time the training process
start_time = time.time()

# Train the model
gradient_boosting_model.fit(X_scaled, y)

# Calculate training time
training_time = time.time() - start_time
print(f'Training Time: {training_time} seconds')

# Predictions
predictions = gradient_boosting_model.predict(X_scaled)

# Calculate MSE and RMSE
mse = mean_squared_error(y, predictions)
print(f'Mean Squared Error (MSE): {mse}')

rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Optionally, you can save the trained model
import joblib
joblib.dump(gradient_boosting_model, 'gradient_boosting_model.pkl')
