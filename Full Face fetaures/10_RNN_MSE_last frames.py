import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras.optimizers import Adam

# Set the paths
#cnn_features_dir = "../featuers_frames/augmented_concatenated_feature"
#target_va_dir = '../featuers_frames/annotations_augmented'

cnn_features_dir = 'cnn_feature_Last50'
target_va_dir =  'annotations_augmented'
prediction = 'prediction/'
csv_file_path = 'metrics/metrics_high_gray_concatenated_Last50.csv'
model_name = 'model/rnn_Last50.h5'

# Define the number of epochs
epochs = 100

# Create the target directory if it doesn't exist
if not os.path.exists(prediction):
    os.makedirs(prediction)

# Define your RNN architecture
input_size = 96 * 96 * 1  # Flattened input size
hidden_size = 10  # Number of hidden units in LSTM layer

# Load features and labels
def load_dataset():
    features = []
    labels = []

    # Get the list of files in cnn_features_dir
    feature_files = os.listdir(cnn_features_dir)

    # Sort the files to get the last 26
    feature_files.sort()  # Sorts in lexicographical order

    # Take only the last 26 files
    feature_files = feature_files[-60:]

    # Iterate through the selected files
    for filename in feature_files:
        if filename.endswith('.csv'):
            # Load the features
            features_df = pd.read_csv(os.path.join(cnn_features_dir, filename))
            features.append(features_df.values.astype('float32'))

            # Load the corresponding valence and arousal labels
            labels_df = pd.read_csv(os.path.join(target_va_dir, filename), header=None, skiprows=1)
            labels_df.columns = labels_df.columns.astype(str)

            labels.append(labels_df.iloc[:, :1].values.astype('float32'))

    # Convert labels to a consistent shape
    labels = [label.flatten() for label in labels]

    return np.array(features), np.array(labels)

# Data Preprocessing
def preprocess_data(features, labels):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42, shuffle=True)
    # Create train and test directories within train_test folder
    train_dir = os.path.join('train_test', 'train')
    test_dir = os.path.join('train_test', 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Save training and testing data
    np.save(os.path.join(train_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(train_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(test_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(test_dir, 'y_test.npy'), y_test)

    return X_train, X_test, y_train, y_test

    return X_train, X_test, y_train, y_test

# Build the RNN Model
# def build_rnn_model(input_shape):
#     model = Sequential()
#     model.add(LSTM(10, input_shape=(input_shape[1], input_shape[2]), activation='relu', return_sequences=True))
#     model.add(LSTM(10, activation='relu', return_sequences=True))
#     model.add(LSTM(15, activation='relu', return_sequences=True))
#     model.add(LSTM(5, activation='relu'))
#     model.add(Dense(104))  # Output layer with 2 units for valence and arousal
#
#     return model

def build_rnn_model(input_shape):
    model = Sequential()
    model.add(LSTM(10, input_shape=(input_shape[1], input_shape[2]), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(10, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(15, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(5, activation='relu'))
    model.add(Dense(39))  # Output layer with 2 units for valence and arousal

    return model

# Train the RNN Model
def train_model(X_train, y_train):
    model = build_rnn_model(X_train.shape)  # Pass input shape excluding the sample dimension
    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mse'])

    # Initialize lists to store metrics for each epoch
    train_loss_history = []
    val_loss_history = []
    val_mse_valence_history = []
    val_mse_arousal_history = []

    # Inside the training loop
    for epoch in range(epochs):
        history = model.fit(X_train, y_train, batch_size=128, epochs=1, validation_split=0.2, verbose=1)

        # Extract training and validation loss from the history object
        train_loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]

        # Append metrics to the respective lists
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # Calculate and store validation MSE
        y_pred = model.predict(X_test)
        mse_valence = mean_squared_error(y_test[:, 0], y_pred[:, 0])
        mse_arousal = mean_squared_error(y_test[:, 1], y_pred[:, 1])
        val_mse_valence_history.append(mse_valence)
        val_mse_arousal_history.append(mse_arousal)

    # Create a dictionary to store the metrics
    metrics_dict = {
        'Epoch': list(range(1, epochs + 1)),
        'Train_Loss': train_loss_history,
        'Val_Loss': val_loss_history,
        'Val_MSE_Valence': val_mse_valence_history,
        'Val_MSE_Arousal': val_mse_arousal_history,
    }

    # Create a Pandas DataFrame from the metrics_dict
    metrics_df = pd.DataFrame(metrics_dict)

    # Save metrics to a CSV file using Pandas
    metrics_df.to_csv(csv_file_path, index=False)

    return model

if __name__ == '__main__':
    features, labels = load_dataset()
    features = features[:120]
    labels = labels[:120]

    X_train, X_test, y_train, y_test = preprocess_data(features, labels)

    model = train_model(X_train, y_train)

    # Predict valence and arousal values from the model
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error for valence and arousal
    mse_valence = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    mse_arousal = mean_squared_error(y_test[:, 1], y_pred[:, 1])

    print("Mean Squared Error for Valence:", mse_valence)
    print("Mean Squared Error for Arousal:", mse_arousal)

    # Save the model
    model.save(model_name)
