import os
import cv2
import csv
import numpy as np
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from skimage.feature import hog

# Define the simple CNN model
# Start timing
start_time1 = time.time()
# def create_cnn_model():
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(96, 96, 3)),  # Increase to 64 filters
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),  # Increase to 128 filters
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),  # Increase to 256 filters
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),  # Increase to 256 filters
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),  # Increase to 512 filters
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(256, activation='relu'),  # Increase to 256 units
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Dense(128, activation='relu'),  # Increase to 128 units
#         tf.keras.layers.Dense(64, activation='relu'),   # Keeping 64 units
#         tf.keras.layers.Dense(32, activation='relu')
#     ])
#     return model
  # 64 features but without bath normalizaiton
# def create_cnn_model():
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         # tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#         # tf.keras.layers.MaxPooling2D((2, 2)),
#         # tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#         # tf.keras.layers.MaxPooling2D((2, 2)),
#
#         tf.keras.layers.Flatten(),
#         # tf.keras.layers.Dense(512, activation='relu'),
#         # tf.keras.layers.Dense(256, activation='relu'),
#         #tf.keras.layers.Dense(128, activation='relu'),
#         #tf.keras.layers.Dropout(0.5)
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dropout(0.5)
#         tf.keras.layers.Dense(64, activation='relu')
#         tf.keras.layers.Dropout(0.5)
#
#     ])
#     return model
import tensorflow as tf


def create_cnn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
        tf.keras.layers.BatchNormalization(),  # Add BatchNormalization
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),  # Add BatchNormalization
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),  # Add BatchNormalization
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),  # Add BatchNormalization
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Add Dropout with a rate of 0.5

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5)  # Add Dropout with a rate of 0.5
    ])
    return model


# Function to extract CNN features from a frame
def extract_cnn_features(frame, model):
    # Preprocess the frame (e.g., resize, normalize, etc.)
    preprocessed_frame = preprocess_frame(frame)

    # Extract CNN features using the model
    features = model.predict(np.expand_dims(preprocessed_frame, axis=0)).flatten()

    return features


def extract_hog_features(frame):
    # Preprocess the frame for Hog feature extraction (resize to 32x32 and convert to grayscale)
    preprocessed_frame = cv2.resize(frame, (32, 32))
    preprocessed_frame = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2GRAY)

    # Create the HOGDescriptor
    hog = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)

    # Compute the HOG features
    features = hog.compute(preprocessed_frame)

    return features.flatten()  # Flatten the feature vector


# Function to preprocess a frame
def preprocess_frame(frame):
    # Preprocess the frame here (e.g., resize, normalize, etc.)
    preprocessed_frame = cv2.resize(frame, (96, 96))
    preprocessed_frame = preprocessed_frame / 255.0  # Normalize pixel values between 0 and 1

    return preprocessed_frame

# Main frames directory
frames_directory = '4Oct/Last_50'
if not os.path.exists(frames_directory):
    os.makedirs(frames_directory)

# Create a directory to save the CNN features
cnn_features_directory = "cnn_feature_Last50_BN"
if not os.path.exists(cnn_features_directory):
    os.makedirs(cnn_features_directory)

# Create a directory to save the Hog features
hog_features_directory = "hog_feature_Last50"
if not os.path.exists(hog_features_directory):
    os.makedirs(hog_features_directory)
# Start timing
start_time = time.time()
# Loop through subdirectories in the frames directory
for folder_name in os.listdir(frames_directory):
    folder_path = os.path.join(frames_directory, folder_name)

    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Create a CSV file for CNN features
        cnn_feature_file = os.path.join(cnn_features_directory, f"{folder_name}.csv")
        cnn_feature_rows = []

        # Create a CSV file for Hog features
        hog_feature_file = os.path.join(hog_features_directory, f"{folder_name}.csv")
        hog_feature_rows = []

        # Create the CNN model
        model = create_cnn_model()

        # Loop through frames in the subdirectory
        for frame_file in os.listdir(folder_path):
            frame_path = os.path.join(folder_path, frame_file)

            # Read the frame
            frame = cv2.imread(frame_path)

            # Check if the frame is valid
            if frame is None:
                print(f"Error reading frame {frame_path}")
                continue  # Skip to the next frame if the current one is invalid

            # Extract CNN features from the frame
            cnn_features = extract_cnn_features(frame, model)

            # Append CNN features to the cnn_feature_rows list
            cnn_feature_rows.append(cnn_features)

            # Extract Hog features from the frame
            hog_features = extract_hog_features(frame)

            # Append Hog features to the hog_feature_rows list
            hog_feature_rows.append(hog_features)

        # Write the CNN features to the CNN features CSV file
        cnn_feature_columns = [f"feature_{i}" for i in range(len(cnn_feature_rows[0]))]
        with open(cnn_feature_file, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(cnn_feature_columns)  # Write column names
            writer.writerows(cnn_feature_rows)

        # Write the Hog features to the Hog features CSV file
        hog_feature_columns = [f"hog_feature_{i}" for i in range(len(hog_feature_rows[0]))]
        with open(hog_feature_file, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(hog_feature_columns)  # Write column names
            writer.writerows(hog_feature_rows)

    #print(f"{video_name} done")
# End timing
end_time = time.time()
execution_time = end_time - start_time
execution_time1= end_time - start_time1
print(f"CNN and Hog features extracted and saved successfully. Execution time: {execution_time} seconds.")
print(f"CNN and Hog features extracted and saved successfully.from first time: {execution_time1} seconds.")
print("CNN and Hog features extracted and saved successfully.")
