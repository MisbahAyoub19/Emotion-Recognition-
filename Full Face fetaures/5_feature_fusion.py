import os
import numpy as np
import time

start_time = time.time()  # Start timing
# Replace with the path to the CNN feature folder
#cnn_feature_folder_path = "cnn_feature"
cnn_feature_folder_path = "cnn_feature_Last50_BN"
# Replace with the path to the HOG feature folder
#hog_feature_folder_path = "hog_feature"
hog_feature_folder_path = "hog_feature_Last50"
# Replace with the path where you want to save the concatenated features
#concatenated_features_folder_path = "concatenated_feature"
concatenated_features_folder_path = "concatenated_feature"
# Create the directory to store concatenated features if it doesn't exist
if not os.path.exists(concatenated_features_folder_path):
    os.makedirs(concatenated_features_folder_path)

# Get the list of files in the CNN feature folder
cnn_feature_files = os.listdir(cnn_feature_folder_path)

# Iterate over the CNN feature files
for cnn_feature_file in cnn_feature_files:
    # Check if the corresponding HOG feature file exists
    hog_feature_file = os.path.join(hog_feature_folder_path, cnn_feature_file)
    if not os.path.exists(hog_feature_file):
        print(f"Skipping {cnn_feature_file} due to missing HOG feature file.")
        continue

    # Read the CNN feature file, skipping the header row, and specifying the delimiter as a comma
    cnn_features = np.loadtxt(os.path.join(cnn_feature_folder_path, cnn_feature_file), skiprows=1, delimiter=',')

    # Read the HOG feature file, skipping the header row, and specifying the delimiter as a comma
    hog_features = np.loadtxt(hog_feature_file, skiprows=1, delimiter=',')

    # Concatenate the CNN and HOG features
    concatenated_features = np.concatenate([cnn_features, hog_features], axis=1)

    # Define header names for the columns
    header_names = ["cnn_feature_" + str(i) for i in range(cnn_features.shape[1])] + ["hog_feature_" + str(i) for i in range(hog_features.shape[1])]

    # Save the concatenated features to a file with header names
    concatenated_feature_file = os.path.join(concatenated_features_folder_path, cnn_feature_file)
    np.savetxt(concatenated_feature_file, concatenated_features, delimiter=',', header=','.join(header_names), comments='')

    print(f"Concatenated features saved for {cnn_feature_file}.")
end_time = time.time()  # End timing
execution_time = end_time - start_time
print(f"Frames saved successfully. Execution time: {execution_time} seconds.")
print("Concatenated features saved successfully.")