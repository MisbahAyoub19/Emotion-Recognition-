import os
import cv2
import time

# Directory path containing the videos
video_directory = 'C:/Users/M.Ayoub19/PycharmProjects/Emotion recognition/Dec2023/ADFES_high'

# Create a directory to save the frames
frames_directory = "frames_high_augmented"
if not os.path.exists(frames_directory):
    os.makedirs(frames_directory)

start_time = time.time()  # Start timing

# Loop through the videos in the directory
for video_file in os.listdir(video_directory):
    video_path = os.path.join(video_directory, video_file)
    video_name = os.path.splitext(video_file)[0]

    # Create a folder for each video to save the frames
    video_frames_directory = os.path.join(frames_directory, video_name)
    if not os.path.exists(video_frames_directory):
        os.makedirs(video_frames_directory)

    # Read the video
    video_capture = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Apply horizontal flips
        flipped_frame = cv2.flip(frame, 1)
        flipped_frame2 = cv2.flip(flipped_frame, 1)
        flipped_frame3 = cv2.flip(flipped_frame2, 1)

        # Save the original frame with the frame number
        frame_path = os.path.join(video_frames_directory, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

        # Increment the frame count
        frame_count += 1

        # Save the flipped frames with the same frame number
        flipped_frame_path = os.path.join(video_frames_directory, f"frame_{frame_count}.jpg")
        cv2.imwrite(flipped_frame_path, flipped_frame)
        frame_count += 1

        flipped_frame2_path = os.path.join(video_frames_directory, f"frame_{frame_count}.jpg")
        cv2.imwrite(flipped_frame2_path, flipped_frame2)
        frame_count += 1

        flipped_frame3_path = os.path.join(video_frames_directory, f"frame_{frame_count}.jpg")
        cv2.imwrite(flipped_frame3_path, flipped_frame3)
        frame_count += 1

    video_capture.release()
end_time = time.time()  # End timing
execution_time = end_time - start_time
print(f"Frames saved successfully. Execution time: {execution_time} seconds.")
