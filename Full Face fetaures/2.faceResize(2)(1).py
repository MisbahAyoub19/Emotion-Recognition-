import os
import cv2

frames_src_path = 'frames_high_augmented'
frames_save_path = 'faceResize_high_gray'
videos_src_path = '../../../ADFES_high'
videos = os.listdir(videos_src_path)
videos = filter(lambda x: x.endswith('wmv'), videos)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def faceResize():
    for each_video in videos:
        print(each_video)

        each_video_name, _ = os.path.splitext(each_video)
        os.makedirs(os.path.join(frames_save_path, each_video_name), exist_ok=True)

        each_video_src_full_path = os.path.join(frames_src_path, each_video_name)
        each_frame_save_full_path = os.path.join(frames_save_path, each_video_name)

        frames = os.listdir(each_video_src_full_path)

        for each_frame in frames:
            print(each_frame)

            each_frame_name, _ = os.path.splitext(each_frame)
            each_frame_src_full_path = os.path.join(each_video_src_full_path, each_frame_name)

            # Read the frame
            image = cv2.imread(each_frame_src_full_path + '.jpg')

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Process each detected face
            for i, (x, y, w, h) in enumerate(faces):
                # Crop the detected face
                face = gray[y:y + h, x:x + w]  # Convert to grayscale

                # Resize the face to 96x96 pixels
                size = (96, 96)
                resized_face = cv2.resize(face, size)

                # Save the resized face as a grayscale image
                save_path = os.path.join(each_frame_save_full_path, f"{each_frame_name}.jpg")
                cv2.imwrite(save_path, resized_face)

if __name__ == '__main__':
    faceResize()
