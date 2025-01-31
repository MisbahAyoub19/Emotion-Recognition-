import glob
import cv2
import os
import dlib
import time
import frames2

# Dr andrew's file path
# vidFolder = 'vids'
# Misbah File Path
#vidFolder = 'C:/Users/M.Ayoub19/PycharmProjects/Emotion recognition/Dec2023/2024/3JAN'
vidFolder = 'C:/Users/M.Ayoub19/PycharmProjects/Emotion recognition/2024/August/LAR'

datasetFolder = os.path.join(vidFolder, 'C:/Users/M.Ayoub19/PycharmProjects/Emotion recognition/2024/August/eyeblink')
videoFiles = glob.glob(os.path.join(datasetFolder, '*.avi'))
#videoFiles = glob.glob(os.path.join(datasetFolder, '*.avi'))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

for vidFile in videoFiles:
    start = time.time()

    Video_path = os.path.join(datasetFolder, vidFile)
    PicturePath = os.path.join(vidFolder, 'pictures')  # path to store pictures
    GaborPath = os.path.join(vidFolder, 'gabor')  # path to store Gabor features
    Gabor_L_EBPath = os.path.join(vidFolder, 'gabor_l_eyebrow')  # path to store gabor feature of lefteyebrowss
    FramePath = os.path.join(vidFolder,  'frames')
    SheetPath = os.path.join(vidFolder, 'sheet')  # path to store SheetPath
    FeaturesPath = os.path.join(vidFolder, 'features')  # path to store sheets
    ResultPath = os.path.join(vidFolder, 'graph2')

    # new variable for shapePoints for right eye, and left eye, nose, and mouth can easily add more
    shapePoints = [[36, 39, 37, 40], [42, 45, 43, 46], [36, 39, 37, 40], [42, 45, 43, 46], [48, 54, 50, 58],
                   [60, 64, 63, 65]]  # original code

    # new variable for border params
    borders = [[-5, 5, -5, 5], [-5, 5, -5, 5], [-25, 10, -5, 10], [-25, 10, -5, 10], [-30, 10, -20, 20],
               [-10, 10, -10, 10]]  # original

    # new array for paths
    imgPaths = []
    # store path detail as an array
    imgPaths.append(os.path.join(vidFolder, 'right_eye/'))  # path to store right eye
    imgPaths.append(os.path.join(vidFolder, 'left_eye/'))  # path to store left eye
    imgPaths.append(os.path.join(vidFolder, 'right_eyebrow/'))  # right eyebrow
    imgPaths.append(os.path.join(vidFolder, 'left_eyebrow/'))  # Path to store left eyebrow
    imgPaths.append(os.path.join(vidFolder, 'Nose/'))  # Path to Nose
    imgPaths.append(os.path.join(vidFolder, 'Mouth/'))  # Path to Mouth
    ResultPath = os.path.join(vidFolder, 'graph2')
    #imgPaths.append(os.path.join(vidFolder, 'Graph/'))
    # Open the video file
    video_capture = cv2.VideoCapture(Video_path)

    # Get the frame rate from the video file
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in {vidFile}: {total_frames}")
    # Continue with the existing code for frame extraction
    frames2.Frame(detector, predictor, shapePoints, borders, Video_path, PicturePath, imgPaths, GaborPath,
                  SheetPath, FeaturesPath, Gabor_L_EBPath, ResultPath, vidFolder, frame_rate)

    video_capture.release()  # Release the video capture object

    end = time.time()
    duration = end - start
    print(f'Duration for {vidFile}: {duration} seconds')
    # # All above defines folder paths but does not create
    # # add new variables
    # frames2.Frame(detector, predictor, shapePoints, borders, Video_path, PicturePath, imgPaths, GaborPath,
    #               SheetPath, FeaturesPath, Gabor_L_EBPath, ResultPath,vidFolder)
    #
    # end = time.time()
    # duration = end - start
    # print(f'Duration for {vidFile}: {duration} seconds')
    #
    #
