import os
import pandas as pd

annotation_path = "annotations_augmented"
videos_path = "../../ADFES_three"
points_dic = {'Anger': [-0.4, 0.8], 'Contempt': [-0.57, 0.66], 'Disgust': [-0.68, 0.5],
              'Embarrass': [-0.31, -0.6], 'Fear': [-0.11, 0.79], 'Joy': [0.95, 0.12], 'Neutral': [0, 0],
              'Pride': [0.31, 0.55], 'Sadness': [-0.5, -0.86], 'Surprise': [0.38, 0.92]}

if not os.path.exists(annotation_path):
    os.makedirs(annotation_path)

def get_emotion(video_name):
    # Extract emotion from video name
    words = video_name.split('-')
    for word in words:
        if word in points_dic:
            return word
    return None


def generateAnnotations(num):
    videos = os.listdir(videos_path)
    for video in videos:
        video_name, extension = os.path.splitext(video)
        emotion = get_emotion(video_name)

        if emotion:
            v_value = points_dic[emotion][0]
            a_value = points_dic[emotion][1]

            annotation_df = pd.DataFrame(columns=["valence", "arousal"])

            for i in range(num):
                frame_id = f"{video_name}_{i}"  # Create frame ID using video name and index
                annotation_df = annotation_df.append({'valence': v_value, 'arousal': a_value}, ignore_index=True)

            annotation_df.to_csv(os.path.join(annotation_path, video_name + '.csv'), index=False,
                                 columns=["valence", "arousal"])
        else:
            print(f"Emotion not found in '{video_name}', skipping.")


if __name__ == '__main__':
    print(points_dic['Disgust'][1])
    generateAnnotations(104)
