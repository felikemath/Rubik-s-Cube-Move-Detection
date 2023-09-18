
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2,
                                       min_hand_presence_confidence=0.2,
                                       min_hand_detection_confidence=0.2)
detector = vision.HandLandmarker.create_from_options(options)

image = mp.Image.create_from_file("moves/sequences/sequence1/frame_0099.png")

def process_img(frame):

    handsop = detector.detect(frame)

    landmarks_dict = {}
    landmarks = []
    print(len(handsop.handedness))
    print(handsop.handedness)
    if len(handsop.handedness) == 2 and handsop.handedness[0][0].display_name == handsop.handedness[1][0].display_name:
        if handsop.handedness[0][0].score > handsop.handedness[1][0].score:
            handsop.handedness[1][0].display_name = "Left" if handsop.handedness[0][0].display_name == 'Right' else 'Right'
        else:
            handsop.handedness[0][0].display_name = "Left" if handsop.handedness[1][
                                                                  0].display_name == 'Right' else 'Right'

    if len(handsop.handedness) > 0:
        for i in range(21):
            if handsop.handedness[0][0].display_name == 'Left':
                landmarks_dict[i] = handsop.hand_landmarks[0][i].x, handsop.hand_landmarks[0][i].y, \
                    handsop.hand_landmarks[0][i].z
            else:
                landmarks_dict[i+21] = handsop.hand_landmarks[0][i].x, handsop.hand_landmarks[0][i].y, \
                handsop.hand_landmarks[0][i].z

    if len(handsop.handedness) > 1:
        for i in range(21):
            if handsop.handedness[1][0].display_name == 'Left':
                landmarks_dict[i] = handsop.hand_landmarks[0][i].x, handsop.hand_landmarks[0][i].y, \
                    handsop.hand_landmarks[0][i].z
            else:
                landmarks_dict[i+21] = handsop.hand_landmarks[0][i].x, handsop.hand_landmarks[0][i].y, \
                handsop.hand_landmarks[0][i].z

    for i in range(42):
        if i in landmarks_dict:
            landmarks.append([landmarks_dict[i][0], landmarks_dict[i][1], landmarks_dict[i][2]])
        else:
            landmarks.append([0, 0, 0])
    return landmarks

print(process_img(image))