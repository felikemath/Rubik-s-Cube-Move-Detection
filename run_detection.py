import cv2
import numpy as np
import os
from ultralytics import YOLO
import shutil
from MovesDataset import RubiksCubeDataset
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
from move_detection import LSTMMovePredictor

# load classification model
model_cls = YOLO('runs/classify/train4/weights/best.pt')

# load object detection model
model_det = YOLO('runs/detect/train23/weights/best.pt')


def check_motion(img):
    cropped_result = model_det.predict(img, conf=0.1, save_crop=True, project="object_detection_crop", name="predict")
    if len(cropped_result[0].boxes.data) == 0:
        return
    pred_path = "object_detection_crop/predict/crops/Rubiks-Cube/image0.jpg"
    results = model_cls.predict(pred_path)
    shutil.rmtree("object_detection_crop/predict")

    if results[0].probs.top1 == 0:
        return True
    else:
        return False


def predict_move(sequence_path):
    sequence = os.listdir(sequence_path)
    tokens_list = []
    deltas = np.zeros(shape=(9, 42, 3))

    pos = {}
    for num_frame, frame_name in enumerate(sequence):
        frame_path = os.path.join(sequence_path, frame_name)

        frame = mp.Image.create_from_file(frame_path)
        # Process the frame to obtain finger landmarks
        landmarks = RubiksCubeDataset.process_img(frame)

        if num_frame > 0:
            for i in range(42):
                landmarks[i][0], landmarks[i][1], landmarks[i][2] = landmarks[i][0] - tokens_list[0][i][0], \
                                                                    landmarks[i][1] - tokens_list[0][i][1] \
                    , landmarks[i][2] - tokens_list[0][i][2]

        # Calculate deltas
        for i in range(42):
            if landmarks[i][0] == 0 and landmarks[i][1] == 0 and landmarks[i][2] == 0:
                continue

            if i in pos:
                deltas[pos[i][3]:num_frame, i, 0] = (landmarks[i][0] - pos[i][0]) / (num_frame - pos[i][3])
                deltas[pos[i][3]:num_frame, i, 1] = (landmarks[i][1] - pos[i][1]) / (num_frame - pos[i][3])
                deltas[pos[i][3]:num_frame, i, 2] = (landmarks[i][2] - pos[i][2]) / (num_frame - pos[i][3])

            pos[i] = landmarks[i][0], landmarks[i][1], landmarks[i][2], num_frame

        tokens_list.append(landmarks)

    tokens_list = [tokens_list]

    data = torch.tensor(tokens_list)
    # torch.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2]))
    print(data.shape)

    # load in prediction model
    input_dim = 3  # Dimensionality of each landmark
    hidden_dim = 128  # Number of hidden units in LSTM
    num_layers = 5  # Number of LSTM layers
    output_dim = 3  # Number of possible cube move classes
    num_landmarks = 42

    model_pred = LSTMMovePredictor(input_dim, hidden_dim, num_layers, output_dim, num_landmarks)
    model_pred.load_state_dict(torch.load("LSTM_model/model.pt"))
    model_pred.eval()

    index_to_label = {0: 'U', 1: 'R', 2: 'F'}

    output = model_pred(data)
    pred_index = torch.argmax(output, dim=1)

    # delete sequence folder
    # shutil.rmtree(sequence_path)

    return index_to_label[int(pred_index[0])]


def run(input_video_path):
    # Open the video file
    video_capture = cv2.VideoCapture(input_video_path)

    # Check if the video is opened successfully
    if not video_capture.isOpened():
        print("Error: Unable to open the video file.")
        return

    in_motion = False
    frame_count = 0

    motion_list = []
    move_list = []

    detection_folder = "detection_frames"

    sequence_count = 0
    sequence_path = None
    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_motion = check_motion(image)

        if current_motion == None:
            continue

        if current_motion != in_motion:
            if current_motion:
                print("Cube is in motion! Current frame: %d" % frame_count)
                sequence_count += 1
                sequence_path = os.path.join(detection_folder, "sequence{0}".format(sequence_count))
                os.mkdir(sequence_path)


            else:
                print("Cube is stationary. Current frame: %d" % frame_count)
                pred_move = predict_move(sequence_path)
                move_list.append(pred_move)
                print("The predicted move for sequence %d is %s" % (sequence_count, pred_move))

            in_motion = current_motion

        if in_motion:
            motion_list.append(frame_count)
            if len(os.listdir(sequence_path)) < 10:
                frame_name = f"frame_{frame_count:04d}.png"
                cv2.imwrite(os.path.join(sequence_path, frame_name), frame)

        frame_count += 1

    print(move_list)



if __name__ == '__main__':
    run("IMG_0201.MOV")
    print("Finished!")
