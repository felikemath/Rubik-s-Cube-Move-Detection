import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO
import shutil

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2,
                                       min_hand_detection_confidence=0.2,
                                       min_hand_presence_confidence=0.2)
detector = vision.HandLandmarker.create_from_options(options)



class RubiksCubeDataset(Dataset):
    def process_img(frame):

        handsop = detector.detect(frame)

        landmarks_dict = {}
        landmarks = []

        if len(handsop.handedness) == 2 and handsop.handedness[0][0].display_name == handsop.handedness[1][
            0].display_name:
            if handsop.handedness[0][0].score > handsop.handedness[1][0].score:
                handsop.handedness[1][0].display_name = "Left" if handsop.handedness[0][
                                                                      0].display_name == 'Right' else 'Right'
            else:
                handsop.handedness[0][0].display_name = "Left" if handsop.handedness[1][
                                                                      0].display_name == 'Right' else 'Right'

        if len(handsop.handedness) > 0:
            for i in range(21):
                if handsop.handedness[0][0].display_name == 'Left':
                    landmarks_dict[i] = handsop.hand_landmarks[0][i].x, handsop.hand_landmarks[0][i].y, \
                        handsop.hand_landmarks[0][i].z
                else:
                    landmarks_dict[i + 21] = handsop.hand_landmarks[0][i].x, handsop.hand_landmarks[0][i].y, \
                        handsop.hand_landmarks[0][i].z

        if len(handsop.handedness) > 1:
            for i in range(21):
                if handsop.handedness[1][0].display_name == 'Left':
                    landmarks_dict[i] = handsop.hand_landmarks[0][i].x, handsop.hand_landmarks[0][i].y, \
                        handsop.hand_landmarks[0][i].z
                else:
                    landmarks_dict[i + 21] = handsop.hand_landmarks[0][i].x, handsop.hand_landmarks[0][i].y, \
                        handsop.hand_landmarks[0][i].z

        for i in range(42):
            if i in landmarks_dict:
                landmarks.append([landmarks_dict[i][0], landmarks_dict[i][1], landmarks_dict[i][2]])
            else:
                landmarks.append([0, 0, 0])
        return landmarks

    def __init__(self, root_dir, max_tokens, transform=None):
        self.root_dir = root_dir
        self.max_tokens = max_tokens
        self.transform = transform
        self.data_frame = pd.read_csv('moves/metadata.csv')


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sequence_dir = os.path.join(self.root_dir, self.data_frame.iloc[idx]['sequence_path'])
        sequence = os.listdir(sequence_dir)

        tokens_list = []
        deltas = np.zeros(shape=(9, 42, 3))

        pos = {}
        for num_frame, frame_name in enumerate(sequence):
            frame_path = os.path.join(sequence_dir, frame_name)

            frame = mp.Image.create_from_file(frame_path)
            # Process the frame to obtain finger landmarks
            landmarks = RubiksCubeDataset.process_img(frame)

            if num_frame > 0:
                for i in range(42):
                    landmarks[i][0], landmarks[i][1], landmarks[i][2] = landmarks[i][0] - tokens_list[0][i][0], landmarks[i][1] - tokens_list[0][i][1]\
                        , landmarks[i][2] - tokens_list[0][i][2]


            # Calculate deltas
            for i in range(42):
                if landmarks[i][0] == 0 and landmarks[i][1] == 0 and landmarks[i][2] == 0:
                    continue

                if i in pos:
                    deltas[pos[i][3]:num_frame, i, 0] = (landmarks[i][0] - pos[i][0])/(num_frame - pos[i][3])
                    deltas[pos[i][3]:num_frame, i, 1] = (landmarks[i][1] - pos[i][1]) / (num_frame - pos[i][3])
                    deltas[pos[i][3]:num_frame, i, 2] = (landmarks[i][2] - pos[i][2]) / (num_frame - pos[i][3])

                pos[i] = landmarks[i][0], landmarks[i][1], landmarks[i][2], num_frame

            tokens_list.append(landmarks)

        return torch.tensor(tokens_list), torch.tensor(deltas, dtype=torch.float32), self.data_frame.iloc[idx]['label']


class CubeCroppedDataset(Dataset):
    def __init__(self, augment=True):
        self.augment = augment
        self.root_dir = "moves"
        self.data_frame = pd.read_csv('moves/metadata.csv')
        self.model_det = YOLO('runs/detect/train23/weights/best.pt')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.ToTensor()  # Convert the image to a tensor
        ])

        self.data = []

        for idx in range(len(self.data_frame)):
            sequence_path = os.path.join("moves", self.data_frame.iloc[idx]['sequence_path'])
            self.model_det.predict(sequence_path, conf=0.1, save_crop=True, project="object_detection_crop",
                                   name="predict")
            cropped_path = "object_detection_crop/predict/crops/Rubiks-Cube"
            tensor = torch.zeros((len(os.listdir(cropped_path)), 3, 224, 224))
            tensor2 = torch.zeros((len(os.listdir(cropped_path)), 3, 224, 224))
            tensor3 = torch.zeros((len(os.listdir(cropped_path)), 3, 224, 224))
            random_rotation = transforms.RandomRotation(degrees=(-30, 30))
            random_flip = transforms.RandomHorizontalFlip(p=0.5)
            for i, frame in enumerate(os.listdir(cropped_path)):
                spatial_img = Image.open(os.path.join(cropped_path, frame))
                spatial_img2 = random_rotation(spatial_img)
                spatial_img3 = random_flip(spatial_img)
                tensor[i] = self.transform(spatial_img)
                tensor2[i] = self.transform(spatial_img2)
                tensor3[i] = self.transform(spatial_img3)

            tensor = tensor[:5]
            tensor2 = tensor2[:5]
            tensor3 = tensor3[:5]
            self.data.append((tensor, self.data_frame.iloc[idx]['label']))
            if self.augment:
                self.data.append((tensor2, self.data_frame.iloc[idx]['label']))
                self.data.append((tensor3, self.data_frame.iloc[idx]['label']))
            shutil.rmtree("object_detection_crop/predict")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]


class BiCNNDataset(Dataset):
    def __init__(self):
        self.root_dir = "moves"
        self.data_frame = pd.read_csv('moves/metadata.csv')
        self.model_det = YOLO('runs/detect/train23/weights/best.pt')
        self.transform_temporal = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()  # Convert the image to a tensor
        ])
        self.transform_spatial = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.ToTensor()  # Convert the image to a tensor
        ])

    def __len__(self):
        return len(self.data_frame)

    def extract(self, seq_path):
        sequence = os.listdir(seq_path)

        spatial_ind = np.random.randint(2, len(sequence) - 3)

        spatial_img = Image.open(os.path.join(seq_path, sequence[spatial_ind]))
        spatial_tensor = self.transform_spatial(spatial_img)

        start_ind = np.random.randint(0, len(sequence) - 6)
        prev_frame = Image.open(os.path.join(seq_path, sequence[start_ind]))
        prev_frame = self.transform_temporal(prev_frame).numpy().reshape((224, 224, 1))
        optical_flows_np = []  # List to hold numpy optical flows
        for idx in range(start_ind + 1, start_ind + 6):
            cur_frame = Image.open(os.path.join(seq_path, sequence[idx]))
            cur_frame = self.transform_temporal(cur_frame).numpy().reshape((224, 224, 1))
            optical_flow = cv2.calcOpticalFlowFarneback(prev_frame, cur_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            optical_flows_np.append(optical_flow)
            prev_frame = cur_frame

        optical_flows = np.array(optical_flows_np)  # Convert list to numpy ndarray
        optical_flows_tensor = torch.tensor(optical_flows)  # Convert numpy ndarray to tensor

        # for i in range(optical_flows_tensor.shape[0]):
        #      self.visualize_optical_flow(optical_flows_tensor[i].numpy())

        return spatial_tensor, optical_flows_tensor

    def visualize_optical_flow(self, flow):
        flow_img = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

        hsv = np.zeros_like(flow_img)
        hsv[..., 1] = 255

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        plt.imshow(flow_img)
        plt.axis('off')
        plt.show()

    def __getitem__(self, idx):
        sequence_dir = os.path.join(self.root_dir, self.data_frame.iloc[idx]['sequence_path'])
        self.model_det.predict(sequence_dir, conf=0.1, save_crop=True, project="object_detection_crop", name="predict")

        cropped_path = "object_detection_crop/predict/crops/Rubiks-Cube"

        spatial_tensor, optical_flows = self.extract(cropped_path)
        optical_flows = optical_flows.view(10, 224, 224)

        shutil.rmtree("object_detection_crop/predict")
        return spatial_tensor, optical_flows, self.data_frame.iloc[idx]['label']


class SpatialImageDataset(Dataset):

    def __init__(self):
        self.data_frame = pd.read_csv('moves/metadata.csv')
        self.data = []
        self.model_det = YOLO('runs/detect/train23/weights/best.pt')
        self.transform_spatial = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.ToTensor()  # Convert the image to a tensor
        ])

        for idx in range(len(self.data_frame)):
            sequence_path = os.path.join("moves", self.data_frame.iloc[idx]['sequence_path'])
            self.model_det.predict(sequence_path, conf=0.1, save_crop=True, project="object_detection_crop",
                                   name="predict")
            cropped_path = "object_detection_crop/predict/crops/Rubiks-Cube"
            for frame in os.listdir(cropped_path):
                spatial_img = Image.open(os.path.join(cropped_path, frame))
                spatial_tensor = self.transform_spatial(spatial_img)

                self.data.append((spatial_tensor, self.data_frame.iloc[idx]['label']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CombinedDataset(Dataset):
    def __init__(self):
        self.data1 = RubiksCubeDataset('moves', 10, transform=None)
        self.data2 = CubeCroppedDataset(augment=False)



    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return (self.data1[idx][0], self.data2[idx][0]), self.data2[idx][1]



def test_RubiksCubeDataset():
    print('hi')
    root_dir = 'moves'
    max_tokens = 10  # Maximum number of tokens (frames) per move

    transform = None

    dataset = RubiksCubeDataset(root_dir, max_tokens, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    data = dataset[18][0]
    delta_data = dataset[18][1]

    # print(dataset[18][2])

    fig, axes = plt.subplots(3, 2, figsize=(30, 20))
    x_data = data[:, :, 0]
    y_data = data[:, :, 1]
    z_data = data[:, :, 2]
    x_delta_data = delta_data[:, :, 0]
    y_delta_data = delta_data[:, :, 1]
    z_delta_data = delta_data[:, :, 2]

    for landmark_num in range(42):
        axes[0][0].plot(np.linspace(1, 10, 10), x_data[:, landmark_num])
        axes[1][0].plot(np.linspace(1, 10, 10), y_data[:, landmark_num])
        axes[2][0].plot(np.linspace(1, 10, 10), z_data[:, landmark_num])
        axes[0][1].plot(np.linspace(1, 9, 9), x_delta_data[:, landmark_num])
        axes[1][1].plot(np.linspace(1, 9, 9), y_delta_data[:, landmark_num])
        axes[2][1].plot(np.linspace(1, 9, 9), z_delta_data[:, landmark_num])

    axes[0][0].set_title('X Coordinates')
    axes[1][0].set_title('Y Coordinates')
    axes[2][0].set_title('Z Coordinates')
    axes[0][1].set_title('X Delta Coordinates')
    axes[1][1].set_title('Y Delta Coordinates')
    axes[2][1].set_title('Z Delta Coordinates')

    plt.show()



def test_CubeCroppedDataset():
    dataset = CubeCroppedDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(dataset[13][0].shape)


def test_BiCNNDataset():
    dataset = BiCNNDataset()
    print(dataset[6])

if __name__ == '__main__':
    dataset = CombinedDataset()
    print(dataset[0])

