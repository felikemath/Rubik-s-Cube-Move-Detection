from roboflow import Roboflow
import os
from ultralytics import YOLO
import multiprocessing
import cv2
import shutil

def download_dataset():
    rf = Roboflow(api_key="Roboflow API Key")
    project = rf.workspace("michael-song").project("move-detection-cube")
    dataset = project.version(2).download("folder")


def preprocess():
    root_dir = "Move-Detection-Cube-2"
    model = YOLO('runs/detect/train23/weights/best.pt')
    for data_type in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, data_type)
        if not os.path.isdir(folder_path):
            continue

        for class_label in os.listdir(folder_path):
            class_path = os.path.join(folder_path, class_label)

            model.predict(class_path, conf=0.1, save_crop=True, project=folder_path)



# training yolo for classification
def train():
    # load baseline model
    model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

    # train on data downloaded
    model.train(data="C:/Users/micha/PycharmProjects/rubiks/Move-Detection-Cube-2/", epochs=100, device=0)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train()