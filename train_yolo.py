from ultralytics import YOLO
import multiprocessing

def main():
    # load baseline model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # train on data downloaded
    model.train(data="C:/Users/micha/PycharmProjects/rubiks/Rubik's-Cube-Detection-4/data.yaml", epochs=100, device=0)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()