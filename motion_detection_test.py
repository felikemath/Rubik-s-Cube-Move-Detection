from ultralytics import YOLO
import os
import shutil
from PIL import Image

# Load the YOLO model
model_cls = YOLO('runs/classify/train2/weights/best.pt')
#
# results = model.predict("C:/Users/micha/PycharmProjects/rubiks/moves/sequences/sequence13/frame_0067.png", show=True)

model_det = YOLO('runs/detect/train23/weights/best.pt')

def classify_img(img_path):
    model_det.predict(img_path, conf=0.1, save_crop=True, project="object_detection_crop", name="predict")
    pred_path = "object_detection_crop/predict/crops/Rubiks-Cube" + "/" + os.path.basename(img_path)
    pred_path = pred_path[:len(pred_path)-3] + "jpg"
    print(pred_path)
    results = model_cls.predict(pred_path)
    print(results[0].probs)
    shutil.rmtree("object_detection_crop/predict")

classify_img("C:/Users/micha/PycharmProjects/rubiks/moves/sequences/sequence13/frame_0067.png")