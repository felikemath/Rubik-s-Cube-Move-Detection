from ultralytics import YOLO
import numpy as np
import cv2
import mediapipe as mp

# Load the YOLO model
model = YOLO('runs/detect/train23/weights/best.pt')

# Initialize the webcam
cap = cv2.VideoCapture('IMG_0193.MOV')  # 0 represents the default webcam index

hands = mp.solutions.hands
hands_mesh = hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

draw = mp.solutions.drawing_utils

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    handsop = hands_mesh.process(rgb)

    if handsop.multi_hand_landmarks:
        for i in handsop.multi_hand_landmarks:
            draw.draw_landmarks(frame, i, hands.HAND_CONNECTIONS)
            for landmark_id, landmark in enumerate(i.landmark):
                # Extract landmark number, x, y, and z coordinates
                landmark_num = landmark_id
                landmark_x = landmark.x
                landmark_y = landmark.y
                landmark_z = landmark.z
                print(landmark_num, landmark_x, landmark_y, landmark_z)


    # Perform object detection
    results = model.predict(frame, show=True, conf=0.1)

    # Loop through detected objects and draw bounding boxes
    # for result in results:
    #     print(result)
    #     bbox = result[:4]
    #     label = result[-1]
    #
    #     x1, y1, x2, y2 = map(int, bbox)
    #
    #     # Draw bounding box
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #
    # # Display the frame with bounding boxes
    # cv2.imshow('Object Detection', frame)


    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()