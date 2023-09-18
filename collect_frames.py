import cv2
import os

def extract_frames_from_video(output_folder_path, input_video_path=0):
    # Open the video file
    video_capture = cv2.VideoCapture(input_video_path)

    # Check if the video is opened successfully
    if not video_capture.isOpened():
        print("Error: Unable to open the video file.")
        return

    os.makedirs(output_folder_path, exist_ok=True)

    frame_count = 0
    added_frames = 0

    while added_frames < 2000:
        # Read a frame from the video
        ret, frame = video_capture.read()

        # Break the loop if no frame is captured (end of video)
        if not ret:
            break

        # Save the frame as an image in the output folder
        if frame_count % 1 == 0:
            output_path = os.path.join(output_folder_path, f"frame_{frame_count:04d}.png")
            cv2.imwrite(output_path, frame)
            added_frames += 1

        frame_count += 1

    # Release the video capture object
    video_capture.release()

    print(f"Frames extracted: {frame_count}")
    print("Extraction completed successfully!")









