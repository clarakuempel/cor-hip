import cv2
import os

def downsample_video(input_path, output_path, spatial_scale=0.5, temporal_rate=2):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the original video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set the new video properties
    downsampled_width = int(original_width * spatial_scale)
    downsampled_height = int(original_height * spatial_scale)
    downsampled_fps = int(original_fps / temporal_rate)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, downsampled_fps, (downsampled_width, downsampled_height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every `temporal_rate` frames
        if frame_count % temporal_rate == 0:
            # Resize (spatial downsample) the frame
            frame = cv2.resize(frame, (downsampled_width, downsampled_height))
            out.write(frame)

        frame_count += 1

    # Release the video objects
    cap.release()
    out.release()
    print(f"Downsampled video saved to {output_path}")

# Define the input and output paths
input_video_path = "/home/ckuempel/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"
output_video_path = "/home/ckuempel/UCF-101-exp/v_ApplyEyeMakeup_g01_c01_downsampled.avi"

# Run the downsampling
downsample_video(input_video_path, output_video_path, spatial_scale=0.5, temporal_rate=2)