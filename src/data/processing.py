# %% Imports

import os
import cv2  #

import numpy as np
from torchvision import transforms
import torchaudio
import librosa

def downsample_video(input_path, output_path, spatial_scale=0.5, temporal_rate=2):
    cap = cv2.VideoCapture(input_path)

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))

    downsampled_width = int(original_width * spatial_scale)
    downsampled_height = int(original_height * spatial_scale)
    downsampled_fps = int(original_fps / temporal_rate)

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        if frame_count % temporal_rate == 0:
            frame = cv2.resize(frame, (downsampled_width, downsampled_height))
            frames.append(frame)

        frame_count += 1

    cap.release()

    return np.array(frames), downsampled_fps  


def extract_audio_from_video(video_path, downsampled_fps):
    audio_samples, sample_rate = torchaudio.load(video_path, format="avi")
    audio_samples = librosa.resample(audio_samples.numpy(), orig_sr=sample_rate, target_sr=downsampled_fps)
    return audio_samples, downsampled_fps


def preprocess_data(data_dir, output_dir, spatial_scale=0.5, temporal_rate=2, categories=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for category in os.listdir(data_dir):
        

        if category not in categories:
            continue
            
        category_path = os.path.join(data_dir, category)
        output_category_path = os.path.join(output_dir, category)

        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        for video_file in os.listdir(category_path):
            video_path = os.path.join(category_path, video_file)
            print(f"Processing {video_path}")

            # Downsample video (both frames and temporal)
            frames, downsampled_fps = downsample_video(video_path, spatial_scale, temporal_rate)

            breakpoint()

            # Extract and downsample audio
            audio_samples, audio_sample_rate = extract_audio_from_video(video_path, downsampled_fps)

            breakpoint()

            # Save frames and audio
            video_name = os.path.splitext(video_file)[0]
            np.save(os.path.join(output_category_path, f"{video_name}_frames.npy"), frames)
            np.save(os.path.join(output_category_path, f"{video_name}_audio.npy"), audio_samples)

            print(f"Saved downsampled frames and audio for {video_name}")

    print("Data preprocessing complete.")


# %% test
# test_categories = ['ApplyEyeMakeup']
# preprocess_data(data_dir="../../data/UCF-101", output_dir="../../data/processed", spatial_scale=0.5, temporal_rate=2, categories=test_categories)

