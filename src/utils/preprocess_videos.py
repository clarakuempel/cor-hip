import os
import cv2
import hydra
from omegaconf import DictConfig


def extract_frames(video_path, output_dir):
    """
    Extract frames from a video and save them as sequential images.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"frame_{frame_id:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_id += 1

    cap.release()
    print(f"Extracted {frame_id} frames from {video_path}")

def preprocess_videos(data_dir, output_dir):
    """
    Process all videos in the dataset and extract frames.
    """
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            continue

        for video in os.listdir(category_path):
            
            
            video_path = os.path.join(category_path, video)
            video_name = os.path.splitext(video)[0]
            output_path = os.path.join(output_dir, category, video_name)

            extract_frames(video_path, output_path)


@hydra.main(config_path="../../conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    import os
    
    data_dir = cfg.dataset.input_folders_small if cfg.data_subset else cfg.dataset.input_folders # os.path.abspath()
    output_dir = os.path.abspath(cfg.dataset.output_folder)
    preprocess_videos(data_dir, output_dir)
    

if __name__ == "__main__":
    main()