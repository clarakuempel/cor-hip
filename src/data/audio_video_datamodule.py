import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import cv2
from torchvision import transforms as T
from PIL import Image
import pytorch_lightning as pl
import subprocess



class VideoAudioDataset(Dataset):

    def __init__(self, cfg, root_dir):
        self.root_dir = root_dir
        self.output_dir = cfg.output_folder
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.frame_rate = cfg.frame_rate  
        self.sample_rate = cfg.sample_rate
        self.n_mels = cfg.n_mels
        self.hop_length = cfg.hop_length
        self.target_length = cfg.target_length
        self.video_files = self.get_video_files()
        self.data = []
        self.preprocess_all()
        
    
    def get_video_files(self):
        """List all .avi files in the root directory and its subdirectories."""
        video_files = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                video_files.append(os.path.join(root, file))
        return video_files


    
    def preprocess_all(self):
        """Process all videos and store aligned video and audio frames."""
        for video_path in self.video_files:
            self.data.extend(self.process_video(video_path))

    
    def process_video(self, video_path):
        """Process a single video to extract frames and aligned audio spectrograms."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_dir, video_name)

        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        waveform, sample_rate = self.extract_audio(video_path)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
        )
        mel_spectrogram = mel_transform(waveform)
        # TODO: check waveform and mel_spectrogram implementation visually
        
       

    def extract_audio(self, video_path):
        """Extract audio from video."""
        if self.has_audio_stream(video_path):
            
            temp_audio_path = os.path.join(self.output_dir, "temp_audio.wav")
            command = f"ffmpeg -i {video_path} -f wav -ar {self.sample_rate} {temp_audio_path} -y"
            os.system(command)
            waveform, sample_rate = torchaudio.load(temp_audio_path)
            os.remove(temp_audio_path)
            return waveform, sample_rate
        else:
            return []

    def has_audio_stream(self, video_path):
        command = [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", video_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout != b"" 

        







    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]



class VideoAudioModule(pl.LightningDataModule):
    def __init__(self, cfg, root_dir):
        super().__init__()
        self.cfg = cfg
        self.root_dir = root_dir
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers

    def setup(self, stage=None):
        self.dataset = VideoAudioDataset(self.cfg.dataset, self.root_dir)

    def train_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )



if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="../../conf", config_name="config", version_base="1.1")
    def main(cfg: DictConfig):
        root_dir = cfg.dataset.input_folders_small if cfg.data_subset else cfg.dataset.input_folders
        data_module = VideoAudioModule(cfg, root_dir)
        data_module.setup()
        for batch in data_module.train_dataloader():
            frame, mel = batch
            print(frame.shape, mel.shape)
            break

    main()