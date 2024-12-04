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
        self.frame_rate = cfg.frame_rate                # Target frames per second
        self.sample_rate = cfg.sample_rate              # Target audio sample rate
        self.n_mels = cfg.n_mels
        self.hop_length = cfg.hop_length
        self.target_length = cfg.target_length
        self.video_files = self.get_video_files()
        self.data = []


        if cfg.preprocessing: 
            self.preprocess_all()
        else:  
            self.load_preprocessed_data()


    def load_preprocessed_data(self):
        """Load preprocessed frames and Mel spectrogram paths."""
        for video_name in os.listdir(self.output_dir):
            video_output_dir = os.path.join(self.output_dir, video_name)
            if os.path.isdir(video_output_dir):
                frame_files = sorted(f for f in os.listdir(video_output_dir) if f.endswith(".jpg"))
                mel_files = sorted(f for f in os.listdir(video_output_dir) if f.endswith(".npy"))
                for frame_file, mel_file in zip(frame_files, mel_files):
                    frame_path = os.path.join(video_output_dir, frame_file)
                    mel_path = os.path.join(video_output_dir, mel_file)
                    self.data.append((frame_path, mel_path))

    
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

        if os.path.exists(video_output_dir) and os.listdir(video_output_dir):
            print(f"Skipping preprocessing for {video_path}. Preprocessed files found.")
            return []

        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)


        if not self.has_audio_stream(video_path):
            print(f"No audio stream found for {video_path}. Skipping.")
            return []


        # Extract audio
        waveform, sample_rate = self.extract_audio(video_path)

        # Calculate hop_length for Mel spectrogram to align with video frame rate
        frame_duration = 1 / self.frame_rate  # Time per video frame in seconds
        hop_length = int(frame_duration * sample_rate)  # Corresponding audio samples per video frame
        
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=self.n_mels,
            hop_length=hop_length,
        )
        mel_spectrogram = mel_transform(waveform)

        # Calculate total duration
        total_duration = waveform.size(1) / sample_rate


        # TODO: check waveform and mel_spectrogram implementation 

        frames = self.extract_video_frames(video_path, total_duration)

        aligned_data = []
        for i, frame in enumerate(frames):
            # Extract the corresponding spectrogram column
            mel_segment = mel_spectrogram[:, :, i]
            mel_segment = self.pad_or_truncate(mel_segment)

            # Save frame and spectrogram
            frame_path = os.path.join(video_output_dir, f"frame_{i}.jpg")
            mel_path = os.path.join(video_output_dir, f"mel_{i}.npy")

            frame.save(frame_path)
            np.save(mel_path, mel_segment.numpy())

            aligned_data.append((frame_path, mel_path))

        return aligned_data



    def extract_audio(self, video_path):
        """Extract audio from video."""
        
            
        temp_audio_path = os.path.join(self.output_dir, "temp_audio.wav")
        command = f"ffmpeg -i {video_path} -f wav -ar {self.sample_rate} {temp_audio_path} -y"
        os.system(command)
        waveform, sample_rate = torchaudio.load(temp_audio_path)
        os.remove(temp_audio_path)
        return waveform, sample_rate
       
    def has_audio_stream(self, video_path):
        command = [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", video_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout != b"" 


    def extract_video_frames(self, video_path, total_duration):
        """Extract frames from the video."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_per_second = int(fps / self.frame_rate)
        frames = []

        for i in range(int(total_duration * self.frame_rate)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * frames_per_second)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

        cap.release()
        return frames



    def pad_or_truncate(self, mel_spectrogram):
        """Pad or truncate the spectrogram to target length."""
        if mel_spectrogram.size(1) < self.target_length:
            pad_length = self.target_length - mel_spectrogram.size(1)
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, pad_length))
        else:
            mel_spectrogram = mel_spectrogram[:, :self.target_length]
        return mel_spectrogram



    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        frame_path, mel_path = self.data[idx]

        # Load the frame image
        frame = Image.open(frame_path).convert("RGB")
        frame_transform = T.Compose([
            T.Resize((320, 240)),  # Example resize
            T.ToTensor(),          # Convert to tensor
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize (optional)
        ])
        frame = frame_transform(frame)

        # Load the Mel spectrogram
          
    
        mel_spectrogram = np.load(mel_path)
        if mel_spectrogram.shape[0] == 2:  # Stereo case
            mel_spectrogram = mel_spectrogram.mean(axis=0, keepdims=True)  # Convert to mono
        
        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)
        mel_spectrogram = self.pad_or_truncate(mel_spectrogram)

        return frame, mel_spectrogram



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
            # image tensor should match (batch_size, 3, imgsize, imgsize)
            # spectrogram tensor should match (batch_size, n_mels, target_length)
            # TODO: mel is wrong shape
            print(f"Frame shape: {frame.shape}, Mel spectrogram shape: {mel.shape}")
            breakpoint()
        

    main()
