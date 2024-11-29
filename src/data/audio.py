import torch
from torch.utils.data import Dataset
import torchaudio
import os
import numpy as np
from torch.nn.functional import normalize
import torchaudio.transforms as transforms
import tempfile
import subprocess

from omegaconf import DictConfig
import hydra



class AudioDataset(Dataset):
    def __init__(self, cfg, root_dir):
        self.target_length = cfg.target_length
        self.sample_rate = cfg.sample_rate
        self.bit_rate = cfg.bit_rate
        self.channels = cfg.channels
        self.audio_files = []
        self.output_dir = cfg.output_folder_audio
        self.hop_length = cfg.hop_length
        self.n_mels = cfg.n_mels
  

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not self.is_data_processed():
            self.preprocess_data(root_dir)

        self.load_audio_files()

    def is_data_processed(self):
        """Check if the data is already processed by verifying the existence of .npy files."""
        
        return any(file.endswith(".npy") for file in os.listdir(self.output_dir))



    def preprocess_data(self, root_dir):
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            
            if os.path.isdir(subdir_path) and (not subdir.endswith("_audio")):
                
                for file in os.listdir(subdir_path):
                    if file.endswith(".avi"):
                        
                        video_path = os.path.join(subdir_path, file)
                        if self.has_audio_stream(video_path):
                            print(video_path)
                            waveform, sample_rate = self.extract_audio(video_path)
                            transform = transforms.MelSpectrogram(
                                sample_rate=sample_rate,         # number of samples per second
                                n_mels=self.n_mels,              # number of mel freq bins (number of freq. bands)
                                hop_length=self.hop_length       # time resolution of the spectrogram
                            )

                            # TODO: check MelSpectrogram with images
                            mel_spectrogram = transform(waveform)
                            mel_spectrogram = self.pad_or_truncate(mel_spectrogram)
                            np.save(os.path.join(self.output_dir, f"{os.path.splitext(file)[0]}.npy"), mel_spectrogram.numpy())


    def load_audio_files(self):
        """Load the list of preprocessed .npy files."""
        for file in os.listdir(self.output_dir):
            self.audio_files.append(os.path.join(self.output_dir, file))


    def extract_audio(self, video_path):
        """Extract audio from .avi video as waveform using FFmpeg if audio is present."""
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_wav:
            command = [
                "ffmpeg",
                "-i", video_path,
                "-f", "wav", 
                "-ab", self.bit_rate,
                "-ac", str(self.channels),
                "-ar", str(self.sample_rate),
                "-vn", tmp_wav.name,
                "-y"
            ]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            # TODO: check with librosa if wav is extracted correctly
            # use wav files for mel spectrogram extraction
            # mel2audio is converting it back

            return torchaudio.load(tmp_wav.name)


    def pad_or_truncate(self, mel_spectrogram):
        # print(mel_spectrogram.size(2))
        # TODO: check if that works correctly as well
        if mel_spectrogram.size(2) < self.target_length:
            pad_length = self.target_length - mel_spectrogram.size(2)
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, pad_length))
        else:
            mel_spectrogram = mel_spectrogram[:, :, :self.target_length]
        return mel_spectrogram
        

    
    def has_audio_stream(self, video_path):
        command = [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", video_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout != b"" 

    def augment(self, spectrogram):
        noise = torch.randn_like(spectrogram) * 0.05
        return spectrogram + noise

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        mel_spectrogram = np.load(self.audio_files[idx])
        mel_spectrogram = torch.tensor(mel_spectrogram)
        positive = self.augment(mel_spectrogram)
        return mel_spectrogram.squeeze(0), positive.squeeze(0)


@hydra.main(config_path="../../conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):

    root_dir = cfg.dataset.input_folders_small if cfg.data_subset else cfg.dataset.input_folders
    AudioDataset(cfg.dataset, root_dir)
    

if __name__ == "__main__":
    main()