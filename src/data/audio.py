import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import numpy as np
from torch.nn.functional import normalize
import torchaudio.transforms as transforms
import tempfile
import subprocess



class AudioDataset(Dataset):
    def __init__(self, root_dir, target_length=64, sample_rate=16000, bit_rate="64k", channels=1):
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.bit_rate = bit_rate
        self.channels = channels
        self.audio_files = []


        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path) and not subdir.endswith("_audio"):
                print('done')
                for file in os.listdir(subdir_path):
                    if file.endswith(".avi"):
                        video_path = os.path.join(subdir_path, file)
                        if self.has_audio_stream(video_path): 
                            self.audio_files.append(video_path)

        # Define a Mel Spectrogram transform
        self.transform = transforms.MelSpectrogram(
            sample_rate=sample_rate, 
            n_mels=64,
            hop_length=512
        )

        

    def __len__(self):
        return len(self.audio_files)


    def __getitem__(self, idx):
        video_path = self.audio_files[idx]
        waveform = self.extract_audio(video_path)
        
        # Convert to Mel Spectrogram
        mel_spectrogram = self.transform(waveform)

        # Ensure the spectrogram has the target length by padding or truncating
        if mel_spectrogram.size(2) < self.target_length:
            pad_length = self.target_length - mel_spectrogram.size(2)
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, pad_length))
        else:
            mel_spectrogram = mel_spectrogram[:, :, :self.target_length]

        positive = self.augment(mel_spectrogram)
        return mel_spectrogram.squeeze(0), positive.squeeze(0)

    def has_audio_stream(self, video_path):

        command = [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", video_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout != b"" 

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

            waveform, sample_rate = torchaudio.load(tmp_wav.name)
            return waveform

    def augment(self, spectrogram):
        noise = torch.randn_like(spectrogram) * 0.05
        return spectrogram + noise