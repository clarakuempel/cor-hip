import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import numpy as np
from torch.nn.functional import normalize
import torchaudio.transforms as transforms


class AudioDataset(Dataset):
    def __init__(self, data_dir, target_length=64):
        self.data_dir = data_dir
        self.audio_files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]
        self.target_length = target_length
        
        # Define a spectrogram transform
        self.transform = transforms.MelSpectrogram(
            sample_rate=16000, 
            n_mels=64,          # Match GRU's input_size of 64
            hop_length=512      # Adjust as necessary
        )

    def __len__(self):
        return len(self.audio_files)


    def __getitem__(self, idx):
        audio_path = os.path.join(self.data_dir, self.audio_files[idx])
        
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Ensure the audio has the correct sample rate
        if sample_rate != 16000:
            resample = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resample(waveform)

        # Convert to Mel Spectrogram
        mel_spectrogram = self.transform(waveform)
        
        # Ensure the spectrogram has the required length by padding or truncating
        if mel_spectrogram.size(2) < self.target_length:
            pad_length = self.target_length - mel_spectrogram.size(2)
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, pad_length))
        else:
            mel_spectrogram = mel_spectrogram[:, :, :self.target_length]
        
        # Apply augmentation to create a positive example
        positive = self.augment(mel_spectrogram)

        return mel_spectrogram.squeeze(0), positive.squeeze(0)

    # def process_audio(self, audio, target_length=16000):
    #     """Pad or truncate audio to a target length."""
    #     if audio.size(1) < target_length:
    #         pad_length = target_length - audio.size(1)
    #         audio = torch.nn.functional.pad(audio, (0, pad_length))
    #     else:
    #         audio = audio[:, :target_length]
    #     return audio

    def augment(self, spectrogram):
        """Example augmentation: Add small random noise."""
        noise = torch.randn_like(spectrogram) * 0.05
        return spectrogram + noise