import numpy as np
import matplotlib.pyplot as plt


npy_file_path = "/home/ckuempel/cor-hip/data/UCF-101-audio/v_PlayingCello_g13_c01.npy"



# Load the .npy file
spectrogram_seq = np.load(npy_file_path)

breakpoint()

# Plot the spectrogram
plt.figure(figsize=(10, 6))
plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label="Amplitude")
plt.title("Spectrogram")
plt.xlabel("Time (frames)")
plt.ylabel("Frequency (bins)")
plt.tight_layout()
plt.savefig('spectrogram.png')
plt.show()