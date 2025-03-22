import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import scipy.io.wavfile as wav

#Charger audio
rate, data = wav.read("audiocut.wav")
if data.ndim > 1:
    data = data[:, 0]
data = data.astype(np.float32)

N = len(data)
t = np.linspace(0, N / rate, N, endpoint=False)

#Affichage
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t, data, color='green', lw=0.8)
plt.title("Signal avant la FFT (domaine temporel)")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.grid(True)



device = torch.device("cuda")
#Convert to GPU
data_tensor = torch.from_numpy(data).to(device)

# Calcul de la FFT
start_time = time.time()
fft_result = torch.fft.fft(data_tensor)
torch.cuda.synchronize()  
end_time = time.time()
print(f"Temps d'exécution de la FFT sur GPU : {end_time - start_time:.6f} secondes")

# Récupération CPU
fft_result_cpu = fft_result.cpu().numpy()

# Calcul des fréquences
freqs = np.fft.fftfreq(N, d=1.0/rate)
mask = freqs >= 0 
freqs_pos = freqs[mask]
magnitude_original = np.abs(fft_result_cpu[mask])
#Affichage
plt.subplot(2, 1, 2)
plt.plot(freqs_pos, magnitude_original, color='blue', lw=0.8)
plt.title("Spectre du signal original (après FFT GPU)")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Reconstruction (IFFT sur GPU) et sauvegarde
ifft_result = torch.fft.ifft(fft_result)
torch.cuda.synchronize()

# Récupération sur CPU
ifft_result_cpu = ifft_result.real.cpu().numpy()

ifft_result_cpu /= np.max(np.abs(ifft_result_cpu))
data_time_int16 = np.int16(ifft_result_cpu * 32767)

# Enregistrer fichier audio
wav.write("mon_fichier_audio_modifie_gpu.wav", rate, data_time_int16)
