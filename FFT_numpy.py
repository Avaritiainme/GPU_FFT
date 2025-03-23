import numpy as np
import scipy.io.wavfile as wav
import time
import matplotlib.pyplot as plt

#Charger audio
rate, data = wav.read("audio.wav")
if data.ndim > 1:
    data = data[:, 0]
data = data.astype(np.float32)

N = len(data)

# Affichage du signal audio
t = np.linspace(0, N / rate, N, endpoint=False)

# On affiche désormais uniquement 2 sous-graphiques
plt.figure(figsize=(12, 8))

plt.subplot(2,1,1)
plt.plot(t, data, color='green', lw=0.8)
plt.title("Signal avant la FFT (domaine temporel)")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# Calcul de la FFT
start_time = time.time()
fft_result = np.fft.fft(data)
end_time = time.time()
print("Temps d'exécution de la FFT : {:.6f} secondes".format(end_time - start_time))

freqs = np.fft.fftfreq(N, d=1.0/rate)
mask = freqs >= 0
freqs_pos = freqs[mask]
magnitude_original = np.abs(fft_result[mask])

# Affichage de la FFT
plt.subplot(2,1,2)
plt.plot(freqs_pos, magnitude_original, color='blue', lw=0.8)
plt.title("Spectre du signal original (après FFT)")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()

# Application du filtre adaptatif pour extraire la voix
delta = 400  # précision du filtrage
spec_filtered = fft_result.copy()

# Dans ce cas, le spectre est 1D, on définit la bande de fréquences de la voix sur la totalité du spectre
bande_voix = (np.abs(freqs) >= 300) & (np.abs(freqs) <= 4000)
energie_bande = np.abs(fft_result[bande_voix])
if energie_bande.size == 0 or np.max(energie_bande) < 1e-6:
    spec_filtered = fft_result * 0.05  # on atténue pour pas mute les endroits sans voix et les perdre
else:
    idx_max = np.argmax(energie_bande)
    freqs_voix = freqs[bande_voix]
    freq_cible = freqs_voix[idx_max]
    masque_doux = np.exp(-0.5 * ((freqs - freq_cible) / delta) ** 2)
    spec_filtered = fft_result * masque_doux

# Reconstruction du signal (IFFT)
start_time = time.time()
data_time = np.fft.ifft(spec_filtered)
end_time = time.time()
print(f"Temps d'exécution de l'IFFT sur CPU (filtré) : {end_time - start_time:.6f} secondes")
data_time = np.real(data_time)
data_time /= np.max(np.abs(data_time))
data_time_int16 = np.int16(data_time * 32767)

# Enregistrer fichier audio
wav.write("mon_fichier_audio_modifie.wav", rate, data_time_int16)