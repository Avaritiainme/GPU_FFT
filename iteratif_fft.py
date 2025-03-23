import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import istft

# Implémentation d'une FFT itérative (à remplacer par ton algorithme si disponible)
def fft_iterative(x):
    # Pour l'exemple, on utilise ici la FFT de NumPy.
    return np.fft.fft(x)

def fft_glissante_iterative(x, fs, nperseg=1024, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2

    step = nperseg - noverlap
    windows = []
    times = []

    for start in range(0, len(x) - nperseg + 1, step):
        segment = x[start:start + nperseg]
        segment = segment * np.hanning(nperseg)
        result = fft_iterative(segment)
        windows.append(result)
        times.append(start / fs)

    spectrogram = np.array(windows).T  
    frequencies = np.fft.fftfreq(nperseg, d=1/fs)  
    frequencies = frequencies[frequencies >= 0]  # vire les negatives
    spectrogram = spectrogram[:len(frequencies), :] 
    return frequencies, times, spectrogram

fs, data = wavfile.read("audio.wav")
if len(data.shape) > 1:
    data = data[:, 0]  # 

# Mesure du temps d'exécution de la FFT glissante
start_time = time.time()
freqs, times, spec = fft_glissante_iterative(data, fs, nperseg=1024)
end_time = time.time()
print("Temps d'exécution de la FFT glissante (iterative) : {:.6f} secondes".format(end_time - start_time))

spec_filtered = np.zeros_like(spec, dtype=complex)
delta = 400  # precision du filtrage 

for i in range(spec.shape[1]):
    colonne = spec[:, i]
    bande_voix = (freqs >= 300) & (freqs <= 4000)
    energie_bande = np.abs(colonne[bande_voix])

    if energie_bande.size == 0 or np.max(energie_bande) < 1e-6:
        colonne_filtree = colonne * 0.05  # on attenue pour pas mute les endroits sans voix et les perdre
    else:
        idx_max = np.argmax(energie_bande)
        freqs_voix = freqs[bande_voix]
        freq_cible = freqs_voix[idx_max]
        masque_doux = np.exp(-0.5 * ((freqs - freq_cible) / delta) ** 2)
        colonne_filtree = colonne * masque_doux

    spec_filtered[:, i] = colonne_filtree

# on affiche le spec pour voir si c'est sensé 
plt.imshow(10 * np.log10(np.abs(spec_filtered) + 1e-10), aspect='auto', origin='lower',
           extent=[times[0], times[-1], freqs[0], freqs[-1]])
plt.xlabel("Temps")
plt.ylabel("Fréquence (Hz)")
plt.title("Spectrogramme après filtrage")
plt.colorbar(label="Amplitude (dB)")
plt.show()

# Reconstruction audio
start_time = time.time()
_, voice_only = istft(spec_filtered, fs)
end_time = time.time()
print("Temps d'exécution de l'IFFT (ISTFT) : {:.6f} secondes".format(end_time - start_time))
voice_only = voice_only / np.max(np.abs(voice_only))  
wavfile.write("voix_extraite_test.wav", fs, (voice_only * 32767).astype(np.int16))

print("finito pipo")
