import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import istft
from numba import cuda, float32, complex64

@cuda.jit #notre seul kernel, c'est la fonction qu'on passe sur le gpu, la fft glissante qui travaille sur nos segments de sons, les "windows"
def fft_kernel(signal, output, nperseg, step):
    i = cuda.grid(1)
    N = nperseg
    if i * step + N > signal.shape[0]:
        return

    bits = 0
    n_temp = N
    while n_temp > 1:
        n_temp >>= 1
        bits += 1

    # reserve de la memoire locale pour stocker partie reelle et imaginaire, ici jusqu'a 4096 echantillons par window
    real = cuda.local.array(4096, dtype=float32)
    imag = cuda.local.array(4096, dtype=float32)

    # on inverse l'ordre des bits pour preparer au diviser pour regner de cooley turkey 
    for j in range(N):
        idx = 0
        n = j
        for b in range(bits):
            if n & 1:
                idx |= 1 << (bits - 1 - b)
            n >>= 1
        window_val = 0.5 - 0.5 * math.cos(2 * math.pi * j / (N - 1))  # Hanning pour lisser les coupures entre window 
        real[idx] = signal[i * step + j] * window_val
        imag[idx] = 0.0

    # Cooley-Tukey en iteratif
    size = 2
    while size <= N:
        half = size // 2
        angle_step = -2 * math.pi / size
        for start in range(0, N, size):
            for k in range(half):
                angle = angle_step * k
                wr = math.cos(angle)
                wi = math.sin(angle)
                a_real = real[start + k]
                a_imag = imag[start + k]
                b_real = real[start + k + half]
                b_imag = imag[start + k + half]

                tr = wr * b_real - wi * b_imag
                ti = wr * b_imag + wi * b_real

                real[start + k] = a_real + tr
                imag[start + k] = a_imag + ti
                real[start + k + half] = a_real - tr
                imag[start + k + half] = a_imag - ti
        size *= 2

    for j in range(N):
        output[j, i] = complex64(real[j] + 1j * imag[j])

# charge l'audio 
fs, data = wavfile.read("audiocut.wav")
if len(data.shape) > 1:
    data = data[:, 0]  # passe en mono 

# listing des paramètres
nperseg = 4096 # nb de segment, lié a lignes 23-24 parce que c'est la taille d'une window donc d'un thread
noverlap = nperseg // 2 #chaque fenetre se chevauche de moitie
step = nperseg - noverlap
n_windows = (len(data) - nperseg) // step + 1

# on envoie sur le gpu
signal_gpu = cuda.to_device(data.astype(np.float32))
spec_gpu = cuda.device_array((nperseg, n_windows), dtype=np.complex64)

threads_per_block = 1024
blocks_per_grid = (n_windows + threads_per_block - 1) // threads_per_block

# on execute nos kernels
fft_kernel[blocks_per_grid, threads_per_block](signal_gpu, spec_gpu, nperseg, step)
cuda.synchronize()

# on recupere le resultat
spec = spec_gpu.copy_to_host()
freqs = np.fft.fftfreq(nperseg, d=1/fs) 
freqs = freqs[freqs >= 0] #frequences positives seulement parce que fft crée les deux
spec = spec[:len(freqs), :]
times = np.arange(n_windows) * step / fs

# prepare le tableau pour le spectrogramme filtre
spec_filtered = np.zeros_like(spec, dtype=complex)

# nb harmonique pour le second filtre
n_harmonics = 3

for i in range(spec.shape[1]):
    colonne = spec[:, i]
    bande_voix = (freqs >= 300) & (freqs <= 2500) # filtre passe bande autour de la voix 
    energie_bande = np.abs(colonne[bande_voix]) # releve l'amplitude / energie a cet endroit 

    if energie_bande.size == 0 or np.max(energie_bande) < 1e-6: # si pas de voix, on attenue pour pas cut completment et avoir des trous a la reconstruction
        colonne_filtree = colonne * 0.05
    else:
        idx_max = np.argmax(energie_bande)
        freqs_voix = freqs[bande_voix]
        freq_fondamentale = freqs_voix[idx_max] # la ou l'energie est max on considere qu'on recupere la freq fondamentale
        
        
        delta_base = 50 + (freq_fondamentale * 0.1)  # largeur du filtre pour la fondamentale
        
        # on initialise le masque comme pour spec mais a la dim de freqs
        masque_total = np.zeros_like(freqs, dtype=float)
        
        # puis on boucle sur les x harmoniques choisies 
        for h in range(1, n_harmonics + 1):
            freq_harmonique = freq_fondamentale * h
            
            # pas d'harmoinique au dela de notre freq max 
            if freq_harmonique > freqs[-1]:
                break
                
            # plus une hamronique est élevé plus elle est instable donc on elargit progresivement notre gausienne pour la capturer pleinement
            delta_harmonique = delta_base * (0.8 + 0.2 * h)
            
            # plus une harmonique est haute plus elle est naturellement basse en volume donc on lui donne un poids moindre pour pas donner trop d'importance au bruit ainsi ajoute
            amplitude_harmonique = 1.0 / h
            
            # applique le filtre gaussien sur l'harmonique ainsi traitee
            masque_harmonique = amplitude_harmonique * np.exp(-0.5 * ((freqs - freq_harmonique) / delta_harmonique) ** 2)
            masque_total = np.maximum(masque_total, masque_harmonique)
        
        # on applique le masque 
        colonne_filtree = colonne * masque_total

    spec_filtered[:, i] = colonne_filtree

# on affiche le spectrogramme
plt.figure(figsize=(10, 6))
plt.imshow(10 * np.log10(np.abs(spec_filtered) + 1e-10), aspect='auto', origin='lower',
           extent=[times[0], times[-1], freqs[0], freqs[-1]])
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.title("Spectrogramme adaptatif avec préservation des harmoniques")
plt.colorbar(label="Amplitude (dB)")
plt.tight_layout()
plt.savefig("spectrogramme_harmonique.png")

# reconstruction du fichier audio 
_, voice_only = istft(spec_filtered, fs)
voice_only = voice_only / np.max(np.abs(voice_only))
wavfile.write("voix_extraite_harmonique.wav", fs, (voice_only * 32767).astype(np.int16))

print("Extraction harmonique terminée. Fichier : voix_extraite_harmonique.wav")