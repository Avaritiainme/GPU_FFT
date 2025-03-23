#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Constantes
#define PI              3.1415926535897932384626433832795
#define TWOPI           6.283185307179586476925286766559
#define LOG10_2_INV     3.3219280948873623478703194294948  // 1/log2(10)

// Structure d’un type complexe générique
template <typename T>
struct myComplex {
    T Re, Im;
};

// Vérification que N est une puissance de 2 et calcul de l’exposant M
template <typename T>
bool isPwrTwo(T N, T* M) {
    *M = (T)ceil(log10((double)N) * LOG10_2_INV);
    T NN = (T)pow(2.0, *M);
    if ((NN != N) || (NN == 0))
        return false;
    return true;
}

// Implémentation de la FFT Radix‑2 (en place)
template <typename T>
void rad2FFT(int N, myComplex<T>* x, myComplex<T>* DFT) {
    int M = 0;
    if (!isPwrTwo(N, &M))
        throw "rad2FFT(): N doit être une puissance de 2.";

    int Bsep, Bwidth, P, stage;
    unsigned int iaddr;
    int ii;
    int MM1 = M - 1;
    unsigned int nMax = (unsigned int)N;
    double twoPiN = TWOPI / (double)N;

    myComplex<T>* pDFT = DFT;
    myComplex<T>* pLo;
    myComplex<T>* pHi;
    myComplex<T> WN, temp;

    // Étape 1 : bit-reversal
    for (unsigned int i = 0; i < nMax; i++) {
        iaddr = i;
        ii = 0;
        for (int l = 0; l < M; l++) {
            if (iaddr & 0x01)
                ii += (1 << (MM1 - l));
            iaddr >>= 1;
        }
        DFT[ii].Re = x[i].Re;
        DFT[ii].Im = x[i].Im;
    }

    // Étape 2 : calcul de la FFT en place
    for (stage = 1; stage <= M; stage++) {
        Bsep = (1 << stage);   // pow(2, stage)
        P = N / Bsep;
        Bwidth = Bsep >> 1;    // Bsep / 2

        for (int j = 0; j < Bwidth; j++) {
            double angle = twoPiN * P * j;
            WN.Re = cos(angle);
            WN.Im = -sin(angle);

            for (int HiIndex = j; HiIndex < N; HiIndex += Bsep) {
                pHi = pDFT + HiIndex;
                pLo = pDFT + HiIndex + Bwidth;

                temp.Re = (pLo->Re * WN.Re) - (pLo->Im * WN.Im);
                temp.Im = (pLo->Re * WN.Im) + (pLo->Im * WN.Re);

                pLo->Re = pHi->Re - temp.Re;
                pLo->Im = pHi->Im - temp.Im;
                pHi->Re = pHi->Re + temp.Re;
                pHi->Im = pHi->Im + temp.Im;
            }
        }
    }
}

// IFFT Radix‑2
template <typename T>
void rad2IFFT(int N, myComplex<T>* DFT, myComplex<T>* x) {
    // Inverse le signe des parties imaginaires
    for (int i = 0; i < N; i++) {
        DFT[i].Im = -DFT[i].Im;
    }
    // Applique la même rad2FFT (IFFT = FFT si Im inversée)
    rad2FFT(N, DFT, x);

    // Normalisation
    for (int i = 0; i < N; i++) {
        x[i].Re /= N;
        x[i].Im = -(x[i].Im / N);
    }
}

// Structure d’entête WAV (pour mono/stéréo, 16 bits PCM)
#pragma pack(push, 1)
struct WAVHeader {
    char riff[4];          // "RIFF"
    uint32_t chunkSize;
    char wave[4];          // "WAVE"
    char fmt[4];           // "fmt "
    uint32_t subchunk1Size;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char data[4];          // "data"
    uint32_t dataSize;
};
#pragma pack(pop)

// Lecture du fichier WAV (16 bits, mono/stéréo -> conversion en mono)
bool readWAV(const std::string &filename, std::vector<double> &samples, uint32_t &sampleRate) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Erreur ouverture fichier " << filename << endl;
        return false;
    }

    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));

    if (strncmp(header.riff, "RIFF", 4) != 0 || strncmp(header.wave, "WAVE", 4) != 0) {
        cerr << "Format WAV invalide." << endl;
        return false;
    }
    if (header.audioFormat != 1) {
        cerr << "Seul le format PCM est supporté." << endl;
        return false;
    }
    if (header.numChannels != 1 && header.numChannels != 2) {
        cerr << "Seul le mono ou stéréo est supporté." << endl;
        return false;
    }

    sampleRate = header.sampleRate;
    size_t totalSamples = header.dataSize / (header.bitsPerSample / 8);
    size_t numFrames = (header.numChannels == 2) ? (totalSamples / 2) : totalSamples;

    samples.resize(numFrames);

    if (header.numChannels == 1) {
        for (size_t i = 0; i < numFrames; i++) {
            int16_t s = 0;
            file.read(reinterpret_cast<char*>(&s), sizeof(int16_t));
            samples[i] = static_cast<double>(s) / 32768.0;
        }
    } else {
        for (size_t i = 0; i < numFrames; i++) {
            int16_t left = 0, right = 0;
            file.read(reinterpret_cast<char*>(&left), sizeof(int16_t));
            file.read(reinterpret_cast<char*>(&right), sizeof(int16_t));
            double avg = (static_cast<double>(left) + static_cast<double>(right)) / 2.0;
            samples[i] = avg / 32768.0;
        }
    }
    return true;
}

// Écriture d’un fichier WAV (16 bits PCM, mono)
bool writeWAV(const char* filename, const std::vector<double> &samples, uint32_t sampleRate) {
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Impossible de créer " << filename << endl;
        return false;
    }

    WAVHeader header;
    memcpy(header.riff, "RIFF", 4);
    memcpy(header.wave, "WAVE", 4);
    memcpy(header.fmt,  "fmt ", 4);
    memcpy(header.data, "data", 4);

    header.subchunk1Size = 16;
    header.audioFormat    = 1;   // PCM
    header.numChannels    = 1;   // mono
    header.sampleRate     = sampleRate;
    header.bitsPerSample  = 16;
    header.byteRate       = sampleRate * 2; // mono * 16 bits
    header.blockAlign     = 2;
    header.dataSize       = samples.size() * 2; // 2 octets par échantillon
    header.chunkSize      = 36 + header.dataSize;

    file.write(reinterpret_cast<const char*>(&header), sizeof(WAVHeader));

    // Échantillons en 16 bits, clampés à [-1,1]
    for (double val : samples) {
        double clamped = max(-1.0, min(1.0, val));
        int16_t s = static_cast<int16_t>(clamped * 32767);
        file.write(reinterpret_cast<const char*>(&s), sizeof(int16_t));
    }
    return true;
}

int main(int argc, char** argv) {
    // Mesure du temps total
    auto startTotal = high_resolution_clock::now();

    if (argc < 2) {
        cout << "Usage: " << argv[0] << " input.wav" << endl;
        return 1;
    }

    string inputFile = argv[1];
    vector<double> audioSamples;
    uint32_t sampleRate = 0;

    // 1) Lecture WAV
    if (!readWAV(inputFile, audioSamples, sampleRate)) {
        cerr << "Erreur de lecture WAV." << endl;
        return 1;
    }

    // Durée du fichier audio (en secondes)
    size_t totalSamples = audioSamples.size();
    double totalDuration = static_cast<double>(totalSamples) / static_cast<double>(sampleRate);
    cout << "Durée totale du signal : " << totalDuration << " secondes" << endl;

    // 2) Choix de N = la puissance de 2 la plus proche
    double log2Val = log2((double)totalSamples);
    size_t lower   = 1 << (int)floor(log2Val);
    size_t higher  = 1 << (int)ceil(log2Val);
    size_t N = (totalSamples - lower <= higher - totalSamples) ? lower : higher;
    cout << "Taille FFT (puissance de 2) = " << N << endl;

    // 3) Allocation mémoire (zéro-padding)
    myComplex<double>* inputSignal  = new myComplex<double>[N];
    myComplex<double>* outputFFT    = new myComplex<double>[N];
    myComplex<double>* outputIFFT   = new myComplex<double>[N];

    for (size_t i = 0; i < N; i++) {
        inputSignal[i].Re = (i < totalSamples) ? audioSamples[i] : 0.0;
        inputSignal[i].Im = 0.0;
    }

    // 4) Calcul FFT (CPU)
    auto startFFT = high_resolution_clock::now();
    try {
        rad2FFT((int)N, inputSignal, outputFFT);
    } catch (const char* err) {
        cerr << "Erreur: " << err << endl;
        delete[] inputSignal;
        delete[] outputFFT;
        delete[] outputIFFT;
        return 1;
    }
    auto endFFT   = high_resolution_clock::now();
    auto durFFT   = duration_cast<microseconds>(endFFT - startFFT).count();

    // 5) Calcul IFFT (CPU)
    auto startIFFT = high_resolution_clock::now();
    rad2IFFT((int)N, outputFFT, outputIFFT);
    auto endIFFT   = high_resolution_clock::now();
    auto durIFFT   = duration_cast<microseconds>(endIFFT - startIFFT).count();

    // (Optionnel) Quelques bins
    // for (size_t i = 0; i < min(N, (size_t)10); i++) {
    //     double mag = sqrt(outputFFT[i].Re * outputFFT[i].Re + outputFFT[i].Im * outputFFT[i].Im);
    //     cout << "Bin[" << i << "] = " << mag << endl;
    // }

    // 6) Écriture du fichier WAV de sortie
    vector<double> finalOutput(N);
    for (size_t i = 0; i < N; i++) {
        finalOutput[i] = outputIFFT[i].Re; // partie réelle
    }

    if (!writeWAV("output_voice_cpu.wav", finalOutput, sampleRate)) {
        cerr << "Erreur écriture fichier WAV." << endl;
    }

    // Mesure du temps total
    auto endTotal = high_resolution_clock::now();
    auto durTotal = duration_cast<milliseconds>(endTotal - startTotal).count();

    // Affichage des temps selon le format demandé
    cout << "Temps total : " << durTotal << " ms" << endl;
    cout << "Temps FFT cumulé (CPU) : " << durFFT << " μs" << endl;
    cout << "Temps IFFT cumulé (CPU): " << durIFFT << " μs" << endl;

    // Libération mémoire
    delete[] inputSignal;
    delete[] outputFFT;
    delete[] outputIFFT;

    return 0;
}
