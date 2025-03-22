#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <chrono>  // Pour la mesure du temps

using namespace std;
using namespace std::chrono;

// Définitions constantes
#define PI       3.1415926535897932384626433832795    
#define TWOPI    6.283185307179586476925286766559     
#define log10_2_INV 3.3219280948873623478703194294948 

// Définition d'un type complexe générique
template <typename T>
struct complex {
    T Re, Im;
};

// Vérification que N est une puissance de 2 et calcul de l'exposant M
template <typename T>
bool isPwrTwo(T N, T *M){
    *M = (T)ceil(log10((double)N) * log10_2_INV);
    T NN = (T)pow(2.0, *M);
    if ((NN != N) || (NN == 0))
        return false;
    return true;
}

// Implémentation de la FFT par Radix-2 (en place)
template <typename T>
void rad2FFT(int N, complex<T> *x, complex<T> *DFT){
    int M = 0;
    if (!isPwrTwo(N, &M))
        throw "rad2FFT(): N doit être une puissance de 2.";
    
    int BSep, BWidth, P, j, stage;
    int HiIndex;
    unsigned int iaddr;
    int ii;
    int MM1 = M - 1;
    unsigned int i;
    int l;
    unsigned int nMax = (unsigned int)N;
    double TwoPi_N = TWOPI / (double)N;
    double TwoPi_NP;
    complex<T> WN, TEMP;
    complex<T> *pDFT = DFT;
    complex<T> *pLo, *pHi, *pX;

    // Réorganisation par renversement de bits
    for (i = 0; i < nMax; i++, DFT++) {
        pX = x + i;
        ii = 0;
        iaddr = i;
        for (l = 0; l < M; l++) {
            if (iaddr & 0x01)
                ii += (1 << (MM1 - l));
            iaddr >>= 1;
            if (!iaddr)
                break;
        }
        DFT = pDFT + ii;
        DFT->Re = pX->Re;
        DFT->Im = pX->Im;
    }

    // Calcul de la FFT
    for (stage = 1; stage <= M; stage++) {
        BSep = (int)(pow(2, stage));
        P = N / BSep;
        BWidth = BSep / 2;
        TwoPi_NP = TwoPi_N * P;
        for (j = 0; j < BWidth; j++) {
            WN.Re = cos(TwoPi_N * P * j);
            WN.Im = -sin(TwoPi_N * P * j);
            for (HiIndex = j; HiIndex < N; HiIndex += BSep) {
                pHi = pDFT + HiIndex;
                pLo = pDFT + HiIndex + BWidth;
                TEMP.Re = (pLo->Re * WN.Re) - (pLo->Im * WN.Im);
                TEMP.Im = (pLo->Re * WN.Im) + (pLo->Im * WN.Re);
                pLo->Re = pHi->Re - TEMP.Re;
                pLo->Im = pHi->Im - TEMP.Im;
                pHi->Re = pHi->Re + TEMP.Re;
                pHi->Im = pHi->Im + TEMP.Im;
            }
        }
    }
}

// Structure WAV modifiée (mono ou stéréo accepté)
#pragma pack(push, 1)
struct WAVHeader {
    char riff[4];
    uint32_t chunkSize;
    char wave[4];
    char fmt[4];
    uint32_t subchunk1Size;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char data[4];
    uint32_t dataSize;
};
#pragma pack(pop)

// Fonction de lecture WAV modifiée pour accepter mono et stéréo (moyennage en stéréo)
bool readWAV(const std::string &filename, std::vector<double> &samples, uint32_t &sampleRate) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Erreur à l'ouverture du fichier " << filename << std::endl;
        return false;
    }
    
    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));
    if (strncmp(header.riff, "RIFF", 4) != 0 || strncmp(header.wave, "WAVE", 4) != 0) {
        std::cerr << "Format de fichier WAV invalide." << std::endl;
        return false;
    }
    
    if (header.audioFormat != 1) {
        std::cerr << "Seul le format PCM est supporté." << std::endl;
        return false;
    }
    
    // Accepter mono ou stéréo
    if (header.numChannels != 1 && header.numChannels != 2) {
        std::cerr << "Seul l'audio mono ou stéréo est supporté." << std::endl;
        return false;
    }
    
    sampleRate = header.sampleRate;
    size_t totalSamples = header.dataSize / (header.bitsPerSample / 8);
    size_t numFrames = (header.numChannels == 2) ? (totalSamples / 2) : totalSamples;
    samples.resize(numFrames);
    
    if (header.numChannels == 1) {
        for (size_t i = 0; i < numFrames; i++) {
            int16_t sample = 0;
            file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t));
            samples[i] = static_cast<double>(sample) / 32768.0;
        }
    } else { // stéréo
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

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " input.wav" << endl;
        return 1;
    }
    
    string filename = argv[1];
    uint32_t sampleRate;
    vector<double> audioSamples;
    if (!readWAV(filename, audioSamples, sampleRate))
        return 1;
    
    // Calcul de N comme la puissance de 2 la plus proche de audioSamples.size()
    size_t totalSamples = audioSamples.size();
    double log2_val = log2((double)totalSamples);
    size_t lower = 1 << (int)floor(log2_val);
    size_t higher = 1 << (int)ceil(log2_val);
    size_t N = (totalSamples - lower <= higher - totalSamples) ? lower : higher;
    
    cout << "Nombre total d'échantillons : " << totalSamples << endl;
    cout << "Taille FFT choisie (puissance de 2 la plus proche) : " << N << endl;
    
    // Allocation du signal d'entrée avec zéro-padding si nécessaire
    complex<double>* inputSignal = new complex<double>[N];
    for (size_t i = 0; i < N; i++) {
        if (i < totalSamples)
            inputSignal[i].Re = audioSamples[i];
        else
            inputSignal[i].Re = 0.0;
        inputSignal[i].Im = 0.0;
    }
    
    complex<double>* outputFFT = new complex<double>[N];
    
    // Mesure du temps d'exécution de la FFT
    auto start = high_resolution_clock::now();
    try {
        rad2FFT((int)N, inputSignal, outputFFT);
    } catch (const char* msg) {
        cerr << "Erreur : " << msg << endl;
        delete[] inputSignal;
        delete[] outputFFT;
        return 1;
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Temps d'exécution de la FFT (CPU): " << duration.count() << " ms" << endl;
    
    // Afficher quelques résultats (magnitudes des 10 premières bins)
    // cout << "Résultats FFT (10 premières bins) :" << endl;
    // for (size_t i = 0; i < min(N, size_t(10)); i++) {
    //     double magnitude = sqrt(outputFFT[i].Re * outputFFT[i].Re + outputFFT[i].Im * outputFFT[i].Im);
    //     cout << "Bin " << i << ": " << magnitude << endl;
    // }
    
    delete[] inputSignal;
    delete[] outputFFT;
    return 0;
}
