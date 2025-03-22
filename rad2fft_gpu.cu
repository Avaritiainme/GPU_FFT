#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include <cuda_runtime.h>
#include <cufft.h>

using namespace std;

#define PI       3.1415926535897932384626433832795    
#define TWOPI    6.283185307179586476925286766559     

// Structure d'en-tête WAV (supporte PCM mono et stéréo)
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

// Fonction de lecture du fichier WAV modifiée pour accepter mono et stéréo (moyennage en stéréo)
bool readWAV(const string &filename, vector<double> &samples, uint32_t &sampleRate) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Erreur à l'ouverture du fichier " << filename << endl;
        return false;
    }
    
    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));
    if (strncmp(header.riff, "RIFF", 4) != 0 || strncmp(header.wave, "WAVE", 4) != 0) {
        cerr << "Format de fichier WAV invalide." << endl;
        return false;
    }
    
    if (header.audioFormat != 1) {
        cerr << "Seul le format PCM est supporté." << endl;
        return false;
    }
    
    // Accepter mono ou stéréo
    if (header.numChannels != 1 && header.numChannels != 2) {
        cerr << "Seul l'audio mono ou stéréo est supporté." << endl;
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
    
    // Allocation mémoire hôte pour le signal (cuFFT utilise cufftDoubleComplex)
    // Avec zéro-padding si nécessaire.
    cufftDoubleComplex* h_signal = new cufftDoubleComplex[N];
    for (size_t i = 0; i < N; i++) {
        if (i < totalSamples)
            h_signal[i].x = audioSamples[i];
        else
            h_signal[i].x = 0.0;
        h_signal[i].y = 0.0;
    }
    
    cufftDoubleComplex* h_result = new cufftDoubleComplex[N];
    
    // Allocation mémoire sur le GPU
    cufftDoubleComplex* d_signal;
    cudaMalloc((void**)&d_signal, sizeof(cufftDoubleComplex) * N);
    
    // Copier le signal d'entrée vers le GPU
    cudaMemcpy(d_signal, h_signal, sizeof(cufftDoubleComplex)*N, cudaMemcpyHostToDevice);
    
    // Création du plan cuFFT
    cufftHandle plan;
    if (cufftPlan1d(&plan, N, CUFFT_Z2Z, 1) != CUFFT_SUCCESS) {
        cerr << "Erreur lors de la création du plan cuFFT." << endl;
        cudaFree(d_signal);
        delete[] h_signal;
        delete[] h_result;
        return 1;
    }
    
    // Mesurer le temps d'exécution de la FFT sur le GPU à l'aide d'événements CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    if (cufftExecZ2Z(plan, d_signal, d_signal, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        cerr << "Erreur lors de l'exécution de cuFFT." << endl;
        cufftDestroy(plan);
        cudaFree(d_signal);
        delete[] h_signal;
        delete[] h_result;
        return 1;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Temps d'exécution de la FFT (GPU): " << elapsedTime << " ms" << endl;
    
    // Copier le résultat du GPU vers l'hôte
    cudaMemcpy(h_result, d_signal, sizeof(cufftDoubleComplex)*N, cudaMemcpyDeviceToHost);
    
    // Afficher quelques résultats (magnitudes des 10 premières bins)
    // cout << "Résultats FFT (10 premières bins) :" << endl;
    // for (size_t i = 0; i < min(N, size_t(10)); i++) {
    //     double magnitude = sqrt(h_result[i].x * h_result[i].x + h_result[i].y * h_result[i].y);
    //     cout << "Bin " << i << ": " << magnitude << endl;
    // }
    
    // Libération des ressources
    cufftDestroy(plan);
    cudaFree(d_signal);
    delete[] h_signal;
    delete[] h_result;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
