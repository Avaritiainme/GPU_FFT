#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define PI 3.14159265358979323846
#define TWOPI 6.28318530717958647692

// Taille de bloc pour le traitement
static const int BLOCKSIZE = 4096;

struct complex_f {
    float Re, Im;
};

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

// ==================== Opérateurs complex_f en device ====================
__device__ complex_f operator+(complex_f a, complex_f b) {
    return { a.Re + b.Re, a.Im + b.Im };
}
__device__ complex_f operator-(complex_f a, complex_f b) {
    return { a.Re - b.Re, a.Im - b.Im };
}
__device__ complex_f operator*(complex_f a, complex_f b) {
    return {
        a.Re * b.Re - a.Im * b.Im,
        a.Re * b.Im + a.Im * b.Re
    };
}

// ==================== Kernels FFT / IFFT / Filtre ====================
__global__ void fftKernel(complex_f* x, complex_f* X, int N, int step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N / 2) return;

    int j = i * step * 2;
    complex_f even = x[j];
    complex_f odd  = x[j + step];

    float angle = -TWOPI * i / N;
    complex_f W = { cosf(angle), sinf(angle) };

    complex_f t = W * odd;
    X[i]          = even + t;
    X[i + N / 2]  = even - t;
}

__global__ void ifftKernel(complex_f* X, complex_f* x, int N, int step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N / 2) return;

    int j = i * step * 2;
    complex_f even = X[i];
    complex_f odd  = X[i + N / 2];

    float angle = TWOPI * i / N;
    complex_f W = { cosf(angle), sinf(angle) };

    complex_f t  = W * odd;
    x[j]         = { (even + t).Re / 2.0f,  (even + t).Im / 2.0f };
    x[j + step]  = { (even - t).Re / 2.0f,  (even - t).Im / 2.0f };
}

__global__ void bandpassFilter(complex_f* data, int N, float sampleRate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float freq = ((float)i / (float)N) * sampleRate;
    if (freq < 80.0f || freq > 8000.0f) {
        data[i].Re = 0.0f;
        data[i].Im = 0.0f;
    }
}

// Fonction pour calculer log2(n) pour les puissances de 2
__host__ __device__ int log2_power2(int n) {
    int log = 0;
    while (n >>= 1) log++;
    return log;
}

bool readWAV(const char* filename, vector<float>& samples, int& sampleRate)
{
    ifstream file(filename, ios::binary);
    if (!file) return false;
    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));
    if (strncmp(header.riff, "RIFF", 4) != 0 ||
        strncmp(header.wave, "WAVE", 4) != 0)
    {
        return false;
    }
    sampleRate = header.sampleRate;
    samples.clear();

    int16_t sample;
    for (uint32_t i = 0; i < header.dataSize / 2; ++i) {
        file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t));
        samples.push_back(static_cast<float>(sample) / 32768.0f);
    }
    return true;
}

bool writeWAV(const char* filename, const vector<float>& samples, int sampleRate)
{
    ofstream file(filename, ios::binary);
    if (!file) return false;

    WAVHeader header;
    memcpy(header.riff, "RIFF", 4);
    memcpy(header.wave, "WAVE", 4);
    memcpy(header.fmt,  "fmt ", 4);
    memcpy(header.data, "data", 4);

    header.subchunk1Size = 16;
    header.audioFormat   = 1;
    header.numChannels   = 1;
    header.sampleRate    = sampleRate;
    header.bitsPerSample = 16;
    header.byteRate      = sampleRate * 2;
    header.blockAlign    = 2;
    header.dataSize      = samples.size() * 2;
    header.chunkSize     = 36 + header.dataSize;

    file.write(reinterpret_cast<const char*>(&header), sizeof(WAVHeader));

    for (float f : samples) {
        float clamped = max(-1.0f, min(1.0f, f));
        int16_t s = static_cast<int16_t>(clamped * 32767);
        file.write(reinterpret_cast<const char*>(&s), sizeof(int16_t));
    }
    return true;
}


// Calcule la plus proche puissance de 2 >= n
int nextPowerOf2(int n)
{
    int power = 1;
    while (power < n) power <<= 1;
    return power;
}

// ==================== Traitement d'un bloc ====================
void processBlockGPU(const float* inBlock, float* outBlock,
                     int blockLen, int sampleRate,
                     long long &accumFFTus, long long &accumIFFTus)
{
    int N = nextPowerOf2(blockLen);
    int numStages = log2_power2(N);

    // Prépare un vecteur host de taille N
    vector<complex_f> h_signal(N);
    for (int i = 0; i < blockLen; ++i) {
        h_signal[i].Re = inBlock[i];
        h_signal[i].Im = 0.0f;
    }
    for (int i = blockLen; i < N; ++i) {
        h_signal[i].Re = 0.0f;
        h_signal[i].Im = 0.0f;
    }

    // Allocation GPU
    complex_f *d_input, *d_output, *d_temp;
    cudaMalloc(&d_input,  N * sizeof(complex_f));
    cudaMalloc(&d_output, N * sizeof(complex_f));
    cudaMalloc(&d_temp,   N * sizeof(complex_f));
    
    cudaMemcpy(d_input, h_signal.data(), N * sizeof(complex_f), cudaMemcpyHostToDevice);

    int blockSize = 256;
    
    // =========== FFT : Radix-2 complet ===========
    auto fft_start = high_resolution_clock::now();
    
    // Première passe (la source est d_input)
    int halfN = N / 2;
    int numBlocks = (halfN + blockSize - 1) / blockSize;
    int step = N/2;
    
    fftKernel<<<numBlocks, blockSize>>>(d_input, d_output, N, step);
    cudaDeviceSynchronize();

    // Passes suivantes (alternance entre d_output et d_temp)
    for (int stage = 1; stage < numStages; stage++) {
        complex_f *src = (stage % 2 == 0) ? d_temp : d_output;
        complex_f *dst = (stage % 2 == 0) ? d_output : d_temp;
        
        step = N >> (stage + 1); // N/2^(stage+1)
        
        for (int k = 0; k < (1 << stage); k++) {
            int offset = k * (N >> stage);
            fftKernel<<<numBlocks, blockSize>>>(src + offset, dst + offset, N >> stage, step);
        }
        cudaDeviceSynchronize();
    }
    
    auto fft_end = high_resolution_clock::now();
    accumFFTus += duration_cast<microseconds>(fft_end - fft_start).count();

    // Déterminer le buffer contenant le résultat final de la FFT
    complex_f *d_fft_result = (numStages % 2 == 0) ? d_temp : d_output;

    // =========== Filtre ===========
    int numBlocksFilter = (N + blockSize - 1) / blockSize;
    bandpassFilter<<<numBlocksFilter, blockSize>>>(d_fft_result, N, (float)sampleRate);
    cudaDeviceSynchronize();

    // =========== IFFT : Radix-2 complet ===========
    auto ifft_start = high_resolution_clock::now();
    
    // Première passe
    ifftKernel<<<numBlocks, blockSize>>>(d_fft_result, d_temp, N, step);
    cudaDeviceSynchronize();

    // Passes suivantes
    for (int stage = 1; stage < numStages; stage++) {
        complex_f *src = (stage % 2 == 0) ? d_output : d_temp;
        complex_f *dst = (stage % 2 == 0) ? d_temp : d_output;
        
        step = N >> (stage + 1);
        
        for (int k = 0; k < (1 << stage); k++) {
            int offset = k * (N >> stage);
            ifftKernel<<<numBlocks, blockSize>>>(src + offset, dst + offset, N >> stage, step);
        }
        cudaDeviceSynchronize();
    }
    
    auto ifft_end = high_resolution_clock::now();
    accumIFFTus += duration_cast<microseconds>(ifft_end - ifft_start).count();

    // Déterminer le buffer contenant le résultat final de l'IFFT
    complex_f *d_ifft_result = (numStages % 2 == 0) ? d_output : d_temp;
    
    // Normalisation (division par N)
    vector<complex_f> h_result(N);
    cudaMemcpy(h_result.data(), d_ifft_result, N * sizeof(complex_f), cudaMemcpyDeviceToHost);

    for (int i = 0; i < blockLen; ++i) {
        outBlock[i] = h_result[i].Re / N;  // Normalisation par N pour l'IFFT complète
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " input.wav" << endl;
        return 1;
    }

    vector<float> input;
    int sampleRate;
    if (!readWAV(argv[1], input, sampleRate)) {
        cerr << "Erreur lecture WAV" << endl;
        return 1;
    }

    auto t0 = high_resolution_clock::now();

    size_t totalSamples = input.size();
    vector<float> output(totalSamples);

    long long accumFFTus = 0;
    long long accumIFFTus = 0;

    // Découpage en blocs
    for (size_t offset = 0; offset < totalSamples; offset += BLOCKSIZE) {
        size_t blockLen = min((size_t)BLOCKSIZE, totalSamples - offset);

        // Traitement GPU sur chaque bloc
        processBlockGPU(&input[offset], &output[offset],
                        (int)blockLen, sampleRate,
                        accumFFTus, accumIFFTus);
    }

    bool ok = writeWAV("output_gpu.wav", output, sampleRate);

    auto t1 = high_resolution_clock::now();
    auto totalTime = duration_cast<milliseconds>(t1 - t0);

    if (!ok) {
        cerr << "Erreur écriture WAV" << endl;
        return 1;
    }

    cout << "Fichier 'output_gpu.wav' généré par blocs de " << BLOCKSIZE << " échantillons." << endl;
    cout << "Temps total : " << totalTime.count() << " ms" << endl;

    cout << "Temps FFT cumulé : "  << accumFFTus  << " μs" << endl;
    cout << "Temps IFFT cumulé : " << accumIFFTus << " μs" << endl;

    return 0;
}
