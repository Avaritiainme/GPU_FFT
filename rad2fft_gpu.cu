#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>  // Ajouté pour strncmp et memcpy
#include <algorithm>

#define PI 3.14159265358979323846
#define TWOPI 6.28318530717958647692

struct complex {
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

__device__ complex operator+(complex a, complex b) {
    return { a.Re + b.Re, a.Im + b.Im };
}
__device__ complex operator-(complex a, complex b) {
    return { a.Re - b.Re, a.Im - b.Im };
}
__device__ complex operator*(complex a, complex b) {
    return {
        a.Re * b.Re - a.Im * b.Im,
        a.Re * b.Im + a.Im * b.Re
    };
}

__global__ void fftKernel(complex* x, complex* X, int N, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N / 2) return;

    int j = i * step * 2;
    complex even = x[j];
    complex odd = x[j + step];

    float angle = -TWOPI * i / N;
    complex W = { cosf(angle), sinf(angle) };

    complex t = W * odd;
    X[i] = even + t;
    X[i + N / 2] = even - t;
}

__global__ void bandpassFilter(complex* data, int N, float sampleRate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float freq = ((float)i / (float)N) * sampleRate;
    if (freq < 80.0f || freq > 8000.0f) {
        data[i].Re = 0.0f;
        data[i].Im = 0.0f;
    }
}

__global__ void ifftKernel(complex* X, complex* x, int N, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N / 2) return;

    int j = i * step * 2;
    complex even = X[i];
    complex odd = X[i + N / 2];

    float angle = TWOPI * i / N;
    complex W = { cosf(angle), sinf(angle) };

    complex t = W * odd;
    x[j] = { (even + t).Re / 2.0f, (even + t).Im / 2.0f };
    x[j + step] = { (even - t).Re / 2.0f, (even - t).Im / 2.0f };
}

bool readWAV(const char* filename, std::vector<float>& samples, int& sampleRate) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;
    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));
    if (strncmp(header.riff, "RIFF", 4) != 0 || strncmp(header.wave, "WAVE", 4) != 0)
        return false;
    sampleRate = header.sampleRate;
    int16_t sample;
    samples.clear();
    for (uint32_t i = 0; i < header.dataSize / 2; ++i) {
        file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t));
        samples.push_back(static_cast<float>(sample) / 32768.0f);
    }
    return true;
}

bool writeWAV(const char* filename, const std::vector<float>& samples, int sampleRate) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;
    WAVHeader header;
    memcpy(header.riff, "RIFF", 4);
    memcpy(header.wave, "WAVE", 4);
    memcpy(header.fmt, "fmt ", 4);
    memcpy(header.data, "data", 4);
    header.subchunk1Size = 16;
    header.audioFormat = 1;
    header.numChannels = 1;
    header.sampleRate = sampleRate;
    header.bitsPerSample = 16;
    header.byteRate = sampleRate * 2;
    header.blockAlign = 2;
    header.dataSize = samples.size() * 2;
    header.chunkSize = 36 + header.dataSize;
    file.write(reinterpret_cast<const char*>(&header), sizeof(WAVHeader));
    for (float f : samples) {
        int16_t s = static_cast<int16_t>(std::max(-1.0f, std::min(1.0f, f)) * 32767);
        file.write(reinterpret_cast<const char*>(&s), sizeof(int16_t));
    }
    return true;
}

int nextPowerOf2(int n) {
    int power = 1;
    while (power < n) power <<= 1;
    return power;
}

#include <chrono>
using namespace std::chrono;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " input.wav" << std::endl;
        return 1;
    }

    std::vector<float> input;
    int sampleRate;
    if (!readWAV(argv[1], input, sampleRate)) {
        std::cerr << "Échec lecture WAV" << std::endl;
        return 1;
    }
    auto t1 = high_resolution_clock::now();
    int N = nextPowerOf2(input.size());

    std::vector<complex> h_signal(N);
    for (int i = 0; i < N; ++i) {
        h_signal[i].Re = (i < input.size()) ? input[i] : 0.0f;
        h_signal[i].Im = 0.0f;
    }

    complex *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(complex));
    cudaMalloc(&d_output, N * sizeof(complex));
    cudaMemcpy(d_input, h_signal.data(), N * sizeof(complex), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N / 2 + blockSize - 1) / blockSize;
    fftKernel<<<numBlocks, blockSize>>>(d_input, d_output, N, 1);
    cudaDeviceSynchronize();

    bandpassFilter<<<(N + blockSize - 1) / blockSize, blockSize>>>(d_output, N, sampleRate);
    cudaDeviceSynchronize();

    ifftKernel<<<numBlocks, blockSize>>>(d_output, d_input, N, 1);
    cudaDeviceSynchronize();

    std::vector<complex> h_result(N);
    cudaMemcpy(h_result.data(), d_input, N * sizeof(complex), cudaMemcpyDeviceToHost);

    std::vector<float> output(N);
    for (int i = 0; i < N; ++i) output[i] = h_result[i].Re;

    if (!writeWAV("output_gpu.wav", output, sampleRate)) {
        std::cerr << "Échec écriture WAV" << std::endl;
        return 1;
    }

    cudaFree(d_input);
    auto t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(t2 - t1);
    std::cout << "Temps d'exécution GPU : " << duration.count() << " ms" << std::endl;
    cudaFree(d_output);

    std::cout << "Voix isolée enregistrée dans output_voice_gpu.wav" << std::endl;
    return 0;
}
