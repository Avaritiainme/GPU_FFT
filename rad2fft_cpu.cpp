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

#define PI 3.14159265358979323846
#define TWOPI 6.28318530717958647692

struct complex {
    double Re, Im;
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

bool readWAV(const char* filename, vector<double>& samples, int& sampleRate) {
    ifstream file(filename, ios::binary);
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
        samples.push_back(static_cast<double>(sample) / 32768.0);
    }
    return true;
}

bool writeWAV(const char* filename, const vector<double>& samples, int sampleRate) {
    ofstream file(filename, ios::binary);
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
    for (double f : samples) {
        int16_t s = static_cast<int16_t>(max(-1.0, min(1.0, f)) * 32767);
        file.write(reinterpret_cast<const char*>(&s), sizeof(int16_t));
    }
    return true;
}

int nextPowerOf2(int n) {
    int power = 1;
    while (power < n) power <<= 1;
    return power;
}

void fft(int N, complex* x, complex* X) {
    for (int k = 0; k < N; ++k) {
        X[k].Re = 0;
        X[k].Im = 0;
        for (int n = 0; n < N; ++n) {
            double angle = -2 * PI * k * n / N;
            X[k].Re += x[n].Re * cos(angle) - x[n].Im * sin(angle);
            X[k].Im += x[n].Re * sin(angle) + x[n].Im * cos(angle);
        }
    }
}

void ifft(int N, complex* X, complex* x) {
    for (int n = 0; n < N; ++n) {
        x[n].Re = 0;
        x[n].Im = 0;
        for (int k = 0; k < N; ++k) {
            double angle = 2 * PI * k * n / N;
            x[n].Re += X[k].Re * cos(angle) - X[k].Im * sin(angle);
            x[n].Im += X[k].Re * sin(angle) + X[k].Im * cos(angle);
        }
        x[n].Re /= N;
        x[n].Im /= N;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " input.wav" << endl;
        return 1;
    }

    vector<double> input;
    int sampleRate;
    if (!readWAV(argv[1], input, sampleRate)) {
        cerr << "Erreur lecture WAV" << endl;
        return 1;
    }

    auto t1 = high_resolution_clock::now();
    int N = nextPowerOf2(input.size());
    vector<complex> signal(N), spectrum(N), output(N);

    for (int i = 0; i < N; ++i) {
        signal[i].Re = (i < input.size()) ? input[i] : 0.0;
        signal[i].Im = 0.0;
    }

    fft(N, signal.data(), spectrum.data());

    for (int i = 0; i < N; ++i) {
        double freq = ((double)i / (double)N) * sampleRate;
        if (freq < 80.0 || freq > 8000.0) {
            spectrum[i].Re = 0.0;
            spectrum[i].Im = 0.0;
        }
    }

    ifft(N, spectrum.data(), output.data());

    vector<double> final(N);
    for (int i = 0; i < N; ++i) final[i] = output[i].Re;

    writeWAV("output__cpu.wav", final, sampleRate);
    auto t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(t2 - t1);
    cout << "Temps d'exécution CPU : " << duration.count() << " ms" << endl;
    return 0;
}
