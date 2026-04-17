#pragma once
#include <iostream>
#include <random>
#include <vector>
#include <fftw3.h>
using namespace std;

class AWGN
{

public:
    // Default constructor
    AWGN();
    // Constructor with initializers
    AWGN(float mean, float variance, unsigned long numberOfSamples);
    // Constructor with initializers (SNR)
    AWGN(float SNR, unsigned long numberOfSamples);

    fftwf_complex *generateNoiseSamples();
    void deallocate();

private:
    float mean;
    float variance;
    unsigned long numberOfSamples;
    float sigma;
    bool isSNRMode;
    fftwf_complex *samples;
};