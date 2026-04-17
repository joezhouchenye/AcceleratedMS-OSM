#include "AWGN.h"

AWGN::AWGN()
{
    // default values are assigned
    this->mean = 0.0;
    this->variance = 1.0;
    this->numberOfSamples = 100;
    this->isSNRMode = false;
}

AWGN::AWGN(float mean, float variance, unsigned long numberOfSamples)
{
    this->mean = mean;
    this->variance = variance;
    this->numberOfSamples = numberOfSamples;
    this->isSNRMode = false;
}

AWGN::AWGN(float SNR, unsigned long numberOfSamples)
{
    this->mean = 0.0;
    this->variance = 1.0;
    this->numberOfSamples = numberOfSamples;
    this->sigma = sqrt(pow(10, (-SNR / 10)));
    this->isSNRMode = true;
}

fftwf_complex *AWGN::generateNoiseSamples()
{
    default_random_engine defaultGeneratorEngine;
    normal_distribution<float> normalDistributionReal(this->mean, this->variance);
    normal_distribution<float> normalDistributionImag(this->mean, this->variance);

    // Dynamically allocate noise samples
    samples = (fftwf_complex *)fftwf_malloc(this->numberOfSamples * sizeof(fftwf_complex));

    if (this->isSNRMode)
    {
        // calculate with SNR
        for (int i = 0; i < this->numberOfSamples; i++)
        {
            samples[i][0] = (this->sigma) / sqrt(2) * normalDistributionReal(defaultGeneratorEngine);
            samples[i][1] = (this->sigma) / sqrt(2) * normalDistributionImag(defaultGeneratorEngine);
        }
    }
    else
    {
        // calculate with mean and variance
        for (int i = 0; i < this->numberOfSamples; i++)
        {
            samples[i][0] = normalDistributionReal(defaultGeneratorEngine);
            samples[i][1] = normalDistributionImag(defaultGeneratorEngine);
        }
    }

    return samples;
}

void AWGN::deallocate()
{
    // Deallocate dynamically allocated noise samples
    fftwf_free(samples);
}