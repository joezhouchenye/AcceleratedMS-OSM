#pragma once
#include "AWGN.h"
#include "globals.h"

using namespace std;

class SimulatedComplexSignal
{
public:
    SimulatedComplexSignal(float bw, float dm, float f0, float period, string mode="complex");
    void generate_pulsar_signal_new(unsigned long repeat = 1);
    void generate_pulsar_signal(unsigned long repeat = 1, bool add_noise = false, float SNR = 0, bool pinned = true);
    void plot_abs(const fftwf_complex *signal, unsigned long size);
    void plot_abs(const uint16_pair *signal, unsigned long size);
    ~SimulatedComplexSignal();

public:
    // Simulated signal
    fftwf_complex *signal;
    // Simulated signal (uint16_pair)
    uint16_pair *signal_u16;
    // Simulated signal data type mode
    string data_type;
    // Size of the simulated signal
    unsigned long signal_size;
    // Period points
    unsigned long Np;
    // Dispersion smearing points
    unsigned long Nd;
    // Filter order for the Multi-segment coherent dedispersion method
    unsigned long M;
    // Minimum filter order for the Multi-segment coherent dedispersion method
    unsigned long M_min;
    // Number of periods the signal will be distributed
    unsigned long range;

private:
    // Bandwidth
    float bw;
    // Sampling frequency
    float fs;
    // Dispersion constant
    float kdm;
    // Dispersion measure
    float dm;
    // Start frequency of the band
    float f0;
    // Start angular frequency
    float w0;
    // Signal period
    float period;
    // Signal power
    float signal_power;
};