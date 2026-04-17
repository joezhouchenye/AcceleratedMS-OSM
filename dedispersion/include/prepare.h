#pragma once

#include "globals.h"

using namespace std;

class Prepare_OSM
{
public:
    Prepare_OSM(float bw = 0, float dm = 0, float f0 = 0);
    void calculate_min_order();
    void override_order(unsigned long order);
    void generate_dedisp_params();

private:
    int next_power_of_2(int n);

public:
    // Filter order for the Overlap-Save method
    unsigned long M;
    // Minimum filter order for the Overlap-Save method
    unsigned long M_min;
    // Dedispersion parameters
    fftwf_complex *dedisp_params;
    // Dispersion smearing points
    unsigned long Nd;

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
};