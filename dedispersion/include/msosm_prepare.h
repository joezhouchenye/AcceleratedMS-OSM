#pragma once

#include "globals.h"

using namespace std;

constexpr int MIN_SEGMENT_POINTS = 256;

class Prepare_MSOSM
{
public:
    Prepare_MSOSM(float bw = 0, float dm = 0, float f0 = 0);
    Prepare_MSOSM(float bw, float *dm, float f0, int numDMs = 1);
    void calculate_min_order();
    void override_order(int order);
    void generate_dedisp_params(int i=0);
    void segmentation(int i=0);

public:
    // Number of different DM values
    int numDMs;
    // Filter order for the Multi-segment Overlap-Save method
    int M;
    // Minimum filter order for the Multi-segment Overlap-Save method
    int M_min;
    // Minimum filter order for different DM values
    int *M_values;

    // Number of delays needed
    int delaycount = 0;
    // Number of delays needed for different DM values
    int *delaycount_values;
    // Dedispersion parameters
    fftwf_complex **dedisp_params;
    // Delay of each FFT point
    int **delay_points;
    // Dispersion smearing points for different DM values
    int *Nd_values;
    int start_Nd;
    int end_Nd;

protected:
    // Segment indexes for the negative and positive frequency components
    int *segneg, *segpos;
    // Segment delays for the negative and positive frequency components
    int *delayneg, *delaypos;
    // Number of segments for the negative and positive frequency components
    int negcount, poscount;
    // Bandwidth
    float bw;
    // Sampling frequency
    float fs;
    // Dispersion constant
    float kdm;
    // Dispersion measure for different DM values
    float *dm_values;
    // Start frequency of the band
    float f0;
    // Start angular frequency
    float w0;
};