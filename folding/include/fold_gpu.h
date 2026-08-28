#pragma once
#include <iostream>
#include <fstream>
#include "gpu_kernel.cuh"
#include "msosm_gpu_batch.h"
#include "osm_gpu_batch.h"

using namespace std;

class Fold_GPU
{
public:
    Fold_GPU(float period, float fs, unsigned long size, string outfileName = "fold.txt", unsigned long time_bin = -1);
    template <typename T>
    void calculate_intensity(T *pol1, T *pol2);
    template <typename T>
    void calculate_intensity(T *pol1);
    void fold_data_phase();
    void get_folded_data();
    void write_to_file();
    ~Fold_GPU();

public:
    float *folded_data;
    unsigned long time_bin;
    unsigned long period_samples;

private:
    double t_start = 0.0;
    bool ready = false;
    int discard_count = 0;
    int current_discard = 0;
    unsigned long size;
    float *total_intensity_d;
    double sampling_frequency;
    double period_seconds;
    double *phase_bin_sum_d;
    unsigned long long *phase_bin_count_d;
    float *folded_data_d;
    float period_samples_float;
    float current_diff = 0.0f;
    unsigned long current_index = 0;
    ofstream outfile;
};