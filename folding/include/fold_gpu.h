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
    void discard_samples(unsigned long Nd);
    template <typename T>
    void calculate_intensity(T *pol1, T *pol2);
    void fold_data();
    void fold_data_bins();
    void write_to_file();
    ~Fold_GPU();

public:
    float *folded_data;
    unsigned long time_bin;
    float *total_intensity;
    unsigned long period_samples;
    unsigned long *fold_count;
    float *tmp_fold;

private:
    bool ready = false;
    int discard_count = 0;
    int current_discard = 0;
    float *folded_data_raw;
    unsigned long size;
    float *total_intensity_d;
    float period_samples_float;
    float diff_samples;
    float current_diff = 0.0f;
    unsigned long current_index = 0;
    ofstream outfile;
};