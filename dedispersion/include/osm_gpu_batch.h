#pragma once

#include "prepare.h"
#include "gpu_kernel.cuh"
#include "psrdada.h"

class OSM_GPU_BATCH : public Prepare_OSM
{
public:
    OSM_GPU_BATCH(float bw = 0, float dm = 0, float f0 = 0);
    void get_device_info();
    void initialize_uint16(unsigned long fftpoint = 0, int batch = 1, bool streaming = false);
    void filter_block_uint16(uint16_pair *input);
    void get_output(Complex *output);
    ~OSM_GPU_BATCH();

private:
    bool reverse_flag = false;
    int current_index = 0;
    bool streaming = false;
    cudaStream_t stream;

public:
    int batch;
    unsigned long total_memory = 0;
    Complex *output_buffer_d;

private:
    unsigned long fftpoint;
    Complex *input_buffer_d;
    uint16_pair *input_buffer_int16_d;
    Complex *fft_block_d;
    Complex *filtered_block_d;
    Complex *ifft_block_d;
    uint16_pair *output_buffer_int16_d;
    Complex *dedisp_params_d;
    cufftHandle p_f, p_b;
};