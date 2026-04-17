#pragma once

#include "msosm_prepare.h"
#include "gpu_kernel.cuh"
#include "psrdada.h"

// Reduce the count of D2D memory copy
constexpr int D2D_REDUCE_COUNT = 16;

// Ensure BLOCK_SIZE is smaller than MIN_SEGMENT_POINTS
#if BLOCK_SIZE > MIN_SEGMENT_POINTS
#error "BLOCK_SIZE should be smaller than MIN_SEGMENT_POINTS"
#endif

class MSOSM_GPU_BATCH : public Prepare_MSOSM
{
public:
    MSOSM_GPU_BATCH(float bw = 0, float dm = 0, float f0 = 0);
    MSOSM_GPU_BATCH(float bw, float *dm, float f0, int numDMs = 1);
    void get_device_info();
    void initialize_uint16(int fftpoint = 0, int batch = 1);
    void filter_block_uint16(uint16_pair *input);
    void get_output(Complex *output);
    void get_output(uint16_pair *output);
    void synchronize();
    void reset_device();
    ~MSOSM_GPU_BATCH();

private:
    bool reverse_flag = false;
    int current_index = 0;

public:
    int batch;
    Complex *output_buffer_d;

private:
    // GPU memory for original 16-bit input
    uint16_pair *input_buffer_int16_d;
    // GPU memory for converted complex input
    Complex *input_buffer_d;
    // Size of converted complex input buffer
    int input_buffer_size;

    cufftHandle p_f;
    // FFT point used
    int fftpoint;
    // Pointer for FFT input
    Complex *input_fft_d;
    // Barrier for FFT input pointer to start over
    int input_fft_barrier;
    // Change the FFT input pointer destination
    void increment_fft_input();
    // Current input index for FFT input pointer
    int input_fft_index = 0;
    // GPU memory for FFT result buffer
    Complex *fft_block_d;
    // Number of FFT results
    int fft_block_size;
    // Pointer for FFT output
    Complex *output_fft_d;
    // Barrier for FFT output pointer to start over
    int output_fft_barrier;
    // Change the FFT output pointer destination
    void increment_fft_output();
    // Current output index for FFT output pointer
    int output_fft_index = 0;

    // GPU memory for segments
    Segment *segments_d;
    // GPU memory for delay block meta
    DelayBlockMeta *delay_block_meta_d;
    // Max segment count for a single DM
    int max_seg_count;
    // GPU memory for dedispersion parameters
    Complex *dedisp_params_d;
    // Read start block index of FFT results
    int read_block_index = 0;

    cufftHandle p_b;
    // Pointer for IFFT input
    Complex *input_ifft_d;
    // Pointer for IFFT output
    Complex *output_ifft_d;

    uint16_pair *output_buffer_int16_d;

private:
    vector<Segment> build_segments(const int* delay_block, int fft_len);
};