#pragma once

#include "gpu_common.cuh"

constexpr int ITEMS_PER_THREAD = 4;
constexpr int BLOCK_SIZE = 256;

// Segment Struct
typedef struct
{
    int start; // Start index of the segment
    int len;   // Length of the segment
    int group; // Which group of the FFT results to fetch
} Segment;

// Delay Block Meta Struct
typedef struct
{
    int seg_offset; // Offset of the segment for all DMs
    int seg_count;  // Number of segments for a single DM
} DelayBlockMeta;

void uint16ToComplex(Complex *dst, uint16_t *src, size_t size, cudaStream_t stream = 0);
void gatherMultiply(
    const Complex *fft_block,
    const Complex *dedisp_params,
    Complex *ifft_input,
    const DelayBlockMeta *delay_block_meta,
    const Segment *segments,
    int &max_seg_count,
    int &batch, int &numDMs, int &fftpoint, int &read_block_index, int &fft_block_size,
    cudaStream_t stream = 0);
void complexMultiply(Complex *a, Complex *b, int N, Complex *block, int batch, cudaStream_t stream = 0);
void discardSamples(Complex *a, Complex *b, int M, int count, cudaStream_t stream = 0);
void complexToUint16(uint16_t *dst, Complex *src, size_t size, cudaStream_t stream = 0);
void discardSamplesToUint16(uint16_t *dst, Complex *src, int M, int count, cudaStream_t stream = 0);
void calculateIntensity(Complex *a, Complex *b, float *total_intensity, unsigned long size, cudaStream_t stream = 0);
void calculateIntensity(Complex *a, float *total_intensity, unsigned long size, cudaStream_t stream = 0);

void initializeBoolArray(bool* array, int size, bool value);
void waitForCPU(bool* flag, cudaStream_t stream);
void unblockCPU(bool* flag, cudaStream_t stream);