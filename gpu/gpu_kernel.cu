#include "gpu_kernel.cuh"

#include <cstdlib>

using namespace std;

static int read_env_int(const char *name, int default_value)
{
    const char *v = std::getenv(name);
    if (v == nullptr)
        return default_value;
    int parsed = atoi(v);
    return (parsed > 0) ? parsed : default_value;
}

static __global__ void uint16ToComplex_kernel(Complex *dst, uint16_t *src, size_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;

    uint32_t val = reinterpret_cast<uint32_t *>(src)[i];
    uint16_t x = val & 0xFFFF;
    uint16_t y = (val >> 16) & 0xFFFF;

    float fx = static_cast<float>(x) - 32768.0f * (x != 0);
    float fy = static_cast<float>(y) - 32768.0f * (y != 0);
    dst[i].x = fx;
    dst[i].y = fy;
}

void uint16ToComplex(Complex *dst, uint16_t *src, size_t size, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (size + block_size - 1) / block_size;
    uint16ToComplex_kernel<<<grid_size, block_size, 0, stream>>>(dst, src, size);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void gatherMultiply_kernel(
    const Complex *__restrict__ fft_block,
    const Complex *__restrict__ dedisp_params,
    Complex *ifft_input,
    const DelayBlockMeta *__restrict__ delay_block_meta,
    const Segment *__restrict__ segments,
    int batch, int numDMs, int fftpoint, int read_block_index, int fft_block_size)
{
    // FFT数据排布方式
    // 总大小为fft_block_size * fftpoint
    // ... front buffer
    // batch * fftpoint start from read_block_index
    // fft[0][batch1],fft[1][batch1],...,fft[fftpoint-1][batch1]
    // fft[0][batch2],fft[1][batch2],...,fft[fftpoint-1][batch2]
    // ...
    // fft[0][batchN],fft[1][batchN],...,fft[fftpoint-1][batchN]
    // ... back buffer
    const int dm_idx = blockIdx.x / batch;
    const int batch_idx = blockIdx.x % batch;
    const int seg_idx = blockIdx.y;
    const int thread_idx = threadIdx.x;

    if (dm_idx >= numDMs || batch_idx >= batch)
        return;

    DelayBlockMeta meta = delay_block_meta[dm_idx];
    if (seg_idx >= meta.seg_count)
        return;

    Segment seg = segments[meta.seg_offset + seg_idx];
    if (thread_idx >= seg.len)
        return;

    int delay_val = seg.group;
    int fft_idx = read_block_index + batch_idx - delay_val;
    fft_idx = ((fft_idx % fft_block_size) + fft_block_size) % fft_block_size;
    int fft_offset = fft_idx * fftpoint;
    int param_offset = dm_idx * fftpoint;
    int dm_offset = dm_idx * batch * fftpoint;
    int batch_offset = batch_idx * fftpoint;

    int n_idx = seg.start + thread_idx;
    Complex inv_N = make_cuFloatComplex(1.0f / fftpoint, 0.0f);
    Complex fft_val = cuCmulf(fft_block[fft_offset + n_idx], inv_N);
    Complex param = dedisp_params[param_offset + n_idx];
    ifft_input[dm_offset + batch_offset + n_idx] = cuCmulf(fft_val, param);
}

void gatherMultiply(
    const Complex *fft_block,
    const Complex *dedisp_params,
    Complex *ifft_input,
    const DelayBlockMeta *delay_block_meta,
    const Segment *segments,
    int &max_seg_count,
    int &batch, int &numDMs, int &fftpoint, int &read_block_index, int &fft_block_size,
    cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    dim3 grid(batch * numDMs, max_seg_count);
    gatherMultiply_kernel<<<grid, block_size, 0, stream>>>(
        fft_block, dedisp_params, ifft_input, delay_block_meta, segments,
        batch, numDMs, fftpoint, read_block_index, fft_block_size);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void complexMultiply_kernel(Complex *a, Complex *b, Complex *block, int N, int batch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * batch)
        return;
    float2 input1, input2;
    input1.x = a[i].x / N;
    input1.y = a[i].y / N;
    input2.x = b[i % N].x;
    input2.y = b[i % N].y;
    block[i] = cuCmulf(input1, input2);
}

void complexMultiply(Complex *a, Complex *b, int N, Complex *block, int batch, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (N * batch + block_size - 1) / block_size;
    complexMultiply_kernel<<<grid_size, block_size, 0, stream>>>(a, b, block, N, batch);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void discardSamples_kernel(Complex *a, Complex *b, int M, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * count)
        return;
    const int group = i / M;
    const int offset = i % M;
    const int index = group * 2 * M + offset + M;

    float2 *a_f2 = reinterpret_cast<float2 *>(a);
    float2 *b_f2 = reinterpret_cast<float2 *>(b);

    const float scale = 1.0f / (2 * M);

    float2 val = a_f2[index];
    val.x *= scale;
    val.y *= scale;

    b_f2[i] = val;
}

void discardSamples(Complex *a, Complex *b, int M, int count, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (M * count + block_size - 1) / block_size;
    discardSamples_kernel<<<grid_size, block_size, 0, stream>>>(a, b, M, count);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void complexToUint16_kernel(uint16_t *dst, Complex *src, size_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    float2 val = reinterpret_cast<float2 *>(src)[i];
    uint16_t x = static_cast<uint16_t>(val.x + 32768.0f * (val.x != 0));
    uint16_t y = static_cast<uint16_t>(val.y + 32768.0f * (val.y != 0));
    uint32_t val32 = (y << 16) | x;
    reinterpret_cast<uint32_t *>(dst)[i] = val32;
}

void complexToUint16(uint16_t *dst, Complex *src, size_t size, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (size + block_size - 1) / block_size;
    complexToUint16_kernel<<<grid_size, block_size, 0, stream>>>(dst, src, size);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void discardSamplesToUint16_kernel(uint16_t *dst, Complex *src, int M, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = M * count;
    if (i >= size)
        return;

    const int group = i / M;
    const int offset = i % M;
    const int index = group * 2 * M + offset + M;
    const float scale = 1.0f / (2 * M);

    float2 val = reinterpret_cast<float2 *>(src)[index];
    val.x *= scale;
    val.y *= scale;

    uint16_t x = static_cast<uint16_t>(val.x + 32768.0f * (val.x != 0));
    uint16_t y = static_cast<uint16_t>(val.y + 32768.0f * (val.y != 0));
    reinterpret_cast<uint32_t *>(dst)[i] = (static_cast<uint32_t>(y) << 16) | x;
}

void discardSamplesToUint16(uint16_t *dst, Complex *src, int M, int count, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    const int size = M * count;
    const int grid_size = (size + block_size - 1) / block_size;
    discardSamplesToUint16_kernel<<<grid_size, block_size, 0, stream>>>(dst, src, M, count);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void calculateIntensity_kernel(Complex *a, Complex *b, float *total_intensity, unsigned long size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    total_intensity[i] = sqrtf(a[i].x * a[i].x + a[i].y * a[i].y + b[i].x * b[i].x + b[i].y * b[i].y);
}

void calculateIntensity(Complex *a, Complex *b, float *total_intensity, unsigned long size, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (size + block_size - 1) / block_size;
    calculateIntensity_kernel<<<grid_size, block_size, 0, stream>>>(a, b, total_intensity, size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void initializeBoolArray_kernel(bool *array, int size, bool value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        array[idx] = value;
    }
}

void initializeBoolArray(bool *array, int size, bool value)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (size + block_size - 1) / block_size;
    initializeBoolArray_kernel<<<grid_size, block_size>>>(array, size, value);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void waitForCPU_kernel(bool *flag)
{
    while (*flag)
        ;
    *flag = true;
}

void waitForCPU(bool *flag, cudaStream_t stream)
{
    waitForCPU_kernel<<<1, 1, 0, stream>>>(flag);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void unblockCPU_kernel(bool *flag)
{
    *flag = false;
}

void unblockCPU(bool *flag, cudaStream_t stream)
{
    unblockCPU_kernel<<<1, 1, 0, stream>>>(flag);
    CUDA_CHECK(cudaGetLastError());
}