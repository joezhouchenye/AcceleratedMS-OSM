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

// static __global__ void gatherMultiply_kernel(
//     const Complex *__restrict__ fft_block,
//     const Complex *__restrict__ dedisp_params,
//     Complex *ifft_input,
//     const DelayBlockMeta *__restrict__ delay_block_meta,
//     const Segment *__restrict__ segments,
//     int batch, int numDMs, int fftpoint, int read_block_index, int fft_block_size)
// {
//     // FFT数据排布方式
//     // 总大小为fft_block_size * fftpoint
//     // ... front buffer
//     // batch * fftpoint start from read_block_index
//     // fft[0][batch1],fft[1][batch1],...,fft[fftpoint-1][batch1]
//     // fft[0][batch2],fft[1][batch2],...,fft[fftpoint-1][batch2]
//     // ...
//     // fft[0][batchN],fft[1][batchN],...,fft[fftpoint-1][batchN]
//     // ... back buffer
//     const int dm_idx = blockIdx.x / batch;
//     const int batch_idx = blockIdx.x % batch;
//     const int seg_idx = blockIdx.y;
//     const int thread_idx = threadIdx.x;

//     if (dm_idx >= numDMs || batch_idx >= batch)
//         return;

//     DelayBlockMeta meta = delay_block_meta[dm_idx];
//     if (seg_idx >= meta.seg_count)
//         return;

//     Segment seg = segments[meta.seg_offset + seg_idx];
//     if (thread_idx >= seg.len)
//         return;

//     int delay_val = seg.group;
//     int fft_idx = read_block_index + batch_idx - delay_val;
//     fft_idx = ((fft_idx % fft_block_size) + fft_block_size) % fft_block_size;
//     const int fft_offset = fft_idx * fftpoint;
//     const int param_offset = dm_idx * fftpoint;
//     const int out_offset = (dm_idx * batch + batch_idx) * fftpoint;

//     int n_idx = seg.start + thread_idx;
//     Complex inv_N = make_cuFloatComplex(1.0f / fftpoint, 0.0f);
//     Complex fft_val = cuCmulf(fft_block[fft_offset + n_idx], inv_N);
//     Complex param = dedisp_params[param_offset + n_idx];
//     ifft_input[out_offset + n_idx] = cuCmulf(fft_val, param);
// }

// void gatherMultiply(
//     const Complex *fft_block,
//     const Complex *dedisp_params,
//     Complex *ifft_input,
//     const DelayBlockMeta *delay_block_meta,
//     const Segment *segments,
//     int &max_seg_count,
//     int &batch, int &numDMs, int &fftpoint, int &read_block_index, int &fft_block_size,
//     cudaStream_t stream)
// {
//     const int block_size = BLOCK_SIZE;
//     dim3 grid(batch * numDMs, max_seg_count);
//     gatherMultiply_kernel<<<grid, block_size, 0, stream>>>(
//         fft_block, dedisp_params, ifft_input, delay_block_meta, segments,
//         batch, numDMs, fftpoint, read_block_index, fft_block_size);
//     CUDA_CHECK(cudaGetLastError());
// }

template <int ITEMS_PER_THREAD>
static __global__ void gatherMultiply_kernel(
    const Complex *__restrict__ fft_block,
    const Complex *__restrict__ dedisp_params,
    Complex *__restrict__ ifft_input,
    const DelayBlockMeta *__restrict__ delay_block_meta,
    const Segment *__restrict__ segments,
    int batch,
    int numDMs,
    int fftpoint,
    int read_block_index,
    int fft_block_size)
{
    const int dm_idx = blockIdx.x / batch;
    const int batch_idx = blockIdx.x % batch;
    const int seg_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (dm_idx >= numDMs || batch_idx >= batch)
        return;

    __shared__ DelayBlockMeta meta;
    __shared__ Segment seg;
    if (threadIdx.x == 0)
    {
        meta = delay_block_meta[dm_idx];
        seg = segments[meta.seg_offset + seg_idx];
    }
    __syncthreads();

    const int local_base = tid * ITEMS_PER_THREAD;

    const int delay_val = seg.group;

    int fft_idx = read_block_index + batch_idx - delay_val;
    fft_idx = ((fft_idx % fft_block_size) + fft_block_size) % fft_block_size;

    const int fft_offset = fft_idx * fftpoint;
    const int param_offset = dm_idx * fftpoint;
    const int out_offset = (dm_idx * batch + batch_idx) * fftpoint;

    Complex fft_val[ITEMS_PER_THREAD];
    Complex param[ITEMS_PER_THREAD];
    bool valid[ITEMS_PER_THREAD];

    const int n_idx0 = seg.start + local_base;

    const float inv_N = 1.0f / static_cast<float>(fftpoint);

#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; ++j)
    {
        const int local = local_base + j;
        valid[j] = local < seg.len;
    }

#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; ++j)
    {
        if (valid[j])
        {
            const int n = n_idx0 + j;
            fft_val[j] = fft_block[fft_offset + n];
            param[j] = dedisp_params[param_offset + n];
        }
    }

#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; ++j)
    {
        if (valid[j])
        {
            Complex v;
            v.x = fft_val[j].x * inv_N;
            v.y = fft_val[j].y * inv_N;

            Complex p = param[j];

            Complex y;
            y.x = v.x * p.x - v.y * p.y;
            y.y = v.x * p.y + v.y * p.x;

            ifft_input[out_offset + n_idx0 + j] = y;
        }
    }
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
    gatherMultiply_kernel<ITEMS_PER_THREAD><<<grid, block_size, 0, stream>>>(
        fft_block, dedisp_params, ifft_input, delay_block_meta, segments,
        batch, numDMs, fftpoint, read_block_index, fft_block_size);
    CUDA_CHECK(cudaGetLastError());
}

// template <int ITEMS_PER_THREAD>
// static __global__ void gatherMultiply_kernel(
//     const Complex *__restrict__ fft_block,
//     const Complex *__restrict__ dedisp_params,
//     Complex *__restrict__ ifft_input,
//     const DelayBlockMeta *__restrict__ delay_block_meta,
//     const Segment *__restrict__ segments,
//     int batch,
//     int numDMs,
//     int fftpoint,
//     int read_block_index,
//     int fft_block_size)
// {
//     const int dm_idx = blockIdx.x / batch;
//     const int batch_idx = blockIdx.x % batch;
//     const int tid = threadIdx.x;

//     if (dm_idx >= numDMs || batch_idx >= batch)
//         return;

//     __shared__ DelayBlockMeta meta;
//     __shared__ Segment segs[ITEMS_PER_THREAD];
//     __shared__ int segs_len[ITEMS_PER_THREAD];
//     __shared__ int segs_start[ITEMS_PER_THREAD];
//     __shared__ int fft_offset[ITEMS_PER_THREAD];
//     if (threadIdx.x == 0)
//     {
//         meta = delay_block_meta[dm_idx];
//         for (int j = 0; j < ITEMS_PER_THREAD; ++j)
//         {
//             const int seg_idx = blockIdx.y * ITEMS_PER_THREAD + j;
//             segs[j] = (seg_idx < meta.seg_count) ? segments[meta.seg_offset + seg_idx] : Segment{0, 0, 0};
//             const Segment seg = segs[j];
//             segs_len[j] = seg.len;
//             segs_start[j] = seg.start;
//             if (seg.len > 0)
//             {
//                 int delay_val = seg.group;
//                 int fft_idx = read_block_index + batch_idx - delay_val;
//                 fft_idx = ((fft_idx % fft_block_size) + fft_block_size) % fft_block_size;
//                 fft_offset[j] = fft_idx * fftpoint;
//             } // else 不会被访问到，后续会通过 valid[j] 判断跳过访问 fft_offset[j]
//         }
//     }
//     __syncthreads();

//     const int param_offset = dm_idx * fftpoint;
//     const int out_offset = (dm_idx * batch + batch_idx) * fftpoint;

//     Complex fft_val[ITEMS_PER_THREAD];
//     Complex param[ITEMS_PER_THREAD];
//     int n_idx[ITEMS_PER_THREAD];
//     bool valid[ITEMS_PER_THREAD];

//     const float inv_N = 1.0f / static_cast<float>(fftpoint);

// #pragma unroll
//     for (int j = 0; j < ITEMS_PER_THREAD; ++j)
//     {
//         valid[j] = tid < segs_len[j];
//         n_idx[j] = segs_start[j] + tid;
//     }

// #pragma unroll
//     for (int j = 0; j < ITEMS_PER_THREAD; ++j)
//     {
//         if (valid[j])
//         {
//             const int n = n_idx[j];
//             fft_val[j] = fft_block[fft_offset[j] + n];
//             param[j] = dedisp_params[param_offset + n];
//         }
//     }

// #pragma unroll
//     for (int j = 0; j < ITEMS_PER_THREAD; ++j)
//     {
//         if (valid[j])
//         {
//             Complex v;
//             v.x = fft_val[j].x * inv_N;
//             v.y = fft_val[j].y * inv_N;

//             Complex p = param[j];

//             Complex y;
//             y.x = v.x * p.x - v.y * p.y;
//             y.y = v.x * p.y + v.y * p.x;

//             ifft_input[out_offset + n_idx[j]] = y;
//         }
//     }
// }

// void gatherMultiply(
//     const Complex *fft_block,
//     const Complex *dedisp_params,
//     Complex *ifft_input,
//     const DelayBlockMeta *delay_block_meta,
//     const Segment *segments,
//     int &max_seg_count,
//     int &batch, int &numDMs, int &fftpoint, int &read_block_index, int &fft_block_size,
//     cudaStream_t stream)
// {
//     constexpr int ITEMS_PER_THREAD = 4;
//     const int block_size = 1024 / ITEMS_PER_THREAD;
//     dim3 grid(batch * numDMs, (max_seg_count + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD);
//     gatherMultiply_kernel<ITEMS_PER_THREAD><<<grid, block_size, 0, stream>>>(
//         fft_block, dedisp_params, ifft_input, delay_block_meta, segments,
//         batch, numDMs, fftpoint, read_block_index, fft_block_size);
//     CUDA_CHECK(cudaGetLastError());
// }

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

    // val.x = __float2uint_rn(val.x + 32768.0f) - 32768.0f;
    // val.y = __float2uint_rn(val.y + 32768.0f) - 32768.0f;

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

static __global__ void discardSamplesToUint16_kernel(uint32_t *dst, Complex *src, int M)
{
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const float scale = 1.0f / (2 * M);
    const int group = blockIdx.y;
    const int src_idx = group * 2 * M + M + offset;
    const int dst_idx = group * M + offset;

    Complex val = src[src_idx];
    float x_f = val.x * scale;
    float y_f = val.y * scale;

    uint32_t x = __float2uint_rn(x_f + (x_f != 0 ? 32768.0f : 0));
    uint32_t y = __float2uint_rn(y_f + (y_f != 0 ? 32768.0f : 0));
    dst[dst_idx] = (y << 16) | x;
}

void discardSamplesToUint16(uint16_t *dst, Complex *src, int M, int count, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    // Both M and BLOCK_SIZE are powers of 2
    const int max_groups = 8;
    dim3 grid(M / block_size, max_groups);
    // This kernel has better performance with loop-based process than a fused single kernel.
    for (int i = 0; i < count; i += max_groups)
    {
        discardSamplesToUint16_kernel<<<grid, block_size, 0, stream>>>(reinterpret_cast<uint32_t *>(dst) + i * M, src + i * 2 * M, M);
    }
    CUDA_CHECK(cudaGetLastError());
}

__global__ void calculateIntensity_kernel(Complex *a, Complex *b, float *total_intensity, unsigned long size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    total_intensity[i] = sqrtf(a[i].x * a[i].x + a[i].y * a[i].y + b[i].x * b[i].x + b[i].y * b[i].y);
}

__global__ void calculateIntensity_single_kernel(Complex *a, float *total_intensity, unsigned long size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    total_intensity[i] = sqrtf(a[i].x * a[i].x + a[i].y * a[i].y);
}

void calculateIntensity(Complex *a, Complex *b, float *total_intensity, unsigned long size, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (size + block_size - 1) / block_size;
    calculateIntensity_kernel<<<grid_size, block_size, 0, stream>>>(a, b, total_intensity, size);
    CUDA_CHECK(cudaGetLastError());
}

void calculateIntensity(Complex *a, float *total_intensity, unsigned long size, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (size + block_size - 1) / block_size;
    calculateIntensity_single_kernel<<<grid_size, block_size, 0, stream>>>(a, total_intensity, size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void foldDataPhase_kernel(
    const float *__restrict__ total_intensity,
    double *phase_bin_sum,
    unsigned long long *phase_bin_count,
    unsigned long size, unsigned long time_bin,
    double t_start, double dt, double period_seconds)
{
    unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;

    float val = total_intensity[i];
    if (val == 0.0f)
        return;

    double t = t_start + static_cast<double>(i) * dt;
    double phase = fmod(t / period_seconds, 1.0);
    unsigned long bin = static_cast<unsigned long>(floor(phase * time_bin));
    if (bin >= time_bin)
        bin = time_bin - 1;

    atomicAdd(&phase_bin_sum[bin], static_cast<double>(val));
    atomicAdd(&phase_bin_count[bin], 1ULL);
}

void foldDataPhase(const float *total_intensity, double *phase_bin_sum, unsigned long long *phase_bin_count,
                    unsigned long size, unsigned long time_bin,
                    double t_start, double dt, double period_seconds, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (size + block_size - 1) / block_size;
    foldDataPhase_kernel<<<grid_size, block_size, 0, stream>>>(
        total_intensity, phase_bin_sum, phase_bin_count, size, time_bin, t_start, dt, period_seconds);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void computeFoldedData_kernel(
    const double *__restrict__ phase_bin_sum,
    const unsigned long long *__restrict__ phase_bin_count,
    float *folded_data, unsigned long time_bin)
{
    unsigned long bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin >= time_bin)
        return;
    unsigned long long cnt = phase_bin_count[bin];
    folded_data[bin] = (cnt > 0) ? static_cast<float>(phase_bin_sum[bin] / cnt) : 0.0f;
}

void computeFoldedData(const double *phase_bin_sum, const unsigned long long *phase_bin_count,
                        float *folded_data, unsigned long time_bin, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (time_bin + block_size - 1) / block_size;
    computeFoldedData_kernel<<<grid_size, block_size, 0, stream>>>(
        phase_bin_sum, phase_bin_count, folded_data, time_bin);
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