#include "osm_gpu_batch.h"

OSM_GPU_BATCH::OSM_GPU_BATCH(float bw, float dm, float f0) : Prepare_OSM(bw, dm, f0) {}

void OSM_GPU_BATCH::get_device_info()
{
    GPU_GetDevInfo();
}

/**
 * @brief Initialize the GPU batch dedispersion object for uint16 data type.
 *
 * @param fftpoint The number of points in the FFT. If 0, the minimum number of points required for the given filter order is used.
 * @param batch The number of overlapping blocks to process.
 * @param streaming Whether to use CUDA streams.
 *
 * This function allocates memory for the input and output buffers, generates the dedispersion parameters, creates the cuFFT plan, and sets up the CUDA stream.
 */
void OSM_GPU_BATCH::initialize_uint16(unsigned long fftpoint, int batch, bool streaming)
{
    this->streaming = streaming;
    if (streaming)
    {
        cudaStreamCreate(&stream);
    }
    if (verbose)
    {
        get_device_info();
    }
    calculate_min_order();
    if (fftpoint != 0)
    {
        M = fftpoint / 2;
        if (M < M_min)
        {
            cout << "Warning: The filter order is too small for the given FFT point. The minimum FFT point " << 2 * M_min << " is used." << endl;
            M = M_min;
        }
    }
    fftpoint = 2 * M;
    this->fftpoint = fftpoint;
    this->batch = batch;
    generate_dedisp_params();

    CUDA_CHECK(cudaMalloc((void **)&input_buffer_d, (batch + 1) * M * sizeof(Complex)));
    total_memory += (batch + 1) * M * sizeof(Complex);
    CUDA_CHECK(cudaMemset(input_buffer_d, 0, (batch + 1) * M * sizeof(Complex)));

    CUDA_CHECK(cudaMalloc((void **)&input_buffer_int16_d, batch * M * sizeof(uint16_pair)));
    total_memory += batch * M * sizeof(uint16_pair);

    CUDA_CHECK(cudaMalloc((void **)&dedisp_params_d, fftpoint * sizeof(Complex)));
    total_memory += fftpoint * sizeof(Complex);
    CUDA_CHECK(cudaMemcpy(dedisp_params_d, dedisp_params, fftpoint * sizeof(Complex), cudaMemcpyHostToDevice));

    // Create cuFTT plan
    int n[1] = {fftpoint};             // 1D FFT Size
    int inembed[] = {(batch + 1) * M}; // Input Size
    int onembed[] = {fftpoint};        // Output Size
    int istride = 1;                   // Input Stride
    int ostride = 1;                   // Output Stride
    int idist = M;                     // Input distance between consecutive FFT batches
    int odist = fftpoint;              // Output distance between consecutive FFT batches
    CUFFT_CHECK(cufftPlanMany(&p_f, 1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch));
    CUFFT_CHECK(cufftPlan1d(&p_b, fftpoint, CUFFT_C2C, batch));
    unsigned long workSize;
    cufftGetSize(p_f, &workSize);
    total_memory += workSize;
    cufftGetSize(p_b, &workSize);
    total_memory += workSize;

    if (streaming)
    {
        cufftSetStream(p_f, stream);
        cufftSetStream(p_b, stream);
    }

    CUDA_CHECK(cudaMalloc((void **)&fft_block_d, batch * fftpoint * sizeof(Complex)));
    total_memory += batch * fftpoint * sizeof(Complex);

    CUDA_CHECK(cudaMalloc((void **)&filtered_block_d, batch * fftpoint * sizeof(Complex)));
    total_memory += batch * fftpoint * sizeof(Complex);

    CUDA_CHECK(cudaMalloc((void **)&ifft_block_d, batch * fftpoint * sizeof(Complex)));
    total_memory += batch * fftpoint * sizeof(Complex);

    CUDA_CHECK(cudaMalloc((void **)&output_buffer_d, batch * M * sizeof(Complex)));
    total_memory += batch * M * sizeof(Complex);

    CUDA_CHECK(cudaMalloc((void **)&output_buffer_int16_d, batch * M * sizeof(uint16_pair)));
    total_memory += batch * M * sizeof(uint16_pair);
}

void OSM_GPU_BATCH::filter_block_uint16(uint16_pair *input)
{
    if (!streaming)
    {
        // 拷贝新的count段输入数据到输入缓冲区
        cudaMemcpy(input_buffer_int16_d, input, batch * M * sizeof(uint16_pair), cudaMemcpyHostToDevice);
        // 拷贝重合部分到输入缓冲区
        cudaMemcpy(input_buffer_d, input_buffer_d + batch * M, M * sizeof(Complex), cudaMemcpyDeviceToDevice);
        // 转换为复数float
        uint16ToComplex(input_buffer_d + M, (uint16_t *)input_buffer_int16_d, batch * M);
        // batch个重合点数为M的FFT同时计算
        cufftExecC2C(p_f, input_buffer_d, fft_block_d, CUFFT_FORWARD);
        // 乘以滤波器系数
        complexMultiply(fft_block_d, dedisp_params_d, fftpoint, filtered_block_d, batch);
        // 同时计算batch个IFFT
        cufftExecC2C(p_b, filtered_block_d, ifft_block_d, CUFFT_INVERSE);
        // 每个IFFT结果都需要丢弃前M个采样点，这里使用核函数来并行处理
        discardSamples(ifft_block_d, output_buffer_d, M, batch);
    }
    else
    {
        CUDA_CHECK(cudaMemcpyAsync(input_buffer_int16_d, input, batch * M * sizeof(uint16_pair), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(input_buffer_d, input_buffer_d + batch * M, M * sizeof(Complex), cudaMemcpyDeviceToDevice, stream));
        uint16ToComplex(input_buffer_d + M, (uint16_t *)input_buffer_int16_d, batch * M, stream);
        CUFFT_CHECK(cufftExecC2C(p_f, input_buffer_d, fft_block_d, CUFFT_FORWARD));
        complexMultiply(fft_block_d, dedisp_params_d, fftpoint, filtered_block_d, batch, stream);
        CUFFT_CHECK(cufftExecC2C(p_b, filtered_block_d, ifft_block_d, CUFFT_INVERSE));
        discardSamples(ifft_block_d, output_buffer_d, M, batch, stream);
    }
}

void OSM_GPU_BATCH::get_output(Complex *output)
{
    if (!streaming)
        cudaMemcpy(output, output_buffer_d, batch * M * sizeof(Complex), cudaMemcpyDeviceToHost);
    else
    {
        CUDA_CHECK(cudaMemcpyAsync(output, output_buffer_d, batch * M * sizeof(Complex), cudaMemcpyDeviceToHost, stream));
    }
}

OSM_GPU_BATCH::~OSM_GPU_BATCH()
{
    cudaDeviceReset();
}
