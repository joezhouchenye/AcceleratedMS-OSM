#include "msosm_gpu_batch.h"

MSOSM_GPU_BATCH::MSOSM_GPU_BATCH(float bw, float dm, float f0) : Prepare_MSOSM(bw, dm, f0) {}

MSOSM_GPU_BATCH::MSOSM_GPU_BATCH(float bw, float *dm, float f0, int numDMs) : Prepare_MSOSM(bw, dm, f0, numDMs) {}

void MSOSM_GPU_BATCH::get_device_info()
{
    GPU_GetDevInfo();
}

void MSOSM_GPU_BATCH::initialize_uint16(int fftpoint, int batch, bool fold)
{
    this->fold = fold;
    if (!fold)
    {
        CUDA_CHECK(cudaStreamCreate(&fft_stream));
        CUDA_CHECK(cudaStreamCreate(&dm_stream));
        CUDA_CHECK(cudaStreamCreate(&output_stream));
        CUDA_CHECK(cudaEventCreate(&fft_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreate(&dm_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreate(&ready_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreate(&output_event, cudaEventDisableTiming));
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

    // Allocate GPU memory for original 16-bit input
    CUDA_CHECK(cudaMalloc((void **)&input_buffer_int16_d, batch * M * sizeof(uint16_pair)));

    // Allocate GPU memory for converted complex input
    input_buffer_size = (batch * D2D_REDUCE_COUNT + 1) * M;
    CUDA_CHECK(cudaMalloc((void **)&input_buffer_d, input_buffer_size * sizeof(Complex)));
    CUDA_CHECK(cudaMemset(input_buffer_d, 0, input_buffer_size * sizeof(Complex)));

    vector<vector<Segment>> segments(numDMs);
    int total_segments = 0;
    for (int i = 0; i < numDMs; i++)
    {
        segmentation(i);
        segments[i] = build_segments(delay_points[i], fftpoint);
        total_segments += segments[i].size();
    }
    // Allocate GPU memory for segments
    CUDA_CHECK(cudaMalloc((void **)&segments_d, total_segments * sizeof(Segment)));
    // Allocate GPU memory for delay block meta
    CUDA_CHECK(cudaMalloc((void **)&delay_block_meta_d, numDMs * sizeof(DelayBlockMeta)));
    vector<DelayBlockMeta> delay_block_meta(numDMs);
    // Copy segments to GPU
    int segment_offset = 0;
    int segment_count = 0;
    max_seg_count = 0;
    for (int i = 0; i < numDMs; i++)
    {
        segment_count = segments[i].size();
        CUDA_CHECK(cudaMemcpy(segments_d + segment_offset, segments[i].data(), segment_count * sizeof(Segment), cudaMemcpyHostToDevice));
        delay_block_meta[i].seg_offset = segment_offset;
        delay_block_meta[i].seg_count = segment_count;
        if (segment_count > max_seg_count)
        {
            max_seg_count = segment_count;
        }
        segment_offset += segment_count;
    }
    // Copy delay block meta to GPU
    CUDA_CHECK(cudaMemcpy(delay_block_meta_d, delay_block_meta.data(), numDMs * sizeof(DelayBlockMeta), cudaMemcpyHostToDevice));

    // Allocate GPU memory for dedispersion parameters
    CUDA_CHECK(cudaMalloc((void **)&dedisp_params_d, numDMs * fftpoint * sizeof(Complex)));
    // Copy dedispersion parameters to GPU
    for (int i = 0; i < numDMs; i++)
    {
        generate_dedisp_params(i);
        CUDA_CHECK(cudaMemcpy(dedisp_params_d + i * fftpoint, dedisp_params[i], fftpoint * sizeof(Complex), cudaMemcpyHostToDevice));
    }

    cout << "FFT point: " << fftpoint << endl;
    input_fft_d = input_buffer_d;
    input_fft_barrier = input_buffer_size - M;
    input_fft_index = 0;
    // Forward cuFTT plan
    int n[1] = {fftpoint};             // 1D FFT Size
    int inembed[] = {(batch + 1) * M}; // Input Size
    int onembed[] = {fftpoint};        // Output Size
    int istride = 1;                   // Input Stride
    int ostride = 1;                   // Output Stride
    int idist = M;                     // Input distance between consecutive FFT batches
    int odist = fftpoint;              // Output distance between consecutive FFT batches
    CUFFT_CHECK(cufftPlanMany(&p_f, 1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch));
    if (!fold)
        CUFFT_CHECK(cufftSetStream(p_f, fft_stream));
    // Allocate GPU memory for FFT result buffer
    // The required delay size is rounded up to the nearest multiple of batch size
    // delaycount is initialized after segmentation(i), don't move this part before segmentation!!!
    fft_block_size = delaycount + batch - 1;
    if (fft_block_size % batch != 0)
    {
        fft_block_size += batch - fft_block_size % batch;
    }
    if (verbose)
    {
        cout << "fft block size: " << fft_block_size << endl;
    }
    CUDA_CHECK(cudaMalloc((void **)&fft_block_d, fft_block_size * fftpoint * sizeof(Complex)));
    output_fft_d = fft_block_d;
    output_fft_barrier = fft_block_size * fftpoint;
    output_fft_index = 0;

    // Allocate GPU memory for IFFT input
    CUDA_CHECK(cudaMalloc((void **)&input_ifft_d, numDMs * batch * fftpoint * sizeof(Complex)));
    // Inverse cuFTT plan
    if (batch % ITEMS_PER_THREAD != 0)
    {
        cout << "Batch size must be a multiple of " << ITEMS_PER_THREAD << " for the current implementation." << endl;
        exit(1);
    }
    CUFFT_CHECK(cufftPlan1d(&p_b, fftpoint, CUFFT_C2C, batch / ITEMS_PER_THREAD));
    if (!fold)
        CUFFT_CHECK(cufftSetStream(p_b, dm_stream));
    // Allocate GPU memory for IFFT output
    CUDA_CHECK(cudaMalloc((void **)&output_ifft_d, numDMs * batch * fftpoint * sizeof(Complex)));

    if (fold)
    {
        // Allocate GPU memory for final output after discarding samples
        CUDA_CHECK(cudaMalloc((void **)&output_buffer_d, numDMs * batch * M * sizeof(Complex)));
    }
    else
    {
        // Allocate GPU memory for final output in uint16 format
        CUDA_CHECK(cudaMalloc((void **)&output_buffer_int16_d, numDMs * batch * M * sizeof(uint16_pair)));
    }
}

void MSOSM_GPU_BATCH::increment_fft_input()
{
    if (input_fft_index >= input_fft_barrier)
    {
        input_fft_index -= input_fft_barrier;
        if (fold)
            CUDA_CHECK(cudaMemcpy(input_buffer_d, input_buffer_d + input_fft_barrier, M * sizeof(Complex), cudaMemcpyDeviceToDevice));
        else
            CUDA_CHECK(cudaMemcpyAsync(input_buffer_d, input_buffer_d + input_fft_barrier, M * sizeof(Complex), cudaMemcpyDeviceToDevice, fft_stream));
    }
    input_fft_d = input_buffer_d + input_fft_index;
    input_fft_index += batch * M;
}

void MSOSM_GPU_BATCH::increment_fft_output()
{
    if (output_fft_index >= output_fft_barrier)
    {
        output_fft_index -= output_fft_barrier;
    }
    output_fft_d = fft_block_d + output_fft_index;
    read_block_index = output_fft_index / fftpoint;
    output_fft_index += batch * fftpoint;
}

void MSOSM_GPU_BATCH::filter_block_uint16(uint16_pair *input)
{
    if (fold)
    {
        // 拷贝新的count段输入数据到输入缓冲区
        CUDA_CHECK(cudaMemcpy(input_buffer_int16_d, input, batch * M * sizeof(uint16_pair), cudaMemcpyHostToDevice));
        // 确定FFT输入指针位置，并拷贝重合部分
        increment_fft_input();
        // 确定FFT输出指针位置
        increment_fft_output();
        // 转换为复数float
        uint16ToComplex(input_fft_d + M, (uint16_t *)input_buffer_int16_d, batch * M);
        // count个重合点数为M的FFT同时计算
        CUFFT_CHECK(cufftExecC2C(p_f, input_fft_d, output_fft_d, CUFFT_FORWARD));
        // 根据分段信息从FFT结果中提取需要的部分并乘以对应的dedispersion参数，得到IFFT输入
        gatherMultiply(fft_block_d, dedisp_params_d, input_ifft_d,
                       delay_block_meta_d, segments_d, max_seg_count,
                       batch, numDMs, fftpoint, read_block_index, fft_block_size);
        CUFFT_CHECK(cufftExecC2C(p_b, input_ifft_d, output_ifft_d, CUFFT_INVERSE));
        // 每个IFFT结果都需要丢弃前M个采样点，这里使用核函数来并行处理
        discardSamples(output_ifft_d, output_buffer_d, M, batch * numDMs);
    }
    else
    {
        cudaStreamWaitEvent(fft_stream, dm_event, 0);
        PUSH_RANGE("H2D", 0);
        CUDA_CHECK(cudaMemcpyAsync(input_buffer_int16_d, input, batch * M * sizeof(uint16_pair), cudaMemcpyHostToDevice, fft_stream));
        POP_RANGE;
        increment_fft_input();
        increment_fft_output();
        PUSH_RANGE("Convert and FFT", 1);
        uint16ToComplex(input_fft_d + M, (uint16_t *)input_buffer_int16_d, batch * M, fft_stream);
        CUFFT_CHECK(cufftExecC2C(p_f, input_fft_d, output_fft_d, CUFFT_FORWARD));
        POP_RANGE;
        cudaEventRecord(fft_event, fft_stream);
        cudaStreamWaitEvent(dm_stream, fft_event, 0);
        PUSH_RANGE("Gather and Multiply", 2);
        gatherMultiply(fft_block_d, dedisp_params_d, input_ifft_d,
                       delay_block_meta_d, segments_d, max_seg_count,
                       batch, numDMs, fftpoint, read_block_index, fft_block_size, dm_stream);
        POP_RANGE;
        cudaEventRecord(dm_event, dm_stream);
        PUSH_RANGE("IFFT", 3);
        for (int i = 0; i < numDMs * ITEMS_PER_THREAD; i++)
        {
            CUFFT_CHECK(cufftExecC2C(p_b, input_ifft_d + i * batch / ITEMS_PER_THREAD * fftpoint, output_ifft_d + i * batch / ITEMS_PER_THREAD * fftpoint, CUFFT_INVERSE));
        }
        POP_RANGE;
        // CUFFT_CHECK(cufftExecC2C(p_b, input_ifft_d, output_ifft_d, CUFFT_INVERSE));
        cudaStreamWaitEvent(dm_stream, output_event, 0);
        PUSH_RANGE("Discard and Convert", 4);
        discardSamplesToUint16((uint16_t *)output_buffer_int16_d, output_ifft_d, M, batch * numDMs, dm_stream);
        POP_RANGE;
        cudaEventRecord(ready_event, dm_stream);
    }
}

void MSOSM_GPU_BATCH::get_output(uint16_pair *output)
{
    cudaStreamWaitEvent(output_stream, ready_event, 0);
    PUSH_RANGE("D2H", 5);
    CUDA_CHECK(cudaMemcpyAsync(output, output_buffer_int16_d, numDMs * batch * M * sizeof(uint16_pair), cudaMemcpyDeviceToHost, output_stream));
    POP_RANGE;
    cudaEventRecord(output_event, output_stream);
}

void MSOSM_GPU_BATCH::synchronize()
{
    CUDA_CHECK(cudaDeviceSynchronize());
}

void MSOSM_GPU_BATCH::reset_device()
{
    CUDA_CHECK(cudaDeviceReset());
}

MSOSM_GPU_BATCH::~MSOSM_GPU_BATCH()
{
    reset_device();
}

vector<Segment> MSOSM_GPU_BATCH::build_segments(const int *delay_block, int fft_len)
{
    vector<Segment> segments;
    if (fft_len <= 0)
        return segments;

    int cur_group = delay_block[0];
    int cur_start = 0;

    for (int k = 1; k < fft_len; ++k)
    {
        // 每个segment最大长度为BLOCK_SIZE，且同一segment内的延迟一致
        // 最大长度对应GPU线程块大小，保证每个线程块处理一个segment时可以高效访问内存
        if ((delay_block[k] != cur_group) || (k - cur_start == BLOCK_SIZE))
        {
            Segment seg;
            seg.start = cur_start;
            seg.len = k - cur_start;
            seg.group = cur_group;
            segments.push_back(seg);

            cur_start = k;
            cur_group = delay_block[k];
        }
    }

    Segment seg;
    seg.start = cur_start;
    seg.len = fft_len - cur_start;
    seg.group = cur_group;
    segments.push_back(seg);

    return segments;
}