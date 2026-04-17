#include "fold_gpu.h"

Fold_GPU::Fold_GPU(float period, float fs, unsigned long size, string outfileName, unsigned long time_bin)
{
    this->size = size;
    period_samples_float = period * fs;
    period_samples = static_cast<unsigned long>(period_samples_float);
    diff_samples = period_samples_float - period_samples;
    if (time_bin == -1)
    {
        this->time_bin = period_samples;
    }
    else
    {
        this->time_bin = time_bin;
    }
    total_intensity = new float[size];
    folded_data_raw = new float[period_samples];
    fold_count = new unsigned long[period_samples];
    tmp_fold = new float[period_samples];
    if (time_bin == -1)
        folded_data = folded_data_raw;
    else
        folded_data = new float[time_bin];
    outfile.open(outfileName);
    cudaMalloc((void **)&total_intensity_d, size * sizeof(float));
}

void Fold_GPU::discard_samples(unsigned long Nd)
{
    discard_count = static_cast<int>(ceil(Nd / period_samples_float));
    current_discard = 0;
}

template <typename T>
void Fold_GPU::calculate_intensity(T *pol1, T *pol2)
{
    // 计算2个极化方向的总强度
    Complex *a = pol1->output_buffer_d;
    Complex *b = pol2->output_buffer_d;
    calculateIntensity(a, b, total_intensity_d, size);
    cudaMemcpy(total_intensity, total_intensity_d, size * sizeof(float), cudaMemcpyDeviceToHost);
}

template void Fold_GPU::calculate_intensity<MSOSM_GPU_BATCH>(MSOSM_GPU_BATCH*, MSOSM_GPU_BATCH*);
template void Fold_GPU::calculate_intensity<OSM_GPU_BATCH>(OSM_GPU_BATCH*, OSM_GPU_BATCH*);

void Fold_GPU::fold_data()
{
    unsigned long current_size = size;
    unsigned long current_input_index = 0;

    // 写入未满一个周期的数据
    if (current_diff > 1.0f)
    {
        current_diff -= 1.0f;
        current_input_index += 1;
        current_size -= 1;
        if (current_size == 0)
            return;
    }
    if (current_index != 0)
    {
        if (current_size < period_samples - current_index)
        {
            for (unsigned long i = 0; i < current_size; i++)
            {
                if (ready)
                {
                    tmp_fold[current_index + i] = total_intensity[current_input_index + i];
                    // folded_data_raw[current_index + i] += total_intensity[current_input_index + i];
                    // fold_count[current_index + i] += (total_intensity[current_input_index + i] == 0) ? 0 : 1;
                }
            }
            current_index += current_size;
            return;
        }
        else
        {
            for (unsigned long i = 0; i < period_samples - current_index; i++)
            {
                if (ready)
                {
                    tmp_fold[current_index + i] = total_intensity[current_input_index + i];
                    // folded_data_raw[current_index + i] += total_intensity[current_input_index + i];
                    // fold_count[current_index + i] += (total_intensity[current_input_index + i] == 0) ? 0 : 1;
                }
            }
            if (ready)
            {
                for (unsigned long i = 0; i < period_samples; i++)
                {
                    folded_data_raw[i] += tmp_fold[i];
                }
            }
            current_size -= period_samples - current_index;
            current_input_index += period_samples - current_index;
            current_index = 0;
            current_diff += diff_samples;
            if (current_diff > 1.0f)
            {
                current_diff -= 1.0f;
                current_input_index += 1;
                current_size -= 1;
                if (current_size == -1)
                {
                    current_diff += 1.0f;
                    return;
                }
            }
            if (!ready)
            {
                current_discard += 1;
                if (current_discard == discard_count)
                {
                    ready = true;
                    current_discard = 0;
                }
            }
        }
    }

    // 写入已满一个周期的数据 （此段未测试，目前处理的数据长度没有超过一个周期）
    while (current_size >= period_samples)
    {
        for (unsigned long i = 0; i < period_samples; i++)
        {
            if (ready)
            {
                folded_data_raw[i] += total_intensity[current_input_index + i];
                fold_count[i] += (total_intensity[current_input_index + i] == 0) ? 0 : 1;
            }
        }
        current_size -= period_samples;
        current_input_index += period_samples;
        current_diff += diff_samples;
        if (current_diff > 1.0f)
        {
            current_diff -= 1.0f;
            current_input_index += 1;
            current_size -= 1;
        }
        if (!ready)
        {
            current_discard += 1;
            if (current_discard == discard_count)
            {
                ready = true;
                current_discard = 0;
            }
        }
    }

    // 写入剩余的数据
    if (current_size == -1)
    {
        current_diff += 1.0f;
        return;
    }
    if (current_size > 0)
    {
        for (unsigned long i = 0; i < current_size; i++)
        {
            if (ready)
            {
                tmp_fold[i] = total_intensity[current_input_index + i];
                // folded_data_raw[i] += total_intensity[current_input_index + i];
                // fold_count[i] += (total_intensity[current_input_index + i] == 0) ? 0 : 1;
            }
        }
        current_index = current_size;
    }
}

void Fold_GPU::fold_data_bins()
{
    // 可能存在无效数据，所以将数据除以该点非零数据的个数加以平均
    // for (unsigned long i = 0; i < period_samples; i++)
    // {
    //     folded_data_raw[i] /= fold_count[i];
    // }
    if (time_bin == period_samples)
        return;
    int bins_per_time_bin = period_samples / time_bin;
    for (unsigned long i = 0; i < time_bin; i++)
    {
        float sum = 0.0f;
        for (unsigned long j = 0; j < bins_per_time_bin; j++)
        {
            sum += folded_data_raw[i * bins_per_time_bin + j];
        }
        folded_data[i] = sum / bins_per_time_bin;
    }
}

void Fold_GPU::write_to_file()
{
    for (unsigned long i = 0; i < time_bin; i++)
    {
        outfile << folded_data[i] << endl;
    }
}

Fold_GPU::~Fold_GPU()
{
    outfile.close();
}