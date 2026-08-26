#include "fold_gpu.h"
#include <cmath>

Fold_GPU::Fold_GPU(float period, float fs, unsigned long size, string outfileName, unsigned long time_bin)
{
    t_start = 0.0;
    this->sampling_frequency = fs;
    this->size = size;
    this->period_seconds = period;
    period_samples_float = period * fs;
    period_samples = static_cast<unsigned long>(period_samples_float);
    if (time_bin == -1)
    {
        time_bin = period_samples;
    }
    this->time_bin = time_bin;
    total_intensity = new float[size]();
    folded_data = new float[time_bin]();
    phase_bin_sum = new double[time_bin]();
    phase_bin_count = new unsigned long long[time_bin]();
    outfile.open(outfileName);
    cudaMalloc((void **)&total_intensity_d, size * sizeof(float));
}

template <typename T>
void Fold_GPU::calculate_intensity(T *pol1, T *pol2)
{
    // 计算2个极化方向的总强度
    Complex *a = pol1->get_output_pointer();
    Complex *b = pol2->get_output_pointer();
    calculateIntensity(a, b, total_intensity_d, size);
    cudaMemcpy(total_intensity, total_intensity_d, size * sizeof(float), cudaMemcpyDeviceToHost);
}

template <typename T>
void Fold_GPU::calculate_intensity(T *pol1)
{
    Complex *a = pol1->get_output_pointer();
    calculateIntensity(a, total_intensity_d, size);
    cudaMemcpy(total_intensity, total_intensity_d, size * sizeof(float), cudaMemcpyDeviceToHost);
}

template void Fold_GPU::calculate_intensity<MSOSM_GPU_BATCH>(MSOSM_GPU_BATCH*, MSOSM_GPU_BATCH*);
template void Fold_GPU::calculate_intensity<OSM_GPU_BATCH>(OSM_GPU_BATCH*, OSM_GPU_BATCH*);
template void Fold_GPU::calculate_intensity<MSOSM_GPU_BATCH>(MSOSM_GPU_BATCH*);
template void Fold_GPU::calculate_intensity<OSM_GPU_BATCH>(OSM_GPU_BATCH*);

void Fold_GPU::fold_data_phase()
{
    const double dt = 1.0 / sampling_frequency;

    for (unsigned long i = 0; i < size; i++)
    {
        double t = t_start + static_cast<double>(i) * dt;

        double phase = fmod(t / period_seconds, 1.0);

        unsigned long bin =
            static_cast<unsigned long>(floor(phase * time_bin));

        if (bin >= time_bin)
            bin = time_bin - 1;

        if (total_intensity[i] == 0)
            continue;
        phase_bin_sum[bin] += total_intensity[i];
        phase_bin_count[bin]++;
    }

    t_start += static_cast<double>(size) * dt;
}


void Fold_GPU::get_folded_data()
{
    for (unsigned long bin = 0; bin < time_bin; bin++)
    {
        if (phase_bin_count[bin] > 0)
        {
            folded_data[bin] =
                static_cast<float>(
                    phase_bin_sum[bin] /
                    phase_bin_count[bin]
                );
        }
        else
        {
            folded_data[bin] = 0.0f;
        }
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
    delete[] phase_bin_sum;
    delete[] phase_bin_count;
}