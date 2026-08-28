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
    folded_data = new float[time_bin]();
    outfile.open(outfileName);
    cudaMalloc((void **)&total_intensity_d, size * sizeof(float));
    cudaMalloc((void **)&phase_bin_sum_d, time_bin * sizeof(double));
    cudaMalloc((void **)&phase_bin_count_d, time_bin * sizeof(unsigned long long));
    cudaMalloc((void **)&folded_data_d, time_bin * sizeof(float));
    cudaMemset(phase_bin_sum_d, 0, time_bin * sizeof(double));
    cudaMemset(phase_bin_count_d, 0, time_bin * sizeof(unsigned long long));
}

template <typename T>
void Fold_GPU::calculate_intensity(T *pol1, T *pol2)
{
    // 计算2个极化方向的总强度
    Complex *a = pol1->get_output_pointer();
    Complex *b = pol2->get_output_pointer();
    calculateIntensity(a, b, total_intensity_d, size);
}

template <typename T>
void Fold_GPU::calculate_intensity(T *pol1)
{
    Complex *a = pol1->get_output_pointer();
    calculateIntensity(a, total_intensity_d, size);
}

template void Fold_GPU::calculate_intensity<MSOSM_GPU_BATCH>(MSOSM_GPU_BATCH*, MSOSM_GPU_BATCH*);
template void Fold_GPU::calculate_intensity<OSM_GPU_BATCH>(OSM_GPU_BATCH*, OSM_GPU_BATCH*);
template void Fold_GPU::calculate_intensity<MSOSM_GPU_BATCH>(MSOSM_GPU_BATCH*);
template void Fold_GPU::calculate_intensity<OSM_GPU_BATCH>(OSM_GPU_BATCH*);

void Fold_GPU::fold_data_phase()
{
    const double dt = 1.0 / sampling_frequency;
    foldDataPhase(total_intensity_d, phase_bin_sum_d, phase_bin_count_d,
                  size, time_bin, t_start, dt, period_seconds);
    t_start += static_cast<double>(size) * dt;
}

void Fold_GPU::get_folded_data()
{
    computeFoldedData(phase_bin_sum_d, phase_bin_count_d, folded_data_d, time_bin);
    cudaMemcpy(folded_data, folded_data_d, time_bin * sizeof(float), cudaMemcpyDeviceToHost);
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
    cudaFree(total_intensity_d);
    cudaFree(phase_bin_sum_d);
    cudaFree(phase_bin_count_d);
    cudaFree(folded_data_d);
}