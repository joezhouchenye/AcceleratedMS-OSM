#include "globals.h"
#include "msosm_gpu_batch.h"
#include "fold_gpu.h"
#include "psrdada.h"

using namespace std;

void run_psrdada_GPU_batch(string filename, float dm, float period = 0);

clock_t start_time, end_time;
double gpu_duration;
unsigned long filter_order_GPU, filter_order_CPU;

unsigned long fft_point = 4194304;

int main(int argc, char *argv[])
{
    line();
    banner();
    line();

    string filename;

    // Whether only to separate interleaved data
    bool separate = false;

    // Pulsar signal parameters
    float bw = 16e6;
    float dm = 750;
    float f0 = 1e9;
    float default_period = 0.032768;

    // Whether to plot
    bool plot_signal = false;

    // Whether to add white Gaussian noise (only for simulation)
    float SNR;
    bool add_noise = false;

    float period = 0;

    const struct option long_options[] = {
        {"verbose", no_argument, nullptr, 'v'},
        {"file", required_argument, nullptr, 'f'},
        {"dm", required_argument, nullptr, 'd'},
        {"period", required_argument, nullptr, 'p'},
        {"show", no_argument, nullptr, 's'},
        {nullptr, 0, nullptr, 0}};

    for (;;)
    {
        switch (getopt_long(argc, argv, "", long_options, nullptr))
        {
        case 'v':
            verbose = true;
            continue;
        case 'f':
            filename = optarg;
            continue;
        case 'd':
            dm = stof(optarg);
            continue;
        case 'p':
            period = stof(optarg);
            continue;
        case 's':
            plot_signal = true;
            continue;
        default:
            continue;
        case -1:
            break;
        }
        break;
    }

    if (period == 0)
    {
        cout << "Please specify the period." << endl;
        exit(1);
    }
    run_psrdada_GPU_batch(filename, dm, period);

    // Performance comparison
    cout << "Performance comparison..." << endl;
    cout << "GPU MS-OSM: " << gpu_duration << "s" << endl;
    line();

    return 0;
}

void run_psrdada_GPU_batch(string filename, float dm, float period)
{
    int count = 1;
    unsigned long fftpoint = 2097152;
    unsigned long time_bin = 1024;

    PSRDADA file(filename);
    line();
    float bw, f0;
    bw = file.bw;
    f0 = file.fc - bw / 2;
    bw = bw * 1e6;
    f0 = f0 * 1e6;
    MSOSM_GPU_BATCH msosm_gpu1(bw, dm, f0);
    MSOSM_GPU_BATCH msosm_gpu2(bw, dm, f0);
    msosm_gpu1.initialize_uint16(fftpoint, count);
    msosm_gpu2.initialize_uint16(fftpoint, count);
    unsigned long M = msosm_gpu1.M;
    file.initBuffer(count * M);
    Fold_GPU fold(period, bw, count * M, file.outfileName, time_bin);
    fold.discard_samples(msosm_gpu1.Nd_values[0]);
    bool first = true;
    for (unsigned long i = 0; i < file.readCount; i++)
    {
        // 每10%显示一次
        if (file.readCount > 100)
        {
            if (i % (file.readCount / 100) == 0)
            {
                cout << "\r" << i * 100 / file.readCount << "%" << flush;
            }
        }
        file.readSamples();
        // 去掉文件头部的无效数据
        if (first)
        {
            if (file.pol1_in[0].first == 0)
                continue;
            else
                first = false;
        }
        // plot_pol1(file.pol1_in, count * M);
        msosm_gpu1.filter_block_uint16(file.pol1_in);
        msosm_gpu2.filter_block_uint16(file.pol2_in);
        fold.calculate_intensity<MSOSM_GPU_BATCH>(&msosm_gpu1, &msosm_gpu2);
        // plot(fold.total_intensity, count * M);
        fold.fold_data();
    }
    // plot(fold.fold_count, fold.period_samples);
    fold.fold_data_bins();
    fold.write_to_file();
    cout << "\rFinish" << endl;
}