#include "globals.h"
#include "osm_gpu_batch.h"
#include "simulated_complex_signal.h"
#include <fstream>

int main(int argc, char *argv[])
{
    verbose = false;
    int batch = 1;
    float bw = 128e6;
    float dm = 75;
    float f0 = 1e9;
    unsigned long fftpoint = 0;
    float period = 0.002048;

    unsigned long osm_process_len = 268435456;
    const struct option long_options[] = {
        {"verbose", no_argument, nullptr, 'v'},
        {"batch", required_argument, nullptr, 'b'},
        {"bw", required_argument, nullptr, 'w'},
        {"dm", required_argument, nullptr, 'd'},
        {"f0", required_argument, nullptr, 'f'},
        {"fftpoint", required_argument, nullptr, 'n'},
        {"period", required_argument, nullptr, 'p'},
        {nullptr, 0, nullptr, 0}};

    for (;;)
    {
        switch (getopt_long(argc, argv, "", long_options, nullptr))
        {
        case 'v':
            verbose = true;
            continue;
        case 'b':
            batch = stoi(optarg);
            continue;
        case 'w':
            bw = stof(optarg);
            continue;
        case 'd':
            dm = stof(optarg);
            continue;
        case 'f':
            f0 = stof(optarg);
            continue;
        case 'n':
            fftpoint = stoul(optarg);
            continue;
        case 'p':
            period = stof(optarg);
            continue;
        default:
            continue;
        case -1:
            break;
        }
        break;
    }

    cout << "Bandwidth: " << bw / 1e6 << " MHz" << endl;
    cout << "Dispersion measure: " << dm << " pc cm^-3" << endl;
    cout << "Start frequency: " << f0 / 1e6 << " MHz" << endl;

    OSM_GPU_BATCH osm(bw, dm, f0);
    osm.initialize_uint16(fftpoint, batch);
    unsigned long process_len = batch * osm.M;

    cout << "FFT point: " << 2 * osm.M << endl;
    cout << "Process Length: " << process_len << endl;

    unsigned long block_size = static_cast<unsigned long>(period * bw);
    unsigned long repeat = process_len / block_size;
    if (repeat == 0)
        repeat = 1;
    if (repeat < osm_process_len / block_size)
        repeat = osm_process_len / block_size;
    if (repeat == 0)
        repeat = 1;

    SimulatedComplexSignal simulated_signal(bw, dm, f0, period, "uint16");
    simulated_signal.generate_pulsar_signal_new(repeat, false, 0, false);
    unsigned long signal_size = simulated_signal.signal_size;
    uint16_pair *input = simulated_signal.signal_u16;
    cout << "Signal Size: " << signal_size << endl;

    cudaError_t error = cudaHostRegister(input, signal_size * sizeof(uint16_pair), cudaHostRegisterDefault);
    if (error != cudaSuccess)
    {
        cout << "Host Memory Registration Failed for input" << endl;
        return 1;
    }

    unsigned long process_count = signal_size / process_len;
    cout << "Process Count: " << process_count << endl;

    Complex *output = (Complex *)malloc(signal_size * sizeof(Complex));
    if (output == nullptr)
    {
        cout << "Memory Allocation Failed" << endl;
        cudaHostUnregister(input);
        return 1;
    }
    error = cudaHostRegister(output, signal_size * sizeof(Complex), cudaHostRegisterDefault);
    if (error != cudaSuccess)
    {
        cout << "Host Memory Registration Failed for output" << endl;
        free(output);
        cudaHostUnregister(input);
        return 1;
    }

    auto start = chrono::high_resolution_clock::now();
    for (unsigned long i = 0; i < process_count; i++)
    {
        osm.filter_block_uint16(input + i * process_len);
        osm.get_output(output + i * process_len);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto stop = chrono::high_resolution_clock::now();

    double time = chrono::duration_cast<chrono::nanoseconds>(stop - start).count() / 1000000.0;
    Complex *plot_output = output + (process_count - 1) * process_len;
    plot_init();
    plot_abs(plot_output, block_size);
    show();

    ofstream abs_output("check_osm_abs.txt");
    for (unsigned long i = 0; i < block_size; i++)
    {
        abs_output << sqrt(plot_output[i].x * plot_output[i].x + plot_output[i].y * plot_output[i].y) << '\n';
    }

    CUDA_CHECK(cudaHostUnregister(output));
    CUDA_CHECK(cudaHostUnregister(input));
    free(output);

    cout << "Time taken (ms): " << time << endl;
    cout << "Real-time data time: " << signal_size * 1000.0 / bw << " ms" << endl;

    return 0;
}
