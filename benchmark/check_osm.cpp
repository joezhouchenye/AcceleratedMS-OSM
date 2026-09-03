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
    string dm_filename_value = "75";
    float f0 = 1e9;
    unsigned long fftpoint = 0;
    float period = 0.002048;

    unsigned long osm_process_len = 2097152 * 8;
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
            dm_filename_value = optarg;
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
    // Use the requested comparison length only, so OSM and MS-OSM fold the
    // same number of simulated pulse periods regardless of their batch sizes.
    unsigned long repeat = osm_process_len / block_size;
    if (repeat == 0)
        repeat = 1;

    // The first M output samples depend on the initially zero-filled overlap
    // buffer. Skip them and begin folding at the next pulse-period boundary.
    const unsigned long warmup_samples = osm.M;
    const unsigned long fold_start =
        ((warmup_samples + block_size - 1) / block_size) * block_size;
    const unsigned long required_samples = fold_start + repeat * block_size;
    const unsigned long required_process_count =
        (required_samples + process_len - 1) / process_len;
    const unsigned long generated_periods =
        (required_process_count * process_len + block_size - 1) / block_size;

    cout << "Warm-up Samples: " << fold_start << endl;
    cout << "Folded Periods: " << repeat << endl;

    SimulatedComplexSignal simulated_signal(bw, dm, f0, period, "uint16");
    simulated_signal.generate_pulsar_signal_new(generated_periods);
    // simulated_signal.generate_pulsar_signal(repeat, false, 0, false);
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

    vector<double> folded_abs(block_size, 0.0);
    for (unsigned long p = 0; p < repeat; p++)
    {
        unsigned long base = fold_start + p * block_size;

        for (unsigned long i = 0; i < block_size; i++)
        {
            const Complex &sample = output[base + i];

            float real = sample.x;
            float imag = sample.y;

            folded_abs[i] += std::sqrt(static_cast<double>(real) * real + static_cast<double>(imag) * imag);
        }
    }
    for (unsigned long i = 0; i < block_size; i++)
        folded_abs[i] /= static_cast<double>(repeat);

    plot_init();
    plot(folded_abs);
    show();

    // Save folded profile
    string filename = "check_osm_abs_DM" + dm_filename_value + ".txt";
    ofstream abs_output(filename);
    for (unsigned long i = 0; i < block_size; i++)
        abs_output << folded_abs[i] << '\n';

    abs_output.close();

    CUDA_CHECK(cudaHostUnregister(output));
    CUDA_CHECK(cudaHostUnregister(input));
    free(output);

    cout << "Time taken (ms): " << time << endl;
    cout << "Real-time data time: " << signal_size * 1000.0 / bw << " ms" << endl;

    return 0;
}
