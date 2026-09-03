#include "globals.h"
#include "msosm_gpu_batch.h"
#include "simulated_complex_signal.h"
#include <fstream>
#include <iomanip>

int main(int argc, char *argv[])
{
    verbose = false;
    int batch = 32;
    // Pulsar signal parameters
    float bw = 128e6;
    float dm = 75;
    string dm_filename_value = "75";
    float f0 = 1e9;
    int numDMs = 1;
    unsigned long fftpoint = 0;
    // Compare proces length with OSM
    unsigned long osm_process_len = 2097152 * 16;
    const struct option long_options[] = {
        {"verbose", no_argument, nullptr, 'v'},
        {"batch", required_argument, nullptr, 'b'},
        {"bw", required_argument, nullptr, 'w'},
        {"dm", required_argument, nullptr, 'd'},
        {"f0", required_argument, nullptr, 'f'},
        {"fftpoint", required_argument, nullptr, 'n'},
        {"compare", required_argument, nullptr, 'p'},
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
            osm_process_len = stoul(optarg);
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

    MSOSM_GPU_BATCH *msosm;
    SimulatedComplexSignal *simulated_signal; // Generate simulated complex signal
    unsigned long signal_size = 0;
    uint16_pair *input;

    // msosm = new MSOSM_GPU_BATCH(bw, dm, f0);
    float dm_values[numDMs];
    for (int i = 0; i < numDMs; i++)
    {
        dm_values[i] = dm + i * 0.01;
    }
    msosm = new MSOSM_GPU_BATCH(bw, dm_values, f0, numDMs);
    msosm->initialize_uint16(fftpoint, batch);
    unsigned long M = msosm->M;
    unsigned long process_len = batch * M;

    cout << "Nd: " << msosm->start_Nd << endl;

    cout << "Process Length: " << process_len << endl;
    cout << "Compared with OSM Process Length: " << osm_process_len << endl;

    float period = 0.002048;
    unsigned long block_size = static_cast<unsigned long>(period * bw);

    // Use exactly the same requested comparison length as check_osm.
    unsigned long repeat = osm_process_len / block_size;
    if (repeat == 0)
        repeat = 1;

    // MS-OSM gathers frequency segments from earlier FFT blocks.  Warm up by
    // enough complete calls to make every delayed source block valid, then
    // start folding at a pulse-period boundary.
    const unsigned long history_blocks =
        msosm->delaycount > 0 ? static_cast<unsigned long>(msosm->delaycount) : 1UL;
    const unsigned long msosm_warmup_samples = M;

    // Match check_osm's warm-up boundary. Its default M is twice the next
    // power of two at least as large as Nd. If --fftpoint overrides it, apply
    // the same lower-bound rule used by OSM_GPU_BATCH.
    unsigned long osm_order = 1;
    while (osm_order < static_cast<unsigned long>(msosm->start_Nd))
        osm_order <<= 1;
    unsigned long common_warmup_samples = 2 * osm_order;
    if (fftpoint != 0 && fftpoint / 2 > common_warmup_samples)
        common_warmup_samples = fftpoint / 2;
    if (msosm_warmup_samples > common_warmup_samples)
    {
        cerr << "The common OSM warm-up is shorter than the MS-OSM history requirement; "
             << "the two profiles cannot use the same simulated periods." << endl;
        return 1;
    }

    const unsigned long fold_start =
        ((common_warmup_samples + block_size - 1) / block_size) * block_size;
    const unsigned long required_samples = fold_start + repeat * block_size;
    const unsigned long required_process_count =
        (required_samples + process_len - 1) / process_len;
    const unsigned long generated_periods =
        (required_process_count * process_len + block_size - 1) / block_size;

    cout << "Warm-up Samples: " << fold_start << endl;
    cout << "Folded Periods: " << repeat << endl;

    simulated_signal = new SimulatedComplexSignal(bw, dm, f0, period, "uint16");
    simulated_signal->generate_pulsar_signal_new(generated_periods);
    // simulated_signal->generate_pulsar_signal(repeat, false, 0, false);
    signal_size = simulated_signal->signal_size;
    cout << "Signal Size: " << signal_size << endl;
    input = simulated_signal->signal_u16;

    // Check and plot simulated signal
    simulated_signal->plot_abs(input, block_size);

    cudaError_t error;
    error = cudaHostRegister(input, signal_size * sizeof(uint16_pair), cudaHostRegisterDefault);
    if (error != cudaSuccess)
    {
        cout << "Host Memory Registration Failed for input" << endl;
        exit(1);
    }

    int process_count = signal_size / process_len;
    cout << "Process Count: " << process_count << endl;

    uint16_pair *output;
    output = (uint16_pair *)malloc(numDMs * signal_size * sizeof(uint16_pair));
    if (output == NULL)
    {
        cout << "Memory Allocation Failed" << endl;
        exit(1);
    }
    error = cudaHostRegister(output, numDMs * signal_size * sizeof(uint16_pair), cudaHostRegisterDefault);
    if (error != cudaSuccess)
    {
        cout << "Host Memory Registration Failed" << endl;
        exit(1);
    }

    uint16_pair *current_input;

    // Start the timer
    auto start = chrono::high_resolution_clock::now();

    for (int k = 0; k < process_count; k++)
    {
        current_input = input + k * process_len;
        msosm->filter_block_uint16(current_input);
        msosm->get_output(output + k * process_len * numDMs);
    }
    msosm->synchronize();

    // Stop the timer
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
    double time = duration.count() / 1000000.0;

    plot_init();
    if (numDMs == 1)
    {
        const unsigned long fold_bins = block_size;
        vector<double> folded_abs(fold_bins, 0.0);
        vector<unsigned long> bin_counts(fold_bins, 0);
        for (unsigned long p = 0; p < repeat; p++)
        {
            unsigned long base = fold_start + p * block_size;

            for (unsigned long i = 0; i < block_size; i++)
            {
                const uint16_pair &sample = output[base + i];
                const unsigned long bin = i * fold_bins / block_size;

                float real = static_cast<float>(sample.first) - 32768.0f * (sample.first != 0);
                float imag = static_cast<float>(sample.second) - 32768.0f * (sample.second != 0);

                folded_abs[bin] += sqrt(
                    static_cast<double>(real) * real +
                    static_cast<double>(imag) * imag);
                bin_counts[bin]++;
            }
        }
        for (unsigned long i = 0; i < fold_bins; i++)
            folded_abs[i] /= static_cast<double>(bin_counts[i]);

        plot(folded_abs);
        show();

        // Save folded profile
        string filename = "check_msosm_folded_abs_" + to_string(MIN_SEGMENT_POINTS) + "_DM" + dm_filename_value + ".txt";

        ofstream abs_output(filename);

        for (unsigned long i = 0; i < fold_bins; i++)
            abs_output << folded_abs[i] << '\n';

        abs_output.close();
    }
    else
    {
        plot_abs(output + process_count * process_len * numDMs - process_len * numDMs, process_len * numDMs);
        show();
    }

    cudaHostUnregister(input);
    cudaHostUnregister(output);
    free(output);
    msosm->reset_device();

    cout << "Time taken (ms):" << time << endl;

    auto timedata = 1.0 / bw * signal_size * 1000;
    cout << "Real-time data time: " << timedata << " ms" << endl;

    return 0;
}