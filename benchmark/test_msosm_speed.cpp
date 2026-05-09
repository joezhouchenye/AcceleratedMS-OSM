#include "globals.h"
#include "msosm_gpu_batch.h"
#include "simulated_complex_signal.h"
#include <fstream>
#include <iomanip>

int main(int argc, char *argv[])
{
    verbose = false;
    int numDMs = 1;
    int batch = 32;
    float startDM = -1;
    float endDM = -1;
    float DMstep = 1;
    int repeat = 10;
    // Pulsar signal parameters
    float bw = 128e6;
    float dm = 75;
    float f0 = 1e9;
    unsigned long fftpoint = 0;
    // Compare proces length with OSM
    unsigned long osm_process_len = 268435456;
    const struct option long_options[] = {
        {"verbose", no_argument, nullptr, 'v'},
        {"batch", required_argument, nullptr, 'b'},
        {"numdms", required_argument, nullptr, 'c'},
        {"lodm", required_argument, nullptr, 's'},
        {"dmstep", required_argument, nullptr, 'e'},
        {"bw", required_argument, nullptr, 'w'},
        {"dm", required_argument, nullptr, 'd'},
        {"f0", required_argument, nullptr, 'f'},
        {"fftpoint", required_argument, nullptr, 'n'},
        {"compare", required_argument, nullptr, 'p'},
        {"repeat", required_argument, nullptr, 'r'},
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
        case 'c':
            numDMs = stoi(optarg);
            continue;
        case 's':
            startDM = stof(optarg);
            continue;
        case 'e':
            DMstep = stof(optarg);
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
            osm_process_len = stoul(optarg);
            continue;
        case 'r':
            repeat = stoi(optarg);
            continue;
        default:
            continue;
        case -1:
            break;
        }
        break;
    }

    cout << "Bandwidth: " << bw / 1e6 << " MHz" << endl;
    cout << "Start frequency: " << f0 / 1e6 << " MHz" << endl;

    // numDMs
    if (startDM == -1)
    {
        cout << "Default dispersion measure: " << dm << " pc cm^-3" << endl;
    }
    else
    {
        endDM = startDM + (numDMs - 1) * DMstep;
        cout << "DM Trial Range: " << startDM << " to " << endDM << endl;
        cout << "DM Step: " << DMstep << endl;
    }

    float *dm_values = new float[numDMs];
    for (int i = 0; i < numDMs; i++)
    {
        if (startDM == -1)
        {
            dm_values[i] = dm;
        }
        else
        {
            dm_values[i] = startDM + i * DMstep;
        }
    }

    MSOSM_GPU_BATCH *msosm;
    SimulatedComplexSignal *simulated_signal;
    unsigned long signal_size = 0;
    uint16_pair *input;

    // For execution time measurement
    double sum = 0;
    double sum_of_squares = 0;

    msosm = new MSOSM_GPU_BATCH(bw, dm_values, f0, numDMs);
    msosm->initialize_uint16(fftpoint, batch);

    unsigned long M = msosm->M;
    unsigned long process_len = batch * M;

    if (startDM == -1)
    {
        cout << "Nd: " << msosm->Nd_values[0] << endl;
    }
    else
    {
        cout << "Nd range: " << msosm->start_Nd << " to " << msosm->end_Nd << endl;
    }

    unsigned long max_process_len;
    max_process_len = batch * M;
    cout << "Max Process Length: " << max_process_len << endl;
    cout << "Compared with OSM Process Length: " << osm_process_len << endl;

    // Generate simulated complex signal
    // Use different parameters here to reduce generation time,
    // since we only need to test the speed
    int inputSize;
    unsigned long block_size = 8388608;
    float period = (float)block_size / 16e6;
    // Assume max_process_len and block_size are powers of 2
    inputSize = max_process_len / block_size;
    if (inputSize == 0)
        inputSize = 1;
    if (inputSize > osm_process_len / block_size)
    {
        cout << "The compared OSM process length is too short" << endl;
    }
    else
    {
        inputSize = osm_process_len / block_size;
        if (inputSize == 0)
            inputSize = 1;
    }
    simulated_signal = new SimulatedComplexSignal(16e6, 75, f0, period, "uint16");
    simulated_signal->generate_pulsar_signal(inputSize, false, 0, false);
    signal_size = simulated_signal->signal_size;
    cout << "Signal Size: " << signal_size << endl;
    input = simulated_signal->signal_u16;

    cudaError_t error;
    error = cudaHostRegister(input, signal_size * sizeof(uint16_pair), cudaHostRegisterDefault);
    if (error != cudaSuccess)
    {
        cout << "Host Memory Registration Failed for input" << endl;
        exit(1);
    }

    uint16_pair *output;
    cudaMallocHost(&output, numDMs * process_len * sizeof(uint16_pair));

    for (int i = 0; i < repeat; i++)
    {
        uint16_pair *current_input;

        // Start the timer
        auto start = chrono::high_resolution_clock::now();

        for (int k = 0; k < signal_size / process_len; k++)
        {
            current_input = input + k * process_len;
            msosm->filter_block_uint16(current_input);
            msosm->get_output(output);
        }
        msosm->synchronize();

        // Stop the timer
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
        double time = duration.count() / 1000000.0;
        sum += time;
        sum_of_squares += time * time;
    }
    cudaHostUnregister(input);
    msosm->reset_device();
    double mean = sum / repeat;
    double variance = (sum_of_squares / repeat) - (mean * mean);
    double standard_deviation = sqrt(variance);
    cout << "Time taken (ms) with " << repeat << " runs:" << endl;
    cout << "Mean time: " << mean << " ms" << endl;
    cout << "Standard deviation: " << standard_deviation << " ms" << endl;

    auto timedata = 1.0 / bw * signal_size * 1000;
    cout << "Real-time data time: " << timedata << " ms" << endl;

    return 0;
}