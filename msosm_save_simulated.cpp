#include "globals.h"
#include "simulated_complex_signal.h"
#include "msosm_data_save_uring.h"

int main(int argc, char *argv[])
{
    verbose = false;
    int numDMs = 1;
    int batch = 32;
    float startDM = -1;
    float endDM = -1;
    float DMstep = 1;
    // Pulsar signal parameters
    float bw = 128e6;
    float dm = 75;
    float f0 = 1e9;
    float testDM = -1;
    unsigned long fftpoint = 0;
    const struct option long_options[] = {
        {"verbose", no_argument, nullptr, 'v'},
        {"batch", required_argument, nullptr, 'b'},
        {"numdms", required_argument, nullptr, 'c'},
        {"lodm", required_argument, nullptr, 's'},
        {"dmstep", required_argument, nullptr, 'e'},
        {"testdm", required_argument, nullptr, 't'},
        {"bw", required_argument, nullptr, 'w'},
        {"dm", required_argument, nullptr, 'd'},
        {"f0", required_argument, nullptr, 'f'},
        {"fftpoint", required_argument, nullptr, 'n'},
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
        case 't':
            testDM = stof(optarg);
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

    // Generate simulated complex signal
    SimulatedComplexSignal *simulated_signal;
    unsigned long signal_size = 268435456;
    int inputSize;
    float period = 0.002048;
    unsigned long block_size = static_cast<unsigned long>(period * bw);
    inputSize = signal_size / block_size;
    if (testDM == -1)
        simulated_signal = new SimulatedComplexSignal(bw, dm_values[0], f0, period, "uint16");
    else
        // A small DM reduces the signal generation time,
        // which is helpful for testing the saving process.
        simulated_signal = new SimulatedComplexSignal(bw, testDM, f0, period, "uint16");
    simulated_signal->generate_pulsar_signal(inputSize, false, 0, false);
    signal_size = simulated_signal->signal_size;
    cout << "Signal Size: " << signal_size << endl;
    uint16_pair *input;
    input = simulated_signal->signal_u16;
    // Check and plot simulated signal
    simulated_signal->plot_abs(input, block_size);

    // Save simulated signal using DataSave_Uring
    MSOSM_DataSave_Uring *msosm;
    msosm = new MSOSM_DataSave_Uring(bw, dm_values, f0, numDMs);
    msosm->initialize_uint16(fftpoint, batch);
    unsigned long M = msosm->M;
    unsigned long process_len = batch * M;
    int process_count = signal_size / process_len;
    cout << "Process Length: " << process_len << endl;
    cout << "Process Count: " << process_count << endl;

    msosm->config_save("", "simulated");
    msosm->initialize_uring(4, process_count);
    msosm->start_saving();

    uint16_pair *current_input;

    // Start the timer
    auto start = chrono::high_resolution_clock::now();

    for (int k = 0; k < process_count; k++)
    {
        current_input = input + k * process_len;
        msosm->filter_block_uint16(current_input);
    }

    msosm->join_saving();
    // Stop the timer
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Total saving time: " << duration.count() << " ms" << endl;
    // You can use the function in ${PROJECT_DIR}/matlab_check to verify the saved data.
    return 0;
}