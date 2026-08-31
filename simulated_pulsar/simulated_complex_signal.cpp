#include "simulated_complex_signal.h"
#include <omp.h>

/**
 * @brief SimulatedComplexSignal constructor
 * @param bw Bandwidth
 * @param dm Dispersion measure
 * @param f0 Start frequency
 * @param period Pulsar period
 *
 * This constructor initializes the parameters of the simulated complex signal.
 */
SimulatedComplexSignal::SimulatedComplexSignal(float bw, float dm, float f0, float period, string mode)
{
    this->data_type = mode;

    this->bw = bw;
    this->dm = dm;
    this->f0 = f0;
    this->period = period;

    this->fs = bw;
    this->kdm = 4.15 * 1e15;
    this->w0 = 2 * pi * f0;
    this->Np = static_cast<unsigned long>(period * fs);
    this->Nd = static_cast<unsigned long>(floor(kdm * dm * (1 / pow(f0, 2) - 1 / pow(f0 + bw, 2)) / (1 / fs)));

    if (verbose)
    {
        cout << "Test signal parameters:" << endl;
        cout << "Bandwidth: " << bw << "Hz" << endl;
        cout << "Dispersion measure: " << dm << "pc cm^-3" << endl;
        cout << "Start frequency: " << f0 << "Hz" << endl;
        cout << "Pulsar period: " << period << "s" << endl;
        cout << "Period samples: " << Np << endl;
        cout << "Dispersion samples: " << Nd << endl;
        line();
    }
}

void SimulatedComplexSignal::generate_pulsar_signal_new(
    unsigned long repeat,
    bool add_noise,
    float SNR,
    bool pinned)
{
    if (verbose)
        cout << "Generating pulsar signal" << endl;

    if (add_noise && verbose)
    {
        cout << "Noise will be added..." << endl;
        cout << "SNR: " << SNR << " dB" << endl;
        line();
    }

    signal_size = repeat * Np;

    if (pinned)
        cudaMallocHost((void **)&this->signal, sizeof(fftwf_complex) * signal_size);
    else
        this->signal = (fftwf_complex *)malloc(sizeof(fftwf_complex) * signal_size);

    if (data_type == "uint16")
    {
        if (pinned)
            cudaMallocHost((void **)&this->signal_u16, sizeof(uint16_pair) * signal_size);
        else
            this->signal_u16 = (uint16_pair *)malloc(sizeof(uint16_pair) * signal_size);
    }

    range = static_cast<unsigned long>(ceil((double)Nd / (double)Np));

    if (verbose)
    {
        cout << "Dispersion spread periods: " << range << endl;
    }

    // ---------------------------------------------------------
    // Generate one original pulse period
    // ---------------------------------------------------------

    fftwf_complex *one_period = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * Np);
    memset(one_period, 0, sizeof(fftwf_complex) * Np);

    unsigned long pulse_position = Np / 2 - 1;

    one_period[pulse_position][0] = 10000.0f;
    one_period[pulse_position][1] = 0.0f;
    one_period[pulse_position + 2][0] = 10000.0f;
    one_period[pulse_position + 2][1] = 0.0f;

    // ---------------------------------------------------------
    // Generate a sufficiently long periodic pulse train
    // ---------------------------------------------------------

    unsigned long guard_periods = range + 2;
    unsigned long num_periods = 2 * guard_periods + 1;
    unsigned long long_period_size = num_periods * Np;
    // FFT length >= signal length + dispersion spread
    unsigned long required_size = long_period_size + Nd;
    unsigned long fftpoint = 1;
    while (fftpoint < required_size)
        fftpoint <<= 1;

    if (verbose)
    {
        cout << "Number of simulated periods: " << num_periods << endl;
        cout << "Using FFT points = " << fftpoint << endl;
    }

    fftwf_complex *signal_long = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftpoint);

    memset(signal_long, 0, sizeof(fftwf_complex) * fftpoint);

    // Repeat original pulse period
    for (unsigned long i = 0; i < num_periods; i++)
    {
        memcpy(signal_long + i * Np, one_period, sizeof(fftwf_complex) * Np);
    }

    // ---------------------------------------------------------
    // Construct theoretical dispersion transfer function
    // directly at fftpoint frequency bins
    // ---------------------------------------------------------

    fftwf_complex *H = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftpoint);

    const double pi_d = 3.14159265358979323846;
    const double kdm_d = 4.15e15;
    const double fs_d = static_cast<double>(fs);
    const double dm_d = static_cast<double>(dm);
    const double f0_d = static_cast<double>(f0);
    const double w0_d = 2.0 * pi_d * f0_d;

    const double freq_step =
        fs_d / static_cast<double>(fftpoint);

#pragma omp parallel for schedule(static)
    for (unsigned long k = 0; k < fftpoint; k++)
    {
        double f_disp;

        if (k < fftpoint / 2)
            f_disp = fs_d / 2.0 +
                     static_cast<double>(k) * freq_step;
        else
            f_disp = static_cast<double>(k) * freq_step -
                     fs_d / 2.0;

        const double w = 2.0 * pi_d * f_disp;

        const double phase =
            4.0 * pi_d * pi_d *
            kdm_d * dm_d *
            w * w /
            ((w + w0_d) * w0_d * w0_d);

        H[k][0] = static_cast<float>(std::cos(phase));
        H[k][1] = static_cast<float>(std::sin(phase));
    }

    // ---------------------------------------------------------
    // Add dispersion:
    // FFT(signal) -> multiply H -> IFFT
    // ---------------------------------------------------------

    fftwf_plan p_f = fftwf_plan_dft_1d(fftpoint, signal_long, signal_long, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_plan p_b = fftwf_plan_dft_1d(fftpoint, signal_long, signal_long, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(p_f);

    for (unsigned long k = 0; k < fftpoint; k++)
    {
        float real = signal_long[k][0] * H[k][0] - signal_long[k][1] * H[k][1];
        float imag = signal_long[k][0] * H[k][1] + signal_long[k][1] * H[k][0];
        signal_long[k][0] = real;
        signal_long[k][1] = imag;
    }

    fftwf_execute(p_b);

    // FFTW backward transform is unnormalized
    const float inv_fftpoint = 1.0f / static_cast<float>(fftpoint);

#pragma omp parallel for schedule(static)
    for (unsigned long i = 0; i < fftpoint; i++)
    {
        signal_long[i][0] *= inv_fftpoint;
        signal_long[i][1] *= inv_fftpoint;
    }

    // ---------------------------------------------------------
    // Extract central period
    // ---------------------------------------------------------

    unsigned long center_period = guard_periods;
    unsigned long start_index = center_period * Np;

    // Save one dispersed period in this->signal[0:Np]
    memcpy(this->signal, signal_long + start_index, sizeof(fftwf_complex) * Np);

    fftwf_destroy_plan(p_f);
    fftwf_destroy_plan(p_b);

    fftwf_free(H);
    fftwf_free(signal_long);
    fftwf_free(one_period);

    // ---------------------------------------------------------
    // Measure signal power
    // ---------------------------------------------------------

    double power_sum = 0.0;

#pragma omp parallel for reduction(+ : power_sum) schedule(static)
    for (unsigned long i = 0; i < Np; i++)
    {
        power_sum += this->signal[i][0] * this->signal[i][0] + this->signal[i][1] * this->signal[i][1];
    }

    signal_power = 10.0 * log10(power_sum / static_cast<double>(Np));

    if (verbose)
    {
        cout << "Signal measured power: " << signal_power << " dB" << endl;
        line();
    }

    // ---------------------------------------------------------
    // Generate noise
    // ---------------------------------------------------------

    fftwf_complex *noise = NULL;
    AWGN g(SNR - signal_power, signal_size);
    if (add_noise)
        noise = g.generateNoiseSamples();

    if (verbose)
    {
        cout << "Repeat signal..." << endl;
        cout << "Repeat: " << repeat << endl;
        cout << "Signal Size: " << signal_size << endl;
    }

    if (!add_noise && verbose)
    {
        cout << "No noise added!" << endl;
    }

    if (verbose)
        line();

    // ---------------------------------------------------------
    // Repeat dispersed period
    // ---------------------------------------------------------

    if (!add_noise)
    {
        if (data_type == "uint16")
        {
#pragma omp parallel for schedule(static)
            for (unsigned long j = 0; j < Np; j++)
            {
                float real = this->signal[j][0];
                float imag = this->signal[j][1];
                this->signal_u16[j].first = static_cast<uint16_t>(real + 32768.0f);
                this->signal_u16[j].second = static_cast<uint16_t>(imag + 32768.0f);
            }

#pragma omp parallel for schedule(static)
            for (unsigned long i = 1; i < repeat; i++)
            {
                memcpy(this->signal + i * Np, this->signal, sizeof(fftwf_complex) * Np);
                memcpy(this->signal_u16 + i * Np, this->signal_u16, sizeof(uint16_pair) * Np);
            }
        }
        else
        {
#pragma omp parallel for schedule(static)
            for (unsigned long i = 1; i < repeat; i++)
            {
                memcpy(this->signal + i * Np, this->signal, sizeof(fftwf_complex) * Np);
            }
        }
    }
    else
    {
#pragma omp parallel for schedule(static)
        for (unsigned long i = 0; i < repeat; i++)
        {
            unsigned long base = i * Np;
            for (unsigned long j = 0; j < Np; j++)
            {
                unsigned long idx = base + j;
                this->signal[idx][0] = this->signal[j][0] + noise[idx][0];
                this->signal[idx][1] = this->signal[j][1] + noise[idx][1];

                if (data_type == "uint16")
                {
                    float real = this->signal[idx][0];
                    float imag = this->signal[idx][1];
                    real = max(-32768.0f, min(32767.0f, real));
                    imag = max(-32768.0f, min(32767.0f, imag));
                    this->signal_u16[idx].first = static_cast<uint16_t>(real + 32768.0f);
                    this->signal_u16[idx].second = static_cast<uint16_t>(imag + 32768.0f);
                }
            }
        }
    }

    if (add_noise)
        g.deallocate();
}

void SimulatedComplexSignal::generate_pulsar_signal(unsigned long repeat, bool add_noise, float SNR, bool pinned)
{
    if (verbose)
        cout << "Generating pulsar signal" << endl;

    if (add_noise && verbose)
    {
        cout << "Noise will be added..." << endl;
        cout << "SNR: " << SNR << " dB" << endl;
        line();
    }

    signal_size = repeat * Np;
    if (pinned)
        cudaMallocHost((void **)&this->signal, sizeof(fftwf_complex) * signal_size);
    else
        this->signal = (fftwf_complex *)malloc(sizeof(fftwf_complex) * signal_size);
    if (data_type == "uint16")
    {
        if (pinned)
            cudaMallocHost((void **)&this->signal_u16, sizeof(uint16_pair) * signal_size);
        else
            this->signal_u16 = (uint16_pair *)malloc(sizeof(uint16_pair) * signal_size);
    }

    range = static_cast<unsigned long>(ceil((float)Nd / (float)Np));
    if (verbose)
    {
        cout << "Dispersion spread periods: " << range << endl;
    }
    unsigned long order = static_cast<unsigned long>(pow(2, ceil(log2(range * Np))));
    unsigned long fftpoint = 2 * order;
    if (verbose)
    {
        cout << "Using FFT points = " << fftpoint;
        cout << " (period points = " << Np << ")" << endl;
    }

    fftwf_complex *signal;
    signal = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * Np);

    // Dispersion filter frequency response
    // float w[fftpoint];
    const double pi_d = 3.14159265358979323846;
    const double kdm_d = 4.15e15;
    const double fs_d = static_cast<double>(fs);
    const double dm_d = static_cast<double>(dm);
    const double f0_d = static_cast<double>(f0);
    const double w0_d = 2.0 * pi_d * f0_d;
    const double step = 2.0 * pi_d * fs_d / static_cast<double>(order);

    fftwf_complex *H1, *H;
    H1 = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * order);
    H = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftpoint);

#pragma omp parallel for schedule(static)
    for (unsigned long i = 0; i < order / 2; i++)
    {
        double w = step * static_cast<double>(i);
        double phase = 4.0 * pi_d * pi_d * kdm_d * dm_d * w * w / ((w + w0_d) * w0_d * w0_d);

        H1[order / 2 + i][0] = static_cast<float>(std::cos(phase));
        H1[order / 2 + i][1] = static_cast<float>(std::sin(phase));
    }
#pragma omp parallel for schedule(static)
    for (unsigned long i = order / 2; i < order; i++)
    {
        double w = step * static_cast<double>(i);
        double phase = 4.0 * pi_d * pi_d * kdm_d * dm_d * w * w / ((w + w0_d) * w0_d * w0_d);

        H1[i - order / 2][0] = static_cast<float>(std::cos(phase));
        H1[i - order / 2][1] = static_cast<float>(std::sin(phase));
    }
    fftwf_plan p;
    p = fftwf_plan_dft_1d(order, H1, H1, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);
    for (unsigned long i = 0; i < order; i++)
    {
        H1[i][0] = H1[i][0] / order;
        H1[i][1] = H1[i][1] / order;
    }

    memset(H, 0, sizeof(fftwf_complex) * fftpoint);
    memcpy(H + order, H1, sizeof(fftwf_complex) * order);
    fftwf_free(H1);
    p = fftwf_plan_dft_1d(fftpoint, H, H, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);

    // Add dispersion
    fftwf_complex *dummy_buf = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftpoint);
    fftwf_plan p_f = fftwf_plan_dft_1d(fftpoint, dummy_buf, dummy_buf, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_plan p_b = fftwf_plan_dft_1d(fftpoint, dummy_buf, dummy_buf, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_free(dummy_buf);

    memset(this->signal, 0, sizeof(fftwf_complex) * Np);

    /// Generate original signal
    memset(signal, 0, sizeof(fftwf_complex) * Np);
    unsigned long pulse_position = Np / 2;
    signal[pulse_position][0] = 32767.0f;
    signal[pulse_position + 2][0] = 32767.0f;

    const float inv_fftpoint = 1.0f / fftpoint;

#pragma omp parallel
    {
        fftwf_complex *thread_signal_fd = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftpoint);
        fftwf_complex *thread_local_signal = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * Np);
        memset(thread_local_signal, 0, sizeof(fftwf_complex) * Np);

#pragma omp for schedule(dynamic)
        for (unsigned long i = 0; i < range; i++)
        {
            memset(thread_signal_fd, 0, sizeof(fftwf_complex) * fftpoint);
            memcpy(thread_signal_fd + i * Np, signal, sizeof(fftwf_complex) * Np);
            fftwf_execute_dft(p_f, thread_signal_fd, thread_signal_fd);
            for (unsigned long j = 0; j < fftpoint; j++)
            {
                float real = (thread_signal_fd[j][0] * H[j][0] - thread_signal_fd[j][1] * H[j][1]) * inv_fftpoint;
                float imag = (thread_signal_fd[j][0] * H[j][1] + thread_signal_fd[j][1] * H[j][0]) * inv_fftpoint;
                thread_signal_fd[j][0] = real;
                thread_signal_fd[j][1] = imag;
            }
            fftwf_execute_dft(p_b, thread_signal_fd, thread_signal_fd);
            for (unsigned long j = 0; j < Np; j++)
            {
                thread_local_signal[j][0] += thread_signal_fd[j][0];
                thread_local_signal[j][1] += thread_signal_fd[j][1];
            }
        }

#pragma omp critical
        {
            for (unsigned long j = 0; j < Np; j++)
            {
                this->signal[j][0] += thread_local_signal[j][0];
                this->signal[j][1] += thread_local_signal[j][1];
            }
        }

        fftwf_free(thread_signal_fd);
        fftwf_free(thread_local_signal);
    }

    fftwf_destroy_plan(p_f);
    fftwf_destroy_plan(p_b);

    fftwf_free(signal);
    fftwf_free(H);

    double power_sum = 0;
#pragma omp parallel for reduction(+ : power_sum) schedule(static)
    for (unsigned long i = 0; i < Np; i++)
    {
        power_sum += this->signal[i][0] * this->signal[i][0] + this->signal[i][1] * this->signal[i][1];
    }
    signal_power = 10 * log10(power_sum / Np);
    if (verbose)
    {
        cout << "Signal measured power: " << signal_power << " dB" << endl;
        line();
    }

    fftwf_complex *noise = NULL;
    AWGN g(SNR - signal_power, signal_size);
    if (add_noise)
    {
        noise = g.generateNoiseSamples();
    }

    if (verbose)
    {
        cout << "Repeat signal..." << endl;
        cout << "Repeat: " << repeat << endl;
        cout << "Signal Size: " << signal_size << endl;
    }

    if (!add_noise && verbose)
    {
        cout << "No noise added!" << endl;
    }

    if (verbose)
        line();

    if (!add_noise)
    {
        if (data_type == "uint16")
        {
#pragma omp parallel for schedule(static)
            for (unsigned long j = 0; j < Np; j++)
            {
                this->signal_u16[j].first = static_cast<uint16_t>(this->signal[j][0] + 32768.0f);
                this->signal_u16[j].second = static_cast<uint16_t>(this->signal[j][1] + 32768.0f);
            }
#pragma omp parallel for schedule(static)
            for (unsigned long i = 1; i < repeat; i++)
            {
                memcpy(this->signal + i * Np, this->signal, sizeof(fftwf_complex) * Np);
                memcpy(this->signal_u16 + i * Np, this->signal_u16, sizeof(uint16_pair) * Np);
            }
        }
        else
        {
#pragma omp parallel for schedule(static)
            for (unsigned long i = 1; i < repeat; i++)
            {
                memcpy(this->signal + i * Np, this->signal, sizeof(fftwf_complex) * Np);
            }
        }
    }
    else
    {
#pragma omp parallel for schedule(static)
        for (unsigned long i = 0; i < repeat; i++)
        {
            unsigned long base = i * Np;
            for (unsigned long j = 0; j < Np; j++)
            {
                unsigned long idx = base + j;
                this->signal[idx][0] = this->signal[j][0] + noise[idx][0];
                this->signal[idx][1] = this->signal[j][1] + noise[idx][1];
                if (data_type == "uint16")
                {
                    this->signal_u16[idx].first = (this->signal[idx][0] == 0) ? 0 : static_cast<uint16_t>(this->signal[idx][0] + 32768.0f);
                    this->signal_u16[idx].second = (this->signal[idx][1] == 0) ? 0 : static_cast<uint16_t>(this->signal[idx][1] + 32768.0f);
                }
            }
        }
    }

    if (add_noise)
    {
        g.deallocate();
    }
}

void SimulatedComplexSignal::plot_abs(const fftwf_complex *signal, unsigned long size)
{
    vector<float> v_signal(size);
    for (unsigned long i = 0; i < size; i++)
    {
        v_signal.at(i) = sqrt(pow(signal[i][0], 2) + pow(signal[i][1], 2));
    }
    if (!Py_IsInitialized())
        Py_Initialize();
    py::module plt = py::module::import("matplotlib.pyplot");
    py::module mplcursors = py::module::import("mplcursors");
    plt.attr("plot")(v_signal);
    mplcursors.attr("cursor")();
    plt.attr("show")();
    cout << "Max: " << *max_element(v_signal.begin(), v_signal.end()) << endl;
}

void SimulatedComplexSignal::plot_abs(const uint16_pair *signal, unsigned long size)
{
    vector<float> v_signal(size);
    float first, second;
    for (unsigned long i = 0; i < size; i++)
    {
        first = (signal[i].first == 0) ? 0.0f : static_cast<float>(signal[i].first) - 32768.0f;
        second = (signal[i].second == 0) ? 0.0f : static_cast<float>(signal[i].second) - 32768.0f;
        v_signal.at(i) = sqrt(pow(first, 2) + pow(second, 2));
    }
    if (!Py_IsInitialized())
        Py_Initialize();
    py::module plt = py::module::import("matplotlib.pyplot");
    py::module mplcursors = py::module::import("mplcursors");
    plt.attr("plot")(v_signal);
    mplcursors.attr("cursor")();
    plt.attr("show")();
    cout << "Max: " << *max_element(v_signal.begin(), v_signal.end()) << endl;
}

SimulatedComplexSignal::~SimulatedComplexSignal()
{
}