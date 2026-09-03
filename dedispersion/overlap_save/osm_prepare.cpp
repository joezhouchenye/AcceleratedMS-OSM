#include "prepare.h"

Prepare_OSM::Prepare_OSM(float bw, float dm, float f0) : bw(bw), dm(dm), f0(f0)
{
    fs = bw;
    kdm = 4.15 * 1e15;
    w0 = 2 * pi * f0;
    Nd = static_cast<unsigned long>(floor(kdm * dm * (1 / pow(f0, 2) - 1 / pow(f0 + bw, 2)) / (1 / fs)));
    if (verbose)
    {
        cout << "Sampling frequency: " << fs << endl;
        cout << "Start frequency: " << f0 << endl;
        cout << "Dispersion smearing points: " << Nd << endl;
    }
}

void Prepare_OSM::calculate_min_order()
{
    // Overlap-Save method
    // Ensure FFT length is 2 * log2(Nd).
    int order;
    order = next_power_of_2(Nd);
    M = 2 * order;
    M_min = M;
    if (verbose)
    {
        cout << "Prepare parameters..." << endl;
        cout << "Minimum filter order: " << M_min << endl;
    }
}

void Prepare_OSM::override_order(unsigned long order)
{
    M = order;
}

static void apply_tukey_window(fftwf_complex *data, int N, double alpha)
{
    const double pi = 3.14159265358979323846;

    if (alpha <= 0.0)
        return;

    if (alpha > 1.0)
        alpha = 1.0;

    for (int i = 0; i < N; i++)
    {
        const double x = static_cast<double>(i) / static_cast<double>(N - 1);
        double win = 1.0;

        if (x < alpha / 2.0)
        {
            win = 0.5 * (1.0 + std::cos(
                pi * (2.0 * x / alpha - 1.0)));
        }
        else if (x > 1.0 - alpha / 2.0)
        {
            win = 0.5 * (1.0 + std::cos(
                pi * (2.0 * x / alpha - 2.0 / alpha + 1.0)));
        }

        data[i][0] *= static_cast<float>(win);
        data[i][1] *= static_cast<float>(win);
    }
}

void Prepare_OSM::generate_dedisp_params()
{
    const double pi_d = 3.14159265358979323846;
    const double kdm_d = 4.15e15;
    const double fs_d = static_cast<double>(fs);
    const double dm_d = static_cast<double>(dm);
    const double f0_d = static_cast<double>(f0);
    const double w0_d = 2.0 * pi_d * f0_d;
    const double step = 2.0 * pi_d * fs_d / static_cast<double>(M);

    unsigned long fftpoint = 2 * M;

    fftwf_complex *H = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftpoint);
    fftwf_complex *h = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * M);

    for (unsigned long i = 0; i < M / 2; i++)
    {
        double w = step * static_cast<double>(i);
        double phase = 4.0 * pi_d * pi_d * kdm_d * dm_d * w * w / ((w + w0_d) * w0_d * w0_d);

        h[M / 2 + i][0] = static_cast<float>(std::cos(phase));
        h[M / 2 + i][1] = static_cast<float>(-std::sin(phase));
    }

    for (unsigned long i = M / 2; i < M; i++)
    {
        double w = step * static_cast<double>(i);
        double phase = 4.0 * pi_d * pi_d * kdm_d * dm_d * w * w / ((w + w0_d) * w0_d * w0_d);

        h[i - M / 2][0] = static_cast<float>(std::cos(phase));
        h[i - M / 2][1] = static_cast<float>(-std::sin(phase));
    }

    fftwf_plan p = fftwf_plan_dft_1d(M, h, h, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);

    const float inv_M = 1.0f / static_cast<float>(M);

    for (unsigned long i = 0; i < M; i++)
    {
        h[i][0] *= inv_M;
        h[i][1] *= inv_M;
    }

    // apply_tukey_window(h, M, 0.5);

    memcpy(H, h, sizeof(fftwf_complex) * M);
    memset(H + M, 0, sizeof(fftwf_complex) * M);

    fftwf_free(h);

    p = fftwf_plan_dft_1d(fftpoint, H, H, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);

    dedisp_params = H;
}

int Prepare_OSM::next_power_of_2(int n)
{
    if (n <= 0)
    {
        return 1;
    }
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}
