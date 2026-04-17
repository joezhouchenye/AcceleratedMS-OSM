#include "msosm_prepare.h"

Prepare_MSOSM::Prepare_MSOSM(float bw, float dm, float f0) : bw(bw), f0(f0)
{
    fs = bw;
    kdm = 4.15 * 1e15;
    w0 = 2 * pi * f0;
    numDMs = 1;
    Nd_values = new int[numDMs];
    dm_values = new float[numDMs];
    dm_values[0] = dm;
    Nd_values[0] = static_cast<int>(floor(kdm * dm_values[0] * (1 / pow(f0, 2) - 1 / pow(f0 + bw, 2)) / (1 / fs)));
    start_Nd = Nd_values[0];
    if (verbose)
    {
        cout << "Sampling frequency: " << fs << endl;
        cout << "Start frequency: " << f0 << endl;
        cout << "Dispersion smearing points: " << Nd_values[0] << endl;
    }
    dedisp_params = new fftwf_complex *[numDMs];
    delay_points = new int *[numDMs];
    delaycount_values = new int[numDMs];
}

Prepare_MSOSM::Prepare_MSOSM(float bw, float *dm, float f0, int numDMs) : bw(bw), dm_values(dm),f0(f0), numDMs(numDMs)
{
    fs = bw;
    kdm = 4.15 * 1e15;
    w0 = 2 * pi * f0;
    Nd_values = new int[numDMs];
    for (int i = 0; i < numDMs; i++)
    {
        Nd_values[i] = static_cast<int>(floor(kdm * dm_values[i] * (1 / pow(f0, 2) - 1 / pow(f0 + bw, 2)) / (1 / fs)));
        if (verbose)
        {
            cout << "Sampling frequency: " << fs << endl;
            cout << "Start frequency: " << f0 << endl;
            cout << "Dispersion smearing points for DM " << dm_values[i] << ": " << Nd_values[i] << endl;
        }
    }
    start_Nd = Nd_values[0];
    end_Nd = Nd_values[numDMs - 1];
    dedisp_params = new fftwf_complex *[numDMs];
    delay_points = new int *[numDMs];
    delaycount_values = new int[numDMs];
}

void Prepare_MSOSM::calculate_min_order()
{
    // Multi-segment Overlap-Save method
    // Ensure each segment at least has 8 points.
    if (verbose)
    {
        cout << "Prepare parameters..." << endl;
    }
    M_values = new int[numDMs];
    for (int i = 0; i < numDMs; i++)
    {
        double order;
        order = sqrt(MIN_SEGMENT_POINTS/2.0) * sqrt(Nd_values[i]);
        M_values[i] = static_cast<int>(pow(2, ceil(log2(order))));
        if (verbose)
        {
            cout << "Minimum filter order for DM " << dm_values[i] << ": " << M_values[i] << endl;
        }
    }
    M = max_element(M_values, M_values + numDMs)[0];
    M_min = M;
}

void Prepare_MSOSM::override_order(int order)
{
    M = order;
}

void Prepare_MSOSM::generate_dedisp_params(int i)
{
    // Dispersion filter frequency response
    float *w;
    w = (float *)fftwf_malloc(sizeof(float) * M);
    float step = 2 * pi * fs / M;
    int fftpoint = 2 * M;
    fftwf_complex *H;
    H = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftpoint);

    fftwf_complex *h;
    h = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * M);

    for (int j = 0; j < M / 2; j++)
    {
        w[j] = -pi * fs + step * j;
        w[j] = w[j] + pi * fs;
        h[M / 2 + j][0] = cos(4 * pi * pi * kdm * dm_values[i] * w[j] * w[j] / (w[j] + w0) / w0 / w0);
        h[M / 2 + j][1] = -sin(4 * pi * pi * kdm * dm_values[i] * w[j] * w[j] / (w[j] + w0) / w0 / w0);
    }
    for (int j = M / 2; j < M; j++)
    {
        w[j] = -pi * fs + step * j;
        w[j] = w[j] + pi * fs;
        h[j - M / 2][0] = cos(4 * pi * pi * kdm * dm_values[i] * w[j] * w[j] / (w[j] + w0) / w0 / w0);
        h[j - M / 2][1] = -sin(4 * pi * pi * kdm * dm_values[i] * w[j] * w[j] / (w[j] + w0) / w0 / w0);
    }

    fftwf_plan p;
    p = fftwf_plan_dft_1d(M, h, h, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    for (int j = 0; j < M; j++)
    {
        H[j][0] = h[j][0] / M;
        H[j][1] = h[j][1] / M;
    }

    memcpy(H, h, sizeof(fftwf_complex) * M);
    fftwf_free(h);
    fftwf_free(w);

    memset(H + M, 0, sizeof(fftwf_complex) * (fftpoint - M));
    p = fftwf_plan_dft_1d(fftpoint, H, H, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);

    dedisp_params[i] = H;
}

void Prepare_MSOSM::segmentation(int i)
{
    delaycount_values[i] = static_cast<int>(ceil(Nd_values[i] / static_cast<float>(M)));
    if (delaycount_values[i] > delaycount)
    {
        delaycount = delaycount_values[i];
    }
    if (verbose)
    {
        cout << "Delaycount for DM " << dm_values[i] << ": " << delaycount_values[i] << endl;
        line();
    }
    float *w;
    w = (float *)malloc(sizeof(float) * (delaycount_values[i] + 1));
    w[0] = -1;
    w[delaycount_values[i]] = 1;
    int pos_index = 1;
    for (int j = 1; j < delaycount_values[i]; j++)
    {
        w[j] = (sqrt(1 / (1 / pow(w0, 2) - j * M / fs / (4 * pi * pi * kdm * dm_values[i]))) - w0 - pi * fs) / pi / fs;
        if (w[j] < 0)
            pos_index = j + 1;
    }
    negcount = pos_index;
    poscount = delaycount - pos_index + 1;
    segneg = (int *)malloc(sizeof(int) * (negcount + 1));
    segpos = (int *)malloc(sizeof(int) * (poscount + 1));
    delayneg = (int *)malloc(sizeof(int) * negcount);
    delaypos = (int *)malloc(sizeof(int) * poscount);
    for (int j = 0; j < negcount; j++)
    {
        segneg[j] = static_cast<int>(floor((w[j] + 2) * M));
    }
    segneg[negcount] = 2 * M;
    segpos[0] = 0;
    for (int j = 0; j < poscount; j++)
    {
        segpos[j + 1] = static_cast<int>(floor(w[j + negcount] * M));
    }
    for (int j = 0; j < negcount; j++)
    {
        delayneg[j] = j;
    }
    for (int j = 0; j < poscount; j++)
    {
        delaypos[j] = delayneg[negcount - 1] + j;
    }
    free(w);

    // Delay of each FFT point
    delay_points[i] = (int *)malloc(sizeof(int) * 2 * M);
    for (int j = 0; j < negcount; j++)
    {
        for (int k = segneg[j]; k < segneg[j + 1]; k++)
        {
            delay_points[i][k] = delayneg[j];
        }
    }
    for (int j = 0; j < poscount; j++)
    {
        for (int k = segpos[j]; k < segpos[j + 1]; k++)
        {
            delay_points[i][k] = delaypos[j];
        }
    }
    free(delayneg);
    free(delaypos);
    free(segneg);
    free(segpos);
}