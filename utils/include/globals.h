#pragma once
#include <iostream>
#include <thread>
#include <mutex>
#include <fstream>
#include <getopt.h>
#include <chrono>
#include <cmath>
#include <vector>
#include <cstring>
#include <fftw3.h>
#include <cufft.h>
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>

#include <pybind11/embed.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;

struct __attribute__((packed)) uint16_pair
{
    uint16_t first;
    uint16_t second;
};

extern bool verbose;
typedef cufftComplex Complex;
constexpr auto pi = M_PI;
extern void banner();
extern void line();

void plot_init();
void show();
void plot_abs(uint16_pair *signal, unsigned long size);
void plot_abs(Complex *signal, unsigned long size);
void plot_pol1(uint16_pair *signal, unsigned long size);
void plot_pol2(uint16_pair *signal, unsigned long size);
template <typename T>
void plot(T *signal, unsigned long size);

void plot(vector<double> &v_signal);