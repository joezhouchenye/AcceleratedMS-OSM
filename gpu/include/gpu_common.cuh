#pragma once

#include <iostream>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <nvtx3/nvToolsExt.h>
#include <cuda.h>

#include "globals.h"

#define USE_NVTX

#ifdef USE_NVTX
const uint32_t colors[] = {0xff0000ff, 0xff00ff00, 0xffffff00, 0xffff0000, 0xff00ffff, 0xffff00ff, 0xffffffff, 0xff000000};
const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid)                       \
    {                                               \
        int color_id = cid;                         \
        color_id = color_id % num_colors;           \
        nvtxEventAttributes_t attr = {0};           \
        attr.version = NVTX_VERSION;                \
        attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;  \
        attr.colorType = NVTX_COLOR_ARGB;           \
        attr.color = colors[color_id];              \
        attr.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        attr.message.ascii = name;                  \
        nvtxRangePushEx(&attr);                     \
    }
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE
#endif

#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t error = call;                                                                                      \
        if (error != cudaSuccess)                                                                                      \
        {                                                                                                              \
            cerr << "CUDA error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

#define CUFFT_CHECK(call)                                                                           \
    do                                                                                              \
    {                                                                                               \
        cufftResult error = call;                                                                   \
        if (error != CUFFT_SUCCESS)                                                                 \
        {                                                                                           \
            cerr << "CUFFT error: " << error << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                                                                \
        }                                                                                           \
    } while (0)

void checkCudaError(cudaError_t err, const char *msg = "");

// Device information
void GPU_GetDevInfo();