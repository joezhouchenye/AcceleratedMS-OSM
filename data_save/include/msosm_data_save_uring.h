#pragma once
#include "globals.h"
#include "msosm_gpu_batch.h"
#include <liburing.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <iomanip>
#include <immintrin.h>

using namespace std;

enum class SaveSlotState
{
    EMPTY,
    COPYING,
    READY,
    SAVING,
    DONE
};

class MSOSM_DataSave_Uring : public MSOSM_GPU_BATCH
{
public:
    MSOSM_DataSave_Uring(float bw, float *dm, float f0, int numDMs);
    void config_save(string dir = "", string prefix = "");
    void initialize_uring(int slot_count, unsigned long save_count);
    void copy_to_slot();
    void poll_cuda_state();
    void save_to_disk(int save_index);
    void start_saving();
    void join_saving();

private:
    unsigned long save_count;

    int slot_count;
    size_t slot_size;
    uint16_pair *slot_data_int16;
    uint16_pair **slot_data_ptr;
    cudaEvent_t *slot_events;
    atomic<SaveSlotState> *slot_states;

    int copy_index = 0;
    thread *save_threads;
    io_uring *rings;

    thread poll_thread;

    int *file_fds;
    int data_offset;
};