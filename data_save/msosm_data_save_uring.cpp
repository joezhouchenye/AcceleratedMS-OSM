#include "msosm_data_save_uring.h"

static void die(const string &msg)
{
    fprintf(stderr, "FATAL: %s (errno=%d, %s)\n", msg.c_str(), errno, strerror(errno));
    exit(1);
}

static string make_filename(const string &dir, const string &prefix, double value, int precision)
{
    ostringstream oss;
    oss << dir << "/" << prefix << "DM" << fixed << setprecision(precision) << value << ".dat";
    return oss.str();
}

static int get_decimal_places_from_dm(float dm, int max_places = 10)
{
    dm = fabs(dm);
    if (dm <= 0.0)
        return 0;
    int ret = max_places;
    for (int places = 0; places <= max_places; ++places)
    {
        float scaled = dm * pow(10.0, places);
        float rounded = std::round(scaled);
        // 误差小于float精度
        if (fabs(scaled - rounded) < 1e-6)
        {
            ret = places;
            break;
        }
    }
    if (ret == 0)
        ret = 2;
    return ret;
}

MSOSM_DataSave_Uring::MSOSM_DataSave_Uring(float bw, float *dm, float f0, int numDMs) : MSOSM_GPU_BATCH(bw, dm, f0, numDMs) {}

void MSOSM_DataSave_Uring::config_save(string dir, string prefix)
{
    if (dir.empty())
    {
        // Use current directory if no directory is specified
        dir = ".";
    }
    if (prefix.empty())
    {
        // Use default prefix if no prefix is specified
        prefix = "";
    }
    else
    {
        prefix += "_";
    }
    file_fds = new int[numDMs];
    // 获取DM的最高精度
    int dm_precision;
    dm_precision = get_decimal_places_from_dm(dm_values[0], 10);
    if (numDMs > 1)
        dm_precision = max(dm_precision, get_decimal_places_from_dm(dm_values[1], 10));
    for (int i = 0; i < numDMs; i++)
    {
        string filename = make_filename(dir, prefix, dm_values[i], dm_precision);
        file_fds[i] = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (file_fds[i] < 0)
        {
            die("Failed to open file: " + filename);
        }
    }
}

void MSOSM_DataSave_Uring::initialize_uring(int slot_count, unsigned long save_count)
{
    this->save_count = save_count;
    this->slot_count = slot_count;
    slot_size = numDMs * batch * M;
    data_offset = batch * M;
    // Allocate memory for save slots
    CUDA_CHECK(cudaMallocHost(&slot_data_int16, slot_count * slot_size * sizeof(uint16_pair)));
    slot_data_ptr = new uint16_pair *[slot_count];
    slot_events = new cudaEvent_t[slot_count];
    slot_states = new atomic<SaveSlotState>[slot_count];
    for (int i = 0; i < slot_count; i++)
    {
        slot_data_ptr[i] = slot_data_int16 + i * slot_size;
        CUDA_CHECK(cudaEventCreate(&slot_events[i], cudaEventDisableTiming));
        slot_states[i] = SaveSlotState::EMPTY;
    }

    rings = new io_uring[slot_count];
    for (int i = 0; i < slot_count; i++)
    {
        if (io_uring_queue_init(numDMs * 32, &rings[i], 0) < 0)
        {
            die("Failed to initialize io_uring queue for slot " + to_string(i));
        }
    }

    save_threads = new thread[slot_count];
}

void MSOSM_DataSave_Uring::copy_to_slot()
{
    while (slot_states[copy_index].load(std::memory_order_acquire) != SaveSlotState::EMPTY)
    {
        _mm_pause();
    }
    get_output(slot_data_ptr[copy_index]);
    cudaEventRecord(slot_events[copy_index], output_stream);
    slot_states[copy_index].store(SaveSlotState::COPYING, std::memory_order_release);
    copy_index = (copy_index + 1) % slot_count;
}

void MSOSM_DataSave_Uring::poll_cuda_state()
{
    unsigned long polled_count = 0;
    int poll_index = 0;
    while (polled_count < save_count)
    {
        bool progressed = false;
        if (slot_states[poll_index].load(std::memory_order_acquire) == SaveSlotState::COPYING)
        {
            cudaError_t status = cudaEventQuery(slot_events[poll_index]);
            if (status == cudaSuccess)
            {
                SaveSlotState expected = SaveSlotState::COPYING;
                if (slot_states[poll_index].compare_exchange_strong(expected, SaveSlotState::READY, std::memory_order_acq_rel))
                {
                    polled_count++;
                    progressed = true;
                    poll_index = (poll_index + 1) % slot_count;
                }
            }
            else if (status != cudaErrorNotReady)
            {
                CUDA_CHECK(status);
            }
        }
        if (!progressed)
        {
            _mm_pause();
        }
    }
    // The process finished. Notify the save thread to exit after saving the remaining data.
    for (int i = 0; i < slot_count; i++)
    {
        while (slot_states[i].load(std::memory_order_acquire) != SaveSlotState::EMPTY)
        {
            _mm_pause();
        }
        slot_states[i].store(SaveSlotState::DONE, std::memory_order_release);
    }
}

void MSOSM_DataSave_Uring::save_to_disk(int save_index)
{
    int slot_busy = 0;
    size_t submitted_count = save_index;

    auto reap_completion = [&](io_uring_cqe *completed_cqe)
    {
        if (completed_cqe->res < 0)
        {
            die("I/O error: " + string(strerror(-completed_cqe->res)));
        }
        slot_busy--;
        io_uring_cqe_seen(rings + save_index, completed_cqe);
        if (slot_busy == 0)
        {
            slot_states[save_index].store(SaveSlotState::EMPTY, std::memory_order_release);
        }
    };

    while (slot_states[save_index].load(std::memory_order_acquire) != SaveSlotState::DONE)
    {
        SaveSlotState expected = SaveSlotState::READY;
        if (slot_states[save_index].compare_exchange_strong(expected, SaveSlotState::SAVING, std::memory_order_acq_rel))
        {
            PUSH_RANGE("Save to Disk", 6);
            const size_t base_offset =
                static_cast<size_t>(submitted_count) *
                static_cast<size_t>(data_offset) *
                sizeof(uint16_pair);

            for (int i = 0; i < numDMs; i++)
            {
                auto sqe = io_uring_get_sqe(rings + save_index);
                if (sqe == nullptr)
                {
                    die("Failed to get submission queue entry");
                }
                io_uring_prep_write(
                    sqe,
                    file_fds[i],
                    slot_data_ptr[save_index] + i * data_offset,
                    data_offset * sizeof(uint16_pair),
                    base_offset);
            }
            if (io_uring_submit(rings + save_index) < 0)
            {
                die("Failed to submit io_uring requests");
            }
            slot_busy = numDMs;
            submitted_count += slot_count;
        }

        while (slot_busy > 0)
        {
            io_uring_cqe *cqe;
            if (io_uring_wait_cqe(rings + save_index, &cqe) < 0)
            {
                die("Failed while waiting for io_uring completion");
            }
            reap_completion(cqe);
            if (slot_busy == 0)
            {
                POP_RANGE;
            }
        }
    }
}

void MSOSM_DataSave_Uring::save_to_disk_pwrite(int save_index)
{
    size_t submitted_count = save_index;
    while (slot_states[save_index].load(std::memory_order_acquire) != SaveSlotState::DONE)
    {
        if (slot_states[save_index].load(std::memory_order_acquire) == SaveSlotState::READY)
        {
            PUSH_RANGE("Save to Disk", 6);
            const size_t base_offset =
                static_cast<size_t>(submitted_count) *
                static_cast<size_t>(data_offset) *
                sizeof(uint16_pair);

            for (int i = 0; i < numDMs; i++)
            {
                ssize_t written = pwrite(
                    file_fds[i],
                    slot_data_ptr[save_index] + i * data_offset,
                    data_offset * sizeof(uint16_pair),
                    base_offset);
                if (written < 0)
                {
                    die("Failed to write to file: " + string(strerror(errno)));
                }
            }
            slot_states[save_index].store(SaveSlotState::EMPTY, std::memory_order_release);
            submitted_count += slot_count;
            POP_RANGE;
        }
        else
        {
            _mm_pause();
        }
    }
}

void MSOSM_DataSave_Uring::start_saving()
{
    poll_thread = thread(&MSOSM_DataSave_Uring::poll_cuda_state, this);
    for (int i = 0; i < slot_count; i++)
    {
        save_threads[i] = thread(&MSOSM_DataSave_Uring::save_to_disk, this, i);
    }
}

void MSOSM_DataSave_Uring::join_saving()
{
    if (poll_thread.joinable())
    {
        poll_thread.join();
    }
    for (int i = 0; i < slot_count; i++)
    {
        if (save_threads[i].joinable())
        {
            save_threads[i].join();
        }
    }
    this->synchronize();
    // Close file descriptors
    for (int i = 0; i < numDMs; i++)
    {
        close(file_fds[i]);
    }
}
