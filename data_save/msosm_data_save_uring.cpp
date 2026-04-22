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
    cout << "Output file: " << oss.str() << endl;
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
    cout << dm_precision << endl;
    if (numDMs > 1)
        dm_precision = max(dm_precision, get_decimal_places_from_dm(dm_values[1], 10));
    cout << dm_precision << endl;
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

    if (io_uring_queue_init(slot_count * numDMs, &ring, 0) < 0)
    {
        die("Failed to initialize io_uring queue");
    }
}

void MSOSM_DataSave_Uring::copy_to_slot()
{
    unsigned long copied_count = 0;
    int copy_index = 0;
    while (copied_count < save_count)
    {
        SaveSlotState expected = SaveSlotState::EMPTY;
        if (slot_states[copy_index].compare_exchange_strong(expected, SaveSlotState::COPYING, std::memory_order_acq_rel))
        {
            get_output(slot_data_ptr[copy_index]);
            cudaEventRecord(slot_events[copy_index], output_stream);
            copied_count++;
            copy_index = (copy_index + 1) % slot_count;
        }
        else
        {
            this_thread::yield();
        }
    }
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
            this_thread::yield();
        }
    }
}

void MSOSM_DataSave_Uring::save_to_disk()
{
    unsigned long saved_count = 0;
    unsigned long submitted_count = 0;
    int save_index = 0;

    vector<int> slot_busy(slot_count, 0);
    int inflight_slots = 0;

    io_uring_cqe *cqe;

    auto reap_completion = [&](io_uring_cqe *completed_cqe)
    {
        if (completed_cqe->res < 0)
        {
            die("I/O error: " + string(strerror(-completed_cqe->res)));
        }
        int completed_index = io_uring_cqe_get_data64(completed_cqe);
        slot_busy[completed_index]--;
        io_uring_cqe_seen(&ring, completed_cqe);
        if (slot_busy[completed_index] == 0)
        {
            slot_states[completed_index].store(SaveSlotState::EMPTY, std::memory_order_release);
            saved_count++;
            inflight_slots--;
        }
    };

    while (saved_count < save_count)
    {
        bool progressed = false;
        SaveSlotState expected = SaveSlotState::READY;
        if (submitted_count < save_count)
        {
            if (slot_states[save_index].compare_exchange_strong(expected, SaveSlotState::SAVING, std::memory_order_acq_rel))
            {
                const size_t base_offset =
                    static_cast<size_t>(submitted_count) *
                    static_cast<size_t>(data_offset) *
                    sizeof(uint16_pair);

                for (int i = 0; i < numDMs; i++)
                {
                    auto sqe = io_uring_get_sqe(&ring);
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
                    io_uring_sqe_set_data64(sqe, (uint64_t)save_index);
                }
                if (io_uring_submit(&ring) < 0)
                {
                    die("Failed to submit io_uring requests");
                }
                slot_busy[save_index] = numDMs;
                inflight_slots++;
                progressed = true;
                save_index = (save_index + 1) % slot_count;
                submitted_count++;
            }
        }

        if (inflight_slots > 0)
        {
            int peek_result = io_uring_peek_cqe(&ring, &cqe);
            if (peek_result == 0)
            {
                reap_completion(cqe);
                progressed = true;
            }
            else if (peek_result != -EAGAIN)
            {
                die("Failed to poll io_uring completion queue");
            }
        }

        if (!progressed)
        {
            if (inflight_slots > 0)
            {
                if (io_uring_wait_cqe(&ring, &cqe) < 0)
                {
                    die("Failed while waiting for io_uring completion");
                }
                reap_completion(cqe);
            }
            else
            {
                this_thread::yield();
            }
        }
    }
}

void MSOSM_DataSave_Uring::start_saving()
{
    copy_thread = thread(&MSOSM_DataSave_Uring::copy_to_slot, this);
    poll_thread = thread(&MSOSM_DataSave_Uring::poll_cuda_state, this);
    save_thread = thread(&MSOSM_DataSave_Uring::save_to_disk, this);
}

void MSOSM_DataSave_Uring::join_saving()
{
    if (copy_thread.joinable())
    {
        copy_thread.join();
    }
    if (poll_thread.joinable())
    {
        poll_thread.join();
    }
    if (save_thread.joinable())
    {
        save_thread.join();
    }
    this->synchronize();
    // Close file descriptors
    for (int i = 0; i < numDMs; i++)
    {
        close(file_fds[i]);
    }
}
