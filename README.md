# Accelerated Multi-Segment Overlap-Save Method for Pulsar Coherent Dedispersion

## Included Examples

- `msosm_psrdata.cpp`: Example code to process a psrdada file using MS-OSM (folding directly)
- `msosm_save_simulated.cpp`: Example code to generate a simulated pulsar signal, process it using MS-OSM, and save the dedispersed data of multiple DM trials to disk using io_uring
- `benchmark/check_msosm.cpp`: Example code to verify the correctness of MS-OSM using a simulated signal
- `benchmark/test_msosm_speed.cpp`: Example code to benchmark the speed of computing process of MS-OSM

## Dependencies

- CMake 3.25 or later
- GCC 13 or later
- Python 3.11 or later (for plotting results)
  - `pybind11`, `matplotlib`, `numpy`, `mplcursors`
- `libfftw3-dev`
- `liburing-dev`
- CUDA Toolkit 12.6 or later