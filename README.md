# Multi-Segment Overlap-Save Method for Pulsar Coherent Dedispersion

My test code for MS-OSM using various GPU implementations. The best GPU implementation still needs to be further evaluated.

## Code Structure

- `build.sh`: Script to build the code
- `CMakeLists.txt`: CMake file to build the code
- `dedispersion`:
  - `msosm_stream`: Simple MS-OSM for a single DM with a stream
  - `msosm_dm_loop`: MS-OSM for multiple DM trials using a loop
  - `msosm_dm_stream`: MS-OSM for multiple DM trials using streams
  - `msosm_dm_concurrent`: Concurrent execution for multiple DM trials using MS-OSM
  - `overlap_save`: Simple Overlap-Save for a single DM
  - `overlap_save_dm_concurrent`: Concurrent execution for multiple DM trials using Overlap-Save
- `folding`: Folding process on GPU
- `gpu`: GPU kernel functions
- `psrdada`: PSRDADA file I/O
- `simulated_pulsar`: Code to generate a simulated pulsar in time-domain
- `utils`: Utility functions
- `msosm_psrdata`: Example code to process a psrdada file using MS-OSM
- `multiple_dm_*.cpp`: Example code for multiple DM trials using different approaches
- `multiple_*.cpp`: Example code for a single DM
- `test_*.cpp`: Test code for performance evaluation