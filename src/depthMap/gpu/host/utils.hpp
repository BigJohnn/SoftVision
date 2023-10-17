#pragma once

// Macros for checking gpu errors
//#define CHECK_GPU_RETURN_ERROR(err)                                                                                   \
//    if(err != gpuSuccess)                                                                                             \
//    {                                                                                                                  \
//        fprintf(stderr, "\n\nGPUError: %s\n", gpuGetErrorString(err));                                               \
//        fprintf(stderr, "  file:       %s\n", __FILE__);                                                               \
//        fprintf(stderr, "  function:   %s\n", __FUNCTION__);                                                           \
//        fprintf(stderr, "  line:       %d\n\n", __LINE__);                                                             \
//        std::stringstream s;                                                                                           \
//        s << "\n  GPU Error: " << gpuGetErrorString(err)                                                             \
//          << "\n  file:  " << __FILE__                                                                                 \
//          << "\n  function:   " << __FUNCTION__                                                                        \
//          << "\n  line:       " << __LINE__                                                                            \
//          << "\n";                                                                                                     \
//        throw std::runtime_error(s.str());                                                                             \
//    }                                                                                                                  \

//#define CHECK_GPU_RETURN_ERROR_NOEXCEPT(err)                                                                          \
//    if(err != gpuSuccess)                                                                                             \
//    {                                                                                                                  \
//        fprintf(stderr, "\n\nGPUError: %s\n", gpuGetErrorString(err));                                               \
//        fprintf(stderr, "  file:       %s\n", __FILE__);                                                               \
//        fprintf(stderr, "  function:   %s\n", __FUNCTION__);                                                           \
//        fprintf(stderr, "  line:       %d\n\n", __LINE__);                                                             \
//    }                                                                                                                  \

//#define CHECK_GPU_ERROR() CHECK_GPU_RETURN_ERROR(gpuGetLastError());

//#define THROW_ON_GPU_ERROR(rcode, message)                                                                            \
//    if(rcode != gpuSuccess)                                                                                           \
//    {                                                                                                                  \
//        std::stringstream s;                                                                                           \
//        s << message << ": " << gpuGetErrorString(err);                                                               \
//        throw std::runtime_error(s.str());                                                                             \
//    }                                                                                                                  \

namespace depthMap {

/**
 * @brief Get and log available GPU devices.
 * @return the number of GPU devices
 */
int listGpuDevices();

/**
 * @brief Get the device id currently used for GPU executions.
 * @return current GPU device id
 */
int getGpuDeviceId();

/**
 * @brief Log current GPU device memory information.
 */
void logDeviceMemoryInfo();

/**
 * @brief Get current GPU device memory information.
 * @param[out] availableMB the available memory in MB on the current GPU device
 * @param[out] usedMB the used memory in MB on the current GPU device
 * @param[out] totalMB the total memory in MB on the current GPU device
 */
void getDeviceMemoryInfo(double& availableMB, double& usedMB, double& totalMB);

} // namespace depthMap
