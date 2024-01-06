#pragma once

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
