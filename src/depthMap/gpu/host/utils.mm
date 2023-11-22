#import <Metal/Metal.h>

#include <depthMap/gpu/host/utils.hpp>
#include <SoftVisionLog.h>

namespace depthMap {


/*
 To use the Metal framework, start by getting a GPU device. All of the objects your app needs to interact with Metal come from a MTLDevice that you acquire at runtime. Some devices, such as those with iOS and tvOS have a single GPU that you can access by calling MTLCreateSystemDefaultDevice().
 */
int listGpuDevices()
{
    int nbDevices = 1; // number of GPUs
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    
    LOG_DEBUG("device 0 : %s", device.description.UTF8String);
    return nbDevices;
}

int getGpuDeviceId()
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device.registryID;
}

void logDeviceMemoryInfo()
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    size_t iavail = device.maxBufferLength - device.currentAllocatedSize;
    size_t itotal = (unsigned long)device.maxBufferLength;

    const double availableMB = double(iavail) / (1024.0 * 1024.0);
    const double totalMB = double(itotal) / (1024.0 * 1024.0);
    const double usedMB = double(itotal - iavail) / (1024.0 * 1024.0);

    LOG_X("Device memory (device id: "<< getGpuDeviceId() <<"):" << std::endl
                      << "\t- used: " << usedMB << " MB" << std::endl
                      << "\t- available: " << availableMB << " MB" << std::endl
                      << "\t- total: " << totalMB << " MB");
}

void getDeviceMemoryInfo(double& availableMB, double& usedMB, double& totalMB)
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    size_t iavail = device.maxBufferLength - device.currentAllocatedSize;
    size_t itotal = (unsigned long)device.maxThreadgroupMemoryLength;

    availableMB = double(iavail) / (1024.0 * 1024.0);
    totalMB = double(itotal) / (1024.0 * 1024.0);
    usedMB = double(itotal - iavail) / (1024.0 * 1024.0);
}

}


