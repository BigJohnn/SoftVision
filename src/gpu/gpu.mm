#import <Metal/Metal.h>

#include <gpu/gpu.hpp>
#include <string>
#include <SoftVisionLog.h>

namespace gpu {

void gpuInformation()
{
    std::string information;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        LOG_DEBUG("GPU Info: \n%s", device.description.UTF8String);
        LOG_DEBUG("device.supportsRenderDynamicLibraries == %d", device.supportsRenderDynamicLibraries);
        LOG_DEBUG("maxThreadgroupMemoryLength == %luk", (unsigned long)device.maxThreadgroupMemoryLength/1024);
        LOG_DEBUG("maxThreadsPerThreadgroup(w,h,depth == %lu,%lu,%lu", (unsigned long)device.maxThreadsPerThreadgroup.width,
                  (unsigned long)device.maxThreadsPerThreadgroup.height,
                  (unsigned long)device.maxThreadsPerThreadgroup.depth);
    }
}

}
