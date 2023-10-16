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
    }
}

}
