//
//  ComputePipeline.hpp
//  SoftVision
//
//  Created by HouPeihong on 2023/11/3.
//

#ifndef ComputePipeline_hpp
#define ComputePipeline_hpp


#import <Metal/Metal.h>

@interface ComputePipeline : NSObject

+(void) Exec:(MTLSize)gridSize ThreadgroupSize:(MTLSize)threadgroupSize KernelFuncName:(NSString*)kernelFuncName Args:(NSArray*)args;

@end

//namespace depthMap {
//    void ComputePipelineExec(vector_float3 gridSize,
//                             vector_float3 threadgroupSize,
//                             const char* kernelFuncName,)
//}
#endif /* ComputePipeline_hpp */
