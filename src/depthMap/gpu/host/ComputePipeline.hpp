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

+(instancetype)createPipeline;

-(void) Exec:(MTLSize)gridSize ThreadgroupSize:(MTLSize)threadgroupSize KernelFuncName:(NSString*)kernelFuncName Args:(NSArray*)args;

@end

#endif /* ComputePipeline_hpp */
