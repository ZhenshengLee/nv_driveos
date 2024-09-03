/* Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * Author: Saurabh Hukerikar, shukerikar@nvidia.com
*/

#ifndef __NVGPU_SDL__
#define __NVGPU_SDL__

typedef enum gpuSDL_result {
    /**
     * The SDL API call returned with no errors. This indicates no
     * faults were detected 
     */
    NVGPUSDL_SUCCESS                              = 0,
    /**
     * The SDL API call returned with error indicating a random 
     * hardware fault was detected in the GPU 
     */
    NVGPUSDL_ERROR_RANDOM_HW_FAULT                = 1,
    /**
     * The SDL API call returned with a cuda error 
     * indicating that the diagnostic failed to execute 
     */
    NVGPUSDL_ERROR_CUDA_ERROR                     = 2,


} NVGPUSDLresult;

namespace nvgpuSDL {
/* Initializes resources associated with the software diagnostic 
 * library
 */
NVGPUSDLresult initialize( void );  

/* Execute all GPU diagnostics in the library for random 
 * hardware faults 
 */
NVGPUSDLresult execute_all( void );

/* Execute subset of GPU diagnostics in the library for random 
 * hardware faults that run in time slice corresponding to 
 * a single frame's processing interval. 
 * The diagnostics in quanta 1, 2 and 3 together represent the 
 * complete set of GPU diagnostics 
 */
NVGPUSDLresult execute_quantum_1( void );
NVGPUSDLresult execute_quantum_2( void );
NVGPUSDLresult execute_quantum_3( void );

/* Explicitly destroys and cleans up all resources associated with the 
 * software diagnostic library
 */
NVGPUSDLresult finalize( void );

}

#endif
