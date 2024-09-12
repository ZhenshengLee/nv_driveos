//
// Cuda Producer client declaration.
//
// Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CUDA_PRODUCER_H
#define CUDA_PRODUCER_H

#include "cuda_common.h"

class CudaProducer:
    public ClientCommon
{
public:
    CudaProducer() = delete;
    CudaProducer(NvSciStreamBlock blockHandle,
                 uint32_t numSyncs = NUM_PROD_SYNCS,
                 uint32_t numElems = NUM_ELEMENTS_PER_PACKET);
    virtual ~CudaProducer(void) = default;

    // Streaming functions
    void getPacket(NvSciStreamCookie &cookie);
    void processPayload(NvSciStreamCookie cookie);
    void sendPacket(NvSciStreamCookie cookie);

private:
    uint32_t numPacketsPresented = 0U;
};

#endif
