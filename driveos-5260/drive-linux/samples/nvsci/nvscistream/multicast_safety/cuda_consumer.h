//
// CUDA consumer client declaration.
//
// Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CUDA_CONSUMER_H
#define CUDA_CONSUMER_H

#include "cuda_common.h"

class CudaConsumer:
    public ClientCommon
{
public:
    CudaConsumer() = delete;
    CudaConsumer(NvSciStreamBlock blockHandle,
                 uint32_t numSyncs = NUM_CONS_SYNCS,
                 uint32_t numElems = NUM_ELEMENTS_PER_PACKET);
    virtual ~CudaConsumer(void) = default;

    // Streaming functions
    bool acquirePacket(NvSciStreamCookie &cookie);
    void processPayload(NvSciStreamCookie cookie);
    void releasePacket(NvSciStreamCookie cookie);

private:
    uint32_t numPacketsRecvd = 0U;
};

#endif
