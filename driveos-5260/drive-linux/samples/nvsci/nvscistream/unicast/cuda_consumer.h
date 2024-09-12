//
// CUDA consumer client declaration.
//
// Copyright (c) 2019-2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CUDA_CONSUMER_H
#define CUDA_CONSUMER_H

#include "client_common.h"

// cuda includes
#include "cuda.h"

namespace NvScistreamSample
{
    class CudaConsumer:
        public ClientCommon
    {
    public:
        CudaConsumer() = delete;
        CudaConsumer(NvSciStreamBlock blockHandle,
                     uint32_t numSyncs = NUM_CONS_SYNCS);
        virtual ~CudaConsumer(void);

        // Buffer setup functions
        virtual void createBufAttrList(NvSciBufModule bufModule);
        virtual void mapPacket(
            NvSciStreamCookie &cookie,
            NvSciStreamPacket packtHandle);
        virtual void mapBuffers(NvSciStreamCookie cookie);
        virtual void unmapBuffers(NvSciStreamCookie cookie);

        // Sync object setup functions
        virtual void createSyncAttrLists(NvSciSyncModule syncModule);
        virtual void reconcileAndAllocSyncObjs(void);
        virtual void mapSyncObjs(void);
        virtual void initFences(NvSciSyncModule syncModule);
        virtual void unmapSyncObjs(void);

        // Event handling
        void handleResourceSetupEvents(NvSciStreamEvent &event);

        // Streaming functions
        bool acquirePacket(NvSciStreamCookie &cookie);
        virtual void processPayload(NvSciStreamCookie cookie);
        void releasePacket(NvSciStreamCookie cookie);

    private:
        NvSciSyncCpuWaitContext waitContext;
        uint32_t numPacketsRecvd;

        CUdevice  dev;
        CUcontext cuctx;
        uint64_t p_size[NUM_ELEMENTS_PER_PACKET];
        unsigned char *pCudaCopyMem[NUM_PACKETS][NUM_ELEMENTS_PER_PACKET];
        CUdeviceptr devPtr[NUM_PACKETS][NUM_ELEMENTS_PER_PACKET];
        CUexternalMemory extMem[NUM_PACKETS][NUM_ELEMENTS_PER_PACKET];
        CUexternalSemaphore signalerSem[MAX_NUM_SYNCS];
        CUexternalSemaphore waiterSem[MAX_NUM_SYNCS];
    };
}

#endif
