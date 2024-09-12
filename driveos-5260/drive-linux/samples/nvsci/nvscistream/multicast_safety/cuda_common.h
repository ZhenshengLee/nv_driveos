//
// NvSciStream Common Client usage
//
// Copyright (c) 2020-2021 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CLIENT_COMMON_H
#define CLIENT_COMMON_H

#include "nvscistream.h"
#include "constants.h"
#include "log.h"

// cuda includes
#include "cuda.h"
#include "cuda_runtime_api.h"

constexpr NvSciStreamCookie cookieBase = 0xC00C1E4U;

// Define Packet struct which is used by cuda client
typedef struct {
    // The client's handle for the packet.
    NvSciStreamCookie       cookie = 0U;
    // The NvSciStream's Handle for the packet
    NvSciStreamPacket       handle = 0U;
    // An array of elements/buffers in the packet
    NvSciBufObj            *buffers = nullptr;
    // An array of pre-allocated prefences for this packet.
    NvSciSyncFence          fences[MAX_NUM_SYNCS];

    // Cuda memory structure
    cudaExternalMemory_t    extMem[NUM_ELEMENTS_PER_PACKET];
    void                   *devPtr[NUM_ELEMENTS_PER_PACKET] = { nullptr };
    uint8_t                *pCudaCopyMem[NUM_ELEMENTS_PER_PACKET] = { nullptr };
} CudaPacket;

class ClientCommon
{
public:
    ClientCommon() = delete;
    ClientCommon(NvSciStreamBlock blockHandle,
                 uint32_t numSyncs,
                 uint32_t numElems);
    virtual ~ClientCommon(void);

    // Buffer setup functions
    void sendPacketAttr(NvSciBufModule bufModule);
    void mapPacket(NvSciStreamEvent &event);
    void mapPacketElement(NvSciStreamEvent &event);
    bool allPacketInfoReceived(void);

    // Sync objects setup functons
    void sendSyncAttr(NvSciSyncModule syncModule);
    void allocAndSendSyncObjs(NvSciStreamEvent &event);
    void mapSyncObj(NvSciStreamEvent &event);
    bool allSyncInfoReceived(void);

    // Event handling
    void handleResourceSetupEvents(NvSciStreamEvent &event);

protected:
    // Packet cookie and id conversion
    inline NvSciStreamCookie assignPacketCookie(void)
    {
        NvSciStreamCookie cookie = cookieBase +
            static_cast<NvSciStreamCookie>(numPackets);
        if (cookie == 0U) {
            LOG_ERR_EXIT("Invalid cookie assignment");
        }
        return cookie;
    }

    inline CudaPacket* getPacketByCookie(const NvSciStreamCookie& cookie)
    {
        if (cookie == 0U) {
            LOG_ERR_EXIT("Invalid cookie assignment");
        }
        uint32_t id = static_cast<uint32_t>(cookie - cookieBase) - 1U;

        return &(packets[id]);
    }

    NvSciStreamBlock        handle = 0U;

    // sync objects
    bool                    hasRecvSyncAttr = false;
    bool                    hasRecvSyncCount = false;

    NvSciSyncAttrList       signalerAttrList = nullptr;

    uint32_t                numSyncObjs = 0U;
    NvSciSyncObj            syncObjs[MAX_NUM_SYNCS] = {nullptr};

    uint32_t                numRecvSyncObjs = 0U;
    uint32_t                numSyncObjsRecvd = 0U;
    NvSciSyncObj            waiterSyncObjs[MAX_NUM_SYNCS] = { nullptr };
    NvSciSyncFence          prefences[MAX_NUM_SYNCS];

    cudaStream_t            endpointStream = nullptr;
    cudaExternalSemaphore_t signalerSem[MAX_NUM_SYNCS];
    cudaExternalSemaphore_t waiterSem[MAX_NUM_SYNCS];

    // packet elements (buffer)
    uint32_t                numElements = 0U;

    uint32_t                numReconciledElements = 0U;
    uint32_t                numReconciledElementsRecvd = 0U;

    uint64_t                p_size[NUM_ELEMENTS_PER_PACKET] = { 0U };

    uint32_t                numPackets = 0U;
    CudaPacket              packets[NUM_PACKETS];


    int                     m_cudaDeviceId = 0;
    NvSciRmGpuId            gpuId;
};

#endif
