//
// CUDA consumer client definition.
//
// Copyright (c) 2020-2021 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include <cstring>
#include "cuda_consumer.h"

CudaConsumer::CudaConsumer(
    NvSciStreamBlock blockHandle,
    uint32_t numSyncs,
    uint32_t numElems):
    ClientCommon(blockHandle, numSyncs, numElems)
{
    LOG_DEBUG("CUDA Consumer Created");
}

// Consumer receives PacketReady event and acquires a packet from
//  the queue.
bool CudaConsumer::acquirePacket(NvSciStreamCookie &cookie)
{
    NvSciStreamEvent event;
    uint32_t timeouts = 0U;
    cookie = 0U;

    while (true) {
        NvSciError err = NvSciStreamBlockEventQuery(handle,
                                                    QUERY_TIMEOUT,
                                                    &event);
        if (err == NvSciError_Success) {
            if (event.type == NvSciStreamEventType_Disconnected) {
                LOG_DEBUG("Receive Disconnected event.");
                return false;
            }
            if (event.type != NvSciStreamEventType_PacketReady) {
                LOG_ERR_EXIT("Consumer failed to Receive PACKET_READY event.");
            }
            LOG_DEBUG("Consumer received PACKET_READY event.");

            // Clear prefences
            for (uint32_t i = 0U; i < numRecvSyncObjs; i++) {
                NvSciSyncFenceClear(&prefences[i]);
            }

            CHECK_NVSCIERR(NvSciStreamConsumerPacketAcquire(handle,
                                                            &cookie,
                                                            prefences));
            if (cookie == 0U) {
                LOG_ERR_EXIT("Invalid payload cookie");
            }
            LOG_DEBUG("Consumer acquired a packet (cookie = " << cookie << ").");

            // Assign prefences value to the corresponding packet
            CudaPacket *packet = getPacketByCookie(cookie);
            if (packet == nullptr) {
                LOG_ERR_EXIT("Invalid packet cookie.");
            }

            for (uint32_t i = 0U; i < numRecvSyncObjs; i++) {
                NvSciSyncFenceDup(&prefences[i], &packet->fences[i]);
                NvSciSyncFenceClear(&prefences[i]);
            }

            return true;
        }  else if (err == NvSciError_Timeout) {
            // if query timeouts - keep waiting for event
            // until wait threshold is reached
            if (timeouts < MAX_QUERY_TIMEOUTS) {
                timeouts++;
            } else {
                LOG_ERR_EXIT("Consumer Query waits seem to be taking forever!");
            }
        } else {
            LOG_ERR("NvSciStreamBlockEventQuery Failed:");
            CHECK_NVSCIERR(err);
        }
    }
}

// Consumer waits for prefences, read buffer data, and then generates
//  postfences.
void CudaConsumer::processPayload(NvSciStreamCookie cookie)
{
    LOG_DEBUG("Consumer processes payload (cookie = " << cookie << ").");

    CudaPacket *packet = getPacketByCookie(cookie);
    if (packet == nullptr) {
        LOG_ERR_EXIT("Invalid packet cookie.");
    }

    // insert for prefences
    for (uint32_t i = 0U; i < numRecvSyncObjs; i++) {
        cudaExternalSemaphoreWaitParams waitParams;
        memset(&waitParams, 0, sizeof(waitParams));
        waitParams.params.nvSciSync.fence = &packet->fences[i];
        waitParams.flags = 0;
        CHECK_CUDARTERR(cudaWaitExternalSemaphoresAsync(&waiterSem[i],
                                                        &waitParams,
                                                        1,
                                                        endpointStream));
        NvSciSyncFenceClear(&packet->fences[i]);
    }

    for (uint32_t i = 0; i < numReconciledElements; i++) {
        CHECK_CUDARTERR(cudaMemcpy2DAsync(packet->pCudaCopyMem[i],
                                          p_size[i],
                                          packet->devPtr[i],
                                          p_size[i],
                                          p_size[i],
                                          1,
                                          cudaMemcpyDeviceToHost,
                                          endpointStream));
    }

    // Generate postfences
    for (uint32_t i = 0U; i < numSyncObjs; i++) {
        cudaExternalSemaphoreSignalParams signalParams;
        memset(&signalParams, 0, sizeof(signalParams));
        signalParams.params.nvSciSync.fence = &packet->fences[i];
        signalParams.flags = 0;
        CHECK_CUDARTERR(cudaSignalExternalSemaphoresAsync(&signalerSem[i],
                                                          &signalParams,
                                                          1,
                                                          endpointStream));
    }
}

// Consumer release the packet and return it to the producer to retuse.
void CudaConsumer::releasePacket(NvSciStreamCookie cookie)
{
    // Get the buffer by consumer cookie.
    CudaPacket *packet = getPacketByCookie(cookie);
    if (packet == nullptr) {
        LOG_ERR_EXIT("Invalid packet cookie.");
    }

    CHECK_NVSCIERR(NvSciStreamConsumerPacketRelease(handle,
                                                    packet->handle,
                                                    packet->fences));

    for (uint32_t i = 0U; i < numSyncObjs; i++) {
        NvSciSyncFenceClear(&packet->fences[i]);
    }

    LOG_DEBUG("Consumer released the packet (cookie = " << packet->cookie << ", handle = " << packet->handle << ").");
}
