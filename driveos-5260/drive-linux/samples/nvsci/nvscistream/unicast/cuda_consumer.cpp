//
// CUDA consumer client definition.
//
// Copyright (c) 2019-2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "cuda_consumer.h"
#include <cstring>

namespace NvScistreamSample
{
    CudaConsumer::CudaConsumer(NvSciStreamBlock blockHandle, uint32_t numSyncs):
        ClientCommon(blockHandle, numSyncs),
        waitContext(nullptr),
        numPacketsRecvd(0U)
    {
        cuInit(0);
        cuDeviceGet (&dev, 0);
        cuCtxCreate(&cuctx, CU_CTX_MAP_HOST, dev);

        LOG_DEBUG("CUDA Consumer Created");
    }

    CudaConsumer::~CudaConsumer(void)
    {
        if (waitContext != nullptr) {
            NvSciSyncCpuWaitContextFree(waitContext);
        }

        for (uint32_t i = 0; i < numPackets; i++) {
            NvSciStreamCookie cookie = getCookieAtIndex(i);
            unmapBuffers(cookie);
        }
        unmapSyncObjs();

        cuCtxDestroy(cuctx);
    }

    // Buffer setup functions
    void CudaConsumer::createBufAttrList(NvSciBufModule bufModule) {
        for (uint32_t i = 0U; i < numElements; i++) {
            CHECK_NVSCIERR(
                NvSciBufAttrListCreate(
                    bufModule,
                    &bufAttrLists[i]));
            LOG_DEBUG("Create NvSciBuf attribute list of element " << i << ".");

            NvSciBufAttrList attrList = bufAttrLists[i];

            NvSciRmGpuId gpuid;
            CUuuid uuid;
            CHECK_CUDAERR(cuDeviceGetUuid(&uuid, dev));
            memcpy(&gpuid.bytes, &uuid.bytes, sizeof(uuid.bytes));

            NvSciBufAttrKeyValuePair genbuffattrs[] = {
                { NvSciBufGeneralAttrKey_GpuId, &gpuid, sizeof(gpuid) },
            };

            CHECK_NVSCIERR(
                NvSciBufAttrListSetAttrs(
                    attrList,
                    genbuffattrs,
                    sizeof(genbuffattrs) / sizeof(NvSciBufAttrKeyValuePair)));

            NvSciBufType bufType = NvSciBufType_Image;
            NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_Readonly;
            bool cpuaccess_flag = true;

            NvSciBufAttrKeyValuePair bufAttrs[] = {
                { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
                { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
                { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag,
                    sizeof(cpuaccess_flag) },
            };

            CHECK_NVSCIERR(
                NvSciBufAttrListSetAttrs(
                    attrList,
                    bufAttrs,
                    sizeof(bufAttrs) / sizeof(NvSciBufAttrKeyValuePair)));

            LOG_DEBUG("Set attribute value of element " << i << ".");
        }
    }

    // Map the received packet in its own space and assigns cookie to it.
    void CudaConsumer::mapPacket(NvSciStreamCookie &cookie, NvSciStreamPacket packetHandle) {
        ClientCommon::mapPacket(cookie, packetHandle);
    }

    // Create client buffer objects from NvSciBufObj in the packet.
    void CudaConsumer::mapBuffers(NvSciStreamCookie cookie) {
        // Get NvSciBufObj from packet
        uint32_t id = packetCookie2Id(cookie);
        Packet *packet = getPacketByCookie(cookie);

        for (uint32_t i = 0; i < numReconciledElements; i++) {
            LOG_DEBUG("Consumer Mapping element: " << i << " of cookie " << std::hex << cookie << ".");
            CUDA_EXTERNAL_MEMORY_HANDLE_DESC memHandleDesc;
            NvSciBufAttrList retList;
            NvSciBufType bufType;
            uint64_t size = 0;

            // Queyy size from Buffer
            CHECK_NVSCIERR(NvSciBufObjGetAttrList(packet->buffers[i], &retList));

            NvSciBufAttrKeyValuePair genattrs[] = {
                {NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
            };

            CHECK_NVSCIERR(NvSciBufAttrListGetAttrs(retList, genattrs, sizeof(genattrs)/sizeof(NvSciBufAttrKeyValuePair)));
            bufType = *(static_cast<const NvSciBufType*>(genattrs[0].value));

            NvSciBufAttrKeyValuePair imgattrs[] = {
                {NvSciBufImageAttrKey_Size, NULL, 0 },
            };

            CHECK_NVSCIERR(NvSciBufAttrListGetAttrs(retList, imgattrs, sizeof(imgattrs)/sizeof(NvSciBufAttrKeyValuePair)));
            size = *(static_cast<const uint64_t*>(imgattrs[0].value));

            memset(&memHandleDesc, 0, sizeof(memHandleDesc));
            memHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF;
            memHandleDesc.handle.nvSciBufObject = packet->buffers[i];

            memHandleDesc.size = size;
            CHECK_CUDAERR(cuImportExternalMemory(&extMem[id][i], &memHandleDesc));

            CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc;
            memset(&bufferDesc, 0, sizeof(bufferDesc));
            bufferDesc.size = size;
            bufferDesc.offset = 0;
            CHECK_CUDAERR(cuExternalMemoryGetMappedBuffer(&devPtr[id][i], extMem[id][i], &bufferDesc));

            pCudaCopyMem[id][i] = (unsigned char *)malloc(size);
            if (pCudaCopyMem[id][i] == nullptr) {
                LOG_ERR_EXIT("malloc for cuda dest copy failed");
            }
            p_size[i] = size;
            memset(pCudaCopyMem[id][i], 0, size);
        }
    }

    // Unmap buffers in its own space.
    void CudaConsumer::unmapBuffers(NvSciStreamCookie cookie) {
        uint32_t id = packetCookie2Id(cookie);
        for (uint32_t i = 0; i < numReconciledElements; i++) {
                CHECK_CUDAERR(cuMemFree(devPtr[id][i]));
                CHECK_CUDAERR(cuDestroyExternalMemory(extMem[id][i]));
            if (pCudaCopyMem[id][i]) {
                free(pCudaCopyMem[id][i]);
                pCudaCopyMem[id][i] = NULL;
            }
        }
    }

    // Sync object setup functions
    void CudaConsumer::createSyncAttrLists(NvSciSyncModule syncModule)
    {
        CHECK_NVSCIERR(NvSciSyncAttrListCreate(syncModule, &signalerAttrList));
        LOG_DEBUG("Create signaler's sync attribute list.");
        CHECK_CUDAERR(cuDeviceGetNvSciSyncAttributes(signalerAttrList, dev, CUDA_NVSCISYNC_ATTR_SIGNAL));
        LOG_DEBUG("Set CUDA-signaler attribute value.");

        CHECK_NVSCIERR(NvSciSyncAttrListCreate(syncModule, &waiterAttrList));
        LOG_DEBUG("Create waiter's sync attribute list.");
        CHECK_CUDAERR(cuDeviceGetNvSciSyncAttributes(waiterAttrList, dev, CUDA_NVSCISYNC_ATTR_WAIT));
        LOG_DEBUG("Set CUDA-waiter attribute value.");
    }

    void CudaConsumer::handleResourceSetupEvents(NvSciStreamEvent &event)
    {
        switch (event.type)
        {
            case NvSciStreamEventType_SyncAttr:
                recvSyncObjAttrs(event);
                LOG_DEBUG("Consumer received producer attributes");
                reconcileAndAllocSyncObjs();
                LOG_DEBUG("Consumer reconciled producer attributes and creates sync object:\n"
                        "\tsending sync objects to producer...");
                sendSyncObjs();
                break;

            case NvSciStreamEventType_SyncCount:
                recvSyncObjCount(event.count);
                break;

            case NvSciStreamEventType_SyncDesc:
                recvSyncObj(event);
                break;

            case NvSciStreamEventType_PacketElementCount:
                recvReconciledPacketElementCount(event.count);
                break;

            case NvSciStreamEventType_PacketAttr:
                recvReconciledPacketAttr(event);
                break;

            case NvSciStreamEventType_PacketCreate:
            {
                NvSciStreamCookie cookie = 0U;
                NvSciStreamPacket packetHandle;
                recvPacket(event, packetHandle);
                mapPacket(cookie, packetHandle);
                registerPacket(cookie);
            }
                break;

            case NvSciStreamEventType_PacketElement:
                recvPacketElement(event);
                break;

            default:
                break;
        }
    }

    void CudaConsumer::reconcileAndAllocSyncObjs(void)
    {
        ClientCommon::reconcileAndAllocSyncObjs();
    }

    void CudaConsumer::mapSyncObjs(void)
    {
        for (uint32_t i = 0U; i < numRecvSyncObjs; i++) {
            // Waiter
            CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC extSemDesc;
            memset(&extSemDesc, 0, sizeof(extSemDesc));
            extSemDesc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC;
            extSemDesc.handle.nvSciSyncObj = waiterSyncObjs[i];
            CHECK_CUDAERR(cuImportExternalSemaphore(&waiterSem[i], &extSemDesc));
        }

        for (uint32_t i = 0U; i < numSyncObjs; i++) {
            // Signaler
            CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC extSemDescSig;
            memset(&extSemDescSig, 0, sizeof(extSemDescSig));
            extSemDescSig.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC;
            extSemDescSig.handle.nvSciSyncObj = syncObjs[i];
            CHECK_CUDAERR(cuImportExternalSemaphore(&signalerSem[i], &extSemDescSig));
        }
    }

    void CudaConsumer::initFences(NvSciSyncModule syncModule)
    {
    }

    void CudaConsumer::unmapSyncObjs(void)
    {
        for (uint32_t i = 0U; i < numRecvSyncObjs; i++) {
            cuDestroyExternalSemaphore(waiterSem[i]);
        }
        for (uint32_t i = 0U; i < numSyncObjs; i++) {
            cuDestroyExternalSemaphore(signalerSem[i]);
        }
    }

    // Streaming functions

    // Consumer receives PacketReady event and acquires a packet from
    //  the queue.
    bool CudaConsumer::acquirePacket(NvSciStreamCookie &cookie)
    {
        NvSciStreamEvent event;
        uint32_t timeouts = 0U;

        while (true) {
            NvSciError err = NvSciStreamBlockEventQuery(handle, QUERY_TIMEOUT, &event);
            if (err == NvSciError_Success) {

                if (event.type == NvSciStreamEventType_Disconnected) {
                    LOG_DEBUG("Receive Disconnected event.");
                    return false;
                }

                if (event.type != NvSciStreamEventType_PacketReady) {
                    LOG_ERR_EXIT("Failed to Receive PACKET_READY event.");
                }
                LOG_DEBUG("Receive PACKET_READY event.");

                // Clear prefences
                for (uint32_t i = 0U; i < numRecvSyncObjs; i++) {
                    NvSciSyncFenceClear(&prefences[i]);
                }

                cookie = 0U;
                CHECK_NVSCIERR(
                    NvSciStreamConsumerPacketAcquire(handle,
                                                     &cookie,
                                                     prefences));
                if (cookie == 0U) {
                    LOG_ERR_EXIT("invalid payload cookie");
                }
                LOG_DEBUG("Acquire a packet (cookie = " << std::hex << cookie << ").");

                // Assign prefences value to the corresponding packet
                Packet *packet = getPacketByCookie(cookie);
                if (packet == nullptr) {
                    LOG_ERR_EXIT("Invalid packet for cookie.");
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

    void CudaConsumer::processPayload(NvSciStreamCookie cookie) {
        LOG_DEBUG("Process payload (cookie = " << std::hex << cookie << ").");

        uint32_t id = packetCookie2Id(cookie);
        Packet *packet = getPacketByCookie(cookie);
        if (packet == nullptr) {
            LOG_ERR_EXIT("Invalid packet for cookie.");
        }

        for (uint32_t i = 0U; i < numRecvSyncObjs; i++) {
            CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS waitParams;
            memset(&waitParams, 0, sizeof(waitParams));
            waitParams.params.nvSciSync.fence = &packet->fences[i];
            CHECK_CUDAERR(cuWaitExternalSemaphoresAsync(&waiterSem[i], &waitParams, 1, 0));

            NvSciSyncFenceClear(&packet->fences[i]);
        }

        for (uint32_t i = 0; i < numReconciledElements; i++) {
            CHECK_CUDAERR(cuMemcpyDtoH(pCudaCopyMem[id][i], devPtr[id][i], p_size[i]));
        }

        for (uint32_t i = 0U; i < numSyncObjs; i++) {
            CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signalParams;
            memset(&signalParams, 0, sizeof(signalParams));
            signalParams.params.nvSciSync.fence = &packet->fences[i];
            CHECK_CUDAERR(cuSignalExternalSemaphoresAsync(&signalerSem[i], &signalParams, 1, 0));
        }
    }

    // Consumer release the packet and return it to the producer to retuse.
    void CudaConsumer::releasePacket(NvSciStreamCookie cookie)
    {
        // Get the buffer by consumer cookie.
        Packet *packet = getPacketByCookie(cookie);
        if (packet == nullptr) {
            LOG_ERR_EXIT("Invalid packet for cookie.");
        }

        CHECK_NVSCIERR(
            NvSciStreamConsumerPacketRelease(
                handle,
                packet->handle,
                packet->fences));

        for (uint32_t i = 0U; i < numSyncObjs; i++) {
            NvSciSyncFenceClear(&packet->fences[i]);
        }

        LOG_DEBUG("Release the packet (cookie = " << packet->cookie << ", handle = " << packet->handle << ").");
    }
}
