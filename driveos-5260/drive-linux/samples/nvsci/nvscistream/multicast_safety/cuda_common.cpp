//
// NvSciStream Common Client definition
//
// Copyright (c) 2020-2021 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "cuda_common.h"

ClientCommon::ClientCommon(
    NvSciStreamBlock blockHandle,
    uint32_t numSyncs,
    uint32_t numElems) :
    handle(blockHandle),
    numSyncObjs(numSyncs),
    numElements(numElems)
{
    // Init CUDA
    size_t unused;
    CHECK_CUDARTERR(cudaDeviceGetLimit(&unused, cudaLimitStackSize));

    CHECK_CUDARTERR(cudaSetDevice(m_cudaDeviceId));

    CUuuid uuid;
    CHECK_CUDAERR(cuDeviceGetUuid(&uuid, m_cudaDeviceId));
    memcpy(&gpuId.bytes, &uuid.bytes, sizeof(uuid.bytes));
}

ClientCommon::~ClientCommon(void)
{
    // Deinit CUDA
    for (uint32_t j = 0; j < numPackets; j++) {
        for (uint32_t i = 0U; i < numReconciledElements; i++) {
            CHECK_CUDARTERR(cudaFree(packets[j].devPtr[i]));
            CHECK_CUDARTERR(cudaDestroyExternalMemory(packets[j].extMem[i]));

            if (packets[j].pCudaCopyMem[i]) {
                free(packets[j].pCudaCopyMem[i]);
            }
        }
    }

    for (uint32_t i = 0U; i < numRecvSyncObjs; i++) {
        cudaDestroyExternalSemaphore(waiterSem[i]);
    }
    for (uint32_t i = 0U; i < numSyncObjs; i++) {
        cudaDestroyExternalSemaphore(signalerSem[i]);
    }

    CHECK_CUDARTERR(cudaStreamDestroy(endpointStream));

    // Deinit other member data
    for (uint32_t i = 0U; i < numPackets; i++) {
        if (packets[i].buffers != nullptr) {
            for (uint32_t j = 0U; j < numReconciledElements; j++) {
                NvSciBufObjFree(packets[i].buffers[j]);
            }
            free(packets[i].buffers);
        }
        for (uint32_t j = 0U; j < MAX_NUM_SYNCS; j++) {
            NvSciSyncFenceClear(&packets[i].fences[j]);
        }
    }

    if (signalerAttrList != nullptr) {
        NvSciSyncAttrListFree(signalerAttrList);
    }

    for (uint32_t i = 0U; i < numSyncObjs; i++) {
        NvSciSyncObjFree(syncObjs[i]);
    }

    for (uint32_t i = 0U; i < numRecvSyncObjs; i++) {
        NvSciSyncObjFree(waiterSyncObjs[i]);
    }

    for (uint32_t i = 0U; i < numRecvSyncObjs; i++) {
        NvSciSyncFenceClear(&prefences[i]);
    }
}

// Send the number of packet elements and packet capabilities/
//  requirements to the pool.
void ClientCommon::sendPacketAttr(NvSciBufModule bufModule)
{
    // Notify the other end the number of elements per packet
    CHECK_NVSCIERR(NvSciStreamBlockPacketElementCount(handle, numElements));
    LOG_DEBUG("Sent the number of elements per packet to the pool, " << numElements << ".\n");

    // Send the packet element attributes one by one.
    for (uint32_t i = 0U; i < numElements; i++) {
        // Create buf attriubte list
        NvSciBufAttrList attrList = nullptr;
        CHECK_NVSCIERR(NvSciBufAttrListCreate(bufModule, &attrList));
        LOG_DEBUG("Created NvSciBuf attribute list of element " << i << ".");

        // Set CUDA buffer attributes
        NvSciBufType bufType = NvSciBufType_RawBuffer;
        NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;
        bool cpuaccess_flag = true;
        uint64_t rawsize = (128 * 1024);
        uint64_t align = (4 * 1024);

        NvSciBufAttrKeyValuePair attrs[] = {
            { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
            { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
            { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag,
                sizeof(cpuaccess_flag) },
            { NvSciBufRawBufferAttrKey_Size, &rawsize, sizeof(rawsize) },
            { NvSciBufRawBufferAttrKey_Align, &align, sizeof(align) },
            { NvSciBufGeneralAttrKey_GpuId, &gpuId, sizeof(gpuId) }
        };

        CHECK_NVSCIERR(NvSciBufAttrListSetAttrs(attrList,
                                                attrs,
                                                sizeof(attrs) / sizeof(NvSciBufAttrKeyValuePair)));
        LOG_DEBUG("Set attribute value of element " << i << ".");

        // Send requirements
        CHECK_NVSCIERR(NvSciStreamBlockPacketAttr(handle,
                                                  i,
                                                  i,
                                                  NvSciStreamElementMode_Asynchronous,
                                                  attrList));
        LOG_DEBUG("Sent buffer attributes of element " << i << ".\n");

        NvSciBufAttrListFree(attrList);
    }
}

// Receive a new packet, map it into own space and assigns cookie to it.
void ClientCommon::mapPacket(NvSciStreamEvent &event)
{
    // Assign cookie for the new packet
    NvSciStreamCookie cookie = assignPacketCookie();

    // Get the slot by cookie and allocate space for all elements (buffers)
    // in this packet.
    CudaPacket *packet = getPacketByCookie(cookie);
    if (packet == nullptr) {
        LOG_ERR_EXIT("\nInvalid packet cookie.\n");
    }
    packet->cookie = cookie;
    packet->handle = event.packetHandle;
    packet->buffers = static_cast<NvSciBufObj*>(
                        calloc(numReconciledElements, sizeof(NvSciBufObj)));

    LOG_DEBUG("Received a new packet (handle = " << packet->handle << "),\n"
        "\t\t\t" << numPackets << " packet(s) received.");

    // Send the packet mapping status to NvSciStream
    CHECK_NVSCIERR(NvSciStreamBlockPacketAccept(handle,
                                                packet->handle,
                                                packet->cookie,
                                                NvSciError_Success));
    LOG_DEBUG("Accepted packet (cookie = " << packet->cookie << ", handle = " << packet->handle << ").\n");
}

// Receive packet elements/buffers one by one.
void ClientCommon::mapPacketElement(NvSciStreamEvent &event)
{
    CudaPacket *packet = getPacketByCookie(event.packetCookie);
    if (packet == nullptr) {
        LOG_ERR_EXIT("\nInvalid packet cookie.\n");
    }

    uint32_t index = event.index;
    packet->buffers[index] = event.bufObj;
    LOG_DEBUG("Received and mapping element " << index << " of cookie " << event.packetCookie << ".");

    // Create CUDA buffer objects from NvSciBufObj in the packet.
    // Query size from buffer object to set CUDA buffer attribute
    NvSciBufAttrList attrList;
    CHECK_NVSCIERR(NvSciBufObjGetAttrList(packet->buffers[index], &attrList));
    NvSciBufAttrKeyValuePair attrs = { NvSciBufRawBufferAttrKey_Size, NULL, 0 };
    CHECK_NVSCIERR(NvSciBufAttrListGetAttrs(attrList, &attrs, 1));
    p_size[index] = *(static_cast<const uint64_t*>(attrs.value));

    cudaExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    memHandleDesc.handle.nvSciBufObject = packet->buffers[index];
    memHandleDesc.size = p_size[index];
    CHECK_CUDARTERR(cudaImportExternalMemory(&packet->extMem[index],
                                             &memHandleDesc));

    // Mapping with pitchlinear buffer
    cudaExternalMemoryBufferDesc bufferDesc;
    memset(&bufferDesc, 0, sizeof(bufferDesc));
    bufferDesc.size = p_size[index];
    bufferDesc.offset = 0;
    CHECK_CUDARTERR(cudaExternalMemoryGetMappedBuffer(&packet->devPtr[index],
                                                      packet->extMem[index],
                                                      &bufferDesc));

    packet->pCudaCopyMem[index] = (unsigned char *)malloc(p_size[index]);
    if (packet->pCudaCopyMem[index] == nullptr) {
        LOG_ERR_EXIT("malloc for cuda dest copy failed");
    }
    memset(packet->pCudaCopyMem[index], 0, p_size[index]);


    // Send the packet element mapping status to NvSciStream
    CHECK_NVSCIERR(NvSciStreamBlockElementAccept(handle,
                                                 packet->handle,
                                                 index,
                                                 NvSciError_Success));
    LOG_DEBUG("Accepted elements " << index << " in packet (handle = " << packet->handle << ").\n");
}

bool ClientCommon::allPacketInfoReceived(void)
{
    return  ((numReconciledElements > 0) &&
            (numReconciledElementsRecvd >= (numReconciledElements * NUM_PACKETS)) &&
            (numPackets >= NUM_PACKETS));
}


// Create signaler and waiter attribute lists and send the waiter requirement
// to the other end.
void ClientCommon::sendSyncAttr(NvSciSyncModule syncModule)
{
    // Create CUDA stream
    CHECK_CUDARTERR(cudaStreamCreateWithFlags(&endpointStream,
                                              cudaStreamNonBlocking));
    LOG_DEBUG("Created cuda stream.");

    // Create CUDA signaler attribute list
    CHECK_NVSCIERR(NvSciSyncAttrListCreate(syncModule, &signalerAttrList));
    LOG_DEBUG("Created signaler's sync attribute list.");
    CHECK_CUDARTERR(cudaDeviceGetNvSciSyncAttributes(signalerAttrList,
                                                     m_cudaDeviceId,
                                                     cudaNvSciSyncAttrSignal));
    LOG_DEBUG("Set CUDA-signaler attribute value.");

    // Create CUDA waiter attribute list
    NvSciSyncAttrList       waiterAttrList = nullptr;
    CHECK_NVSCIERR(NvSciSyncAttrListCreate(syncModule, &waiterAttrList));
    LOG_DEBUG("Created waiter's sync attribute list.");
    CHECK_CUDARTERR(cudaDeviceGetNvSciSyncAttributes(waiterAttrList,
                                                     m_cudaDeviceId,
                                                     cudaNvSciSyncAttrWait));
    LOG_DEBUG("Set CUDA-waiter attribute value.");

    // Send waiter attribute list
    CHECK_NVSCIERR(NvSciStreamBlockSyncRequirements(handle,
                                                    false,
                                                    waiterAttrList));
    LOG_DEBUG("Sent waiter's sync object requirement.\n");

    NvSciSyncAttrListFree(waiterAttrList);
}

// Receive sync object requirement from the other end.
// Reconciles its own sync object attribute and the received sync
//  object object attribute. Then it recates a sync object based on the
//  reconciled attribute list.
void ClientCommon::allocAndSendSyncObjs(NvSciStreamEvent &event)
{
    if (event.synchronousOnly) {
        return;
    }

    LOG_DEBUG("Received waiter's sync object requirement.\n");

    // Reconcile sinaler and waiter sync attributes
    NvSciSyncAttrList unreconciledList[2] = { nullptr };
    unreconciledList[0] = signalerAttrList;
    unreconciledList[1] = event.syncAttrList;
    NvSciSyncAttrList reconciledList = nullptr;
    NvSciSyncAttrList newConflictList = nullptr;
    CHECK_NVSCIERR(NvSciSyncAttrListReconcile(unreconciledList,
                                              2,
                                              &reconciledList,
                                              &newConflictList));
    LOG_DEBUG("Reconcile signaler and the reveived waiter attributes.\n");

    NvSciSyncAttrListFree(event.syncAttrList);

    // Ensure the number of sync objects is supported.
    int32_t maxNumSyncObjs = 0;
    CHECK_NVSCIERR(NvSciStreamAttributeQuery(NvSciStreamQueryableAttrib_MaxSyncObj,
                                             &maxNumSyncObjs));
    LOG_DEBUG("Query max number of sync objects allowed: " << maxNumSyncObjs << ".\n");

    if (numSyncObjs > static_cast<uint32_t>(maxNumSyncObjs)) {
        LOG_ERR_EXIT("\nnum of sync objs exceed max allowed.\n");
    }

    CHECK_NVSCIERR(NvSciStreamBlockSyncObjCount(handle, numSyncObjs));
    LOG_DEBUG("Sent number of sync objects, " << numSyncObjs << ".\n");

    // Create, map and send sync objects.
    for (uint32_t i = 0U; i < numSyncObjs; i++) {
        // Signaler
        NvSciSyncObjAlloc(reconciledList, &syncObjs[i]);
        LOG_DEBUG("Created NvSciSync object " << i << ".\n");

        // Map CUDA Signaler sync objects
        cudaExternalSemaphoreHandleDesc extSemDescSig;
        memset(&extSemDescSig, 0, sizeof(extSemDescSig));
        extSemDescSig.type = cudaExternalSemaphoreHandleTypeNvSciSync;
        extSemDescSig.handle.nvSciSyncObj = syncObjs[i];
        CHECK_CUDARTERR(cudaImportExternalSemaphore(&signalerSem[i],
                                                    &extSemDescSig));

        // Send sync object to the other end
        CHECK_NVSCIERR(NvSciStreamBlockSyncObject(handle, i, syncObjs[i]));
        LOG_DEBUG("Sent sync object " << i << ".\n");
    }

    // Free resources
    NvSciSyncAttrListFree(reconciledList);
    NvSciSyncAttrListFree(newConflictList);
}

// Receive and map sync object.
void ClientCommon::mapSyncObj(NvSciStreamEvent &event)
{
    uint32_t index = event.index;
    if (index >= numRecvSyncObjs) {
        LOG_ERR_EXIT("\nInvalid sync object index received\n");
    }
    waiterSyncObjs[index] = event.syncObj;
    LOG_DEBUG("Received sync object " << index << ".\n");

    // Map CUDA waiter sync object
    cudaExternalSemaphoreHandleDesc extSemDesc;
    memset(&extSemDesc, 0, sizeof(extSemDesc));
    extSemDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    extSemDesc.handle.nvSciSyncObj = waiterSyncObjs[index];
    CHECK_CUDARTERR(cudaImportExternalSemaphore(&waiterSem[index],
                                                &extSemDesc));
}

bool ClientCommon::allSyncInfoReceived(void)
{
    return (hasRecvSyncAttr &&
            hasRecvSyncCount &&
            (numSyncObjsRecvd == numRecvSyncObjs));
}

void ClientCommon::handleResourceSetupEvents(NvSciStreamEvent &event)
{
    switch (event.type)
    {
    case NvSciStreamEventType_SyncAttr:
        hasRecvSyncAttr = true;
        allocAndSendSyncObjs(event);
        break;
    case NvSciStreamEventType_SyncCount:
        hasRecvSyncCount = true;
        numRecvSyncObjs = event.count;
        LOG_DEBUG("Received the number of sync objects, " << numRecvSyncObjs << ".\n");
        break;
    case NvSciStreamEventType_SyncDesc:
        numSyncObjsRecvd++;
        mapSyncObj(event);
        break;
    case NvSciStreamEventType_PacketElementCount:
        numReconciledElements = event.count;
        LOG_DEBUG("Received the number of elements per packet from pool: " << numReconciledElements << ".\n");
        break;
    case NvSciStreamEventType_PacketAttr:
        // Receive reconciled packet element attributes:
        // userData, syncMode and bufAttrList, which are not used in this app.
        // The app should free the attribute list if not used any more.
        NvSciBufAttrListFree(event.bufAttrList);
        LOG_DEBUG("Received attributes of element " << event.index << " from pool.");
        break;
    case NvSciStreamEventType_PacketCreate:
        numPackets++;
        mapPacket(event);
        break;
    case NvSciStreamEventType_PacketElement:
        numReconciledElementsRecvd++;
        mapPacketElement(event);
        break;
    default:
        break;
    }
}
