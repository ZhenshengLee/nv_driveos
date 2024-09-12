//
// NvSciStream Common Client definition
//
// Copyright (c) 2019-2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "client_common.h"

namespace NvScistreamSample
{
    ClientCommon::ClientCommon(NvSciStreamBlock blockHandle, uint32_t numSyncs) :
        handle(blockHandle),
        hasRecvSyncAttr(false),
        hasRecvSyncCount(false),
        signalerAttrList(nullptr),
        waiterAttrList(nullptr),
        numSyncObjs(numSyncs),
        numRecvSyncObjs(0U),
        numSyncObjsRecvd(0U),
        waiterSyncObjs(nullptr),
        prefences(nullptr),
        numElements(NUM_ELEMENTS_PER_PACKET),
        numReconciledElements(0U),
        numReconciledElementsRecvd(0U),
        reconciledElementAttrs(nullptr),
        numPackets(0U)
    {
        recvSyncAttrList = nullptr;

        for (uint32_t i = 0U; i < numSyncObjs; i++) {
            syncObjs[i] = nullptr;
        }
        for (uint32_t i = 0U; i < numElements; i++) {
            bufAttrLists[i] = nullptr;
        }
    }

    ClientCommon::~ClientCommon(void)
    {
        if (signalerAttrList != nullptr) {
            NvSciSyncAttrListFree(signalerAttrList);
            signalerAttrList = nullptr;
        }

        if (waiterAttrList != nullptr) {
            NvSciSyncAttrListFree(waiterAttrList);
            waiterAttrList = nullptr;
        }

        if (recvSyncAttrList != nullptr) {
            NvSciSyncAttrListFree(recvSyncAttrList);
            recvSyncAttrList = nullptr;
        }

        for (uint32_t i = 0U; i < numSyncObjs; i++) {
            NvSciSyncObjFree(syncObjs[i]);
            syncObjs[i] = nullptr;
        }

        freeSyncObjs(waiterSyncObjs, numRecvSyncObjs);
        for (uint32_t i = 0U; i < numRecvSyncObjs; i++) {
            NvSciSyncFenceClear(&prefences[i]);
            prefences = nullptr;
        }

        for (uint32_t i = 0U; i < numElements; i++) {
            NvSciBufAttrListFree(bufAttrLists[i]);
            bufAttrLists[i] = nullptr;
        }

        freeElementAttrs(reconciledElementAttrs, numReconciledElements);

        for (uint32_t i = 0U; i < numPackets; i++) {
            for (uint32_t j = 0U; j < numElements; j++) {
                NvSciBufObjFree(packets[i].buffers[j]);
            }
            for (uint32_t j = 0U; j < MAX_NUM_SYNCS; j++) {
                NvSciSyncFenceClear(&packets[i].fences[j]);
            }
        }
    }

    // Send the number of packet elements and packet capabilities/
    //  requirements to the pool.
    void ClientCommon::sendPacketAttr(void)
    {
        // Need to notify the other end elements per packet
        CHECK_NVSCIERR(NvSciStreamBlockPacketElementCount(handle, numElements));
        LOG_DEBUG("Send the number of elements per packet to the pool, " << numElements << ".\n");

        // Send the packet element attributes one by one.
        for (uint32_t i = 0U; i < numElements; i++) {
            CHECK_NVSCIERR(
                NvSciStreamBlockPacketAttr(handle, i, 0U,
                                           NvSciStreamElementMode_Asynchronous,
                                           bufAttrLists[i]));
            LOG_DEBUG("Send buffer attributes of element " << i <<".\n");
        }
    }

    // receive the determined packet layout from pool.
    void ClientCommon::recvReconciledPacketElementCount(uint32_t count)
    {
        numReconciledElements = count;
        LOG_DEBUG("Receive the number of elements per packet from pool: " << count << ".\n");

        // Allocate space for sync objects
        reconciledElementAttrs = static_cast<PacketElementAttr*>(
            calloc(numReconciledElements, sizeof(PacketElementAttr)));
    }

    // receive the determined packet layout from pool.
    void ClientCommon::recvReconciledPacketAttr(NvSciStreamEvent &event)
    {
        if (reconciledElementAttrs == nullptr) {
            LOG_ERR_EXIT("\nReceived Element Attr before Element count\n");
        }

        // Receive reconciled packet attr
        uint32_t index = event.index;
        if (index >= numReconciledElements) {
            LOG_ERR_EXIT("\nInvalid packet attr index received\n");
        }
        reconciledElementAttrs[index].userData = event.userData;
        reconciledElementAttrs[index].syncMode = event.syncMode;
        reconciledElementAttrs[index].bufAttrList = event.bufAttrList;
        LOG_DEBUG("Receive reconciled attributes of element" << index << "from pool.");
    }

    // Receive PacketCreate event.
    void ClientCommon::recvPacket(NvSciStreamEvent &event,
                                  NvSciStreamPacket &packetHandle)
    {
        packetHandle = event.packetHandle;
        numPackets++;
        LOG_DEBUG("Receive a new packet (handle = " << packetHandle << "),\n"
                "\t\t\t" << numPackets << " packet(s) received.\n");
    }

    // Map the received packet in its own space and assigns cookie to it.
    void ClientCommon::mapPacket(
        NvSciStreamCookie &cookie,
        NvSciStreamPacket packetHandle)
    {
        // Assign cookie for the new packet
        cookie = assignPacketCookie();

        // Get the slot by cookie and allocate space for all elements (buffers)
        // in this packet.
        Packet *packet = getPacketByCookie(cookie);
        if (packet == nullptr) {
            LOG_ERR_EXIT("\nInvalid packet for cookie.\n");
        }

        packet->cookie = cookie;
        packet->handle = packetHandle;
        packet->buffers = static_cast<NvSciBufObj*>(
            calloc(numReconciledElements, sizeof(NvSciBufObj)));

        for (uint32_t i = 0U; i < MAX_NUM_SYNCS; i++) {
            packet->fences[i] = NvSciSyncFenceInitializer;
        }
    }

    // Send the packet cookie and mapping status to NvSciStream.
    void ClientCommon::registerPacket(
        NvSciStreamCookie cookie,
        NvSciError status)
    {
        Packet *packet = getPacketByCookie(cookie);
        if (packet == nullptr) {
            LOG_ERR_EXIT("\nInvalid packet for cookie.\n");
        }

        // TODO: send the actual mapping status
        CHECK_NVSCIERR(
            NvSciStreamBlockPacketAccept(
                handle,
                packet->handle,
                packet->cookie,
                status));
        LOG_DEBUG("Assign cookie " << std::hex << packet->cookie << " to the packet (handle = " << packet-handle << ").\n");
    }

    // Receive packet elements/buffers one by one.
    void ClientCommon::recvPacketElement(NvSciStreamEvent &event)
    {
        Packet *packet = getPacketByCookie(event.packetCookie);
        if (packet == nullptr) {
            LOG_ERR_EXIT("\nInvalid packet for cookie.\n");
        }

        uint32_t index = event.index;
        packet->buffers[index] = event.bufObj;
        numReconciledElementsRecvd++;
        LOG_DEBUG("Receive buffer object of element " << index <<".\n");
    }

    bool ClientCommon::allPacketInfoReceived(void)
    {
        return  ((numReconciledElements > 0) &&
                (numReconciledElementsRecvd >= (numReconciledElements * NUM_PACKETS)) &&
                (numPackets >= NUM_PACKETS));
    }

    // Send the packet element mapping status to NvSciStream.
    void ClientCommon::registerPacketElements(
        NvSciStreamCookie cookie,
        NvSciError status)
    {
        Packet *packet = getPacketByCookie(cookie);
        if (packet == nullptr) {
            LOG_ERR_EXIT("\nInvalid packet for cookie.\n");
        }

        // TODO: send the actual mapping status
        for (uint32_t i = 0U; i < numReconciledElements; i++) {
            CHECK_NVSCIERR(
                NvSciStreamBlockElementAccept(
                    handle,
                    packet->handle,
                    i,
                    status));
        }
        LOG_DEBUG("Accept elements in packet (handle = " << packet->handle << ").\n");
    }

    // Send sync object requirement.
    void ClientCommon::sendSyncAttr(void)
    {
        CHECK_NVSCIERR(
            NvSciStreamBlockSyncRequirements(handle, false, waiterAttrList));
        LOG_DEBUG("Send waiter's sync object requirement.\n");
    }

    // Receive sync object requirement from the other end.
    void ClientCommon::recvSyncObjAttrs(NvSciStreamEvent &event)
    {
        if (!event.synchronousOnly) {
            recvSyncAttrList = event.syncAttrList;
        }
        hasRecvSyncAttr = true;
        LOG_DEBUG("Receive waiter's sync object requirement.\n");
    }

    // Query the max number of sync objects allowed, and validate
    // its number of sync objects.
    void ClientCommon::validateNumSyncObjs(void)
    {
        int32_t maxNumSyncObjs = 0;
        CHECK_NVSCIERR(
            NvSciStreamAttributeQuery(
                NvSciStreamQueryableAttrib_MaxSyncObj,
                &maxNumSyncObjs));
        LOG_DEBUG("Query max number of sync objects allowed: " << maxNumSyncObjs << ".\n");

        // Ensure the number of sync objects is supported.
        if (numSyncObjs > static_cast<uint32_t>(maxNumSyncObjs)) {
            LOG_ERR_EXIT("\nnum of sync objs exceed max allowed.\n");
        }
        LOG_DEBUG("Number of waiter sync objects: " << numElements << ".\n");
    }

    // Reconciles its own sync object attribute and the received sync
    //  object object attribute. Then it recates a sync object based on the
    //  reconciled attribute list.
    void ClientCommon::reconcileAndAllocSyncObjs(void)
    {
        NvSciSyncAttrList unreconciledList[2] = { nullptr };
        unreconciledList[0] = signalerAttrList;
        unreconciledList[1] = recvSyncAttrList;
        NvSciSyncAttrList reconciledList = nullptr;
        NvSciSyncAttrList newConflictList = nullptr;
        CHECK_NVSCIERR(
            NvSciSyncAttrListReconcile(
                unreconciledList,
                2,
                &reconciledList,
                &newConflictList));
        LOG_DEBUG("Common: Reconcile its signaler attributes and the reveived waiter attributes.\n");

        // Create sync objects.
        for (uint32_t i = 0U; i < numSyncObjs; i++) {
            NvSciSyncObjAlloc(reconciledList, &syncObjs[i]);
            LOG_DEBUG("Common: Create NvSciSync object " << i << " with the reconciled attribute list.\n");
        }

        // Free resources
        NvSciSyncAttrListFree(reconciledList);
        NvSciSyncAttrListFree(newConflictList);
    }

    // Send the number of sync objects and the sync objects.
    void ClientCommon::sendSyncObjs(void)
    {
        CHECK_NVSCIERR(
            NvSciStreamBlockSyncObjCount(handle, numSyncObjs));
        LOG_DEBUG("Send number of sync objects, " << numSyncObjs << ".\n");

        // Send the sync objects one by one.
        for (uint32_t i = 0U; i < numSyncObjs; i++) {
            CHECK_NVSCIERR(
                NvSciStreamBlockSyncObject(handle, i, syncObjs[i]));
            LOG_DEBUG("Send sync object " << i << ".\n");
        }
    }

    // Receive the number of sync objects.
    void ClientCommon::recvSyncObjCount(uint32_t count)
    {
        numRecvSyncObjs = count;
        LOG_DEBUG("Receive the number of sync objects, " << numRecvSyncObjs << ".\n");

        if (numRecvSyncObjs > 0U) {
            // Allocate space for sync objects
            waiterSyncObjs = static_cast<NvSciSyncObj*>(
                calloc(numRecvSyncObjs, sizeof(NvSciSyncObj)));

            // Allocate space for temporary prefences
            prefences = static_cast<NvSciSyncFence*>(
                calloc(numRecvSyncObjs, sizeof(NvSciSyncFence)));
            for (uint32_t i = 0U; i < numRecvSyncObjs; i++) {
                prefences[i] = NvSciSyncFenceInitializer;
            }
        }
        hasRecvSyncCount = true;
    }

    // Receive sync object.
    void ClientCommon::recvSyncObj(NvSciStreamEvent &event)
    {
        if (waiterSyncObjs == nullptr) {
            LOG_ERR_EXIT("\nReceived sync object before sync obj count\n");
        }

        uint32_t index = event.index;
        if (index >= numRecvSyncObjs) {
            LOG_ERR_EXIT("\nInvalid sync object index received\n");
        }

        waiterSyncObjs[index] = event.syncObj;
        numSyncObjsRecvd++;
        LOG_DEBUG("Receive sync object " << index << ".\n");
    }

    bool ClientCommon::allSyncInfoReceived(void) {
        return (hasRecvSyncAttr &&
                hasRecvSyncCount &&
                (numSyncObjsRecvd == numRecvSyncObjs));
    }
}
