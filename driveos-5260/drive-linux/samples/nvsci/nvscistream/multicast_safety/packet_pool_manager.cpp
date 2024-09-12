//
// Pool manager client declaration definition.
//
// Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "packet_pool_manager.h"

PacketPoolManager::PacketPoolManager(
    NvSciStreamBlock blockHandle):
    handle(blockHandle)
{
}

PacketPoolManager::~PacketPoolManager(void)
{
    if (prodElementAttrs != nullptr) {
        for (uint32_t i = 0U; i < numProdElements; i++) {
            if (prodElementAttrs[i].bufAttrList != nullptr) {
                NvSciBufAttrListFree(prodElementAttrs[i].bufAttrList);
            }
        }
        free(prodElementAttrs);
    }

    if (consElementAttrs != nullptr) {
        for (uint32_t i = 0U; i < numConsElements; i++) {
            if (consElementAttrs[i].bufAttrList != nullptr) {
                NvSciBufAttrListFree(consElementAttrs[i].bufAttrList);
            }
        }
        free(consElementAttrs);
    }

    if (elementAttrs != nullptr) {
        for (uint32_t i = 0U; i < numElements; i++) {
            if (elementAttrs[i].bufAttrList != nullptr) {
                NvSciBufAttrListFree(elementAttrs[i].bufAttrList);
            }
        }
        free(elementAttrs);
    }

    for (uint32_t i = 0U; i < numPackets; i++) {
        if (packets[i].buffers != nullptr) {
            for (uint32_t j = 0; j < numElements; j++) {
                NvSciBufObjFree(packets[i].buffers[j]);
            }
            free(packets[i].buffers);
        }
    }
}

// Receive producer's packet capabilities and consumer's packet
//  requirements.
void PacketPoolManager::recvPacketAttrs(NvSciStreamEvent &event)
{
    uint32_t index;

    switch (event.type) {
        case NvSciStreamEventType_PacketElementCountProducer:
            numProdElements = event.count;
            LOG_DEBUG("Pool received number of elements per packet from producer: " << event.count << ".\n");

            // Allocate space for producer packet element attr list
            prodElementAttrs = static_cast<PacketElementAttr*>(
                calloc(numProdElements, sizeof(PacketElementAttr)));

            break;

        case NvSciStreamEventType_PacketElementCountConsumer:
            numConsElements = event.count;
            LOG_DEBUG("Pool received number of elements per packet from consumer: " << event.count << ".\n");

            // Allocate space for producer packet element attr list
            consElementAttrs = static_cast<PacketElementAttr*>(
                calloc(numConsElements, sizeof(PacketElementAttr)));

            break;

        case NvSciStreamEventType_PacketAttrProducer:
            if (numProdElements == 0) {
                LOG_ERR_EXIT("\nInvalid number of producer elements");
            }

            index = event.index;
            prodElementAttrs[index].userData = event.userData;
            prodElementAttrs[index].syncMode = event.syncMode;
            prodElementAttrs[index].bufAttrList = event.bufAttrList;
            LOG_DEBUG("Pool received packet capabilities of element " << index << " from producer.\n");
            numProdElementsRecv++;
            break;

        case NvSciStreamEventType_PacketAttrConsumer:
            if (numConsElements == 0) {
                LOG_ERR_EXIT("\nInvalid number of consumer elements");
            }

            index = event.index;
            consElementAttrs[index].userData = event.userData;
            consElementAttrs[index].syncMode = event.syncMode;
            consElementAttrs[index].bufAttrList = event.bufAttrList;
            LOG_DEBUG("Pool received packet requirements of element " << index << " from consumer.\n");
            numConsElementsRecv++;
            break;

        default:
            break;
    }
}

bool PacketPoolManager::recvAllPacketAttrs(void)
{
    return ((numProdElements > 0) &&
            (numProdElementsRecv >= numProdElements) &&
            (numConsElements > 0) &&
            (numConsElementsRecv >= numConsElements));
}

// Reconcile the producer's and consumer's attribute lists for each
//  packet element.
void PacketPoolManager::reconcilePacketAttrs(void)
{
    int32_t maxNumElements = 0;
    CHECK_NVSCIERR(NvSciStreamAttributeQuery(NvSciStreamQueryableAttrib_MaxElements,
                                             &maxNumElements));
    LOG_DEBUG("Query max number of elements per packet allowed: " << maxNumElements << ".\n");

    // Allocate space for packet element attributes
    elementAttrs = static_cast<PacketElementAttr*>(
                    calloc(maxNumElements, sizeof(PacketElementAttr)));

    bool *reconciled = static_cast<bool*>(
                        calloc(numConsElements, sizeof(bool)));

    for (uint32_t i = 0U; i < numProdElements; i++) {
        elementAttrs[i].syncMode = prodElementAttrs[i].syncMode;
        elementAttrs[i].userData = prodElementAttrs[i].userData;

        // Reconcile the element attributes of the same userData in producer's
        // and consumer's packet.
        uint32_t attrListCount = 1U;
        NvSciBufAttrList oldBufAttr[2] = { prodElementAttrs[i].bufAttrList,
                                           nullptr };
        for (uint32_t j = 0U; j < numConsElements; j++) {
            if (consElementAttrs[j].userData ==
                prodElementAttrs[i].userData) {
                oldBufAttr[1] = consElementAttrs[j].bufAttrList;
                attrListCount = 2U;

                if (consElementAttrs[j].syncMode ==
                    NvSciStreamElementMode_Immediate) {
                    elementAttrs[i].syncMode = NvSciStreamElementMode_Immediate;
                }
                reconciled[j] = true;
                break;
            }
        }

        // Reconcile buffer attribute list
        NvSciBufAttrList conflictlist = nullptr;
        CHECK_NVSCIERR(NvSciBufAttrListReconcile(oldBufAttr,
                                                 attrListCount,
                                                 &elementAttrs[i].bufAttrList,
                                                 &conflictlist));

        // If there is conflict attributes, dump the conflict list.
        // Free the conflict attribute lists
        if (conflictlist != nullptr) {
            NvSciBufAttrListFree(conflictlist);
            conflictlist = nullptr;
        }
    }
    numElements = numProdElements;

    // Copy the remaining element attributes in consumer's packet
    for (uint32_t i = 0U; i < numConsElements; i++) {
        if (reconciled[i]) {
            continue;
        }

        // Ensure the number of packet elements is supported.
        if (numElements >= static_cast<uint32_t>(maxNumElements)) {
            LOG_ERR_EXIT("Reach the max number of elements per packet.\n");
        }
        elementAttrs[numElements].syncMode = consElementAttrs[i].syncMode;
        elementAttrs[numElements].userData = consElementAttrs[i].userData;
        elementAttrs[numElements].bufAttrList = consElementAttrs[i].bufAttrList;
        consElementAttrs[i].bufAttrList = nullptr;

        numElements++;
    }

    free(reconciled);
    LOG_DEBUG("Pool determined packet layout.\n");
}

// Send the determined packet layout to both producer and consumer.
void PacketPoolManager::sendReconciledPacketAttrs(void)
{
    CHECK_NVSCIERR(NvSciStreamBlockPacketElementCount(handle, numElements));
    LOG_DEBUG("Pool sent number of elements per packet: " << numElements << ".\n");

    for (uint32_t i = 0U; i < numElements; i++) {
        CHECK_NVSCIERR(NvSciStreamBlockPacketAttr(handle,
                                                  i,
                                                  elementAttrs[i].userData,
                                                  elementAttrs[i].syncMode,
                                                  elementAttrs[i].bufAttrList));
        LOG_DEBUG("Pool sent the reconciled attribute of element " << i << " to endpoints.\n");
    }
}

// Create a buffer object for each element.
void PacketPoolManager::allocBuffers(NvSciStreamCookie &cookie)
{
    // Assign cookie for the new packet
    cookie = assignPacketCookie();

    // Get the slot by pool's cookie and allocate space for all elements
    // (buffers) in this packet.
    PoolPacket *packet = getPacketByCookie(cookie);
    if (packet == nullptr) {
        LOG_ERR_EXIT("\nInvalid packet cookie.\n");
    }

    packet->cookie = cookie;
    packet->buffers = static_cast<NvSciBufObj*>(
                        calloc(numElements, sizeof(NvSciBufObj)));

    // Create buffer object for each element.
    for (uint32_t i = 0U; i < numElements; i++) {
        CHECK_NVSCIERR(NvSciBufObjAlloc(elementAttrs[i].bufAttrList,
                                        &packet->buffers[i]));
        LOG_DEBUG("Pool created buffer object of element " << i << ".\n");
    }

    numPackets++;
}

// Send a new packet and all the packet elements to the producer
//  and consumer.
void PacketPoolManager::sendPacket(NvSciStreamCookie cookie)
{
    // Send pool's packet cookie to the NvSciStream, and save the packet handle
    // returned from the NvSciStream.
    PoolPacket *packet = getPacketByCookie(cookie);
    if (packet == nullptr) {
        LOG_ERR_EXIT("\nInvalid packet for cookie.\n");
    }

    CHECK_NVSCIERR(NvSciStreamPoolPacketCreate(handle,
                                               packet->cookie,
                                               &packet->handle));
    LOG_DEBUG("Pool sent packet (cookie = " << packet->cookie << ", handle = " << packet->handle << ") to endpoints.\n");

    // Send each element (buffer) one by one.
    for (uint32_t i = 0U; i < numElements; i++) {
        CHECK_NVSCIERR(NvSciStreamPoolPacketInsertBuffer(handle,
                                                         packet->handle,
                                                         i,
                                                         packet->buffers[i]));
        LOG_DEBUG("Pool sent buffer object of element " << i << " to endpoints.\n");
    }
}

void PacketPoolManager::reconcileAndMapPackets(void)
{
    reconcilePacketAttrs();
    sendReconciledPacketAttrs();

    // Pool allocates buffers and sends packets to producer
    // + consumer
    for (uint32_t i = 0U; i < NUM_PACKETS; i++) {
        NvSciStreamCookie poolCookie = 0U;
        LOG_DEBUG("Pool creates "<< static_cast<int>(i + 1) << " packet(s):\n");
        allocBuffers(poolCookie);
        sendPacket(poolCookie);
    }
}

// Receive the packet mapping status from producer and consumer.
//  Buffer registration is finished.
void PacketPoolManager::recvPacketStatus(NvSciStreamEvent &event)
{
    // process packet status event
    switch(event.type) {
        case NvSciStreamEventType_PacketStatusProducer:
            numProdPacketStatusRecv++;
            if (numProdPacketStatusRecv > NUM_PACKETS) {
                LOG_ERR_EXIT("\nReceived excess packet status from producer: "
                    << numProdPacketStatusRecv << "\n");
            }
            LOG_DEBUG("Pool received PacketStatus "
                "(cookie = " << event.packetCookie << ") from producer,\n"
                "\t\t\ttotal received = " << numProdPacketStatusRecv << ".\n");
            break;
        case NvSciStreamEventType_PacketStatusConsumer:
            numConsPacketStatusRecv++;
            if (numConsPacketStatusRecv > NUM_PACKETS) {
                LOG_ERR_EXIT("\nReceived excess packet status from consumer: "
                    << numConsPacketStatusRecv << "\n");
            }
            LOG_DEBUG("Pool received PacketStatus "
                "(cookie = " << event.packetCookie << ") from consumer,\n"
                "\t\t\ttotal received = " << numConsPacketStatusRecv << ".\n");
            break;
        case NvSciStreamEventType_ElementStatusProducer:
            numProdElementStatusRecv++;
            if (numProdElementStatusRecv > (NUM_PACKETS * numElements)) {
                LOG_ERR_EXIT("\nReceived excess element status from producer: "
                    << numProdElementStatusRecv << "\n");
            }
            LOG_DEBUG("Pool received ElementStatus "
                "(cookie = " << event.packetCookie << ", idx = " << event.index << ") from producer,\n"
                "\t\t\ttotal received = " << numProdElementStatusRecv << ".\n");
            break;
        case NvSciStreamEventType_ElementStatusConsumer:
            numConsElementStatusRecv++;
            if (numConsElementStatusRecv > (NUM_PACKETS * numElements)) {
                LOG_ERR_EXIT("\nReceived excess element status from consumer: "
                    << numConsElementStatusRecv << "\n");
            }
            LOG_DEBUG("Pool received ElementStatus "
                "(cookie = " << event.packetCookie << ", idx = " << event.index << ") from consumer,\n"
                "\t\t\ttotal received = " << numConsElementStatusRecv << ".\n");
            break;
        default:
            break;
    }
}

bool PacketPoolManager::recvAllPacketStatus(void)
{
    const uint32_t expectedElementCount = NUM_PACKETS * numElements;
    return ((numProdPacketStatusRecv == NUM_PACKETS) &&
            (numProdElementStatusRecv == expectedElementCount) &&
            (numConsPacketStatusRecv == NUM_PACKETS) &&
            (numConsElementStatusRecv == expectedElementCount));
}
