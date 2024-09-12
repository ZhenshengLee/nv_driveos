//
// Pool manager client declaration definition.
//
// Copyright (c) 2019 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "packet_pool_manager.h"

namespace NvScistreamSample
{
    PacketPoolManager::PacketPoolManager(NvSciStreamBlock blockHandle):
        handle(blockHandle),
        numProdElements(0U),
        numProdElementsRecv(0U),
        prodElementAttrs(nullptr),
        numConsElements(0U),
        numConsElementsRecv(0U),
        consElementAttrs(nullptr),
        numElements(0U),
        elementAttrs(nullptr),
        numPackets(NUM_PACKETS),
        numPacketsCreated(0U),
        numProdPacketStatusRecv(0U),
        numConsPacketStatusRecv(0U),
        numProdElementStatusRecv(0U),
        numConsElementStatusRecv(0U)
    {
    }

    PacketPoolManager::~PacketPoolManager(void)
    {
        freeElementAttrs(prodElementAttrs, numProdElements);
        freeElementAttrs(consElementAttrs, numConsElements);
        freeElementAttrs(elementAttrs, numElements);

        for (uint32_t i = 0U; i < numPackets; i++) {
            for (uint32_t j = 0; j < numElements; j++) {
                NvSciBufObjFree(packets[i].buffers[j]);
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
                LOG_DEBUG("Pool Receives number of elements per packet from producer: " << event.count << ".\n");

                // Allocate space for producer packet element attr list
                prodElementAttrs = static_cast<PacketElementAttr*>(
                    calloc(numProdElements, sizeof(PacketElementAttr)));

                break;

            case NvSciStreamEventType_PacketElementCountConsumer:
                numConsElements = event.count;
                LOG_DEBUG("Pool Receives number of elements per packet from consumer: " << event.count << ".\n");

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
                LOG_DEBUG("Pool Receives packet capabilities of element " << index << " from producer.\n");
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
                LOG_DEBUG("Pool Receives packet requirements of element " << index << " from consumer.\n");
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
        CHECK_NVSCIERR(NvSciStreamAttributeQuery(
            NvSciStreamQueryableAttrib_MaxElements,
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
                        elementAttrs[i].syncMode =
                            NvSciStreamElementMode_Immediate;
                    }
                    reconciled[j] = true;
                    break;
                }
            }

            // Reconcile buffer attribute list
            NvSciBufAttrList conflictlist = nullptr;
            CHECK_NVSCIERR(NvSciBufAttrListReconcile(
                    oldBufAttr,
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
        LOG_DEBUG("Determine packet layout.\n");
    }

    // Send the determined packet layout to both producer and consumer.
    void PacketPoolManager::sendReconciledPacketAttrs(void)
    {
        // The default number of elements per packet is 1. If the number of
        // elements per packet is not 1, need to notify the other end.
        if (numElements != 1U) {
            CHECK_NVSCIERR(NvSciStreamBlockPacketElementCount(handle, numElements));
            LOG_DEBUG("Send the determined number of packet elements " << numElements << ".\n");
        }

        for (uint32_t i = 0U; i < numElements; i++) {
            NvSciError err = NvSciStreamBlockPacketAttr(handle,
                i,
                elementAttrs[i].userData,
                elementAttrs[i].syncMode,
                elementAttrs[i].bufAttrList);
            LOG_DEBUG("Send the reconciled attribute list of element " << i << " to producer and consumer.\n");
            CHECK_NVSCIERR(err);
        }
    }

    // Create a buffer object for each element.
    void PacketPoolManager::allocBuffers(NvSciStreamCookie &cookie)
    {
        // Assign cookie for the new packet
        cookie = assignPacketCookie();

        // Get the slot by pool's cookie and allocate space for all elements
        // (buffers) in this packet.
        Packet *packet = getPacketByCookie(cookie);
        if (packet == nullptr) {
            LOG_ERR_EXIT("\nInvalid packet for cookie.\n");
        }

        packet->cookie = cookie;
        packet->buffers = static_cast<NvSciBufObj*>(
            calloc(numElements, sizeof(NvSciBufObj)));

        // Create buffer object for each element.
        for (uint32_t i = 0U; i < numElements; i++) {
            CHECK_NVSCIERR(NvSciBufObjAlloc(
                    elementAttrs[i].bufAttrList,
                    &packet->buffers[i]));
            LOG_DEBUG("Create buffer object of element " << i << " with the reconciled attribute list.\n");
        }

        numPacketsCreated++;
    }

    // Send a new packet and all the packet elements to the producer
    //  and consumer.
    void PacketPoolManager::sendPacket(NvSciStreamCookie cookie)
    {
        // Send pool's packet cookie to the NvSciStream, and save the packet handle
        // returned from the NvSciStream.
        Packet *packet = getPacketByCookie(cookie);
        if (packet == nullptr) {
            LOG_ERR_EXIT("\nInvalid packet for cookie.\n");
        }

        CHECK_NVSCIERR(NvSciStreamPoolPacketCreate(
                handle,
                packet->cookie,
                &packet->handle));
        LOG_DEBUG("Assign pool cookie " << packet->cookie << " to the packet, and\n"
                "\t\t\tsend this new packet (handle = " << std::hex << packet->handle << ") to producer and consumer.\n");

        // Send each element (buffer) one by one.
        for (uint32_t i = 0U; i < numElements; i++) {
            CHECK_NVSCIERR(NvSciStreamPoolPacketInsertBuffer(
                    handle,
                    packet->handle,
                    i,
                    packet->buffers[i]));
            LOG_DEBUG("Send buffer object of element " << i << " to producer and consumer.\n");
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
            LOG_DEBUG("Pool creates a new packet " << i << ":\n");
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
                    LOG_ERR_EXIT("\nReceived excess packet status from producer: " << numProdPacketStatusRecv << "\n");
                }
                LOG_DEBUG("Received PacketStatus (cookie = " << event.packetCookie << ")"
                            " from producer, total received = " << numProdPacketStatusRecv << ".\n");
                break;
            case NvSciStreamEventType_PacketStatusConsumer:
                numConsPacketStatusRecv++;
                if (numConsPacketStatusRecv > NUM_PACKETS) {
                    LOG_ERR_EXIT("\nReceived excess packet status from producer: " << numProdPacketStatusRecv << "\n");
                }
                LOG_DEBUG("Received PacketStatus (cookie = " << event.packetCookie << ")"
                            " from consumer, total received = " << numConsPacketStatusRecv << ".\n");
                break;
            case NvSciStreamEventType_ElementStatusProducer:
                numProdElementStatusRecv++;
                if (numProdElementStatusRecv > (NUM_PACKETS * numElements)) {
                    LOG_ERR_EXIT("\nReceived excess element status from producer: " << numProdElementStatusRecv << "\n");
                }
                LOG_DEBUG("Received ElementStatus (cookie = " << event.packetCookie << ", idx = " << event.index << ")"
                            " from producer, total received = " << numProdElementStatusRecv << ".\n");
                break;
            case NvSciStreamEventType_ElementStatusConsumer:
                numConsElementStatusRecv++;
                if (numConsElementStatusRecv > (NUM_PACKETS * numElements)) {
                    LOG_ERR_EXIT("\nReceived excess element status from consumer: " << numConsElementStatusRecv << "\n");
                }
                LOG_DEBUG("Received ElementStatus (cookie = " << event.packetCookie << ", idx = " << event.index << ")"
                            " from consumer, total received = " << numConsElementStatusRecv << ".\n");
                break;
            default:
                break;
        }
    }

    bool PacketPoolManager::recvAllPacketStatus(void)
    {
        const uint32_t expectedElementCount = NUM_PACKETS * numElements;
        return ((numProdPacketStatusRecv >= NUM_PACKETS) &&
                (numProdElementStatusRecv >= expectedElementCount) &&
                (numConsPacketStatusRecv >= NUM_PACKETS) &&
                (numConsElementStatusRecv >= expectedElementCount));
    }
}
