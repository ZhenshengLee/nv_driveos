//
// NvSciStream Consumer channel usage defintion.
//
// Copyright (c) 2019-2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "channel_consumer.h"
#include "log.h"
#include "util.h"

namespace NvScistreamSample
{
    ConsumerChannel::ConsumerChannel(NvSciBufModule bufMod,
                                     NvSciSyncModule syncMod,
                                     NvSciIpcEndpoint endpoint) :
                                     consumerHandle(0U),
                                     queueHandle(0U),
                                     bufModule(bufMod),
                                     syncModule(syncMod),
                                     ipcEndpoint(endpoint)
    {
    }


    ConsumerChannel::~ConsumerChannel()
    {
        LOG_DEBUG("Closing Consumer channel");

        // Delete blocks
        if (consumerHandle != 0U) {
            NvSciStreamBlockDelete(consumerHandle);
        }
        if (queueHandle != 0U) {
            NvSciStreamBlockDelete(queueHandle);
        }

        if (ipcDstHandle != 0U) {
            NvSciStreamBlockDelete(ipcDstHandle);
        }

        if (consumer) {
            delete consumer;
        }
    }

    void ConsumerChannel::createBlocks(QueueType qType)
    {
        queueType = qType;

        // Create Queue
        if (queueType == QueueType_Mailbox) {
            CHECK_NVSCIERR(NvSciStreamMailboxQueueCreate(&queueHandle));
        } else {
            CHECK_NVSCIERR(NvSciStreamFifoQueueCreate(&queueHandle));
        }
        // Create consumer associated with above queue
        CHECK_NVSCIERR(NvSciStreamConsumerCreate(queueHandle, &consumerHandle));
        // Create ipc dst block
        NvSciStreamIpcDstCreate(ipcEndpoint, syncModule, bufModule, &ipcDstHandle);

        // Create consumer client
        consumer = new CudaConsumer(consumerHandle);

        LOG_DEBUG("Consumer channel created");

        // Global stream queries
        consumer->validateNumSyncObjs();
    }

    void ConsumerChannel::connectToStream(void)
    {
        NvSciStreamEvent event;
        LOG_DEBUG("\n\n ========================= Consumer Connection setup =========================\n");
        // Connect blocks
        LOG_DEBUG("Connecting over Ipc... waiting for producer connection");
        CHECK_NVSCIERR(NvSciStreamBlockConnect(ipcDstHandle, consumerHandle));

        CHECK_NVSCIERR(NvSciStreamBlockEventQuery(ipcDstHandle, QUERY_TIMEOUT_FOREVER, &event));
        if (event.type != NvSciStreamEventType_Connected) {
            LOG_ERR_EXIT("\nipcDst failed to receive Connected event\n");
        }

        CHECK_NVSCIERR(NvSciStreamBlockEventQuery(queueHandle, QUERY_TIMEOUT_FOREVER, &event));
        if (event.type != NvSciStreamEventType_Connected) {
            LOG_ERR_EXIT("\nqueue failed to receive Connected event\n");
        }

        CHECK_NVSCIERR(NvSciStreamBlockEventQuery(consumerHandle, QUERY_TIMEOUT_FOREVER, &event));
        if (event.type != NvSciStreamEventType_Connected) {
            LOG_ERR_EXIT("\nconsumer failed to receive Connected event\n");
        }

        LOG_DEBUG("Consumer channel connected\n");
    }

    void ConsumerChannel::setupForStreaming(void)
    {
        LOG_DEBUG("\n\n ========================= Consumer Sync requirement setup =========================\n");
        // Create and send sync attributes
        LOG_DEBUG("Consumer creates NvSciSync attributes:\n");
        consumer->createSyncAttrLists(syncModule);
        LOG_DEBUG("Consumer sends sync attribute to producer:\n");
        consumer->sendSyncAttr();
        LOG_DEBUG("========================================= ==================================================");

        // Create and send Packet requirements
        LOG_DEBUG("\n\n ========================= Consumer Packet requirement setup =========================\n");
        LOG_DEBUG("Consumer creates NvSciBuf attributes:\n");
        consumer->createBufAttrList(bufModule);
        LOG_DEBUG("Consumer sends packet requirements to the pool:\n");
        consumer->sendPacketAttr();
        LOG_DEBUG("========================================= ==================================================");

        LOG_DEBUG("\n\n ========================= Consumer finalize requirements and map resources =========================\n");
        NvSciStreamEvent event;
        uint32_t timeouts = 0;

        // Receive producer and pool reconciled requirements
        while ((!consumer->allPacketInfoReceived()) ||
               (!consumer->allSyncInfoReceived()) ) {
            NvSciError err = NvSciStreamBlockEventQuery(consumerHandle, QUERY_TIMEOUT, &event);
            if (err == NvSciError_Success) {
                // reset timeout counter
                timeouts = 0;
                // handle event
                consumer->handleResourceSetupEvents(event);
            } else if (err == NvSciError_Timeout) {
                // if query timeouts - keep waiting for event
                // until wait threshold is reached
                if (timeouts < MAX_QUERY_TIMEOUTS) {
                    timeouts++;
                } else {
                    LOG_ERR_EXIT("\nConsumer Query waits seem to be taking forever!\n");
                }
            } else {
                LOG_ERR("\nNvSciStreamBlockEventQuery Failed:");
                CHECK_NVSCIERR(err);
            }
        }

        // map resources
        LOG_DEBUG("Consumer maps sync objects:\n");
        consumer->mapSyncObjs();
        LOG_DEBUG("Consumer initializes NvSciSyncFence:\n");
        consumer->initFences(syncModule);
        // For each packet received,
        // use the corresponding cookie to map buffers
        LOG_DEBUG("Consumer maps buffer objects:\n");
        for (uint32_t i = 0; i < NUM_PACKETS; i++)
        {
            NvSciStreamCookie cookie = ClientCommon::getCookieAtIndex(i);
            consumer->mapBuffers(cookie);
            consumer->registerPacketElements(cookie);
        }

        LOG_DEBUG("========================================= ==================================================");
    }

    void ConsumerChannel::runStream(void)
    {
        LOG_DEBUG("\n\n ========================= Consumer enters Streaming phase =========================\n");
        uint32_t numPackets = 0U;
        NvSciStreamCookie cookie = 0U;
        while (consumer->acquirePacket(cookie)) {
            LOG_DEBUG("Consumer acquired " << std::dec << (numPackets + 1) << " packet(s).");
            numPackets++;
            consumer->processPayload(cookie);
            consumer->releasePacket(cookie);
        }
        // Fifo stream should receive all packets
        if ((queueType == QueueType_FIFO) && (numPackets != NUM_FRAMES)) {
            LOG_ERR_EXIT("Consumer didn't receive all packets from producer.");
        }
        LOG_DEBUG("\n\n ========================= Consumer Streaming phase End =========================\n");
    }
}
