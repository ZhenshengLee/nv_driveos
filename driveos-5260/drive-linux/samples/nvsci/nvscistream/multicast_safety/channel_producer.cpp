//
// NvSciStream Producer channel usage defintion.
//
// Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "channel_producer.h"
#include "packet_pool_manager.h"
#include "log.h"

ProducerChannel::ProducerChannel(
    NvSciBufModule bufMod,
    NvSciSyncModule syncMod,
    NvSciIpcEndpoint endpoint[MAX_CONSUMERS],
    uint32_t nConsumer) :
    bufModule(bufMod),
    syncModule(syncMod),
    consumerCount(nConsumer)
{
    for (uint32_t n = 0U; n < consumerCount; n++) {
        ipcEndpoint[n] = endpoint[n];
    }
}

ProducerChannel::~ProducerChannel()
{
    LOG_DEBUG("Closing Producer channel");

    // Delete blocks
    if (producerHandle != 0U) {
        NvSciStreamBlockDelete(producerHandle);
    }
    if (poolHandle != 0U) {
        NvSciStreamBlockDelete(poolHandle);
    }
    if (multicast != 0U) {
        NvSciStreamBlockDelete(multicast);
    }
    for (uint32_t n=0U; n < consumerCount; n++) {
        if (ipcSrcHandle[n] != 0U) {
            NvSciStreamBlockDelete(ipcSrcHandle[n]);
        }
    }

    if (producer) {
        delete producer;
    }
}

void ProducerChannel::createBlocks(void)
{
    // Create pool
    CHECK_NVSCIERR(NvSciStreamStaticPoolCreate(NUM_PACKETS, &poolHandle));
    // Create producer block associated with above pool
    CHECK_NVSCIERR(NvSciStreamProducerCreate(poolHandle, &producerHandle));
    // Create multicast block
    if (consumerCount > 1U) {
        CHECK_NVSCIERR(NvSciStreamMulticastCreate(consumerCount, &multicast));
    }
    // Create ipc source block
    // Multi-Process consumer needs to create multiple ipcSrcEndpoint
    for(uint32_t n = 0U; n < consumerCount; n++) {
        NvSciStreamIpcSrcCreate(ipcEndpoint[n],
                                syncModule,
                                bufModule,
                                &ipcSrcHandle[n]);
    }

    // Create producer client
    producer = new CudaProducer(producerHandle);

    LOG_DEBUG("Producer channel created");
}

void ProducerChannel::connectToStream(void)
{
    LOG_DEBUG("\n\n========================= Producer Connection setup =======================\n");

    // Connect blocks
    LOG_DEBUG("Connecting over Ipc... waiting for consumer connection\n");
    if (consumerCount == 1U) {
        LOG_DEBUG("Single Cast Case!!!\n");
        CHECK_NVSCIERR(NvSciStreamBlockConnect(producerHandle, ipcSrcHandle[0]));
    } else {
        LOG_DEBUG("Multi-Cast Case!!!\n");
        CHECK_NVSCIERR(NvSciStreamBlockConnect(producerHandle, multicast));

        for (uint32_t n=0U; n < consumerCount; n++) {
            CHECK_NVSCIERR(NvSciStreamBlockConnect(multicast, ipcSrcHandle[n]));
        }
    }

    NvSciStreamEvent event;
    for (uint32_t n = 0U; n < consumerCount; n++) {
        CHECK_NVSCIERR(NvSciStreamBlockEventQuery(ipcSrcHandle[n], QUERY_TIMEOUT_FOREVER, &event));
        if (event.type != NvSciStreamEventType_Connected) {
            LOG_ERR_EXIT("\nipcSrc failed to receive Connected event\n");
        }
    }

    if (consumerCount > 1U) {
        CHECK_NVSCIERR(NvSciStreamBlockEventQuery(multicast, QUERY_TIMEOUT_FOREVER, &event));
        if (event.type != NvSciStreamEventType_Connected) {
            LOG_ERR_EXIT("\nmulticast failed to receive Connected event\n");
        }
    }

    CHECK_NVSCIERR(NvSciStreamBlockEventQuery(poolHandle, QUERY_TIMEOUT_FOREVER, &event));
    if (event.type != NvSciStreamEventType_Connected) {
        LOG_ERR_EXIT("\npool failed to receive Connected event\n");
    }

    CHECK_NVSCIERR(NvSciStreamBlockEventQuery(producerHandle, QUERY_TIMEOUT_FOREVER, &event));
    if (event.type != NvSciStreamEventType_Connected) {
        LOG_ERR_EXIT("\nproducer failed to receive Connected event\n");
    }

    LOG_DEBUG("Producer channel connected");
    LOG_DEBUG("=============================================================================");
}

void ProducerChannel::setupForStreaming(void)
{
    LOG_DEBUG("\n======================== Producer Sync requirement setup ====================\n");
    producer->sendSyncAttr(syncModule);
    LOG_DEBUG("Producer sent sync attribute to consumer.\n");
    LOG_DEBUG("=============================================================================");

    LOG_DEBUG("\n====================== Producer Packet requirement setup ====================\n");
    producer->sendPacketAttr(bufModule);
    LOG_DEBUG("Producer sent packet requirements to the pool.\n");
    LOG_DEBUG("=============================================================================");

    LOG_DEBUG("\n============ Producer finalize requirements and map resources ===============\n");
    finalizeSetup();
    LOG_DEBUG("=================================================================================");
}

void ProducerChannel::runStream(void)
{
    LOG_DEBUG("\n\n======================= Producer enters Streaming phase ===================\n");
    for (uint32_t i = 0U; i < NUM_FRAMES; i++) {
        // Producer acquires a new packet
        LOG_DEBUG("\nProducer presents " << static_cast<uint32_t>(i + 1U) << " packet(s):");
        NvSciStreamCookie cookie = 0U;
        producer->getPacket(cookie);
        producer->processPayload(cookie);
        producer->sendPacket(cookie);
    }
    LOG_DEBUG("\n\n========================== Producer Streaming phase End ===================\n");
}

static void* pollPool(void *arg)
{
    // Create pool manager
    NvSciStreamBlock poolHandle = *(static_cast<NvSciStreamBlock*>(arg));
    PacketPoolManager pool(poolHandle);

    bool poolRecvdPacketAttrs = false;
    bool poolRecvdPacketStatus = false;
    uint32_t timeouts = 0U;

    // Receive pool events
    while (!poolRecvdPacketAttrs || !poolRecvdPacketStatus) {
        NvSciStreamEvent event;
        NvSciError err = NvSciStreamBlockEventQuery(poolHandle,
                                                    QUERY_TIMEOUT,
                                                    &event);
        if (err == NvSciError_Success) {
            // Process PacketAttr events
            if (!poolRecvdPacketAttrs) {
                pool.recvPacketAttrs(event);
                // If pool has received all producer's and consumers'
                // packet attributes, it can create packets
                // and send them to producer and consumers.
                if (pool.recvAllPacketAttrs()) {
                    LOG_DEBUG("Pool received all packet attribute.");
                    pool.reconcileAndMapPackets();
                    poolRecvdPacketAttrs = true;
                }
            }

            // Process PacketStatus events
            if (!poolRecvdPacketStatus) {
                pool.recvPacketStatus(event);
                // If pool has received all producer's and consumers'
                // packet status, it can move onto to streaming.
                if (pool.recvAllPacketStatus()) {
                    LOG_DEBUG("Pool received all packet status.");
                    poolRecvdPacketStatus = true;
                }
            }
        } else if (err == NvSciError_Timeout) {
            if (++timeouts == MAX_QUERY_TIMEOUTS) {
                LOG_ERR_EXIT("\nPool Query waits seem to be taking forever!\n");
            }
        } else {
            LOG_ERR("\nPool NvSciStreamBlockEventQuery Failed:");
            CHECK_NVSCIERR(err);
        }
    }
    return nullptr;
}

void ProducerChannel::finalizeSetup(void)
{
    // Creaet a thread for pool event tracking
    pthread_t poolTid;
    if (0 != pthread_create(&poolTid, nullptr, pollPool, &poolHandle)) {
        LOG_ERR_EXIT("Failed to create pool thread.");
    }

    // Producer event tracking
    uint32_t timeouts = 0U;
    while (!producer->allPacketInfoReceived() ||
           !producer->allSyncInfoReceived()) {
        // Receive producer events
        NvSciStreamEvent event;
        NvSciError err = NvSciStreamBlockEventQuery(producerHandle,
                                                    QUERY_TIMEOUT,
                                                    &event);
        if (err == NvSciError_Success) {
            // process event
            producer->handleResourceSetupEvents(event);
        } else if (err == NvSciError_Timeout) {
            if (++timeouts == MAX_QUERY_TIMEOUTS) {
                LOG_ERR_EXIT("\nProducer Query waits seem to be taking forever!\n");
            }
        } else {
            LOG_ERR("\nProducer NvSciStreamBlockEventQuery Failed:");
            CHECK_NVSCIERR(err);
        }
    }

    pthread_join(poolTid, nullptr);
}
