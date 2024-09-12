//
// NvSciStream Producer channel usage defintion.
//
// Copyright (c) 2019-2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "channel_producer.h"
#include "log.h"

namespace NvScistreamSample
{
    ProducerChannel::ProducerChannel(NvSciBufModule bufMod,
                                     NvSciSyncModule syncMod,
                                     NvSciIpcEndpoint endpoint) :
                                     producerHandle(0U),
                                     poolHandle(0U),
                                     ipcSrcHandle(0U),
                                     bufModule(bufMod),
                                     syncModule(syncMod),
                                     ipcEndpoint(endpoint),
                                     producer(nullptr),
                                     pool(nullptr)
    {
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
        if (ipcSrcHandle != 0U) {
            NvSciStreamBlockDelete(ipcSrcHandle);
        }

        if (producer) {
            delete producer;
        }

        if (pool) {
            delete pool;
        }
    }

    void ProducerChannel::createBlocks(void)
    {
        // Create pool
        CHECK_NVSCIERR(NvSciStreamStaticPoolCreate(NUM_PACKETS, &poolHandle));
        // Create producer block associated with above pool
        CHECK_NVSCIERR(NvSciStreamProducerCreate(poolHandle, &producerHandle));
        // Create ipc source block
        NvSciStreamIpcSrcCreate(ipcEndpoint, syncModule, bufModule, &ipcSrcHandle);

        // Create producer client
        producer = new NvMediaProducer(producerHandle);
        // Create pool manager
        pool = new PacketPoolManager(poolHandle);

        LOG_DEBUG("Producer channel created");

        // Global stream queries
        producer->validateNumSyncObjs();
    }

    void ProducerChannel::connectToStream(void)
    {
        LOG_DEBUG("\n\n ========================= Producer Connection setup =========================\n");
        NvSciStreamEvent event;
        // Connect blocks
        LOG_DEBUG("Connecting over Ipc... waiting for consumer connection\n");
        CHECK_NVSCIERR(NvSciStreamBlockConnect(producerHandle, ipcSrcHandle));

        CHECK_NVSCIERR(NvSciStreamBlockEventQuery(ipcSrcHandle, QUERY_TIMEOUT_FOREVER, &event));
        if (event.type != NvSciStreamEventType_Connected) {
            LOG_ERR_EXIT("\nipcSrc failed to receive Connected event\n");
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
        LOG_DEBUG("========================================= ==================================================");
    }

    void ProducerChannel::setupPacketRequirements(void)
    {
        // Receive producer / pool requirements
        NvSciStreamEvent event;
        uint32_t timeouts = 0;
        NvSciError err = NvSciError_Success;

        // prod event tracking
        bool pollProd = true;

        // pool event tracking
        bool poolRecvdPacketAttrs = false;
        bool poolRecvdPacketStatus = false;
        bool pollPool = true;

        while (pollProd || pollPool) {
            bool didTimeout = false;

            if (pollProd) {
                // Receive producer events
                err = NvSciStreamBlockEventQuery(producerHandle, QUERY_TIMEOUT, &event);
                if (err == NvSciError_Success) {
                    // reset timeout counter
                    timeouts = 0;
                    // process event
                    producer->handleResourceSetupEvents(event);
                    // Keep listening till prod can move onto
                    // buffer mapping and packet registration
                    if (producer->allPacketInfoReceived() &&
                        producer->allSyncInfoReceived()) {
                        // Sync mapping
                        LOG_DEBUG("Producer receives consumer's sync object:\n");
                        producer->mapSyncObjs();
                        LOG_DEBUG("Producer initializes NvSciSyncFence:\n");
                        producer->initFences(syncModule);
                        // For each packet received,
                        // use the corresponding cookie to map buffers
                        LOG_DEBUG("Producer maps buffer objects:\n");
                        for (uint32_t i = 0; i < NUM_PACKETS; i++)
                        {
                            NvSciStreamCookie cookie = ClientCommon::getCookieAtIndex(i);
                            producer->mapBuffers(cookie);
                            producer->registerPacketElements(cookie);
                        }
                        pollProd = false;
                    }
                } else if (err == NvSciError_Timeout) {
                    didTimeout |= true;
                } else {
                    LOG_ERR("\nProducer NvSciStreamBlockEventQuery Failed:");
                    CHECK_NVSCIERR(err);
                }
            }

            // Receive pool events
            if (pollPool) {
                err = NvSciStreamBlockEventQuery(poolHandle, QUERY_TIMEOUT, &event);
                if (err == NvSciError_Success) {
                    // reset timeout counter
                    timeouts = 0;

                    // Process PacketAttr events
                    if (!poolRecvdPacketAttrs) {
                        pool->recvPacketAttrs(event);
                        // if pool has received all prod + cons
                        // attributes, it can create packets
                        // and send them to prod + cons
                        if (pool->recvAllPacketAttrs()) {
                            LOG_DEBUG("Pool received all prod / cons packet Attrs");
                            pool->reconcileAndMapPackets();
                            poolRecvdPacketAttrs = true;
                        }
                    }

                    // Process PacketStatus events
                    if (!poolRecvdPacketStatus) {
                        pool->recvPacketStatus(event);
                        // if pool has received all prod + cons
                        // packet status, we can move onto
                        // streaming
                        if (pool->recvAllPacketStatus()) {
                            LOG_DEBUG("Pool received all prod / cons packet Status");
                            poolRecvdPacketStatus = true;
                        }
                    }

                    // stop polling pool for attr and status events
                    // once they're all received
                    if (poolRecvdPacketAttrs && poolRecvdPacketStatus) {
                        pollPool = false;
                    }
                } else if (err == NvSciError_Timeout) {
                    didTimeout |= true;
                } else {
                    LOG_ERR("\nPool NvSciStreamBlockEventQuery Failed:");
                    CHECK_NVSCIERR(err);
                }
            }

            // Handle Timeouts
            if (didTimeout) {
                // if query timeouts - keep waiting for event
                // until wait threshold is reached
                if (timeouts < MAX_QUERY_TIMEOUTS) {
                    timeouts++;
                } else {
                    LOG_ERR_EXIT("\nProducer Channel Query waits seem to be taking forever!\n");
                }
            }
        }
    }

    void ProducerChannel::setupForStreaming(void)
    {
        LOG_DEBUG("\n\n ========================= Producer Sync requirement setup =========================\n");
        // Create and send sync attributes
        LOG_DEBUG("Producer creates NvSciSync attributes:\n");
        producer->createSyncAttrLists(syncModule);
        LOG_DEBUG("Producer sends sync attribute to producer:\n");
        producer->sendSyncAttr();
        LOG_DEBUG("========================================= ==================================================");

        // Create and send Packet requirements
        LOG_DEBUG("\n\n ========================= Producer Packet requirement setup =========================\n");
        LOG_DEBUG("Producer creates NvSciBuf attributes:\n");
        producer->createBufAttrList(bufModule);
        LOG_DEBUG("Producer sends packet requirements to the pool:\n");
        producer->sendPacketAttr();
        LOG_DEBUG("========================================= ==================================================");

        LOG_DEBUG("\n\n ========================= Producer finalize requirements and map resources =========================\n");
        LOG_DEBUG("Pool + Producer receive packet and resource requirements.");
        LOG_DEBUG("and process them once all are received\n");
        setupPacketRequirements();
        LOG_DEBUG("========================================= ==================================================");
    }

    void ProducerChannel::runStream(void)
    {
        LOG_DEBUG("\n\n ========================= Producer enters Streaming phase =========================\n");
        for (uint32_t i = 0U; i < NUM_FRAMES; i++) {
            // Producer acquires a new packet
            LOG_DEBUG("Producer acquires a packet " << i << ":\n");
            NvSciStreamCookie cookie = 0U;
            producer->getPacket(cookie);
            // process payload and send it out again
            producer->processPayload(cookie);
            producer->sendPacket(cookie);
        }
        LOG_DEBUG("\n\n ========================= Producer Streaming phase End =========================\n");
    }
}