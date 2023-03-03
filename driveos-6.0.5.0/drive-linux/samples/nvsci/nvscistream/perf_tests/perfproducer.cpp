//! \file
//! \brief NvSciStream test Producer client declaration.
//!
//! \copyright
//! Copyright (c) 2019-2022 NVIDIA Corporation. All rights reserved.
//!
//! NVIDIA Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from NVIDIA Corporation is strictly prohibited.

#include <unistd.h>
#include <thread>
#include "perfproducer.h"
#include "nvplayfair.h"

extern TestArg          testArg;

extern NvpPerfData_t    prodPacketWait;
extern uint64_t         testStartTime;

PerfProducer::PerfProducer(NvSciBufModule buf,
                           NvSciSyncModule sync) :
    PerfClient(buf, sync)
{
    if (testArg.testType == CrossProcProd) {
        c2cQueue.resize(testArg.numConsumers, 0U);
        ipcSrc.resize(testArg.numConsumers, 0U);
    }
}

PerfProducer::~PerfProducer(void)
{
    // Delete blocks
    if (producer != 0U) {
        NvSciStreamBlockDelete(producer);
        producer = 0U;
    }
    // Wait for pool event handling thread
    if (poolThread.joinable()) {
        poolThread.join();
    }
    if (pool != 0U) {
        NvSciStreamBlockDelete(pool);
        pool = 0U;
    }
    if (multicast != 0U) {
        NvSciStreamBlockDelete(multicast);
        multicast = 0U;
    }
    for (uint32_t i{ 0U }; i < c2cQueue.size(); i++) {
        if (c2cQueue[i] != 0U) {
            NvSciStreamBlockDelete(c2cQueue[i]);
            c2cQueue[i] = 0U;
        }
    }
    for (uint32_t i{ 0U }; i < ipcSrc.size(); i++) {
        if (ipcSrc[i] != 0U) {
            NvSciStreamBlockDelete(ipcSrc[i]);
            ipcSrc[i] = 0U;
        }
    }
}

void PerfProducer::setEndpointBufAttr(NvSciBufAttrList attrList)
{
    NvSciBufType bufType{ NvSciBufType_RawBuffer };
    // Convert buffer size from MB to Bytes
    uint64_t rawsize{ static_cast<uint64_t>(1024U * 1024U * testArg.bufSize) };
    uint64_t align{ 4 * 1024 };
    NvSciBufAttrValAccessPerm perm{ NvSciBufAccessPerm_ReadWrite };
    // Disable cpu access for vidmem
    bool cpuaccess_flag{ testArg.vidmem ? false : true };

    NvSciBufAttrKeyValuePair rawbuffattrs[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufRawBufferAttrKey_Size, &rawsize, sizeof(rawsize) },
        { NvSciBufRawBufferAttrKey_Align, &align, sizeof(align) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag,
            sizeof(cpuaccess_flag) },
    };

    CHECK_NVSCIERR(NvSciBufAttrListSetAttrs(attrList,
                    rawbuffattrs,
                    sizeof(rawbuffattrs) / sizeof(NvSciBufAttrKeyValuePair)));

    // Set vidmem
    if (testArg.vidmem) {
        NvSciRmGpuId gpuid;

        memcpy(&gpuid, &testArg.gpuUUID, sizeof(gpuid));

        NvSciBufAttrKeyValuePair vidmemBuffattrs[] = {
            { NvSciBufGeneralAttrKey_GpuId, &gpuid, sizeof(gpuid) },
            { NvSciBufGeneralAttrKey_VidMem_GpuId, &gpuid, sizeof(gpuid) },
        };
        CHECK_NVSCIERR(NvSciBufAttrListSetAttrs(attrList, vidmemBuffattrs, 2));
    }
}

void PerfProducer::getCpuPtr(Packet& pkt)
{
    for (uint32_t i{ 0U }; i < NUM_ELEMENTS; i++) {
        void* vaPtr{ nullptr };
        CHECK_NVSCIERR(NvSciBufObjGetCpuPtr(pkt.buffers[i], &vaPtr));
        pkt.cpuPtr[i] = static_cast<uint64_t*>(vaPtr);
    }
}

NvSciStreamBlock PerfProducer::createStream(
    std::vector<NvSciIpcEndpoint>* ipcEndpoint)
{
    // Create blocks
    CHECK_NVSCIERR(NvSciStreamStaticPoolCreate(testArg.numPackets, &pool));
    CHECK_NVSCIERR(NvSciStreamProducerCreate(pool, &producer));

    if (testArg.numConsumers > 1U) {
        CHECK_NVSCIERR(NvSciStreamMulticastCreate(testArg.numConsumers,
                                                  &multicast));
    }

    if (testArg.testType == CrossProcProd) {
        for (uint32_t i{ 0U }; i < testArg.numConsumers; i++) {
            if (testArg.isC2c) {
                CHECK_NVSCIERR(NvSciStreamFifoQueueCreate(&c2cQueue[i]));
            }

            CHECK_NVSCIERR(NvSciStreamIpcSrcCreate2((*ipcEndpoint)[i],
                                                    syncModule,
                                                    bufModule,
                                                    c2cQueue[i],
                                                    &ipcSrc[i]));
        }
    }

    // Connect blocks
    NvSciStreamBlock upstreamBlock{ 0U };
    if (testArg.testType == CrossProcProd) {
        if (testArg.numConsumers > 1U) {
            CHECK_NVSCIERR(NvSciStreamBlockConnect(producer, multicast));
            for (uint32_t i{ 0U }; i < testArg.numConsumers; i++) {
                CHECK_NVSCIERR(NvSciStreamBlockConnect(multicast, ipcSrc[i]));
            }
        } else {
            CHECK_NVSCIERR(NvSciStreamBlockConnect(producer, ipcSrc[0]));
        }
    } else {
        // Set the upstream block, which will be connected to
        //  the consumer chain in the single-process use case.
        if (testArg.numConsumers > 1U) {
            CHECK_NVSCIERR(NvSciStreamBlockConnect(producer, multicast));
            upstreamBlock = multicast;
        } else {
            upstreamBlock = producer;
        }
    }

    endpointHandle = producer;

    return upstreamBlock;
}

void PerfProducer::run(void)
{
    // Spawn a new thread to handle pool's events
    poolThread = std::thread(&PerfClient::runPoolEventsHandler, this, pool);

    // Handle producer's events
    handleEvents();
}

void PerfProducer::handlePayload(void)
{
    // Skip if all payloads are sent.
    if (testArg.numFrames == numPayloads) {
        return;
    }

    if (testArg.latency) {
        // Record the time that PacketReady event arrived and
        //  save timestamp data
        packetWaitEndTime = NvpGetTimeMark();
        NVP_CHECKERR_EXIT(NvpRecordSample(&prodPacketWait,
                                          packetWaitStartTime,
                                          packetWaitEndTime));
    }

    // Get a packet
    NvSciStreamCookie cookie{ 0U };
    CHECK_NVSCIERR(NvSciStreamProducerPacketGet(endpointHandle,
                                                &cookie));

    // Record time the producer receives the packet from consumer.
    //  Skip if the packet it's used at its first time,
    //  which is not released from the consumer.
    uint64_t releaseEndTime{ 0U };
    if (testArg.latency && (numPayloads >= numRecvPackets)) {
        releaseEndTime = NvpGetTimeMark();
    }

    if (numPayloads == 0) {
        // Set frame rate and mark the start of the first payload processing.
        NVP_CHECKERR_EXIT(NvpRateLimitInit(&rateLimitInfo,
                                           testArg.fps));
        NVP_CHECKERR_EXIT(NvpMarkPeriodicExecStart(&rateLimitInfo));
    } else {
        // Wait for some time before processing the next payload
        //  to simulate the specified producer-present rate.
        NVP_CHECKERR_EXIT(NvpRateLimitWait(&rateLimitInfo));
    }

    uint32_t const id{ static_cast<uint32_t>(cookie) - 1U };
    CHECK_ERR(id < numRecvPackets, "Invalid cookie");
    Packet* packet{ &packets[id] };

    // Wait for prefences from consumer
    for (uint32_t j{ 0U }; j <testArg.numSyncs; j++) {
        for (uint32_t i{ 0U }; i <testArg.numConsumers; i++) {
            NvSciSyncFence prefence;
            CHECK_NVSCIERR(NvSciStreamBlockPacketFenceGet(endpointHandle,
                                                          packet->handle,
                                                          i,
                                                          j,
                                                          &prefence));
            CHECK_NVSCIERR(NvSciSyncFenceWait(&prefence,
                                              waitContext,
                                              FENCE_WAIT_INFINITE));
            NvSciSyncFenceClear(&prefence);
        }
    }

    // Write timestamps to buffer
    if (testArg.latency) {
        // Start the present timer
        uint64_t *timerPtr{ packet->cpuPtr[0] };
        uint64_t presentStartTime{ NvpGetTimeMark() };

        // Write the present start time and the release stop time
        // to the first element
        timerPtr[0] = presentStartTime;
        timerPtr[1] = releaseEndTime;
    }

    // Generate postfences
    for (uint32_t i{ 0U }; i < testArg.numSyncs; i++) {
        postfences[i] = NvSciSyncFenceInitializer;
        if (syncs[i] != nullptr) {
            CHECK_NVSCIERR(NvSciSyncObjGenerateFence(syncs[i],
                                                     &postfences[i]));
            CHECK_NVSCIERR(NvSciStreamBlockPacketFenceSet(endpointHandle,
                                                          packet->handle,
                                                          i,
                                                          &postfences[i]));
        }
    }

    // Present the packet to the consumer
    CHECK_NVSCIERR(NvSciStreamProducerPacketPresent(endpointHandle,
                                                    packet->handle));
    // Signal postfences
    for (uint32_t i{ 0U }; i < testArg.numSyncs; i++) {
        if (syncs[i] != nullptr) {
            CHECK_NVSCIERR(NvSciSyncObjSignal(syncs[i]));
            NvSciSyncFenceClear(&postfences[i]);
        }
    }

    ++numPayloads;
}

void PerfProducer::handleStreamComplete(void)
{
    // Record streaming end time
    streamingEndTime = NvpGetTimeMark();

    printf("\nProducer presented %d packet(s)\n", numPayloads);

    printf("Producer total init time:                    %8.5f us\n",
           getTimeInUs(streamingStartTime - testStartTime));
    printf("Producer setup phase (after stream connect): %8.5f us\n",
           getTimeInUs(streamingStartTime - setupStartTime));
    printf("Producer streaming phase:                    %8.5f us\n\n",
           getTimeInUs(streamingEndTime - streamingStartTime));

    // Mark streaming done
    streamingDone = true;
}
