//! \file
//! \brief NvSciStream test class Consumer client declaration.
//!
//! \copyright
//! SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//! SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//!
//! NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
//! property and proprietary rights in and to this material, related
//! documentation and any modifications thereto. Any use, reproduction,
//! disclosure or distribution of this material and related documentation
//! without an express license agreement from NVIDIA CORPORATION or
//! its affiliates is strictly prohibited.

#include <thread>
#include "perfconsumer.h"

extern TestArg testArg;

extern NvpPerfData_t    presentLatency;
extern NvpPerfData_t    presentLatency2;
extern NvpPerfData_t    releaseLatency;
extern NvpPerfData_t    consPacketWait;

extern uint64_t testStartTime;

PerfConsumer::PerfConsumer(NvSciBufModule buf,
                           NvSciSyncModule sync):
    PerfClient(buf, sync)
{
}

PerfConsumer::~PerfConsumer(void)
{
    // Delete blocks
    if (consumer != 0U) {
        NvSciStreamBlockDelete(consumer);
        consumer = 0U;
    }
    if (queue != 0U) {
        NvSciStreamBlockDelete(queue);
        queue = 0U;
    }
    // Wait for c2c pool event handling thread
    //  for inter-chip stream
    if (poolThread.joinable()) {
        poolThread.join();
    }
    if (c2cPool != 0U) {
        NvSciStreamBlockDelete(c2cPool);
        c2cPool = 0U;
    }
    if (ipcDst != 0U) {
        NvSciStreamBlockDelete(ipcDst);
        ipcDst = 0U;
    }
}

void PerfConsumer::setEndpointBufAttr(NvSciBufAttrList attrList)
{
    NvSciBufType bufType{ NvSciBufType_RawBuffer };
    NvSciBufAttrValAccessPerm perm{ NvSciBufAccessPerm_Readonly };
    // Disable cpu access for vidmem
    bool cpuaccess_flag{ testArg.vidmem ? false : true };

    NvSciBufAttrKeyValuePair bufAttrs[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag,
            sizeof(cpuaccess_flag) },
    };

    CHECK_NVSCIERR(
        NvSciBufAttrListSetAttrs(attrList,
            bufAttrs,
            sizeof(bufAttrs) / sizeof(NvSciBufAttrKeyValuePair)));

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

void PerfConsumer::getCpuPtr(Packet& pkt)
{
    for (uint32_t i{ 0U }; i < NUM_ELEMENTS; i++) {
        void const* vaPtr{ nullptr };
        CHECK_NVSCIERR(NvSciBufObjGetConstCpuPtr(pkt.buffers[i], &vaPtr));
        pkt.constCpuPtr[i] = static_cast<uint64_t const*>(vaPtr);
    }
}

void PerfConsumer::createStream(
    NvSciIpcEndpoint const ipcEndpoint,
    NvSciStreamBlock const upstreamBlock)
{
    // Create blocks
    CHECK_NVSCIERR(NvSciStreamFifoQueueCreate(&queue));
    CHECK_NVSCIERR(NvSciStreamConsumerCreate(queue, &consumer));

    if (testArg.testType == CrossProcCons) {
        ipcEP = ipcEndpoint;

        if (testArg.isC2c) {
            CHECK_NVSCIERR(NvSciStreamStaticPoolCreate(testArg.numPackets,
                                                       &c2cPool));
        }
        CHECK_NVSCIERR(NvSciStreamIpcDstCreate2(ipcEP,
                                                syncModule,
                                                bufModule,
                                                c2cPool,
                                                &ipcDst));

        if (testArg.syncSend) {
            // Config the block to send payloads synchronously at runtime
            CHECK_NVSCIERR(NvSciStreamPacketSendSync(ipcDst));
        }
    }

    // Connect blocks
    if (testArg.testType == CrossProcCons) {
        CHECK_NVSCIERR(NvSciStreamBlockConnect(ipcDst, consumer));
    } else {
        CHECK_NVSCIERR(NvSciStreamBlockConnect(upstreamBlock, consumer));
    }

    endpointHandle = consumer;
}

void PerfConsumer::run(void)
{
    if (testArg.isC2c) {
        // Spawn a new thread to handle c2c pool's events
        poolThread = std::thread(&PerfClient::runPoolEventsHandler, this,
                                 c2cPool);
    }

    // Handle consumer's events
    handleEvents();
}

void PerfConsumer::handleInternalEvents(void)
{
    // IpcDst block processes the internal message from the pair ipcSrc
    NvSciError err = NvSciStreamBlockHandleInternalEvents(ipcDst);
    if (NvSciError_TryItAgain != err) {
        CHECK_NVSCIERR(err);
    }
}

void PerfConsumer::handleSetupComplete(void)
{
    // Init array to record release time for each packet
    releaseStartTime.resize(numRecvPackets);

    // Disable IPC event notification at runtime
    if (testArg.ipcEventNotifyDisabled) {
        CHECK_NVSCIERR(NvSciIpcEnableNotification(ipcEP, false));
        ipcNotifyOff = true;
    }

    PerfClient::handleSetupComplete();
}

void PerfConsumer::handlePayload(void)
{
    if (testArg.latency) {
        // Record the time that PacketReady event arrived and
        //  save timestamp data
        packetWaitEndTime = NvpGetTimeMark();
        NVP_CHECKERR_EXIT(NvpRecordSample(&consPacketWait,
                                          packetWaitStartTime,
                                          packetWaitEndTime));
    }

    // Acquire a payload
    NvSciStreamCookie cookie{ 0U };
    CHECK_NVSCIERR(NvSciStreamConsumerPacketAcquire(endpointHandle, &cookie));

    // Record time that the consumer acquires the packet.
    uint64_t presentEndTime{ 0U };
    if (testArg.latency) {
        presentEndTime = getTimeStamp(testArg.ptpTime);
    }

    uint32_t const id{ static_cast<uint32_t>(cookie) - 1U };
    CHECK_ERR(id < numRecvPackets, "Invalid cookie");
    Packet* packet{ &packets[id] };

    // Wait for prefences from producer
    for (uint32_t i{ 0U }; i < testArg.numSyncs; i++) {
        NvSciSyncFence fence;
        CHECK_NVSCIERR(NvSciStreamBlockPacketFenceGet(endpointHandle,
                                                      packet->handle,
                                                      0U,
                                                      i,
                                                      &fence));
        CHECK_NVSCIERR(NvSciSyncFenceWait(&fence,
                                          waitContext,
                                          FENCE_WAIT_INFINITE));
        NvSciSyncFenceClear(&fence);
    }

    // Stop the present timer when the consumer can read the buffer data.
    uint64_t presentEndTime2{ 0U };
    if (testArg.latency) {
        presentEndTime2 = getTimeStamp(testArg.ptpTime);
    }

    ++numPayloads;

    // Read timestamps from buffer, (present start time and
    //  release stop time of the last payload which uses this packet).
    if (testArg.latency) {
        uint64_t const *timerPtr{ packet->constCpuPtr[0] };

        // Save timestamp data for packet present from producer to consumer
        perfRecordSample(&presentLatency,
                          timerPtr[0],
                          presentEndTime,
                          testArg.ptpTime);
        perfRecordSample(&presentLatency2,
                          timerPtr[0],
                          presentEndTime2,
                          testArg.ptpTime);

        // Save timestamp data for packet-release latency from consumer to
        //  producer only for non inter-chip (c2c) stream.
        //  In c2c stream, after triggering copy submit, the packet in the
        //  upstream pool will be returned to the producer, and send a packet
        //  in the downstream (c2c) pool to the consumer in another chip.
        if (!testArg.isC2c && (timerPtr[1] != 0U)) {
            NVP_CHECKERR_EXIT(NvpRecordSample(&releaseLatency,
                                              releaseStartTime[id],
                                              timerPtr[1]));
        }
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

    // Record time the consumer releases the packet to the producer
    if (testArg.latency && !testArg.isC2c) {
        releaseStartTime[id] = NvpGetTimeMark();
    }

    // Release the packet back to the producer
    CHECK_NVSCIERR(NvSciStreamConsumerPacketRelease(endpointHandle,
                                                    packet->handle));
    // Signal postfences
    for (uint32_t i{ 0U }; i < testArg.numSyncs; i++) {
        if (syncs[i] != nullptr) {
            CHECK_NVSCIERR(NvSciSyncObjSignal(syncs[i]));
            NvSciSyncFenceClear(&postfences[i]);
        }
    }

    // Mark streaming done after processing all payloads
    if (numPayloads == testArg.numFrames) {
        handleStreamComplete();
    }
}

void PerfConsumer::handleStreamComplete(void)
{
    if (streamingDone) {
        return;
    }

    // Record streaming end time
    streamingEndTime = NvpGetTimeMark();

    // Total time of the streaming phase
    double duration = getTimeInUs(streamingEndTime - streamingStartTime);

    printf("\nConsumer received %d packet(s)\n", numPayloads);

    printf("Consumer total init time:                    %8.5f us\n",
           getTimeInUs(streamingStartTime - testStartTime));
    printf("Consumer setup phase (after stream connect): %8.5f us\n",
           getTimeInUs(streamingStartTime - setupStartTime));
    printf("Consumer streaming phase:                    %8.5f us\n", duration);

    if (testArg.isC2c) {
        // Convert unit of buffer size to (GB): bufSize(MB) / 1024
        // Convert unit of duration to (s): duration(us) / 10^6
        double bandwidth =
            (testArg.bufSize * numPayloads / 1024.0) *
            (1000000.0 / duration);
        printf("\nBuffer size per packet: %8.5f MB\n", testArg.bufSize);
        printf("PCIe bandwidth (buffer size received by consumer): "
               "%8.5f GBps\n\n", bandwidth);
    }

    streamingDone = true;
}
