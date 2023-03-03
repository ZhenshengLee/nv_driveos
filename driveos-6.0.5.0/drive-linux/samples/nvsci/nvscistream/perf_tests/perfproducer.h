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

#ifndef PERFPRODUCER_H
#define PERFPRODUCER_H
#include "perfclient.h"
#include "nvplayfair.h"

class PerfProducer: public PerfClient
{
public:
    PerfProducer(NvSciBufModule buf,
                 NvSciSyncModule sync);
    ~PerfProducer(void) final;

    // Create producer, pool, mutlicast (for multicast streams) and
    //  ipcSrc (for inter-process stream) blocks and connect them.
    //  Return the last block of the producer chain, which will be
    //  connected to the consumer chain.
    NvSciStreamBlock createStream(
        std::vector<NvSciIpcEndpoint>* ipcEndpoint = nullptr);

    // Spawn a separate thread to handle events on pool block, and
    //  start waiting for and handling events on producer block.
    void run(void);

protected:
    // Setup element attributes supported by the producer.
    void setEndpointBufAttr(NvSciBufAttrList attrList) final;

    // Get CPU access pointer to each buffer in the packet
    //  with write permission.
    void getCpuPtr(Packet& pkt) final;


    // Get a packet from pool and wait for prefences from the consumer.
    //  Record and write timestamps into the buffer, generate postfences
    //  and send payloads to the consumer.
    void handlePayload(void) final;

    // Mark streaming done,
    //  after receiving the disconnect event from the consumer.
    void handleStreamComplete(void) final;

protected:
    NvSciStreamBlock                producer{ 0U };
    NvSciStreamBlock                pool{ 0U };
    NvSciStreamBlock                multicast{ 0U };
    std::vector<NvSciStreamBlock>   c2cQueue;
    std::vector<NvSciStreamBlock>   ipcSrc;

    // NvPlayFair rate-limit object to set the wait period for
    //  the specified frame rate.
    NvpRateLimitInfo_t              rateLimitInfo;
} ;

#endif // PERFPRODUCER_H
