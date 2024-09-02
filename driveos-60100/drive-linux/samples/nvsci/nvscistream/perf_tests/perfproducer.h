//! \file
//! \brief NvSciStream test Producer client declaration.
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

    // Trigger ipcSrc handler to process internal message from ipcDst
    void handleInternalEvents(void) final;

    // Finalize all local setup,
    //  after receiving the setup complete message.
    void handleSetupComplete(void) final;

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

    std::vector <NvSciIpcEndpoint>  ipcEP;

    // NvPlayFair rate-limit object to set the wait period for
    //  the specified frame rate.
    NvpRateLimitInfo_t              rateLimitInfo;
} ;

#endif // PERFPRODUCER_H
