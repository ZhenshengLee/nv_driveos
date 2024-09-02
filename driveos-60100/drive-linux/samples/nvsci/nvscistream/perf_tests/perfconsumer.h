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

#ifndef PERFCONSUMER_H
#define PERFCONSUMER_H

#include "perfclient.h"
#include "nvplayfair.h"

class PerfConsumer : public PerfClient
{
public:
    PerfConsumer(NvSciBufModule buf,
                 NvSciSyncModule sync);
    ~PerfConsumer(void) final;

    // Create consumer, queue, ipcDst (for inter-process streams) and
    //  pool (which is attached to ipcDst block in inter-chip streams)
    //  blocks, and connect them.
    //  Connect producer and consumer by connecting the first block of
    //  the consumer chain to the provided upstream block from the
    //  producer chain.
    void createStream(NvSciIpcEndpoint const ipcEndpoint,
                      NvSciStreamBlock const upstreamBlock);

    // Spawn a separate thread to handle events on c2c pool block,
    //  if it's an inter-chip stream, and start waiting for and
    //  handling events on consumer block.
    void run(void);

protected:
    // Setup element attributes supported by the consumer.
    void setEndpointBufAttr(NvSciBufAttrList attrList) final;

    // Get CPU access pointer to each buffer in the packet
    //  with read permission.
    void getCpuPtr(Packet& pkt) final;

    // Trigger ipcDst handler to process internal message from ipcSrc
    void handleInternalEvents(void) final;

    // Finalize all local setup,
    //  after receiving the setup complete message.
    void handleSetupComplete(void) final;

    // Get a payload from queue and wait for prefences from the producer.
    //  Record and read timestamps from the buffer, generate postfences
    //  and release the packet to the producer for reuse.
    void handlePayload(void) final;

    // Mark streaming done,
    //  after processing all payloads from the producer.
    void handleStreamComplete(void) final;

protected:
    NvSciStreamBlock        ipcDst{ 0U };
    NvSciStreamBlock        c2cPool{ 0U };
    NvSciStreamBlock        queue{ 0U };
    NvSciStreamBlock        consumer{ 0U };

    NvSciIpcEndpoint        ipcEP{ 0U };

    // Array of timestamps to record releasing time for each packet.
    //  The array will be resized with the number of packets.
    std::vector<uint64_t>   releaseStartTime;
};

#endif // PERFCONSUMER_H
