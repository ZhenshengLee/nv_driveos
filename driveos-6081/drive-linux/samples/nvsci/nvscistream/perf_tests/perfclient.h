//! \file
//! \brief NvSciStream test client declaration.
//!
//! \copyright
//! Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//!
//! NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
//! property and proprietary rights in and to this material, related
//! documentation and any modifications thereto. Any use, reproduction,
//! disclosure or distribution of this material and related documentation
//! without an express license agreement from NVIDIA CORPORATION or
//! its affiliates is strictly prohibited.

#ifndef PERFCLIENT_H
#define PERFCLIENT_H

#include <array>
#include <vector>
#include <thread>
#include "util.h"

class PerfClient
{
public:
    PerfClient(NvSciBufModule buf,
               NvSciSyncModule sync);

    virtual ~PerfClient(void);

    // Run a separate thread to handle pool events
    void runPoolEventsHandler(NvSciStreamBlock pool);

protected:
    // Setup synchronization attributes at endpoint.
    //  This test uses CPU wait.
    NvSciError setCpuSyncAttrList(NvSciSyncAccessPerm cpuPerm,
                                  NvSciSyncAttrList attrList);

    // Setup synchronization attributes at endpoint
    virtual void setEndpointBufAttr(NvSciBufAttrList attrList) = 0;

    // Get CPU access pointer to each buffer in the packet
    virtual void getCpuPtr(Packet& pkt) = 0;

    // Handle all events at this endpoint
    //  with the following handle*() helper functions.
    void handleEvents(void);

    // Provide endpoint element attributes,
    //  after stream connected.
    void handleElemSupport(void);

    // Provide endpoint waiter attributes for each element
    //  after receiving the reconciled attributes from the pool.
    void handleElemSetting(void);

    // Reconcile endpoint signaler and remote waiter attributes,
    //  and provide synchornization objects,
    //  after receiving the remote waiter information.
    void handleSyncExport(void);

    // Save buffers and get CPU access to each buffer,
    //  after receiving a new packet.
    void handlePacketCreate(void);

    // Finalize all local setup if not done and mark setup done,
    //  after receiving the setup complete message.
    virtual void handleSetupComplete(void);

    // Process data in the payload and record timestamps,
    //  after receiving a new payload.
    virtual void handlePayload(void) = 0;

    // Mark streaming done,
    //  after receiving disconnect event or process all payloads.
    virtual void handleStreamComplete(void) = 0;

protected:
    NvSciBufModule                  bufModule{ nullptr };
    NvSciSyncModule                 syncModule{ nullptr };

    NvSciStreamBlock                endpointHandle{ 0U };
    std::thread                     poolThread;

    NvSciSyncCpuWaitContext         waitContext{ nullptr };
    std::vector<NvSciSyncObj>       syncs;

    std::vector<Packet>             packets;
    uint32_t                        numRecvPackets{ 0U };

    uint32_t                        numPayloads{ 0U };
    std::vector<NvSciSyncFence>     postfences;

    bool                            setupDone{ false };
    bool                            streamingDone{ false };

    uint64_t                        setupStartTime{ 0UL };
    uint64_t                        streamingStartTime{ 0UL };
    uint64_t                        streamingEndTime{ 0UL };
    uint64_t                        packetWaitStartTime{ 0UL };
    uint64_t                        packetWaitEndTime{ 0UL };
} ;

#endif // PERFCLIENT_H
