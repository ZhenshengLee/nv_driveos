//
// NvSciStream Producer channel declaration
//
// Copyright (c) 2019-2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef NVSCISTREAM_PRODUCER_CH_H
#define NVSCISTREAM_PRODUCER_CH_H

#include "nvscistream.h"
#include "nvmedia_producer.h"
#include "packet_pool_manager.h"

namespace NvScistreamSample
{
    //=============================================================================
    // Define NvSciStreamSample test suite.
    //=============================================================================
    class ProducerChannel
    {
    private:
        NvSciStreamBlock producerHandle;
        NvSciStreamBlock poolHandle;
        NvSciStreamBlock ipcSrcHandle[MAX_CONSUMERS];
        NvSciStreamBlock multicast = 0U;

        NvSciBufModule bufModule;
        NvSciSyncModule syncModule;
        NvSciIpcEndpoint ipcEndpoint[MAX_CONSUMERS];

        NvMediaProducer* producer;
        PacketPoolManager* pool;

        void setupPacketRequirements(void);

        uint32_t    consumerCount = 1U;

    public:
        ProducerChannel() = delete;
        ProducerChannel(NvSciBufModule bufMod,
                        NvSciSyncModule syncMod,
                        NvSciIpcEndpoint endpoint[MAX_CONSUMERS],
                        uint32_t nConsumer);
        ~ProducerChannel();

        void createBlocks(void);
        void connectToStream(void);
        void setupForStreaming(void);
        void runStream(void);
    };
}

#endif