//
// NvSciStream Consumer channel declaration
//
// Copyright (c) 2019-2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef NVSCISTREAM_CONSUMER_CH_H
#define NVSCISTREAM_CONSUMER_CH_H

#include "nvscistream.h"
#include "cuda_consumer.h"

namespace NvScistreamSample
{
    typedef enum {
        QueueType_Mailbox,
        QueueType_FIFO
    } QueueType;

    class ConsumerChannel
    {
    private:
        QueueType queueType;

        NvSciStreamBlock consumerHandle;
        NvSciStreamBlock queueHandle;
        NvSciStreamBlock ipcDstHandle;

        NvSciBufModule bufModule;
        NvSciSyncModule syncModule;
        NvSciIpcEndpoint ipcEndpoint;

        CudaConsumer* consumer;

    public:
        ConsumerChannel() = delete;
        ConsumerChannel(NvSciBufModule bufMod,
                        NvSciSyncModule syncMod,
                        NvSciIpcEndpoint endpoint);
        ~ConsumerChannel();

        void createBlocks(QueueType queueType);
        void connectToStream(void);
        void setupForStreaming(void);
        void runStream(void);
    };
}

#endif
