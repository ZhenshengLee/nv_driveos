//
// NvSciStream Consumer channel declaration
//
// Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
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

typedef enum {
    QueueType_Mailbox,
    QueueType_FIFO
} QueueType;

class ConsumerChannel
{
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

private:
    void finalizeSetup(void);

    QueueType queueType = QueueType_FIFO;

    NvSciStreamBlock consumerHandle = 0U;
    NvSciStreamBlock queueHandle = 0U;
    NvSciStreamBlock ipcDstHandle = 0U;

    NvSciBufModule bufModule = nullptr;
    NvSciSyncModule syncModule = nullptr;
    NvSciIpcEndpoint ipcEndpoint = 0U;

    CudaConsumer* consumer = nullptr;
};

#endif
