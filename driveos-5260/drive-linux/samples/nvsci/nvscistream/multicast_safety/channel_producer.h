//
// NvSciStream Producer channel declaration
//
// Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef NVSCISTREAM_PRODUCER_CH_H
#define NVSCISTREAM_PRODUCER_CH_H

#include "nvscistream.h"
#include "cuda_producer.h"

class ProducerChannel
{
public:
    ProducerChannel() = delete;
    ProducerChannel(NvSciBufModule bufMod,
                    NvSciSyncModule syncMod,
                    NvSciIpcEndpoint endpoint[MAX_CONSUMERS],
                    uint32_t nConsumer = 1U);
    ~ProducerChannel();

    void createBlocks(void);
    void connectToStream(void);
    void setupForStreaming(void);
    void runStream(void);

private:
    void finalizeSetup(void);

    NvSciStreamBlock producerHandle = 0U;
    NvSciStreamBlock poolHandle = 0U;
    NvSciStreamBlock ipcSrcHandle[MAX_CONSUMERS] = { 0U };
    NvSciStreamBlock multicast = 0U;

    NvSciBufModule bufModule = nullptr;
    NvSciSyncModule syncModule = nullptr;
    NvSciIpcEndpoint ipcEndpoint[MAX_CONSUMERS];

    CudaProducer* producer = nullptr;

    uint32_t    consumerCount = 0U;
};

#endif
