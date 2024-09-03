//! \file
//! \brief NvSciStream perf test definition.
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

#include <thread>
#include "perftest.h"

extern TestArg testArg;

//==================== PerfTest ====================//

PerfTest::PerfTest(NvSciBufModule buf,
                   NvSciSyncModule sync):
    bufModule(buf),
    syncModule(sync)
{
}

void PerfTest::run(void)
{
    // Create and connect blocks in producer thread
    PerfProducer prod(bufModule, syncModule);
    NvSciStreamBlock upstreamBlock = prod.createStream();

    // Spawn seperate thread for each consumer
    std::vector<std::thread> consThread(testArg.numConsumers);
    for (uint32_t i{ 0U }; i < testArg.numConsumers; i++) {
        consThread[i] = std::thread(&PerfTest::runConsumer, this, upstreamBlock);
    }

    // Start producer
    prod.run();

    // Wait for consumer threads join
    for (uint32_t i{ 0U }; i < consThread.size(); i++) {
        if (consThread[i].joinable()) {
            consThread[i].join();
        }
    }
}

void PerfTest::runConsumer(NvSciStreamBlock upstreamBlock)
{
    // Create and connect blocks in consumer thread
    PerfConsumer cons(bufModule, syncModule);
    cons.createStream(0U, upstreamBlock);

    // Start Consumer
    cons.run();
}

//==================== PerfTestProd ====================//

PerfTestProd::PerfTestProd(std::vector<NvSciIpcEndpoint>& ipc,
                           NvSciBufModule buf,
                           NvSciSyncModule sync) :
    bufModule(buf),
    syncModule(sync),
    ipcEndpoint(ipc)
{
}

void PerfTestProd::run(void)
{
    // Create and connect blocks in producer proceess
    PerfProducer prod(bufModule, syncModule);
    prod.createStream(&ipcEndpoint);

    // Start producer
    prod.run();
}

//==================== PerfTestCons ====================//

PerfTestCons::PerfTestCons(NvSciIpcEndpoint ipc,
                           NvSciBufModule buf,
                           NvSciSyncModule sync) :
    bufModule(buf),
    syncModule(sync),
    ipcEndpoint(ipc)
{
}

void PerfTestCons::run(void)
{
    // Create and connect blocks in consumer process
    PerfConsumer cons(bufModule, syncModule);
    cons.createStream(ipcEndpoint, 0U);

    // Start Consumer
    cons.run();
}