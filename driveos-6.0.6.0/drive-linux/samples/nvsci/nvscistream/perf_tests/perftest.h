//! \file
//! \brief NvSciStream perf test declaration.
//!
//! \copyright
//! Copyright (c) 2019-2022 NVIDIA Corporation. All rights reserved.
//!
//! NVIDIA Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from NVIDIA Corporation is strictly prohibited.

#ifndef PERFPTEST_H
#define PERFPTEST_H

#include <vector>
#include "perfproducer.h"
#include "perfconsumer.h"

// Single process
class PerfTest
{
public:
    PerfTest(NvSciBufModule buf,
             NvSciSyncModule sync);
    virtual ~PerfTest(void) = default;
    void run(void);
    void runConsumer(NvSciStreamBlock upstreamBlock);

private:
    NvSciBufModule                  bufModule{ nullptr };
    NvSciSyncModule                 syncModule{ nullptr };
};

// Producer process
class PerfTestProd
{
public:
    PerfTestProd(std::vector<NvSciIpcEndpoint>& ipc,
                 NvSciBufModule buf,
                 NvSciSyncModule sync);
    virtual ~PerfTestProd(void) = default;
    void run(void);

private:
    NvSciBufModule                  bufModule{ nullptr };
    NvSciSyncModule                 syncModule{ nullptr };

    std::vector<NvSciIpcEndpoint>   ipcEndpoint;
};

// Consumer process
class PerfTestCons
{
public:
    PerfTestCons(NvSciIpcEndpoint ipc,
                 NvSciBufModule buf,
                 NvSciSyncModule sync);
    virtual ~PerfTestCons(void) = default;
    void run(void);

private:
    NvSciBufModule                  bufModule{ nullptr };
    NvSciSyncModule                 syncModule{ nullptr };

    NvSciIpcEndpoint                ipcEndpoint;
};

#endif // PERFPTEST_H
