//! \file
//! \brief Utility functions for nvscistrem perf tests.
//!
//! \copyright
//! Copyright (c) 2019-2022 NVIDIA Corporation. All rights reserved.
//!
//! NVIDIA Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from NVIDIA Corporation is strictly prohibited.

#ifndef UTIL_H
#define UTIL_H

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <array>
#include "nvscistream.h"
#include "nvplayfair.h"

// Number of elements per packet
#define NUM_ELEMENTS  (1U)

#define QUERY_WAIT_INFINITE (-1L)
#define FENCE_WAIT_INFINITE (-1L)
#define TIMEUNIT (NvpTimeUnits_t::USEC)



// Check the return error of NvSci* APIs
#define CHECK_NVSCIERR(e) {                                 \
    if (e != NvSciError_Success) {                          \
        printf ("%s:\n %s:%d, NvSci ERR: %0x\n",            \
            __FILE__, __func__, __LINE__, e);               \
        exit(-1);                                           \
    }                                                       \
}

// Check error and print error message for failure case
#define CHECK_ERR(e, s) {                                   \
    if (!(e)) {                                             \
        printf ("%s, %s:%d, ERR: %s\n",                     \
            __FILE__, __func__, __LINE__, s);               \
        exit(-1);                                           \
    }                                                       \
}

// The test support single-process, inter-process and
//  inter-chip (C2C) use cases. Need to run separate
//  producer and consumer processes for inter-process/
//  inter-chip use case.
enum TestType {
    SingleProc,
    CrossProcProd,
    CrossProcCons
};

// Test config
struct TestArg {
    TestType            testType{ SingleProc };
    bool                isC2c{ false };
    uint32_t            numConsumers{ 1U };
    uint32_t            consIndex{ 0U };
    uint32_t            numPackets{ 1U };
    uint32_t            numFrames{ 100U };
    double              bufSize{ 1.0f };
    uint32_t            numSyncs{ 1U };
    bool                vidmem{ false };
    uint32_t            fps{ 0U };
    bool                latency{ false };
    bool                verbose{ false };
    double              avgTarget{ 0.0f };
    double              pct99Target{ 0.0f };
    NvSciRmGpuId        gpuUUID;
};

// Packet data
struct Packet {
    Packet(void) {
        buffers.fill(nullptr);
        cpuPtr.fill(nullptr);
        constCpuPtr.fill(nullptr);
    };

    NvSciStreamCookie                         cookie{ 0U };
    NvSciStreamPacket                         handle{ 0U };

    std::array<NvSciBufObj, NUM_ELEMENTS>     buffers;

    // CPU pointer to write data into buffers,
    //  used by producer.
    std::array<uint64_t*, NUM_ELEMENTS>       cpuPtr;

    // CPU pointer to read data from buffers,
    //  used by consumer.
    std::array<uint64_t const*, NUM_ELEMENTS> constCpuPtr;
};

#endif
