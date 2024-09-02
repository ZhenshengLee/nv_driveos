//! \file
//! \brief Utility functions for nvscistrem perf tests.
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

#ifndef UTIL_H
#define UTIL_H

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <array>
#if (QNX == 1)
#include "nvtime2.h"
#else
#if !defined(__x86_64__)
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/nvpps_ioctl.h>
#include <unistd.h>
#endif
#endif
#include "nvscistream.h"
#include "nvplayfair.h"

// Number of elements per packet
#define NUM_ELEMENTS  (1U)

#define QUERY_WAIT_INFINITE (-1L)
#define FENCE_WAIT_INFINITE (-1L)
#define TIMEUNIT (NvpTimeUnits_t::USEC)
#define NS_RES 1000000000U



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
    bool                ptpTime{ false };
    bool                ipcEventNotifyDisabled{ false };
    bool                syncSend{ false };
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

#if (QNX == 0) && !defined(__x86_64__)
// File descriptor for nvpps device
extern int fd;

// Utility function to init/deinit NvPPS on Linux.
//  We need to sync time on different chips using PTP and
//  call NvPPS APIs to get the PTP timestamp to measure the
//  C2C packet-delivery latency.
inline bool nvppsInit(void)
{

    const char  *dev_name = "/dev/nvpps0";

    // Open the device
    fd = open(dev_name, O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        printf("ERR: Failed to open the device %s errno = %d",
               dev_name, errno);
        return false;
    }
    // set the mode
    struct nvpps_params params = {NVPPS_MODE_TIMER, NVPPS_TSC_NSEC};
    if (ioctl(fd, NVPPS_SETPARAMS, &params) != 0) {
        printf("ERR: ioctl failed for NVPPS_SETPARAMS err %s\n",
               strerror(errno));
        return false;
    }
    return true;
}

inline void nvppsDeinit(void) {
    close(fd);
}
#endif

// Utility functions to init/deinit ptp.
inline int ptpInit(void)
{
#if (QNX == 1)
    // On QNX, get PTP time with NvTime APIs.
    // Validate whether ptp setup done.
    struct timespec ts_ptr;
    if (NVTIME_SUCCESS == nvtime_gettime_ptp(&ts_ptr, "mgbe2")) {
        return true;
    }
#elif (!defined(__x86_64__))
    // On Linux, get PTP time with NvPPS APIs.
    if (nvppsInit()) {
        struct nvpps_timestamp_struct ts;
        if (ioctl(fd, NVPPS_GETTIMESTAMP, &ts) == 0) {
            return true;
        }
    }
#endif
    return false;
}

inline void ptpDeinit(void)
{
#if (QNX == 0) && !defined(__x86_64__)
    nvppsDeinit();
#endif
}


// Utility function to get the time stamp.
// For C2C case, get the PTP time.
// For non-C2C case, call NvPlayFair API to get system time.
inline uint64_t getTimeStamp(bool ptpTime)
{
    uint64_t timeStamp { 0U };

    if(!ptpTime) {
        // Use NvPlayFair API to get timestamp for non-C2C.
        timeStamp = NvpGetTimeMark();
    } else {
#if (QNX == 1)
        struct timespec ts_ptr;
        if (NVTIME_SUCCESS != nvtime_gettime_ptp(&ts_ptr, "mgbe2")) {
            printf("ERR: nvtime_gettime_ptp failed\n");
        } else {
            timeStamp = ((NS_RES*ts_ptr.tv_sec)+ts_ptr.tv_nsec);
        }
#elif (!defined(__x86_64__))
        struct nvpps_timestamp_struct ts;
        if (ioctl(fd, NVPPS_GETTIMESTAMP, &ts) != 0) {
            printf("ERR: ioctl failed for NVPPS_GETTIMESTAMP err %s\n",
                   strerror(errno));
        } else {
            timeStamp = ((NS_RES*ts.hw_ptp_ts.tv_sec)+ts.hw_ptp_ts.tv_nsec);
        }
#endif
    }
    return timeStamp;
}

// Utility function to record the given timestamp and latency values
//  to NvPlayFair data object.
inline void perfRecordSample(NvpPerfData_t *perfData,
                             uint64_t sampleStartTimeMark,
                             uint64_t sampleEndTimeMark,
                             bool ptpTime)
{
    if (!ptpTime) {
        NVP_CHECKERR_EXIT(NvpRecordSample(perfData,
                                          sampleStartTimeMark,
                                          sampleEndTimeMark));
    } else {
        uint64_t timestamp, latency;
        uint64_t sampleNumber;

        timestamp = sampleStartTimeMark;
        latency = (sampleEndTimeMark - sampleStartTimeMark);

        sampleNumber = __atomic_fetch_add(&perfData->sampleNumber, 1U,
                                          __ATOMIC_SEQ_CST);
        sampleNumber %= perfData->maxSamples;

        perfData->timestamps[sampleNumber] = timestamp;
        perfData->latencies[sampleNumber] = latency;
    }
}

// Utility function to get time in microsecond (us).
inline double getTimeInUs(uint64_t time)
{
    return NvpConvertTimeMarkToNsec(time) / 1000.0f;
};

#endif
