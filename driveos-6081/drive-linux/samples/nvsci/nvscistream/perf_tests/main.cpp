//! \file
//! \brief NvSciStream perf test main.
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

#include <limits>
#include <algorithm>
#include <cmath>
#include <unistd.h>
#include <string>
#include <unordered_map>
#include "util.h"
#include "perftest.h"
#include "nvplayfair.h"

// Test configs
TestArg             testArg{};

#if (QNX == 0) && !defined(__x86_64__)
// File descriptor for nvpps device
int fd{ -1 };
#endif

// NvPlayFair data object to record the duration
//  that the producer waits for the PacketReady event.
// Data is recorded in the producer process.
NvpPerfData_t       prodPacketWait;

// NvPlayFair data object to record the duration
//  that the consumer waits for the PacketReady event.
// Data is recorded in the consumer process.
NvpPerfData_t       consPacketWait;

// NvPlayFair data object to record packet delivery latency
//  from the time the producer presents the packet
//  to the time the consumer acquires the packet, without waiting on fences.
// Data is recorded in the consumer process.
NvpPerfData_t       presentLatency;

// NvPlayFair data object to record packet delivery latency
//  from the time the producer presents the packet
//  to the time the consumer acquires the packet and waits on pre-fences.
// Data is recorded in the consumer process.
NvpPerfData_t       presentLatency2;

// NvPlayFair data object to record packet delivery latency
//  from the time the consumer releases the packet
//  to the time the producer gets the packet, without waiting on fences.
// Data is recorded in the consumer process.
NvpPerfData_t       releaseLatency;

// Record time the test starts
uint64_t            testStartTime{ 0UL };

// Utility functions
static void help(void);
static void runTest(std::unordered_map<uint32_t, std::string>& ipcEpNames);
static void perfAnalyze(NvpPerfData_t* data,
                        char const* msg,
                        bool ignoreTarget = false);


static void help(void)
{
    printf("\n===============================================================");
    printf("\n NvSciStream Perf Test App: Single-process use caes by default");
    printf("\n [-h]            Print this usage.");
    printf("\n Inter-process test:");
    printf("\n [-p]            Inter-process producer.");
    printf("\n [-c] <N>        Inter-process consumer, consumer index.");
    printf("\n Inter-chip test:");
    printf("\n [-P] <N> <s>    Inter-SoC producer, NvSciIpc endpoint name"
                               " connected to indexed consumer.");
    printf("\n [-C] <N> <s>    Inter-SoC consumer, consumer index and "
                               "NvSciIpc endpoint name for this consumer.");
    printf("\n Common options:");
    printf("\n [-n] <N>        Multicast stream with N Consumer(s).");
    printf("\n                 Default: 1.");
    printf("\n                 (Set by producer.)");
    printf("\n [-k] <N>        Number of packets in pool. ");
    printf("\n                 Default: 1.");
    printf("\n                 (Set by producer for primary pool. "
                               "Set by consumer for c2c pool)");
    printf("\n [-f] <N>        Number of frames.");
    printf("\n                 Default: 100.");
    printf("\n                 (Set the same number by both producer and "
                               "consumer.)");
    printf("\n [-b] <N>        Buffer size: N MB.");
    printf("\n                 Default: 1MB.");
    printf("\n                 (Set by producer.)");
    printf("\n [-s] <N>        Number of sync objects per client.");
    printf("\n                 Valid: [0, %u]. Default: 1.",
                               static_cast<uint32_t>(NUM_ELEMENTS));
    printf("\n [-r] <N>        Frame present rate: N fps.");
    printf("\n                 Default: No limit.");
    printf("\n                 (Set by producer.)");
    printf("\n [-t] <0|1>      Memory type. 0 for sysmem and 1 for vidmem.");
    printf("\n                 Pass dGPU UUID using -u, if using vidmem.");
    printf("\n                 Default: 0.");
    printf("\n [-u] <GPU_UUID> Required for vidmem buffers.");
    printf("\n                 Can be retrieved from 'nvidia-smi -L' command on x86.");
    printf("\n [-l]            Measure latency. Default: false.");
    printf("\n                 Skip if using vidmem.");
    printf("\n                 (Set by both producer and consumer.)");
    printf("\n Latency measurement options:");
    printf("\n (Ignore if not measuring latency)");
    printf("\n [-v]            Verbose. Save latency raw data in csv file");
    printf("\n                 Default: Not set.");
    printf("\n                 (Set by both producer and consumer.)");
    printf("\n [-a] <num>      Average KPI target (us) for packet delivery"
                               "latency.");
    printf("\n                 (Set by consumer.)");
    printf("\n [-m] <num>      99.99 percentile KPI target (us) for packet"
                               "delivery latency.");
    printf("\n                 (Set by consumer.)\n");
}

int main(int argc, char **argv)
{
    // Parse test options
    int32_t opt;
    bool gpuUUIDSet{ false };
    std::unordered_map<uint32_t, std::string> ipcEpNames;
    while ((opt = getopt(argc, argv, "pP:c:C:n:k:f:b:s:r:t:a:m:u:lvh")) != -1) {
        switch (opt) {
        case 'p':
            testArg.testType = CrossProcProd;
            break;
        case 'P':
            testArg.testType = CrossProcProd;
            testArg.isC2c = true;
            ipcEpNames[atoi(optarg)] = argv[optind++];
            break;
        case 'c':
            testArg.testType = CrossProcCons;
            testArg.consIndex = atoi(optarg);
            break;
        case 'C':
            testArg.testType = CrossProcCons;
            testArg.consIndex = atoi(optarg);
            testArg.isC2c = true;
            ipcEpNames[testArg.consIndex] = argv[optind++];
            break;
        case 'n':
            testArg.numConsumers = atoi(optarg);
            if (testArg.numConsumers == 0U) {
                printf("ERR: Number of consumers is zero.\n");
                help();
                return -1;
            }
            break;
        case 'k':
            testArg.numPackets = atoi(optarg);
            break;
        case 'f':
            testArg.numFrames = atoi(optarg);
            break;
        case 'b':
            testArg.bufSize = atof(optarg);
            break;
        case 's':
            testArg.numSyncs = atoi(optarg);
            if (testArg.numSyncs > NUM_ELEMENTS) {
                printf("ERR: Number of synch objects out of range\n");
                help();
                return -1;
            }
            break;
        case 'r':
            testArg.fps = atoi(optarg);
            break;
        case 't':
            if (atoi(optarg) == 0) {
                testArg.vidmem = false;
            } else if (atoi(optarg) == 1 ) {
                testArg.vidmem = true;
            } else {
                printf("ERR: Memory type not supported.\n");
                help();
                return -1;
            }
            break;
        case 'l':
            testArg.latency = true;
            break;
        case 'v':
            testArg.verbose = true;
            break;
        case 'a':
            testArg.avgTarget = atof(optarg);
            break;
        case 'm':
            testArg.pct99Target = atof(optarg);
            break;
        case 'u':
            for (uint32_t i=0; i<sizeof(testArg.gpuUUID.bytes); i+=1) {
                char* strOptArg = (char*)optarg;
                sscanf(strOptArg + 2*i, "%02hx",(uint16_t*)&testArg.gpuUUID.bytes[i]);
            }
            gpuUUIDSet = true;
            break;
        default:
            help();
            return 0;
        }
    }

    // As we could not access the vidmem on tegra with CPU pointer,
    // we could not read/write timestamps into the buffer directly.
    // Disable the latency measurement if using vidmem.
    // We only measure the PCIe bandwidth for c2c caes with vidmem now.
    if (testArg.vidmem) {
        testArg.latency = false;
        if (!gpuUUIDSet) {
            printf("ERR: GPU UUID needed for using vidmem\n");
            help();
            return -1;
        }
    }

    // We need to sync time using PTP to measure inter-chip latency
    if (testArg.latency && testArg.isC2c) {
#if !defined(__x86_64__)
        // If init fail, fall back to use NvPlayFair API.
        // We can get other perf number, but the inter-chip packet-delivery
        // latency may not be correct.
        if (ptpInit()) {
            testArg.ptpTime = true;
        }
#else
    // With PTP, x86 is the host, which can use NvPlayFair API
    // to get timestamp.
#endif
    }

    // Initialize NvPlayFair data objects
    if (testArg.latency) {

        // Perf data on saved on producer side
        if ((testArg.testType == SingleProc) ||
            (testArg.testType == CrossProcProd)) {

            // Number of samples
            uint64_t numOfSamples = testArg.numFrames;

            NVP_CHECKERR_EXIT(
                NvpConstructPerfData(&prodPacketWait,
                                     numOfSamples,
                                     "nvscistream_prod_pktwait_latency.csv"));
        }

        // Perf data saved on consumer side
        if ((testArg.testType == SingleProc) ||
            (testArg.testType == CrossProcCons)) {

            // Number of samples
            uint64_t numOfSamples = testArg.numFrames * testArg.numConsumers;

            NVP_CHECKERR_EXIT(
                NvpConstructPerfData(&presentLatency,
                                     numOfSamples,
                                     "nvscistream_present_latency.csv"));

            NVP_CHECKERR_EXIT(
                NvpConstructPerfData(&presentLatency2,
                                     numOfSamples,
                                     "nvscistream_present_latency_2.csv"));

            NVP_CHECKERR_EXIT(
                NvpConstructPerfData(&consPacketWait,
                                     numOfSamples,
                                     "nvscistream_cons_pktwait_latency.csv"));

            // Only record release time from consumer to producer in
            //  non C2C (chip-to-chip) case.
            if (!testArg.isC2c) {
                NVP_CHECKERR_EXIT(
                    NvpConstructPerfData(&releaseLatency,
                                         numOfSamples,
                                         "nvscistream_release_latency.csv"));
            }
        }
    }

    // Record test start time
    testStartTime = NvpGetTimeMark();

    // Run test
    runTest(ipcEpNames);

    // Analyze perf and cleanup data
    if (testArg.latency) {
        // Perf data on producer side
        if (prodPacketWait.sampleNumber > 0UL) {
            perfAnalyze(&prodPacketWait,
                        "Producer packet wait",
                        true);
        }

        // Perf data on proconsumer side
        if (presentLatency.sampleNumber > 0UL) {
            perfAnalyze(&presentLatency, "Producer packet present");
        }
        if (presentLatency2.sampleNumber > 0UL) {
            perfAnalyze(&presentLatency2,
                        testArg.isC2c ?
                        "Producer packet present + C2C copy done" :
                        "Producer packet present + engine write done");
        }
        if (consPacketWait.sampleNumber > 0UL) {
            perfAnalyze(&consPacketWait,
                        "Consumer packet wait",
                        true);
        }
        if (releaseLatency.sampleNumber > 0UL) {
            perfAnalyze(&releaseLatency, "Consumer packet release");
        }
    }

    if (testArg.latency && testArg.isC2c) {
        ptpDeinit();
    }

    return 0;
}

static void runTest(std::unordered_map<uint32_t, std::string>& ipcEpNames)
{
    // Open NvSciSync/NvSciBuf modules
    NvSciBufModule bufModule{ nullptr };
    CHECK_NVSCIERR(NvSciBufModuleOpen(&bufModule));

    NvSciSyncModule syncModule{ nullptr };
    CHECK_NVSCIERR(NvSciSyncModuleOpen(&syncModule));

    switch (testArg.testType) {
    case SingleProc:
    {
        // Run single-proc test
        PerfTest perfTest(bufModule, syncModule);
        perfTest.run();

        break;
    }
    case CrossProcProd:
    {
        // Init NvSciIpc channel and open endpoints
        CHECK_NVSCIERR(NvSciIpcInit());

        std::vector<NvSciIpcEndpoint> ipcEndpoint(testArg.numConsumers, 0U);
        for (uint32_t i = 0U; i < testArg.numConsumers; i++) {
            if (testArg.isC2c) {
                CHECK_NVSCIERR(NvSciIpcOpenEndpoint(ipcEpNames[i].c_str(),
                                                    &ipcEndpoint[i]));
            } else {
                char ipcEpName[32];
                sprintf(ipcEpName, "%s%d", "nvscistream_", 2 * i);
                CHECK_NVSCIERR(NvSciIpcOpenEndpoint(ipcEpName,
                                                    &ipcEndpoint[i]));
            }
            CHECK_NVSCIERR(NvSciIpcResetEndpointSafe(ipcEndpoint[i]));
        }

        // Run producer test
        PerfTestProd* prodTest = new PerfTestProd(ipcEndpoint,
                                                  bufModule,
                                                  syncModule);
        prodTest->run();
        delete prodTest;

        // Close NvSciIpc endpoint and deinit
        for (uint32_t i = 0U; i < testArg.numConsumers; i++) {
            CHECK_NVSCIERR(NvSciIpcCloseEndpointSafe(ipcEndpoint[i], false));
        }
        NvSciIpcDeinit();

        break;
    }
    case CrossProcCons:
    {
        // Init NvSciIpc channel and open endpoints
        CHECK_NVSCIERR(NvSciIpcInit());

        NvSciIpcEndpoint ipcEndpoint{ 0U };
        if (testArg.isC2c) {
            CHECK_NVSCIERR(
                NvSciIpcOpenEndpoint(ipcEpNames[testArg.consIndex].c_str(),
                                     &ipcEndpoint));
        } else {
            char ipcEpName[32];
            sprintf(ipcEpName, "%s%d",
                    "nvscistream_", 2 * testArg.consIndex + 1);
            CHECK_NVSCIERR(NvSciIpcOpenEndpoint(ipcEpName, &ipcEndpoint));
        }
        CHECK_NVSCIERR(NvSciIpcResetEndpointSafe(ipcEndpoint));

        // Run consumer process
        PerfTestCons* consTest = new PerfTestCons(ipcEndpoint,
                                                  bufModule,
                                                  syncModule);
        consTest->run();
        delete consTest;

        // Close NvSciIpc endpoint and deinit
        CHECK_NVSCIERR(NvSciIpcCloseEndpointSafe(ipcEndpoint, false));
        NvSciIpcDeinit();

        break;
    }
    default:
        help();
        break;
    }

    // Close NvSciBuf and NvSciSync modules
    if (bufModule != nullptr) {
        NvSciBufModuleClose(bufModule);
        bufModule = nullptr;
    }
    if (syncModule != nullptr) {
        NvSciSyncModuleClose(syncModule);
        syncModule = nullptr;
    }
}

static void perfAnalyze(NvpPerfData_t* data,
                        char const* msg,
                        bool ignoreTarget)
{
    // Save the raw data to nvscistream_*.csv
    if (testArg.verbose) {
        NVP_CHECKERR_EXIT(NvpDumpData(data));
    }

    // Calculate common stat with the benchmark data
    NvpPerfStats_t stats;
    NVP_CHECKERR_EXIT(NvpCalcStats(data, &stats, TIMEUNIT));

    // Print the summary of perf stat
    NVP_CHECKERR_EXIT(NvpPrintStats(data, &stats, TIMEUNIT, msg, false));

    if (!ignoreTarget) {
        // Compare with average and 99.99 percentile target, if set,
        if (testArg.avgTarget > 0.0f) {
            printf("----------------------------------------\n");
            printf("** Average Target (us): %8.5f. \n", testArg.avgTarget);
            // 5% tolerance range.
            printf("  [%s]\n", stats.mean <= testArg.avgTarget * 1.05 ?
                  "Passed" :
                  "Failed");
        }
        if (testArg.pct99Target > 0.0f) {
            printf("----------------------------------------\n");
            printf("** 99.99 percentile Target (us):  %8.5f. \n",
                   testArg.pct99Target);
            // 5% tolerance range.
            printf("  [%s]\n", (stats.pct99 <= testArg.pct99Target * 1.05) ?
                  "Passed" :
                  "Failed");
        }
    }

    // Destroy NvPlayFair data object
    NVP_CHECKERR_EXIT(NvpDestroyPerfData(data));
}
