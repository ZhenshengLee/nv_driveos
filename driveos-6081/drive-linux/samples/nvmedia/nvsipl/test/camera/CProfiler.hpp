/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <mutex>
#include <queue>
#include <fstream>
#include <condition_variable>

#include "nvmedia_core.h"

#include "NvSIPLClient.hpp"
#include "CUtils.hpp"

#ifdef NVMEDIA_QNX
#include <sys/syspage.h>
#include <sys/neutrino.h>
#else
#define PTP_DEV "/dev/ptp0"
#define CLOCKFD (3)
#define FD2CLOCKID(fd) ((~(clockid_t)(fd) << 3) | (CLOCKFD))
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <linux/ethtool.h>
#include <net/if.h>
#include <linux/sockios.h>
#include <sys/ioctl.h>
#endif // NVMEDIA_QNX

#ifndef CPROFILER_HPP
#define CPROFILER_HPP

#define NVMEDIA_IMAGE_STATUS_TIMEOUT_MS (100U)
static constexpr uint32_t TSC_TO_NS_MULTIPLIER {32U};

using namespace nvsipl;

#ifndef NVMEDIA_QNX
#define MAX_NETWORK_INTERFACE_NAME_LEN (IFNAMSIZ)
class CClockIdMgr
{
 public:
    SIPLStatus GetClockId(NvSiplTimeBase timeBase, clockid_t &clockId)
    {
       char node[20];

        switch (timeBase) {
            case NVSIPL_TIME_BASE_CLOCK_PTP:
                if (m_ptpFD < 0) {
                    if (GetPtpDeviceNode(node) != 0) {
                        LOG_ERR("GetPtpDeviceNode failed\n");
                        return NVSIPL_STATUS_ERROR;
                    }
                    m_ptpFD = open(node, O_RDONLY);
                    if (m_ptpFD < 0) {
                        LOG_ERR("Unable to open %s! (errno:%d)\n", node, errno);
                        return NVSIPL_STATUS_ERROR;
                    }
                    LOG_INFO("Using PTP clock\n");
                }
                clockId = FD2CLOCKID(m_ptpFD);
                break;
            case NVSIPL_TIME_BASE_CLOCK_MONOTONIC:
                clockId = CLOCK_MONOTONIC;
                break;
            default:
                LOG_ERR("Unsupported time base: %u\n", timeBase);
                return NVSIPL_STATUS_ERROR;
        }

        return NVSIPL_STATUS_OK;
    }

    ~CClockIdMgr()
    {
        if (m_ptpFD != -1) {
            close(m_ptpFD);
        }
    }

 private:
    int m_ptpFD = -1;

    int GetPtpDeviceNodeByNetworkInterface(const char* interface_name, char* node) {
        int sd;
        int ret;
        struct ethtool_ts_info info;
        struct ifreq ifr;

        sd = socket( AF_UNIX, SOCK_DGRAM, 0 );
        if( sd < 0 ) {
            LOG_ERR("GetPtpDeviceNodeIndex: failed to open socket\n");
            ret = -1;
            goto done;
        }
        memset( &ifr, 0, sizeof(ifr));
        memset( &info, 0, sizeof(info));
        info.cmd = ETHTOOL_GET_TS_INFO;
        strcpy(ifr.ifr_name, interface_name);
        ifr.ifr_data = (char *) &info;
        if( ioctl( sd, SIOCETHTOOL, &ifr ) < 0 ) {
            LOG_ERR("GetPtpDeviceNodeIndex: ioctl(SIOETHTOOL) failed\n");
            ret = -1;
            node[0] = '\0';
        } else {
            ret = 0;
            sprintf(node,"/dev/ptp%d", info.phc_index);
            LOG_INFO("GetPtpDeviceNode() selected %s\n", node);
        }
        close(sd);
    done:
        return ret;
    }

    int GetPtpDeviceNode(char* node) {
        int rc = -1;
        int i;
        const char network_interface_names[][MAX_NETWORK_INTERFACE_NAME_LEN] = {"mgbe2_0",
                                                                                "eth0",
                                                                                "eqos_0",
                                                                                "mgbe3_0"};
        int number_of_network_interfaces = sizeof(network_interface_names)/sizeof(network_interface_names[0]);

        for (i = 0; i < number_of_network_interfaces; i++) {

            rc = GetPtpDeviceNodeByNetworkInterface(network_interface_names[i], node);

            if (rc == 0) {
                break;
            }
        }

        return rc;
    }
};

#endif // NVMEDIA_QNX

class CProfiler
{
 public:
    typedef struct {
        std::mutex profDataMut;
        uint64_t uFrameCount;
        uint64_t uPrevFrameCount;
        uint64_t uFirstCaptureDelayUs;
        uint64_t uFirstReceivedDelayUs;
        uint64_t uFirstReadyDelayUs;
    } ProfilingData;

    SIPLStatus Init(uint32_t uSensor,
                    INvSIPLClient::ConsumerDesc::OutputType outputType,
                    std::vector<uint64_t> uInitTimeArrUs,
                    NvSciSyncModule sciSyncModule,
                    bool bEnableProfiling,
                    std::string &sProfilePrefix,
                    bool bEnableInitProfiling = false)
    {
        m_uSensor = uSensor;
        m_outputType = outputType;

        m_profData.profDataMut.lock();
        m_profData.uFrameCount = 0U;
        m_profData.uPrevFrameCount = 0U;
        m_profData.profDataMut.unlock();

        m_bQuit = false;
        m_bEnableProfiling = bEnableProfiling;

        auto sciErr = NvSciSyncCpuWaitContextAlloc(sciSyncModule, &m_cpuWaitContext);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncCpuWaitContextAlloc");

        if (m_bEnableProfiling) {
            m_thread = std::thread(sFrameWaiterThreadFunc, this);
        }

        if (bEnableProfiling) {
            std::string filename = sProfilePrefix + "_cam_" + std::to_string(m_uSensor) +
                "_out_" + std::to_string(int(m_outputType)) + ".csv";

            m_profileFile.open(filename, std::fstream::out);
            if (!m_profileFile.is_open()) {
                LOG_ERR("Fail to open output CSV file\n");
                return NVSIPL_STATUS_ERROR;
            }

            if (m_outputType == INvSIPLClient::ConsumerDesc::OutputType::ICP) {
                m_profileFile << "Capture Timestamp (us),";
            }
            else {
                m_profileFile << "Capture Timestamp (us),";
                m_profileFile << "Capture Done Event Timestamp (us),";
            }
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus ProfileFrame(INvSIPLClient::INvSIPLNvMBuffer *pNvMBuffer)
    {
        if (pNvMBuffer == nullptr) {
            LOG_ERR("Invalid INvSIPLClient::INvSIPLNvMBuffer pointer\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        if (m_bEnableProfiling) {
            pNvMBuffer->AddRef();

            m_oBufferQueueMutex.lock();
            m_oBufferQueue.push(pNvMBuffer);
            m_oBufferQueueMutex.unlock();
            m_oBufferQueueCond.notify_one();
        }
        else {
            m_profData.profDataMut.lock();
            m_profData.uFrameCount++;
            m_profData.profDataMut.unlock();
        }
        return NVSIPL_STATUS_OK;
    }

    void RecordCaptureDoneEventTime(uint64_t frameCaptureTSC, std::vector<uint64_t> uCaptureEventTimeArrUs)
    {
	if (m_bEnableProfiling) {
            m_oTimePairQueueMutex.lock();
            m_oTimePairQueue.push(std::make_pair(frameCaptureTSC, uCaptureEventTimeArrUs));
            m_oTimePairQueueMutex.unlock();

            m_oTimePairQueueCondition.notify_all();
	}
    }

    void Deinit()
    {
        m_bQuit = true;
        m_oTimePairQueueCondition.notify_all();
        m_oBufferQueueCond.notify_one();
        if (m_thread.joinable()) {
            m_thread.join();
        }

        if (m_cpuWaitContext != nullptr) {
            NvSciSyncCpuWaitContextFree(m_cpuWaitContext);
            m_cpuWaitContext = nullptr;
        }

        if (m_profileFile.is_open()) {
            m_profileFile.close();
        }
    }

    ~CProfiler()
    {
        Deinit();
    }

    uint32_t m_uSensor = UINT32_MAX;
    INvSIPLClient::ConsumerDesc::OutputType m_outputType;
    ProfilingData m_profData;

 private:

    static void sFrameWaiterThreadFunc(CProfiler *pProf)
    {
        pthread_setname_np(pthread_self(), "FrameWaiter");
        pProf->FrameWaiterThreadFunc();
    }

    SIPLStatus WaitForPostFence(INvSIPLClient::INvSIPLNvMBuffer *pNvMBuffer)
    {
        NvSciSyncFence fence = NvSciSyncFenceInitializer;
        SIPLStatus status = pNvMBuffer->GetEOFNvSciSyncFence(&fence);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("INvSIPLClient::INvSIPLNvMBuffer::GetEOFNvSciSyncFence failed\n");
            return NVSIPL_STATUS_ERROR;
        }

        NvSciError sciErr = NvSciSyncFenceWait(&fence,
                                               m_cpuWaitContext,
                                               FENCE_FRAME_TIMEOUT_MS * 1000UL);
        if (sciErr != NvSciError_Success) {
            if (sciErr == NvSciError_Timeout) {
                LOG_ERR("Frame done NvSciSyncFenceWait timed out\n");
            } else {
                LOG_ERR("Frame done NvSciSyncFenceWait failed\n");
            }
            return NVSIPL_STATUS_ERROR;
        }
        NvSciSyncTaskStatus taskStatus;
        taskStatus.status = NvSciSyncTaskStatus_Invalid;
        sciErr = NvSciSyncFenceGetTaskStatus(&fence, &taskStatus);
        if (sciErr != NvSciError_Success) {
            LOG_ERR("NvSciSyncFenceGetTaskStatus failed\n");
            return NVSIPL_STATUS_ERROR;
        }
        LOG_INFO("Task Status was %d\n", taskStatus.status);

        if (taskStatus.status != NvSciSyncTaskStatus_Success) {
            if (taskStatus.status == NvSciSyncTaskStatus_Invalid) {
                LOG_WARN("TaskStatus was not populated by engine\n");
            } else {
                LOG_ERR("TaskStatus was not a success\n");
                return NVSIPL_STATUS_ERROR;
            }
        }

        uint64_t timestampUS = 0;
        sciErr = NvSciSyncFenceGetTimestamp(&fence, &timestampUS);
        if (sciErr != NvSciError_Success) {
            LOG_ERR("NvSciSyncFenceGetTimestamp failed\n");
            return NVSIPL_STATUS_ERROR;
        }
        LOG_INFO("Timestamp in us was %ld\n", timestampUS);

        NvSciSyncFenceClear(&fence);

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus GetCaptureDoneEventTimePair(uint64_t frameCaptureTSC, std::pair<uint64_t, std::vector<uint64_t>> &timePair)
    {
        std::unique_lock<std::mutex> lock(m_oTimePairQueueMutex);
        while (m_oTimePairQueue.empty()) {
            m_oTimePairQueueCondition.wait(lock);
            if (m_bQuit) {
                return NVSIPL_STATUS_ERROR;
            }
        }

        timePair = m_oTimePairQueue.front();
        m_oTimePairQueue.pop();
        lock.unlock();

        if (timePair.first != frameCaptureTSC) {
            LOG_ERR("Timestamps don't match: expected: %lu received: %lu\n",
                     timePair.first, frameCaptureTSC);
            return NVSIPL_STATUS_ERROR;
        }

        return NVSIPL_STATUS_OK;
    }

    void FrameWaiterThreadFunc()
    {
        while (!m_bQuit) {
            uint64_t uFrameTimeUs = 0U;
            uint64_t uCaptureDoneTimeUs = 0U;
            SIPLStatus status;

            // Wait for buffer to be delivered to thread
            std::unique_lock<std::mutex> queueLock(m_oBufferQueueMutex);
            while (m_oBufferQueue.empty()) {
                m_oBufferQueueCond.wait(queueLock);
                if (m_bQuit) {
                    return;
                }
            }

            INvSIPLClient::INvSIPLNvMBuffer *pNvMBuffer = m_oBufferQueue.front();
            m_oBufferQueue.pop();
            queueLock.unlock();

#ifdef NVMEDIA_QNX
            const INvSIPLClient::ImageMetaData &md = pNvMBuffer->GetImageData();

            uFrameTimeUs = (uint64_t)((((double_t)md.frameCaptureTSC) / m_cps) * 1000000.0);

            if (m_outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP) {
                auto status = WaitForPostFence(pNvMBuffer);
                if (status != NVSIPL_STATUS_OK) {
                    return;
                }

                std::pair<uint64_t, std::vector <uint64_t>> timePair;
                status = GetCaptureDoneEventTimePair(md.frameCaptureTSC, timePair);
                if (status != NVSIPL_STATUS_OK) {
                    return;
                }
                uCaptureDoneTimeUs = timePair.second[0];
            }
#elif !NV_IS_SAFETY
            const INvSIPLClient::ImageMetaData &md = pNvMBuffer->GetImageData();

            uFrameTimeUs = (md.frameCaptureTSC * TSC_TO_NS_MULTIPLIER) / 1000UL;

            if (m_outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP) {
                auto status = WaitForPostFence(pNvMBuffer);
                if (status != NVSIPL_STATUS_OK) {
                    return;
                }

                std::pair<uint64_t, std::vector <uint64_t>> timePair;
                status = GetCaptureDoneEventTimePair(md.frameCaptureTSC, timePair);
                if (status != NVSIPL_STATUS_OK) {
                    return;
                }
                uCaptureDoneTimeUs = timePair.second[0];
            }
#endif // !NV_IS_SAFETY

            status = pNvMBuffer->Release();
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("INvSIPLClient::INvSIPLBuffer::Release failed\n");
                return;
            }

            m_profData.profDataMut.lock();
            m_profData.uFrameCount++;
            m_profData.profDataMut.unlock();

            if (m_profileFile.is_open()) {
                if (m_outputType == INvSIPLClient::ConsumerDesc::OutputType::ICP) {
                    m_profileFile << uFrameTimeUs << ",";
                }
                else {
                    m_profileFile << uFrameTimeUs << ",";
                    m_profileFile << uCaptureDoneTimeUs << ",";
                }
            }
        }
        return;
    }

#if !defined(NVMEDIA_QNX) && !NV_IS_SAFETY
    CClockIdMgr m_clockIdMgr;
#endif
    NvSciSyncCpuWaitContext m_cpuWaitContext = nullptr;
    std::thread m_thread;

    bool m_bQuit = false;
    bool m_bEnableProfiling = false;
    uint64_t m_uPrevReceiveLatencyUs = 0U;
    uint64_t m_uPrevReadyLatencyUs = 0U;

    std::mutex m_oTimePairQueueMutex;
    std::condition_variable m_oTimePairQueueCondition;
    std::queue<std::pair<uint64_t, std::vector<uint64_t>>> m_oTimePairQueue;

    std::mutex m_oBufferQueueMutex;
    std::condition_variable m_oBufferQueueCond;
    std::queue<INvSIPLClient::INvSIPLNvMBuffer*> m_oBufferQueue;

    std::fstream m_profileFile;

#ifdef NVMEDIA_QNX
    const uint64_t m_cps = SYSPAGE_ENTRY(qtime)->cycles_per_sec;
#endif
};

#endif // CPROFILER_HPP
