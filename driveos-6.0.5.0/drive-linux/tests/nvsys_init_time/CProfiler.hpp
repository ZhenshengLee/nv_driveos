/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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
#endif // NVMEDIA_QNX

#ifndef CPROFILER_HPP
#define CPROFILER_HPP

#define NVMEDIA_IMAGE_STATUS_TIMEOUT_MS (100U)

using namespace nvsipl;

#ifndef NVMEDIA_QNX
class CClockIdMgr
{
 public:
    SIPLStatus GetClockId(NvSiplTimeBase timeBase, clockid_t &clockId)
    {
        switch (timeBase) {
            case NVSIPL_TIME_BASE_CLOCK_PTP:
// /dev/ptp0 doesnt exist when using a hardcoded config to run from systemd
// ToDo: Timebases should be removed. This is a temporary workaround
#if 0
                if (m_ptpFD < 0) {
                    m_ptpFD = open(PTP_DEV, O_RDONLY);
                    if (m_ptpFD < 0) {
                        LOG_ERR("Unable to open %s! (errno:%d)\n", PTP_DEV, errno);
                        return NVSIPL_STATUS_ERROR;
                    }
                    LOG_INFO("Using PTP clock\n");
                }
                clockId = FD2CLOCKID(m_ptpFD);
#endif
                clockId = CLOCK_MONOTONIC;
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
};

static uint64_t GetCurrentTimeUs(clock_t clockId)
{
    struct timespec currentTimespec;
    clock_gettime(clockId, &currentTimespec);
    return ((uint64_t)currentTimespec.tv_sec * 1000000) +
           ((uint64_t)currentTimespec.tv_nsec / 1000);

}
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
                    NvSciSyncModule sciSyncModule)
    {
        m_uSensor = uSensor;
        m_outputType = outputType;
        m_uInitTimeArrUs = uInitTimeArrUs;

        m_profData.profDataMut.lock();
        m_profData.uFrameCount = 0U;
        m_profData.uPrevFrameCount = 0U;
        m_profData.uFirstCaptureDelayUs = 0U;
        m_profData.uFirstReceivedDelayUs = 0U;
        m_profData.uFirstReadyDelayUs = 0U;
        m_profData.profDataMut.unlock();

        m_bQuit = false;

        auto sciErr = NvSciSyncCpuWaitContextAlloc(sciSyncModule, &m_cpuWaitContext);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncCpuWaitContextAlloc");

        m_thread = std::thread(sFrameWaiterThreadFunc, this);

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus ProfileFrame(INvSIPLClient::INvSIPLNvMBuffer *pNvMBuffer)
    {
        if (pNvMBuffer == nullptr) {
            LOG_ERR("Invalid INvSIPLClient::INvSIPLNvMBuffer pointer\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        pNvMBuffer->AddRef();

        m_oBufferQueueMutex.lock();
        m_oBufferQueue.push(pNvMBuffer);
        m_oBufferQueueMutex.unlock();
        m_oBufferQueueCond.notify_one();

        return NVSIPL_STATUS_OK;
    }

    void RecordCaptureDoneEventTime(uint64_t frameCaptureTSC, std::vector<uint64_t> uCaptureEventTimeArrUs)
    {
        m_oTimePairQueueMutex.lock();
        m_oTimePairQueue.push(std::make_pair(frameCaptureTSC, uCaptureEventTimeArrUs));
        m_oTimePairQueueMutex.unlock();

        m_oTimePairQueueCondition.notify_all();
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
        if (taskStatus.status != NvSciSyncTaskStatus_Success) {
            if (taskStatus.status == NvSciSyncTaskStatus_Invalid) {
                LOG_WARN("TaskStatus was not populated by engine\n");
            } else {
                LOG_ERR("TaskStatus was not a success\n");
                return NVSIPL_STATUS_ERROR;
            }
        }

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
            uint64_t uInitTimeUs = 0U;
            uint64_t uFrameTimeUs = 0U;
            uint64_t uReceivedTimeUs = 0U;
            uint64_t uReadyTimeUs = 0U;
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

            const INvSIPLClient::ImageMetaData &md = pNvMBuffer->GetImageData();

#ifdef NVMEDIA_QNX
            uInitTimeUs = m_uInitTimeArrUs[0];
            uFrameTimeUs = (uint64_t)((((double_t)md.frameCaptureTSC) / m_cps) * 1000000.0);
            uReceivedTimeUs = (uint64_t)((((double_t)ClockCycles()) / m_cps) * 1000000.0);

            if (m_outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP) {
                auto status = WaitForPostFence(pNvMBuffer);
                if (status != NVSIPL_STATUS_OK) {
                    return;
                }

                uReadyTimeUs = (uint64_t)((((double_t)ClockCycles()) / m_cps) * 1000000.0);

                std::pair<uint64_t, std::vector <uint64_t>> timePair;
                status = GetCaptureDoneEventTimePair(md.frameCaptureTSC, timePair);
                if (status != NVSIPL_STATUS_OK) {
                    return;
                }
            }
#elif !NV_IS_SAFETY
            uInitTimeUs = m_uInitTimeArrUs[md.timeBase];
            uFrameTimeUs = md.captureGlobalTimeStamp;

            clockid_t clockId;
            status = m_clockIdMgr.GetClockId(md.timeBase, clockId);
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("CClockIdMgr::GetClockId failed\n");
                return;
            }
            uReceivedTimeUs = GetCurrentTimeUs(clockId);

            if (m_outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP) {
                auto status = WaitForPostFence(pNvMBuffer);
                if (status != NVSIPL_STATUS_OK) {
                    return;
                }

                uReadyTimeUs = GetCurrentTimeUs(clockId);

                std::pair<uint64_t, std::vector <uint64_t>> timePair;
                status = GetCaptureDoneEventTimePair(md.frameCaptureTSC, timePair);
                if (status != NVSIPL_STATUS_OK) {
                    return;
                }
            }
#endif // !NV_IS_SAFETY

            status = pNvMBuffer->Release();
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("INvSIPLClient::INvSIPLBuffer::Release failed\n");
                return;
            }

            m_profData.profDataMut.lock();
            if (m_profData.uFrameCount == 0U) {
                m_profData.uFirstCaptureDelayUs = uFrameTimeUs - uInitTimeUs;
                m_profData.uFirstReceivedDelayUs = uReceivedTimeUs - uInitTimeUs;
                if (m_outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP) {
                    m_profData.uFirstReadyDelayUs = uReadyTimeUs - uInitTimeUs;
                } else {
                    // For capture output, since the image is ready when received.
                    // First ready delay is same as first received delay.
                    m_profData.uFirstReadyDelayUs = m_profData.uFirstReceivedDelayUs;
                }
            }
            m_profData.uFrameCount++;
            m_profData.profDataMut.unlock();
        }
        return;
    }

#if !defined(NVMEDIA_QNX) && !NV_IS_SAFETY
    CClockIdMgr m_clockIdMgr;
#endif
    std::vector<uint64_t> m_uInitTimeArrUs;
    NvSciSyncCpuWaitContext m_cpuWaitContext = nullptr;
    std::thread m_thread;

    bool m_bQuit = false;
    uint64_t m_uPrevReceiveLatencyUs = 0U;
    uint64_t m_uPrevReadyLatencyUs = 0U;

    std::mutex m_oTimePairQueueMutex;
    std::condition_variable m_oTimePairQueueCondition;
    std::queue<std::pair<uint64_t, std::vector<uint64_t>>> m_oTimePairQueue;

    std::mutex m_oBufferQueueMutex;
    std::condition_variable m_oBufferQueueCond;
    std::queue<INvSIPLClient::INvSIPLNvMBuffer*> m_oBufferQueue;

#ifdef NVMEDIA_QNX
    const uint64_t m_cps = SYSPAGE_ENTRY(qtime)->cycles_per_sec;
#endif
};

#endif // CPROFILER_HPP
