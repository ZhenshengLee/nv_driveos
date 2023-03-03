/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/**
 * @file
 * <b>NvSIPL Camera sample application. </b>
 *
 * @b Description: This application demonstrates the usage of NvSIPL APIs to,
 *  1. Select a pre-defined platform configuration using Query APIs.
 *  2. Create and configure camera pipelines using Camera APIs.
 *  3. Create an NvSIPL client to consume images using NvSIPL Client APIs.
 *  4. Implement callbacks to receive and process the outputs a pipeline using Client APIs.
 */

/* STL Headers */
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <csignal>
#include <thread>
#include <chrono>
#include <ctime>
#include <atomic>
#include <cmath>
#include <pthread.h>
#include <dlfcn.h>
#include <sys/time.h>

/* NvSIPL Headers */
#include "NvSIPLVersion.hpp" // Version
#include "NvSIPLTrace.hpp" // Trace
#include "NvSIPLCommon.hpp" // Common
#include "NvSIPLQuery.hpp" // Query
#include "NvSIPLQueryTrace.hpp" // Query Trace
#include "NvSIPLCamera.hpp" // Camera
#include "NvSIPLPipelineMgr.hpp" // Pipeline manager
#include "NvSIPLClient.hpp" // Client

#include "CNvSIPLMaster.hpp"
#include "CNvSIPLMasterNvSci.hpp"
#include "CProfiler.hpp"
#include "CNvSIPLConsumer.hpp"
#include "CCmdLineParser.hpp"
#include "CAutoRecovery.hpp"
#include "CFileReader.hpp"
#include "CFrameFeeder.hpp"
#if !NV_IS_SAFETY
#include "CFileWriter.hpp"
#include "CComposite.hpp"
#include "CCompositeNvSci.hpp"
#endif // !NV_IS_SAFETY

#ifdef NVMEDIA_QNX
#include "nvdvms_client.h"
#endif // NVMEDIA_QNX

#define BUFFER_8MP (8000000U)
#define MAX_NUM_SENSORS (16U)
#define SECONDS_PER_ITERATION (2)
#define EVENT_QUEUE_TIMEOUT_US (1000000U)
#define MAX_LINE_LENGTH (256U)

/* Quit flag. */
std::atomic<bool> bQuit;

/* Ignore Error flag. */
std::atomic<bool> bIgnoreError;

/* Ignore Error status enum for Link Recovery. */
enum eLinkErrorIgnoreStatus {
    DontIgnore = 0,
    StartIgnore,
    StopIgnore
};

/* Ignore Error status array for Link Recovery. */
std::atomic<eLinkErrorIgnoreStatus> eLinkErrorIgnoreArr[MAX_NUM_SENSORS];

/* SIPL Master. */
std::unique_ptr<CNvSIPLMaster> upMaster(nullptr);

/* Error reported by library via callbacks */
std::unique_ptr<CAutoRecovery> m_upAutoRecovery {nullptr};

/* Dynamic library handle */
void *pluginlib_handle = NULL;

/** Signal handler.*/
static void SigHandler(int signum)
{
    LOG_WARN("Received signal: %u. Quitting\n", signum);
    signal(SIGINT, SIG_IGN);
    signal(SIGTERM, SIG_IGN);
    signal(SIGQUIT, SIG_IGN);
    signal(SIGHUP, SIG_IGN);

    bQuit = true;

    signal(SIGINT, SIG_DFL);
    signal(SIGTERM, SIG_DFL);
    signal(SIGQUIT, SIG_DFL);
    signal(SIGHUP, SIG_DFL);
}

/** Sets up signal handler.*/
static void SigSetup(void)
{
    struct sigaction action { };
    action.sa_handler = SigHandler;

    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);
    sigaction(SIGQUIT, &action, NULL);
    sigaction(SIGHUP, &action, NULL);
}

class CDeviceBlockNotificationHandler : public NvSIPLPipelineNotifier
{
public:
    uint32_t m_uDevBlkIndex = -1U;

    SIPLStatus Init(uint32_t uDevBlkIndex, DeviceBlockInfo &oDeviceBlockInfo,
                    INvSIPLNotificationQueue *notificationQueue)
    {
        if (notificationQueue == nullptr) {
            LOG_ERR("Invalid Notification Queue\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }
        m_uDevBlkIndex = uDevBlkIndex;
        m_oDeviceBlockInfo = oDeviceBlockInfo;
        m_pNotificationQueue = notificationQueue;

        SIPLStatus status = upMaster->GetMaxErrorSize(m_uDevBlkIndex, m_uErrorSize);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("DeviceBlock: %u, GetMaxErrorSize failed\n", m_uDevBlkIndex);
            return status;
        }

        if (m_uErrorSize != 0U) {
            m_oDeserializerErrorInfo.upErrorBuffer.reset(new uint8_t[m_uErrorSize]);
            m_oDeserializerErrorInfo.bufferSize = m_uErrorSize;

            m_oSerializerErrorInfo.upErrorBuffer.reset(new uint8_t[m_uErrorSize]);
            m_oSerializerErrorInfo.bufferSize = m_uErrorSize;

            m_oSensorErrorInfo.upErrorBuffer.reset(new uint8_t[m_uErrorSize]);
            m_oSensorErrorInfo.bufferSize = m_uErrorSize;
        }

        m_upThread.reset(new std::thread(EventQueueThreadFunc, this));
        return NVSIPL_STATUS_OK;
    }

    void Deinit()
    {
        m_bQuit = true;
        if (m_upThread != nullptr) {
            m_upThread->join();
            m_upThread.reset();
        }
    }

    bool IsDeviceBlockInError() {
        return m_bInError;
    }

    virtual ~CDeviceBlockNotificationHandler()
    {
        Deinit();
    }

private:
    void HandleDeserializerError()
    {
        bool isRemoteError {false};
        uint8_t linkErrorMask {0U};

        /* Get detailed error information (if error size is non-zero) and
         * information about remote error and link error. */
        SIPLStatus status = upMaster->GetDeserializerErrorInfo(
                                        m_uDevBlkIndex,
                                        (m_uErrorSize > 0) ? &m_oDeserializerErrorInfo : nullptr,
                                        isRemoteError,
                                        linkErrorMask);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("DeviceBlock: %u, GetDeserializerErrorInfo failed\n", m_uDevBlkIndex);
            m_bInError = true;
            return;
        }

        if ((m_uErrorSize > 0) && (m_oDeserializerErrorInfo.sizeWritten != 0)) {
            cout << "DeviceBlock[" << m_uDevBlkIndex << "] Deserializer Error Buffer: ";
            for (uint32_t i = 0; i < m_oDeserializerErrorInfo.sizeWritten; i++) {
                printf("0x%x ", m_oDeserializerErrorInfo.upErrorBuffer[i]);
            }
            printf("\n");

            if (!m_upAutoRecovery) {
                m_bInError = true;
            }
        }

        if (isRemoteError) {
            cout << "DeviceBlock[" << m_uDevBlkIndex << "] Deserializer Remote Error.\n";
            for (uint32_t i = 0; i < m_oDeviceBlockInfo.numCameraModules; i++) {
                HandleCameraModuleError(m_oDeviceBlockInfo.cameraModuleInfoList[i].sensorInfo.id);
            }
        }

        if (linkErrorMask != 0U) {
            if (m_upAutoRecovery != nullptr) {
                for (uint32_t i = 0; i < m_oDeviceBlockInfo.numCameraModules; i++) {
                    if ((linkErrorMask & (1 << (m_oDeviceBlockInfo.cameraModuleInfoList[i].linkIndex))) != 0) {
                        m_upAutoRecovery->OnLinkFailure(m_oDeviceBlockInfo.cameraModuleInfoList[i].sensorInfo.id);
                    }
                }
            }
            else {
                LOG_ERR("DeviceBlock: %u, Deserializer link error. mask: %u\n", m_uDevBlkIndex, linkErrorMask);
                m_bInError = true;
            }
        }
    }

    void HandleCameraModuleError(uint32_t index)
    {
        if (m_uErrorSize > 0) {
            /* Get detailed error information. */
            SIPLStatus status = upMaster->GetModuleErrorInfo(
                                            index,
                                            &m_oSerializerErrorInfo,
                                            &m_oSensorErrorInfo);
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("index: %u, GetModuleErrorInfo failed\n", index);
                if (!m_upAutoRecovery) {
                    m_bInError = true;
                }
                return;
            }

            if (m_oSerializerErrorInfo.sizeWritten != 0) {
                cout << "Pipeline[" << index << "] Serializer Error Buffer: ";
                for (uint32_t i = 0; i < m_oSerializerErrorInfo.sizeWritten; i++) {
                    printf("0x%x ", m_oSerializerErrorInfo.upErrorBuffer[i]);
                }
                printf("\n");
                if (!m_upAutoRecovery) {
                    m_bInError = true;
                }
            }

            if (m_oSensorErrorInfo.sizeWritten != 0) {
                cout << "Pipeline[" << index << "] Sensor Error Buffer: ";
                for (uint32_t i = 0; i < m_oSensorErrorInfo.sizeWritten; i++) {
                    printf("0x%x ", m_oSensorErrorInfo.upErrorBuffer[i]);
                }
                printf("\n");
                if (!m_upAutoRecovery) {
                    m_bInError = true;
                }
            }
        }
    }

    bool isTrueGPIOInterrupt(const uint32_t *gpioIdxs, uint32_t numGpioIdxs)
    {
        /*
         * Get disambiguated GPIO interrupt event codes, to determine whether
         * true interrupts or propagation functionality fault occurred.
         */

        bool true_interrupt = false;

        for (uint32_t i = 0U; i < numGpioIdxs; i++) {
            SIPLGpioEvent code;
            SIPLStatus status = upMaster->GetErrorGPIOEventInfo(m_uDevBlkIndex,
                                                                gpioIdxs[i],
                                                                code);
            if (status == NVSIPL_STATUS_NOT_SUPPORTED) {
                LOG_INFO("GetErrorGPIOEventInfo is not supported by OS backend currently!\n");
                /* Allow app to fetch detailed error info, same as in case of true interrupt. */
                return true;
            } else if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("DeviceBlock: %u, GetErrorGPIOEventInfo failed\n", m_uDevBlkIndex);
                m_bInError = true;
                return false;
            }

            /*
             * If no error condition code is returned, and at least one GPIO has
             * NVSIPL_GPIO_EVENT_INTR status, return true.
             */
            if (code == NVSIPL_GPIO_EVENT_INTR) {
                true_interrupt = true;
            } else if (code != NVSIPL_GPIO_EVENT_NOTHING) {
                // GPIO functionality fault (treat as fatal)
                m_bInError = true;
                return false;
            }
        }

        return true_interrupt;
    }

    //! Notifier function
    void OnEvent(NotificationData &oNotificationData)
    {
        switch (oNotificationData.eNotifType) {
        case NOTIF_ERROR_DESERIALIZER_FAILURE:
            LOG_ERR("DeviceBlock: %u, NOTIF_ERROR_DESERIALIZER_FAILURE\n", m_uDevBlkIndex);
            if (!bIgnoreError) {
                if (isTrueGPIOInterrupt(oNotificationData.gpioIdxs, oNotificationData.numGpioIdxs)) {
                    HandleDeserializerError();
                }
            }
            break;
        case NOTIF_ERROR_SERIALIZER_FAILURE:
            LOG_ERR("DeviceBlock: %u, NOTIF_ERROR_SERIALIZER_FAILURE\n", m_uDevBlkIndex);
            if (!bIgnoreError) {
                for (uint32_t i = 0; i < m_oDeviceBlockInfo.numCameraModules; i++) {
                    if (isTrueGPIOInterrupt(oNotificationData.gpioIdxs, oNotificationData.numGpioIdxs)) {
                        HandleCameraModuleError(m_oDeviceBlockInfo.cameraModuleInfoList[i].sensorInfo.id);
                    }
                }
            }
            break;
        case NOTIF_ERROR_SENSOR_FAILURE:
            LOG_ERR("DeviceBlock: %u, NOTIF_ERROR_SENSOR_FAILURE\n", m_uDevBlkIndex);
            if (!bIgnoreError) {
                for (uint32_t i = 0; i < m_oDeviceBlockInfo.numCameraModules; i++) {
                    if (isTrueGPIOInterrupt(oNotificationData.gpioIdxs, oNotificationData.numGpioIdxs)) {
                        HandleCameraModuleError(m_oDeviceBlockInfo.cameraModuleInfoList[i].sensorInfo.id);
                    }
                }
            }
            break;
        case NOTIF_ERROR_INTERNAL_FAILURE:
            LOG_ERR("DeviceBlock: %u, NOTIF_ERROR_INTERNAL_FAILURE\n", m_uDevBlkIndex);
            m_bInError = true;
            break;
        default:
            LOG_WARN("DeviceBlock: %u, Unknown/Invalid notification\n", m_uDevBlkIndex);
            break;
        }
        return;
    }

    static void EventQueueThreadFunc(CDeviceBlockNotificationHandler *pThis)
    {
        SIPLStatus status = NVSIPL_STATUS_OK;
        NotificationData notificationData;

        if ((pThis == nullptr) || (pThis->m_pNotificationQueue == nullptr)) {
            LOG_ERR("Invalid thread data\n");
            return;
        }

        pthread_setname_np(pthread_self(), "DevBlkEvent");

        while (!pThis->m_bQuit) {
            status = pThis->m_pNotificationQueue->Get(notificationData, EVENT_QUEUE_TIMEOUT_US);
            if (status == NVSIPL_STATUS_OK) {
                pThis->OnEvent(notificationData);
            } else if (status == NVSIPL_STATUS_TIMED_OUT) {
                LOG_DBG("Queue timeout\n");
            } else if (status == NVSIPL_STATUS_EOF) {
                LOG_DBG("Queue shutdown\n");
                pThis->m_bQuit = true;
            } else {
                LOG_ERR("Unexpected queue return status\n");
                pThis->m_bQuit = true;
            }
        }
    }

    bool m_bQuit = false;
    bool m_bInError = false;
    std::unique_ptr<std::thread> m_upThread = nullptr;
    INvSIPLNotificationQueue *m_pNotificationQueue = nullptr;
    DeviceBlockInfo m_oDeviceBlockInfo;

    size_t m_uErrorSize {};
    SIPLErrorDetails m_oDeserializerErrorInfo {};
    SIPLErrorDetails m_oSerializerErrorInfo {};
    SIPLErrorDetails m_oSensorErrorInfo {};
};

class CPipelineNotificationHandler : public NvSIPLPipelineNotifier
{
public:
    uint32_t m_uSensor = -1U;

    //! Initializes the Pipeline Notification Handler
    SIPLStatus Init(uint32_t uSensor,
                    INvSIPLNotificationQueue *notificationQueue,
                    std::vector<CProfiler*> vpProfilerISP)
    {
        if (notificationQueue == nullptr) {
            LOG_ERR("Invalid Notification Queue\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }
        m_uSensor = uSensor;
        m_pNotificationQueue = notificationQueue;
        m_vpProfilerISP = vpProfilerISP;
        m_upThread.reset(new std::thread(EventQueueThreadFunc, this));

#if !defined(NVMEDIA_QNX) && !NV_IS_SAFETY
        SIPLStatus status = m_clockIdMgr.GetClockId(NVSIPL_TIME_BASE_CLOCK_PTP, m_ptpClockId);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("CClockIdMgr::GetClockId failed\n");
            return status;
        }
#endif

        return NVSIPL_STATUS_OK;
    }

    void Deinit()
    {
        m_bQuit = true;
        if (m_upThread != nullptr) {
            m_upThread->join();
            m_upThread.reset();
        }
    }

    //! Returns true to pipeline encountered any fatal error.
    bool IsPipelineInError(void)
    {
        return m_bInError;
    }

    //! Get number of frame drops for this pipeline
    uint32_t GetNumFrameDrops()
    {
        return m_uNumFrameDrops;
    }

    //! Get number of frame discontinuities for this pipeline
    uint32_t GetNumFrameDiscontinuities()
    {
        return m_uNumFrameDiscontinuities;
    }

    virtual ~CPipelineNotificationHandler()
    {
        Deinit();
    }

private:
    //! Notifier function
    void OnEvent(NotificationData &oNotificationData)
    {
        std::vector<uint64_t> uCaptureDoneTimeUs = {0U, 0U};
        eLinkErrorIgnoreStatus ignoreStatus = StopIgnore;

        switch (oNotificationData.eNotifType) {
        case NOTIF_INFO_ICP_PROCESSING_DONE:
#ifdef NVMEDIA_QNX
            uCaptureDoneTimeUs[0] = (uint64_t)((((double_t)ClockCycles()) / m_cps) * 1000000.0);
#elif !NV_IS_SAFETY
            struct timespec currentTimespec;
            clock_gettime(m_ptpClockId, &currentTimespec);
            uCaptureDoneTimeUs[NVSIPL_TIME_BASE_CLOCK_PTP] = ((uint64_t)currentTimespec.tv_sec * 1000000) +
                                                             ((uint64_t)currentTimespec.tv_nsec / 1000);

            clock_gettime(CLOCK_MONOTONIC, &currentTimespec);
            uCaptureDoneTimeUs[NVSIPL_TIME_BASE_CLOCK_MONOTONIC] = ((uint64_t)currentTimespec.tv_sec * 1000000) +
                                                                   ((uint64_t)currentTimespec.tv_nsec / 1000);
#endif // NVMEDIA_QNX

            for (auto &pProfilerISP : m_vpProfilerISP) {
                pProfilerISP->RecordCaptureDoneEventTime(oNotificationData.frameCaptureTSC, uCaptureDoneTimeUs);
            }
            LOG_INFO("Pipeline: %u, NOTIF_INFO_ICP_PROCESSING_DONE\n", oNotificationData.uIndex);

            if (eLinkErrorIgnoreArr[oNotificationData.uIndex].compare_exchange_strong(ignoreStatus, DontIgnore)) {
                LOG_INFO("Pipeline: %u, Stop ignoring bad input stream error. Current state=%d\n",ignoreStatus);
            }
            break;
        case NOTIF_INFO_ISP_PROCESSING_DONE:
            LOG_INFO("Pipeline: %u, NOTIF_INFO_ISP_PROCESSING_DONE\n", oNotificationData.uIndex);
            break;
        case NOTIF_INFO_ACP_PROCESSING_DONE:
            LOG_INFO("Pipeline: %u, NOTIF_INFO_ACP_PROCESSING_DONE\n", oNotificationData.uIndex);
            break;
        case NOTIF_INFO_CDI_PROCESSING_DONE:
            LOG_INFO("Pipeline: %u, NOTIF_INFO_CDI_PROCESSING_DONE\n", oNotificationData.uIndex);
            break;
        case NOTIF_WARN_ICP_FRAME_DROP:
            LOG_WARN("Pipeline: %u, NOTIF_WARN_ICP_FRAME_DROP\n", oNotificationData.uIndex);
            m_uNumFrameDrops++;
            break;
        case NOTIF_WARN_ICP_FRAME_DISCONTINUITY:
            LOG_WARN("Pipeline: %u, NOTIF_WARN_ICP_FRAME_DISCONTINUITY\n", oNotificationData.uIndex);
            m_uNumFrameDiscontinuities++;
            break;
        case NOTIF_WARN_ICP_CAPTURE_TIMEOUT:
            LOG_WARN("Pipeline: %u, NOTIF_WARN_ICP_CAPTURE_TIMEOUT\n", oNotificationData.uIndex);
            break;
        case NOTIF_ERROR_ICP_BAD_INPUT_STREAM:
            LOG_ERR("Pipeline: %u, NOTIF_ERROR_ICP_BAD_INPUT_STREAM\n", oNotificationData.uIndex);
            if (m_upAutoRecovery == nullptr and !bIgnoreError and
               (eLinkErrorIgnoreArr[oNotificationData.uIndex].load() == DontIgnore)) {
                m_bInError = true; // Treat this as fatal error only if state is DontIgnore.
            }
            break;
        case NOTIF_ERROR_ICP_CAPTURE_FAILURE:
            LOG_ERR("Pipeline: %u, NOTIF_ERROR_ICP_CAPTURE_FAILURE\n", oNotificationData.uIndex);
            m_bInError = true;
            break;
        case NOTIF_ERROR_ICP_EMB_DATA_PARSE_FAILURE:
            LOG_ERR("Pipeline: %u, NOTIF_ERROR_ICP_EMB_DATA_PARSE_FAILURE\n", oNotificationData.uIndex);
            m_bInError = true;
            break;
        case NOTIF_ERROR_ISP_PROCESSING_FAILURE:
            LOG_ERR("Pipeline: %u, NOTIF_ERROR_ISP_PROCESSING_FAILURE\n", oNotificationData.uIndex);
            m_bInError = true;
            break;
        case NOTIF_ERROR_ISP_PROCESSING_FAILURE_RECOVERABLE:
            LOG_WARN("Pipeline: %u, NOTIF_ERROR_ISP_PROCESSING_FAILURE_RECOVERABLE\n", oNotificationData.uIndex);
            break;
        case NOTIF_ERROR_ACP_PROCESSING_FAILURE:
            LOG_ERR("Pipeline: %u, NOTIF_ERROR_ACP_PROCESSING_FAILURE\n", oNotificationData.uIndex);
            m_bInError = true;
            break;
        case NOTIF_ERROR_CDI_SET_SENSOR_CTRL_FAILURE:
            LOG_ERR("Pipeline: %u, NOTIF_ERROR_CDI_SET_SENSOR_CTRL_FAILURE\n", oNotificationData.uIndex);
            if (m_upAutoRecovery == nullptr and !bIgnoreError) {
                m_bInError = true; // Treat this as fatal error only if link recovery is not enabled.
            }
            break;
        case NOTIF_ERROR_INTERNAL_FAILURE:
            LOG_ERR("Pipeline: %u, NOTIF_ERROR_INTERNAL_FAILURE\n", oNotificationData.uIndex);
            m_bInError = true;
            break;
        default:
            LOG_WARN("Pipeline: %u, Unknown/Invalid notification\n", oNotificationData.uIndex);
            break;
        }

        return;
    }

    static void EventQueueThreadFunc(CPipelineNotificationHandler *pThis)
    {
        SIPLStatus status = NVSIPL_STATUS_OK;
        NotificationData notificationData;

        if ((pThis == nullptr) || (pThis->m_pNotificationQueue == nullptr)) {
            LOG_ERR("Invalid thread data\n");
            return;
        }

        pthread_setname_np(pthread_self(), "PipelineEvent");

        while ((!pThis->m_bQuit) || (pThis->m_pNotificationQueue->GetCount() > 0U)) {
            status = pThis->m_pNotificationQueue->Get(notificationData, EVENT_QUEUE_TIMEOUT_US);
            if (status == NVSIPL_STATUS_OK) {
                pThis->OnEvent(notificationData);
            } else if (status == NVSIPL_STATUS_TIMED_OUT) {
                LOG_DBG("Queue timeout\n");
            } else if (status == NVSIPL_STATUS_EOF) {
                LOG_DBG("Queue shutdown\n");
                pThis->m_bQuit = true;
            } else {
                LOG_ERR("Unexpected queue return status\n");
                pThis->m_bQuit = true;
            }
        }
    }

#ifdef NVMEDIA_QNX
    const uint64_t m_cps = SYSPAGE_ENTRY(qtime)->cycles_per_sec;
#elif !NV_IS_SAFETY
    CClockIdMgr m_clockIdMgr;
    clockid_t m_ptpClockId;
#endif

    uint32_t m_uNumFrameDrops = 0U;
    uint32_t m_uNumFrameDiscontinuities = 0U;
    bool m_bInError = false;
    std::unique_ptr<std::thread> m_upThread = nullptr;
    INvSIPLNotificationQueue *m_pNotificationQueue = nullptr;
    std::vector<CProfiler*> m_vpProfilerISP;
    bool m_bQuit = false;
};

class CPipelineFrameQueueHandler
{
public:
    uint32_t m_uSensor = -1U;

    //! Initializes the Pipeline Frame Queue Handler
    SIPLStatus Init(uint32_t uSensor,
                    std::vector<std::pair<INvSIPLFrameCompletionQueue *,
                    CNvSIPLConsumer*>> &vQueueConsumerPair,
                    NvSciSyncModule sciSyncModule)
    {
        m_uSensor = uSensor;
        m_vQueueConsumerPair = vQueueConsumerPair;

        auto sciErr = NvSciSyncCpuWaitContextAlloc(sciSyncModule, &m_cpuWaitContext);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncCpuWaitContextAlloc");

        m_upThread.reset(new std::thread(FrameCompletionQueueThreadFunc, this));

        return NVSIPL_STATUS_OK;
    }

    void Deinit()
    {
        m_bQuit = true;
        if (m_upThread != nullptr) {
            m_upThread->join();
            m_upThread.reset();
        }

        if (m_cpuWaitContext != nullptr) {
            NvSciSyncCpuWaitContextFree(m_cpuWaitContext);
            m_cpuWaitContext = nullptr;
        }
    }

    static void FrameCompletionQueueThreadFunc(CPipelineFrameQueueHandler *pThis)
    {
        SIPLStatus status = NVSIPL_STATUS_OK;
        INvSIPLClient::INvSIPLBuffer *pBuffer = nullptr;
        bool bFramesAvailable = true;

        pthread_setname_np(pthread_self(), "FrameQueue");

        while ((!pThis->m_bQuit) || bFramesAvailable) {
            bFramesAvailable = false;
            for (auto &queueConsPair : pThis->m_vQueueConsumerPair) {
                status = queueConsPair.first->Get(pBuffer, IMAGE_QUEUE_TIMEOUT_US);
                if (status == NVSIPL_STATUS_OK) {
                    status = queueConsPair.second->OnFrameAvailable(pBuffer,
                                                                    pThis->m_cpuWaitContext);
                    if (status != NVSIPL_STATUS_OK) {
                        LOG_ERR("OnFrameAvailable failed. (status:%u)\n", status);
                        pThis->m_bQuit = true;
                        pBuffer->Release();
                        return;
                    }
#if !NV_IS_SAFETY
                    if(queueConsPair.second->IsLEDControlEnabled()) {
                        uint32_t uSensor = queueConsPair.second->m_uSensor;
                        if ((queueConsPair.second->IsLEDEnabled()) && (!queueConsPair.second->IsPrevFrameLEDEnabled())) {
                            upMaster->ToggleLED(uSensor, true);
                            cout << "LED toggled: ON, sensor: " << uSensor << endl;
                        } else if ((!queueConsPair.second->IsLEDEnabled()) && (queueConsPair.second->IsPrevFrameLEDEnabled())) {
                            upMaster->ToggleLED(uSensor, false);
                            cout << "LED toggled: OFF, sensor: " << uSensor << endl;
                        }
                    }
#endif //!NV_IS_SAFETY
                    status = pBuffer->Release();
                    if (status != NVSIPL_STATUS_OK) {
                        LOG_ERR("Buffer release failed\n");
                        pThis->m_bQuit = true;
                        return;
                    }
                } else if (status == NVSIPL_STATUS_TIMED_OUT) {
                    LOG_DBG("Queue timeout\n");
                } else if (status == NVSIPL_STATUS_EOF) {
                    LOG_DBG("Queue shutdown\n");
                    pThis->m_bQuit = true;
                    return;
                } else {
                    LOG_ERR("Unexpected queue return status: %u\n", status);
                    pThis->m_bQuit = true;
                    return;
                }
                if (queueConsPair.first->GetCount() > 0U) {
                    bFramesAvailable = true;
                }
            }
        }
    }

    virtual ~CPipelineFrameQueueHandler()
    {
        Deinit();
    }

    std::unique_ptr<std::thread> m_upThread = nullptr;
    std::vector<std::pair<INvSIPLFrameCompletionQueue *, CNvSIPLConsumer *>> m_vQueueConsumerPair;
    NvSciSyncCpuWaitContext m_cpuWaitContext = nullptr;
    bool m_bQuit = false;
};

static SIPLStatus GetOV2311CustomInterface(char *line) {
    SIPLStatus status = NVSIPL_STATUS_OK;
    uint32_t param = 0U;
    uint32_t i = 0U;

    if ((line[0] == 'g') && (line[1] == 'c')) {
        i = 2U; /* Starting point on the line */
    }

    // Obtain sensor ID from provided string
    while (line[i] != '\0' && i < MAX_LINE_LENGTH) {
        if ((line[i] >= '0') && (line[i] <= '9')) {
            param = (param * 10U) + (uint32_t)(line[i] - '0');
        } else {
            // else ignore all other characters
        }
        i++;
    }

    if (param < MAX_CAMERAMODULES_PER_PLATFORM) {
        // Get custom interface for user specified sensor ID
        upMaster->ov2311NonFuSaCustomInterface =
            upMaster->GetOV2311NonFuSaCustomInterface(param);
        if (upMaster->ov2311NonFuSaCustomInterface != nullptr) {
            LOG_DBG("OV2311 custom interface found\n");
            status = NVSIPL_STATUS_OK;
        } else {
            LOG_ERR("Failed to obtain OV2311 custom interface\n");
            status = NVSIPL_STATUS_ERROR;
        }
    } else {
        LOG_ERR("The wrong sensor ID used : %d\n", param);
    }
printf("++ %s %d\n", __func__, __LINE__);
    return status;
}

static void processOV2311CustomCommand(char *line) {
    SIPLStatus status = NVSIPL_STATUS_OK;
    uint32_t param = 0U;
    uint32_t readBackValue = 0;
    uint32_t i = 0U;

    if (upMaster->ov2311NonFuSaCustomInterface != nullptr) {
        // Get custom interface
        OV2311NonFuSaCustomInterface* ov2311NonFuSaCustomInterface =
            upMaster->ov2311NonFuSaCustomInterface;

        if ((line[0] == 'c') && (line[1] == 'u') && (line[2] == 's') && (line[3] == 't')) {
            i = 4U; /* Starting point on the line */
            // Obtain set value from provided string
            while (line[i] != '\0' && i < MAX_LINE_LENGTH) {
                if ((line[i] >= '0') && (line[i] <= '9')) {
                    param = (param * 10U) + (uint32_t)(line[i] - '0');
                } else {
                    // else ignore all other characters
                }
                i++;
            }

            LOG_DBG("OV2311 custom interface found\n");
            status = ov2311NonFuSaCustomInterface->SetCustomValue(param);
            if (status == NVSIPL_STATUS_OK) {
                // Read back to validate value set
                std::this_thread::sleep_for(std::chrono::milliseconds(66));

                status = ov2311NonFuSaCustomInterface->GetCustomValue(&readBackValue);
                if (status == NVSIPL_STATUS_OK) {
                    if (readBackValue == param) {
                        LOG_MSG("Custom value set successfully\n");
                    } else {
                        LOG_ERR("Incorrect value set: %u", readBackValue);
                    }
                } else {
                    LOG_ERR("Failed to read back custom value\n");
                }
            }  else {
                LOG_ERR("Failed to set custom value\n");
            }
        } else if ((line[0] == 'c') && (line[1] == 'm')) {
            if (ov2311NonFuSaCustomInterface != nullptr) {
                LOG_DBG("OV2311 custom interface found\n");
                status = ov2311NonFuSaCustomInterface->CheckModuleStatus();
                if (status == NVSIPL_STATUS_OK) {
                    LOG_MSG("Detected the module successfully\n");
                }  else {
                    LOG_ERR("Failed to detect the module\n");
                }
            } else {
                LOG_ERR("Failed to obtain OV2311 custom interface\n");
            }
        }
    } else {
        LOG_ERR("Failed to obtain OV2311 custom interface\n");
    }
}

#if !NV_IS_SAFETY
/**
 * SIPL Fetch NITO Metadata Interface Sample Usage
 */
static SIPLStatus callFetchNITOMetadataAPI(const std::vector<uint8_t>& nitoFile,
                                           const size_t numParameterSets) {

    SIPLStatus status = NVSIPL_STATUS_OK;
    /**
     * Setup API Arguments:
     * 1. nitoFile loaded into memory - using LoadNITOFile() function in CUtils.hpp/cpp, NITO file
     * was loaded into an std::vector<uint_8> nitoFile. Such functionality is required to
     * use the API.
     * 2. nitoFile memory length - in this case, provided through std::vector::size.
     * 3. NvSIPLNitoMetadata struct array, mdArray - instantiation below.
     * Note: Length of mdArray must be >= number of parameter sets in NITO file, else API will throw
     * error.
     * Note: By default, this is set to 10u. Client has option to specify this via
     * command line argument --showNitoMetadata, --numParameterSetsInNITO.
     * 4. mdArray length - in this case, provided through numParameterSets.
     * 5. outCount variable - instantiated.
     */

    // Instantiate metadata array to be populated by API, using maximum number of parameter sets in
    // NITO file.
    NvSIPLNitoMetadata mdArray[numParameterSets];
    // Counter, populated by API to indicate how many parameter sets were read.
    size_t outCount = 0U;

    /**
     * Calling the API entrypoint, GetNitoMetadataFromMemory.
     */
    status = GetNitoMetadataFromMemory(nitoFile.data(), nitoFile.size(),
                                       mdArray, numParameterSets, &outCount);
    if (status != NVSIPL_STATUS_OK)
    {
        LOG_ERR("Failed to retrieve metadata from NITO file \n");
        return status;
    }

    /**
     * Output metadata to console in human readable format.
     * Parameter Set ID Format: 00000000-0000-0000-0000-000000000000 in hex
     * Hash Format: Hex string with no spaces
     */
    LOG_MSG("Total number of knobsets in NITO file is %lu \n", outCount);
    for (auto i = 0U; i < outCount; ++i)
    {
        cout << "For Parameter Set:" << i + 1U << endl;
        cout << "Parameter Set ID" << endl;
        status = PrintParameterSetID(mdArray[i].parameterSetID,
                                     nvsipl::NITO_PARAMETER_SET_ID_SIZE);
        if (status != NVSIPL_STATUS_OK)
        {
            LOG_ERR("Failed to print NITO Parameter Set ID \n");
            return status;
        }

        cout << "Parameter Set Schema Hash" << endl;
        status = PrintParameterSetSchemaHash(mdArray[i].schemaHash,
                                             nvsipl::NITO_SCHEMA_HASH_SIZE);
        if (status != NVSIPL_STATUS_OK)
        {
            LOG_ERR("Failed to print NITO Schema Hash \n");
            return status;
        }

        cout << "Parameter Set Data Hash" << endl;
        status = PrintParameterSetDataHash(mdArray[i].dataHash,
                                           nvsipl::NITO_DATA_HASH_SIZE);
        if (status != NVSIPL_STATUS_OK)
        {
            LOG_ERR("Failed to print NITO Data Hash \n");
            return status;
        }
    }
    cout << "\n";
    return status;
}
#endif // !NV_IS_SAFETY

#ifdef NVMEDIA_QNX
static inline uint32_t sec_to_ms(uint32_t sec)
{
    return (sec * 1000);
}
static inline uint32_t us_to_ms(uint32_t us)
{
    const double MICROSECONDS_TO_MILLISECONDS = 1.0 / 1000.0;

    //Precise within 1,000 microseconds.
    return MICROSECONDS_TO_MILLISECONDS*us;
}
static void suspend_drive_os(void)
{
    struct timeval start, finish, delta;
    uint32_t time_in_suspend_ms;

    std::cout << "Suspending DriveOS..." << std::endl;

    gettimeofday(&start, NULL);
    nvdvms_set_vm_state(NVDVMS_SUSPEND);
    gettimeofday(&finish, NULL);

    timersub(&finish, &start, &delta);

    time_in_suspend_ms = sec_to_ms(delta.tv_sec);
    time_in_suspend_ms += us_to_ms(delta.tv_usec);

    std::cout << "DriveOs resumed, DriveOS suspended for " << time_in_suspend_ms << "ms" << std::endl;
}
#endif //NVMEDIA_QNX
int main(int argc, char *argv[])
{
    pthread_setname_np(pthread_self(), "Main");

    bQuit = false;

    for(auto linkIndex = 0U;  linkIndex < MAX_NUM_SENSORS; linkIndex++) {
        eLinkErrorIgnoreArr[linkIndex].store(DontIgnore);
    }

    LOG_INFO("Checking SIPL version\n");
    NvSIPLVersion oVer;
    NvSIPLGetVersion(oVer);

    LOG_INFO("NvSIPL library version: %u.%u.%u\n", oVer.uMajor, oVer.uMinor, oVer.uPatch);
    LOG_INFO("NVSIPL header version: %u %u %u\n", NVSIPL_MAJOR_VER, NVSIPL_MINOR_VER, NVSIPL_PATCH_VER);
    if (oVer.uMajor != NVSIPL_MAJOR_VER || oVer.uMinor != NVSIPL_MINOR_VER || oVer.uPatch != NVSIPL_PATCH_VER) {
        LOG_ERR("NvSIPL library and header version mismatch\n");
    }

    // INvSIPLQuery
    auto pQuery = INvSIPLQuery::GetInstance();
    CHK_PTR_AND_RETURN(pQuery, "INvSIPLQuery::GetInstance");

    SIPLStatus status = pQuery->ParseDatabase();
    CHK_STATUS_AND_RETURN(status, "INvSIPLQuery::ParseDatabase");

    LOG_INFO("Parsing command line arguments\n");
    CCmdLineParser cmdline;
    auto ret = cmdline.Parse(argc, argv);
    if (ret != 0) {
        // No need to print any error, Parse() would have printed error.
        return -1;
    }

#if !NV_IS_SAFETY
    if (cmdline.verbosity >= INvSIPLTrace::LevelInfo) {
        cmdline.PrintArgs();
    }

    std::unique_ptr<CComposite> upComposite(nullptr);
    if (cmdline.uNumDisplays > 0U) {
        LOG_INFO("Creating compositor\n");
        if (cmdline.bNvSci) {
            upComposite.reset(new CCompositeNvSci());
            CHK_PTR_AND_RETURN(upComposite, "NvSci compositor creation");
        } else {
            upComposite.reset(new CComposite());
            CHK_PTR_AND_RETURN(upComposite, "Compositor creation");
        }
    }
#endif // !NV_IS_SAFETY

    // Set verbosity level
    LOG_INFO("Setting verbosity level: %u\n", cmdline.verbosity);
    INvSIPLQueryTrace::GetInstance()->SetLevel((INvSIPLQueryTrace::TraceLevel)cmdline.verbosity);
#if !NV_IS_SAFETY
    INvSIPLTrace::GetInstance()->SetLevel((INvSIPLTrace::TraceLevel)cmdline.verbosity);
#endif // !NV_IS_SAFETY
    CLogger::GetInstance().SetLogLevel((CLogger::LogLevel) cmdline.verbosity);

    LOG_INFO("Setting up signal handler\n");
    SigSetup();

    if (cmdline.sTestConfigFile != "") {
        status = pQuery->ParseJsonFile(cmdline.sTestConfigFile);
        CHK_STATUS_AND_RETURN(status, "INvSIPLQuery::ParseJsonFile");
    }

    LOG_INFO("Getting platform configuration for %s\n", cmdline.sConfigName.c_str());
    PlatformCfg oPlatformCfg;
    status = pQuery->GetPlatformCfg(cmdline.sConfigName, oPlatformCfg);
    CHK_STATUS_AND_RETURN(status, "INvSIPLQuery::GetPlatformCfg");

    // Apply mask
    if (cmdline.vMasks.size() != 0) {
        LOG_INFO("Setting link masks\n");
        status = pQuery->ApplyMask(oPlatformCfg, cmdline.vMasks);
        CHK_STATUS_AND_RETURN(status, "INvSIPLQuery::ApplyMask");
    }

    // Get to ignore the fatal error
    bIgnoreError = cmdline.bIgnoreError;

    // Check if file mode.
    // If so we need to verify/update device block info and create fileReader
    std::unique_ptr<CFileReader> fileReader(nullptr);
    if (!cmdline.vInputRawFiles.empty()) {
        if (oPlatformCfg.numDeviceBlocks != 1) {
            LOG_ERR("Only one device block is supported in simulator mode. Please correct mask.\n");
            return -1;
        }
        if (oPlatformCfg.deviceBlockList[0].numCameraModules != 1) {
            LOG_ERR("Only one camera module is supported in simulator mode. Please correct mask.\n");
            return -1;
        }

        oPlatformCfg.deviceBlockList[0].isSimulatorModeEnabled = true;
        // Create new file reader
        fileReader.reset(new CFileReader());
        CHK_PTR_AND_RETURN(fileReader, "FileReader creation");

        // Initialize the feeder
        const auto& vcinfo = oPlatformCfg.deviceBlockList[0].cameraModuleInfoList[0].sensorInfo.vcInfo;
        auto status = fileReader->Init(cmdline.vInputRawFiles, vcinfo, &bQuit);
        CHK_STATUS_AND_RETURN(status, "FileReader initialization");
    }

    // Check if passive mode is enabled.
    // If so we need to modify device block info.
    for (auto d = 0u; d != oPlatformCfg.numDeviceBlocks; d++ ) {
        if (cmdline.bEnablePassive) {
            oPlatformCfg.deviceBlockList[d].isPassiveModeEnabled = true;
        }
    }

    std::unique_ptr<CFrameFeeder> upFrameFeeder(nullptr);
    uint32_t uTwoPassIspProducerId = UINT32_MAX;
    uint32_t uTwoPassIspConsumerId = UINT32_MAX;
    if (cmdline.bTwoPassIsp) {
        // Pair where the first entry is the pipeline ID and the second entry is the device block ID
        std::pair<uint32_t, uint32_t> uPipelineIds[2U] {{UINT32_MAX, UINT32_MAX}};
        uint32_t uNumPipelines = 0U;
        for (uint32_t d = 0U; d < oPlatformCfg.numDeviceBlocks; d++) {
            const DeviceBlockInfo &db = oPlatformCfg.deviceBlockList[d];
            if (db.numCameraModules > 1U) {
                LOG_ERR("For two-pass ISP only one pipeline can be enabled per device block\n");
                return -1;
            }
            if (db.numCameraModules > 0U) {
                if (uNumPipelines < 2U) {
                    uPipelineIds[uNumPipelines] = {db.cameraModuleInfoList[0].sensorInfo.id, d};
                }
                uNumPipelines++;
            }
        }
        if (uNumPipelines != 2U) {
            LOG_ERR("Exactly two pipelines must be enabled for two-pass ISP processing: %u != 2\n",
                    uNumPipelines);
            return -1;
        }
        uint32_t uSimulatorDevBlk = UINT32_MAX;
        if (uPipelineIds[0].first > uPipelineIds[1].first) {
            uTwoPassIspProducerId = (cmdline.bTwoPassLowToHigh ? uPipelineIds[1].first : uPipelineIds[0].first);
            uTwoPassIspConsumerId = (cmdline.bTwoPassLowToHigh ? uPipelineIds[0].first : uPipelineIds[1].first);
            uSimulatorDevBlk = (cmdline.bTwoPassLowToHigh ? uPipelineIds[0].second : uPipelineIds[1].second);
        } else {
            uTwoPassIspProducerId = (cmdline.bTwoPassLowToHigh ? uPipelineIds[0].first : uPipelineIds[1].first);
            uTwoPassIspConsumerId = (cmdline.bTwoPassLowToHigh ? uPipelineIds[1].first : uPipelineIds[0].first);
            uSimulatorDevBlk = (cmdline.bTwoPassLowToHigh ? uPipelineIds[1].second : uPipelineIds[0].second);
        }
        oPlatformCfg.deviceBlockList[uSimulatorDevBlk].isSimulatorModeEnabled = true;
        upFrameFeeder.reset(new CFrameFeeder());
        CHK_PTR_AND_RETURN(upFrameFeeder, "CFrameFeeder creation");
    }

#if !NV_IS_SAFETY
    // Configure the recorder configuration
    for (auto d = 0u; d != oPlatformCfg.numDeviceBlocks; d++) {
        oPlatformCfg.deviceBlockList[d].deserInfo.camRecCfg = cmdline.uCamRecCfg;
    }

    // Warn if file dumping is requested with more than one output enabled
    if (cmdline.sFiledumpPrefix != "") {
        uint32_t uNumOutputsPerSensor = (uint32_t)(!cmdline.bDisableRaw)
                                      + (uint32_t)(!cmdline.bDisableISP0)
                                      + (uint32_t)(!cmdline.bDisableISP1)
                                      + (uint32_t)(!cmdline.bDisableISP2);
        if (!((oPlatformCfg.numDeviceBlocks == 1)
              && (oPlatformCfg.deviceBlockList[0].numCameraModules == 1)
              && (uNumOutputsPerSensor == 1))) {
            LOG_WARN("More than one output is requested for file dump. Frame drops may occur.\n");
        }
    }
#endif // !NV_IS_SAFETY

    uint32_t activeSensorCount = 0U;
    // Check platform configuration and pipeline configuration compatibility.
    for (auto d = 0u; d != oPlatformCfg.numDeviceBlocks; d++ ) {
        auto db = oPlatformCfg.deviceBlockList[d];
        for (auto m = 0u; m != db.numCameraModules; m++) {
            auto module = db.cameraModuleInfoList[m];
            auto sensor = module.sensorInfo;
            activeSensorCount++;
#if !NV_IS_SAFETY
            if (sensor.isTPGEnabled and
                (!cmdline.bDisableISP0 or
                 !cmdline.bDisableISP1 or
                 !cmdline.bDisableISP2)) {
                LOG_ERR("Cannot enable ISP output in TPG mode.\n");
                return -1;
            }
#endif // !NV_IS_SAFETY
        }
    }

    // Get the current timestamp in all time bases
    // Later, when frames become available, compare to the correct one
    std::vector<uint64_t> uInitTimeArrUs;
    uInitTimeArrUs.resize(NVSIPL_TIME_BASE_CLOCK_USER_DEFINED);
    if (cmdline.bEnableInitProfiling) {
#ifdef NVMEDIA_QNX
        uint64_t cps = SYSPAGE_ENTRY(qtime)->cycles_per_sec;
#else
        clockid_t clockId;
        struct timespec initTimespec;
        CClockIdMgr clockIdMgr;
#endif // NVMEDIA_QNX
        for (uint32_t i = NVSIPL_TIME_BASE_CLOCK_PTP; i < NVSIPL_TIME_BASE_CLOCK_USER_DEFINED; i++) {
#ifdef NVMEDIA_QNX
            // On QNX, set the timestamp to be the same in all time bases
            if (i == (uint32_t)NVSIPL_TIME_BASE_CLOCK_PTP) {
                uInitTimeArrUs[i] = (uint64_t)((((double_t)ClockCycles()) / cps) * 1000000.0);
            } else {
                uInitTimeArrUs[i] = uInitTimeArrUs[(uint32_t)NVSIPL_TIME_BASE_CLOCK_PTP];
            }
#else
            status = clockIdMgr.GetClockId((NvSiplTimeBase)i, clockId);
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("CClockIdMgr::GetClockId failed\n");
                return status;
            }
            clock_gettime(clockId, &initTimespec);
            uInitTimeArrUs[i] = ((uint64_t)initTimespec.tv_sec * 1000000) +
                                ((uint64_t)initTimespec.tv_nsec / 1000);
#endif // NVMEDIA_QNX
        }
    }

    LOG_INFO("Creating camera master\n");
    if (cmdline.bNvSci) {
        upMaster.reset(new CNvSIPLMasterNvSci());
        CHK_PTR_AND_RETURN(upMaster, "NvSci camera master creation");
    } else {
        upMaster.reset(new CNvSIPLMaster());
        CHK_PTR_AND_RETURN(upMaster, "Camera master creation");
    }

    NvSciBufModule bufModule = nullptr;
    NvSciSyncModule syncModule = nullptr;

    LOG_INFO("Setting up master\n");
    status = upMaster->Setup(&bufModule, &syncModule);
    CHK_STATUS_AND_RETURN(status, "Master setup");

    // Enable following outputs for all sensors
    std::vector<INvSIPLClient::ConsumerDesc::OutputType> eOutputList;
    if (!cmdline.bDisableRaw) {
        eOutputList.push_back(INvSIPLClient::ConsumerDesc::OutputType::ICP);
    }

    if (!cmdline.bDisableISP0) {
        eOutputList.push_back(INvSIPLClient::ConsumerDesc::OutputType::ISP0);
    }

    if (!cmdline.bDisableISP1) {
        eOutputList.push_back(INvSIPLClient::ConsumerDesc::OutputType::ISP1);
    }

    if (!cmdline.bDisableISP2) {
        eOutputList.push_back(INvSIPLClient::ConsumerDesc::OutputType::ISP2);
    }

    // All ISP outputs are signaled by the same fence, so only one ISP output needs to interact with
    // the frame feeder. By default, set that output to be ISP0.
    INvSIPLClient::ConsumerDesc::OutputType frameFeederIspOutput =
            INvSIPLClient::ConsumerDesc::OutputType::ISP0;
    if (upFrameFeeder != nullptr) {
        status = upFrameFeeder->Init(syncModule, &bQuit);
        CHK_STATUS_AND_RETURN(status, "CFrameFeeder initialization");
        // Select the first enabled ISP output to be the one to interact with the frame feeder.
        // Since the capture output and at least one ISP output must be enabled for two-pass ISP
        // processing, we know for certain that the first enabled ISP output will be at position one
        // in eOutputList.
        frameFeederIspOutput = eOutputList[1];
    }

    LOG_INFO("Creating consumers\n");
    vector<CProfiler*> vpProfilerISP;
    vector<unique_ptr<CProfiler>> vupProfilers;
    vector<unique_ptr<CNvSIPLConsumer>> vupConsumers;
    vector<unique_ptr<CPipelineNotificationHandler>> vupNotificationHandler;
    vector<unique_ptr<CDeviceBlockNotificationHandler>> vupDeviceBlockNotifyHandler;
    vector<unique_ptr<CPipelineFrameQueueHandler>> vupFrameCompletionQueueHandler;
    vector<std::pair<INvSIPLFrameCompletionQueue *, CNvSIPLConsumer *>> vQueueConsumerPair;
    NvSciStreamBlock consumer[MAX_NUM_SENSORS][MAX_OUTPUTS_PER_SENSOR];
    NvSciStreamBlock *consumerUpstream[MAX_NUM_SENSORS][MAX_OUTPUTS_PER_SENSOR] = {{nullptr}};
    NvSciStreamBlock queue[MAX_NUM_SENSORS][MAX_OUTPUTS_PER_SENSOR];

    NvSIPLDeviceBlockQueues deviceBlockQueues;
#if !NV_IS_SAFETY
    bool bDisplay = (cmdline.uNumDisplays > 0U);
#else
    bool bDisplay = false;
#endif // !NV_IS_SAFETY
    bool bFileWrite = (cmdline.sFiledumpPrefix != "");
    status = upMaster->SetPlatformConfig(&oPlatformCfg, deviceBlockQueues, cmdline.bIsParkingStream);
    CHK_STATUS_AND_RETURN(status, "Master SetPlatformConfig");

    // for each sensor
    for (auto d = 0u; d != oPlatformCfg.numDeviceBlocks; d++) {
        auto db = oPlatformCfg.deviceBlockList[d];
        for (auto m = 0u; m != db.numCameraModules; m++) {
            auto module = db.cameraModuleInfoList[m];
            auto sensor = module.sensorInfo;
            uint32_t uSensor = sensor.id;

            NvSIPLPipelineConfiguration pipelineCfg {};
            NvSIPLPipelineQueues pipelineQueues {};

            pipelineCfg.enableSubframe = cmdline.bEnableSubframe;
            pipelineCfg.captureOutputRequested = !cmdline.bDisableRaw;
            pipelineCfg.isp0OutputRequested = !cmdline.bDisableISP0;
            pipelineCfg.isp1OutputRequested = !cmdline.bDisableISP1;
            pipelineCfg.isp2OutputRequested = !cmdline.bDisableISP2;

            NvSIPLDownscaleCropCfg &downscaleCropCfg = pipelineCfg.downscaleCropCfg;
            downscaleCropCfg.ispInputCropEnable = cmdline.bIspInputCropEnable;
            downscaleCropCfg.ispInputCrop.x0 = 0u;
            downscaleCropCfg.ispInputCrop.y0 = cmdline.uIspInputCropY;
            downscaleCropCfg.ispInputCrop.x1 = sensor.vcInfo.resolution.width;
            downscaleCropCfg.ispInputCrop.y1 = cmdline.uIspInputCropY + cmdline.uIspInputCropH;

            if (cmdline.bIsParkingStream && (uSensor == 0U)) {
               downscaleCropCfg.isp1DownscaleEnable = true;
               downscaleCropCfg.isp1DownscaleWidth = 1920U;
               downscaleCropCfg.isp1DownscaleHeight = 1080U;
            }

          if (((eOutputList.size() + activeSensorCount) > 2U) &&
               (bDisplay) &&
               ((sensor.vcInfo.resolution.width * sensor.vcInfo.resolution.height) > BUFFER_8MP)) {
                /* It is not possible to stream multiple outputs from 8MP sensors into Display
                 * at 30fps since we rely on VIC for format downscale and format conversion.
                 * This is because of the limited bandwidth available with VIC.
                 * To avoid VIC from having to do both downscale and convert images:
                 * we use ISP downscale when display is enabled (bDisplay)
                 * while using more than 1 output or more than 1 sensor
                 * i.e., ((eOutputList.size() + activeSensorCount) > 2U)
                 * and when using 8MP cameras ((width * height) > BUFFER_8MP)
                 */
               downscaleCropCfg.isp0DownscaleEnable = true;
               downscaleCropCfg.isp0DownscaleWidth = 1920U;
               downscaleCropCfg.isp0DownscaleHeight = 1080U;
               downscaleCropCfg.isp1DownscaleEnable = true;
               downscaleCropCfg.isp1DownscaleWidth = 1920U;
               downscaleCropCfg.isp1DownscaleHeight = 1080U;
               downscaleCropCfg.isp2DownscaleEnable = true;
               downscaleCropCfg.isp2DownscaleWidth = 1920U;
               downscaleCropCfg.isp2DownscaleHeight = 1080U;
            }

            if (cmdline.bEnableStatsOverrideTest) {
                NvSIPLIspStatsOverrideSetting &overrideSetting = pipelineCfg.statsOverrideSettings;

                // override ISP HIST1 statistics
                overrideSetting.enableHistStatsOverride[1] =true;
                overrideSetting.histStats[1].enable = true;
                overrideSetting.histStats[1].offset = 0.0;
                uint8_t knees[NVSIPL_ISP_HIST_KNEE_POINTS] = {4u, 24u, 48u, 96u, 140u, 192u, 248u, 255u};
                uint8_t ranges[NVSIPL_ISP_HIST_KNEE_POINTS] = {4u, 8u, 10u, 12u, 14u, 16u, 19u, 19u};
                for (uint8_t i = 0; i < NVSIPL_ISP_HIST_KNEE_POINTS; i++) {
                    overrideSetting.histStats[1].knees[i] = knees[i];
                    overrideSetting.histStats[1].ranges[i] = ranges[i];
                }
                overrideSetting.histStats[1].rectangularMask.x0 = 2;
                overrideSetting.histStats[1].rectangularMask.y0 = 2;
                overrideSetting.histStats[1].rectangularMask.x1 = 200;
                overrideSetting.histStats[1].rectangularMask.y1 = 200;

                // override ISP LAC1 statistcs
                overrideSetting.enableLacStatsOverride[1] = true;
                overrideSetting.lacStats[1].enable = true;
                for (uint8_t i = 0; i < NVSIPL_ISP_MAX_COLOR_COMPONENT; i++) {
                    overrideSetting.lacStats[1].min[i] = 0.0;
                    overrideSetting.lacStats[1].max[i] = 1.0;
                }

                for (uint8_t j = 0; j < NVSIPL_ISP_MAX_LAC_ROI; j++) {
                    overrideSetting.lacStats[1].roiEnable[j] = true;
                    overrideSetting.lacStats[1].ellipticalMaskEnable[j] = true;
                }

                NvSiplISPStatisticsWindows window[] = {{4, 4, 32, 32, 4, 4, {0, 0}},
                                                {4, 4, 32, 32, 4, 4, {128, 0}},
                                                {4, 4, 32, 32, 4, 4, {0, 128}},
                                                {4, 4, 32, 32, 4, 4, {128, 128}}};
                memcpy(&overrideSetting.lacStats[1].windows, window, sizeof(NvSiplISPStatisticsWindows) * 4);
                overrideSetting.lacStats[1].ellipticalMask = {{128.0F, 128.0F}, 256, 256, 180.0F};
            }


            // Check if simulator mode.
            // If so, we need to set up a fileReader to feed frames to SIPL
            if (db.isSimulatorModeEnabled) {
                pipelineCfg.imageGroupWriter = fileReader.get();
            }
            if (cmdline.bTwoPassIsp && (uSensor == uTwoPassIspConsumerId)) {
                // For two-pass ISP processing we want the consumer/reprocessing pipeline's input to
                // be the images captured by the producer/capture pipeline
                pipelineCfg.imageGroupWriter = upFrameFeeder.get();
            }

            status = upMaster->SetPipelineConfig(uSensor, pipelineCfg, pipelineQueues);
            CHK_STATUS_AND_RETURN(status, "Master SetPipelineConfig");

            INvSIPLFrameCompletionQueue *frameCompletionQueue[MAX_OUTPUTS_PER_SENSOR];

            frameCompletionQueue[(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ICP] = pipelineQueues.captureCompletionQueue;
            frameCompletionQueue[(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP0] = pipelineQueues.isp0CompletionQueue;
            frameCompletionQueue[(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP1] = pipelineQueues.isp1CompletionQueue;
            frameCompletionQueue[(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP2] = pipelineQueues.isp2CompletionQueue;

#if !NV_IS_SAFETY
            auto isRawSensor = false;
            if ((sensor.vcInfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW6) or
                (sensor.vcInfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW7) or
                (sensor.vcInfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW8) or
                (sensor.vcInfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10) or
                (sensor.vcInfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12) or
                (sensor.vcInfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW14) or
                (sensor.vcInfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW16) or
                (sensor.vcInfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW20)) {
                isRawSensor = true;
            }
#endif // !NV_IS_SAFETY

            vQueueConsumerPair.clear();
            vpProfilerISP.clear();

            // for each output
            for (INvSIPLClient::ConsumerDesc::OutputType eOutput : eOutputList) {
                // Create NvSIPL consumer using client descriptor
                LOG_INFO("Creating consumer for output:%u of sensor:%u\n", eOutput, uSensor);
                auto upCons = unique_ptr<CNvSIPLConsumer>(new CNvSIPLConsumer());
                CHK_PTR_AND_RETURN(upCons, "Consumer creation");

                // Register the consumer with the compositor as source
                auto uID = -1u;
#if !NV_IS_SAFETY
                if (upComposite != nullptr) {
                    auto grpIndex = 0;
                    switch (db.csiPort) {
                        default:
                        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_AB:
                        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_A:
                        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_B:
                            grpIndex = 0;
                            break;
                        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_CD:
                        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_C:
                        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_D:
                            grpIndex = 1;
                            break;
                        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_EF:
                        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_E:
                        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_F:
                            grpIndex = 2;
                            break;
                        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_GH:
                        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_G:
                        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_H:
                            grpIndex = 3;
                            break;
                    }

                    LOG_INFO("Register with compositor for output:%u of sensor:%u\n", eOutput, uSensor);
                    auto outIndex = int(eOutput) - int(INvSIPLClient::ConsumerDesc::OutputType::ICP);
                    auto isRawOutput = (eOutput == INvSIPLClient::ConsumerDesc::OutputType::ICP) and isRawSensor;
                    status = upComposite->RegisterSource(grpIndex,
                                                         module.linkIndex,
                                                         outIndex,
                                                         isRawOutput,
                                                         uID,
                                                         db.isSimulatorModeEnabled,
                                                         &consumer[uSensor][(uint32_t)eOutput],
                                                         &consumerUpstream[uSensor][(uint32_t)eOutput],
                                                         &queue[uSensor][(uint32_t)eOutput]);
                    CHK_STATUS_AND_RETURN(status, "Composite register");

                    if (!cmdline.bNvSci) {
                        /**
                         * If the synchronization isn't being handled by NvSciStream we need to
                         * store the compositor ID for later. It will be used when setting up the
                         * synchronization.
                         */
                        status = upMaster->SetCompositorID(uSensor, eOutput, upComposite.get(), uID);
                        if (status != NVSIPL_STATUS_OK) {
                            LOG_ERR("SetCompositorID failed for sensor pipeline:%u\n", uSensor);
                            return -1;
                        }
                    }
                }
#endif // !NV_IS_SAFETY

                CProfiler *pProfiler = nullptr;
                if (cmdline.bShowFPS || cmdline.bEnableProfiling || cmdline.bEnableInitProfiling) {
                    LOG_INFO("Profiler initialization for output:%u of sensor:%u\n", eOutput, uSensor);
                    unique_ptr<CProfiler> upProfiler = unique_ptr<CProfiler>(new CProfiler());
                    CHK_PTR_AND_RETURN(upProfiler, "Profiler creation");
                    status = upProfiler->Init(uSensor,
                                              eOutput,
                                              uInitTimeArrUs,
                                              syncModule,
                                              cmdline.bEnableProfiling,
                                              cmdline.sProfilePrefix,
                                              cmdline.bEnableInitProfiling);
                    if (status != NVSIPL_STATUS_OK) {
                        LOG_ERR("Failed to initialize profiler for output:%u of sensor:%u\n", eOutput, uSensor);
                        return -1;
                    }
                    pProfiler = upProfiler.get();
                    vupProfilers.push_back(move(upProfiler));
                }
                CFrameFeeder *pFrameFeeder = nullptr;
                if (((uSensor == uTwoPassIspProducerId)
                     && (eOutput == INvSIPLClient::ConsumerDesc::OutputType::ICP))
                    || ((uSensor == uTwoPassIspConsumerId)
                        && (eOutput == frameFeederIspOutput))) {
                    pFrameFeeder = upFrameFeeder.get();
                }

                LOG_INFO("Consumer initialization for output:%u of sensor:%u\n", eOutput, uSensor);
                if (cmdline.bNvSci) {
                    status = upCons->Init(
#if !NV_IS_SAFETY
                                          nullptr,
                                          (upComposite != nullptr) ? (CNvSIPLMasterNvSci *)upMaster.get() : nullptr,
#endif // !NV_IS_SAFETY
                                          -1,
                                          uSensor,
                                          eOutput,
                                          pProfiler,
                                          pFrameFeeder,
                                          cmdline.sFiledumpPrefix,
                                          cmdline.uNumSkipFrames,
                                          cmdline.uNumWriteFrames);
                } else {
                    status = upCons->Init(
#if !NV_IS_SAFETY
                                          upComposite.get(),
                                          nullptr,
#endif // !NV_IS_SAFETY
                                          uID,
                                          uSensor,
                                          eOutput,
                                          pProfiler,
                                          pFrameFeeder,
                                          cmdline.sFiledumpPrefix,
                                          cmdline.uNumSkipFrames,
                                          cmdline.uNumWriteFrames);
                }
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("Failed to initialize consumer for output:%u of sensor:%u\n", eOutput, uSensor);
                    return -1;
                }

                if (cmdline.bShowMetadata) {
                    upCons->EnableMetadataLogging();
                }
#if !NV_IS_SAFETY
                if (cmdline.bAutoLEDControl) {
                    upCons->EnableLEDControl();
                }
#endif // !NV_IS_SAFETY

                if (eOutput != INvSIPLClient::ConsumerDesc::OutputType::ICP) {
                    if (pProfiler) {
                        vpProfilerISP.push_back(pProfiler);
                    }
                }

                vQueueConsumerPair.push_back(std::make_pair(frameCompletionQueue[(uint32_t)eOutput], upCons.get()));
                vupConsumers.push_back(move(upCons));
            } // output

            auto upFrameCompletionQueueHandler = std::unique_ptr<CPipelineFrameQueueHandler>(new CPipelineFrameQueueHandler());
            CHK_PTR_AND_RETURN(upFrameCompletionQueueHandler, "Frame Completion Queues handler creation");

            upFrameCompletionQueueHandler->Init(uSensor, vQueueConsumerPair, syncModule);
            CHK_STATUS_AND_RETURN(status, "Frame Completion Queues Handler Init");

            vupFrameCompletionQueueHandler.push_back(move(upFrameCompletionQueueHandler));

            auto upNotificationHandler = std::unique_ptr<CPipelineNotificationHandler>(new CPipelineNotificationHandler());
            CHK_PTR_AND_RETURN(upNotificationHandler, "Notification handler creation");

            status = upNotificationHandler->Init(uSensor, pipelineQueues.notificationQueue, vpProfilerISP);
            CHK_STATUS_AND_RETURN(status, "Notification Handler Init");

            vupNotificationHandler.push_back(move(upNotificationHandler));
        } // module
    } // device block

#ifdef NVMEDIA_QNX
    if (cmdline.bEnableSc7Boot) {
        suspend_drive_os();
    }
#endif //NVMEDIA_QNX

#if !NV_IS_SAFETY
    if (upComposite != nullptr) {
        LOG_INFO("Initializing compositor\n");
        SIPLStatus status = NVSIPL_STATUS_ERROR;
        status = upComposite->Init(cmdline.uNumDisplays,
                                   cmdline.bRectSet ? &cmdline.oDispRect : nullptr,
                                   bufModule,
                                   syncModule);
        CHK_STATUS_AND_RETURN(status, "Compositor initialization");
    }
#endif // !NV_IS_SAFETY

    LOG_INFO("Initializing master interface\n");
    status = upMaster->Init();
    CHK_STATUS_AND_RETURN(status, "Master initialization");

    if (cmdline.bEnablePassive) {
        cout << "Press any key to continue to process\n";
        char line[256];
        cout << "-\n";
        cin.getline(line, 256);
    }

    for (auto d = 0u; d != oPlatformCfg.numDeviceBlocks; d++) {
        auto upDeviceBlockNotifyHandler = std::unique_ptr<CDeviceBlockNotificationHandler>(new CDeviceBlockNotificationHandler());
        CHK_PTR_AND_RETURN(upDeviceBlockNotifyHandler, "Device Block Notification handler creation");

        status = upDeviceBlockNotifyHandler->Init(d, oPlatformCfg.deviceBlockList[d],
                                                  deviceBlockQueues.notificationQueue[d]);
        CHK_STATUS_AND_RETURN(status, "Device Block Notification Handler Init");

        vupDeviceBlockNotifyHandler.push_back(move(upDeviceBlockNotifyHandler));
    }

    if (cmdline.bNvSci) {
        // for each sensor
        for (auto d = 0u; d != oPlatformCfg.numDeviceBlocks; d++) {
            auto db = oPlatformCfg.deviceBlockList[d];
            for (auto m = 0u; m != db.numCameraModules; m++) {
                auto module = db.cameraModuleInfoList[m];
                auto sensor = module.sensorInfo;
                uint32_t uSensor = sensor.id;

                // for each output
                for (INvSIPLClient::ConsumerDesc::OutputType eOutput : eOutputList) {
                    auto status = upMaster->RegisterSource(uSensor,
                                                           eOutput,
                                                           db.isSimulatorModeEnabled,
#if !NV_IS_SAFETY
                                                           (upComposite != nullptr),
#else // !NV_IS_SAFETY
                                                           false,
#endif // !NV_IS_SAFETY
                                                           &consumer[uSensor][(uint32_t)eOutput],
                                                           consumerUpstream[uSensor][(uint32_t)eOutput],
                                                           &queue[uSensor][(uint32_t)eOutput],
                                                           bDisplay);
                    if (status != NVSIPL_STATUS_OK) {
                        LOG_ERR("Failed to register source for output:%u of sensor:%u\n", eOutput, uSensor);
                        return -1;
                    }
                } // output
                // ICP is needed even if not an output; register so that buffers can be allocated
                if (cmdline.bDisableRaw) {
                    status = upMaster->RegisterSource(uSensor,
                                                      INvSIPLClient::ConsumerDesc::OutputType::ICP,
                                                      db.isSimulatorModeEnabled,
                                                      false,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      bDisplay);
                    if (status != NVSIPL_STATUS_OK) {
                        LOG_ERR("Failed to register source for ICP output of sensor:%u\n", uSensor);
                        return -1;
                    }
                }
            } // module
        } // device block
    }

    if (cmdline.bAutoRecovery) {
        m_upAutoRecovery.reset(new CAutoRecovery(upMaster.get()));
        CHK_PTR_AND_RETURN(m_upAutoRecovery, "Auto recovery creation");
    }

#if !NV_IS_SAFETY
    if (upComposite != nullptr) {
        LOG_INFO("Starting compositor\n");
        status = upComposite->Start();
        CHK_STATUS_AND_RETURN(status, "Compositor start");

        if (cmdline.bNvSci) {
            status = upMaster->SetupElements();
            if(status != NVSIPL_STATUS_OK) {
                LOG_ERR("Failed to set up producer streaming\n");
                return -1;
            }
        }
    }
#endif // !NV_IS_SAFETY

    bool bTwoPassIspProducerRegistered = false;
    for (auto d = 0u; d != oPlatformCfg.numDeviceBlocks; d++ ) {
        auto db = oPlatformCfg.deviceBlockList[d];
        for (auto m = 0u; m != db.numCameraModules; m++) {
            auto module = db.cameraModuleInfoList[m];
            auto sensor = module.sensorInfo;
            uint32_t uSensor = sensor.id;

            if (cmdline.bShowEEPROM) {
                LOG_MSG("EEPROM of Sensor %d\n", uSensor);
                upMaster->ShowEEPROM(uSensor);
            }
#if !NV_IS_SAFETY
            if (cmdline.uExpNo != -1u) {
                status = upMaster->SetCharMode(uSensor, cmdline.uExpNo);
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("SetCharMode failed for sensor pipeline:%u\n", uSensor);
                    return -1;
                }
            }
#endif // !NV_IS_SAFETY
            if (cmdline.bNvSci) {
                status = upMaster->AllocateBuffers(uSensor,
                                                   !cmdline.bDisableISP0,
                                                   !cmdline.bDisableISP1,
                                                   !cmdline.bDisableISP2);
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("AllocateBuffers failed for sensor pipeline:%u\n", uSensor);
                    return -1;
                }
            } else {
                uint32_t uAllocRegisterPipeline = uSensor;
                if (cmdline.bTwoPassIsp) {
                    // If necessary, for two-pass ISP processing swap the index used for
                    // allocation/registration so that the producer buffers get allocated and
                    // registered before the consumer buffers. This is necessary so that the
                    // consumer/reprocessing pipeline can register the buffers that were previously
                    // allocated by the producer/capture pipeline.
                    if ((uSensor == uTwoPassIspConsumerId) && (!bTwoPassIspProducerRegistered)) {
                        uAllocRegisterPipeline = uTwoPassIspProducerId;
                        bTwoPassIspProducerRegistered = true;
                    } else if (uSensor == uTwoPassIspProducerId) {
                        if (bTwoPassIspProducerRegistered) {
                            uAllocRegisterPipeline = uTwoPassIspConsumerId;
                        } else {
                            bTwoPassIspProducerRegistered = true;
                        }
                    }
                }
                status = upMaster->AllocateAndRegisterBuffers(uAllocRegisterPipeline,
                                                              oPlatformCfg.platformConfig,
                                                              !cmdline.bDisableRaw,
                                                              !cmdline.bDisableISP0,
                                                              !cmdline.bDisableISP1,
                                                              !cmdline.bDisableISP2,
                                                              bFileWrite,
                                                              cmdline.bTwoPassIsp);
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("AllocateAndRegisterBuffers failed for sensor pipeline:%u\n",
                            uAllocRegisterPipeline);
                    return -1;
                }
            }
#if !NV_IS_SAFETY
            if (!(cmdline.bNvSci && (upComposite != nullptr))) {
                /**
                 * If the synchronization is not being handled by NvSciStream (bNvSci and non-null
                 * upComposite) then we need to set it up here. If there is a compositor, set up
                 * synchronization between SIPL and NvMedia 2D via SetupNvSciSync2DSignalerWaiter().
                 * Otherwise, if there isn't a compositor, call SetNvSciSyncCPUWaiter() to allocate
                 * and register a simple CPU waiter sync object.
                 * It is worth noting, however, that upComposite doesn't exist in the safety build
                 * so in that case we always want to call SetNvSciSyncCPUWaiter().
                 */
                if (upComposite != nullptr) {
                    status = upMaster->SetupNvSciSync2DSignalerWaiter(uSensor,
                                                                      eOutputList,
                                                                      db.isSimulatorModeEnabled);
                    if (status != NVSIPL_STATUS_OK) {
                        LOG_ERR("SetupNvSciSync2DSignalerWaiter failed for sensor pipeline:%u\n", uSensor);
                        return -1;
                    }
                } else {
#endif // !NV_IS_SAFETY
                    if (!cmdline.bTwoPassIsp || (uSensor != uTwoPassIspConsumerId)) {
                        status = upMaster->SetNvSciSyncCPUWaiter(uSensor,
                                                                 !cmdline.bDisableISP0,
                                                                 !cmdline.bDisableISP1,
                                                                 !cmdline.bDisableISP2);
                        if (status != NVSIPL_STATUS_OK) {
                            LOG_ERR("SetNvSciSyncCPUWaiter failed for sensor pipeline:%u\n", uSensor);
                            return -1;
                        }
                    }
#if !NV_IS_SAFETY
                }
            }
#endif // !NV_IS_SAFETY
        }
    }

#if !NV_IS_SAFETY
    if ((upComposite != nullptr) && cmdline.bNvSci) {
        status = upMaster->SetupBuffers();
        if(status != NVSIPL_STATUS_OK) {
            LOG_ERR("Failed to set up producer buffers\n");
            return -1;
        }

        status = upMaster->SetupSync(bFileWrite
                                     || cmdline.bEnableProfiling
                                     || cmdline.bEnableInitProfiling);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("Failed to set up producer synchronization\n");
            return -1;
        }

        status = upMaster->SetupComplete();
        if(status != NVSIPL_STATUS_OK) {
            LOG_ERR("Failed to finalize producer setup\n");
            return -1;
        }
    }
#endif // !NV_IS_SAFETY

    if (cmdline.bTwoPassIsp) {
        status = upMaster->SetTwoPassIspSync(uTwoPassIspProducerId, uTwoPassIspConsumerId);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("SetTwoPassIspSync failed for sensor pipelines: %u and %u\n",
                    uTwoPassIspProducerId,
                    uTwoPassIspConsumerId);
            return -1;
        }
    }

    if (cmdline.bNvSci) {
        for (auto d = 0u; d != oPlatformCfg.numDeviceBlocks; d++ ) {
            auto db = oPlatformCfg.deviceBlockList[d];
            for (auto m = 0u; m != db.numCameraModules; m++) {
                auto module = db.cameraModuleInfoList[m];
                auto sensor = module.sensorInfo;
                uint32_t uSensor = sensor.id;
                status = upMaster->RegisterBuffers(uSensor,
                                                   !cmdline.bDisableISP0,
                                                   !cmdline.bDisableISP1,
                                                   !cmdline.bDisableISP2);
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("RegisterBuffers failed for sensor pipeline:%u\n", uSensor);
                    return -1;
                }
            }
        }
    }

    nvsipl::ISiplControlAuto* upCustomPlugins[MAX_SENSORS_PER_PLATFORM] {nullptr};

    for (auto d = 0u; d != oPlatformCfg.numDeviceBlocks; d++ ) {
        auto db = oPlatformCfg.deviceBlockList[d];
        for (auto m = 0u; m != db.numCameraModules; m++) {
            auto module = db.cameraModuleInfoList[m];
            auto sensor = module.sensorInfo;
            uint32_t uSensor = sensor.id;

            if (!cmdline.bDisableISP0 ||
                !cmdline.bDisableISP1 ||
                !cmdline.bDisableISP2) {
                std::vector<uint8_t> blob;
                status = LoadNITOFile(cmdline.sNitoFolderPath,
                                      module.name,
                                      blob);
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("Failed to load NITO file\n");
                    return -1;
                }
#if !NV_IS_SAFETY
                if (cmdline.bShowNitoMetadata) {
                    /**
                     * SIPL Fetch NITO Metadata Interface Sample Usage
                     */
                    LOG_MSG("Retrieving metadata from NITO file for sensorID %u \n", uSensor);
                    status = callFetchNITOMetadataAPI(blob,
                                                      cmdline.uShowNitoMetadataNumParamSets);
                    if (status != NVSIPL_STATUS_OK) {
                        LOG_ERR("Failed to run Fetch NITO Metadata API \n");
                        return -1;
                    }
                }
#endif // !NV_IS_SAFETY
                if (cmdline.autoPlugin == NV_PLUGIN) {
                    status = upMaster->RegisterAutoControl(uSensor, NV_PLUGIN,
                                                           nullptr, blob);
                    if (status != NVSIPL_STATUS_OK) {
                        LOG_ERR("SetSiplControl(NV) failed for ISP output of sensor:%u\n", uSensor);
                        return -1;
                    }
                } else {
                    pluginlib_handle = dlopen("libnvsipl_sampleplugin.so", RTLD_LAZY);
                    if (!pluginlib_handle) {
                        LOG_ERR("Failed to open lib libnvsipl_sampleplugin.so\n");
                        return -1;
                    }
                    nvsipl::ISiplControlAuto* (*libCreatePlugin)() = (nvsipl::ISiplControlAuto*(*)())dlsym(pluginlib_handle, "CreatePlugin");
                    if (!libCreatePlugin) {
                        LOG_ERR("Failed to create function pointer for CreatePlugin\n");
                        return -1;
                    }
                    upCustomPlugins[uSensor] = libCreatePlugin();
                    CHK_PTR_AND_RETURN(upCustomPlugins[uSensor], "AutoControl plugin creation");

                    status = upMaster->RegisterAutoControl(uSensor, CUSTOM_PLUGIN0,
                                                           upCustomPlugins[uSensor], blob);
                    if (status != NVSIPL_STATUS_OK) {
                        LOG_ERR("SetAutoControl(CUST0) failed for ISP output of sensor:%u\n", uSensor);
                        return -1;
                    }
                }
            }

        }
    }

    // Retrive OV2311's custom interface
    // The client can get the custom interface handle between SIPL Init() API and SIPL Start() API called
    SIPLStatus statusOV2311 = NVSIPL_STATUS_ERROR;
    if (cmdline.sConfigName.find("OV2311") != std::string::npos) {
        cout << "Enter 'gc <sensor ID>' to get OV2311 custom inteface\n";
        char line[MAX_LINE_LENGTH];
        cout << "-\n";
        cin.getline(line, MAX_LINE_LENGTH);
        if ((line[0] == 'g') && (line[1] == 'c')) {
            statusOV2311 = GetOV2311CustomInterface(line);
        }
    }

    LOG_INFO("Starting master\n");
    status = upMaster->Start();
    CHK_STATUS_AND_RETURN(status, "Master start");

    if (m_upAutoRecovery != nullptr) {
        LOG_INFO("Starting auto recovery\n");
        status = m_upAutoRecovery->Start();
        CHK_STATUS_AND_RETURN(status, "Auto recovery start");
    }

    // Spawn a background thread to accept user's runtime command
    std::thread([&]
    {
        pthread_setname_np(pthread_self(), "RuntimeMenu");

        while (!bQuit) {
#if !NV_IS_SAFETY
            if (upComposite != nullptr) {
                cout << "Enter 'ld' to list display-able outputs.\n";
                cout << "Enter 'e' followed by group ID and display ID to enable display for specific camera group\n";
            }
            cout << "Enter 'les' followed by sensor ID to enable LED\n";
            cout << "Enter 'lds' followed by sensor ID to disable LED\n";
#endif // !NV_IS_SAFETY
            cout << "Enter 'dl' followed by sensor ID to disable the link\n";
            cout << "Enter 'el' followed by sensor ID to enable the link without module reset\n";
            cout << "Enter 'elr' followed by sensor ID to enable the link with module reset\n";
            cout << "Enter 'cm' to check the module availabilty\n";

            if (cmdline.sConfigName.find("OV2311") != std::string::npos) {
                // Show custom interface command
                cout << "Enter 'cust <value>' to set an example value for OV2311 custom API\n";
            }

            cout << "Enter 'q' to quit the application\n";
            char line[256];
            cout << "-\n";
            cin.getline(line, 256);
            if (line[0] == 'q') {
                bQuit = true;
            }
#if !NV_IS_SAFETY
            else if (upComposite != nullptr) {
                if (line[0] == 'l' and line[1] == 'd') {
                    upComposite->PrintDisplayableGroups();
                } else if ((line[0] == 'e') && (line[1] != 'l')) {
                    uint32_t uGroupId = isdigit(line[1]) ? (line[1] - '0') : 0U;
                    uint32_t uDispId = isdigit(line[2]) ? (line[2] - '0') : 0U;
                    if (uDispId >= cmdline.uNumDisplays) {
                        uDispId = 0U;
                    }
                    status = upComposite->SetActiveGroup(uDispId, uGroupId);
                    if (status == NVSIPL_STATUS_OK) {
                        cout << "Enabled output: ";
                    } else {
                        cout << "Failed to enable output: ";
                    }
                    cout << uGroupId << " on display: " << uDispId << endl;
                }
            }

            if((line[0] == 'l') && (line[2] == 's')) {
                auto id = atoi(&line[3]);
                if ((id >= 0) && (id < 16)) {
                    if (line[1] == 'e') {
                        upMaster->ToggleLED(id, true);
                        cout << "Enable LED: " << id << endl;
                    } else if (line[1] == 'd') {
                        upMaster->ToggleLED(id, false);
                        cout << "Disable LED: " << id << endl;
                    }
                } else {
                    cout << "The sensor id " << id << " is out of the range. The valid sensor id is from 0 to 15" << endl;
                }
            }
#endif // !NV_IS_SAFETY

            if ((line[0] == 'd') && (line[1] == 'l')) {
                auto id = atoi(&line[2]);
                if ((id >= 0) && (id < 16)) {
                    SIPLStatus status = upMaster->DisableLink(id);
                    if (status != NVSIPL_STATUS_OK) {
                        cout << "Disable Link failed Link id: " << id << "Error:"<<status << endl;
                    }
                    eLinkErrorIgnoreArr[id].store(StartIgnore);
                    cout << "Disable Link: " << id << endl;
                } else {
                    cout << "The sensor id " << id << " is out of the range. The valid sensor id is from 0 to 15" << endl;
                }
            }

            if ((line[0] == 'e') && (line[1] == 'l')) {
                uint32_t id = -1U;
                bool resetModule = false;
                if (line[2] == 'r') {
                    id = atoi(&line[3]);
                    resetModule = true;
                }
                else {
                    id = atoi(&line[2]);
                }
                if ((id >= 0) && (id < 16)) {
                    SIPLStatus status = upMaster->EnableLink(id, resetModule);
                    if (status != NVSIPL_STATUS_OK) {
                        cout << "Enable Link failed Link id: " << id << "Error:"<<status << endl;
                    }
                    eLinkErrorIgnoreArr[id].store(StopIgnore);
                    cout << "Enable Link: " << id << endl;
                } else {
                    cout << "The sensor id " << id << " is out of the range. The valid sensor id is from 0 to 15" << endl;
                }
            }

            if (((line[0] == 'c') && (line[1] == 'u') && (line[2] == 's') && (line[3] == 't')) ||
                ((line[0] == 'c') && (line[1] == 'm'))) {
                if (statusOV2311 == NVSIPL_STATUS_OK) {
                    processOV2311CustomCommand(line);
                }
            }
        }
    }).detach();

    bool bFirstIteration = true;
    bool bValidFrameCount = false;
    uint64_t uFrameCountDelta = 0u;
    double firstCaptureDelayMs = 0.0;
    double firstReceivedDelayMs = 0.0;
    double firstReadyDelayMs = 0.0;
    // Wait for quit
    while (!bQuit) {
        // Wait for SECONDS_PER_ITERATION
        auto oStartTime = chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::seconds(SECONDS_PER_ITERATION));

        if (cmdline.uRunDurationSec != -1) {
            cmdline.uRunDurationSec -= SECONDS_PER_ITERATION;
            if (cmdline.uRunDurationSec < SECONDS_PER_ITERATION) {
                bQuit = true;
            }
        }

#if !NV_IS_SAFETY
       // check if frames dumping is done
        if (cmdline.sFiledumpPrefix != "") {
            bool bFrameWritePending = false;
            for (auto &cons : vupConsumers) {
                if (!cons->IsFrameWriteComplete()) {
                    bFrameWritePending = true;
                    break;
                }
            }
            if (!bFrameWritePending) {
                bQuit = true;
            }
        }
#endif // !NV_IS_SAFETY

        // Check for any asynchronous fatal errors reported by pipeline threads in the library
        for (auto &notificationHandler : vupNotificationHandler) {
            if (notificationHandler->IsPipelineInError()) {
                bQuit = true;
            }
        }

        // Check for any asynchronous errors reported by the device blocks
        for (auto &notificationHandler : vupDeviceBlockNotifyHandler) {
            if (notificationHandler->IsDeviceBlockInError()) {
                bQuit = true;
            }
        }

        // Log FPS
        if (cmdline.bShowFPS || cmdline.bEnableProfiling || cmdline.bEnableInitProfiling) {
            auto uTimeElapsedMs = chrono::duration<double, std::milli> (chrono::steady_clock::now() - oStartTime).count();
            cout << "Output" << endl;
            for (auto &prof : vupProfilers) {
                prof->m_profData.profDataMut.lock();
                uFrameCountDelta = prof->m_profData.uFrameCount - prof->m_profData.uPrevFrameCount;
                prof->m_profData.uPrevFrameCount = prof->m_profData.uFrameCount;
                bValidFrameCount = uFrameCountDelta != 0u;
                if (bValidFrameCount) {
                    if (cmdline.bEnableInitProfiling && bFirstIteration) {
                        firstCaptureDelayMs = prof->m_profData.uFirstCaptureDelayUs / 1000.0;
                        firstReceivedDelayMs = prof->m_profData.uFirstReceivedDelayUs / 1000.0;
                        firstReadyDelayMs = prof->m_profData.uFirstReadyDelayUs / 1000.0;
                    }
                }
                prof->m_profData.profDataMut.unlock();

                string profName = "Sensor" + to_string(prof->m_uSensor) + "_Out"
                                  + to_string(int(prof->m_outputType)) + "\t";

                auto fps = uFrameCountDelta / (uTimeElapsedMs / 1000.0);
                cout << profName << "Frame rate (fps):\t\t" << fps << endl;

                if (cmdline.bEnableInitProfiling && bFirstIteration) {
                    cout << profName << "First capture delay (ms):\t";
                    if (bValidFrameCount) {
                        cout << firstCaptureDelayMs << endl;
                    } else {
                        cout << "N/A" << endl;
                    }
                    cout << profName << "First received delay (ms):\t";
                    if (bValidFrameCount) {
                        cout << firstReceivedDelayMs << endl;
                    } else {
                        cout << "N/A" << endl;
                    }
                    cout << profName << "First ready delay (ms):\t\t";
                    if (bValidFrameCount) {
                        cout << firstReadyDelayMs << endl;
                    } else {
                        cout << "N/A" << endl;
                    }
                }
                if (cmdline.bEnableInitProfiling && bFirstIteration) {
                    cout << endl;
                }
            }

            if (bFirstIteration) {
                bFirstIteration = false;
            }
            cout << endl;
        }
    }

    for (auto &notificationHandler : vupNotificationHandler) {
        cout << "Sensor" + to_string(notificationHandler->m_uSensor) + "\t\t"
             << "Frame drops: " << to_string(notificationHandler->GetNumFrameDrops()) + "\t\t"
             << "Frame discontinuities: " << to_string(notificationHandler->GetNumFrameDiscontinuities()) << endl;
    }

    if (m_upAutoRecovery != nullptr) {
        LOG_INFO("Stopping auto recovery\n");
        m_upAutoRecovery->Stop();
    }

    if (upFrameFeeder != nullptr) {
        upFrameFeeder->Stop();
    }

    if (upMaster != nullptr) {
        LOG_INFO("Stopping master\n");
        status = upMaster->Stop();
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("Failed to Stop master %u\n", status);
        }
    }

#if !NV_IS_SAFETY
    if (upComposite != nullptr) {
        LOG_INFO("Stopping compositor\n");
        upComposite->Stop();
    }
#endif // !NV_IS_SAFETY

    if (fileReader != nullptr) {
        fileReader->Deinit();
    }

#if !NV_IS_SAFETY
    if (upComposite != nullptr) {
        LOG_INFO("Deinitializing compositor\n");
        upComposite->Deinit();
    }
#endif // !NV_IS_SAFETY

    bool bDeviceBlockError = false;
    for (auto &notificationHandler : vupDeviceBlockNotifyHandler) {
        LOG_INFO("Deinitializing devblk notificationHandler: %u\n", notificationHandler->m_uDevBlkIndex);
        bDeviceBlockError |= notificationHandler->IsDeviceBlockInError();
        notificationHandler->Deinit();
    }

    bool bPipelineError = false;
    for (auto &notificationHandler : vupNotificationHandler) {
        LOG_INFO("Deinitializing pipeline notificationHandler: %u\n", notificationHandler->m_uSensor);
        bPipelineError |= notificationHandler->IsPipelineInError();
        notificationHandler->Deinit();
    }

    for (auto &frameCompletionQueueHandler : vupFrameCompletionQueueHandler) {
        LOG_INFO("Deinitializing frameCompletionQueueHandler: %u\n", frameCompletionQueueHandler->m_uSensor);
        frameCompletionQueueHandler->Deinit();
    }

    for (auto &cons : vupConsumers) {
        string consumerName = "Sensor" + to_string(cons->m_uSensor) + "_Out"
                              + to_string(int(cons->m_outputType));
        LOG_INFO("Deinitializing consumer %s\n", consumerName.c_str());
        cons->Deinit();
    }

    for (auto &prof : vupProfilers) {
        string profName = "Sensor" + to_string(prof->m_uSensor) + "_Out"
                          + to_string(int(prof->m_outputType));
        LOG_INFO("Deinitializing profiler %s\n", profName.c_str());
        prof->Deinit();
    }

    if (upFrameFeeder != nullptr) {
        upFrameFeeder->Deinit();
    }

    if (upMaster != nullptr) {
        LOG_INFO("Deinitializing master\n");
        upMaster->Deinit();
    }

    if (pluginlib_handle) {
        dlclose(pluginlib_handle);
    }

    if (bPipelineError) {
        LOG_ERR("Pipeline failure\n");
        return -1;
    }

    if (bDeviceBlockError) {
        LOG_ERR("Device Block failure\n");
        return -1;
    }

    LOG_MSG("SUCCESS\n");
    return 0;
}
