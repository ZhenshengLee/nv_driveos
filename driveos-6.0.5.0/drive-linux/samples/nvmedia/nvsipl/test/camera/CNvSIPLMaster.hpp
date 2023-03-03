/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/* STL Headers */
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <mutex>

#include "NvSIPLCamera.hpp" // Camera
#include "NvSIPLPipelineMgr.hpp" // Pipeline manager
#include "CUtils.hpp"

#include "nvscibuf.h"
#if !NV_IS_SAFETY
#include "CComposite.hpp"
#endif // !NV_IS_SAFETY

#include "OV2311NonFuSaCustomInterface.hpp" // Custom interface for OV2311, sample

#ifndef CNVSIPLMASTER_HPP
#define CNVSIPLMASTER_HPP

#define NUM_PACKETS (6U)
#define MAX_PARAM_SIZE (sizeof(float)*4*10*10)

#define MAX_SENSORS (16U)
#define MAX_OUTPUTS_PER_SENSOR (4U)

using namespace std;
using namespace nvsipl;

/** NvSIPL Master class */
class CNvSIPLMaster
{
 protected:
    typedef struct {
        INvSIPLClient::INvSIPLNvMBuffer *buffer = nullptr;
        NvSciStreamPacket packet;
        NvSciStreamCookie cookie;
    } BufferInfo;
    typedef struct {
        uint32_t uSensor;
        INvSIPLClient::ConsumerDesc::OutputType outputType;
        bool isSimulatorMode;
        NvSciStreamBlock staticPool;
        NvSciStreamBlock producer;
        NvSciStreamBlock queue;
        NvSciStreamBlock downstream;
        NvSciSyncObj producerSyncObj;
        NvSciSyncObj objFromConsumer;
        NvSciBufAttrList bufAttrList;
        NvSciSyncAttrList signalerAttrList;
        NvSciSyncAttrList waiterAttrList;
        NvSciSyncAttrList cpuAttrList;
        NvSciSyncAttrList consumerAttrList;
        NvSciSyncCpuWaitContext cpuWaitContext;
        std::vector<NvSciBufObj> sciBufObjs; // one per packet
        std::thread producerThread;
        BufferInfo bufferInfo[NUM_PACKETS];
        std::atomic<uint32_t> numBuffersWithConsumer;
    } ProducerStream;

 public:
    SIPLStatus Setup(NvSciBufModule *bufModule, NvSciSyncModule *syncModule)
    {
        if ((bufModule == nullptr) || (syncModule == nullptr)) {
            LOG_ERR("Setup: Received unexpected nullptr\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        // Camera Master setup
        m_pCamera = INvSIPLCamera::GetInstance();
        CHK_PTR_AND_RETURN(m_pCamera, "INvSIPLCamera::GetInstance()");

        NvSciError sciErr = NvSciBufModuleOpen(&m_sciBufModule);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufModuleOpen");

        sciErr = NvSciSyncModuleOpen(&m_sciSyncModule);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncModuleOpen");

        *bufModule = m_sciBufModule;
        *syncModule = m_sciSyncModule;

        return NVSIPL_STATUS_OK;
    }

#if !NV_IS_SAFETY
    SIPLStatus SetCompositorID(uint32_t pip,
                               INvSIPLClient::ConsumerDesc::OutputType outputType,
                               CComposite *pComposite,
                               uint32_t id)
    {
        const uint32_t outIndex = static_cast<uint32_t>(outputType);
        if ((pip >= MAX_SENSORS) || (outIndex >= MAX_OUTPUTS_PER_SENSOR)) {
            LOG_ERR("Invalid input identifier for compositor ID\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }
        if (m_pComposite == nullptr) {
            m_pComposite = pComposite;
            // Initialize array of compositor IDs with invalid values
            for (uint32_t i = 0U; i < MAX_SENSORS; i++) {
                for (uint32_t j = 0U; j < MAX_OUTPUTS_PER_SENSOR; j++) {
                    m_compositorIDs[i][j] = UINT32_MAX;
                }
            }
        }
        m_compositorIDs[pip][outIndex] = id;
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus SetupNvSciSync2DSignalerWaiter(uint32_t pip,
                                              std::vector<INvSIPLClient::ConsumerDesc::OutputType> outputList,
                                              bool isSimulatorMode)
    {
        static constexpr uint32_t NUM_SYNC_INTERFACES = 2U;
        static constexpr uint32_t MAX_NUM_SYNC_ACTORS = 3U;
        static constexpr uint32_t MAX_NUM_DUP_ISP_SYNC_OBJS = (MAX_OUTPUTS_PER_SENSOR - 2U);
        NvMediaNvSciSyncClientType clientType2d[NUM_SYNC_INTERFACES] = { NVMEDIA_WAITER,
                                                                         NVMEDIA_SIGNALER };
        NvSiplNvSciSyncClientType clientTypeSipl[NUM_SYNC_INTERFACES] = { SIPL_SIGNALER,
                                                                          SIPL_WAITER };
        NvMediaNvSciSyncObjType syncObjType2d[NUM_SYNC_INTERFACES] = { NVMEDIA_PRESYNCOBJ,
                                                                       NVMEDIA_EOFSYNCOBJ };
        NvSiplNvSciSyncObjType syncObjTypeSipl[NUM_SYNC_INTERFACES] = { NVSIPL_EOFSYNCOBJ,
                                                                        NVSIPL_PRESYNCOBJ };
        uint32_t uNumIspStreamsEnabled = 0U;
        uint32_t uNumIspStreamsSeen = 0U;
        std::unique_ptr<NvSciSyncObj, CloseNvSciSyncObj> ispDupSyncObjs[MAX_NUM_DUP_ISP_SYNC_OBJS] {nullptr};

        if (pip >= MAX_SENSORS) {
            LOG_ERR("Invalid pipeline identifier for NvSciSync setup\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }
        for (INvSIPLClient::ConsumerDesc::OutputType outputType : outputList) {
            const uint32_t outIndex = static_cast<uint32_t>(outputType);
            if (outIndex >= MAX_OUTPUTS_PER_SENSOR) {
                LOG_ERR("Invalid output identifier for NvSciSync setup\n");
                return NVSIPL_STATUS_BAD_ARGUMENT;
            }
            if (outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP) {
                uNumIspStreamsEnabled++;
            }
        }

        // Set CPU waiter/signaler attributes, they will get used later if necessary
        NvSciSyncAttrKeyValuePair keyVals[2];
        bool cpuAccess = true;
        NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
        keyVals[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
        keyVals[0].value = (void *)&cpuAccess;
        keyVals[0].len = sizeof(cpuAccess);
        keyVals[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
        keyVals[1].value = (void*)&cpuPerm;
        keyVals[1].len = sizeof(cpuPerm);

        SIPLStatus status = NVSIPL_STATUS_OK;
        for (INvSIPLClient::ConsumerDesc::OutputType outputType : outputList) {
            const uint32_t outIndex = static_cast<uint32_t>(outputType);
            for (uint32_t i = 0U; i < NUM_SYNC_INTERFACES; i++) {
                if ((syncObjTypeSipl[i] == NVSIPL_EOFSYNCOBJ)
                    && (outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP) // ISP stream
                    && (uNumIspStreamsSeen != 0U)) { // ISP sync object has already been allocated
                    // Register previously allocated sync object with NvMedia 2D
                    status = m_pComposite->RegisterNvSciSyncObj(m_compositorIDs[pip][outIndex],
                                                                syncObjType2d[i],
                                                                std::move(ispDupSyncObjs[uNumIspStreamsSeen - 1U]));
                    CHK_STATUS_AND_RETURN(status, "2D RegisterNvSciSyncObj");
                } else {
                    // Create attribute lists (with automatic destructors)
                    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> attrListSipl;
                    attrListSipl.reset(new NvSciSyncAttrList());
                    NvSciError sciErr = NvSciSyncAttrListCreate(m_sciSyncModule, attrListSipl.get());
                    CHK_NVSCISTATUS_AND_RETURN(sciErr, "SIPL NvSciSyncAttrListCreate");
                    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> attrList2d;
                    attrList2d.reset(new NvSciSyncAttrList());
                    sciErr = NvSciSyncAttrListCreate(m_sciSyncModule, attrList2d.get());
                    CHK_NVSCISTATUS_AND_RETURN(sciErr, "2D NvSciSyncAttrListCreate");
                    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> reconciledAttrList;
                    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> conflictAttrList;
                    reconciledAttrList.reset(new NvSciSyncAttrList());
                    conflictAttrList.reset(new NvSciSyncAttrList());
                    // Fill attribute lists
                    if (isSimulatorMode && (outputType == INvSIPLClient::ConsumerDesc::OutputType::ICP)) {
                        // There is no hardware engine involved so set CPU attributes
                        cpuPerm = (syncObjTypeSipl[i] == NVSIPL_EOFSYNCOBJ) ?
                            NvSciSyncAccessPerm_SignalOnly : NvSciSyncAccessPerm_WaitOnly;
                        sciErr = NvSciSyncAttrListSetAttrs(*attrListSipl, keyVals, 2);
                        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListSetAttrs");
                    } else {
                        status = m_pCamera->FillNvSciSyncAttrList(pip,
                                                                  outputType,
                                                                  *attrListSipl,
                                                                  clientTypeSipl[i]);
                        CHK_STATUS_AND_RETURN(status, "SIPL FillNvSciSyncAttrList");
                    }
                    status = m_pComposite->FillNvSciSyncAttrList(m_compositorIDs[pip][outIndex],
                                                                 *attrList2d,
                                                                 clientType2d[i]);
                    CHK_STATUS_AND_RETURN(status, "2D FillNvSciSyncAttrList");
                    NvSciSyncAttrList attrListsForReconcile[MAX_NUM_SYNC_ACTORS] = { *attrListSipl,
                                                                                     *attrList2d };
                    uint32_t uNumAttrLists = 2U;
                    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> attrListCustom;
                    if (syncObjTypeSipl[i] == NVSIPL_EOFSYNCOBJ) {
                        attrListCustom.reset(new NvSciSyncAttrList());
                        sciErr = NvSciSyncAttrListCreate(m_sciSyncModule, attrListCustom.get());
                        CHK_NVSCISTATUS_AND_RETURN(sciErr, "Custom NvSciSyncAttrListCreate");
                        // Add CPU waiter attributes for profiling and file writing use cases
                        cpuPerm = NvSciSyncAccessPerm_WaitOnly;
                        sciErr = NvSciSyncAttrListSetAttrs(*attrListCustom, keyVals, 2);
                        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListSetAttrs");
                        attrListsForReconcile[uNumAttrLists] = *attrListCustom;
                        uNumAttrLists++;
                    }
                    // Reconcile attribute lists
                    sciErr = NvSciSyncAttrListReconcile(attrListsForReconcile,
                                                        uNumAttrLists,
                                                        reconciledAttrList.get(),
                                                        conflictAttrList.get());
                    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListReconcile");
                    // Allocate sync object
                    std::unique_ptr<NvSciSyncObj, CloseNvSciSyncObj> syncObj;
                    syncObj.reset(new NvSciSyncObj());
                    CHK_PTR_AND_RETURN(syncObj, "NvSciSyncObj creation");
                    sciErr = NvSciSyncObjAlloc(*reconciledAttrList, syncObj.get());
                    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncObjAlloc");
                    if ((syncObjTypeSipl[i] == NVSIPL_EOFSYNCOBJ)
                        && (outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP) // ISP stream
                        && (uNumIspStreamsSeen == 0U)) { // ISP sync object hasn't been allocated yet
                        // Duplicate this sync object so it can be used by the other ISP streams
                        for (uint32_t j = 0U; j < (uNumIspStreamsEnabled - 1U); j++) {
                            ispDupSyncObjs[j].reset(new NvSciSyncObj());
                            CHK_PTR_AND_RETURN(ispDupSyncObjs[j], "NvSciSyncObj creation");
                            sciErr = NvSciSyncObjDup(*syncObj, ispDupSyncObjs[j].get());
                            CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncObjDup");
                        }
                    }
                    // Register sync object
                    if (!isSimulatorMode || (outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP)) {
                        status = m_pCamera->RegisterNvSciSyncObj(pip,
                                                                 outputType,
                                                                 syncObjTypeSipl[i],
                                                                 *syncObj);
                        CHK_STATUS_AND_RETURN(status, "SIPL RegisterNvSciSyncObj");
                    }
                    status = m_pComposite->RegisterNvSciSyncObj(m_compositorIDs[pip][outIndex],
                                                                syncObjType2d[i],
                                                                std::move(syncObj));
                    CHK_STATUS_AND_RETURN(status, "2D RegisterNvSciSyncObj");
                }
            }
            if (outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP) {
                uNumIspStreamsSeen++;
            }
        }

        return NVSIPL_STATUS_OK;
    }
#endif // !NV_IS_SAFETY

    SIPLStatus SetTwoPassIspSync(uint32_t producerIdx, uint32_t consumerIdx)
    {
        if ((producerIdx >= MAX_SENSORS) || (consumerIdx >= MAX_SENSORS)) {
            LOG_ERR("Invalid pipeline identifier for two-pass ISP NvSciSync setup\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        // Set up synchronization so that the ISP output of the SIPL consumer/reprocessing pipeline
        // can signal the capture output of the SIPL producer/capture pipeline; also add CPU wait
        // functionality

        // Create attribute lists (with automatic destructors)
        std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> attrListSignaler;
        attrListSignaler.reset(new NvSciSyncAttrList());
        NvSciError sciErr = NvSciSyncAttrListCreate(m_sciSyncModule, attrListSignaler.get());
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "Signaler NvSciSyncAttrListCreate");
        std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> attrListWaiter;
        attrListWaiter.reset(new NvSciSyncAttrList());
        sciErr = NvSciSyncAttrListCreate(m_sciSyncModule, attrListWaiter.get());
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "Waiter NvSciSyncAttrListCreate");
        std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> reconciledAttrList;
        std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> conflictAttrList;
        reconciledAttrList.reset(new NvSciSyncAttrList());
        conflictAttrList.reset(new NvSciSyncAttrList());

        // Fill attribute lists
        // Use ISP0 for the consumer/reprocessing pipeline because the synchronization attributes
        // are the same for all ISP outputs
        // For the producer/capture pipeline it is guaranteed that the synchronization is with the
        // capture output
        SIPLStatus status = m_pCamera->FillNvSciSyncAttrList(consumerIdx,
                                                    INvSIPLClient::ConsumerDesc::OutputType::ISP0,
                                                    *attrListSignaler,
                                                    SIPL_SIGNALER);
        CHK_STATUS_AND_RETURN(status, "Signaler FillNvSciSyncAttrList");
        status = m_pCamera->FillNvSciSyncAttrList(producerIdx,
                                                  INvSIPLClient::ConsumerDesc::OutputType::ICP,
                                                  *attrListWaiter,
                                                  SIPL_WAITER);
        CHK_STATUS_AND_RETURN(status, "Waiter FillNvSciSyncAttrList");

        // Create and initialize CPU waiter attribute list
        std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> attrListCpuWaiter;
        attrListCpuWaiter.reset(new NvSciSyncAttrList());
        sciErr = NvSciSyncAttrListCreate(m_sciSyncModule, attrListCpuWaiter.get());
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "CPU waiter NvSciSyncAttrListCreate");
        NvSciSyncAttrKeyValuePair keyValue[2U];
        bool cpuAccess = true;
        keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
        keyValue[0].value = (void *)&cpuAccess;
        keyValue[0].len = sizeof(cpuAccess);
        NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
        keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
        keyValue[1].value = (void *)&cpuPerm;
        keyValue[1].len = sizeof(cpuPerm);
        sciErr = NvSciSyncAttrListSetAttrs(*attrListCpuWaiter, keyValue, 2U);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "CPU waiter NvSciSyncAttrListSetAttrs");

        // Reconcile attribute lists
        NvSciSyncAttrList attrListsForReconcile[3U] = { *attrListSignaler,
                                                        *attrListWaiter,
                                                        *attrListCpuWaiter };
        sciErr = NvSciSyncAttrListReconcile(attrListsForReconcile,
                                            3U,
                                            reconciledAttrList.get(),
                                            conflictAttrList.get());
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListReconcile");

        // Allocate sync object
        std::unique_ptr<NvSciSyncObj, CloseNvSciSyncObj> syncObj;
        syncObj.reset(new NvSciSyncObj());
        CHK_PTR_AND_RETURN(syncObj, "NvSciSyncObj creation");
        sciErr = NvSciSyncObjAlloc(*reconciledAttrList, syncObj.get());
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncObjAlloc");

        // Register sync object
        // Use ISP0 for the consumer/reprocessing pipeline because only one ISP synchronization
        // object is allowed and the specific choice of output type doesn't matter
        // For the producer/capture pipeline it is guaranteed that the synchronization is with the
        // capture output
        status = m_pCamera->RegisterNvSciSyncObj(consumerIdx,
                                                 INvSIPLClient::ConsumerDesc::OutputType::ISP0,
                                                 NVSIPL_EOFSYNCOBJ,
                                                 *syncObj);
        CHK_STATUS_AND_RETURN(status, "Signaler RegisterNvSciSyncObj");
        status = m_pCamera->RegisterNvSciSyncObj(producerIdx,
                                                 INvSIPLClient::ConsumerDesc::OutputType::ICP,
                                                 NVSIPL_PRESYNCOBJ,
                                                 *syncObj);
        CHK_STATUS_AND_RETURN(status, "Waiter RegisterNvSciSyncObj");

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus SetNvSciSyncCPUWaiter(uint32_t pip, bool isp0Enabled, bool isp1Enabled, bool isp2Enabled)
    {
        if (!isp0Enabled && !isp1Enabled && !isp2Enabled)
        {
            return NVSIPL_STATUS_OK;
        }

        NvSciSyncAttrList signalerAttrList;
        NvSciSyncAttrList waiterAttrList;

        // SIPL signalers across all ISP outputs will be the same and thus only need to create a
        // single attribute list
        auto sciErr = NvSciSyncAttrListCreate(m_sciSyncModule,
                                              &signalerAttrList);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer signaler NvSciSyncAttrListCreate");

        // All ISP outputs have the same signaler attributes, so just using ISP0
        auto status = m_pCamera->FillNvSciSyncAttrList(pip,
                                                       INvSIPLClient::ConsumerDesc::OutputType::ISP0,
                                                       signalerAttrList,
                                                       SIPL_SIGNALER);
        CHK_STATUS_AND_RETURN(status, "Producer signaler INvSIPLCamera::FillNvSciSyncAttrList");

        // Waiters across all ISP outputs will be the same for our purposes (CPU wait) and thus
        // only need a single attribute list to represent all consumers
        sciErr = NvSciSyncAttrListCreate(m_sciSyncModule,
                                         &waiterAttrList);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer waiter NvSciSyncAttrListCreate");

        // Create application's NvSciSync attributes for CPU waiting, reconcile with SIPL's signaler
        // attributes, create NvSciSyncObj with the reconciled attributes, register the object with
        // SIPL as EOF sync obj
        NvSciSyncAttrList unreconciledLists[2];
        NvSciSyncAttrList reconciledList = NULL;
        NvSciSyncAttrList conflictList = NULL;

        NvSciSyncAttrKeyValuePair keyValue[2];
        bool cpuWaiter = true;
        keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
        keyValue[0].value = (void *)&cpuWaiter;
        keyValue[0].len = sizeof(cpuWaiter);
        NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
        keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
        keyValue[1].value = (void*)&cpuPerm;
        keyValue[1].len = sizeof(cpuPerm);
        sciErr = NvSciSyncAttrListSetAttrs(waiterAttrList, keyValue, 2);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "CPU waiter NvSciSyncAttrListSetAttrs");

        size_t inputCount = 0U;
        unreconciledLists[inputCount++] = signalerAttrList;
        unreconciledLists[inputCount++] = waiterAttrList;

        // Reconcile the  waiter and signaler through the unreconciledLists
        sciErr = NvSciSyncAttrListReconcile(unreconciledLists,
                                            inputCount,
                                            &reconciledList,
                                            &conflictList);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "Signaler and waiter NvSciSyncAttrListReconcile");

        NvSciSyncAttrListFree(signalerAttrList);
        NvSciSyncAttrListFree(waiterAttrList);

        NvSciSyncObj syncObj;

        // Allocate the sync object
        sciErr = NvSciSyncObjAlloc(reconciledList, &syncObj);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "Reconciled signaler and waiter NvSciSyncObjAlloc");

        NvSciSyncAttrListFree(reconciledList);
        if (conflictList != nullptr) {
            NvSciSyncAttrListFree(conflictList);
        }

        // Register with SIPL, SIPL expects to register only one NvSciSyncObj for all the enabled
        // ISP outputs of a given pipeline
        status = m_pCamera->RegisterNvSciSyncObj(pip,
                                                 INvSIPLClient::ConsumerDesc::OutputType::ISP0,
                                                 NVSIPL_EOFSYNCOBJ,
                                                 syncObj);
        CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::RegisterNvSciSyncObj");

        NvSciSyncObjFree(syncObj);

        return NVSIPL_STATUS_OK;
    }

#if !NV_IS_SAFETY
    SIPLStatus SetCharMode(uint32_t uIndex, uint8_t expNo)
    {
        return m_pCamera->SetSensorCharMode(uIndex, expNo);
    }
#endif // !NV_IS_SAFETY

    SIPLStatus SetPlatformConfig(PlatformCfg* pPlatformCfg, NvSIPLDeviceBlockQueues &queues, bool bIsParkingStream)
    {
        m_isParkingStream = bIsParkingStream;
        return m_pCamera->SetPlatformCfg(pPlatformCfg, queues);
    }

    SIPLStatus SetPipelineConfig(uint32_t uIndex, NvSIPLPipelineConfiguration &pipelineCfg, NvSIPLPipelineQueues &pipelineQueues)
    {
        return m_pCamera->SetPipelineCfg(uIndex, pipelineCfg, pipelineQueues);
    }

    SIPLStatus RegisterAutoControl(uint32_t uIndex, PluginType type, ISiplControlAuto* customPlugin, std::vector<uint8_t>& blob)
    {
        return m_pCamera->RegisterAutoControlPlugin(uIndex, type, customPlugin, blob);
    }

#if !NV_IS_SAFETY
    SIPLStatus ToggleLED(uint32_t uIndex, bool enable)
    {
        return m_pCamera->ToggleLED(uIndex, enable);
    }
#endif // !NV_IS_SAFETY

    SIPLStatus Init()
    {
        auto status = m_pCamera->Init();
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("NvSIPLCamera Init failed\n");
            return status;
        }

        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus Start()
    {
        return m_pCamera->Start();
    }

    virtual SIPLStatus Stop()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_pCamera->Stop();
    }

    virtual void Deinit(void)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        auto status = m_pCamera->Deinit();
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("INvSIPLCamera::Deinit failed. status: %x\n", status);
        }

        for (uint32_t i = 0U; i < MAX_SENSORS; i++) {
            for (uint32_t j = 0U; j < MAX_OUTPUTS_PER_SENSOR; j++) {
                for (NvSciBufObj &bufObj : m_sciBufObjs[i][j]) {
#if !NV_IS_SAFETY
                    if ((m_pComposite != nullptr) && (m_compositorIDs[i][j] != UINT32_MAX)) {
                        status = m_pComposite->UnregisterNvSciBufObj(m_compositorIDs[i][j], bufObj);
                        if (status != NVSIPL_STATUS_OK) {
                            LOG_ERR("2D UnregisterNvSciBufObj failed. status: %x\n", status);
                        }
                    }
#endif // !NV_IS_SAFETY
                    NvSciBufObjFree(bufObj);
                }
            }
        }

        if (m_sciBufModule != NULL) {
            NvSciBufModuleClose(m_sciBufModule);
            m_sciBufModule = NULL;
        }

        if (m_sciSyncModule != NULL) {
            NvSciSyncModuleClose(m_sciSyncModule);
            m_sciSyncModule = NULL;
        }
    }

    SIPLStatus GetMaxErrorSize(const uint32_t devBlkIndex, size_t &size)
    {
        return m_pCamera->GetMaxErrorSize(devBlkIndex, size);
    }

    SIPLStatus GetErrorGPIOEventInfo(const uint32_t devBlkIndex,
                                     const uint32_t gpioIndex,
                                     SIPLGpioEvent &event)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_pCamera->GetErrorGPIOEventInfo(devBlkIndex, gpioIndex, event);
    }

    SIPLStatus GetDeserializerErrorInfo(const uint32_t devBlkIndex,
                                        SIPLErrorDetails * const deserializerErrorInfo,
                                        bool & isRemoteError,
                                        uint8_t& linkErrorMask)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_pCamera->GetDeserializerErrorInfo(devBlkIndex, deserializerErrorInfo,
                                                   isRemoteError, linkErrorMask);
    }

    SIPLStatus GetModuleErrorInfo(const uint32_t index,
                                  SIPLErrorDetails * const serializerErrorInfo,
                                  SIPLErrorDetails * const sensorErrorInfo)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_pCamera->GetModuleErrorInfo(index, serializerErrorInfo, sensorErrorInfo,
                                             NVSIPL_MODULE_ERROR_READ_ALL);
    }

    SIPLStatus DisableLink(uint32_t index)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_pCamera->DisableLink(index);
    }

    SIPLStatus EnableLink(uint32_t index, bool resetModule)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_pCamera->EnableLink(index, resetModule);
    }

    SIPLStatus ShowEEPROM(uint32_t pip)
    {
        SIPLStatus status = NVSIPL_STATUS_OK;
        uint16_t addr = 0x4a;
        uint32_t length = 4;
        uint8_t buf[length];

        memset(buf, 0, sizeof(buf));

        LOG_MSG("Performing example EEPROM register access\n");
        LOG_MSG("Reading %u byte(s) at 0x%X\n", length, addr);

        status = m_pCamera->ReadEEPROMData(pip, addr, length, buf);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("Failed to read EEPROM data via SIPL API");
            goto done;
        }

        LOG_MSG("Read 0x%X 0x%X 0x%X 0x%X\n",
            buf[0], buf[1], buf[2], buf[3]);

    done:
        return status;
    }

    SIPLStatus CaptureFillAttributesAllocateSciBuf(uint32_t pip)
    {
        NvSciBufAttrList reconciledAttrlist;
        NvSciBufAttrList unreconciledAttrlist = NULL;
        NvSciBufAttrList conflictlist = NULL;
        NvSciError err = NvSciBufAttrListCreate(m_sciBufModule, &unreconciledAttrlist);
        CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListCreate");
        // Need CPU Read permission for RAW->RGB conversion on compositor
        // Need CPU Write permission for FileReader
        // TODO: Determine the permission based on exact config instead of hardcoded value.
        NvSciBufType bufType = NvSciBufType_Image;
        NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_ReadWrite;
        NvSciBufAttrKeyValuePair attrKvp[] = {
            { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
            { NvSciBufGeneralAttrKey_RequiredPerm, &accessPerm, sizeof(accessPerm) }
        };
        err = NvSciBufAttrListSetAttrs(unreconciledAttrlist, attrKvp, 2);
        CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListSetAttrs");
        auto status = m_pCamera->GetImageAttributes(pip, INvSIPLClient::ConsumerDesc::OutputType::ICP, unreconciledAttrlist);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("GetImageAttributes failed\n");
            return status;
        }

        err = NvSciBufAttrListReconcile(&unreconciledAttrlist, 1, &reconciledAttrlist, &conflictlist);
        CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListReconcile");

        for (auto i = 0U; i < NUM_CAPTURE_BUFFERS_PER_POOL; i++) {
            NvSciBufObj imageGrpSciBufObj;
            err = NvSciBufObjAlloc(reconciledAttrlist, &imageGrpSciBufObj);
            CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufObjAlloc");
            const uint32_t outIndex = static_cast<uint32_t>(INvSIPLClient::ConsumerDesc::OutputType::ICP);
            m_sciBufObjs[pip][outIndex].push_back(imageGrpSciBufObj);
        }

        if (conflictlist != NULL) {
            NvSciBufAttrListFree(conflictlist);
        }

        if (unreconciledAttrlist != NULL) {
            NvSciBufAttrListFree(unreconciledAttrlist);
        }

        if (reconciledAttrlist != NULL) {
            NvSciBufAttrListFree(reconciledAttrlist);
        }
        return NVSIPL_STATUS_OK;
    }

    void PrintISPOutputFormat(uint32_t pip,
                              INvSIPLClient::ConsumerDesc::OutputType outputType,
                              NvSciBufAttrList attrlist)
    {
        NvSciBufAttrKeyValuePair keyVals[] = {
            { NvSciBufImageAttrKey_PlaneCount, NULL, 0 },
        };

        size_t length = sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair);
        if (NvSciBufAttrListGetAttrs(attrlist, keyVals, length) != NvSciError_Success) {
            LOG_ERR("NvSciBufAttrListGetAttrs Failed");
            return;
        }

        uint32_t planeCount = *(uint32_t*) keyVals[0].value;

        if (planeCount == 1) {
            NvSciBufAttrKeyValuePair keyVals[] = {
                { NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0 },
                { NvSciBufImageAttrKey_PlaneColorStd, NULL, 0 },
                { NvSciBufImageAttrKey_Layout, NULL, 0 },
            };

            size_t length = sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair);
            if (NvSciBufAttrListGetAttrs(attrlist, keyVals, length) != NvSciError_Success) {
                LOG_ERR("NvSciBufAttrListGetAttrs Failed");
                return;
            }

            NvSciBufAttrValColorFmt planeColorFormat = *(NvSciBufAttrValColorFmt*)keyVals[0].value;
            NvSciBufAttrValColorStd planeColorStd = *(NvSciBufAttrValColorStd*)keyVals[1].value;
            NvSciBufAttrValImageLayoutType layoutType =
                    *(NvSciBufAttrValImageLayoutType*)keyVals[2].value;

            std::string layout = "PL";
            std::string colorStd = "SENSOR_RGBA";
            std::string colorFormat = "RGBA PACKED FLOAT16";
            if (layoutType == NvSciBufImage_BlockLinearType) {
                layout = "BL";
            }
            if (planeColorStd == NvSciColorStd_REC709_ER) {
                colorStd = "REC709_ER";
            } else if (planeColorStd == NvSciColorStd_SRGB) {
                colorStd = "SRGB";
            }
            if (planeColorFormat == NvSciColor_Y16) {
                colorFormat = "LUMA PACKED UINT16";
            } else if (planeColorFormat == NvSciColor_A8Y8U8V8) {
                colorFormat = "VUYX PACKED UINT8";
            } else if (planeColorFormat == NvSciColor_A16Y16U16V16) {
                colorFormat = "VUYX PACKED UINT16";
            }

            cout << "Pipeline: " << pip
                 << " ISP Output: " << ((uint32_t)outputType - 1)
                 << " is using "
                 << colorFormat << " " << layout << " " << colorStd << "\n";

        } else {
            NvSciBufAttrKeyValuePair keyVals[] = {
                { NvSciBufImageAttrKey_SurfMemLayout, NULL, 0 },
                { NvSciBufImageAttrKey_SurfColorStd, NULL, 0 },
                { NvSciBufImageAttrKey_Layout, NULL, 0 },
                { NvSciBufImageAttrKey_SurfSampleType, NULL, 0 },
                { NvSciBufImageAttrKey_SurfBPC, NULL, 0 },
            };

            size_t length = sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair);
            if (NvSciBufAttrListGetAttrs(attrlist, keyVals, length) != NvSciError_Success) {
                LOG_ERR("NvSciBufAttrListGetAttrs Failed");
                return;
            }

            NvSciBufSurfMemLayout surfMemLayout = *(NvSciBufSurfMemLayout*)keyVals[0].value;
            if (surfMemLayout != NvSciSurfMemLayout_SemiPlanar) {
                LOG_ERR("Only Semi Planar surfaces are supported with Surf attributes");
                return;
            }

            NvSciBufAttrValColorStd surfColorStd = *(NvSciBufAttrValColorStd*)keyVals[1].value;
            if (surfColorStd != NvSciColorStd_REC709_ER) {
                LOG_ERR("Color standard not supported for YUV images");
                return;
            }
            NvSciBufAttrValImageLayoutType layoutType =
                    *(NvSciBufAttrValImageLayoutType*)keyVals[2].value;
            NvSciBufSurfSampleType surfSampleType = *(NvSciBufSurfSampleType*)keyVals[3].value;
            NvSciBufSurfBPC surfBPC = *(NvSciBufSurfBPC*)keyVals[4].value;

            std::string layout = "PL";
            std::string sampleType = "420";
            std::string bpc = "8";

            if (layoutType == NvSciBufImage_BlockLinearType) {
                layout = "BL";
            }
            if (surfSampleType == NvSciSurfSampleType_444) {
                sampleType = "444";
            }
            if (surfBPC == NvSciSurfBPC_16) {
                bpc = "16";
            }
            cout << "Pipeline: " << pip
                 << " ISP Output: " << ((uint32_t)outputType - 1)
                 << " is using "
                 << "YUV " << sampleType << " SEMI-PLANAR UINT" << bpc
                 << " " << layout << " REC_709ER" << "\n";
        }

    }

    SIPLStatus OverrideImageAttributes(NvSciBufAttrList attrlist, ISPOutputFormats fmt)
    {
        SIPLStatus status = NVSIPL_STATUS_OK;
        if ((fmt >= FMT_UPPER_BOUND) || (fmt <= FMT_LOWER_BOUND)) {
            LOG_ERR("Unsupported format\n");
            return NVSIPL_STATUS_ERROR;
        }
        if (fmt < FMT_VUYX_UINT8_BL) {
            /* For YUV Semi Planar Images we use High Level NvSciBuf Attributes
             * in case the user wants any format apart from FMT_YUV_420SP_UINT8_PL
             * the user is expected to set the following attributes directly as shown below
             * 1) NvSciBufGeneralAttrKey_Types, value NvSciBufType_Image
             * 2) NvSciBufImageAttrKey_SurfType, value NvSciSurfType_YUV
             * 3) NvSciBufImageAttrKey_SurfBPC, valid value one of NvSciSurfBPC_8 or 16
             * 4) NvSciBufImageAttrKey_SurfMemLayout, value NvSciSurfMemLayout_SemiPlanar
             * 5) NvSciBufImageAttrKey_SurfSampleType, valid value one of  NvSciSurfSampleType_420 or 444
             * 6) NvSciBufImageAttrKey_SurfComponentOrder, value NvSciSurfComponentOrder_YUV
             * 7) NvSciBufImageAttrKey_SurfColorStd, value NvSciColorStd_REC709_ER
             * 8) NvSciBufImageAttrKey_Layout, , valid NvSciBufAttrValImageLayoutType
             */
            NvSciBufAttrValImageLayoutType layout = NvSciBufImage_BlockLinearType;
            NvSciBufSurfSampleType surfSampleType = NvSciSurfSampleType_420;
            NvSciBufSurfBPC surfBPC = NvSciSurfBPC_8;
            switch(fmt) {
            case FMT_YUV_420SP_UINT8_PL:
                layout = NvSciBufImage_PitchLinearType;
                surfSampleType = NvSciSurfSampleType_420;
                surfBPC = NvSciSurfBPC_8;
                break;
            case FMT_YUV_420SP_UINT16_PL:
                layout = NvSciBufImage_PitchLinearType;
                surfSampleType = NvSciSurfSampleType_420;
                surfBPC = NvSciSurfBPC_16;
                break;
            case FMT_YUV_444SP_UINT8_PL:
                layout = NvSciBufImage_PitchLinearType;
                surfSampleType = NvSciSurfSampleType_444;
                surfBPC = NvSciSurfBPC_8;
                break;
            case FMT_YUV_444SP_UINT16_PL:
                layout = NvSciBufImage_PitchLinearType;
                surfSampleType = NvSciSurfSampleType_444;
                surfBPC = NvSciSurfBPC_16;
                break;
            case FMT_YUV_420SP_UINT8_BL:
                layout = NvSciBufImage_BlockLinearType;
                surfSampleType = NvSciSurfSampleType_420;
                surfBPC = NvSciSurfBPC_8;
                break;
            case FMT_YUV_420SP_UINT16_BL:
                layout = NvSciBufImage_BlockLinearType;
                surfSampleType = NvSciSurfSampleType_420;
                surfBPC = NvSciSurfBPC_16;
                break;
            case FMT_YUV_444SP_UINT8_BL:
                layout = NvSciBufImage_BlockLinearType;
                surfSampleType = NvSciSurfSampleType_444;
                surfBPC = NvSciSurfBPC_8;
                break;
            case FMT_YUV_444SP_UINT16_BL:
                layout = NvSciBufImage_BlockLinearType;
                surfSampleType = NvSciSurfSampleType_444;
                surfBPC = NvSciSurfBPC_16;
                break;
                default:
                    LOG_ERR("Unsupported format\n");
                    status = NVSIPL_STATUS_ERROR;
            }
            NvSciBufType bufType = NvSciBufType_Image;
            NvSciBufSurfType surfType = NvSciSurfType_YUV;
            NvSciBufSurfMemLayout surfMemLayout = NvSciSurfMemLayout_SemiPlanar;
            NvSciBufSurfComponentOrder surfCompOrder = NvSciSurfComponentOrder_YUV;
            NvSciBufAttrValColorStd surfColorStd[] = { NvSciColorStd_REC709_ER };
            NvSciBufAttrKeyValuePair keyVals[] = {
                { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
                { NvSciBufImageAttrKey_SurfType, &surfType, sizeof(surfType) },
                { NvSciBufImageAttrKey_SurfBPC, &surfBPC, sizeof(surfBPC) },
                { NvSciBufImageAttrKey_SurfMemLayout, &surfMemLayout, sizeof(surfMemLayout)},
                { NvSciBufImageAttrKey_SurfSampleType, &surfSampleType, sizeof(surfSampleType) },
                { NvSciBufImageAttrKey_SurfComponentOrder, &surfCompOrder, sizeof(surfCompOrder) },
                { NvSciBufImageAttrKey_SurfColorStd, &surfColorStd, sizeof(surfColorStd) },
                { NvSciBufImageAttrKey_Layout, &layout, sizeof(layout) },
            };
            size_t length = sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair);
            auto err = NvSciBufAttrListSetAttrs(attrlist, keyVals, length);
            CHK_NVSCISTATUS_AND_RETURN(err,"NvSciBufAttrListSetAttrs Failed");
        } else {
            /* For Packed Images we use Low Level NvSciBuf Attributes
             * In case a user wants to use a single plane/packed formats
             * the user is expected to set the following attributes directly
             * 1) NvSciBufGeneralAttrKey_Types, value NvSciBufType_Image
             * 2) NvSciBufImageAttrKey_PlaneCount, value 1
             * 3) NvSciBufImageAttrKey_Layout, valid NvSciBufAttrValImageLayoutType
             * 4) NvSciBufImageAttrKey_PlaneColorFormat, valid and supported NvSciBufAttrValColorFmt
             * 5) NvSciBufImageAttrKey_PlaneColorStd, valid and supported NvSciBufAttrValColorStd
             */
            NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
            NvSciBufAttrValColorFmt colorFormat[] = { NvSciColor_A8Y8U8V8 };
            NvSciBufAttrValColorStd colorStd[] = { NvSciColorStd_REC709_ER };
            switch(fmt) {
                case FMT_VUYX_UINT8_BL:
                    layout = NvSciBufImage_BlockLinearType;
                    colorFormat[0] = { NvSciColor_A8Y8U8V8 };
                    colorStd[0] = { NvSciColorStd_REC709_ER };
                    break;
                case FMT_VUYX_UINT8_PL:
                    layout = NvSciBufImage_PitchLinearType;
                    colorFormat[0] = { NvSciColor_A8Y8U8V8 };
                    colorStd[0] = { NvSciColorStd_REC709_ER };
                    break;
                case FMT_VUYX_UINT16_PL:
                    layout = NvSciBufImage_PitchLinearType;
                    colorFormat[0] = NvSciColor_A16Y16U16V16;
                    colorStd[0] = { NvSciColorStd_REC709_ER };
                    break;
                case FMT_RGBA_FLOAT16_PL:
                    layout = NvSciBufImage_PitchLinearType;
                    colorFormat[0] = NvSciColor_Float_A16B16G16R16;
                    colorStd[0] = NvSciColorStd_SENSOR_RGBA;
                    break;
#if !NV_IS_SAFETY
                case FMT_LUMA_UINT16_PL:
                    layout = NvSciBufImage_PitchLinearType;
                    colorFormat[0] = NvSciColor_Y16;
                    colorStd[0] = NvSciColorStd_REC709_ER;
                    break;
#endif // !NV_IS_SAFETY
                default:
                    LOG_ERR("Unsupported format\n");
                    status = NVSIPL_STATUS_ERROR;
                }
            NvSciBufType bufType = NvSciBufType_Image;
            uint32_t planeCount = 1;

            NvSciBufAttrKeyValuePair keyVals[] = {
                { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
                { NvSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(planeCount) },
                { NvSciBufImageAttrKey_Layout, &layout, sizeof(layout) },
                { NvSciBufImageAttrKey_PlaneColorFormat, colorFormat, sizeof(NvSciBufAttrValColorFmt) },
                { NvSciBufImageAttrKey_PlaneColorStd, colorStd, sizeof(NvSciBufAttrValColorStd) * planeCount },
            };
            size_t length = sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair);
            auto err = NvSciBufAttrListSetAttrs(attrlist, keyVals, length);
            CHK_NVSCISTATUS_AND_RETURN(err,"NvSciBufAttrListSetAttrs Failed");
        }

        return status;
    }

    SIPLStatus IMX728IMX623SurfTypeOverride(uint32_t pip,
                                            INvSIPLClient::ConsumerDesc::OutputType outputType,
                                            NvSciBufAttrList attrlist)
    {
        SIPLStatus status = NVSIPL_STATUS_OK;
        if (pip == 0U) {
            // This is to override the surface type for IMX728 for DOS-SHR-5819
            switch (outputType) {
                case INvSIPLClient::ConsumerDesc::OutputType::ISP0:
                case INvSIPLClient::ConsumerDesc::OutputType::ISP1:
                    // Override surface type to YUV 444 SEMI-PLANAR UINT8 PL
                    status = OverrideImageAttributes(attrlist, FMT_YUV_444SP_UINT8_PL);
                    break;
                case INvSIPLClient::ConsumerDesc::OutputType::ISP2:
                    // Override surface type to RGBA FP16 PL
                    status = OverrideImageAttributes(attrlist, FMT_RGBA_FLOAT16_PL);
                    break;
                default:
                    LOG_ERR("Invalid output type\n");
                    status = NVSIPL_STATUS_BAD_ARGUMENT;
            }
        } else if ((pip >= 4U) && (pip <= 7U)) {
            // This is to override the surface type for IMX623 for DOS-SHR-5819
            switch (outputType) {
                case INvSIPLClient::ConsumerDesc::OutputType::ISP0:
                    // Override surface type to YUV 444 SEMI-PLANAR UINT16 PL
                    status = OverrideImageAttributes(attrlist, FMT_YUV_444SP_UINT16_PL);
                    break;
                case INvSIPLClient::ConsumerDesc::OutputType::ISP1:
                    // Override surface type to YUV 444 SEMI-PLANAR UINT8 PL
                    status = OverrideImageAttributes(attrlist, FMT_YUV_444SP_UINT8_PL);
                    break;
                case INvSIPLClient::ConsumerDesc::OutputType::ISP2:
                    // Override surface type to RGBA FP16 PL
                    status = OverrideImageAttributes(attrlist, FMT_RGBA_FLOAT16_PL);
                    break;
                default:
                    LOG_ERR("Invalid output type\n");
                    status =  NVSIPL_STATUS_BAD_ARGUMENT;
            }
        } else {
            LOG_ERR("Invalid pipeline index\n");
            status = NVSIPL_STATUS_BAD_ARGUMENT;
        }

        return status;
    }

    SIPLStatus ProcessFillAttributesAllocateSciBuf(uint32_t pip,
                                                   std::string platformConfigName,
                                                   INvSIPLClient::ConsumerDesc::OutputType outType,
                                                   bool bDisplay,
                                                   bool bFileWrite)
    {
        NvSciBufAttrList reconciledAttrlist;
        NvSciBufAttrList unreconciledAttrlist = NULL;
        NvSciBufAttrList conflictlist = NULL;
        NvSciError err = NvSciBufAttrListCreate(m_sciBufModule, &unreconciledAttrlist);
        CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListCreate");
        // Need CPU Read permission for RAW->RGB conversion on compositor
        // Need CPU Write permission for FileReader
        // TODO: Determine the permission based on exact config instead of hardcoded value.
        NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_ReadWrite;
        NvSciBufAttrKeyValuePair attrKvp = { NvSciBufGeneralAttrKey_RequiredPerm,
                                             &accessPerm,
                                             sizeof(accessPerm) };
        err = NvSciBufAttrListSetAttrs(unreconciledAttrlist, &attrKvp, 1);
        CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListSetAttrs");

        if (m_isParkingStream) {
            auto status = IMX728IMX623SurfTypeOverride(pip, outType, unreconciledAttrlist);
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("IMX728IMX623SurfTypeOverride failed\n");
                return status;
            }
        }
#if !NV_IS_SAFETY
        else if (bDisplay) {
            if (outType == INvSIPLClient::ConsumerDesc::OutputType::ISP2) {
                // VIC does not support "floating point" or "16bit" input hence we override ISP2 format
                auto status = OverrideImageAttributes(unreconciledAttrlist, FMT_YUV_420SP_UINT8_BL);
                if (status != NVSIPL_STATUS_OK) {
                    return status;
                }
            }
        }
#endif // !NV_IS_SAFETY

        // CPU_ACCESS and CACHED attributes are needed for for FileWriter
        bool isCpuAcccessReq = bFileWrite;
        bool isCpuCacheEnabled = bFileWrite;
        NvSciBufAttrKeyValuePair setAttrs[] = {
            { NvSciBufGeneralAttrKey_NeedCpuAccess, &isCpuAcccessReq, sizeof(isCpuAcccessReq) },
            { NvSciBufGeneralAttrKey_EnableCpuCache, &isCpuCacheEnabled, sizeof(isCpuCacheEnabled) },
        };
        err = NvSciBufAttrListSetAttrs(unreconciledAttrlist, setAttrs, 2);
        CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListSetAttrs");

        auto status = m_pCamera->GetImageAttributes(pip, outType, unreconciledAttrlist);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("GetImageAttributes failed\n");
            return status;
        }

        err = NvSciBufAttrListReconcile(&unreconciledAttrlist, 1, &reconciledAttrlist, &conflictlist);
        CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListReconcile");

        for (auto i = 0u; i < NUM_ISP_BUFFERS_PER_POOL; i++) {
            NvSciBufObj imageGrpSciBufObj;
            err = NvSciBufObjAlloc(reconciledAttrlist, &imageGrpSciBufObj);
            CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufObjAlloc");
            const uint32_t outIndex = static_cast<uint32_t>(outType);
            m_sciBufObjs[pip][outIndex].push_back(imageGrpSciBufObj);
        }

        PrintISPOutputFormat(pip, outType, unreconciledAttrlist);

        if (conflictlist != NULL) {
            NvSciBufAttrListFree(conflictlist);
        }

        if (unreconciledAttrlist != NULL) {
            NvSciBufAttrListFree(unreconciledAttrlist);
        }

        if (reconciledAttrlist != NULL) {
            NvSciBufAttrListFree(reconciledAttrlist);
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus AllocateAndRegisterBuffers(uint32_t pip,
                                          std::string platformConfigName,
                                          bool isRawEnabled,
                                          bool isp0Enabled,
                                          bool isp1Enabled,
                                          bool isp2Enabled,
                                          bool bFileWrite,
                                          bool bTwoPassIsp)
    {
        bool registerWithSipl[MAX_OUTPUTS_PER_SENSOR] = {true, isp0Enabled, isp1Enabled, isp2Enabled};
#if !NV_IS_SAFETY
        bool registerWithCompositor[MAX_OUTPUTS_PER_SENSOR] = {isRawEnabled, isp0Enabled, isp1Enabled, isp2Enabled};
#endif // !NV_IS_SAFETY
        INvSIPLClient::ConsumerDesc::OutputType outputType[MAX_OUTPUTS_PER_SENSOR] = {
            INvSIPLClient::ConsumerDesc::OutputType::ICP,
            INvSIPLClient::ConsumerDesc::OutputType::ISP0,
            INvSIPLClient::ConsumerDesc::OutputType::ISP1,
            INvSIPLClient::ConsumerDesc::OutputType::ISP2
        };

        SIPLStatus status = NVSIPL_STATUS_OK;
        for (uint32_t i = 0U; i < MAX_OUTPUTS_PER_SENSOR; i++) {
            if (registerWithSipl[i]) {
                // For two-pass ISP processing, capture buffers will be shared across two pipelines.
                // To accommodate this, create a variable to store the ID of the pipeline whose
                // buffers should be used. In the default case this will be the same as the pipeline
                // ID but in the case of the two-pass consumer/reprocessing pipeline it will be the
                // two-pass producer/capture pipeline.
                uint32_t uBufObjIdx = pip;
                if (outputType[i] == INvSIPLClient::ConsumerDesc::OutputType::ICP) {
                    // If using two-pass ISP and the pipeline ID of the shared buffers hasn't been
                    // set yet, set it
                    if (bTwoPassIsp && (m_uSharedBufObjIdx == UINT32_MAX)) {
                        m_uSharedBufObjIdx = pip;
                    }
                    if ((!bTwoPassIsp) || (pip == m_uSharedBufObjIdx)) {
                        // Allocate buffers if 1) not using two-pass ISP or 2) using two-pass ISP
                        // but this pipeline is the one whose buffers are to be shared
                        status = CaptureFillAttributesAllocateSciBuf(pip);
                    } else {
                        // For two-pass ISP where this is the pipeline to share the
                        // previously-allocated buffers, set the buffer pipeline ID to that of the
                        // shared buffers
                        uBufObjIdx = m_uSharedBufObjIdx;
                    }
                } else {
                    status = ProcessFillAttributesAllocateSciBuf(pip,
                                                                 platformConfigName,
                                                                 outputType[i],
#if NV_IS_SAFETY
                                                                 false,
#else
                                                                 (m_pComposite != nullptr),
#endif // NV_IS_SAFETY
                                                                 bFileWrite);
                }
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("FillAttributesAllocateSciBuf failed for output type:%u\n", outputType[i]);
                    return status;
                }
                const uint32_t outIndex = static_cast<uint32_t>(outputType[i]);
                status = m_pCamera->RegisterImages(pip, outputType[i], m_sciBufObjs[uBufObjIdx][outIndex]);
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("RegisterImages for failed for output type:%u\n", outIndex);
                    return status;
                }
#if !NV_IS_SAFETY
                if ((m_pComposite != nullptr) && registerWithCompositor[i]) {
                    for (NvSciBufObj &bufObj : m_sciBufObjs[pip][outIndex]) {
                        status = m_pComposite->RegisterNvSciBufObj(m_compositorIDs[pip][outIndex], bufObj);
                        CHK_STATUS_AND_RETURN(status, "2D RegisterNvSciBufObj");
                    }
                }
#endif // !NV_IS_SAFETY
            }
        }

        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus AllocateNvSciBuffers(ProducerStream *pStream)
    {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    virtual SIPLStatus AllocateBuffers(uint32_t uSensor,
                                       bool isp0Enabled,
                                       bool isp1Enabled,
                                       bool isp2Enabled)
    {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    virtual SIPLStatus RegisterImageGroups(ProducerStream *pStream)
    {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    virtual SIPLStatus RegisterImages(ProducerStream *pStream)
    {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    virtual SIPLStatus RegisterBuffers(uint32_t pip,
                                       bool isp0Enabled,
                                       bool isp1Enabled,
                                       bool isp2Enabled)
    {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    virtual SIPLStatus RegisterSource(uint32_t uSensor,
                                      INvSIPLClient::ConsumerDesc::OutputType outputType,
                                      bool isSimulatorMode,
                                      bool streamingEnabled,
                                      NvSciStreamBlock *consumer,
                                      NvSciStreamBlock *consumerUpstream,
                                      NvSciStreamBlock *queue,
                                      bool bDisplay)
    {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    virtual SIPLStatus Post(uint32_t uSensor,
                            INvSIPLClient::ConsumerDesc::OutputType outputType,
                            INvSIPLClient::INvSIPLNvMBuffer *pBuffer)
    {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    virtual SIPLStatus SetupElements()
    {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    virtual SIPLStatus SetupBuffers()
    {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    virtual SIPLStatus SetupSync(bool bNeedCpuWaiter)
    {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    virtual SIPLStatus SetupComplete()
    {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    OV2311NonFuSaCustomInterface* ov2311NonFuSaCustomInterface = nullptr;

    OV2311NonFuSaCustomInterface* GetOV2311NonFuSaCustomInterface(const uint32_t uSensor) {
        IInterfaceProvider* moduleInterfaceProvider = nullptr;
        OV2311NonFuSaCustomInterface* ov2311NonFuSaCustomInterface = nullptr;

        // Get the interface provider
        SIPLStatus status = m_pCamera->GetModuleInterfaceProvider(uSensor,
            moduleInterfaceProvider);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("Error %d while getting module interface provider for sensor ID: %d\n",
                status, uSensor);
        } else if (moduleInterfaceProvider != nullptr) {
            // Get the custom interface and cast
            ov2311NonFuSaCustomInterface = static_cast<OV2311NonFuSaCustomInterface*>
                (moduleInterfaceProvider->GetInterface(OV2311_NONFUSA_CUSTOM_INTERFACE_ID));
            if (ov2311NonFuSaCustomInterface != nullptr) {
                // Verify that the ID matches expected - we have the correct custom interface
                if (ov2311NonFuSaCustomInterface->getInstanceInterfaceID() ==
                    OV2311_NONFUSA_CUSTOM_INTERFACE_ID) {
                    LOG_DBG("OV2311 (non-fusa) custom interface found\n");
                } else {
                    LOG_ERR("Incorrect interface obtained from module\n");
                    // Set the return pointer to NULL because the obtained pointer
                    // does not point to correct interface.
                    ov2311NonFuSaCustomInterface = nullptr;
                }
            }
        }

        return ov2311NonFuSaCustomInterface;
    }

 protected:
    unique_ptr<INvSIPLCamera> m_pCamera;
    NvSciBufModule m_sciBufModule {NULL};
    NvSciSyncModule m_sciSyncModule {NULL};
    bool m_isParkingStream {false};

 private:
    std::mutex m_mutex;
    std::vector<NvSciBufObj> m_sciBufObjs[MAX_SENSORS][MAX_OUTPUTS_PER_SENSOR];
    uint32_t m_uSharedBufObjIdx {UINT32_MAX};
#if !NV_IS_SAFETY
    CComposite *m_pComposite {nullptr};
    uint32_t m_compositorIDs[MAX_SENSORS][MAX_OUTPUTS_PER_SENSOR];
#endif // !NV_IS_SAFETY
};

#endif //CNVSIPLMASTER_HPP
