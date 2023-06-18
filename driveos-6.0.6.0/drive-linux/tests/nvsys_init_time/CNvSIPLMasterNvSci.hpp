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

#include "NvSIPLCamera.hpp" // Camera
#include "NvSIPLPipelineMgr.hpp" // Pipeline manager
#include "CNvSIPLMaster.hpp" // Master base class
#include "CUtils.hpp"

#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvscistream.h"

#ifndef CNVSIPLMASTERNVSCI_HPP
#define CNVSIPLMASTERNVSCI_HPP

#define MAX_SENSORS (16U)
#define STREAMING_TIMEOUT (250000U)
#define SIMULATOR_MODE_CPU_WAIT_TIMEOUT_US (100000U)
#define MAX_OTHER_ISP_STREAMS (2U)
#define MAX_ATTR_LISTS_TO_RECONCILE (5U)

using namespace std;
using namespace nvsipl;

/** NvSIPL Master class */
class CNvSIPLMasterNvSci final : public CNvSIPLMaster
{
 public:
    SIPLStatus Start(void)
    {
        m_bRunning = true;
        for (ProducerStream *pStream : m_usedStreams) {
            // Spawn a thread to collect and release used buffers
            pStream->numBuffersWithConsumer = 0U;
            pStream->producerThread = std::thread(&CNvSIPLMasterNvSci::ThreadFunc, this, pStream);
            LOG_INFO("Created producer thread: ID:%u\n", pStream->producerThread.get_id());
        }

        const SIPLStatus status = m_pCamera->Start();
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("Failed to start SIPL\n");
            return status;
        }

        return status;
    }

    SIPLStatus Stop(void)
    {
        const SIPLStatus status = m_pCamera->Stop();
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("Failed to stop SIPL\n");
            return status;
        }

        // Signal producer threads to stop
        m_bRunning = false;

        // Wait for the producer threads
        for (ProducerStream *pStream : m_usedStreams) {
            LOG_INFO("Waiting to join producer thread: ID:%u\n", pStream->producerThread.get_id());
            pStream->producerThread.join();
        }

        return status;
    }

    void Deinit()
    {
        auto status = m_pCamera->Deinit();
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("INvSIPLCamera::Deinit failed. status: %x\n", status);
        }

        for (auto i = 0u; i < MAX_SENSORS; i++) {
          for (auto j = 0u; j < MAX_OUTPUTS_PER_SENSOR; j++) {
            ProducerStream *pStream = &m_streams[i][j];
            if (pStream->bufAttrList != nullptr) {
              NvSciBufAttrListFree(pStream->bufAttrList);
            }
            if (pStream->signalerAttrList != nullptr) {
              NvSciSyncAttrListFree(pStream->signalerAttrList);
            }
            if (pStream->waiterAttrList != nullptr) {
              NvSciSyncAttrListFree(pStream->waiterAttrList);
            }
            if (pStream->consumerAttrList != nullptr) {
              NvSciSyncAttrListFree(pStream->consumerAttrList);
            }
            if (pStream->cpuWaitContext != nullptr) {
              NvSciSyncCpuWaitContextFree(pStream->cpuWaitContext);
            }
            for (NvSciBufObj bufObj : pStream->sciBufObjs) {
              if (bufObj != nullptr) {
                NvSciBufObjFree(bufObj);
              }
            }
            if (pStream->objFromConsumer != nullptr) {
              NvSciSyncObjFree(pStream->objFromConsumer);
            }
            if (pStream->producerSyncObj != nullptr) {
              NvSciSyncObjFree(pStream->producerSyncObj);
            }
            if (pStream->queue != 0U) {
              NvSciStreamBlockDelete(pStream->queue);
            }
            if (pStream->downstream != 0U) {
              NvSciStreamBlockDelete(pStream->downstream);
            }
            if (pStream->staticPool != 0U) {
              NvSciStreamBlockDelete(pStream->staticPool);
            }
            if (pStream->producer != 0U) {
              NvSciStreamBlockDelete(pStream->producer);
            }
            std::vector<NvSciBufObj>().swap(pStream->sciBufObjs);

          }
        }

        vector<ProducerStream*>().swap(m_usedStreams);
        if (m_sciBufModule != NULL) {
          NvSciBufModuleClose(m_sciBufModule);
        }

        if (m_sciSyncModule != NULL) {
          NvSciSyncModuleClose(m_sciSyncModule);
        }
    }

    SIPLStatus AllocateNvSciBuffers(ProducerStream *pStream)
    {
        pStream->sciBufObjs.resize(NUM_PACKETS);
        for(auto i = 0u; i < NUM_PACKETS; i++) {
           auto sciErr = NvSciBufObjAlloc(pStream->bufAttrList, &pStream->sciBufObjs[i]);
           CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjAlloc");
        }
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus AllocateBuffers(uint32_t uSensor,
                               bool isp0Enabled,
                               bool isp1Enabled,
                               bool isp2Enabled)
    {
        SIPLStatus status;

        status = AllocateNvSciBuffers(&m_streams[uSensor][0]);
        CHK_STATUS_AND_RETURN(status, "ICP CNvSIPLMaster::AllocateNvSciBuffers");

        if (isp0Enabled) {
            status = AllocateNvSciBuffers(&m_streams[uSensor][1]);
            CHK_STATUS_AND_RETURN(status, "ISP0 CNvSIPLMaster::AllocateNvSciBuffers");
        }

        if (isp1Enabled) {
            status = AllocateNvSciBuffers(&m_streams[uSensor][2]);
            CHK_STATUS_AND_RETURN(status, "ISP1 CNvSIPLMaster::AllocateNvSciBuffers");
        }

        if (isp2Enabled) {
            status = AllocateNvSciBuffers(&m_streams[uSensor][3]);
            CHK_STATUS_AND_RETURN(status, "ISP2 CNvSIPLMaster::AllocateNvSciBuffers");
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus RegisterImages(ProducerStream *pStream)
    {
        auto status = m_pCamera->RegisterImages(pStream->uSensor, pStream->outputType, pStream->sciBufObjs);
        if (status != NVSIPL_STATUS_OK) {
            LOG_WARN("RegisterImages for Capture failed\n");
            return status;
        }
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus RegisterBuffers(uint32_t pip,
                               bool isp0Enabled,
                               bool isp1Enabled,
                               bool isp2Enabled)
    {
        SIPLStatus status;
        status = RegisterImages(&m_streams[pip][0]);
        CHK_STATUS_AND_RETURN(status, "RegisterImages for ICP");

        if (isp0Enabled) {
            status = RegisterImages(&m_streams[pip][1]);
            CHK_STATUS_AND_RETURN(status, "RegisterImages for ISP0");
        }

        if (isp1Enabled) {
            status = RegisterImages(&m_streams[pip][2]);
            CHK_STATUS_AND_RETURN(status, "RegisterImages for ISP1");
        }

        if (isp2Enabled) {
            status = RegisterImages(&m_streams[pip][3]);
            CHK_STATUS_AND_RETURN(status, "RegisterImages for ISP2");
        }

        return NVSIPL_STATUS_OK;
    }


    SIPLStatus RegisterSource(uint32_t uSensor,
                              INvSIPLClient::ConsumerDesc::OutputType outputType,
                              bool isSimulatorMode,
                              bool streamingEnabled,
                              NvSciStreamBlock *consumer,
                              NvSciStreamBlock *consumerUpstream,
                              NvSciStreamBlock *queue,
                              bool bDisplay)
    {
        NvSciStreamEventType eventType;
        NvSciBufAttrList reconciledAttrlist = nullptr;
        NvSciBufAttrList unreconciledAttrlist = nullptr;
        NvSciBufAttrList conflictlist = nullptr;
        ProducerStream *pStream = &m_streams[uSensor][int(outputType)];

        if (streamingEnabled) {
            if ((consumer == nullptr) || (consumerUpstream == nullptr) || (queue == nullptr)) {
                LOG_ERR("RegisterSource: Received unexpected nullptr\n");
                return NVSIPL_STATUS_BAD_ARGUMENT;
            }

            auto sciErr = NvSciStreamStaticPoolCreate(NUM_PACKETS, &pStream->staticPool);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamStaticPoolCreate");

            sciErr = NvSciStreamProducerCreate(pStream->staticPool, &pStream->producer);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamProducerCreate");

            pStream->downstream = *consumer;
            *consumerUpstream = pStream->producer;
            pStream->queue = *queue;

            sciErr = NvSciStreamBlockConnect(pStream->producer, pStream->downstream);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockConnect");

            sciErr = NvSciStreamBlockEventQuery(pStream->staticPool, STREAMING_TIMEOUT, &eventType);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Static Pool NvSciStreamBlockEventQuery");
            if (eventType != NvSciStreamEventType_Connected) {
                LOG_ERR("RegisterSource: Did not receive expected static pool Connected event\n");
                return NVSIPL_STATUS_ERROR;
            }

            sciErr = NvSciStreamBlockEventQuery(pStream->producer, STREAMING_TIMEOUT, &eventType);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockEventQuery");
            if (eventType != NvSciStreamEventType_Connected) {
                LOG_ERR("RegisterSource: Did not receive expected producer Connected event\n");
                return NVSIPL_STATUS_ERROR;
            }

            sciErr = NvSciStreamBlockEventQuery(pStream->queue, STREAMING_TIMEOUT, &eventType);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Queue NvSciStreamBlockEventQuery");
            if (eventType != NvSciStreamEventType_Connected) {
                LOG_ERR("RegisterSource: Did not receive expected queue Connected event\n");
                return NVSIPL_STATUS_ERROR;
            }

            sciErr = NvSciStreamBlockEventQuery(pStream->downstream, STREAMING_TIMEOUT, &eventType);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciStreamBlockEventQuery");
            if (eventType != NvSciStreamEventType_Connected) {
                LOG_ERR("RegisterSource: Did not receive expected consumer Connected event\n");
                return NVSIPL_STATUS_ERROR;
            }
        }
        auto sciErr = NvSciBufAttrListCreate(m_sciBufModule, &unreconciledAttrlist);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate");
        // Need CPU Read permission for RAW->RGB conversion on compositor
        // Need CPU Write permission for FileReader
        // TODO: Determine the permission based on exact config instead of hardcoded value.
        NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_ReadWrite;
        NvSciBufAttrKeyValuePair attrKvp = { NvSciBufGeneralAttrKey_RequiredPerm,
                                                &accessPerm,
                                                sizeof(accessPerm)};
        sciErr = NvSciBufAttrListSetAttrs(unreconciledAttrlist, &attrKvp, 1);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

        if ((outputType == INvSIPLClient::ConsumerDesc::OutputType::ISP0) ||
            (outputType == INvSIPLClient::ConsumerDesc::OutputType::ISP1) ||
            (outputType == INvSIPLClient::ConsumerDesc::OutputType::ISP2)) {
            // Add CPU_ACCESS_CACHED attribute if not already set, to be backward compatible
            bool isCpuAcccessReq = true;
            bool isCpuCacheEnabled = true;

            NvSciBufAttrKeyValuePair setAttrs[] = {
                { NvSciBufGeneralAttrKey_NeedCpuAccess, &isCpuAcccessReq, sizeof(isCpuAcccessReq) },
                { NvSciBufGeneralAttrKey_EnableCpuCache, &isCpuCacheEnabled, sizeof(isCpuCacheEnabled) },
            };
            sciErr = NvSciBufAttrListSetAttrs(unreconciledAttrlist, setAttrs, 2);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");
        }

        if (bDisplay && (outputType == INvSIPLClient::ConsumerDesc::OutputType::ISP2)) {
            // Set buffer attributes to be YUV, semi-planar, pitch-linear, UINT8 for display workflow
            SIPLStatus status = OverrideImageAttributes(unreconciledAttrlist, FMT_YUV_420SP_UINT8_PL);
            CHK_STATUS_AND_RETURN(status, "OverrideImageAttributes");
        }

        auto status = m_pCamera->GetImageAttributes(uSensor, outputType, unreconciledAttrlist);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("GetImageAttributes failed\n");
            return status;
        }

        pStream->uSensor = uSensor;
        pStream->outputType = outputType;
        pStream->isSimulatorMode = isSimulatorMode;

        if (streamingEnabled) {
            pStream->bufAttrList = unreconciledAttrlist;
            m_usedStreams.push_back(pStream);
        } else {
            auto sciErr = NvSciBufAttrListReconcile(&unreconciledAttrlist, 1, &reconciledAttrlist, &conflictlist);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciBufAttrListReconcile");
            pStream->bufAttrList = reconciledAttrlist;
            NvSciBufAttrListFree(unreconciledAttrlist);
            if (conflictlist != nullptr) {
                NvSciBufAttrListFree(conflictlist);
            }
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus Post(uint32_t uSensor,
                    INvSIPLClient::ConsumerDesc::OutputType outputType,
                    INvSIPLClient::INvSIPLNvMBuffer *pBuffer)
    {
        ProducerStream *pStream = &m_streams[uSensor][int(outputType)];
        NvSciSyncFence fence = NvSciSyncFenceInitializer;
        NvSciStreamCookie cookie = 0U;

        CHK_PTR_AND_RETURN(pBuffer, "Post INvSIPLClient::INvSIPLNvMBuffer");
        CHK_PTR_AND_RETURN(pStream, "Post ProducerStream");

        if (!pStream->isSimulatorMode || (pStream->outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP)) {
            auto status = pBuffer->GetEOFNvSciSyncFence(&fence);
            CHK_STATUS_AND_RETURN(status, "INvSIPLClient::INvSIPLNvMBuffer::GetEOFNvSciSyncFence");
        }
        NvSciBufObj sciBufObj = pBuffer->GetNvSciBufImage();
        CHK_PTR_AND_RETURN(sciBufObj, "INvSIPLClient::INvSIPLNvMBuffer::GetNvSciBufImage");
        for (uint32_t i = 0U; i < NUM_PACKETS; i++) {
            if (sciBufObj == pStream->sciBufObjs[i]) {
                cookie = i + 1U;
                break;
            }
        }
        if (cookie == 0U) {
            // Didn't find matching buffer so cookie was not set
            LOG_ERR("Failed to get cookie for buffer\n");
            return NVSIPL_STATUS_ERROR;
        }

        // Subtract one from cookie to get index (since it was incremented on initialization)
        pStream->bufferInfo[cookie - 1U].buffer = pBuffer;
        pBuffer->AddRef();

        auto sciErr = NvSciStreamBlockPacketFenceSet(pStream->producer,
                                                     pStream->bufferInfo[cookie - 1U].packet,
                                                     0U,
                                                     &fence);
        if (sciErr != NvSciError_Success) {
            LOG_ERR("NvSciStreamBlockPacketFenceSet failed\n");
            return NVSIPL_STATUS_ERROR;
        }

        sciErr = NvSciStreamProducerPacketPresent(pStream->producer,
                                                  pStream->bufferInfo[cookie - 1U].packet);
        if (sciErr != NvSciError_Success) {
            LOG_ERR("NvSciStreamProducerPacketPresent failed\n");
            return NVSIPL_STATUS_ERROR;
        }

        pStream->numBuffersWithConsumer++;
        NvSciSyncFenceClear(&fence);

        return NVSIPL_STATUS_OK;
    }

    // All required sync attributes are available if this is an ICP stream or the last ISP stream
    SIPLStatus AreSyncAttrsAvailable(INvSIPLClient::ConsumerDesc::OutputType outputType,
                                     uint32_t uNumIspStreamsEnabled,
                                     ProducerStream *pOtherIspStreams[MAX_OTHER_ISP_STREAMS],
                                     bool &bAvail)
    {
        // ICP streams do not share sync attributes with other streams, the required attributes are
        // therefore available
        if (outputType == INvSIPLClient::ConsumerDesc::OutputType::ICP) {
            bAvail = true;
            return NVSIPL_STATUS_OK;
        }
        if (uNumIspStreamsEnabled < 1U) {
            LOG_ERR("Number of enabled ISP streams should only be zero if the output is ICP\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }
        if (uNumIspStreamsEnabled > (MAX_OTHER_ISP_STREAMS + 1U)) {
            LOG_ERR("Invalid number of enabled ISP streams\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }
        if (pOtherIspStreams == nullptr) {
            LOG_ERR("Invalid array of pointers to the other ISP streams\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }
        // Check if all consumer attributes have been collected from the other ISP streams
        bool isConsumerAttrListMissing = false;
        for (uint32_t i = 0U; i < (uNumIspStreamsEnabled - 1U); i++) {
            if ((pOtherIspStreams[i] == nullptr) || (pOtherIspStreams[i]->consumerAttrList == nullptr)) {
                isConsumerAttrListMissing = true;
                break;
            }
        }
        bAvail = (!isConsumerAttrListMissing);
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus SetupSync(bool bNeedCpuWaiter)
    {
        NvSciStreamEventType eventType;

        // Get camera waiter and signaler NvSciSync attributes and send waiter attributes to consumer
        for (ProducerStream *pStream : m_usedStreams) {
            NvSciError sciErr = NvSciSyncAttrListCreate(m_sciSyncModule, &pStream->signalerAttrList);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer signaler NvSciSyncAttrListCreate");
            sciErr = NvSciSyncAttrListCreate(m_sciSyncModule, &pStream->waiterAttrList);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer waiter NvSciSyncAttrListCreate");

            // Set CPU waiter attributes, they will get used later if necessary
            NvSciSyncAttrKeyValuePair keyVals[2];
            bool cpuAccess = true;
            NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
            keyVals[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
            keyVals[0].value = (void *)&cpuAccess;
            keyVals[0].len = sizeof(cpuAccess);
            keyVals[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
            keyVals[1].value = (void*)&cpuPerm;
            keyVals[1].len = sizeof(cpuPerm);

            if (bNeedCpuWaiter) {
                sciErr = NvSciSyncAttrListCreate(m_sciSyncModule, &pStream->cpuAttrList);
                CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer CPU NvSciSyncAttrListCreate");
                sciErr = NvSciSyncAttrListSetAttrs(pStream->cpuAttrList, keyVals, 2);
                CHK_NVSCISTATUS_AND_RETURN(sciErr, "CPU waiter NvSciSyncAttrListSetAttrs");
            }

            if (pStream->isSimulatorMode && (pStream->outputType == INvSIPLClient::ConsumerDesc::OutputType::ICP)) {
                // CFileReader is the source for ICP, so fill CPU waiter and signaler attributes
                sciErr = NvSciSyncAttrListSetAttrs(pStream->waiterAttrList, keyVals, 2);
                CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer waiter NvSciSyncAttrListSetAttrs for simulator mode");
                cpuPerm = NvSciSyncAccessPerm_SignalOnly;
                sciErr = NvSciSyncAttrListSetAttrs(pStream->signalerAttrList, keyVals, 2);
                CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer signaler NvSciSyncAttrListSetAttrs for simulator mode");
                sciErr = NvSciSyncCpuWaitContextAlloc(m_sciSyncModule, &pStream->cpuWaitContext);
                CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciSyncCpuWaitContextAlloc for simulator mode");
            } else {
                SIPLStatus status = m_pCamera->FillNvSciSyncAttrList(pStream->uSensor,
                                                                     pStream->outputType,
                                                                     pStream->signalerAttrList,
                                                                     SIPL_SIGNALER);
                CHK_STATUS_AND_RETURN(status, "Signaler INvSIPLCamera::FillNvSciSyncAttrList");
                status = m_pCamera->FillNvSciSyncAttrList(pStream->uSensor,
                                                          pStream->outputType,
                                                          pStream->waiterAttrList,
                                                          SIPL_WAITER);
                CHK_STATUS_AND_RETURN(status, "Waiter INvSIPLCamera::FillNvSciSyncAttrList");
            }

            sciErr = NvSciStreamBlockElementWaiterAttrSet(pStream->producer, 0U, pStream->waiterAttrList);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockElementWaiterAttrSet");
            sciErr = NvSciStreamBlockSetupStatusSet(pStream->producer, NvSciStreamSetup_WaiterAttrExport, true);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockSetupStatusSet(WaiterAttrExport)");
        }

       // Determine how many ISP streams are enabled for each sensor
       uint32_t numIspStreamsEnabled[MAX_SENSORS] = {0U};
       for (ProducerStream *pStream : m_usedStreams) {
           if (pStream->outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP) {
               numIspStreamsEnabled[pStream->uSensor]++;
           }
       }

       // Get consumer's NvSciSync attributes, reconcile with our own signaler attributes, create
       // producer NvSciSyncObjs with the reconciled attributes, register the objects with Camera
       // as EOF sync objs, and send objects to the consumer.
       for (ProducerStream *pStream : m_usedStreams) {
          NvSciSyncAttrList unreconciledList[MAX_ATTR_LISTS_TO_RECONCILE];
          NvSciSyncAttrList reconciledList = NULL;
          NvSciSyncAttrList newConflictList = NULL;

          auto sciErr = NvSciStreamBlockEventQuery(pStream->producer, STREAMING_TIMEOUT, &eventType);
          CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer WaiterAttr NvSciStreamBlockEventQuery");

          if (eventType != NvSciStreamEventType_WaiterAttr) {
              LOG_ERR("SetupSync: did not receive expected WaiterAttr event\n");
              return NVSIPL_STATUS_ERROR;
          }
          sciErr = NvSciStreamBlockElementWaiterAttrGet(pStream->producer, 0U, &pStream->consumerAttrList);
          CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockElementWaiterAttrGet");
          sciErr = NvSciStreamBlockSetupStatusSet(pStream->producer, NvSciStreamSetup_WaiterAttrImport, true);
          CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockSetupStatusSet(WaiterAttrImport)");

          // If two/three ISP outputs are enabled for a single sensor, they should share one NvSciSyncObj
          // Get a pointer to the other ISP streams for this sensor for later use
          ProducerStream *pOtherIspStreams[MAX_OTHER_ISP_STREAMS] = {nullptr};
          if (pStream->outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP) {
              if (numIspStreamsEnabled[pStream->uSensor] == 3U) {
                  if (pStream->outputType == INvSIPLClient::ConsumerDesc::OutputType::ISP0) {
                      pOtherIspStreams[0] = &m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP1];
                      pOtherIspStreams[1] = &m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP2];
                  } else if (pStream->outputType == INvSIPLClient::ConsumerDesc::OutputType::ISP1) {
                      pOtherIspStreams[0] = &m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP0];
                      pOtherIspStreams[1] = &m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP2];
                  } else if (pStream->outputType == INvSIPLClient::ConsumerDesc::OutputType::ISP2) {
                      pOtherIspStreams[0] = &m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP0];
                      pOtherIspStreams[1] = &m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP1];
                  }
              }
              if (numIspStreamsEnabled[pStream->uSensor] == 2U) {
                  if (pStream->outputType == INvSIPLClient::ConsumerDesc::OutputType::ISP0) {
                      if (&m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP1] != nullptr) {
                          pOtherIspStreams[0] = &m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP1];
                      }
                      else if (&m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP2] != nullptr) {
                          pOtherIspStreams[0] = &m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP2];
                      }
                  } else if (pStream->outputType == INvSIPLClient::ConsumerDesc::OutputType::ISP1) {
                      if (&m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP0] != nullptr) {
                          pOtherIspStreams[0] = &m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP0];
                      }
                      else if (&m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP2] != nullptr) {
                          pOtherIspStreams[0] = &m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP2];
                      }
                  } else if (pStream->outputType == INvSIPLClient::ConsumerDesc::OutputType::ISP2) {
                      if (&m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP1] != nullptr) {
                          pOtherIspStreams[0] = &m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP1];
                      } else if (&m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP0] != nullptr) {
                          pOtherIspStreams[0] = &m_streams[pStream->uSensor][(uint32_t)INvSIPLClient::ConsumerDesc::OutputType::ISP0];
                      }
                  }
              }
          }

          // When two or three ISP streams are sharing one NvSciSyncObj it is necessary to reconcile
          // the consumer attributes from all streams
          // If this is not the last of the two/three streams to be encountered in the loop we don't
          // yet have the consumer attributes from the other stream(s); call helper function to
          // determine whether or not to proceed with NvSciSyncObj allocation and registration
          bool bSyncAttrsAvail = false;
          SIPLStatus status = AreSyncAttrsAvailable(pStream->outputType,
                                                    numIspStreamsEnabled[pStream->uSensor],
                                                    pOtherIspStreams,
                                                    bSyncAttrsAvail);
          CHK_STATUS_AND_RETURN(status, "AreSyncAttrsAvailable");
          if (bSyncAttrsAvail) {
              size_t inputCount = 0U;
              unreconciledList[inputCount++] = pStream->signalerAttrList;
              unreconciledList[inputCount++] = pStream->consumerAttrList;
              if (bNeedCpuWaiter) {
                unreconciledList[inputCount++] = pStream->cpuAttrList;
              }

              for (auto i = 0U; i < MAX_OTHER_ISP_STREAMS; i++) {
                  // In the case of two/three ISP streams, add the attribute list(s) from the other consumer(s)
                  if (pOtherIspStreams[i] != nullptr) {
                      // It is not possible to reconcile multiple signaler attribute lists
                      // All ISP signaler attribute lists are the same anyways, so only use the
                      // consumer list(s) from the other stream(s) (omit the signaler list(s) fom the other stream(s))
                      unreconciledList[inputCount++] = pOtherIspStreams[i]->consumerAttrList;
                  }
              }

              sciErr = NvSciSyncAttrListReconcile(unreconciledList,
                                                  inputCount,
                                                  &reconciledList,
                                                  &newConflictList);
              CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciSyncAttrListReconcile");

              sciErr = NvSciSyncObjAlloc(reconciledList, &pStream->producerSyncObj);
              CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciSyncObjAlloc");

              NvSciSyncAttrListFree(reconciledList);
              if (newConflictList != nullptr) {
                  NvSciSyncAttrListFree(newConflictList);
              }

              if (!pStream->isSimulatorMode || (pStream->outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP)) {
                  auto status = m_pCamera->RegisterNvSciSyncObj(pStream->uSensor,
                                                                pStream->outputType,
                                                                NVSIPL_EOFSYNCOBJ,
                                                                pStream->producerSyncObj);
                  CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::RegisterNvSciSyncObj");
              }

              sciErr = NvSciStreamBlockElementSignalObjSet(pStream->producer, 0, pStream->producerSyncObj);
              CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockElementSignalObjSet");
              sciErr = NvSciStreamBlockSetupStatusSet(pStream->producer, NvSciStreamSetup_SignalObjExport, true);
          CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockSetupStatusSet(SignalObjExport)");

              // In the case of multiple ISP streams, duplicate the object for the other stream(s) and set
              // it with NvSciStream (don't register it with SIPL, that was already done)
              for (auto i = 0U; i < MAX_OTHER_ISP_STREAMS; i++) {
                  if (pOtherIspStreams[i] != nullptr) {
                      sciErr = NvSciSyncObjDup(pStream->producerSyncObj, &pOtherIspStreams[i]->producerSyncObj);
                      CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciSyncObjDup");

                      sciErr = NvSciStreamBlockElementSignalObjSet(pOtherIspStreams[i]->producer, 0, pOtherIspStreams[i]->producerSyncObj);
                      CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockElementSignalObjSet");

                      sciErr = NvSciStreamBlockSetupStatusSet(pOtherIspStreams[i]->producer, NvSciStreamSetup_SignalObjExport, true);
                      CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockSetupStatusSet");
                  }
              }
           }
       }

       // Get sync objs from consumer and register as presync objects.
       // Send SciBuf attributes to pool
       for(ProducerStream *pStream : m_usedStreams) {
           auto sciErr = NvSciStreamBlockEventQuery(pStream->producer, STREAMING_TIMEOUT, &eventType);
           CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer SignalObj NvSciStreamBlockEventQuery");

           if(eventType != NvSciStreamEventType_SignalObj) {
               LOG_ERR("SetupSync: did not receive expected SignalObj event type\n");
               return NVSIPL_STATUS_ERROR;
           }
           sciErr = NvSciStreamBlockElementSignalObjGet(pStream->producer, 0U, 0U, &pStream->objFromConsumer);
           CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockElementSignalObjGet");
          sciErr = NvSciStreamBlockSetupStatusSet(pStream->producer, NvSciStreamSetup_SignalObjImport, true);
          CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockSetupStatusSet(SignalObjImport)");

           if (!pStream->isSimulatorMode || (pStream->outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP)) {
               auto status = m_pCamera->RegisterNvSciSyncObj(pStream->uSensor,
                                                             pStream->outputType,
                                                             NVSIPL_PRESYNCOBJ,
                                                             pStream->objFromConsumer);
               CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::RegisterNvSciSyncObj");
           }
       }

       return NVSIPL_STATUS_OK;
    }

    SIPLStatus SetupElements(void)
    {
        NvSciStreamEventType eventType;

        for(ProducerStream *pStream : m_usedStreams) {
            auto sciErr = NvSciStreamBlockElementAttrSet(pStream->producer, 0,
                                                         pStream->bufAttrList);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockElementAttrSet");

            sciErr = NvSciStreamBlockSetupStatusSet(pStream->producer,
                                                    NvSciStreamSetup_ElementExport,
                                                    true);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockSetupStatusSet(ElementExport)");
        }

        // At pool, get buffer attributes from producer and consumer, reconcile them, send to both.
        for(ProducerStream *pStream : m_usedStreams) {
            NvSciBufAttrList unreconciledAttrlist[2] {};
            NvSciBufAttrList reconciledAttrlist = nullptr;
            NvSciBufAttrList conflictlist = nullptr;

            auto sciErr = NvSciStreamBlockEventQuery(pStream->staticPool, STREAMING_TIMEOUT, &eventType);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool Elements NvSciStreamBlockEventQuery");
            if (eventType != NvSciStreamEventType_Elements) {
                LOG_ERR("SetupElements: did not receive expected Elements event type\n");
                return NVSIPL_STATUS_ERROR;
            }

            sciErr = NvSciStreamBlockElementAttrGet(pStream->staticPool,
                                                    NvSciStreamBlockType_Consumer,
                                                    0U, nullptr,
                                                    &unreconciledAttrlist[0]);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool NvSciStreamBlockElementAttrGet(Consumer)");

            sciErr = NvSciStreamBlockElementAttrGet(pStream->staticPool,
                                                    NvSciStreamBlockType_Producer,
                                                    0U, nullptr,
                                                    &unreconciledAttrlist[1]);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool NvSciStreamBlockElementAttrGet(Producer)");

            sciErr = NvSciStreamBlockSetupStatusSet(pStream->staticPool,
                                                    NvSciStreamSetup_ElementImport,
                                                    true);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool NvSciStreamBlockSetupStatusSet(ElementImport)");

            sciErr = NvSciBufAttrListReconcile(unreconciledAttrlist, 2, &reconciledAttrlist, &conflictlist);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool NvSciBufAttrListReconcile");

            sciErr = NvSciStreamBlockElementAttrSet(pStream->staticPool, 0,
                                                    reconciledAttrlist);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool NvSciStreamBlockElementAttrSet");

            sciErr = NvSciStreamBlockSetupStatusSet(pStream->staticPool,
                                                    NvSciStreamSetup_ElementExport,
                                                    true);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool NvSciStreamBlockSetupStatusSet(ElementExport)");

            NvSciBufAttrListFree(unreconciledAttrlist[0]);
            NvSciBufAttrListFree(unreconciledAttrlist[1]);
            NvSciBufAttrListFree(reconciledAttrlist);
            if(conflictlist != nullptr) {
                NvSciBufAttrListFree(conflictlist);
            }
        }

        // Get reconciled buffer attributes from pool and replace our unreconciled list
        for(ProducerStream *pStream : m_usedStreams) {

            auto sciErr = NvSciStreamBlockEventQuery(pStream->producer, STREAMING_TIMEOUT, &eventType);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer Elements NvSciStreamBlockEventQuery");
            if(eventType != NvSciStreamEventType_Elements) {
                LOG_ERR("SetupElements: did not receive Elements event as expected\n");
                return NVSIPL_STATUS_ERROR;
            }

            NvSciBufAttrListFree(pStream->bufAttrList);  // replace it
            sciErr = NvSciStreamBlockElementAttrGet(pStream->producer,
                                                    NvSciStreamBlockType_Pool,
                                                    0U, 0U,
                                                    &pStream->bufAttrList);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockElementAttrGet");
            sciErr = NvSciStreamBlockSetupStatusSet(pStream->producer,
                                                    NvSciStreamSetup_ElementImport,
                                                    true);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockSetupStatusSet(ElementImport)");
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus SetupBuffers(void)
    {
        NvSciStreamEventType eventType;
        NvSciStreamPacket packetHandle;
        NvSciError sciErr;

        // Pool sends buffers to everyone
        for(ProducerStream *pStream : m_usedStreams) {
            for(auto i = 0U; i < NUM_PACKETS; i++) {

                sciErr = NvSciStreamPoolPacketCreate(pStream->staticPool, i+1, &packetHandle);
                CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamPoolPacketCreate");

                sciErr = NvSciStreamPoolPacketInsertBuffer(pStream->staticPool, packetHandle, 0,
                                                           pStream->sciBufObjs[i]);
                CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamPoolPacketInsertBuffer");

                sciErr = NvSciStreamPoolPacketComplete(pStream->staticPool, packetHandle);
                CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamPoolPacketComplete");
            }

            sciErr = NvSciStreamBlockSetupStatusSet(pStream->staticPool,
                                                    NvSciStreamSetup_PacketExport,
                                                    true);

            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool NvSciStreamBlockSetupStatusSet(PacketExport)");
        }

        // Producer replaces buffers
        for (ProducerStream *pStream : m_usedStreams) {
            for (uint32_t i = 0U; i < NUM_PACKETS; i++) {
                sciErr = NvSciStreamBlockEventQuery(pStream->producer,
                                                    STREAMING_TIMEOUT,
                                                    &eventType);
                CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer PacketCreate NvSciStreamBlockEventQuery");
                if(eventType != NvSciStreamEventType_PacketCreate) {
                    LOG_ERR("SetupBuffers: did not receive PacketCreate event as expected\n");
                    return NVSIPL_STATUS_ERROR;
                }

                sciErr = NvSciStreamBlockPacketNewHandleGet(pStream->producer,
                                                            &pStream->bufferInfo[i].packet);
                CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockPacketNewHandleGet");

                NvSciBufObjFree(pStream->sciBufObjs[i]); // Replace
                sciErr = NvSciStreamBlockPacketBufferGet(pStream->producer,
                                                         pStream->bufferInfo[i].packet,
                                                         0U,
                                                         &pStream->sciBufObjs[i]);
                CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockPacketBufferGet");

                // Set cookie to one more than the index
                pStream->bufferInfo[i].cookie = i + 1U;
                sciErr = NvSciStreamBlockPacketStatusSet(pStream->producer,
                                                         pStream->bufferInfo[i].packet,
                                                         pStream->bufferInfo[i].cookie,
                                                         NvSciError_Success);
                CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockPacketStatusSet");
            }

            sciErr = NvSciStreamBlockEventQuery(pStream->producer, STREAMING_TIMEOUT, &eventType);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer PacketsComplete NvSciStreamBlockEventQuery");
            if(eventType != NvSciStreamEventType_PacketsComplete) {
                LOG_ERR("SetupBuffers: did not receive PacketsComplete event as expected\n");
                return NVSIPL_STATUS_ERROR;
            }

            sciErr = NvSciStreamBlockSetupStatusSet(pStream->producer,
                                                    NvSciStreamSetup_PacketImport,
                                                    true);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockSetupStatusSet(PacketImport)");
        }

        // Pool receives acceptance from producer and consumer
        for(ProducerStream *pStream : m_usedStreams) {
            for(auto i = 0u; i < NUM_PACKETS; i++) {
                sciErr = NvSciStreamBlockEventQuery(pStream->staticPool, STREAMING_TIMEOUT, &eventType);
                CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool PacketStatus NvSciStreamBlockEventQuery");
                if(eventType != NvSciStreamEventType_PacketStatus) {
                    LOG_ERR("SetupBuffers: did not receive PacketStatus event as expected\n");
                    return NVSIPL_STATUS_ERROR;
                }
            }

            auto sciErr = NvSciStreamBlockSetupStatusSet(pStream->staticPool,
                                                         NvSciStreamSetup_PacketImport,
                                                         true);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool NvSciStreamBlockSetupStatusSet(PacketImport)");
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus SetupComplete(void)
    {
        NvSciStreamEventType eventType;
        NvSciStreamCookie cookie;

        // Producer receives notification and takes initial ownership of packets
        for (ProducerStream *pStream : m_usedStreams) {
            NvSciError sciErr = NvSciStreamBlockEventQuery(pStream->producer, STREAMING_TIMEOUT, &eventType);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer SetupComplete NvSciStreamBlockEventQuery");
            if (eventType != NvSciStreamEventType_SetupComplete) {
                LOG_ERR("Didn't receive expected SetupComplete event\n");
                return NVSIPL_STATUS_ERROR;
            }
            for (uint32_t i = 0u; i < NUM_PACKETS; i++) {
                NvSciError sciErr = NvSciStreamBlockEventQuery(pStream->producer, STREAMING_TIMEOUT, &eventType);
                if (sciErr != NvSciError_Success) {
                    LOG_ERR("Failed to get initial ownership of packet\n");
                    return NVSIPL_STATUS_ERROR;
                }
                if (eventType != NvSciStreamEventType_PacketReady) {
                    LOG_ERR("Didn't receive expected PacketReady event\n");
                    return NVSIPL_STATUS_ERROR;
                }
                sciErr = NvSciStreamProducerPacketGet(pStream->producer, &cookie);
                if (sciErr != NvSciError_Success) {
                    LOG_ERR("NvSciStreamProducerPacketGet failed\n");
                    return NVSIPL_STATUS_ERROR;
                }
            }
        }

        return NVSIPL_STATUS_OK;
    }

 private:
    void ThreadFunc(ProducerStream *pStream)
    {
        NvSciStreamCookie cookie;
        NvSciSyncFence fence = NvSciSyncFenceInitializer;
        NvSciStreamEventType eventType;
        INvSIPLClient::INvSIPLNvMBuffer *pBuffer = nullptr;

        pthread_setname_np(pthread_self(), "ProdPacketGet");

        while (m_bRunning || (pStream->numBuffersWithConsumer > 0U)) {
            pBuffer = nullptr;
            NvSciError sciErr = NvSciStreamBlockEventQuery(pStream->producer,
                                                           STREAMING_TIMEOUT,
                                                           &eventType);
            if (sciErr == NvSciError_Timeout) {
                continue;
            }
            if (sciErr != NvSciError_Success) {
                LOG_ERR("Failed to get returned packet\n");
                m_bRunning = false;
                return;
            }
            if (eventType != NvSciStreamEventType_PacketReady) {
                LOG_ERR("Didn't receive expected PacketReady event\n");
                m_bRunning = false;
                return;
            }
            sciErr = NvSciStreamProducerPacketGet(pStream->producer, &cookie);
            if (sciErr != NvSciError_Success) {
                LOG_ERR("NvSciStreamProducerPacketGet failed\n");
                m_bRunning = false;
                return;
            }
            sciErr = NvSciStreamBlockPacketFenceGet(pStream->producer,
                                                    pStream->bufferInfo[cookie - 1U].packet,
                                                    0U, 0U,
                                                    &fence);
            if (sciErr != NvSciError_Success) {
                LOG_ERR("NvSciStreamBlockPacketFenceGet failed\n");
                m_bRunning = false;
                return;
            }

            pBuffer = pStream->bufferInfo[cookie - 1U].buffer;
            pStream->numBuffersWithConsumer--;

            if (pBuffer == nullptr) {
                LOG_ERR("Couldn't match cookie to buffer\n");
                m_bRunning = false;
                return;
            }

            if (!pStream->isSimulatorMode || (pStream->outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP)) {
                const SIPLStatus status = pBuffer->AddNvSciSyncPrefence(fence);
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("AddNvSciSyncPrefence failed\n");
                    m_bRunning = false;
                    return;
                }
            } else {
                sciErr = NvSciSyncFenceWait(&fence,
                                            pStream->cpuWaitContext,
                                            SIMULATOR_MODE_CPU_WAIT_TIMEOUT_US);
                if (sciErr != NvSciError_Success) {
                    if (sciErr == NvSciError_Timeout) {
                        LOG_ERR("Producer NvSciSyncFenceWait timed out (simulator mode)\n");
                    } else {
                        LOG_ERR("Producer NvSciSyncFenceWait failed (simulator mode)\n");
                    }
                    m_bRunning = false;
                    return;
                }
            }
            NvSciSyncFenceClear(&fence);

            pBuffer->Release();
        }
    }

    ProducerStream m_streams[MAX_SENSORS][MAX_OUTPUTS_PER_SENSOR];
    std::vector<ProducerStream*> m_usedStreams;
    std::atomic<bool> m_bRunning; // Flag indicating if producer threads are running
};

#endif //CNVSIPLMASTERNVSCI_HPP
