/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "CCompositeNvSci.hpp"

#include "nvscisync.h"
#include "nvscistream.h"
#include "nvmedia_2d_sci.h"

#define STREAMING_TIMEOUT (100000U)

using namespace std;

SIPLStatus CCompositeNvSci::CreateHelpers(uint32_t uNumDisplays) {
    for (uint32_t i = 0U; i < uNumDisplays; i++) {
        m_helpers[i].reset(new CCompositeHelper<IndexItem>(m_groupInfos));
        CHK_PTR_AND_RETURN(m_helpers[i], "CCompositeHelper creation");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCompositeNvSci::RegisterSource(uint32_t groupIndex,   // Group
                                           uint32_t modIndex,     // Row
                                           uint32_t outIndex,     // Column
                                           bool isRaw,
                                           uint32_t &id,
                                           bool isSimulatorMode,
                                           NvSciStreamBlock *consumer,
                                           NvSciStreamBlock **upstream,
                                           NvSciStreamBlock *queue,
                                           QueueType type)
{
    NvSciError sciErr;

    if ((groupIndex >= NUM_OF_GROUPS) or (modIndex >= NUM_OF_ROWS) or (outIndex >= NUM_OF_COLS)) {
        LOG_ERR("Compositor: RegisterSource: Invalid argument\n");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    if ((consumer == nullptr) || (upstream == nullptr)) {
        LOG_ERR("Compositor: RegisterSource: Received unexpected nullptr\n");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    // Compute global ID for the source
    id = groupIndex * (NUM_OF_COLS * NUM_OF_ROWS) + modIndex * NUM_OF_COLS + outIndex;

    SIPLStatus status = m_iGroupInfos[groupIndex]->AddInput(id, isSimulatorMode);
    if (status != NVSIPL_STATUS_OK) {
        return status;
    }
    CInputInfo<IndexItem> *pInput = m_groupInfos[groupIndex].GetInput(id);
    if (pInput == nullptr) {
        return NVSIPL_STATUS_ERROR;
    }

    std::unique_ptr<ConsumerStream> upStream(new ConsumerStream());
    CHK_PTR_AND_RETURN(upStream.get(), "ConsumerStream creation");

    switch (type) {
        case QueueType_Mailbox:
            sciErr = NvSciStreamMailboxQueueCreate(&upStream->queue);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamMailboxQueueCreate");
            break;
        case QueueType_Fifo:
        default:
            sciErr = NvSciStreamFifoQueueCreate(&upStream->queue);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamFifoQueueCreate");
            break;
    }

    sciErr = NvSciStreamConsumerCreate(upStream->queue, &upStream->consumer);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamConsumerCreate");

    *consumer = upStream->consumer;
    *upstream = &upStream->upstream;
    *queue = upStream->queue;

    upStream->id = id;
    upStream->row = modIndex;
    upStream->group = groupIndex;
    m_usedStreams.push_back(std::move(upStream));

    LOG_INFO("Compositor: Registered output:%u from link:%u of quad:%u as id:%u\n", outIndex, modIndex, groupIndex, id);
    return NVSIPL_STATUS_OK;
}

SIPLStatus CCompositeNvSci::Start()
{
    // Start thread
    m_pThread.reset(new (std::nothrow) std::thread(&CCompositeNvSci::ThreadFunc, this));
    CHK_PTR_AND_RETURN(m_pThread.get(), "Compositor thread creation");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCompositeNvSci::Stop()
{
    // Signal thread to stop
    m_bRunning = false;

    // Wait for the thread
    if ((m_pThread != nullptr) && (m_pThread->joinable())) {
        LOG_INFO("Waiting to join compositor thread ID:%u\n", m_pThread->get_id());
        m_pThread->join();
    }

    return NVSIPL_STATUS_OK;
}

void CCompositeNvSci::ThreadFunc()
{
    pthread_setname_np(pthread_self(), "CCompositeNvSci");

    SIPLStatus status = SetupStreaming();
    if (status != NVSIPL_STATUS_OK) {
       LOG_ERR("Compositor: SetupStreaming failed\n");
       return;
    }

    // Start helpers
    for (uint32_t i = 0U; i < MAX_SUPPORTED_DISPLAYS; i++) {
        if (m_helpers[i] != NULL) {
            SIPLStatus status = m_helpers[i]->Start();
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("Compositor: Helper start failed\n");
                return;
            }
        }
    }
    m_bRunning = true;

    while (m_bRunning) {
        status = CollectEvents();
        if (status != NVSIPL_STATUS_OK) {
           LOG_ERR("Compositor: CollectEvents failed\n");
           return;
        }
    }

    // Stop helpers
    for (uint32_t i = 0U; i < MAX_SUPPORTED_DISPLAYS; i++) {
        if (m_helpers[i] != NULL) {
            SIPLStatus status = m_helpers[i]->Stop();
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("Compositor: Helper stop failed\n");
                return;
            }
        }
    }

    return;
}

SIPLStatus CCompositeNvSci::CollectEvents(void)
{
    NvSciStreamEventType eventType;
    NvSciStreamCookie cookie = 0U;
    NvSciSyncFence fence = NvSciSyncFenceInitializer;

    for (std::unique_ptr<ConsumerStream> &pStream : m_usedStreams) {
        auto sciErr = NvSciStreamBlockEventQuery(pStream->consumer, 0, &eventType);
        if(sciErr == NvSciError_Timeout) {
            continue;   // nothing ready on this stream
        }
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer PacketReady NvSciStreamBlockEventQuery");
        if(eventType != NvSciStreamEventType_PacketReady) {
             LOG_ERR("Compositor: did not receive expected PacketReady event\n");
             return NVSIPL_STATUS_ERROR;
        }

        sciErr = NvSciStreamConsumerPacketAcquire(pStream->consumer, &cookie);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamConsumerPacketAcquire");
        // Index is one less than cookie, it was incremented on initialization since zero is invalid
        uint32_t index = cookie - 1U;

        sciErr = NvSciStreamBlockPacketFenceGet(pStream->consumer,
                                                pStream->packets[index],
                                                0U, 0U,
                                                &fence);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockPacketFenceGet");

        CInputInfo<IndexItem> *pInput = m_groupInfos[pStream->group].GetInput(pStream->id);
        if (pInput == nullptr) {
            return NVSIPL_STATUS_ERROR;
        }
        IndexItem item = IndexItem(pStream.get(), index);

        if (m_iGroupInfos[pStream->group]->m_bGroupInUse) {
            sciErr = NvSciSyncFenceDup(&fence, &pStream->prefences[index]);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceDup");
            pInput->QueueAdd(item);
        } else {
            pStream->postfences[index] = NvSciSyncFenceInitializer;
        }
        NvSciSyncFenceClear(&fence);
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCompositeNvSci::SetupStreaming(void)
{
    NvSciBufAttrList bufAttrList;
    NvSciSyncAttrList signalerAttrList, waiterAttrList;
    NvSciStreamEventType eventType;

    // Create SciBuf attributes.
    auto sciErr = NvSciBufAttrListCreate(m_sciBufModule, &bufAttrList);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciBufAttrListCreate");

    NvSciBufType bufType = NvSciBufType_Image;
    bool cpuAccess = true;
    NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_Readonly;
    NvSciBufAttrKeyValuePair attrKvp[] = {
                                            { NvSciBufGeneralAttrKey_Types,
                                              &bufType,
                                              sizeof(bufType)
                                            },
                                            { NvSciBufGeneralAttrKey_NeedCpuAccess,
                                              &cpuAccess,
                                              sizeof(cpuAccess)
                                            },
                                            { NvSciBufGeneralAttrKey_RequiredPerm,
                                              &accessPerm,
                                              sizeof(accessPerm)
                                            }
                                         };

    sciErr = NvSciBufAttrListSetAttrs(bufAttrList, attrKvp, 3);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciBufAttrListSetAttrs");

    // Get waiter NvSciSync attributes from 2D
    sciErr = NvSciSyncAttrListCreate(m_sciSyncModule, &waiterAttrList);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer waiter NvSciSyncAttrListCreate");

    auto nvmStatus = NvMedia2DFillNvSciSyncAttrList(m_helpers[0]->m_p2dDevices[0].get(), waiterAttrList, NVMEDIA_WAITER);
    CHK_NVMSTATUS_AND_RETURN(nvmStatus, "Consumer waiter NvMedia2DFillNvSciSyncAttrList");

    // Send SciBuf attributes to pool
    for(std::unique_ptr<ConsumerStream> &pStream : m_usedStreams) {
       sciErr = NvSciStreamBlockElementAttrSet(pStream->consumer, 0,
                                               bufAttrList);
       CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciStreamBlockElementAttrSet");

       sciErr = NvSciStreamBlockSetupStatusSet(pStream->consumer,
                                               NvSciStreamSetup_ElementExport,
                                               true);
       CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciStreamBlockSetupStatusSet(ElementExport)");
    }
    NvSciBufAttrListFree(bufAttrList);

    // Get element notification from pool (no info queried because it isn't needed)
    for(std::unique_ptr<ConsumerStream> &pStream : m_usedStreams) {
        auto sciErr = NvSciStreamBlockEventQuery(pStream->consumer, STREAMING_TIMEOUT, &eventType);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer Elements NvSciStreamBlockEventQuery");
        if(eventType != NvSciStreamEventType_Elements) {
            LOG_ERR("Compositor: did not receive Elements event as expected\n");
            return NVSIPL_STATUS_ERROR;
        }
        sciErr = NvSciStreamBlockSetupStatusSet(pStream->consumer,
                                                NvSciStreamSetup_ElementImport,
                                                true);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool NvSciStreamBlockSetupStatusSet(ElementImport)");
    }

    // Get SciBufObjs from pool, acknowledge.
    // Create Images from SciBufObjs.
    for (std::unique_ptr<ConsumerStream> &pStream : m_usedStreams) {
        for (uint32_t i = 0U; i < NUM_PACKETS; i++) {
            NvSciError sciErr = NvSciStreamBlockEventQuery(pStream->consumer,
                                                           STREAMING_TIMEOUT,
                                                           &eventType);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer PacketCreate NvSciStreamBlockEventQuery");
            if(eventType != NvSciStreamEventType_PacketCreate) {
                LOG_ERR("Compositor: did not receive PacketCreate event as expected\n");
                return NVSIPL_STATUS_ERROR;
            }

            sciErr = NvSciStreamBlockPacketNewHandleGet(pStream->consumer,
                                                        &pStream->packets[i]);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciStreamBlockPacketNewHandleGet");

            NvSciBufObj recvBufObj;
            sciErr = NvSciStreamBlockPacketBufferGet(pStream->consumer,
                                                     pStream->packets[i],
                                                     0U,
                                                     &recvBufObj);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciStreamBlockPacketBufferGet");

            // Duplicate received object to this class's copy, then free the received object
            // Duplication is necessary because otherwise the producer and consumer side both end up
            // using the same NvSciBufObj
            sciErr = NvSciBufObjDup(recvBufObj, &pStream->sciBufObjs[i]);
            NvSciBufObjFree(recvBufObj);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciBufObjDup");

            NvSciStreamCookie cookie = i + 1U;
            sciErr = NvSciStreamBlockPacketStatusSet(pStream->consumer,
                                                     pStream->packets[i],
                                                     cookie,
                                                     NvSciError_Success);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciStreamBlockPacketStatusSet");
        }
    }

    for (std::unique_ptr<ConsumerStream> &pStream : m_usedStreams) {
        for (uint32_t i = 0U; i < NUM_PACKETS; i++) {
            SIPLStatus status = RegisterNvSciBufObj(pStream->id, pStream->sciBufObjs[i]);
            CHK_STATUS_AND_RETURN(status, "RegisterNvSciBufObj");
        }
    }

    // Consumer completes packet setup
    for (std::unique_ptr<ConsumerStream> &pStream : m_usedStreams) {
        auto sciErr = NvSciStreamBlockEventQuery(pStream->consumer, STREAMING_TIMEOUT, &eventType);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer PacketsComplete NvSciStreamBlockEventQuery");
        if(eventType != NvSciStreamEventType_PacketsComplete) {
            LOG_ERR("Compositor: did not receive PacketsComplete event as expected\n");
            return NVSIPL_STATUS_ERROR;
        }
        sciErr = NvSciStreamBlockSetupStatusSet(pStream->consumer,
                                                NvSciStreamSetup_PacketImport,
                                                true);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciStreamBlockSetupStatusSet(PacketImport)");
    }

    // Send the waiter attributes to the producer
    for(std::unique_ptr<ConsumerStream> &pStream : m_usedStreams) {
        sciErr = NvSciStreamBlockElementWaiterAttrSet(pStream->consumer, 0U, waiterAttrList);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciStreamBlockElementWaiterAttrSet");
        sciErr = NvSciStreamBlockSetupStatusSet(pStream->consumer,
                                                NvSciStreamSetup_WaiterAttrExport,
                                                true);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciStreamBlockSetupStatusSet(WaiterAttrExport)");
    }

    // Get SciSync attributes from producer, reconcile with our own signaler attributes,
    // create consumer sync objs with the reconciled attributes, register objs with 2D and
    // send to producer.
    for(std::unique_ptr<ConsumerStream> &pStream : m_usedStreams) {
       NvSciSyncAttrList unreconciledList[2];
       NvSciSyncAttrList reconciledList = NULL;
       NvSciSyncAttrList newConflictList = NULL;

       auto sciErr = NvSciStreamBlockEventQuery(pStream->consumer, STREAMING_TIMEOUT, &eventType);
       CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer WaiterAttr NvSciStreamBlockEventQuery");
       if(eventType != NvSciStreamEventType_WaiterAttr) {
           LOG_ERR("Compositor: did not receive expected event type\n");
           return NVSIPL_STATUS_ERROR;
       }
       sciErr = NvSciStreamBlockElementWaiterAttrGet(pStream->consumer,
                                                     0U,
                                                     &unreconciledList[1]);
       CHK_NVSCISTATUS_AND_RETURN(sciErr, "Producer NvSciStreamBlockElementAttrGet");

       sciErr = NvSciStreamBlockSetupStatusSet(pStream->consumer,
                                               NvSciStreamSetup_WaiterAttrImport,
                                               true);
       CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciStreamBlockSetupStatusSet(WaiterAttrImport)");

       // Get signaler NvSciSync attributes from 2D
       sciErr = NvSciSyncAttrListCreate(m_sciSyncModule, &signalerAttrList);
       CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer signaler NvSciSyncAttrListCreate");
       nvmStatus = NvMedia2DFillNvSciSyncAttrList(m_helpers[0]->m_p2dDevices[pStream->row].get(),
                                                  signalerAttrList,
                                                  NVMEDIA_SIGNALER);
       CHK_NVMSTATUS_AND_RETURN(nvmStatus, "Consumer signaler NvMedia2DFillNvSciSyncAttrList");
       unreconciledList[0] = signalerAttrList;

       sciErr = NvSciSyncAttrListReconcile(unreconciledList, 2, &reconciledList, &newConflictList);
       CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciSyncAttrListReconcile");

       std::unique_ptr<NvSciSyncObj, CloseNvSciSyncObj> eofSyncObj(new NvSciSyncObj());
       sciErr = NvSciSyncObjAlloc(reconciledList, eofSyncObj.get());
       CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciSyncObjAlloc");

       NvSciSyncAttrListFree(reconciledList);
       NvSciSyncAttrListFree(signalerAttrList);
       NvSciSyncAttrListFree(unreconciledList[1]);
       if(newConflictList != nullptr) {
          NvSciSyncAttrListFree(newConflictList);
       }

       sciErr = NvSciStreamBlockElementSignalObjSet(pStream->consumer, 0, *eofSyncObj);
       CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciStreamBlockElementSignalObjSet");

       SIPLStatus status = RegisterNvSciSyncObj(pStream->id, NVMEDIA_EOFSYNCOBJ, std::move(eofSyncObj));
       CHK_STATUS_AND_RETURN(status, "Consumer RegisterNvSciSyncObj");

       sciErr = NvSciStreamBlockSetupStatusSet(pStream->consumer,
                                               NvSciStreamSetup_SignalObjExport,
                                               true);
       CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciStreamBlockSetupStatusSet(SignalObjExport)");
    }

    NvSciSyncAttrListFree(waiterAttrList);

    // Get sync objs from producer and register with 2D.
    for(std::unique_ptr<ConsumerStream> &pStream : m_usedStreams) {
       auto sciErr = NvSciStreamBlockEventQuery(pStream->consumer, STREAMING_TIMEOUT, &eventType);
       CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer SignalObj NvSciStreamBlockEventQuery");
       if(eventType != NvSciStreamEventType_SignalObj) {
           LOG_ERR("Compositor: did not receive expected SignalObj event type\n");
           return NVSIPL_STATUS_ERROR;
       }

       std::unique_ptr<NvSciSyncObj, CloseNvSciSyncObj> preSyncObj(new NvSciSyncObj());
       sciErr = NvSciStreamBlockElementSignalObjGet(pStream->consumer, 0U, 0U, preSyncObj.get());
       CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciStreamBlockElementSignalObjGet");

       SIPLStatus status = RegisterNvSciSyncObj(pStream->id, NVMEDIA_PRESYNCOBJ, std::move(preSyncObj));
       CHK_STATUS_AND_RETURN(status, "Consumer RegisterNvSciSyncObj");

       sciErr = NvSciStreamBlockSetupStatusSet(pStream->consumer,
                                               NvSciStreamSetup_SignalObjImport,
                                               true);
       CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer NvSciStreamBlockSetupStatusSet(SignalObjImport)");

    }

    for(std::unique_ptr<ConsumerStream> &pStream : m_usedStreams) {
        auto sciErr = NvSciStreamBlockEventQuery(pStream->consumer, STREAMING_TIMEOUT, &eventType);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "Consumer SetupComplete NvSciStreamBlockEventQuery");
        if (eventType != NvSciStreamEventType_SetupComplete) {
            LOG_ERR("Didn't receive expected SetupComplete event\n");
            return NVSIPL_STATUS_ERROR;
        }
    }

    return NVSIPL_STATUS_OK;
}
