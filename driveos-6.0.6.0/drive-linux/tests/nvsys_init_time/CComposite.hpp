/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "CCompositeHelper.hpp"
#include "NvSIPLClient.hpp"

#ifndef CCOMPOSITE_HPP
#define CCOMPOSITE_HPP

// Base class for input queue data items
class Item
{
public:
    virtual NvSciBufObj GetBuffer() = 0;
    virtual SIPLStatus GetFence(NvSciSyncFence *fence) = 0;
    virtual SIPLStatus AddFence(const NvSciSyncFence &fence) = 0;
};

// Derived class for input queue data items for default (non-NvSci) case
class BufferItem final : public Item
{
public:
    BufferItem() = default;

    ~BufferItem() = default;

    BufferItem(INvSIPLClient::INvSIPLNvMBuffer *buffer,
               NvSciSyncCpuWaitContext *cpuWaitContext) {
        if (buffer != nullptr) {
            buffer->AddRef();
            m_bufferDeleter = INvSIPLNvMBufferWaitAndRelease(cpuWaitContext);
            m_spBuffer.reset(buffer, m_bufferDeleter);
            m_bIsSimulatorMode = (cpuWaitContext != nullptr);
        }
    }

    NvSciBufObj GetBuffer() override {
        NvSciBufObj sciBufObj = nullptr;
        if (m_spBuffer != nullptr) {
            sciBufObj = m_spBuffer->GetNvSciBufImage();
        }
        return sciBufObj;
    }

    SIPLStatus GetFence(NvSciSyncFence *fence) override {
        if (m_spBuffer == nullptr) {
            return NVSIPL_STATUS_NOT_INITIALIZED;
        }
        if (fence == nullptr) {
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }
        SIPLStatus status = NVSIPL_STATUS_OK;
        if (m_bIsSimulatorMode) {
            *fence = NvSciSyncFenceInitializer;
        } else {
            status = m_spBuffer->GetEOFNvSciSyncFence(fence);
        }
        return status;
    }

    SIPLStatus AddFence(const NvSciSyncFence &fence) override {
        if (m_spBuffer == nullptr) {
            return NVSIPL_STATUS_NOT_INITIALIZED;
        }
        SIPLStatus status = NVSIPL_STATUS_OK;
        if (m_bIsSimulatorMode) {
            NvSciError sciErr = NvSciSyncFenceDup(&fence, &(m_bufferDeleter.m_fence));
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "AddFence NvSciSyncFenceDup");
        } else {
            status = m_spBuffer->AddNvSciSyncPrefence(fence);
        }
        return status;
    }

private:
    struct INvSIPLNvMBufferWaitAndRelease {
        INvSIPLNvMBufferWaitAndRelease() = default;

        INvSIPLNvMBufferWaitAndRelease(NvSciSyncCpuWaitContext *cpuWaitContext) {
            m_cpuWaitContext = cpuWaitContext;
        }

        void operator()(INvSIPLClient::INvSIPLNvMBuffer *pBuffer) {
            if (pBuffer != nullptr) {
                if (m_cpuWaitContext != nullptr) {
                    NvSciError sciErr = NvSciSyncFenceWait(&m_fence,
                                                           *m_cpuWaitContext,
                                                           NvSciSyncFenceMaxTimeout);
                    if (sciErr != NvSciError_Success) {
                        LOG_ERR("INvSIPLNvMBufferWaitAndRelease NvSciSyncFenceWait failed\n");
                    }
                    NvSciSyncFenceClear(&m_fence);
                }
                pBuffer->Release();
            }
        }

        NvSciSyncFence m_fence {NvSciSyncFenceInitializer};
        NvSciSyncCpuWaitContext *m_cpuWaitContext {nullptr};
    };

    bool m_bIsSimulatorMode {false};
    INvSIPLNvMBufferWaitAndRelease m_bufferDeleter {nullptr};
    std::shared_ptr<INvSIPLClient::INvSIPLNvMBuffer> m_spBuffer {nullptr};
};

// Default composition/display manager
class CComposite
{
protected:
    typedef enum {
        QueueType_Mailbox,
        QueueType_Fifo
    } QueueType;

public:
    CComposite()
    {
        for (uint32_t i = 0U; i < NUM_OF_GROUPS; i++) {
            m_iGroupInfos[i] = static_cast<IGroupInfo *>(&(m_groupInfos[i]));
        }
    }

    virtual ~CComposite()
    {
        if (m_bRunning) {
            Stop();
        }
        if (!m_bDeinitialized) {
            Deinit();
        }
    }

    virtual SIPLStatus CreateHelpers(uint32_t uNumDisplays);

    // Initializes compositor
    SIPLStatus Init(uint32_t uNumDisplays,
                    NvSiplRect *pRect,
                    NvSciBufModule bufModule,
                    NvSciSyncModule syncModule);

    // Registers a source and returns the ID assigned to it
    virtual SIPLStatus RegisterSource(uint32_t groupIndex,
                                      uint32_t modIndex,
                                      uint32_t outIndex,
                                      bool isRaw,
                                      uint32_t &id,
                                      bool isSimulatorMode,
                                      NvSciStreamBlock *consumer,
                                      NvSciStreamBlock **upstream,
                                      NvSciStreamBlock *queue,
                                      QueueType queueType=QueueType_Fifo);

    virtual SIPLStatus Start();

    virtual SIPLStatus Stop();

    SIPLStatus Deinit();

    SIPLStatus FillNvSciSyncAttrList(uint32_t id,
                                     NvSciSyncAttrList &attrList,
                                     NvMediaNvSciSyncClientType clientType);

    SIPLStatus RegisterNvSciSyncObj(uint32_t id,
                                    NvMediaNvSciSyncObjType syncObjType,
                                    std::unique_ptr<NvSciSyncObj, CloseNvSciSyncObj> syncObj);

    SIPLStatus UnregisterNvSciSyncObjs();

    SIPLStatus FillNvSciBufAttrList(NvSciBufAttrList &attrList);

    SIPLStatus RegisterNvSciBufObj(uint32_t id, NvSciBufObj &bufObj);

    SIPLStatus UnregisterNvSciBufObj(uint32_t id, NvSciBufObj &bufObj);

    SIPLStatus Post(uint32_t id, INvSIPLClient::INvSIPLNvMBuffer *pBuffer);

    SIPLStatus SetActiveGroup(uint32_t dispId, uint32_t groupIndex);

    void PrintDisplayableGroups() const;

protected:
    SIPLStatus InitDisplay(uint32_t uNumDisplays);

    NvSciBufModule m_sciBufModule {};
    NvSciSyncModule m_sciSyncModule {};
    NvSciSyncCpuWaitContext m_cpuWaitContext {nullptr};
    std::unique_ptr<IDisplayManager> m_dispMgr;
    std::atomic<bool> m_bRunning; // Flag indicating if compositor is running
    bool m_bDeinitialized {false}; // Flag indicating if compositor has been deinitialized
    std::unique_ptr<ICompositeHelper> m_helpers[MAX_SUPPORTED_DISPLAYS];
    IGroupInfo *m_iGroupInfos[NUM_OF_GROUPS] = {nullptr};

private:
    CGroupInfo<BufferItem> m_groupInfos[NUM_OF_GROUPS]; // Groups (of sources to composite)
};

#endif // CCOMPOSITE_HPP
