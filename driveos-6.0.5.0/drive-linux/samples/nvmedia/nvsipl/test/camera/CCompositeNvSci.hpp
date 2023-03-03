/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "CComposite.hpp"

#ifndef CCOMPOSITENVSCI_HPP
#define CCOMPOSITENVSCI_HPP

#define NUM_PACKETS (6U)

using namespace nvsipl;

// Information related to incoming (input) stream
class ConsumerStream
{
public:
   uint32_t id;
   uint32_t row;
   uint32_t group;
   NvSciStreamBlock queue;
   NvSciStreamBlock consumer;
   NvSciStreamBlock upstream;
   NvSciBufObj sciBufObjs[NUM_PACKETS];
   NvSciSyncFence prefences[NUM_PACKETS];
   NvSciSyncFence postfences[NUM_PACKETS];
   NvSciStreamPacket packets[NUM_PACKETS];

   ConsumerStream() {
        // Initialize fence structures
        for (uint32_t i = 0U; i < NUM_PACKETS; i++) {
            prefences[i] = NvSciSyncFenceInitializer;
            postfences[i] = NvSciSyncFenceInitializer;
        }
    }

    ~ConsumerStream() {
        for (uint32_t i = 0U; i < NUM_PACKETS; i++) {
            if (sciBufObjs[i] != nullptr) {
                NvSciBufObjFree(sciBufObjs[i]);
            }
        }
    }

    SIPLStatus ReleasePacket(uint32_t index) {
        NvSciError sciErr = NvSciStreamBlockPacketFenceSet(consumer,
                                                           packets[index],
                                                           0U,
                                                           &postfences[index]);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockPacketFenceSet");

        sciErr = NvSciStreamConsumerPacketRelease(consumer, packets[index]);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamConsumerPacketRelease");

        NvSciSyncFenceClear(&postfences[index]);

        return NVSIPL_STATUS_OK;
    }
};

// Derived class for input queue data items for NvSci case
class IndexItem final : public Item
{
public:
    IndexItem() = default;

    ~IndexItem() = default;

    IndexItem(ConsumerStream *pStream, uint32_t index) {
        m_index = index;
        m_spStream.reset(pStream, [index](ConsumerStream *pStream){ pStream->ReleasePacket(index); });
    }

    uint32_t GetIndex() {
        return m_index;
    }

    ConsumerStream * GetStream() {
        return m_spStream.get();
    }

    NvSciBufObj GetBuffer() override {
        NvSciBufObj sciBufObj = nullptr;
        if ((m_spStream != nullptr) && (m_index < NUM_PACKETS)) {
            sciBufObj = m_spStream->sciBufObjs[m_index];
        }
        return sciBufObj;
    }

    SIPLStatus GetFence(NvSciSyncFence *fence) override {
        if (fence == nullptr) {
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }
        if (m_spStream == nullptr) {
            return NVSIPL_STATUS_NOT_INITIALIZED;
        }
        if (m_index >= NUM_PACKETS) {
            return NVSIPL_STATUS_INVALID_STATE;
        }
        NvSciError sciErr = NvSciSyncFenceDup(&(m_spStream->prefences[m_index]), fence);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceDup");
        NvSciSyncFenceClear(&(m_spStream->prefences[m_index]));
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus AddFence(const NvSciSyncFence &fence) override {
        if (m_spStream == nullptr) {
            return NVSIPL_STATUS_NOT_INITIALIZED;
        }
        if (m_index >= NUM_PACKETS) {
            return NVSIPL_STATUS_INVALID_STATE;
        }
        NvSciError sciErr = NvSciSyncFenceDup(&fence, &(m_spStream->postfences[m_index]));
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceDup");
        return NVSIPL_STATUS_OK;
    }

private:
    std::shared_ptr<ConsumerStream> m_spStream {nullptr};
    uint32_t m_index {UINT32_MAX};
};

// NvSci composition/display manager
class CCompositeNvSci final : public CComposite
{
public:
    CCompositeNvSci()
    {
        for (uint32_t i = 0U; i < NUM_OF_GROUPS; i++) {
            m_iGroupInfos[i] = static_cast<IGroupInfo *>(&(m_groupInfos[i]));
        }
    }

    ~CCompositeNvSci() override
    {
        if (m_bRunning) {
            Stop();
        }
        if (!m_bDeinitialized) {
            Deinit();
        }
    }

    SIPLStatus CreateHelpers(uint32_t uNumDisplays) override;
    SIPLStatus RegisterSource(uint32_t groupIndex,
                              uint32_t modIndex,
                              uint32_t outIndex,
                              bool isRaw,
                              uint32_t &id,
                              bool isSimulatorMode,
                              NvSciStreamBlock *consumer,
                              NvSciStreamBlock **upstream,
                              NvSciStreamBlock *queue,
                              QueueType queueType=QueueType_Fifo) override;
    SIPLStatus Start() override;
    SIPLStatus Stop() override;

private:
    void ThreadFunc();
    SIPLStatus SetupStreaming();
    SIPLStatus CollectEvents();

    std::unique_ptr<std::thread> m_pThread {nullptr};
    CGroupInfo<IndexItem> m_groupInfos[NUM_OF_GROUPS]; // Groups (of sources to composite)

    // Streaming
    std::vector<std::unique_ptr<ConsumerStream>> m_usedStreams;
};

#endif // CCOMPOSITENVSCI_HPP
