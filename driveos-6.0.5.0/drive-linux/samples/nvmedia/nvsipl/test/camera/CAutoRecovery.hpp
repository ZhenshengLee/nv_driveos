/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <unistd.h>

#include "NvSIPLDeviceBlockInfo.hpp"
#include "CNvSIPLMaster.hpp"

#ifndef CAUTORECOVERY_HPP
#define CAUTORECOVERY_HPP

using namespace nvsipl;

#define RECOVERY_INTERVAL_MS (100U)

/** CAutoRecovery class */
class CAutoRecovery
{
public:
    CAutoRecovery(CNvSIPLMaster *pMaster = nullptr) :m_pMaster(pMaster)
    {
        for (auto i = 0u; i < MAX_SENSORS_PER_PLATFORM; i++) {
            m_linkNeedsRecovery[i] = false;
        }
    }

    ~CAutoRecovery()
    {
    }

    SIPLStatus Start()
    {
        if (m_pMaster == nullptr) {
            LOG_ERR("Start(), m_pMaster is not assigned.\n");
            return NVSIPL_STATUS_NOT_INITIALIZED;
        }
        m_upThread.reset(new std::thread(&CAutoRecovery::ThreadFunc, this));
        if (m_upThread == nullptr) {
            LOG_ERR("Failed to create auto recovery thread\n");
            return NVSIPL_STATUS_OUT_OF_MEMORY;
        }

        LOG_INFO("Created auto recovery thread: ID:%u\n", m_upThread->get_id());
        return NVSIPL_STATUS_OK;
    }

    void OnLinkFailure(uint32_t linkIndex)
    {
        if (linkIndex >= MAX_SENSORS_PER_PLATFORM) {
            LOG_ERR("OnLinkFailure, invalid linkIndex: %u\n", linkIndex);
            return;
        }
        std::unique_lock<std::mutex> lck(m_mutex);
        if (!m_linkNeedsRecovery[linkIndex]) {
            LOG_WARN("OnLinkFailure, link: %u\n", linkIndex);
            m_linkNeedsRecovery[linkIndex] = true;
            m_condition.notify_all();
        } else {
            LOG_WARN("OnLinkFailure, link %u is already in error state.\n", linkIndex);
        }
    }

    void Stop()
    {
        // Signal thread to stop
        m_bRunning = false;
        std::unique_lock<std::mutex> lck(m_mutex);

        m_condition.notify_all();
        lck.unlock();

        // Wait for the thread
        if (m_upThread != nullptr) {
            LOG_INFO("Waiting to join auto recovery thread: ID: %u\n", m_upThread->get_id());
            m_upThread->join();
        }
    }
protected:
    void ThreadFunc(void)
    {
        uint32_t errorLinks[MAX_SENSORS_PER_PLATFORM];
        uint32_t errorLinkCount = 0;
        bool anyRecoveryFailure = false;

        pthread_setname_np(pthread_self(), "CAutoRecovery");
        m_bRunning = true;
        while (m_bRunning) {
            std::unique_lock<std::mutex> lck(m_mutex);
            if (!ErrlinkExists()) {
                 m_condition.wait(lck, [this]{return ErrlinkExists() || !m_bRunning;});
            }

            //Copy error links to a temp array
            errorLinkCount = 0;
            for (auto i = 0u; i < MAX_SENSORS_PER_PLATFORM; i++) {
                if (m_linkNeedsRecovery[i]) {
                    errorLinks[errorLinkCount++] = i;
                    m_linkNeedsRecovery[i] = false;
                }
            }

            //Try to recover each error link
            lck.unlock();
            anyRecoveryFailure = false;
            for (auto i = 0u; i < errorLinkCount; i++) {
                auto status = m_pMaster->DisableLink(errorLinks[i]);
                if (status == NVSIPL_STATUS_OK) {
                    LOG_MSG("DisableLink: %u successful\n", errorLinks[i]);

                    status = m_pMaster->EnableLink(errorLinks[i], true);
                    if (status == NVSIPL_STATUS_OK) {
                        LOG_MSG("EnableLink: %u successful\n", errorLinks[i]);
                    }
                }

                if (status != NVSIPL_STATUS_OK) {
                    lck.lock();
                    m_linkNeedsRecovery[errorLinks[i]] = true;
                    lck.unlock();
                    anyRecoveryFailure = true;
                    LOG_ERR("Attempt to recover link: %u failed! with error: %x\n", errorLinks[i], status);
                }
            }
            if (anyRecoveryFailure) {
                std::this_thread::sleep_for(std::chrono::milliseconds(RECOVERY_INTERVAL_MS));
            }
        }
    }

 private:
    //This function can only be called while the caller is holding the mutex.
    bool ErrlinkExists()
    {
        for (auto i = 0u; i < MAX_SENSORS_PER_PLATFORM; i++) {
            if (m_linkNeedsRecovery[i]) {
                return true;
            }
        }
        return false;
    }

    bool m_linkNeedsRecovery[MAX_SENSORS_PER_PLATFORM];
    std::mutex m_mutex;
    std::condition_variable m_condition;
    std::unique_ptr<std::thread> m_upThread {nullptr};
    std::atomic<bool> m_bRunning; // Flag indicating if link recovery is running.
    CNvSIPLMaster *m_pMaster = nullptr;
};

#endif //CAUTORECOVERY_HPP
