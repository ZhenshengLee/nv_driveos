/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#define RECOVERY_INTERVAL_MS (1000U)

/** CAutoRecovery class */
class CAutoRecovery
{
public:
    CAutoRecovery(CNvSIPLMaster *pMaster = nullptr) :m_pMaster(pMaster)
    {
        for (auto i = 0u; i < MAX_SENSORS_PER_PLATFORM; i++) {
            m_elinkRecoveryAction[i] = RecoveryAction_NULL;
            m_linkRecoverTrial[i] = 0;
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

    void TriggerRecover(uint32_t linkIndex) {
        if (linkIndex >= MAX_SENSORS_PER_PLATFORM) {
            LOG_ERR("OnLinkFailure, invalid linkIndex: %u\n", linkIndex);
            return;
        }
        std::unique_lock<std::mutex> lck(m_mutex);
        if (m_elinkRecoveryAction[linkIndex] == RecoveryAction_NULL){
            LOG_WARN("OnLinkFailure, link: %u\n", linkIndex);
            m_elinkRecoveryAction[linkIndex] = RecoveryAction_Disable;
            m_condition.notify_all();
        } else {
            LOG_WARN("OnLinkFailure, link %u is already in error state.\n", linkIndex);
        }
    }


    void OnLinkFailure(uint32_t linkIndex)
    {
        if (linkIndex >= MAX_SENSORS_PER_PLATFORM) {
            LOG_ERR("OnLinkFailure, invalid linkIndex: %u\n", linkIndex);
            return;
        }
        std::unique_lock<std::mutex> lck(m_mutex);
        if(m_elinkRecoveryAction[linkIndex] == RecoveryAction_NULL){
            LOG_WARN("OnLinkFailure, link: %u\n", linkIndex);
            m_elinkRecoveryAction[linkIndex] = RecoveryAction_Disable;
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
        SIPLStatus status = NVSIPL_STATUS_OK;
        pthread_setname_np(pthread_self(), "CAutoRecovery");
        m_bRunning = true;
        for (auto i = 0u; i < MAX_SENSORS_PER_PLATFORM; i++) {
            m_linkDisabled[i] = false;
        }
        while (m_bRunning) {
            std::unique_lock<std::mutex> lck(m_mutex);
            if (!NeedRecovery()) {
                 m_condition.wait(lck, [this]{return NeedRecovery() || !m_bRunning;});
            }

            //Try to recover each error link
            lck.unlock();
            for (auto i = 0u; i < MAX_SENSORS_PER_PLATFORM; i++) {
                switch(m_elinkRecoveryAction[i]) {
                    case RecoveryAction_Disable:
                        if (!m_linkDisabled[i]) {
                            status = m_pMaster->DisableLink(i);
                            if (status == NVSIPL_STATUS_OK) {
                                LOG_MSG("DisableLink: %u successful\n", i);
                                lck.lock();
                                m_elinkRecoveryAction[i] = RecoveryAction_Check;
                                m_linkDisabled[i] = true;
                                lck.unlock();
                            }
                        }
                        break;
                    case RecoveryAction_Check:
                        status = m_pMaster->CheckSensorStatus(i);
                        if (status == NVSIPL_STATUS_OK) {
                            LOG_MSG("Detected the module successfully\n");
                            lck.lock();
                            m_elinkRecoveryAction[i] = RecoveryAction_Enable;
                            lck.unlock();
                        } else if (status == NVSIPL_STATUS_NOT_SUPPORTED) {
                            LOG_MSG("Module detection not supported\n");
                            lck.lock();
                            m_elinkRecoveryAction[i] = RecoveryAction_Enable;
                            lck.unlock();
                        } else {
                            LOG_ERR("Failed to detect the module\n");
                            lck.lock();
                            m_elinkRecoveryAction[i] = RecoveryAction_Check;
                            lck.unlock();
                        }
                        break;
                    case RecoveryAction_Enable:
                        status = m_pMaster->EnableLink(i, true);
                        if (status == NVSIPL_STATUS_OK) {
                            LOG_MSG("EnableLink: %u successful\n", i);
                            lck.lock();
                            m_elinkRecoveryAction[i] = RecoveryAction_NULL;
                            m_linkDisabled[i] = false;
                            lck.unlock();
                        } else if(status == NVSIPL_STATUS_ERROR) {
                            lck.lock();
                            LOG_MSG("EnableLink: %u failed, check again\n", i); 
                            m_elinkRecoveryAction[i] = RecoveryAction_Check;
                            lck.unlock();
                        } else if(status == NVSIPL_STATUS_INVALID_STATE) {
                            lck.lock();
                            m_elinkRecoveryAction[i] = RecoveryAction_NULL;
                            lck.unlock();
                        }
                        break;
                    default:
                        break;
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(RECOVERY_INTERVAL_MS));
        }
    }

 private:
    //This function can only be called while the caller is holding the mutex.
    bool NeedRecovery()
    {
        for (auto i = 0u; i < MAX_SENSORS_PER_PLATFORM; i++) {
            if (m_elinkRecoveryAction[i] != RecoveryAction_NULL &&
                m_elinkRecoveryAction[i] != RecoveryAction_Invalid) {
                return true;
            }
        }
        return false;
    }
    enum RecoveryAction {
        RecoveryAction_NULL,
        RecoveryAction_Disable,
        RecoveryAction_Check,
        RecoveryAction_Enable,
        RecoveryAction_Invalid
    };

    bool m_linkDisabled[MAX_SENSORS_PER_PLATFORM];
    RecoveryAction m_elinkRecoveryAction[MAX_SENSORS_PER_PLATFORM];
    uint64_t m_uLastTs[MAX_SENSORS_PER_PLATFORM];
    int m_linkRecoverTrial[MAX_SENSORS_PER_PLATFORM];
    std::mutex m_mutex;
    std::condition_variable m_condition;
    std::unique_ptr<std::thread> m_upThread {nullptr};
    std::atomic<bool> m_bRunning; // Flag indicating if link recovery is running.
    CNvSIPLMaster *m_pMaster = nullptr;
};

#endif //CAUTORECOVERY_HPP
