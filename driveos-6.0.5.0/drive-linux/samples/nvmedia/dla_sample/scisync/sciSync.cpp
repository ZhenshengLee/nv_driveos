/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <chrono>

#include "sciSync.h"
#include "utils.h"

Semaphore::Semaphore(uint32_t uInitCount, uint32_t uMaxCount)
{
    if (uInitCount > uMaxCount) {
        uInitCount = uMaxCount;
    }

    m_maxCount = uMaxCount;
    m_count = uInitCount;
}

Semaphore::~Semaphore()
{
}

void Semaphore::Increment()
{
    std::unique_lock<std::mutex> lck(m_mutex);
    m_count++;
    if (m_count > m_maxCount) {
        m_count = m_maxCount;
    } else {
        m_condition.notify_all();
    }
}

bool Semaphore::Decrement(int32_t timeoutMs)
{
    while (true) {
        std::unique_lock <std::mutex> lck(m_mutex);
        if (m_count > 0u) {
            m_count--;
            return true;
        }

        if (timeoutMs == 0) {
            return false; // Could not decrement as the count is 0.
        }

        if (timeoutMs == -1) {
            m_condition.wait(lck);
        } else {
            m_condition.wait_for(lck, std::chrono::milliseconds(timeoutMs));
            timeoutMs = 0;
        }
    }
    return true;
}

Worker::Worker(
        std::string name) :
    m_name(name),
    m_thread(nullptr),
    m_threadStartFlag(0, 1),
    m_bQuit(false)
{
}

NvMediaStatus Worker::Start()
{
    // Create new thread
    m_thread.reset(new (std::nothrow) std::thread(&Worker::m_FuncStatic, this));
    if (m_thread == nullptr) {
        LOG_ERR("Failed to create thread\n");
        return NVMEDIA_STATUS_ERROR;
    }

    // Wait for thread to get created
    LOG_INFO("Waiting for thread to be created\n");
    m_threadStartFlag.Decrement(-1);

    LOG_INFO("ThreadName = %s: Thread started\n", m_name.c_str());

    return NVMEDIA_STATUS_OK;
}

void Worker::Stop()
{
    m_bQuit = true;

    // Wait for thread to end
    if (m_thread && m_thread->joinable()) {
        LOG_INFO("Waiting for thread %s to end\n", m_name.c_str());
        m_thread->join();
    }

    LOG_INFO("Thread %s ended\n", m_name.c_str());
}

void* Worker::m_FuncStatic(void* vpParam)
{
    LOG_INFO("Thread created.\n");
    Worker* pThis = reinterpret_cast <Worker*>(vpParam);
    return pThis->m_Func();
}

void* Worker::m_Func()
{
    LOG_INFO("ThreadName = %s: Notifying the creator.\n", m_name.c_str());

    m_threadStartFlag.Increment();

    LOG_INFO("ThreadName = %s: Thread running\n", m_name.c_str());
    // Use do-while to ensure that the ThreadFunc is called atleast once.

    bool bDoMore = false;
    do {
        bDoMore = DoWork();
        if (m_bQuit) {
            break;
        }
    } while (bDoMore);

    LOG_INFO("ThreadName = %s: Thread exiting\n", m_name.c_str());
    return nullptr;
}

CPUWaiter::CPUWaiter(
        std::string name,
        NvSciSyncCpuWaitContext cpuWaitContext,
        ShareBuf* shareBuf) :
    Worker(name),
    m_cpuWaitContext(cpuWaitContext),
    m_shareBuf(shareBuf)
{
}

NvMediaStatus CPUWaiter::FenceWait()
{
    NvSciError err;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    for (auto i = 0; i < m_shareBuf->BufferSize; ++i) {
        /* Generate NvSciSyncFenceWait */
        LOG_INFO("CPUWaiter: wait on %dth fence\n", i);
        err = NvSciSyncFenceWait(&m_shareBuf->fence_nvm2cpu[i], m_cpuWaitContext, -1);
        PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "CPUWaiter: NvSciSyncFenceWait on %dth fence", i);
    }

fail:
    return status;
}

NvMediaStatus CPUWaiter::GetAttrList(NvSciSyncModule module, NvSciSyncAttrList &attrList)
{
    NvSciError err;
    NvSciSyncAttrKeyValuePair keyValue[2] = {};
    bool cpuWaiter = true;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvSciSyncAccessPerm cpuPerm;

    /* Fill NvSciSyncAttrList for CPU signaler */
    keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void*) &cpuWaiter;
    keyValue[0].len = sizeof(cpuWaiter);
    cpuPerm = NvSciSyncAccessPerm_WaitOnly;
    keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*)&cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);
    err = NvSciSyncAttrListSetAttrs(attrList, keyValue, 2);
    PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciSyncAttrListSetAttrs");

fail:
    return status;
}

bool CPUWaiter::DoWork()
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    LOG_INFO(">>>>>>>>>>> CPUWaiter: DoWork Start >>>>>>>>> \n");

    if (m_shareBuf->sema_nvm2cpu) {
        LOG_INFO("CPUWaiter: Wait for Dla fence to be ready \n");
        m_shareBuf->sema_nvm2cpu->Decrement(-1);
        LOG_INFO("CPUWaiter: Wait for Dla fence to be ready done\n");
    }

    if (m_shareBuf->sema_cpu2cpu) {
        LOG_INFO("CPUWaiter: Signal CPUSignaler to start \n");
        m_shareBuf->sema_cpu2cpu->Increment();
        LOG_INFO("CPUWaiter: Signal CPUSignaler to start done\n");
    }

    LOG_INFO("CPUWaiter: FenceWait start\n");
    status = FenceWait();
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "FenceWait");
    LOG_INFO("CPUWaiter: FenceWait done\n");

    LOG_INFO(">>>>>>>>>>> CPUWaiter: DoWork End >>>>>>>>> \n");

fail:
    return false;
}

CPUSignaler::CPUSignaler(
        std::string name,
        ShareBuf* shareBuf) :
    Worker(name),
    m_shareBuf(shareBuf)
{
}

NvMediaStatus CPUSignaler::GenerateFences()
{
    NvSciError err;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    for (auto i = 0; i < m_shareBuf->BufferSize; ++i) {
        /* Generate NvSciSyncFence */
        err = NvSciSyncObjGenerateFence(m_shareBuf->syncObj_cpu2nvm, &m_shareBuf->fence_cpu2nvm[i]);
        PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciSyncObjGenerateFence");
    }

fail:
    return status;
}

NvMediaStatus CPUSignaler::SignalFences()
{
    NvSciError err;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    for (auto i = 0; i < m_shareBuf->BufferSize; ++i) {
        /* Signal the NvSciSyncFence */
        LOG_INFO("CPU signaler signal %dth fence\n", i);
        err = NvSciSyncObjSignal(m_shareBuf->syncObj_cpu2nvm);
        PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciSyncObjSignal");
    }

fail:
    return status;
}

NvMediaStatus CPUSignaler::GetAttrList(NvSciSyncModule module, NvSciSyncAttrList &attrList)
{
    NvSciError err;
    NvSciSyncAttrKeyValuePair keyValue[2] = {};
    bool cpuSignaler = true;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvSciSyncAccessPerm cpuPerm;

    /* Fill NvSciSyncAttrList for CPU signaler */
    keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void*) &cpuSignaler;
    keyValue[0].len = sizeof(cpuSignaler);
    cpuPerm = NvSciSyncAccessPerm_SignalOnly;
    keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*)&cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);
    err = NvSciSyncAttrListSetAttrs(attrList, keyValue, 2);
    PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciSyncAttrListSetAttrs");

fail:
    return status;
}

bool CPUSignaler::DoWork()
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    LOG_INFO(">>>>>>>>>>> CPUSignaler: DoWork Start >>>>>>>>> \n");
    LOG_INFO("CPUSignaler: Generate fences \n");
    status = GenerateFences();
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "GenerateFences");
    LOG_INFO("CPUSignaler: Generate fences done\n");

    if (m_shareBuf->sema_cpu2nvm) {
        LOG_INFO("CPUSignaler: Signal Dla to start \n");
        m_shareBuf->sema_cpu2nvm->Increment();
        LOG_INFO("CPUSignaler: Signal Dla to start done\n");
    }

    if (m_shareBuf->sema_cpu2cpu) {
        LOG_INFO("CPUSignaler: Wait for CPUWaiter \n");
        m_shareBuf->sema_cpu2cpu->Decrement(-1);
        LOG_INFO("CPUSignaler: Wait for CPUWaiter done\n");
    }

    LOG_INFO("CPUSignaler: SignalFences Start\n");
    status = SignalFences();
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "SignalFences");
    LOG_INFO("CPUSignaler: SignalFences Done\n");

    LOG_INFO("<<<<<<<<<<< CPUSignaler: DoWork End <<<<<<<<<<< \n");

fail:
    return false;
}

DlaWorker::DlaWorker(
        uint32_t dlaId,
        uint32_t numTasks,
        std::string profileName,
        std::string name,
        ShareBuf* shareBuf) :
    Worker(name),
    m_shareBuf(shareBuf),
    m_rtDla(dlaId, numTasks, profileName)
{
}

NvMediaStatus DlaWorker::CreateDla()
{
    return m_rtDla.SetUp();
}

NvMediaStatus DlaWorker::WaitAndSignal()
{
    auto status = m_rtDla.RunTest(false);
    CHECK_FAIL( status == NVMEDIA_STATUS_OK, "WaitAndSignal");

fail:
    return status;
}

NvMediaStatus DlaWorker::GetAttrList(
                NvSciSyncModule module,
                NvSciSyncAttrList &attrList,
                NvMediaNvSciSyncClientType syncType)
{
    auto status = m_rtDla.m_upDla->GetAttrList(module, attrList, syncType);
    CHECK_FAIL( status == NVMEDIA_STATUS_OK, "GetAttrList");

fail:
    return status;
}

NvMediaStatus DlaWorker::RegisterSyncObj(
                NvMediaNvSciSyncObjType syncObjType,
                NvSciSyncObj syncObj)
{
    auto status = m_rtDla.m_upDla->RegisterSyncObj(syncObjType, syncObj);
    CHECK_FAIL( status == NVMEDIA_STATUS_OK, "RegisterSyncObj");

fail:
    return status;
}

NvMediaStatus DlaWorker::UnRegisterSyncObj(
                NvSciSyncObj syncObj)
{
    auto status = m_rtDla.m_upDla->UnRegisterSyncObj(syncObj);
    CHECK_FAIL( status == NVMEDIA_STATUS_OK, "UnRegisterSyncObj");

fail:
    return status;
}

NvMediaStatus DlaWorker::SetEOFSyncObj(
                NvSciSyncObj syncObj)
{
    auto status = m_rtDla.m_upDla->SetEOFSyncObj(syncObj);
    CHECK_FAIL( status == NVMEDIA_STATUS_OK, "SetEOFSyncObj");

fail:
    return status;
}

bool DlaWorker::DoWork()
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    LOG_INFO(">>>>>>>>>>> DlaWorker: DoWork Start >>>>>>>>> \n");
    if (m_shareBuf->sema_cpu2nvm) {
        LOG_INFO("DlaWorker: Wait for CPUSignaler to generate fences \n");
        m_shareBuf->sema_cpu2nvm->Decrement(-1);
        LOG_INFO("DlaWorker: Wait for CPUSignaler to generate fences done\n");
    }

    for (auto i = 0; i < m_shareBuf->BufferSize; i++) {
        LOG_INFO("DlaWorker: Dla insert pre fences \n");
        status = m_rtDla.m_upDla->InsertPreSciFences(&m_shareBuf->fence_cpu2nvm[i]);;
        CHECK_FAIL( status == NVMEDIA_STATUS_OK, "InsertPreSciFences");
        LOG_INFO("DlaWorker: Dla insert pre fences done\n");

        LOG_INFO("DlaWorker: Dla submit operation \n");
        status = WaitAndSignal();
        CHECK_FAIL( status == NVMEDIA_STATUS_OK, "WaitAndSignal");
        LOG_INFO("DlaWorker: Dla submit operation done\n");

        LOG_INFO("DlaWorker: Dla get EOF fences \n");
        status = m_rtDla.m_upDla->GetEOFSciFences(m_shareBuf->syncObj_nvm2cpu, &m_shareBuf->fence_nvm2cpu[i]);;
        CHECK_FAIL( status == NVMEDIA_STATUS_OK, "GetEOFSciFences");
        LOG_INFO("DlaWorker: Dla get EOF fences done\n");
    }

    if (m_shareBuf->sema_nvm2cpu) {
        LOG_INFO("DlaWorker: Signal CPU waiter EOF fence ready \n");
        m_shareBuf->sema_nvm2cpu->Increment();
        LOG_INFO("DlaWorker: Signal CPU waiter EOF fence ready done\n");
    }

    LOG_INFO("<<<<<<<<<<< DlaWorker: DoWork End <<<<<<<<<<< \n");

fail:
    return false;
}