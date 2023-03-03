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

#ifndef _SCISYNC_H_
#define _SCISYNC_H_

#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>

#include "nvmedia_core.h"
#include "nvscisync.h"
#include "nvmedia_dla.h"
#include "nvmedia_dla_nvscisync.h"

#include "testRuntime.h"
#include "utils.h"

//! Semaphore class.
class Semaphore final
{
public:

    //! Construct a semaphore.
    //! \param uInitCount The initial semaphore count.
    //! \param uMaxCount The maximum semaphore count.
    Semaphore(uint32_t uInitCount,
              uint32_t uMaxCount);

    //! Disallow default constructor
    Semaphore() = delete;

    //! Dtor.
    ~Semaphore();

    //! Increment the semaphore count.
    void Increment();

    //! Decrement the semaphore count.
    //! \param timeoutMs Maximum amount of time to wait (in milliseconds).
    //!        -1 means infinite wait;
    //! \retval false The semaphore wasn't decremented.
    //! \retval true The sempahore was decremented.
    bool Decrement(int32_t timeoutMs);

private:
    std::condition_variable m_condition;
    std::mutex m_mutex;
    uint32_t m_maxCount = 0U;
    uint32_t m_count = 0U;
};

//! Buffer objects shared between different thread
class ShareBuf final
{
public:
    ShareBuf(int32_t numOfBuffers) :
        BufferSize(numOfBuffers),
        sema_cpu2nvm(new Semaphore(0, 1)),
        sema_nvm2cpu(new Semaphore(0, 1)),
        sema_cpu2cpu(new Semaphore(0, 1)),
        syncObj_cpu2nvm(nullptr),
        syncObj_nvm2cpu(nullptr)
    {
        for (auto i = 0; i < BufferSize; i++) {
            NvSciSyncFence fence {};
            fence_cpu2nvm.push_back(fence);

            fence_nvm2cpu.push_back(fence);
        }
    }

    ~ShareBuf() {
        for (auto &item : fence_cpu2nvm) {
            NvSciSyncFenceClear(&item);
        }

        for (auto &item : fence_nvm2cpu) {
            NvSciSyncFenceClear(&item);
        }

        if (syncObj_cpu2nvm) {
            NvSciSyncObjFree(syncObj_cpu2nvm);
            syncObj_cpu2nvm = nullptr;
        }

        if (syncObj_nvm2cpu) {
            NvSciSyncObjFree(syncObj_nvm2cpu);
            syncObj_nvm2cpu = nullptr;
        }
    }

    int32_t BufferSize;

    std::shared_ptr<Semaphore> sema_cpu2nvm;

    std::shared_ptr<Semaphore> sema_nvm2cpu;

    std::shared_ptr<Semaphore> sema_cpu2cpu;

    std::vector<NvSciSyncFence> fence_cpu2nvm;

    std::vector<NvSciSyncFence> fence_nvm2cpu;

    NvSciSyncObj syncObj_cpu2nvm;

    NvSciSyncObj syncObj_nvm2cpu;
};

//! Base class for a thread
class Worker
{
public:
    Worker(
        std::string name);

    NvMediaStatus Start();

    void Stop();

    virtual ~Worker() = default;

protected:

    virtual bool DoWork() = 0;

private:
    static void* m_FuncStatic(void* vpParam);

    void* m_Func();

    std::string m_name;

    std::unique_ptr<std::thread> m_thread;

    Semaphore m_threadStartFlag;

    bool m_bQuit;
};

//! Class for CPU waiter thread
class CPUWaiter : public Worker
{
public:
    CPUWaiter(
        std::string name,
        NvSciSyncCpuWaitContext cpuWaitContext,
        ShareBuf* shareBuf);

    NvMediaStatus GetAttrList(NvSciSyncModule module, NvSciSyncAttrList &attrList);

    ~CPUWaiter() = default;

protected:
    NvMediaStatus FenceWait();

    virtual bool DoWork() override;

private:
    NvSciSyncCpuWaitContext m_cpuWaitContext;

    ShareBuf* m_shareBuf = nullptr;
};

//! Class for CPU signaler thread
class CPUSignaler : public Worker
{
public:
    CPUSignaler(
        std::string name,
        ShareBuf* shareBuf);

    NvMediaStatus GetAttrList(NvSciSyncModule module, NvSciSyncAttrList &attrList);

    ~CPUSignaler() = default;

protected:
    NvMediaStatus GenerateFences();

    NvMediaStatus SignalFences();

    virtual bool DoWork() override;

private:
    ShareBuf* m_shareBuf = nullptr;
};

//! Class for Dla operation thread
class DlaWorker : public Worker
{
public:
    DlaWorker(
        uint32_t dlaId,
        uint32_t numTasks,
        std::string profileName,
        std::string name,
        ShareBuf* shareBuf);

    NvMediaStatus CreateDla();

    NvMediaStatus GetAttrList(
                    NvSciSyncModule module,
                    NvSciSyncAttrList &attrList,
                    NvMediaNvSciSyncClientType syncType);

    NvMediaStatus RegisterSyncObj(NvMediaNvSciSyncObjType syncObjType, NvSciSyncObj syncObj);

    NvMediaStatus UnRegisterSyncObj(NvSciSyncObj syncObj);

    NvMediaStatus SetEOFSyncObj(NvSciSyncObj syncObj);

    ~DlaWorker() = default;

protected:
    NvMediaStatus WaitAndSignal();

    virtual bool DoWork() override;

private:
    ShareBuf* m_shareBuf = nullptr;

    TestRuntime m_rtDla;
};

#endif // _SCISYNC_H_
