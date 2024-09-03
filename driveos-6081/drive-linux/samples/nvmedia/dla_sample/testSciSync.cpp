/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "testSciSync.h"

TestSciSync::TestSciSync(
        uint32_t dlaId,
        uint32_t numTasks,
        std::string profileName) :
    m_dlaId(dlaId),
    m_numTasks(numTasks),
    m_profileName(profileName),
    m_shareBuf(new ShareBuf(numTasks))
{
}

TestSciSync::~TestSciSync()
{
    if (m_spSig) {
        m_spSig->Stop();
    }

    if (m_spDla) {
        m_spDla->Stop();
    }

    if (m_spWaiter) {
        m_spWaiter->Stop();
    }

    // Unregister sync obj
    if (m_spDla) {
        m_spDla->UnRegisterSyncObj(m_shareBuf->syncObj_cpu2nvm);
        m_spDla->UnRegisterSyncObj(m_shareBuf->syncObj_nvm2cpu);
    }

    m_shareBuf.reset();

    if (m_cpuWaitContext != nullptr) {
        NvSciSyncCpuWaitContextFree(m_cpuWaitContext);
    }

    if (m_module != nullptr) {
        NvSciSyncModuleClose(m_module);
    }
}

NvMediaStatus TestSciSync::SetUp()
{
    NvSciError err;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvSciSyncAttrList cpu_signalerAttrList = NULL;
    NvSciSyncAttrList Dla_waiterAttrList = NULL;
    NvSciSyncAttrList Dla_signalerAttrList = NULL;
    NvSciSyncAttrList cpu_waiterAttrList = NULL;

    /* Initialize the NvSciSync module */
    err = NvSciSyncModuleOpen(&m_module);
    PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciSyncModuleOpen");

    // Create CpuWaitContextg
    err = NvSciSyncCpuWaitContextAlloc(m_module, &m_cpuWaitContext);
    PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciSyncCpuWaitContextAlloc");

    m_spSig.reset(new CPUSignaler(
                            "CPUSignaler",
                            m_shareBuf.get()));

    m_spDla.reset(new DlaWorker(
                            m_dlaId, m_numTasks, m_profileName,
                            "DlaWorker",
                            m_shareBuf.get()));

    status = m_spDla->CreateDla();
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "CreateDla");

    m_spWaiter.reset(new CPUWaiter(
                            "CPUWaiter",
                            m_cpuWaitContext,
                            m_shareBuf.get()));

    err = NvSciSyncAttrListCreate(m_module, &cpu_signalerAttrList);
    PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciSyncAttrListCreate");

    status = m_spSig->GetAttrList(m_module, cpu_signalerAttrList);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Signal GetAttrList");

    err = NvSciSyncAttrListCreate(m_module, &Dla_waiterAttrList);
    PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciSyncAttrListCreate");

    status = m_spDla->GetAttrList(m_module, Dla_waiterAttrList, NVMEDIA_WAITER);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Dla GetAttrList as waiter");

    // Reconcile and alloc sync obj
    status = CreateSyncObjFromAttrList(cpu_signalerAttrList, Dla_waiterAttrList, &m_shareBuf->syncObj_cpu2nvm);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "CreateSyncObjFromAttrList between CPU signaler and Dla waiter");

    err = NvSciSyncAttrListCreate(m_module, &Dla_signalerAttrList);
    PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciSyncAttrListCreate");

    status = m_spDla->GetAttrList(m_module, Dla_signalerAttrList, NVMEDIA_SIGNALER);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Dla GetAttrList as signaler");

    err = NvSciSyncAttrListCreate(m_module, &cpu_waiterAttrList);
    PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciSyncAttrListCreate");

    status = m_spWaiter->GetAttrList(m_module, cpu_waiterAttrList);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Signal GetAttrList");

    // Reconcile and alloc sync obj
    status = CreateSyncObjFromAttrList(Dla_signalerAttrList, cpu_waiterAttrList, &m_shareBuf->syncObj_nvm2cpu);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "CreateSyncObjFromAttrList between Dla signaler and CPU waiter");

    // Register Syncobj for Dla
    status = m_spDla->RegisterSyncObj(NVMEDIA_PRESYNCOBJ, m_shareBuf->syncObj_cpu2nvm);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Dla RegisterSyncObj");

    status = m_spDla->RegisterSyncObj(NVMEDIA_EOFSYNCOBJ, m_shareBuf->syncObj_nvm2cpu);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Dla RegisterSyncObj");

    // Set EOF for Dla
    status = m_spDla->SetEOFSyncObj(m_shareBuf->syncObj_nvm2cpu);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Dla SetEOFSyncObj");

fail:
    /* Free Attribute list objects */
    FreeSyncAttrList(cpu_signalerAttrList);
    FreeSyncAttrList(Dla_waiterAttrList);
    FreeSyncAttrList(Dla_signalerAttrList);
    FreeSyncAttrList(cpu_waiterAttrList);

    return status;
}

NvMediaStatus TestSciSync::RunTest()
{
    m_spWaiter->Start();

    m_spDla->Start();

    m_spSig->Start();

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus TestSciSync::CreateSyncObjFromAttrList(
            NvSciSyncAttrList list1,
            NvSciSyncAttrList list2,
            NvSciSyncObj *syncObj)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvSciError err;
    NvSciSyncAttrList unreconcileDlast[2] = {};
    NvSciSyncAttrList reconcileDlalist = nullptr;
    NvSciSyncAttrList newConflictList = nullptr;

    unreconcileDlast[0] = list1;
    unreconcileDlast[1] = list2;

    // Reconcile Signaler and Waiter NvSciSyncAttrList
    err = NvSciSyncAttrListReconcile(unreconcileDlast, 2, &reconcileDlalist,
            &newConflictList);
    PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciSyncAttrListReconcile");

    // Create NvSciSync object and get the syncObj
    err = NvSciSyncObjAlloc(reconcileDlalist, syncObj);
    PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciSyncObjAlloc");

fail:
    FreeSyncAttrList(reconcileDlalist);
    FreeSyncAttrList(newConflictList);

    return status;
}
