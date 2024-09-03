/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _TESTSCISYNC_H_
#define _TESTSCISYNC_H_

#include "sciSync.h"
#include "nvscisync.h"

//! Class to test scisync mode

class TestSciSync final
{
public:
    TestSciSync(
        uint32_t dlaId,
        uint32_t numTasks,
        std::string profileName);

    ~TestSciSync();

    NvMediaStatus SetUp();

    NvMediaStatus RunTest();

protected:
    NvMediaStatus CreateSyncObjFromAttrList(
        NvSciSyncAttrList list1,
        NvSciSyncAttrList list2,
        NvSciSyncObj *syncOjb);

    inline void FreeSyncAttrList(NvSciSyncAttrList list) {
        if( list != nullptr) {
            NvSciSyncAttrListFree(list);
            list = nullptr;
        }
    }

private:
    uint32_t m_dlaId;

    uint32_t m_numTasks;

    std::string m_profileName;

    std::unique_ptr<ShareBuf> m_shareBuf;

    std::shared_ptr<CPUSignaler> m_spSig = nullptr;

    std::shared_ptr<DlaWorker> m_spDla = nullptr;

    std::shared_ptr<CPUWaiter> m_spWaiter = nullptr;

    NvSciSyncCpuWaitContext m_cpuWaitContext = nullptr;

    NvSciSyncModule m_module = nullptr;
};

#endif // _TESTSCISYNC_H_
