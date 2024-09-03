/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <iterator>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>

#include "testRuntime.h"
#include "testMT.h"
#include "utils.h"

TestMT::TestMT(
        std::string profileName) :
    m_profileName(profileName)
{
}

DlaThread::DlaThread(
        uint32_t dlaId,
        uint32_t numTasks,
        std::string profileName,
        std::string name) :
    Worker(name),
    m_rtDla(dlaId, numTasks, profileName)
{
}

NvMediaStatus DlaThread::CreateDla()
{
    return m_rtDla.SetUp();
}

bool DlaThread::DoWork()
{
    auto status = m_rtDla.RunTest(true);
    CHECK_FAIL( status == NVMEDIA_STATUS_OK, "Dla submit operation");
    LOG_INFO("<<<<<<<<<<< DlaThread: DoWork End <<<<<<<<<<< \n");

fail:
    return false;
}

TestMT::~TestMT()
{
 //De-init
    if (m_Thread1) {
        m_Thread1->Stop();
    }

    if (m_Thread2) {
        m_Thread2->Stop();
    }

    if (m_Thread3) {
        m_Thread3->Stop();
    }

    if (m_Thread4) {
        m_Thread4->Stop();
    }
}

NvMediaStatus TestMT::SetUp()
{
 //Init
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint32_t dlaId0 = 0u;
    uint32_t dlaId1 = 1u;
    uint32_t num_tasks = 1u; // since we are only submitting 1 task per context

    m_Thread1.reset(new DlaThread(dlaId0, num_tasks, m_profileName,"m_Thread1"));

    status = m_Thread1->CreateDla();
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Thread1");

    m_Thread2.reset(new DlaThread(dlaId1, num_tasks, m_profileName,"m_Thread2"));

    status = m_Thread2->CreateDla();
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Thread2");

    m_Thread3.reset(new DlaThread(dlaId0, num_tasks, m_profileName,"m_Thread3"));

    status = m_Thread3->CreateDla();
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Thread3");

    m_Thread4.reset(new DlaThread(dlaId1, num_tasks, m_profileName,"m_Thread4"));

    status = m_Thread4->CreateDla();
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Thread4");

fail:
    return status;
}

NvMediaStatus TestMT::RunTest(bool CheckStatus)
{
 //Run
    m_Thread1->Start();
    m_Thread2->Start();
    m_Thread3->Start();
    m_Thread4->Start();

    return NVMEDIA_STATUS_OK;
}

