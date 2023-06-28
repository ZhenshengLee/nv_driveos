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

#ifndef _TESTMT_H_
#define _TESTMT_H_

#include <memory>
#include <string>

#include "dla.h"
#include "tensor.h"
#include "sciSync.h"

//! Class for Dla operation thread
class DlaThread : public Worker
{
public:
    DlaThread(
        uint32_t dlaId,
        uint32_t numTasks,
        std::string profileName,
        std::string name);

    NvMediaStatus CreateDla();

    ~DlaThread() = default;

protected:

    virtual bool DoWork() override;

private:
    TestRuntime m_rtDla;
};


//! Class to test multithread mode
class TestMT final
{
friend class DlaThread;

public:
    TestMT(std::string profileName);

    ~TestMT();

    NvMediaStatus SetUp();

    NvMediaStatus RunTest(bool CheckStatus = true);

private:
    std::string m_profileName;

    std::shared_ptr<DlaThread> m_Thread1 = nullptr;

    std::shared_ptr<DlaThread> m_Thread2 = nullptr;

    std::shared_ptr<DlaThread> m_Thread3 = nullptr;

    std::shared_ptr<DlaThread> m_Thread4 = nullptr;

};

#endif // end of _TESTMT_H_
