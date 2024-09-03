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

#ifndef _TESTRUNTIME_H_
#define _TESTRUNTIME_H_

#include <memory>
#include <string>

#include "dla.h"
#include "tensor.h"

//! Class to test runtime mode

class TestRuntime final
{
friend class DlaWorker;

public:
    TestRuntime(
        uint32_t dlaId,
        uint32_t numTasks,
        std::string profileName,
        bool IsPingTest = false);

    ~TestRuntime();

    NvMediaStatus SetUp();

    NvMediaStatus RunTest(bool CheckStatus = true);

protected:
    NvMediaStatus InitNvSciBuf(void);

    void DeinitNvSciBuf(void);

    NvMediaStatus ReconcileAndAllocSciBufObj(
        NvMediaTensorAttr tensorAttrs[],
        uint32_t numAttrs,
        NvSciBufObj *sciBuf);

private:
    uint32_t m_dlaId;

    uint32_t m_numTasks;

    std::string m_profileName;

    bool m_isPingTest;

    uint32_t m_loadableIndex;

    std::unique_ptr<Dla> m_upDla;

    std::vector<NvSciBufObj> m_pInputTensorScibuf;

    std::vector<std::unique_ptr<Tensor>> m_vupInputTensor;

    std::vector<NvSciBufObj> m_pOutputTensorScibuf;

    std::vector<std::unique_ptr<Tensor>> m_vupOutputTensor;

    NvSciBufModule m_NvscibufModule = nullptr;

};

#endif // end of _TESTRUNTIME_H_
