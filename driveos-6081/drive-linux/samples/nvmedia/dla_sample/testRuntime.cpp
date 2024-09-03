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

#include <iterator>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>

#include "testRuntime.h"
#include "utils.h"

TestRuntime::TestRuntime(
        uint32_t dlaId,
        uint32_t numTasks,
        std::string profileName,
        bool IsPingTest) :
    m_dlaId(dlaId),
    m_numTasks(numTasks),
    m_profileName(profileName),
    m_isPingTest(IsPingTest),
    m_loadableIndex(Dla::INVALID_LOADABLEINDEX)
{
}

TestRuntime::~TestRuntime()
{
    if (m_isPingTest) {
        return;
    }

    for (auto i = 0u; i < m_vupInputTensor.size(); i++) {
        m_upDla->DataUnregister(m_loadableIndex, m_vupInputTensor[i].get());
    }

    for (auto i = 0u; i < m_vupOutputTensor.size(); i++) {
        m_upDla->DataUnregister(m_loadableIndex, m_vupOutputTensor[i].get());
    }

    m_upDla->RemoveLoadable(m_loadableIndex);

    m_upDla.reset();

    for (auto i = 0u; i < m_pInputTensorScibuf.size(); i++) {
        if (m_pInputTensorScibuf[i]) {
            NvSciBufObjFree(m_pInputTensorScibuf[i]);
        }
    }

    for (auto i = 0u; i < m_vupInputTensor.size(); i++) {
        m_vupInputTensor[i].reset();
    }

    for (auto i = 0u; i < m_pOutputTensorScibuf.size(); i++) {
        if (m_pOutputTensorScibuf[i]) {
            NvSciBufObjFree(m_pOutputTensorScibuf[i]);
        }
    }

    for (auto i = 0u; i < m_vupOutputTensor.size(); i++) {
        m_vupOutputTensor[i].reset();
    }

    DeinitNvSciBuf();
}

NvMediaStatus TestRuntime::SetUp()
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    std::vector<NvMediaDlaTensorDescriptor> vInputTensorDesc;
    std::vector<NvMediaDlaTensorDescriptor> vOutputTensorDesc;
    NvSciBufObj sciBufObj;

    if (m_isPingTest) {
        LOG_INFO("Ping test \n");
        goto fail;
    }

    status = InitNvSciBuf();
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "InitNvSciBuf");

    m_upDla = Dla::Create();
    PROPAGATE_ERROR_FAIL(m_upDla != nullptr, "Create");

    status = m_upDla->Init(m_dlaId, m_numTasks);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Init");

    status = m_upDla->AddLoadable(m_profileName, m_loadableIndex);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "AddLoadable");

    status = m_upDla->GetDesc(m_loadableIndex, vInputTensorDesc, vOutputTensorDesc);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "GetDesc");

    // input tensor allocation
    for (auto i = 0u; i < vInputTensorDesc.size(); i++) {
        status = ReconcileAndAllocSciBufObj(vInputTensorDesc[i].tensorAttrs, vInputTensorDesc[i].numAttrs, &sciBufObj);
        CHECK_FAIL(status == NVMEDIA_STATUS_OK, "ReconcileAndAllocSciBufObj");

        m_pInputTensorScibuf.push_back(sciBufObj);

        std::unique_ptr<Tensor> upTensor(new Tensor());

        status = upTensor->Create(sciBufObj);
        CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Tensor Create");

        status = upTensor->SetData(0);
        CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Tensor SetData");

        status = m_upDla->DataRegister(m_loadableIndex, upTensor.get());

        m_vupInputTensor.push_back(std::move(upTensor));
    }

    // output tensor allocation
    for (auto i = 0u; i < vOutputTensorDesc.size(); i++) {
        status = ReconcileAndAllocSciBufObj(vOutputTensorDesc[i].tensorAttrs, vOutputTensorDesc[i].numAttrs, &sciBufObj);
        CHECK_FAIL(status == NVMEDIA_STATUS_OK, "ReconcileAndAllocSciBufObj");

        m_pOutputTensorScibuf.push_back(sciBufObj);

        std::unique_ptr<Tensor> upTensor(new Tensor());

        status = upTensor->Create(sciBufObj);
        CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Tensor Create");

        status = upTensor->SetData(0);
        CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Tensor SetData");

        status = m_upDla->DataRegister(m_loadableIndex, upTensor.get());

        m_vupOutputTensor.push_back(std::move(upTensor));
    }

fail:
    return status;
}

NvMediaStatus TestRuntime::RunTest(bool CheckStatus)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    std::vector<Tensor *> vInputTensor;
    std::vector<Tensor *> vOutputTensor;

    if (m_isPingTest) {
        status = Dla::PingById(m_dlaId);
        CHECK_FAIL(status == NVMEDIA_STATUS_OK, "PingById");
        return status;
    }

    for (auto i = 0u; i < m_vupInputTensor.size(); i++) {
        vInputTensor.push_back(m_vupInputTensor[i].get());
    }

    for (auto i = 0u; i < m_vupOutputTensor.size(); i++) {
        vOutputTensor.push_back(m_vupOutputTensor[i].get());
    }

    status = m_upDla->Submit(m_loadableIndex, vInputTensor, vOutputTensor);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Submit");

    if (CheckStatus) {
        status = m_vupOutputTensor[0]->GetStatus();
        CHECK_FAIL(status == NVMEDIA_STATUS_OK, "GetStatus");
    }

fail:
    return status;
}

NvMediaStatus TestRuntime::InitNvSciBuf(void)
{
    NvSciError err = NvSciError_Success;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;

    err = NvSciBufModuleOpen(&m_NvscibufModule);
    PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciBufModuleOpen");

    status = NvMediaTensorNvSciBufInit();
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaTensorNvSciBufInit");

fail:
    return status;
}

void TestRuntime::DeinitNvSciBuf()
{
    NvSciBufModuleClose(m_NvscibufModule);

    NvMediaTensorNvSciBufDeinit();
}

NvMediaStatus TestRuntime::ReconcileAndAllocSciBufObj(
    NvMediaTensorAttr tensorAttrs[],
    uint32_t numAttrs,
    NvSciBufObj *sciBuf)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvSciError err = NvSciError_Success;
    NvSciBufAttrValAccessPerm access_perm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrList unreconciled_attrlistTensor = NULL;
    NvSciBufAttrList reconciled_attrlist = NULL;
    NvSciBufAttrList conflictlist = NULL;

    NvSciBufAttrKeyValuePair attr_kvp = {NvSciBufGeneralAttrKey_RequiredPerm,
                                         &access_perm,
                                         sizeof(access_perm)};

    err = NvSciBufAttrListCreate(m_NvscibufModule, &unreconciled_attrlistTensor);
    PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciBufAttrListCreate");

    err = NvSciBufAttrListSetAttrs(unreconciled_attrlistTensor, &attr_kvp, 1);
    PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciBufAttrListSetAttrs");

    status = Tensor::FillNvSciBufTensorAttrs(tensorAttrs, numAttrs, unreconciled_attrlistTensor);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "GetNvSciBufTensorAttrs");

    err = NvSciBufAttrListReconcile(&unreconciled_attrlistTensor,
                                    1,
                                    &reconciled_attrlist,
                                    &conflictlist);
    PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciBufAttrListReconcile");

    err = NvSciBufObjAlloc(reconciled_attrlist,
                           sciBuf);
    PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciBufAttrListReconcile");

    if(unreconciled_attrlistTensor) {
        NvSciBufAttrListFree(unreconciled_attrlistTensor);
    }
    if(reconciled_attrlist) {
        NvSciBufAttrListFree(reconciled_attrlist);
    }
    if(conflictlist) {
        NvSciBufAttrListFree(conflictlist);
    }

fail:
    return status;
}
