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

#include "dla.h"
#include "utils.h"

#define MODULE "Dla"

Dla::~Dla()
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    for (auto it = m_vLoadables.begin(); it != m_vLoadables.end(); it++) {
        if (*it) {
            status = NvMediaDlaLoadableDestroy(m_pDla, *it);
            CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaLoadableDestroy");
        }
    }

    m_vLoadables.clear();

    if (m_pDla) {
        status = NvMediaDlaDestroy(m_pDla);
        CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaDestroy");
        m_pDla = nullptr;
    }

fail:
    return;
}

Dla::Dla(NvMediaDla *dla) :
    m_pDla(dla)
{
}

NvMediaStatus Dla::GetDlaVersion(NvMediaVersion *version)
{
    NvMediaStatus status = NvMediaDlaGetVersion(version);

    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaGetVersion");

fail:
    return status;
}

NvMediaStatus Dla::PingById(const uint32_t dlaId)
{
    NvMediaStatus status = NvMediaDlaPingById(dlaId);

    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaPingById");

fail:
    return status;
}

std::unique_ptr<Dla> Dla::Create()
{
    NvMediaDla *dla = NvMediaDlaCreate();
    CHECK_FAIL(dla != nullptr, "NvMediaDlaCreate");

fail:
    return std::unique_ptr<Dla>(new Dla(dla));
}

NvMediaStatus Dla::Init(uint32_t dlaId, uint32_t numTasks)
{
    NvMediaStatus status = NvMediaDlaInit(m_pDla, dlaId, numTasks);

    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaInit");

fail:
    return status;
}

NvMediaStatus Dla::AddLoadable(std::string profileName, uint32_t &loadableIndex)
{
    NvMediaDlaBinaryLoadable binaryLoadable {};
    NvMediaStatus status {NVMEDIA_STATUS_OK};
    NvMediaDlaLoadable *loadable {nullptr};

    LOG_DBG("Reading loadable file to memory = %s\n", profileName.c_str());
    binaryLoadable.loadable = readFileToMemory(profileName.c_str(), binaryLoadable.loadableSize);
    PROPAGATE_ERROR_FAIL(binaryLoadable.loadable != nullptr, "readFileToMemory");

    // Create loadable handle
    status = NvMediaDlaLoadableCreate(m_pDla, &loadable);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaLoadableCreate");

    // Append loadable
    status = NvMediaDlaAppendLoadable(m_pDla, binaryLoadable, loadable);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaAppendLoadable");

    status = NvMediaDlaSetCurrentLoadable(m_pDla, loadable);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaSetCurrentLoadable");

    // Load loadable
    status = NvMediaDlaLoadLoadable(m_pDla);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaLoadLoadable");

    m_vLoadables.push_back(loadable);

    loadableIndex = m_vLoadables.size() - 1;
fail:
    if (binaryLoadable.loadable) {
        delete [] binaryLoadable.loadable;
    }

    return status;
}

NvMediaStatus Dla::GetDesc(
    uint32_t loadableIndex,
    std::vector<NvMediaDlaTensorDescriptor> &vInputTensorDesc,
    std::vector<NvMediaDlaTensorDescriptor> &vOutputTensorDesc
)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvMediaDlaTensorDescriptor tensorDesc {};
    int32_t num {0};

    PROPAGATE_ERROR_FAIL(
        vInputTensorDesc.size() == 0 && vOutputTensorDesc.size() == 0,
        "Check descriptor argument");

    PROPAGATE_ERROR_FAIL(
        loadableIndex >= 0 && loadableIndex < m_vLoadables.size(),
        "Check loadable index argument");

    status = NvMediaDlaSetCurrentLoadable(m_pDla, m_vLoadables[loadableIndex]);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaSetCurrentLoadable");

    // Input tensor
    status = NvMediaDlaGetNumOfInputTensors(m_pDla, &num);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaGetNumOfInputTensors");

    for (auto i = 0; i < num; i++) {
        status = NvMediaDlaGetInputTensorDescriptor(m_pDla, i, &tensorDesc);
        CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaGetInputTensorDescriptor");
        status = PrintTensorDesc(&tensorDesc);
        CHECK_FAIL(status == NVMEDIA_STATUS_OK, "PrintTensorDesc of input tensor");
        vInputTensorDesc.push_back(tensorDesc);
    }

    // Output tensor
    status = NvMediaDlaGetNumOfOutputTensors(m_pDla, &num);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaGetNumOfOutputTensors");

    for (auto i = 0; i < num; i++) {
        status = NvMediaDlaGetOutputTensorDescriptor(m_pDla, i, &tensorDesc);
        CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaGetOutputTensorDescriptor");
        status = PrintTensorDesc(&tensorDesc);
        CHECK_FAIL(status == NVMEDIA_STATUS_OK, "PrintTensorDesc of output tensor");
        vOutputTensorDesc.push_back(tensorDesc);
    }

fail:
    return status;
}

NvMediaStatus Dla::DataRegister(
    uint32_t loadableIndex,
    Tensor *tensor
)
{
    NvMediaDlaData dlaData {};
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    PROPAGATE_ERROR_FAIL(
        loadableIndex >= 0 && loadableIndex < m_vLoadables.size(),
        "Check loadable index argument");

    status = NvMediaDlaSetCurrentLoadable(m_pDla, m_vLoadables[loadableIndex]);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaSetCurrentLoadable");

    dlaData.type = NVMEDIA_DLA_DATA_TYPE_TENSOR;
    dlaData.pointer.tensor = tensor->GetPtr();
    status = NvMediaDlaDataRegister(m_pDla, &dlaData, 0);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaDataRegister");

fail:
    return status;
}

NvMediaStatus Dla::DataUnregister(
    uint32_t loadableIndex,
    Tensor *tensor
)
{
    NvMediaDlaData dlaData {};
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    PROPAGATE_ERROR_FAIL(
        loadableIndex >= 0 && loadableIndex < m_vLoadables.size(),
        "Check loadable index argument");

    status = NvMediaDlaSetCurrentLoadable(m_pDla, m_vLoadables[loadableIndex]);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaSetCurrentLoadable");

    dlaData.type = NVMEDIA_DLA_DATA_TYPE_TENSOR;
    dlaData.pointer.tensor = tensor->GetPtr();
    status = NvMediaDlaDataUnregister(m_pDla, &dlaData);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaDataUnregister");

fail:
    return status;
}

NvMediaStatus Dla::RemoveLoadable(uint32_t loadableIndex)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    PROPAGATE_ERROR_FAIL(
        loadableIndex >= 0 && loadableIndex < m_vLoadables.size(),
        "Check loadable index argument");

    status = NvMediaDlaSetCurrentLoadable(m_pDla, m_vLoadables[loadableIndex]);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaSetCurrentLoadable");

    status = NvMediaDlaRemoveLoadable(m_pDla);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaRemoveLoadable");

fail:
    return status;
}

NvMediaStatus Dla::Submit(
    uint32_t loadableIndex,
        std::vector<Tensor*>  &vpInputTensor,
        std::vector<Tensor*>  &vpOutputTensor
)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvMediaDlaArgs inputArgs {};
    NvMediaDlaArgs outputArgs {};

    PROPAGATE_ERROR_FAIL(
        vpInputTensor.size() < MAX_NUM_OF_DLA_DATA &&
        vpOutputTensor.size() < MAX_NUM_OF_DLA_DATA,
        "Check input args");

    PROPAGATE_ERROR_FAIL(
        loadableIndex >= 0 && loadableIndex < m_vLoadables.size(),
        "Check loadable index argument");

    // input tensor
    for (auto i = 0u; i < vpInputTensor.size(); i++) {
        m_aInputDlaData[i].type = NVMEDIA_DLA_DATA_TYPE_TENSOR;
        m_aInputDlaData[i].pointer.tensor = vpInputTensor[i]->GetPtr();
    }
    inputArgs.dlaData = m_aInputDlaData.data();
    inputArgs.numArgs = vpInputTensor.size();

    // output tensor
    for (auto i = 0u; i < vpOutputTensor.size(); i++) {
        m_aOutputDlaData[i].type = NVMEDIA_DLA_DATA_TYPE_TENSOR;
        m_aOutputDlaData[i].pointer.tensor = vpOutputTensor[i]->GetPtr();
    }
    outputArgs.dlaData = m_aOutputDlaData.data();
    outputArgs.numArgs = vpOutputTensor.size();

    status = NvMediaDlaSetCurrentLoadable(m_pDla, m_vLoadables[loadableIndex]);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaSetCurrentLoadable");

    status = NvMediaDlaSubmit(m_pDla, &inputArgs, NULL, &outputArgs, NVMEDIA_DLA_DEFAULT_TASKTIMEOUT);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaSubmit");

fail:
    return status;
}

NvMediaStatus Dla::PrintTensorDesc(NvMediaDlaTensorDescriptor *tensorDesc)
{
    uint32_t i = 0;
    LOG_DBG("Tensor descripor \n");
    LOG_DBG("\t name = %s \n", tensorDesc->name);

    for (i = 0; i < tensorDesc->numAttrs; i++) {
        switch(tensorDesc->tensorAttrs[i].type) {
        case NVM_TENSOR_ATTR_DATA_TYPE:
            LOG_DBG("\t Data type = %d \n", tensorDesc->tensorAttrs[i].value);
            break;
        case NVM_TENSOR_ATTR_BITS_PER_ELEMENT:
            LOG_DBG("\t Bits per element = %d \n", tensorDesc->tensorAttrs[i].value);
            break;
        case NVM_TENSOR_ATTR_DIMENSION_ORDER:
            LOG_DBG("\t dimension order = %d \n", tensorDesc->tensorAttrs[i].value);
            break;
        case NVM_TENSOR_ATTR_CPU_ACCESS:
            LOG_DBG("\t CPU access = %d \n", tensorDesc->tensorAttrs[i].value);
            break;
        case NVM_TENSOR_ATTR_ALLOC_TYPE:
            LOG_DBG("\t Alloc type = %d \n", tensorDesc->tensorAttrs[i].value);
            break;
        case NVM_TENSOR_ATTR_4D_N:
            LOG_DBG("\t N = %d \n", tensorDesc->tensorAttrs[i].value);
            break;
        case NVM_TENSOR_ATTR_4D_C:
            LOG_DBG("\t C = %d \n", tensorDesc->tensorAttrs[i].value);
            break;
        case NVM_TENSOR_ATTR_4D_H:
            LOG_DBG("\t H = %d \n", tensorDesc->tensorAttrs[i].value);
            break;
        case NVM_TENSOR_ATTR_4D_W:
            LOG_DBG("\t W = %d \n", tensorDesc->tensorAttrs[i].value);
            break;
        case NVM_TENSOR_ATTR_4D_X:
            LOG_DBG("\t X = %d \n", tensorDesc->tensorAttrs[i].value);
            break;
        default:
            return NVMEDIA_STATUS_ERROR;
        }
    }

    return NVMEDIA_STATUS_OK;
}

// SciSync related API
NvMediaStatus Dla::GetAttrList(
                NvSciSyncModule module,
                NvSciSyncAttrList &attrList,
                NvMediaNvSciSyncClientType syncType)
{
    NvMediaStatus status = NvMediaDlaFillNvSciSyncAttrList(m_pDla, attrList, syncType);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaFillNvSciSyncAttrList");

fail:
    return status;
}

NvMediaStatus Dla::RegisterSyncObj(
                NvMediaNvSciSyncObjType syncObjType,
                NvSciSyncObj syncObj)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = NvMediaDlaRegisterNvSciSyncObj(m_pDla, syncObjType, syncObj);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaRegisterNvSciSyncObj");

fail:
    return status;
}

NvMediaStatus Dla::UnRegisterSyncObj(
                NvSciSyncObj syncObj)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = NvMediaDlaUnregisterNvSciSyncObj(m_pDla, syncObj);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaUnregisterNvSciSyncObj");

fail:
    return status;
}

NvMediaStatus Dla::SetEOFSyncObj(
                NvSciSyncObj syncObj)
{
    NvMediaStatus status = NvMediaDlaSetNvSciSyncObjforEOF(m_pDla, syncObj);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaSetNvSciSyncObjforEOF");

fail:
    return status;
}

NvMediaStatus Dla::InsertPreSciFences(NvSciSyncFence *EOFfence)
{
    NvMediaStatus status = NvMediaDlaInsertPreNvSciSyncFence(m_pDla, EOFfence);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaInsertPreNvSciSyncFence");

fail:
    return status;
}

NvMediaStatus Dla::GetEOFSciFences(
                NvSciSyncObj eofSyncObj,
                NvSciSyncFence *EOFfence)
{
    NvMediaStatus status = NvMediaDlaGetEOFNvSciSyncFence(m_pDla, eofSyncObj, EOFfence);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaDlaGetEOFNvSciSyncFence");

fail:
    return status;
}
