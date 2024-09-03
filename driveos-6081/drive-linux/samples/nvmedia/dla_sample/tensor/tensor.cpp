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

#include <cstring>
#include <memory>
#include "tensor.h"
#include "utils.h"

#define MODULE "DlaTensor"

Tensor::Tensor() :
    m_pTensor(nullptr)
{
}

Tensor::~Tensor()
{
    if (m_pTensor) {
        NvMediaTensorDestroy(m_pTensor);
    }
}

NvMediaStatus Tensor::FillNvSciBufTensorAttrs(
    NvMediaTensorAttr tensorAttrs[],
    uint32_t numAttrs,
    NvSciBufAttrList attr_h)
{
    return NvMediaTensorFillNvSciBufAttrs(NULL, tensorAttrs, numAttrs, 0, attr_h);
}

NvMediaStatus Tensor::Create(NvSciBufObj bufObj)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    status = NvMediaTensorCreateFromNvSciBuf(NULL,
                                             bufObj,
                                             &m_pTensor);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaTensorCreateFromNvSciBuf");

fail:
    return status;
}

NvMediaStatus Tensor::SetData(uint8_t value)
{
    NvMediaTensorSurfaceMap tensorMap {};
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = NvMediaTensorLock(m_pTensor, NVMEDIA_TENSOR_ACCESS_WRITE, &tensorMap);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaTensorLock");

    memset(tensorMap.mapping, value, tensorMap.size);
    NvMediaTensorUnlock(m_pTensor);

fail:
    return status;
}

// Fill tensor with data from buffer
NvMediaStatus Tensor::FillDataIntoTensor(uint32_t size, void *p)
{
    NvMediaTensorSurfaceMap tensorMap {};
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = NvMediaTensorLock(m_pTensor, NVMEDIA_TENSOR_ACCESS_WRITE, &tensorMap);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaTensorLock");

    CHECK_FAIL(tensorMap.size >= size, "Tensor size check");

    memcpy(tensorMap.mapping, p, size);
    NvMediaTensorUnlock(m_pTensor);

fail:
    return status;
}

NvMediaTensor* Tensor::GetPtr() const
{
    return m_pTensor;
}

NvMediaStatus Tensor::GetStatus()
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvMediaTensorTaskStatus taskStatus{};

    status = NvMediaTensorGetStatus(m_pTensor, NVMEDIA_TENSOR_TIMEOUT_INFINITE, &taskStatus);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaTensorGetStatus");

    LOG_INFO("Operation duration: %d us.\n", taskStatus.durationUs);
    if(taskStatus.status != NVMEDIA_STATUS_OK) {
        status = taskStatus.status;
        LOG_ERR("Engine returned error.\n");
        goto fail;
    }

fail:
    return status;
}

NvMediaStatus Tensor::CompareWithRef(uint32_t size, void *p)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvMediaTensorSurfaceMap tensorMap {};

    // Read output and check results.
    status = NvMediaTensorLock(m_pTensor, NVMEDIA_TENSOR_ACCESS_READ, &tensorMap);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaTensorLock");

    if(0 != memcmp(tensorMap.mapping, p, size)) {
        LOG_ERR("Output does not match expected\n");
        status = NVMEDIA_STATUS_ERROR;
        NvMediaTensorUnlock(m_pTensor);
        goto fail;
    } else {
        LOG_INFO("Compare with ref data: pass\n");
    }

    NvMediaTensorUnlock(m_pTensor);

fail:
    return status;
}
