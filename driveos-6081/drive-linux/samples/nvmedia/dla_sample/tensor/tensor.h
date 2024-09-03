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

#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <string>

#include "nvmedia_core.h"
#include "nvmedia_tensor.h"
#include "nvmedia_tensor_nvscibuf.h"

//! Class for create and destroy NvMedia Tensor from NvSciBuf
class Tensor
{
public:
    static NvMediaStatus FillNvSciBufTensorAttrs(
                            NvMediaTensorAttr tensorAttrs[],
                            uint32_t numAttrs,
                            NvSciBufAttrList attr_h);

    Tensor();

    NvMediaStatus Create(NvSciBufObj bufObj);

    // Fill tensor with single value
    NvMediaStatus SetData(uint8_t value);

    //! Fill tensor with data from buffer
    NvMediaStatus FillDataIntoTensor(uint32_t size, void *p);

    // Fill tensor with data from pgm image file
    virtual NvMediaStatus FillDataIntoTensor(std::string pgmImageFileName) {
        return NVMEDIA_STATUS_NOT_SUPPORTED;
    }

    NvMediaTensor *GetPtr() const;

    NvMediaStatus GetStatus();

    NvMediaStatus CompareWithRef(uint32_t size, void *p);

    virtual ~Tensor();

protected:
    NvMediaTensor *m_pTensor;
};

#endif // end of _TENSOR_H_
