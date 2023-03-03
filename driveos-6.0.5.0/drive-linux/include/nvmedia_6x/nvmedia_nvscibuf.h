/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * \file
 * \brief <b> Utility functions for NvSciBufObj </b>
 *
 */

#ifndef NVMEDIA_NVSCIBUF_H
#define NVMEDIA_NVSCIBUF_H

#ifdef __cplusplus
extern "C" {
#endif

#include "nvscibuf.h"
#include "nvmedia_core.h"

NvMediaStatus
NvMediaNvSciBufObjPutBits(
    NvSciBufObj dstBufObj,
    void **srcPntrs,
    const uint32_t srcPlanePitches[]
);

NvMediaStatus
NvMediaNvSciBufObjGetBits(
    NvSciBufObj srcBufObj,
    void **dstPntrs,
    const uint32_t dstPlanePitches[]
);

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_NVSCIBUF_H */
