/* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef IMAGE_SCIBUF_H
#define IMAGE_SCIBUF_H

#include "nvmedia_core.h"
#include "nvmedia_surface.h"
#include "nvmedia_image.h"

/*This function creates NvMedia Image using NvScibuf.
 * The application is expected to call NvMediaImageNvSciBufInit()
 * before using NvMediaImageCreateUsingNvScibuf API to create image and
 * NvMediaImageNvSciDeBufinit() after destroying the image.
 */

#ifdef NVMEDIA_NVSCI_ENABLE
NvMediaImage *
NvMediaImageCreateUsingNvScibuf(
    NvMediaDevice *device,
    NvMediaSurfaceType type,
    const NvMediaSurfAllocAttr *attrs,
    uint32_t numAttrs,
    uint32_t flags
);
#endif

#endif /* IMAGE_SCIBUF_H */
