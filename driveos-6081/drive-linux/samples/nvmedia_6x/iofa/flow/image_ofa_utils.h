/*
 * Copyright (c) 2021-2023 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _NVMEDIA_TEST_IMAGE_OFA_UTILS_H_
#define _NVMEDIA_TEST_IMAGE_OFA_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "log_utils.h"
#include "commandline.h"
#include "misc_utils.h"
#include "nvmedia_core.h"
#include "nvmedia_iofa.h"
#include "scibuf_utils.h"

//  ReadImage
//
//    ReadImage()  Read image from file
//
//  Arguments:
//
//   filename
//      (in) Input file name
//
//   frameNum
//      (in) Frame number to read. Use for stream input files.
//
//   width
//      (in) Surface width
//
//   height
//      (in) Surface height
//
//   image
//      (out) Pointer to pre-allocated output surface
//
//   uvOrderFlag
//      (in) Flag for UV order. If true - UV; If false - VU;
//
//   bytesPerPixel
//      (in) Bytes per pixel. Nedded for RAW image types handling.
//         RAW8 - 1 byte per pixel
//         RAW10, RAW12, RAW14 - 2 bytes per pixel
//
//   pixelAlignment
//      (in) Alignment of bits in pixel.
//         0 - LSB Aligned
//         1 - MSB Aligned

NvMediaStatus
ReadImageOFA (
    char            *fileName,
    uint32_t        frameNum,
    uint32_t        width,
    uint32_t        height,
    NvSciBufObj     image[NVMEDIA_IOFA_MAX_PYD_LEVEL],
    bool            uvOrderFlag,
    uint32_t        pixelAlignment,
    uint8_t         pyrLevel,
    uint32_t        bitdepth,
    ChromaFormat    Format);

#ifdef __cplusplus
}
#endif

#endif /* _NVMEDIA_TEST_IMAGE_OFA_UTILS_H_ */
