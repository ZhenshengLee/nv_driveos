/*
 * Copyright (c) 2021-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _NVMEDIA_SCIBUF_UTILS_H_
#define _NVMEDIA_SCIBUF_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "nvscibuf.h"
#include "nvmedia_core.h"


typedef enum {
    YUV400P_8bit,
    YUV400P_10bit,
    YUV400P_12bit,
    YUV400P_16bit,

    YUV420SP_8bit,
    YUV420SP_10bit,
    YUV420SP_12bit,
    YUV420SP_16bit,

    YUV422SP_8bit,
    YUV422SP_10bit,
    YUV422SP_12bit,
    YUV422SP_16bit,

    YUV444SP_8bit,
    YUV444SP_10bit,
    YUV444SP_12bit,
    YUV444SP_16bit,

    YUV420P_8bit,
    YUV420P_10bit,
    YUV420P_12bit,
    YUV420P_16bit,

    YUV422P_8bit,
    YUV422P_10bit,
    YUV422P_12bit,
    YUV422P_16bit,

    YUV444P_8bit,
    YUV444P_10bit,
    YUV444P_12bit,
    YUV444P_16bit,

    RGBA_8bit,
    ARGB_8bit,
    RG16,
    A16,
    A8,

    CHROMA_FORMAT_UNSUPPORTED,
} ChromaFormat;

typedef struct {
    NvSciBufAttrValImageLayoutType layout;
    uint32_t planeCount;
    uint32_t planeWidth[NV_SCI_BUF_IMAGE_MAX_PLANES];
    uint32_t planeHeight[NV_SCI_BUF_IMAGE_MAX_PLANES];
    NvSciBufAttrValColorFmt planeColorFormat[NV_SCI_BUF_IMAGE_MAX_PLANES];
    uint32_t planeBitsPerPixel[NV_SCI_BUF_IMAGE_MAX_PLANES];
    uint8_t planeChannelCount[NV_SCI_BUF_IMAGE_MAX_PLANES];
    uint32_t planePitchBytes[NV_SCI_BUF_IMAGE_MAX_PLANES];
    bool needCpuAccess;
} BufAttrValues;

NvMediaStatus
ClearSurface(
    uint32_t width,
    uint32_t height,
    NvSciBufObj bufObj,
    ChromaFormat DataFormat);


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
//   bufObj
//      (out) Pointer to pre-allocated output surface
//
//   inputFileChromaFormat
//      (in) Chroma format of the input YUV file being read
//
//   uvOrderFlag
//      (in) Flag for UV order. If true - UV; If false - VU;
//
//   pixelAlignment
//      (in) Alignment of bits in pixel.
//         0 - LSB Aligned
//         1 - MSB Aligned

NvMediaStatus
ReadInput(
    char *fileName,
    uint32_t frameNum,
    uint32_t width,
    uint32_t height,
    NvSciBufObj bufObj,
    ChromaFormat inputFileChromaFormat,
    bool uvOrderFlag,
    uint32_t pixelAlignment);

//  WriteOutput
//
//    WriteOutput()  Save RGB or YUV image to a file
//
//  Arguments:
//
//   filename
//      (in) Output file name
//
//   bufObj
//      (in) Pointer to the image
//
//   uvOrderFlag
//      (in) Flag for UV order. If true - UV; If false - VU;
//           Used only in YUV type surface case
//
//   appendFlag
//      (in) Apped to exisitng file if true otherwise create new file
//
//   srcRect
//      (in) structure containing co-ordinates of the rectangle in the source surface
//           from which the client surface is to be copied. Setting srcRect to NULL
//           implies rectangle of full surface size.
NvMediaStatus
WriteOutput(
    char *filename,
    NvSciBufObj bufObj,
    bool uvOrderFlag,
    bool appendFlag,
    NvMediaRect *srcRect);

NvMediaStatus
GetNvSciBufObjCrc(
    NvSciBufObj bufObj,
    NvMediaRect *srcRect,
    bool monochromeFlag,
    uint32_t *crcOut);

NvMediaStatus
GetNvSciBufObjCrcNoSrcRect(
    NvSciBufObj bufObj,
    bool monochromeFlag,
    uint32_t *crcOut);

NvMediaStatus
CheckSurfaceCrc(
    NvSciBufObj surf,
    NvMediaRect *srcRect,
    bool        monochromeFlag,
    uint32_t    ref,
    bool       *isMatching);

NvMediaStatus
PopulateNvSciBufAttrList (
    ChromaFormat                    chromaFormat,
    uint32_t                        width,
    uint32_t                        height,
    bool                            needCpuAccess,
    NvSciBufAttrValImageLayoutType  layout,
    uint32_t                        planeCount,
    NvSciBufAttrValAccessPerm       access_perm,
    uint32_t                        lumaBaseAddressAlign,
    NvSciBufAttrValColorStd         lumaColorStd,
    NvSciBufAttrValImageScanType    scanType,
    NvSciBufAttrList                bufAttributeList
);

NvMediaStatus
GetNvSciBufAttributes (
    NvSciBufAttrList bufAttrList,
    BufAttrValues *imgAttrValues);

#ifdef __cplusplus
}
#endif

#endif /* _NVMEDIA_SCIBUF_UTILS_H_ */
