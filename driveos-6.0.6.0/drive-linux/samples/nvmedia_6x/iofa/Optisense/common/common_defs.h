/* Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef COMMON_DEFS_H
#define COMMON_DEFS_H

#include <string>

#define GET_EXTENSION(str) str.substr(str.find_last_of(".") + 1);

using namespace std;

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

    SF_RG16,
    SF_A16,
    SF_A8,

    SURFACE_FORMAT_UNSUPPORTED
} SurfaceFormat;

typedef enum
{
    YUV_400,
    YUV_420,
    YUV_422,
    YUV_444,
    RG16,
    A16,
    A8,
    NONE_CF
} ChromaFormat;


typedef enum
{
    SGMPARAM_P1_OVERRIDE         = (1<<0),
    SGMPARAM_P2_OVERRIDE         = (1<<1),
    SGMPARAM_DIAGONAL_OVERRIDE   = (1<<2),
    SGMPARAM_ADAPTIVEP2_OVERRIDE = (1<<3),
    SGMPARAM_NUMPASSES_OVERRIDE  = (1<<4),
    SGMPARAM_ALPHA_OVERRIDE      = (1<<5),
} SGMPARAM_OVERRIDE;

#endif
