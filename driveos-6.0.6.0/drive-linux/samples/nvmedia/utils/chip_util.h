/* Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef _NVMEDIA_CHIP_UTIL_H_
#define _NVMEDIA_CHIP_UTIL_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TEGRA_UKNOWN,
    TEGRA_T18x = 0x18,
    TEGRA_T19x = 0x19,
} TegraChipId;

TegraChipId
GetTegraChipId(void);

#ifdef __cplusplus
}
#endif

#endif

