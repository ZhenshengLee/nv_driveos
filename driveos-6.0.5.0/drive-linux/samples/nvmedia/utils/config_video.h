/* Copyright (c) 2015-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __CONFIG_VIDEO_H__
#define __CONFIG_VIDEO_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "nvmedia_core.h"
#include "nvmedia_surface.h"
#include "nvmedia_isc.h"

#define MAX_BRIDGE_SUB_DEVICES 5

typedef enum {
    ISC_ADV7281 = 0,
    ISC_UNKNOWN_INPUT_SOURCE
} ConfigVideoInputSource;

typedef enum {
    ISC_VIDEO_STANDARD_NTSC = 0,
    ISC_VIDEO_STANDARD_PAL
} ConfigVideoStandard;


typedef struct {
    // ISC
    NvMediaISCRootDevice       *iscRootDevice;
    NvMediaISCDevice           *iscVideoBridgeDevice[MAX_BRIDGE_SUB_DEVICES];
    ConfigVideoInputSource      inputSourceType;
} ConfigVideoDevices;

typedef struct {
    uint32_t                    video_bridge_address[MAX_BRIDGE_SUB_DEVICES];
    uint32_t                    bridgeSubDevicesNum;
    NvMediaBool                 useColorBarMode;
    ConfigVideoStandard         videoStandard;
    ConfigVideoInputSource      inputSourceType;
    int                         i2cDevice;
//    char                        *board;
//    char                        *resolution;
    NvMediaICPInterfaceType     csi_link;
} ConfigVideoInfo;

NvMediaStatus
ConfigVideoCreateDevices (
    ConfigVideoInfo         *videoConfigInfo,
    ConfigVideoDevices      *isc,
    NvMediaBool             reconfigureFlag,
    char                    *captureModuleName
);

void
ConfigVideoDestroyDevices (
    ConfigVideoInfo           *videoConfigInfo,
    ConfigVideoDevices        *isc
);

#ifdef __cplusplus
}
#endif

#endif // __CONFIG_VIDEO_H__
