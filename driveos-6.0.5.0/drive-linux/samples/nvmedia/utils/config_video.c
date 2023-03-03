/* Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include <unistd.h>
#include <string.h>

#include "nvmedia_core.h"
#include "nvmedia_surface.h"
#include "nvmedia_isc.h"
#include "log_utils.h"
#include "config_isc_adv7281.h"
#include "config_video.h"

static ConfigVideoInputSource
GetInputSourceType(
    char *inputSource
    )
{
    if(!strncasecmp(inputSource, "adv7281", 7)) {
        return ISC_ADV7281;
    } else {
        LOG_ERR("%s: Unknown ISC device: %s\n",
                __func__, inputSource);
        return ISC_UNKNOWN_INPUT_SOURCE;
    }
}

NvMediaStatus
ConfigVideoCreateDevices (
    ConfigVideoInfo         *videoConfigInfo,
    ConfigVideoDevices      *isc,
    NvMediaBool             reconfigureFlag,
    char                    *captureModuleName
    )
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (!videoConfigInfo)
        return NVMEDIA_STATUS_ERROR;
    if (!isc)
        return NVMEDIA_STATUS_ERROR;
    if (!captureModuleName)
        return NVMEDIA_STATUS_ERROR;

    videoConfigInfo->inputSourceType = GetInputSourceType(captureModuleName);
    // Config ISC Video Bridge Devices
    switch(videoConfigInfo->inputSourceType) {
        case ISC_ADV7281:
        default:
            status = ConfigISC_adv7281(videoConfigInfo, isc, reconfigureFlag);
            break;
    }
    if (status == NVMEDIA_STATUS_OK)
        isc->inputSourceType = videoConfigInfo->inputSourceType;

    return status;
}

void
ConfigVideoDestroyDevices (
    ConfigVideoInfo           *videoConfigInfo,
    ConfigVideoDevices        *isc
    )
{
    unsigned int i;

    switch(videoConfigInfo->inputSourceType) {
        case ISC_ADV7281:
            ConfigISC_adv7281_destroy(videoConfigInfo, isc);
            break;
        default:
            break;
    }

    for(i = 0; i < videoConfigInfo->bridgeSubDevicesNum; i++) {
        if(isc->iscVideoBridgeDevice[i])
            NvMediaISCDeviceDestroy(isc->iscVideoBridgeDevice[i]);
    }
    if(isc->iscRootDevice)
        NvMediaISCRootDeviceDestroy(isc->iscRootDevice);
}
