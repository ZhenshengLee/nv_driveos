/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#if !NV_IS_SAFETY
#include "log_utils.h"
#endif

#include "sipl_util.h"
#include "devblk_cdi.h"
#include "cdi_ds90ub971_tpg.h"

static NvMediaStatus
DriverCreate(
    DevBlkCDIDevice *handle,
    void const* clientContext)
{
    return NVMEDIA_STATUS_OK;
}

static NvMediaStatus
DriverDestroy(
    DevBlkCDIDevice *handle)
{
    return NVMEDIA_STATUS_OK;
}

#if !NV_IS_SAFETY
static NvMediaStatus
GetModuleConfig(
    DevBlkCDIDevice *handle,
    DevBlkCDIModuleConfig *cameraModuleConfig)
{
    return NVMEDIA_STATUS_OK;
}
#endif

static DevBlkCDIDeviceDriver deviceDriver = {
    .deviceName = "TI DS90UB971 TPG Dummy Image Sensor",
    .regLength = 2u,
    .dataLength = 1u,
    .DriverCreate = DriverCreate,
    .DriverDestroy = DriverDestroy,
#if !NV_IS_SAFETY
    .GetModuleConfig = GetModuleConfig,
#endif
#if 0
    .GetSensorAttributes = GetSensorAttributes,
    .SetSensorControls = SetSensorControls,
    .ParseTopEmbDataInfo = NULL,
    .ParseBotEmbDataInfo = NULL,
#endif
};

DevBlkCDIDeviceDriver *
GetDS90UB971TPGDriver(void)
{
    return &deviceDriver;
}
