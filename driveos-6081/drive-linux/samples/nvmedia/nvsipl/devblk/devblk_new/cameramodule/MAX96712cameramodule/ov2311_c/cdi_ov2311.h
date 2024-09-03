/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2021-2022, OmniVision Technologies.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __CDI_OV2311_H__
#define __CDI_OV2311_H__

#include "devblk_cdi.h"

#define OV2311_NUM_HDR_EXP         1u
#define SIZE_FUSE_ID   16u

typedef enum {
    CDI_CONFIG_OV2311_SOFTWARE_RESET = 0,
    CDI_CONFIG_OV2311_ENABLE_STREAMING,
    CDI_CONFIG_OV2311_ENABLE_PG,
    CDI_CONFIG_OV2311_STREAM_1600x1300_RAW10_60FPS,
    CDI_CONFIG_OV2311_STREAM_1600x1300_RAW10_30FPS,
    CDI_CONFIG_OV2311_EMBLINE_TOP,
    CDI_CONFIG_OV2311_EMBLINE_BOTTOM,
    CDI_CONFIG_OV2311_FUSEID,
    CDI_CONFIG_OV2311_SETTINGINFO,
}ConfigSetsOV2311;

typedef enum {
    CDI_OV2311_REV_0 = 0,
    CDI_OV2311_REV_1 = 1,
    CDI_OV2311_INVALID_REV,
} RevisionOV2311;

typedef enum {
    CDI_WRITE_PARAM_CMD_EXPOSURE = 0,
    CDI_WRITE_PARAM_CMD_STROBE,
} WriteParametersCmdOV2311;

typedef enum {
    ASIL_STATUS_MAX_ERR = 0
} ErrGrpOV2311;

typedef struct {
    int32_t enumeratedDeviceConfig;
    float_t    frameRate;
} ConfigInfoOV2311;

typedef struct {
    float_t expo;
    float_t gain;
} ExpoGainOV2311;

typedef struct {
    union {
        ExpoGainOV2311 expogain;
        float_t strobe_ms;
    };
} WriteReadParametersParamOV2311;

typedef struct {
#if !NV_IS_SAFETY
    DevBlkCDIModuleConfig  moduleConfig;
#endif
    uint32_t    oscMHz;
    float_t     maxGain;
} ContextOV2311;

DevBlkCDIDeviceDriver *GetOV2311Driver(void);
NvMediaStatus GetOV2311ConfigSet(
    char *resolution,
    NvSiplCapInputFormatType inputFormat,
    int *configSet,
    uint32_t framerate);

NvMediaStatus
OV2311CheckPresence(
    DevBlkCDIDevice *handle);

NvMediaStatus
OV2311SetDefaults(
    DevBlkCDIDevice *handle);

NvMediaStatus
OV2311SetDeviceConfig(
    DevBlkCDIDevice *handle,
    uint32_t enumeratedDeviceConfig);

NvMediaStatus
OV2311ReadRegister(
    DevBlkCDIDevice *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff);

NvMediaStatus
OV2311WriteRegister(
    DevBlkCDIDevice *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff);

NvMediaStatus
OV2311ReadParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter);

NvMediaStatus
OV2311WriteParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter);

NvMediaStatus
OV2311DumpRegisters(
    DevBlkCDIDevice *handle);

NvMediaStatus
OV2311ReadErrorSize(
    size_t *errSize);

NvMediaStatus
OV2311ReadErrorData(
    DevBlkCDIDevice *handle,
    size_t bufSize,
    uint8_t * const buffer);

NvMediaStatus
OV2311SetDeviceValue(
    DevBlkCDIDevice *handle,
    uint32_t const valueToSet);

NvMediaStatus
OV2311GetDeviceValue(
    DevBlkCDIDevice *handle,
    uint32_t * const valueToGet);

#endif //__CDI_OV2311_H__