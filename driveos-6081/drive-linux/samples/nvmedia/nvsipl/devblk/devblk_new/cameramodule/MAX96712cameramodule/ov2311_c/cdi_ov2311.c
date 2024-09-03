/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2021-2022, OmniVision Technologies.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <errno.h>

#if !NV_IS_SAFETY
 #include "log_utils.h"
#endif
#include "sipl_util.h"
#include "devblk_cdi.h"
#include "devblk_cdi_i2c.h"
#include "cdi_ov2311.h"
#include "cdi_ov2311_setting.h"
#include "cdi_ov2311_regmap.h"
#include "os_common.h"

/********************************************************************************/
// OV2311 (OV2311 RGGB version) Information
/********************************************************************************/
/* OV2311 Register Address: 16bit */
#define REG_ADDRESS_BYTES     2
/* OV2311 Register Value: 8bit */
#define REG_DATA_BYTES        1

#define REG_WRITE_BUFFER      256

#define OV2311_CHIP_ID        0x2311u

/* OV2311 HDR: HCG,LCG,S,VS */
#define OV2311_MAXN_HDR_EXP     4u

#define OV2311_AGAIN_MAX       15.9375
#define OV2311_AGAIN_MIN       1.0
#define OV2311_DGAIN_MAX       2.0
#define OV2311_DGAIN_MIN       1.0
#define OV2311_WBGAIN_MAX      1.0
#define OV2311_WBGAIN_MIN      1.0

/* OV2311 Register value - 1x gain */
#define OV2311_REGV_AGAIN_1X         0x100
#define OV2311_REGV_DGAIN_1X         0x10000
#define OV2311_REGV_CGAIN_1X         0x1000
#define OV2311_REGV_WBGAIN_1x        0x400

#define OV2311_AGAIN_SCALES      128

/* Usethe fixed exposure time */
#define OV2311_FIXED_EXPOSURE_TIME 1U

typedef struct {
    RevisionOV2311 revId;
    uint32_t revMajorVal;
    uint32_t revMinorVal;
} Revision;

/* These values must include all of values in the RevisionOV2311 enum */
static Revision supportedRevisions[] = {
    { CDI_OV2311_REV_0, 0xa0, 0x00 },
};

typedef struct {
    uint8_t fuseId[SIZE_FUSE_ID];
    float   sensitivity[OV2311_MAXN_HDR_EXP];
} OtpInfo;

typedef struct {
    DevBlkCDII2CPgmr i2cProgrammer;

    RevisionOV2311 revision;
    OtpInfo        otp;
    CDIRegSetOV2311 regsel_table[OV2311_EMB_DATA_NUM_REGISTERS];

    const DevBlkCDII2CRegList   *default_setting;
    uint8_t     numActiveExposures;
    uint32_t    tclk; /* timing clock, Hz */

    uint32_t    vts;  /* VTS */
    uint32_t    hts;  /* HTS */

    uint32_t    Tstrobe;
#if !NV_IS_SAFETY
    DevBlkCDIModuleConfig moduleCfg;
#endif

    ConfigInfoOV2311 configinfo;

    NvMediaBool charModeEnabled;
    uint8_t     charModeExpNo;

    uint8_t     numFrameReportBytes;

    NvMediaBool initDone;

    DevBlkCDISensorControl exposureControl_prev;
    NvMediaBool updateSensor;
} _DriverHandle;

uint32_t ov2311_custom_value = 0;

static uint32_t colorIndex[4] = {
    3u, /* B - index 1; index order: R, Gr, Gb, B */
    2u, /* Gb  - index 0 */
    1u, /* Gr  - index 3 */
    0u  /* R - index 2 */
};

static NvMediaStatus
SetExposure(
    DevBlkCDIDevice const* handle,
    const DevBlkCDIExposure *exposureControl);

static NvMediaStatus
SetSensorWbGain(
    DevBlkCDIDevice const* handle,
    const DevBlkCDIWhiteBalance *wbControl);

static NvMediaStatus
SetSensorFrameReport(
    DevBlkCDIDevice const* handle,
    const DevBlkCDIFrameReport *frameReport);

/********************************************************************************/
// Export Functions
/********************************************************************************/
static NvMediaStatus
GetTimingInfo(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle=NULL;

    uint16_t  hts;
    uint16_t  vts;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to GetTimingInfo");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to GetTimingInfo");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    /* VTS */
    status = DevBlkCDII2CPgmrReadUint16(drvHandle->i2cProgrammer,
                                        REG_VTS,
                                        &vts);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("OX03C10: VTS Registers read failed", (int32_t)status);
        goto done;
    }

    /* HTS */
    status = DevBlkCDII2CPgmrReadUint16(drvHandle->i2cProgrammer,
                                        REG_HTS,
                                        &hts);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("OV2311: HTS Registers read failed", (int32_t)status);
        goto done;
    }

    drvHandle->hts = hts;
    drvHandle->vts = vts;

done:
    return status;
}

static NvMediaStatus
GetFuseId (
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    uint8_t readBuff[SIZE_FUSE_ID] = {0u};

    if ((handle == NULL) || ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = DevBlkCDII2CPgmrReadBlock(drvHandle->i2cProgrammer,
                                REG_FUSE_ID,
                                SIZE_FUSE_ID,
                                readBuff);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("OV2311: Fuse ID Registers read failed", (int32_t)status);
        return status;
    }

    /* Save Fuse ID */
    (void)memcpy(drvHandle->otp.fuseId, readBuff, SIZE_FUSE_ID);

    return status;
}

NvMediaStatus
OV2311CheckPresence(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    RevisionOV2311 rev = CDI_OV2311_INVALID_REV;
    uint16_t chip_id;
    uint8_t  revision_id;
    uint32_t i = 0, numRev = sizeof(supportedRevisions) / sizeof(supportedRevisions[0]);

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to OV2311CheckPresence");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to OV2311CheckPresence");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    /* Chip ID */
    status = DevBlkCDII2CPgmrReadUint16(drvHandle->i2cProgrammer,
                                        REG_CHIP_ID,
                                        &chip_id);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("OV2311: Chip ID Register read failed", (int32_t)status);
        goto done;
    }
    if (chip_id != OV2311_CHIP_ID) {
        LOG_MSG("OV2311: chip id", chip_id);
        SIPL_LOG_ERR_STR_INT("OV2311: Chip ID Mismatch", (int32_t)status);
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    /* Revision ID */
    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                        REG_CHIP_ID + 2,
                                        &revision_id);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("OV2311: Revision ID Register read failed", (int32_t)status);
        goto done;
    }

    for (i = 0; i < numRev; i++) {
        if (revision_id == supportedRevisions[i].revMajorVal) {
            LOG_MSG("OV2311: revision 0x%x\n", revision_id);
            rev = supportedRevisions[i].revId;
            status = NVMEDIA_STATUS_OK;
            goto done;
        }
    }

    status = NVMEDIA_STATUS_NOT_SUPPORTED;
    rev = CDI_OV2311_INVALID_REV;
    LOG_MSG("OV2311: revision id 0x%x\n", revision_id);
    SIPL_LOG_ERR_STR_INT("OV2311: Revision not suppported ", (int32_t)status);

done:
    drvHandle->revision = rev;
    return status;
}

NvMediaStatus
OV2311SetDefaults(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to OV2311SetDefaults");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to OV2311SetDefaults");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,  drvHandle->default_setting);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("OV2311: I2C write for set defaults failed", (int32_t)status);
        goto done;
    }

done:
    return status;
}

NvMediaStatus
OV2311SetDeviceConfig(
    DevBlkCDIDevice *handle,
    uint32_t enumeratedDeviceConfig)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to OV2311SetDeviceConfig");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to OV2311SetDeviceConfig");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    switch (enumeratedDeviceConfig) {
        case CDI_CONFIG_OV2311_STREAM_1600x1300_RAW10_60FPS:
            drvHandle->configinfo.frameRate = 60.0f;
            break;

        case CDI_CONFIG_OV2311_STREAM_1600x1300_RAW10_30FPS:
            drvHandle->configinfo.frameRate = 30.0f;
            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                                &ov2311_reglist_config_30fps);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("OV2311: Failed to set the framerate", (int32_t)status);
                goto done;
            }
            break;

        case CDI_CONFIG_OV2311_SOFTWARE_RESET:
            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                                &ov2311_reglist_software_reset);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("OV2311: Failed to software reset", (int32_t)status);
                goto done;
            }
            break;
        case CDI_CONFIG_OV2311_ENABLE_STREAMING:
            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                                &ov2311_reglist_stream_on);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("OV2311: Failed to enable streaming", (int32_t)status);
                goto done;
            }
            break;
        case CDI_CONFIG_OV2311_ENABLE_PG:
            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                                &ov2311_reglist_config_testpattern);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("OV2311: Failed to set test pattern", (int32_t)status);
                goto done;
            }
            break;
        case CDI_CONFIG_OV2311_EMBLINE_TOP:
            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                                &ov2311_reglist_emb_top);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("OV2311: Failed to set top embedded line", (int32_t)status);
                goto done;
            }
            break;
        case CDI_CONFIG_OV2311_EMBLINE_BOTTOM:
            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                                &ov2311_reglist_emb_bottom);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("OV2311: Failed to set bottom embedded line", (int32_t)status);
                goto done;
            }
            break;
        case CDI_CONFIG_OV2311_FUSEID:
            status = GetFuseId(handle);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("OV2311: Failed to get fuse id", (int32_t)status);
                goto done;
            }
            break;
        case CDI_CONFIG_OV2311_SETTINGINFO:
            status = GetTimingInfo(handle);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("OV2311: Failed to get setting info", (int32_t)status);
                goto done;
            }
            break;
        default:
            SIPL_LOG_ERR_STR_INT("OV2311: Unsupported device config", (int32_t)status);
            status = NVMEDIA_STATUS_NOT_SUPPORTED;
            break;
    }

done:
    return status;
}

NvMediaStatus
OV2311ReadParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    WriteReadParametersParamOV2311 *param;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to OV2311ReadParameters");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (parameter == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null output parameter passed to OV2311ReadParameters");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    param = (WriteReadParametersParamOV2311 *)parameter;
    if (param == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Output parameter structure contained null pointer in OV2311ReadParameters");
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    /* Todo: ... */

done:
    return status;
}

NvMediaStatus
OV2311DumpRegisters(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

#if !NV_IS_SAFETY
    _DriverHandle *drvHandle = NULL;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to OV2311DumpRegisters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to OV2311DumpRegisters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }
    status = DevBlkCDII2CPgmrDumpArray(drvHandle->i2cProgrammer, drvHandle->default_setting);
#endif

    return status;
}

NvMediaStatus
GetOV2311ConfigSet(
    char *resolution,
    NvSiplCapInputFormatType inputFormat,
    int *configSet,
    uint32_t framerate)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    /* set input mode setting for OV2311 */
    if ((inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10) &&
        (resolution != NULL) && (strcasecmp(resolution, "1600x1300") == 0)) {
        if (framerate == 60)
            *configSet = CDI_CONFIG_OV2311_STREAM_1600x1300_RAW10_60FPS;
        else if (framerate == 30)
            *configSet = CDI_CONFIG_OV2311_STREAM_1600x1300_RAW10_30FPS;
        else {
            SIPL_LOG_ERR_STR_INT("OV2311: Not supported framerate", (int32_t)framerate);
            status = NVMEDIA_STATUS_ERROR;
            goto done;
        }
    }
    else {
        SIPL_LOG_ERR_STR_INT("OV2311: Not supported %s ", (int32_t)inputFormat);
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

done:
    return status;
}

/********************************************************************************/
// Embedded Data
/********************************************************************************/
static NvMediaStatus
ParseWBGain(
    _DriverHandle *drvHandle,
    DevBlkCDIWhiteBalance *parsedWbInfo)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint16_t gain[DEVBLK_CDI_MAX_COLOR_COMPONENT];
    uint16_t i;
    float_t value = 0u;

    if ((drvHandle == NULL)) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to ParseWBGain");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (parsedWbInfo == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null white balance info passed to ParseWBGain");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    (void)memset(parsedWbInfo, 0u, sizeof(DevBlkCDIWhiteBalance));

    gain[0] = (drvHandle->regsel_table[REG_AWB_GAIN_0_OV2311].data<<8) //B
                +drvHandle->regsel_table[REG_AWB_GAIN_1_OV2311].data;
    gain[1] = (drvHandle->regsel_table[REG_AWB_GAIN_2_OV2311].data<<8) //G
                +drvHandle->regsel_table[REG_AWB_GAIN_3_OV2311].data;
    gain[2] = gain[1];
    gain[3] = (drvHandle->regsel_table[REG_AWB_GAIN_4_OV2311].data<<8) //R
                +drvHandle->regsel_table[REG_AWB_GAIN_5_OV2311].data;
    for (i = 0u; i < DEVBLK_CDI_MAX_COLOR_COMPONENT; i++) {  /* Gr, B, R, Gb */
        value = (float_t)(gain[i]) / (float_t)OV2311_REGV_WBGAIN_1x;
        parsedWbInfo->wbGain[0].value[colorIndex[i]] = value;
    }
    parsedWbInfo->wbValid = NVMEDIA_TRUE;

done:
    return status;
}

static NvMediaStatus
ParseExposure(
    _DriverHandle *drvHandle,
    DevBlkCDIExposure *sensorExpInfo)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint32_t expRegv,againRegv,dgainRegv;
    float_t exposureTime,analogGain, digitalGain;

    if ((drvHandle == NULL)) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to ParseExposure");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (sensorExpInfo == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null exposure info passed to ParseExposure");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    memset(sensorExpInfo, 0, sizeof(DevBlkCDIExposure));
    /***
     * Exposure Register value
     */
    expRegv = (drvHandle->regsel_table[REG_EXPOSURE_B1_OV2311].data<<8u)+
                 drvHandle->regsel_table[REG_EXPOSURE_B0_OV2311].data;

    /***
     * Gain Register Value
     */
    againRegv = (drvHandle->regsel_table[REG_AGAIN_B1_OV2311 ].data<<8u)+
                    drvHandle->regsel_table[REG_AGAIN_B0_OV2311 ].data;
    dgainRegv = (drvHandle->regsel_table[REG_DGAIN_B2_OV2311 ].data<<16u)+
                    (drvHandle->regsel_table[REG_DGAIN_B1_OV2311].data<<8u)+
                    drvHandle->regsel_table[REG_DGAIN_B0_OV2311 ].data;

    /***
     * Exposure Time and Real gain
     */
    exposureTime= (float)(expRegv * drvHandle->hts)/(float)drvHandle->tclk;
    analogGain  = (float)againRegv/(float)OV2311_REGV_AGAIN_1X;
    digitalGain = (float)dgainRegv/(float)OV2311_REGV_DGAIN_1X;

    sensorExpInfo->expTimeValid = NVMEDIA_TRUE;
    sensorExpInfo->gainValid = NVMEDIA_TRUE;

    sensorExpInfo->sensorGain[0] = analogGain*digitalGain;
    sensorExpInfo->exposureTime[0] = exposureTime;
done:
    return status;
}

static NvMediaStatus
ParsePWL(
    _DriverHandle *drvHandle,
    DevBlkCDIPWL *sensorPWLInfo)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to ParsePWL");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }
    if (sensorPWLInfo == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null sensor PWL info passed to ParsePWL");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    sensorPWLInfo->pwlValid = NVMEDIA_FALSE;
done:
    return status;
}

static NvMediaStatus
DepackEmbeddedLine(
    CDIRegSetOV2311* regset,
    uint32_t lineLength,
    const uint8_t *lineData,
    uint16_t selLength)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    const uint8_t* praw = lineData;
    uint16_t i, j;
    uint8_t tag = 0u;
    uint8_t dat = 0u;

     if (regset == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Regsiter set was null in DepackEmbeddedLine");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (lineData == NULL ) {
        SIPL_LOG_ERR_STR("OV2311: line data was null in DepackEmbeddedLine");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (lineLength == 0) {
        SIPL_LOG_ERR_STR("OV2311: line length was 0 in DepackEmbeddedLine");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (selLength == 0) {
        SIPL_LOG_ERR_STR("OV2311: sel length was null in DepackEmbeddedLine");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    j = 0;
    for (i = 0u; i < lineLength / 12u; i++) {
        //Fixed LE endian
        tag = praw[9];
        dat = praw[11];
        if ((j<selLength) && (tag==0x00)) {
            regset[j].data = dat;
            j++;
        }
        praw+=12;
        if (j>=selLength) {
            goto done;
        }
    }

    if (j<selLength) {
        SIPL_LOG_ERR_STR("OV2311: selLength doesn't match lineLength");
        status = NVMEDIA_STATUS_ERROR;
    }

done:
    return status;
}

static NvMediaStatus
UpdateSensor(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    if ((handle == NULL)) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to SetSensorControls");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle*)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to SetSensorControls");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (drvHandle->exposureControl_prev.numSensorContexts == 1) {
        /* exposure control calculation */
        if ((drvHandle->exposureControl_prev.exposureControl[0].expTimeValid == NVMEDIA_TRUE) ||
            (drvHandle->exposureControl_prev.exposureControl[0].gainValid == NVMEDIA_TRUE) ||
            (drvHandle->exposureControl_prev.wbControl[0].wbValid == NVMEDIA_TRUE) ||
            (drvHandle->exposureControl_prev.frameReportControl.frameReportValid == NVMEDIA_TRUE)) {
            if (drvHandle->initDone == NVMEDIA_FALSE) {
                drvHandle->initDone = NVMEDIA_TRUE;
            }

            /* group hold start */
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer, REG_GROUP_HOLD, 0x00);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("OV2311: Failed to write group hold start register", (int32_t)status);
                goto done;
            }

            /* exposure control calculation */
            if ((drvHandle->exposureControl_prev.exposureControl[0].expTimeValid == NVMEDIA_TRUE) ||
                (drvHandle->exposureControl_prev.exposureControl[0].gainValid == NVMEDIA_TRUE)) {
                status = SetExposure(handle, &drvHandle->exposureControl_prev.exposureControl[0]);
                if (status != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR_INT("OV2311: Failed to Set Exposure", (int32_t)status);
                    goto done;
                }
            }

            /* wb control calculation */
            if (drvHandle->exposureControl_prev.wbControl[0].wbValid == NVMEDIA_TRUE) {
                status = SetSensorWbGain(handle, &drvHandle->exposureControl_prev.wbControl[0]);
                if (status != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR_INT("OV2311: Failed to Set Sensor White Balance Gain", (int32_t)status);
                    goto done;
                }
            }

            /* frame report control calculation */
            if (drvHandle->exposureControl_prev.frameReportControl.frameReportValid == NVMEDIA_TRUE) {
                status = SetSensorFrameReport(handle, &drvHandle->exposureControl_prev.frameReportControl);
                if (status != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR_INT("OV2311: Failed to Set Sensor Frame Report", (int32_t)status);
                    goto done;
                }
            }

            /* group hold end */
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer, REG_GROUP_HOLD, 0x10);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("OV2311: Failed to write group hold end register", (int32_t)status);
                goto done;
            }

            /* group hold launch */
            //launch mode 2,v-blanking.
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer, REG_GROUP_HOLD, 0xa0);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("OV2311: Failed to write group hold launch register", (int32_t)status);
                goto done;
            }

            memset(&drvHandle->exposureControl_prev, 0U, sizeof(DevBlkCDISensorControl));
        }
    }
done:
    return status;
}

static NvMediaStatus
ParseTopEmbDataInfo(
    DevBlkCDIDevice const* handle,
    const struct DevBlkCDIEmbeddedDataChunk *embeddedTopDataChunk,
    const size_t dataChunkStructSize,
    struct DevBlkCDIEmbeddedDataInfo *parsedInfo,
    const size_t parsedInfoStructSize)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to ParseTopEmbDataInfo");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to ParseTopEmbDataInfo");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (embeddedTopDataChunk == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null top data chunk passed to ParseTopEmbDataInfo");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (parsedInfo == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to ParseTopEmbDataInfo");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (dataChunkStructSize == 0) {
        SIPL_LOG_ERR_STR("OV2311: 0 data chunk size passed to ParseTopEmbDataInfo");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (parsedInfoStructSize == 0) {
        SIPL_LOG_ERR_STR("OV2311: Info struct size was 0 in ParseTopEmbDataInfo");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (dataChunkStructSize != sizeof(DevBlkCDIEmbeddedDataChunk)) {
        SIPL_LOG_ERR_STR("OV2311: Data chunk size version mismatch. Please re-compile the client application.");
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    if (parsedInfoStructSize != sizeof(DevBlkCDIEmbeddedDataInfo)) {
        SIPL_LOG_ERR_STR("OV2311: Parsed info size version mismatch. Please re-compile the client application.");
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    if (drvHandle->updateSensor) {
        /* Silicon revision 0x1C has an issue that the embedded data is corrupted
         * if SW programs the sensor with the group hold while the sensor outputs the embedded data
         * To avoid this possibility, programs the sensor after getting the top embedded data lines
         * in ParseTopEmbDataInfo()
         */
        status = UpdateSensor(handle);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("OV2311: UpdateSensor failed", (int32_t)status);
            goto done;
        }
    }

    /* Reset register table */
    memcpy(drvHandle->regsel_table, regsel_ov2311, sizeof(regsel_ov2311));

    /* decoding embedded line & filling register set */
    if (embeddedTopDataChunk->lineLength != 0 && embeddedTopDataChunk->lineData != NULL) {
        (void)memset(parsedInfo, 0u, sizeof(DevBlkCDIEmbeddedDataInfo));
        status = DepackEmbeddedLine (drvHandle->regsel_table,
                                        embeddedTopDataChunk->lineLength,
                                        embeddedTopDataChunk->lineData,
                                        OV2311_EMB_DATA_NUM_REGISTERS);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("OV2311: DepackEmbeddedLine failed", (int32_t)status);
            goto done;
        }
    } else {
        SIPL_LOG_ERR_STR("OV2311: Invalid Top Embedded data");
        status = NVMEDIA_STATUS_NOT_SUPPORTED;
        goto done;
    }

    /* frame number of exposures info */
    parsedInfo->numExposures = drvHandle->numActiveExposures;

    /* sensor exposure info */
    status = ParseExposure(drvHandle, &parsedInfo->sensorExpInfo);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("OV2311: ParseExposure failed", (int32_t)status);
        goto done;
    }

    /* sensor white balance info */
    status = ParseWBGain(drvHandle, &parsedInfo->sensorWBInfo);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("OV2311: ParseWBGain failed", (int32_t)status);
        goto done;
    }

    /* sensor PWL info */
    status = ParsePWL(drvHandle, &parsedInfo->sensorPWLInfo);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("OV2311: ParsePWL failed", (int32_t)status);
        goto done;
    }

    /* sensor temperature info */
    parsedInfo->sensorTempInfo.tempValid = NVMEDIA_TRUE;
    parsedInfo->sensorTempInfo.numTemperatures = 1; /* top and bottom */
    parsedInfo->sensorTempInfo.sensorTempCelsius[0] =
        (drvHandle->regsel_table[REG_TPM_INT_RDOUT_OV2311].data) +
        (drvHandle->regsel_table[REG_TPM_DEC_RDOUT_OV2311].data/256.0);

    /* sensor frame sequence number */
    parsedInfo->frameSeqNumInfo.frameSequenceNumber =
        (drvHandle->regsel_table[REG_FRAME_CNT_B3_OV2311].data << 24u) |
        (drvHandle->regsel_table[REG_FRAME_CNT_B2_OV2311].data << 16u) |
        (drvHandle->regsel_table[REG_FRAME_CNT_B1_OV2311].data << 8u) |
        (drvHandle->regsel_table[REG_FRAME_CNT_B0_OV2311].data << 0u) ;
    parsedInfo->frameSeqNumInfo.frameSeqNumValid = NVMEDIA_TRUE;

    /*error information */
    parsedInfo->errorFlag = 0U; // no error info but this would be how it is indicated
done:
    return status;
}

static NvMediaStatus
ParseBotEmbDataInfo(
    DevBlkCDIDevice const* handle,
    const struct DevBlkCDIEmbeddedDataChunk *embeddedBotDataChunk,
    const size_t dataChunkStructSize,
    struct DevBlkCDIEmbeddedDataInfo *parsedInfo,
    const size_t parsedInfoStructSize)
{
    (void)handle;
    (void)embeddedBotDataChunk;
    (void)dataChunkStructSize;
    (void)parsedInfo;
    (void)parsedInfoStructSize;

    return NVMEDIA_STATUS_OK;
}

/********************************************************************************/
// Sensor Control
/********************************************************************************/
static NvMediaStatus
Exposure_Time2Regv(
    DevBlkCDIDevice const* handle,
    const DevBlkCDIExposure *exposureControl,
    uint32_t *exposureVal)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle;

    /*
     * input parameter validation check
     */
    if (handle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to SetExposure");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to SetExposure");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (exposureControl == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null exposure control passed to SetExposure");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (drvHandle->tclk == 0) {
        SIPL_LOG_ERR_STR("OV2311: Exposure control must have a valid gain or exposure time");
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    /* Exposure time to regv */
    exposureVal[0] = (exposureControl->exposureTime[0] *
                     drvHandle->tclk + drvHandle->hts / 2U) / drvHandle->hts;
    if (exposureVal[0]<2) {
        exposureVal[0] = 0x01U;
    }

done:
    return status;
}

static NvMediaStatus
Gain_Amp2AgainDgain(
    float realgain,
    float *again,
    float *dgain)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    float base[] = {1.0f, 2.0f, 4.0f, 8.0f, 16.0f};
    uint32_t n,i;

    if (realgain < OV2311_AGAIN_MIN)
    {
        again[0] = OV2311_AGAIN_MIN;
        dgain[0] = OV2311_DGAIN_MIN;
        goto done;
    }

    if (realgain > (OV2311_AGAIN_MAX*OV2311_DGAIN_MAX))
    {
        again[0] = OV2311_AGAIN_MAX;
        dgain[0] = OV2311_DGAIN_MAX;
        goto done;
    }

    /* Coarse Step */
    for (n = 0U; n < 3U; n++) {
        if (realgain < (base[n + 1U] * (1.0f + 1.0f / OV2311_AGAIN_SCALES))) {
            break;
        }
    }

    /* Fine Step */
    for (i = 0U; i < (OV2311_AGAIN_SCALES - 1U); i++) {
        if (realgain <= (base[n] * (1.0f + 1.0f * i / OV2311_AGAIN_SCALES))) {
            break;
        }
    }

    again[0] = base[n] * (1.0f + 1.0f * i / OV2311_AGAIN_SCALES);
    dgain[0] = realgain / again[0];
done:
    return status;
}

static NvMediaStatus
Gain_Amp2Regv(
    DevBlkCDIDevice const* handle,
    const DevBlkCDIExposure *exposureControl,
    uint32_t *aGainVal,
    uint32_t *dGainVal)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    float again, dgain;

    /*
     * input parameter validation check
     */
    if (handle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to SetExposure");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to SetExposure");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (exposureControl == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null exposure control passed to SetExposure");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (drvHandle->tclk == 0) {
        SIPL_LOG_ERR_STR("OV2311: Exposure control must have a valid gain or exposure time");
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    /*
     * Gain to again and dgain
     */
    Gain_Amp2AgainDgain(exposureControl->sensorGain[0], &again, &dgain);

    aGainVal[0] = (uint32_t)(again * (float)OV2311_REGV_AGAIN_1X);
    aGainVal[0] &= 0x1ff0u;
    dGainVal[0] = (uint32_t)(dgain * (float)OV2311_REGV_DGAIN_1X);
    dGainVal[0] &= 0x0fffc0u;

done:
    return status;
}

static NvMediaStatus
SetExposure(
    DevBlkCDIDevice const* handle,
    const DevBlkCDIExposure *exposureControl)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    const DevBlkCDIExposure *exposureCtrlSettings;

    uint32_t exposureRegv = 0u;
    uint32_t dGainRegv = 0u;
    uint32_t aGainRegv = 0u;

    uint8_t  regv8;
    uint16_t regv16;

    /*
     * input parameter validation check
     */
    if (handle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to SetExposure");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to SetExposure");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (exposureControl == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null exposure control passed to SetExposure");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if ((exposureControl->gainValid == NVMEDIA_FALSE) &&
        (exposureControl->expTimeValid == NVMEDIA_FALSE)) {
        SIPL_LOG_ERR_STR("OV2311: Exposure control must have a valid gain or exposure time");
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    exposureCtrlSettings = exposureControl;
    if (drvHandle->initDone == NVMEDIA_FALSE) {
        if (NVMEDIA_TRUE == exposureCtrlSettings->expTimeValid) {
            status = Exposure_Time2Regv(handle, exposureCtrlSettings, &exposureRegv);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("OV2311: Exposure Time to Regv Failed", (int32_t)status);
                goto done;
            }

            /* Write Exposure Registers */
            regv16 = exposureRegv & 0xFFFFu;
            status = DevBlkCDII2CPgmrWriteUint16(drvHandle->i2cProgrammer, REG_EXPOSURE, regv16);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("OV2311: Failed to write exposure register", (int32_t)status);
                goto done;
            }

            /* Write Strobe Registers */
            if (drvHandle->Tstrobe == 0) {
                regv16 = exposureRegv & 0xFFFFu;
                //LOG_MSG("\n...Exposure Strobe Width: 0x%x\n", regv16);
            } else {
                regv16 = drvHandle->Tstrobe & 0xFFFFu;
                //LOG_MSG("\n...Fixed Strobe Width: exp - 0x%x strobe - 0x%x\n", exposureRegv, regv16);
            }
            status = DevBlkCDII2CPgmrWriteUint16(drvHandle->i2cProgrammer, REG_STROBE_WIDTH, regv16);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("OV2311: Failed to write strobe register", (int32_t)status);
                goto done;
            }

            if (drvHandle->vts - 7U > exposureRegv) {
                exposureRegv = drvHandle->vts - 7U - exposureRegv;
            } else {
                exposureRegv = 0U;
            }
            regv16 = exposureRegv & 0xFFFFU;
            status = DevBlkCDII2CPgmrWriteUint16(drvHandle->i2cProgrammer, REG_STROBE_ST, regv16);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("OV2311: Failed to write strobe register", (int32_t)status);
                goto done;
            }
            //LOG_MSG("\n...Exposure Strobe Start: 0x%x 0x%x\n", drvHandle->vts, regv16);
        }
    }

    if (NVMEDIA_TRUE == exposureCtrlSettings->gainValid) {
        status = Gain_Amp2Regv(handle, exposureCtrlSettings, &aGainRegv, &dGainRegv);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("OV2311: Gain to Regv Failed", (int32_t)status);
            goto done;
        }

        /* Write Real Gain Registers */
        regv16 = aGainRegv & 0xFFFFu;
        status = DevBlkCDII2CPgmrWriteUint16(drvHandle->i2cProgrammer, REG_AGAIN, regv16);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("OV2311: Failed to write real gain register", (int32_t)status);
            goto done;
        }

        /* Write Digital Gain Registers */
        regv8 = (uint8_t)((dGainRegv>>16) & 0xFFu);
        status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer, REG_DGAIN, regv8);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("OV2311: Failed to write coarse digital gain register", (int32_t)status);
            goto done;
        }

        regv16 = (uint16_t)(dGainRegv & 0xFFFFu);
        status = DevBlkCDII2CPgmrWriteUint16(drvHandle->i2cProgrammer, REG_DGAIN+1, regv16);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("OV2311: Failed to write fine digital gain register", (int32_t)status);
            goto done;
        }
    }
done:
    return status;
}

static NvMediaStatus
SetStrobe(
    DevBlkCDIDevice *handle,
    const DevBlkCDIExposure *exposureControl)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    const DevBlkCDIExposure *exposureCtrlSettings;

    uint32_t exposureRegv = 0u;

    /*
     * input parameter validation check
     */
    if (handle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to SetExposure");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to SetExposure");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (exposureControl == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null exposure control passed to SetExposure");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if ((exposureControl->gainValid == NVMEDIA_FALSE) &&
        (exposureControl->expTimeValid == NVMEDIA_FALSE)) {
        SIPL_LOG_ERR_STR("OV2311: Exposure control must have a valid gain or exposure time");
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    exposureCtrlSettings = exposureControl;

    if (NVMEDIA_TRUE == exposureCtrlSettings->expTimeValid) {
        status = Exposure_Time2Regv(handle, exposureCtrlSettings, &exposureRegv);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("OV2311: Failed to set strobe width", (int32_t)status);
            goto done;
        }
        drvHandle->Tstrobe = exposureRegv;
    }

done:
    return status;
}

static NvMediaStatus
SetSensorWbGain(
    DevBlkCDIDevice const* handle,
    const DevBlkCDIWhiteBalance *wbControl)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (wbControl == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null white balance control passed to SetSensorWbGain");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to SetSensorWbGain");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

//done:
    return status;
}

static NvMediaStatus
SetSensorFrameReport(
    DevBlkCDIDevice const* handle,
    const DevBlkCDIFrameReport *frameReport)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    return status;
}

NvMediaStatus
OV2311WriteParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    WriteReadParametersParamOV2311 *param;
    DevBlkCDIExposure expCtrlApply;
    memset(&expCtrlApply, 0u, sizeof(DevBlkCDIExposure));

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to OV2311ReadParameters");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (parameter == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null output parameter passed to OV2311ReadParameters");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    param = (WriteReadParametersParamOV2311 *)parameter;
    if (param == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Input parameter structure contained null pointer in OV2311ReadParameters");
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    switch (parameterType)
    {
    case CDI_WRITE_PARAM_CMD_EXPOSURE: //Used in initial process, no group write
        if (parameterSize != (sizeof(param->expogain))) {
            SIPL_LOG_ERR_STR("OV2311: ExpoGain parameter size mismatch");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            goto done;
        }
        expCtrlApply.expTimeValid = true;
        expCtrlApply.gainValid = true;
        expCtrlApply.exposureTime[0] = param->expogain.expo;
        expCtrlApply.sensorGain[0] = param->expogain.gain;

        status = SetExposure(handle, &expCtrlApply);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("OV2311: Failed to set exposure time", (int32_t)status);
            goto done;
        }
        break;

    case CDI_WRITE_PARAM_CMD_STROBE://Used in initial process, no group write
        if (param->strobe_ms > 0) {
            expCtrlApply.expTimeValid = true;
            expCtrlApply.exposureTime[0] = param->strobe_ms;
            status = SetStrobe(handle, &expCtrlApply);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("OV2311: Failed to set strobe width", (int32_t)status);
                goto done;
            }
        }
        break;

    default:
        break;
    }
done:
    return status;
}

static NvMediaStatus
SetSensorControls(
    DevBlkCDIDevice const* handle,
    const struct DevBlkCDISensorControl *sensorControl,
    const size_t sensrCtrlStructSize)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    if ((handle == NULL)) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to SetSensorControls");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle*)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to SetSensorControls");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (sensorControl == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null sensor control passed to SetSensorControls");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (sensrCtrlStructSize == 0) {
        SIPL_LOG_ERR_STR("OV2311: Sensor control structure had 0 size in SetSensorControls");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (sensrCtrlStructSize != sizeof(DevBlkCDISensorControl)) {
        SIPL_LOG_ERR_STR("OV2311: Version mismatch detected in SetSensorControls, please update client application");
        status = NVMEDIA_STATUS_NOT_SUPPORTED;
        goto done;
    }

    /* Silicon revision 0x1C has an issue that the embedded data is corrupted
     * if SW programs the sensor with the group hold while the sensor outputs the embedded data
     * To avoid this possibility, programs the sensor after getting the top embedded data lines
     * in ParseTopEmbDataInfo()
     */
    memcpy(&drvHandle->exposureControl_prev,
           sensorControl,
           sizeof(DevBlkCDISensorControl));
    drvHandle->updateSensor = NVMEDIA_TRUE;
done:
    return status;
}

static NvMediaStatus GetDeviceDriverName(char *name);
static NvMediaStatus
GetSensorAttributes(
    DevBlkCDIDevice const* handle,
    struct DevBlkCDISensorAttributes *sensorAttr,
    const size_t sensorAttrStructSize)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    char name[DEVBLK_CDI_MAX_SENSOR_NAME_LENGTH];
    float_t minVal = 0.0f;
    float_t maxVal = 0.0f;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to GetSensorAttributes");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to GetSensorAttributes");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (sensorAttr == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null sensor attributes passed to GetSensorAttributes");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (sensorAttrStructSize == 0) {
        SIPL_LOG_ERR_STR("OV2311: Sensor attribute structure size is 0 in GetSensorAttributes");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (sensorAttrStructSize != sizeof(DevBlkCDISensorAttributes)) {
        SIPL_LOG_ERR_STR("OV2311: Version mismatch in GetSensorAttributes, please recompile client application");
        status = NVMEDIA_STATUS_NOT_SUPPORTED;
        goto done;
    }

    /* Reset attributes */
    (void)memset(sensorAttr, 0u, sizeof(DevBlkCDISensorAttributes));
    status = GetDeviceDriverName(name);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("OV2311: Failed to get device driver name", (int32_t)status);
        goto done;
    }


     /**
      * sensor name
      */
    memcpy(sensorAttr->sensorName, name, DEVBLK_CDI_MAX_SENSOR_NAME_LENGTH);

    /**
     * Holds the CFA attribute. If not supported, set to zero.
     */
    sensorAttr->sensorCFA = NVSIPL_PIXEL_ORDER_CCCC;

     /**
      * Number of active exposure
      */
    sensorAttr->numActiveExposures = drvHandle->numActiveExposures;

    /**
     * Exposure & Gain limitation
     */
#ifdef OV2311_FIXED_EXPOSURE_TIME
    minVal = 0.0005F;
    maxVal = 0.0005F;
#else
    minVal = ((float)(drvHandle->hts)) / ((float) drvHandle->tclk);
    maxVal = ((float)((drvHandle->vts - 12U) * drvHandle->hts)) / ((float) drvHandle->tclk);
#endif

    sensorAttr->sensorExpRange[0].min = minVal;
    sensorAttr->sensorExpRange[0].max = maxVal;
    sensorAttr->sensorGainRange[0].min = OV2311_DGAIN_MIN * OV2311_AGAIN_MIN;
    sensorAttr->sensorGainRange[0].max = OV2311_DGAIN_MAX * OV2311_AGAIN_MAX;
    sensorAttr->sensorWhiteBalanceRange[0].min = OV2311_WBGAIN_MIN;
    sensorAttr->sensorWhiteBalanceRange[0].max = OV2311_WBGAIN_MAX;

    sensorAttr->sensorGainFactor[0] = 1.0f;

    /**
     * Holds the number of frame report bytes supported by the sensor. If not
     * supported, set to zero.
     * Supported values : [1 , NVMEDIA_CDI_MAX_FRAME_REPORT_BYTES]
     */
    sensorAttr->numFrameReportBytes = drvHandle->numFrameReportBytes;

    /**
     * Holds the fuse ID attribute. If not supported, set to NULL.
     */
    (void)memcpy(sensorAttr->sensorFuseId, drvHandle->otp.fuseId, SIZE_FUSE_ID);

done:
    return status;
}


#if !NV_IS_SAFETY
static NvMediaStatus
SetSensorCharMode (
    DevBlkCDIDevice *handle,
    uint8_t expNo)
{
     NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    if ((handle == NULL)) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to SetSensorCharMode");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to SetSensorCharMode");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle->charModeEnabled = NVMEDIA_TRUE;
    drvHandle->charModeExpNo = expNo;
    drvHandle->numActiveExposures = 1;

done:
    return status;
}

static NvMediaStatus
GetModuleConfig(
    DevBlkCDIDevice *handle,
    DevBlkCDIModuleConfig *cameraModuleConfig)
{
     NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    if ((handle == NULL)) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to GetModuleConfig");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to GetModuleConfig");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (cameraModuleConfig == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null camera module config passed to GetModuleConfig");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    status = fusa_strncpy(cameraModuleConfig->cameraModuleCfgName,
                          (drvHandle)->moduleCfg.cameraModuleCfgName,
                          sizeof(cameraModuleConfig->cameraModuleCfgName));
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("[%s:%d] Failed to allocate storage for camera module name\n", __func__, __LINE__);
        goto done;
    }

    cameraModuleConfig->cameraModuleConfigPass1 =
        drvHandle->moduleCfg.cameraModuleConfigPass1;
    cameraModuleConfig->cameraModuleConfigPass2 =
        drvHandle->moduleCfg.cameraModuleConfigPass2;

done:
    return status;
}
#endif

static NvMediaStatus
DriverDestroy(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to DriverDestroy");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to DriverDestroy");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    // Destroy the I2C programmer
    DevBlkCDII2CPgmrDestroy(drvHandle->i2cProgrammer);

    if (handle->deviceDriverHandle != NULL) {
        free(handle->deviceDriverHandle);
        handle->deviceDriverHandle = NULL;
    }

done:
    return status;
}

static NvMediaStatus
DriverCreate(
    DevBlkCDIDevice *handle,
    void const* clientContext)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *driverHandle;

#if !NV_IS_SAFETY
    const ContextOV2311 *ctx = NULL;
#endif

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null handle to DriverCreate");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    driverHandle = calloc(1, sizeof(_DriverHandle));
    if (driverHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Failed to allocate memory for driver handle");
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    driverHandle->revision = CDI_OV2311_INVALID_REV;
    driverHandle->default_setting = &ov2311_reglist_default;

    //(void)memset(driverHandle->fuseId, 0, SIZE_FUSE_ID);
    handle->deviceDriverHandle = (void *)driverHandle;

#if !NV_IS_SAFETY
    if (clientContext != NULL) {  /* ok to be NULL, then use default values */
        ctx = clientContext;

        status = fusa_strncpy(driverHandle->moduleCfg.cameraModuleCfgName,
                              ctx->moduleConfig.cameraModuleCfgName,
                              sizeof(driverHandle->moduleCfg.cameraModuleCfgName));
        if (NVMEDIA_STATUS_OK != status) {
            LOG_ERR("[%s:%d] Out of Memory\n", __func__, __LINE__);
            goto done;
        }

        driverHandle->moduleCfg.cameraModuleConfigPass1 = ctx->moduleConfig.cameraModuleConfigPass1;
        driverHandle->moduleCfg.cameraModuleConfigPass2 = ctx->moduleConfig.cameraModuleConfigPass2;
    }
#endif

    /* copy configInfo to current handle */
    (void)memcpy(driverHandle->regsel_table, regsel_ov2311,
                                        sizeof(CDIRegSetOV2311)*SEL_EMB_LENGTH_OV2311);

    driverHandle->numActiveExposures = OV2311_NUM_HDR_EXP; /* One */
    driverHandle->tclk = OV2311_SCLK; /* initial value related to default setting */
    driverHandle->hts = 0x03aa;  /* Total HTS, initial value related to default setting */
    driverHandle->vts = 0x0588; /* Reg380E, Reg380F, initial value related to default setting */
    driverHandle->numFrameReportBytes = 0u; /* not support */

    driverHandle->Tstrobe = 0u; /* 0: same as exposure, others for fixed width */

    // create the I2C programmer for register read/write
    driverHandle->i2cProgrammer = DevBlkCDII2CPgmrCreate(handle, REG_ADDRESS_BYTES, REG_DATA_BYTES);
    if (driverHandle->i2cProgrammer == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Failed to initialize I2C programmer");
        status = NVMEDIA_STATUS_ERROR;
    }

    driverHandle->initDone = NVMEDIA_FALSE;
    driverHandle->updateSensor = NVMEDIA_FALSE;
done:
    return status;
}

static DevBlkCDIDeviceDriver deviceDriver = {
    .deviceName = "OV2311 Image Sensor",
    .regLength = REG_ADDRESS_BYTES,
    .dataLength = REG_DATA_BYTES,
    .DriverCreate = DriverCreate,
    .DriverDestroy = DriverDestroy,
#if !NV_IS_SAFETY
    .GetModuleConfig = GetModuleConfig,
#endif
    .SetSensorControls = SetSensorControls,
#if !NV_IS_SAFETY
    .SetSensorCharMode = SetSensorCharMode,
#endif
    .GetSensorAttributes = GetSensorAttributes,
    .ParseTopEmbDataInfo = ParseTopEmbDataInfo,
    .ParseBotEmbDataInfo = ParseBotEmbDataInfo,
};

DevBlkCDIDeviceDriver *
GetOV2311Driver(void)
{
    return &deviceDriver;
}

static NvMediaStatus
GetDeviceDriverName(char *name)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

     if (name == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null name passed to GetDeviceDriverName");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    memcpy(name, deviceDriver.deviceName, DEVBLK_CDI_MAX_SENSOR_NAME_LENGTH);
done:
    return status;
}

NvMediaStatus
OV2311ReadErrorSize(size_t *errSize)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;


    if (errSize == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null buffer passed to ReadErrorData");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    *errSize = (sizeof(uint16_t) * (size_t) ASIL_STATUS_MAX_ERR);

done:
    return status;
}

NvMediaStatus
OV2311ReadErrorData(DevBlkCDIDevice *handle, size_t bufSize, uint8_t * const buffer) {
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    if ((handle == NULL)) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to ReadErrorData");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to ReadErrorData");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (buffer == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null buffer passed to ReadErrorData");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    status = NVMEDIA_STATUS_NOT_SUPPORTED;

done:
    return status;
}

NvMediaStatus
OV2311SetDeviceValue(DevBlkCDIDevice *handle, uint32_t const valueToSet) {
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    if ((handle == NULL)) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to SetDeviceValue");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to SetDeviceValue");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    /*
        This type of custom API can be used to set a register value, etc.
        The exact behavior can depend on the supported functionality of the sensor.
    */
    // Setting dummy value for example
    ov2311_custom_value = valueToSet;
    status = NVMEDIA_STATUS_OK;

done:
    return status;
}

NvMediaStatus
OV2311GetDeviceValue(DevBlkCDIDevice *handle, uint32_t * const valueToGet) {
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    if ((handle == NULL)) {
        SIPL_LOG_ERR_STR("OV2311: Null handle passed to GetDeviceValue");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null driver handle passed to GetDeviceValue");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (valueToGet == NULL) {
        SIPL_LOG_ERR_STR("OV2311: Null valueToGet passed to GetDeviceValue");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }
    /*
    This type of custom API can be used to read a register value,
    parse custom embedded data, etc.
    The exact behavior can depend on the supported functionality of the sensor.
    */
    // Getting previously set value for example
    *valueToGet = ov2311_custom_value;
    status = NVMEDIA_STATUS_OK;

done:
    return status;
}
