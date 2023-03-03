/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if !NV_IS_SAFETY
#include "log_utils.h"
#endif
#include "sipl_error.h"
#include "devblk_cdi.h"
#include "os_common.h"
#include "cdi_max96759.h"
#include "cdi_max96759_setting.h"

#define MIN(a, b)            (((a) < (b)) ? (a) : (b))
#define REG_WRITE_BUFFER                    32
#define REG_DEV_ID_ADDR                     0x0D
#define MAX96777_DEV_ID                     0x87
#define MAX96759_DEV_ID                     0xB0
#define MAX_I2C_ADDRESS                     0x7F
#define REG_DEV_REV_ADDR                    0x0E

#define MAX96759_EDID_TIMING_OFFSET         54

#define GET_BLOCK_LENGTH(x) (x[0])
#define GET_SIZE(x)         (sizeof(x)/sizeof(x[0]))
#define GET_BLOCK_DATA(x)   (&x[1])
#define SET_NEXT_BLOCK(x)   x += (x[0] + 1)

/* Pack two values into a byte as follows
   x3 x2 x1 x0 y3 y2 y1 y0 */
#define PACK_TWO(x, y) \
    ((((x) << 4) & 0xF0) | (((y) << 0) & 0x0F))

/* Pack four values into a byte as follows
   x1 x0 y1 y0 z1 z0 w1 w0 */
#define PACK_FOUR(x, y, z, w) \
    ((((x) << 6) & 0xC0) | (((y) << 4) & 0x30) | (((z) << 2) & 0x0C) | (((y) << 0) & 0x03))

typedef struct {
    ContextMAX96759 ctx;
    DevBlkCDII2CPgmr i2cProgrammer;
} _DriverHandle;

typedef struct {
    uint32_t devId;
    RevisionMax96777Max96759 revId;
    uint32_t revVal;
} Revision;

/* These values must include all of values in the RevisionMax96777Max96759 enum */
static Revision supportedRevisions[] = {
    { MAX96777_DEV_ID, CDI_MAX96777_REV_3, 0x3 },
    { MAX96777_DEV_ID, CDI_MAX96777_REV_4, 0x4 },
    { MAX96759_DEV_ID, CDI_MAX96759_REV_5, 0x5 },
    { MAX96759_DEV_ID, CDI_MAX96759_REV_7, 0x7 },
};

const uint8_t max96759_display_name_header[] = {
    0x00, 0x00, 0x00, 0xFC, 0x00
};

typedef struct {
    uint32_t active;
    uint32_t frontPorch;
    uint32_t backPorch;
    uint32_t sync;
} ImageAxis;

typedef struct {
    ImageAxis horizontal;
    ImageAxis vertical;
    uint32_t pclk;
    uint32_t frameRate;
    const DevBlkCDII2CReg *pgSetting;
    const size_t pgSettingSize;
} Max96759EDID;

static const Max96759EDID EDIDTable[] = {
    {
        .horizontal = {
            .active = 1920,
            .frontPorch = 48,
            .backPorch = 80,
            .sync = 32,
        },
        .vertical = {
            .active = 1236,
            .frontPorch = 3,
            .backPorch = 9,
            .sync = 6
        },
        .pclk = 78000000,
        .frameRate = 30,
        .pgSetting = max96759_pg_setting_1920x1236_regs,
        .pgSettingSize = (size_t)I2C_ARRAY_SIZE(max96759_pg_setting_1920x1236_regs)
    },
    {
        .horizontal = {
            .active = 3848,
            .frontPorch = 48,
            .backPorch = 80,
            .sync = 32,
        },
        .vertical = {
            .active = 2174,
            .frontPorch = 3,
            .backPorch = 23,
            .sync = 5
        },
        .pclk = 265120000,
        .frameRate = 30,
        .pgSetting = NULL,
        .pgSettingSize = 0
    },
    {
        .horizontal = {
            .active = 1936,
            .frontPorch = 48,
            .backPorch = 80,
            .sync = 32,
        },
        .vertical = {
            .active = 1223,
            .frontPorch = 3,
            .backPorch = 25,
            .sync = 5
        },
        .pclk = 79000000,
        .frameRate = 30,
        .pgSetting = NULL,
        .pgSettingSize = 0
    },
    {
        .horizontal = {
            .active = 1600,
            .frontPorch = 48,
            .backPorch = 80,
            .sync = 32,
        },
        .vertical = {
            .active = 1302,
            .frontPorch = 30,
            .backPorch = 21,
            .sync = 20
        },
        .pclk = 145000000,
        .frameRate = 60,
        .pgSetting = NULL,
        .pgSettingSize = 0
    }
};

NvMediaStatus
MAX96759SetDefaults(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    const DevBlkCDII2CReg max96759_defaults_regs[] = {
        {0x001A, 0x03},   /* Enable decoding error reporting */
        {0x01C8, 0x13},   /* Video timing generation: freerunning, Disable VSYNC & HSYNC */
        {0x0053, 0x30},   /* Enable both links when in splitter mode */
        {0x20F5, 0x01},   /* Assert HPD */
        {0x0001, 0x88}
    };
    DevBlkCDII2CRegList max96759_defaults = {
        .regs = max96759_defaults_regs,
        .numRegs = I2C_ARRAY_SIZE(max96759_defaults_regs),
    };

    if ((NULL == handle) || (NULL == drvHandle)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &max96759_defaults);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    nvsleep(50000);

    return status;
}

static NvMediaStatus
SetTranslator(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    ReadWriteParams96759 *param)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;

    DevBlkCDII2CReg max96759_set_translator_regs[] = {
        {0x0042, 0x00},
        {0x0043, 0x00},
    };
    DevBlkCDII2CRegList max96759_set_translator = {
        .regs = max96759_set_translator_regs,
        .numRegs = I2C_ARRAY_SIZE(max96759_set_translator_regs),
    };

    if ((NULL == handle) || (NULL == param) || (NULL == drvHandle)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (parameterType == CDI_WRITE_PARAM_CMD_MAX96759_SET_TRANSLATOR_B) {
        max96759_set_translator_regs[0].address = 0x44;
        max96759_set_translator_regs[1].address = 0x45;
    }

    max96759_set_translator_regs[0].data = param->Translator.source << 1;
    max96759_set_translator_regs[1].data = param->Translator.destination << 1;

    return DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                      &max96759_set_translator);
}

static NvMediaStatus
SetDeviceAddress(
    DevBlkCDIDevice *handle,
    unsigned char address)
{
    NvMediaStatus status;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg dev_addr = {0x0000, 0x0};

    if (NULL == handle) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (address > MAX_I2C_ADDRESS) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    dev_addr.data = address << 1;

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        dev_addr.address,
                                        dev_addr.data);

    return status;
}

static NvMediaStatus
SetReset(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg reset = {0x0010, 0x91};

    if (NULL == handle) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        reset.address,
                                        reset.data);

    return status;
}

static NvMediaStatus
SetOneShotReset(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    uint8_t data;

    if ((NULL == handle) || (NULL == drvHandle)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                       0x0010,
                                       &data);
    data |= 0x20;

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        0x0010,
                                        data);

    nvsleep(50000);
    return status;
}

static NvMediaStatus
EnableExternalFrameSync(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg gpio_regs[] = {
        {0x0233, 0x84},
        {0x0235, 0x02}
    };
    DevBlkCDII2CRegList gpio = {
        .regs = gpio_regs,
        .numRegs = I2C_ARRAY_SIZE(gpio_regs),
    };

    if ((NULL == handle) || (NULL == drvHandle)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &gpio);

    return status;
}

static NvMediaStatus
SetupDualView(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg dualview_regs[] = {
        {0x01A0, 0x8D},
        {0x01A2, 0x20}
    };
    DevBlkCDII2CRegList dualview = {
        .regs = dualview_regs,
        .numRegs = I2C_ARRAY_SIZE(dualview_regs),
    };

    if ((NULL == handle) || (NULL == drvHandle)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &dualview);

    return status;
}

static const Max96759EDID*
GetEDIDSpec(
    uint32_t width,
    uint32_t height,
    uint32_t frameRate)
{
    uint32_t i;
    for (i = 0; i < GET_SIZE(EDIDTable); i++) {
        if ((EDIDTable[i].vertical.active == height) &&
            (EDIDTable[i].horizontal.active == width)   &&
            (EDIDTable[i].frameRate == frameRate)) {
            return &EDIDTable[i];
        }
    }
    return NULL;
}

static NvMediaStatus
SetEDID(
    DevBlkCDIDevice *handle,
    uint32_t height,
    uint32_t width,
    uint32_t frameRate,
    const char *identifier)
{
    NvMediaStatus status;
    unsigned int i;
    uint8_t edid[MAX96759_EDID_SIZE];
    uint32_t length;
    uint32_t checksum = 0;
    uint32_t hBlank;
    uint32_t vBlank;
    const Max96759EDID *edidSpec = NULL;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;

    if ((NULL == handle) || (NULL == drvHandle)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* Find the EDID that matches the provided settings */
    edidSpec = GetEDIDSpec(width, height, frameRate);
    if (NULL == edidSpec) {
        SIPL_LOG_ERR_STR_2INT("MAX96759: Could not find EDID settings for resolution", width, height);
        SIPL_LOG_ERR_STR_INT("MAX96759:   and frames per second", frameRate);
        return NVMEDIA_STATUS_NOT_SUPPORTED;
    }

    /* Copy the edid to a local, editable buffer */
    memcpy(edid, max96759_default_edid, MAX96759_EDID_SIZE);

    /* Write the pclk to the edid (10KHz increments little endian)*/
    edid[MAX96759_EDID_TIMING_OFFSET + 0] = ((edidSpec->pclk / 10000) >> 0) & 0xFF;
    edid[MAX96759_EDID_TIMING_OFFSET + 1] = ((edidSpec->pclk / 10000) >> 8) & 0xFF;

    /* Write the active and blanking pixels */
    hBlank = edidSpec->horizontal.frontPorch +
             edidSpec->horizontal.sync +
             edidSpec->horizontal.backPorch;
    edid[MAX96759_EDID_TIMING_OFFSET + 2] = edidSpec->horizontal.active & 0xFF;
    edid[MAX96759_EDID_TIMING_OFFSET + 3] = hBlank;
    edid[MAX96759_EDID_TIMING_OFFSET + 4] = PACK_TWO(edidSpec->horizontal.active >> 8,
                                                     hBlank >> 8);
    vBlank = edidSpec->vertical.frontPorch +
             edidSpec->vertical.sync +
             edidSpec->vertical.backPorch;
    edid[MAX96759_EDID_TIMING_OFFSET + 5] = edidSpec->vertical.active & 0xFF;
    edid[MAX96759_EDID_TIMING_OFFSET + 6] = vBlank;
    edid[MAX96759_EDID_TIMING_OFFSET + 7] = PACK_TWO(edidSpec->vertical.active >> 8, vBlank >> 8);

    /* Write front porch and back porch sync */
    edid[MAX96759_EDID_TIMING_OFFSET + 8] = edidSpec->horizontal.frontPorch & 0xFF;
    edid[MAX96759_EDID_TIMING_OFFSET + 9] = edidSpec->horizontal.sync & 0xFF;
    edid[MAX96759_EDID_TIMING_OFFSET + 10] = PACK_TWO(edidSpec->vertical.frontPorch,
                                                      edidSpec->vertical.sync);
    edid[MAX96759_EDID_TIMING_OFFSET + 11] = PACK_FOUR(edidSpec->horizontal.frontPorch >> 8,
                                                       edidSpec->horizontal.sync >> 8,
                                                       edidSpec->vertical.frontPorch >> 4,
                                                       edidSpec->vertical.sync >> 4);

    /* If an identifier was provided, use it to set the display name descriptor */
    if (NULL != identifier) {
        /* Copy the header to the beginning of the display name descriptor */
        memcpy(&edid[MAX96759_DISPLAY_NAME_OFFSET],
               max96759_display_name_header,
               sizeof(max96759_display_name_header));

        /* Copy the user provided string to the display name descriptor */
        length = strnlen(identifier, MAX96759_DISPLAY_NAME_MAX_LENGTH);
        memcpy(&edid[MAX96759_DISPLAY_NAME_OFFSET + sizeof(max96759_display_name_header)],
               identifier,
               length);

        /* If there is space left, terminate with \n and fill the rest of the space with ' ' */
        if (length < MAX96759_DISPLAY_NAME_MAX_LENGTH) {
            edid[MAX96759_DISPLAY_NAME_OFFSET +
                 sizeof(max96759_display_name_header) +
                 (length++)] = '\n';
        }

        while(length < MAX96759_DISPLAY_NAME_MAX_LENGTH) {
            edid[MAX96759_DISPLAY_NAME_OFFSET +
                 sizeof(max96759_display_name_header) +
                 (length++)] = ' ';
        }
    }

    /* Now recalculate the checksum, ((sum of all bytes from [0 - 127]) % 256) should be 0 */
    for (i = 0; i < MAX96759_EDID_CHECKSUM_OFFSET; i++) {
        checksum += edid[i];
    }
    checksum = 256 - (checksum % 256);
    edid[MAX96759_EDID_CHECKSUM_OFFSET] = checksum % 256;

    /* Write the EDID to the edid table in the max96759 */
    /* Break the write up into smaller writes depending on the size of REG_WRITE_BUFFER */
    for (i = 0;  i < MAX96759_EDID_SIZE; i++) {
        status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                            MAX96759_EDID_START_REGISTER + i,
                                            edid[i]);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_INT("MAX96759: Failed to write to EDID", (int32_t)status);
            return status;
        }
    }

    nvsleep(10000);

    return status;
}

static NvMediaStatus
SetTPG(
    DevBlkCDIDevice *handle,
    uint32_t height,
    uint32_t width,
    uint32_t frameRate)
{
    NvMediaStatus status;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    const Max96759EDID *edidSpec;

    edidSpec = GetEDIDSpec(width, height, frameRate);
    if ((NULL == edidSpec) || (NULL == edidSpec->pgSetting)) {
        SIPL_LOG_ERR_STR_2INT("MAX96759: Could not find TPG settings for resolution", width, height);
        SIPL_LOG_ERR_STR_INT("MAX96759:   and frames per second", frameRate);
        return NVMEDIA_STATUS_NOT_SUPPORTED;
    }

    tpg.regs = edidSpec->pgSetting;
    tpg.numRegs = edidSpec->pgSettingSize;

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &tpg);
    return status;
}

static NvMediaStatus
SetLinkMode(
    DevBlkCDIDevice *handle,
    LinkModeMax96759 mode)
{
    NvMediaStatus status;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg link_mode = {0x0010, 0x00};

    if ((NULL == handle) || (NULL == drvHandle)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    switch (mode) {
        case LINK_MODE_MAX96759_AUTO:
            link_mode.data = 0x30;
            break;
        case LINK_MODE_MAX96759_SPLITTER:
            link_mode.data = 0x33;
            break;
        case LINK_MODE_MAX96759_DUAL:
            link_mode.data = 0x23;
            break;
        case LINK_MODE_MAX96759_LINK_A:
            link_mode.data = 0x31;
            break;
        case LINK_MODE_MAX96759_LINK_B:
            link_mode.data = 0x32;
            break;
        default:
            return NVMEDIA_STATUS_NOT_SUPPORTED;
    }

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        link_mode.address,
                                        link_mode.data);

    return status;
}

static NvMediaStatus
SetBPP(
    DevBlkCDIDevice *handle,
    uint8_t bpp)
{
    NvMediaStatus status;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg bpp_reg = {0x0101, 0x40};

    if ((NULL == handle) || (NULL == drvHandle)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (bpp > MAX96759_MAX_BPP) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    bpp_reg.data |= bpp;

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        bpp_reg.address,
                                        bpp_reg.data);

    return status;
}

static NvMediaStatus
GetRevId (
    DevBlkCDIDevice *handle,
    RevisionMax96777Max96759 *rev)
{
    NvMediaStatus status;
    _DriverHandle *drvHandle = NULL;
    uint8_t readBuff;
    uint8_t deviceId;
    uint8_t revision;
    uint32_t i = 0u, numRev = sizeof(supportedRevisions) / sizeof(supportedRevisions[0]);

    if ((handle == NULL) || ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* First Read the device ID */
    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                       REG_DEV_ID_ADDR,
                                       &readBuff);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }
    deviceId = readBuff;

    /* Read the revision */
    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                       REG_DEV_REV_ADDR,
                                       &readBuff);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }
    revision = readBuff & 0x0F;

    for (i = 0u; i < numRev; i++) {
        if ((deviceId == supportedRevisions[i].devId) &&
            (revision == supportedRevisions[i].revVal)) {
            *rev = supportedRevisions[i].revId;
            LOG_MSG("MAX%s Rev %u detected!\n", (deviceId == MAX96777_DEV_ID) ? "96777" :
                                                                                "96759",
                                                revision);
            return NVMEDIA_STATUS_OK;
        }
    }

    *rev = CDI_MAX96777_MAX96759_INVALID_REV;
    LOG_MSG("Unsupported MAX%s revision %u detected! Supported revisions are:\n",
            (deviceId == MAX96777_DEV_ID) ? "96777" : "96759",
            revision);
    for (i = 0u; i < numRev; i++) {
        LOG_MSG("MAX%s Rev %u\n", (supportedRevisions[i].devId == MAX96777_DEV_ID) ? "96777" :
                                                                                     "96759",
                                  supportedRevisions[i].revVal);
    }

    return NVMEDIA_STATUS_OK;
}

static NvMediaStatus
DriverCreate(
    DevBlkCDIDevice *handle,
    void const* clientContext)
{
    _DriverHandle *drvHandle;

    if (NULL == handle) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = calloc(1, sizeof(_DriverHandle));
    if (!drvHandle) {
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }


    if (clientContext) {
        memcpy(&drvHandle->ctx, clientContext, sizeof(ContextMAX96759));
    }

    handle->deviceDriverHandle = (void *)drvHandle;

    // Create the I2C programmer for register read/write
    drvHandle->i2cProgrammer = DevBlkCDII2CPgmrCreate(handle,
                                                      MAX96759_NUM_ADDR_BYTES,
                                                      MAX96759_NUM_DATA_BYTES);
    if(drvHandle->i2cProgrammer == NULL) {
        SIPL_LOG_ERR_STR("MAX96759: Failed to initialize the I2C programmer");
        free(drvHandle);
        return NVMEDIA_STATUS_ERROR;
    }

    return NVMEDIA_STATUS_OK;

}
static NvMediaStatus
DriverDestroy(
    DevBlkCDIDevice *handle)
{
    _DriverHandle *driverHandle = NULL;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96759: Null handle passed to DriverDestroy");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    driverHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (driverHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96759: Null driver handle passed to DriverDestroy");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    // Destroy the I2C programmer
    DevBlkCDII2CPgmrDestroy(driverHandle->i2cProgrammer);

    if (handle->deviceDriverHandle != NULL) {
        free(handle->deviceDriverHandle);
        handle->deviceDriverHandle = NULL;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
MAX96759SetDeviceConfig(
    DevBlkCDIDevice *handle,
    uint32_t enumeratedDeviceConfig)
{

    if (NULL == handle) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    switch(enumeratedDeviceConfig) {
        case CDI_CONFIG_MAX96759_SET_ONESHOT_RESET:
            return SetOneShotReset(handle);
        case CDI_CONFIG_MAX96759_SET_RESET:
            return SetReset(handle);
        case CDI_CONFIG_MAX96759_SETUP_DUAL_VIEW:
            return SetupDualView(handle);
        case CDI_CONFIG_MAX96759_ENABLE_EXT_FRAME_SYNC:
            return EnableExternalFrameSync(handle);
        default:
            return NVMEDIA_STATUS_NOT_SUPPORTED;
    }
}

NvMediaStatus
MAX96759WriteParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter)
{
    NvMediaStatus status;
    ReadWriteParams96759 *param = parameter;

    if ((NULL == handle) || (NULL == parameter)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    switch(parameterType) {
        case CDI_WRITE_PARAM_CMD_MAX96759_SET_TRANSLATOR_A:
        case CDI_WRITE_PARAM_CMD_MAX96759_SET_TRANSLATOR_B:
            if (parameterSize != sizeof(param->Translator)) {
                return NVMEDIA_STATUS_BAD_PARAMETER;
            }
            status = SetTranslator(handle,
                                   parameterType,
                                   param);
           break;

        case CDI_WRITE_PARAM_CMD_MAX96759_SET_DEVICE_ADDRESS:
            if (parameterSize != sizeof(param->DeviceAddress)) {
                return NVMEDIA_STATUS_BAD_PARAMETER;
            }
            status = SetDeviceAddress(handle,
                                      param->DeviceAddress.address);
           break;
        case CDI_WRITE_PARAM_CMD_MAX96759_SET_EDID:
            if (parameterSize != sizeof(param->EDID)) {
                return NVMEDIA_STATUS_BAD_PARAMETER;
            }
            status = SetEDID(handle,
                             param->EDID.height,
                             param->EDID.width,
                             param->EDID.frameRate,
                             param->EDID.identifier);
           break;
        case CDI_WRITE_PARAM_CMD_MAX96759_SET_TPG:
            if (parameterSize != sizeof(param->TPG)) {
                return NVMEDIA_STATUS_BAD_PARAMETER;
            }
            status = SetTPG(handle,
                            param->TPG.height,
                            param->TPG.width,
                            param->TPG.frameRate);
           break;
        case CDI_WRITE_PARAM_CMD_MAX96759_SET_LINK_MODE:
            if (parameterSize != sizeof(param->LinkMode)) {
                return NVMEDIA_STATUS_BAD_PARAMETER;
            }
            status = SetLinkMode(handle,
                                 param->LinkMode.mode);
           break;
        case CDI_WRITE_PARAM_CMD_MAX96759_SET_BPP:
            if (parameterSize != sizeof(param->BitsPerPixel)) {
                return NVMEDIA_STATUS_BAD_PARAMETER;
            }
            status = SetBPP(handle,
                            param->BitsPerPixel);
           break;

        default:
            status = NVMEDIA_STATUS_NOT_SUPPORTED;
            break;
    }

    return status;
}

NvMediaStatus
MAX96759ReadParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter)
{
    ReadWriteParams96759 *param = parameter;

    if (!handle || !parameter) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    switch(parameterType) {
        case CDI_READ_PARAM_CMD_MAX96759_REV_ID:
            if (parameterSize != sizeof(param->Revision))
                return NVMEDIA_STATUS_BAD_PARAMETER;
            return GetRevId(handle,
                            &param->Revision);
        default:
            break;
    }
    return NVMEDIA_STATUS_ERROR;
}

NvMediaStatus
MAX96759DumpRegisters(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    uint8_t data = 0;
    uint32_t i = 0u;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96759: Null handle passed to MAX96759DumpRegisters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96759: Null driver handle passed to MAX96759DumpRegisters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    for (i = 0u; i <= MAX96759_MAX_REG_ADDRESS; i++) {
        status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                           ((i / 256u) << 8) | (i % 256u),
                                           &data);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_2INT("MAX96759: Failed to read register with status", (int32_t)i, (int32_t)status);
            return status;
        }

        LOG_MSG("MAX96759: 0x%04X%02X - 0x%02X\n", (i / 256u), (i % 256u), data);
    }

    return status;
}

NvMediaStatus
MAX96759CheckPresence(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status;
    _DriverHandle *drvHandle = NULL;
    RevisionMax96777Max96759 rev;

    if ((handle == NULL) || ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = GetRevId(handle,
                      &rev);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    if (rev == CDI_MAX96777_MAX96759_INVALID_REV) {
        SIPL_LOG_ERR_STR("MAX96759: Detected unsupported serializer");
        return NVMEDIA_STATUS_NOT_SUPPORTED;
    }

    drvHandle->ctx.revision = rev;

    return NVMEDIA_STATUS_OK;
}

static DevBlkCDIDeviceDriver deviceDriver = {
    .deviceName = "Maxim 96759 Serializer",
    .regLength = 2,
    .dataLength = 1,
    .DriverCreate = DriverCreate,
    .DriverDestroy = DriverDestroy,
};

DevBlkCDIDeviceDriver *
GetMAX96759Driver(void)
{
    return &deviceDriver;
}
