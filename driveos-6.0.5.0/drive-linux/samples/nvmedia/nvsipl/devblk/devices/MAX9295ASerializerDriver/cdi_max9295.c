/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include "sipl_error.h"
#if !NV_IS_SAFETY
#include "log_utils.h"
#endif
#include "devblk_cdi.h"
#include "devblk_cdi_i2c.h"
#include "cdi_max9295.h"
#include "os_common.h"

#define GET_BLOCK_LENGTH(x)            x[0]
#define GET_SIZE(x)                    sizeof(x)
#define GET_BLOCK_DATA(x)              &x[1]
#define SET_NEXT_BLOCK(x)              x += (x[0] + 1)

#define MAX9295_NUM_ADDR_BYTES         2u
#define MAX9295_NUM_DATA_BYTES         1u
#define REG_WRITE_BUFFER_BYTES         MAX9295_NUM_DATA_BYTES
#define MAX9295_CDI_DEVICE_INDEX       0u

#define REG_DEV_ID_ADDR                0x0D
#define MAX9295A_DEV_ID                0x91
#define MAX9295B_DEV_ID                0x93
#define REG_DEV_REV_ADDR               0x0E
#define REG_LFLT_INT                   0x1B
#define MAX9295_REG_MAX_ADDRESS        0x1576

typedef struct {
    RevisionMAX9295 revId;
    uint32_t revVal;
} Revision;

typedef struct {
    DevBlkCDII2CPgmr i2cProgrammer;
} _DriverHandle;

/* These values must include all of values in the RevisionMAX9295 enum */
static Revision supportedRevisions[] = {
    {CDI_MAX9295_REV_5, 5u},
    {CDI_MAX9295_REV_7, 7u},
    {CDI_MAX9295_REV_8, 8u},
};

static NvMediaStatus
SetDeviceAddress(
    DevBlkCDIDevice *handle,
    uint8_t address)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    DevBlkCDII2CReg setAddrRegs[] = {{0x0000, 0x0}};
    DevBlkCDII2CRegList setAddr = {
        .regs = setAddrRegs,
        .numRegs = I2C_ARRAY_SIZE(setAddrRegs),
    };

    if ((handle == NULL)) {
        SIPL_LOG_ERR_STR("MAX9295: null handle passed to SetDeviceAddress");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: null driver handle passed to SetDeviceAddress");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    /* Check for 7 bit I2C address */
    if (address >= 0x80) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX9295: Bad parameter: Address is greater than 0x80", (uint32_t)address);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    setAddrRegs[0].data = address << 1;

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &setAddr);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Failed to set device address", (int32_t)status);
        SIPL_LOG_ERR_STR_HEX_UINT("MAX9295:  Attempting to set address", (uint32_t)address);
    }

done:
    return status;
}

static NvMediaStatus
SetTranslator(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint8_t source,
    uint8_t destination)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    DevBlkCDII2CReg setTranslatorRegs[] = {
        {0x0042, 0x00},
        {0x0043, 0x00},
    };
    DevBlkCDII2CRegList setTranslator = {
        .regs = setTranslatorRegs,
        .numRegs = I2C_ARRAY_SIZE(setTranslatorRegs),
    };

    if ((handle == NULL) || ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        SIPL_LOG_ERR_STR("MAX9295: Null handle passed to SetTranslator");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    /* Check for 7 bit I2C address */
    if ((source >= 0x80) || (destination >= 0x80)) {
        SIPL_LOG_ERR_STR("MAX9295: Source or destination address out of 7-bit range");
        SIPL_LOG_ERR_STR_HEX_UINT("MAX9295:  - source address", (uint32_t) source);
        SIPL_LOG_ERR_STR_HEX_UINT("MAX9295:  - destination address", (uint32_t) destination);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (parameterType == CDI_WRITE_PARAM_CMD_MAX9295_SET_TRANSLATOR_B) {
        setTranslatorRegs[0].address += 2;
        setTranslatorRegs[1].address += 2;
    }

    setTranslatorRegs[0].data = source << 1;
    setTranslatorRegs[1].data = destination << 1;

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &setTranslator);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Set translation I2C write failed", (int32_t)status);
    }

done:
    return status;
}

static NvMediaStatus
ConfigPipelinesDoublePipe(
    DevBlkCDIDevice *handle,
    DataTypeMAX9295 dataType,
    bool embDataType)
{
    /* Configure pipeline X for pixel data and
     * pipeline Y for emb data if enabled */
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    DevBlkCDII2CReg startPipeFromPortRegs[] = {{0x0311, 0x00}};
    DevBlkCDII2CRegList startPipeFromPort = {
        .regs = startPipeFromPortRegs,
        .numRegs = I2C_ARRAY_SIZE(startPipeFromPortRegs),
    };

    DevBlkCDII2CReg pipePortMapRegs[] = {{0x0308, 0x40}};
    DevBlkCDII2CRegList pipePortMap = {
        .regs = pipePortMapRegs,
        .numRegs = I2C_ARRAY_SIZE(pipePortMapRegs),
    };

    DevBlkCDII2CReg videoTxEnRegs[] = {{0x0002, 0x13}};
    DevBlkCDII2CRegList videoTxEn = {
        .regs = videoTxEnRegs,
        .numRegs = I2C_ARRAY_SIZE(videoTxEnRegs),
    };

    DevBlkCDII2CReg mappingPixelRegs[] = {
                        {0x0314, 0x6C}, /* Route 12bit RAW to VIDEO_X (MSB enable) */
                        {0x0053, 0x10}, /* Stream ID for packets for VIDEO_X */
                    };
    DevBlkCDII2CRegList mappingPixel = {
        .regs = mappingPixelRegs,
        .numRegs = I2C_ARRAY_SIZE(mappingPixelRegs),
    };

    DevBlkCDII2CReg mappingEmbRegs[] = {
                        {0x0316, 0x52}, /* Route EMBEDDED8 to VIDEO_Y (MSB enable) */
                        {0x0057, 0x11}, /* Stream ID for packets for VIDEO_Y */
                    };
    DevBlkCDII2CRegList mappingEmb = {
        .regs = mappingEmbRegs,
        .numRegs = I2C_ARRAY_SIZE(mappingEmbRegs),
    };

    DevBlkCDII2CReg doubleBpp12Bpp8Regs[] = {
                        {0x0312, 0x02},
                        {0x0313, 0x10},
                        {0x031C, 0x38},
                        {0x031D, 0x30},
                    };
    DevBlkCDII2CRegList doubleBpp12Bpp8 = {
        .regs = doubleBpp12Bpp8Regs,
        .numRegs = I2C_ARRAY_SIZE(doubleBpp12Bpp8Regs),
    };

    DevBlkCDII2CReg disableHeartbeatRegs[] = {
                        {0x0102, 0x0E},
                        {0x010A, 0x0E},
                    };
    DevBlkCDII2CRegList disableHeartbeat = {
        .regs = disableHeartbeatRegs,
        .numRegs = I2C_ARRAY_SIZE(disableHeartbeatRegs),
    };

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver handle passed to ConfigPipelines");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver driver handle passed to ConfigPipelines");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    /* Update mapping table with data type */
    switch (dataType) {
        case CDI_MAX9295_DATA_TYPE_RAW10:
            mappingPixelRegs[0].data = 0x6B;
            break;
        case CDI_MAX9295_DATA_TYPE_RAW12:
            mappingPixelRegs[0].data = 0x6C;
            break;
        case CDI_MAX9295_DATA_TYPE_RAW16:
            mappingPixelRegs[0].data = 0x6E;
            break;
        default:
            SIPL_LOG_ERR_STR_HEX_UINT("MAX9295: Invalid data type passed to ConfigPipelines", (uint32_t)dataType);
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            goto done;
    }

    startPipeFromPortRegs[0].data = 0x10;
    pipePortMapRegs[0].data |= 0x21;
    if (embDataType) {
        startPipeFromPortRegs[0].data |= 0x20;
        pipePortMapRegs[0].data |= 0x02;
        videoTxEnRegs[0].data |= 0x20;
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &startPipeFromPort);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Start pipe from port write failed", (int32_t)status);
        goto done;
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &pipePortMap);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Pipe port map write failed", (int32_t)status);
        goto done;
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &mappingPixel);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Mapping pixel write failed", (int32_t)status);
        goto done;
    }

    if (embDataType) {
        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &mappingEmb);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX9295: Mapping emb write failed", (int32_t)status);
            goto done;
        }
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &videoTxEn);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Video tx enable write failed", (int32_t)status);
        goto done;
    }

    /* Turn on double mode only if emb data type is enabled and pixel data type is RAW12/RAW10 */
    if (dataType == CDI_MAX9295_DATA_TYPE_RAW10) {
        doubleBpp12Bpp8Regs[1].data = 0x1;
        doubleBpp12Bpp8Regs[2].data = 0x34;
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &doubleBpp12Bpp8);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Double BPP12BPP8 write failed", (int32_t)status);
        goto done;
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &disableHeartbeat);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Disable heart beat write failed", (int32_t)status);
        goto done;
    }

done:
    return status;
}


static NvMediaStatus
ConfigPipelinesSinglePipe(
    DevBlkCDIDevice *handle,
    DataTypeMAX9295 dataType,
    bool embDataType)
{
    /* Configure pipeline Z for pixel data and/or emb data if enabled */
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    DevBlkCDII2CReg videoTxEnRegs[] = {{0x0002, 0x43}}; /* Enable pipe Z */
    DevBlkCDII2CRegList videoTxEn = {
        .regs = videoTxEnRegs,
        .numRegs = I2C_ARRAY_SIZE(videoTxEnRegs),
    };

    DevBlkCDII2CReg doubleBpp12Bpp8Regs[] = {
                        {0x0312, 0x04}, /* Double EMB8 in pipe Z */
                        {0x031E, 0x2C}, /* Min BPP = 12 in pipe Z */
                        {0x0111, 0x50}, /* Max BPP = 16 in pipe Z */
                        {0x0110, 0x60}, /* Disable auto BPP in pipe Z */
                        {0x0112, 0x0C}, /* Limit heartbeat and disable drift detect on pipe Z */
    };
    DevBlkCDII2CRegList doubleBpp12Bpp8 = {
        .regs = doubleBpp12Bpp8Regs,
        .numRegs = I2C_ARRAY_SIZE(doubleBpp12Bpp8Regs),
    };

    DevBlkCDII2CReg doubleBpp12BppRegs[] = {
                        {0x0313, 0x40}, /* Double 12-bit data on pipe Z */
                        {0x031E, 0x38}, /* Min BPP = 24 in pipe Z */
                        {0x0118, 0x6C}, /* RAW12 to pipe Z */
                        {0x0112, 0x0E}, /* Limit heartbeat on pipe Z */
    };
    DevBlkCDII2CRegList doubleBpp12Bpp = {
        .regs = doubleBpp12BppRegs,
        .numRegs = I2C_ARRAY_SIZE(doubleBpp12BppRegs),
    };

    DevBlkCDII2CReg mappingPixelRegs[] = {
                        {0x0318, 0x6C}, /* Route 12bit RAW to VIDEO_Z (MSB enable) */
                    };
    DevBlkCDII2CRegList mappingPixel = {
        .regs = mappingPixelRegs,
        .numRegs = I2C_ARRAY_SIZE(mappingPixelRegs),
    };

    DevBlkCDII2CReg mappingEmbRegs[] = {
                        {0x0319, 0x52}, /* Route EMBEDDED8 to VIDEO_Z (MSB enable) */
                    };
    DevBlkCDII2CRegList mappingEmb = {
        .regs = mappingEmbRegs,
        .numRegs = I2C_ARRAY_SIZE(mappingEmbRegs),
    };

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver handle passed to ConfigPipelines");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver driver handle passed to ConfigPipelines");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &mappingPixel);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Mapping pixel write failed", (int32_t)status);
        goto done;
    }

    if (embDataType) {
        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &mappingEmb);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX9295: Mapping emb write failed", (int32_t)status);
            goto done;
        }
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &videoTxEn);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Video tx enable write failed", (int32_t)status);
        goto done;
    }

    if (embDataType) {
       status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &doubleBpp12Bpp8);
       if (status != NVMEDIA_STATUS_OK) {
           SIPL_LOG_ERR_STR_INT("MAX9295: Double BPP12BPP8 write failed", (int32_t)status);
           goto done;
       }
    } else {
       status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &doubleBpp12Bpp);
       if (status != NVMEDIA_STATUS_OK) {
           SIPL_LOG_ERR_STR_INT("MAX9295: Double BPP12 write failed", (int32_t)status);
           goto done;
       }
    }

done:
    return status;
}

static NvMediaStatus
ConfigPipelines(
    DevBlkCDIDevice *handle,
    DataTypeMAX9295 dataType,
    bool embDataType,
    bool isSinglePipeline)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver handle passed to ConfigPipelines");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver driver handle passed to ConfigPipelines");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (isSinglePipeline) {
        status = ConfigPipelinesSinglePipe(handle, dataType, embDataType);
    } else {
        status = ConfigPipelinesDoublePipe(handle, dataType, embDataType);
    }

done:
    return status;
}

static NvMediaStatus
ConfigPhy(
    DevBlkCDIDevice *handle,
    phyMapMAX9295 *mapping,
    phyPolarityMAX9295 *polarity,
    uint8_t numDataLanes)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    DevBlkCDII2CReg phyMapRegs[] = {
        {0x0330, 0x00},
        {0x0331, 0x00},
        {0x0332, 0xEE},
        {0x0333, 0xE4}
    };
    DevBlkCDII2CRegList phyMap = {
        .regs = phyMapRegs,
        .numRegs = I2C_ARRAY_SIZE(phyMapRegs),
    };

    DevBlkCDII2CReg phyPolarityRegs[] = {
        {0x0334, 0x00},
        {0x0335, 0x00}
    };
    DevBlkCDII2CRegList phyPolarity = {
        .regs = phyPolarityRegs,
        .numRegs = I2C_ARRAY_SIZE(phyPolarityRegs),
    };

    if ((handle == NULL)) {
        SIPL_LOG_ERR_STR("MAX9295: Null handle passed to ConfigPhy");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver handle passed to ConfigPhy");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (mapping == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null mapping passed to ConfigPhy");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (polarity == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null polarity passed to ConfigPhy");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if ((numDataLanes != 2) && (numDataLanes != 4)) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Number of data lanes was not 2 or 4", (int32_t)numDataLanes);
        status = NVMEDIA_STATUS_NOT_SUPPORTED;
        goto done;
    }

    if ((numDataLanes == 2) && (mapping->enableMapping)) {
        SIPL_LOG_ERR_STR("MAX9295: Lane swapping is supported only in 4 lane mode");
        status = NVMEDIA_STATUS_NOT_SUPPORTED;
        goto done;
    }

    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, phyMapRegs[1].address, (uint8_t *) &(phyMapRegs[1].data));
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Set phy map read failed", (int32_t)status);
        goto done;
    }

    /* data lanes indexing starts at 0 (0 = 1 lane, 1 = 2 lanes, etc) */
    numDataLanes -= 1;
    /* clear the data lanes settings for Port B */
    phyMapRegs[1].data &= ~0x30;
    /* Set num data lanes for Port B */
    phyMapRegs[1].data |= (numDataLanes << 4);

    if (mapping->enableMapping) {
        phyMapRegs[2].data = (mapping->phy1_d1 << 6) |
                             (mapping->phy1_d0 << 4) |
                             (mapping->phy0_d1 << 2) |
                             (mapping->phy0_d0);
        phyMapRegs[3].data = (mapping->phy3_d1 << 6) |
                             (mapping->phy3_d0 << 4) |
                             (mapping->phy2_d1 << 2) |
                             (mapping->phy2_d0);
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &phyMap);
    if (status != NVMEDIA_STATUS_OK) {
        NvOsDebugPrintStrInt(NVOS_SLOG_CODE_CAMERA, NVOS_LOG_SEVERITY_ERROR,
                "MAX9295: Set phy map write failed", (int)status);
        goto done;
    }

    if (polarity->setPolarity) {
        phyPolarityRegs[0].data = (polarity->phy1_clk << 6) |
                                  (polarity->phy1_d1  << 5) |
                                  (polarity->phy1_d0  << 4);
        phyPolarityRegs[1].data = (polarity->phy2_clk << 2) |
                                  (polarity->phy2_d1  << 1) |
                                  (polarity->phy2_d0);

        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &phyPolarity);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX9295: Set phy polarity write failed", (int32_t)status);
            goto done;
        }
    }

done:
    return status;
}

static NvMediaStatus
SetGPIOOutput(
    DevBlkCDIDevice *handle,
    GPIOTypeMAX9295 gpio,
    bool level)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    DevBlkCDII2CReg setGPIOAModeRegs[] = {{0x02BE, 0x80}}; /* pull-up 1M ohm, output driver enabled */
    DevBlkCDII2CRegList setGPIOAMode = {
        .regs = setGPIOAModeRegs,
        .numRegs = I2C_ARRAY_SIZE(setGPIOAModeRegs),
    };

    DevBlkCDII2CReg setGPIOBModeRegs[] = {{0x02BF, 0x60}}; /* pull-up, output push-pull */
    DevBlkCDII2CRegList setGPIOBMode = {
        .regs = setGPIOBModeRegs,
        .numRegs = I2C_ARRAY_SIZE(setGPIOBModeRegs),
    };

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null handle passed to SetGPIOOutput");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver handle passed to SetGPIOOutput");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (gpio == CDI_MAX9295_GPIO_TYPE_INVALID) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Invalid GPIO pin passed to SetGPIOOutput", (int32_t)gpio);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (gpio >= CDI_MAX9295_GPIO_TYPE_NUM) {
        SIPL_LOG_ERR_STR_INT("MAX9295: GPIO pin passed to SetGPIOOutput exceeds maximum", (int32_t)gpio);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    setGPIOAModeRegs[0].address += ((uint8_t) gpio - 1u) * 3u;
    setGPIOBModeRegs[0].address += ((uint8_t) gpio - 1u) * 3u;

    if (level) {
        setGPIOAModeRegs[0].data |= (1 << 4);
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &setGPIOAMode);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Set GPIO A mode write failed", (int32_t)status);
        goto done;
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &setGPIOBMode);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Set GPIO B mode write failed", (int32_t)status);
        goto done;
    }

done:
    return status;
}

static NvMediaStatus
SetFsyncGPIO(
    DevBlkCDIDevice *handle,
    GPIOTypeMAX9295 gpio,
    uint8_t rxID)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    DevBlkCDII2CReg setGPIOModeRegs[] = {{0x02BE, 0x84}}; /* pull-up 1M ohm, GPIO source en for GMSL2 */
    DevBlkCDII2CRegList setGPIOMode = {
        .regs = setGPIOModeRegs,
        .numRegs = I2C_ARRAY_SIZE(setGPIOModeRegs),
    };

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null handle passed to SetFsyncGPIO");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver handle passed to SetFsyncGPIO");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (gpio == CDI_MAX9295_GPIO_TYPE_INVALID) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Invalid GPIO pin passed to SetFsyncGPIO", (int32_t)gpio);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (gpio >= CDI_MAX9295_GPIO_TYPE_NUM) {
        SIPL_LOG_ERR_STR_INT("MAX9295: GPIO pin passed to SetFsyncGPIO exceeds maximum", (int32_t)gpio);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    setGPIOModeRegs[0].address += ((uint16_t) gpio - 1u) * 3u;

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &setGPIOMode);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Set FSync write failed", (int32_t)status);
        goto done;
    }

    /* Update the offset from GPIO A to GPIO C*/
    setGPIOModeRegs[0].address += 2u;
    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, setGPIOModeRegs[0].address, (uint8_t *) &(setGPIOModeRegs[0].data));
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Set FSync read failed", (int32_t)status);
        goto done;
    }

    setGPIOModeRegs[0].data &= 0xE0;
    setGPIOModeRegs[0].data |= (rxID & 0x1F); /* GPIO receive ID */
    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &setGPIOMode);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Set FSync GPIO write failed", (int32_t)status);
        goto done;
    }

done:
    return status;
}

static NvMediaStatus
EnableRefClock(
    DevBlkCDIDevice *handle,
    GPIOTypeMAX9295 gpio,
    bool enableRClk)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    DevBlkCDII2CReg enablePCLKOutRegs[] = {{0x3F1, 0x00}};
    DevBlkCDII2CRegList enablePCLKOut = {
        .regs = enablePCLKOutRegs,
        .numRegs = I2C_ARRAY_SIZE(enablePCLKOutRegs),
    };

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null handle passed to EnableRefClock");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver handle passed to EnableRefClock");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (gpio == CDI_MAX9295_GPIO_TYPE_INVALID) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Invalid GPIO pin passed to EnableRefClock", (int32_t)gpio);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (gpio >= CDI_MAX9295_GPIO_TYPE_NUM) {
        SIPL_LOG_ERR_STR_INT("MAX9295: GPIO pin passed to EnableRefClock exceeds GPIO maximum", (int32_t)gpio);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (enableRClk) {
        enablePCLKOutRegs[0].data |= 0x80;
    }

    enablePCLKOutRegs[0].data |= ((((uint8_t) gpio - 1u) << 1) | 0x1);

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &enablePCLKOut);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Enable ref clock write failed", (int32_t)status);
        goto done;
    }

done:
    return status;
}

static NvMediaStatus
ForwardGPIO(
    DevBlkCDIDevice *handle,
    uint8_t srcGpio,
    uint8_t dstGpio)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    DevBlkCDII2CReg setGPIOModeRegs[] = {{0x02BE, 0x1C}};
    DevBlkCDII2CRegList setGPIOMode = {
        .regs = setGPIOModeRegs,
        .numRegs = I2C_ARRAY_SIZE(setGPIOModeRegs),
    };

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null handle passed to ForwardGPIO");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver handle passed to ForwardGPIO");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (srcGpio > 10U) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Source GPIO to forward exceeds 10", (int32_t)srcGpio);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (dstGpio > 31U) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Destination GPIO exceeds 31", (int32_t)dstGpio);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    setGPIOModeRegs[0].address += (srcGpio * 3u);

    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, setGPIOModeRegs[0].address, (uint8_t *) &(setGPIOModeRegs[0].data));
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Set forward GPIO read failed", (int32_t)status);
        goto done;
    }

    setGPIOModeRegs[0].data |= 0x3; /* Set GPIO_TX_EN, GPIO_OUT_DIS */
    setGPIOModeRegs[0].data &= ~(1 << 2); /* Unset GPIO_RX_EN */

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &setGPIOMode);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Set Forward GPIO write failed", (int32_t)status);
        goto done;
    }

    /* Update the offset from GPIO A to GPIO B */
    setGPIOModeRegs[0].address += 1u;
    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, setGPIOModeRegs[0].address, (uint8_t *) &(setGPIOModeRegs[0].data));
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Set Forward GPIIO read failed", (int32_t)status);
        goto done;
    }

    setGPIOModeRegs[0].data &= 0xE0;
    setGPIOModeRegs[0].data |= (dstGpio & 0x1F);
    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &setGPIOMode);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Set Forward GPIO mode write failed", (int32_t)status);
        goto done;
    }

done:
    return status;
}

static NvMediaStatus
GetRevId(
    DevBlkCDIDevice *handle,
    RevisionMAX9295 *rev)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    uint32_t numRev = sizeof(supportedRevisions) / sizeof(supportedRevisions[0]);
    uint8_t revision = 0u;
    uint32_t i = 0u;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null handle passed to GetRevId");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver handle passed to GetRevId");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, REG_DEV_REV_ADDR, &revision);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: GetRevId read failed", (int32_t)status);
        goto done;
    }

    revision &= 0x0F;
    for (i = 0u; i < numRev; i++) {
        if (revision == supportedRevisions[i].revVal) {
            *rev = supportedRevisions[i].revId;
            LOG_MSG("MAX9295: Revision %u detected!\n", (int)revision);
            goto done;
        }
    }

    LOG_MSG("MAX9295: Unsupported MAX9295 revision %u detected!\nSupported revisions are:", (int)revision);
    for (i = 0u; i < numRev; i++) {
        LOG_MSG("MAX9295: Revision %u\n", supportedRevisions[i].revVal);
    }
    status = NVMEDIA_STATUS_NOT_SUPPORTED;

done:
    return status;
}

static NvMediaStatus
DriverCreate(
    DevBlkCDIDevice *handle,
    void const* clientContext)
{
    _DriverHandle *drvHandle = NULL;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null handle passed to DriverCreate");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (clientContext != NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null context passed to DriverCreate");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = calloc(1, sizeof(_DriverHandle));
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Unable to allocate memory for driver handle");
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
    }

    handle->deviceDriverHandle = (void *)drvHandle;

    // Create the I2C programmer for register read/write
    drvHandle->i2cProgrammer = DevBlkCDII2CPgmrCreate(handle, MAX9295_NUM_ADDR_BYTES, MAX9295_NUM_DATA_BYTES);
    if(drvHandle->i2cProgrammer == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Failed to initialize the I2C programmer");
        status = NVMEDIA_STATUS_ERROR;
    }

done:
    return status;
}

static NvMediaStatus
DriverDestroy(
    DevBlkCDIDevice *handle)
{
    _DriverHandle *drvHandle = NULL;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null handle passed to DriverDestroy");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver handle passed to DriverDestroy");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }
    // Destroy the I2C programmer
    DevBlkCDII2CPgmrDestroy(drvHandle->i2cProgrammer);

    free(drvHandle);

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
MAX9295SetDefaults(
    DevBlkCDIDevice *handle)
{
    _DriverHandle *drvHandle = NULL;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null handle passed to MAX9295SetDefaults");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver handle passed to MAX9295SetDefaults");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
MAX9295WriteParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    bool isValidSize = false;
    ReadWriteParamsMAX9295 *param = (ReadWriteParamsMAX9295 *) parameter;
    _DriverHandle *drvHandle = NULL;
    static const char *cmdString[] = {
        [CDI_WRITE_PARAM_CMD_MAX9295_SET_TRANSLATOR_A] =
            "CDI_WRITE_PARAM_CMD_MAX9295_SET_TRANSLATOR_A",
        [CDI_WRITE_PARAM_CMD_MAX9295_SET_TRANSLATOR_B] =
            "CDI_WRITE_PARAM_CMD_MAX9295_SET_TRANSLATOR_B",
        [CDI_WRITE_PARAM_CMD_MAX9295_SET_DEVICE_ADDRESS] =
            "CDI_WRITE_PARAM_CMD_MAX9295_SET_DEVICE_ADDRESS",
        [CDI_WRITE_PARAM_CMD_MAX9295_SET_FSYNC_GPIO] =
            "CDI_WRITE_PARAM_CMD_MAX9295_SET_FSYNC_GPIO",
        [CDI_WRITE_PARAM_CMD_MAX9295_SET_GPIO_OUTPUT] =
            "CDI_WRITE_PARAM_CMD_MAX9295_SET_GPIO_OUTPUT",
        [CDI_WRITE_PARAM_CMD_MAX9295_ENABLE_REF_CLOCK] =
            "CDI_WRITE_PARAM_CMD_MAX9295_ENABLE_REF_CLOCK",
        [CDI_WRITE_PARAM_CMD_MAX9295_CONFIG_VIDEO_PIPELINES] =
            "CDI_WRITE_PARAM_CMD_MAX9295_CONFIG_VIDEO_PIPELINES",
        [CDI_WRITE_PARAM_CMD_MAX9295_CONFIG_PHY] =
            "CDI_WRITE_PARAM_CMD_MAX9295_CONFIG_PHY",
        [CDI_WRITE_PARAM_CMD_MAX9295_GPIO_FORWARD] =
            "CDI_WRITE_PARAM_CMD_MAX9295_GPIO_FORWARD",
    };

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null handle passed to MAX9295WriteParameters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (parameter == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null parameter passed to MAX9295WriteParameters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if  (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver handle passed to MAX9295WriteParameters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (parameterType == CDI_WRITE_PARAM_CMD_MAX9295_INVALID) {
        SIPL_LOG_ERR_STR_UINT("MAX9295: Invalid parameter type passed to MAX9295WriteParameters", (uint32_t)parameterType);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if  (parameterType >= CDI_WRITE_PARAM_CMD_MAX9295_NUM) {
        SIPL_LOG_ERR_STR_UINT("MAX9295: Out of range parameter type passed to MAX9295WriteParameters", (uint32_t)parameterType);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    LOG_DBG("MAX9295: Executing command %s", cmdString[parameterType]);
    switch (parameterType) {
        case CDI_WRITE_PARAM_CMD_MAX9295_SET_TRANSLATOR_A:
        case CDI_WRITE_PARAM_CMD_MAX9295_SET_TRANSLATOR_B:
            if (parameterSize == sizeof(param->Translator)) {
                isValidSize = true;
                status = SetTranslator(handle,
                                       parameterType,
                                       param->Translator.source,
                                       param->Translator.destination);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX9295_SET_DEVICE_ADDRESS:
            if (parameterSize == sizeof(param->DeviceAddress)) {
                isValidSize = true;
                status = SetDeviceAddress(handle,
                                          param->DeviceAddress.address);
            };
            break;
        case CDI_WRITE_PARAM_CMD_MAX9295_SET_GPIO_OUTPUT:
            if (parameterSize == sizeof(param->GPIOOutp)) {
                isValidSize = true;
                status = SetGPIOOutput(handle,
                                       param->GPIOOutp.gpioInd,
                                       param->GPIOOutp.level);
            };
            break;
        case CDI_WRITE_PARAM_CMD_MAX9295_SET_FSYNC_GPIO:
            if (parameterSize == sizeof(param->FSyncGPIO)) {
                isValidSize = true;
                status = SetFsyncGPIO(handle,
                                       param->FSyncGPIO.gpioInd,
                                       param->FSyncGPIO.rxID);
            };
            break;
        case CDI_WRITE_PARAM_CMD_MAX9295_ENABLE_REF_CLOCK:
            if (parameterSize == sizeof(param->RefClkGPIO)) {
                isValidSize = true;
                status = EnableRefClock(handle,
                                        param->RefClkGPIO.gpioInd,
                                        param->RefClkGPIO.enableRClk);
            };
            break;
        case CDI_WRITE_PARAM_CMD_MAX9295_GPIO_FORWARD:
            if (parameterSize == sizeof(param->GPIOForward)) {
                isValidSize = true;
                status = ForwardGPIO(handle,
                                     param->GPIOForward.srcGpio,
                                     param->GPIOForward.dstGpio);
            };
            break;
        case CDI_WRITE_PARAM_CMD_MAX9295_CONFIG_VIDEO_PIPELINES:
            if (parameterSize == sizeof(param->ConfigVideoPipeline)) {
                isValidSize = true;
                status = ConfigPipelines(handle,
                                         param->ConfigVideoPipeline.dataType,
                                         param->ConfigVideoPipeline.embDataType,
                                         param->ConfigVideoPipeline.isSinglePipeline);
            };
            break;
        case CDI_WRITE_PARAM_CMD_MAX9295_CONFIG_PHY:
            if (parameterSize == sizeof(param->ConfigPhy)) {
                isValidSize = true;
                status = ConfigPhy(handle,
                                   &param->ConfigPhy.mapping,
                                   &param->ConfigPhy.polarity,
                                   param->ConfigPhy.numDataLanes);
            };
            break;
        default:
            SIPL_LOG_ERR_STR_INT("MAX9295: Invalid command", (int32_t)parameterType);
            isValidSize = true;
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }

    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_2STR("MAX9295: Command failed", cmdString[parameterType]);
    }

    if (!isValidSize) {
        SIPL_LOG_ERR_STR("MAX9295: Invalid parameter size");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return status;
}
#if !NV_IS_SAFETY
NvMediaStatus
MAX9295DumpRegisters(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint32_t i = 0u;
    _DriverHandle *drvHandle = NULL;
    DevBlkCDII2CReg dumpRegs[] = {{0x0000, 0x00}};

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null handle passed to MAX9295DumpRegisters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver handle passed to MAX9295DumpRegisters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    for (i = 0u; i <= MAX9295_REG_MAX_ADDRESS; i++) {
        dumpRegs[0].address = (uint16_t) i;
        status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, dumpRegs[0].address, (uint8_t *) &(dumpRegs[0].data));
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_2INT("MAX9295: Failed to dump register", (int32_t)i, (int32_t)status);
            return status;
        }

        SIPL_LOG_ERR_STR_2INT("MAX9295: Regsiter has value", (int32_t)dumpRegs[0].address, (int32_t)dumpRegs[0].data);
    }

    return status;
}
#endif
NvMediaStatus
MAX9295CheckPresence(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    RevisionMAX9295 rev = CDI_MAX9295_INVALID_REV;
    uint8_t devID = 0u;
    _DriverHandle *drvHandle = NULL;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null handle passed to MAX9295CheckPresence");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver handle passed to MAX9295CheckPresence");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, REG_DEV_ID_ADDR, &devID);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Read device ID failed", (int32_t)status);
        goto done;
    };

    if ((devID != MAX9295A_DEV_ID) && devID != MAX9295B_DEV_ID) {
        SIPL_LOG_ERR_STR_2INT("MAX9295: Device ID mismatch (expected, expected)", MAX9295A_DEV_ID, MAX9295B_DEV_ID);
        SIPL_LOG_ERR_STR_INT("MAX9295: Device ID mismatch (returned)", (int32_t)devID);
        status = NVMEDIA_STATUS_NOT_SUPPORTED;
        goto done;
    }

    status = GetRevId(handle,
                      &rev);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: GetRevId failed", (int32_t)status);
        goto done;
    }

done:
    return status;
}

NvMediaStatus
MAX9295ReadErrorStatus(
    DevBlkCDIDevice *handle,
    uint32_t dataLength,
    uint8_t *dataBuff)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null handle passed to MAX9295ReadErrorStatus");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX9295: Null driver handle passed to MAX9295ReadErrorStatus");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, REG_LFLT_INT, dataBuff);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: Read error status failed", (int32_t)status);
        goto done;
    };

done:
    return status;
}

static DevBlkCDIDeviceDriver deviceDriver = {
    .deviceName = "Maxim 9295 Serializer",
    .regLength = MAX9295_NUM_ADDR_BYTES,
    .dataLength = MAX9295_NUM_DATA_BYTES,
    .DriverCreate = DriverCreate,
    .DriverDestroy = DriverDestroy,
};

DevBlkCDIDeviceDriver *
GetMAX9295Driver(
    void)
{
    return &deviceDriver;
}
