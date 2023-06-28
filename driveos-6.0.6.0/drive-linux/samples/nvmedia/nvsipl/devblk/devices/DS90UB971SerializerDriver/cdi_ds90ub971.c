/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#if !NV_IS_SAFETY
#include "log_utils.h"
#endif
#include "sipl_error.h"
#include "devblk_cdi.h"
#include "cdi_ds90ub971.h"
#include "cdi_ds90ub971_setting.h"
#include "os_common.h"

#define GET_BLOCK_LENGTH(x)         x[0]
#define GET_SIZE(x)                 (sizeof(x)/sizeof(x[0]))
#define GET_BLOCK_DATA(x)           &x[1]
#define SET_NEXT_BLOCK(x)           x += (x[0] + 1)

#define DS90UB971_NUM_ADDR_BYTES    (1U)
#define DS90UB971_NUM_DATA_BYTES    (1U)
#define REG_WRITE_BUFFER_BYTES      (DS90UB971_NUM_DATA_BYTES)
#define DS90UB971_CDI_DEVICE_INDEX  (0U)

#define REG_DEV_ID_ADDR             (0x00)
#define DS90UB971A_DEV_ID           (0x32)
#define REG_DEV_REV_ADDR            (0x50)
#define DS90UB971_REG_MAX_ADDRESS   (0x1576)

typedef struct {
    RevisionDS90UB971 revId;
    uint32_t revVal;
} Revision;

/* These values must include all of values in the RevisionDS90UB971 enum */
static Revision supportedRevisions[] = {
    {CDI_DS90UB971_REV_1, 0x00u},
    {CDI_DS90UB971_REV_2, 0x10u},
};

static NvMediaStatus
GetRevId(
    DevBlkCDIDevice *handle,
    RevisionDS90UB971 *rev)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint32_t numRev = sizeof(supportedRevisions) / sizeof(supportedRevisions[0]);
    uint8_t revision = 0u;
    uint32_t i = 0u;

    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                       REG_DEV_REV_ADDR,
                                       &revision);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    for (i = 0u; i < numRev; i++) {
        if (revision == supportedRevisions[i].revVal) {
            *rev = supportedRevisions[i].revId;
            LOG_MSG("DS90UB971: Revision %u detected!\n", revision);
            return NVMEDIA_STATUS_OK;
        }
    }

    SIPL_LOG_ERR_STR_INT("DS90UB971: Unsupported DS90UB971 revision detected",
        (int32_t)revision);

    return NVMEDIA_STATUS_NOT_SUPPORTED;
}

static NvMediaStatus
DriverCreate(
    DevBlkCDIDevice *handle,
    void const* clientContext)
{
    _DriverHandle *drvHandle = NULL;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB971: Null handle passed to DriverCreate");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (clientContext != NULL) {
        SIPL_LOG_ERR_STR("DS90UB971: Context must not be supplied");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = calloc(1, sizeof(_DriverHandle));
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB971: Memory allocation for context failed");
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }

    // Create the I2C programmer for register read/write
    drvHandle->i2cProgrammer = DevBlkCDII2CPgmrCreate(handle,
                                                      DS90UB971_NUM_ADDR_BYTES,
                                                      DS90UB971_NUM_DATA_BYTES);
    if(drvHandle->i2cProgrammer == NULL) {
        SIPL_LOG_ERR_STR("DS90UB971: Failed to initialize the I2C programmer\n");
        free(drvHandle);
        return NVMEDIA_STATUS_ERROR;
    }

    handle->deviceDriverHandle = (void *)drvHandle;

    return NVMEDIA_STATUS_OK;

}

static NvMediaStatus
DriverDestroy(
    DevBlkCDIDevice *handle)
{
    _DriverHandle *drvHandle = NULL;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB971: Null handle passed to DriverDestroy");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB971: Null driver handle passed to DriverDestroy");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    // Destroy the I2C programmer
    DevBlkCDII2CPgmrDestroy(drvHandle->i2cProgrammer);

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
DS90UB971SetDefaults(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    if (handle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB971: Null handle passed to DS90UB971SetDefaults");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer, // Reset ser with registers
                                        0x01,
                                        0x01);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("DS90UB971: Failed to write to serializer device");
    }

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer, // Set CSI 4 lane continuous clock
                                        0x02,
                                        0x73);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("DS90UB971: Failed to write to serializer device");
    }

    return status;
}

static NvMediaStatus
SetTPG(
    DevBlkCDIDevice *handle,
    uint32_t height,
    uint32_t width,
    uint32_t frameRate)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if ((width == 3840) && (height == 1928)) {
        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                            &ds90ub971_pg_setting_1920x1236);
    }

    return status;
}

static NvMediaStatus
SetFsyncGPIO(
    DevBlkCDIDevice *handle,
    GPIODS90UB971 gpio,
    uint8_t rxGpioID)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint16_t regAddr = 0U;
    uint8_t regData = 0U;

    if (gpio >= CDI_DS90UB971_GPIO_NUM) {
        SIPL_LOG_ERR_STR("DS90UB971: Bad parameter: Invalid GPIO pin");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    regAddr = 0x0E;
    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                       regAddr,
                                       &regData);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    regData |= ((1 << gpio) << 4);
    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        regAddr,
                                        regData);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    regAddr = 0x0D;
    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                       regAddr,
                                       &regData);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    regData |= ((rxGpioID << 4) & 0xF0) | (regData & 0x0F);
    return DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                      regAddr,
                                      regData);
}

static NvMediaStatus
setLocalGpioOutput(DevBlkCDIDevice *handle,
    GPIODS90UB971 gpio,
    uint8_t level)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;

    static int runonce = 1;

    DevBlkCDII2CReg ds90ub971_reset_sensor_regs[] = {
        {0x0D, 0x08, 10000},
        {0x0E, 0x96, 10000},
        {0x0D, 0x00, 10000},
        {0x0D, 0x08, 10000},
        {0x0E, 0x87, 10000}
    };

    DevBlkCDII2CRegList ds90ub971_reset_sensor = {
        .regs = ds90ub971_reset_sensor_regs,
        .numRegs = (uint32_t)I2C_ARRAY_SIZE(ds90ub971_reset_sensor_regs),
    };

    if(runonce) {
        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                &ds90ub971_reset_sensor);
        runonce =0;
    }
    return status;
}

static NvMediaStatus
SetClkOut(DevBlkCDIDevice *handle,
    uint8_t N,
    uint8_t M)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;

    DevBlkCDII2CReg ds90ub971_clkout_regs[] = {
        {0x06, M}, /* M value for clockout divider */
        {0x07, N} /* N Value for said divider */
    };

    DevBlkCDII2CRegList ds90ub971_clkout = {
        .regs = ds90ub971_clkout_regs,
        .numRegs = (uint32_t)I2C_ARRAY_SIZE(ds90ub971_clkout_regs),
    };

    if (M <= 0x10 || 1) {
        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                            &ds90ub971_clkout);
    }

    return status;
}

NvMediaStatus
DS90UB971WriteParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    bool isValidSize = false;
    ReadWriteParamsDS90UB971 *param = (ReadWriteParamsDS90UB971 *) parameter;
    static const char *cmdString[] = {
        [CDI_WRITE_PARAM_CMD_DS90UB971_SET_TPG] =
            "CDI_WRITE_PARAM_CMD_DS90UB971_SET_TPG",
        [CDI_WRITE_PARAM_CMD_DS90UB971_SET_CLKOUT] =
            "CDI_WRITE_PARAM_CMD_DS90UB971_SET_CLKOUT",
        [CDI_WRITE_PARAM_CMD_DS90UB971_SET_FSYNC_GPIO] =
            "CDI_WRITE_PARAM_CMD_DS90UB971_SET_FSYNC_GPIO",
    };

    if ((handle == NULL) || (parameter == NULL)) {
        SIPL_LOG_ERR_STR("DS90UB971: Bad parameter: Null ptr");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if ((parameterType == CDI_WRITE_PARAM_CMD_DS90UB971_INVALID) ||
        (parameterType >= CDI_WRITE_PARAM_CMD_DS90UB971_NUM)) {
        SIPL_LOG_ERR_STR("DS90UB971: Bad parameter: Invalid command");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    switch (parameterType) {
        case CDI_WRITE_PARAM_CMD_DS90UB971_SET_TPG:
            if (parameterSize == sizeof(param->TPG)) {
                isValidSize = true;
                status = SetTPG(handle,
                                param->TPG.height,
                                param->TPG.width,
                                param->TPG.frameRate);
           }
           break;
        case CDI_WRITE_PARAM_CMD_DS90UB971_SET_CLKOUT:
            if (parameterSize == sizeof(param->ClkOut)) {
                isValidSize = true;
                status = SetClkOut(handle,
                                param->ClkOut.N,
                                param->ClkOut.M);
           }
           break;
        case CDI_WRITE_PARAM_CMD_DS90UB971_SET_FSYNC_GPIO:
            if (parameterSize == sizeof(param->fsyncGpio)) {
                isValidSize = true;
                status = SetFsyncGPIO(handle,
                                      param->fsyncGpio.gpioInd,
                                      param->fsyncGpio.rxGpioID);
            }
            break;
        case CDI_WRITE_PARAM_CMD_DS90UB971_SET_LOCAL_GPIO:
            if(parameterSize == sizeof(param->localGpio)) {
                isValidSize = true;
                setLocalGpioOutput(handle,
                                        param->localGpio.gpio,
                                        param->localGpio.level);
            }
            break;
        default:
            SIPL_LOG_ERR_STR("DS90UB971: Bad parameter: Invalid command");
            isValidSize = true;
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }

    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_2STR("DS90UB971: Command failed", cmdString[parameterType]);
    }

    if (!isValidSize) {
        SIPL_LOG_ERR_2STR("DS90UB971: Bad parameter: Invalid param size", cmdString[parameterType]);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return status;
}

#if !NV_IS_SAFETY
NvMediaStatus
DS90UB971DumpRegisters(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    uint8_t data = 0;
    uint32_t i = 0u;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB971: Null handle passed to DS90UB971DumpRegisters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB971: Null driver handle passed to DS90UB971DumpRegisters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    for (i = 0u; i <= DS90UB971_REG_MAX_ADDRESS; i++) {
        status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                           ((i / 256u) << 8) | (i % 256u),
                                           &data);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_2INT("DS90UB971: Failed to read register (status)", (int32_t)i, (int32_t)status);
            return status;
        }
    }

    return status;
}
#endif

NvMediaStatus
DS90UB971CheckPresence(
    DevBlkCDIDevice *handle)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    RevisionDS90UB971 rev = CDI_DS90UB971_INVALID_REV;
    uint8_t devID = 0u;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB971: Null handle passed to DS90UB971CheckPresence");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                       REG_DEV_ID_ADDR,
                                       &devID);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("DS90UB971: Failed to read serializer device ID");
        return status;
    }

    if (devID != DS90UB971A_DEV_ID) {
        SIPL_LOG_ERR_STR_2INT("DS90UB971: Device ID mismatch",
            (int32_t)DS90UB971A_DEV_ID, (int32_t)devID);
        return NVMEDIA_STATUS_ERROR;
    }

    status = GetRevId(handle, &rev);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    return NVMEDIA_STATUS_OK;
}

static DevBlkCDIDeviceDriver deviceDriver = {
    .deviceName = "TI DS90UB971 Serializer",
    .regLength = DS90UB971_NUM_ADDR_BYTES,
    .dataLength = DS90UB971_NUM_DATA_BYTES,
    .DriverCreate = DriverCreate,
    .DriverDestroy = DriverDestroy,
};

DevBlkCDIDeviceDriver *
GetDS90UB971Driver(void)
{
    return &deviceDriver;
}
