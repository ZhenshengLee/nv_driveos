/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <stdlib.h>
#include <string.h>

#include "devblk_cdi.h"
#include "devblk_cdi_i2c.h"
#include "cdi_max20087.h"
#include "os_common.h"
#include "sipl_error.h"

#define REGISTER_ADDRESS_LENGTH  1
#define REGISTER_DATA_LENGTH 1

#define REG_MASK        0U
#define REG_CONFIG      1U
#define REG_DEV_REV_ID  2U
#define MAX20087_DEV_ID 0x20U
#define MAX20087_REV_ID 0x1U

typedef struct {
    DevBlkCDII2CPgmr    i2cProgrammer;
} DriverHandleMAX20087;

static
DriverHandleMAX20087* getHandlePrivMAX20087(DevBlkCDIDevice const* handle)
{
    DriverHandleMAX20087* drvHandle;

    if (NULL != handle) {
        drvHandle = (DriverHandleMAX20087 *)handle->deviceDriverHandle;
    } else {
        drvHandle = NULL;
    }
    return drvHandle;
}

static NvMediaStatus
MAX20087DriverCreate(
    DevBlkCDIDevice *handle,
    void const* clientContext)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX20087 *drvHandle = NULL;

    if (NULL == handle) {
        SIPL_LOG_ERR_STR("Bad Parameter");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else if (NULL != clientContext) {
        SIPL_LOG_ERR_STR("Context must not be supplied");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        drvHandle = calloc(1U, sizeof(*drvHandle));
        if (NULL == drvHandle) {
            SIPL_LOG_ERR_STR_HEX_UINT("memory allocation failed", (uint32_t)status);
            status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        } else {
            handle->deviceDriverHandle = (void *)drvHandle;

            /* Create the I2C Programmer for register read/write */
            drvHandle->i2cProgrammer = DevBlkCDII2CPgmrCreate(handle,
                                                              REGISTER_ADDRESS_LENGTH,
                                                              REGISTER_DATA_LENGTH);
            if (NULL == drvHandle->i2cProgrammer) {
                free(drvHandle);
                handle->deviceDriverHandle = NULL;
                SIPL_LOG_ERR_STR("Failed initialize the I2C Programmer");
                status = NVMEDIA_STATUS_ERROR;
            }
        }
    }
    return status;
}

static NvMediaStatus
MAX20087DriverDestroy(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX20087 const* drvHandle = getHandlePrivMAX20087(handle);

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("Bad Parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        /* Destroy the I2C Programmer */
        DevBlkCDII2CPgmrDestroy(drvHandle->i2cProgrammer);

        free(handle->deviceDriverHandle);
        handle->deviceDriverHandle = NULL;
    }
    return status;
}

NvMediaStatus
MAX20087ReadRegister(
    DevBlkCDIDevice const *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX20087 const* drvHandle = getHandlePrivMAX20087(handle);

    (void) deviceIndex; // unused parameter

    if ((NULL == drvHandle) || (NULL == dataBuff)) {
        SIPL_LOG_ERR_STR("Bad Parameter");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    for (uint32_t i = 0U; i < dataLength; i++) {
        status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                           (uint16_t)(registerNum + i),
                                           &dataBuff[i]);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("failed with status", (int32_t)status);
            break;
        }
    }

    return status;
}

NvMediaStatus
MAX20087WriteRegister(
    DevBlkCDIDevice const *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t const *dataBuff)
{

    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX20087 const* drvHandle = getHandlePrivMAX20087(handle);

    (void) deviceIndex; // unused parameter

    if ((NULL == drvHandle) || (NULL == dataBuff)) {
        SIPL_LOG_ERR_STR("Bad Parameter");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    for (uint32_t i = 0U; i < dataLength; i++) {
        status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                           (uint16_t)(registerNum + i),
                                           dataBuff[i]);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("failed with status", (int32_t)status);
            break;
        }
    }

    return status;
}

NvMediaStatus
MAX20087Init(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t data = 0;

    if (NULL == handle) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* Mask UV event */
    data = 0x1;
    status = MAX20087WriteRegister(handle,
                                   0,
                                   REG_MASK,
                                   REGISTER_DATA_LENGTH,
                                   &data);
    if (status != NVMEDIA_STATUS_OK) {
        goto exit;
    }

    /* Disable power for all links*/
    status = MAX20087ReadRegister(handle,
                                  0,
                                  REG_CONFIG,
                                  REGISTER_DATA_LENGTH,
                                  &data);
    if (status != NVMEDIA_STATUS_OK) {
        goto exit;
    }

    data &= 0xF0;
    status = MAX20087WriteRegister(handle,
                                   0,
                                   REG_CONFIG,
                                   REGISTER_DATA_LENGTH,
                                   &data);

exit:
    return status;
}

NvMediaStatus
MAX20087CheckPresence(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t dev_rev_id = 0U;

    if (handle == NULL) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto exit;
    }

    status = MAX20087ReadRegister(handle,
                                  0,
                                  REG_DEV_REV_ID,
                                  1,
                                  &dev_rev_id);
    if (status != NVMEDIA_STATUS_OK) {
        goto exit;
    }

    if (((dev_rev_id & 0x30) != MAX20087_DEV_ID) ||
        ((dev_rev_id & 0xF) != MAX20087_REV_ID)) {
        SIPL_LOG_ERR_STR("Dev/Rev ID not match");
        status = NVMEDIA_STATUS_ERROR;
    }

exit:
    return status;
}

NvMediaStatus
MAX20087SetLinkPower(
    DevBlkCDIDevice const* handle,
    uint8_t const linkIndex,
    bool const enable)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t data;

    status = MAX20087ReadRegister(handle,
                                  0,
                                  REG_CONFIG,
                                  REGISTER_DATA_LENGTH,
                                  &data);
    if (status != NVMEDIA_STATUS_OK) {
        goto exit;
    }

    if (enable) {
        data |= (1 << linkIndex);
    } else {
        data &= ~(1 << linkIndex);
    }
    status = MAX20087WriteRegister(handle,
                                   0,
                                   REG_CONFIG,
                                   REGISTER_DATA_LENGTH,
                                   &data);
exit:
    return status;
}

static DevBlkCDIDeviceDriver deviceDriver = {
    .deviceName = "MAX20087 Power Load Switch",
    .regLength = REGISTER_ADDRESS_LENGTH,
    .dataLength = REGISTER_DATA_LENGTH,
    .DriverCreate = MAX20087DriverCreate,
    .DriverDestroy = MAX20087DriverDestroy,
    .ReadRegister = MAX20087ReadRegister,
    .WriteRegister = MAX20087WriteRegister,
};

DevBlkCDIDeviceDriver *
GetMAX20087Driver(void)
{
    return &deviceDriver;
}
