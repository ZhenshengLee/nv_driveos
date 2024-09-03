/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <stdlib.h>
#include <string.h>

#include "devblk_cdi.h"
#include "devblk_cdi_i2c.h"
#include "cdi_max25614.h"
#include "os_common.h"
#include "sipl_error.h"

#define REGISTER_ADDRESS_LENGTH  1
#define REGISTER_DATA_LENGTH 1

#define VCSEL_SIZE_BYTES 16U

#define REG_DEV_ID       0U
#define REG_REV_ID       1U
#define REG_POWER_SETUP  2U
#define REG_SET_ILED     3U
#define REG_SET_IBOOST   4U
#define REG_SET_VBOOST   5U
#define REG_SET_TSLEW    6U
#define REG_SET_VLED_MAX 8U
#define REG_SET_TON_MAX  9U

#define ILED_MAX        0x7FU
#define IBOOST_MAX      0x7FU
#define VBOOST_MAX      0x7FU
#define TSLEW_MAX       0x7FU
#define TON_MAX_MAX     0x7FU //Max allowable value for the TOn Max setting

#define MAX25614_DEV_ID 1U
#define MAX25614_REV_ID 1U

typedef struct {
    DevBlkCDII2CPgmr    i2cProgrammer;
} DriverHandleMAX25614;


static
DriverHandleMAX25614* getHandlePrivMAX25614(DevBlkCDIDevice const* handle)
{
    DriverHandleMAX25614* drvHandle;

    if (NULL != handle) {
        /* coverity[misra_c_2012_rule_11_5_violation] */
        drvHandle = (DriverHandleMAX25614 *)handle->deviceDriverHandle;
    } else {
        drvHandle = NULL;
    }
    return drvHandle;
}

static NvMediaStatus
DriverCreateMAX25614(
    DevBlkCDIDevice *handle,
    void const* clientContext)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX25614 *drvHandle = NULL;

    if (NULL == handle) {
        SIPL_LOG_ERR_STR("Bad Parameter");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else if (NULL != clientContext) {
        /* clientContext should be NULL, there are no settable parameters for the VCSEL device */
        SIPL_LOG_ERR_STR("Context must not be supplied");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        /* allocation MAX25614VCSEL drvHandle */
        /* coverity[misra_c_2012_rule_11_5_violation] */
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
                SIPL_LOG_ERR_STR("Failed initialize the I2C Programmer");
                status = NVMEDIA_STATUS_ERROR;
            }
        }
    }
    return status;
}

static NvMediaStatus
DriverDestroyMAX25614(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX25614 const* drvHandle = getHandlePrivMAX25614(handle);

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
MAX25614ReadRegister(
    DevBlkCDIDevice const* handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX25614 const* drvHandle = getHandlePrivMAX25614(handle);

    (void) deviceIndex; // unused parameter

    if ((NULL == drvHandle) || (NULL == dataBuff)) {
        SIPL_LOG_ERR_STR("Bad Parameter");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else if ((dataLength + registerNum) > VCSEL_SIZE_BYTES) {
        SIPL_LOG_ERR_STR_HEX_UINT("Overflow for read bytes", (dataLength + registerNum));
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        for (uint32_t i = 0U; i < dataLength; i++) {
            status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                               (uint16_t)(registerNum + i),
                                               &dataBuff[i]);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("failed with status", (int32_t)status);
                break;
            }
        }
    }
    return status;
}

NvMediaStatus
MAX25614WriteRegister(
    DevBlkCDIDevice const* handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t const* dataBuff)
{

    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX25614 const* drvHandle = getHandlePrivMAX25614(handle);

    (void) deviceIndex; // unused parameter

    if ((NULL == drvHandle) || (NULL == dataBuff)) {
        SIPL_LOG_ERR_STR("Bad Parameter");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else if ((dataLength + registerNum) > VCSEL_SIZE_BYTES) {
        SIPL_LOG_ERR_STR_HEX_UINT("Overflow for write bytes", (dataLength + registerNum));
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        for (uint32_t i = 0U; i < dataLength; i++) {
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                               (uint16_t)(registerNum + i),
                                               dataBuff[i]);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("failed with status", (int32_t)status);
                break;
            }
        }
    }
    return status;

}

static NvMediaStatus
MAX25614SetIBOOST(
    DevBlkCDIDevice const* handle, uint8_t iBoost)
{
    // TODO: IBOOST set originally at 0xFU caused 0x5B sensor to stop streaming presumed to be a
    // power issue, thus currently allowing it to be set by client.
    NvMediaStatus status;

    if (NULL == handle) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = MAX25614WriteRegister(handle,
                                       0,
                                       REG_SET_IBOOST,
                                       1,
                                       &iBoost);
    }

    return status;
}

static NvMediaStatus
MAX25614SetVBOOST(
    DevBlkCDIDevice const* handle, uint8_t vBoost)
{
    NvMediaStatus status;

    if (NULL == handle) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = MAX25614WriteRegister(handle,
                                       0,
                                       REG_SET_VBOOST,
                                       1,
                                       &vBoost);
    }

    return status;
}

static NvMediaStatus
MAX25614SetILED(
    DevBlkCDIDevice const* handle, uint8_t iLed)
{
    NvMediaStatus status;

    if (NULL == handle) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = MAX25614WriteRegister(handle,
                                       0,
                                       REG_SET_ILED,
                                       1,
                                       &iLed);
    }

    return status;
}

static NvMediaStatus
MAX25614SetTSlew(
    DevBlkCDIDevice const* handle, uint8_t tSlew)
{
    NvMediaStatus status;

    if (NULL == handle) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = MAX25614WriteRegister(handle,
                                       0,
                                       REG_SET_TSLEW,
                                       1,
                                       &tSlew);
    }

    return status;
}

static NvMediaStatus
MAX25614SetVLEDMAX(
    DevBlkCDIDevice const* handle, uint8_t vLedMax)
{
    NvMediaStatus status;

    if (NULL == handle) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = MAX25614WriteRegister(handle,
                                       0,
                                       REG_SET_VLED_MAX,
                                       1,
                                       &vLedMax);
    }

    return status;
}

static NvMediaStatus
MAX25614SetTOnMax(
    DevBlkCDIDevice const* handle, uint8_t tOnMax)
{
    NvMediaStatus status;

    if (NULL == handle) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = MAX25614WriteRegister(handle,
                                       0,
                                       REG_SET_TON_MAX,
                                       1,
                                       &tOnMax);
    }

    return status;
}

static NvMediaStatus
MAX25614Start(
    DevBlkCDIDevice const* handle)
{
    uint8_t start;
    uint8_t val = 0xE1U;
    NvMediaStatus status;

    if (NULL == handle) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = MAX25614WriteRegister(handle,
                                       0,
                                       REG_POWER_SETUP,
                                       1,
                                       &val);
        if (status == NVMEDIA_STATUS_OK) {
            status = MAX25614ReadRegister(handle,
                                          0,
                                          REG_POWER_SETUP,
                                          1,
                                          &start);
            if (status == NVMEDIA_STATUS_OK) {
                if (start != 0xE1) {
                    status = NVMEDIA_STATUS_ERROR;
                }
            }
        }
    }

    return status;
}

NvMediaStatus
MAX25614Init(
    DevBlkCDIDevice const* handle, MAX25614Params params)
{
    NvMediaStatus status;

    if ( (NULL == handle)              ||
         (params.iBoost > IBOOST_MAX)  ||
         (params.vBoost > VBOOST_MAX)  ||
         (params.iLed   > ILED_MAX)    ||
         (params.tOnMax > TON_MAX_MAX) ||
         (params.tSlew > TSLEW_MAX) )
    {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = MAX25614SetIBOOST(handle, params.iBoost);
        if (status == NVMEDIA_STATUS_OK) {
            status = MAX25614SetVBOOST(handle, params.vBoost);
            if (status == NVMEDIA_STATUS_OK) {
                status = MAX25614SetILED(handle, params.iLed);
                if (status == NVMEDIA_STATUS_OK) {
                    status = MAX25614SetTOnMax(handle, params.tOnMax);
                    if (status == NVMEDIA_STATUS_OK) {
                        status = MAX25614SetTSlew(handle, params.tSlew);
                        if (status == NVMEDIA_STATUS_OK) {
                            status = MAX25614SetVLEDMAX(handle, params.vLedMax);
                            if (status == NVMEDIA_STATUS_OK) {
                                status = MAX25614Start(handle);
                            }
                        }
                    }
                }
            }
        }
    }

    return status;
}

NvMediaStatus
MAX25614CheckPresence(
    DevBlkCDIDevice const* handle)
{
    uint8_t dev_id = 0U;
    uint8_t rev_id = 0U;
    NvMediaStatus status;

    if (NULL == handle) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = MAX25614ReadRegister(handle,
                                      0,
                                      REG_DEV_ID,
                                      1,
                                      &dev_id);
        if (status == NVMEDIA_STATUS_OK) {
            status = MAX25614ReadRegister(handle,
                                          0,
                                          REG_REV_ID,
                                          1,
                                          &rev_id);
        }
    }

    if (status == NVMEDIA_STATUS_OK) {
        if ((dev_id == MAX25614_DEV_ID) && (rev_id == MAX25614_REV_ID)) {
            status = NVMEDIA_STATUS_OK;
        } else {
            status = NVMEDIA_STATUS_ERROR;
        }
    }

    return status;
}

DevBlkCDIDeviceDriver *
GetMAX25614Driver(
    void)
{
    /** Structure that holds function pointers for the driver operations */
    static DevBlkCDIDeviceDriver deviceDriverMAX25614 = {
        /** Name of VCSEL chip */
        .deviceName = "MAX25614",

        /** Number of bytes in register address */
        .regLength = REGISTER_ADDRESS_LENGTH,

        /** Number of bytes in register data */
        .dataLength = REGISTER_DATA_LENGTH,

        /** Function pointer to create driver function */
        .DriverCreate = DriverCreateMAX25614,

        /** Function pointer to destroy driver function */
        .DriverDestroy = DriverDestroyMAX25614,

        /** Function pointer to read from VCSEL function */
        .ReadRegister = MAX25614ReadRegister,

        /** Function pointer to write to VCSEL function */
        .WriteRegister = MAX25614WriteRegister,
    };

    return &deviceDriverMAX25614;
}
