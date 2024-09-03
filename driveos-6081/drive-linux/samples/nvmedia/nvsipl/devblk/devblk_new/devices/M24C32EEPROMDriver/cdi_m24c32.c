/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <stdlib.h>
#include <string.h>

#include "cdi_m24c32.h"
#include "os_common.h"
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
#include "log_utils.h"
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
#include "devblk_cdi_i2c.h"
#include "sipl_error.h"

/** size of M24C32 address in bytes */
#define M24C32_REGISTER_ADDRESS_LENGTH   2

/** size of M24C32 data register in bytes */
#define M24C32_REGISTER_DATA_LENGTH      1

/** M24C32's EEPROM size in bytes */
#define M24C32_EEPROM_SIZE_BYTES         4096U

/** first address of M24C32 */
#define M24C32_FIRST_ADDRESS             0U

/** M24C32's device driver handle structure */
typedef struct {
    /** pointer for I2C programmer function */
    DevBlkCDII2CPgmr i2cProgrammer;
} DriverHandleM24C32;

/** @brief returns the device driver handle from DEV BLK handle
 *
 * Implements
 * - verify handle is not NULL
 * - extracts device driver handle from handle
 * @param[in] handle DEVBLK handle
 * @return device driver handle
 * @return NULL if DEVBLK handle is NULL  */
static DriverHandleM24C32 *
getHandlePrivM24C32(DevBlkCDIDevice const *handle)
{
    DriverHandleM24C32 *drvHandle;

    if (NULL == handle) {
        drvHandle = NULL;
    } else {
        /* coverity[misra_c_2012_rule_11_5_violation] : intentional TID-1417 */
        drvHandle = (DriverHandleM24C32 *)handle->deviceDriverHandle;
    }
    return drvHandle;
}

/** @brief creates an instance of M24C32 device driver
 *
 * Implements:
 * - verify handle is not NULL
 * - verifiy that clientContext is NOT NULL
 *  - clientContext should be NULL, there are no settable parameters for the EEPROM device
 *  .
 * - alllocate memory for device driver handle by
 *  - drvHandle =(DriverHandleM24C32 *)calloc(1, sizeof(DriverHandleM24C32))
 *  .
 * - verify memory was allocated
 * - save new device driver handle in handle by
 *  - handle->deviceDriverHandle = (void *)drvHandle
 *  .
 * - Create the I2C programmer for register read/write by
 *  - drvHandle->i2cProgrammer = DevBlkCDII2CPgmrCreate(
 *   - handle,
 *   - (uint8_t)M24C32_REGISTER_ADDRESS_LENGTH,
 *   - (uint8_t)M24C32_REGISTER_DATA_LENGTH)
 *   .
 *  .
 * - verify drvHandle->i2cProgrammer is NOT NULL
 * @param[in] handle DEV BLK handle
 * @param[in] clientContext caller context pointer , not used
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code
 * when any of the called functions returns error */
static NvMediaStatus
DriverCreateM24C32(
    DevBlkCDIDevice *handle,
    void const* clientContext)
{
    DriverHandleM24C32 *drvHandle = NULL;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (NULL == handle) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        SIPL_LOG_ERR_STR("M24C32: Bad parameter: Null ptr");
    } else if(clientContext != NULL) {
    /* clientContext should be NULL, there are no settable parameters for the EEPROM device */
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        SIPL_LOG_ERR_STR("Bad input parameter");
    } else {
        /* coverity[misra_c_2012_rule_11_5_violation] : intentional TID-1417 */
        /* coverity[misra_c_2012_rule_21_3_violation] : intentional TID-1493 */
        drvHandle =(DriverHandleM24C32 *)calloc(1, sizeof(DriverHandleM24C32));
        if (drvHandle == NULL) {
            status = NVMEDIA_STATUS_OUT_OF_MEMORY;
            SIPL_LOG_ERR_STR("Unable allocate memory to driver handle");
        } else {
            handle->deviceDriverHandle = (void *)drvHandle;

            /* Create the I2C programmer for register read/write */
            drvHandle->i2cProgrammer = DevBlkCDII2CPgmrCreate(handle,
                                                              (uint8_t)M24C32_REGISTER_ADDRESS_LENGTH,
                                                              (uint8_t)M24C32_REGISTER_DATA_LENGTH);
            if(drvHandle->i2cProgrammer == NULL) {
                status = NVMEDIA_STATUS_ERROR;
                SIPL_LOG_ERR_STR("M24C32: Failed to initialize the I2C programmer");
            }
        }
    }
    return status;
}

/** @brief destroys the instance of the M24C32 device driver handle
 *
 * Implements:
 * - extracts device driver handle from handle
 * - verify device driver handle is not NULL
 * - Destroy the I2C programmer by
 *  - DevBlkCDII2CPgmrDestroy(drvHandle->i2cProgrammer)
 *  .
 * - free the device handle memory by
 *  - free(handle->deviceDriverHandle)
 *  .
 * - clear device driver handle in handle
 *  - handle->deviceDriverHandle = NULL
 *  .
 * @param[in] handle DEV BLK handle
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code
 * when any of the called functions returns error */
static NvMediaStatus
DriverDestroyM24C32(
  DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleM24C32 const* drvHandle = getHandlePrivM24C32(handle);

    if (NULL == drvHandle) {
        status =  NVMEDIA_STATUS_BAD_PARAMETER;
        SIPL_LOG_ERR_STR("M24C32: Bad parameter: Null ptr");
    } else {
        /* Destroy the I2C programmer */
        DevBlkCDII2CPgmrDestroy(drvHandle->i2cProgrammer);
        /* coverity[misra_c_2012_rule_21_3_violation] : intentional TID-1493 */
        free(handle->deviceDriverHandle);
        handle->deviceDriverHandle = NULL;
    }

    return status;
}

/** @brief M24C32 read register, to get EEPROM content
 *
 * Implements:
 * - extracts device driver handle from handle
 * - verify device driver handle is not NULL
 * - verify dataBuff is not NULL
 * - verify eeprom memory  address calculation will not roll over by
 *  - (((uint64_t)dataLength + registerNum) <= 0xFFFFFFFFU))
 *  .
 * - verify that register address calculation within allowed range by
 *  - ((dataLength + registerNum) <= M24C32_EEPROM_SIZE_BYTES)
 *  .
 * - read data from eeprom by
 *  - status = DevBlkCDII2CPgmrReadBlock(
 *   - drvHandle->i2cProgrammer,
 *   - (uint16_t)registerNum,
 *   - (uint16_t)dataLength,
 *   - dataBuff)
 *   .
 *  .
 * @param[in] handle DEV BLK handle
 * @param[in] deviceIndex device index
 * @param[in] registerNum register number
 * @param[in] dataLength data length to be read
 * @param[out] dataBuff caller buffer where data is to be stored in
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code
 * when any of the called functions returns error */
static NvMediaStatus
M24C32ReadRegister(
    DevBlkCDIDevice const* handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleM24C32 const* drvHandle = getHandlePrivM24C32(handle);
    (void)deviceIndex;

    if ((NULL == drvHandle) ||
        (NULL == dataBuff)  ||
        (((uint64_t)dataLength + registerNum) > 0xFFFFFFFFU)) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        SIPL_LOG_ERR_STR("M24C32: Bad parameter");
    } else if ((dataLength + registerNum) > M24C32_EEPROM_SIZE_BYTES) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        SIPL_LOG_ERR_STR_HEX_UINT("M24C32: Overflow for read bytes", (dataLength + registerNum));
    } else {
        status = DevBlkCDII2CPgmrReadBlock(drvHandle->i2cProgrammer,
                                           (uint16_t)registerNum,
                                           (uint16_t)dataLength,
                                           dataBuff);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("M24C32: EEPROM read failed", (uint32_t)status);
        } else {
            uint8_t  eepromCmpBytes[M24C32_EEPROM_SIZE_BYTES];
            /* Now read again because it has been ordained */
            status = DevBlkCDII2CPgmrReadBlock(drvHandle->i2cProgrammer,
                                               (uint16_t)registerNum,
                                               (uint16_t)dataLength,
                                               &eepromCmpBytes[0]);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("M24C32: Read failed with status", (int32_t)status);
            } else {
                /* Compare if both are equal */
                if (0 != memcmp(eepromCmpBytes, dataBuff, dataLength)) {
                    status = NVMEDIA_STATUS_ERROR;
                    SIPL_LOG_ERR_STR("EEPROM Read mismatch");
                }
            }
        }
    }
    return status;
}

/** function description is in header file */
NvMediaStatus
M24C32CheckPresence(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleM24C32 const* drvHandle = getHandlePrivM24C32(handle);
    uint8_t data = 0U;

    if (NULL == drvHandle) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        SIPL_LOG_ERR_STR("Bad input parameter");
    } else {
        /* Attempt to read the first byte on the EEPROM chip */
        status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                           (uint16_t)M24C32_FIRST_ADDRESS,
                                           &data);

        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("M24C32: Check presence failed");
        }
    }
    return status;
}

/** function description is in header file */
DevBlkCDIDeviceDriver *
GetM24C32Driver(
    void)
{
    static DevBlkCDIDeviceDriver deviceDriverM24C32 = {
        .deviceName = "M24C32",
        .regLength = M24C32_REGISTER_ADDRESS_LENGTH,
        .dataLength = M24C32_REGISTER_DATA_LENGTH,
        .DriverCreate = DriverCreateM24C32,
        .DriverDestroy = DriverDestroyM24C32,
        .ReadRegister = M24C32ReadRegister,
    };
    return &deviceDriverM24C32;
}
