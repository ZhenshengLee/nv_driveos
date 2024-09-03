/*
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <stdlib.h>
#include <string.h>

#include "devblk_cdi.h"
#include "cdi_n24c64.h"
#include "os_common.h"

#define REGISTER_ADDRESS_LENGTH  2
#define REGISTER_DATA_LENGTH 1

#define MAXIMUM_READ_SIZE 128
#define EEPROM_PAGE_SIZE_BYTES 32
#define EEPROM_SIZE_BYTES 8192
#define FIRST_ADDRESS 0

#define MICROSECONDS_BETWEEN_WRITES 4000

#define MIN(a,b)            (((a) < (b)) ? (a) : (b))

static NvMediaStatus
DriverCreate(
    DevBlkCDIDevice *handle,
    void const* clientContext)
{
    if (NULL == handle) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* This should be NULL, there are no settable parameters for the EEPROM device */
    if (clientContext) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    handle->deviceDriverHandle = NULL;

    return NVMEDIA_STATUS_OK;
}

static NvMediaStatus
DriverDestroy(
    DevBlkCDIDevice *handle)
{
    if (NULL == handle) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
N24C64ReadRegister(
    DevBlkCDIDevice const* handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff)
{
    uint8_t address[REGISTER_ADDRESS_LENGTH];
    uint32_t bytesToRead;
    uint32_t totalBytesRead = 0;
    NvMediaStatus status;

    if ((NULL == handle) || (NULL == dataBuff)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (dataLength + registerNum > EEPROM_SIZE_BYTES) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    while (totalBytesRead < dataLength) {
        bytesToRead = MIN(dataLength - totalBytesRead, MAXIMUM_READ_SIZE);

        address[1] = (registerNum + totalBytesRead) & 0xFF;
        address[0] = (registerNum + totalBytesRead) >> 8;

        status = DevBlkCDIDeviceRead(handle,
            deviceIndex,    /* device index */
            REGISTER_ADDRESS_LENGTH, /* regLength */
            address,        /* regData */
            bytesToRead,     /* dataLength */
            &dataBuff[totalBytesRead]);      /* data */

        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }

        totalBytesRead += bytesToRead;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
N24C64WriteRegister(
    DevBlkCDIDevice const* handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t const* dataBuff)
{
    uint8_t buffer[REGISTER_ADDRESS_LENGTH + EEPROM_PAGE_SIZE_BYTES];
    uint8_t * const address = buffer;
    uint8_t * const data = &buffer[REGISTER_ADDRESS_LENGTH];
    NvMediaStatus status;

    if (NULL == handle || NULL == dataBuff) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (dataLength + registerNum > EEPROM_SIZE_BYTES) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }


    /* The EEPROM can only write to one page at a time.  If one attempts to write past */
    /* the edge of a page boundary, the write will need to be split into multiple I2C transactions */
    while (dataLength > 0) {
        uint32_t bytesToWrite = MIN(dataLength, EEPROM_PAGE_SIZE_BYTES - (registerNum % EEPROM_PAGE_SIZE_BYTES));

        address[0]  = registerNum >> 8;
        address[1] = registerNum & 0xFF;
        memcpy(data, dataBuff, bytesToWrite);

        status = DevBlkCDIDeviceWrite(handle,
            deviceIndex,
            bytesToWrite + REGISTER_ADDRESS_LENGTH,
            buffer);

        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }

        dataLength -= bytesToWrite;
        registerNum += bytesToWrite;
        dataBuff += bytesToWrite;

        nvsleep(MICROSECONDS_BETWEEN_WRITES);
    }

    return status;
}

NvMediaStatus
N24C64CheckPresence(
    DevBlkCDIDevice *handle)
{
    uint8_t data[1] = {0};
    NvMediaStatus status;

    if (NULL == handle) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* Attempt to read the first byte on the EEPROM chip */
    status = N24C64ReadRegister(
        handle,
        0,
        FIRST_ADDRESS,
        1,
        data
    );

    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }
    return NVMEDIA_STATUS_OK;
}

static DevBlkCDIDeviceDriver deviceDriver = {
    .deviceName = "N24C64",
    .regLength = REGISTER_ADDRESS_LENGTH,
    .dataLength = REGISTER_DATA_LENGTH,
    .DriverCreate = DriverCreate,
    .DriverDestroy = DriverDestroy,
    .ReadRegister = N24C64ReadRegister,
    .WriteRegister = N24C64WriteRegister,
};

DevBlkCDIDeviceDriver *
GetN24C64Driver(
    void)
{
    return &deviceDriver;
}
