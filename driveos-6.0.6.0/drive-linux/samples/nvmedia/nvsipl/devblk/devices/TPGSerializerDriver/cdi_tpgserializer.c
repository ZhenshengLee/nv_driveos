/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if !NV_IS_SAFETY
#include "log_utils.h"
#endif
#include "sipl_error.h"
#include "devblk_cdi.h"
#include "cdi_tpgserializer.h"
#include "os_common.h"
#include "TPGSensorSerializerUtility.h"

#define REGISTER_ADDRESS_BYTES  1
#define REG_WRITE_BUFFER        32

#define GET_BLOCK_LENGTH(x) x[0]
#define GET_BLOCK_DATA(x)   &x[1]
#define SET_NEXT_BLOCK(x)   x += (x[0] + 1)

static NvMediaStatus
DriverCreate(
    DevBlkCDIDevice *handle,
    void const* clientContext)
{
    _DriverHandle *driverHandle = NULL;

    if(!handle)
        return NVMEDIA_STATUS_BAD_PARAMETER;

    driverHandle = calloc(1, sizeof(_DriverHandle));
    if (driverHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96705: Memory allocation for context failed");
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }
    handle->deviceDriverHandle = (void *)driverHandle;

    return NVMEDIA_STATUS_OK;
}

static NvMediaStatus
DriverDestroy(
    DevBlkCDIDevice *handle)
{
    _DriverHandle *driverHandle = NULL;

    if((handle == NULL) || ((driverHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (handle->deviceDriverHandle != NULL) {
        free(handle->deviceDriverHandle);
        handle->deviceDriverHandle = NULL;
    }

    return NVMEDIA_STATUS_OK;
}

static DevBlkCDIDeviceDriver deviceDriver = {
    .deviceName = "Maxim 96712 TPG Serializer",
    .regLength = 2,
    .dataLength = 1,
    .DriverCreate = DriverCreate,
    .DriverDestroy = DriverDestroy,
};

DevBlkCDIDeviceDriver *
GetTPGSerializerDriver(void)
{
    return &deviceDriver;
}

size_t
TPGSerializerReadErrorSize(void) {
    /*
        errSize should be the cumulative maximum size (in 16 bit) of all data we'd
        expect to retrieve.
    */
    return (sizeof(uint16_t) * (size_t)TPGSERIALIZER_STATUS_MAX_ERR);
}

NvMediaStatus
TPGSerializerReadErrorData(size_t bufSize,
                           uint8_t* const buffer) {
    /* Define Error Handling REGISTER string LUT */
    char const *TPG_SERIALIZER_REGISTER_NAME[] = {
    TPGSERIALIZER_REGISTER(GENERATE_ENUM_STRING)
    };
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (buffer == NULL) {
        SIPL_LOG_ERR_STR("TPGSerializer: Bad input parameter\n");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }
    else {
        /*
           Read REG_STATUS_xx, STATC_ERR reigsters to check which error trigger
           SYS_CHECK
           note that here we'd do register reads as needed and add all data to the
           provided buffer in whatever order is determined with and communicated to
           the client who will be reading the buffer

           Mimic REG_ASIL by file I/O
        */
        if (((size_t)TPGSERIALIZER_STATUS_MAX_ERR * sizeof(uint16_t)) > bufSize) {
            SIPL_LOG_ERR_STR("TPGSerializer: Bad buffer length \n");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
        }
        else {
            uint32_t i = 0U;
            const char *TPGSerializerErrorCodePath = "/tmp/TPGserializerErrCode.err";
            uint16_t TPGSerializerErrCode[TPGSERIALIZER_STATUS_MAX_ERR];
            (void)memset(TPGSerializerErrCode, 0, sizeof(TPGSerializerErrCode));
            status = TPGReadErrorData((size_t)TPGSERIALIZER_STATUS_MAX_ERR,
                                      TPG_SERIALIZER_REGISTER_NAME,
                                      TPGSerializerErrorCodePath,
                                      TPGSerializerErrCode);

            if (status == NVMEDIA_STATUS_OK) {
                //! parse register data to buffer
                for (i = 0U; i < (uint32_t)TPGSERIALIZER_STATUS_MAX_ERR; ++i) {
                    buffer[i * 2U] =
                        (uint8_t)((TPGSerializerErrCode[i] >> 8) & 0xFFU);
                    buffer[(i * 2U) + 1U] =
                        (uint8_t)(TPGSerializerErrCode[i] & 0xFFU);
                    LOG_DBG("[%s:%d] Error value of %s : 0x%.2X%.2X \n", __func__, __LINE__, TPG_SERIALIZER_REGISTER_NAME[i], buffer[i * 2U], buffer[(i * 2U) + 1U]);
                }
            }
            else {
                SIPL_LOG_ERR_STR("TPGSerializer: unable to read error code in TPGSerializerReadErrorData\n");
            }
        }
    }
    return status;
}

