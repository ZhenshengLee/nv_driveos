/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <ctype.h>
#include "devblk_cdi.h"
#include <limits.h>
#include "sipl_error.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <TPGSensorSerializerUtility.h>

#ifndef LINE_BUFFER_SIZE
#define LINE_BUFFER_SIZE 1024
#endif

static NvMediaStatus
TPGErrorCodeSetting(const char* buffer,
                    const char** errGrpLUT,
                    const uint16_t errGrpSize,
                    uint16_t* errGrpCode);


static NvMediaStatus
TPGErrorCodeSetting(const char* buffer,
                    const char** errGrpLUT,
                    const uint16_t errGrpSize,
                    uint16_t* errGrpCode) {
    uint32_t i = 0U;
    uint32_t j = 0U;
    /* return OK if no buffer found */
    int32_t offset = 0;
    int16_t numDigit = 0;

    if (buffer != NULL) {
        /* parse the buffer and make set appropriate return code to each register */
        /*expected format TPGXXXX_ASIL_STATUS_REG_0 = 0x0001
          or
                          TPGXXXX_ASIL_STATUS_REG_0 = 1
         */
        /* parse line */
        NvMediaStatus found = NVMEDIA_STATUS_ERROR;
        const char* substring = NULL;
        const char* finalSubstring = NULL;
        for (i = 0U; i < errGrpSize; ++i) {
            if (strstr(buffer, errGrpLUT[i]) != NULL) {
                substring = strchr(buffer, '=');
                if (substring != NULL) {
                    substring += 1; /* move to the character after the '=' character*/
                    for (j = 0U; substring[j]; ++j) {
                        if (substring[j] == 'x' || substring[j] == 'X') {
                            finalSubstring = substring + j;
                            break;
                        }
                    }
                    if (finalSubstring != NULL) {
                        finalSubstring += 1; /* move to the character after x*/
                        for (j = 0U; finalSubstring[j]; ++j) {
                            if (!isxdigit(finalSubstring[j])) {
                                SIPL_LOG_ERR_STR("TPGSensorSerializerUtility: unknown error code format");
                                return NVMEDIA_STATUS_ERROR;
                            }
                            ++numDigit;
                        }
                        if (numDigit > 4) // 16bits hexdecimal
                        {
                            SIPL_LOG_ERR_STR("TPGSensorSerializerUtility: error code is greater than the maximum of uint16_t datatype");
                            return NVMEDIA_STATUS_ERROR;
                        }
                        errGrpCode[i] = (uint16_t)strtoul((finalSubstring), NULL, 16);
                    }
                    else
                    {
                        for (j = 1U; substring[j]; ++j) {
                            if (isspace(substring[j])) {
                                offset = j;
                                continue;
                            }
                            if (!isdigit(substring[j])) {
                                SIPL_LOG_ERR_2STR("TPGSensorSerializerUtility: unknown error code format", substring);
                                return NVMEDIA_STATUS_ERROR;
                            }
                        }
                        uint32_t data = (uint32_t)strtoul((substring + offset), NULL,10);
                        if (data > UINT16_MAX) {
                            SIPL_LOG_ERR_STR("TPGSensorSerializerUtility: error code is greater than Maximum value of an uint16_t datatype");
                            return NVMEDIA_STATUS_ERROR;
                        }
                        errGrpCode[i] = (uint16_t)data;
                    }
                }
                found = NVMEDIA_STATUS_OK;
                break;
            }
        }
        if (found == NVMEDIA_STATUS_ERROR) {
            SIPL_LOG_ERR_2STR("TPGSensorSerializerUtility: unknown ASIL info", buffer);
            return NVMEDIA_STATUS_ERROR;
        }
    }
    return NVMEDIA_STATUS_OK;
};

NvMediaStatus
TPGReadErrorData(size_t errGrpSize,
                 const char** errGrpLUT,
                 const char* errCodeFileName,
                 uint16_t* errGrpCode) {

    char stringBuffer[LINE_BUFFER_SIZE];
    uint16_t i = 0U;
    FILE *fp = NULL;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    char* strtokP = NULL;
    char* savedPtr = NULL;
    /* simulate a GPIO error by Reading the setting in the file */
    fp = fopen(errCodeFileName, "r");
    if (fp != NULL) {
        status = NVMEDIA_STATUS_OK;
        while (fgets(stringBuffer, sizeof(stringBuffer), fp)  != NULL) {
            strtokP = strtok_r(stringBuffer, "\n", &savedPtr);
            if (strtokP == NULL) {
                SIPL_LOG_ERR_STR("TPGSensorSerializerUtility: unknown string");
                status = NVMEDIA_STATUS_ERROR;
            }
            if (status == NVMEDIA_STATUS_OK) {
                status = TPGErrorCodeSetting(strtokP,
                                            errGrpLUT,
                                            errGrpSize,
                                            errGrpCode);
                if (status != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR("TPGSensorSerializerUtility: failed to read ErrorData");
                    break;
                }
            }
        }
        fclose(fp);
    }
    else {
        uint16_t defaultValue = 0x1234U;
        fp = fopen(errCodeFileName, "w");
        if (fp != NULL) {
            status  = NVMEDIA_STATUS_OK;
            for (i = 0U; i < errGrpSize; ++i) {
                /* set default to 0x1234; more obvious to the client to see the TPGReadErrorData is triggered */
                if (0 > fprintf(fp, "%s = 0x%04x\n", errGrpLUT[i], defaultValue)){
                    SIPL_LOG_ERR_STR("TPGSensorSerializerUtility: failed to write to file");
                    status = NVMEDIA_STATUS_ERROR;
                    break;
                };
                errGrpCode[i] = defaultValue;
            }
            if (fclose(fp)) {
                SIPL_LOG_ERR_STR("TPGSensorSerializerUtility: failed  to close file");
                status  = NVMEDIA_STATUS_ERROR;
            }
        }
        else {
            SIPL_LOG_ERR_2STR("TPGSensorSerializerUtility: Cannot create ASIL error file", errCodeFileName);
            status =  NVMEDIA_STATUS_ERROR;
        }
    }

    return status;
}
