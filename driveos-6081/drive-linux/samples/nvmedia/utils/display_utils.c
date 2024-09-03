/*
 * Copyright (c) 2018 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>

#include "log_utils.h"
#include "display_utils.h"


NvMediaStatus
GetAvailableVideoDisplayDevices(
    int *outputDevicesNum,
    NvMediaVideoOutputDeviceParams *outputDevicesList)
{
    NvMediaStatus status;

    /* Get device information for all devices */
    status = NvMediaVideoOutputDevicesQuery(outputDevicesNum, outputDevicesList);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("GetAvailableVideoDisplayDevices: Failed querying devices. Error: %d\n", status);
        return status;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
GetAvailableIDPDisplayDevices(
    int32_t *outputDevicesNum,
    NvMediaIDPDeviceParams *outputDevicesList)
{
    NvMediaStatus status;

    /* Get device information for all devices */
    status = NvMediaIDPQuery(outputDevicesNum, outputDevicesList);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("GetAvailableIDPDisplayDevices: Failed querying devices. Error: %d\n", status);
        return status;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
CheckIDPDisplayDeviceID(
    unsigned int displayId,
    NvMediaBool *enabled)
{
    int32_t outputDevices;
    NvMediaIDPDeviceParams *outputParams;
    NvMediaStatus status;
    unsigned int found = 0;
    int i;

    /* By default set it as not enabled (initialized) */
    *enabled = NVMEDIA_FALSE;

    /* Get the number of devices */
    status = NvMediaIDPQuery(&outputDevices, NULL);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("CheckIDPDisplayDeviceID: Failed querying the number of devices. Error: %d\n",
                status);
        return status;
    }

    /* Allocate memory for information for all devices */
    outputParams = malloc(outputDevices * sizeof(NvMediaIDPDeviceParams));
    if(outputParams == NULL) {
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }

    /* Get device information for all devices */
    status = NvMediaIDPQuery(&outputDevices, outputParams);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("CheckIDPDisplayDeviceID: Failed querying devices. Error: %d\n", status);
        free(outputParams);
        return status;
    }

    /* Find desired device */
    for(i = 0; i < outputDevices; i++) {
        if((outputParams + i)->displayId == displayId) {
            /* Return information */
            *enabled = (outputParams + i)->enabled;
            found = 1;
            break;
        }
    }

    free(outputParams);

    if(!found) {
        LOG_ERR("CheckIDPDisplayDeviceID: Requested display id is invalid (%d)\n", displayId);
        return NVMEDIA_STATUS_ERROR;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
CheckVideoDisplayDeviceID(
    unsigned int displayId,
    NvMediaBool *enabled)
{
    int outputDevices;
    NvMediaVideoOutputDeviceParams *outputParams;
    NvMediaStatus status;
    unsigned int found = 0;
    int i;

    /* By default set it as not enabled (initialized) */
    *enabled = NVMEDIA_FALSE;

    /* Get the number of devices */
    status = NvMediaVideoOutputDevicesQuery(&outputDevices, NULL);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("CheckVideoDisplayDeviceID: Failed querying the number of devices. Error: %d\n",
                status);
        return status;
    }

    /* Allocate memory for information for all devices */
    outputParams = malloc(outputDevices * sizeof(NvMediaVideoOutputDeviceParams));
    if(outputParams == NULL) {
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }

    /* Get device information for all devices */
    status = NvMediaVideoOutputDevicesQuery(&outputDevices, outputParams);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("CheckVideoDisplayDeviceID: Failed querying devices. Error: %d\n", status);
        free(outputParams);
        return status;
    }

    /* Find desired device */
    for(i = 0; i < outputDevices; i++) {
        if((outputParams + i)->displayId == displayId) {
            /* Return information */
            *enabled = (outputParams + i)->enabled;
            found = 1;
            break;
        }
    }

    free(outputParams);

    if(!found) {
        LOG_ERR("CheckVideoDisplayDeviceID: Requested display id is invalid (%d)\n", displayId);
        return NVMEDIA_STATUS_ERROR;
    }

    return NVMEDIA_STATUS_OK;
}

