/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CDI_MAX20087_LINUX_H
#define CDI_MAX20087_LINUX_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "dirent.h"

#define DEVICE_TREE     "/proc/device-tree"

static NvMediaStatus
GetNumfromDT(
    const char *path,
    uint32_t *number)
{
    FILE* fp = NULL;
    uint32_t sz = 0u;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    int8_t valDT[4] = {-1};

    /* Open device property */
    fp = fopen(path, "r");
    if (fp != NULL) {
        /* Get size of property size */
        if (fseek(fp, 0L, SEEK_END) == 0) {
            sz = ftell(fp);
            if (fseek(fp, 0L, SEEK_SET) == 0) {

                if (sz != 0u) {
                    if (fread((void *)&valDT, sizeof(int8_t), 4, fp) == 4u) {
                        /* i2c-bus value is stored as MSB on 0 byte position
                        and LSB on 3rd byte position in file */
                        *number = ((uint32_t)(valDT[0] << 24) & 0xFF000000);
                        *number |= ((uint32_t)(valDT[1] << 16) & 0xFF0000);
                        *number |= ((uint32_t)(valDT[2] << 8) & 0xFF00);
                        *number |= ((uint32_t)valDT[3] & 0xFF);
                        status = NVMEDIA_STATUS_OK;
                    }
                    else {
                        SIPL_LOG_ERR_2STR("GetNumfromDT: Reading property from DT"
                                          " failed with error", strerror(errno));
                    }
                }
            }
        }
        if (fclose(fp) != 0) {
            SIPL_LOG_ERR_2STR("GetNumfromDT: fclose failed for property with "
                              "error", strerror(errno));
            status = NVMEDIA_STATUS_ERROR;
        }
    }
    return status;
}

/*  Reads the device tree to get the property sent by caller from MAX20087 node.
 *  Returns NVMEDIA_STATUS_ERROR if device tree property cannot be read.
 *  Sets the value of the property if property is found.
 */
static NvMediaStatus
GetDTPropU32(
    const char * propName,
    uint32_t * const propValue,
    int32_t const csiPort)
{
    DIR *dirp = NULL;
    struct dirent *dir_entry = NULL;
    uint32_t csiPortDT;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;

    *propValue = 1;    // Failure value 1 if property not found in DT

    if (csiPort < 0) {
        SIPL_LOG_ERR_STR("GetDTPropU32: Incorrect csiPort received");
        goto done;
    }

    /* Get the device-tree entry directory */
    dirp = opendir(DEVICE_TREE);
    if (dirp != NULL) {
        char path_buffer[PATH_MAX];
        if (realpath(DEVICE_TREE, path_buffer) != NULL) {
            while ((dir_entry = readdir(dirp)) != NULL) {

                if (strncmp(dir_entry->d_name, "sipl_devblk", strlen("sipl_devblk")) == 0) {

                    /* Get the full path of csi-port to match with csiPort received */
                    (void)snprintf( path_buffer, PATH_MAX,
                                    "%s/%s/tegra/csi-port", DEVICE_TREE, dir_entry->d_name);
                    status = GetNumfromDT(path_buffer, &csiPortDT);
                    if (status != NVMEDIA_STATUS_OK) {
                        SIPL_LOG_ERR_STR("GetDTPropU32: Failed to get csi-Port");
                        goto done;
                    }
                    // Match the csi-Port from DT to the csi-Port received from
                    // cameraModuleCfg for getting correct sipl_devblk node
                    if (csiPortDT != (uint32_t)csiPort) {
                        continue;
                    }

                    /* Get the full path of required property */
                    (void)snprintf( path_buffer, PATH_MAX,
                                    "%s/%s/pwr_ctrl/max20087/%s", DEVICE_TREE,
                                    dir_entry->d_name, propName );
                    /* Get property value */
                    status = GetNumfromDT(path_buffer, propValue);
                    if (status != NVMEDIA_STATUS_OK) {
                        *propValue = 1;
                        SIPL_LOG_ERR_2STR("GetDTPropU32: Failed to get property",
                                          propName);
                    }
                    break;
                }
            }
        }
        else {
            SIPL_LOG_ERR_2STR("GetDTPropU32: get real path failed: ", strerror(errno));
        }
        if (closedir(dirp) < 0) {
            SIPL_LOG_ERR_2STR("GetDTPropU32: closedir() failed: ", strerror(errno));
            status = NVMEDIA_STATUS_ERROR;
        }
    }
    else {
        SIPL_LOG_ERR_2STR("GetDTPropU32: DT File Not accessible: ", strerror(errno));
    }
done:
    return status;
}

#endif // CDI_MAX20087_LINUX_H
