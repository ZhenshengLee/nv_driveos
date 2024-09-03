/* Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <string.h>
#include <stdlib.h>
#include "log_utils.h"
#include "chip_util.h"
#include "nvmedia_core.h"

#ifndef TEGRA_CHIP_ID_NODE
#ifdef NVMEDIA_QNX
#define TEGRA_CHIP_ID_NODE "/dev/nvsys/tegra_chip_id"
#else
#define TEGRA_CHIP_ID_NODE "/sys/module/tegra_fuse/parameters/tegra_chip_id"
#endif
#endif

TegraChipId
GetTegraChipId(void) {

#ifndef NVMEDIA_GHSI
#ifndef TARGET_T194_SIL
    FILE * fp = NULL;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    unsigned long cid = 0;
    char buf[128];
    TegraChipId chipId = TEGRA_UKNOWN;

    memset(buf, 0, sizeof(buf));

    /* Open device tree */
    fp = fopen(TEGRA_CHIP_ID_NODE, "r");
    if (fp == NULL) {
        LOG_ERR("%s : Failed to open TEGRA_CHIP_ID_NODE: %s \n", TEGRA_CHIP_ID_NODE, __func__);
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    /* Read chip-id */
    if (fgets(buf, 127, fp) == NULL) {
        LOG_ERR("%s : Failed to read TEGRA_CHIP_ID_NODE\n", __func__);
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    cid = strtol(buf, NULL, 0);

    switch (cid) {
        case TEGRA_T18x :
            chipId = TEGRA_T18x;
            LOG_INFO("TEGRA CHIP DETECTED : PARKER \n", __func__);
            break;
        case TEGRA_T19x :
            chipId = TEGRA_T19x;
            LOG_INFO("TEGRA CHIP DETECTED : XAVIER \n", __func__);
            break;
        default:
            chipId = TEGRA_UKNOWN;
            LOG_INFO("TEGRA CHIP DETECTED : UNKNOWN \n", __func__);
            break;
    }

done:
    if (fp) {
        if (fclose(fp) != 0) {
            LOG_ERR("%s: fclose() failed \n", __func__);
        }
    }

    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s : Failed to read TEGRA_CHIP_ID \n", __func__);
    }

    return chipId;
#else
    return TEGRA_T19x;
#endif
#else
    return TEGRA_T18x;
#endif
}


