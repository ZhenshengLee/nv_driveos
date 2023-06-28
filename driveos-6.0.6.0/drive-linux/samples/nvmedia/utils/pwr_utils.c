/* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdio.h>
#include <pthread.h>
#include <dlfcn.h>

#include "pwr_utils.h"
#include "log_utils.h"
#include "ccp.h"

#if (NV_IS_SAFETY == 1)
#define MODEL_NODE  "/dev/nvsku/model"
#else /* NV_IS_SAFETY */
#if defined(NVMEDIA_QNX)
#define MODEL_NODE  "/dev/nvdt/model"
#else  /* NVMEDIA_QNX */
#define MODEL_NODE "/proc/device-tree/model"
#endif
#endif /* NV_IS_SAFETY */
#define MAX_BOARD_NAME_SIZE (128u)

typedef struct {
    const char* interface;
    PowerPort port;
} StringToPort;

typedef struct {
    NvMediaICPInterfaceType interface;
    PowerPort port;
} InterfaceToPort;

typedef enum {
    DISABLED = 0,
    OWNED,
    POWERED
} PowerPortOwnership;

typedef struct {
    PowerPortOwnership ownership;
    bool slave;
    bool ccp_supported;
} PowerPortState;

static PowerPortState powerPortStates[] = {
    [POWER_PORT_A] = {.ownership = DISABLED, .slave = false},
    [POWER_PORT_B] = {.ownership = DISABLED, .slave = false},
    [POWER_PORT_C] = {.ownership = DISABLED, .slave = false},
    [POWER_PORT_D] = {.ownership = DISABLED, .slave = false}
};

static nvccp_cam_group_id
sGetGroupId(PowerPort port) {
    switch(port) {
        case POWER_PORT_A:
            return NVCCP_GROUP_A;
        case POWER_PORT_B:
            return NVCCP_GROUP_B;
        case POWER_PORT_C:
            return NVCCP_GROUP_C;
        default:
            return NVCCP_GROUP_D;
    };
}

static bool
sIsCCPSupported(
    void)
{
    static const char* unsupportedBoards[] = {
        "p2382_t186"
    };

    FILE* file = NULL;
    char name[MAX_BOARD_NAME_SIZE]= {0};
    int size = 0;
    uint32_t i = 0;

    /* Open the board node file */
    file = fopen(MODEL_NODE, "r");
    if (file == NULL) {
        LOG_ERR("Failed to open board model file \"%s\" with errno (%d)\n", MODEL_NODE, errno);
        return false;
    }

    /* Read the board model */
    size = fread(name, sizeof(char), MAX_BOARD_NAME_SIZE-1, file);
    if (size <= 0) {
        LOG_ERR("Could not read board model from file \"%s\" with errno (%d)\n", MODEL_NODE, errno);
        return false;
    }
    name[size] = '\0';

    /* Compare with the unsupported boards */
    for (i = 0; i < sizeof(unsupportedBoards) / sizeof(unsupportedBoards[0]); i++) {
        if (strncmp(unsupportedBoards[i], name, strlen(unsupportedBoards[i])) == 0) {
            return false;
        }
    }

    return true;
}

PowerPort
PowerControlPortFromString(
    const char* interface)
{
    static const StringToPort stringToPortMap[] = {
        {.interface = "csi-a",  .port = POWER_PORT_A},
        {.interface = "trio-a", .port = POWER_PORT_A},
        {.interface = "csi-b",  .port = POWER_PORT_A},
        {.interface = "trio-b", .port = POWER_PORT_A},
        {.interface = "csi-c",  .port = POWER_PORT_B},
        {.interface = "trio-c", .port = POWER_PORT_B},
        {.interface = "csi-d",  .port = POWER_PORT_B},
        {.interface = "trio-d", .port = POWER_PORT_B},
        {.interface = "csi-e",  .port = POWER_PORT_C},
        {.interface = "trio-e", .port = POWER_PORT_C},
        {.interface = "csi-f",  .port = POWER_PORT_C},
        {.interface = "trio-f", .port = POWER_PORT_C},
        {.interface = "csi-g",  .port = POWER_PORT_D},
        {.interface = "trio-g", .port = POWER_PORT_D},
        {.interface = "csi-h",  .port = POWER_PORT_D},
        {.interface = "trio-h", .port = POWER_PORT_D},
    };
    unsigned int i = 0;

    for (i = 0; i < sizeof(stringToPortMap) / sizeof(stringToPortMap[0]); i++) {
        if (strncmp(interface, stringToPortMap[i].interface, strlen(stringToPortMap[i].interface)) == 0) {
            return stringToPortMap[i].port;
            break;
        }
    }
    return POWER_PORT_INVALID;
}

PowerPort
PowerControlPortFromInterface(
    NvMediaICPInterfaceType type)
{
    static const InterfaceToPort interfaceToPortMap[] = {
        {.interface = NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_A,  .port = POWER_PORT_A},
        {.interface = NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_B,  .port = POWER_PORT_A},
        {.interface = NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_AB, .port = POWER_PORT_A},
        {.interface = NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_C,  .port = POWER_PORT_B},
        {.interface = NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_D,  .port = POWER_PORT_B},
        {.interface = NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_CD, .port = POWER_PORT_B},
        {.interface = NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_E,  .port = POWER_PORT_C},
        {.interface = NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_F,  .port = POWER_PORT_C},
        {.interface = NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_EF, .port = POWER_PORT_C},
        {.interface = NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_G,  .port = POWER_PORT_D},
        {.interface = NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_H,  .port = POWER_PORT_D},
        {.interface = NVMEDIA_IMAGE_CAPTURE_CSI_INTERFACE_TYPE_CSI_GH, .port = POWER_PORT_D},
    };
    unsigned int i = 0;

    for (i = 0; i < sizeof(interfaceToPortMap) / sizeof(interfaceToPortMap[0]); i++) {
        if (interfaceToPortMap[i].interface == type) {
            return interfaceToPortMap[i].port;
            break;
        }
    }
    return POWER_PORT_INVALID;
}

NvMediaStatus
PowerControlRequestOwnership(
    PowerPort port,
    bool slave)
{
    nvccp_return_t status =  NVCCP_STATUS_OK;

    /* Verify input is valid */
    if (port >= POWER_PORT_INVALID) {
        LOG_ERR("%s: Invalid port provided!\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* Check if CCP is supported for this board */
    if(sIsCCPSupported()) {
        powerPortStates[port].ccp_supported = true;
    } else {
        powerPortStates[port].ccp_supported = false;
        return NVMEDIA_STATUS_OK;
    }

    /* Make sure we aren't requesting ownership for something that is already owned */
    if (powerPortStates[port].ownership != DISABLED) {
        LOG_ERR("%s: Cannot request ownership for a port that is already owned!\n", __func__);
        return NVMEDIA_STATUS_ERROR;
    }


    /* Request ownership */
    status = nvccp_request_ownership(
        sGetGroupId(port),
        slave ? NVCCP_CAM_SLAVE : NVCCP_CAM_MASTER);
    if (status != NVCCP_STATUS_OK) {
        LOG_ERR("%s: nvccp_request_ownership failed with status (%d)\n", __func__, status);
        return NVMEDIA_STATUS_ERROR;
    }

    /* Update the status */
    powerPortStates[port].slave = slave;
    powerPortStates[port].ownership = OWNED;
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
PowerControlReleaseOwnership(
    PowerPort port)
{
    nvccp_return_t status =  NVCCP_STATUS_OK;
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;

    /* Verify input is valid */
    if (port >= POWER_PORT_INVALID) {
        LOG_ERR("%s: Invalid port provided!\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* Do nothing if CCP is not supported */
    if (!powerPortStates[port].ccp_supported) {
        return NVMEDIA_STATUS_OK;
    }

    switch(powerPortStates[port].ownership) {
        /* If the port is powered, first unpower it */
        case POWERED:
            nvmStatus = PowerControlSetPower(port, false);
            if (nvmStatus != NVMEDIA_STATUS_OK) {
                return nvmStatus;
            }

        /* If the port is owned, then release it */
        case OWNED:
            status = nvccp_release_ownership(
                sGetGroupId(port),
                powerPortStates[port].slave ? NVCCP_CAM_SLAVE : NVCCP_CAM_MASTER);
            if (status != NVCCP_STATUS_OK) {
                LOG_ERR("%s: nvccp_release_ownership failed with status (%d)\n", __func__, status);
                return NVMEDIA_STATUS_ERROR;
            }
        default:
            break;
    }
    /* Now reset the ownership status and return */
    powerPortStates[port].ownership = DISABLED;
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
PowerControlSetPower(
    PowerPort port,
    bool poweredOn)
{
    nvccp_return_t status =  NVCCP_STATUS_OK;
    /* Verify input is valid */
    if (port >= POWER_PORT_INVALID) {
        LOG_ERR("%s: Invalid port provided!\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* Do nothing if CCP is not supported */
    if (!powerPortStates[port].ccp_supported) {
        return NVMEDIA_STATUS_OK;
    }

    /* If the port is being powered on */
    if (poweredOn) {
        /* Verify that the state is correct */
        if (powerPortStates[port].ownership != OWNED) {
            return NVMEDIA_STATUS_ERROR;
        }

        /* Power on the aggregator */
        status = nvccp_set_aggreg_pwr_on(sGetGroupId(port));
        if (status != NVCCP_STATUS_OK && status != NVCCP_STATUS_ALREADY_ON) {
            LOG_ERR("%s: nvccp_set_aggreg_pwr_on failed with status (%d)\n", __func__, status);
            return NVMEDIA_STATUS_ERROR;
        }

        /* Power on the individual cameras */
        status = nvccp_set_cam_pwr_on(sGetGroupId(port));
        if (status != NVCCP_STATUS_OK && status != NVCCP_STATUS_ALREADY_ON) {
            LOG_ERR("%s: nvccp_set_cam_pwr_on failed with status (%d)\n", __func__, status);
            return NVMEDIA_STATUS_ERROR;
        }

        /* Update the power status */
        powerPortStates[port].ownership = POWERED;

    /* If the port is being powered off */
    } else {
        /* Verify that the state is correct */
        if (powerPortStates[port].ownership != POWERED) {
            return NVMEDIA_STATUS_ERROR;
        }

        /* Power off the aggregator */
        status = nvccp_set_aggreg_pwr_off(sGetGroupId(port));
        if (status != NVCCP_STATUS_OK && status != NVCCP_STATUS_ALREADY_OFF) {
            LOG_ERR("%s: nvccp_set_aggreg_pwr_off failed with status (%d)\n", __func__, status);
            return NVMEDIA_STATUS_ERROR;
        }

        /* Power on the individual cameras */
        status = nvccp_set_cam_pwr_off(sGetGroupId(port));
        if (status != NVCCP_STATUS_OK && status != NVCCP_STATUS_ALREADY_OFF) {
            LOG_ERR("%s: nvccp_set_cam_pwr_off failed with status (%d)\n", __func__, status);
            return NVMEDIA_STATUS_ERROR;
        }

        /* Update the power status */
        powerPortStates[port].ownership = OWNED;
    }

    return NVMEDIA_STATUS_OK;
}
