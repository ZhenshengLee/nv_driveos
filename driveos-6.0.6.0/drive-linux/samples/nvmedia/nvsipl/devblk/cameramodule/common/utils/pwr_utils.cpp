/* Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <chrono>
#include <dlfcn.h>
#include <cerrno>
#include <cstring>
#include <thread>
#include <unistd.h>

#include "pwr_utils.h"
#include "CNvMTrace.hpp"
#include "utils.hpp"
#include "sipl_error.h"

/* coverity[misra_cpp_2008_rule_16_0_1_violation] : intentional TID-1434 */
extern "C" {
#include "ccp.h"
}


/**
 * \brief Which port to enable or disable the power for.
 *
 * Corresponds to NvCCP deserializers. Power port values must always start with
 * 0 and be sequential, as they are used elsewhere in this module as array
 * indicies into arrays of length POWER_PORT_INVALID.
 */
namespace nvsipl {
typedef enum {
    POWER_PORT_A = 0,
    POWER_PORT_B,
    POWER_PORT_C,
    POWER_PORT_D,
    /**
     * \brief Reserved value to represent an invalid power port.
     *
     * Must always be one greater than the last power port, so it can be used
     * as the length of lists of pwoer ports.
     */
    POWER_PORT_INVALID
} PowerPort;

/**
 * \brief Internal port state machine states.
 *
 * Each port may only be in one of these three states.
 *
 * \todo: add a dynamic diagram showing how state transition occurs in terms of
 * internal and external API functions.
 */
typedef enum {
    /**
     * \brief For use when the power port is not owned at all by the current application.
     *
     * In the disabled state, power may not be controlled for this port and it
     * is assumed to be powered off.
     */
    DISABLED = 0,
    /**
     * \brief For use when the power port is owned by the current application,
     * but not powered on.
     *
     * In the owned state, power may be controlled for this port and it is
     * assumed to be powered off.
     */
    OWNED,
    /**
     * \brief For use when the power port is owned by the current application and powered on.
     *
     * In the POWERED state, power may be controlled for this port and it is
     * assumed to be powered on.
     */
    POWERED
} PowerPortOwnership;

/**
 * \brief The full state of a given power port.
 *
 * Stores the ownership state machine (see `PowerPortOwnership`) and any
 * additional metadata about the port required for correct power control.
 */
typedef struct {
    /**
     * \brief The power port state machine state
     */
    PowerPortOwnership ownership;
    /**
     * \brief Whether this port is in passive or master mode.
     *
     * When true, the port is in passive mode. Otherwise, it is assumed to be in
     * master mode. NvCCP needs to know this information when controlling the
     * power state of a port, so we track it.
     */
    bool passive;
} PowerPortState;

/**
 * \brief Complete list of power port states.
 *
 * This array contains an entry for each power port.
 * All ports are initialized in the DISABLED state and in master mode.
 */
static PowerPortState powerPortStates[POWER_PORT_INVALID] = {
    [POWER_PORT_A] = {.ownership = DISABLED, .passive = false},
    [POWER_PORT_B] = {.ownership = DISABLED, .passive = false},
    [POWER_PORT_C] = {.ownership = DISABLED, .passive = false},
    [POWER_PORT_D] = {.ownership = DISABLED, .passive = false}
};

/**
 * \brief Map a `PowerPort` to the corresponding NvCCP camera group
 *
 * The mapping is as follows:
 *
 * `port`         | `nvccp_cam_group_id`
 * -------------- | -------------------
 * `POWER_PORT_A` | `NVCCP_GROUP_A`
 * `POWER_PORT_B` | `NVCCP_GROUP_B`
 * `POWER_PORT_C` | `NVCCP_GROUP_C`
 * `POWER_PORT_D` | `NVCCP_GROUP_D`
 *
 * Any other value for `port` is invalid and will not be mapped correctly.
 *
 * @param[in] port The port to map.
 *
 * @returns The mapped camera group ID according to the table above.
 */
static nvccp_cam_group_id
sGetGroupId(PowerPort const port)
{
    nvccp_cam_group_id group_id;

    switch(port) {
        case POWER_PORT_A:
            group_id = NVCCP_GROUP_A;
            break;
        case POWER_PORT_B:
            group_id = NVCCP_GROUP_B;
            break;
        case POWER_PORT_C:
            group_id = NVCCP_GROUP_C;
            break;
        case POWER_PORT_D:
            group_id = NVCCP_GROUP_D;
            break;
        default:
            group_id = NVCCP_GROUP_D;
            SIPL_LOG_ERR_STR_INT("pwr_utils::sGetGroupId port out of range, defaulting to NVCCP_GROUP_D",
                                static_cast<int32_t>(port));
            break;
    }
    return group_id;
}

/**
 * \brief Maps a power portA and link number to the corresponding NvCCP camera.
 *
 * The mapping is as follows:
 *
 * `port`         | `link` | `nvccp_cam_id`
 * -------------- | ------ | --------------
 * `POWER_PORT_A` | 0      | `NVCCP_CAM_A0`
 * `POWER_PORT_A` | 1      | `NVCCP_CAM_A1`
 * `POWER_PORT_A` | 2      | `NVCCP_CAM_A2`
 * `POWER_PORT_A` | 3      | `NVCCP_CAM_A3`
 *
 * Passing values not on the above table will result in erroneous behavior and
 * should be avoided by the caller.
 *
 * @param[in] link The link on which the camera is located
 *
 * @returns A `nvccp_cam_id` according to the table above.
 *
 */
static nvccp_cam_id
getPowerPortA(uint8_t const link)
{
    nvccp_cam_id camid = NVCCP_CAM_A0;

    switch(link) {
        case 0:
            camid = NVCCP_CAM_A0;
            break;
        case 1:
            camid = NVCCP_CAM_A1;
            break;
        case 2:
            camid = NVCCP_CAM_A2;
            break;
        case 3:
            camid = NVCCP_CAM_A3;
            break;
        default:
            // do nothing
            break;
    }
    return camid;
}

/**
 * \brief Maps a power portB and link number to the corresponding NvCCP camera.
 *
 * The mapping is as follows:
 *
 * `port`         | `link` | `nvccp_cam_id`
 * -------------- | ------ | --------------
 * `POWER_PORT_B` | 0      | `NVCCP_CAM_B0`
 * `POWER_PORT_B` | 1      | `NVCCP_CAM_B1`
 * `POWER_PORT_B` | 2      | `NVCCP_CAM_B2`
 * `POWER_PORT_B` | 3      | `NVCCP_CAM_B3`
 *
 * Passing values not on the above table will result in erroneous behavior and
 * should be avoided by the caller.
 *
 * @param[in] link The link on which the camera is located
 *
 * @returns A `nvccp_cam_id` according to the table above.
 *
 */
static nvccp_cam_id
getPowerPortB(uint8_t const link)
{
    nvccp_cam_id camid = NVCCP_CAM_B0;

    switch(link) {
        case 0:
            camid = NVCCP_CAM_B0;
            break;
        case 1:
            camid = NVCCP_CAM_B1;
            break;
        case 2:
            camid = NVCCP_CAM_B2;
            break;
        case 3:
            camid = NVCCP_CAM_B3;
            break;
        default:
            /* do nothing */
            break;
    }
    return camid;
}

/**
 * \brief Maps a power portC and link number to the corresponding NvCCP camera.
 *
 * The mapping is as follows:
 *
 * `port`         | `link` | `nvccp_cam_id`
 * -------------- | ------ | --------------
 * `POWER_PORT_C` | 0      | `NVCCP_CAM_C0`
 * `POWER_PORT_C` | 1      | `NVCCP_CAM_C1`
 * `POWER_PORT_C` | 2      | `NVCCP_CAM_C2`
 * `POWER_PORT_C` | 3      | `NVCCP_CAM_C3`
 *
 * Passing values not on the above table will result in erroneous behavior and
 * should be avoided by the caller.
 *
 * @param[in] link The link on which the camera is located
 *
 * @returns A `nvccp_cam_id` according to the table above.
 *
 */
static nvccp_cam_id
getPowerPortC(uint8_t const link)
{
    nvccp_cam_id camid = NVCCP_CAM_C0;

    switch(link) {
        case 0:
            camid = NVCCP_CAM_C0;
            break;
        case 1:
            camid = NVCCP_CAM_C1;
            break;
        case 2:
            camid = NVCCP_CAM_C2;
            break;
        case 3:
            camid = NVCCP_CAM_C3;
            break;
        default:
            /* do nothing */
            break;
    }
    return camid;
}

/**
 * \brief Maps a power portD and link number to the corresponding NvCCP camera.
 *
 * The mapping is as follows:
 *
 * `port`         | `link` | `nvccp_cam_id`
 * -------------- | ------ | --------------
 * `POWER_PORT_D` | 0      | `NVCCP_CAM_D0`
 * `POWER_PORT_D` | 1      | `NVCCP_CAM_D1`
 * `POWER_PORT_D` | 2      | `NVCCP_CAM_D2`
 * `POWER_PORT_D` | 3      | `NVCCP_CAM_D3`
 *
 * Passing values not on the above table will result in erroneous behavior and
 * should be avoided by the caller.
 *
 * @param[in] link The link on which the camera is located
 *
 * @returns A `nvccp_cam_id` according to the table above.
 *
 */
static nvccp_cam_id
getPowerPortD(uint8_t const link)
{
    nvccp_cam_id camid = NVCCP_CAM_D0;

    switch(link) {
        case 0:
            camid = NVCCP_CAM_D0;
            break;
        case 1:
            camid = NVCCP_CAM_D1;
            break;
        case 2:
            camid = NVCCP_CAM_D2;
            break;
        case 3:
            camid = NVCCP_CAM_D3;
            break;
        default:
            /* do nothing */
            break;
    }
    return camid;
}

/**
 * \brief Maps a power port and link number to the corresponding NvCCP camera.
 *
 * The mapping is as follows:
 *
 * `port`         | `link` | `nvccp_cam_id`
 * -------------- | ------ | --------------
 * `POWER_PORT_A` | 0      | `NVCCP_CAM_A0`
 * `POWER_PORT_A` | 1      | `NVCCP_CAM_A1`
 * `POWER_PORT_A` | 2      | `NVCCP_CAM_A2`
 * `POWER_PORT_A` | 3      | `NVCCP_CAM_A3`
 * `POWER_PORT_B` | 0      | `NVCCP_CAM_B0`
 * `POWER_PORT_B` | 1      | `NVCCP_CAM_B1`
 * `POWER_PORT_B` | 2      | `NVCCP_CAM_B2`
 * `POWER_PORT_B` | 3      | `NVCCP_CAM_B3`
 * `POWER_PORT_C` | 0      | `NVCCP_CAM_C0`
 * `POWER_PORT_C` | 1      | `NVCCP_CAM_C1`
 * `POWER_PORT_C` | 2      | `NVCCP_CAM_C2`
 * `POWER_PORT_C` | 3      | `NVCCP_CAM_C3`
 * `POWER_PORT_D` | 0      | `NVCCP_CAM_D0`
 * `POWER_PORT_D` | 1      | `NVCCP_CAM_D1`
 * `POWER_PORT_D` | 2      | `NVCCP_CAM_D2`
 * `POWER_PORT_D` | 3      | `NVCCP_CAM_D3`
 *
 * Passing values not on the above table will result in erroneous behavior and
 * should be avoided by the caller.
 *
 * @param[in] port The power port on which the camera is located
 * @param[in] link The link on which the camera is located
 *
 * @returns A `nvccp_cam_id` according to the table above.
 *
 */
static nvccp_cam_id
sGetCamId(
    PowerPort const port,
    uint8_t const link)
{
    nvccp_cam_id camid;

    switch(port) {
        case POWER_PORT_A:
            camid = getPowerPortA(link);
            break;
        case POWER_PORT_B:
            camid = getPowerPortB(link);
            break;
        case POWER_PORT_C:
            camid = getPowerPortC(link);
            break;
        default:
            camid = getPowerPortD(link);
            break;
    }

    return camid;
}

/**
 * \brief Determines which power port is associated with the provided index..
 *
 * The mapping is as follows:
 *
 * `pwrPortNum` | `PowerPort`
 * ------------ | --------------------
 * `0`          | `POWER_PORT_A`
 * `1`          | `POWER_PORT_B`
 * `2`          | `POWER_PORT_C`
 * `3`          | `POWER_PORT_D`
 *
 * Passing a `NvSiplCapInterfaceType` not on the above table will result in `POWER_PORT_INVALID`.
 *
 * @param[in] type The input `NvSiplCapInterfaceType`
 *
 * @returns `type` mapped as specified above, or `POWER_PORT_INVALID` if not on
 *          the table.
 *
 */
static PowerPort
PowerControlGetPowerPort(uint32_t const pwrPortNum)
{
    PowerPort port;

    switch(pwrPortNum) {
        case 0:
            port = POWER_PORT_A;
            break;
        case 1:
            port = POWER_PORT_B;
            break;
        case 2:
            port = POWER_PORT_C;
            break;
        case 3:
            port = POWER_PORT_D;
            break;
        default:
            port = POWER_PORT_INVALID;
            break;
    }
    return port;
}

/**
 * \brief Requests ownership of a power port via NvCCP
 *
 * The provided port must currently be `DISABLED`.
 * Following successful invocation of this function, `port` will be in the
 * `OWNED` state, and `powerPortStates[port].passive` will be set to the value of
 * `passive`. On failure, neither the port state nor the `passive` configuration
 * will be updated.
 *
 * The NvCCP camera group is determined from the port as specified by
 * `sGetGroupId`.
 *
 * @param[in] port The port to request ownership of
 * @param[in] passive
 *  True indicates the port should be requested in passive mode,
 *  false requests master mode.
 *
 * @retval NVSIPL_STATUS_OK if all operations succeed.
 * @retval NVSIPL_STATUS_BAD_ARGUMENT on parameter validation failure. (Failure)
 * @retval NVSIPL_STATUS_ERROR if the provided port is already owned. (Failure)
 * @retval NVSIPL_STATUS_ERROR if requesting ownership via NvCCP fails. (Failure)
 *
 */
static SIPLStatus
PowerControlRequestOwnership(
    PowerPort const port,
    bool const passive)
{
    SIPLStatus  nvsiplStatus {NVSIPL_STATUS_OK};

    if ((port >= POWER_PORT_INVALID) || (port < POWER_PORT_A)) {
        nvsiplStatus = NVSIPL_STATUS_BAD_ARGUMENT;
    } else if (powerPortStates[port].ownership != DISABLED) { /* Make sure
        we aren't requesting ownership for something that is already owned */
        nvsiplStatus = NVSIPL_STATUS_ERROR;
    } else {
        nvccp_return_t status;

        /* Request ownership */
        status = nvccp_request_ownership(sGetGroupId(port),
                                         passive ? NVCCP_CAM_SLAVE : NVCCP_CAM_MASTER);
        if (status != NVCCP_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("nvccp_request_ownership failed with status",
                                  static_cast<int32_t>(status));
        } else {
            /* Update the status */
            powerPortStates[port].passive = passive;
            powerPortStates[port].ownership = OWNED;
            nvsiplStatus = NVSIPL_STATUS_OK;
        }
    }
    return nvsiplStatus;
}

/**
 * \brief Performs the inverse of PowerControlRequestOwnership, releasing
 * ownership of the port via NvCCP.
 *
 * The port must be in the `OWNED` state prior to invocation of this function.
 * Following the successful completion of this function, it will be in the
 * `DISABLED` stated. The port will be released via `nvccp_release_ownership`.
 *
 * The NvCCP camera group is determined from the port as specified by
 * `sGetGroupId`.
 *
 * @param[in] port The port to release.
 *
 * @retval NVSIPL_STATUS_OK if all operations succeed.
 * @retval NVSIPL_STATUS_BAD_ARGUMENT if the provided `port` is invalid. (Failure)
 * @retval NVSIPL_STATUS_ERROR if `port` is not in the `OWNED` state. (Failure)
 * @retval NVSIPL_STATUS_ERROR releasing the port via NvCCP fails for any reason. (Failure)
 */
static SIPLStatus
PowerControlReleaseOwnership(
    PowerPort const port)
{
    SIPLStatus pwrStatus {NVSIPL_STATUS_OK};

    nvccp_return_t status;
    if ((port >= POWER_PORT_INVALID) || (port < POWER_PORT_A)) {
       pwrStatus = NVSIPL_STATUS_BAD_ARGUMENT;
       goto done;
    }
    status = nvccp_release_ownership(sGetGroupId(port),
                                         powerPortStates[port].passive ? NVCCP_CAM_SLAVE : NVCCP_CAM_MASTER);
    if (status != NVCCP_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("nvccp_release_ownership failed with status",
                              static_cast<int32_t>(status));
        pwrStatus = NVSIPL_STATUS_ERROR;
    } else {
        /* Now reset the ownership status and return */
        powerPortStates[port].ownership = DISABLED;
    }
done:
    return pwrStatus;
}

static SIPLStatus
PowerOnAggregator(
    PowerPort const port)
{
    /* Power on the aggregator */
    SIPLStatus pwrStatus {NVSIPL_STATUS_OK};
    nvccp_return_t const status = nvccp_set_aggreg_pwr_on(sGetGroupId(port));
    if ((status != NVCCP_STATUS_OK) && (status != NVCCP_STATUS_ALREADY_ON)) {
        SIPL_LOG_ERR_STR_INT("nvccp_set_aggreg_pwr_on failed with status",
                              static_cast<int32_t>(status));
        pwrStatus =  NVSIPL_STATUS_ERROR;
    } else {
        /* Update the power status */
        powerPortStates[port].ownership = POWERED;
    }
    return pwrStatus;
}

static SIPLStatus
PowerOffAggregator(
    PowerPort const port)
{
    /* Power off the aggregator */
    SIPLStatus pwrStatus {NVSIPL_STATUS_OK};
    nvccp_return_t const status = nvccp_set_aggreg_pwr_off(sGetGroupId(port));
    if ((status != NVCCP_STATUS_OK) && (status != NVCCP_STATUS_ALREADY_OFF)) {
        SIPL_LOG_ERR_STR_INT("nvccp_set_aggreg_pwr_off failed with status",
                              static_cast<int32_t>(status));
        pwrStatus = NVSIPL_STATUS_ERROR;
    } else {
        /* Update the power status */
        powerPortStates[port].ownership = OWNED;
        (void)PowerControlReleaseOwnership(port);
    }
    return pwrStatus;
}
} /* namespace nvsipl */

nvsipl::SIPLStatus
PowerControlSetAggregatorPower(
    uint32_t const pwrPortNum,
    bool const passive,
    bool const poweredOn)
{
    nvsipl::SIPLStatus pwrStatus;
    nvsipl::PowerPort const port {nvsipl::PowerControlGetPowerPort(pwrPortNum)};

    /* Verify port is valid */
    if ((port >= nvsipl::POWER_PORT_INVALID) || (port < nvsipl::POWER_PORT_A)) {
        pwrStatus = nvsipl::NVSIPL_STATUS_BAD_ARGUMENT;
    } else if (poweredOn) {    /* If the port is being powered on */
        nvsipl::SIPLStatus const nvsiplStatus = PowerControlRequestOwnership(port, passive);
        if (nvsiplStatus != nvsipl::NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("PowerControlSetAggregatorPower failed with SIPL error",
                                  static_cast<int32_t>(nvsiplStatus));
            pwrStatus = nvsiplStatus;
        } else if (nvsipl::powerPortStates[port].ownership != nvsipl::OWNED) {    /* Verify that the state is correct */
            pwrStatus = nvsipl::NVSIPL_STATUS_ERROR;
        } else {
            pwrStatus = PowerOnAggregator(port); /* Power on the aggregator */
        }
    /* If the port is being powered off */
    } else if (nvsipl::powerPortStates[port].ownership != nvsipl::POWERED) {    /* Verify that the state is correct */
        pwrStatus = nvsipl::NVSIPL_STATUS_ERROR;
    } else {
        pwrStatus = PowerOffAggregator(port); /* Power off the aggregator */
    }
    return pwrStatus;
}

nvsipl::SIPLStatus
PowerControlSetUnitPower(
    uint32_t const pwrPortNum,
    uint8_t const uLinkIndex,
    bool const poweredOn)
{
    nvsipl::SIPLStatus pwrStatus {nvsipl::NVSIPL_STATUS_OK};
    nvsipl::PowerPort const port {nvsipl::PowerControlGetPowerPort(pwrPortNum)};
    nvccp_return_t status;
    /* Verify port is valid */
     if ((port >= nvsipl::POWER_PORT_INVALID) || (port < nvsipl::POWER_PORT_A)) {
        pwrStatus = nvsipl::NVSIPL_STATUS_BAD_ARGUMENT;
        goto done;
    } else if (poweredOn) {    /* If the port is being powered on */
        /* Power on the unit */
        status = nvccp_set_cam_unit_pwr_on(sGetCamId(port, uLinkIndex));
        if ((status != NVCCP_STATUS_OK) && (status != NVCCP_STATUS_ALREADY_ON)) {
            SIPL_LOG_ERR_STR_INT("nvccp_set_cam_unit_pwr_on failed with status",
                                  static_cast<int32_t>(status));
            pwrStatus = nvsipl::NVSIPL_STATUS_ERROR;
        }
    } else {
        /* Power off the unit */
        status = nvccp_set_cam_unit_pwr_off(sGetCamId(port, uLinkIndex));
        if (status == NVCCP_STATUS_NOT_REGISTERED) {
            /* Power was already turned off or the deserilzier power is already turned off */
            pwrStatus = nvsipl::NVSIPL_STATUS_OK;
        } else if ((status != NVCCP_STATUS_OK) && (status != NVCCP_STATUS_ALREADY_OFF)) {
            SIPL_LOG_ERR_STR_INT("nvccp_set_cam_unit_pwr_off failed with status",
                                  static_cast<int32_t>(status));
            pwrStatus = nvsipl::NVSIPL_STATUS_ERROR;
        } else {
            /* Nothing */
        }
    }
done:
    return pwrStatus;
}

