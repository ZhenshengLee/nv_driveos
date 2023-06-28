/* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __POWER_UTILS_H__
#define __POWER_UTILS_H__

#include <stdbool.h>

#include "nvmedia_icp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Which port to enable or disable the power for */
typedef enum {
    POWER_PORT_A = 0,
    POWER_PORT_B,
    POWER_PORT_C,
    POWER_PORT_D,
    POWER_PORT_INVALID
} PowerPort;

/* Convert a string like "trio-ab" to a port */
PowerPort
PowerControlPortFromString(
    const char* interface);

/* Convert an interface to a port */
PowerPort
PowerControlPortFromInterface(
    NvMediaICPInterfaceType type);

/* Request master or slave ownership of the given port */
NvMediaStatus
PowerControlRequestOwnership(
    PowerPort port,
    bool slave);

/* Release an owned port */
NvMediaStatus
PowerControlReleaseOwnership(
    PowerPort port);

/* Set the power status of the given port */
NvMediaStatus
PowerControlSetPower(
    PowerPort port,
    bool poweredOn);

#ifdef __cplusplus
}
#endif

#endif /* __POWER_UTILS_H__ */
