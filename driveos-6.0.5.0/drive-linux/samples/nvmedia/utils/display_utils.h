/*
 * Copyright (c) 2018-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "nvmedia_idp.h"
#include "nvmedia_vop.h"

//  GetAvailableVideoDisplayDevices
//
//    GetAvailableVideoDisplayDevices()  Returns a list of available display devices using
//                                       video APIs.
//                                       It's the responsibility of the calling function
//                                       to release this memory after use.
//
//  Arguments:
//
//   outputDevicesNum
//      (out) Pointer to number of devices
//
//   outputDevicesList
//      (out) Pointer to a list of available output devices

NvMediaStatus
GetAvailableVideoDisplayDevices(
    int *outputDevicesNum,
    NvMediaVideoOutputDeviceParams *outputDevicesList);

//  GetAvailableIDPDisplayDevices
//
//    GetAvailableIDPDisplayDevices()    Returns a list of available display devices using
//                                       image APIs.
//                                       It's the responsibility of the calling function
//                                       to release this memory after use.
//
//  Arguments:
//
//   outputDevicesNum
//      (out) Pointer to number of devices
//
//   outputDevicesList
//      (out) Pointer to a list of available output devices

NvMediaStatus
GetAvailableIDPDisplayDevices(
    int32_t *outputDevicesNum,
    NvMediaIDPDeviceParams *outputDevicesList);

//  CheckVideoDisplayDeviceID
//
//    CheckVideoDisplayDeviceID()  Check display by display id using video APIs
//
//  Arguments:
//
//   displayId
//      (in) Requested display id
//
//   enabled
//      (out) Pointer to display enabled flag

NvMediaStatus
CheckVideoDisplayDeviceID(
    unsigned int displayId,
    NvMediaBool *enabled);

//  CheckIDPDisplayDeviceID
//
//    CheckIDPDisplayDeviceID()  Check display by display id using image APIs
//
//  Arguments:
//
//   displayId
//      (in) Requested display id
//
//   enabled
//      (out) Pointer to display enabled flag

NvMediaStatus
CheckIDPDisplayDeviceID(
    unsigned int displayId,
    NvMediaBool *enabled);

#ifdef __cplusplus
}    /* extern "C" */
#endif
