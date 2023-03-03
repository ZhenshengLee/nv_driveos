/*
* Copyright (c) 2014-2015 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA Corporation is strictly prohibited.
*/


#ifndef _NVFX_SDK_WIRE_H_
#define _NVFX_SDK_WIRE_H_

#include "nvfx.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SDK_WIRE_GUID 0x75f11827, 0x15464b20, 0xad2409ca, 0x9a2b625c
	NVID(NVID_SDK_WIRE, SDK_WIRE_GUID);

/*
 * No external call parameters to define
 */

/*
 * Required export
 */
extern const invfx_t* nvfx_sdk_wire;

#ifdef __cplusplus
}
#endif

#endif /* #ifndef _NVFX_SDK_WIRE_H_ */
