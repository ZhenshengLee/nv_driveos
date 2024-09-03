/*
 * Copyright (c) 2007-2021 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef INCLUDED_nvrm_init_H
#define INCLUDED_nvrm_init_H


#if defined(__cplusplus)
extern "C"
{
#endif

/**
 * This file, APIs and data struct in it are deprecated and shouldn't be
 * used any longer. Don't include this file any more in new code.
 * The inclusion of the header in old files would be cleaned up.
 */

#include "nvcommon.h"
#include "nverror.h"

/**
 * The usage of NvRmDeviceHandle is deprecated and all the APIs that
 * take NvRmDeviceHandle as arg should replace it with void pointer and
 * remove any validations on it and ignore the arg.
 */
#include "nvrm_defs.h"

/** Macro to detect presence of NvRmOpenNew/NvRmClose API definition in
 * the header itself.
 */
#define NVRM_OPEN_CLOSE_DEFINED

/**
 * Deprecated API. Don't use this API.
 */
static NvError NV_UNUSED NvRmOpenNew(NvRmDeviceHandle *pHandle) {
	*pHandle = (NvRmDeviceHandle)1;
	return NvSuccess;
}

/**
 * Deprecated API. Don't use this API.
 */
static void NV_UNUSED NvRmClose(NvRmDeviceHandle NV_UNUSED hDevice) { }

/** @} */

#if defined(__cplusplus)
}
#endif

#endif
