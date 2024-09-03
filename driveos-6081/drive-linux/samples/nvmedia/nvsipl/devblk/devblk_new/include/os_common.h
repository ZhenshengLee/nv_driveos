/*
 * Copyright (c) 2015-2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_OS_COMMON_H
#define INCLUDED_OS_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

#ifdef NVMEDIA_QNX
#define nvsleep(usec) (((usec) < 10000) ? nanospin_ns((usec) * 1000) : usleep(usec))
#else
#define nvsleep(usec) usleep(usec)
#endif

#ifdef __cplusplus
}
#endif

#endif /* INCLUDED_OS_COMMON_H */

