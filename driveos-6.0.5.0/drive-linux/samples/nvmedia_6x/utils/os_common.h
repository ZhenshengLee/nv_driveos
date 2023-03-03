/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _NVMEDIA_OS_COMMON_H_
#define _NVMEDIA_OS_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

#if defined(__INTEGRITY)
#include <INTEGRITY.h>
#else
#include <malloc.h>
#endif

#ifdef NVMEDIA_QNX
#define nvsleep(usec) (((usec) < 10000) ? nanospin_ns((usec) * 1000) : usleep(usec))
#else
#define nvsleep(usec) usleep(usec)
#endif

#if !defined(__GNUC__) || defined(__INTEGRITY)
#define NVM_PREFETCH(ptr)  \
    __asm__ __volatile__( "pld [%[memloc]]     \n\t"  \
    : : [memloc] "r" (ptr) : "cc" );
#else
#define NVM_PREFETCH(ptr)  \
        __builtin_prefetch(ptr, 1, 1);
#endif

#ifdef __cplusplus
}
#endif

#endif /* _NVMEDIA_OS_COMMON_H_ */

