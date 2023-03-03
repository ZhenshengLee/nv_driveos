/*
 * Copyright (c) 2014 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */


#include "nvfx.h"

#if NVFX_ADSP_OFFLOAD
#include "kernel/thread.h"

#else

#include "stdio.h"
#include "stdarg.h"
#include "time.h"

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#endif /* NVFX_ADSP_OFFLOAD */

#ifndef INFO
#define INFO    1
#endif

#if NVFX_ADSP_OFFLOAD && !defined __ARCH_ARM_OPS_H
static inline uint32_t arch_cycle_count(void)
{
	uint32_t count;
	__asm__ volatile("mrc		p15, 0, %0, c9, c13, 0"
			 : "=r" (count)
			 );
	return count;
}
#endif

uint64_t nvfx_get_sys_ts()
{
	uint64_t t;
#if NVFX_ADSP_OFFLOAD
	t = arch_cycle_count();
#else
	struct timespec tv;
#ifdef __MACH__
	clock_serv_t cclock;
	mach_timespec_t mts;
	host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
	clock_get_time(cclock, &mts);
	mach_port_deallocate(mach_task_self(), cclock);
	tv.tv_sec = mts.tv_sec;
	tv.tv_nsec = mts.tv_nsec;
#else
	clock_gettime(CLOCK_MONOTONIC, &tv);
#endif
	/* printf("sec:%d nsec:%d\n", (int)tv.tv_sec, (int)tv.tv_nsec); */
	t = tv.tv_sec * 1000000 + tv.tv_nsec / 1000;
#endif
	/*    printf("nvfx_get_sys_ts t:%llu\n", t); */
	return t;
}


uint32_t nvfx_get_exec_ts()
{
	uint64_t t;

#if NVFX_ADSP_OFFLOAD
	t = thread_get_cycle_count();
#else
	t = nvfx_get_sys_ts();
#endif
	/*    printf("nvfx_get_exec_ts t:%llu\n", t); */
	return (uint32_t)t;
}

#if !NVFX_ADSP_OFFLOAD
static __thread void* s_ptr;
#else
#define ADSP_TLS_ENTRY_FOR_PLUGIN  0
#endif
void nvfx_set_instance(void* ptr)
{
#if NVFX_ADSP_OFFLOAD
	tls_set(ADSP_TLS_ENTRY_FOR_PLUGIN, (uint32_t)ptr);
#else
	s_ptr = ptr;
#endif
}

void* nvfx_get_instance(void)
{
#if NVFX_ADSP_OFFLOAD
	return (void*)tls_get(ADSP_TLS_ENTRY_FOR_PLUGIN);
#else
	return s_ptr;
#endif
}



