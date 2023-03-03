/*
 * Copyright (c) 2007-2014 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "nvfx.h"

/*
 * Atomic APIs
 */
#if NVFX_ADSP_OFFLOAD
#define atomic_fetch_add(...)  __atomic_fetch_add(__VA_ARGS__, __ATOMIC_RELAXED);
#elif defined WINAPI_FAMILY
#include <Windows.h>
#define atomic_fetch_add(...)  InterlockedAdd(__VA_ARGS__)
#elif defined USE_STDATOMIC
#include "stdatomic.h"
#else
/* Race condition here.  Add MUST be atomic in multithreaded environment! */
#define atomic_fetch_add(ptr, num)  { /*assert(0);*/ *(ptr) += (num); }
#endif

#include <kernel/thread.h>
#ifndef CONFIG_APE_1_0
#include <lib/apralloc.h>
#endif

#define min(a, b) \
	({ __typeof__ (a) _a = (a); \
	   __typeof__ (b) _b = (b); \
	   _a < _b ? _a : _b; })

/**
 * NVFX Memory APIs
 *
 */
void nvfx_mem_init(nvfx_mem_t* mem,
		   variant_t addr,
		   const int32_t size,
		   const uint32_t flags)
{
	mem->addr = addr;
	mem->size = size;
	mem->flags = flags;
}


/**
 * NVFX Buffer APIs
 *
 */
void nvfx_buffer_init(nvfx_buffer_t* buffer,
		      variant_t addr,
		      const int32_t size,
		      const uint32_t flags)
{
	nvfx_mem_init(&buffer->mem, addr, size, flags);
	buffer->read = 0;
	buffer->write = 0;
	buffer->valid_bytes = 0;
}

#ifndef CONFIG_APE_1_0
/*
* This function will try to allocate APR memory in ARAM region, if it fails
* then it will try allocating on APR DRAM.
*/
void *nvfx_apr_alloc(size_t size,
		      unsigned int alignment,
		      nvfx_t* fx,
		      int32_t apr_mem_flag)
{
	void *ptr = NULL;

	if ((apr_mem_flag & APR_MEM_ARAM) == APR_MEM_ARAM) {
		ptr = aramapr_alloc(size, alignment);
		fx->apr_mem_flag = APR_MEM_ARAM;
		if (ptr == NULL)
			printf("APR ARAM allocation Failed!! Trying to allocate from APR DRAM\n");
	}

	if ((ptr == NULL) && (apr_mem_flag != APR_MEM_NONE)) {
		ptr = dramapr_alloc(size, alignment);
		fx->apr_mem_flag = APR_MEM_DRAM;
	}

	if (ptr == NULL) {
		printf("APR allocation Failed\n");
		fx->apr_mem_flag = APR_MEM_NONE;
	}

	return ptr;
}

void nvfx_apr_free(nvfx_t* fx)
{
	int32_t apr_mem_flag = fx->apr_mem_flag;

	if (fx->apr_mem == NULL)
		return;

	if (apr_mem_flag == APR_MEM_ARAM)
		aramapr_free(fx->apr_mem);
	else if (apr_mem_flag == APR_MEM_DRAM)
		dramapr_free(fx->apr_mem);

	fx->apr_mem_flag = APR_MEM_NONE;
	fx->apr_mem = NULL;
}
#endif

int32_t nvfx_buffer_bytes_to_end(const nvfx_buffer_t* buffer,
				 const int32_t offset)
{
	return buffer->mem.size - offset;
}


int32_t nvfx_buffer_valid_bytes(const nvfx_buffer_t* buffer)
{
	return buffer->valid_bytes;
}

int32_t nvfx_buffer_valid_bytes_contiguous(const nvfx_buffer_t* buffer)
{
	int32_t valid_bytes = buffer->valid_bytes;
	return valid_bytes > buffer->mem.size - buffer->read ?
	buffer->mem.size - buffer->read :
	valid_bytes;
}

int32_t nvfx_buffer_bytes_free(const nvfx_buffer_t* buffer)
{
	return buffer->mem.size - buffer->valid_bytes;
}

int32_t nvfx_buffer_bytes_free_contiguous(const nvfx_buffer_t* buffer)
{
	int32_t bytes_free = nvfx_buffer_bytes_free(buffer);
	return bytes_free > buffer->mem.size - buffer->write ?
	buffer->mem.size - buffer->write :
	bytes_free;
}

void nvfx_buffer_copy_in(nvfx_buffer_t* buffer,
			 const void* source,
			 const int32_t num_bytes)
{
	int32_t write = buffer->write;
	int32_t bytes_to_end = nvfx_buffer_bytes_to_end(buffer, write);
	if (num_bytes <= bytes_to_end) {
		memcpy(buffer->mem.addr.puint8 + write, source, num_bytes);
	} else {
		memcpy(buffer->mem.addr.puint8 + write, source, bytes_to_end);
		memcpy(buffer->mem.addr.puint8, (uint8_t*)source + bytes_to_end,
		       num_bytes - bytes_to_end);
	}
}

void nvfx_buffer_copy_out(nvfx_buffer_t* buffer,
			  void* destination,
			  const int32_t num_bytes)
{
	int32_t read = buffer->read;
	int32_t bytes_to_end = nvfx_buffer_bytes_to_end(buffer, read);
	if (num_bytes <= bytes_to_end) {
		memcpy(destination, buffer->mem.addr.puint8 + read, num_bytes);
	} else {
		memcpy(destination, buffer->mem.addr.puint8 + read, bytes_to_end);
		memcpy((uint8_t*)destination + bytes_to_end, buffer->mem.addr.pvoid,
		       num_bytes - bytes_to_end);
	}
}

void nvfx_buffer_set(nvfx_buffer_t* buffer,
		     int8_t value,
		     const int32_t num_bytes)
{
	int32_t write = buffer->write;
	int32_t bytes_to_end = nvfx_buffer_bytes_to_end(buffer, write);
	if (num_bytes <= bytes_to_end) {
		memset(buffer->mem.addr.puint8 + write, value, num_bytes);
	} else {
		memset(buffer->mem.addr.puint8 + write, value, bytes_to_end);
		memset(buffer->mem.addr.puint8, value,
		       num_bytes - bytes_to_end);
	}
}

int32_t nvfx_buffer_add_bytes(nvfx_buffer_t* buffer, const int32_t num_bytes)
{
	enter_critical_section();
	int32_t write = buffer->write;
	int32_t bytes_added = min(buffer->mem.size - buffer->valid_bytes, num_bytes);

	if (bytes_added < num_bytes) {
		printf("Overrun in %p by %d bytes\n", buffer->mem.addr.pint,
		       num_bytes - bytes_added);
	}

	buffer->write = (write + bytes_added) % buffer->mem.size;
	atomic_fetch_add(&buffer->valid_bytes, bytes_added);
	exit_critical_section();
	return num_bytes - bytes_added;
}

int32_t nvfx_buffer_consume_bytes(nvfx_buffer_t* buffer,
				  const int32_t num_bytes)
{
	enter_critical_section();
	int32_t read = buffer->read;
	int32_t bytes_read = min(buffer->valid_bytes, num_bytes);

	if (bytes_read < num_bytes) {
		printf("Underrun in %p by %d bytes\n", buffer->mem.addr.pint,
		       num_bytes - bytes_read);
	}

	buffer->read = (read + bytes_read) % buffer->mem.size;
	atomic_fetch_add(&buffer->valid_bytes, -bytes_read);
	exit_critical_section();
	return num_bytes - bytes_read;
}
