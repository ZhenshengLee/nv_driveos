/*
 * Copyright (c) 2014-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */


#ifndef _NVFX_H_
#define _NVFX_H_

#include <stdbool.h>
#include "stdio.h"
#include "stdint.h"
#include "string.h"
#include "tegra_nvfx.h"

#ifndef NVFX_ADSP_OFFLOAD
#define NVFX_ADSP_OFFLOAD       0
#endif

#ifndef NVFX_USE_STD_INTERFACE
#define NVFX_USE_STD_INTERFACE  1
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Forward Declarations
 */
typedef struct invfx_t invfx_t;

/**
 * nvfx_get_sys_ts - Returns current system timestamp
 *     This timestamp should be used to note the last time any processing
 *     occurred.
 */
uint64_t nvfx_get_sys_ts(void);

/**
 * nvfx_get_exec_ts - Returns execution timestamp
 *    This timestamp should be used in calculations of all execution
 *    time related variables.
 */
uint32_t nvfx_get_exec_ts(void);

/**
 * nvfx_ts_diff - Returns difference between two timestamps
 *      taking rollover into account.
 *
 * @start           Start timestamp
 * @end             End timestamp
 *
 */
inline static uint32_t nvfx_ts_diff(uint32_t start, uint32_t end)
{
	uint32_t t;
	if (start <= end) {
		t = end - start;
	}
	else {
		t = UINT32_MAX - start + end;
	}

	return t;
}

/**
 * nvfx_set_instance - Stores an instance pointer in thread local storage.
 *      The pointer will only be valid for the duration of the current
 *      call to the plugin.
 *
 * @ptr             Instance data to store in the thread context
 */
void nvfx_set_instance(void* ptr);

/**
 * nvfx_get_instance - Gets the instance pointer from thread local storage.
 *
 */
void* nvfx_get_instance(void);


/**
 * nvid_t
 *
 */
typedef struct {
	uint32_t data1;
	uint32_t data2;
	uint32_t data3;
	uint32_t data4;
} nvid_t;

#ifdef NVFX_DEFINE_CONSTS
#define NVID(name, g)   const nvid_t name = { g }
#else
#define NVID(name, g)   extern const nvid_t name;
#endif

#define NVID_NULL_GUID  0x00000000, 0x00000000, 0x00000000, 0x00000000
NVID(NVID_NULL, NVID_NULL_GUID);

#define nvid_is_equal(id1, id2) \
((id1)->data1 == (id2)->data1 && (id1)->data2 == (id2)->data2 && \
(id1)->data3 == (id2)->data3 && (id1)->data4 == (id2)->data4))

/**
 * Compiler Specific Variable and Memory Section Attributes
 *
 * @NVFX_VAR_UNUSED             Compiler specific unused variable attribute
 * @NVFX_VAR_USED               Compiler specific used variable attribute
 * @NVFX_MEM_SECTION            Compiler specific memory section specifier
 * @NVFX_MEM_INSTANCE           Instance memory
 * @NVFX_MEM_SHARED             Cross-processor, non-cached memory
 * @NVFX_MEM_FAST_SHARED        Cross-processor, shared memory
 * @NVFX_MEM_INTERNAL           Internal memory if available or default
 * @NVFX_MEM_REQ_INTERNAL       Required internal memory
 *
 */
#if NVFX_ADSP_OFFLOAD
#define NVFX_VAR_UNUSED         __attribute__((unused))
#define NVFX_VAR_USED           __attribute__((used))
#define NVFX_MEM_SECTION(s)     __attribute__((section(s))) NVFX_VAR_USED
#define NVFX_MEM_INSTANCE       NVFX_MEM_SECTION(".dram_data")
#define NVFX_MEM_SHARED         NVFX_MEM_SECTION(".dram_shared")
#define NVFX_MEM_FAST_SHARED    NVFX_MEM_SECTION(".dram_shared_wc")
#define NVFX_MEM_INTERNAL       NVFX_MEM_SECTION(".aram_data")
#define NVFX_MEM_REQ_INTERNAL   NVFX_MEM_SECTION(".aram_x_data")
#else  /* #if NV_IS_ADSP */
#define NVFX_VAR_USED
#define NVFX_VAR_UNUSED
#define NVFX_MEM_SECTION(s)
#define NVFX_MEM_INSTANCE
#define NVFX_MEM_SHARED
#define NVFX_MEM_FAST_SHARED
#define NVFX_MEM_INTERNAL
#define NVFX_MEM_REQ_INTERNAL
#endif /* #if NV_IS_ADSP */

#ifndef CONFIG_APE_1_0
#define NVFX_APR_MEM_ALIGN    (16)
/**
 * APR MEMORY FLAG
 *
 * @APR_MEM_NONE    No APR memory allocation
 * @APR_MEM_ARAM    Allocate APR memory in ARAM
 * @APR_MEM_DRAM    Allocate APR memory in DRAM
*/
enum {
	APR_MEM_NONE = 0,
	APR_MEM_ARAM,
	APR_MEM_DRAM
};
#endif

/**
 * nvfx_mem_t
 *
 * @buffer          Base address
 * @size            Size in bytes
 *
 */
typedef struct {
	variant_t addr;
	int32_t size;
	uint32_t flags;
} nvfx_mem_t;

/**
 * nvfx_mem_init - Initializes the NVFX Memory
 *
 * @mem             Memory instance
 * @addr            Base address
 * @size            Size in bytes
 * @flags           Flags
 *
 */
void nvfx_mem_init(nvfx_mem_t* mem,
		   variant_t addr,
		   const int32_t size,
		   const uint32_t flags);


/**
 * nvfx_buffer_t
 *
 * @mem             Memory instance
 * @read            Read byte offset
 * @write           Write byte offset
 * @valid_bytes     Valid bytes in the buffer; Use atomic operations to modify
 *
 */
typedef struct {
	nvfx_mem_t mem;
	volatile int32_t read;
	volatile int32_t write;
	volatile int32_t valid_bytes;
} nvfx_buffer_t;

/**
 * nvfx_buffer_init - Initializes the NVFX Buffer
 *
 * @nvfx_buffer:    NVFX buffer instance
 * @addr:           Base address
 * @size:           Size in bytes
 * @flags:          Flags
 *
 */
void nvfx_buffer_init(nvfx_buffer_t* buffer,
		      variant_t addr,
		      const int32_t size,
		      const uint32_t flags);

/**
 * nvfx_buffer_bytes_to_end - Returns the number of bytes from the offset
 *                  to the end of the buffer
 *
 * @nvfx_buffer     NVFX buffer instance
 * @offset          Offset
 *
 */
int32_t nvfx_buffer_bytes_to_end(const nvfx_buffer_t* buffer,
				 const int32_t offset);

/**
 * nvfx_buffer_valid_bytes - Returns the number of valid bytes in the buffer
 *
 * @nvfx_buffer     NVFX buffer instance
 *
 */
int32_t nvfx_buffer_valid_bytes(const nvfx_buffer_t* buffer);

/**
 * nvfx_buffer_valid_bytes_contiguous - Returns the number of contiguous
 * valid bytes in the buffer
 *
 * @nvfx_buffer     NVFX buffer instance
 *
 */
int32_t nvfx_buffer_valid_bytes_contiguous(const nvfx_buffer_t* buffer);

/**
 * nvfx_buffer_bytes_free - Returns the number of free bytes in the buffer
 *
 * @nvfx_buffer     NVFX buffer instance
 *
 */
int32_t nvfx_buffer_bytes_free(const nvfx_buffer_t* buffer);

/**
 * nvfx_buffer_bytes_free_contiguous - Returns the number of contiguous
 * free bytes in the buffer
 *
 * @nvfx_buffer     NVFX buffer instance
 *
 */
int32_t nvfx_buffer_bytes_free_contiguous(const nvfx_buffer_t* nvfx_buffer);

/**
 * nvfx_buffer_copy_in - Copy data from the source buffer to the NVFX buffer
 *
 * @nvfx_buffer     NVFX buffer instance
 * @source          Source buffer
 * @num_bytes       Number of bytes to copy
 *
 */
void nvfx_buffer_copy_in(nvfx_buffer_t* buffer,
			 const void* source,
			 const int32_t num_bytes);

/**
 * nvfx_buffer_copy_out - Copy data from the NVFX buffer to the destination
 *
 * @nvfx_buffer     NVFX buffer instance
 * @destination     Destination buffer
 * @num_bytes       Number of bytes to copy
 *
 */
void nvfx_buffer_copy_out(nvfx_buffer_t* buffer,
			  void* destination,
			  const int32_t num_bytes);

/**
 * nvfx_buffer_set - Sets memory to the specified value
 *
 * @nvfx_buffer     NVFX buffer instance
 * @value           Value to set
 * @num_bytes       Number of bytes to set
 *
 */
void nvfx_buffer_set(nvfx_buffer_t* buffer,
		     int8_t value,
		     const int32_t num_bytes);

/**
 * nvfx_buffer_add_bytes - Updates the write pointer with the number of
 *                  bytes added and returns overrun bytes
 *
 * @nvfx_buffer     NVFX buffer instance
 * @num_bytes       Number of bytes to added
 *
 */
int32_t nvfx_buffer_add_bytes(nvfx_buffer_t* buffer, const int32_t num_bytes);

/**
 * nvfx_buffer_consume_bytes - Updates the read pointer with the number of
 *                  bytes consumed and returns underrun bytes
 *
 * @nvfx_buffer     NVFX buffer instance
 * @num_bytes       Number of bytes consumed
 *
 */
int32_t nvfx_buffer_consume_bytes(nvfx_buffer_t* buffer,
				  const int32_t num_bytes);

#define PRINT_NVFX_BUFFER(b) \
	dprintf(INFO, "%s@%p: sz: %d r: %d w: %d vb: %d\n",\
	#b, b->mem.addr.pint, b->mem.size, b->read, b->write, b->valid_bytes)

/**
 * nvfx_t - Required instance data
 *
 * @ifx                 Constant FX interface
 * @cached_state        Cached, shared state
 * @shared_state        Non-cached, shared state
 * @input_buffer        Pointer to array of input buffer instance data
 * @output_buffer       Pointer to array of output buffer instance data
 * @is_apr              Flag for apr or non apr memory usage
 * @apr_mem_flag        Flag to choose APR memory in ARAM/DRAM
 * @non_apr_mem         Non apr memory address
 * @apr_mem             Apr memory adress
 *
 */
typedef struct {
	const invfx_t* ifx;
	nvfx_shared_state_t* cached_state;
	nvfx_shared_state_t* noncached_state;
	nvfx_buffer_t** input_buffer;
	nvfx_buffer_t* output_buffer;
	/* custom params */
	bool is_apr;
	int32_t apr_mem_flag;
	void *non_apr_mem;
	void *apr_mem;
} nvfx_t;

#ifndef CONFIG_APE_1_0
/**
 * nvfx_apr_alloc - Allocate the APR memory in ARAM/DRAM based on the apr_mem_flag.
 *
 * @size                Size of the APR buffer in bytes
 * @alignment           Alignment of APR buffer memory start address
 * @fx                  FX instance memory address
 * @apr_mem_flag        Flag to choose APR memory in ARAM/DRAM
 *
*/
void *nvfx_apr_alloc(size_t size,
		     unsigned int alignment,
		     nvfx_t* fx,
		     int32_t apr_mem_flag);

/**
 * nvfx_apr_free - Free the APR memory in ARAM/DRAM.
 *
 * @fx                  FX instance memory address
 *
*/
void nvfx_apr_free(nvfx_t* fx);
#endif

/**
 * nvfx_init_t - Initializes the FX
 *
 * @ifx                 Pointer to FX interface
 * @fx                  FX instance memory address
 * @shared_state        Shared memory address
 * @fast_shared_mem     Fast shared memory address
 * @internal_mem        Internal memory address
 * @req_internal_mem    Required internal memory address
 *
 */
typedef void (*nvfx_init_t)(const invfx_t* ifx,
			    nvfx_t* fx,
			    nvfx_shared_state_t* shared_state,
			    void* fast_shared_mem,
			    void* internal_mem,
			    void* req_internal_mem);

/**
 * nvfx_close_t - Cleans up the FX
 *
 * @fx                  FX instance
 *
 */
typedef void (*nvfx_close_t)(nvfx_t* fx);

/**
 * nvfx_call_t - Function to get/set runtime parameters
 *
 * @fx                  FX instance
 * @call_params         Pointer to 'call' parameters
 *
 */
typedef void (*nvfx_call_t)(nvfx_t* fx, nvfx_call_params_t* call_params);

/**
 * nvfx_process_t - Function to process data
 *
 * @fx                  FX instance
 *
 */
typedef void (*nvfx_process_t)(nvfx_t* fx);

/**
 * invfx_t - Required library variable for linking
 *
 * Constants
 * ---------
 * @size                    Size of the invfx_t struct
 * @id                      FX ID
 * @version                 Version
 * @type                    Type
 *
 * @instance_mem_size       Instance memory size
 * @shared_mem_size         Shared memory size
 * @fast_shared_mem_size    Fast shared memory size
 * @internal_mem_size       Internal memory size
 * @req_internal_mem_size   Required internal memory size
 *
 * @max_call_params_size    Maximum size of unioned call_params
 *
 * @num_input_buffers       Number of input buffers
 * @num_output_buffers      Number of output buffers
 * @output_buffer_size[]    Size of output buffer
 *
 * @max_process_time        Max process time
 * @period                  Likely scheduling period
 *
 *
 * Function Pointers
 * -----------------
 * @init                    Function to initialize the FX
 * @close                   Function to close the FX
 * @call                    Function to call internal function
 * @process                 Function to process data
 *
 */
struct invfx_t {
	uint32_t size;
	nvid_t id;
	uint32_t version;
	uint32_t type;

	uint32_t instance_mem_size;
	uint32_t shared_mem_size;
	uint32_t fast_shared_mem_size;
	uint32_t internal_mem_size;
	uint32_t req_internal_mem_size;

	uint32_t max_call_params_size;

	uint32_t num_input_pins;
	uint32_t num_output_pins;

	uint32_t max_process_time;
	uint32_t period;

	/* Function Pointers */
	nvfx_init_t init;
	nvfx_close_t close;
	nvfx_call_t call;
	nvfx_process_t process;
};

#if NVFX_USE_STD_INTERFACE
extern const invfx_t* nvfx;
#endif

/**
 * nvfx_init - Base struct initializer
 *
 */
inline static void nvfx_init(nvfx_t* fx, const invfx_t* ifx,
			     nvfx_shared_state_t* cached_state,
			     nvfx_shared_state_t* noncached_state,
			     nvfx_buffer_t** input_buffer,
			     nvfx_buffer_t* output_buffer)
{
	memset(fx, 0, ifx->instance_mem_size);

	fx->ifx = ifx;
	fx->cached_state = cached_state;
	fx->noncached_state = noncached_state;
	fx->input_buffer = input_buffer;
	fx->output_buffer = output_buffer;

	memset(cached_state, 0, sizeof(nvfx_shared_state_t));
	cached_state->process.time_low = UINT32_MAX;
	*noncached_state = *cached_state;
}

/**
 * nvfx_process_start - Initialize timestamps
 *
 */
inline static void nvfx_process_start(nvfx_t* fx, uint32_t* ts_start)
{
	nvfx_process_state_t* csp = &fx->cached_state->process;
	nvfx_process_state_t* ncsp = &fx->noncached_state->process;

	ncsp->ts_last = csp->ts_last = nvfx_get_sys_ts();
	*ts_start = nvfx_get_exec_ts();
}

/**
 * nvfx_process - Process struct update
 *
 */
inline static void nvfx_process(nvfx_t* fx, uint32_t ts_start, uint32_t period)
{
	nvfx_process_state_t* csp = &fx->cached_state->process;
	nvfx_process_state_t* ncsp = &fx->noncached_state->process;
	uint32_t ts_diff = nvfx_ts_diff(ts_start, nvfx_get_exec_ts());

	if (ts_start > 0) {
		csp->time_last = ts_diff;
		csp->time_total += ts_diff;
		if (ts_diff < csp->time_low) {
			csp->time_low = ts_diff;
		}
		else if (csp->time_high < ts_diff) {
			csp->time_high = ts_diff;
		}

		csp->count++;
		csp->period = period;
	}

	*ncsp = *csp;
}

/**
 * nvfx_reset - Reset buffers and state
 *
 */
inline static void nvfx_reset(nvfx_t* fx)
{
	uint32_t i;

	if (fx->noncached_state->process.count > 0) {
		printf("\nProcessing Cycles for GUID 0x%x,0x%x,0x%x,0x%x\n",
			fx->ifx->id.data1, fx->ifx->id.data2,
			fx->ifx->id.data2, fx->ifx->id.data3);
		printf("  Calls: %u\n", fx->noncached_state->process.count);
		printf("  High: %u\n", fx->noncached_state->process.time_high);
		printf("  Low: %u\n", fx->noncached_state->process.time_low);
		printf("  Last: %u\n", fx->noncached_state->process.time_last);
		printf("  Total: %llu\n", fx->noncached_state->process.time_total);
		printf("  Avg: %llu\n", fx->noncached_state->process.time_total /
			fx->noncached_state->process.count);
	}

	for (i = 0; i < fx->ifx->num_input_pins; i++) {
		if (fx->input_buffer[i] && fx->input_buffer[i]->mem.addr.pint) {
			nvfx_buffer_init(fx->input_buffer[i],
				fx->input_buffer[i]->mem.addr,
				fx->input_buffer[i]->mem.size,
				fx->input_buffer[i]->mem.flags);
		}
	}

	for (i = 0; i < fx->ifx->num_output_pins; i++) {
		if (fx->output_buffer[i].mem.addr.pint) {
			nvfx_buffer_init(&fx->output_buffer[i],
				fx->output_buffer[i].mem.addr,
				fx->output_buffer[i].mem.size,
				fx->output_buffer[i].mem.flags);
			nvfx_buffer_set(&fx->output_buffer[i], 0,
				fx->output_buffer[i].mem.size);
		}
	}

	memset(fx->cached_state, 0, sizeof(nvfx_shared_state_t));
	memset(fx->noncached_state, 0, sizeof(nvfx_shared_state_t));
}

#ifdef __cplusplus
}
#endif

#endif // #ifndef _NVFX_H_
