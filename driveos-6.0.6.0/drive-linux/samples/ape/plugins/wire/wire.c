/*
* Copyright (c) 2012-2021, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA Corporation is strictly prohibited.
*/

#include <string.h>
#include <stdlib.h>
#include "plugin.h"

#include "wire.h"

#define min(a, b) \
    ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a < _b ? _a : _b; })

#define NVFX_WIRE_SAMPLES_PER_BLOCK    (128)
#define NVFX_WIRE_BLOCK_SIZE           (NVFX_WIRE_SAMPLES_PER_BLOCK * 2 * 2)
#define NVFX_WIRE_OUTPUT_BUFFER_SIZE   (NVFX_WIRE_BLOCK_SIZE * 16)

typedef struct
{
	nvfx_t fx;
	nvfx_shared_state_t cached_state;
	nvfx_buffer_t* input_buffer[1];
	nvfx_buffer_t output_buffer[1];

	int32_t bytes_avail;
	uint8_t buffer[NVFX_WIRE_BLOCK_SIZE];

} wire_instance_t;

static void wire_reset(nvfx_t* fx)
{
	wire_instance_t* wire = (wire_instance_t*)fx;

	/* Initialize internal buffer */
	wire->bytes_avail = 0;
	memset(wire->buffer, 0, sizeof(wire->buffer));

	nvfx_reset(fx);
}

static void wire_init(
	const invfx_t* ifx,
	nvfx_t* fx,
	nvfx_shared_state_t* shared_state,
	void* fast_shared_mem,
	void* internal_mem,
	void* req_internal_mem
	)
{
	wire_instance_t* wire = (wire_instance_t*)fx;
	variant_t addr;

	nvfx_init(fx, ifx, &wire->cached_state, shared_state,
		wire->input_buffer, wire->output_buffer);

	/* Our output buffer is located at start of fast, shared memory */
	addr.puint8 = (uint8_t*)fast_shared_mem;
	nvfx_buffer_init(fx->output_buffer, addr, NVFX_WIRE_OUTPUT_BUFFER_SIZE,
		NVFX_MEM_ALL_ACCESS);

	wire_reset(fx);
}

static int32_t do_something(wire_instance_t* wire,
	nvfx_buffer_t* ibuffer,
	nvfx_buffer_t* obuffer,
	int32_t* bytes_consumed)
{
	int32_t bytes_to_copy;
	int32_t bytes_produced = 0;
	int32_t bytes_free = nvfx_buffer_bytes_free(obuffer);
	int32_t bytes_in = nvfx_buffer_valid_bytes(ibuffer);

	bytes_to_copy = min(NVFX_WIRE_BLOCK_SIZE - wire->bytes_avail, bytes_in);
	if (bytes_to_copy > 0) {
		nvfx_buffer_copy_out(ibuffer, wire->buffer + wire->bytes_avail, bytes_to_copy);
		wire->bytes_avail += bytes_to_copy;
	}
	*bytes_consumed = bytes_to_copy;

	/* If we have one block of data, the process it */
	if (wire->bytes_avail == NVFX_WIRE_BLOCK_SIZE &&
		bytes_free >= NVFX_WIRE_BLOCK_SIZE) {
		/* Just copy input to output */
		nvfx_buffer_copy_in(obuffer, wire->buffer, NVFX_WIRE_BLOCK_SIZE);
		wire->bytes_avail = 0;
		bytes_produced = NVFX_WIRE_BLOCK_SIZE;
	}

	return bytes_produced;
}

static void wire_process(nvfx_t* fx)
{
	wire_instance_t* wire = (wire_instance_t*)fx;

	nvfx_buffer_t* ibuf = wire->fx.input_buffer[0];
	nvfx_buffer_t* obuf = wire->fx.output_buffer;
	nvfx_shared_state_t* cs = fx->cached_state;
	nvfx_shared_state_t* ncs = fx->noncached_state;

	int32_t bytes_produced;
	uint32_t ts_start;
	nvfx_process_start(fx, &ts_start);
	do {
		int32_t bytes_consumed = 0;
		/* Process data */
		bytes_produced = do_something(wire,
			ibuf,
			obuf,
			&bytes_consumed);

		nvfx_buffer_consume_bytes(ibuf, bytes_consumed);
		cs->input[0].bytes += bytes_consumed;

		nvfx_buffer_add_bytes(obuf, bytes_produced);
		cs->output[0].bytes += bytes_produced;
		cs->output[0].frames++;
	} while (nvfx_buffer_valid_bytes(ibuf) && bytes_produced);

	/* Update non-cached state */
	ncs->input[0] = cs->input[0];
	ncs->output[0] = cs->output[0];
	nvfx_process(fx, ts_start, 0);
}

static void wire_close(nvfx_t* fx)
{
}

static void wire_set_state(nvfx_t* fx, nvfx_set_state_params_t* set_state_params)
{
	int32_t state = set_state_params->state;

	switch (state)
	{
	case nvfx_state_inactive:
	case nvfx_state_active:
		/* Nothing to do */
		break;
	default:
		break;
	}

	fx->noncached_state->process.state = fx->cached_state->process.state = state;
}

static void wire_call(nvfx_t* fx, nvfx_call_params_t* call_params)
{
	switch (call_params->method)
	{
	case nvfx_method_set_state:
	{
		wire_set_state(fx, (nvfx_set_state_params_t*)call_params);
	}
		break;

	case nvfx_method_reset:
	{
		wire_reset(fx);
	}
		break;

	default:
		break;
	}
}

static const invfx_t invfx_wire =
{
	sizeof(invfx_t),

	{ SDK_WIRE_GUID },
	1,
	0,

	sizeof(wire_instance_t),
	sizeof(nvfx_shared_state_t),
	NVFX_WIRE_OUTPUT_BUFFER_SIZE,
	0,
	0,

	/* Only required calls supported */
	sizeof(nvfx_req_call_params_t),

	1,
	1,

	0,
	0,

	wire_init,
	wire_close,
	wire_call,
	wire_process
};


#if NVFX_USE_STD_INTERFACE
const invfx_t* nvfx = &invfx_wire;
#endif
const invfx_t* nvfx_wire = &invfx_wire;

CREATE_PLUGIN(nvfx_wire,
                sizeof(wire_instance_t),
                sizeof(nvfx_shared_state_t),
                NVFX_WIRE_OUTPUT_BUFFER_SIZE,
                0,
                0,
                "0.1.0")
