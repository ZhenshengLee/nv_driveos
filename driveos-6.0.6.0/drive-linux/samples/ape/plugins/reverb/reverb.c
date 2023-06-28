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

#include "reverb.h"

#define min(a, b) \
    ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a < _b ? _a : _b; })

#define NVFX_REVERB_SAMPLES_PER_BLOCK    (128)
#define NVFX_REVERB_BLOCK_SIZE           (NVFX_REVERB_SAMPLES_PER_BLOCK * 2 * 2)
#define NVFX_REVERB_OUTPUT_BUFFER_SIZE   (NVFX_REVERB_BLOCK_SIZE * 16)
#define MAX_DELAY_SAMPLES 48000
#define MAX_DELAY_LINE_SIZE (48000 * 2)

typedef struct
{
	nvfx_t fx;
	nvfx_shared_state_t cached_state;
	nvfx_buffer_t* input_buffer[1];
	nvfx_buffer_t output_buffer[1];
	nvfx_reverb_params_t params;
	int32_t delay_idx;
	int32_t delay_line[MAX_DELAY_LINE_SIZE];

	int32_t bytes_avail;
	uint8_t buffer[NVFX_REVERB_BLOCK_SIZE];
} reverb_instance_t;

static void reverb_reset(nvfx_t* fx)
{
	reverb_instance_t* reverb = (reverb_instance_t*)fx;

	/* Initialize internal buffers and parameters */
	reverb->bytes_avail = 0;
	memset(reverb->buffer, 0, sizeof(reverb->buffer));
	reverb->delay_idx = 0;
	reverb->params.delay_in_samples = 4500;
	reverb->params.gain_q15 = 0x6000;
	reverb->params.forward_gain_q15 = 0x4000;
	reverb->params.word_length = 2;
	reverb->params.number_of_channels = 2;
	memset(reverb->delay_line, 0, sizeof(reverb->delay_line));

	nvfx_reset(fx);
}

static void reverb_init(
	const invfx_t* ifx,
	nvfx_t* fx,
	nvfx_shared_state_t* shared_state,
	void* fast_shared_mem,
	void* internal_mem,
	void* req_internal_mem
	)
{
	reverb_instance_t* reverb = (reverb_instance_t*)fx;
	variant_t addr;

	nvfx_init(fx, ifx, &reverb->cached_state, shared_state,
		reverb->input_buffer, reverb->output_buffer);

	/* Our output buffer is located at start of fast, shared memory */
	addr.puint8 = (uint8_t*)fast_shared_mem;
	nvfx_buffer_init(fx->output_buffer, addr, NVFX_REVERB_OUTPUT_BUFFER_SIZE,
		NVFX_MEM_ALL_ACCESS);

	reverb_reset(fx);
}

static int32_t do_something(reverb_instance_t* reverb,
	nvfx_buffer_t* ibuffer,
	nvfx_buffer_t* obuffer,
	int32_t* bytes_consumed)
{
	int32_t bytes_to_copy;
	int32_t bytes_produced = 0;

	int32_t bytes_free = nvfx_buffer_bytes_free(obuffer);
	int32_t bytes_in = nvfx_buffer_valid_bytes(ibuffer);

	bytes_to_copy = min(NVFX_REVERB_BLOCK_SIZE - reverb->bytes_avail, bytes_in);
	if (bytes_to_copy > 0) {
		nvfx_buffer_copy_out(ibuffer, reverb->buffer + reverb->bytes_avail, bytes_to_copy);
		reverb->bytes_avail += bytes_to_copy;
	}
	*bytes_consumed = bytes_to_copy;

	/* If we have one block of data, then process it */
	/* y(n) = forward_gain * x(n) + gain * y(n -delay) */
	if (reverb->bytes_avail == NVFX_REVERB_BLOCK_SIZE &&
		bytes_free >= NVFX_REVERB_BLOCK_SIZE) {
		int32_t n_samples = NVFX_REVERB_BLOCK_SIZE / reverb->params.word_length;
		int16_t *ptr = (int16_t *)reverb->buffer;
		int16_t *optr = ptr;
		int16_t *delay_ptr = (int16_t *)reverb->delay_line;
		int32_t sample;
		int32_t delay = reverb->params.number_of_channels * reverb->params.delay_in_samples;
		while (n_samples-- > 0) {
			sample = (reverb->params.gain_q15 * (int32_t)delay_ptr[reverb->delay_idx]);
			sample += (((int32_t)(*ptr++)) * reverb->params.forward_gain_q15);
			sample = (sample + 0x00004000) >> 15;
			if (sample > 32767l)
				sample = 32767l;
			if (sample < -32768l)
				sample = -32768l;
			delay_ptr[reverb->delay_idx] = (int16_t)sample;
			reverb->delay_idx++;
			if (reverb->delay_idx >= delay) {
				reverb->delay_idx = 0;
			}
			*optr++ = sample;
		}
		reverb->bytes_avail = 0;
		bytes_produced = NVFX_REVERB_BLOCK_SIZE;
		nvfx_buffer_copy_in(obuffer, reverb->buffer, NVFX_REVERB_BLOCK_SIZE);
	}

	return bytes_produced;
}

static void reverb_process(nvfx_t* fx)
{
	reverb_instance_t* reverb = (reverb_instance_t*)fx;

	nvfx_buffer_t* ibuf = reverb->fx.input_buffer[0];
	nvfx_buffer_t* obuf = reverb->fx.output_buffer;
	nvfx_shared_state_t* cs = fx->cached_state;
	nvfx_shared_state_t* ncs = fx->noncached_state;

	int32_t bytes_produced;
	uint32_t ts_start;
	nvfx_process_start(fx, &ts_start);
	do {
		int32_t bytes_consumed = 0;
		/* Process data */
		bytes_produced = do_something(reverb,
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

static void reverb_close(nvfx_t* fx)
{
}

static void reverb_set_state(nvfx_t* fx, nvfx_set_state_params_t* set_state_params)
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

static void reverb_call(nvfx_t* fx, nvfx_call_params_t* call_params)
{
	switch (call_params->method)
	{
	case nvfx_method_set_state:
	{
		reverb_set_state(fx, (nvfx_set_state_params_t*)call_params);
	}
		break;
	case nvfx_method_reset:
	{
		reverb_reset(fx);
	}
		break;
	case nvfx_reverb_method_init:
	{
		nvfx_reverb_init_params_t* init_params =
			(nvfx_reverb_init_params_t*)call_params;

		reverb_instance_t* reverb = (reverb_instance_t*)fx;

		reverb->params.number_of_channels = 2; /* init_params->params.number_of_channels; */
		reverb->params.word_length = 2; /* init_params->params.word_length */;
		reverb->params.delay_in_samples = init_params->params.delay_in_samples;
		reverb->params.gain_q15 = init_params->params.gain_q15;
		reverb->params.forward_gain_q15 = init_params->params.forward_gain_q15;
	}
		break;
	case nvfx_reverb_method_set_single_param:
	{
		nvfx_reverb_single_param_t* param =
			(nvfx_reverb_single_param_t *)call_params;

		reverb_instance_t* reverb = (reverb_instance_t*)fx;
		switch (param->id)
		{
		case nvfx_reverb_param_delay:
			reverb->params.delay_in_samples = param->value;
			if (reverb->params.delay_in_samples > MAX_DELAY_SAMPLES) {
				reverb->params.delay_in_samples = MAX_DELAY_SAMPLES;
			}
			if (reverb->params.delay_in_samples < 0) {
				reverb->params.delay_in_samples = 0;
			}
			break;
		case nvfx_reverb_param_gain:
			reverb->params.gain_q15 = param->value;
			if (reverb->params.gain_q15 > 0x7fffl) {
				reverb->params.gain_q15 = 0x7fffl;
			}
			if (reverb->params.gain_q15 < 0) {
				reverb->params.gain_q15 = 0;
			}
			break;
		case nvfx_reverb_param_forward_gain:
			reverb->params.forward_gain_q15 = param->value;
			if (reverb->params.forward_gain_q15 > 0x7fffl) {
				reverb->params.forward_gain_q15 = 0x7fffl;
			}
			if (reverb->params.forward_gain_q15 < 0) {
				reverb->params.forward_gain_q15 = 0;
			}
			break;
		case nvfx_reverb_param_word_length:
			reverb->params.word_length = 2; /* param->value; */
			break;
		case nvfx_reverb_param_num_chans:
			reverb->params.number_of_channels = 2; /* param->value */;
			break;
		default:
			break;
		}
	}
	default:
		break;
	}
}

static const invfx_t invfx_reverb =
{
	sizeof(invfx_t),

	{ SDK_REVERB_GUID },
	1,
	0,

	sizeof(reverb_instance_t),
	sizeof(nvfx_shared_state_t),
	NVFX_REVERB_OUTPUT_BUFFER_SIZE,
	0,
	0,

	/* Only required calls supported */
	sizeof(nvfx_reverb_req_call_params_t),

	1,
	1,

	0,
	0,

	reverb_init,
	reverb_close,
	reverb_call,
	reverb_process
};


#if NVFX_USE_STD_INTERFACE
const invfx_t* nvfx = &invfx_reverb;
#endif
const invfx_t* nvfx_reverb = &invfx_reverb;

CREATE_PLUGIN(nvfx_reverb,
                sizeof(reverb_instance_t),
                sizeof(nvfx_shared_state_t),
                NVFX_REVERB_OUTPUT_BUFFER_SIZE,
                0,
                0,
                "0.1.0")
