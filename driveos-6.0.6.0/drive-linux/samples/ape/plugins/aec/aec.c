/*
 * Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifdef HAVE_CONFIG_H
#include "speex_config.h"
#endif

#include <stdbool.h>
#include "plugin.h"

#include "os_support.h"
#include "speex/speex_echo.h"
#include "speex/speex_preprocess.h"
#include "nvfx_aec.h"

#define DEBUG_LEVEL 2
#define DEBUG_PRINT(level, str, x...) do { if ((level) >= DEBUG_LEVEL) { printf(str, ## x); } } while (0)


#define min(a, b) \
	({ __typeof__ (a) _a = (a); \
	   __typeof__ (b) _b = (b); \
	   _a < _b ? _a : _b; })

/* set limits on AEC config */

#define NVFX_AEC_MAX_WORD_LENGH           (2)
#define NVFX_AEC_MAX_IN_CHANNELS          (2)
#define NVFX_AEC_MAX_REF_CHANNELS         (1)
#define NVFX_AEC_MAX_OUT_CHANNELS         (2)

#define NVFX_AEC_MAX_SAMPLES_PER_BLOCK    (4*128)
#define NVFX_AEX_MAX_TAIL_LENGTH          (1024)
#define NVFX_AEC_MAX_SAMPLE_RATE          (3*8000)
#define NVFX_AEC_MAX_BULK_DELAY           (1024)
#define NVFX_AEC_MAX_HEAP_SIZE            (256 * 1024)
#define NVFX_AEC_MAX_INPUT_PINS           (2)

/* set default values */
#define NVFX_AEC_DEFAULT_SAMPLE_RATE      (2*8000)
#define NVFX_AEC_DEFAULT_BLOCK_LENGTH     (128)
#define NVFX_AEC_DEFAULT_TAIL_LENGTH      (512)
#define NVFX_AEC_DEFAULT_IN_CHANNELS      (2)
#define NVFX_AEC_DEFAULT_REF_CHANNELS     (1)
#define NVFX_AEC_DEFAULT_OUT_CHANNELS     (2)
#define NVFX_AEC_DEFAULT_ERR_CHANNELS     (1)
#define NVFX_AEC_DEFAULT_WORD_LENGTH      (2)
#define NVFX_AEC_DEFAULT_BULK_DELAY       (0)
#define NVFX_AEC_DEFAULT_INPUT_MODE       (nvfx_aec_input_single_interleaved)
#define NVFX_AEC_DEFAULT_PROCESSING_MODE  (nvfx_aec_mode_active)
#define NVFX_AEC_DEFAULT_ACTIVE_SUPRESS   (-15)
#define NVFX_AEC_DEFAULT_INACTIVE_SUPRESS (-40)


#define NVFX_AEC_MAX_IN_BLOCK_SIZE        (NVFX_AEC_MAX_SAMPLES_PER_BLOCK * \
						(NVFX_AEC_MAX_IN_CHANNELS + \
						NVFX_AEC_MAX_REF_CHANNELS ) * \
						NVFX_AEC_MAX_WORD_LENGH)
#define NVFX_AEC_MAX_OUT_BLOCK_SIZE       (NVFX_AEC_MAX_SAMPLES_PER_BLOCK * \
						NVFX_AEC_MAX_OUT_CHANNELS * \
						NVFX_AEC_MAX_WORD_LENGH)
#define NVFX_AEC_MAX_OUTPUT_BUFFER_SIZE   (NVFX_AEC_MAX_OUT_BLOCK_SIZE * 16)
#define NVFX_AEC_MAX_BULK_DELAY_BUFF_LEN  (NVFX_AEC_MAX_BULK_DELAY * NVFX_AEC_MAX_REF_CHANNELS)
#define NVFX_AEC_MAX_TEMP_BUF_SIZE        (NVFX_AEC_MAX_IN_BLOCK_SIZE)

typedef struct {
	nvfx_shared_state_t noncached_state;
	int32_t status_updated;
} aec_noncached_state_t;

typedef struct
{
	nvfx_t fx;
	nvfx_shared_state_t cached_state;
	nvfx_buffer_t *input_buffer[NVFX_AEC_MAX_INPUT_PINS];
	nvfx_buffer_t output_buffer[1];

	int32_t block_length_bytes_ec;
	int32_t block_length_bytes_ref;
	int32_t block_length_bytes_out;
	int32_t block_length_bytes_in[NVFX_AEC_MAX_INPUT_PINS];
	uint32_t ref_delay_idx;
	bool output_available;
	bool input_available;
	void *current_ptr;
	int remaining_mem;
	SpeexEchoState *st;
	SpeexPreprocessState *den;
	uint32_t mode;
	nvfx_aec_params_t aec_params;
	uint8_t aec_heap[NVFX_AEC_MAX_HEAP_SIZE + 7];
	int32_t bytes_available[NVFX_AEC_MAX_INPUT_PINS];
	spx_int16_t echo_buf[NVFX_AEC_MAX_SAMPLES_PER_BLOCK * NVFX_AEC_MAX_IN_CHANNELS];
	spx_int16_t ref_buf[NVFX_AEC_MAX_SAMPLES_PER_BLOCK * NVFX_AEC_MAX_REF_CHANNELS];
	spx_int16_t e_buf[NVFX_AEC_MAX_SAMPLES_PER_BLOCK * NVFX_AEC_MAX_OUT_CHANNELS];
	spx_int16_t ref_delay_line[NVFX_AEC_MAX_BULK_DELAY_BUFF_LEN];
	spx_int16_t scratch_buf[(NVFX_AEC_MAX_SAMPLES_PER_BLOCK *
		(NVFX_AEC_MAX_IN_CHANNELS + NVFX_AEC_MAX_OUT_CHANNELS))];
	uint32_t buffer[NVFX_AEC_MAX_INPUT_PINS][NVFX_AEC_MAX_TEMP_BUF_SIZE / sizeof(uint32_t)];
} aec_instance_t;

aec_instance_t aec_instance NVFX_MEM_INSTANCE;
aec_noncached_state_t aec_noncached_state NVFX_MEM_SHARED;
uint8_t aec_output_buffer[NVFX_AEC_MAX_OUTPUT_BUFFER_SIZE] NVFX_MEM_FAST_SHARED;

static void aec_initialize(nvfx_t *fx, nvfx_aec_reset_params_t *reset_params, bool apply_params);

void *libnvaecfx_alloc(int size)
{
	aec_instance_t *aec = (aec_instance_t *)nvfx_get_instance();
	void *rptr = aec->current_ptr;


	if (aec->remaining_mem < size) {
		DEBUG_PRINT(5, "libnvaecfx_alloc: out of memory\n");
		return NULL;
	}
	aec->current_ptr = (uint8_t *)aec->current_ptr + (((size + 7) >> 3) << 3);
	aec->remaining_mem -= ((size + 7) >> 3) << 3;
	DEBUG_PRINT(0, "libnvaecfx_alloc: ptr = %x, size = %d, remaining memory = %d\n", (uint32_t)rptr, size, aec->remaining_mem);
	return rptr;
}

void libnvaecfx_free(void *ptr)
{
}

static void aec_reset(nvfx_t *fx)
{
	aec_instance_t *aec = (aec_instance_t *)fx;
	aec_noncached_state_t *aec_state = (aec_noncached_state_t *)fx->noncached_state;

	/* set default parameters */
	aec->aec_params.flags = 0;
	aec->aec_params.data_fmt.sample_rate = NVFX_AEC_DEFAULT_SAMPLE_RATE;
	aec->aec_params.data_fmt.word_length = NVFX_AEC_DEFAULT_WORD_LENGTH;
	aec->aec_params.data_fmt.n_input_channels = NVFX_AEC_DEFAULT_IN_CHANNELS;
	aec->aec_params.data_fmt.n_output_channels = NVFX_AEC_DEFAULT_OUT_CHANNELS;
	aec->aec_params.data_fmt.n_reference_channels = NVFX_AEC_DEFAULT_REF_CHANNELS;

	aec->aec_params.f_config.bulk_delay = (NVFX_AEC_DEFAULT_BULK_DELAY * NVFX_AEC_DEFAULT_REF_CHANNELS);
	aec->aec_params.f_config.input_mode = NVFX_AEC_DEFAULT_INPUT_MODE;
	aec->aec_params.f_config.block_length = NVFX_AEC_DEFAULT_BLOCK_LENGTH;
	aec->aec_params.f_config.tail_length = NVFX_AEC_DEFAULT_TAIL_LENGTH;

	aec->aec_params.pp_config.residual_supression_threshold_inactive =
		NVFX_AEC_DEFAULT_INACTIVE_SUPRESS;
	aec->aec_params.pp_config.residual_supression_threshold_active =
		NVFX_AEC_DEFAULT_ACTIVE_SUPRESS;


	aec->mode = NVFX_AEC_DEFAULT_PROCESSING_MODE;

	aec->current_ptr = (void *)((((int)aec->aec_heap + 7) >> 3) << 3);
	aec->remaining_mem = (sizeof(aec->aec_heap) >> 3) << 3;

	/* Initialize internal buffers */
	aec->bytes_available[0] = 0;
	aec->bytes_available[1] = 0;
	aec->output_available = false;
	aec->input_available = false;
	aec->ref_delay_idx = 0;
	memset(aec->echo_buf, 0, sizeof(aec->echo_buf));
	memset(aec->ref_buf, 0, sizeof(aec->echo_buf));
	memset(aec->e_buf, 0, sizeof(aec->echo_buf));
	memset(aec->aec_heap, 0, sizeof(aec->aec_heap));
	memset(aec->ref_delay_line, 0, sizeof(aec->ref_delay_line));

	/* Initialize shared state */
	aec_state->status_updated = 0;

}

static void aec_init(const invfx_t *ifx,
	nvfx_t *fx,
	nvfx_shared_state_t *shared_state,
	void *fast_shared_mem,
	void *internal_mem,
	void *req_internal_mem)
{
	aec_instance_t *aec = (aec_instance_t *)fx;
	variant_t addr;

	nvfx_set_instance(aec);

	nvfx_init(fx, ifx, &aec->cached_state, shared_state,
		aec->input_buffer, aec->output_buffer);

	/* Our output buffer is located at start of fast, shared memory */
	addr.puint8 = (uint8_t *)fast_shared_mem;
	nvfx_buffer_init(fx->output_buffer, addr, NVFX_AEC_MAX_OUTPUT_BUFFER_SIZE,
		NVFX_MEM_ALL_ACCESS);
	aec_reset(fx);
	aec_initialize(fx, NULL, false);
	DEBUG_PRINT(3, "AEC_INIT\n");
}

static void stereo_to_mono(int16_t* in, int16_t* out, int16_t samples)
{
	int16_t j;
	int16_t *left = in;
	int16_t *right = in + 1;

	for (j=0; j<samples; j++) {
		out[j] = ((int32_t)*left + (int32_t)*right)/2;
		left += 2;
		right += 2;
	}
}

static void mono_to_stereo(int16_t* in, int16_t samples)
{
	int i;
	int16_t *dst = in + samples * 2 - 1;
	int16_t *src = in + samples - 1;
	for (i = 0; i < samples; i++) {
		*dst-- = *src; *dst-- = *src; src--;
	}
}

static void deinterleave_input(aec_instance_t *aec)
{
	spx_int16_t *echo_buf = aec->echo_buf;
	spx_int16_t *ref_buf = aec->ref_buf;
	int length = aec->aec_params.f_config.block_length;
	nvfx_aec_data_fmt_t *fmt = &aec->aec_params.data_fmt;
	spx_int16_t *iptr = (spx_int16_t *)&aec->buffer[0][0];

	while (length-- > 0) {
		for (unsigned int e_len = 0; e_len < fmt->n_input_channels; e_len++) {
			*echo_buf++ = *iptr++;
		}
		for (unsigned int r_len = 0; r_len < fmt->n_reference_channels; r_len++) {
			aec->ref_delay_line[aec->ref_delay_idx] = *iptr++;
			aec->ref_delay_idx++;
			if (aec->aec_params.f_config.bulk_delay <= aec->ref_delay_idx) {
				aec->ref_delay_idx = 0;
			}
			*ref_buf++ = aec->ref_delay_line[aec->ref_delay_idx];
		}
	}
}

static void interleave_samples(int16_t *obuf,
	int16_t *buf1,
	int16_t *buf2,
	int16_t *buf3,
	int32_t nchan1,
	int32_t nchan2,
	int32_t nchan3,
	int32_t length)
{
	while (length-- > 0) {
		for (int e_len = 0; e_len < nchan1; e_len++) {
			*obuf++ = *buf1++;
		}
		for (int e_len = 0; e_len < nchan2; e_len++) {
			*obuf++ = *buf2++;
		}
		if (buf3) {
			for (int e_len = 0; e_len < nchan3; e_len++) {
				*obuf++ = *buf3++;
			}
		}
	}
}

static void delay_ref(aec_instance_t *aec)
{
	spx_int16_t *ref_buf = aec->ref_buf;
	int length = aec->aec_params.f_config.block_length;
	nvfx_aec_data_fmt_t *fmt = &aec->aec_params.data_fmt;
	spx_int16_t *iptr = (spx_int16_t *)&aec->buffer[1][0];;

	while (length-- > 0) {
		for (unsigned int r_len = 0; r_len < fmt->n_reference_channels; r_len++) {
			aec->ref_delay_line[aec->ref_delay_idx] = *iptr++;
			aec->ref_delay_idx++;
			if (aec->aec_params.f_config.bulk_delay <= aec->ref_delay_idx) {
				aec->ref_delay_idx = 0;
			}
			*ref_buf++ = aec->ref_delay_line[aec->ref_delay_idx];
		}
	}
}

static void aec_update_block_length_info(aec_instance_t *aec)
{
	aec->block_length_bytes_ec = aec->aec_params.f_config.block_length *
		aec->aec_params.data_fmt.word_length *
		aec->aec_params.data_fmt.n_input_channels;

	DEBUG_PRINT(0, "aec_update_block_length_info: block_length_bytes_ec %d\n", aec->block_length_bytes_ec);

	aec->block_length_bytes_ref = aec->aec_params.f_config.block_length *
		aec->aec_params.data_fmt.word_length *
		aec->aec_params.data_fmt.n_reference_channels;

	DEBUG_PRINT(0, "aec_update_block_length_info: block_length_bytes_ref %d\n", aec->block_length_bytes_ref);

	if (aec->aec_params.f_config.input_mode == nvfx_aec_input_single_interleaved) {
		aec->block_length_bytes_in[0] = aec->block_length_bytes_ec + aec->block_length_bytes_ref;
		aec->block_length_bytes_in[1] = 0;
	}
	else {
		aec->block_length_bytes_in[0] = aec->block_length_bytes_ec;
		aec->block_length_bytes_in[1] = aec->block_length_bytes_ref;
	}

	switch (aec->mode) {
	case nvfx_aec_mode_char:
		aec->block_length_bytes_out = aec->block_length_bytes_in[0];
		if (aec->aec_params.f_config.input_mode == nvfx_aec_input_multi_pin) {
			aec->block_length_bytes_out += aec->block_length_bytes_in[1];
		}
		break;
	case nvfx_aec_mode_cmp:
		aec->block_length_bytes_out = aec->block_length_bytes_ref;
		aec->block_length_bytes_out += aec->block_length_bytes_ec;
		aec->block_length_bytes_out += aec->aec_params.f_config.block_length *
			aec->aec_params.data_fmt.word_length *
			aec->aec_params.data_fmt.n_output_channels;
		break;
	default:
		aec->block_length_bytes_out = aec->aec_params.f_config.block_length *
			aec->aec_params.data_fmt.word_length *
			aec->aec_params.data_fmt.n_output_channels;
		break;
	}
	DEBUG_PRINT(0, "aec_update_block_length_info: block_length_bytes_in[0]  %d\n", aec->block_length_bytes_in[0]);
	DEBUG_PRINT(0, "aec_update_block_length_info: block_length_bytes_in[1]  %d\n", aec->block_length_bytes_in[1]);
	DEBUG_PRINT(0, "aec_update_block_length_info: block_length_bytes_out  %d\n", aec->block_length_bytes_out);
}

static void aec_get_input(aec_instance_t *aec, nvfx_shared_state_t *cs)
{
	int32_t bytes_to_copy;


	bytes_to_copy = min(aec->block_length_bytes_in[0] - aec->bytes_available[0],
		nvfx_buffer_valid_bytes(aec->fx.input_buffer[0]));

	DEBUG_PRINT(0, "aec_get_input: memcpy %d\n", bytes_to_copy);

	nvfx_buffer_copy_out(aec->fx.input_buffer[0],
		&aec->buffer[0][0] + aec->bytes_available[0],
		bytes_to_copy);
	aec->bytes_available[0] += bytes_to_copy;

	DEBUG_PRINT(0, "aec_get_input: bytes_available[0] %d\n", aec->bytes_available[0]);

	nvfx_buffer_consume_bytes(aec->fx.input_buffer[0], bytes_to_copy);
	cs->input[0].bytes += bytes_to_copy;


	if (aec->block_length_bytes_in[1] > 0) {
		bytes_to_copy = min(aec->block_length_bytes_in[1] - aec->bytes_available[1],
			nvfx_buffer_valid_bytes(aec->fx.input_buffer[1]));
		DEBUG_PRINT(0, "aec_get_input: memcpy %d\n", bytes_to_copy);
		nvfx_buffer_copy_out(aec->fx.input_buffer[1],
			&aec->buffer[1][0] + aec->bytes_available[0],
			bytes_to_copy);
		aec->bytes_available[1] += bytes_to_copy;
		DEBUG_PRINT(0, "aec_get_input: bytes_available[1] %d\n", aec->bytes_available[1]);
		nvfx_buffer_consume_bytes(aec->fx.input_buffer[1], bytes_to_copy);
		cs->input[1].bytes += bytes_to_copy;
	}
	if ((aec->bytes_available[0] >= aec->block_length_bytes_in[0]) &&
		(aec->bytes_available[1] >= aec->block_length_bytes_in[1]) &&
		(nvfx_buffer_bytes_free(aec->fx.output_buffer) >=
		aec->block_length_bytes_out)) {

		if (aec->aec_params.f_config.input_mode == nvfx_aec_input_single_interleaved) {
			DEBUG_PRINT(0, "aec_get_input: deinterleave_input\n");;
			deinterleave_input(aec);
		}
		else {
			memcpy(aec->echo_buf, &aec->buffer[0][0], aec->block_length_bytes_in[0]);
			delay_ref(aec);
		}
		aec->bytes_available[0] = 0;
		aec->bytes_available[1] = 0;
		aec->input_available = true;
	}
	else {
		DEBUG_PRINT(0, "aec_get_input: %d %d %d\n",
			aec->block_length_bytes_in[0], aec->block_length_bytes_in[1],
			nvfx_buffer_bytes_free(aec->fx.output_buffer));
	}
}

static void aec_process_input(aec_instance_t *aec)
{
	DEBUG_PRINT(0, "aec_process_input\n");

	if (aec->input_available){
		if (((nvfx_aec_mode_active == aec->mode) ||
			(nvfx_aec_mode_bypass == aec->mode) ||
			(nvfx_aec_mode_cmp == aec->mode)) &&
			(nvfx_state_active == ((nvfx_t *)aec)->cached_state->process.state)) {
			stereo_to_mono(aec->echo_buf, aec->scratch_buf, aec->aec_params.f_config.block_length);
			DEBUG_PRINT(0, "aec_process_input: speex_echo_cancellation\n");
			speex_echo_cancellation(aec->st, aec->scratch_buf, aec->ref_buf, aec->e_buf);
			DEBUG_PRINT(0, "aec_process_input: speex_preprocess_run\n");
			speex_preprocess_run(aec->den, aec->e_buf);
			mono_to_stereo(aec->e_buf, aec->aec_params.f_config.block_length);
		}
		else  if (nvfx_aec_mode_char == aec->mode) {
			DEBUG_PRINT(0, "aec_process_input: copy in to out\n");
		}
		aec->input_available = false;
		aec->output_available = true;
	}
	DEBUG_PRINT(0, "aec_process_input: done\n");
}

static int32_t aec_put_output(aec_instance_t *aec)
{
	int32_t bytes_produced = 0;
	DEBUG_PRINT(0, "aec_put_output: \n");
	if (aec->output_available) {
		nvfx_buffer_t *obuf = aec->fx.output_buffer;
		switch (aec->mode) {
		case nvfx_aec_mode_char:
			DEBUG_PRINT(0, "aec_put_output: nvfx_aec_mode_char\n");
			if (aec->aec_params.f_config.input_mode == nvfx_aec_input_single_interleaved) {
				nvfx_buffer_copy_in(obuf, &aec->buffer[0][0], aec->block_length_bytes_out);
			}
			else {
				interleave_samples(aec->scratch_buf,
					(spx_int16_t *)&aec->buffer[0][0],
					(spx_int16_t *)&aec->buffer[1][0],
					NULL,
					aec->aec_params.data_fmt.n_input_channels,
					aec->aec_params.data_fmt.n_reference_channels,
					0,
					aec->aec_params.f_config.block_length);
				nvfx_buffer_copy_in(obuf, aec->scratch_buf, aec->block_length_bytes_out);
			}
			break;
		case nvfx_aec_mode_active:
			nvfx_buffer_copy_in(obuf, aec->e_buf, aec->block_length_bytes_out);
			break;
		case nvfx_aec_mode_bypass:
			nvfx_buffer_copy_in(obuf, aec->echo_buf, aec->block_length_bytes_out);
			break;
		case nvfx_aec_mode_cmp:
			interleave_samples(aec->scratch_buf,
				aec->echo_buf,
				aec->ref_buf,
				aec->e_buf,
				aec->aec_params.data_fmt.n_input_channels,
				aec->aec_params.data_fmt.n_reference_channels,
				aec->aec_params.data_fmt.n_output_channels,
				aec->aec_params.f_config.block_length);
			nvfx_buffer_copy_in(obuf, aec->scratch_buf, aec->block_length_bytes_out);
			break;
		default:
			break;
		}
		aec->output_available = false;
		bytes_produced = aec->block_length_bytes_out;
		nvfx_buffer_add_bytes(obuf, bytes_produced);
	}
	DEBUG_PRINT(0, "aec_put_output: done\n");

	return bytes_produced;
}

static void aec_process(nvfx_t *fx)
{
	aec_instance_t *aec = (aec_instance_t *)fx;
	uint32_t ts_start;
	int32_t  bytes_produced = 0;

	nvfx_set_instance(aec);

	DEBUG_PRINT(1, "AEC_PROCESS\n");

	nvfx_process_start(fx, &ts_start);
	aec_update_block_length_info(aec);
	aec_get_input(aec, fx->cached_state);
	aec_process_input(aec);
	bytes_produced = aec_put_output(aec);
	if (bytes_produced) {
		fx->cached_state->output[0].frames++;
		fx->cached_state->output[0].bytes += bytes_produced;
		nvfx_process(fx, ts_start, 0);
	}
	fx->noncached_state->input[0] = fx->cached_state->input[0];
	fx->noncached_state->input[1] = fx->cached_state->input[1];
	fx->noncached_state->output[0] = fx->cached_state->output[0];

	DEBUG_PRINT(1, "AEC_PROCESS: done\n");
}

static void aec_close(nvfx_t *fx)
{
	aec_instance_t *aec = (aec_instance_t *)fx;
	nvfx_set_instance(aec);

	DEBUG_PRINT(1, "AEC_CLOSE\n");
	speex_echo_state_destroy(aec->st);
	speex_preprocess_state_destroy(aec->den);
}

static void aec_set_state(nvfx_t *fx, nvfx_set_state_params_t *set_state_params)
{
	int32_t state = set_state_params->state;

	switch (state)
	{
	case nvfx_state_inactive:
	case nvfx_state_active:
	default:
		break;
	}
	fx->noncached_state->process.state = fx->cached_state->process.state = state;
	DEBUG_PRINT(2, "AEC: aec_set_state exit\n");
}

static void aec_verify_parameters(nvfx_aec_params_t *params)
{
	if (8000 > params->data_fmt.sample_rate) {
		params->data_fmt.sample_rate = 8000;
	}
	if (NVFX_AEC_MAX_WORD_LENGH < params->data_fmt.word_length) {
		params->data_fmt.word_length = NVFX_AEC_MAX_WORD_LENGH;
	}
	if (2 > params->data_fmt.word_length) {
		params->data_fmt.word_length = 2;
	}
	if (NVFX_AEC_MAX_IN_CHANNELS < params->data_fmt.n_input_channels) {
		params->data_fmt.n_input_channels = NVFX_AEC_MAX_IN_CHANNELS;
	}
	if (0 >= params->data_fmt.n_input_channels) {
		params->data_fmt.n_input_channels = 1;
	}
	if (NVFX_AEC_MAX_OUT_CHANNELS < params->data_fmt.n_output_channels) {
		params->data_fmt.n_output_channels = NVFX_AEC_MAX_OUT_CHANNELS;
	}
	if (0 >= params->data_fmt.n_output_channels) {
		params->data_fmt.n_output_channels = 1;
	}
	if (NVFX_AEC_MAX_REF_CHANNELS < params->data_fmt.n_reference_channels) {
		params->data_fmt.n_reference_channels = NVFX_AEC_MAX_REF_CHANNELS;
	}
	if (0 >= params->data_fmt.n_reference_channels) {
		params->data_fmt.n_reference_channels = 1;
	}
	if (NVFX_AEC_MAX_BULK_DELAY_BUFF_LEN < params->f_config.bulk_delay) {
		params->f_config.bulk_delay = NVFX_AEC_MAX_BULK_DELAY_BUFF_LEN;
	}
	if (nvfx_aec_input_mode_max <= params->f_config.input_mode) {
		params->f_config.input_mode = nvfx_aec_input_single_interleaved;
	}
	if (NVFX_AEC_MAX_INPUT_PINS == 1) {
		params->f_config.input_mode = nvfx_aec_input_single_interleaved;
	}
	if (NVFX_AEC_MAX_SAMPLES_PER_BLOCK < params->f_config.block_length) {
		params->f_config.block_length = NVFX_AEC_MAX_SAMPLES_PER_BLOCK;
	}
	if (NVFX_AEX_MAX_TAIL_LENGTH < params->f_config.tail_length) {
		params->f_config.tail_length = NVFX_AEX_MAX_TAIL_LENGTH;
	}
	if (0 >= params->f_config.tail_length) {
		params->f_config.tail_length = NVFX_AEX_MAX_TAIL_LENGTH;
	}
}

static void aec_initialize(nvfx_t *fx, nvfx_aec_reset_params_t *reset_params, bool apply_params)
{
	aec_instance_t *aec = (aec_instance_t *)fx;
	spx_int32_t tmp;
	nvfx_set_instance(aec);

	DEBUG_PRINT(2, "AEC_INITIALIZE\n");

	if (apply_params) {
		aec->aec_params.data_fmt.sample_rate = reset_params->params.data_fmt.sample_rate;
		aec->aec_params.data_fmt.word_length = reset_params->params.data_fmt.word_length;
		aec->aec_params.data_fmt.n_input_channels = reset_params->params.data_fmt.n_input_channels;
		aec->aec_params.data_fmt.n_output_channels = reset_params->params.data_fmt.n_output_channels;
		aec->aec_params.data_fmt.n_reference_channels = reset_params->params.data_fmt.n_reference_channels;
		aec->aec_params.f_config.bulk_delay = reset_params->params.f_config.bulk_delay * reset_params->params.data_fmt.n_reference_channels;
		aec->aec_params.f_config.input_mode = reset_params->params.f_config.input_mode;
		aec->aec_params.f_config.block_length = reset_params->params.f_config.block_length;
		aec->aec_params.f_config.tail_length = reset_params->params.f_config.tail_length;
		aec->aec_params.pp_config.residual_supression_threshold_inactive =
			reset_params->params.pp_config.residual_supression_threshold_inactive;
		aec->aec_params.pp_config.residual_supression_threshold_active =
			reset_params->params.pp_config.residual_supression_threshold_active;
	}
	aec_verify_parameters(&aec->aec_params);
	DEBUG_PRINT(1, "speex_echo_state_init\n");
	aec->st = speex_echo_state_init(aec->aec_params.f_config.block_length, aec->aec_params.f_config.tail_length);
	DEBUG_PRINT(1, "speex_preprocess_state_init\n");
	aec->den = speex_preprocess_state_init(aec->aec_params.f_config.block_length, aec->aec_params.data_fmt.sample_rate);
	DEBUG_PRINT(1, "speex_echo_ctl\n");
	speex_echo_ctl(aec->st, SPEEX_ECHO_SET_SAMPLING_RATE, &aec->aec_params.data_fmt.sample_rate);
	DEBUG_PRINT(1, "speex_preprocess_ctl\n");
	speex_preprocess_ctl(aec->den, SPEEX_PREPROCESS_SET_ECHO_STATE, aec->st);
	if (aec->aec_params.flags & denoise_disable) {
		tmp = 2;
		speex_preprocess_ctl(aec->den, SPEEX_PREPROCESS_SET_DENOISE, &tmp);
	}
	if (aec->aec_params.flags & residual_supression_disable) {
		tmp = 0;
		speex_preprocess_ctl(aec->den, SPEEX_PREPROCESS_SET_ECHO_SUPPRESS, &tmp);
		speex_preprocess_ctl(aec->den, SPEEX_PREPROCESS_SET_ECHO_SUPPRESS_ACTIVE, &tmp);
	}
	DEBUG_PRINT(2, "AEC_INITIALIZE OUT\n");
}

static void aec_set_param(nvfx_t *fx, nvfx_aec_param_group_t *set_params)
{
	aec_instance_t *aec = (aec_instance_t *)fx;

	switch (set_params->id) {
	case nvfx_aec_param_aec:
		aec->aec_params.f_config.bulk_delay = set_params->param.f_config.bulk_delay * set_params->param.data_fmt.n_reference_channels;
		aec->aec_params.f_config.input_mode = set_params->param.f_config.input_mode;
		aec->aec_params.f_config.block_length = set_params->param.f_config.block_length;
		aec->aec_params.f_config.tail_length = set_params->param.f_config.tail_length;
		break;
	case nvfx_aec_param_data_fmt:
		aec->aec_params.data_fmt.sample_rate = set_params->param.data_fmt.sample_rate;
		aec->aec_params.data_fmt.word_length = set_params->param.data_fmt.word_length;
		aec->aec_params.data_fmt.n_input_channels = set_params->param.data_fmt.n_input_channels;
		aec->aec_params.data_fmt.n_output_channels = set_params->param.data_fmt.n_output_channels;
		aec->aec_params.data_fmt.n_reference_channels = set_params->param.data_fmt.n_reference_channels;
		break;
	case nvfx_aec_param_post_proc:
		aec->aec_params.pp_config.residual_supression_threshold_inactive =
			set_params->param.pp_config.residual_supression_threshold_inactive;
		aec->aec_params.pp_config.residual_supression_threshold_active =
			set_params->param.pp_config.residual_supression_threshold_active;
		break;
	default:
		break;
	}
	aec_verify_parameters(&aec->aec_params);
}

static void aec_set_mode(nvfx_t *fx, nvfx_aec_mode_t *mode_params)
{
	aec_instance_t *aec = (aec_instance_t *)fx;

	switch (mode_params->mode) {
	case nvfx_aec_mode_bypass:
	case nvfx_aec_mode_active:
	case nvfx_aec_mode_char:
	case nvfx_aec_mode_cmp:
		DEBUG_PRINT(3, "AEC: set mode: %d\n", mode_params->mode);
		aec->mode = mode_params->mode;
		break;
	default:
		DEBUG_PRINT(3, "AEC: set mode: unknown\n");
		break;
	}
}

static void aec_update_status(nvfx_t *fx)
{
	aec_noncached_state_t *aec_state = (aec_noncached_state_t *)fx->noncached_state;

	aec_state->status_updated++;;

}

static void aec_set_controls(nvfx_t *fx, nvfx_aec_controls_t *control_params)
{
	aec_instance_t *aec = (aec_instance_t *)fx;
	uint32_t tmp = control_params->mask & control_params->flags;
	aec->aec_params.flags &= (~control_params->mask | tmp);

	if (control_params->mask & denoise_disable) {
		if (aec->aec_params.flags & denoise_disable) {
			tmp = 2;
		}
		else {
			tmp = 1;
		}
		speex_preprocess_ctl(aec->den, SPEEX_PREPROCESS_SET_DENOISE, &tmp);
	}
	if (control_params->mask & residual_supression_disable) {
		if (aec->aec_params.flags & residual_supression_disable) {
			tmp = 0;
			speex_preprocess_ctl(aec->den, SPEEX_PREPROCESS_SET_ECHO_SUPPRESS, &tmp);
			speex_preprocess_ctl(aec->den, SPEEX_PREPROCESS_SET_ECHO_SUPPRESS_ACTIVE, &tmp);
		}
		else {
			tmp = aec->aec_params.pp_config.residual_supression_threshold_inactive;
			speex_preprocess_ctl(aec->den, SPEEX_PREPROCESS_SET_ECHO_SUPPRESS, &tmp);
			tmp = aec->aec_params.pp_config.residual_supression_threshold_active;
			speex_preprocess_ctl(aec->den, SPEEX_PREPROCESS_SET_ECHO_SUPPRESS_ACTIVE, &tmp);
		}
	}
}

static void aec_call(nvfx_t *fx, nvfx_call_params_t *call_params)
{

	aec_instance_t *aec = (aec_instance_t *)fx;
	nvfx_set_instance(aec);

	switch (call_params->method)
	{
	case nvfx_method_set_state:
		DEBUG_PRINT(3, "AEC: set state\n");
		aec_set_state(fx, (nvfx_set_state_params_t *)call_params);
		break;
	case nvfx_method_flush:
		DEBUG_PRINT(3, "AEC: flush\n");
		nvfx_reset(fx);
		break;
	case nvfx_method_reset:
	{
		bool apply_params = call_params->size == sizeof(nvfx_aec_reset_params_t) ? true : false;
		DEBUG_PRINT(3, "AEC: reset\n");
		nvfx_reset(fx);
		aec_reset(fx);
		aec_initialize(fx, (nvfx_aec_reset_params_t *)call_params, apply_params);
	}
		break;
	case nvfx_aec_method_set_params:
		DEBUG_PRINT(3, "AEC: set param\n");
		aec_set_param(fx, (nvfx_aec_param_group_t *)call_params);
		break;
	case nvfx_aec_method_set_mode:
		DEBUG_PRINT(3, "AEC: set mode method\n");
		aec_set_mode(fx, (nvfx_aec_mode_t *)call_params);
		break;
	case nvfx_aec_method_set_controls:
		DEBUG_PRINT(3, "AEC: set controls method\n");
		aec_set_controls(fx, (nvfx_aec_controls_t *)call_params);
		break;
	case nvfx_aec_method_update_status:
		DEBUG_PRINT(3, "AEC: update status method\n");
		aec_update_status(fx);
		break;
	default:
		DEBUG_PRINT(3, "AEC: unknown call %d\n", call_params->method);
		break;
	}
}

static const invfx_t invfx_aec =
{
	sizeof(invfx_t),

	{ AEC_GUID },
	1,
	0,

	sizeof(aec_instance),
	sizeof(aec_noncached_state_t),
	sizeof(aec_output_buffer),
	0,
	0,

	sizeof(nvfx_aec_req_call_params_t),

	NVFX_AEC_MAX_INPUT_PINS,
	1,

	0,
	0,

	aec_init,
	aec_close,
	aec_call,
	aec_process
};

#if NVFX_USE_STD_INTERFACE
const invfx_t *nvfx = &invfx_aec;
#endif
const invfx_t *nvfx_aec = &invfx_aec;

CREATE_PLUGIN(nvfx_aec,
                sizeof(aec_instance),
                sizeof(aec_noncached_state_t),
                sizeof(aec_output_buffer),
                0,
                0,
                "0.1.0")
