/*
 * Copyright (c) 2014-2015 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */


#ifndef _NVFX_AEC_H_
#define _NVFX_AEC_H_

#include "nvfx.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AEC_GUID 0xf0ae075a, 0x8db549d0, 0x90fbb2c2, 0xb671c9b5
NVID(NVID_AEC, AEC_GUID);

/* input modes */
enum
{
	nvfx_aec_input_single_interleaved = 0,
	nvfx_aec_input_multi_pin,
	nvfx_aec_input_mode_max,
};

/* aec modes*/
enum
{
	nvfx_aec_mode_bypass = 0,
	nvfx_aec_mode_active,
	nvfx_aec_mode_char,
	nvfx_aec_mode_cmp,
};

/* aec params*/
enum
{
	nvfx_aec_param_aec = 0,
	nvfx_aec_param_data_fmt,
	nvfx_aec_param_post_proc,
};

/* flags:  bit positions of control flags*/
enum
{
	denoise_disable = 0,
	residual_supression_disable,
};

/* nvfx call methods*/
enum
{
	nvfx_aec_method_set_params = nvfx_method_external_start,
	nvfx_aec_method_set_mode,
	nvfx_aec_method_set_controls,
	nvfx_aec_method_update_status,
};

/* data formats*/
typedef struct {
	uint32_t word_length;
	uint32_t sample_rate;
	uint32_t n_input_channels;
	uint32_t n_output_channels;
	uint32_t n_reference_channels;
} nvfx_aec_data_fmt_t;

/* AEC filter parameters*/
typedef struct {
	uint32_t input_mode;
	uint32_t bulk_delay;
	uint32_t tail_length;
	uint32_t block_length;
} nvfx_aec_filter_config;

/* AEC post processor parameters*/
/* supression thresholds in db*/
typedef struct {
	int32_t residual_supression_threshold_inactive;
	int32_t residual_supression_threshold_active;
} nvfx_aec_post_proc_config;

/* controls */
typedef struct {
	nvfx_call_params_t call_params;
	uint32_t flags;
	uint32_t mask;
} nvfx_aec_controls_t;

/* AEC parameters*/
typedef struct {
	uint32_t flags;
	nvfx_aec_data_fmt_t data_fmt;
	nvfx_aec_filter_config f_config;
	nvfx_aec_post_proc_config pp_config;
} nvfx_aec_params_t;

/*
* initialization parameters
*/
typedef struct {
	nvfx_call_params_t call_params;
	nvfx_aec_params_t params;
} nvfx_aec_reset_params_t;

/*
* initialization parameters
*/
typedef struct {
	nvfx_call_params_t call_params;
	uint32_t mode;
} nvfx_aec_mode_t;

/*
* set single parameter
*/
typedef struct {
	nvfx_call_params_t call_params;
	int32_t id;
	union {
		nvfx_aec_data_fmt_t data_fmt;
		nvfx_aec_filter_config f_config;
		nvfx_aec_post_proc_config pp_config;
	} param;
} nvfx_aec_param_group_t;

typedef union {
	nvfx_req_call_params_t req_call;
	nvfx_aec_controls_t controls;
	nvfx_aec_mode_t mode;
	nvfx_aec_reset_params_t reset_params;
	nvfx_aec_param_group_t param_group;
} nvfx_aec_req_call_params_t;

/*
 * Required export
 */
extern const invfx_t* nvfx_aec;

#ifdef __cplusplus
}
#endif

#endif /* #ifndef _NVFX_AEC_H_ */
