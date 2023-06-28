/*
* Copyright (c) 2014-2015 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA Corporation is strictly prohibited.
*/


#ifndef _NVFX_REVERB_H_
#define _NVFX_REVERB_H_

#include "nvfx.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SDK_REVERB_GUID 0Xfa2db1ca, 0X350041b2, 0Xa8fb887b, 0xcab448e6
	NVID(NVID_REVERB, SDK_REVERB_GUID);

/* nvfx call methods*/
enum
{
	nvfx_reverb_method_init = nvfx_method_external_start,
	nvfx_reverb_method_set_single_param,
};

/* reverb params*/
enum
{
	nvfx_reverb_param_delay = 0,
	nvfx_reverb_param_gain,
	nvfx_reverb_param_forward_gain,
	nvfx_reverb_param_word_length,
	nvfx_reverb_param_num_chans,
};

typedef struct {
	int32_t number_of_channels;
	int32_t word_length;
	int32_t delay_in_samples;
	/* gain is 16 bit q15 right justified, sign extended*/
	int32_t gain_q15;
	int32_t forward_gain_q15;
} nvfx_reverb_params_t;

/*
 * initialization parameters
 */
typedef struct {
	nvfx_call_params_t call_params;
	nvfx_reverb_params_t params;
} nvfx_reverb_init_params_t;

/*
* set single parameter
*/
typedef struct {
	nvfx_call_params_t call_params;
	int32_t id;
	int32_t value;
} nvfx_reverb_single_param_t;

typedef union {
	nvfx_req_call_params_t req_call;
	nvfx_reverb_init_params_t init_params;
	nvfx_reverb_single_param_t single_params;
} nvfx_reverb_req_call_params_t;
/*
 * Required export
 */
extern const invfx_t* nvfx_reverb;

#ifdef __cplusplus
}
#endif

#endif /* #ifndef _NVFX_REVERB_H_ */
