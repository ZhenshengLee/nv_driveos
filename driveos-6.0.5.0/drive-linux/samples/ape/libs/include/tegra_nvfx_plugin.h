/*
 * tegra_nvfx_plugin.h - Shared Plugin interface between Tegra host driver and
 *			 ADSP user space code.
 *
 * Copyright (c) 2014, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef _TEGRA_NVFX_PLUGIN_H_
#define _TEGRA_NVFX_PLUGIN_H_

#include <list.h>

/**
 * Plugin state structure shared between ADSP & CPU
 */
typedef struct {
	variant_t plugin;
	/* NVFX specific shared memory follows */
} plugin_shared_mem_t;

typedef struct {
	struct list_node node;
} plugin_t;

#define PLUGIN_SHARED_MEM(x)	((plugin_shared_mem_t*)x)
#define NVFX_SHARED_STATE(x)	(nvfx_shared_state_t*)(PLUGIN_SHARED_MEM(x) + 1)
#define FX_FROM_PLUGIN(x)	(x ? (nvfx_t*)(x+1) : NULL)

#endif /* #ifndef _TEGRA_NVFX_PLUGIN_H_ */
