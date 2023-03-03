/*
 * plugin.h - Header for defining plugin interface
 *
 * Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef _PLUGIN_H_
#define _PLUGIN_H_

#include <list.h>
#include <adsp_app.h>
#include <err.h>
#include "nvfx.h"
#include "tegra_nvfx_plugin.h"

#define PLUGIN_APP_START(name) ADSP_APP_START(name)

#define CREATE_PLUGIN(nvfx, \
				dram_size,          /* NVFX_MEM_INSTANCE */ \
				dram_shared_size,   /* NVFX_MEM_SHARED */ \
				dram_shared_wc_size,/* NVFX_MEM_FAST_SHARED */ \
				aram_size,          /* NVFX_MEM_INTERNAL */ \
				aram_x_size,        /* NVFX_MEM_REQ_INTERNAL */ \
				plugin_version \
) \
static status_t plugin_init(const struct adsp_app_descriptor* app, void* reserved) \
{ \
	plugin_t *plugin = (plugin_t*)app->mem.dram; \
	nvfx_t* fx = (nvfx_t*)(plugin + 1); \
	plugin_shared_mem_t* plugin_shared = PLUGIN_SHARED_MEM(app->mem.shared); \
	nvfx_shared_state_t* shared_state = NVFX_SHARED_STATE(app->mem.shared); \
\
	/* Initialize the FX */ \
	nvfx->init(nvfx, fx, shared_state, \
		   app->mem.shared_wc, app->mem.aram, app->mem.aram_x); \
\
	list_clear_node(&plugin->node); \
\
	/* Initialize the shared memory with the FX interface */ \
	plugin_shared->plugin.pint = (int32_t*)plugin; \
\
	return NO_ERROR; \
} \
PLUGIN_APP_START(PLUGIN_NAME) \
	.init = plugin_init, \
	.entry = NULL, \
	.mem_size = { \
		.dram = dram_size + sizeof(plugin_t), \
		.dram_shared = dram_shared_size + sizeof(plugin_shared_mem_t), \
		.dram_shared_wc = dram_shared_wc_size, \
		.aram = aram_size, \
		.aram_x = aram_x_size \
	}, \
	.flags = 0, \
	.version = plugin_version, \
ADSP_APP_END

#endif /* #ifndef _PLUGIN_H_ */
