/*
 * Copyright (c) 2019 - 2020, NVIDIA Corporation.  All Rights Reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation
 * is strictly prohibited.
 */

/*
 * Copyright © 2012 John Kåre Alsaker
 * Copyright © 2013 Vasily Khoruzhick <anarsoul@gmail.com>
 * Copyright © 2019-2020 NVIDIA Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _WESTON_HAL_RENDERER_H_
#define _WESTON_HAL_RENDERER_H_

#include "config.h"

#include <stdint.h>

#include "compositor.h"
#include "gbm.h"
#include "pixman.h"

struct hal_renderer_interface {
	int (*create)(struct weston_compositor *compositor, int drm_fd);

	int (*output_set_buffer)(struct weston_output *output, int idx);

	int (*output_create)(struct weston_output *output,
			     struct gbm_bo **gbm_bo,
			     int num_bo);

	int (*output_add_vpr_buffers) (struct weston_output *output,
			     struct gbm_bo **gbm_bo);

	bool (*output_has_vpr_buffers) (struct weston_output *output);

	void (*output_set_hw_extra_damage)(struct weston_output *output,
					   pixman_region32_t *extra_damage);

	void (*output_destroy)(struct weston_output *output);

	/* Create fence sync FD to wait for HAL rendering.
	 * Return FD on success, -1 on failure.
	 * Caller takes ownership of FD.
	 */
	int (*create_fence_fd)(struct weston_output *output);

	/* Get fence sync FD to wait for HAL rendering.
	 * Return FD on success, -1 on failure.
	 * Caller does not take ownership of FD.
	 */
	int (*get_fence_fd)(struct weston_output *output);
	bool (*is_protected)(struct weston_output *output);
};

#endif // _WESTON_HAL_RENDERER_H_
