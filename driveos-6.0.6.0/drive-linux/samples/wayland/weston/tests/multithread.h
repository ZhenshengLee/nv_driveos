/* Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __WESTON_MULTITHREAD_TEST_H
#define __WESTON_MULTITHREAD_TEST_H
#include <stdio.h>
#include <errno.h>
#include <assert.h>
#include <wayland-client.h>
#include <wayland-egl.h>
#include <EGL/egl.h>
#include <GLES2/gl2.h>

#define _GNU_SOURCE
#include <pthread.h>

#include "xdg-shell-unstable-v6-client-protocol.h"

#define TEST_FLAG_IVI_SHELL   (1<<0)

// debug log output
#define dlog(...)						\
	if(options.debug) {					\
		printf("%s[%d]: ", __FUNCTION__, __LINE__);	\
		printf(__VA_ARGS__);				\
	}
// error log
#define elog(...) {						\
		fprintf(stderr, "Error(%d): [%s:%d] ", errno, __FUNCTION__, __LINE__); \
		fprintf(stderr, __VA_ARGS__);			\
	}
// verbose message
#define vlog(...)					    \
	if (options.verbose) {				    \
		printf("[%s:%d] ", __FUNCTION__, __LINE__); \
		printf( __VA_ARGS__);			\
	}

#define FAILED_IF(rc)				\
	if (rc) {				\
		elog("Failed, rc = %d \n", rc); \
		return rc;			\
	}


struct winsys {
	struct wl_display * wl_display;
	struct wl_registry * wl_registry;
	struct wl_compositor *wl_compositor;
	struct zxdg_shell_v6 * xdg_shell;
	struct ivi_application *ivi_app;

	EGLDisplay egl_display;

	int status;
	pthread_t tid;
};

struct options {
	int debug;
	int verbose;

	int test_flag;

	int win_size[2];
	int timeout;
	int thread_num;
	int frame_limit;

	int thread_loops;
	int win_loops;
	int egl_ctx_loops;
	int egl_surf_loops;
	int win_resize;
};

extern struct options options;
extern struct winsys winsys;

int  win_manager_init();
int  win_manager_report();
void win_manager_fini();

#endif
