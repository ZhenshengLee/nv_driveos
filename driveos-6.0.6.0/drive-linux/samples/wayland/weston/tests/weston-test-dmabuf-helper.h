/*
 * Copyright © 2020 NVIDIA Corporation.
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

#ifndef _WESTON_TEST_DMABUF_HELPER_H_
#define _WESTON_TEST_DMABUF_HELPER_H_

#include "weston-test-client-helper.h"

#include <xf86drm.h>
#include <drm_fourcc.h>
#include <linux-dmabuf-unstable-v1-client-protocol.h>
#include <linux-explicit-synchronization-unstable-v1-client-protocol.h>

#define MAX_BUFFER_PLANES 4

typedef void *EGLImageKHR;
typedef unsigned int GLuint;

struct dma_buffer_allocator;

struct dma_buffer {
	struct wl_buffer *buffer;

	struct dma_buffer_allocator *allocator;

	bool use_explicit_sync;
	int busy;

	struct gbm_bo *bo;

	int width;
	int height;
	int format;
	uint64_t modifier;
	int plane_count;
	int dmabuf_fds[MAX_BUFFER_PLANES];
	uint32_t strides[MAX_BUFFER_PLANES];
	uint32_t offsets[MAX_BUFFER_PLANES];

	EGLImageKHR egl_image;
	GLuint gl_texture;
	GLuint gl_fbo;

	struct zwp_linux_buffer_release_v1 *buffer_release;
	/* The buffer owns the release_fence_fd, until it passes ownership
	 * to it to EGL (see wait_for_buffer_release_fence). */
	int release_fence_fd;
};

// NOTE caller must do a display_roundtrip before create_dmabuf_buffer()
struct dma_buffer_allocator *
create_dma_buffer_allocator(struct zwp_linux_dmabuf_v1 *dmabuf,
			    char const *drm_render_node,
			    bool protected_ctx);

void
free_dma_buffer_allocator(struct dma_buffer_allocator *allocator);

int
create_dmabuf_buffer(struct dma_buffer_allocator *allocator,
		     struct dma_buffer *buffer,
		     int width,
		     int height,
		     bool req_dmabuf_immediate,
		     bool alloc_protected,
		     bool use_explicit_sync);

void
free_dmabuf_buffer(struct dma_buffer *buffer);

int
create_fbo_for_buffer(struct dma_buffer_allocator *display,
		      struct dma_buffer *buffer,
		      bool protected);

void
fill_dma_buffer_solid(struct dma_buffer *buffer, pixman_color_t color);

#endif