/*
 * Copyright (c) 2019 - 2022, NVIDIA Corporation.  All Rights Reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation
 * is strictly prohibited.
 */

/*
 * Copyright © 2012 Intel Corporation
 * Copyright © 2013 Vasily Khoruzhick <anarsoul@gmail.com>
 * Copyright © 2015 Collabora, Ltd.
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
 * NONINFRINGEMENT.	 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "hal-renderer.h"
#include "nvcomposer.h"

#include "shared/fd-util.h"
#include "shared/helpers.h"
#include "shared/platform.h"

#include "xf86drm.h"
#include <drm_fourcc.h>
#include "linux-dmabuf.h"

#include "pixel-formats.h"

#include "linux-dmabuf-unstable-v1-server-protocol.h"
#include "linux-explicit-synchronization.h"

#include "gl-renderer.h"

#include "compositor/weston.h"
/* For the WL_*_PRESENT_MODE enums */
#include "wayland-eglstream-controller-server-protocol.h"

#include "weston-debug.h"

#include <gbm.h>

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_DMABUFEXPORT_PLANES 3
 // TODO: Check if this is fine, or need to be determined at runtime.
 // There are several places where there are hard-coded values
#define MAX_EGLIMG_CONSUMER_FRAMES 3

// NV24 is not yet part of the Wayland protocol. Assume that it will follow
// the convention of other formats and have a value equal to the corresponding
// DRM format.
#define WL_SHM_FORMAT_NV24 (DRM_FORMAT_NV24)

#ifndef DRM_RDWR
#define DRM_RDWR O_RDWR
#endif

struct hal_image {
	int32_t width;
	int32_t height;
	struct hal_renderer *renderer;
	nvcomposer_buffer_t *nvc_buffer;
	struct gbm_bo *gbm_bo_scratch;
	int refcount;
	bool is_solid_color;
};

struct dmabuf_image {
	struct linux_dmabuf_buffer *dmabuf;
	struct hal_image *hal_image;
	struct wl_list link;
};

struct hal_shm_buffer_state {
	struct weston_buffer *buffer;
	struct hal_image *hal_image;
	struct wl_listener buffer_destroy_listener;
	bool full_damage;
};

struct yuv_plane_descriptor {
	int width_divisor;
	int height_divisor;
	uint32_t format;
	int plane_index;
};

struct yuv_format_descriptor {
	uint32_t format;
	int input_planes;
	int output_planes;
	int texture_type;
	struct yuv_plane_descriptor plane[4];
};

struct hal_renderer {
	struct weston_renderer base;
	struct nvcomposer *composer;

	struct wl_signal destroy_signal;

	int has_dmabuf_import;
	int has_dmabuf_import_modifiers;

	struct wl_list dmabuf_images;

	int drm_fd;
	bool own_drm_fd;
	struct gbm_device *gbm;
	int protected_buffer_count;

	struct weston_debug_scope *debug;
	bool use_gamma_correction;
};

struct hal_output_state {
	struct hal_image *hw_buffer[4];
	int current_buffer;
	int end_render_fence_fd;
	pixman_region32_t *hw_extra_damage;
	pixman_region32_t *buffer_damage;
	bool do_prepare_read_pixels;
};

struct hal_surface_state {
	struct weston_surface *surface;

	int pitch;  /* in pixels */
	int height; /* in pixels */

	struct hal_image *hal_image;
	struct weston_buffer_reference buffer_ref;
	struct weston_buffer_release_reference buffer_release_ref;

	struct wl_listener surface_destroy_listener;
	struct wl_listener renderer_destroy_listener;

	bool used_in_output_repaint;

	struct {
		EGLImage pending_egl_image;
		EGLImage cur_egl_image;
		EGLImage prev_egl_image;
		EGLImage egl_image_list[MAX_EGLIMG_CONSUMER_FRAMES];
		struct linux_dmabuf_buffer dmabufs[MAX_EGLIMG_CONSUMER_FRAMES];
		int fds[MAX_DMABUFEXPORT_PLANES][MAX_DMABUFEXPORT_PLANES];
		int num_frames;
	} egl_img_cons;

	GLenum target;
	int num_images;

	EGLStreamKHR egl_stream;
	bool new_stream;

	int32_t present_mode;
	int32_t fifo_length;

	pixman_region32_t surface_damage;
};

#define MAX_PLANES 4
const static struct shm_format_info {
	uint32_t shm_format;
	int num_planes;
	struct {
		int w;
		int h;
		int bpp;
	} planes[MAX_PLANES];
} drm_formats[] = {
	{WL_SHM_FORMAT_ARGB8888, 1, {{1, 1, 32}, {0, 0, 0}, {0, 0, 0}}},
	{WL_SHM_FORMAT_XRGB8888, 1, {{1, 1, 32}, {0, 0, 0}, {0, 0, 0}}},
	{WL_SHM_FORMAT_RGB565,   1, {{1, 1, 16}, {0, 0, 0}, {0, 0, 0}}},
	{WL_SHM_FORMAT_NV12,     2, {{1, 1, 8}, {2, 2, 16}, {0, 0, 0}}},
	{WL_SHM_FORMAT_NV16,     2, {{1, 1, 8}, {2, 1, 16}, {0, 0, 0}}},
	{WL_SHM_FORMAT_NV24,     2, {{1, 1, 8}, {1, 1, 16}, {0, 0, 0}}},
	{WL_SHM_FORMAT_YUV420,   3, {{1, 1, 8}, {2, 2,  8}, {2, 2, 8}}},
	{WL_SHM_FORMAT_YUYV,     1, {{1, 1, 16}, {0, 0, 0}, {0, 0, 0}}},
};

#define hal_debug(b, ...) \
	weston_debug_scope_printf((b)->debug, __VA_ARGS__)

static inline const char *
dump_format(uint32_t format, char out[4])
{
#if BYTE_ORDER == BIG_ENDIAN
	format = __builtin_bswap32(format);
#endif
	memcpy(out, &format, 4);
	return out;
}

static inline nvcomposer_transform_t
wl_transform_to_nv_transform(uint32_t transform)
{
	switch (transform) {
	    case WL_OUTPUT_TRANSFORM_NORMAL:
		    return NVCOMPOSER_TRANSFORM_NONE;
	    case WL_OUTPUT_TRANSFORM_90:
		    return NVCOMPOSER_TRANSFORM_ROT_90;
	    case WL_OUTPUT_TRANSFORM_180:
		    return NVCOMPOSER_TRANSFORM_ROT_180;
	    case WL_OUTPUT_TRANSFORM_270:
		    return NVCOMPOSER_TRANSFORM_ROT_270;
	    case WL_OUTPUT_TRANSFORM_FLIPPED:
		    return NVCOMPOSER_TRANSFORM_FLIP_H;
	    case WL_OUTPUT_TRANSFORM_FLIPPED_90:
		    return NVCOMPOSER_TRANSFORM_INV_TRANSPOSE;
	    case WL_OUTPUT_TRANSFORM_FLIPPED_180:
		    return NVCOMPOSER_TRANSFORM_FLIP_V;
	    case WL_OUTPUT_TRANSFORM_FLIPPED_270:
		    return (NVCOMPOSER_TRANSFORM_FLIP_V |
			    NVCOMPOSER_TRANSFORM_ROT_90);
	    default:
		    assert(!"Invalid Wayland transform");
		    return NVCOMPOSER_TRANSFORM_NONE;
	}
}

static inline struct hal_output_state *
get_output_state(struct weston_output *output)
{
	return (struct hal_output_state *)output->renderer_state;
}

static int
hal_renderer_create_surface(struct weston_surface *surface);

static inline struct hal_surface_state *
get_surface_state(struct weston_surface *surface)
{
	if (!surface->renderer_state)
		hal_renderer_create_surface(surface);

	return (struct hal_surface_state *)surface->renderer_state;
}

static struct hal_shm_buffer_state*
create_shm_buffer_state(struct weston_buffer *buffer)
{
	struct hal_shm_buffer_state *bs;

	bs = zalloc(sizeof *bs);
	if (bs == NULL) {
		return NULL;
	}

	buffer->renderer_state = bs;

	bs->buffer = buffer;

	return bs;
}

static struct hal_shm_buffer_state*
get_shm_buffer_state(struct weston_buffer *buffer)
{
	assert(buffer->shm_buffer);
	return (struct hal_shm_buffer_state *)buffer->renderer_state;
}

static inline struct hal_renderer *
get_renderer(struct weston_compositor *ec)
{
	return (struct hal_renderer *)ec->renderer;
}

static inline int
scratch_blit_shm_buffer(struct weston_view *ev)
{
	struct weston_surface *surface = ev->surface;
	struct hal_surface_state *vs = get_surface_state(surface);

	struct hal_image *hal_image = vs->hal_image;
	if (!hal_image) {
		assert(0);
		return -1;
	}

	struct hal_renderer *vr = hal_image->renderer;
	nvcomposer_buffer_t *nvc_buffer = hal_image->nvc_buffer;
	if (!nvc_buffer) {
		assert(0);
		weston_log("Error: %s: missing nvcomposer_buffer\n", __func__);
		return -1;
	}

	struct weston_buffer *weston_buffer = vs->buffer_ref.buffer;
	struct hal_shm_buffer_state *bs = get_shm_buffer_state(weston_buffer);
	if (hal_image != bs->hal_image) {
		weston_log("Error: %s: hal_image cache mismatch\n", __func__);
		assert(0);
		return -1;
	}

	hal_debug(vr, "\t\t\t\t[hal][scratch_blit_shm_buffer][weston_view] "
		  "full_damage=%d, surface_damage=(x1=%d, y1=%d, x2=%d, y2=%d)\n",
		  bs->full_damage,
		  vs->surface_damage.extents.x1, vs->surface_damage.extents.y1,
		  vs->surface_damage.extents.x2, vs->surface_damage.extents.y2);

	// Map an SHM buffer to a dmabuf only if there's damage, otherwise
	// return here and use the cached version.
	if (!bs->full_damage && !pixman_region32_not_empty(&vs->surface_damage)) {
		hal_debug(vr, "\t\t\t\t[hal][scratch_blit_shm_buffer] Using a cached SHM buffer\n");
		return 0;
	}

	// The hal_image instance has a gbm bo scratch buffer if it's created
	// from an SHM buffer or with hal_renderer_surface_set_color. In this
	// function we're only interested in the former.
	struct gbm_bo *gbo_scratch = hal_image->gbm_bo_scratch;
	assert(gbo_scratch);

	uint32_t src_format =
		wl_shm_buffer_get_format(weston_buffer->shm_buffer);
	const struct shm_format_info *format_info = NULL;
	for (int ii = 0; ii < ARRAY_LENGTH(drm_formats); ii++) {
		if (drm_formats[ii].shm_format == src_format) {
			format_info = &drm_formats[ii];
			break;
		}
	}
	if (!format_info) {
		weston_log("Error: %s: SHM Format unknown\n", __func__);
		return -1;
	}

	uint32_t map_x;
	uint32_t map_y;
	uint32_t map_width;
	uint32_t map_height;

	if (bs->full_damage || format_info->num_planes > 1) {
		if (format_info->num_planes > 1) {
			// We can't get proper offsets and strides for
			// multi-planar formats when mapping a sub-rectangle, so
			// we only perform sub-rectangle mapping for single
			// plane formats.
			hal_debug(vr, "\t\t\t\t[hal][scratch_blit_shm_buffer] Performance warning!\n");
		}

		map_x = 0;
		map_y = 0;
		map_width = hal_image->width;
		map_height = hal_image->height;

	} else {
		pixman_box32_t damage_box_buffer;
		weston_matrix_transform_rect(&damage_box_buffer,
					     &surface->surface_to_buffer_matrix,
					     &vs->surface_damage.extents);

		hal_debug(vr, "\t\t\t\t[hal][scratch_blit_shm_buffer] "
			  "damage_box_buffer=(x1=%d, y1=%d, x2=%d, y2=%d)\n",
			  damage_box_buffer.x1, damage_box_buffer.y1,
			  damage_box_buffer.x2, damage_box_buffer.y2);

		map_x = CLAMP(damage_box_buffer.x1, 0, hal_image->width);
		map_y = CLAMP(damage_box_buffer.y1, 0, hal_image->height);
		map_width = CLAMP(damage_box_buffer.x2 - map_x, 0, hal_image->width);
		map_height = CLAMP(damage_box_buffer.y2 - map_y, 0, hal_image->height);
	}

	hal_debug(vr, "\t\t\t\t[hal][scratch_blit_shm_buffer][gmb_bo] "
		  "map_x=%d, map_y=%d, map_width=%d, map_height=%d\n",
		  map_x, map_y, map_width, map_height);

	uint32_t dst_stride;
	void *gbo_mapping = NULL;
	char *dst_ptr = gbm_bo_map(gbo_scratch,
				   map_x,
				   map_y,
				   map_width,
				   map_height,
				   GBM_BO_TRANSFER_READ_WRITE,
				   &dst_stride,
				   &gbo_mapping);

	if (!dst_ptr) {
		weston_log("Error: %s: gbm_bo_map failed\n", __func__);
		return -1;
	}

	hal_debug(vr, "\t\t\t\t[hal][scratch_blit_shm_buffer][gmb_bo] dst_stride=%d\n",
		  dst_stride);

	wl_shm_buffer_begin_access(weston_buffer->shm_buffer);
	char *src_ptr =
		wl_shm_buffer_get_data(weston_buffer->shm_buffer);

	assert(format_info->num_planes == gbm_bo_get_plane_count(gbo_scratch));
	for (int ii = 0; ii < format_info->num_planes; ii++) {
		int bpp = format_info->planes[ii].bpp;
		int plane_src_width = ((hal_image->width + format_info->planes[ii].w-1) /
				       format_info->planes[ii].w);
		uint32_t src_stride = plane_src_width * (bpp / 8);

		int plane_map_width = ((map_width + format_info->planes[ii].w-1) /
				       format_info->planes[ii].w);
		int plane_map_height = ((map_height + format_info->planes[ii].h-1) /
					format_info->planes[ii].h);
		uint32_t src_map_stride = plane_map_width * (bpp / 8);

		uint32_t src_y = map_y * plane_src_width * (bpp / 8);
		uint32_t src_x = map_x * (bpp / 8);

		uint32_t dst_plane_stride = ii == 0 ? dst_stride : gbm_bo_get_stride_for_plane(gbo_scratch, ii);
		uint32_t dst_offset = gbm_bo_get_offset(gbo_scratch, ii);
		assert((ii == 0 && dst_offset == 0) || ii > 0);

		hal_debug(vr, "\t\t\t\t[hal][scratch_blit_shm_buffer][memcpy] "
			  "src_stride=%d, plane_map_width=%d, src_map_stride=%d, src_x=%d, src_y=%d, "
			  "dst_plane_stride=%d, dst_offset=%d\n",
			  src_stride, plane_map_width, src_map_stride, src_x, src_y, dst_plane_stride, dst_offset);

		char *dst_plane_ptr = dst_ptr + dst_offset;
		src_ptr += src_y;
		for (int jj = 0; jj < plane_map_height; jj++) {
			memcpy(dst_plane_ptr, src_ptr + src_x, src_map_stride);
			src_ptr += src_stride;
			dst_plane_ptr += dst_plane_stride;
		}
	}

	gbm_bo_unmap(gbo_scratch, gbo_mapping);
	wl_shm_buffer_end_access(weston_buffer->shm_buffer);

	return 0;
}

static int
init_nvcomposer_layer(nvcomposer_layer_t *layer,
		      struct weston_view *ev,
		      struct weston_output *output)
{
	struct weston_surface *surface = ev->surface;

	struct hal_surface_state *vs = get_surface_state(ev->surface);

	struct hal_image *hal_image = vs->hal_image;
	if (!hal_image) {
		return -1;
	}

	struct hal_renderer *vr = hal_image->renderer;

	nvcomposer_buffer_t *nvc_buffer = hal_image->nvc_buffer;
	if (!nvc_buffer) {
		weston_log("Error: %s: missing nvcomposer_buffer\n", __func__);
		return -1;
	}

	hal_debug(vr, "\t\t[hal][init_nvcomposer_layer][weston_view] %p\n",
		  ev);
	hal_debug(vr, "\t\t\t[hal][init_nvcomposer_layer][hal_image] %p\n",
		  hal_image);
	hal_debug(vr, "\t\t\t[hal][init_nvcomposer_layer][hal_image] width=%d, height=%d, "
		  "nvc_buffer=%p, gbm_bo_scratch=%p, is_solid_color=%d\n",
		  hal_image->width, hal_image->height, hal_image->nvc_buffer,
		  hal_image->gbm_bo_scratch, hal_image->is_solid_color);

	// Handle buffer transformations and culling

	uint32_t bbox_transform = output->transform;
	nvcomposer_transform_t nv_transform = NVCOMPOSER_TRANSFORM_NONE;

	// Only pass the output transformation to the NV Composer backend, if
	// the buffer doesn't have a transformation applied.
	if (surface->buffer_viewport.buffer.transform == WL_OUTPUT_TRANSFORM_NORMAL) {
		nv_transform = wl_transform_to_nv_transform(output->transform);
	}

	hal_debug(vr, "\t\t\t[hal][init_nvcomposer_layer][weston_output] transform=%d\n",
		  output->transform);
	hal_debug(vr, "\t\t\t[hal][init_nvcomposer_layer][weston_output] nv_transform=%d\n",
		  nv_transform);

	// Transform the view bounding box from global to output coordinates:
	pixman_box32_t *bbox_global = &ev->transform.boundingbox.extents;

	hal_debug(vr, "\t\t\t[hal][init_nvcomposer_layer][weston_output] region.extents.(x1=%d, y1=%d, x2=%d, y2=%d)\n",
		  output->region.extents.x1, output->region.extents.y1, output->region.extents.x2, output->region.extents.y2);
	hal_debug(vr, "\t\t\t[hal][init_nvcomposer_layer][weston_view] clip.extents(x1=%d, y1=%d, x2=%d, y2=%d)\n",
		  ev->clip.extents.x1, ev->clip.extents.y1, ev->clip.extents.x2, ev->clip.extents.y2);
	hal_debug(vr, "\t\t\t[hal][init_nvcomposer_layer][weston_view] bbox_global->(x1=%d, y1=%d, x2=%d, y2=%d)\n",
		  bbox_global->x1, bbox_global->y1, bbox_global->x2, bbox_global->y2);

	// If the view is out of current display, then don't composite it to the
	// current display
	if (pixman_region32_contains_rectangle(&output->region, bbox_global) == PIXMAN_REGION_OUT) {
		hal_debug(vr, "\t\t\t[hal][init_nvcomposer_layer] PIXMAN_REGION_OUT\n");
		return -1;
	}

	pixman_box32_t bbox_output;
	weston_matrix_transform_rect(&bbox_output,
				     &output->matrix,
				     bbox_global);

	hal_debug(vr, "\t\t\t[hal][init_nvcomposer_layer][weston_view] bbox_output->(x1=%d, y1=%d, x2=%d, y2=%d)\n",
		  bbox_output.x1, bbox_output.y1, bbox_output.x2, bbox_output.y2);

	// This can be NULL, since hal_renderer_surface_set_color doesn't
	// provide a weston_buffer instance:
	struct weston_buffer *weston_buffer = vs->buffer_ref.buffer;

	hal_debug(vr, "\t\t\t[hal][init_nvcomposer_layer] weston_buffer=%p\n",
		  weston_buffer);

	// By default Weston sets weston_buffer->y_inverted = 1. This boolean is
	// relative to the OpenGL coordinate system where the origin is at the
	// bottom-left corner. So if y_inverted == 0, the buffer uses GL
	// coordinates and HAL should flip it vertically, i.e. around the
	// horizontal axis.
	if (weston_buffer) {
		hal_debug(vr, "\t\t\t[hal][init_nvcomposer_layer] weston_buffer->y_inverted=%d\n",
			  weston_buffer->y_inverted);
	    if (weston_buffer->y_inverted == 0) {
		    nv_transform = nv_transform ^ NVCOMPOSER_TRANSFORM_FLIP_V;
	    }
	}

	// Handle SHM buffers
	if (weston_buffer && weston_buffer->shm_buffer) {
		assert(surface->acquire_fence_fd < 0);

		hal_debug(vr, "\t\t\t[hal][init_nvcomposer_layer][weston_buffer] shm_buffer=%p\n",
			  weston_buffer->shm_buffer);

		int err = scratch_blit_shm_buffer(ev);
		if (err) {
			return err;
		}

		struct hal_shm_buffer_state *bs = get_shm_buffer_state(weston_buffer);
		bs->full_damage = false;
	}

	pixman_region32_clear(&vs->surface_damage);

	// Setup the NV Composer Layer instance

	nvcomposer_rect_t source_crop;
	if (ev->geometry.scissor_enabled) {
		source_crop.left = ev->geometry.scissor.extents.x1;
		source_crop.top = ev->geometry.scissor.extents.y1;
		source_crop.right = ev->geometry.scissor.extents.x2;
		source_crop.bottom = ev->geometry.scissor.extents.y2;
	} else {
		source_crop.left = 0;
		source_crop.top = 0;
		source_crop.right = hal_image->width;
		source_crop.bottom = hal_image->height;
	}

	nvcomposer_rect_t display_frame;
	display_frame.left = bbox_output.x1;
	display_frame.right = bbox_output.x2;
	display_frame.top = bbox_output.y1;
	display_frame.bottom = bbox_output.y2;

	memset(layer, 0, sizeof(*layer));
	layer->flags = hal_image->is_solid_color ? NVCOMPOSER_LAYER_FLAG_IS_SOLID_COLOR : 0;
	layer->dataSpace = 0;

	layer->transform = nv_transform;

	layer->planeAlpha = ev->alpha * 255;
	layer->blending = NVCOMPOSER_BLENDING_PREMULT;

	layer->sourceCrop = source_crop;
	layer->displayFrame = display_frame;
	layer->buffer = nvc_buffer;

	layer->buffertype = NVCOMPOSER_BUFFERTYPE_DMABUF;

	// nvcomposer does NOT take ownership of this FD, so no need to dup it.
	layer->acquireFence = surface->acquire_fence_fd;

	switch (ev->surface->blend_mode) {
	case WESTON_SURFACE_BLEND_NONE:
		layer->blending = NVCOMPOSER_BLENDING_NONE;
		break;
	case WESTON_SURFACE_BLEND_PREMULT:
		layer->blending = NVCOMPOSER_BLENDING_PREMULT;
		break;
	case WESTON_SURFACE_BLEND_COVERAGE:
		layer->blending = NVCOMPOSER_BLENDING_COVERAGE;
		break;
	}

	vs->used_in_output_repaint = true;
	return 0;
}

static int
repaint_surfaces(struct weston_output *output, pixman_region32_t *hw_damage)
{
	struct weston_compositor *compositor = output->compositor;
	struct hal_renderer *vr =
		(struct hal_renderer *)output->compositor->renderer;
	struct hal_output_state *vo = get_output_state(output);
	struct weston_mode *mode = output->current_mode;
	struct weston_view *view;

	nvcomposer_rect_t clip_rect;
	clip_rect.left = 0;
	clip_rect.right = mode->width;
	clip_rect.top = 0;
	clip_rect.bottom = mode->height;

	nvcomposer_contents_t contents;
	nvcomposer_contents_init(&contents);
	contents.numLayers = 0;
	contents.minScale = 1.0f;
	contents.maxScale = 1.0f;
	contents.clip = clip_rect;
	contents.flags = NVCOMPOSER_FLAG_GEOMETRY_CHANGED;
	if (vr->use_gamma_correction)
		contents.flags |= NVCOMPOSER_FLAG_GAMMA_CORRECTION;

	nvcomposer_target_t target;
	nvcomposer_target_init(&target);
	target.buffer = vo->hw_buffer[vo->current_buffer]->nvc_buffer;
	target.buffertype = NVCOMPOSER_BUFFERTYPE_DMABUF;
	target.acquireFence = -1;
	target.releaseFence = -1;
	target.flags = NVCOMPOSER_FLAG_GEOMETRY_CHANGED;
	if (vo->do_prepare_read_pixels == true) {
		target.flags |= NVCOMPOSER_FLAG_PREPARE_READ_PIXELS;
	}

	hal_debug(vr, "\t\t[hal][repaint_surfaces] hw_damage->extents.(x1=%d, y1=%d, x2=%d, y2=%d)\n",
		  hw_damage->extents.x1, hw_damage->extents.y1, hw_damage->extents.x2, hw_damage->extents.y2);

	int ii = 0;
	wl_list_for_each_reverse(view, &compositor->view_list, link) {
		if (view->plane == &compositor->primary_plane) {
			nvcomposer_layer_t *layer = &contents.layers[ii];

			if (init_nvcomposer_layer(layer,
						  view,
						  output) == 0) {
				ii++;
				contents.numLayers++;
			}
		}
	}

	if (contents.numLayers < 1) {
		vo->do_prepare_read_pixels = false;
		return 0;
	}

	if (contents.numLayers > NVCOMPOSER_MAX_LAYERS) {
		weston_log("Error: %s: maximum layer count exceeded\n", __func__);
		return -1;
	}

	if (nvcomposer_prepare(vr->composer, &contents, &target)) {
		weston_log("Error: %s: nvcomposer_prepare failed\n", __func__);
		return -1;
	}

	if (nvcomposer_set(vr->composer, &contents, &target)) {
		weston_log("Error: %s: nvcomposer_set failed\n", __func__);
		return -1;
	}

	fd_move(&vo->end_render_fence_fd, &target.releaseFence);

	vo->do_prepare_read_pixels = false;

	return 0;
}

static int
hal_renderer_create_fence_fd(struct weston_output *output);

/* Updates the release fences of surfaces that were used in the current output
 * repaint. Should only be used from hal_renderer_repaint_output, so that the
 * information in hal_surface_state.used_in_output_repaint is accurate.
 */
static void
update_buffer_release_fences(struct weston_compositor *compositor,
			     struct weston_output *output)
{
	struct weston_view *view;

	wl_list_for_each_reverse (view, &compositor->view_list, link) {
		struct hal_surface_state *vs;
		struct weston_buffer_release *buffer_release;
		int fence_fd;

		if (view->plane != &compositor->primary_plane)
			continue;

		vs = get_surface_state(view->surface);
		buffer_release = vs->buffer_release_ref.buffer_release;

		if (!vs->used_in_output_repaint || !buffer_release)
			continue;

		fence_fd = hal_renderer_create_fence_fd(output);

		/* If we have a buffer_release then it means we support fences,
		 * and we should be able to create the release fence. If we
		 * can't, something has gone horribly wrong, so disconnect the
		 * client.
		 */
		if (fence_fd < 0) {
			linux_explicit_synchronization_send_server_error(
				buffer_release->resource,
				"Failed to create release fence");
			fd_clear(&buffer_release->fence_fd);
			continue;
		}

		/* At the moment it is safe to just replace the fence_fd,
		 * discarding the previous one:
		 *
		 * 1. If the previous fence fd represents a sync fence from
		 *     a previous repaint cycle, that fence fd is now not
		 *     sufficient to provide the release guarantee and should
		 *     be replaced.
		 *
		 * 2. If the fence fd represents a sync fence from another
		 *    output in the same repaint cycle, it's fine to replace.
		 *    So a fence issued for a later output rendering is
		 *    guaranteed to signal after fences for previous output
		 *    renderings.
		 *
		 * Note that the above is only valid if the buffer_release
		 * fences only originate from the HAL renderer, which guarantees
		 * a total order of operations and fences.	If we introduce
		 * fences from other sources (e.g., plane out-fences), we will
		 * need to merge fences instead.
		 */
		fd_move(&buffer_release->fence_fd, &fence_fd);
	}
}

static void
hal_renderer_repaint_output(struct weston_output *output,
			    pixman_region32_t *output_damage)
{
	struct hal_output_state *vo = get_output_state(output);
	struct weston_compositor *compositor = output->compositor;
	struct weston_view *view;
	pixman_region32_t hw_damage;
	struct hal_renderer *vr = get_renderer(compositor);
	bool is_protected = vo->current_buffer >= (ARRAY_LENGTH(vo->hw_buffer) / 2) ;

	hal_debug(vr, "\t[hal][repaint_output] %s output=%s (%lu)\n",
		  is_protected ? "protected" : "",
		  output->name, (unsigned long) output->id);

	if (!vo->hw_buffer[vo->current_buffer]) {
		weston_log("Error: %s: missing hw_buffer[%d]\n",
			   __func__,
			   vo->current_buffer);
		vo->hw_extra_damage = NULL;
		return;
	}

	/* Clear the used_in_output_repaint flag, so that we can properly track
	 * which surfaces were used in this output repaint. */
	wl_list_for_each_reverse (view, &compositor->view_list, link) {
		if (view->plane == &compositor->primary_plane) {
			struct hal_surface_state *vs =
				get_surface_state(view->surface);
			vs->used_in_output_repaint = false;
		}
	}

	pixman_region32_init(&hw_damage);
	if (vo->hw_extra_damage) {
		pixman_region32_union(&hw_damage,
				      vo->hw_extra_damage,
				      output_damage);
		vo->hw_extra_damage = NULL;
	} else {
		pixman_region32_copy(&hw_damage, output_damage);
	}

	if (repaint_surfaces(output, &hw_damage)) {
		weston_log("Error: %s: repaint failed\n", __func__);
		pixman_region32_fini(&hw_damage);
		return;
	}

	pixman_region32_fini(&hw_damage);

	pixman_region32_copy(&output->previous_damage, output_damage);
	wl_signal_emit(&output->frame_signal, output);

	/* Actual flip should be done by caller */

	update_buffer_release_fences(compositor, output);
}

static int
gbm_bo_to_dmabuf_attributes(const struct hal_renderer *vr,
			    struct gbm_bo *gbm_bo,
			    struct dmabuf_attributes *attribs);

static struct hal_image *
hal_image_create(struct hal_renderer *vr,
		 const struct dmabuf_attributes *attributes)
{
	struct hal_image *image = NULL;
	assert(attributes);

	image = zalloc(sizeof *image);
	image->renderer = vr;
	image->refcount = 1;
	image->width = attributes->width;
	image->height = attributes->height;
	image->nvc_buffer = nvcomposer_attach_dmabuf(vr->composer,
						     attributes->width,
						     attributes->height,
						     attributes->format,
						     attributes->n_planes,
						     attributes->fd,
						     attributes->stride,
						     attributes->offset,
						     attributes->modifier);
	if (!image->nvc_buffer) {
		free(image);
		weston_log("Error: %s: nvcomposer_attach_dmabuf failed\n",
			   __func__);
		return NULL;
	}

	return image;
}

static struct hal_image *
hal_image_create_from_gbm_bo_scratch(struct hal_renderer *vr, struct gbm_bo *gbm_bo_scratch)
{
	struct dmabuf_attributes attributes;
	struct hal_image *image = NULL;

	if (gbm_bo_to_dmabuf_attributes(vr, gbm_bo_scratch, &attributes)) {
		weston_log("Error: %s: gbm_bo_to_dmabuf_attributes failed\n",
			   __func__);
		return NULL;
	}

	image = hal_image_create(vr, &attributes);
	if (!image) {
		weston_log("Error: %s: hal_image_create failed\n", __func__);
		return NULL;
	}

	image->gbm_bo_scratch = gbm_bo_scratch;

	return image;
}

static struct hal_image *
hal_image_ref(struct hal_image *image)
{
	image->refcount++;

	return image;
}

static int
hal_image_unref(struct hal_image *image)
{
	if (!image) {
		return 0;
	}

	struct hal_renderer *vr = image->renderer;

	assert(image->refcount > 0);

	image->refcount--;
	if (image->refcount > 0) {
		return image->refcount;
	}

	if (image->nvc_buffer) {
		nvcomposer_detach_dmabuf(vr->composer, image->nvc_buffer);
		image->nvc_buffer = NULL;
	}

	if (image->gbm_bo_scratch) {
		gbm_bo_destroy(image->gbm_bo_scratch);
		image->gbm_bo_scratch = NULL;
	}

	free(image);

	return 0;
}

static struct dmabuf_image *
dmabuf_image_create(struct linux_dmabuf_buffer *dmabuf, struct hal_image *hal_image)
{
	assert(dmabuf && hal_image);
	struct dmabuf_image *dmabuf_image;

	dmabuf_image = zalloc(sizeof *dmabuf_image);
	wl_list_init(&dmabuf_image->link);

	dmabuf_image->dmabuf = dmabuf;
	dmabuf_image->hal_image = hal_image;

	return dmabuf_image;
}

static void
dmabuf_image_destroy(struct dmabuf_image *image)
{
	if (image->dmabuf) {
		linux_dmabuf_buffer_set_user_data(image->dmabuf, NULL, NULL);
	}

	if (image->hal_image) {
		hal_image_unref(image->hal_image);
		image->hal_image = NULL;
	}

	wl_list_remove(&image->link);
	free(image);
}

static void
hal_renderer_destroy_dmabuf(struct linux_dmabuf_buffer *dmabuf)
{
	struct dmabuf_image *image = linux_dmabuf_buffer_get_user_data(dmabuf);

	dmabuf_image_destroy(image);
}

static void
hal_renderer_query_dmabuf_formats(struct weston_compositor *wc,
				  int **out_formats,
				  int *out_num_formats)
{
	struct hal_renderer *vr = get_renderer(wc);
	assert(vr->has_dmabuf_import);

	const int *formats = NULL;
	int num_formats = vr->composer->numSupportedDrmFormats;

	if (num_formats > 0) {
		formats = vr->composer->supportedDrmFormats;
	} else {
		static const int fallback_formats[] = {
			DRM_FORMAT_ARGB8888, DRM_FORMAT_XRGB8888,
		};
		num_formats = ARRAY_LENGTH(fallback_formats);
		formats = fallback_formats;
	}

	*out_formats = calloc(num_formats, sizeof(**out_formats));
	if (*out_formats == NULL) {
		*out_num_formats = 0;
		return;
	}

	memcpy(*out_formats, formats, num_formats * sizeof(**out_formats));
	*out_num_formats = num_formats;
}

static void
hal_renderer_query_dmabuf_modifiers(struct weston_compositor *wc,
				    int format,
				    uint64_t **out_format_modifiers,
				    int *out_num_format_modifiers)
{
	struct hal_renderer *vr = get_renderer(wc);
	assert(vr->has_dmabuf_import);

	int ii;
	bool found_format = false;
	for (ii = 0; ii < vr->composer->numSupportedDrmFormats; ii++) {
		if (format == vr->composer->supportedDrmFormats[ii]) {
			found_format = true;
		}
	}

	if (!found_format) {
		goto fail;
	}

	const uint64_t *format_modifiers = NULL;
	int num_format_modifiers = vr->composer->numSupportedDrmFormatModifiers;

	if (num_format_modifiers > 0) {
		format_modifiers = vr->composer->supportedDrmFormatModifiers;
	} else {
		goto fail;
	}

	*out_format_modifiers = calloc(num_format_modifiers, sizeof(**out_format_modifiers));
	if (*out_format_modifiers == NULL) {
		goto fail;
	}

	memcpy(*out_format_modifiers, format_modifiers,
	       num_format_modifiers * sizeof(**out_format_modifiers));
	*out_num_format_modifiers = num_format_modifiers;

	return;

    fail:
	*out_format_modifiers = NULL;
	*out_num_format_modifiers = 0;
}

static struct dmabuf_image *
import_dmabuf(struct hal_renderer *vr, struct linux_dmabuf_buffer *dmabuf)
{
	struct hal_image *hal_image = NULL;
	struct dmabuf_image *dmabuf_image = NULL;

	hal_image = hal_image_create(vr, &dmabuf->attributes);
	if (hal_image) {
		dmabuf_image = dmabuf_image_create(dmabuf, hal_image);
	} else {
		weston_log("Error: %s: hal_image_create failed\n",
			   __func__);
	}

	return dmabuf_image;
}

static bool
hal_renderer_import_dmabuf(struct weston_compositor *ec,
			   struct linux_dmabuf_buffer *dmabuf)
{
	struct hal_renderer *vr = get_renderer(ec);
	struct dmabuf_image *dmabuf_image;
	int i;

	assert(vr->has_dmabuf_import);

	for (i = 0; i < dmabuf->attributes.n_planes; i++) {
		/* return if the renderer doesn't support import modifiers */
		if (dmabuf->attributes.modifier[i] != DRM_FORMAT_MOD_INVALID) {
			if (!vr->has_dmabuf_import_modifiers) {
				weston_log(
					"Error: %s: modifiers no supported\n",
					__func__);
				return false;
			}
		}

		/* return if modifiers passed are unequal */
		if (dmabuf->attributes.modifier[i] !=
		    dmabuf->attributes.modifier[0]) {
			weston_log("Error: %s: modifier mismatch\n", __func__);
			return false;
		}
	}

	/* reject all flags we do not recognize or handle */
	if (dmabuf->attributes.flags & ~ZWP_LINUX_BUFFER_PARAMS_V1_FLAGS_Y_INVERT) {
		weston_log("Error: %s: invalid flags\n", __func__);
		return false;
	}

	dmabuf_image = import_dmabuf(vr, dmabuf);
	if (!dmabuf_image) {
		weston_log("Error: %s: import_dmabuf failed\n", __func__);
		return false;
	}

	wl_list_insert(&vr->dmabuf_images, &dmabuf_image->link);
	linux_dmabuf_buffer_set_user_data(dmabuf,
					  dmabuf_image,
					  hal_renderer_destroy_dmabuf);

	return true;
}

static bool
import_known_dmabuf(struct hal_renderer *vr, struct dmabuf_image *dmabuf_image)
{
	if (dmabuf_image->hal_image == NULL) {
		weston_log("Error: missing dmabuf_image->hal_image!\n");
		return false;
	}

	return true;
}

static void
buffer_state_handle_buffer_destroy(struct wl_listener *listener, void *data);

static int
hal_renderer_attach_shm(struct weston_surface *es,
			struct weston_buffer *buffer,
			struct wl_shm_buffer *shm_buffer)
{
	assert(es && buffer && shm_buffer);

	struct weston_compositor *ec = es->compositor;
	struct hal_renderer *vr = get_renderer(ec);
	struct hal_surface_state *vs = get_surface_state(es);

	buffer->shm_buffer = shm_buffer;
	buffer->width = wl_shm_buffer_get_width(shm_buffer);
	buffer->height = wl_shm_buffer_get_height(shm_buffer);

	struct hal_shm_buffer_state *bs = get_shm_buffer_state(buffer);
	if (bs == NULL) {
		bs = create_shm_buffer_state(buffer);
	}

	if (bs->buffer_destroy_listener.notify) {
		wl_list_remove(&bs->buffer_destroy_listener.link);
		bs->buffer_destroy_listener.notify = NULL;
	}

	int gbm_format = 0;
	switch (wl_shm_buffer_get_format(shm_buffer)) {
	case WL_SHM_FORMAT_XRGB8888:
		gbm_format = GBM_FORMAT_XRGB8888;
		break;
	case WL_SHM_FORMAT_ARGB8888:
		gbm_format = GBM_FORMAT_ARGB8888;
		break;
	case WL_SHM_FORMAT_RGB565:
		gbm_format = GBM_FORMAT_RGB565;
		break;
	case WL_SHM_FORMAT_YUV420:
		gbm_format = GBM_FORMAT_YUV420;
		break;
	case WL_SHM_FORMAT_NV12:
		gbm_format = GBM_FORMAT_NV12;
		break;
	case  WL_SHM_FORMAT_YUYV:
		gbm_format = GBM_FORMAT_YUYV;
		break;
	case WL_SHM_FORMAT_NV16:
		gbm_format = GBM_FORMAT_NV16;
		break;
	case WL_SHM_FORMAT_NV24:
		gbm_format = DRM_FORMAT_NV24;
		break;
	default:
		weston_log("warning: unknown shm buffer format: %08x\n",
			   wl_shm_buffer_get_format(shm_buffer));
		return -1;
	}

	if (bs->hal_image != NULL) {
		struct gbm_bo* gbm_bo = bs->hal_image->gbm_bo_scratch;
		if (gbm_bo) {
			// Since the gbm_bo in the hal_image is always created
			// from an SHM buffer, we only need to check this
			// limited number of parameters.
			if (buffer->width != gbm_bo_get_width(gbm_bo) ||
			    buffer->height != gbm_bo_get_height(gbm_bo) ||
			    gbm_format != gbm_bo_get_format(gbm_bo))
			{
				hal_image_unref(bs->hal_image);
				bs->hal_image = NULL;
			}
		}
	}

	if (bs->hal_image == NULL) {
		struct gbm_bo *gbm_bo = gbm_bo_create(vr->gbm,
						      buffer->width,
						      buffer->height,
						      gbm_format,
						      GBM_BO_USE_LINEAR);
		if (gbm_bo == NULL) {
			return -1;
		}

		bs->hal_image = hal_image_create_from_gbm_bo_scratch(vr, gbm_bo);

		if (bs->hal_image == NULL) {
			gbm_bo_destroy(gbm_bo);
			return -1;
		}

		bs->full_damage = true;
	}

	hal_image_unref(vs->hal_image);
	vs->hal_image = hal_image_ref(bs->hal_image);

	vs->pitch = buffer->width;
	vs->height = buffer->height;

	bs->buffer_destroy_listener.notify = buffer_state_handle_buffer_destroy;
	wl_signal_add(&buffer->destroy_signal, &bs->buffer_destroy_listener);

	return 0;
}

static int
hal_renderer_attach_dmabuf(struct weston_surface *surface,
			   struct weston_buffer *buffer,
			   struct linux_dmabuf_buffer *dmabuf)
{
	struct hal_renderer *vr = get_renderer(surface->compositor);
	struct hal_surface_state *vs = get_surface_state(surface);
	struct dmabuf_image *dmabuf_image = NULL;

	if (dmabuf && dmabuf->user_data == NULL) {
		return -1;
	}

	if (!vr->has_dmabuf_import) {
		linux_dmabuf_buffer_send_server_error(
			dmabuf, "dmabuf import not supported");
		return -1;
	}

	buffer->width = dmabuf->attributes.width;
	buffer->height = dmabuf->attributes.height;

	// Y_INVERT indicates that the buffer uses the OpenGL coordinate system,
	// and should be flipped vertically.
	buffer->y_inverted =
		!(dmabuf->attributes.flags & ZWP_LINUX_BUFFER_PARAMS_V1_FLAGS_Y_INVERT);

	dmabuf_image = linux_dmabuf_buffer_get_user_data(dmabuf);
	/* The dmabuf_image should have been created during the import */
	assert(dmabuf_image != NULL);

	if (!import_known_dmabuf(vr, dmabuf_image)) {
		linux_dmabuf_buffer_send_server_error(
			dmabuf, "hal dmabuf import failed");
		return -1;
	}

	vs->pitch = buffer->width;
	vs->height = buffer->height;

	if (vs->hal_image) {
		hal_image_unref(vs->hal_image);
		vs->hal_image = NULL;
	} else {

	/* This path indicates that the surface was just created, then we need to
	 * record that whether this surface is protected.
	 * */
	if(nvcomposer_buffer_is_protected(vr->composer, dmabuf_image->hal_image->nvc_buffer))
		vr->protected_buffer_count++;
	}
	vs->hal_image = dmabuf_image->hal_image;
	hal_image_ref(vs->hal_image);

	return 0;
}

static bool
export_and_copy_dmabuf_attribs(struct gl_renderer *gr,
				struct hal_surface_state *vs,
				struct weston_buffer *buffer,
				struct linux_dmabuf_buffer *dmabuf)
{
#ifdef EGL_NV_stream_consumer_eglimage
	int ret1;
	int ret2;
	int fourcc;
	int fds[MAX_DMABUFEXPORT_PLANES];
	EGLint strides[MAX_DMABUFEXPORT_PLANES];
	EGLint offsets[MAX_DMABUFEXPORT_PLANES];
	int num_planes;
	EGLuint64KHR mod[MAX_DMABUFEXPORT_PLANES];

	ret1 = gr->export_dmabuf_image_query(
				gr->egl_display,
				vs->egl_img_cons.egl_image_list[vs->egl_img_cons.num_frames],
				&fourcc,
				&num_planes,
				mod);

	if (num_planes > MAX_DMABUFEXPORT_PLANES) {
		weston_log("Exporting dmabuf failed!\n");
		return false;
	}

	ret2 = gr->export_dmabuf_image(
				gr->egl_display,
				vs->egl_img_cons.egl_image_list[vs->egl_img_cons.num_frames],
				fds,
				strides,
				offsets);

	if (!ret1 || !ret2) {
		weston_log("Exporting dmabuf failed!\n");
		return false;
	}

	dmabuf->attributes.width    = buffer->width;
	dmabuf->attributes.height   = buffer->height;
	dmabuf->attributes.format   = fourcc;
	dmabuf->attributes.n_planes = num_planes;

	memcpy(dmabuf->attributes.modifier, mod,
		sizeof(EGLuint64KHR) * MAX_DMABUFEXPORT_PLANES);
	memcpy(dmabuf->attributes.offset, offsets,
		sizeof(EGLint) * MAX_DMABUFEXPORT_PLANES);
	memcpy(dmabuf->attributes.stride, strides,
		sizeof(EGLint) * MAX_DMABUFEXPORT_PLANES);
	memcpy(dmabuf->attributes.fd, fds,
		sizeof(int) * MAX_DMABUFEXPORT_PLANES);
	memcpy(vs->egl_img_cons.fds[vs->egl_img_cons.num_frames], fds,
		sizeof(int) * MAX_DMABUFEXPORT_PLANES);

	return true;
#else
	return false;
#endif
}


static void
egl_stream_destroy(struct hal_surface_state *vs)
{
#ifdef EGL_NV_stream_consumer_eglimage
	int ii;
	struct gl_renderer* gr = (struct gl_renderer*)vs->surface->compositor->gl_renderer_stash;

	vs->egl_img_cons.pending_egl_image = EGL_NO_IMAGE;
	vs->egl_img_cons.cur_egl_image = EGL_NO_IMAGE;
	vs->egl_img_cons.prev_egl_image = EGL_NO_IMAGE;

	for (ii = 0; ii < vs->egl_img_cons.num_frames; ii++) {
		if (vs->egl_img_cons.egl_image_list[ii]) {
			gr->destroy_image(gr->egl_display,
					  vs->egl_img_cons.egl_image_list[ii]);
			vs->egl_img_cons.egl_image_list[ii] = EGL_NO_IMAGE;
		}
	}

	if (vs->egl_stream != EGL_NO_STREAM_KHR) {
		gr->destroy_stream(gr->egl_display, vs->egl_stream);
		vs->egl_stream = EGL_NO_STREAM_KHR;
	}

	for (ii = 0; ii < vs->egl_img_cons.num_frames; ii++) {
		// TODO: close just the one handle that gets used.
		close(vs->egl_img_cons.fds[ii][0]);
		vs->egl_img_cons.fds[ii][0] = -1;
	}

	for (ii = 0; ii < vs->egl_img_cons.num_frames; ii++) {
		struct dmabuf_image *image = linux_dmabuf_buffer_get_user_data
			(&vs->egl_img_cons.dmabufs[ii]);
		assert(image);
		dmabuf_image_destroy(image);
	}

	vs->egl_img_cons.num_frames = 0;
#endif // EGL_NV_stream_consumer_eglimage
}

static void
hal_renderer_commit_eglstream_consumer_image(struct weston_surface *es)
{
	struct hal_surface_state *vs = get_surface_state(es);

	vs->egl_img_cons.cur_egl_image = vs->egl_img_cons.pending_egl_image;
}

static bool
hal_renderer_eglimage_consumer_release(struct gl_renderer *gr, struct hal_surface_state *vs)
{
	bool ret = true;
	if (vs->egl_img_cons.prev_egl_image &&
			vs->egl_img_cons.prev_egl_image != vs->egl_img_cons.cur_egl_image) {
		ret = gr->stream_release_image(
					gr->egl_display,
					vs->egl_stream,
					vs->egl_img_cons.prev_egl_image,
					EGL_NO_SYNC);
		if (!ret)
			weston_log("Error: EGLImage consumer release failed!\n");
	}
	vs->egl_img_cons.prev_egl_image = vs->egl_img_cons.cur_egl_image;
	return ret;
}

static bool
hal_renderer_release_eglstream_consumer_image(struct weston_surface *es)
{
	struct gl_renderer *gr = es->compositor->gl_renderer_stash;
	struct hal_surface_state *vs = get_surface_state(es);

	return hal_renderer_eglimage_consumer_release(gr, vs);
}

static bool
hal_renderer_eglimage_consumer_acquire(struct weston_surface *es,
				       struct gl_renderer *gr,
				       struct hal_surface_state *vs,
				       struct weston_buffer *buffer,
				       struct linux_dmabuf_buffer **ret_dmabuf)
{
#ifdef EGL_NV_stream_consumer_eglimage
	EGLenum event = 0;
	const int timeout = 500;//500us
	int cur_frame = -1;
	int i;
	EGLBoolean ret;
	*ret_dmabuf = NULL;

	/**
	 *  First, we create EGLImages for all producer buffers. After that,
	 *  we can start acquiring the frames into the EGLImages. We then
	 *  export dmabuf FDs from these EGLImages.
	 */
	do {
		gr->stream_query_consumer_event(gr->egl_display,
						vs->egl_stream,
						timeout,
						&event,
						NULL);

		if (event == EGL_STREAM_IMAGE_ADD_NV) {
			vs->egl_img_cons.egl_image_list[vs->egl_img_cons.num_frames] = gr->create_image(
						gr->egl_display,
						EGL_NO_CONTEXT,
						EGL_STREAM_CONSUMER_IMAGE_NV,
						vs->egl_stream,
						NULL);

			if (vs->egl_img_cons.egl_image_list[vs->egl_img_cons.num_frames] == NULL) {
				weston_log("Failed to create an EGLImage!\n");
				goto fail_stream_image_add;
			}

			if (!export_and_copy_dmabuf_attribs(gr, vs, buffer, &vs->egl_img_cons.dmabufs[vs->egl_img_cons.num_frames])) {
				weston_log("dmabuf export failed!\n");
				goto fail_stream_image_add;
			}

			if (!hal_renderer_import_dmabuf(es->compositor, &vs->egl_img_cons.dmabufs[vs->egl_img_cons.num_frames])) {
				weston_log("Failed to import a dmabuf (num_frames=%d)!\n", vs->egl_img_cons.num_frames);
				goto fail_stream_image_add;
			}

			vs->egl_img_cons.num_frames++;
		}
	} while (event == EGL_STREAM_IMAGE_ADD_NV);

	if (event == EGL_STREAM_IMAGE_AVAILABLE_NV) {
		EGLImage old_pending = vs->egl_img_cons.pending_egl_image;
		ret = gr->stream_acquire_image(
					gr->egl_display,
					vs->egl_stream,
					&vs->egl_img_cons.pending_egl_image,
					EGL_NO_SYNC);
		if (!ret) {
			weston_log("EGLImage consumer acquire failed!\n");
			goto fail_stream_image_add;
		}

		for (i = 0; i < vs->egl_img_cons.num_frames; i++) {
			if (vs->egl_img_cons.egl_image_list[i] == vs->egl_img_cons.pending_egl_image) {
				cur_frame = i;
				break;
			}
		}
		assert(cur_frame != -1);
		*ret_dmabuf = &vs->egl_img_cons.dmabufs[cur_frame];

		// Handle new attach without intermediate commit
		// TODO: This could be used to implement swap interval 0 properly
		if (old_pending &&
				old_pending != vs->egl_img_cons.pending_egl_image &&
				old_pending != vs->egl_img_cons.cur_egl_image &&
				old_pending != vs->egl_img_cons.prev_egl_image) {
			ret = gr->stream_release_image(
						gr->egl_display,
						vs->egl_stream,
						old_pending,
						EGL_NO_SYNC);
			if (!ret) {
				weston_log("Error: EGLImage consumer release failed!\n");
				goto fail_stream_image_add;
			}
		}
	} else if (event == EGL_STREAM_IMAGE_REMOVE_NV) {
		egl_stream_destroy(vs);
		return true;
	} else {
		weston_log("Error: Bad EGLStream!\n");
		goto fail_stream_image_add;
	}

	return true;

fail_stream_image_add:
	egl_stream_destroy(vs);
#endif
	return false;
}

static EGLint
vic_renderer_attach_eglstream_consumer_common(struct weston_surface *es,
					     struct wl_resource *wl_eglstream)
{
#ifdef EGL_NV_stream_attrib
	struct weston_compositor *ec = es->compositor;
	struct gl_renderer *gr = es->compositor->gl_renderer_stash;
	struct hal_surface_state *vs = get_surface_state(es);
	EGLStreamKHR stream = EGL_NO_STREAM_KHR;
	EGLAttrib stream_attribs[] = {
#ifdef EGL_WL_wayland_eglstream
		EGL_WAYLAND_EGLSTREAM_WL, (EGLAttrib)wl_eglstream,
#endif
		EGL_NONE, EGL_NONE, EGL_NONE, EGL_NONE, EGL_NONE
	};

	if (vs->present_mode != WL_EGLSTREAM_CONTROLLER_PRESENT_MODE_MAILBOX &&
	    gr->has_egl_stream_fifo &&
	    gr->has_egl_stream_fifo_synchronous) {
		stream_attribs[2] = EGL_STREAM_FIFO_SYNCHRONOUS_NV;
		stream_attribs[3] = 1;
		stream_attribs[4] = EGL_STREAM_FIFO_LENGTH_KHR;
		stream_attribs[5] =
		vs->present_mode == WL_EGLSTREAM_CONTROLLER_PRESENT_MODE_DONT_CARE
		? 1 : vs->fifo_length;
	}
	/* Check for required extensions. If they arent supported, there's no
	 *  way the given resource corresponds to an EGLStream */
	if (!gr->has_egl_stream_attrib ||
	    !gr->has_egl_stream_consumer_gltexture ||
	    !gr->has_egl_wayland_eglstream)
		return EGL_BAD_ACCESS;

	stream = gr->create_stream_attrib(gr->egl_display, stream_attribs);

	if (stream == EGL_NO_STREAM_KHR)
		return gr->egl_get_error_func();

	if (vs->egl_stream != EGL_NO_STREAM_KHR)
		egl_stream_destroy(vs);

	vs->egl_stream = stream;
	memset(&vs->egl_img_cons, 0, sizeof(vs->egl_img_cons));

	vs->new_stream = gr->stream_consumer_connect(
					gr->egl_display,
					vs->egl_stream,
					0,
					NULL,
					NULL);


	if (!vs->new_stream) {
		EGLint err = gr->egl_get_error_func();
		weston_log("Failed to set stream consumer\n");

		egl_stream_destroy(vs);

		return err;
	}

	return EGL_SUCCESS;
#else
	return EGL_BAD_ACCESS;
#endif
}

static bool
hal_renderer_attach_stream_eglimage_consumer(struct weston_surface *es,
					     struct weston_buffer *buffer,
					     struct linux_dmabuf_buffer **ret_dmabuf)
{
	struct weston_compositor *ec = es->compositor;
	struct gl_renderer *gr = es->compositor->gl_renderer_stash;
	struct hal_surface_state *vs = get_surface_state(es);
	EGLint stream_state = EGL_STREAM_STATE_EMPTY_KHR;
	*ret_dmabuf = NULL;

	EGLint err = vic_renderer_attach_eglstream_consumer_common(es, buffer->resource);
	switch (err) {
	case EGL_BAD_ACCESS:
		weston_log("Error: EGL_BAD_ACCESS for EGLStream!\n");
		return false;

	case EGL_BAD_STREAM_KHR:
		/* EGL_BAD_STREAM_KHR is generated whenever
		 * buffer->resource corresponds to a previously created
		 * stream so we must have a valid stream handle already
		 * we can use to acquire next frame */
		break;

	case EGL_SUCCESS:
		/* The first time the connection is done, we don't do an acquire
		 * since the dmabuf isn't created yet.
		 */
		return true;

	default:
		/* An unknown error was generated */
		weston_log("Error: Attaching an EGLStream failed!\n");
		assert(0);
		return false;
	}

	/* At this point we should have a valid stream handle */
	assert(vs->egl_stream != EGL_NO_STREAM_KHR);

	/* Check whether there are new frames available */
	if (gr->query_stream(gr->egl_display,
			     vs->egl_stream,
			     EGL_STREAM_STATE_KHR,
			     &stream_state) != EGL_TRUE) {
		weston_log("Error: Failed to query stream state!\n");
		return false;
	}

	gr->query_buffer(gr->egl_display, buffer->resource,
			 EGL_WIDTH, &buffer->width);
	gr->query_buffer(gr->egl_display, buffer->resource,
			 EGL_HEIGHT, &buffer->height);
	gr->query_buffer(gr->egl_display, buffer->resource,
			 EGL_WAYLAND_Y_INVERTED_WL, &buffer->y_inverted);

	if (!hal_renderer_eglimage_consumer_acquire(es, gr, vs, buffer, ret_dmabuf)) {
			weston_log("Error: Failed to acquire a dmabuf for EGLStream!\n");
			return false;
	}

	buffer->eglstream_dmabuf_export = *ret_dmabuf;

	if (*ret_dmabuf) {
		(*ret_dmabuf)->compositor = ec;
	}

	return true;
}


static bool
vic_renderer_attach_eglstream_consumer(struct weston_surface *es,
				       struct wl_resource *stream,
				       struct wl_array *attribs)
{
	intptr_t                *attr;
	struct hal_surface_state *gs = get_surface_state(es);

	wl_array_for_each(attr, attribs) {
		switch (attr[0]) {
			case WL_EGLSTREAM_CONTROLLER_ATTRIB_PRESENT_MODE:
				gs->present_mode = attr[1];
				break;
			case WL_EGLSTREAM_CONTROLLER_ATTRIB_FIFO_LENGTH:
				gs->fifo_length = attr[1];
				break;
			default:
				assert(!"Unknown attribute");
				break;
		}

		/* Attribs processed in pairs */
		attr += 1;
	}

	if ((gs->present_mode == WL_EGLSTREAM_CONTROLLER_PRESENT_MODE_FIFO && gs->fifo_length < 1) ||
	    (gs->present_mode != WL_EGLSTREAM_CONTROLLER_PRESENT_MODE_FIFO && gs->fifo_length > 0)) {
		weston_log("inconsistent client attributes");
		return false;
	}
	EGLint err = vic_renderer_attach_eglstream_consumer_common(es, stream);
	return (err == EGL_SUCCESS);
}


static void
hal_renderer_attach(struct weston_surface *es, struct weston_buffer *buffer)
{
	struct weston_compositor *ec = es->compositor;
	struct hal_renderer *vr = get_renderer(ec);
	struct gl_renderer* gr = (struct gl_renderer*)es->compositor->gl_renderer_stash;

	struct hal_surface_state *vs = get_surface_state(es);
	struct linux_dmabuf_buffer *dmabuf;
	int i = 0, j = 0;

	if (!buffer) {
#ifdef EGL_NV_stream_consumer_eglimage
		egl_stream_destroy(vs);
#endif
		hal_image_unref(vs->hal_image);
		vs->hal_image = NULL;

		goto fail;
	}

	weston_buffer_reference(&vs->buffer_ref, buffer);
	weston_buffer_release_reference(&vs->buffer_release_ref,
					es->buffer_release_ref.buffer_release);

	struct wl_shm_buffer *shm_buffer = wl_shm_buffer_get(buffer->resource);

	if (shm_buffer) {
		if (hal_renderer_attach_shm(es, buffer, shm_buffer)) {
			weston_log("Error: hal_renderer_attach_shm failed!\n");
			goto fail;
		}
	} else if ((dmabuf = linux_dmabuf_buffer_get(buffer->resource))) {
		if (hal_renderer_attach_dmabuf(es, buffer, dmabuf)) {
			weston_log(
				"Error: hal_renderer_attach_dmabuf failed!\n");
			goto fail;
		}
	} else if ((hal_renderer_attach_stream_eglimage_consumer(es, buffer, &dmabuf))) {
		if (dmabuf) {
			hal_renderer_attach_dmabuf(es, buffer, dmabuf);
		}
	} else {
		weston_log("Error: unhandled buffer type!\n");
		weston_buffer_send_server_error(
			buffer, "disconnecting due to unhandled buffer type");
		goto fail;
	}

	return;

fail:
	weston_buffer_reference(&vs->buffer_ref, NULL);
	weston_buffer_release_reference(&vs->buffer_release_ref, NULL);
}

static void
hal_renderer_flush_damage(struct weston_surface *surface)
{
	struct hal_renderer *vr = get_renderer(surface->compositor);
	struct hal_surface_state *vs = get_surface_state(surface);

	hal_debug(vr, "\t[hal][flush_damage] surface=%p\n", surface);

	pixman_box32_t *surface_damage_extents = pixman_region32_extents(&surface->damage);
	hal_debug(vr, "\t[hal][flush_damage] surface->damage=(x1=%d, y1=%d, x2=%d, y2=%d)\n",
		  surface_damage_extents->x1, surface_damage_extents->y1,
		  surface_damage_extents->x2, surface_damage_extents->y2);

	pixman_region32_union(&vs->surface_damage,
			      &vs->surface_damage, &surface->damage);
}

static void
hal_renderer_surface_set_color(struct weston_surface *surface,
			       float red,
			       float green,
			       float blue,
			       float alpha)
{
	// We create a small dmabuf that is filled with a single color, and tell
	// the NV Composer API backend to scale it. We set the size of the
	// buffer so that the scale factor to 4k is 16.
	//
	// TODO: Check the max scale factor from the NV Composer API backend.
	static const int width = 256;
	static const int height = 256;
	static const int gbm_format = GBM_FORMAT_ARGB8888;
	static const int bpp = 32;
	uint32_t src_stride = width * (bpp / 8);

	struct weston_compositor *ec = surface->compositor;
	struct hal_renderer *vr = get_renderer(ec);
	struct hal_surface_state *vs = get_surface_state(surface);
	if (vs->hal_image) {
		hal_image_unref(vs->hal_image);
		vs->hal_image = NULL;
	}

	// We follow the precedent set by Pixman and GL renderers, where calling
	// surface_set_color requires that the buffer is attached again, if the
	// color fill is no longer intended for the surface.
	weston_buffer_reference(&vs->buffer_ref, NULL);
	weston_buffer_release_reference(&vs->buffer_release_ref, NULL);

	struct gbm_bo *gbm_bo = gbm_bo_create(vr->gbm,
					      width,
					      height,
					      gbm_format,
					      GBM_BO_USE_LINEAR);
	if (gbm_bo == NULL) {
		return;
	}

	uint32_t dst_stride = 0;
	void *gbm_mapping = NULL;
	uint8_t *dst_ptr = gbm_bo_map(gbm_bo,
				      0,
				      0,
				      width,
				      height,
				      GBM_BO_TRANSFER_READ_WRITE,
				      &dst_stride,
				      &gbm_mapping);

	for (int ii = 0; ii < height; ii++) {
		int jj = 0;
		while (jj < src_stride) {
			dst_ptr[jj++] = 0xff * blue;
			dst_ptr[jj++] = 0xff * green;
			dst_ptr[jj++] = 0xff * red;
			dst_ptr[jj++] = 0xff * alpha;
		}
		dst_ptr += dst_stride;
	}

	gbm_bo_unmap(gbm_bo, gbm_mapping);

	vs->hal_image = hal_image_create_from_gbm_bo_scratch(vr, gbm_bo);
	vs->hal_image->is_solid_color = true;

	hal_debug(vr, "[hal][surface_set_color] surface=%p, hal_image=%p\n",
		  surface, vs->hal_image);

	hal_debug(vr, "[hal][surface_set_color] red=%f, green=%f, blue=%f, alpha=%f\n",
		  red, green, blue, alpha);

	if (vs->hal_image == NULL) {
		gbm_bo_destroy(gbm_bo);
		return;
	}
}

static void
buffer_state_handle_buffer_destroy(struct wl_listener *listener, void *data)
{
	struct weston_buffer *buffer = (struct weston_buffer*) data;
	struct hal_shm_buffer_state* bs = NULL;

	bs = container_of(listener,
			  struct hal_shm_buffer_state,
			  buffer_destroy_listener);

	if (bs->hal_image) {
		hal_image_unref(bs->hal_image);
		bs->hal_image = NULL;
	}

	if (bs->buffer_destroy_listener.notify) {
		wl_list_remove(&bs->buffer_destroy_listener.link);
		bs->buffer_destroy_listener.notify = NULL;
	}

	free(bs);
}

static void
surface_state_destroy(struct hal_surface_state *vs, struct hal_renderer *vr)
{
	wl_list_remove(&vs->surface_destroy_listener.link);
	wl_list_remove(&vs->renderer_destroy_listener.link);

	vs->surface->renderer_state = NULL;

	pixman_region32_fini(&vs->surface_damage);

	if (vs->hal_image) {
		if(nvcomposer_buffer_is_protected(vr->composer,vs->hal_image->nvc_buffer)) {
			if (vr->protected_buffer_count > 0)
				vr->protected_buffer_count--;
		}
		hal_image_unref(vs->hal_image);
		vs->hal_image = NULL;
	}

	weston_buffer_reference(&vs->buffer_ref, NULL);
	weston_buffer_release_reference(&vs->buffer_release_ref, NULL);

	if (vs->egl_stream != EGL_NO_STREAM_KHR) {
		egl_stream_destroy(vs);
	}

	free(vs);
}

static void
surface_state_handle_surface_destroy(struct wl_listener *listener, void *data)
{
	struct hal_surface_state *vs;
	struct hal_renderer *vr;

	vs = container_of(listener,
			  struct hal_surface_state,
			  surface_destroy_listener);

	vr = get_renderer(vs->surface->compositor);

	surface_state_destroy(vs, vr);
}

static void
surface_state_handle_renderer_destroy(struct wl_listener *listener, void *data)
{
	struct hal_surface_state *vs;
	struct hal_renderer *vr;

	vr = data;

	vs = container_of(listener,
			  struct hal_surface_state,
			  renderer_destroy_listener);

	surface_state_destroy(vs, vr);
}

static int
hal_renderer_create_surface(struct weston_surface *surface)
{
	struct hal_surface_state *vs;
	struct hal_renderer *vr = get_renderer(surface->compositor);

	vs = zalloc(sizeof *vs);
	if (vs == NULL) {
		return -1;
	}

	pixman_region32_init(&vs->surface_damage);

	surface->renderer_state = vs;

	vs->surface = surface;

	vs->surface_destroy_listener.notify =
		surface_state_handle_surface_destroy;
	wl_signal_add(&surface->destroy_signal, &vs->surface_destroy_listener);

	vs->renderer_destroy_listener.notify =
		surface_state_handle_renderer_destroy;
	wl_signal_add(&vr->destroy_signal, &vs->renderer_destroy_listener);

	return 0;
}

static void
hal_renderer_destroy(struct weston_compositor *ec)
{
	struct hal_renderer *vr = get_renderer(ec);
	struct dmabuf_image *image, *next;

	wl_signal_emit(&vr->destroy_signal, vr);

	wl_list_for_each_safe (image, next, &vr->dmabuf_images, link) {
		dmabuf_image_destroy(image);
	}

	if (vr->own_drm_fd) {
		close(vr->drm_fd);
	}
	gbm_device_destroy(vr->gbm);

	weston_debug_scope_destroy(vr->debug);
	vr->debug = NULL;

	nvcomposer_close(vr->composer);

	free(vr);

	ec->renderer = NULL;
}

static void
hal_renderer_surface_get_content_size(struct weston_surface *surface,
				      int *width,
				      int *height)
{
	struct hal_surface_state *vs = get_surface_state(surface);

	*width = vs->pitch;
	*height = vs->height;
}

static int
gbm_bo_to_dmabuf_attributes(const struct hal_renderer *vr,
			    struct gbm_bo *gbm_bo,
			    struct dmabuf_attributes *attribs)
{
	int ii;
	int ret = 0;

	memset(attribs, 0, sizeof(*attribs));
	attribs->width = gbm_bo_get_width(gbm_bo);
	attribs->height = gbm_bo_get_height(gbm_bo);
	attribs->format = gbm_bo_get_format(gbm_bo);
	attribs->n_planes = gbm_bo_get_plane_count(gbm_bo);

	for (ii = 0; ii < attribs->n_planes; ii++) {
		union gbm_bo_handle handle =
			gbm_bo_get_handle_for_plane(gbm_bo, ii);
		if (handle.s32 != -1) {
			ret = drmPrimeHandleToFD(vr->drm_fd,
						 handle.u32,
						 DRM_RDWR,
						 &attribs->fd[ii]);

			if (ret) {
				weston_log("Error: %s: drmPrimeHandleFD "
					   "failed, errno=%d\n",
					   __func__,
					   ret);
				goto fail;
			}
		} else {
			ret = -1;
			weston_log("Error: %s: gbm_bo_get_handle_for_plane "
				   "failed\n",
				   __func__);
			goto fail;
		}
		attribs->offset[ii] = gbm_bo_get_offset(gbm_bo, ii);
		attribs->stride[ii] = gbm_bo_get_stride_for_plane(gbm_bo, ii);
		attribs->modifier[ii] = gbm_bo_get_modifier(gbm_bo);
	}

	return ret;

fail:
	memset(attribs, 0, sizeof(*attribs));
	return ret;
}

WL_EXPORT int
hal_renderer_output_set_buffer(struct weston_output *output, int idx)
{
	struct hal_output_state *vo = get_output_state(output);

	if (idx >= ARRAY_LENGTH(vo->hw_buffer)) {
		weston_log("Error: Invalid HAL output buffer index (%d)\n", idx);
		assert(idx < ARRAY_LENGTH(vo->hw_buffer));
		return -1;
	}

	if (vo->hw_buffer[idx] == NULL) {
		weston_log("Error: Missing a HAL output buffer (idx=%d)\n", idx);
		assert(vo->hw_buffer[idx]);
		return -1;
	}

	vo->current_buffer = idx;
	return 0;
}

static int
hal_renderer_output_create(struct weston_output *output,
			   struct gbm_bo **gbm_bo,
			   int num_bo)
{
	struct weston_compositor *ec = output->compositor;
	struct hal_renderer *vr = get_renderer(ec);
	struct hal_output_state *vo = NULL;

	vo = zalloc(sizeof *vo);
	if (vo == NULL) {
		return -1;
	}
	assert(num_bo <= ARRAY_LENGTH(vo->hw_buffer));

	for (int ii = 0; ii < ARRAY_LENGTH(vo->hw_buffer); ii++) {
		vo->hw_buffer[ii] = NULL;
	}

	vo->current_buffer = 0;
	vo->end_render_fence_fd = -1;
	vo->hw_extra_damage = NULL;
	vo->buffer_damage = NULL;

	for (int ii = 0; ii < num_bo; ii++) {
		struct dmabuf_attributes dmabuf;

		if (gbm_bo_to_dmabuf_attributes(vr, gbm_bo[ii], &dmabuf)) {
			weston_log("Error: %s: gbm_bo_to_dmabuf_attributes "
				   "failed\n",
				   __func__);
			goto fail;
		}

		vo->hw_buffer[ii] = hal_image_create(vr, &dmabuf);

		if (!vo->hw_buffer[ii]) {
			weston_log("Error: %s: hal_image_create failed\n",
				   __func__);
			goto fail;
		}
	}

	output->renderer_state = vo;

	return 0;

fail:
	for (int ii = 0; ii < num_bo; ii++) {
		if (vo->hw_buffer[ii]) {
			hal_image_unref(vo->hw_buffer[ii]);
			vo->hw_buffer[ii] = NULL;
		}
	}

	if (vo) {
		free(vo);
	}

	output->renderer_state = NULL;
	return -1;
}

static int
hal_renderer_add_vpr_buffers(struct weston_output *output,
			   struct gbm_bo **gbm_bo)
{
	assert(output);
	assert(output->renderer_state);
	struct hal_output_state *vo = output->renderer_state;
	struct weston_compositor *ec = output->compositor;
	struct hal_renderer *vr = get_renderer(ec);

	int vpr_end = ARRAY_LENGTH(vo->hw_buffer);
	int vpr_start = vpr_end / 2;

	for (int ii = vpr_start; ii < vpr_end; ii++) {
		struct dmabuf_attributes dmabuf;

		assert(vo->hw_buffer[ii] == NULL);

		if (gbm_bo_to_dmabuf_attributes(vr, gbm_bo[ii], &dmabuf)) {
			weston_log("Error: %s: gbm_bo_to_dmabuf_attributes "
				   "failed\n",
				   __func__);
			goto fail;
		}

		vo->hw_buffer[ii] = hal_image_create(vr, &dmabuf);

		if (!vo->hw_buffer[ii]) {
			weston_log("Error: %s: hal_image_create failed\n",
				   __func__);
			goto fail;
		}
	}
	return 0;
fail:
	for (int ii = vpr_start; ii < vpr_end; ii++) {
		if (vo->hw_buffer[ii]) {
			hal_image_unref(vo->hw_buffer[ii]);
			vo->hw_buffer[ii] = NULL;
		}
	}

	return -1;
}

static bool
hal_renderer_has_vpr_buffers(struct weston_output *output) {
	assert(output);
	assert(output->renderer_state);
	struct hal_output_state *vo = output->renderer_state;
	int vpr_end = ARRAY_LENGTH(vo->hw_buffer);
	int vpr_start = vpr_end / 2;

	for (int ii = vpr_start; ii < vpr_end; ii++) {
		if (!vo->hw_buffer[ii]) {
			return false;
		}
	}
	return true;
}

static void
hal_renderer_output_destroy(struct weston_output *output)
{
	struct hal_output_state *vo = get_output_state(output);

	fd_clear(&vo->end_render_fence_fd);

	for (int ii = 0; ii < ARRAY_LENGTH(vo->hw_buffer); ii++) {
		if (vo->hw_buffer[ii]) {
			hal_image_unref(vo->hw_buffer[ii]);
			vo->hw_buffer[ii] = NULL;
		}
	}

	if (vo->buffer_damage) {
		for (int i = 0; i < 2; i++) {
			pixman_region32_fini(&vo->buffer_damage[i]);
		}
	}

	free(vo);
}

static int
hal_renderer_create_fence_fd(struct weston_output *output)
{
	const struct hal_output_state *vo = get_output_state(output);

	if (vo->end_render_fence_fd < 0) {
		return -1;
	}

	return dup(vo->end_render_fence_fd);
}

static int
hal_renderer_get_fence_fd(struct weston_output *output)
{
	const struct hal_output_state *vo = get_output_state(output);
	return vo->end_render_fence_fd;
}

void
hal_renderer_prepare_read_pixels(struct weston_output *output)
{
	struct hal_output_state *vo = get_output_state(output);
	vo->do_prepare_read_pixels = true;
}

static int
hal_renderer_read_pixels(struct weston_output *output,
			pixman_format_code_t format, void *pixels,
			uint32_t x, uint32_t y,
			uint32_t width, uint32_t height)
{
	GLenum gl_format;
	struct hal_output_state *vo = get_output_state(output);
	struct hal_renderer *vr =
		(struct hal_renderer *)output->compositor->renderer;

	switch (format) {
	case PIXMAN_a8r8g8b8:
		gl_format = GL_BGRA_EXT;
		break;
	case PIXMAN_a8b8g8r8:
		gl_format = GL_RGBA;
		break;
	default:
		return -1;
	}

	if (!nvcomposer_read_pixels(vr->composer, gl_format, pixels,
			x, y, width, height)) {
		weston_log("nvcomposer_read_pixels failed\n");
		return -1;
	}

	return 0;
}

static int
hal_renderer_create(struct weston_compositor *compositor, int drm_fd)
{
	struct hal_renderer *vr;
	vr = zalloc(sizeof *vr);
	if (vr == NULL) {
		return -1;
	}

	struct nvcomposer *(*nvsupercomposer_create)(void);
	// TODO HAL: It's ugly to harcode the path
	nvsupercomposer_create = weston_load_module(
		"/usr/lib/aarch64-linux-gnu/tegra/libnvsupercomposer.so",
		"nvsupercomposer_create");
	vr->composer = nvsupercomposer_create();

	vr->base.repaint_output = hal_renderer_repaint_output;
	vr->base.flush_damage = hal_renderer_flush_damage;
	vr->base.attach = hal_renderer_attach;
	vr->base.surface_set_color = hal_renderer_surface_set_color;
	vr->base.destroy = hal_renderer_destroy;
	vr->base.surface_get_content_size =
		hal_renderer_surface_get_content_size;
	vr->base.attach_eglstream_consumer = vic_renderer_attach_eglstream_consumer;
	vr->base.commit_eglstream_consumer_image = hal_renderer_commit_eglstream_consumer_image;
	vr->base.release_eglstream_consumer_image = hal_renderer_release_eglstream_consumer_image;
	vr->base.prepare_read_pixels = hal_renderer_prepare_read_pixels;
	vr->base.read_pixels = hal_renderer_read_pixels;

	compositor->renderer = &vr->base;

	compositor->capabilities |= WESTON_CAP_EXPLICIT_SYNC;
	// Reset the YFLIP capabilities if already set
	compositor->capabilities &= (~WESTON_CAP_CAPTURE_YFLIP);

	wl_display_add_shm_format(compositor->wl_display, WL_SHM_FORMAT_RGB565);
	wl_display_add_shm_format(compositor->wl_display, WL_SHM_FORMAT_YUV420);
	wl_display_add_shm_format(compositor->wl_display, WL_SHM_FORMAT_NV12);
	wl_display_add_shm_format(compositor->wl_display, WL_SHM_FORMAT_YUYV);
	wl_display_add_shm_format(compositor->wl_display, WL_SHM_FORMAT_NV16);
	wl_display_add_shm_format(compositor->wl_display, WL_SHM_FORMAT_NV24);

	vr->has_dmabuf_import = 1;
	vr->has_dmabuf_import_modifiers = 1;

	wl_list_init(&vr->dmabuf_images);
	if (vr->has_dmabuf_import) {
		vr->base.import_dmabuf = hal_renderer_import_dmabuf;
		vr->base.query_dmabuf_formats =
			hal_renderer_query_dmabuf_formats;
		vr->base.query_dmabuf_modifiers =
			hal_renderer_query_dmabuf_modifiers;
	}

	if (drm_fd == -1) {
		vr->drm_fd = open("/dev/dri/card0", O_RDWR);
		if (vr->drm_fd == -1) {
			weston_log("Error: failed to open the drm fd");
			return -1;
		}
		vr->own_drm_fd = true;
	}
	else {
		vr->drm_fd = drm_fd;
		vr->own_drm_fd = false;
	}

	vr->gbm = gbm_create_device(vr->drm_fd);

	vr->debug = weston_compositor_add_debug_scope(compositor, "hal-renderer",
						      "Debug messages from HAL Renderer\n",
						      NULL, NULL);

	struct weston_config *config = wet_get_config(compositor);
	struct weston_config_section *section;
	int use_gamma_correction;
	section = weston_config_get_section(config, "hal-renderer", NULL, NULL);
	weston_config_section_get_bool(section, "gamma-correction",
	                               &use_gamma_correction, 1);
	vr->use_gamma_correction = (use_gamma_correction == 1) ? true : false;

	wl_signal_init(&vr->destroy_signal);

	return 0;
}

static bool hal_renderer_is_protected(struct weston_output *output)
{
	struct weston_compositor *ec = output->compositor;
	struct hal_renderer *vr = get_renderer(ec);
	bool is_protected = false;
	struct hal_surface_state *vs;
	struct weston_view *view;
	pixman_box32_t *bbox_global;

	wl_list_for_each_reverse(view, &ec->view_list, link) {
		if (view->plane == &ec->primary_plane) {
			// Transform the view bounding box from global to output coordinates:
			bbox_global = &view->transform.boundingbox.extents;
			vs = get_surface_state(view->surface);
			if ((pixman_region32_contains_rectangle(&output->region, bbox_global) != PIXMAN_REGION_OUT) &&
				nvcomposer_buffer_is_protected(vr->composer, vs->hal_image->nvc_buffer)) {
				is_protected = true;
				break;
			}
		}
	}
	return is_protected;
}

WL_EXPORT struct hal_renderer_interface hal_renderer_interface =
	{ .create = hal_renderer_create,
	  .output_set_buffer = hal_renderer_output_set_buffer,
	  .output_create = hal_renderer_output_create,
	  .output_add_vpr_buffers = hal_renderer_add_vpr_buffers,
	  .output_has_vpr_buffers = hal_renderer_has_vpr_buffers,
	  .output_destroy = hal_renderer_output_destroy,
	  .create_fence_fd = hal_renderer_create_fence_fd,
	  .get_fence_fd = hal_renderer_get_fence_fd,
	  .is_protected = hal_renderer_is_protected};
