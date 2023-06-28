/*
 * Copyright © 2011 Benjamin Franzke
 * Copyright © 2010 Intel Corporation
 * Copyright © 2014,2018 Collabora Ltd.
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

#include "weston-test-dmabuf-helper.h"
#include <shared/platform.h>

#include <fcntl.h>
#include <gbm.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

// WAR till GBM_BO_USE_PROTECTED gets upstreamed
#ifndef GBM_BO_USE_PROTECTED
#define GBM_BO_USE_PROTECTED (1 << 10)
#endif

#define BUFFER_FORMAT DRM_FORMAT_ARGB8888

#ifndef DRM_RDWR
#define DRM_RDWR O_RDWR
#endif

struct dma_buffer_allocator {
	struct zwp_linux_dmabuf_v1 *dmabuf;

	struct {
		int drm_fd;
		struct gbm_device *device;
	} gbm;

	uint64_t *modifiers;
	int modifiers_count;

	struct {
		EGLDisplay display;
		EGLContext context;
		bool has_dma_buf_import_modifiers;
		PFNEGLQUERYDMABUFMODIFIERSEXTPROC query_dma_buf_modifiers;
		PFNEGLCREATEIMAGEKHRPROC create_image;
		PFNEGLDESTROYIMAGEKHRPROC destroy_image;
		PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target_texture_2d;
		PFNEGLCREATESYNCKHRPROC create_sync;
		PFNEGLDESTROYSYNCKHRPROC destroy_sync;
		PFNEGLCLIENTWAITSYNCKHRPROC client_wait_sync;
		PFNEGLDUPNATIVEFENCEFDANDROIDPROC dup_native_fence_fd;
		PFNEGLWAITSYNCKHRPROC wait_sync;
		uint64_t *modifiers;
		int modifiers_count;
	} egl;
};

static EGLDeviceEXT chooseEGLDevice(char const *drm_render_node)
{
	int num_devices, i;
	const char *drm_device_file;
	PFNEGLQUERYDEVICESEXTPROC query_devices;
	PFNEGLQUERYDEVICESTRINGEXTPROC query_device_string;
	EGLDeviceEXT *egl_devs, egl_dev;

	query_devices = (void *) eglGetProcAddress("eglQueryDevicesEXT");
	query_device_string = (void *) eglGetProcAddress("eglQueryDeviceStringEXT");
	if (!query_devices || !query_device_string)
		goto error;

	if (!query_devices(0, NULL, &num_devices)) {
		fprintf(stderr, "failed to get max egl devices\n");
		goto error;
	}

	egl_devs = malloc(num_devices * sizeof(*egl_devs));
	if (!egl_devs)
		goto error;

	if (!query_devices(num_devices, egl_devs, &num_devices)) {
		fprintf(stderr, "failed to get egl devices\n");
		free(egl_devs);
		goto error;
	}

	for (i = 0; i < num_devices; i++) {
		drm_device_file = query_device_string(egl_devs[i],
				EGL_DRM_DEVICE_FILE_EXT);
		if (drm_device_file == NULL)
			continue;
		if (strcmp(drm_device_file, drm_render_node) == 0) {
			egl_dev = egl_devs[i];
			break;
		}
	}

	free(egl_devs);
	return egl_dev;

error:
	return EGL_NO_DEVICE_EXT;
}

static void
dmabuf_modifiers(void *data, struct zwp_linux_dmabuf_v1 *zwp_linux_dmabuf,
		 uint32_t format, uint32_t modifier_hi, uint32_t modifier_lo)
{
	int i;
	struct dma_buffer_allocator *allocator = data;
	bool supported = false;
	uint64_t modifier = ((uint64_t)modifier_hi << 32) | modifier_lo;

	//printf("Got modifier %u:%u for format %u\n", modifier_hi, modifier_lo, format);

	switch (format) {
	case BUFFER_FORMAT:
		/* Poor person's set intersection: d->modifiers INTERSECT
		 * egl_modifiers.  If a modifier is not supported, replace it with
		 * DRM_FORMAT_MOD_INVALID in the d->modifiers array.
		 */
		for (i = 0; i < allocator->egl.modifiers_count; ++i) {
			if (allocator->egl.modifiers[i] == modifier) {
				supported = true;
			}
		}
		if (!supported) {
			break;
		}

		++allocator->modifiers_count;
		allocator->modifiers =
			realloc(allocator->modifiers,
				allocator->modifiers_count *
					sizeof(*allocator->modifiers));
		allocator->modifiers[allocator->modifiers_count - 1] =
			modifier;
		break;
	default:
		break;
	}
}

static void
dmabuf_format(void *data, struct zwp_linux_dmabuf_v1 *zwp_linux_dmabuf, uint32_t format)
{
	/* XXX: deprecated */
}

static const struct zwp_linux_dmabuf_v1_listener dmabuf_listener = {
	dmabuf_format,
	dmabuf_modifiers
};


static bool
get_supported_modifiers_for_egl(struct dma_buffer_allocator *allocator, uint64_t **egl_modifiers, int *num_egl_modifiers)
{
	EGLBoolean ret;
	int i;
	*egl_modifiers = NULL;
	*num_egl_modifiers = 0;

	/* If EGL doesn't support modifiers, don't use them at all. */
	if (!allocator->egl.has_dma_buf_import_modifiers) {
		allocator->modifiers_count = 0;
		free(allocator->modifiers);
		allocator->modifiers = NULL;
		return true;
	}

	ret = allocator->egl.query_dma_buf_modifiers(allocator->egl.display,
				             BUFFER_FORMAT,
				             0,    /* max_modifiers */
				             NULL, /* modifiers */
				             NULL, /* external_only */
				             num_egl_modifiers);
	if (ret == EGL_FALSE || *num_egl_modifiers == 0) {
		fprintf(stderr, "Failed to query num EGL modifiers for format\n");
		goto error;
	}

	*egl_modifiers = malloc(*num_egl_modifiers * sizeof(*egl_modifiers));

	ret = allocator->egl.query_dma_buf_modifiers(allocator->egl.display,
					     BUFFER_FORMAT,
					     *num_egl_modifiers,
					     *egl_modifiers,
					     NULL, /* external_only */
					     num_egl_modifiers);
	if (ret == EGL_FALSE) {
		fprintf(stderr, "Failed to query EGL modifiers for format\n");
		goto error;
	}

	return true;

error:
	if (*egl_modifiers) {
		free(*egl_modifiers);
	}

	return false;
}

struct dma_buffer_allocator *
create_dma_buffer_allocator(struct zwp_linux_dmabuf_v1 *dmabuf,
			    char const *drm_render_node,
			    bool protected_ctx)
{
	struct dma_buffer_allocator *allocator;
	const char *egl_extensions = NULL;
	const char *gl_extensions = NULL;
	EGLint major, minor;
	EGLDeviceEXT egl_dev;
	EGLint device_platform_attribs[] = {
		EGL_DRM_MASTER_FD_EXT, -1,
		EGL_NONE
	};
	static EGLint context_attribs[] = {
		EGL_CONTEXT_CLIENT_VERSION, 2,
		EGL_PROTECTED_CONTENT_EXT, EGL_FALSE,
		EGL_NONE
	};

	if (protected_ctx) {
		assert(context_attribs[2] == EGL_PROTECTED_CONTENT_EXT);
		context_attribs[3] = EGL_TRUE;
	}

	egl_dev = chooseEGLDevice(drm_render_node);
	if (egl_dev == EGL_NO_DEVICE_EXT) {
		fprintf(stderr, "failed to query DRM device name\n");
		goto error;
	}

	allocator = calloc(1, sizeof(*allocator));
	if (!allocator) {
		fprintf(stderr, "malloc failed\n");
		goto error;
	}

	memset(allocator, 0, sizeof(*allocator));

	allocator->dmabuf = dmabuf;

	allocator->gbm.drm_fd = open(drm_render_node, O_RDWR);
	if (allocator->gbm.drm_fd < 0) {
		fprintf(stderr, "Failed to open drm render node %s\n",
			drm_render_node);
		goto error;
	}

	allocator->gbm.device = gbm_create_device(allocator->gbm.drm_fd);
	if (allocator->gbm.device == NULL) {
		fprintf(stderr, "Failed to create gbm device\n");
		goto error;
	}

	assert(device_platform_attribs[0] == EGL_DRM_MASTER_FD_EXT);
	device_platform_attribs[1] = allocator->gbm.drm_fd;
	allocator->egl.display = weston_platform_get_egl_display(EGL_PLATFORM_DEVICE_EXT,
			(void*)egl_dev, device_platform_attribs);

	if (eglInitialize(allocator->egl.display, &major, &minor) == EGL_FALSE) {
		fprintf(stderr, "Failed to initialize EGLDisplay\n");
		goto error;
	}

	if (eglBindAPI(EGL_OPENGL_ES_API) == EGL_FALSE) {
		fprintf(stderr, "Failed to bind OpenGL ES API\n");
		goto error;
	}

	egl_extensions = eglQueryString(allocator->egl.display, EGL_EXTENSIONS);
	assert(egl_extensions != NULL);

	if (!weston_check_egl_extension(egl_extensions,
					"EGL_EXT_image_dma_buf_import")) {
		fprintf(stderr, "EGL_EXT_image_dma_buf_import not supported\n");
		goto error;
	}

	allocator->egl.context = eglCreateContext(allocator->egl.display,
						EGL_NO_CONFIG_KHR,
						EGL_NO_CONTEXT,
						context_attribs);
	if (allocator->egl.context == EGL_NO_CONTEXT) {
		fprintf(stderr, "Failed to create EGLContext\n");
		goto error;
	}

	if (eglMakeCurrent(allocator->egl.display,
			   EGL_NO_SURFACE,
			   EGL_NO_SURFACE,
			   allocator->egl.context) != EGL_TRUE) {
		fprintf(stderr, "Failed eglMakeCurrent\n");
		goto error;
	}

	gl_extensions = (const char *)glGetString(GL_EXTENSIONS);
	assert(gl_extensions != NULL);

	if (!weston_check_egl_extension(gl_extensions,
					"GL_OES_EGL_image")) {
		fprintf(stderr, "GL_OES_EGL_image not supported\n");
		goto error;
	}

	if (weston_check_egl_extension(egl_extensions,
				       "EGL_EXT_image_dma_buf_import_modifiers")) {
		allocator->egl.has_dma_buf_import_modifiers = true;
		allocator->egl.query_dma_buf_modifiers =
			(void *) eglGetProcAddress("eglQueryDmaBufModifiersEXT");
		assert(allocator->egl.query_dma_buf_modifiers);
	}

	allocator->egl.create_image =
		(void *) eglGetProcAddress("eglCreateImageKHR");
	assert(allocator->egl.create_image);

	allocator->egl.destroy_image =
		(void *) eglGetProcAddress("eglDestroyImageKHR");
	assert(allocator->egl.destroy_image);

	allocator->egl.image_target_texture_2d =
		(void *) eglGetProcAddress("glEGLImageTargetTexture2DOES");
	assert(allocator->egl.image_target_texture_2d);

	if (!get_supported_modifiers_for_egl(allocator,
					     &allocator->egl.modifiers,
					     &allocator->egl.modifiers_count)) {
		fprintf(stderr, "failed to query supported egl modifiers\n");
		goto error;
	}

	zwp_linux_dmabuf_v1_add_listener(allocator->dmabuf, &dmabuf_listener, allocator);

	return allocator;
error:
	if (allocator) {
		free_dma_buffer_allocator(allocator);
	}
	return NULL;
}

void free_dma_buffer_allocator(struct dma_buffer_allocator *allocator)
{
	if (allocator->gbm.device)
		gbm_device_destroy(allocator->gbm.device);

	if (allocator->gbm.drm_fd >= 0)
		close(allocator->gbm.drm_fd);

	if (allocator->egl.context != EGL_NO_CONTEXT)
		eglDestroyContext(allocator->egl.display, allocator->egl.context);

	if (allocator->egl.display != EGL_NO_DISPLAY)
		eglTerminate(allocator->egl.display);

	free(allocator);
}

static void
buffer_release(void *data, struct wl_buffer *buffer)
{
	struct dma_buffer *mybuf = data;
	mybuf->busy = 0;
}

static const struct wl_buffer_listener buffer_listener = {
	buffer_release
};

static void
create_succeeded(void *data,
		 struct zwp_linux_buffer_params_v1 *params,
		 struct wl_buffer *new_buffer)
{
	struct dma_buffer *dmabuf = data;

	dmabuf->buffer = new_buffer;
	/* When not using explicit synchronization listen to wl_buffer.release
	 * for release notifications, otherwise we are going to use
	 * zwp_linux_buffer_release_v1. */
	if (!dmabuf->use_explicit_sync) {
		wl_buffer_add_listener(dmabuf->buffer, &buffer_listener,
				       dmabuf);
	}

	zwp_linux_buffer_params_v1_destroy(params);
}

static void
create_failed(void *data, struct zwp_linux_buffer_params_v1 *params)
{
	struct dma_buffer *dmabuf = data;

	dmabuf->buffer = NULL;

	zwp_linux_buffer_params_v1_destroy(params);

	fprintf(stderr, "Error: zwp_linux_buffer_params.create failed.\n");
}

static const struct zwp_linux_buffer_params_v1_listener params_listener = {
	create_succeeded,
	create_failed
};

int
create_dmabuf_buffer(struct dma_buffer_allocator *allocator,
		     struct dma_buffer *buffer,
		     int width,
		     int height,
		     bool req_dmabuf_immediate,
		     bool alloc_protected,
		     bool use_explicit_sync)
{
	/* TODO VIC: We'd like to Y-invert the buffer image, since we are
	 * going to renderer to the buffer through an FBO. However, VIC
	 * requires all buffers to have the same Y-flip. This is
	 * unfeasible, if there are shm-buffers. So we disable the flag
	 * until vic-renderer supports multi-pass composition. */
	// static const uint32_t flags = ZWP_LINUX_BUFFER_PARAMS_V1_FLAGS_Y_INVERT;
	static const uint32_t flags = 0;

	struct zwp_linux_buffer_params_v1 *params;
	int i;
	int use_flags = GBM_BO_USE_RENDERING;

	memset(buffer, 0, sizeof(*buffer));

	buffer->use_explicit_sync = use_explicit_sync;
	buffer->allocator = allocator;
	buffer->width = width;
	buffer->height = height;
	buffer->format = BUFFER_FORMAT;
	buffer->release_fence_fd = -1;

	if (!allocator->modifiers_count) {
		printf("dmabuf_allocator has no format modifiers\n");
	}

#ifdef HAVE_GBM_MODIFIERS
	if (allocator->modifiers_count > 0 && !alloc_protected) {
		buffer->bo = gbm_bo_create_with_modifiers(allocator->gbm.device,
							  buffer->width,
							  buffer->height,
							  buffer->format,
							  allocator->modifiers,
							  allocator->modifiers_count);
		if (buffer->bo)
			buffer->modifier = gbm_bo_get_modifier(buffer->bo);
	}
#endif

	if (!buffer->bo) {
	    if (alloc_protected)
		use_flags |= GBM_BO_USE_PROTECTED;

		buffer->bo = gbm_bo_create(allocator->gbm.device,
					   buffer->width,
					   buffer->height,
					   buffer->format,
					   use_flags);
		buffer->modifier = gbm_bo_get_modifier(buffer->bo);
	}

	if (!buffer->bo) {
		fprintf(stderr, "create_bo failed\n");
		goto error;
	}

#ifdef HAVE_GBM_MODIFIERS
	buffer->plane_count = gbm_bo_get_plane_count(buffer->bo);
	for (i = 0; i < buffer->plane_count; ++i) {
		int ret;
		union gbm_bo_handle handle;

		handle = gbm_bo_get_handle_for_plane(buffer->bo, i);
		if (handle.s32 == -1) {
			fprintf(stderr, "error: failed to get gbm_bo_handle\n");
			goto error;
		}

		ret = drmPrimeHandleToFD(allocator->gbm.drm_fd, handle.u32, DRM_RDWR,
					 &buffer->dmabuf_fds[i]);
		if (ret < 0 || buffer->dmabuf_fds[i] < 0) {
			fprintf(stderr, "error: failed to get dmabuf_fd\n");
			goto error;
		}
		buffer->strides[i] = gbm_bo_get_stride_for_plane(buffer->bo, i);
		buffer->offsets[i] = gbm_bo_get_offset(buffer->bo, i);
	}
#else
	buffer->plane_count = 1;
	buffer->strides[0] = gbm_bo_get_stride(buffer->bo);
	buffer->dmabuf_fds[0] = gbm_bo_get_fd(buffer->bo);
	if (buffer->dmabuf_fds[0] < 0) {
		fprintf(stderr, "error: failed to get dmabuf_fd\n");
		goto error;
	}
#endif

	params = zwp_linux_dmabuf_v1_create_params(allocator->dmabuf);
	for (i = 0; i < buffer->plane_count; ++i) {
		zwp_linux_buffer_params_v1_add(params,
					       buffer->dmabuf_fds[i],
					       i,
					       buffer->offsets[i],
					       buffer->strides[i],
					       buffer->modifier >> 32,
					       buffer->modifier & 0xffffffff);
	}

	zwp_linux_buffer_params_v1_add_listener(params, &params_listener, buffer);
	if (req_dmabuf_immediate) {
		buffer->buffer =
			zwp_linux_buffer_params_v1_create_immed(params,
								buffer->width,
								buffer->height,
								buffer->format,
								flags);
		/* When not using explicit synchronization listen to
		 * wl_buffer.release for release notifications, otherwise we
		 * are going to use zwp_linux_buffer_release_v1. */
		if (!buffer->use_explicit_sync) {
			wl_buffer_add_listener(buffer->buffer,
					       &buffer_listener,
					       buffer);
		}
	}
	else {
		zwp_linux_buffer_params_v1_create(params,
						  buffer->width,
						  buffer->height,
						  buffer->format,
						  flags);
	}

	return 0;

error:
	assert(buffer->allocator);
	free_dmabuf_buffer(buffer);
	return -1;
}

void
free_dmabuf_buffer(struct dma_buffer *buffer)
{
	int i;
	struct dma_buffer_allocator *allocator = buffer->allocator;

	if (buffer->release_fence_fd >= 0)
		close(buffer->release_fence_fd);

	if (buffer->buffer_release)
		zwp_linux_buffer_release_v1_destroy(buffer->buffer_release);

	if (buffer->gl_fbo)
		glDeleteFramebuffers(1, &buffer->gl_fbo);

	if (buffer->gl_texture)
		glDeleteTextures(1, &buffer->gl_texture);

	if (buffer->egl_image) {
		allocator->egl.destroy_image(allocator->egl.display,
						   buffer->egl_image);
	}

	if (buffer->buffer)
		wl_buffer_destroy(buffer->buffer);

	if (buffer->bo)
		gbm_bo_destroy(buffer->bo);

	for (i = 0; i < buffer->plane_count; ++i) {
		if (buffer->dmabuf_fds[i] >= 0)
			close(buffer->dmabuf_fds[i]);
	}
}

int
create_fbo_for_buffer(struct dma_buffer_allocator *allocator,
		      struct dma_buffer *buffer,
		      bool protected)
{
	static const int general_attribs = 4;
	static const int plane_attribs = 5;
	static const int entries_per_attrib = 2;
	EGLint attribs[(general_attribs + plane_attribs * MAX_BUFFER_PLANES) *
			entries_per_attrib + 1];
	unsigned int atti = 0;

	attribs[atti++] = EGL_WIDTH;
	attribs[atti++] = buffer->width;
	attribs[atti++] = EGL_HEIGHT;
	attribs[atti++] = buffer->height;
	attribs[atti++] = EGL_LINUX_DRM_FOURCC_EXT;
	attribs[atti++] = buffer->format;
	attribs[atti++] = EGL_PROTECTED_CONTENT_EXT;
	attribs[atti++] = protected;

#define ADD_PLANE_ATTRIBS(plane_idx) { \
	attribs[atti++] = EGL_DMA_BUF_PLANE ## plane_idx ## _FD_EXT; \
	attribs[atti++] = buffer->dmabuf_fds[plane_idx]; \
	attribs[atti++] = EGL_DMA_BUF_PLANE ## plane_idx ## _OFFSET_EXT; \
	attribs[atti++] = (int) buffer->offsets[plane_idx]; \
	attribs[atti++] = EGL_DMA_BUF_PLANE ## plane_idx ## _PITCH_EXT; \
	attribs[atti++] = (int) buffer->strides[plane_idx]; \
	if (allocator->egl.has_dma_buf_import_modifiers) { \
		attribs[atti++] = EGL_DMA_BUF_PLANE ## plane_idx ## _MODIFIER_LO_EXT; \
		attribs[atti++] = buffer->modifier & 0xFFFFFFFF; \
		attribs[atti++] = EGL_DMA_BUF_PLANE ## plane_idx ## _MODIFIER_HI_EXT; \
		attribs[atti++] = buffer->modifier >> 32; \
	} \
	}

	if (buffer->plane_count > 0)
		ADD_PLANE_ATTRIBS(0);

	if (buffer->plane_count > 1)
		ADD_PLANE_ATTRIBS(1);

	if (buffer->plane_count > 2)
		ADD_PLANE_ATTRIBS(2);

	if (buffer->plane_count > 3)
		ADD_PLANE_ATTRIBS(3);

#undef ADD_PLANE_ATTRIBS

	attribs[atti] = EGL_NONE;

	assert(atti < ARRAY_LENGTH(attribs));

	buffer->egl_image = allocator->egl.create_image(allocator->egl.display,
						      EGL_NO_CONTEXT,
						      EGL_LINUX_DMA_BUF_EXT,
						      NULL, attribs);
	if (buffer->egl_image == EGL_NO_IMAGE_KHR) {
		fprintf(stderr, "EGLImageKHR creation failed\n");
		return -1;
	}

	eglMakeCurrent(allocator->egl.display, EGL_NO_SURFACE, EGL_NO_SURFACE,
			allocator->egl.context);

	glGenTextures(1, &buffer->gl_texture);
	glBindTexture(GL_TEXTURE_2D, buffer->gl_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	if (protected) {
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_PROTECTED_EXT, GL_TRUE);
	}

	allocator->egl.image_target_texture_2d(GL_TEXTURE_2D, buffer->egl_image);

	glGenFramebuffers(1, &buffer->gl_fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, buffer->gl_fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			       GL_TEXTURE_2D, buffer->gl_texture, 0);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		fprintf(stderr, "FBO creation failed\n");
		return -1;
	}

	return 0;
}

void fill_dma_buffer_solid(struct dma_buffer *buffer, pixman_color_t color)
{
	const float full = (float)((1 << 16) - 1);
	glBindFramebuffer(GL_FRAMEBUFFER, buffer->gl_fbo);
	glClearColor(color.red / full, color.green / full, color.blue / full, color.alpha / full);
	glClear(GL_COLOR_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glFinish();
}
