/*
 * Copyright © 2012 John Kåre Alsaker
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

#include "config.h"

#include <stdint.h>

#include "compositor.h"

#ifdef ENABLE_EGL

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#else

typedef int EGLint;
typedef int EGLenum;
typedef void *EGLDeviceEXT;
typedef void *EGLDisplay;
typedef void *EGLSurface;
typedef void *EGLConfig;
typedef intptr_t EGLNativeDisplayType;
typedef intptr_t EGLNativeWindowType;
#define EGL_DEFAULT_DISPLAY ((EGLNativeDisplayType)0)
#define EGL_NO_DEVICE_EXT   ((EGLDeviceEXT)0)

#endif /* ENABLE_EGL */

#define NO_EGL_PLATFORM 0

enum gl_renderer_border_side {
	GL_RENDERER_BORDER_TOP = 0,
	GL_RENDERER_BORDER_LEFT = 1,
	GL_RENDERER_BORDER_RIGHT = 2,
	GL_RENDERER_BORDER_BOTTOM = 3,
};

struct gl_renderer_image {
	unsigned int gl_texture;
	unsigned int gl_fbo;
};

struct gl_shader {
	GLuint program;
	GLuint vertex_shader, fragment_shader;
	GLint proj_uniform;
	GLint tex_uniforms[3];
	GLint alpha_uniform;
	GLint color_uniform;
	const char *vertex_source, *fragment_source;
};

struct gl_renderer {
	struct weston_renderer base;
	int fragment_shader_debug;
	int fan_debug;
	struct weston_binding *fragment_binding;
	struct weston_binding *fan_binding;

	EGLDisplay egl_display;
	EGLContext egl_context;
	EGLConfig egl_config;

	EGLSurface dummy_surface;

	uint32_t gl_version;

	struct wl_array vertices;
	struct wl_array vtxcnt;

	PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target_texture_2d;
	PFNEGLCREATEIMAGEKHRPROC create_image;
	PFNEGLDESTROYIMAGEKHRPROC destroy_image;
	PFNEGLSWAPBUFFERSWITHDAMAGEEXTPROC swap_buffers_with_damage;
	PFNEGLCREATEPLATFORMWINDOWSURFACEEXTPROC create_platform_window;

	int has_unpack_subimage;

	PFNEGLBINDWAYLANDDISPLAYWL bind_display;
	PFNEGLUNBINDWAYLANDDISPLAYWL unbind_display;
	PFNEGLQUERYWAYLANDBUFFERWL query_buffer;
	int has_bind_display;

	int has_context_priority;

	int has_egl_image_external;

	int has_egl_buffer_age;

	int has_configless_context;

	int has_surfaceless_context;

	PFNEGLGETOUTPUTLAYERSEXTPROC get_output_layers;
	PFNEGLQUERYOUTPUTLAYERATTRIBEXTPROC query_output_layer_attrib;
	int has_egl_output_base;
	int has_egl_output_drm;
	int has_egl_output_drm_flip_event;

	PFNEGLCREATESTREAMKHRPROC create_stream;
	PFNEGLDESTROYSTREAMKHRPROC destroy_stream;
	PFNEGLQUERYSTREAMKHRPROC query_stream;
	int has_egl_stream;

	PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC create_stream_producer_surface;
	int has_egl_stream_producer_eglsurface;

	PFNEGLSTREAMCONSUMEROUTPUTEXTPROC stream_consumer_output;
	int has_egl_stream_consumer_egloutput;

#ifdef EGL_NV_stream_attrib
	PFNEGLCREATESTREAMATTRIBNVPROC create_stream_attrib;
	PFNEGLSTREAMCONSUMERACQUIREATTRIBNVPROC stream_consumer_acquire_attrib;
#endif

#ifdef EGL_NV_stream_consumer_eglimage
	PFNEGLQUERYSTREAMCONSUMEREVENTNVPROC   stream_query_consumer_event;
	PFNEGLSTREAMIMAGECONSUMERCONNECTNVPROC stream_consumer_connect;
	PFNEGLSTREAMACQUIREIMAGENVPROC         stream_acquire_image;
	PFNEGLSTREAMRELEASEIMAGENVPROC         stream_release_image;
	PFNEGLEXPORTDMABUFIMAGEMESAPROC        export_dmabuf_image;
	PFNEGLEXPORTDMABUFIMAGEQUERYMESAPROC   export_dmabuf_image_query;
#endif

	bool use_eglimage_consumer;
	int has_egl_stream_attrib;
	int has_egl_stream_acquire_mode;

	int has_egl_stream_fifo;
	int has_egl_stream_fifo_synchronous;

	PFNEGLSTREAMCONSUMERGLTEXTUREEXTERNALKHRPROC stream_consumer_gltexture;
	int has_egl_stream_consumer_gltexture;
	int has_egl_wayland_eglstream;

	int has_dmabuf_import;
	struct wl_list dmabuf_images;

	int has_gl_texture_rg;

	struct gl_shader texture_shader_rgba;
	struct gl_shader texture_shader_rgbx;
	struct gl_shader texture_shader_egl_external;
	struct gl_shader texture_shader_y_uv;
	struct gl_shader texture_shader_y_u_v;
	struct gl_shader texture_shader_y_xuxv;
	struct gl_shader invert_color_shader;
	struct gl_shader solid_shader;
	struct gl_shader *current_shader;

	struct wl_signal destroy_signal;

	struct wl_listener output_destroy_listener;

	int has_dmabuf_import_modifiers;
	PFNEGLQUERYDMABUFFORMATSEXTPROC query_dmabuf_formats;
	PFNEGLQUERYDMABUFMODIFIERSEXTPROC query_dmabuf_modifiers;

	int has_native_fence_sync;
	PFNEGLCREATESYNCKHRPROC create_sync;
	PFNEGLDESTROYSYNCKHRPROC destroy_sync;
	PFNEGLDUPNATIVEFENCEFDANDROIDPROC dup_native_fence_fd;

	int has_wait_sync;
	PFNEGLWAITSYNCKHRPROC wait_sync;

	// Helper for use inside the vic renderer.
	EGLint (*egl_get_error_func)(void);
};

enum timeline_render_point_type {
	TIMELINE_RENDER_POINT_TYPE_BEGIN,
	TIMELINE_RENDER_POINT_TYPE_END
};

struct gl_renderer_interface {
	const EGLint *opaque_attribs;
	const EGLint *alpha_attribs;
	const EGLint *alpha_stream_attribs;
	const EGLint *surfaceless_attribs;

	int (*display_create)(struct weston_compositor *ec,
			      EGLenum platform,
			      void *native_window,
			      const EGLint *platform_attribs,
			      const EGLint *config_attribs,
			      const EGLint *visual_id,
			      const int n_ids);

	EGLDisplay (*display)(struct weston_compositor *ec);

	int (*output_window_create)(struct weston_output *output,
				    EGLNativeWindowType window_for_legacy,
				    void *window_for_platform,
				    const EGLint *config_attribs,
				    const EGLint *visual_id,
				    const int n_ids);

	int (*output_surfaceless_create)(struct weston_output *output, const EGLint *config_attribs);

	void (*output_destroy)(struct weston_output *output);

	EGLSurface (*output_surface)(struct weston_output *output);

	/* Sets the output border.
	 *
	 * The side specifies the side for which we are setting the border.
	 * The width and height are the width and height of the border.
	 * The tex_width patemeter specifies the width of the actual
	 * texture; this may be larger than width if the data is not
	 * tightly packed.
	 *
	 * The top and bottom textures will extend over the sides to the
	 * full width of the bordered window.  The right and left edges,
	 * however, will extend only to the top and bottom of the
	 * compositor surface.  This is demonstrated by the picture below:
	 *
	 * +-----------------------+
	 * |          TOP          |
	 * +-+-------------------+-+
	 * | |                   | |
	 * |L|                   |R|
	 * |E|                   |I|
	 * |F|                   |G|
	 * |T|                   |H|
	 * | |                   |T|
	 * | |                   | |
	 * +-+-------------------+-+
	 * |        BOTTOM         |
	 * +-----------------------+
	 */
	void (*output_set_border)(struct weston_output *output,
				  enum gl_renderer_border_side side,
				  int32_t width, int32_t height,
				  int32_t tex_width, unsigned char *data);

	/* Create fence sync FD to wait for GPU rendering.
	 *
	 * Return FD on success, -1 on failure or unsupported
	 * EGL_ANDROID_native_fence_sync extension.
	 */
	int (*create_fence_fd)(struct weston_output *output);

	void (*print_egl_error_state)(void);

	int (*get_devices)(EGLint max_devices,
			   EGLDeviceEXT *devices,
			   EGLint *num_devices);

	int (*get_drm_device_file)(EGLDeviceEXT device,
				   const char **drm_device_file);

	/* Binds a previously imported dmabuf image as the renderer output image. */
	int (*bind_output_dmabuf)(struct weston_output *output,
							  struct linux_dmabuf_buffer *dmabuf,
							  unsigned int buffer_age);
};
