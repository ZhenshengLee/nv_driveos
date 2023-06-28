/*
 * Copyright © 2011 Benjamin Franzke
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "config.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <signal.h>

#ifndef NO_LIBINPUT
#include <linux/input.h>
#endif

#if defined(__QNX__)
#include <sys/time.h>
#endif
#include <wayland-client.h>
#include <wayland-egl.h>
#include <wayland-cursor.h>

#include <GLES2/gl2.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "xdg-shell-unstable-v6-client-protocol.h"
#include <sys/types.h>
#include <unistd.h>
#include "ivi-application-client-protocol.h"
#define IVI_SURFACE_ID 9000
#ifdef ENABLE_IVI_CONTROLLER
#include <ilm_client.h>
#endif

#include "shared/helpers.h"
#include "shared/platform.h"
#include "weston-egl-ext.h"

#include <pthread.h>
#include <poll.h>
#include <errno.h>

#define DEFAULT_WIDTH 250
#define DEFAULT_HEIGHT 250
#define DEFAULT_SURFACE_ID 9000

// NV_MULTHREADED is set as build parameter when building
// weston-simple-egl-multithreaded
#define DEFAULT_THREADS 2

#define FRAME_SYNC_APP     (1 << 0)
#define FRAME_SYNC_DRIVER  (1 << 1)

struct window;
struct seat;
struct wl_output *output = NULL;

struct display {
	struct wl_display *display;
	struct wl_registry *registry;
	struct wl_compositor *compositor;
	struct zxdg_shell_v6 *shell;
	struct wl_seat *seat;
	struct wl_pointer *pointer;
	struct wl_touch *touch;
	struct wl_keyboard *keyboard;
	struct wl_shm *shm;
	struct wl_cursor_theme *cursor_theme;
	struct wl_cursor *default_cursor;
	struct wl_surface *cursor_surface;
	struct {
		EGLDisplay dpy;
		EGLConfig conf;
	} egl;
	struct ivi_application *ivi_application;
#ifdef ENABLE_IVI_CONTROLLER
	bool has_ivi_controller;
#endif
	PFNEGLSWAPBUFFERSWITHDAMAGEEXTPROC swap_buffers_with_damage;
};

struct geometry {
	int width, height;
};

struct window {
	struct display *display;
	struct geometry geometry, window_size;
	struct {
		GLuint rotation_uniform;
		GLuint pos;
		GLuint col;
	} gl;

	struct wl_event_queue * wl_event_queue;
	EGLContext threadCtx;

	uint32_t benchmark_time, frames;
	struct wl_egl_window *native;
	struct wl_surface *surface;
	struct zxdg_surface_v6 *xdg_surface;
	struct zxdg_toplevel_v6 *xdg_toplevel;
	struct ivi_surface *ivi_surface;
	EGLSurface egl_surface;
	struct wl_callback *callback;
	int fullscreen, opaque, buffer_size, frame_sync, delay;
	bool wait_for_configure;
	unsigned int ivi_surfaceId;
	struct wl_callback    *throttle_callback;
};

struct threads_ctx {
	int num;
	pthread_t * tids;
	struct window * wins;
	struct display * dpy;
};

int surfaceCount = DEFAULT_THREADS;

static const char *vert_shader_text =
	"uniform mat4 rotation;\n"
	"attribute vec4 pos;\n"
	"attribute vec4 color;\n"
	"varying vec4 v_color;\n"
	"void main() {\n"
	"  gl_Position = rotation * pos;\n"
	"  v_color = color;\n"
	"}\n";

static const char *frag_shader_text =
	"precision mediump float;\n"
	"varying vec4 v_color;\n"
	"void main() {\n"
	"  gl_FragColor = v_color;\n"
	"}\n";

static volatile int running = 1;

static GLuint
create_shader(struct window *window, const char *source, GLenum shader_type)
{
	GLuint shader;
	GLint status;

	shader = glCreateShader(shader_type);
	assert(shader != 0);

	glShaderSource(shader, 1, (const char **) &source, NULL);
	glCompileShader(shader);

	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (!status) {
		char log[1000];
		GLsizei len;
		glGetShaderInfoLog(shader, 1000, &len, log);
		fprintf(stderr, "Error: compiling %s: %*s\n",
			shader_type == GL_VERTEX_SHADER ? "vertex" : "fragment",
			len, log);
		exit(1);
	}

	return shader;
}

static void
init_gl(struct window *window)
{
	GLuint frag, vert;
	GLuint program;
	GLint status;

	frag = create_shader(window, frag_shader_text, GL_FRAGMENT_SHADER);
	vert = create_shader(window, vert_shader_text, GL_VERTEX_SHADER);

	program = glCreateProgram();
	glAttachShader(program, frag);
	glAttachShader(program, vert);
	glLinkProgram(program);

	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (!status) {
		char log[1000];
		GLsizei len;
		glGetProgramInfoLog(program, 1000, &len, log);
		fprintf(stderr, "Error: linking:\n%*s\n", len, log);
		exit(1);
	}

	glUseProgram(program);

	window->gl.pos = 0;
	window->gl.col = 1;

	glBindAttribLocation(program, window->gl.pos, "pos");
	glBindAttribLocation(program, window->gl.col, "color");
	glLinkProgram(program);

	window->gl.rotation_uniform =
		glGetUniformLocation(program, "rotation");
}

static void
handle_surface_configure(void *data, struct zxdg_surface_v6 *surface,
			 uint32_t serial)
{
	struct window *window = data;

	zxdg_surface_v6_ack_configure(surface, serial);

	window->wait_for_configure = false;
}

static const struct zxdg_surface_v6_listener xdg_surface_listener = {
	handle_surface_configure
};

static void
handle_toplevel_configure(void *data, struct zxdg_toplevel_v6 *toplevel,
			  int32_t width, int32_t height,
			  struct wl_array *states)
{
	struct window *window = data;
	uint32_t *p;

	window->fullscreen = 0;
	wl_array_for_each(p, states) {
		uint32_t state = *p;
		switch (state) {
		case ZXDG_TOPLEVEL_V6_STATE_FULLSCREEN:
			window->fullscreen = 1;
			break;
		}
	}

	if (width > 0 && height > 0) {
		if (!window->fullscreen) {
			window->window_size.width = width;
			window->window_size.height = height;
		}
		window->geometry.width = width;
		window->geometry.height = height;
	} else if (!window->fullscreen) {
		window->geometry = window->window_size;
	}

	if (window->native)
		wl_egl_window_resize(window->native,
				     window->geometry.width,
				     window->geometry.height, 0, 0);
}

static void
handle_toplevel_close(void *data, struct zxdg_toplevel_v6 *xdg_toplevel)
{
	running = 0;
}

static const struct zxdg_toplevel_v6_listener xdg_toplevel_listener = {
	handle_toplevel_configure,
	handle_toplevel_close,
};


static void display_handle_mode(void *data, struct wl_output *wl_output,
				uint32_t flags, int32_t width, int32_t height,
				int32_t refresh) {
	struct window *win = data;
	if (flags & WL_OUTPUT_MODE_CURRENT) {
		win->geometry.width = width;
		win->geometry.height = height;
	}
}

static void display_handle_geometry(void *data, struct wl_output *wl_output,
             int32_t x, int32_t y, int32_t physical_width, int32_t physical_height,
             int32_t subpixel, const char *make, const char *model, int32_t transform) {
	// this space intentionally left blank
}

static void display_handle_done(void *data, struct wl_output *wl_output) {
	// this space intentionally left blank
}

static void display_handle_scale(void *data, struct wl_output *wl_output, int32_t factor) {
	// this space intentionally left blank
}


static const struct wl_output_listener output_listener = {
	.mode = display_handle_mode,
	.geometry = display_handle_geometry,
	.done = display_handle_done,
	.scale = display_handle_scale
};

static void
handle_ivi_surface_configure(void *data, struct ivi_surface *ivi_surface,
                             int32_t width, int32_t height)
{
	struct window *window = data;

	if (window->native)
		wl_egl_window_resize(window->native, width, height, 0, 0);

	window->geometry.width = width;
	window->geometry.height = height;

	if (!window->fullscreen)
		window->window_size = window->geometry;
}

static const struct ivi_surface_listener ivi_surface_listener = {
	handle_ivi_surface_configure,
};

static void
create_xdg_surface(struct window *window, struct display *display)
{
	window->xdg_surface = zxdg_shell_v6_get_xdg_surface(display->shell,
							    window->surface);
	zxdg_surface_v6_add_listener(window->xdg_surface,
				     &xdg_surface_listener, window);

	window->xdg_toplevel =
		zxdg_surface_v6_get_toplevel(window->xdg_surface);
	zxdg_toplevel_v6_add_listener(window->xdg_toplevel,
				      &xdg_toplevel_listener, window);

	zxdg_toplevel_v6_set_title(window->xdg_toplevel, "simple-egl");

	window->wait_for_configure = true;
	wl_surface_commit(window->surface);
}

static void
create_ivi_surface(struct window *window, struct display *display)
{
#ifdef ENABLE_IVI_CONTROLLER
	if (display->has_ivi_controller) {
		ilmErrorTypes error;
		t_ilm_nativedisplay native_display = (t_ilm_nativedisplay)display->display;
		printf("Using surface id: %u\n", window->ivi_surfaceId);

		error = ilmClient_init(native_display);
		assert(error == ILM_SUCCESS);

		error = ilm_surfaceCreate((t_ilm_nativehandle)window->surface,
					  window->geometry.width,
					  window->geometry.height,
					  ILM_PIXELFORMAT_RGBA_8888,
					  &window->ivi_surfaceId);
		assert(error == ILM_SUCCESS);
	} else
#endif
	{
		uint32_t id_ivisurf = window->ivi_surfaceId + (uint32_t)getpid();
		window->ivi_surface =
			ivi_application_surface_create(display->ivi_application,
						       id_ivisurf, window->surface);

		if (window->ivi_surface == NULL) {
			fprintf(stderr, "Failed to create ivi_client_surface\n");
			abort();
		}

		ivi_surface_add_listener(window->ivi_surface,
					 &ivi_surface_listener, window);
	}
}

static void
wayland_throttle_callback(void *data,
                          struct wl_callback *callback,
                          uint32_t time)
{
   struct window * w = data;

    wl_callback_destroy(callback);
    w->throttle_callback = NULL;
}

static const struct wl_callback_listener throttle_listener = {
   .done = wayland_throttle_callback
};

static void
create_surface(struct window *window)
{
	EGLBoolean ret;
	struct display *display = window->display;

	window->wl_event_queue = wl_display_create_queue(display->display);
	assert(window->wl_event_queue);

	window->surface = wl_compositor_create_surface(display->compositor);
	wl_proxy_set_queue((struct wl_proxy *)window->surface,
			   window->wl_event_queue);

	window->throttle_callback = NULL;
	window->native = wl_egl_window_create(window->surface,
					      window->geometry.width,
					      window->geometry.height);

	if (display->shell) {
		create_xdg_surface(window, display);
	} else if (display->ivi_application) {
		create_ivi_surface(window, display);
	} else {
		assert(0);
	}
	if (window->xdg_surface) {
		wl_proxy_set_queue((struct wl_proxy *)window->xdg_surface,
				   window->wl_event_queue);
	}
	window->egl_surface =
		weston_platform_create_egl_surface(display->egl.dpy,
						   display->egl.conf,
						   window->native, NULL);

	ret = eglMakeCurrent(window->display->egl.dpy, window->egl_surface,
			     window->egl_surface, window->threadCtx);
	assert(ret == EGL_TRUE);

	if (!(window->frame_sync & FRAME_SYNC_DRIVER)) {
		eglSwapInterval(display->egl.dpy, 0);
	}
	if (!display->shell)
		return;

	if (window->fullscreen)
		zxdg_toplevel_v6_set_fullscreen(window->xdg_toplevel, NULL);
}

static void
destroy_surface(struct window *window)
{
	if (window->xdg_toplevel)
		zxdg_toplevel_v6_destroy(window->xdg_toplevel);
	if (window->xdg_surface)
		zxdg_surface_v6_destroy(window->xdg_surface);
	if (window->display->ivi_application) {
#ifdef ENABLE_IVI_CONTROLLER
		if (window->display->has_ivi_controller) {
			ilm_surfaceRemove(window->ivi_surfaceId);
		} else
#endif
		{
			ivi_surface_destroy(window->ivi_surface);
		}
	}

	wl_surface_destroy(window->surface);

	if (window->callback)
		wl_callback_destroy(window->callback);
}

static void
redraw(void *data, struct wl_callback *callback, uint32_t time)
{
	struct window *window = data;
	struct display *display = window->display;
	static const GLfloat verts[3][2] = {
		{ -0.5, -0.5 },
		{  0.5, -0.5 },
		{  0,    0.5 }
	};
	static const GLfloat colors[3][3] = {
		{ 1, 0, 0 },
		{ 0, 1, 0 },
		{ 0, 0, 1 }
	};
	GLfloat angle;
	GLfloat rotation[4][4] = {
		{ 1, 0, 0, 0 },
		{ 0, 1, 0, 0 },
		{ 0, 0, 1, 0 },
		{ 0, 0, 0, 1 }
	};
	static const uint32_t speed_div = 5, benchmark_interval = 5;
	struct wl_region *region;
	EGLint rect[4];
	EGLint buffer_age = 0;
	struct timeval tv;

	assert(window->callback == callback);
	window->callback = NULL;

	if (callback)
		wl_callback_destroy(callback);

	gettimeofday(&tv, NULL);
	time = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	if (window->frames == 0)
		window->benchmark_time = time;
	if (time - window->benchmark_time > (benchmark_interval * 1000)) {
		printf("%u frames in %u seconds: %f fps\n",
		       window->frames,
		       benchmark_interval,
		       (float) window->frames / benchmark_interval);
		window->benchmark_time = time;
		window->frames = 0;
	}

	angle = (time / speed_div) % 360 * M_PI / 180.0;
	rotation[0][0] =  cos(angle);
	rotation[0][2] =  sin(angle);
	rotation[2][0] = -sin(angle);
	rotation[2][2] =  cos(angle);

	if (display->swap_buffers_with_damage)
		eglQuerySurface(display->egl.dpy, window->egl_surface,
				EGL_BUFFER_AGE_EXT, &buffer_age);

	glViewport(0, 0, window->geometry.width, window->geometry.height);

	glUniformMatrix4fv(window->gl.rotation_uniform, 1, GL_FALSE,
			   (GLfloat *) rotation);

	glClearColor(0.0, 0.0, 0.0, 0.5);
	glClear(GL_COLOR_BUFFER_BIT);

	glVertexAttribPointer(window->gl.pos, 2, GL_FLOAT, GL_FALSE, 0, verts);
	glVertexAttribPointer(window->gl.col, 3, GL_FLOAT, GL_FALSE, 0, colors);
	glEnableVertexAttribArray(window->gl.pos);
	glEnableVertexAttribArray(window->gl.col);

	glDrawArrays(GL_TRIANGLES, 0, 3);

	glDisableVertexAttribArray(window->gl.pos);
	glDisableVertexAttribArray(window->gl.col);

	usleep(window->delay);

	if (window->opaque || window->fullscreen) {
		region = wl_compositor_create_region(window->display->compositor);
		wl_region_add(region, 0, 0,
			      window->geometry.width,
			      window->geometry.height);
		wl_surface_set_opaque_region(window->surface, region);
		wl_region_destroy(region);
	} else {
		wl_surface_set_opaque_region(window->surface, NULL);
	}

	if (display->swap_buffers_with_damage && buffer_age > 0) {
		rect[0] = window->geometry.width / 4 - 1;
		rect[1] = window->geometry.height / 4 - 1;
		rect[2] = window->geometry.width / 2 + 2;
		rect[3] = window->geometry.height / 2 + 2;
		display->swap_buffers_with_damage(display->egl.dpy,
						  window->egl_surface,
						  rect, 1);
	} else {
		eglSwapBuffers(display->egl.dpy, window->egl_surface);
	}
	window->frames++;
}

static void
pointer_handle_enter(void *data, struct wl_pointer *pointer,
		     uint32_t serial, struct wl_surface *surface,
		     wl_fixed_t sx, wl_fixed_t sy)
{
	struct display *display = data;
	struct wl_buffer *buffer;
	struct wl_cursor *cursor = display->default_cursor;
	struct wl_cursor_image *image;

#if !NV_MULTITHREADED
	if (display->mainWindow->fullscreen)
		wl_pointer_set_cursor(pointer, serial, NULL, 0, 0);
	else
#endif
	if (cursor) {
		image = display->default_cursor->images[0];
		buffer = wl_cursor_image_get_buffer(image);
		if (!buffer)
			return;
		wl_pointer_set_cursor(pointer, serial,
				      display->cursor_surface,
				      image->hotspot_x,
				      image->hotspot_y);
		wl_surface_attach(display->cursor_surface, buffer, 0, 0);
		wl_surface_damage(display->cursor_surface, 0, 0,
				  image->width, image->height);
		wl_surface_commit(display->cursor_surface);
	}
}

static void
pointer_handle_leave(void *data, struct wl_pointer *pointer,
		     uint32_t serial, struct wl_surface *surface)
{
}

static void
pointer_handle_motion(void *data, struct wl_pointer *pointer,
		      uint32_t time, wl_fixed_t sx, wl_fixed_t sy)
{
}

static void
pointer_handle_button(void *data, struct wl_pointer *wl_pointer,
		      uint32_t serial, uint32_t time, uint32_t button,
		      uint32_t state)
{
#if !defined(NO_LIBINPUT) && !defined(NV_MULTITHREADED)
	struct display *display = data;

	if (!display->mainWindow->xdg_toplevel)
		return;

	if (button == BTN_LEFT && state == WL_POINTER_BUTTON_STATE_PRESSED)
		zxdg_toplevel_v6_move(display->mainWindow->xdg_toplevel,
				      display->seat, serial);
#endif
}

static void
pointer_handle_axis(void *data, struct wl_pointer *wl_pointer,
		    uint32_t time, uint32_t axis, wl_fixed_t value)
{
}

static const struct wl_pointer_listener pointer_listener = {
	pointer_handle_enter,
	pointer_handle_leave,
	pointer_handle_motion,
	pointer_handle_button,
	pointer_handle_axis,
};

static void
touch_handle_down(void *data, struct wl_touch *wl_touch,
		  uint32_t serial, uint32_t time, struct wl_surface *surface,
		  int32_t id, wl_fixed_t x_w, wl_fixed_t y_w)
{
	struct display *d = (struct display *)data;

	if (!d->shell)
		return;

#if !NV_MULTITHREADED
	zxdg_toplevel_v6_move(d->mainWindow->xdg_toplevel, d->seat, serial);
#endif
}

static void
touch_handle_up(void *data, struct wl_touch *wl_touch,
		uint32_t serial, uint32_t time, int32_t id)
{
}

static void
touch_handle_motion(void *data, struct wl_touch *wl_touch,
		    uint32_t time, int32_t id, wl_fixed_t x_w, wl_fixed_t y_w)
{
}

static void
touch_handle_frame(void *data, struct wl_touch *wl_touch)
{
}

static void
touch_handle_cancel(void *data, struct wl_touch *wl_touch)
{
}

static const struct wl_touch_listener touch_listener = {
	touch_handle_down,
	touch_handle_up,
	touch_handle_motion,
	touch_handle_frame,
	touch_handle_cancel,
};

static void
keyboard_handle_keymap(void *data, struct wl_keyboard *keyboard,
		       uint32_t format, int fd, uint32_t size)
{
}

static void
keyboard_handle_enter(void *data, struct wl_keyboard *keyboard,
		      uint32_t serial, struct wl_surface *surface,
		      struct wl_array *keys)
{
}

static void
keyboard_handle_leave(void *data, struct wl_keyboard *keyboard,
		      uint32_t serial, struct wl_surface *surface)
{
}

static void
keyboard_handle_key(void *data, struct wl_keyboard *keyboard,
		    uint32_t serial, uint32_t time, uint32_t key,
		    uint32_t state)
{
#ifndef NO_LIBINPUT
	struct display *d = data;

	if (!d->shell)
		return;

	if (key == KEY_F11 && state) {
#if !NV_MULTITHREADED
		if (d->mainWindow->fullscreen)
			zxdg_toplevel_v6_unset_fullscreen(d->mainWindow->xdg_toplevel);
		else
			zxdg_toplevel_v6_set_fullscreen(d->mainWindow->xdg_toplevel,
							NULL);
#endif
	} else if (key == KEY_ESC && state)
		running = 0;
#endif
}

static void
keyboard_handle_modifiers(void *data, struct wl_keyboard *keyboard,
			  uint32_t serial, uint32_t mods_depressed,
			  uint32_t mods_latched, uint32_t mods_locked,
			  uint32_t group)
{
}

static const struct wl_keyboard_listener keyboard_listener = {
	keyboard_handle_keymap,
	keyboard_handle_enter,
	keyboard_handle_leave,
	keyboard_handle_key,
	keyboard_handle_modifiers,
};

static void
seat_handle_capabilities(void *data, struct wl_seat *seat,
			 enum wl_seat_capability caps)
{
	struct display *d = data;

	if ((caps & WL_SEAT_CAPABILITY_POINTER) && !d->pointer) {
		d->pointer = wl_seat_get_pointer(seat);
		wl_pointer_add_listener(d->pointer, &pointer_listener, d);
	} else if (!(caps & WL_SEAT_CAPABILITY_POINTER) && d->pointer) {
		wl_pointer_destroy(d->pointer);
		d->pointer = NULL;
	}

	if ((caps & WL_SEAT_CAPABILITY_KEYBOARD) && !d->keyboard) {
		d->keyboard = wl_seat_get_keyboard(seat);
		wl_keyboard_add_listener(d->keyboard, &keyboard_listener, d);
	} else if (!(caps & WL_SEAT_CAPABILITY_KEYBOARD) && d->keyboard) {
		wl_keyboard_destroy(d->keyboard);
		d->keyboard = NULL;
	}

	if ((caps & WL_SEAT_CAPABILITY_TOUCH) && !d->touch) {
		d->touch = wl_seat_get_touch(seat);
		wl_touch_set_user_data(d->touch, d);
		wl_touch_add_listener(d->touch, &touch_listener, d);
	} else if (!(caps & WL_SEAT_CAPABILITY_TOUCH) && d->touch) {
		wl_touch_destroy(d->touch);
		d->touch = NULL;
	}
}

static const struct wl_seat_listener seat_listener = {
	seat_handle_capabilities,
};

static void
xdg_shell_ping(void *data, struct zxdg_shell_v6 *shell, uint32_t serial)
{
	zxdg_shell_v6_pong(shell, serial);
}

static const struct zxdg_shell_v6_listener xdg_shell_listener = {
	xdg_shell_ping,
};

static void
registry_handle_global(void *data, struct wl_registry *registry,
		       uint32_t name, const char *interface, uint32_t version)
{
	struct display *d = data;

	if (strcmp(interface, "wl_compositor") == 0) {
		d->compositor =
			wl_registry_bind(registry, name,
					 &wl_compositor_interface,
					 MIN(version, 4));
	} else if (strcmp(interface, "zxdg_shell_v6") == 0) {
		d->shell = wl_registry_bind(registry, name,
					    &zxdg_shell_v6_interface, 1);
		zxdg_shell_v6_add_listener(d->shell, &xdg_shell_listener, d);
	} else if (strcmp(interface, "wl_seat") == 0) {
		d->seat = wl_registry_bind(registry, name,
					   &wl_seat_interface, 1);
		wl_seat_add_listener(d->seat, &seat_listener, d);
	} else if (strcmp(interface, "wl_shm") == 0) {
		d->shm = wl_registry_bind(registry, name,
					  &wl_shm_interface, 1);
		d->cursor_theme = wl_cursor_theme_load(NULL, 32, d->shm);
		if (!d->cursor_theme) {
			fprintf(stderr, "unable to load default theme\n");
			return;
		}
		d->default_cursor =
			wl_cursor_theme_get_cursor(d->cursor_theme, "left_ptr");
		if (!d->default_cursor) {
			fprintf(stderr, "unable to load default left pointer\n");
			// TODO: abort ?
		}
	} else if (strcmp(interface, "ivi_application") == 0) {
		d->ivi_application =
			wl_registry_bind(registry, name,
					 &ivi_application_interface, 1);
#ifdef ENABLE_IVI_CONTROLLER
	} else if (strcmp(interface, "ivi_wm") == 0) {
		d->has_ivi_controller = true;
#endif
	}
}

static void
registry_handle_global_remove(void *data, struct wl_registry *registry,
			      uint32_t name)
{
}

static const struct wl_registry_listener registry_listener = {
	registry_handle_global,
	registry_handle_global_remove
};

static void
usage(int error_code)
{
	fprintf(stderr, "Usage: simple-egl-multithreaded [OPTIONS]\n\n"
		"  -d <us>\t\tBuffer swap delay in microseconds\n"
		"  -f\t\tRun in fullscreen mode\n"
		"  -o\t\tCreate an opaque surface\n"
		"  -s\t\tUse a 16 bpp EGL config\n"
		"  -sync <val>\tFrame sync option(default is 1):\n"
		"          0 :\tno sync, set eglSwapInterval to 0\n"
		"          1 :\tapp wait for wl_surface_interface::frame before redraw\n"
		"          2 :\teglSwapBuffers wait for wl_surface_interface::frame in egl driver. "
		"This is the default behavior in egl driver(eglSwapInterval=1)\n"
		"          3 :\tboth 1 & 2\n"
		"  -threads <val>\tFor multithreaded client application\n"
		"  -width <val>\tSurface width (default: %d)\n"
		"  -height <val>\tSurface height (default: %d)\n"
		"  -surface <id>\tSet surface id (default: %d)\n"
		"  -h\t\tThis help text\n\n"
		, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_SURFACE_ID);

	exit(error_code);
}

static void
destroy_wl_resources(struct display * display) {
	if (display->pointer)
		wl_pointer_destroy(display->pointer);

	if (display->keyboard)
		wl_keyboard_destroy(display->keyboard);

	if (display->touch)
		wl_touch_destroy(display->touch);

	wl_surface_destroy(display->cursor_surface);
	if (display->cursor_theme)
		wl_cursor_theme_destroy(display->cursor_theme);

	if (display->shm)
		wl_shm_destroy(display->shm);

	if (display->seat)
		wl_seat_destroy(display->seat);

	if (display->shell)
		zxdg_shell_v6_destroy(display->shell);

	if (display->ivi_application) {
#ifdef ENABLE_IVI_CONTROLLER
		if (display->has_ivi_controller) {
			ilmClient_destroy();
		} else
#endif
		{
			ivi_application_destroy(display->ivi_application);
		}
	}
	if (display->compositor)
		wl_compositor_destroy(display->compositor);
	wl_registry_destroy(display->registry);
	wl_display_flush(display->display);
	wl_display_disconnect(display->display);
}

static void
sig_handler(int signo) {
       running = 0;
}

static int
render_frame(struct window * w)
{
	int ret = 0;
	/*If surface is invisible, app will not receive frame
	  event from server. So if app calls eglSwapBuffers and
	  surface is invisible, server will not respond and app will
	  be blocked inside of eglSwapBuffers.
	  To avoid such issue, app should throttle redraw by waiting
	  surface::frame,  once it receives surface::frame, it's
	  safe to call eglSwapBuffers and it will not be blocked.
	  Unfortunately there is no way to avoid it if surface is
	  invisible in the beginning.*/
	if (w->frame_sync & FRAME_SYNC_APP) {
		w->throttle_callback = wl_surface_frame(w->surface);
		if (!w->throttle_callback) {
			return -1;
		}
		/* install frame callback */
		wl_callback_add_listener(w->throttle_callback,
					 &throttle_listener, w);

		redraw(w, NULL, 0);

		/* wait until server sent back frame callback.
		   so we will not draw and post surface to server if
		   surface is invisible. */
		while (ret != -1 && running && w->throttle_callback) {
			ret = wl_display_roundtrip_queue(w->display->display,
							 w->wl_event_queue);
			if (w->throttle_callback) {
				/* delay to avoid sending too many messages*/
				usleep(10);
			}
		}
		/* free callback if got error or be interrupted */
		if (w->throttle_callback) {
			assert(ret == -1 || !running);
			wl_callback_destroy(w->throttle_callback);
		}
	} else {
		redraw(w, NULL, 0);
		ret = wl_display_roundtrip_queue(w->display->display,
						 w->wl_event_queue);
	}
	return ret;
}

static int
display_init(struct display *display, struct window *window)
{
	const struct {
		char *extension, *entrypoint;
	} swap_damage_ext_to_entrypoint[] = {
		{
			.extension = "EGL_EXT_swap_buffers_with_damage",
			.entrypoint = "eglSwapBuffersWithDamageEXT",
		},
		{
			.extension = "EGL_KHR_swap_buffers_with_damage",
			.entrypoint = "eglSwapBuffersWithDamageKHR",
		},
	};

	const char *extensions;

	EGLint config_attribs[] = {
		EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
		EGL_RED_SIZE, 1,
		EGL_GREEN_SIZE, 1,
		EGL_BLUE_SIZE, 1,
		EGL_ALPHA_SIZE, 1,
		EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
		EGL_NONE
	};

	EGLint major, minor, n, count, i, size;
	EGLConfig *configs;
	EGLBoolean ret;

	display->display = wl_display_connect(NULL);
	assert(display->display);

	if (!display->display) {
		fprintf(stderr, "wl_display_connect failed\n");
		return -1;
	}

	display->registry = wl_display_get_registry(display->display);
	wl_registry_add_listener(display->registry,
				 &registry_listener, display);

	wl_display_roundtrip(display->display);

	if (output && display->ivi_application) {
		wl_output_add_listener(output, &output_listener, &window);
		wl_display_roundtrip(display->display);
	}

	if (window->opaque || window->buffer_size == 16)
		config_attribs[9] = 0;

	display->egl.dpy =
		weston_platform_get_egl_display(EGL_PLATFORM_WAYLAND_KHR,
						display->display, NULL);
	assert(display->egl.dpy);

	ret = eglInitialize(display->egl.dpy, &major, &minor);
	assert(ret == EGL_TRUE);
	if (ret != EGL_TRUE) {
		fprintf(stderr, "eglInitialize failed\n");
		return -1;
	}

	ret = eglBindAPI(EGL_OPENGL_ES_API);
	assert(ret == EGL_TRUE);

	if (!eglGetConfigs(display->egl.dpy, NULL, 0, &count) || count < 1)
		assert(0);

	configs = calloc(count, sizeof *configs);
	assert(configs);

	ret = eglChooseConfig(display->egl.dpy, config_attribs,
			      configs, count, &n);
	assert(ret && n >= 1);

	for (i = 0; i < n; i++) {
		eglGetConfigAttrib(display->egl.dpy,
				   configs[i], EGL_BUFFER_SIZE, &size);
		if (window->buffer_size == size) {
			display->egl.conf = configs[i];
			break;
		}
	}
	free(configs);
	if (display->egl.conf == NULL) {
		fprintf(stderr, "did not find config with buffer size %d\n",
			window->buffer_size);
		exit(EXIT_FAILURE);
	}

	display->swap_buffers_with_damage = NULL;
	extensions = eglQueryString(display->egl.dpy, EGL_EXTENSIONS);
	if (extensions &&
	    weston_check_egl_extension(extensions, "EGL_EXT_buffer_age")) {
		for (i = 0; i < (int) ARRAY_LENGTH(swap_damage_ext_to_entrypoint); i++) {
			if (weston_check_egl_extension(extensions,
						       swap_damage_ext_to_entrypoint[i].extension)) {
				/* The EXTPROC is identical to the KHR one */
				display->swap_buffers_with_damage =
					(PFNEGLSWAPBUFFERSWITHDAMAGEEXTPROC)
					eglGetProcAddress(swap_damage_ext_to_entrypoint[i].entrypoint);
				break;
			}
		}
	}

	display->cursor_surface =
		wl_compositor_create_surface(display->compositor);

	if (display->swap_buffers_with_damage)
		printf("has EGL_EXT_buffer_age and %s\n", swap_damage_ext_to_entrypoint[i].extension);

	return 0;
}

static void
display_fini(struct display *display)
{
	eglTerminate(display->egl.dpy);
	eglReleaseThread();
	destroy_wl_resources(display);
}


static int
thread_init(struct window * win)
{
	const EGLint context_attribs[] = {
		EGL_CONTEXT_CLIENT_VERSION, 2,
		EGL_NONE
	};

	win->threadCtx = eglCreateContext(win->display->egl.dpy,
					  win->display->egl.conf,
					  EGL_NO_CONTEXT,
					  context_attribs);
	if (win->threadCtx == EGL_NO_CONTEXT)
		return -1;
	create_surface(win);
	init_gl(win);

	return 0;
}

static void
thread_fini(struct window * win)
{
	eglMakeCurrent(win->display->egl.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE,
		       EGL_NO_CONTEXT);

	weston_platform_destroy_egl_surface(win->display->egl.dpy,
					    win->egl_surface);
	wl_egl_window_destroy(win->native);

	eglReleaseThread();

	destroy_surface(win);
	wl_event_queue_destroy(win->wl_event_queue);
}

static void*
thread_func(void *arg) {
	int rc;
	struct window*  window = (struct window*)arg;
	struct display* display = window->display;

	rc = thread_init(window);
	if (rc) {
		return NULL;
	}

	while (running) {
		rc = wl_display_dispatch_queue_pending(display->display,
						       window->wl_event_queue);
		if (rc == -1) {
			fprintf(stderr, "wl_display_dispatch_queue_pending failed\n");
			break;
		}
		if (window->wait_for_configure) 
			continue;

		rc = render_frame(window);
		if (rc == -1) {
			fprintf(stderr, "render_frame failed\n");
			break;
		}
	}

	thread_fini(window);

	return NULL;
}

static int
threads_init(struct threads_ctx * ctx,
	     int num,
	     struct display * dpy,
	     struct window * src)
{
	int i;

	ctx->dpy = dpy;
	assert(num > 0);
	ctx->num = num;
	ctx->tids = calloc(sizeof(pthread_t), num);
	assert(ctx->tids);
	if (!ctx->tids) {
		return -1;
	}
	ctx->wins = calloc(sizeof(struct window), num);
	assert(ctx->wins);
	if (!ctx->wins) {
		return -1;
	}
	for (i = 0; i < num; i++) {
		ctx->wins[i].display = dpy;
		ctx->wins[i].geometry.width  = src->geometry.width;
		ctx->wins[i].geometry.height = src->geometry.height;
		ctx->wins[i].window_size = src->geometry;
		ctx->wins[i].buffer_size = src->buffer_size;
		ctx->wins[i].frame_sync = src->frame_sync;
#ifdef ENABLE_IVI_CONTROLLER
		ctx->wins[i].ivi_surfaceId = src->ivi_surfaceId + i;
#endif
		ctx->wins[i].geometry.width  = src->geometry.width;
		ctx->wins[i].geometry.height = src->geometry.height;

		pthread_create(&ctx->tids[i], NULL, thread_func, (void *)&ctx->wins[i]);
	}

	return 0;
}
static void
threads_fini(struct threads_ctx * ctx)
{
	int i;
	running = 0;

	for (i = 0; i < ctx->num; i++) {
		pthread_join(ctx->tids[i], NULL);
	}

	free(ctx->tids);
	free(ctx->wins);
}

int
main(int argc, char **argv)
{
	struct display display = { 0 };
	struct sigaction sigint;
	struct window  window  = { 0 };
	int i, ret = 0;

	window.display = &display;
	window.geometry.width  = DEFAULT_WIDTH;
	window.geometry.height = DEFAULT_HEIGHT;
#ifdef ENABLE_IVI_CONTROLLER
	display.has_ivi_controller = false;
#endif
	window.buffer_size = 32;
	window.frame_sync = FRAME_SYNC_APP;
	window.delay = 0;
	window.ivi_surfaceId = DEFAULT_SURFACE_ID;

	for (i = 1; i < argc; i++) {
		if (strcmp("-d", argv[i]) == 0 && i+1 < argc)
			window.delay = atoi(argv[++i]);
		else if (strcmp("-f", argv[i]) == 0)
			window.fullscreen = 1;
		else if (strcmp("-o", argv[i]) == 0)
			window.opaque = 1;
		else if (strcmp("-s", argv[i]) == 0)
			window.buffer_size = 16;
		else if (strcmp("-sync", argv[i]) == 0)
			sscanf(argv[++i], "%d", &window.frame_sync);
		else if (strcmp("-threads",argv[i]) == 0)
			sscanf(argv[++i], "%d", &surfaceCount);
		else if (strcmp("-width", argv[i]) == 0)
			sscanf(argv[++i], "%d", &window.geometry.width);
		else if (strcmp("-height", argv[i]) == 0)
			sscanf(argv[++i], "%d", &window.geometry.height);
		else if (strcmp("-surface", argv[i]) == 0)
			sscanf(argv[++i], "%u", &window.ivi_surfaceId);
		else if (strcmp("-h", argv[i]) == 0)
			usage(EXIT_SUCCESS);
		else
			usage(EXIT_FAILURE);
	}
	switch(window.frame_sync) {
		case 1:
			window.frame_sync = FRAME_SYNC_APP;
			break;
		case 2:
			window.frame_sync = FRAME_SYNC_DRIVER;
			break;
		case 3:
			window.frame_sync = FRAME_SYNC_APP | FRAME_SYNC_DRIVER;
			break;
		case 0:
			break;
		default:
			usage(EXIT_FAILURE);
			break;
	}
	window.window_size = window.geometry;

	ret = display_init(&display, &window);
	if (ret ) {
		return -1;
	}

	struct threads_ctx ctx = { 0 };
	ret = threads_init(&ctx, surfaceCount, &display, &window);
	if (ret) {
		return -1;
	}
	signal(SIGINT, sig_handler);
	while (running && ret != -1) {
		ret = wl_display_dispatch(display.display);
	}
	threads_fini(&ctx);
	display_fini(&display);

	return 0;
}
