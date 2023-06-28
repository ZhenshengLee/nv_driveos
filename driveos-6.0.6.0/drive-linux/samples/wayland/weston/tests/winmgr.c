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

#include <stdlib.h>
#include <assert.h>
#include <signal.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdbool.h>
#include "multithread.h"
#include "xdg-shell-unstable-v6-client-protocol.h"
#include "ivi-application-client-protocol.h"
#include "ilm_control.h"

#define IVI_LAYER_ID   1000
#define IVI_SURFACE_ID 1000

#define WIN_MGR_STATUS_RUN          0
#define WIN_MGR_STATUS_STOP         1

#define WIN_MGR_NOTIFY_TEST_START   0
#define WIN_MGR_NOTIFY_TEST_FAIL    1
#define WIN_MGR_NOTIFY_TEST_PASS    2
#define WIN_MGR_NOTIFY_STOP         3
#define WIN_MGR_NOTIFY_THREAD_DONE  4
#define WIN_MGR_NOTIFY_THREAD_ERROR 5

#define win_has_next_test(win, i, bound)			      \
	({							      \
		int _rc = 1;					      \
		if (!win_manager_next_test() || (win)->error.code) {  \
			_rc = 0;				      \
		} else if((bound) > 0) {			      \
			if (bound > (i)) {			      \
				(i)++;				      \
				_rc = 1;			      \
			} else {				      \
				_rc = 0;			      \
			}					      \
		}						      \
		_rc;						      \
	})

#define win_wl_next_test(w)				\
	win_has_next_test((w), (w)->tests.win.iterator, options.win_loops)

#define win_egl_ctx_next_test(w)			\
	win_has_next_test((w), (w)->tests.egl_ctx.iterator, options.egl_ctx_loops)

#define win_egl_surf_next_test(w)			\
	win_has_next_test((w), (w)->tests.egl_surf.iterator, options.egl_surf_loops)

#define win_render_next_test(w)			\
	win_has_next_test((w), (w)->tests.render.iterator, options.frame_limit)

#define WIN_TEST_BLOCK_BEGIN(w, name)			\
	while(name ## _next_test(w)) {			\
		rc = name ## _init(w);			\
		if (rc) {				\
			elog( #name "_init failed\n");	\
			break;				\
		}					\

#define WIN_TEST_BLOCK_END(w, name)			\
		name ## _fini(w);			\
	}


struct error {
	const char * str;
	int  code;
};
struct tests {
	struct {
		int iterator;
	} thread;
	struct {
		int iterator;
	} win;
	struct {
		int iterator;
		int config_num;
		int config;
	} egl_ctx;
	struct {
		int iterator;
		int size[2];
	} egl_surf;
	struct {
		int iterator;
		float det;
		float color;
	} render;
	int all_done;
};

struct win {
	struct error error;
	struct tests tests;

	EGLConfig egl_conf;
	EGLSurface egl_surface;
	EGLContext egl_context;

	struct wl_egl_window * wl_egl_window;
	struct wl_surface * wl_surface;
	struct wl_region * wl_region;
	struct wl_event_queue * wl_event_queue;

	struct {
		// wait_for_configure will be cleared from
		// xdg_surface_listener::configure callback,
		// configure callback will be called from
		// wl_display_dispatch_queue_pending, since each window use
		// its own event queue, so all these functions run in
		// the same thread. it's thread-safe to check/change its
		// value inside of same win_thread.
		bool wait_for_configure;
		struct zxdg_surface_v6 * surface;
		struct zxdg_toplevel_v6 *toplevel;
	} xdg_shell;

	struct {
		unsigned surf_id;
		unsigned layer;
		struct ivi_surface * surface;
	} ivi_shell;

	// the two fields are used by win_manager only.
	// win_manager maintain the list of notify windows.
	// notify: store notify code.
	// notify_next: point to next notify win.
	int notify;
	struct win * notify_next;

	int id, status;
	pthread_t tid;
};

struct win_manager {
	int win_num;
	int ivi_ready_num;
	unsigned * ivi_layers;
	struct win * win;

	struct {
		int done;
	} thread_test;

	struct error error;

	struct {
		unsigned total;
		unsigned skip;
		unsigned pass;
		unsigned fail;
	} test_status;

	int status;

	// the head of list notify windows.
	struct win * notify_win;

	pthread_t tid;
	pthread_t tid_event;
	pthread_mutex_t mutex;
	pthread_cond_t cond;
};

static struct win_manager manager = {
	.mutex = PTHREAD_MUTEX_INITIALIZER,
	.cond  = PTHREAD_COND_INITIALIZER,
};

static void win_do_resize(struct win * win, int w, int h, int dx, int dy);
static int  win_manager_next_test();
static void win_manager_notify(struct win * win, int notify);
static void xdg_handle_toplevel_close(void *data, struct zxdg_toplevel_v6 *xdg_toplevel);
static void xdg_handle_toplevel_configure(void *data, struct zxdg_toplevel_v6 *toplevel,
				      int32_t width, int32_t height,
				      struct wl_array *states);
static void xdg_handle_surface_configure(void *data, struct zxdg_surface_v6 *surface,
				     uint32_t serial);

static const struct zxdg_toplevel_v6_listener xdg_toplevel_listener = {
	xdg_handle_toplevel_configure,
	xdg_handle_toplevel_close,
};
static const struct zxdg_surface_v6_listener xdg_surface_listener = {
	xdg_handle_surface_configure
};

static void
xdg_handle_toplevel_close(void *data, struct zxdg_toplevel_v6 *xdg_toplevel)
{
	//TODO. handle toplevel events.
}
static void
xdg_handle_toplevel_configure(void *data, struct zxdg_toplevel_v6 *toplevel,
			  int32_t width, int32_t height,
			  struct wl_array *states)
{
	struct win * win = data;
	dlog("handle_toplevel_configure w = %d, h = %d\n", width, height);
	if (width > 0 && height > 0) {
		win_do_resize(win, width, height, 0, 0);
	}
}

// random_r is not available on QNX.
// random is not thread safe on some platform.
// so make a wrapper & use a mutex to protect it.
static inline int
nvtest_random_r() {
	static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
	pthread_mutex_lock(&mutex);
	int r = random();
	pthread_mutex_unlock(&mutex);
	return r;
}

static void
win_set_error(struct win * win, const char * str, int rc)
{
	assert(win->error.code == 0);
	win->error.str = str;
	win->error.code = rc;
	elog("%s failed, rc=%d\n", str, rc);
	// notice manager an error happened,
	// manager will save the first error and stop all threads & exit.
	win_manager_notify(win, WIN_MGR_NOTIFY_THREAD_ERROR);
}

static int
win_dispatch_events(struct win * win) {
	int rc = wl_display_dispatch_queue_pending(winsys.wl_display,
						   win->wl_event_queue);
	if (rc == -1) {
		elog("wl_display_dispatch_queue_pending failed\n");
		return -1;
	}
	return 0;
}

static void
win_do_resize(struct win * win, int w, int h, int dx, int dy)
{
	if (w > 0 && h > 0 && win->wl_egl_window) {
		win->tests.egl_surf.size[0] = w;
		win->tests.egl_surf.size[1] = h;

		wl_egl_window_resize(win->wl_egl_window,
				     w, h, dx, dy);
	} else {
		elog("window size(%d,%d) is invalid or egl_window(%p) is invalid\n",
		     w, h, win->wl_egl_window);
	}
}

static int
win_resize(struct win * win)
{
	if (options.win_resize) {
		int i, d[2], size[2];
		static int off[] = {
			-7, 2, 3, 6, -6, 8. -3, 10, -8. -1, -10, 3, 1, 20,-12,
		};
		const int len = sizeof(off)/sizeof(off[0]);

		d[0] = off[(len + (nvtest_random_r() % len)) % len];
		d[1] = off[(len + (nvtest_random_r() % len)) % len];

		for(i=0; i < 2; i++) {
			size[i] = win->tests.egl_surf.size[i] + d[i];
			// negtive size also make the window invisible
			// which will block eglSwapBuffer forever.
			if (size[i] >= 900 ||  size[i] <= 0) {
				size[i] = 400;
			}
		}

		dlog("w = %d, h = %d\n", size[0], size[1]);

		// TODO. add a movement test but without moving window
		// out side of screen. need to find out a way to do it.
		// if window move out of visible area, wayland compositor
		// will never send back frame callback and it will stuck
		// here forever.
		d[0] = d[1] = 0;
		win_do_resize(win, size[0], size[1], d[0], d[1]);
	}
	return 0;
}

static int
win_render(struct win * win)
{
	win_manager_notify(win, WIN_MGR_NOTIFY_TEST_START);

	win->tests.render.color += win->tests.render.det;
	if (win->tests.render.color >= 1.0f)
		win->tests.render.det = -0.003f;
	if (win->tests.render.color <= 0)
		win->tests.render.det = 0.003f;

	glClearColor(win->tests.render.color,
		     win->tests.render.color,
		     win->tests.render.color,
		     1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glFlush();

	if (!eglSwapBuffers(winsys.egl_display, win->egl_surface)) {
		elog("eglSwapBuffers failed. num=%d\n", win->tests.render.iterator);
		win_set_error(win, "eglSwapBuffers", eglGetError());
		win_manager_notify(win, WIN_MGR_NOTIFY_TEST_FAIL);
		return -1;
	}
	if (win_resize(win)) {
		elog("win_surf_resize failed.\n");
		win_manager_notify(win, WIN_MGR_NOTIFY_TEST_FAIL);
		return -1;
	}
	win_manager_notify(win, WIN_MGR_NOTIFY_TEST_PASS);

	if (options.test_flag & TEST_FLAG_IVI_SHELL &&
	    !win->tests.render.iterator ) {
		ilm_layerSetVisibility(win->ivi_shell.layer, ILM_TRUE);
		ilm_commitChanges();
	}

	return 0;
}

static int
win_render_test(struct win * win)
{
	int rc;
	while (win->xdg_shell.wait_for_configure) {
		dlog("waiting for config \n");
		rc = win_dispatch_events(win);
		if (rc) {
			elog("win_dispatch_events failed\n");
			return rc;
		}
	}
	while(win_render_next_test(win)) {
		dlog("win_dispatch_events\n");
		rc = win_dispatch_events(win);
		if (rc) {
			elog("win_dispatch_events failed\n");
			return rc;
		}

		dlog("win_render\n");
		rc = win_render(win);
		if (rc) {
			elog("win_render failed\n");
			return rc;
		}
	}

	dlog("win_render_test done! \n");
	return 0;
}

static int
win_egl_ctx_init(struct win * win)
{
	EGLint count, n;
	EGLConfig *configs;

	EGLint config_attribs[] = {
		EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
		EGL_RED_SIZE, 8,
		EGL_GREEN_SIZE, 8,
		EGL_BLUE_SIZE, 8,
		EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
		EGL_NONE
	};

	static const EGLint context_attribs[] = {
		EGL_CONTEXT_CLIENT_VERSION, 2,
		EGL_NONE
	};

	eglGetConfigs(winsys.egl_display, NULL, 0, &count);
	dlog("EGL has %d configs\n", count);

	configs = calloc(count, sizeof *configs);
	eglChooseConfig(winsys.egl_display, config_attribs,
			configs, count, &n);

	win->tests.egl_ctx.config_num = n;
	assert(win->tests.egl_ctx.config < n);

	win->egl_conf = configs[win->tests.egl_ctx.config];

	win->egl_context =
		eglCreateContext(winsys.egl_display,
				 win->egl_conf,
				 EGL_NO_CONTEXT, context_attribs);
	free(configs);

	if (!win->egl_context) {
		win_set_error(win, "eglCreateContext", eglGetError());
		return -1;
	}
	return 0;
}

static void
win_egl_ctx_fini(struct win * win)
{
	(void)win;
	eglMakeCurrent(winsys.egl_display,
		       EGL_NO_SURFACE,
		       EGL_NO_SURFACE,
		       EGL_NO_CONTEXT);
}

static int
win_egl_surf_init(struct win * win)
{
	win->wl_egl_window =
		wl_egl_window_create(win->wl_surface,
				     win->tests.egl_surf.size[0],
				     win->tests.egl_surf.size[1]);
	if (!win->wl_egl_window ) {
		win_set_error(win, "wl_egl_window_create", ENOMEM);
		return -1;
	}
	dlog("wl_egl_window_create succeed\n");

	win->egl_surface = eglCreateWindowSurface(winsys.egl_display,
						  win->egl_conf,
						  win->wl_egl_window, NULL);
	if (win->egl_surface == EGL_NO_SURFACE) {
		win_set_error(win, "eglCreateWindowSurface", (int)eglGetError());
		return -1;
	}
	dlog("eglCreateWindowSurface succeed\n");

	if (!eglMakeCurrent(winsys.egl_display,
			   win->egl_surface,
			   win->egl_surface,
			   win->egl_context)) {
		win_set_error(win, "eglMakeCurrent", (int)eglGetError());
		return -1;
	}
	dlog("eglMakeCurrent succeed\n");

	return 0;
}

static void
win_egl_surf_fini(struct win * win)
{
	dlog("eglDestroySurface\n");
	eglDestroySurface(winsys.egl_display, win->egl_surface);

	dlog("wl_egl_window_destroy\n");
	wl_egl_window_destroy(win->wl_egl_window);
}

//
// xdg shell
//
static int
win_wl_xdg_shell_init(struct win * win)
{
	if (!winsys.xdg_shell) {
		win->xdg_shell.wait_for_configure = false;
		return -1;
	}
	win->xdg_shell.surface =
		zxdg_shell_v6_get_xdg_surface(winsys.xdg_shell,
					      win->wl_surface);
	if (!win->xdg_shell.surface) {
		win_set_error(win, "zxdg_shell_v6_get_xdg_surface", errno);
		return -1;
	}
	dlog("zxdg_shell_v6_get_shell_surface successed\n");

	dlog("win->wl_event_queue = %p \n",  win->wl_event_queue);
	wl_proxy_set_queue((struct wl_proxy *)win->xdg_shell.surface,
			   win->wl_event_queue);
	dlog("wl_proxy_set_queue successed!\n");

	if (zxdg_surface_v6_add_listener(win->xdg_shell.surface,
					 &xdg_surface_listener, win)) {
		win_set_error(win, "zxdg_surface_v6_add_listener", errno);
		return -1;
	}
	dlog("zxdg_surface_v6_add_listener successed!\n");

	win->xdg_shell.toplevel =
		zxdg_surface_v6_get_toplevel(win->xdg_shell.surface);
	if (!win->xdg_shell.toplevel) {
		win_set_error(win, "zxdg_surface_v6_get_toplevel", errno);
		return -1;
	}
	dlog("zxdg_surface_v6_get_toplevel successed!\n");

	wl_proxy_set_queue((struct wl_proxy *)win->xdg_shell.toplevel,
			   win->wl_event_queue);
	if (zxdg_toplevel_v6_add_listener(win->xdg_shell.toplevel,
					  &xdg_toplevel_listener, win)) {
		win_set_error(win, "zxdg_toplevel_v6_add_listener", errno);
		return -1;
	}
	dlog("zxdg_toplevel_v6_add_listener successed!\n");

	zxdg_toplevel_v6_set_title(win->xdg_shell.toplevel, "multithread-test");
	win->xdg_shell.wait_for_configure = true;

	return 0;
}
static void
win_wl_xdg_shell_fini(struct win * win)
{
	dlog("destroy wl_xdg_surface\n");
	if (win->xdg_shell.toplevel) {
		zxdg_toplevel_v6_destroy(win->xdg_shell.toplevel);
		win->xdg_shell.toplevel = NULL;
	}
	dlog("destroy wl_xdg_surface\n");
	if (win->xdg_shell.surface) {
		zxdg_surface_v6_destroy(win->xdg_shell.surface);
		win->xdg_shell.surface = NULL;
	}
}

static void
xdg_handle_surface_configure(void *data, struct zxdg_surface_v6 *surface,
			 uint32_t serial)
{
	dlog("handle_surface_configure \n");

	struct win * win = data;
	zxdg_surface_v6_ack_configure(surface, serial);
	win->xdg_shell.wait_for_configure = false;
}

//
// ivi shell
//
static void
ivi_handle_surface_configure(void *data, struct ivi_surface *ivi_surface,
                             int32_t width, int32_t height)
{
	struct win *win = data;
	if (win->wl_egl_window && width > 0 && height > 0) {
		win_do_resize(win, width, height, 0, 0);
	}
}
static const struct ivi_surface_listener ivi_surface_listener = {
	ivi_handle_surface_configure,
};

static int
win_wl_ivi_shell_init(struct win * win)
{
	if (!winsys.ivi_app) {
		elog("ivi_application interface is not available.\n");
		return -1;
	}
	win->ivi_shell.surf_id = IVI_SURFACE_ID + win->id;
	win->ivi_shell.surface =
		ivi_application_surface_create(winsys.ivi_app,
					       win->ivi_shell.surf_id,
					       win->wl_surface);
	if (!win->ivi_shell.surface) {
		win_set_error(win, "ivi_application_surface_create", errno);
		elog("ivi_application_surface_create failed\n");
		return -1;
	}
	wl_proxy_set_queue((struct wl_proxy *)win->ivi_shell.surface,
			   win->wl_event_queue);

	ivi_surface_add_listener(win->ivi_shell.surface,
				 &ivi_surface_listener, win);

	if (options.test_flag & TEST_FLAG_IVI_SHELL) {
		ilmErrorTypes rc;
		win->ivi_shell.layer = IVI_LAYER_ID + win->id;
		rc = ilm_layerRemove(win->ivi_shell.layer);
		dlog("ilm_layerRemove, rc = %d\n", (int)rc);

		rc = ilm_commitChanges();
		dlog("ilm_commitChanges, rc = %d\n", (int)rc);
		if (ILM_SUCCESS != rc) {
			elog("ilm_commitChanges failed\n");
			return -1;
		}

		rc = ilm_layerCreateWithDimension(&win->ivi_shell.layer,
						  600, 600);
		dlog("ilm_layerCreateWithDimension , rc = %d\n", (int)rc);
		if (ILM_SUCCESS != rc) {
			elog("ilm_layerCreateWithDimension failed\n");
			return -1;
		}
		rc = ilm_commitChanges();
		dlog("ilm_commitChanges, rc = %d\n", (int)rc);
		if (ILM_SUCCESS != rc) {
			elog("ilm_commitChanges failed\n");
			return -1;
		}

		rc = ilm_layerAddSurface(win->ivi_shell.layer,
					 win->ivi_shell.surf_id);
		dlog("ilm_layerAddSurface, rc = %d\n", (int)rc);
		if (ILM_SUCCESS != rc) {
			elog("ilm_layerAddSurface failed\n");
			return -1;
		}

		rc = ilm_surfaceSetVisibility(win->ivi_shell.surf_id, ILM_TRUE);
		dlog("ilm_surfaceSetVisibility, rc = %d\n", (int)rc);

		rc = ilm_layerSetRenderOrder((t_ilm_layer)win->ivi_shell.layer,
					     (t_ilm_layer *)&win->ivi_shell.surf_id,
					     1);
		dlog("ilm_layerSetRenderOrder, rc = %d\n", (int)rc);

		rc = ilm_commitChanges();
		dlog("ilm_commitChanges, rc = %d\n", (int)rc);
		if (ILM_SUCCESS != rc) {
			elog("ilm_commitChanges failed\n");
			return -1;
		}

		rc = ilm_layerSetVisibility(win->ivi_shell.layer, ILM_TRUE);
		dlog("ilm_layerSetVisibility, rc = %d\n", (int)rc);

		if (ILM_SUCCESS != rc) {
			elog("ilm_layerSetVisibility failed\n");
			return -1;
		}

		rc = ilm_commitChanges();
		if (ILM_SUCCESS != rc) {
			elog("ilm_commitChanges failed\n");
			return -1;
		}

		pthread_mutex_lock(&manager.mutex);
		manager.ivi_layers[win->id] = win->ivi_shell.layer;
	        if (++manager.ivi_ready_num == options.thread_num) {
			pthread_cond_broadcast(&manager.cond);
		}
		pthread_mutex_unlock(&manager.mutex);
	}
	return 0;
}
static void
win_wl_ivi_shell_fini(struct win * win)
{
	if (win->ivi_shell.surface) {
		if (options.test_flag & TEST_FLAG_IVI_SHELL) {
			ilmErrorTypes rc;
			rc = ilm_layerRemoveSurface(win->ivi_shell.layer,
						    win->ivi_shell.surf_id);
			assert(rc == ILM_SUCCESS);
			dlog("ilm_layerRemoveSurface, rc = %d\n", (int)rc);

			rc = ilm_layerRemove(win->ivi_shell.layer);
			assert(rc == ILM_SUCCESS);
			dlog("ilm_layerRemove, rc = %d\n", (int)rc);

			rc = ilm_commitChanges();
			assert(ILM_SUCCESS == rc);
			dlog("ilm_commitChanges, rc = %d\n", (int)rc);
		}

		ivi_surface_destroy(win->ivi_shell.surface);
		win->ivi_shell.surface = NULL;
	}
}


static int
win_wl_init(struct win * win)
{
	int rc;

	win->wl_surface = wl_compositor_create_surface(winsys.wl_compositor);
	if (!win->wl_surface) {
		win_set_error(win, "wl_compositor_create_surface", errno);
		return -1;
	}
	dlog("wl_compositor_create_surface successed\n");

	win->wl_event_queue = wl_display_create_queue(winsys.wl_display);
	if (!win->wl_event_queue) {
		win_set_error(win, "wl_display_create_queue", errno);
		return -1;
	}
	dlog("win->wl_event_queue = %p \n",  win->wl_event_queue);
	dlog("wl_display_create_queue successed\n");

	dlog("win->wl_event_queue = %p \n",  win->wl_event_queue);

	wl_proxy_set_queue((struct wl_proxy *)win->wl_surface,
			   win->wl_event_queue);

	dlog("win->wl_event_queue = %p \n",  win->wl_event_queue);
	win->wl_region = wl_compositor_create_region(winsys.wl_compositor);
	if (!win->wl_region) {
		win_set_error(win, "wl_compositor_create_region", errno);
		return -1;
	}
	dlog("win->wl_event_queue = %p \n",  win->wl_event_queue);
	dlog("wl_compositor_create_region successed\n");

	dlog("surf size: %d x %d \n",
	     win->tests.egl_surf.size[0],
	     win->tests.egl_surf.size[1]);

	wl_region_add(win->wl_region, 0, 0,
		      win->tests.egl_surf.size[0],
		      win->tests.egl_surf.size[1]);

	dlog("win->wl_event_queue = %p \n",  win->wl_event_queue);
	wl_surface_set_opaque_region(win->wl_surface, win->wl_region);

	/* try ivi-shell first.
	 * if ivi-shell is not available, we will try xdg-shell.
	 */
	rc = win_wl_ivi_shell_init(win);
	dlog("win_wl_ivi_shell_init, rc=%d \n", rc);
	if (rc) {
		/* return fail if ivi-test is enabled but ivi-shell init failed*/
		if (options.test_flag & TEST_FLAG_IVI_SHELL) {
			elog("win_wl_ivi_init failed\n");
			return -1;
		}
		/* then try xdg shell */
		rc = win_wl_xdg_shell_init(win);
		dlog("win_wl_xdg_shell_init, rc=%d \n", rc);
	}
	if (rc) {
		elog("win_wl_ivi_init & win_wl_xdg_shell_init failed\n");
		return -1;
	}

	wl_surface_commit(win->wl_surface);

	dlog("win_wl_init done!\n")
	return 0;
}

static void
win_wl_fini(struct win * win)
{
	win_wl_xdg_shell_fini(win);
	win_wl_ivi_shell_fini(win);

	dlog("destroy wl_region\n");
	if (win->wl_region) {
		wl_region_destroy(win->wl_region);
	}
	dlog("destroy wl_event_queue\n");
	if (win->wl_event_queue) {
		wl_event_queue_destroy(win->wl_event_queue);
		win->wl_event_queue = NULL;
	}
	dlog("destroy wl_surface\n");
	if (win->wl_surface) {
		wl_surface_destroy(win->wl_surface);
		win->wl_surface = NULL;
	}
}

static void *
win_thread(void *param) {
	int rc;
	struct win * win = param;
	char name[64];
	snprintf(name, sizeof(name), "win-thread-%03d", win->id);
	pthread_setname_np(pthread_self(), name);

	vlog("win-thread: %03d (%03d/%03d) start\n",
	     win->id,
	     win->tests.thread.iterator,
	     options.thread_loops);

	WIN_TEST_BLOCK_BEGIN(win, win_wl)
	WIN_TEST_BLOCK_BEGIN(win, win_egl_ctx)
	WIN_TEST_BLOCK_BEGIN(win, win_egl_surf)

	rc = win_render_test(win);
	if (rc) {
		elog("win_render_test failed\n");
		break;
	}
	WIN_TEST_BLOCK_END(win, win_egl_surf)
	WIN_TEST_BLOCK_END(win, win_egl_ctx)
	WIN_TEST_BLOCK_END(win, win_wl)

	vlog("win-thread: %03d (%03d/%03d) end\n",
	     win->id,
	     win->tests.thread.iterator,
	     options.thread_loops);

	win_manager_notify(win, WIN_MGR_NOTIFY_THREAD_DONE);
	return NULL;
}


static int
win_manager_next_test()
{
	int rc;
	pthread_mutex_lock(&manager.mutex);
	rc = manager.status != WIN_MGR_STATUS_STOP;
	pthread_mutex_unlock(&manager.mutex);
	return rc;
}

static void
win_manager_notify(struct win * win, int notify)
{
	int rc;
	rc = pthread_mutex_lock(&manager.mutex);
	assert(rc == 0);
	switch(notify){
		case WIN_MGR_NOTIFY_TEST_START:
			manager.test_status.total ++;
			break;
		case WIN_MGR_NOTIFY_TEST_FAIL:
			manager.test_status.fail ++;
			break;
		case WIN_MGR_NOTIFY_TEST_PASS:
			manager.test_status.pass ++;
			break;
		case WIN_MGR_NOTIFY_STOP:
		case WIN_MGR_NOTIFY_THREAD_DONE:
		case WIN_MGR_NOTIFY_THREAD_ERROR:
			win->notify = notify;
			win->notify_next = manager.notify_win;
			manager.notify_win = win;
			rc = pthread_cond_broadcast(&manager.cond);
			assert(rc == 0);
			break;
		default:
			assert(0 && "unexpected notify code");
			break;
	}
	rc = pthread_mutex_unlock(&manager.mutex);
	assert(rc == 0);
	(void)rc;
}

//
// all handler functions are locked.
//
static void
win_manager_handle_error(struct win * win, int error, const char * str)
{
	// only save the first error
	assert(win || error);

	if (!manager.error.code) {
		if (win) {
			manager.error = win->error;
		} else {
			manager.error.code = error;
			manager.error.str = str;
		}
		manager.status = WIN_MGR_STATUS_STOP;
		elog("Get an error. emit term signal to exit\n");
		kill(getpid(), SIGTERM);
	}
}

static void
win_manager_handle_thread_done(struct win * win)
{
	if (win->error.code) {
		win_manager_handle_error(win, 0, NULL);
		return ;
	}
	if (options.thread_loops > 0) {
		assert(!win->tests.all_done);
		assert(options.thread_loops > win->tests.thread.iterator);
		if (options.thread_loops == ++win->tests.thread.iterator) {
			++manager.thread_test.done;
			win->tests.all_done = 1;
			vlog("win-thread[%03d] done! (%03d/%03d)\n",
			     win->id,
			     manager.thread_test.done,
			     manager.win_num);
		}
	}
	if (manager.win_num == manager.thread_test.done) {
		vlog("all done. sending signal to main thread \n");
		// done. send signal to main thread.
		manager.status = WIN_MGR_STATUS_STOP;
		kill(getpid(), SIGTERM);
	} else if (!win->tests.all_done) {
		// TODO. unlock mutex since bellow operations will block
		// other threads.
		// recreate this thread to run test again.
		dlog("pthread_join \n");
		//pthread_mutex_unlock(&manager.mutex);
		int rc = pthread_join(win->tid, NULL);
		//pthread_mutex_lock(&manager.mutex);
		if (rc) {
			win_manager_handle_error(NULL, rc, "pthread_join");
			return;
		}

		struct win w = { 0 };
		w.id = win->id;
		w.tests.egl_surf.size[0] = options.win_size[0];
		w.tests.egl_surf.size[1] = options.win_size[1];
		w.tests.thread.iterator = win->tests.thread.iterator;
		*win = w;

		vlog("recreate thread \n");
		rc = pthread_create(&win->tid, NULL, win_thread, (void *)win);
		assert(rc == 0);
		if(rc) {
			win->tid = 0;
			win_manager_handle_error(NULL, rc, "pthread_create");
		}
	}
	dlog("done \n");
}

static void *
win_manager_thread(void *arg)
{
	int i;
	dlog("win_manager_thread ...\n");
	pthread_setname_np(pthread_self(), "win-manager-thread");

	// create threads
	dlog("creating threads ...\n");
	manager.win_num = 0;
	for (i=0; i < options.thread_num; i++) {
		// init win test parameters.
		manager.win[i].id = i;
		manager.win[i].tests.egl_surf.size[0] = options.win_size[0];
		manager.win[i].tests.egl_surf.size[1] = options.win_size[1];
		int rc = pthread_create(&manager.win[i].tid,
				    NULL,
				    win_thread,
				    (void *)&manager.win[i]);
		if (rc) {
			pthread_mutex_lock(&manager.mutex);
			win_manager_handle_error(NULL, rc, "pthread_create");
			pthread_mutex_unlock(&manager.mutex);
			break;
		}
		manager.win_num ++;
	}
	dlog("created %d threads\n", manager.win_num);

	if ((options.test_flag & TEST_FLAG_IVI_SHELL) && winsys.ivi_app) {
		t_ilm_display* screenIDs;
		t_ilm_uint screenCount;
		ilmErrorTypes rc;
		rc = ilm_getScreenIDs(&screenCount, &screenIDs);
		dlog("ilm_getScreenIDs, rc = %d, screenCount=%d\n",
		     (int)rc, screenCount);

		pthread_mutex_lock(&manager.mutex);
		while(manager.ivi_ready_num != options.thread_num) {
			pthread_cond_wait(&manager.cond, &manager.mutex);
		}
		pthread_mutex_unlock(&manager.mutex);

		rc = ilm_displaySetRenderOrder((t_ilm_nativedisplay)screenIDs[0],
					       manager.ivi_layers,
					       options.thread_num);
		dlog("ilm_displaySetRenderOrder, rc = %d\n", (int)rc);

		rc = ilm_commitChanges();
		dlog("ilm_commitChanges, rc = %d\n", (int)rc);

		free(screenIDs);
	}

	int quit = manager.win_num == 0;
	while(!quit) {
		pthread_mutex_lock(&manager.mutex);
		while(!quit && !manager.notify_win) {
			pthread_cond_wait(&manager.cond, &manager.mutex);
		}
		struct win * win = manager.notify_win;
		while(win) {
			switch(win->notify) {
				case WIN_MGR_NOTIFY_STOP:
					manager.status = WIN_MGR_STATUS_STOP;
					break;
				case WIN_MGR_NOTIFY_THREAD_ERROR:
					win_manager_handle_error(win, 0, NULL);
					break;
				case WIN_MGR_NOTIFY_THREAD_DONE:
					assert(win);
					win_manager_handle_thread_done(win);
					break;
				default:
					assert(0 && "invalid notify received!");
					break;
			}
			win = win->notify_next;
		}
		manager.notify_win = NULL;
		quit = manager.status == WIN_MGR_STATUS_STOP;
		pthread_mutex_unlock(&manager.mutex);
	}

	dlog("join all threads...\n");
	for(i=0; i< manager.win_num; i++) {
		pthread_join(manager.win[i].tid, NULL);
	}
	dlog("join all threads done!\n");

	dlog("win_manager_thread done!\n");
	return NULL;
}

int
win_manager_init()
{
	int rc;
	dlog("win_manager_init ...\n");

	assert(options.thread_num >=1);
	manager.status = WIN_MGR_STATUS_RUN;
	manager.thread_test.done = 0;
	manager.notify_win = NULL;
	manager.win = calloc(options.thread_num, sizeof(struct win));
	if (!manager.win) {
		win_manager_handle_error(NULL, errno, "calloc");
		return -ENOMEM;
	}
	manager.ivi_ready_num = 0;
	manager.ivi_layers = calloc(options.thread_num, sizeof(unsigned));
	if (!manager.ivi_layers) {
		win_manager_handle_error(NULL, errno, "calloc");
		return -ENOMEM;
	}

	rc = pthread_create(&manager.tid, NULL, win_manager_thread, NULL);
	assert(rc == 0);
	if (rc) {
		elog("pthread_create failed");
		pthread_mutex_lock(&manager.mutex);
		win_manager_handle_error(NULL, rc, "pthread_create");
		pthread_mutex_unlock(&manager.mutex);
	}

	dlog("win_manager_init done!\n");
	return rc;
}

void
win_manager_fini()
{
	dlog("win_manager_fini ...\n");
	struct win dummy;
	win_manager_notify(&dummy, WIN_MGR_NOTIFY_STOP);
	pthread_join(manager.tid, NULL);

	if (manager.win) {
		free(manager.win);
		manager.win = NULL;
	}
	if (manager.ivi_layers) {
		free(manager.ivi_layers);
		manager.ivi_layers = NULL;
	}
	dlog("win_manager_fini done!\n");
}

int
win_manager_report()
{
	fprintf(stdout,
		"\n%u test, %u pass, %u skip, %u fail\n",
		manager.test_status.total,
		manager.test_status.pass,
		manager.test_status.skip,
		manager.test_status.fail);
	if (manager.error.str) {
		fprintf(stderr,
			"Error in %s , error code = %d\n",
			manager.error.str,
			manager.error.code);
	}
	return manager.error.code;
}
