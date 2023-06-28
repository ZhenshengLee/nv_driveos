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

/* Brief description.
 * This is the simple framework of multithread stress test for weston/wayland.
 * Basically the test will create some threads to run, each thread
 * create its own wayland window/surface/queue, run its own message
 * loop and do gles rendering.
 *
 * It can be used as a manual test, visually it will create some windows on
 * wayland screen and keep rendering something in each window.
 * when user slay it(Press Ctrl-C or kill), windows will disappear one by one
 * and application will exit normally if no error happenend.
 *
 * It also can be used as an auto test, run it as a stress test and wait
 * until deadlock happened. user can specify a timeout or N frames renderings,
 * when the app exits, it will print out message 'Passed'.
 * It's easy to write a script to run the app and detect status automatically.
 *
 * Usage :
 * multithread-test.weston [options]
 * options can be:
 * [-test-ivi]
 * [-threads=<val>]
 * [-timeout=<val>]
 * [-thread-loops=<val>]
 * [-win-loops=<val>]
 * [-egl-ctx-loops=<val>]
 * [-egl-surf-loops=<val>]
 * [-frame-limit=val]
 * [-size=wxh]
 * [-resize]
 *
 * Examples:
 * create 100 threads and set timeout to 5s:
 * multithread-test.weston -threads=100 -timeout=5

 * create 100 threads and set frame-limit to 15
 * multithread-test.weston -threads=100 -frame-limit=15

 * By default, thread num is 1, timeout & frame-limit is -1 (infinity).
 *
 * Check usage message for more details of command line options.
 */

/*
 * Internal threads design.
 * main-thread :
 *      It handle signals, it will stop other threads if it get one of
 *      signals: SIGINT, SIGTERM or SIGALRM.
 *      signals may come from user input, or from win-manager-thread.
 * win-manager-thread:
 *      It create all the win-threads and monitor/control them. If any thread
 *      report an error, it will save the error and try to stop all test threads.
 *      Once all the win-threads finish test, manager-thread will notice
 *      main-thread to exit by sending signal SIGTERM.
 * win-thread:
 *      It create windows and egl resources, do rendering/swapbuffer tests.
 *      If it finish all the tests, win-manager-thread may re-create a new thread
 *      if option thread-loops is specified with a positive value.
 *      If an error happened(may come from egl,wayland or system), then it
 *      will try to stop and report the error to win-manager-thread,
 *      win-manager-thread will try to stop all the win-threads and exit.
 *      Error will be reported when app exit.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include "multithread.h"
#include "ivi-application-client-protocol.h"
#include "ilm_common.h"

#define MAX_THREAD_NUM 256

static void registry_handle_global(void *data,
				   struct wl_registry *registry,
				   uint32_t id,
				   const char *interface,
				   uint32_t version);
static void registry_handle_global_remover(void *data,
					   struct wl_registry *registry,
					   uint32_t id);


// global vars
struct winsys winsys = {
	.wl_display = NULL,
	.wl_registry = NULL,
	.wl_compositor = NULL,
	.xdg_shell = NULL,
	.ivi_app = NULL,
	.egl_display = EGL_NO_DISPLAY,
};

struct options options = {
	.debug = 0,
	.verbose = 0,
	.test_flag = 0,
	.win_size = { 320, 200 },
	.thread_num = 1,
	.timeout = -1,     // seconds, -1 means infinity
	.frame_limit = -1, // max frames in rendering.
	.thread_loops = 1,
	.win_loops = 1,
	.egl_ctx_loops = 1,
	.egl_surf_loops = 1,
};

static const struct wl_registry_listener registry_listener = {
	registry_handle_global,
	registry_handle_global_remover
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
registry_handle_global(void *data,
		       struct wl_registry *registry,
		       uint32_t id,
		       const char *interface,
		       uint32_t version)
{
	struct winsys * s = data;
	dlog("Got a registry event for %s id %u\n", interface, id);
	if (strcmp(interface, "wl_compositor") == 0) {
		s->wl_compositor = wl_registry_bind(registry,
						 id,
						 &wl_compositor_interface,
						 1);
	}  else if (strcmp(interface, "zxdg_shell_v6") == 0) {
		s->xdg_shell = wl_registry_bind(registry, id,
					    &zxdg_shell_v6_interface, 1);
		zxdg_shell_v6_add_listener(s->xdg_shell, &xdg_shell_listener, s);
	} else if (strcmp(interface, "ivi_application") == 0) {
		s->ivi_app = wl_registry_bind(registry, id,
					      &ivi_application_interface, 1);

	}
}

static void
registry_handle_global_remover(void *data,
			       struct wl_registry *registry,
			       uint32_t id)
{
	dlog("Got a registry losing event for %u\n", id);
}

static int
winsys_egl_dpy_init(struct winsys * s)
{
	EGLint major, minor;
	s->egl_display = eglGetDisplay((EGLNativeDisplayType) s->wl_display);
	if (s->egl_display == EGL_NO_DISPLAY) {
		elog("Can't create egl display\n");
		return -1;
	}
	dlog("Created egl display\n");

	if (eglInitialize(s->egl_display, &major, &minor) != EGL_TRUE) {
		elog("Can't initialise egl display\n");
		return -1;
	}
	dlog("eglInitialize succeed! EGL major: %d, minor %d\n", major, minor);

	return 0;
}

static void
winsys_egl_dpy_fini(struct winsys * s)
{
	eglTerminate(s->egl_display);
	eglReleaseThread();
}


//
// main-event-thread
// there are some multi-thread issues if app operate fd (read/poll) directly.
// so to avoid lockup, do not use fd directly in multithread app and always
// call wayland APIs:
//     wl_display_dispatch
//     wl_display_round_trip
//
// A simple multi-thread model work like this:
//    main-event-thread:
//        while(1) {
//            ...
//            wl_display_dispatch(...);
//            ...
//        }
//    other-threads:
//        create its own event queue &
//        while(1) {
//            ...
//           wl_display_dispatch_queue_pending(...queue);
//            ...
//        }
// this model guarantee that each thread's handler will be called
// in its own thread context.
//
static void *
main_event_thread(void *arg)
{
	int rc = 0;
	struct winsys * ws = arg;
	struct wl_display * display = ws->wl_display;

	dlog("main_event_thread start\n");
	pthread_setname_np(pthread_self(), "main_event_thread");
	while(rc != -1) {
		// check exit signal
		int quit = *(volatile int *)&ws->status == -1;
		__asm__ __volatile__("": : :"memory");
		if (quit) {
			dlog("exiting...\n");
			break;
		}
		// Don't use wl_display_dispatch since there is no safe way
		// to exit from it.
		// wl_display_dispatch() on its own will wait for incoming
		// events from the server, but those events will be in response
		// of client requests. Most requests don't guarantee a response
		// from the server in a finite period of time, so we could
		// potentially block forever waiting for incoming events.
		//
		// wl_display_roundtrip() will in turn create a wl_display::sync
		// request, which is guaranteed to have a server response
		// in a finite period of time. Therefore, we would always
		// return from it and would be given the chance to break out
		// of the loop if needed for any reason.
		rc = wl_display_roundtrip(display);
	}
	dlog("main_event_thread exit\n");
	return NULL;
}

static int
winsys_init(struct winsys * s)
{
	s->wl_display = wl_display_connect(NULL);
	if (!s->wl_display) {
		elog("wl_display_connect failed\n");
		return -1;
	}
	dlog("connected to display\n");

	s->wl_registry = wl_display_get_registry(s->wl_display);
	if (!s->wl_registry) {
		elog("wl_display_get_registry failed\n");
		return -1;
	}
	wl_registry_add_listener(s->wl_registry, &registry_listener, &winsys);

	wl_display_dispatch(s->wl_display);
	wl_display_roundtrip(s->wl_display);

	if (!s->wl_compositor ||
	    (!s->xdg_shell && !s->ivi_app)) {
		elog("Failed to get wl_compositor/xdg_shell or ivi_application interface\n");
		return -1;
	}

	if (options.test_flag & TEST_FLAG_IVI_SHELL) {
		if (!s->ivi_app) {
			elog("Failed to get ivi_application interface\n"
			     "Make sure start weston with ivi-shell, \n"
			     "then set ivi-module=ivi-controller.so in weston.init\n");
			return -1;
		}
		ilmErrorTypes rc;
		rc = ilm_initWithNativedisplay((t_ilm_nativedisplay)s->wl_display);
		dlog("ilm_initWithNativedisplay, rc=%d\n", (int)rc);
		if (rc != ILM_SUCCESS) {
			elog("ilm_initWithNativedisplay failed\n");
			return -1;
		}
		wl_display_flush(s->wl_display);
	}

	if (winsys_egl_dpy_init(s)) {
		elog("winsys_egl_dpy_init failed\n");
		return -1;
	}
	int rc = pthread_create(&s->tid, NULL,
				main_event_thread, s);
	assert(rc == 0);
	if (rc) {
		elog("Failed to create pthread\n");
		return -1;
	}
	return 0;
}

static void
winsys_fini(struct winsys *s)
{
	winsys_egl_dpy_fini(s);

	if (s->tid) {
		// wait until main-event-thread exit.
		*(volatile int *)&s->status = -1;
		__asm__ __volatile__("": : :"memory");
		pthread_join(s->tid, NULL);
		s->tid = 0;
	}
	if (s->ivi_app) {
		assert(options.test_flag & TEST_FLAG_IVI_SHELL);
		ilm_destroy();
	}
	wl_registry_destroy(s->wl_registry);
	wl_compositor_destroy(s->wl_compositor);
	wl_display_flush(s->wl_display);
	wl_display_disconnect(s->wl_display);

	dlog("disconnected from display\n");
}

static void
usage()
{
	printf("Usage: multithread-test.weston [OPTIONS]\n\n"
	       "  -test-ivi             Run ivi-shell tests\n"
	       "  -size=<wxh>           Set window size\n"
	       "  -threads=<val>        Set thread number, max thread num is %d\n"
	       "  -timeout=<val>        Set timeout value in seconds,-1 means infinite.\n"
	       "  -frame-limit=<val>    Set max frame limite in rendering,-1 means infinite.\n"
	       "  -thread-loops=<val>   Set max thread create/destroy number in test,-1 means infinite.\n"
	       "  -win-loops=<val>      Set max window create/destroy number in test,-1 means infinite.\n"
	       "  -egl-ctx-loops=<val>  Set max egl context create/destroy number in test,-1 means infinite.\n"
	       "  -egl-surf-loops=<val> Set max egl surface create/destroy number in test,-1 means infinite.\n"
	       "  -resize               Enable wayland window resize test.\n"
	       "  -d                    Enable debug output\n"
	       "  -v                    Enable verbose output\n"
	       "  -h                    Show this help\n",
	       MAX_THREAD_NUM);
}

static int
parse_options(int argc, char **argv)
{
	int i;
	// parse options
	for (i = 1; i < argc; i++) {
		if (!strncmp(argv[i], "-d", strlen("-d"))) {
			options.debug = 1;
		}
		else if (!strncmp(argv[i], "-v", strlen("-v"))) {
			options.verbose = 1;
		}
		else if (!strncmp(argv[i], "-resize", strlen("-resize"))) {
			options.win_resize = 1;
		}
		else if (!strncmp(argv[i], "-size=", strlen("-size="))) {
			int w, h;
			if (sscanf(argv[i] + strlen("-size="),
				   "%dx%d",
				   &w, &h) != 2) {
				usage();
				return 1;
			}
			options.win_size[0] = w;
			options.win_size[1] = h;
		}
		else if (!strncmp(argv[i], "-threads=", strlen("-threads="))) {
			options.thread_num = atoi(argv[i] + strlen("-threads="));
			if (options.thread_num > MAX_THREAD_NUM) {
				options.thread_num = MAX_THREAD_NUM;
			}
		}
		else if (!strncmp(argv[i], "-test-ivi", strlen("-test-ivi"))) {
			options.test_flag |= TEST_FLAG_IVI_SHELL;
		}
		else if (!strncmp(argv[i], "-timeout=", strlen("-timeout="))) {
			options.timeout = atoi(argv[i] + strlen("-timeout="));
		}
		else if (!strncmp(argv[i], "-frame-limit=", strlen("-frame-limit="))) {
			options.frame_limit = atoi(argv[i] + strlen("-frame-limit="));
		}
		else if (!strncmp(argv[i], "-thread-loops=", strlen("-thread-loops="))) {
			options.thread_loops = atoi(argv[i] + strlen("-thread-loops="));
		}
		else if (!strncmp(argv[i], "-win-loops=", strlen("-win-loops="))) {
			options.win_loops = atoi(argv[i] + strlen("-win-loops="));
		}
		else if (!strncmp(argv[i], "-egl-ctx-loops=", strlen("-egl-ctx-loops="))) {
			options.egl_ctx_loops = atoi(argv[i] + strlen("-egl-ctx-loops="));
		}
		else if (!strncmp(argv[i], "-egl-surf-loops=", strlen("-egl-surf-loops="))) {
			options.egl_surf_loops = atoi(argv[i] + strlen("-egl-surf-loops="));
		}
		else {
			usage();
			return 1;
		}
	}
	return 0;
}

int main(int argc, char **argv)
{
	int rc = 0;

	rc = parse_options(argc, argv);
	if (rc) {
		return rc;
	}

	// Block all signals except SIGTSTP.
	// Leave SIGTSTP to default will allow user to do job control in terminal.
	sigset_t sigset;
	sigfillset(&sigset);
	sigdelset(&sigset, SIGTSTP);
	pthread_sigmask(SIG_BLOCK, &sigset, NULL);

	// Init wayland winsys
	dlog("winsys_init..\n");
	rc = winsys_init(&winsys);
	if (rc) {
		elog("winsys_init failed\n");
		return -1;
	}
	// create threads
	rc = win_manager_init();
	if (rc) {
		elog("win_manager_init failed\n");
		goto failed;
	}

	dlog("timeout = %d seconds \n", options.timeout);
	if (options.timeout > 0) {
		struct itimerval itval;
		itval.it_interval.tv_usec = 0;
		itval.it_interval.tv_sec = 0;
		itval.it_value.tv_usec = 0;
		itval.it_value.tv_sec = options.timeout;

		rc = setitimer(ITIMER_REAL, &itval, NULL);
		if (rc) {
			elog("settimer failed\n");
			goto failed;
		}
	}

	// main thread loop, waiting for signals to exit.
	siginfo_t siginfo;
	while (1) {
		rc = sigwaitinfo(&sigset, &siginfo);
		dlog("siginfo: si_signo = %d \n", siginfo.si_signo);
		if (rc == -1) {
			continue;
		}
		// exit if get SIGTERM or SIGINT
		if (siginfo.si_signo == SIGINT ||
		    siginfo.si_signo == SIGTERM ||
		    siginfo.si_signo == SIGALRM ) {
			rc = 0;
			break;
		}
	}

  failed:
	dlog("cleanup ... \n");

	dlog("win_threads_fini\n");
	win_manager_fini();

	dlog("winsys_fini");
	winsys_fini(&winsys);

	rc = win_manager_report();
	return rc;
}
