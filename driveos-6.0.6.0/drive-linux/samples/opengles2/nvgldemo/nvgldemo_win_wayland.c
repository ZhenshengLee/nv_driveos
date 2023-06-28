/*
 * nvgldemo_win_wayland.c
 *
 * Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

//
// This file illustrates how to access the display and create a window for
//   a GL application using WAYLAND.
//

#include "nvgldemo.h"
#include "nvgldemo_win_wayland.h"

#ifdef NVGLDEMO_ENABLE_DMABUF
#include <drm_fourcc.h>
#if !defined(WAYLAND_PROTOCOLS_ADDPREFIX)
#include <linux-dmabuf-unstable-v1-client-protocol.h>
#include <linux-explicit-synchronization-unstable-v1-client-protocol.h>
#else
#include <wayland-linux-dmabuf-unstable-v1-client-protocol.h>
#include <wayland-linux-explicit-synchronization-unstable-v1-client-protocol.h>
#endif
#endif

static NvGlDemoCloseCB   closeCB   = NULL;
static NvGlDemoResizeCB  resizeCB  = NULL;
static NvGlDemoKeyCB     keyCB     = NULL;
static NvGlDemoPointerCB pointerCB = NULL;
static NvGlDemoButtonCB  buttonCB  = NULL;

static void
handle_ping(void *data, struct wl_shell_surface *wlShellSurface,
        uint32_t serial)
{
    wl_shell_surface_pong(wlShellSurface, serial);
}

static void
handle_configure(void *data, struct wl_shell_surface *shell_surface,
             uint32_t edges, int32_t width, int32_t height)
{
    struct Window *window = data;

    if (window->wlEGLNativeWindow) {
        wl_egl_window_resize(window->wlEGLNativeWindow, width, height, 0, 0);
    }

    window->geometry.width = width;
    window->geometry.height = height;

    if (!window->fullscreen) {
        window->window_size = window->geometry;
    }

    if(resizeCB) {
       resizeCB(width, height);
    }
}

static const struct wl_shell_surface_listener shell_surface_listener =
{
    handle_ping,
    handle_configure,
};

#ifdef ENABLE_IVI_SHELL
static void handle_ivi_surface_configure(void *data, struct ivi_surface *ivi_surface,
            int32_t width, int32_t height)
{
    struct Window *window = data;

    if (window->wlEGLNativeWindow) {
        wl_egl_window_resize(window->wlEGLNativeWindow, width, height, 0, 0);
    }

    window->geometry.width = width;
    window->geometry.height = height;

   if (!window->fullscreen)
       window->window_size = window->geometry;

    if(resizeCB) {
       resizeCB(width, height);
    }
}

static const struct ivi_surface_listener ivi_surface_listener = {
        handle_ivi_surface_configure,
};
#endif

static void
configure_callback(void *data, struct wl_callback *callback, uint32_t time)
{
    struct Window *window = data;

    wl_callback_destroy(callback);

    window->configured = 1;
}

static struct wl_callback_listener configure_callback_listener =
{
    configure_callback,
};

static void
toggle_fullscreen(struct Window *window, int fullscreen)
{

    struct wl_callback *callback;

    window->fullscreen = fullscreen;
    window->configured = 0;

    if (fullscreen) {
        wl_shell_surface_set_fullscreen(
            window->wlShellSurface,
            WL_SHELL_SURFACE_FULLSCREEN_METHOD_DEFAULT,0, NULL);
    } else {
        wl_shell_surface_set_toplevel(window->wlShellSurface);
        handle_configure(window, window->wlShellSurface, 0,
            window->window_size.width,
            window->window_size.height);
    }

    callback = wl_display_sync(window->display->wlDisplay);
    wl_callback_add_listener(callback, &configure_callback_listener,
        window);

}

static void
pointer_handle_enter(void *data, struct wl_pointer *pointer,
                     uint32_t serial, struct wl_surface *surface,
                     wl_fixed_t sx, wl_fixed_t sy)
{
}

static void
pointer_handle_leave(void *data, struct wl_pointer *pointer,
                     uint32_t serial, struct wl_surface *surface)
{
}

static void
pointer_handle_motion(void *data, struct wl_pointer *pointer,
                      uint32_t time, wl_fixed_t sx_w, wl_fixed_t sy_w)
{
    if (pointerCB) {
        float sx = wl_fixed_to_double(sx_w);
        float sy = wl_fixed_to_double(sy_w);
        pointerCB(sx, sy);
    }
}

static void
pointer_handle_button(void *data, struct wl_pointer *wl_pointer,
                      uint32_t serial, uint32_t time, uint32_t button,
                      uint32_t state)
{
#ifdef HAVE_LIBINPUT
    if (buttonCB) {
        buttonCB((button == BTN_LEFT) ? 1 : 0,
            (state == WL_POINTER_BUTTON_STATE_PRESSED) ? 1 : 0);
    }
#endif
}

static void
pointer_handle_axis(void *data, struct wl_pointer *wl_pointer,
                    uint32_t time, uint32_t axis, wl_fixed_t value)
{
}

static const struct wl_pointer_listener pointer_listener =
{
    pointer_handle_enter,
    pointer_handle_leave,
    pointer_handle_motion,
    pointer_handle_button,
    pointer_handle_axis,
};

static void
keyboard_handle_keymap(void *data, struct wl_keyboard *keyboard,
                       uint32_t format, int fd, uint32_t size)
{
#ifdef HAVE_XKBCOMMON
    struct Display *input = data;
    struct xkb_keymap *keymap;
    struct xkb_state *state;
    char *map_str;

    if (!data) {
        close(fd);
        return;
    }

    if (format != WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1) {
        close(fd);
        return;
    }

    map_str = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (map_str == MAP_FAILED) {
        close(fd);
        return;
    }

    keymap = xkb_map_new_from_string(input->xkb_context,
                map_str, XKB_KEYMAP_FORMAT_TEXT_V1, 0);
    munmap(map_str, size);
    close(fd);

    if (!keymap) {
        NvGlDemoLog("failed to compile keymap\n");
        return;
    }

    state = xkb_state_new(keymap);
    if (!state) {
        NvGlDemoLog("failed to create XKB state\n");
        xkb_map_unref(keymap);
        return;
    }

    xkb_keymap_unref(input->xkb.keymap);
    xkb_state_unref(input->xkb.state);
    input->xkb.keymap = keymap;
    input->xkb.state = state;

    input->xkb.control_mask =
        1 << xkb_map_mod_get_index(input->xkb.keymap, "Control");
    input->xkb.alt_mask =
        1 << xkb_map_mod_get_index(input->xkb.keymap, "Mod1");
    input->xkb.shift_mask =
        1 << xkb_map_mod_get_index(input->xkb.keymap, "Shift");
#else
    close(fd);
#endif
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
#ifdef HAVE_XKBCOMMON
    struct Display *input = data;
    uint32_t code, num_syms;
    const xkb_keysym_t *syms;
    xkb_keysym_t sym;
    input->serial = serial;
    code = key + 8;
    if (!input->xkb.state) {
        return;
    }

    num_syms = xkb_key_get_syms(input->xkb.state, code, &syms);

    sym = XKB_KEY_NoSymbol;
    if (num_syms == 1) {
        sym = syms[0];
    }

    if (keyCB) {
        if (sym) {
            char buf[16];
            xkb_keysym_to_utf8(sym, &buf[0], 16);
            keyCB(buf[0], (state == WL_KEYBOARD_KEY_STATE_PRESSED)? 1 : 0);
        }
    }
#endif
}

static void
keyboard_handle_modifiers(void *data, struct wl_keyboard *keyboard,
                          uint32_t serial, uint32_t mods_depressed,
                          uint32_t mods_latched, uint32_t mods_locked,
                          uint32_t group)
{
#ifdef HAVE_XKBCOMMON
    struct Display *input = data;
    xkb_mod_mask_t mask;

    /* If we're not using a keymap, then we don't handle PC-style modifiers */
    if (!input->xkb.keymap) {
        return;
    }

    xkb_state_update_mask(input->xkb.state, mods_depressed, mods_latched,
        mods_locked, 0, 0, group);

    mask = xkb_state_serialize_mods(input->xkb.state,
        XKB_STATE_DEPRESSED |
        XKB_STATE_LATCHED);
    input->modifiers = 0;

    if (mask & input->xkb.control_mask) {
        input->modifiers |= MOD_CONTROL_MASK;
    }
    if (mask & input->xkb.alt_mask) {
        input->modifiers |= MOD_ALT_MASK;
    }
    if (mask & input->xkb.shift_mask) {
        input->modifiers |= MOD_SHIFT_MASK;
    }
#endif
}

static const struct wl_keyboard_listener keyboard_listener =
{
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
    struct Display *d = data;

    if ((caps & WL_SEAT_CAPABILITY_POINTER) && !d->pointer) {
        d->pointer = wl_seat_get_pointer(seat);
        wl_pointer_add_listener(d->pointer, &pointer_listener, d);
    }

    if ((caps & WL_SEAT_CAPABILITY_KEYBOARD) && !d->keyboard) {
        d->keyboard = wl_seat_get_keyboard(seat);
        wl_keyboard_add_listener(d->keyboard, &keyboard_listener, d);
    }
}

static const struct wl_seat_listener seat_listener =
{
    seat_handle_capabilities,
};

#ifdef NVGLDEMO_ENABLE_DMABUF

static void
dmabuf_modifiers(void *data, struct zwp_linux_dmabuf_v1 *zwp_linux_dmabuf,
		 uint32_t format, uint32_t modifier_hi, uint32_t modifier_lo)
{
    struct Display *d = data;
    if (format == DRM_FORMAT_XRGB8888) {
        int i = 0;
        while (i < NVGLDEMO_MAX_DRM_MODIFIERS - 1 && d->formatModifiers_XRGB8[i] != 0xffffffffffffffff) {
            ++i;
        }
        if (i == NVGLDEMO_MAX_DRM_MODIFIERS - 1) {
            return;
        }
        d->formatModifiers_XRGB8[i] = ((uint64_t)(modifier_hi) << 32) | modifier_lo;
    }
    if (format == DRM_FORMAT_ARGB8888) {
        int i = 0;
        while (i < NVGLDEMO_MAX_DRM_MODIFIERS - 1 && d->formatModifiers_ARGB8[i] != 0xffffffffffffffff) {
            ++i;
        }
        if (i == NVGLDEMO_MAX_DRM_MODIFIERS - 1) {
            return;
        }
        d->formatModifiers_ARGB8[i] = ((uint64_t)(modifier_hi) << 32) | modifier_lo;
    }
}

static void
dmabuf_format(void *data, struct zwp_linux_dmabuf_v1 *zwp_linux_dmabuf, uint32_t format)
{
}

static const struct zwp_linux_dmabuf_v1_listener dmabuf_listener = {
	dmabuf_format,
	dmabuf_modifiers
};
#endif

// Registry handling static function
static void
registry_handle_global(void *data, struct wl_registry *registry,
               uint32_t name, const char *interface, uint32_t version)
{
    struct Display *d = data;

    if (strcmp(interface, "wl_compositor") == 0) {
        d->wlCompositor = wl_registry_bind(registry, name,
                        &wl_compositor_interface, 1);
    } else if (strcmp(interface, "wl_subcompositor") == 0) {
        d->wlSubcompositor = wl_registry_bind(registry, name,
                        &wl_subcompositor_interface, 1);
#ifdef NVGLDEMO_ENABLE_DMABUF
    } else if (strcmp(interface, "zwp_linux_dmabuf_v1") == 0) {
        d->wlDmabuf = wl_registry_bind(registry, name,
                        &zwp_linux_dmabuf_v1_interface, 3);
        memset(d->formatModifiers_XRGB8, 0xff, sizeof(d->formatModifiers_XRGB8));
        memset(d->formatModifiers_ARGB8, 0xff, sizeof(d->formatModifiers_ARGB8));
        zwp_linux_dmabuf_v1_add_listener(d->wlDmabuf, &dmabuf_listener, d);
    } else if (strcmp(interface, "zwp_linux_explicit_synchronization_v1") == 0) {
        d->wlExplicitSync = wl_registry_bind(registry, name,
                        &zwp_linux_explicit_synchronization_v1_interface, 1);
#endif
    } else if (strcmp(interface, "wl_shell") == 0) {
        d->wlShell = wl_registry_bind(registry, name,
                        &wl_shell_interface, 1);
    } else if (strcmp(interface, "wl_seat") == 0) {
        d->wlSeat = wl_registry_bind(registry, name,
                        &wl_seat_interface, 1);
        wl_seat_add_listener(d->wlSeat, &seat_listener, d);
#ifdef ENABLE_IVI_SHELL
    } else if (strcmp(interface, "ivi_application") == 0) {
        d->ivi_application = wl_registry_bind(registry, name,
                        &ivi_application_interface, 1);

        d->ivi_type = IVI_CONTROLLER;

      // weston 2.0 has ivi_controller_interface and weston 3.0 has ivi_wm_interface
      // Embedded Linux is still pointing to weston 2.0, so we need to support both
    } else if ((strcmp(interface, "ivi_wm") == 0) || (strcmp(interface, "ivi_controller") == 0)) {
        d->ivi_type = IVI_CONTROLLER;
    } else if (!strcmp(interface, "ivi_hmi_controller")) {
        d->ivi_type = HMI_CONTROLLER;
        // TODO: Add HMI controller related code.
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
signal_int(int signum)
{
    if (closeCB) {
        closeCB();
    }
}

//
// Window system startup/shutdown
//

int
NvGlDemoGetDisplayAttribs(EGLint** displayAttribs)
{
    // Nothing to do
    return 0;
}

// Initialize access to the display system
int
NvGlDemoDisplayInit(void)
{
    // Allocate a structure for the platform-specific state
    demoState.platform =
        (NvGlDemoPlatformState*)MALLOC(sizeof(NvGlDemoPlatformState));

    if (!demoState.platform) {
        NvGlDemoLog("Could not allocate platform specific storage memory.\n");
        goto fail;
    }

    MEMSET(demoState.platform, 0, sizeof(NvGlDemoPlatformState));

    demoState.platform->window.display = &demoState.platform->display;
    demoState.platform->display.window = &demoState.platform->window;

    // Open WAYLAND display
    demoState.platform->display.wlDisplay = wl_display_connect(NULL);
    if (demoState.platform->display.wlDisplay == NULL) {
        NvGlDemoLog("connect to a wayland socket failed.\n");
        goto fail;
    }

#ifdef HAVE_XKBCOMMON
    demoState.platform->display.xkb_context = xkb_context_new(0);
    if (demoState.platform->display.xkb_context == NULL) {
        NvGlDemoLog("Failed to create XKB context\n");
        goto fail;
    }
#endif

    demoState.platform->display.wlRegistry = wl_display_get_registry(
       demoState.platform->display.wlDisplay);
    if (demoState.platform->display.wlRegistry == NULL) {
        NvGlDemoLog("Failed to get registry.\n");
        goto fail;
    }

    wl_registry_add_listener(demoState.platform->display.wlRegistry,
       &registry_listener, &demoState.platform->display);

    wl_display_dispatch(demoState.platform->display.wlDisplay);

    demoState.nativeDisplay = (NativeDisplayType)demoState.platform->display.wlDisplay;
    demoState.platformType = NvGlDemoInterface_Wayland;

    // If display option is specified, but isn't supported, then exit.
    if (demoOptions.displayName[0]) {
        NvGlDemoLog("Setting display output is not supported. Exiting.\n");
        goto fail;
    }

    if (demoOptions.displayRate) {
        NvGlDemoLog("Setting display refresh is not supported. Exiting.\n");
        goto fail;
    }

    if ((demoOptions.displayBlend >= NvGlDemoDisplayBlend_None) ||
        (demoOptions.displayAlpha >= 0.0) ||
        (demoOptions.displayColorKey[0] >= 0.0) ||
        (demoOptions.displayLayer > 0)) {
        NvGlDemoLog("Display layers are not supported. Exiting.\n");
        goto fail;
    }

    // But do not fail on the unsupported "displaysize"
    // option.
    // We will ignore it on WAYLAND.

    if (demoOptions.displaySize[0]) {
        NvGlDemoLog("Setting display size is not supported. Ignoring.\n");
    }

    demoState.platform->display.sigint.sa_handler = signal_int;
    sigemptyset(&demoState.platform->display.sigint.sa_mask);
    demoState.platform->display.sigint.sa_flags = SA_RESETHAND;
    sigaction(SIGINT, &demoState.platform->display.sigint, NULL);

    return 1;

    fail:
        NvGlDemoDisplayTerm();
    return 0;
}

// Terminate access to the display system
void
NvGlDemoDisplayTerm(void)
{

    if (!demoState.platform) {
        return;
    }

    if (demoState.platform->display.wlRegistry) {
        wl_registry_destroy(demoState.platform->display.wlRegistry);
    }

    if (demoState.platform->display.wlSeat) {
        wl_seat_destroy(demoState.platform->display.wlSeat);
    }

    if (demoState.platform->display.wlShell) {
        wl_shell_destroy(demoState.platform->display.wlShell);
    }

#ifdef ENABLE_IVI_SHELL
    if (demoState.platform->display.ivi_application) {
        ivi_application_destroy(demoState.platform->display.ivi_application);
    }
#endif

    if (demoState.platform->display.wlCompositor) {
        wl_compositor_destroy(demoState.platform->display.wlCompositor);
    }

    // Explicitly destroy pointer object as traditional(weston clients) way
    // to destroy them in seat_handle_capabilies (a wl_seat_listner) is not
    // triggered for reasons still to be discovered.
    if (demoState.platform->display.pointer) {
        wl_pointer_destroy(demoState.platform->display.pointer);
        demoState.platform->display.pointer = NULL;
    }

    // Explicitly destroy keyboard object as traditional(weston clients) way
    // to destroy them in seat_handle_capabilies (a wl_seat_listner) is not
    // triggered for reasons still to be discovered.
    if (demoState.platform->display.keyboard) {
        wl_keyboard_destroy(demoState.platform->display.keyboard);
        demoState.platform->display.keyboard = NULL;
    }

#ifdef HAVE_XKBCOMMON
    if (demoState.platform->display.xkb_context) {

        if (demoState.platform->display.xkb.keymap) {
            xkb_keymap_unref(demoState.platform->display.xkb.keymap);
        }

        if (demoState.platform->display.xkb.state) {
            xkb_state_unref(demoState.platform->display.xkb.state);
        }

        xkb_context_unref(demoState.platform->display.xkb_context);
    }
#endif

    if (demoState.platform->display.wlDisplay) {
        wl_display_flush(demoState.platform->display.wlDisplay);
        wl_display_disconnect(demoState.platform->display.wlDisplay);
        demoState.platform->display.wlDisplay = NULL;
    }

    FREE(demoState.platform);
    demoState.platform = NULL;
    demoState.platformType = NvGlDemoInterface_Unknown;
}

static int create_shell_surface(struct Window *window, struct Display *display)
{
    window->wlShellSurface = wl_shell_get_shell_surface(
            display->wlShell, window->wlSurface);

    if (window->wlShellSurface == NULL) {
        NvGlDemoLog("Failed to create wayland shell surface.\n");
        return 0;
    }

    wl_shell_surface_add_listener(window->wlShellSurface,
            &shell_surface_listener, window);

    return 1;
}

#ifdef ENABLE_IVI_SHELL
static int create_ivi_surface(struct Window *window, struct Display *display)
{
    // By default, assume IVI controller.
    uint32_t id_ivisurf = window->ivi_surfaceId;
    if (display->ivi_type == HMI_CONTROLLER) {
        id_ivisurf = window->ivi_surfaceId + (uint32_t)getpid();
    }

    window->ivi_surface = ivi_application_surface_create(display->ivi_application,
                                                         id_ivisurf,
                                                         window->wlSurface);

    if (window->ivi_surface == NULL) {
        NvGlDemoLog("Failed to create ivi_client_surface\n");
        return 0;
    }

    ivi_surface_add_listener(window->ivi_surface,
                           &ivi_surface_listener,
                            window);

    return 1;
}
#endif

// Create the window
int
NvGlDemoWindowInit(
    int *argc,
    char **argv,
    const char *appName)
{
    if (!demoState.platform) {
        return 0;
    }

    if (!demoState.platform->display.wlCompositor ||
        !(demoState.platform->display.wlShell
#ifdef ENABLE_IVI_SHELL
        || demoState.platform->display.ivi_application
#endif
    )){
        return 0;
    }

    // If not specified, use default window size
    if (!demoOptions.windowSize[0])
        demoOptions.windowSize[0] = NVGLDEMO_DEFAULT_WIDTH;
    if (!demoOptions.windowSize[1])
        demoOptions.windowSize[1] = NVGLDEMO_DEFAULT_HEIGHT;

    if (!demoOptions.surface_id)
        demoOptions.surface_id = NVGLDEMO_DEFAULT_SURFACE_ID;

    demoState.platform->window.fullscreen = demoOptions.fullScreen;

#ifdef ENABLE_IVI_SHELL
    demoState.platform->window.ivi_surfaceId = demoOptions.surface_id;
#endif

    demoState.platform->window.wlSurface  =
       wl_compositor_create_surface(demoState.platform->display.wlCompositor);
    if (demoState.platform->window.wlSurface == NULL) {
        NvGlDemoLog("Failed to create wayland surface.\n");
        goto fail;
    }

    if (demoOptions.windowOffsetValid == 1) {
        NvGlDemoLog("windowoffset not supported.\n");
        goto fail;
    }

// With IVI-shell+ivi-controller, wl_shell interface is also exposed, but if
// ivi_application is exposed, we need to choose that.
#ifdef ENABLE_IVI_SHELL
    if (demoState.platform->display.ivi_application) {
        if (!create_ivi_surface(&demoState.platform->window,
                                &demoState.platform->display)) {
            goto fail;
        }
    } else
#endif
    {
        if (demoState.platform->display.wlShell) {
            if (!create_shell_surface(&demoState.platform->window,
                                     &demoState.platform->display)) {
                goto fail;
            }
        } else {
            assert(!"Wayland application interface not found");
        }
    }

    demoState.platform->window.geometry.width = demoOptions.windowSize[0];
    demoState.platform->window.geometry.height = demoOptions.windowSize[1];
    demoState.platform->window.window_size = demoState.platform->window.geometry;
    demoState.platform->window.wlEGLNativeWindow =
        wl_egl_window_create(demoState.platform->window.wlSurface,
                             demoState.platform->window.geometry.width,
                             demoState.platform->window.geometry.height);

    if (demoState.platform->window.wlEGLNativeWindow == NULL) {
        NvGlDemoLog("Failed to create wayland EGL window.\n");
        goto fail;
    }

    if (demoState.platform->display.wlShell
#ifdef ENABLE_IVI_SHELL
        && !demoState.platform->display.ivi_application
#endif
        ) {
        toggle_fullscreen(&demoState.platform->window, demoState.platform->window.fullscreen);
    }
    demoState.nativeWindow = (NativeWindowType)demoState.platform->window.wlEGLNativeWindow;

    return 1;

    fail: NvGlDemoWindowTerm();

    return 0;
}

int NvGlDemoPostSwap(void)
{
    // Nothing to do
    return 1;
}

// Close the window
void NvGlDemoWindowTerm(void)
{
    if (!demoState.platform) {
        return;
    }

    /* Required, otherwise segfault in egl_dri2.c: dri2_make_current()
     * on eglReleaseThread(). */
    // Close the native window
    if (demoState.platform->window.wlEGLNativeWindow) {
        wl_egl_window_destroy(demoState.platform->window.wlEGLNativeWindow);
        demoState.platform->window.wlEGLNativeWindow = 0;
    }

    if (demoState.platform->window.wlShellSurface) {
        wl_shell_surface_destroy(demoState.platform->window.wlShellSurface);
        demoState.platform->window.wlShellSurface = 0;
    }

#ifdef ENABLE_IVI_SHELL
    if (demoState.platform->window.ivi_surface) {
        ivi_surface_destroy(demoState.platform->window.ivi_surface);
        demoState.platform->window.ivi_surface = 0;
    }
#endif

    if (demoState.platform->window.wlSurface) {
        wl_surface_destroy(demoState.platform->window.wlSurface);
        demoState.platform->window.wlSurface = 0;
    }

    if (demoState.platform->window.callback) {
        wl_callback_destroy(demoState.platform->window.callback);
        demoState.platform->window.callback = 0;
    }

    demoState.nativeWindow = 0;
}
//
// No Pixmap support
//

EGLNativePixmapType NvGlDemoPixmapCreate(
    unsigned int width,
    unsigned int height,
    unsigned int depth)
{
    NvGlDemoLog("Wayland pixmap functions not supported\n");
    return (EGLNativePixmapType)0;
}

void
NvGlDemoPixmapDelete(
    EGLNativePixmapType pixmap)
{
    NvGlDemoLog("Wayland pixmap functions not supported\n");
}

//
// Callback handling
//

void NvGlDemoSetCloseCB(NvGlDemoCloseCB cb)
{
    closeCB  = cb;
}

void NvGlDemoSetResizeCB(NvGlDemoResizeCB cb)
{
    resizeCB = cb;

    // If the callback is registered after initialization, call it now.
    if (demoState.platform) {
        resizeCB(demoState.platform->window.geometry.width,
                 demoState.platform->window.geometry.height);
    }
}

void NvGlDemoSetKeyCB(NvGlDemoKeyCB cb)
{
    keyCB = cb;
}

void NvGlDemoSetPointerCB(NvGlDemoPointerCB cb)
{
    pointerCB = cb;
}

void NvGlDemoSetButtonCB(NvGlDemoButtonCB cb)
{
    buttonCB = cb;
}

void NvGlDemoCheckEvents(void)
{
    if ((!demoState.platform) ||
        (!demoState.platform->display.wlDisplay) ||
        (wl_display_dispatch_pending(demoState.platform->display.wlDisplay) == -1)) {
            if (closeCB) {
                closeCB();
            }
        }
    }

void NvGlDemoWaitEvents(void)
{
    if ((!demoState.platform) ||
        (!demoState.platform->display.wlDisplay) ||
        (wl_display_dispatch(demoState.platform->display.wlDisplay) == -1)) {
            if (closeCB) {
                closeCB();
            }
        }
}

EGLBoolean NvGlDemoSwapInterval(EGLDisplay dpy, EGLint interval)
{
    return eglSwapInterval(dpy, interval);
}

