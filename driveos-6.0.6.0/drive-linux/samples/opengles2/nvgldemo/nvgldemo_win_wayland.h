/*
 * nvgldemo_win_wayland.h
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

// Exposes wayland display system objects created by nvgldemo to applications
//   which demonstrate wayland-specific features.

#ifndef __NVGLDEMO_WIN_WAYLAND_H
#define __NVGLDEMO_WIN_WAYLAND_H

#ifdef HAVE_LIBINPUT
#include <linux/input.h>
#endif
#include <unistd.h>
#include <sys/mman.h>
#include <signal.h>
#ifdef HAVE_XKBCOMMON
#include <xkbcommon/xkbcommon.h>
#endif
#include "wayland-client.h"
#include "wayland-egl.h"

#ifdef ENABLE_IVI_SHELL
#include "ivi-application-client-protocol.h"
#endif

#define MOD_SHIFT_MASK    0x01
#define MOD_ALT_MASK    0x02
#define MOD_CONTROL_MASK    0x04

struct Window;

enum ivi_interface_type {
    IVI_DISABLED = 0,
    HMI_CONTROLLER = 1,
    IVI_CONTROLLER = 2,
    IVI_COUNT
};

#define NVGLDEMO_MAX_DRM_MODIFIERS 32

struct Display {
    struct wl_display *wlDisplay;
    struct wl_registry *wlRegistry;
    struct wl_compositor *wlCompositor;
    struct wl_subcompositor *wlSubcompositor;
#ifdef NVGLDEMO_ENABLE_DMABUF
    struct zwp_linux_dmabuf_v1 *wlDmabuf;
    struct zwp_linux_explicit_synchronization_v1 *wlExplicitSync;
    uint64_t formatModifiers_XRGB8[NVGLDEMO_MAX_DRM_MODIFIERS];
    uint64_t formatModifiers_ARGB8[NVGLDEMO_MAX_DRM_MODIFIERS];
#endif
    struct wl_shell *wlShell;
#ifdef ENABLE_IVI_SHELL
    struct ivi_application *ivi_application;
    enum ivi_interface_type ivi_type;
#endif
    struct wl_seat *wlSeat;
    struct wl_pointer *pointer;
    struct wl_keyboard *keyboard;
#ifdef HAVE_XKBCOMMON
    struct xkb_context *xkb_context;
    struct {
        struct xkb_keymap *keymap;
        struct xkb_state *state;
        xkb_mod_mask_t control_mask;
        xkb_mod_mask_t alt_mask;
        xkb_mod_mask_t shift_mask;
    } xkb;
#endif
    uint32_t modifiers;
    uint32_t serial;
    struct sigaction sigint;
    struct Window *window;
};

struct Geometry {
    int width, height;
};

struct Window {
    struct Display *display;
    struct wl_egl_window *wlEGLNativeWindow;
    struct wl_surface *wlSurface;
    struct wl_shell_surface *wlShellSurface;
#ifdef ENABLE_IVI_SHELL
    struct ivi_surface *ivi_surface;
    unsigned int ivi_surfaceId;
#endif
    struct wl_callback *callback;
    int fullscreen, configured, opaque;
    struct Geometry geometry,window_size;
};

// Platform-specific state info
struct NvGlDemoPlatformState
{
    struct Display display;
    int screen;
    struct Window window;
};
#endif // __NVGLDEMO_WIN_WAYLAND_H
