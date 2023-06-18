/* 
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef WL_EGLSTREAM_CONTROLLER_SERVER_PROTOCOL_H
#define WL_EGLSTREAM_CONTROLLER_SERVER_PROTOCOL_H

#ifdef  __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include "wayland-server.h"

struct wl_client;
struct wl_resource;

struct wl_buffer;
struct wl_eglstream_controller;
struct wl_surface;

extern const struct wl_interface wl_eglstream_controller_interface;

#ifndef WL_EGLSTREAM_CONTROLLER_PRESENT_MODE_ENUM
#define WL_EGLSTREAM_CONTROLLER_PRESENT_MODE_ENUM
/**
 * wl_eglstream_controller_present_mode - Stream present mode
 * @WL_EGLSTREAM_CONTROLLER_PRESENT_MODE_DONT_CARE: Let the Server decide
 *	present mode
 * @WL_EGLSTREAM_CONTROLLER_PRESENT_MODE_FIFO: Use a fifo present mode
 * @WL_EGLSTREAM_CONTROLLER_PRESENT_MODE_MAILBOX: Use a mailbox mode
 *
 * - dont_care: Using this enum will tell the server to make its own
 * decisions regarding present mode.
 *
 * - fifo: Tells the server to use a fifo present mode. The decision to use
 * fifo synchronous is left up to the server.
 *
 * - mailbox: Tells the server to use a mailbox present mode.
 */
enum wl_eglstream_controller_present_mode {
	WL_EGLSTREAM_CONTROLLER_PRESENT_MODE_DONT_CARE = 0,
	WL_EGLSTREAM_CONTROLLER_PRESENT_MODE_FIFO = 1,
	WL_EGLSTREAM_CONTROLLER_PRESENT_MODE_MAILBOX = 2,
};
#endif /* WL_EGLSTREAM_CONTROLLER_PRESENT_MODE_ENUM */

#ifndef WL_EGLSTREAM_CONTROLLER_ATTRIB_ENUM
#define WL_EGLSTREAM_CONTROLLER_ATTRIB_ENUM
/**
 * wl_eglstream_controller_attrib - Stream consumer attachment attributes
 * @WL_EGLSTREAM_CONTROLLER_ATTRIB_PRESENT_MODE: Tells the server the
 *	desired present mode
 * @WL_EGLSTREAM_CONTROLLER_ATTRIB_FIFO_LENGTH: Tells the server the
 *	desired fifo length when the desired presenation_mode is fifo.
 *
 * - present_mode: Must be one of wl_eglstream_controller_present_mode.
 * Tells the server the desired present mode that should be used.
 *
 * - fifo_length: Only valid when the present_mode attrib is provided and
 * its value is specified as fifo. Tells the server the desired fifo length
 * to be used when the desired present_mode is fifo.
 */
enum wl_eglstream_controller_attrib {
	WL_EGLSTREAM_CONTROLLER_ATTRIB_PRESENT_MODE = 0,
	WL_EGLSTREAM_CONTROLLER_ATTRIB_FIFO_LENGTH = 1,
};
#endif /* WL_EGLSTREAM_CONTROLLER_ATTRIB_ENUM */

struct wl_eglstream_controller_interface {
	/**
	 * attach_eglstream_consumer - Create server stream and attach
	 *	consumer
	 * @wl_surface: wl_surface corresponds to the client surface
	 *	associated with newly created eglstream
	 * @wl_resource: wl_resource corresponding to an EGLStream
	 *
	 * Creates the corresponding server side EGLStream from the given
	 * wl_buffer and attaches a consumer to it.
	 */
	void (*attach_eglstream_consumer)(struct wl_client *client,
					  struct wl_resource *resource,
					  struct wl_resource *wl_surface,
					  struct wl_resource *wl_resource);
	/**
	 * attach_eglstream_consumer_attribs - List of attributes with
	 *	consumer attachment data
	 * @wl_surface: wl_surface corresponds to the client surface
	 *	associated with newly created eglstream
	 * @wl_resource: wl_resource corresponding to an EGLStream
	 * @attribs: Stream consumer attachment attribs
	 *
	 * It contains key-value pairs compatible with intptr_t type. A
	 * key must be one of wl_eglstream_controller_attrib enumeration
	 * values. What a value represents is attribute-specific.
	 * @since: 2
	 */
	void (*attach_eglstream_consumer_attribs)(struct wl_client *client,
						  struct wl_resource *resource,
						  struct wl_resource *wl_surface,
						  struct wl_resource *wl_resource,
						  struct wl_array *attribs);
};


#ifdef  __cplusplus
}
#endif

#endif
