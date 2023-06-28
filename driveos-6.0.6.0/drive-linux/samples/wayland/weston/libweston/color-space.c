/*
 ** Copyright © 2018 Intel Corporation
 ** Copyright _ 2022 NVIDIA Corporation
 **
 ** Permission is hereby granted, free of charge, to any person obtaining
 ** a copy of this software and associated documentation files (the
 ** "Software"), to deal in the Software without restriction, including
 ** without limitation the rights to use, copy, modify, merge, publish,
 ** distribute, sublicense, and/or sell copies of the Software, and to
 ** permit persons to whom the Software is furnished to do so, subject to
 ** the following conditions:
 **
 ** The above copyright notice and this permission notice (including the
 ** next paragraph) shall be included in all copies or substantial
 ** portions of the Software.
 **
 ** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 ** EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 ** MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 ** NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 ** BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 ** ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 ** CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 ** SOFTWARE.
 **/
#include "weston-color-space-server-protocol.h"
#include "compositor.h"

static void
set_colorspace(struct wl_client *client,
	       struct wl_resource *resource,
	       struct wl_resource *surface_resource,
	       uint32_t color_space)
{
	struct weston_surface *surface =
		wl_resource_get_user_data(surface_resource);

	surface->pending.color_space = (enum weston_color_space) color_space;
}

static void
destroy_color_space(struct wl_client *client,
		    struct wl_resource *resource)
{
	wl_resource_destroy(resource);
}

static const struct color_space_interface color_space_implementation = {
	set_colorspace,
	destroy_color_space,
};

static void bind_colorspace(struct wl_client *client, void *data,
			    uint32_t version, uint32_t id)
{
	struct wl_resource *resource;
	resource = wl_resource_create(client,
				      &color_space_interface,
				      version, id);
	if (!resource) {
		wl_client_post_no_memory(client);
		return;
	}

	wl_resource_set_implementation(resource,
				       &color_space_implementation,
				       NULL, NULL);
}

WL_EXPORT int
weston_colorspace_setup(struct weston_compositor *compositor)
{
	if (!wl_global_create(compositor->wl_display,
			      &color_space_interface, 1,
			      compositor, bind_colorspace))
		return -1;

	return 0;
}
