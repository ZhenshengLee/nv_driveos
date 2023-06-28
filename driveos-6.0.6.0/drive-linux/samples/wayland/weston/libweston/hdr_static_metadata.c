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

#include "weston-hdr-static-metadata-server-protocol.h"
#include "compositor.h"

#define SET_METADATA(x) surface->pending.hdr_metadata->x

static void
set_metadata(struct wl_client *client,
	     struct wl_resource *resource,
	     wl_fixed_t display_primary_r_x,
	     wl_fixed_t display_primary_r_y,
	     wl_fixed_t display_primary_g_x,
	     wl_fixed_t display_primary_g_y,
	     wl_fixed_t display_primary_b_x,
	     wl_fixed_t display_primary_b_y,
	     wl_fixed_t white_point_x,
	     wl_fixed_t white_point_y,
	     wl_fixed_t min_luminance,
	     wl_fixed_t max_luminance,
	     uint32_t max_cll,
	     uint32_t max_fall,
	     struct wl_resource *surface_resource)
{
	struct weston_surface *surface =
		wl_resource_get_user_data(surface_resource);
	if (!surface->pending.hdr_metadata) {
		surface->pending.hdr_metadata =
			zalloc(sizeof(*(surface->pending.hdr_metadata)));
		if (!surface->pending.hdr_metadata)
			return;
	}

	SET_METADATA(primaries.r.x) = wl_fixed_to_double(display_primary_r_x);
	SET_METADATA(primaries.r.y) = wl_fixed_to_double(display_primary_r_y);
	SET_METADATA(primaries.g.x) = wl_fixed_to_double(display_primary_g_x);
	SET_METADATA(primaries.g.y) = wl_fixed_to_double(display_primary_g_y);
	SET_METADATA(primaries.b.x) = wl_fixed_to_double(display_primary_b_x);
	SET_METADATA(primaries.b.y) = wl_fixed_to_double(display_primary_b_y);
	SET_METADATA(primaries.white_point.x) = wl_fixed_to_double(white_point_x);
	SET_METADATA(primaries.white_point.y) = wl_fixed_to_double(white_point_y);
	SET_METADATA(max_luminance) = wl_fixed_to_double(max_luminance);
	SET_METADATA(min_luminance) = wl_fixed_to_double(min_luminance);
	SET_METADATA(max_cll) = max_cll;
	SET_METADATA(max_fall) = max_fall;
}

static void
destroy_metadata(struct wl_client *client,
		 struct wl_resource *resource,
		 struct wl_resource *surface_resource)
{
	struct weston_surface *surface =
		wl_resource_get_user_data(surface_resource);
	free(surface->pending.hdr_metadata);
	surface->pending.hdr_metadata = NULL;
	wl_resource_destroy(resource);
}

static const struct hdr_surface_interface hdr_surface_metadata_implementation = {
	set_metadata,
	destroy_metadata,
};

static void bind_hdr_metadata(struct wl_client *client, void *data,
			      uint32_t version, uint32_t id)
{
	struct wl_resource *resource;
	resource = wl_resource_create(client,
				      &hdr_surface_interface,
				      version, id);
	if (!resource) {
		wl_client_post_no_memory(client);
		return;
	}

	wl_resource_set_implementation(resource,
				       &hdr_surface_metadata_implementation,
				       NULL, NULL);
}

WL_EXPORT int
weston_hdr_static_metadata_setup(struct weston_compositor *compositor)
{
	if (!wl_global_create(compositor->wl_display,
			      &hdr_surface_interface, 1,
			      compositor, bind_hdr_metadata))
		return -1;

	return 0;
}
