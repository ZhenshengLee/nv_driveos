/*
 * Copyright © 2012 Collabora, Ltd.
 * Copyright © 2020 NVIDIA Corporation.
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

// This test creates a surface for each blend mode and places it over a
// background surface. If IVI-shell is available, the surface order is then
// rotated preiodically. Results must be inspected visually so the test never
// exits.

#include "config.h"

#include <stdio.h>
#include <string.h>

#include "weston-test-client-helper.h"
#include "weston-test-dmabuf-helper.h"
#include "weston-blending-client-protocol.h"
#include "ivi-application-client-protocol.h"
#include "ivi-wm-client-protocol.h"
#include "ivi-test.h"

enum weston_blending_blend_mode all_blend_modes[] = {
	WESTON_BLENDING_BLEND_MODE_NONE,
	WESTON_BLENDING_BLEND_MODE_PREMULT,
	WESTON_BLENDING_BLEND_MODE_COVERAGE
};

#define NUM_BLEND_MODES (sizeof(all_blend_modes) / sizeof(all_blend_modes[0]))

struct test_interfaces {
	bool use_ivishell;
	bool use_dmabuf;

	struct weston_blending *blending;
	struct wl_subcompositor *subcompositor;
	struct ivi_application *iviapp;
	struct ivi_wm *iviwm;
	struct wl_shell *wl_shell;
	struct wl_output *output;
	struct zwp_linux_dmabuf_v1 *dmabuf;

	uint64_t nv12_modifier;
	struct dma_buffer_allocator *dmabuf_allocator;
};

struct test_buffer {
	struct wl_buffer *buf;
	struct buffer *shmbuf;
	struct dma_buffer *dmabuf;
};

// A number of subsurfaces are created to visually test each blend mode
struct compound_surface {
	struct wl_surface *parent;
	struct test_buffer *parent_buf;
	struct ivi_surface *parent_ivi_surf;
	struct wl_shell_surface *parent_shell_surf;

	struct wl_surface *child[NUM_BLEND_MODES];
	struct wl_subsurface *sub[NUM_BLEND_MODES];
	struct test_buffer *buffers[NUM_BLEND_MODES];
	struct ivi_surface *ivi_surf[NUM_BLEND_MODES];

	uint32_t parent_surface_id;
	uint32_t child_surface_ids[NUM_BLEND_MODES];
};

static void
get_interfaces(struct client *client, struct test_interfaces *ifcs)
{
	struct global *g;
	struct weston_blending *blending;

	wl_list_for_each (g, &client->global_list, link) {
		if (strcmp(g->interface, "weston_blending") == 0) {
			if (ifcs->blending)
				assert(0 && "multiple weston_blending objects");
			assert(g->version == 1);
			ifcs->blending =
				wl_registry_bind(client->wl_registry,
						 g->name,
						 &weston_blending_interface,
						 1);
		}
		if (strcmp(g->interface, "wl_subcompositor") == 0) {
			if (ifcs->subcompositor)
				assert(0 &&
				       "multiple wl_subcompositor objects");
			assert(g->version == 1);
			ifcs->subcompositor =
				wl_registry_bind(client->wl_registry,
						 g->name,
						 &wl_subcompositor_interface,
						 1);
		}
		if (strcmp(g->interface, "ivi_application") == 0) {
			if (ifcs->iviapp)
				assert(0 &&
				       "multiple ivi_application objects");
			assert(g->version == 1);
			ifcs->iviapp =
				wl_registry_bind(client->wl_registry,
						 g->name,
						 &ivi_application_interface,
						 1);
		}
		if (strcmp(g->interface, "ivi_wm") == 0) {
			if (ifcs->iviwm)
				assert(0 &&
				       "multiple ivi_wm objects");
			assert(g->version == 1);
			ifcs->iviwm =
				wl_registry_bind(client->wl_registry,
						 g->name,
						 &ivi_wm_interface,
						 1);
		}
		if (strcmp(g->interface, "wl_shell") == 0) {
			ifcs->wl_shell = wl_registry_bind(client->wl_registry,
						g->name,
						&wl_shell_interface,
						1);
		}
		if (strcmp(g->interface, "wl_output") == 0) {
			if (ifcs->output) {
				continue;
			}
			assert(g->version == 3);
			ifcs->output =
				wl_registry_bind(client->wl_registry,
						 g->name,
						 &wl_output_interface,
						 3);
		}
		if (strcmp(g->interface, "zwp_linux_dmabuf_v1") == 0) {
			if (ifcs->output) {
				continue;
			}
			assert(g->version == 3);
			ifcs->dmabuf =
				wl_registry_bind(client->wl_registry,
						 g->name,
						 &zwp_linux_dmabuf_v1_interface,
						 3);
		}
	}

	assert(ifcs->blending && "no weston_blending found");
	assert(ifcs->subcompositor && "no wl_subcompositor found");

	if (ifcs->iviapp) {
		ifcs->use_ivishell = true;
		assert(ifcs->iviwm && "no ivi_wm found");
		assert(ifcs->output && "no wl_output found");
	} else {
		printf("no ivi_application found\n");
	}

	if (ifcs->dmabuf) {
		ifcs->use_dmabuf = true;
	} else {
		printf("no zwp_linux_dmabuf_v1 found\n");
	}
}

struct buffer *
create_test_shm_buffer(struct client *client, int w, int h, pixman_color_t color)
{
	struct buffer *buffer;
	pixman_image_t *solid;
	buffer = create_shm_buffer_a8r8g8b8(client, w, h);
	solid = pixman_image_create_solid_fill(&color);
	pixman_image_composite32(PIXMAN_OP_SRC,
				 solid, /* src */
				 NULL, /* mask */
				 buffer->image, /* dst */
				 0, 0, /* src x,y */
				 0, 0, /* mask x,y */
				 0, 0, /* dst x,y */
				 w, h);
	pixman_image_unref(solid);
	return buffer;
}

static struct test_buffer*
create_test_buffer(struct test_interfaces *ifcs,
		   struct client *client,
		   int w,
		   int h,
		   pixman_color_t color)
{
	struct test_buffer *buf;
	buf = malloc(sizeof(*buf));
	memset(buf, 0, sizeof(*buf));

	if (ifcs->use_dmabuf) {
		buf->dmabuf = malloc(sizeof(*buf->dmabuf));
		create_dmabuf_buffer(ifcs->dmabuf_allocator, buf->dmabuf, w, h, false, false, false);
		create_fbo_for_buffer(ifcs->dmabuf_allocator, buf->dmabuf, false);
		fill_dma_buffer_solid(buf->dmabuf, color);

		client_roundtrip(client);
		assert(buf->dmabuf->buffer);
		buf->buf = buf->dmabuf->buffer;
	} else {
		buf->shmbuf = create_test_shm_buffer(client, w, h, color);
		buf->buf = buf->shmbuf->proxy;
	}
	return buf;
}

static void
free_test_buffer(struct test_buffer* buf)
{
	if (buf->dmabuf) {
		free_dmabuf_buffer(buf->dmabuf);
		free(buf->dmabuf);
	}
	if (buf->shmbuf) {
		buffer_destroy(buf->shmbuf);
	}
	free(buf);
}

static void
handle_ping(void *data, struct wl_shell_surface *shell_surface,
	uint32_t serial)
{
	wl_shell_surface_pong(shell_surface, serial);
}

static void
handle_configure(void *data, struct wl_shell_surface *shell_surface,
	uint32_t edges, int32_t width, int32_t height)
{
}


static const struct wl_shell_surface_listener shell_surface_listener =
{
	handle_ping,
	handle_configure,
};

static int
populate_compound_surface(struct test_interfaces *ifcs,
			  struct compound_surface *com,
			  struct client *client,
			  int xoffset,
			  int yoffset,
			  int width,
			  int height,
			  uint32_t layer_id,
			  uint32_t *surface_id)
{
	int i;
	com->parent = wl_compositor_create_surface(client->wl_compositor);

	if (ifcs->use_ivishell) {
		com->parent_ivi_surf =
			ivi_application_surface_create(ifcs->iviapp,
						       *surface_id,
						       com->parent);
		ivi_wm_layer_add_surface(ifcs->iviwm, layer_id, *surface_id);
		ivi_wm_set_surface_visibility(ifcs->iviwm, *surface_id, true);
		ivi_wm_set_surface_source_rectangle(ifcs->iviwm, *surface_id, 0, 0, width, height);
		ivi_wm_set_surface_destination_rectangle(ifcs->iviwm, *surface_id, xoffset, yoffset, width, height);
		printf("Created surface %u\n", *surface_id);
		com->parent_surface_id = (*surface_id)++;
	} else {
		com->parent_shell_surf = wl_shell_get_shell_surface(ifcs->wl_shell, com->parent);
		if (com->parent_shell_surf == NULL) {
			printf("Failed to create wayland shell surface.\n");
		}
		wl_shell_surface_add_listener(com->parent_shell_surf, &shell_surface_listener, NULL);
		wl_shell_surface_set_toplevel(com->parent_shell_surf);
	}

	pixman_color_t parent_color = { 16384, 16384, 32768, 65535 };
	com->parent_buf = create_test_buffer(ifcs, client, width, height, parent_color);
	wl_surface_attach(com->parent, com->parent_buf->buf, 0, 0);
	wl_surface_damage(com->parent, 0, 0, width, height);
	wl_surface_commit(com->parent);

	assert(sizeof(all_blend_modes) / sizeof(all_blend_modes[0]) == NUM_BLEND_MODES);

	for (i = 0; i < NUM_BLEND_MODES; i++) {
		int x = (i + 2) * 40;
		int y = (i + 2) * 20;
		int w = width - (NUM_BLEND_MODES + 4) * 40;
		int h = height - (NUM_BLEND_MODES + 4) * 20;

		com->child[i] =
			wl_compositor_create_surface(client->wl_compositor);

		if (ifcs->use_ivishell) {
			com->ivi_surf[i] = ivi_application_surface_create(
				ifcs->iviapp,
				*surface_id,
				com->child[i]);

			ivi_wm_layer_add_surface(ifcs->iviwm, layer_id, *surface_id);
			ivi_wm_set_surface_visibility(ifcs->iviwm, *surface_id, true);
			ivi_wm_set_surface_source_rectangle(ifcs->iviwm, *surface_id, 0, 0, w, h);
			ivi_wm_set_surface_destination_rectangle(ifcs->iviwm, *surface_id, xoffset + x, yoffset + y, w, h);

			printf("Created surface %u\n", *surface_id);
			com->child_surface_ids[i] = (*surface_id)++;
		} else {
			com->sub[i] =
				wl_subcompositor_get_subsurface(ifcs->subcompositor,
								com->child[i],
								com->parent);

			/* Move the blended subsurfaces a little so they don't cover the
			 * parent entirely. */
			wl_subsurface_set_position(com->sub[i], x, y);
		}

		/* Give each a different blend mode. */
		weston_blending_set_surface_blend_mode(ifcs->blending,
						       com->child[i],
						       all_blend_modes[i]);

		/* Create, attach and commit a buffer to force the blend mode
		 * through the renderer. This will also show a visual result. */
		pixman_color_t color = { 32768, 0, 0, 32768 };
		com->buffers[i] = create_test_buffer(ifcs, client, w, h, color);
		wl_surface_attach(com->child[i], com->buffers[i]->buf, 0, 0);
		wl_surface_damage(com->child[i], 0, 0, w, h);
		wl_surface_commit(com->child[i]);
		wl_surface_commit(com->parent);
	}

	return layer_id;
}

static void free_compound_surface(struct compound_surface *com)
{
	free_test_buffer(com->parent_buf);
	for (int i = 0; i < NUM_BLEND_MODES; i++) {
		free_test_buffer(com->buffers[i]);
	}
}

TEST(test_blending_basic_protocol)
{
	struct test_interfaces ifcs = {0};
	struct client *client;
	struct compound_surface com1;
	struct compound_surface com2;
	struct ivi_wm_screen *screen;
	uint32_t layer_id = IVI_TEST_LAYER_ID(0);
	uint32_t surface_id = IVI_TEST_SURFACE_ID(0);
	char const *drm_render_node = "/dev/dri/card0";

	client = create_client();
	assert(client);
	get_interfaces(client, &ifcs);

	if (ifcs.use_dmabuf) {
		ifcs.dmabuf_allocator =
			create_dma_buffer_allocator(ifcs.dmabuf,
						    drm_render_node, false);
		client_roundtrip(client);
	}

	// Set up a single layer to put all surfaces on
	if (ifcs.use_ivishell) {
		int w = 1000;
		int h = 500;
		ivi_wm_create_layout_layer(ifcs.iviwm, layer_id, w, h);
		ivi_wm_set_layer_visibility(ifcs.iviwm, layer_id, true);
		ivi_wm_set_layer_source_rectangle(ifcs.iviwm, layer_id, 0, 0, w, h);
		ivi_wm_set_layer_destination_rectangle(ifcs.iviwm, layer_id, 0, 0, w, h);
		ivi_wm_layer_clear(ifcs.iviwm, layer_id);
		screen = ivi_wm_create_screen(ifcs.iviwm, ifcs.output);
		ivi_wm_screen_clear(screen);
		ivi_wm_screen_add_layer(screen, layer_id);
		printf("Created layer %u\n", layer_id);
	}

	// Create a background surface and a number of child surfaces with blending properties set.
	populate_compound_surface(&ifcs, &com1, client, 0, 0, 500, 500, layer_id, &surface_id);

	// Create a second set of surfaces, but using shared memory buffers
	ifcs.use_dmabuf = false;
	populate_compound_surface(&ifcs, &com2, client, 500, 0, 500, 500, layer_id, &surface_id);

	if (ifcs.use_ivishell) {
		ivi_wm_commit_changes(ifcs.iviwm);
	}

	client_roundtrip(client);

	printf("Verify that blending is happening.\n"
	       "There should be two sets of three red rectangles.\n"
	       "\tWithin each set,\n"
	       "\t- The first should be solid 50% dark red (no blending)\n"
	       "\t- The second should be 50% transparent 100% bright red (premultiplied alpha)\n"
	       "\t- The second should be 50% transparent 50% dark red (regular \"coverage\" alpha)\n"
	       "\tBoth sets should be identical. The left uses dma buffers "
	       "that can be assigned overlays while the right uses shared "
	       "memory buffers.\n"
	       "Press Ctrl-C to exit.\n");
	fflush(stdout);

	// Indefinitely cycle through surface ordering, re-creating the layer's
	// surface order, but with child surface order rotated.
	int rot = 0;
	for (int i = 0; i < 20 && ifcs.use_ivishell; ++i) {
		++rot;
		ivi_wm_layer_clear(ifcs.iviwm, layer_id);

		// Add com1 surfaces
		ivi_wm_layer_add_surface(ifcs.iviwm,
					 layer_id,
					 com1.parent_surface_id);
		for (int j = 0; j < NUM_BLEND_MODES; ++j) {
			uint32_t surface_id =
				com1.child_surface_ids[(j + rot) %
						       NUM_BLEND_MODES];
			ivi_wm_layer_add_surface(ifcs.iviwm,
						 layer_id,
						 surface_id);
		}

		// Add com2 surfaces
		ivi_wm_layer_add_surface(ifcs.iviwm,
					 layer_id,
					 com2.parent_surface_id);
		for (int j = 0; j < NUM_BLEND_MODES; ++j) {
			uint32_t surface_id =
				com2.child_surface_ids[(j + rot) %
						       NUM_BLEND_MODES];
			ivi_wm_layer_add_surface(ifcs.iviwm,
						 layer_id,
						 surface_id);
		}

		// Commit
		ivi_wm_commit_changes(ifcs.iviwm);
		client_roundtrip(client);
		usleep(200000);
	}
	if (!ifcs.use_ivishell) {
		usleep(3000000);
	}

	if (ifcs.use_ivishell) {
		ivi_wm_destroy_layout_layer(ifcs.iviwm, layer_id);
		ivi_wm_screen_destroy(screen);
	}

	free_compound_surface(&com1);

	if (ifcs.dmabuf_allocator) {
		free_dma_buffer_allocator(ifcs.dmabuf_allocator);
	}
}
