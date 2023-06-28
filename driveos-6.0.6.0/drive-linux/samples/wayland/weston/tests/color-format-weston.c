/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
/*
 * Based upon software provided under the following terms:
 *
 * Copyright © 2012 Intel Corporation
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
#include "shared/xalloc.h"

#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#include "shared/os-compatibility.h"
#include "weston-test-client-helper.h"

#define SURF_WIDTH 128
#define SURF_HEIGHT 128
#define SURF_SPACING 20
#define SWATCH_ROWS 3
#define SWATCH_COLUMNS 3

#define TRACE() \
	printf("TRACE: %d\n", __LINE__)

// The Wayland protocol does not technically support an NV24 format, but we still
// need to support one. WL_SHM_FORMAT_* values are equal to corresponding fourcc codes.
#if !defined(WL_SHM_FORMAT_NV24)
#define WL_SHM_FORMAT_NV24 \
	(((uint32_t)'N') | \
	 ((uint32_t)'V' << 8) | \
	 ((uint32_t)'2' << 16) | \
	 ((uint32_t)'4' << 24))
#endif

typedef void *RawImage;

struct Color
{
	float red;
	float green;
	float blue;
};

static size_t
get_buf_stride(uint32_t format)
{
	switch (format) {
	case WL_SHM_FORMAT_XRGB8888: return SURF_WIDTH * 4;
	case WL_SHM_FORMAT_ARGB8888: return SURF_WIDTH * 4;
	case WL_SHM_FORMAT_RGB565: return SURF_WIDTH * 2;
	case WL_SHM_FORMAT_YUV420: return SURF_WIDTH;
	case WL_SHM_FORMAT_NV12: return SURF_WIDTH;
	case WL_SHM_FORMAT_YUYV: return SURF_WIDTH * 2;
	case WL_SHM_FORMAT_NV16: return SURF_WIDTH;
	case WL_SHM_FORMAT_NV24: return SURF_WIDTH;
	}
	assert(!"Invalid format");
	return 0;
}

static size_t
get_buf_len(uint32_t format)
{
	switch (format) {
	    case WL_SHM_FORMAT_XRGB8888: return SURF_WIDTH * SURF_HEIGHT * 4;
	    case WL_SHM_FORMAT_ARGB8888: return SURF_WIDTH * SURF_HEIGHT * 4;
	    case WL_SHM_FORMAT_RGB565: return SURF_WIDTH * SURF_HEIGHT * 2;
	    case WL_SHM_FORMAT_YUV420: return SURF_WIDTH * SURF_HEIGHT + (SURF_WIDTH * SURF_HEIGHT / 2);
	    case WL_SHM_FORMAT_NV12: return SURF_WIDTH * SURF_HEIGHT + (SURF_WIDTH * SURF_HEIGHT / 2);
	    case WL_SHM_FORMAT_YUYV: return SURF_WIDTH * SURF_HEIGHT * 2;
	    case WL_SHM_FORMAT_NV16: return SURF_WIDTH * SURF_HEIGHT * 2;
	    case WL_SHM_FORMAT_NV24: return SURF_WIDTH * SURF_HEIGHT * 3;
	}
	assert(!"Invalid format");
	return 0;
}

static struct buffer *
create_shm_buffer(struct client *client, uint32_t format, RawImage *out_data)
{
	struct wl_shm *shm = client->wl_shm;
	struct buffer *buf;
	struct wl_shm_pool *pool;
	int fd;
	RawImage data;
	size_t bytes_pp;

	buf = xzalloc(sizeof *buf);

	buf->len = get_buf_len(format);

	fd = os_create_anonymous_file(buf->len);
	assert(fd >= 0);

	data = mmap(NULL, buf->len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (data == MAP_FAILED) {
		close(fd);
		assert(data != MAP_FAILED);
	}

	pool = wl_shm_create_pool(shm, fd, buf->len);
	buf->proxy = wl_shm_pool_create_buffer(pool, 0, SURF_WIDTH, SURF_HEIGHT,
					       get_buf_stride(format), format);
	wl_shm_pool_destroy(pool);
	close(fd);

	assert(buf->proxy);

	// We don't actually use a pixman image, because not all formats are supported.
	buf->image = NULL;

    *out_data = data;
	return buf;
}

static uint8_t
round_clamp(float x, uint8_t min_val, uint8_t max_val)
{
	int result;

	if (x <= (float)min_val) {
		return min_val;
	} else if (x >= (float)(max_val)) {
		return max_val;
	}
	return (uint8_t)(x + 0.5f);
}

static void
calculate_components(uint8_t* components, const struct Color *color, uint32_t format)
{
	float y, u, v;

	switch (format) {
	case WL_SHM_FORMAT_XRGB8888:
	case WL_SHM_FORMAT_ARGB8888:
		components[0] = 255;
		components[1] = round_clamp(color->red * 255.0f, 0, 255);
		components[2] = round_clamp(color->green * 255.0f, 0, 255);
		components[3] = round_clamp(color->blue * 255.0f, 0, 255);
		break;
	case WL_SHM_FORMAT_RGB565:
		components[0] = round_clamp(color->red * 31.0f, 0, 31);
		components[1] = round_clamp(color->green * 63.0f, 0, 63);
		components[2] = round_clamp(color->blue * 31.0f, 0, 31);
		break;
	case WL_SHM_FORMAT_YUV420:
	case WL_SHM_FORMAT_NV12:
	case WL_SHM_FORMAT_YUYV:
	case WL_SHM_FORMAT_NV16:
	case WL_SHM_FORMAT_NV24:
		components[0] = round_clamp( 16.0f +  65.481f * color->red + 128.553f * color->green +  24.966f * color->blue, 16, 235);
		components[1] = round_clamp(128.0f -  37.797f * color->red -  74.203f * color->green + 112.0f   * color->blue, 16, 240);
		components[2] = round_clamp(128.0f + 112.0f   * color->red -  93.786f * color->green -  18.214f * color->blue, 16, 240);
		break;
	default:
		assert(!"Unrecognized color format");
	}
}

static void
set_pixel(RawImage data, int x, int y, uint8_t *components, uint32_t format)
{
	uint8_t *data8;
	uint16_t *data16;
	uint32_t *data32;

	data8 = (uint8_t*)data;
	data16 = (uint16_t*)data;
	data32 = (uint32_t*)data;

	switch (format) {
	case WL_SHM_FORMAT_XRGB8888:
	case WL_SHM_FORMAT_ARGB8888:
		data32[y * SURF_WIDTH + x] = (components[0] << 24) | (components[1] << 16) | (components[2] << 8) | components[3];
		break;
	case WL_SHM_FORMAT_RGB565:
		data16[y * SURF_WIDTH + x] = (components[0] << 11) | (components[1] << 5) | components[2];
		break;
	case WL_SHM_FORMAT_YUV420:
		data8[y * SURF_WIDTH + x] = components[0];
		data8[SURF_WIDTH * SURF_HEIGHT + (y / 2) * (SURF_WIDTH / 2) + (x / 2)] = components[1];
		data8[SURF_WIDTH * SURF_HEIGHT + (SURF_WIDTH * SURF_HEIGHT / 4)  + (y / 2) * (SURF_WIDTH / 2) + (x / 2)] = components[2];
		break;
	case WL_SHM_FORMAT_NV12:
		data8[y * SURF_WIDTH + x] = components[0];
		data8[SURF_WIDTH * SURF_HEIGHT + ((y / 2) * (SURF_WIDTH / 2) + (x / 2)) * 2] = components[1];
		data8[SURF_WIDTH * SURF_HEIGHT + ((y / 2) * (SURF_WIDTH / 2) + (x / 2)) * 2 + 1] = components[2];
		break;
	case WL_SHM_FORMAT_YUYV:
		data8[(y * SURF_WIDTH + x) * 2] = components[0];
		data8[(((y * SURF_WIDTH + x) * 2) & ~0x3) + 1] = components[1];
		data8[(((y * SURF_WIDTH + x) * 2) & ~0x3) + 3] = components[2];
		break;
	case WL_SHM_FORMAT_NV16:
		data8[y * SURF_WIDTH + x] = components[0];
		data8[SURF_WIDTH * SURF_HEIGHT + (y * (SURF_WIDTH / 2) + (x / 2)) * 2] = components[1];
		data8[SURF_WIDTH * SURF_HEIGHT + (y * (SURF_WIDTH / 2) + (x / 2)) * 2 + 1] = components[2];
		break;
	case WL_SHM_FORMAT_NV24:
		data8[y * SURF_WIDTH + x] = components[0];
		data8[SURF_WIDTH * SURF_HEIGHT + (y * SURF_WIDTH + x) * 2] = components[1];
		data8[SURF_WIDTH * SURF_HEIGHT + (y * SURF_WIDTH + x) * 2 + 1] = components[2];
		break;
	default:
		assert(!"Unrecognized color format");
	}
}

static void
draw_swatch(RawImage data, const struct Color *color, int x_start, int y_start,
		int x_end, int y_end, uint32_t format)
{
	uint8_t components[4];
	int x, y;

	calculate_components(components, color, format);

	for (y = y_start; y < y_end; ++y) {
		for (x = x_start; x < x_end; ++x) {
			set_pixel(data, x, y, components, format);
		}
	}
}

static void
draw_swatches(RawImage buffer_data, uint32_t format)
{
	static const struct Color colors[SWATCH_ROWS * SWATCH_COLUMNS] = {
		//  Red   Green    Blue
		{ 0.1f, 0.1f, 0.1f },
		{ 0.9f, 0.9f, 0.9f },
		{ 0.9f, 0.1f, 0.1f },
		{ 0.1f, 0.9f, 0.1f },
		{ 0.5f, 0.5f, 0.5f },
		{ 0.9f, 0.9f, 0.1f },
		{ 0.1f, 0.1f, 0.9f },
		{ 0.9f, 0.1f, 0.9f },
		{ 0.1f, 0.9f, 0.9f },
	};

	int row, col;
	int x_start, y_start, x_end, y_end;
	int colorIndex;

	for (row = 0; row < SWATCH_ROWS; ++row) {
		y_start = row * SURF_HEIGHT / SWATCH_ROWS;
		y_end = (row + 1) * SURF_HEIGHT / SWATCH_ROWS;

		for (col = 0; col < SWATCH_COLUMNS; ++col) {
			colorIndex = row * SWATCH_COLUMNS + col;
			x_start = col * SURF_WIDTH / SWATCH_COLUMNS;
			x_end = (col + 1) * SURF_WIDTH / SWATCH_COLUMNS;
			draw_swatch(buffer_data, &colors[colorIndex], x_start, y_start, x_end, y_end, format);
		}
	}
}

struct client *
create_client_and_test_surface_format(int x, int y, uint32_t format, RawImage *out_data)
{
	struct client *client;
	struct surface *surface;

	client = create_client();

	/* initialize the client surface */
	surface = create_test_surface(client);
	client->surface = surface;

	surface->width = SURF_WIDTH;
	surface->height = SURF_HEIGHT;
	surface->buffer = create_shm_buffer(client, format, out_data);
	move_client(client, x, y);

	return client;
}

void draw_window_with_format(size_t index, uint32_t format)
{
	struct client *client;
	struct wl_surface *surface;
	struct buffer *buf;
	RawImage buffer_data;
	int frame;
	int x = 100 + (SURF_WIDTH + SURF_SPACING) * index;
	int y = 50;

	client = create_client_and_test_surface_format(x, y, format, &buffer_data);
	assert(client);

	surface = client->surface->wl_surface;
	buf = client->surface->buffer;

	draw_swatches(buffer_data, format);

	wl_surface_attach(surface, buf->proxy, 0, 0);
	wl_surface_damage(surface, 0, 0, SURF_WIDTH, SURF_HEIGHT);
	frame_callback_set(surface, &frame);
	wl_surface_commit(surface);
	frame_callback_wait_nofail(client, &frame);
}

TEST(color_formats)
{
	uint32_t formats[] = {
		WL_SHM_FORMAT_ARGB8888,
		WL_SHM_FORMAT_XRGB8888,
		WL_SHM_FORMAT_RGB565,
		WL_SHM_FORMAT_YUV420,
		WL_SHM_FORMAT_NV12,
		WL_SHM_FORMAT_YUYV,
		WL_SHM_FORMAT_NV16,
		WL_SHM_FORMAT_NV24,
	};

	// Test infrastructure clients seem to assume a single surface with a single format,
	// so use multiple clients for multiple formats.
	for (size_t iFormat = 0; iFormat < sizeof(formats) / sizeof(formats[0]); ++iFormat) {
		draw_window_with_format(iFormat, formats[iFormat]);
	}

	printf("Verify that all test patterns appear identical, subject to the\n"
			"limitations of the formats.  In particular, this test does not\n"
			"perform proper chrominance filtering for subsampled YUV formats,\n"
			"so edge artifacts are to be expected.\n"
			"Press Ctrl-C to exit.\n");
	fflush(stdout);
	(void)getchar();
}
