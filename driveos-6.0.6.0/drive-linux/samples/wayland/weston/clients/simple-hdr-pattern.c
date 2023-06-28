/*
 * Copyright _ 2022 NVIDIA Corporation
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
#include <math.h>
#include <assert.h>
#include <sys/mman.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>

#include <linux/input.h>
#include <wayland-client.h>

#include <xf86drm.h>
#include <drm_fourcc.h>

#include <gbm.h>

#include "weston-hdr-static-metadata-client-protocol.h"
#include "weston-color-space-client-protocol.h"
#include "linux-dmabuf-unstable-v1-client-protocol.h"

#include "shared/os-compatibility.h"
#include "shared/helpers.h"
#include "shared/platform.h"
#include "shared/xalloc.h"
#include "shared/zalloc.h"
#include "window.h"

#define NUM_BUFFERS 3
#define MAX_PLANES 3

#ifndef DRM_FORMAT_MOD_LINEAR
#define DRM_FORMAT_MOD_LINEAR 0
#endif

#define ARRAY_LEN(_arr) (sizeof(_arr)/sizeof((_arr)[0]))

static int32_t option_help;
static int32_t option_fullscreen;
static int32_t option_sdr;
static int32_t option_width = 3840;
static int32_t option_height = 2160;
static int32_t option_count = 100;
static int32_t option_testCase = 1;
static char *option_format;

static const struct weston_option options[] = {
	{ WESTON_OPTION_BOOLEAN, "fullscreen", 'f', &option_fullscreen },
	{ WESTON_OPTION_BOOLEAN, "sdr", 's', &option_sdr },
	{ WESTON_OPTION_INTEGER, "width", 'w', &option_width },
	{ WESTON_OPTION_INTEGER, "height", 'h', &option_height },
	{ WESTON_OPTION_INTEGER, "count", 'c', &option_count },
	{ WESTON_OPTION_INTEGER, "testCase", 't', &option_testCase },
	{ WESTON_OPTION_STRING, "format", 'F', &option_format },
	{ WESTON_OPTION_BOOLEAN, "help", 'H', &option_help },
};

static const char help_text[] =
"Usage: %s [options]\n"
"\n"
"  -f, --fullscreen\t\tRun in fullscreen mode\n"
"  -s, --sdr\t\tRun with no metadata\n"
"  -w, --width\t\twidth of window\n"
"  -h, --height\t\theight of window\n"
"  -c, --count\t\tno of frames to run\n"
"  -t, --testCase\t\t test case number to run. FP16 and RGB10 formats have 4 test cases, RGB8 has 1\n"
"  -F, --format\t\tsurface format\n"
"  -H, --help\t\tShow this help text\n"
"\n";

struct app;
struct buffer;
int count = 0;
int caseNumber = 0;

struct buffer {
	struct wl_buffer *buffer;
	int busy;

	int drm_fd;
	int dmabuf_fd;
	uint8_t *mmap;

	int width;
	int height;
	unsigned long stride;
	int format;
	int offset[MAX_PLANES];

	struct gbm_device *gbm;
	int cpp;
	struct gbm_bo *bo;
};

struct app {
	struct display *display;
	struct window *window;
	struct widget *widget;

	struct color_space *colorspace;
	struct hdr_surface *hdr_surface;

	struct zwp_linux_dmabuf_v1 *dmabuf;
	struct buffer buffers[NUM_BUFFERS];
};

struct metadata {
	double rx;
	double ry;
	double gx;
	double gy;
	double bx;
	double by;
	double wx;
	double wy;
	double maxl;
	double minl;
	uint32_t maxCLL;
	uint32_t maxFALL;
};

struct metadata hdr10Default =
{
	// DCI-P3 color primaries
	.rx = 0.680,
	.ry = 0.320,
	.gx = 0.265,
	.gy = 0.690,
	.bx = 0.150,
	.by = 0.060,
	// D65 white point
	.wx = 0.3127,
	.wy = 0.3290,

	.maxl = 1000,
	.minl = 0.03,
	.maxCLL = 1000,
	.maxFALL = 400,
};

struct metadata hdr10Double =
{
	// DCI-P3 color primaries
	.rx = 0.680,
	.ry = 0.320,
	.gx = 0.265,
	.gy = 0.690,
	.bx = 0.150,
	.by = 0.060,
	// D65 white point
	.wx = 0.3127,
	.wy = 0.3290,

	.maxl = 2000,
	.minl = 0.03,
	.maxCLL = 2000,
	.maxFALL = 800,
};

struct metadata hdr10Half =
{
	// DCI-P3 color primaries
	.rx = 0.680,
	.ry = 0.320,
	.gx = 0.265,
	.gy = 0.690,
	.bx = 0.150,
	.by = 0.060,
	// D65 white point
	.wx = 0.3127,
	.wy = 0.3290,

	.maxl = 500,
	.minl = 0.03,
	.maxCLL = 500,
	.maxFALL = 200,
};

struct metadata hdr10SdrRange =
{
	// DCI-P3 color primaries
	.rx = 0.680,
	.ry = 0.320,
	.gx = 0.265,
	.gy = 0.690,
	.bx = 0.150,
	.by = 0.060,
	// D65 white point
	.wx = 0.3127,
	.wy = 0.3290,

	.maxl = 80,
	.minl = 0.03,
	.maxCLL = 80,
	.maxFALL = 32,
};

struct metadata scRGBDefault =
{
	// sRGB color primaries
	.rx = 0.640,
	.ry = 0.330,
	.gx = 0.300,
	.gy = 0.600,
	.bx = 0.150,
	.by = 0.060,
	// D65 white point
	.wx = 0.3127,
	.wy = 0.3290,

	.maxl = 1000,
	.minl = 0.03,
	.maxCLL = 1000,
	.maxFALL = 400,
};

struct metadata scRGBDouble =
{
	// sRGB color primaries
	.rx = 0.640,
	.ry = 0.330,
	.gx = 0.300,
	.gy = 0.600,
	.bx = 0.150,
	.by = 0.060,
	// D65 white point
	.wx = 0.3127,
	.wy = 0.3290,

	.maxl = 2000,
	.minl = 0.03,
	.maxCLL = 2000,
	.maxFALL = 800,
};

struct metadata scRGBHalf =
{
	// sRGB color primaries
	.rx = 0.640,
	.ry = 0.330,
	.gx = 0.300,
	.gy = 0.600,
	.bx = 0.150,
	.by = 0.060,
	// D65 white point
	.wx = 0.3127,
	.wy = 0.3290,

	.maxl = 500,
	.minl = 0.03,
	.maxCLL = 500,
	.maxFALL = 200,
};

struct metadata scRGBSdrRange =
{
	// sRGB color primaries
	.rx = 0.640,
	.ry = 0.330,
	.gx = 0.300,
	.gy = 0.600,
	.bx = 0.150,
	.by = 0.060,
	// D65 white point
	.wx = 0.3127,
	.wy = 0.3290,

	.maxl = 80,
	.minl = 0.03,
	.maxCLL = 80,
	.maxFALL = 32,
};

typedef float (*transferFunctionPtr) (float);

static float PQOOTF(float e)
{
	float ep;

	if (e <= 0.0003024) {
		ep = 267.84 * e;
	} else {
		 ep = 1.099 * pow(59.5208 * e, 0.45) - 0.099;
	}

	return 100 * pow(ep, 2.4);
}

static float invPQEOTF(float fd)
{
	float m1 = (2610. / 16384);
	float m2 = (2523. / 4096) * 128;
	float c1 = (3424. / 4096);
	float c2 = (2413. / 4096) * 32;
	float c3 = (2392. / 4096) * 32;

	float y = fd / 10000;

	return pow(((c1 + (c2 * pow(y, m1))) / (1 + (c3 * pow(y, m1)))), m2);
}

// Encode linear color value with BT2100 PQ OETF.
// https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2100-2-201807-I!!PDF-E.pdf
static float PQOETF(float color)
{
	return invPQEOTF(PQOOTF(color));
}

static float hdr10DefaultTF(float color)
{
	return PQOETF(color / (10000 / hdr10Default.maxCLL));
}

static float hdr10DoubleTF(float color)
{
	return PQOETF(color / (10000 / hdr10Double.maxCLL));
}

static float hdr10HalfTF(float color)
{
	return PQOETF(color / (10000 / hdr10Half.maxCLL));
}

static float hdr10SdrRangeTF(float color)
{
	return PQOETF(color / (10000 / hdr10SdrRange.maxCLL));
}

static float scRGBLinearDefaultTF(float color)
{
	return color * (125. / (10000. / hdr10Default.maxCLL));
}

static float scRGBLinearDoubleTF(float color)
{
	return color * (125. / (10000. / hdr10Double.maxCLL));
}

static float scRGBLinearHalfTF(float color)
{
	return color * (125. / (10000. / hdr10Half.maxCLL));
}

float scRGBLinearSdrRangeTF(float color)
{
	return color * (125. / (10000. / hdr10SdrRange.maxCLL));
}

float identityTF(float color)
{
	return color;
}

struct testCases {
	char name[50];
	struct metadata *md;
	transferFunctionPtr tf;
	int colorSpace;
};

struct testCases hdr10TestCases [] = {
	{ "80 nit HDR10", &hdr10SdrRange, hdr10SdrRangeTF, COLOR_SPACE_WELL_KNOWN_COLOR_SPACE_BT2100_PQ},
	{ "500 nit HDR10", &hdr10Half, hdr10HalfTF, COLOR_SPACE_WELL_KNOWN_COLOR_SPACE_BT2100_PQ},
	{ "1000 nit HDR10", &hdr10Default, hdr10DefaultTF, COLOR_SPACE_WELL_KNOWN_COLOR_SPACE_BT2100_PQ},
	{ "2000 nit HDR10", &hdr10Double, hdr10DoubleTF, COLOR_SPACE_WELL_KNOWN_COLOR_SPACE_BT2100_PQ},
};

struct testCases scRGBTestCases [] = {
	{ "80 nit scRGB Linear", &scRGBSdrRange, scRGBLinearSdrRangeTF, COLOR_SPACE_WELL_KNOWN_COLOR_SPACE_SCRGB_LINEAR},
	{ "500 nit scRGB Linear", &scRGBHalf, scRGBLinearHalfTF, COLOR_SPACE_WELL_KNOWN_COLOR_SPACE_SCRGB_LINEAR},
	{ "1000 nit scRGB Linear", &scRGBDefault, scRGBLinearDefaultTF, COLOR_SPACE_WELL_KNOWN_COLOR_SPACE_SCRGB_LINEAR},
	{ "2000 nit scRGB Linear", &scRGBDouble, scRGBLinearDoubleTF, COLOR_SPACE_WELL_KNOWN_COLOR_SPACE_SCRGB_LINEAR},
};

struct testCases defaultTestCases [] = {
	{ "SDR", NULL, identityTF, 0},
};

static int
create_dmabuf_buffer(struct app *app, struct buffer *buffer,
		     int width, int height, int format);

static void
gbm_fini(struct buffer *my_buf);

static void
destroy_dmabuf_buffer(struct buffer *buffer)
{
	if (buffer->buffer) {
		wl_buffer_destroy(buffer->buffer);
		close(buffer->dmabuf_fd);
		gbm_bo_destroy(buffer->bo);
	}
}

static void
buffer_release(void *data, struct wl_buffer *buffer)
{
	struct buffer *mybuf = data;
	mybuf->busy = 0;
}

static const struct wl_buffer_listener buffer_listener = {
	buffer_release
};

static inline void
destroy_hdr_surface(struct app *app, struct wl_surface *surface)
{
	if (app->hdr_surface) {
		hdr_surface_destroy(app->hdr_surface, surface);
		app->hdr_surface = NULL;
	}
}

static struct buffer *
next_buffer(struct app *a)
{
	int i;

	for (i = 0; i < NUM_BUFFERS; i++)
		if (!a->buffers[i].busy)
			return &a->buffers[i];

	return NULL;
}

void drawPixelAtPoint(struct buffer *buffer,
		      const uint64_t color,
		      const uint64_t x,
		      const uint64_t y)
{
	uint8_t *pixel = NULL;
	uint32_t bytesPerPixel = 0;

	switch (buffer->format) {
		case DRM_FORMAT_ABGR16161616F:
		case DRM_FORMAT_XBGR16161616F:
			bytesPerPixel = 8;
			break;
		case DRM_FORMAT_ABGR2101010:
		case DRM_FORMAT_XBGR2101010:
			bytesPerPixel = 4;
			break;
		case DRM_FORMAT_ARGB8888:
			bytesPerPixel = 4;
			break;
	}
	pixel = (uint8_t *)buffer->mmap + x * bytesPerPixel + y * buffer->stride;

	switch (bytesPerPixel) {
		case 4:
			*((uint32_t *)pixel) = color;
			break;
		case 8:
			*((uint64_t *)pixel) = color;
			break;
		default:
			return;
	}
}

static inline uint64_t packComponent(uint16_t value, uint8_t offset)
{
	return ((uint64_t) value) << offset;
}

static uint64_t packARGBHelper(uint16_t alpha,
			       uint16_t red,
			       uint16_t green,
			       uint16_t blue,
			       int format)
{
	switch (format) {
		case DRM_FORMAT_ABGR16161616F:
		case DRM_FORMAT_XBGR16161616F:
			return packComponent(alpha, 48) |
			       packComponent(red, 0)    |
			       packComponent(green, 16) |
			       packComponent(blue, 32);
		case DRM_FORMAT_ABGR2101010:
		case DRM_FORMAT_XBGR2101010:
			return packComponent(alpha, 30) |
			       packComponent(red, 0)    |
			       packComponent(green, 10) |
			       packComponent(blue, 20);
		case DRM_FORMAT_ARGB8888:
			return packComponent(alpha, 24) |
			       packComponent(red, 16)   |
			       packComponent(green, 8)  |
			       packComponent(blue, 0);
	}
	return -1;
}

static inline uint16_t scaleFloat(float value, uint8_t width)
{
	if (width == 0) {
		return 0;
	}

	uint16_t max = (1 << width) - 1;
	return value >= 1.0f ? max : (value <= 0.0f ? 0 : (uint16_t)floor(value * max));
}

uint32_t as_uint(const float x) {
	return *(uint32_t *)&x;
}

static inline uint16_t halfFloat(float value)
{
	const uint32_t b = as_uint(value)+0x00001000;
	const uint32_t e = (b&0x7F800000)>>23;
	const uint32_t m = b&0x007FFFFF;
	return (b&0x80000000)>>16 | (e>112)*((((e-112)<<10)&0x7C00)|m>>13) | ((e<113)&(e>101))*((((0x007FF000+m)>>(125-e))+1)>>1) | (e>143)*0x7FFF;

}

/* input floats are in range [0.0 - 1.0] */
uint64_t packARGB(float alpha,
		  float red,
		  float green,
		  float blue,
		  int format)
{
	switch (format) {
		case DRM_FORMAT_ABGR16161616F:
		case DRM_FORMAT_XBGR16161616F:
			return packARGBHelper(halfFloat(alpha),
					      halfFloat(red),
					      halfFloat(green),
					      halfFloat(blue),
					      format);
		case DRM_FORMAT_ABGR2101010:
		case DRM_FORMAT_XBGR2101010:
			return packARGBHelper(scaleFloat(alpha, 2),
					      scaleFloat(red, 10),
					      scaleFloat(green, 10),
					      scaleFloat(blue, 10),
					      format);
		case DRM_FORMAT_ARGB8888:
			return packARGBHelper(scaleFloat(alpha, 8),
					      scaleFloat(red, 8),
					      scaleFloat(green, 8),
					      scaleFloat(blue, 8),
					      format);
	}
	return -1;
}

void drawSingleGradientBand(struct buffer *buffer,
			    uint32_t startRow,
			    uint32_t endRow,
			    float *color,
			    transferFunctionPtr pTf)
{
	uint32_t row = 0;
	uint32_t col = 0;
	uint64_t pixel = 0;

	float rowDelta = 1.0f / (endRow - startRow);
	float colDelta[3] = { 0.0f, 0.0f, 0.0f};
	float gradient[3] = { 1.0f, 1.0f, 1.0f};

	for (row = startRow; row < endRow; row++) {

		/* Start with gray scale */
		gradient[0] = 1.0f - (rowDelta * (row - startRow));
		gradient[1] = 1.0f - (rowDelta * (row - startRow));
		gradient[2] = 1.0f - (rowDelta * (row - startRow));

		colDelta[0] = gradient[0] * (1.0f - color[0]) / buffer->width;
		colDelta[1] = gradient[0] * (1.0f - color[1]) / buffer->width;
		colDelta[2] = gradient[0] * (1.0f - color[2]) / buffer->width;

		for (col = 0; col < buffer->width; col++) {
			pixel = packARGB(1.0f,
					 (*pTf)(gradient[0]),
					 (*pTf)(gradient[1]),
					 (*pTf)(gradient[2]),
					 buffer->format);
			drawPixelAtPoint(buffer, pixel, col, row);

			gradient[0] -= colDelta[0];
			gradient[1] -= colDelta[1];
			gradient[2] -= colDelta[2];
		}
	}
}

void drawBands(struct buffer *buffer, transferFunctionPtr pTf)
{
	uint32_t band = 0;
	uint32_t row = 0;
	const uint32_t bandHeight = buffer->height / 6;

	float colors[21] = { 0.0f, 1.0f, 1.0f, /* Cyan */
			     1.0f, 0.0f, 1.0f, /* Magenta */
			     1.0f, 1.0f, 0.0f, /* Yellow */
			     1.0f, 0.0f, 0.0f, /* Red */
			     0.0f, 1.0f, 0.0f, /* Green */
			     0.0f, 0.0f, 1.0f, /* Blue */
			     0.0f, 0.0f, 0.0f};/* Black */

	for (band = 0; band < 6; band++) {
		drawSingleGradientBand(buffer,
				       row,
				       row + bandHeight,
				       &colors[band * 3],
				       pTf);
		row += bandHeight;
	}

	if (row < buffer->height) {
		drawSingleGradientBand(buffer,
				       row,
				       buffer->height,
				       &colors[18],
				       pTf);
	}
}

static int
fill_buffer(struct buffer *buffer) {
	void *map_data = NULL;
	int32_t dst_stride;
	struct testCases *t;

	buffer->mmap = gbm_bo_map(buffer->bo, 0, 0, buffer->width, buffer->height,
				  GBM_BO_TRANSFER_WRITE, &dst_stride, &map_data);
	if (!buffer->mmap) {
		fprintf(stderr, "Unable to mmap buffer\n");
		return 0;
	}

	switch (buffer->format) {
		case DRM_FORMAT_XBGR2101010:
		case DRM_FORMAT_ABGR2101010:
			t = &hdr10TestCases[caseNumber];
			break;
		case DRM_FORMAT_XBGR16161616F:
		case DRM_FORMAT_ABGR16161616F:
			t = &scRGBTestCases[caseNumber];
			break;
		case DRM_FORMAT_ARGB8888:
			t = &defaultTestCases[caseNumber];
			break;
	}
	drawBands(buffer, t->tf);
	gbm_bo_unmap(buffer->bo, map_data);
	return 1;
}

void set_md_and_cs(struct app *app, struct wl_surface *surface, struct buffer *buffer)
{
	struct testCases *t;
	int max_cll, max_fall;
	wl_fixed_t rx, ry, gx, gy, bx, by, wx, wy, max_luma, min_luma;

	switch (buffer->format) {
		case DRM_FORMAT_ABGR16161616F:
		case DRM_FORMAT_XBGR16161616F:
			caseNumber = option_testCase > 4 ? 0 : option_testCase - 1;
			t = &scRGBTestCases[caseNumber];
			break;
		case DRM_FORMAT_ABGR2101010:
		case DRM_FORMAT_XBGR2101010:
			caseNumber = option_testCase > 4 ? 0 : option_testCase - 1;
			t = &hdr10TestCases[caseNumber];
			break;
		case DRM_FORMAT_ARGB8888:
			caseNumber = option_testCase > 1 ? 0 : option_testCase - 1;
			t = &defaultTestCases[caseNumber];
			break;
	}
	printf("test case : %s \n", t->name);
	if (!option_sdr && t->md != NULL) {
		printf("app sets metadata\n");
		max_cll = t->md->maxCLL;
		max_fall = t->md->maxFALL;
		rx = wl_fixed_from_double(t->md->rx);
		ry = wl_fixed_from_double(t->md->ry);
		gx = wl_fixed_from_double(t->md->gx);
		gy = wl_fixed_from_double(t->md->gy);
		bx = wl_fixed_from_double(t->md->bx);
		by = wl_fixed_from_double(t->md->by);
		wx = wl_fixed_from_double(t->md->wx);
		wy = wl_fixed_from_double(t->md->wy);
		max_luma = wl_fixed_from_double(t->md->maxl);
		min_luma = wl_fixed_from_double(t->md->minl);
		hdr_surface_set_metadata(
				app->hdr_surface,
				rx,
				ry,
				gx,
				gy,
				bx,
				by,
				wx,
				wy,
				min_luma,
				max_luma,
				max_cll,
				max_fall,
				surface);
		printf("app sets colorspace\n");
		color_space_set(app->colorspace,
			        surface,
				t->colorSpace);
	}
}

static void
redraw_handler(struct widget *widget, void *data)
{
	struct app *app = data;
	struct buffer *buffer;
	struct wl_buffer *wlbuffer;
	struct wl_surface *surface;

	surface = widget_get_wl_surface(widget);

	if (count == option_count) {
		display_exit(app->display);
		return;
	}
	if (count == (option_count - 1)) {
		if (!option_sdr) {
			destroy_hdr_surface(app, surface);
		}
	}
	if (count == 0) {
		printf("Preparing buffers\n");
		set_md_and_cs(app, surface, &app->buffers[0]);
		for (int i = 0; i < NUM_BUFFERS; i++) {
			fill_buffer(&app->buffers[i]);
		}
		printf("fill buffers done\n");
	}

	buffer = next_buffer(app);

	// If no free buffers available schedule redraw and return;
	if(!buffer) {
		widget_schedule_redraw(widget);
		return;
	}

	wlbuffer = buffer->buffer;

	wl_surface_attach(surface, wlbuffer, 0, 0);
	wl_surface_damage(surface, 0, 0, buffer->width, buffer->height);
	wl_surface_commit(surface);
	widget_schedule_redraw(widget);
	buffer->busy = 1;
	count++;
}

static void
dmabuf_modifiers(void *data, struct zwp_linux_dmabuf_v1 *zwp_linux_dmabuf,
		 uint32_t format, uint32_t modifier_hi, uint32_t modifier_lo)
{
}

static void
dmabuf_format(void *data, struct zwp_linux_dmabuf_v1 *zwp_linux_dmabuf, uint32_t format)
{
	/* XXX: deprecated */
}


static const struct zwp_linux_dmabuf_v1_listener dmabuf_listener = {
	dmabuf_format,
	dmabuf_modifiers
};

static void
global_handler(struct display *display, uint32_t id,
	       const char *interface, uint32_t version, void *data)
{

	struct app *app = data;

	if (strcmp(interface, "color_space") == 0) {
		app->colorspace =
			display_bind(display, id,
				     &color_space_interface, 1);
	} else if (strcmp(interface, "hdr_surface") == 0) {
		app->hdr_surface =
			display_bind(display, id,
				     &hdr_surface_interface, 1);
	} else if (strcmp(interface, "zwp_linux_dmabuf_v1") == 0) {
		if (version < 3)
			return;
		app->dmabuf =
			display_bind(display, id,
				     &zwp_linux_dmabuf_v1_interface, 3);
		zwp_linux_dmabuf_v1_add_listener(app->dmabuf,
						 &dmabuf_listener,
						 app);
	}

}

static void
global_handler_remove(struct display *display, uint32_t id,
		      const char *interface, uint32_t version, void *data)
{
}

static int
gbm_init(struct buffer *my_buf)
{
	my_buf->drm_fd = open("/dev/dri/card0", O_RDWR);
	if (my_buf->drm_fd < 0)
		return 0;

	if (!my_buf->gbm)
		my_buf->gbm = gbm_create_device(my_buf->drm_fd);

	if (!my_buf->gbm)
		return 0;

	return 1;
}

static void
gbm_fini(struct buffer *my_buf)
{
	gbm_device_destroy(my_buf->gbm);
	close(my_buf->drm_fd);
}

static int
create_dmabuf_buffer(struct app *app, struct buffer *buffer,
		     int width, int height, int format)
{
	struct zwp_linux_buffer_params_v1 *params;
	uint64_t modifier = DRM_FORMAT_MOD_LINEAR;
	uint32_t flags = 0;
	unsigned buf_w, buf_h;
	int pixel_format;

	memset(buffer, 0, sizeof(*buffer));
	if (!gbm_init(buffer)) {
		fprintf(stderr, "drm_connect failed\n");
		goto error;
	}

	buffer->width = width;
	buffer->height = height;
	buffer->format = format;

	switch (format) {
	case DRM_FORMAT_ARGB8888:
		pixel_format = DRM_FORMAT_ARGB8888;
		buf_w = width;
		buf_h = height;
		buffer->cpp = 1;
		break;
	case DRM_FORMAT_XBGR2101010:
		pixel_format = DRM_FORMAT_XBGR2101010;
		buf_w = width;
		buf_h = height;
		buffer->cpp = 1;
		break;
	case DRM_FORMAT_ABGR2101010:
		pixel_format = DRM_FORMAT_ABGR2101010;
		buf_w = width;
		buf_h = height;
		buffer->cpp = 1;
		break;
	case DRM_FORMAT_XBGR16161616F:
		pixel_format = DRM_FORMAT_XBGR16161616F;
		buf_w = width;
		buf_h = height;
		buffer->cpp = 1;
		break;
	case DRM_FORMAT_ABGR16161616F:
		pixel_format = DRM_FORMAT_ABGR16161616F;
		buf_w = width;
		buf_h = height;
		buffer->cpp = 1;
		break;
	default:
		buffer->height = height;
		buffer->cpp = 1;
	}

	buffer->bo = gbm_bo_create(buffer->gbm, buf_w, buf_h, pixel_format, GBM_BO_USE_LINEAR);
	if (!buffer->bo) {
		fprintf(stderr, "error: unable to allocate bo\n");
		goto error1;
	}

	buffer->dmabuf_fd = gbm_bo_get_fd(buffer->bo);
	buffer->stride = gbm_bo_get_stride(buffer->bo);
	buffer->offset[0] = gbm_bo_get_offset(buffer->bo , 0);

	if (buffer->dmabuf_fd < 0) {
		fprintf(stderr, "error: dmabuf_fd < 0\n");
		goto error2;
	}

	params = zwp_linux_dmabuf_v1_create_params(app->dmabuf);
	zwp_linux_buffer_params_v1_add(params,
				       buffer->dmabuf_fd,
				       0, /* plane_idx */
				       buffer->offset[0], /* offset */
				       buffer->stride,
				       modifier >> 32,
				       modifier & 0xffffffff);

	buffer->buffer = zwp_linux_buffer_params_v1_create_immed(params,
								 buffer->width,
								 buffer->height,
								 format,
								 flags);
	wl_buffer_add_listener(buffer->buffer, &buffer_listener, buffer);

	return 0;

error2:
	gbm_bo_destroy(buffer->bo);
error1:
	gbm_fini(buffer);
error:
	return -1;
}

static char* format_to_string(int format)
{
	switch (format) {
		case DRM_FORMAT_ARGB8888:
			return "DRM_FORMAT_ARGB8888";
		case DRM_FORMAT_XBGR2101010:
			return "DRM_FORMAT_XBGR2101010";
		case DRM_FORMAT_ABGR2101010:
			return "DRM_FORMAT_ABGR2101010";
		case DRM_FORMAT_XBGR16161616F:
			return "DRM_FORMAT_XBGR16161616F";
		case DRM_FORMAT_ABGR16161616F:
			return "DRM_FORMAT_ABGR16161616F";
		default:
			return "Undefined";
	}
}

static uint32_t string_to_format(char *format)
{
	if (format) {
		if (!strcmp(format, "DRM_FORMAT_ARGB8888")) {
			return DRM_FORMAT_ARGB8888;
		} else if (!strcmp(format, "DRM_FORMAT_XBGR2101010")) {
			return DRM_FORMAT_XBGR2101010;
		} else if (!strcmp(format, "DRM_FORMAT_ABGR2101010")) {
			return DRM_FORMAT_ABGR2101010;
		} else if (!strcmp(format, "DRM_FORMAT_XBGR16161616F")) {
			return DRM_FORMAT_XBGR16161616F;
		} else if (!strcmp(format, "DRM_FORMAT_ABGR16161616F")) {
			return DRM_FORMAT_ABGR16161616F;
		}
	}
	return DRM_FORMAT_XBGR2101010;
}

static struct app *
create(struct display *display)
{
	struct app *app;
	struct wl_surface *surface;
	struct wl_display *wldisplay;
	uint32_t i, width, height, format;
	int ret;
	struct buffer *buffer;

	app = xzalloc(sizeof *app);

	app->display = display;
	display_set_user_data(app->display, app);
	display_set_global_handler(display, global_handler);
	display_set_global_handler_remove(display, global_handler_remove);

	// Ensure that we have received the DMABUF format and modifier support
	wldisplay = display_get_display(display);
	wl_display_roundtrip(wldisplay);

	app->window = window_create(app->display);
	app->widget = window_add_widget(app->window, app);
	window_set_title(app->window, "Wayland Simple pattern");

	window_set_user_data(app->window, app);

	widget_set_redraw_handler(app->widget, redraw_handler);

	widget_set_use_cairo(app->widget, 0);

	surface = window_get_wl_surface(app->window);

	if (option_count < 20) {
		option_count = 20;
	}

	width = option_width;
	height = option_height;
	format = string_to_format(option_format);
	printf("displaying  pattern \n"
		"\t format = %s\n"
		"\t width = %d, height = %d\n",
		format_to_string(format), width, height);

	if (option_fullscreen) {
		window_set_fullscreen(app->window, 1);
	} else {
		/* if not fullscreen, resize as per the video size */
		widget_schedule_resize(app->widget, width, height);
	}

	for (i = 0; i < NUM_BUFFERS; i++) {
		buffer = &app->buffers[i];
		ret = create_dmabuf_buffer(app, buffer, width, height, format);

		if (ret < 0)
			goto err;

	}

	return app;

err:
	free(app);
	return NULL;
}

static void
destroy(struct app *app)
{
	int i;

	for (i = 0; i < NUM_BUFFERS; i++) {
		destroy_dmabuf_buffer(&(app->buffers[i]));
	}
	widget_destroy(app->widget);
	window_destroy(app->window);
	free(app);
}

int
main(int argc, char *argv[])
{
	struct display *display;
	struct app *app;

	parse_options(options, ARRAY_LENGTH(options), &argc, argv);
	if (option_help) {
		printf(help_text, argv[0]);
		return 0;
	}

	display = display_create(&argc, argv);
	if (display == NULL) {
		fprintf(stderr, "failed to create display: %m\n");
		return -1;
	}

	if (!display_has_subcompositor(display)) {
		fprintf(stderr, "compositor does not support "
			"the subcompositor extension\n");
		return -1;
	}

	app = create(display);
	if (!app) {
		fprintf(stderr, "Failed to initialize!\n");
		exit(EXIT_FAILURE);
	}

	display_run(display);

	destroy(app);
	display_destroy(display);

	return 0;
}
