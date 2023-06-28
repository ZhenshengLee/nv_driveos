/*
 * Copyright © 2019 Intel Corporation
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

#ifndef WESTON_DRM_HDR_METADATA_H
#define WESTON_DRM_HDR_METADATA_H

#include <stdint.h>

/* Monitor's HDR metadata */
struct drm_edid_hdr_metadata_static {
	uint8_t eotf;
	uint8_t metadata_type;
	uint8_t desired_max_ll;
	uint8_t desired_max_fall;
	uint8_t desired_min_ll;
};

void
drm_release_hdr_metadata(struct drm_edid_hdr_metadata_static *md);

struct drm_edid_hdr_metadata_static *
drm_get_display_hdr_metadata(const uint8_t *edid, uint32_t edid_len);

uint16_t
drm_get_display_colorspace(const uint8_t *edid, uint32_t edid_len);

#endif
