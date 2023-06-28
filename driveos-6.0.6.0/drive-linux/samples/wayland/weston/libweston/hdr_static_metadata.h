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

#ifndef WESTON_HDR_STATIC_METADATA_H
#define WESTON_HDR_STATIC_METADATA_H

struct cie_xy {
	double x;
	double y;
};

struct color_primaries {
	struct cie_xy r;
	struct cie_xy g;
	struct cie_xy b;
	struct cie_xy white_point;
};

struct weston_hdr_static_metadata {
	struct color_primaries primaries;
	double min_luminance;
	double max_luminance;
	uint32_t max_cll;
	uint32_t max_fall;
};

#endif
