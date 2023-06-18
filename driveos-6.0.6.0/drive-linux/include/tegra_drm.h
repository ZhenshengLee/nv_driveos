/*
 * Copyright (c) 2012-2019, NVIDIA CORPORATION.  All rights reserved.
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
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef _UAPI_TEGRA_DRM_H_
#define _UAPI_TEGRA_DRM_H_

#include <drm.h>

/* TODO remove this when all downstream additions from this header are removed */
#define DOWNSTREAM_TEGRA_DRM

#define fourcc_code_tegra(a,b,c,d) ((__u32)(a) | ((__u32)(b) << 8) | \
        ((__u32)(c) << 16) | ((__u32)(d) << 24))

/*
 * 2 plane YCrCb, 10 bits per channel
 * index 0 = Y plane, [15:0] Y
 * index 1 = Cr:Cb plane, [31:0] Cr:Cb
 *
 * Pixel packing:
 *
 *          Y plane
 * MS-Byte            LS-Byte
 * Y9Y8Y7Y6Y5Y4Y3Y2 | Y1Y0XXXXXX
 *
 *          UV plane
 * MS-Byte                                            LS-Byte
 * V9V8V7V6V5V4V3V2 | V1V0XXXXXX | U9U8U7U6U5U4U3U2 | U1U0XXXXXX
 */
#define DRM_FORMAT_TEGRA_P010_709 fourcc_code_tegra('H', 'D', '0', '1') /* 2x2 subsampled Cr:Cb plane BT.709 */
#define DRM_FORMAT_TEGRA_P010_2020 fourcc_code_tegra('U', 'H', 'D', '0') /* 2x2 subsampled Cr:Cb plane BT.2020 */
#define DRM_FORMAT_TEGRA_ARGB2101010_709 fourcc_code_tegra('R', '1', '0', '1') /* [31:0] A:R:G:B 2:10:10:10 BT.709 little endian  */
#define DRM_FORMAT_TEGRA_ARGB2101010_2020 fourcc_code_tegra('R', '1', '0', '0') /* [31:0] A:R:G:B 2:10:10:10 BT.2020 little endian */
#define DRM_FORMAT_TEGRA_ABGR2101010_709 fourcc_code_tegra('B', '1', '0', '1') /* [31:0] A:B:G:R 2:10:10:10 BT.709 little endian */
#define DRM_FORMAT_TEGRA_ABGR2101010_2020 fourcc_code_tegra('B', '1', '0', '0') /* [31:0] A:B:G:R 2:10:10:10 BT.2020 little endian */
#define DRM_FORMAT_TEGRA_ARGB8888_sRGB fourcc_code_tegra('A', 'S', '2', '4') /* [31:0] A:R:G:B 8:8:8:8 sRGB little endian */
#define DRM_FORMAT_TEGRA_XRGB8888_sRGB fourcc_code_tegra('X', 'S', '2', '4') /* [31:0] A:R:G:B 8:8:8:8 sRGB little endian */

#define DRM_TEGRA_GEM_CREATE_TILED        (1 << 0)
#define DRM_TEGRA_GEM_CREATE_BOTTOM_UP    (1 << 1)
#define DRM_TEGRA_GEM_CREATE_COMPRESSIBLE (1 << 2)

/* TODO Upstream DRM_TEGRA_GEM_CREATE_PROTECTED */
#define DRM_TEGRA_GEM_CREATE_PROTECTED    (1 << 3)

struct drm_tegra_gem_create {
	__u64 size;
	__u32 flags;
	__u32 handle;
};

struct drm_tegra_gem_mmap {
	__u32 handle;
	__u32 offset;
};

struct drm_tegra_syncpt_read {
	__u32 id;
	__u32 value;
};

struct drm_tegra_syncpt_incr {
	__u32 id;
	__u32 pad;
};

struct drm_tegra_syncpt_wait {
	__u32 id;
	__u32 thresh;
	__u32 timeout;
	__u32 value;
};

#define DRM_TEGRA_NO_TIMEOUT	(0xffffffff)

struct drm_tegra_open_channel {
	__u32 client;
	__u32 pad;
	__u64 context;
};

struct drm_tegra_close_channel {
	__u64 context;
};

struct drm_tegra_get_syncpt {
	__u64 context;
	__u32 index;
	__u32 id;
};

struct drm_tegra_get_syncpt_base {
	__u64 context;
	__u32 syncpt;
	__u32 id;
};

struct drm_tegra_syncpt {
	__u32 id;
	__u32 incrs;
};

struct drm_tegra_cmdbuf {
	__u32 handle;
	__u32 offset;
	__u32 words;
	__u32 pad;
};

struct drm_tegra_reloc {
	struct {
		__u32 handle;
		__u32 offset;
	} cmdbuf;
	struct {
		__u32 handle;
		__u32 offset;
	} target;
	__u32 shift;
	__u32 pad;
};

struct drm_tegra_waitchk {
	__u32 handle;
	__u32 offset;
	__u32 syncpt;
	__u32 thresh;
};

struct drm_tegra_submit {
	__u64 context;
	__u32 num_syncpts;
	__u32 num_cmdbufs;
	__u32 num_relocs;
	__u32 num_waitchks;
	__u32 waitchk_mask;
	__u32 timeout;
	__u64 syncpts;
	__u64 cmdbufs;
	__u64 relocs;
	__u64 waitchks;
	__u32 fence;		/* Return value */

	__u32 reserved[5];	/* future expansion */
};

#define DRM_TEGRA_GEM_TILING_MODE_PITCH 0
#define DRM_TEGRA_GEM_TILING_MODE_TILED 1
#define DRM_TEGRA_GEM_TILING_MODE_BLOCK 2

struct drm_tegra_gem_set_tiling {
	/* input */
	__u32 handle;
	__u32 mode;
	__u32 value;
	__u32 pad;
};

struct drm_tegra_gem_get_tiling {
	/* input */
	__u32 handle;
	/* output */
	__u32 mode;
	__u32 value;
	__u32 pad;
};

#define DRM_TEGRA_GEM_BOTTOM_UP    (1 << 0)
#define DRM_TEGRA_GEM_COMPRESSIBLE (1 << 1)
#define DRM_TEGRA_GEM_FLAGS        (DRM_TEGRA_GEM_BOTTOM_UP | \
                                    DRM_TEGRA_GEM_COMPRESSIBLE)

struct drm_tegra_gem_set_flags {
	/* input */
	__u32 handle;
	/* output */
	__u32 flags;
};

struct drm_tegra_gem_get_flags {
	/* input */
	__u32 handle;
	/* output */
	__u32 flags;
};

struct drm_tegra_set_crtc_flip_syncpt {
	/* input */
	__u32 crtc_id;
	__u32 id;
	__u32 value;
};

struct drm_tegra_get_crtc_flip_syncpt {
	/* input */
	__u32 crtc_id;
	/* output */
	__u32 id;
	__u32 value;
};

struct drm_tegra_set_plane_flip_syncpt {
	/* input */
	__u32 plane_id;
	__u32 id;
	__u32 value;
};

struct drm_tegra_get_plane_flip_syncpt {
	/* input */
	__u32 plane_id;
	/* output */
	__u32 id;
	__u32 value;
};

struct drm_tegra_get_client_cap {
	/* input */
	__u64 cap;
	/* output */
	__u32 val;
};

struct drm_tegra_set_crtc_flip_timestamp {
	/* input */
	__u32 crtc_id;
	__u64 value;
};

struct drm_tegra_set_plane_flip_timestamp {
	/* input */
	__u32 plane_id;
	__u64 value;
};

struct drm_tegra_hdr_metadata_smpte_2086 {
    // idx 0 : G, 1 : B, 2 : R
    __u16 display_primaries_x[3];          // normalized x chromaticity cordinate. It shall be in the range of 0 to 50000
    __u16 display_primaries_y[3];          // normalized y chromaticity cordinate. It shall be in the range of 0 to 50000
    __u16 white_point_x;                   // normalized x chromaticity cordinate of white point of mastering display
    __u16 white_point_y;                   // normalized y chromaticity cordinate of white point of mastering display
    __u32 max_display_parameter_luminance; // nominal maximum display luminance in units of 0.0001 candelas per square metre
    __u32 min_display_parameter_luminance; // nominal minimum display luminance in units of 0.0001 candelas per square metre
};

#define DRM_TEGRA_GEM_CREATE		0x00
#define DRM_TEGRA_GEM_MMAP		0x01
#define DRM_TEGRA_SYNCPT_READ		0x02
#define DRM_TEGRA_SYNCPT_INCR		0x03
#define DRM_TEGRA_SYNCPT_WAIT		0x04
#define DRM_TEGRA_OPEN_CHANNEL		0x05
#define DRM_TEGRA_CLOSE_CHANNEL		0x06
#define DRM_TEGRA_GET_SYNCPT		0x07
#define DRM_TEGRA_SUBMIT		0x08
#define DRM_TEGRA_GET_SYNCPT_BASE	0x09
#define DRM_TEGRA_GEM_SET_TILING	0x0a
#define DRM_TEGRA_GEM_GET_TILING	0x0b
#define DRM_TEGRA_GEM_SET_FLAGS		0x0c
#define DRM_TEGRA_GEM_GET_FLAGS		0x0d
#define DRM_TEGRA_SET_CRTC_FLIP_SYNCPT	0x0e
#define DRM_TEGRA_GET_CRTC_FLIP_SYNCPT	0x0f
#define DRM_TEGRA_SET_PLANE_FLIP_SYNCPT	0x10
#define DRM_TEGRA_GET_PLANE_FLIP_SYNCPT	0x11
#define DRM_TEGRA_GET_CLIENT_CAP	0x12
#define DRM_TEGRA_SET_CRTC_FLIP_TIMESTAMP  0x13
#define DRM_TEGRA_SET_PLANE_FLIP_TIMESTAMP 0x14

// This client capability controls drm-nvdc behavior in some cases
// where drm-nvdc has diverged from upstream.
// Permissive=1: Maintain legacy drm-nvdc behavior (default)
// Permissive=0: Behave consistently with upstream libdrm
#define DRM_CLIENT_CAP_DRM_NVDC_PERMISSIVE 6

#define DRM_IOCTL_TEGRA_GEM_CREATE DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_GEM_CREATE, struct drm_tegra_gem_create)
#define DRM_IOCTL_TEGRA_GEM_MMAP DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_GEM_MMAP, struct drm_tegra_gem_mmap)
#define DRM_IOCTL_TEGRA_SYNCPT_READ DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_SYNCPT_READ, struct drm_tegra_syncpt_read)
#define DRM_IOCTL_TEGRA_SYNCPT_INCR DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_SYNCPT_INCR, struct drm_tegra_syncpt_incr)
#define DRM_IOCTL_TEGRA_SYNCPT_WAIT DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_SYNCPT_WAIT, struct drm_tegra_syncpt_wait)
#define DRM_IOCTL_TEGRA_OPEN_CHANNEL DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_OPEN_CHANNEL, struct drm_tegra_open_channel)
#define DRM_IOCTL_TEGRA_CLOSE_CHANNEL DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_CLOSE_CHANNEL, struct drm_tegra_open_channel)
#define DRM_IOCTL_TEGRA_GET_SYNCPT DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_GET_SYNCPT, struct drm_tegra_get_syncpt)
#define DRM_IOCTL_TEGRA_SUBMIT DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_SUBMIT, struct drm_tegra_submit)
#define DRM_IOCTL_TEGRA_GET_SYNCPT_BASE DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_GET_SYNCPT_BASE, struct drm_tegra_get_syncpt_base)
#define DRM_IOCTL_TEGRA_GEM_SET_TILING DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_GEM_SET_TILING, struct drm_tegra_gem_set_tiling)
#define DRM_IOCTL_TEGRA_GEM_GET_TILING DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_GEM_GET_TILING, struct drm_tegra_gem_get_tiling)
#define DRM_IOCTL_TEGRA_GEM_SET_FLAGS DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_GEM_SET_FLAGS, struct drm_tegra_gem_set_flags)
#define DRM_IOCTL_TEGRA_GEM_GET_FLAGS DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_GEM_GET_FLAGS, struct drm_tegra_gem_get_flags)
#define DRM_IOCTL_TEGRA_SET_CRTC_FLIP_SYNCPT DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_SET_CRTC_FLIP_SYNCPT, struct drm_tegra_set_crtc_flip_syncpt)
#define DRM_IOCTL_TEGRA_GET_CRTC_FLIP_SYNCPT DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_GET_CRTC_FLIP_SYNCPT, struct drm_tegra_get_crtc_flip_syncpt)
#define DRM_IOCTL_TEGRA_SET_PLANE_FLIP_SYNCPT DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_SET_PLANE_FLIP_SYNCPT, struct drm_tegra_set_plane_flip_syncpt)
#define DRM_IOCTL_TEGRA_GET_PLANE_FLIP_SYNCPT DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_GET_PLANE_FLIP_SYNCPT, struct drm_tegra_get_plane_flip_syncpt)
#define DRM_IOCTL_TEGRA_GET_CLIENT_CAP DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_GET_CLIENT_CAP, struct drm_tegra_get_client_cap)
#define DRM_IOCTL_TEGRA_SET_CRTC_FLIP_TIMESTAMP DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_SET_CRTC_FLIP_TIMESTAMP, struct drm_tegra_set_crtc_flip_timestamp)
#define DRM_IOCTL_TEGRA_SET_PLANE_FLIP_TIMESTAMP DRM_IOWR(DRM_COMMAND_BASE + DRM_TEGRA_SET_PLANE_FLIP_TIMESTAMP, struct drm_tegra_set_plane_flip_timestamp)

#endif
