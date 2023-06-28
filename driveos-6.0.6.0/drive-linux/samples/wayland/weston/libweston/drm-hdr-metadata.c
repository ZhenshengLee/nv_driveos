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
#include <string.h>
#include "compositor.h"
#include "drm-hdr-metadata.h"
#include "shared/helpers.h"

#define EDID_BLOCK_LENGTH 				128
#define EDID_CEA_EXT_ID 				0x02
#define EDID_CEA_TAG_EXTENDED 				0x7

/* CEA-861-G new EDID blocks for HDR */
#define EDID_CEA_TAG_COLORIMETRY			0x5
#define EDID_CEA_EXT_TAG_STATIC_METADATA 		0x6
#define EDID_CEA_EXT_TAG_DYNAMIC_METADATA 		0x7

static const uint8_t *
edid_find_cea_extension_block(const uint8_t *edid)
{
	uint8_t ext_blks;
	int blk;
	const uint8_t *ext = NULL;

	if (!edid) {
		weston_log("No EDID\n");
		return NULL;
	}

	ext_blks = edid[126];
	if (!ext_blks) {
		weston_log("EDID doesn't have any extension block\n");
		return NULL;
	}

	for (blk = 0; blk < ext_blks; blk++) {
		ext = edid + EDID_BLOCK_LENGTH * (blk + 1);
		if (ext[0] == EDID_CEA_EXT_ID)
			break;
	}

	if (blk == ext_blks)
		return NULL;

	return ext;
}

static const uint8_t *
edid_find_extended_data_block(const uint8_t *edid,
			      uint8_t *data_len,
			      uint32_t block_tag)
{
	uint8_t d;
	uint8_t tag;
	uint8_t extended_tag;
	uint8_t dblen;

	const uint8_t *dbptr;
	const uint8_t *cea_db_start;
	const uint8_t *cea_db_end;
	const uint8_t *cea_ext_blk;

	if (!edid) {
		weston_log("No EDID in blob\n");
		return NULL;
	}

	cea_ext_blk = edid_find_cea_extension_block(edid);
	if (!cea_ext_blk) {
		weston_log("No CEA extension block available\n");
		return NULL;
	}

	/* CEA DB starts at blk[4] and ends at blk[d] */
	d = cea_ext_blk[2];
	cea_db_start = cea_ext_blk + 4;
	cea_db_end = cea_ext_blk + d - 1;

	for (dbptr = cea_db_start; dbptr < cea_db_end; dbptr += (dblen + 1)) {

		/* First data byte contains db length and tag */
		dblen = dbptr[0] & 0x1F;
		tag = dbptr[0] >> 5;

		/* Metadata bock is extended tag block */
		if (tag != EDID_CEA_TAG_EXTENDED)
			continue;

		/* Extended block uses one extra byte for extended tag */
		extended_tag = dbptr[1];
		if (extended_tag != block_tag)
			continue;

		*data_len = dblen - 1;
		return dbptr + 2;
	}

	return NULL;
}

static struct drm_edid_hdr_metadata_static *
drm_get_hdr_static_metadata(const uint8_t *hdr_db, uint32_t data_len)
{
	struct drm_edid_hdr_metadata_static *s;

	if (data_len < 2) {
		weston_log("Invalid metadata input to static parser\n");
		return NULL;
	}

	s = zalloc(sizeof (struct drm_edid_hdr_metadata_static));
	if (!s) {
		weston_log("OOM while parsing static metadata\n");
		return NULL;
	}

	memset(s, 0, sizeof(struct drm_edid_hdr_metadata_static));

	s->eotf = hdr_db[0] & 0x3F;
	s->metadata_type = hdr_db[1];

	if (data_len >  2 && data_len < 6) {
		s->desired_max_ll = hdr_db[2];
		s->desired_max_fall = hdr_db[3];
		s->desired_min_ll = hdr_db[4];

		if (!s->desired_max_ll)
			s->desired_max_ll = 0xFF;
	}
	return s;
}

uint16_t
drm_get_display_colorspace(const uint8_t *edid, uint32_t edid_len)
{
	uint8_t data_len = 0;
	const uint8_t *clr_db;
	uint16_t colorspaces = 0;

	clr_db = edid_find_extended_data_block(edid, &data_len,
			EDID_CEA_TAG_COLORIMETRY);
	if (clr_db && data_len != 0)
		/* db[4] bit 7 is DCI-P3 support information (added in CTA-861-G) */
		colorspaces = ((!!(clr_db[1] & 0x80)) << 8) | (clr_db[0]);

	return colorspaces;
}

struct drm_edid_hdr_metadata_static *
drm_get_display_hdr_metadata(const uint8_t *edid, uint32_t edid_len)
{
	uint8_t data_len = 0;
	const uint8_t *hdr_db;
	struct drm_edid_hdr_metadata_static *md = NULL;

	if (!edid) {
		weston_log("Invalid EDID\n");
		return NULL;
	}

	hdr_db = edid_find_extended_data_block(edid, &data_len,
			EDID_CEA_EXT_TAG_STATIC_METADATA);
	if (hdr_db && data_len != 0) {
		md = drm_get_hdr_static_metadata(hdr_db, data_len);
		if (!md) {
			weston_log("Can't find static HDR metadata in EDID\n");
			return NULL;
		}
		weston_log("Monitor supports HDR\n");
	}

	return md;
}

void
drm_release_hdr_metadata(struct drm_edid_hdr_metadata_static *md)
{
	free(md);
}
