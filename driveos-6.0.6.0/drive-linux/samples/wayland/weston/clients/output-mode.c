/*
 * Copyright © 2020 NVIDIA Corporation
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

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <sys/mman.h>
#include <assert.h>

#include <wayland-client.h>
#include "weston-ctm-client-protocol.h"
#include "weston-gamma-client-protocol.h"
#include "weston-output-mode-client-protocol.h"
#include "shared/xalloc.h"

#define TEST_RESOLUTION_CHANGE  0x1
#define TEST_COLOR_RANGE        0x2
#define TEST_SET_GAMMA          0x4
#define TEST_GET_GAMMA          0x8
#define TEST_CTM                0x10

#define SUBTEST_PASSTHROUGH 0
#define SUBTEST_INVERSE 1
#define RESET_GAMMA 2

uint32_t gammaSize;

struct mode_output {
	struct wl_output *output;
	struct wl_buffer *buffer;
	void *data;
	enum weston_quant_color_range color_range;
	struct wl_list link;
};

struct mode_data {
	struct wl_display *display;
	struct wl_list output_list;

	struct weston_quant *color_range;
	struct weston_gamma *gamma;
	struct wl_shm *shm;
	struct weston_resolution *res_change;
	struct weston_ctm *ctm;
	int res_change_done;
};


static void
resolution_reply(void *data,
		 struct weston_resolution *res,
		 struct wl_output *output,
		 char *name,
		 struct wl_array *modes)
{
	struct resolution_data {
		int32_t width;
		int32_t height;
		int32_t refresh;
		int32_t flags;
	};

	struct mode_data *mdata = data;
	struct resolution_data *ptr;
	int datasize = sizeof(struct resolution_data);
	int num = 0, size = 0;
	int input = -1;

	printf("Output %s supported resolutions:\n", name);
	while (size < modes->size) {
		ptr = (struct resolution_data*)(modes->data + size);
		printf("Mode %d: %d_%d_%.3f: The resolution is %d x %d, "
		       "and the refresh rate is %.3f Hz. %s %s\n",
		       num, ptr->width, ptr->height, ptr->refresh / 1000.0,
		       ptr->width, ptr->height, ptr->refresh / 1000.0,
		       (ptr->flags & WL_OUTPUT_MODE_CURRENT) ? "Current" : "",
		       (ptr->flags & WL_OUTPUT_MODE_PREFERRED) ? "Preferred" : "");
		size = (++num) * datasize;
	}
	printf("\nInput (0-%d) to request a mode change or other key to exit:\n", num-1);
	scanf("%d", &input);
	if (input >= 0 && input < num) {
		ptr = (struct resolution_data*)(modes->data + input * datasize);
		if (ptr->flags & WL_OUTPUT_MODE_CURRENT) {
			printf("Mode %d: %d_%d_%.3f is already current\n",
			       input, ptr->width, ptr->height, ptr->refresh / 1000.0);
		} else {
			printf("Request resolution change to Mode %d: %d_%d_%.3f\n",
			       input, ptr->width, ptr->height, ptr->refresh / 1000.0);
			weston_resolution_change(mdata->res_change, output,
						 ptr->width, ptr->height, ptr->refresh);
			wl_display_roundtrip(mdata->display);
		}
	} else {
		printf("Exit without resolution change\n");
		mdata->res_change_done = 0;
	}
}

static void
resolution_change_done(void *data,
		       struct weston_resolution *res,
		       int32_t changed)
{
	struct mode_data *mdata = data;
	mdata->res_change_done = changed;
	if (changed) {
		printf("Resolution is changed.\n");
	} else {
		printf("Resolution change failed.\n");
	}
}

static const struct weston_resolution_listener resolution_listener = {
	resolution_reply,
	resolution_change_done
};

static void
gamma_size(void *data,
	   struct weston_gamma *gamma,
	   uint32_t size)
{
	gammaSize = size;

	printf("Gamma size: %u\n",size);
	printf("Element count of red, green and blue should be %u each\n\n",size);
}

static const struct weston_gamma_listener gamma_listener = {
	gamma_size
};

static void
handle_global(void *data, struct wl_registry *registry, uint32_t name,
		const char *interface, uint32_t version)
{
	static struct mode_output *output;
	struct mode_data *mdata = data;

	if (strcmp(interface, "wl_output") == 0) {
		output = xmalloc(sizeof *output);
		memset(output, 0, sizeof *output);
		output->output = wl_registry_bind(registry, name,
						  &wl_output_interface, 1);
		wl_list_insert(&mdata->output_list, &output->link);
	} else if (strcmp(interface, "weston_quant") == 0) {
		mdata->color_range = wl_registry_bind(registry, name,
						      &weston_quant_interface,
						      1);
	} else if (strcmp(interface, "weston_resolution") == 0) {
		mdata->res_change = wl_registry_bind(registry, name,
						     &weston_resolution_interface,
						     1);
		weston_resolution_add_listener(mdata->res_change,
					       &resolution_listener,
					       mdata);
	} else if (strcmp(interface, "weston_ctm") == 0) {
		mdata->ctm = wl_registry_bind(registry, name,
					      &weston_ctm_interface,
					      1);
	} else if (strcmp(interface, "weston_gamma") == 0) {
		mdata->gamma = wl_registry_bind(registry, name,
						&weston_gamma_interface,
						1);
		weston_gamma_add_listener(mdata->gamma,
					  &gamma_listener,
					  mdata);
	} else if (strcmp(interface, "wl_shm") == 0) {
		mdata->shm = wl_registry_bind(registry, name,
					      &wl_shm_interface,
					      1);
	}
}

static void
handle_global_remove(void *data, struct wl_registry *registry, uint32_t name)
{
}

static const struct wl_registry_listener registry_listener = {
	handle_global,
	handle_global_remove
};

void print_usage_and_exit()
{
	printf("usage flags:\n"
		   "\t'--color-range=<>'\n\t\t0 for full, 1 for limited\n"
		   "\t'--res-change'\n"
		   "\t'--set-ctm/C\n"
		   "\t'--set-gamma/l\n"
		   "\t'--get-gamma/g\n");
	exit(0);
}

enum weston_quant_color_range get_color_range(int color_range) {
	switch(color_range) {
		case 1:
			return WESTON_QUANT_COLOR_RANGE_LIMITED;
		default:
			return WESTON_QUANT_COLOR_RANGE_FULL;
	}
}

static void
destroy_mode_data(struct mode_data *data)
{
	struct mode_output *output, *tmp;
	struct mode_data *mdata = data;

	if (mdata->shm)
		wl_shm_destroy(mdata->shm);

	wl_list_for_each_safe(output, tmp, &data->output_list, link) {
		wl_list_remove(&output->link);
		free(output);
	}
}

void set_ctm(struct weston_ctm *ctm, struct wl_output *output)
{

	struct wl_array matrix;
	uint64_t *f;
	/* Below are the default values for identity CSC matrix.
	 * Set the appropriate CTM values here
	 */
	uint64_t values[9] = {0, 0x100000000, 0, 0, 0, 0x100000000, 0x100000000, 0, 0};
	int i;

	printf("Setting CTM matrix: ");
	for (i = 0; i < 9; i++) {
		printf("%llu  ", values[i]);
	}
	printf("\n");

	wl_array_init(&matrix);
	for (i = 0; i < 9; i++) {
		f = wl_array_add(&matrix, sizeof *f);
		*f = values[i];
	}
	weston_ctm_set_ctm(ctm, output, &matrix);
	wl_array_release(&matrix);
}

static void gamma_ramp(void *data, int width, int subtest)
{
	int i, j;
	uint16_t *val = data;

	switch(subtest) {
		case(SUBTEST_PASSTHROUGH):
			printf("set_gamma_passthrough\n");
			break;
		case(SUBTEST_INVERSE):
			printf("set_gamma_inverse\n");
			break;
		case(RESET_GAMMA):
			printf("reset_gamma\n");
			break;
	}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < width; j++) {
			int step = (1 << 16) / width;
			switch (subtest) {
				case SUBTEST_PASSTHROUGH:
					*val++ = j * step;
					break;
				case SUBTEST_INVERSE:
					*val++ = (width - j - 1) * step;
					break;
				case RESET_GAMMA:
					*val++ = j * step;
					break;
				default:
					break;
			}
		}
	}
}

static struct wl_buffer *
gamma_create_shm_buffer(int width, int height, void **data_out,
			struct wl_shm *shm, int set_gamma, int subtest,
			int *data_size)
{
	struct wl_shm_pool *pool;
	struct wl_buffer *buffer;
	int fd, size, stride;
	void *data;

	stride = width * 2;
	size = stride * height;
	*data_size = size;

	fd = os_create_anonymous_file(size);
	if (fd < 0) {
		fprintf(stderr, "Creating a buffer file %d failed: %s\n",
			size, strerror(errno));
		return NULL;
	}

	data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (data == MAP_FAILED) {
		fprintf(stderr, "MMAP failed: %s\n", strerror(errno));
		close(fd);
		return NULL;
	}

	pool = wl_shm_create_pool(shm, fd, size);
	buffer = wl_shm_pool_create_buffer(pool, 0, width, height, stride,
					   WL_SHM_FORMAT_XRGB8888);
	wl_shm_pool_destroy(pool);
	if (set_gamma)
		gamma_ramp(data, width, subtest);
	*data_out = data;
	close(fd);
	return buffer;
}

void check_gamma_values(void **data) {
	uint16_t *val = *data, value;
	char *colors[] = {"Red", "Green", "Blue"};
	for (int i = 0; i < 3; i++) {
		printf("%s values:\n",colors[i]);
		for (int j = 0; j < 1025; j++) {
			value = *val;
			val++;
			printf("%u ",value);
		}
		printf("\n\n");
	}
}

int main (int argc, char *argv[])
{
	struct wl_display *display;
	struct wl_registry *registry;
	struct mode_output *output;
	struct mode_data mdata = {};
	int c, option_index;
	int test = 0, ret = 0;
	enum weston_quant_color_range color_range = WESTON_QUANT_COLOR_RANGE_FULL;
	int sleeptime = 3;

	static struct option long_options[] = {
		{"color-range",      required_argument, 0,  'c' },
		{"res-change",       no_argument      , 0,  'r' },
		{"sleep-time",       required_argument, 0,  's' },
		{"set-gamma",        no_argument,       0,  'l' },
		{"get-gamma",        no_argument,       0,  'g' },
		{"set-ctm",          no_argument,       0,  'C' },
		{"help",             no_argument      , 0,  'h' },
		{0, 0, 0, 0}
	};

	while ((c = getopt_long(argc, argv, "hc:lgs:rC", long_options, &option_index)) != -1) {
		switch (c) {
		case 'c':
			color_range = get_color_range(atoi(optarg));
			test |= TEST_COLOR_RANGE;
			break;
		case 'r':
			test |= TEST_RESOLUTION_CHANGE;
			break;
		case 'C':
			test |= TEST_CTM;
			break;
		case 'l':
			test |= TEST_SET_GAMMA;
			break;
		case 'g':
			test |= TEST_GET_GAMMA;
			break;
		case 's':
			sleeptime = atoi(optarg);
			break;
		default:
			print_usage_and_exit();
		}
	}

	display = wl_display_connect(NULL);
	if (display == NULL) {
		fprintf(stderr, "Failed to create display: %s\n",
			strerror(errno));
		return -1;
	}
	mdata.display = display;

	wl_list_init(&mdata.output_list);
	registry = wl_display_get_registry(display);
	wl_registry_add_listener(registry, &registry_listener, &mdata);
	wl_display_dispatch(display);
	wl_display_roundtrip(display);

	if (test & TEST_COLOR_RANGE) {
		wl_list_for_each(output, &mdata.output_list, link) {
			output->color_range = color_range;
			weston_quant_set_output_color_range(mdata.color_range,
							    output->output,
							    output->color_range);
			wl_display_roundtrip(display);
		}
	}

	if (test & TEST_RESOLUTION_CHANGE) {
		wl_list_for_each(output, &mdata.output_list, link) {
			weston_resolution_query(mdata.res_change,
						output->output);
			mdata.res_change_done = -1;
			wl_display_roundtrip(display);
		}
	}

	if (test & TEST_CTM) {
		wl_list_for_each(output, &mdata.output_list, link) {
			set_ctm(mdata.ctm, output->output);
			wl_display_roundtrip(display);
		}
	}

	if (test & TEST_SET_GAMMA) {
		wl_list_for_each(output, &mdata.output_list, link) {
			weston_gamma_get_size(mdata.gamma, output->output);
			wl_display_dispatch(display);
			assert(gammaSize > 0);

			printf("Setting gamma values\n");
			for (int subtest = 0; subtest < 3; subtest++) {
				int width = gammaSize;
				int height = 3, data_size = 0;
				output->buffer =
					gamma_create_shm_buffer(width, height,
								&output->data,
								mdata.shm, 1, subtest,
								&data_size);
				weston_gamma_set_gamma(mdata.gamma,
						       output->output,
						       output->buffer);
				wl_display_roundtrip(display);
				sleep(sleeptime);
				if (output->buffer)
					free(output->buffer);
				if (output->data)
					munmap(output->data, data_size);
			}
		}
	}

	if (test & TEST_GET_GAMMA) {
		wl_list_for_each(output, &mdata.output_list, link) {
			weston_gamma_get_size(mdata.gamma, output->output);
			wl_display_dispatch(display);
			assert(gammaSize > 0);

			printf("Reading gamma values\n");
			int width = gammaSize;
			int height = 3, data_size = 0;
			output->buffer =
				gamma_create_shm_buffer(width, height,
							&output->data,
							mdata.shm, 0, 0,
							&data_size);
			weston_gamma_get_gamma(mdata.gamma,
					       output->output,
					       output->buffer);
			wl_display_roundtrip(display);
			check_gamma_values(&output->data);
			if (output->buffer)
				free(output->buffer);
			if (output->data)
				munmap(output->data, data_size);
		}
	}

	destroy_mode_data(&mdata);
	wl_registry_destroy(registry);
	wl_display_disconnect(display);
}
