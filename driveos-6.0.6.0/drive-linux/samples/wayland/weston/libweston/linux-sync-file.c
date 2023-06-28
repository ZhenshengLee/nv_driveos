/*
 * Copyright © 2018 Collabora, Ltd
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

#include <assert.h>
#include <errno.h>
#include <poll.h>
#include <stddef.h>
#include <sys/ioctl.h>

#ifdef HAVE_LINUX_SYNC_FILE_H
#include <linux/sync_file.h>
#else
#include "linux-sync-file-uapi.h"
#endif

#include "linux-sync-fence-uapi.h"

#include "linux-sync-file.h"
#include "shared/timespec-util.h"

/* Query sync_fence_info_data associated with an FD. This is the
 * interface used by Tegra devices. */
static struct sync_fence_info_data *
get_sync_fence_info_data(void *buffer, size_t buffer_length, int fd)
{
	struct sync_fence_info_data *result =
		(struct sync_fence_info_data*)buffer;
	result->len = buffer_length;

	if (ioctl(fd, SYNC_IOC_FENCE_INFO, result) != 0)
		return NULL;

	// Lower bound on the size of a fence_info containing at least one
	// fence.
	if (result->len < sizeof(struct sync_fence_info_data) +
			  sizeof(struct sync_pt_info))
		return NULL;

	return result;
}

/* Check that a file descriptor represents a valid sync file
 *
 * \param fd[in] a file descriptor
 * \return true if fd is a valid sync file, false otherwise
 */
bool
linux_sync_file_is_valid(int fd)
{
	struct sync_file_info file_info = { { 0 } };
	char fence_info_buffer[1024] = { 0 };

	if (get_sync_fence_info_data(fence_info_buffer,
				     sizeof(fence_info_buffer), fd) != NULL)
		return true;

	if (ioctl(fd, SYNC_IOC_FILE_INFO, &file_info) < 0)
		return false;

	return file_info.num_fences > 0;
}

/* Read the timestamp stored in a sync file
 *
 * \param fd[in] fd a file descriptor for a sync file
 * \param ts[out] the timespec struct to fill with the timestamp
 * \return 0 if a timestamp was read, -1 on error
 */
int
linux_sync_file_read_timestamp(int fd, struct timespec *ts)
{
	// Ensure that the buffer is sufficiently aligned.
	uint64_t fence_info_buffer[1024 / sizeof(uint64_t)] = { 0 };
	struct sync_fence_info_data *fence_info_data;
	struct sync_file_info file_info = { { 0 } };
	struct sync_fence_info fence_info = { { 0 } };

	assert(ts != NULL);

	fence_info_data = get_sync_fence_info_data(fence_info_buffer,
						   sizeof(fence_info_buffer),
						   fd);
	if (fence_info_data != NULL)
	{
		struct sync_pt_info *pt_info =
			(struct sync_pt_info*)(fence_info_data->pt_info);
		timespec_from_nsec(ts, pt_info->timestamp_ns);
		return 0;
	}

	file_info.sync_fence_info = (uint64_t)(uintptr_t)&fence_info;
	file_info.num_fences = 1;

	if (ioctl(fd, SYNC_IOC_FILE_INFO, &file_info) < 0)
		return -1;

	timespec_from_nsec(ts, fence_info.timestamp_ns);

	return 0;
}

int
linux_sync_file_wait(int fd, int timeout)
{
	__s32 to = timeout;
	return ioctl(fd, SYNC_IOC_WAIT, &to);
}
