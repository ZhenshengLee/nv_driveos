/*
 * Copyright (c) 2019 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

// Define the WESTON_TEST_NV_ASSERT_MSG and WESTON_TEST_NV_ASSERT_STATUS
// before including assert.h to override the default ones.
#define WESTON_TEST_NV_ASSERT_MSG "Expected assertion at: (%s:%d: \"%s\")\n"
#define WESTON_TEST_NV_ASSERT_STATUS EXIT_SUCCESS

#include <assert.h>

int main()
{
	fprintf(stderr, "This is a negative assertion test and the next assertion is benign.\n");
	assert(!"Negative assertion test. This message is expected.");
	fprintf(stderr, "Test failed: Assertion did not fire at (%s:%d)\n", __FILE__, __LINE__-1);
	return 0;
}
