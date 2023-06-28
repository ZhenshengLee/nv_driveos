/*
* Copyright (c) 2014-2015 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA Corporation is strictly prohibited.
*/

#ifndef OS_SUPPORT_OVERRIDE_H
#define OS_SUPPORT_OVERRIDE_H

#ifdef OVERRIDE_SPEEX_ALLOC
extern void *libnvaecfx_alloc(int size);
static inline void *speex_alloc(int size)
{
	return libnvaecfx_alloc(size);
}
#endif
#ifdef OVERRIDE_SPEEX_FREE
extern void libnvaecfx_free(void *ptr);
static inline void speex_free(void *ptr)
{
	return libnvaecfx_free(ptr);
}
#endif
#ifdef OVERRIDE_SPEEX_FATAL
static inline void _speex_fatal(const char *str, const char *file, int line)
{
	fprintf(stderr, "Fatal (internal) error in %s, line %d: %s\n", file, line, str);
#ifdef AEC_C_MODEL
	exit(1);
#endif
}
#endif

#endif /* OS_SUPPORT_OVERRIDE_H */

