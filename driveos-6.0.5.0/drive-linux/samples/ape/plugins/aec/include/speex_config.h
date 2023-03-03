/*
* Copyright (c) 2014-2015 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA Corporation is strictly prohibited.
*/

#ifndef _CONFIG_H_
#define _CONFIG_H_

#if defined(_WIN32)
// Microsoft version of 'inline'
#define inline __inline
#endif

// Visual Studio support alloca(), but it always align variables to 16-bit
// boundary, while SSE need 128-bit alignment. So we disable alloca() when
// SSE is enabled.

#define FIXED_POINT 1

#ifndef _USE_SSE
#  define USE_ALLOCA
#endif

/* Default to floating point */
#ifndef FIXED_POINT
#define FLOATING_POINT
#define USE_SMALLFT
#else
#define USE_KISS_FFT
#endif

/* We don't support visibility on Win32 */
/* and on ADSP*/

#define EXPORT

#define OS_SUPPORT_CUSTOM

#define xOVERRIDE_SPEEX_COPY
#define xOVERRIDE_SPEEX_WARNING
#define xOVERRIDE_SPEEX_WARNING_INT

#define OVERRIDE_SPEEX_ALLOC
#define OVERRIDE_SPEEX_FREE
#define OVERRIDE_SPEEX_FATAL
#define OVERRIDE_SPEEX_ALLOC_SCRATCH
#define OVERRIDE_SPEEX_REALLOC
#define OVERRIDE_SPEEX_FREE_SCRATCH
#define OVERRIDE_SPEEX_MOVE
#define OVERRIDE_SPEEX_MEMSET
#define OVERRIDE_SPEEX_NOTIFY
#define OVERRIDE_SPEEX_PUTC

#endif /* _CONFIG_H_ */
