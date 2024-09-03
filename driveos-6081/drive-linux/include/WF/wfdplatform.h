/* Copyright (c) 2009 The Khronos Group Inc.
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */

/*! \ingroup wfd
 *  \file wfdplatform.h
 *
 *  \brief Platform specific type definitions
 */


#ifndef WFDPLATFORM_H
#define WFDPLATFORM_H
#include <KHR/khrplatform.h>

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef WFD_API_CALL
#define WFD_API_CALL KHRONOS_APICALL
#endif

#ifndef WFD_APIENTRY
#define WFD_APIENTRY KHRONOS_APIENTRY
#endif

#ifndef WFD_APIEXIT
#define WFD_APIEXIT KHRONOS_APIATTRIBUTES
#endif

typedef khronos_int32_t             WFDint;
typedef khronos_float_t             WFDfloat;
typedef khronos_uint32_t            WFDbitfield;
typedef khronos_uint32_t            WFDHandle;
typedef khronos_uint32_t            WFDuint32;
typedef khronos_uint64_t            WFDuint64;

#define WFD_FOREVER (0xFFFFFFFFFFFFFFFF)

#ifndef NV_SAFETY_BUILD

typedef khronos_uint8_t             WFDuint8;
typedef khronos_utime_nanoseconds_t WFDtime;

typedef void*                       WFDEGLDisplay; /* An opaque handle to an EGLDisplay */
typedef void*                       WFDEGLSync; /* An opaque handle to an EGLSyncKHR */
typedef void*                       WFDEGLImage; /* An opaque handle to an EGLImage */
typedef WFDHandle                   WFDNativeStreamType;

#define WFD_INVALID_SYNC ((WFDEGLSync)0)

#endif // !NV_SAFETY_BUILD

#ifdef __cplusplus
}
#endif
#endif /* WFDPLATFORM_H */

