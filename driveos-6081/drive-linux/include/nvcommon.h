/*
 * Copyright (c) 2006-2021 NVIDIA Corporation.  All Rights Reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef INCLUDED_NVCOMMON_H
#define INCLUDED_NVCOMMON_H
#if (defined(__QNX__) && defined(NV_IS_SAFETY) && (NV_IS_SAFETY == 1))
#include "nvcommon_tegra_safety.h"
#else
/**
 * @file
 * <b> NVIDIA Common Definitions</b>
 *
 * @b Description: This file contains standard definitions used by various
 *    interfaces.
 */

/**
 * @if SWDOCS_VB_LNX_G3
 * @defgroup nvcommon Core Library: Common Declarations
 * @else
 * @defgroup nvcommon Common Declarations
 * @endif
 *
 * Contains standard definitions used by various interfaces.
 *
 * @ingroup common_utils
 * @{
 */

// Pull in some headers that are mirrored from desktop.
#include "nvtypes.h"

/// Include headers that provide NULL, size_t, offsetof, and [u]intptr_t.  In
/// the event that the toolchain doesn't provide these, provide them ourselves.
#include <stddef.h>
#if defined(__hos__) || (defined(__linux__) && !defined(__KERNEL__)) || defined(__arm) || defined(__APPLE__) || defined(__QNX__) || defined(__INTEGRITY) || defined(NV_HYPERVISOR)
#include <stdint.h>
#endif

// Rest of this file is mobile-only definitions, and appending to them
// is discouraged since it prevents header portability between mobile
// and desktop.
#define NV_FORCE_INLINE NV_FORCEINLINE
#define NV_ALIGN NV_ALIGN_BYTES

#if (NVCPU_IS_X86 && NVOS_IS_WINDOWS)
#   define NVOS_IS_WINDOWS_X86 1
#else
#   define NVOS_IS_WINDOWS_X86 0
#endif

#if defined(__APPLE__)
#  define NVOS_IS_DARWIN 1
#  define NVOS_IS_UNIX 1
#endif

#define NVOS_IS_LINUX_KERNEL 0

/// Min/Max values for NvF32
#define NV_MIN_F32  (1.1754944e-38f)
#define NV_MAX_F32  (3.4028234e+38f)

/// Declares a 64-bit aligned pointer.
#if defined(__GNUC__)
#define NV_ALIGN_POINTER_NAME(f) f##Align
#define NV_ALIGN_POINTER(t, f)                  \
    union {                                     \
        t f;                                    \
        NvU64 NV_ALIGN_POINTER_NAME(f);         \
    }
#else
/* rvds builds don't support anonymous unions but also don't need
 * 64-bit alignment.
 */
#define NV_ALIGN_POINTER(t, f) t f
#endif

/// Function attributes are lumped in here too.
/// NAKED - Create a function without a prologue or an epilogue.
#if NVOS_IS_WINDOWS

#define NV_NAKED __declspec(naked)
#define NV_LIKELY(c)   (c)
#define NV_UNLIKELY(c) (c)
#define NV_UNUSED

#elif defined(__ghs__) // GHS COMP
#define NV_NAKED
#define NV_LIKELY(c)   (c)
#define NV_UNLIKELY(c) (c)
#define NV_UNUSED __attribute__((unused))

#elif defined(__GNUC__)
#define NV_NAKED __attribute__((naked))
#define NV_LIKELY(c)   __builtin_expect((c),1)
#define NV_UNLIKELY(c) __builtin_expect((c),0)
#define NV_UNUSED __attribute__((unused))

#elif defined(__arm) // ARM RVDS compiler
#define NV_NAKED __asm
#define NV_LIKELY(c)   (c)
#define NV_UNLIKELY(c) (c)
#define NV_UNUSED

#else
#error Unknown compiler
#endif

/// Symbol attributes.
/// WEAK  - Define the symbol weakly so it can be overridden by the user.
#if NVOS_IS_WINDOWS
#define NV_WEAK
#elif defined(__ghs__)
#define NV_WEAK __attribute__((weak))
#elif defined(__GNUC__)
#define NV_WEAK __attribute__((weak))
#elif defined(__arm)
#define NV_WEAK __weak
#else
#error Unknown compiler
#endif

// NV_DEBUG_CODE conflicts with the definition of the same name
// in OpenGL/nvInc/nvDebug.h.
#if !defined(NV_DEBUG_CODE)
/**
 * This macro wraps its argument with the equivalent of "#if NV_DEBUG", but
 * also can be used where "#ifdef"'s can't, like inside a macro.
 */
#if NV_DEBUG
#define NV_DEBUG_CODE(x) x
#else
#define NV_DEBUG_CODE(x)
#endif
#endif

/** Macro for determining the size of an array */
#define NV_ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

/** Macro for taking min or max of a pair of numbers */
#ifndef NV_MIN
#define NV_MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef NV_MAX
#define NV_MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif

/** Macro for determining the offset of "member" in "type". */
#if !defined(NV_OFFSETOF)
    #if defined(__GNUC__)
        #define NV_OFFSETOF(type, member)   __builtin_offsetof(type, member)
    #else
        #define NV_OFFSETOF(type, member)   ((NvUPtr)(&(((type *)0)->member)))
    #endif
#endif

/** Macro for determining the size of an element e in struct s. */
#define NV_SIZEOF(s,e)          (sizeof(((s*)0)->e))

/** Get just the lowest bit of the 32 bit number */
#define NV_LOWEST_BIT_ONLY(v)   ((NvU32)(v) & (NvU32)-(NvS32)(v))

/** True if unsigned int v is a power of 2 */
#define NV_IS_POWER_OF_2(v)     (NV_LOWEST_BIT_ONLY(v) == (NvU32)(v))

/**
 * By convention, we use this value to represent an infinite wait interval in
 * APIs that expect a timeout argument.  A value of zero should not be
 * interpreted as infinite -- it should be interpreted as "time out immediately
 * and simply check whether the event has already happened."
 */
#define NV_WAIT_INFINITE 0xFFFFFFFF

/// Macro to help with MSVC Code Analysis false positives
#if defined(_PREFAST_)
#define NV_ANALYSIS_ASSUME(x) __analysis_assume(x)
#else
#define NV_ANALYSIS_ASSUME(x)
#endif

/**
 * A physical address type sized such that it matches the addressing support of
 * the hardware modules with which HW drivers typically interfaces.
 */
typedef NvU64 NvPhysAddr64;

/**
 * This is deprecated and shouldn't be used.
 *
 * NvU32 is no longer enough to hold physical address.
 * NvPhysAddr64 should be used in place of NvRmPhysAddr.
 */
typedef NvU32 NvRmPhysAddr;

/**
 * Performs the 64-bit division and returns the quotient.
 *
 * If the divisor is 0, returns 0.
 *
 * It is not guaranteed to have 64-bit divide on all the platforms. So,
 * portable code should call this function instead of using / % operators on
 * 64-bit variables.
 */
static NV_FORCE_INLINE  NvU64
NvDiv64Inline(NvU64 dividend, NvU32 divisor)
{
    if (divisor == 0U) {
        return 0;
    }
    return dividend / divisor;
}

#define NvDiv64(dividend, divisor) NvDiv64Inline(dividend, divisor)

/**
 * Union that can be used to view a 32-bit word as your choice of a 32-bit
 * unsigned integer, a 32-bit signed integer, or an IEEE single-precision
 * float.  Here is an example of how you might use it to extract the (integer)
 * bitwise representation of a floating-point number:
 *   NvData32 data;
 *   data.f = 1.0f;
 *   printf("%x", data.u);
 */
typedef union NvData32Rec
{
    NvU32 u;
    NvS32 i;
    NvF32 f;
} NvData32;

/**
 * Generic data representation for both 32 and 64 bits data
 */
typedef union NvData64Rec
{
    NvU32 u;
    NvS32 i;
    NvF32 f;
    NvU64 u64;
    NvS64 i64;
    NvF64 d;
} NvData64;

/**
 * This structure is used to determine a location on a 2-dimensional object,
 * where the coordinate (0,0) is located at the top-left of the object.  The
 * values of x and y are in pixels.
 */
typedef struct NvPointRec
{
    /** horizontal location of the point */
    NvS32 x;

    /** vertical location of the point */
    NvS32 y;
} NvPoint;

typedef struct NvPointF32Rec
{
    /** horizontal location of the point */
    NvF32 x;

    /** vertical location of the point */
    NvF32 y;
} NvPointF32;
/**
 * This structure is used to define a 2-dimensional rectangle where the
 * rectangle is bottom right exclusive (that is, the right most column, and the
 * bottom row of the rectangle is not included).
 */
typedef struct NvRectRec
{
    /** left column of a rectangle */
    NvS32 left;

    /** top row of a rectangle*/
    NvS32 top;

    /** right column of a rectangle */
    NvS32 right;

    /** bottom row of a rectangle */
    NvS32 bottom;
} NvRect;

/**
 * This structure is used to define a 2-dimensional rectangle with floating
 * point coordinates.
 */
typedef struct NvRectF32Rec
{
    NvF32 left;
    NvF32 top;
    NvF32 right;
    NvF32 bottom;
} NvRectF32;

/**
 * This structure is used to define a 2-dimensional surface where the surface is
 * determined by it's height and width in pixels.
 */
typedef struct NvSizeRec
{
    /* width of the surface in pixels */
    NvS32 width;

    /* height of the surface in pixels */
    NvS32 height;
} NvSize;

/** @} */
#endif
#endif // INCLUDED_NVCOMMON_H
