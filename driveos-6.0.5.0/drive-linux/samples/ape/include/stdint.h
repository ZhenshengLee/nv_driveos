/*
 * Copyright (c) 2008 Travis Geiselbrecht
 * Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifndef __STDINT_H
#define __STDINT_H

#include <limits.h> // for ULONG_MAX

typedef unsigned char		uint8_t;
typedef unsigned short		uint16_t;
typedef unsigned int		uint32_t;
typedef unsigned long long	uint64_t;
typedef signed char		int8_t;
typedef short			int16_t;
typedef int			int32_t;
typedef long long		int64_t;

typedef int8_t			int_least8_t;
typedef int16_t			int_least16_t;
typedef int32_t			int_least32_t;
typedef int64_t			int_least64_t;
typedef uint8_t			uint_least8_t;
typedef uint16_t		uint_least16_t;
typedef uint32_t		uint_least32_t;
typedef uint64_t		uint_least64_t;

typedef int8_t			int_fast8_t;
typedef int16_t			int_fast16_t;
typedef int32_t			int_fast32_t;
typedef int64_t			int_fast64_t;
typedef uint8_t			uint_fast8_t;
typedef uint16_t		uint_fast16_t;
typedef uint32_t		uint_fast32_t;
typedef uint64_t		uint_fast64_t;

typedef long			intptr_t;
typedef unsigned long		uintptr_t;

typedef long long		intmax_t;
typedef unsigned long long	uintmax_t;

#define INT8_MIN		(-128)
#define INT16_MIN		(-32768)
#define INT32_MIN		(-2147483648)
#define INT64_MIN		(-9223372036854775807LL-1LL)

#define INT8_MAX		(+128-1)
#define INT16_MAX		(+32768-1)
#define INT32_MAX		(+2147483648-1)
#define INT64_MAX		(+9223372036854775807LL)

#define UINT8_MAX		(256-1)
#define UINT16_MAX		(65536-1)
#define UINT32_MAX		(4294967296U-1U)
#define UINT64_MAX		(18446744073709551615ULL)

#define INT_LEAST8_MIN		(-(128-1))
#define INT_LEAST16_MIN		(-(32768-1))
#define INT_LEAST32_MIN		(-(2147483648-1))
#define INT_LEAST64_MIN		(-(9223372036854775807LL))

#define INT_LEAST8_MAX		INT8_MAX
#define INT_LEAST16_MAX		INT16_MAX
#define INT_LEAST32_MAX		INT32_MAX
#define INT_LEAST64_MAX		INT64_MAX

#define UINT_LEAST8_MAX		UINT8_MAX
#define UINT_LEAST16_MAX	UINT16_MAX
#define UINT_LEAST32_MAX	UINT32_MAX
#define UINT_LEAST64_MAX	UINT64_MAX

#define INT_FAST8_MIN		(-(128-1))
#define INT_FAST16_MIN		(-(32768-1))
#define INT_FAST32_MIN		(-(2147483648-1))
#define INT_FAST64_MIN		(-(9223372036854775807LL))

#define INT_FAST8_MAX		INT8_MAX
#define INT_FAST16_MAX		INT16_MAX
#define INT_FAST32_MAX		INT32_MAX
#define INT_FAST64_MAX		INT64_MAX

#define UINT_FAST8_MAX		UINT8_MAX
#define UINT_FAST16_MAX		UINT16_MAX
#define UINT_FAST32_MAX		UINT32_MAX
#define UINT_FAST64_MAX		UINT64_MAX

#define INTPTR_MIN		LONG_MIN
#define INTPTR_MAX		LONG_MAX
#define UINTPTR_MAX		ULONG_MAX

#define INTMAX_MIN		(-(9223372036854775807LL))
#define INTMAX_MAX		(9223372036854775807LL)
#define UINTMAX_MAX		(18446744073709551615ULL)

#define SIZE_MAX		ULONG_MAX

#endif
