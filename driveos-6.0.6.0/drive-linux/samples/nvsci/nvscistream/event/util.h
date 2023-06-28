/* NvSciStream Event Loop Driven Sample App - utilities
 *
 * Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software, related documentation and any
 * modifications thereto. Any use, reproduction, disclosure or distribution
 * of this software and related documentation without an express license
 * agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _UTIL_H
#define _UTIL_H 1

#include <stdint.h>

/* CRC checksum generator */
extern uint32_t generateCRC(
    uint8_t *data_ptr,
    uint32_t height,
    uint32_t width,
    uint32_t pitch);

#endif // _UTIL_H
