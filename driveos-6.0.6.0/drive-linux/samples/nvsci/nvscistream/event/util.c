/* NvSciStream Event Loop Driven Sample App - utility functions
 *
 * Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software, related documentation and any
 * modifications thereto. Any use, reproduction, disclosure or distribution
 * of this software and related documentation without an express license
 * agreement from NVIDIA Corporation is strictly prohibited.
 */

#include "util.h"

#define CRC32_POLYNOMIAL 0xEDB88320L

static uint32_t calculateBufferCRC(
    uint32_t count,
    uint32_t crc,
    uint8_t* buffer)
{
    static uint32_t crcTable[256];
    static int initialized = 0;
    uint32_t i, j, tmp;

    if (!initialized) {
        for (i = 0; i <= 255; i++) {
            tmp = i;
            for (j = 8; j > 0; j--) {
                if (tmp & 1) {
                    tmp = (tmp >> 1) ^ CRC32_POLYNOMIAL;
                } else {
                    tmp >>= 1;
                }
            }
            crcTable[i] = tmp;
        }
        initialized = 1;
    }

    while (count-- != 0) {
        tmp = (crc >> 8) & 0x00FFFFFFL;
        crc = tmp ^ crcTable[((uint32_t) crc ^ *buffer++) & 0xFF];
    }

    return crc;
}

uint32_t generateCRC(
    uint8_t *data_ptr,
    uint32_t height,
    uint32_t width,
    uint32_t pitch)
{
    uint32_t y = 0U;
    uint32_t crc = 0U;
    for (y = 0U; y < height; y++) {
        crc = calculateBufferCRC(width, crc, data_ptr);
        data_ptr += pitch;
    }
    return crc;
}
