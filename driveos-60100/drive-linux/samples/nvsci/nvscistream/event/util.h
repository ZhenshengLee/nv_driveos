/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/*
 * NvSciStream Event Loop Driven Sample App - utilities
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
