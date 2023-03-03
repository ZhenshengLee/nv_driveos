/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _UTILS_H_
#define _UTILS_H_

#include <cstdint>
#include <string>
#include "cLogger.h"

#define CHECK_FAIL(condition, message, ...)                        \
    do {                                                           \
        if( !(condition)) {                                        \
            LOG_ERR("%s Fail\n", message, ## __VA_ARGS__);         \
            goto fail ;                                            \
        } else {                                                   \
            LOG_DBG("%s Successful\n", message, ## __VA_ARGS__);  \
        }                                                          \
    } while (0)

#define PROPAGATE_ERROR_FAIL(condition, message, ...) \
    do {                                                           \
        if( !(condition)) {                                        \
            LOG_ERR("%s Fail\n", message, ## __VA_ARGS__);         \
            status = NVMEDIA_STATUS_ERROR;                         \
            goto fail ;                                            \
        } else {                                                   \
            LOG_DBG("%s Successful\n", message, ## __VA_ARGS__);  \
        }                                                          \
    } while (0)

uint8_t *
readFileToMemory(
    std::string fileName,
    uint64_t &fileSize
);

#endif // end of _UTILS_H_
