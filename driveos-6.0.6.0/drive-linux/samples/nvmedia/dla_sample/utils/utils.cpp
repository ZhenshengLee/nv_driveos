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

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

#include "utils.h"

uint8_t *
readFileToMemory(
    std::string fileName,
    uint64_t &fileSize
)
{
    char *pBuffer {nullptr};

    std::ifstream is(fileName, std::ifstream::binary | std::ios::ate);
    CHECK_FAIL(is, "Open filename = %s", fileName.c_str());

    fileSize = is.tellg();

    pBuffer = new char[fileSize];
    CHECK_FAIL(pBuffer != nullptr, "alloc buffer");

    is.seekg (0, is.beg);

    is.read(pBuffer, fileSize);

fail:
    return reinterpret_cast<uint8_t *>(pBuffer);
}
