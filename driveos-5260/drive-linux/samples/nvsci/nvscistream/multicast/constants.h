//
// Streaming constants
//
// Copyright (c) 2019-2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace NvScistreamSample
{
    constexpr uint32_t NUM_PACKETS = 4;
    constexpr uint32_t NUM_ELEMENTS_PER_PACKET = 1;
    constexpr uint32_t NUM_PROD_SYNCS = 1;
    constexpr uint32_t NUM_CONS_SYNCS = 1;
    constexpr uint32_t MAX_NUM_SYNCS = 8;
    constexpr uint32_t NUM_FRAMES = 10;
    constexpr int QUERY_TIMEOUT = 10000; // usecs
    constexpr int QUERY_TIMEOUT_FOREVER = -1;
    constexpr int MAX_QUERY_TIMEOUTS = 1000U;

    constexpr uint32_t WIDTH = 1920;
    constexpr uint32_t HEIGHT = 1080;
}

#endif
