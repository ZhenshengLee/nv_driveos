/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef NVSIPLPLATFORMCFG_HPP
#define NVSIPLPLATFORMCFG_HPP

#include "NvSIPLDeviceBlockInfo.hpp"

#include <string>
#include <cstdint>
#include <vector>

/**
 * @file
 *
 * @brief <b> NVIDIA SIPL: Camera Platform Configuration </b>
 *
 */

namespace nvsipl
{

/** @addtogroup NvSIPL
 * @{
 */

/** @brief Defines the camera platform configuration.
 *
 * Describes up to @ref MAX_DEVICEBLOCKS_PER_PLATFORM deserializers connected to Tegra. */
struct PlatformCfg
{
    /** Holds the platform name. For example, "ddpx-a". */
    std::string platform = "";
    /** Holds the platform configuration name. */
    std::string platformConfig = "";
    /** Holds the platform configuration description. */
    std::string description = "";
    /** Holds the number of device blocks.
     * This value must be less than or equal to @ref MAX_DEVICEBLOCKS_PER_PLATFORM. */
    uint32_t numDeviceBlocks = 0U;
    /** Holds an array of @ref DeviceBlockInfo. */
    DeviceBlockInfo deviceBlockList[MAX_DEVICEBLOCKS_PER_PLATFORM];
};

/** @} */

}// namespace nvsipl

#endif // NVSIPLPLATFORMCFG_HPP
