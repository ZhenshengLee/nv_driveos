/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited
 */

#ifndef INCLUDE_NVTYPES_H
#define INCLUDE_NVTYPES_H

#if (defined(__QNX__) && defined(NV_IS_SAFETY) && (NV_IS_SAFETY == 1))
#include "nvtypes_tegra_safety.h"
#else
#include "p4-mirror/sdk/nvidia/inc/nvtypes.h"
#endif

#endif /* End of INCLUDE_NVTYPES_H */
