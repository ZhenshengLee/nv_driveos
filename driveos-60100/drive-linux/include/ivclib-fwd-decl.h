/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef IVCLIB_FWD_DECL_H
#define IVCLIB_FWD_DECL_H

#include "ivclib-static-analysis.h"

#ifdef __cplusplus
namespace Ivc {
extern "C" {
#endif

struct tegra_hv_queue_data;
struct ivc_queue;
struct sivc_queue;

STAN_2_3_PD_HYP_10576
struct ivc_shared_area;
STAN_2_3_PD_HYP_10576
struct ivc_info_page;
STAN_2_3_PD_HYP_10576
struct hyp_server_page;

#ifdef __cplusplus
} // extern "C"
} // namespace Ivc
#endif

#endif // IVCLIB_FWD_DECL_H
