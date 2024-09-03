/* Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "pwr_utils.h"

nvsipl::SIPLStatus
PowerControlSetAggregatorPower(
    uint32_t const pwrPortNum,
    bool const passive,
    bool const poweredOn)
{
    (void)pwrPortNum;
    (void)passive;
    (void)poweredOn;
    return nvsipl::NVSIPL_STATUS_OK;
}

nvsipl::SIPLStatus
PowerControlSetUnitPower(
    uint32_t const pwrPortNum,
    uint8_t const uLinkIndex,
    bool const poweredOn)
{
    (void)pwrPortNum;
    (void)uLinkIndex;
    (void)poweredOn;
    return nvsipl::NVSIPL_STATUS_OK;
}
