/* Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/**
 * \file
 * \brief <b> NVIDIA Device Block: Power Control Utilities </b>
 *
 * This file contains the DeviceBlock internal API for interacting with NVidia
 * Camera Control Protocol (NVCCP)
 */

#ifndef POWER_UTILS_H
#define POWER_UTILS_H

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
#include <cstdbool>
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#else
#include <stdbool.h>
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif
#include "NvSIPLCommon.hpp"

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
extern "C" {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif

/**
 * \brief Set the power status of a CSI aggretator.
 *
 * If requesting power on (by passing `true` for the `powerOn` parameter), the
 * interface must currently be powered down. A poweron request is sent to both
 * the aggregator. On success, the aggregator will be powered on.
 *
 * If requesting power off (by passing `false` for the `powerOn` parameter),
 * the interface must be owned and powered up. A poweroff request is sent to both
 * the aggregator and the interface's cameras. On success, the interface will
 * powered off.
 *
 * Failure may leave the interface in an unrecoverable indeterminate state.
 *
 * The interface is mapped to an internal power port name as specified by
 * `PowerControlPortFromInterface`.
 *
 * @param[in]   pwrPortNum  The port to request power change on. Must be in the
 *                          range 0 <= pwrPortNum <= 3.
 * @param[in]   passive       True if the interface should be acquired in passive
 *                          mode, false otherwise.
 * @param[in]   poweredOn   The requested power state.
 *                          `true` means power should be applied,
 *                          `false` means power should be withdrawn.
 *
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if the provided interface is invalid. (Failure)
 * @retval NVMEDIA_STATUS_ERROR if any internal operations fail. (Failure)
 * @retval NVMEDIA_STATUS_OK on success.
 */
nvsipl::SIPLStatus
PowerControlSetAggregatorPower(
    uint32_t const pwrPortNum,
    bool const passive,
    bool const poweredOn);

/**
 * \brief Control the power state of an individual link on the given interface.
 *
 * If the provided port number/link index combination is invalid, the Port A
 * Link 0 camera will be controlled.
 *
 * If `poweredOn` is true, on success the given link will be powered up.
 * If `poweredOn` is false, on success the given link will be powered down.
 *
 * The port number/link index combination is mapped to a  NvCCP camera as
 * specified by `sGetCamId`.
 *
 * @todo We probably want to validate the interface/link combination rather than
 * blindly sending power on/off requests to A0
 *
 * @param[in]   pwrPortNum  The port to request power change on. Must be in the
 *                          range 0 <= pwrPortNum <= 3.
 * @param[in]   uLinkIndex  The link index on which to request a power state
 *                          change.
 * @param[in]   poweredOn   Whether to request power up (`true`) or power down
 *                          (`false`).
 *
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if the given port is invalid. (Failure)
 * @retval NVMEDIA_STATUS_ERROR if an internal operation fails. (Failure)
 * @retval NVMEDIA_STATUS_OK if the operation completes successfully.
 *
 */
nvsipl::SIPLStatus
PowerControlSetUnitPower(
    uint32_t const pwrPortNum,
    uint8_t const uLinkIndex,
    bool const poweredOn);

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif

/** @} */ // ends NVSIPLDevBlk_PowerControl

#endif /* POWER_UTILS_H */
