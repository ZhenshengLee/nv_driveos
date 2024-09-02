/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef LIB_CAM_FSYNC_H
#define LIB_CAM_FSYNC_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status codes for fsync group program start time operation.
 */
typedef enum CAM_FSYNC_STATUS
{
    /** The operation was successful. */
    CAM_FSYNC_OK = 0,
    /** Group is busy. Generator in group already running. */
    CAM_FSYNC_GROUP_BUSY,
    /** Group not found for specified group id. */
    CAM_FSYNC_GROUP_NOT_FOUND,
    /** The operation encountered fault. Start time may be in past or failed to start the generator. */
    CAM_FSYNC_GROUP_FAULT,
    /** A generic error occured during the operation. */
    CAM_FSYNC_ERROR,
    /** The requested operation is not supported. */
    CAM_FSYNC_UNSUPPORTED,
#if !(NV_IS_SAFETY)
    /** Group is idle. Generator in group is not running. */
    CAM_FSYNC_GROUP_IDLE,
    /** The generator configuration is invalid. */
    CAM_FSYNC_INVALID_CONFIG,
    /** Generator not found for specified generator id. */
    CAM_FSYNC_GENERATOR_NOT_FOUND,
#endif
} CAM_FSYNC_STATUS;


/**
 * @brief Program and start generators in group with absolute start time in TSC ticks
 *
 * This function programs and start all the generator in group identified by group ID
 * at an absolute start time given in TSC ticks.
 *
 * @pre None
 *
 * @param[in] group_id                     Group id for the node
 *                                         Valid range: [ @ref MIN_GROUP_ID_DEFINED_IN_DT, @ref MAX_GROUP_ID_DEFINED_IN_DT]
 * @param[in] start_time_in_tsc_ticks      Start time in TSC ticks to program generators in group
 *                                         Valid range: [ @ref TSC_CURRENT_TICKS, @ref MAX_UINT64]
 *
 * @retval CAM_FSYNC_OK     Success to program and start fsync in @a group_id with @a start_time_in_tsc_ticks
 * @retval Error status defined in @ref CAM_FSYNC_STATUS.
 */
CAM_FSYNC_STATUS cam_fsync_program_abs_start_value(uint32_t group_id, uint64_t start_time_in_tsc_ticks);

#if !(NV_IS_SAFETY)

/**
 * @brief Stop all generators in the requested group.
 *
 * @pre None
 *
 * @param[in] group_id                     Group id for the node
 *                                         Valid range: [ @ref MIN_GROUP_ID_DEFINED_IN_DT, @ref MAX_GROUP_ID_DEFINED_IN_DT]
 *
 * @retval CAM_FSYNC_OK                    Successfully stopped all generators in the requested group.
 * @retval CAM_FSYNC_GROUP_IDLE            Failed to stop the generators - the group is already stopped.
 * @retval CAM_FSYNC_GROUP_NOT_FOUND       The input group ID was not found.
 * @retval CAM_FSYNC_UNSUPPORTED           The operation is not supported (Linux only).
 * @retval CAM_FSYNC_ERROR                 A generic error occurred.
 */
CAM_FSYNC_STATUS cam_fsync_stop_group(uint32_t group_id);

/**
 * @brief Reconfigure the period, duty cycle, and relative offset of a given generator.
 *
 * @pre The generator and its group must be in a stopped state.
 *
 * @param[in] group_id                     Group id for the node
 *                                         Valid range: [@ref MIN_GROUP_ID_DEFINED_IN_DT, @ref MAX_GROUP_ID_DEFINED_IN_DT]
 * @param[in] generator_id                 ID of the generator within the group. The ID of a
 *                                         generator is its index within the group (defined in DT).
 *                                         Valid range: [@ref MIN_GENERATOR_ID_DEFINED_IN_DT, @ref MAX_GENERATOR_ID_DEFINED_IN_DT]
 * @param[in] frequnecy_hz                 The new frequency, in Hz. On Linux, all generators within the group
 *                                         must have a least-common-multiple frequency not greater than 120Hz.
 *                                         On QNX, all generators must not be greater than 120Hz and rational multiples
 *                                         of the highest frequency.
 *                                         Valid range: [> 0, 120]
 * @param[in] duty_cycle                   The new duty cycle, in whole percent.
 *                                         Valid range: [> 0, < 100]
 * @param[in] offset_ms                    The new relative offset, in milliseconds.
 *                                         Value range: [non-zero]
 *
 * @retval CAM_FSYNC_OK                    The requested generator was successfully reconfigured.
 * @retval CAM_FSYNC_GROUP_BUSY            The group is not stopped - the generator cannot be reconfigured.
 * @retval CAM_FSYNC_INVALID_CONFIG        The input frequency, duty cycle, or relative offset is out-of-range.
 *                                         The input frequency, duty cycle, or relative offset conflicts with
 *                                         other generators in the group.
 * @retval CAM_FSYNC_GROUP_NOT_FOUND       The input group ID was not found.
 * @retval CAM_FSYNC_GENERATOR_NOT_FOUND   The input generator ID was not found.
 * @retval CAM_FSYNC_UNSUPPORTED           The operation is not supported (Linux only).
 * @retval CAM_FSYNC_ERROR                 A generic error occurred.
 */
CAM_FSYNC_STATUS cam_fsync_reconfigure_generator(uint32_t group_id,
                                                 uint32_t generator_id,
                                                 uint32_t frequency_hz,
                                                 uint32_t duty_cycle,
                                                 uint32_t offset_ms);

#endif

#ifdef __cplusplus
}
#endif

#endif
