/* Copyright (c) 2023 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef LIB_CAM_FSYNC_H
#define LIB_CAM_FSYNC_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Enum defining status codes for fsync group program start time operation
 * The CAM_FSYNC_STATUS enum defines the following status codes:
 * * CAM_FSYNC_OK: The operation was successful
 * * CAM_FSYNC_GROUP_BUSY: Group is busy. Generator in group already running
 * * CAM_FSYNC_GROUP_NOT_FOUND: Group not found for specified group id
 * * CAM_FSYNC_GROUP_FAULT: The operation encountered fault. Start time may be in past or failed to start the generator
 * * CAM_FSYNC_ERROR: A generic error occured during the operation
 * * CAM_FSYNC_UNSUPPORTED: The requested operation is not supported
*/
typedef enum CAM_FSYNC_STATUS
{
    CAM_FSYNC_OK = 0,
    CAM_FSYNC_GROUP_BUSY,
    CAM_FSYNC_GROUP_NOT_FOUND,
    CAM_FSYNC_GROUP_FAULT,
    CAM_FSYNC_ERROR,
    CAM_FSYNC_UNSUPPORTED
} CAM_FSYNC_STATUS;


/*
 * @brief Program and start generators in group with absolute start time in TSC ticks
 * This function programs and start all the generator in group identified by group ID
 * at an abosolute start time given in TSC ticks.
 * The generators for the group are defined in the device tree.
 *
 * @param[in] group_id                     Group id for the node
 *                                         [MIN_GROUP_ID_DEFINED_IN_DT: MAX_GROUP_ID_DEFINED_IN_DT]
 * @param[in] start_time_in_tsc_ticks      Start time in TSC ticks to program generators in group
 *                                         [TSC_CURRENT_TICKS < value < MAX_UINT64]
 *
 * @retval CAM_FSYNC_STATUS (success = CAM_FSYNC_OK)
 */
CAM_FSYNC_STATUS cam_fsync_program_abs_start_value(uint32_t group_id, uint64_t start_time_in_tsc_ticks);

#ifdef __cplusplus
}
#endif

#endif
