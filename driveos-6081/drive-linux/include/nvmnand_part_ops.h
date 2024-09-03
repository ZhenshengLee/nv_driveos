/*
 * Copyright (c) 2017-2018 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

/**
 * @file nvmnand_part_ops.h
 * @brief <b> NVIDIA mNAND library part specific operations header file </b>
 *
 * @b Description: This file declares the private interface in mnand library
 * for using mNAND with encapsulation of different vendor commands for health
 * related information.
 *
 */

#ifndef __NV_MNAND_PART_OPS_H__
#define __NV_MNAND_PART_OPS_H__

/**
 *
 * "Opening" the mNAND chip. This function needs to be called first to
 * identify the mNAND device and update device related content including
 * dev_type, desc, refresh_available, summary_available, storage_type
 * and refresh properties(rfsh_properties) specific to this chip. Allocates
 * memory if needed to be used in later functions. For properties not
 * supported initialize appropriately by setting flag to 0 and pointers to
 * NULL.
 *
 *
 * mnand_open and mnand_close to be used in pair.
 *
 * @param chip - the mNAND chip handle to be initialized
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
typedef MNAND_STATUS (*__mnand_open)(struct __mnand_chip *chip);

/**
 * "Closing" (aka releasing) the mNAND chip. This function needs to be
 * called last to free up any resources and data structures allocated in open
 *
 * mnand_open and mnand_close to be used in pair.
 *
 * @param chip - the mNAND chip handle to be used
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
typedef MNAND_STATUS (*__mnand_close)(struct __mnand_chip *chip);

/**
 * Updating the list of block info(if supported by the chip) inside mNAND
 * chip handle.
 *
 * @param chip - the mNAND chip handle to be updated
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
typedef MNAND_STATUS (*__mnand_update_block_info)(struct __mnand_chip *chip);

/**
 * Updating the summary info inside mNAND chip handle.
 *
 * @param chip - the mNAND chip handle to be updated
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
typedef MNAND_STATUS (*__mnand_update_summary)(struct __mnand_chip *chip);

/**
 * Checking EOL status – check EOL status from device and update status
 *
 * @param chip - the mNAND chip handle to be updated
 * @param eol_status - EOL status (MNAND_EOL_DETECTED = EOL set)
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
typedef MNAND_STATUS (*__mnand_check_eol)(struct __mnand_chip *chip,
    MNAND_EOL_STATUS *eol_status);

/**
 * Send "refresh" command to the mNAND chip. The purpose is to force data
 * rotation to improve the data retention.
 *
 * @param chip - the mNAND chip handle to be used
 * @param cached - specifies if mNAND device cached is enabled or not so that
 *                 special handling can be done of the device if needed.
 * @param block_type - which region (SLC/MLC) to be Refreshed (if supported)
 * @param num_blocks - number of blocks to be refreshed
 * @param rfsh_unit_time_us - time for which refresh operation is allowed
 *                            to execute(optional)
 * @param progress - refresh progress (obtained from device)
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
typedef MNAND_STATUS (*__mnand_send_refresh)(struct __mnand_chip *chip,
    int cached, uint8_t block_type, int num_blocks, uint32_t rfsh_unit_time_us,
    mnand_refresh_progress *progress);

/**
 * Extract refresh progress from the mNAND chip.
 *
 * @param chip - the mNAND chip handle to be used
 * @param rfsh_progress - current refresh progress (in percentage) as
 *                        obtained from the device. 100% means all the blocks
 *                        in the device is refreshed once. Similarly, 200% means
 *                        all block in the device is refreshed twice and so on.
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
typedef MNAND_STATUS (*__mnand_get_rfsh_progress)(struct __mnand_chip *chip,
    double *rfsh_progress);

/**
 * mNAND part/vendor specific operation block
 */
struct __mnand_operations {
    __mnand_open               open;             /* mandatory */
    __mnand_close              close;            /* mandatory */
    __mnand_send_refresh       send_rfsh;        /* optional */
    __mnand_get_rfsh_progress  get_rfsh_progress;/* optional */
    __mnand_update_block_info  update_blk_info;  /* optional */
    __mnand_update_summary     update_summary;   /* optional */
    __mnand_check_eol          check_eol;        /* optional */
};
typedef struct __mnand_operations mnand_operations;

#endif /* __NV_MNAND_PART_OPS_H__ */
