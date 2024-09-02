/*
 * Copyright (c) 2017-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef __NV_MNAND_UFS_UTIL_H__
#define __NV_MNAND_UFS_UTIL_H__

#include <stdio.h>
#include <stdint.h>
#include "nvmnand.h"
#include "nvmnand_part_ops.h"

/**
 * Common functions/definitions used on UFS. Chip independent
 */

#define UPIU_QUERY_OPCODE_READ_DESC    0x01
#define UPIU_QUERY_OPCODE_WRITE_DESC   0x02
#define UPIU_QUERY_OPCODE_READ_ATTR    0x03
#define UPIU_QUERY_OPCODE_WRITE_ATTR   0x04
#define UPIU_QUERY_OPCODE_READ_FLAG    0x05
#define UPIU_QUERY_OPCODE_SET_FLAG     0x06
#define UPIU_QUERY_OPCODE_CLEAR_FLAG   0x07
#define UPIU_QUERY_OPCODE_TOGGLE_FLAG  0x08

/* Descriptor IDN definitions */
#define QUERY_DESC_IDN_DEVICE          0x0
#define QUERY_DESC_IDN_CONFIGURAION    0x1
#define QUERY_DESC_IDN_UNIT            0x2
#define QUERY_DESC_IDN_GEOMETRY        0x7
#define QUERY_DESC_IDN_DEVICE_HEALTH   0x9

/* Flag IDN definitions */
#define QUERY_FLAG_IDN_BKOPS_EN        0x4
#define QUERY_FLAG_IDN_PURGE_EN        0x6
#define QUERY_FLAG_IDN_RFSH_EN         0x7

/* Attribute IDN definitions */
#define QUERY_ATTR_IDN_BKOPS_STATUS    0x5
#define QUERY_ATTR_IDN_PURGE_STATUS    0x6

/* UFS Descriptor size definitions */
#define MNAND_UFS_GEO_DESC_SIZE        0x48
#define MNAND_UFS_CFG_DESC_SIZE        0x90
#define MNAND_UFS_DEV_DESC_SIZE        0x40
#define MNAND_UFS_UNIT_DESC_SIZE       0x23
#define MNAND_UFS_HEALTH_DESC_SIZE     0x30

/* SCSI Command code definitions */
#define SCSI_INQUIRY_CMD_CODE          0x12
#define SCSI_MODE_SENSE_CMD_CODE       0x5A
#define SCSI_MODE_SELECT_CMD_CODE      0x55
#define SCSI_SYNC_CACHE10_CMD_CODE     0x35
#define SCSI_UNMAP_CMD_CODE            0x42
#define SCSI_READ_CAPACITY_CMD_CODE    0x25

/* SCSI Command block lenth definitions */
#define SCSI_INQUIRY_CMD_BLK_LEN       6
#define SCSI_MODE_SENSE_CMD_BLK_LEN    10
#define SCSI_MODE_SELECT_CMD_BLK_LEN   10
#define SCSI_SYNC_CACHE10_CMD_BLK_LEN  10
#define SCSI_UNMAP_CMD_BLK_LEN         10
#define SCSI_READ_CAPACITY_CMD_BLK_LEN 16

/* SCSI Command - Reply length definitions */
#define SCSI_STD_INQUIRY_REPLY_LEN     36

/* UFS - Power Down Power mode (used for PON) */
#define UFS_POWER_MODE_POWER_DOWN      3

/* Max number for commands per UFS Query combo */
#define MAX_CMD_PER_QUERY_COMBO        10

/* Max blocks that can be unmapped at a time */
#define UNMAP_BLOCK_COUNT_MAX          0xFFFFFFFF

/* Purge status definitions */
#define PURGE_STATUS_INPROGRESS        1
#define PURGE_STATUS_COMPLETED         3


/**
 * Send UFS combo query request
 *
 * @param fd - file descriptor to opened mNAND device
 * @param query_buf - pointer to query request buffer
 * @param num_cmds  - number of query requests in this combo request
 * @param need_cq_empty - Flag to specify if this request needs cq to be empty
 * @param ret_on_err - Flag to specify to return on error or continue furthur
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_send_combo_query_command(int fd,
    mnand_ufs_query_req *query_buf, uint8_t num_cmds, uint8_t need_cq_empty,
    uint8_t ret_on_err);

/**
 * Send UFS Query - Descriptor Read Request
 *
 * @param fd - file descriptor to opened mNAND device
 * @param idn - Descriptor identifier
 * @param index  - index
 * @param sel - selector
 * @param buffer - buffer to store the descriptor read
 * @param buf_size - size of the buffer passed/size of the descriptor read
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_query_desc_read(int fd, uint8_t idn, uint8_t index,
    uint8_t sel, uint8_t *buffer, uint32_t *buf_size);

/**
 * Send UFS Query - Descriptor Write Request
 *
 * @param fd - file descriptor to opened mNAND device
 * @param idn - Descriptor identifier
 * @param index  - index
 * @param sel - selector
 * @param buffer - buffer to store the descriptor to be written
 * @param buf_size - size of the buffer passed/size of the descriptor written
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_query_desc_write(int fd, uint8_t idn, uint8_t index,
    uint8_t sel, uint8_t *buffer, uint32_t *buf_size);

/**
 * Send UFS Query - Device Descriptor Read
 *
 * @param fd - file descriptor to opened mNAND device
 * @param buffer - buffer to store the descriptor read
 * @param buf_size - size of the buffer passed/size of the descriptor read
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_query_device_desc_read(int fd, uint8_t *buffer,
    uint32_t *buf_size);

/**
 * Send UFS Query - Config Descriptor Read
 *
 * @param fd - file descriptor to opened mNAND device
 * @param index  - index
 * @param buffer - buffer to store the descriptor read
 * @param buf_size - size of the buffer passed/size of the descriptor read
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_query_config_desc_read(int fd, uint8_t index,
    uint8_t *buffer, uint32_t *buf_size);

/**
 * Send UFS Query - Config Descriptor Write
 *
 * @param fd - file descriptor to opened mNAND device
 * @param index  - index
 * @param buffer - buffer to store the descriptor data to be updated in device
 * @param buf_size - size of the buffer passed/size of the descriptor written
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_query_config_desc_write(int fd, uint8_t index,
    uint8_t *buffer, uint32_t *buf_size);

/**
 * Send UFS Query - Geometry Descriptor Read
 *
 * @param fd - file descriptor to opened mNAND device
 * @param buffer - buffer to store the descriptor read
 * @param buf_size - size of the buffer passed/size of the descriptor read
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_query_geo_desc_read(int fd, uint8_t *buffer,
    uint32_t *buf_size);

/**
 * Send UFS Query - Unit Descriptor Read
 *
 * @param fd - file descriptor to opened mNAND device
 * @param index  - index
 * @param buffer - buffer to store the descriptor read
 * @param buf_size - size of the buffer passed/size of the descriptor read
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_query_unit_desc_read(int fd, uint8_t index,
    uint8_t *buffer, uint32_t *buf_size);

/**
 * Send UFS Query - Health Descriptor Read
 *
 * @param fd - file descriptor to opened mNAND device
 * @param index  - index
 * @param buffer - buffer to store the descriptor read
 * @param buf_size - size of the buffer passed/size of the descriptor read
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_query_health_desc_read(int fd, uint8_t index,
    uint8_t *buffer, uint32_t *buf_size);

/**
 * Send UFS Query - Attribute Read Request
 *
 * @param fd - file descriptor to opened mNAND device
 * @param idn - Attribute identifier
 * @param index  - index
 * @param sel - selector
 * @param attr - buffer to store the attribute read
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_query_attr_read(int fd, uint8_t idn, uint8_t index,
    uint8_t sel, uint32_t *attr);

/**
 * Send UFS Query - Attribute Write Request
 *
 * @param fd - file descriptor to opened mNAND device
 * @param idn - Attribute identifier
 * @param index  - index
 * @param sel - selector
 * @param attr - buffer to store the attribute to be written
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_query_attr_write(int fd, uint8_t idn, uint8_t index,
    uint8_t sel, uint32_t *attr);

/**
 * Send UFS Query - Attribute Read Request
 *
 * @param fd - file descriptor to opened mNAND device
 * @param idn - flag identifier
 * @param index  - index
 * @param sel - selector
 * @param attr - buffer to store the flag read
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_query_flag_read(int fd, uint8_t idn, uint8_t index,
    uint8_t sel, uint8_t *flag);

/**
 * Send UFS Query - Set Flag Request
 *
 * @param fd - file descriptor to opened mNAND device
 * @param idn - Flag identifier
 * @param index  - index
 * @param sel - selector
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_query_flag_set(int fd, uint8_t idn, uint8_t index,
    uint8_t sel);

/**
 * Send UFS Query - Clear Flag Request
 *
 * @param fd - file descriptor to opened mNAND device
 * @param idn - Flag identifier
 * @param index  - index
 * @param sel - selector
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_query_flag_clear(int fd, uint8_t idn, uint8_t index,
    uint8_t sel);

/**
 * Send UFS Query - Toggle Flag Request
 *
 * @param fd - file descriptor to opened mNAND device
 * @param idn - Flag identifier
 * @param index  - index
 * @param sel - selector
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_query_flag_toggle(int fd, uint8_t idn, uint8_t index,
    uint8_t sel);


/**
 * Send SCSI Inquiry Request
 *
 * @param fd - file descriptor to opened mNAND device
 * @param evpd - evpd flag for inquiry request
 * @param page_code  -page code for inquiry request
 * @param buffer - pointer to the buffer to store the inquiry data response
 * @param buf_size - size of the buffer/size of inquiry data read
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_scsi_inquiry(int fd, int8_t evpd, uint8_t page_code,
    uint8_t *buffer, uint32_t *buf_size);

/**
 * Send SCSI Mode Sense Request
 *
 * @param fd - file descriptor to opened mNAND device
 * @param page_code  - page code for mode sense request
 * @param sub_page_code  - sub page code for mode sense request
 * @param buffer - pointer to the buffer to store the mode sense data response
 * @param buf_size - size of the buffer/size of mode sense data read
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_scsi_mode_sense(int fd, uint8_t page_code,
    uint8_t sub_page_code, uint8_t *buffer, uint32_t *buf_size);

/**
 * Send SCSI Mode Select Request
 *
 * @param fd - file descriptor to opened mNAND device
 * @param page_code  - save pages option for mode select request
 * @param buffer - pointer to the buffer to store the mode select data
 * @param buf_size - size of the buffer/size of mode sense  data written
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_scsi_mode_select(int fd, int8_t save_pages,
    uint8_t *buffer, uint32_t *buf_size);

/**
 * Send SCSI Sync Cache Request
 *
 * @param fd - file descriptor to opened mNAND device
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_scsi_sync_cache(int fd);

/**
 * Send SCSI Mode Select Request
 *
 * @param fd - file descriptor to opened mNAND device
 * @param buffer - pointer to the buffer to store the unmap data
 * @param buf_size - size of the buffer/size of unmap data written
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_unmap(int fd, uint8_t *buffer, uint32_t *buf_size);

/**
 * Send SCSI Read Capacity Request
 *
 * @param fd - file descriptor to opened mNAND device
 * @param buffer - pointer to the buffer to store the read capacity response data
 * @param buf_size - size of the buffer/size of read capacity data received
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_read_capacity(int fd, uint8_t *buffer,
    uint32_t *buf_size);

/**
 * Set UFS power mode
 *
 * @param fd - file descriptor to opened mNAND device
 * @param power_mode - power mode to be set
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_set_power_mode(int fd, uint32_t power_mode);

/**
 * Send SCSI Read Buffer
 *
 * @param fd - file descriptor to opened mNAND device
 * @param cmd_blk - pointer to the buffer for command block
 * @param cmd_blk_len - command block buffer length
 * @param buffer - pointer to the buffer that stores the data read
 * @param buf_size - size of the read buffer
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_scsi_read_buffer(int fd, uint8_t *cmd_blk,
    uint32_t cmd_blk_len, uint8_t *buffer, uint32_t *buf_size);

/**
 * Send SCSI Write Buffer
 *
 * @param fd - file descriptor to opened mNAND device
 * @param cmd_blk - pointer to the buffer for command block
 * @param cmd_blk_len - command block buffer length
 * @param buffer - pointer to the buffer which has the data to be written
 * @param buf_size - size of the write buffer
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_ufs_scsi_write_buffer(int fd, uint8_t *cmd_blk,
    uint32_t cmd_blk_len, uint8_t *buffer, uint32_t *buf_size);

#endif /* __NV_MNAND_UFS_UTIL_H__ */
