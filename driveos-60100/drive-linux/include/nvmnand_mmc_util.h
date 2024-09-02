/*
 * Copyright (c) 2013-2018 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef __NV_MNAND_UTIL_H__
#define __NV_MNAND_UTIL_H__

#include <stdio.h>
#include <stdint.h>
#include "nvmnand.h"
#include "nvmnand_part_ops.h"

#ifdef QNX /* QNX */
#include <sys/dcmd_cam.h>
#else /* Linux */
#include "nvcommon.h"
#endif


#define max(a,b) NV_MAX(a, b)
#define min(a,b) NV_MIN(a, b)
#define ARRAY_SIZE(array) (sizeof((array))/sizeof((array)[0]))

/**
 * Common functions/definitions used on mNAND. Chip independent
 */

/* MMCSD passtrough commands used */
#define MMC_SEND_CID                2
#define MMC_SWITCH                  6
#define MMC_SEND_EXT_CSD            8
#define MMC_IF_COND                 8
#define MMC_SEND_CSD                9
#define MMC_SEND_STATUS             13
#define MMC_READ_SINGLE_BLOCK       17     /* read single block */
#define MMC_READ_MULTIPLE_BLOCK     18     /* read multiple blocks */
#define MMC_WRITE_BLOCK             24     /* write single block */
#define MMC_WRITE_MULTIPLE_BLOCK    25     /* write multiple blocks */
#define MMC_GEN_CMD                 56     /* Hynix F26/Toshiba specific */
#define MMC_MANF0_CMD               60     /* Hynix F20 specific */
#define MMC_STOP_TRANSMISSION       12
#define MMC_SECTOR_START            32     /* S */
#define MMC_SECTOR_END              33     /* S */
#define MMC_ERASE_GROUP_START       35     /* S */
#define MMC_ERASE_GROUP_END         36     /* S */
#define MMC_ERASE                   38     /* S */
/* Erase flags */
#define MMC_ERASE_NORM              0x00000000
#define MMC_ERASE_TRIM              0x00000001
#define MMC_ERASE_GARBAGE_COLLECT   0x80008001
#define MMC_ERASE_SECURE_TRIM       0x80000001
#define MMC_ERASE_SECURE            0x80000000
#ifdef QNX
#define MMC_COMBO_CMD               128    /* pseudo command for combos */
#define MMC_GET_CXD_CMD             129    /* pseudo command for CSD/CID */
#define MMC_TRIM_ALL_CMD            130    /* pseudo command for trimming the chip */
#endif

/* Parameters for setting EXT CSD */
#define MMC_SWITCH_CMDSET_DFLT   0x01
#define MMC_SWITCH_MODE_WRITE    0x3

/* CID content */
#define CID_MID    15
#define CID_PRV    6
#define CID_PNM    7

/* sector size */
#ifndef SECTOR_SZ
#define SECTOR_SZ               512
#endif

#define ARRAY_SIZE(array) (sizeof((array))/sizeof((array)[0]))

/* Some general EXT CSD content */
#define EXT_CSD_FLUSH_CACHE             32
#define EXT_CSD_CACHE_CTRL              33
#define EXT_CSD_PWR_OFF_NOTIFICATION    34
#define EXT_CSD_PWR_OFF_NOTIFICATION_NO_PWR_NOTIFY    0
#define EXT_CSD_PWR_OFF_NOTIFICATION_PWR_ON           1
#define EXT_CSD_PWR_OFF_NOTIFICATION_PWR_OFF_SHORT    2
#define EXT_CSD_PWR_OFF_NOTIFICATION_PWR_OFF_LONG     3
#define EXT_CSD_ENH_START_ADDR0         136
#define EXT_CSD_ENH_START_ADDR1         137
#define EXT_CSD_ENH_START_ADDR2         138
#define EXT_CSD_ENH_START_ADDR3         139
#define EXT_CSD_ENH_SIZE_MULT0          140
#define EXT_CSD_ENH_SIZE_MULT1          141
#define EXT_CSD_ENH_SIZE_MULT2          142
#define EXT_CSD_MAX_ENH_SIZE_MULT0      157
#define EXT_CSD_MAX_ENH_SIZE_MULT1      158
#define EXT_CSD_MAX_ENH_SIZE_MULT2      159
#define EXT_CSD_BKOPS_EN            	163
#define EXT_CSD_BKOPS_ENABLED       	1
#define EXT_CSD_BKOPS_START	        	164
#define EXT_CSD_BKOPS_INITIATE	    	1
#define EXT_CSD_REV                     192
#define EXT_CSD_REV_4_4                 5
#define EXT_CSD_REV_4_5                 6
#define EXT_CSD_REV_5_0                 7
#define EXT_CSD_REV_5_1                 8
#define EXT_CSD_BKOPS_SUPPORT       	502
#define EXT_CSD_SEC_COUNT0      		212
#define EXT_CSD_SEC_COUNT1      		213
#define EXT_CSD_SEC_COUNT2      		214
#define EXT_CSD_SEC_COUNT3      		215
#define EXT_CSD_BKOPS_STATUS            246
#define EXT_CSD_PWR_OFF_LONG_TIME       247
#define EXT_CSD_GENERIC_CMD6_TIME       248
#define EXT_CSD_FIRMWARE_VER0           254
#define EXT_CSD_FIRMWARE_VER1           255
#define EXT_CSD_FIRMWARE_VER2           256
#define EXT_CSD_FIRMWARE_VER3           257
#define EXT_CSD_FIRMWARE_VER4           258
#define EXT_CSD_FIRMWARE_VER5           259
#define EXT_CSD_FIRMWARE_VER6           260
#define EXT_CSD_FIRMWARE_VER7           261

#define HOURS_PER_DAY                   24
#define SECS_PER_HOUR                   3600
#define SECS_PER_DAY                    ((HOURS_PER_DAY) * (SECS_PER_HOUR))

/**
 * combo command
 */
typedef struct __mmc_combo_cmd_t {
    uint32_t cmd;
    uint32_t arg;
#ifndef QNX
    uint8_t *data_ptr;
    uint32_t data_len;
#endif
} mmc_combo_cmd_t;

#ifdef QNX
/**
 * CAM passthru using QNX SCSI layer (complex form).
 *
 * @param fd - file descriptor to opened mNAND device
 * @param cmd - MMCSD command
 * @param argument - MMCSD command argument
 * @param flags - CAM flags
 * @param senseptr - uses SCSI sense data to pass over extra information
 * @param senslen - the length of SCSI sense data
 * @param dataptr - input/output data
 * @param datalen - the length of I/O data
 * @param errmsg_prepend - in case of error, the header of error message
 * @param check_error - checking for error?
 *
 * @retval  0 success
 * @retval  else problem with operation
 */
int scsi_passthru_complex(int fd, uint8_t cmd, uint32_t argument,
    uint32_t flags, uint8_t *senseptr, int senselen,
    uint8_t *dataptr, int datalen, char *errmsg_prepend, int check_error);

/**
 * CAM passthru using QNX SCSI layer (simple form).
 *
 * @param fd - file descriptor to opened mNAND device
 * @param cmd - MMCSD command
 * @param argument - MMCSD command argument
 * @param dataptr - input/output data
 * @param datalen - the length of I/O data
 * @param errmsg_prepend - in case of error, the header of error message
 * @param check_error - checking for error?
 *
 * @retval  0 success
 * @retval  else problem with operation
 */
int scsi_passthru_simple(int fd, uint8_t cmd, uint32_t argument,
    uint8_t *dataptr, int datalen, char *errmsg_prepend, int check_error);
#endif

/**
 * Updating the card status.
 *
 * @param fd - file descriptor to opened mNAND device
 * @param card_status - pointer to card status to be updated
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_mmc_common_update_card_status(int fd, uint32_t *card_status);

/**
 * Updating the CID.
 *
 * @param fd - file descriptor to opened mNAND device
 * @param cid - pointer to CID to be updated
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_mmc_common_update_cid(int fd, mnand_cid_info *cid);

/**
 * Updating the CSD.
 *
 * @param fd - file descriptor to opened mNAND device
 * @param csd - pointer to CSD to be updated
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_mmc_common_update_csd(int fd, mnand_csd_info *csd);

/**
 * Updating the EXT CSD.
 *
 * @param fd - file descriptor to opened mNAND device
 * @param xcsd - pointer to EXT CSD to be updated
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_mmc_common_update_xcsd(int fd, mnand_ext_csd_info *xcsd);

/**
 * Setting the EXT CSD register.
 *
 * @param fd - file descriptor to opened mNAND device
 * @param index - which EXT CSD register to be set
 * @param value - the value of EXT CSD register to be set
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_mmc_common_set_xcsd(int fd, uint8_t index, uint8_t value);

/**
 * Trimming the whole device.
 *
 * @param fd - file descriptor point to mNAND device
 *
 * @retval   0: success
 * @retval  -1: problem with operation
 */
MNAND_STATUS mnand_mmc_common_trim_all(mnand_chip *chip);

#ifdef QNX
/**
 * Reading sectors into given buffer (pointed by virtual address).
 *
 * @param fd - file descriptor point to mNAND device
 * @param start_sector - the starting sector to be read
 * @param sector_cnt - the number of sectors to be read
 * @param virtaddr - buffer address (in virtual address form)
 *
 * @retval  number of sectors read
 * @retval  -1: problem with operation
 */
int mnand_mmc_common_read_sectors_virt(int fd, unsigned int start_sector,
    unsigned int sector_cnt, void *virtaddr);

/**
 * Reading sectors into given buffer (pointed by physical address).
 *
 * @param fd - file descriptor point to mNAND device
 * @param start_sector - the starting sector to be read
 * @param sector_cnt - the number of sectors to be read
 * @param physaddr - buffer address (in physical address form)
 *
 * @retval  number of sectors read
 * @retval  -1: problem with operation
 */
int mnand_mmc_common_read_sectors_phys(int fd, unsigned int start_sector,
    unsigned int sector_cnt, off_t physaddr);

#else /* Linux */
/**
 * send a list of eMMC command to mnand device
 *
 * @param fd - file descriptor point to mNAND device
 * @param ccs - the buffer address as array of commands
 * @param num_cmd - the number of commands encapsulate in ccs
 *
 * @retval  MNAND_OK: for success
 * @retval  -1: problem with operation
 */
int mnand_send_combo_cmd(int fd, mmc_combo_cmd_t *ccs, int num_cmd);

/**
 * send a list of eMMC command to mnand device
 *
 * @param fd - file descriptor point to mNAND device
 * @param ccs - the buffer address for one commands
 *
 * @retval  MNAND_OK: for success
 * @retval  -1: problem with operation
 */

int mnand_send_cmd_one(int fd, mmc_combo_cmd_t *ccs);

/**
 * Get the power-of-2 aligned user sector count
 *
 * @param chip - the mNAND chip handle to be used
 *
 * @retval   aligned sector count
 */
uint32_t mnand_mmc_common_get_aligned_sector_count(mnand_chip *chip);

/**
 * Setting the EXT CSD register and update the xcsd.
 *
 * @param fd - file descriptor to opened mNAND device
 * @param index - which EXT CSD register to be set
 * @param value - the value of EXT CSD register to be set
 * @param xcsd - update the value of EXT CSD register
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_mmc_common_set_update_xcsd(int fd, uint8_t index,
    uint8_t enabled, mnand_ext_csd_info *xcsd);
#endif
#endif /* __NV_MNAND_UTIL_H__ */
