/*
 * Copyright (c) 2013-2022 NVIDIA Corporation.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * Neither the name of the NVIDIA Corporation nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

/**
 * @file nvmnand.h
 * @brief <b> NVIDIA mNAND library </b>
 *
 * @b Description: This files declares the interface for using mNAND
 * with encapsulation of different vendor commands for health related
 * information.
 *
 */

#ifndef __NV_MNAND_H__
#define __NV_MNAND_H__

/**
 * @defgroup mNAND interface.
 *
 * This interface can be used to communicate with mNAND via vendor
 * specific commands, or more generic commands (JDEC spec. 4.51).
 *
 */

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#if defined(__cplusplus)
extern "C"
{
#endif

/**
 * max. erase count a block can have
 */
#define MAX_ERASE_CNT    0xFFFF

/**
 * sector size
 */
#ifndef SECTOR_SZ
#define SECTOR_SZ               512
#endif

/**
 * Read/Write Cache enable bits used in cache handling API
 */
#define  MNAND_READ_CACHE_ENABLE   (1 << 0)
#define  MNAND_WRITE_CACHE_ENABLE  (1 << 1)

/* forward declaration */
struct __mnand_chip;
struct __mnand_operations;

/**
 * information per block
 */
typedef struct __mnand_block_info {
    uint16_t block_number;    /* block#, absolute */
    uint16_t block_age;       /* age (aka erase, p/e) count */
    uint16_t block_type;      /* block type */
        #define MNAND_UNKNOWN_BLOCK         (1 << 0)
        #define MNAND_MLC_BLOCK             (1 << 1)
        #define MNAND_SLC_BLOCK             (1 << 2)
        #define MNAND_UNKNOWN_BAD_BLOCK     (1 << 8)
        #define MNAND_MANF_BAD_BLOCK        (1 << 9)
        #define MNAND_RUNTIME_BAD_BLOCK     (1 << 10)
        #define MNAND_SYSTEM_BLOCK          (1 << 11)
    uint16_t valid;           /* is this entry valid? */
} mnand_block_info;

/**
 * information on life time element
 */
typedef union __mnand_life_time_element {
    uint16_t spare_blocks;    /* #spare block in this region */
    double   percentage;      /* Life time in percentage */
} mnand_life_time_element;

/**
 * information on life time
 */
typedef struct __mnand_life_time_info {
    mnand_life_time_element life_time;
    uint8_t  life_time_type;  /* how the lifetime being presented */
        #define MNAND_LIFETIME_INVALID     (1 << 0)
        #define MNAND_LIFETIME_IN_BLOCK    (1 << 1)
        #define MNAND_LIFETIME_IN_PERCENT  (1 << 2)
} mnand_life_time_info;

/**
 * information on refresh element
 */
typedef union __mnand_refresh_element {
    uint16_t block;           /* last block being refresh */
    double   percentage;      /* refresh percentage */
} mnand_refresh_element;

/**
 * information on refresh progress
 */
typedef struct __mnand_refresh_progress {
    mnand_refresh_element progress;
    uint8_t progress_type;    /* how the progress being presented */
        #define MNAND_RFSH_PROG_INVALID     (1 << 0)
        #define MNAND_RFSH_PROG_IN_BLOCK    (1 << 1)
        #define MNAND_RFSH_PROG_IN_PERCENT  (1 << 2)
} mnand_refresh_progress;

/**
 * information on refresh properties
 */
typedef struct __mnand_refresh_properties {
    uint32_t rfsh_days;               /* whole chip refreshes in given days */
    uint32_t def_rfsh_interval;       /* default refresh interval(msec) */
    uint32_t min_rfsh_interval;       /* minimum refresh interval(msec) */
    uint32_t shutdown_rfsh_interval;  /* min shutdown refresh interval(msec) */
    uint32_t rfsh_calibrate_cnt;      /* min refreshes before re-calibration */
    uint32_t rfsh_unit_time_us;       /* time for each refresh operation(UFS only) */
} mnand_refresh_properties;

/**
 * summary of ages
 */
typedef struct __mnand_ages_element {
    uint16_t min_age;         /* min. age in this region */
    uint16_t max_age;         /* max. age in this region */
    double   avg_age;         /* avg. age in this region */
    uint32_t total_age;       /* total age (if available) */
    uint8_t valid;            /* is this entry valid? */
} mnand_ages_info;

/**
 * summary of given block type (or region)
 */
typedef struct __mnand_summary_element {
    mnand_life_time_info life;/* life time information */
    mnand_ages_info ages;     /* ages information */
    uint8_t valid;            /* is this entry valid? */
} mnand_summary_element;

/**
 * aggregation of summaries
 */
typedef struct __mnand_summary_info {
    mnand_summary_element mlc;    /* MLC region */
    mnand_summary_element slc;    /* SLC region */
    mnand_summary_element total;  /* per chip */
    int factory_bb_cnt;       /* #factory bad block, -1=invalid */
    int runtime_bb_cnt;       /* #runtime bad block, -1=invalid */
} mnand_summary_info;

/**
 * CID - eMMC only
 */
typedef struct __mnand_cid_info {
    uint8_t data[16];    /* JDEC spec. 4.51, table 64, 128bit wide */
} mnand_cid_info;

/**
 * CSD - eMMC only
 */
typedef struct __mnand_csd_info {
    uint8_t data[16];    /* JDEC spec. 4.51, table 67, 128bit wide */
} mnand_csd_info;

/**
 * EXT CSD - eMMC only
 */
typedef struct __mnand_ext_csd_info {
    uint8_t data[512];   /* JDEC spec. 4.51, table 82, 512byte wide */
} mnand_ext_csd_info;

/**
 * DEVICE DESC - UFS Only
 */
typedef struct __mnand_device_desc_info {
    uint8_t data[64];   /* JESD220C spec. table 14.4, 64byte wide */
} mnand_device_desc_info;

/**
 * CONFIG DESC - UFS Only
 */
typedef struct __mnand_config_desc_info {
    uint8_t data1[144];   /* JESD220C spec. table 14.10, 144byte wide */
    uint8_t data2[144];   /* JESD220C spec. table 14.10, 144byte wide */
    uint8_t data3[144];   /* JESD220C spec. table 14.10, 144byte wide */
    uint8_t data4[144];   /* JESD220C spec. table 14.10, 144byte wide */
} mnand_config_desc_info;

/**
 * GEOMETRY DESC - UFS Only
 */
typedef struct __mnand_geo_desc_info {
    uint8_t data[72];   /* JESD220C spec. table 14.13, 72byte wide */
} mnand_geo_desc_info;

/**
 * UNIT DESC - UFS Only
 */
typedef struct __mnand_unit_desc_info {
    uint8_t data[19];   /* JESD220C spec. table 14.14, 19byte wide */
} mnand_unit_desc_info;

/**
 * MMC GEOMETRY INFO - eMMC only
 */
typedef struct __mnand_mmc_geo_info {
    uint64_t sec_count;
    uint64_t enh_size_mult;
    uint64_t enh_start_addr;
    uint64_t hc_wp_grp_size;
    uint64_t hc_erase_grp_size;
    uint64_t max_enh_size_mult;
    uint64_t partition_size_alignment;
    uint64_t user_capacity;
} mnand_mmc_geo_info;

/**
 * UFS GEOMETRY INFO - UFS only
 */
typedef struct __mnand_ufs_geo_info {
    uint64_t total_raw_capacity;
    uint8_t  max_lu;
    uint32_t segment_size;
    uint8_t  alloc_unit_size;
    uint16_t supported_memory_types;
    uint32_t sys_code_max_alloc;
    uint16_t sys_code_cap_adj_fac;
    uint32_t non_persist_max_alloc;
    uint16_t non_persist_cap_adj_fac;
    uint32_t enhanced1_max_alloc;
    uint16_t enhanced1_cap_adj_fac;
    uint32_t enhanced2_max_alloc;
    uint16_t enhanced2_cap_adj_fac;
    uint32_t enhanced3_max_alloc;
    uint16_t enhanced3_cap_adj_fac;
    uint32_t enhanced4_max_alloc;
    uint16_t enhanced4_cap_adj_fac;
} mnand_ufs_geo_info;

/**
 * GEOMETRY INFO
 */
typedef union __mnand_geo_info {
    mnand_mmc_geo_info mmc_geo_info;
    mnand_ufs_geo_info ufs_geo_info;
} mnand_geo_info;

/**
 * LOGICAL UNIT INFO - UFS Only
 */
typedef struct __mnand_lu_info {
    uint8_t lu_enabled;
    uint8_t memory_type;
    uint8_t logical_blk_sz;
    uint32_t  num_alloc_units;
} mnand_lu_info;

/**
 * mNAND device enumeration
 */
typedef enum __mnand_device_type {
    MNAND_CATCH_ALL = 0,   /* "catch-all", used when none matched */
    MNAND_HYNIX_F26 = 1,   /* Hynix F26 */
    MNAND_HYNIX_F20 = 2,   /* Hynix F20 */
    MNAND_HYNIX_F16 = 3,   /* Hynix F16 */
    MNAND_MICRON    = 4,   /* Samsung eMMC5.0 */
    MNAND_SAMSUNG50 = 5,   /* Samsung eMMC5.0 */
    MNAND_SAMSUNG51 = 6,   /* Samsung eMMC5.1 */
    MNAND_TOSHIBA   = 7,   /* Toshiba */
    MNAND_UFS_TOSHIBA = 8,   /* Toshiba UFS */
    MNAND_UFS_SAMSUNG = 9,   /* Samsung UFS */
    MNAND_UFS_MICRON = 10,   /* Micron UFS */
} MNAND_DEVICE_TYPE;


/**
 * mNAND storage type enumeration
 */
typedef enum __mnand_storage_type {
    MNAND_EMMC = 0,   /* eMMC storage device */
    MNAND_UFS = 1,   /* UFS storage device */
    MNAND_UNKNOWN = 2,   /* unknown storage - error */
} MNAND_STORAGE_TYPE;

/**
 * mNAND EOL status enumeration
 */
typedef enum __mnand_eol_status {
    MNAND_EOL_OK = 0,        /* Not yet EOL */
    MNAND_EOL_DETECTED = 1,  /* EOL detected */
} MNAND_EOL_STATUS;

/**
 * mNAND operation status used in the API
 */
typedef enum __mnand_status {
    MNAND_OK = 0,            /* OK */
    MNAND_ENOMEM = 1,        /* memory allocation error */
    MNAND_EIO = 2,           /* I/O error */
    MNAND_EINVAL = 3,        /* invalid operation */
    MNAND_EOPEN = 4,         /* error opening device node */
    MNAND_UNSUPPORTED = 5,   /* not supported operation */
    MNAND_UNKNOWN_ERR = 6,   /* unknown error */
    MNAND_BKOPS_EN_ERR = 7   /* BKOPS en state not set to requested value */
} MNAND_STATUS;

/**
 * mNAND chip information
 */
struct __mnand_chip {
    int fd;                         /* file descriptor */
    char desc[80];                  /* chip desciption, mNAND specific */
    void *priv_ops;                 /* mNAND vendor specific priv operations */
    void *priv_chip_ops;            /* mNAND storage specific priv operations */
    int num_blocks;                 /* #blocks available, 0=unavailable */
    mnand_block_info *block_info;   /* block info list, indexd by num_blocks */
    int refresh_available;          /* force refresh is available */
    mnand_refresh_properties rfsh_properties;  /* refresh properties */
    int summary_available;          /* summary info is available */
    mnand_summary_info *summary;    /* summary info, not NULL when available */
    mnand_cid_info cid;             /* eMMC CID */
    mnand_csd_info csd;             /* eMMC CSD */
    mnand_ext_csd_info xcsd;        /* eMMC EXT CSD */
    mnand_device_desc_info device_desc; /* UFS Device Desccriptior */
    mnand_config_desc_info cfg_desc; /* UFS Configuration Descriptor */
    mnand_geo_desc_info geo_desc;    /* UFS Geometry Descriptor */
    mnand_unit_desc_info unit_desc; /* UFS Unit Descriptor */
    uint32_t card_status;           /* card status */
    MNAND_DEVICE_TYPE dev_type;     /* device type */
    MNAND_STORAGE_TYPE storage_type; /* storage type */
    void *priv;                     /* mNAND specific private "handle" */
};

typedef struct __mnand_chip mnand_chip;

/**
 * UFS Query Request Info (UFS Only)
 */
struct __mnand_ufs_query_req {
    uint8_t opcode;   /* Query operation opcode */
    uint8_t idn;      /* Query operation idn */
    uint8_t index;    /* index - optional in some cases */
    uint8_t selector; /* index - optional in some cases */
    uint8_t buf_size; /* buf_size - buffer size in bytes pointed by buffer */
    uint8_t *buffer;  /* user buffer pointer for query data
                       * should be 4 bytes for Attribute Read/Write
                       * should be 1 byte for Flag Read
                       * size varies for Read/Write Descriptor */
    uint32_t delay;   /* delay after query command completion */
    MNAND_STATUS status; /* return status of this query request */
};

/**
 * UFS Combo Query Request Info (UFS Only)
 */
struct __mnand_ufs_combo_query_req {
	uint8_t num_cmds;      /* Number of Query Commands in this Combo */
	uint8_t need_cq_empty; /* Command Queue need to be empty or not */
	struct __mnand_ufs_query_req *query; /* Query Request info start ptr */
};

typedef struct __mnand_ufs_query_req mnand_ufs_query_req;
typedef struct __mnand_ufs_combo_query_req mnand_ufs_combo_query_req;

/* inline helpers */

/**
 * Return the validity of the block, given the pointer of block info
 *
 * @param blk - pointer to block info
 *
 * @retval  non-zero, block info is valid
 * @retval  0, block info is invalid
 */
static inline int mnand_is_valid_block(mnand_block_info *blk)
{
    assert(blk != NULL);

    return (blk->valid != 0);
}

/**
 * Determine if given block is a bad block.
 *
 * @param blk - pointer to block info
 *
 * @retval  non-zero, block is bad block
 * @retval  0, block is not bad block
 */
static inline int mnand_is_bad_block(mnand_block_info *blk)
{
    assert(blk != NULL);

    return mnand_is_valid_block(blk) && ((blk->block_type &
        (MNAND_UNKNOWN_BAD_BLOCK | MNAND_MANF_BAD_BLOCK |
        MNAND_RUNTIME_BAD_BLOCK)) || (blk->block_age == MAX_ERASE_CNT)) ? 1 : 0;
}

/**
 * Determine if given block is a usable block. Note, in addition to bad block,
 * system block is not usable, neither.
 *
 * @param blk - pointer to block info
 *
 * @retval  non-zero, block is usable
 * @retval  0, block is not usable
 */
static inline int mnand_is_usable_block(mnand_block_info *blk)
{
    assert(blk != NULL);

    return (!mnand_is_valid_block(blk) || mnand_is_bad_block(blk) ||
        (blk->block_type & MNAND_SYSTEM_BLOCK)) ? 0 : 1;
}

/**
 * Determine if given block is a usable block, and matches given block type.
 *
 * @param blk      - pointer to block info
 * @param blk_type - block type to match
 *
 * @retval  non-zero, block is usable
 * @retval  0, block is not usable
 */
static inline int mnand_is_usable_block_with_type(mnand_block_info *blk,
    uint16_t blk_type)
{
    assert((blk != NULL) && (blk_type != 0));

    return (mnand_is_usable_block(blk) && (blk->block_type & blk_type)) ? 1 : 0;
}

/**
 * Determine if given block is in SLC region.
 *
 * @param blk - pointer to block info
 *
 * @retval  non-zero, block is in SLC region
 * @retval  0, block is not in SLC region
 */
static inline int mnand_is_slc_block(mnand_block_info *blk)
{
    assert(blk != NULL);

    return mnand_is_valid_block(blk) &&
        (blk->block_type & MNAND_SLC_BLOCK) ? 1 : 0;
}

/**
 * Determine if given block is in MLC region.
 *
 * @param blk - pointer to block info
 *
 * @retval  non-zero, block is in MLC region
 * @retval  0, block is not in MLC region
 */
static inline int mnand_is_mlc_block(mnand_block_info *blk)
{
    assert(blk != NULL);

    return mnand_is_valid_block(blk) &&
        (blk->block_type & MNAND_MLC_BLOCK) ? 1 : 0;
}

/**
 * Return if the summary is available
 *
 * @param chip - pointer to chip handle
 *
 * @retval  non-zero, summary is available
 * @retval  0, summary is not available
 */
static inline int mnand_is_summary_available(mnand_chip *chip)
{
    assert(chip != NULL);

    return (chip->summary_available != 0);
}

/**
 * Return if force refresh is available
 *
 * @param chip - pointer to chip handle
 *
 * @retval  non-zero, force refresh is available
 * @retval  0, force refresh is not available
 */
static inline int mnand_is_refresh_available(mnand_chip *chip)
{
    assert(chip != NULL);

    return (chip->refresh_available != 0);
}

/**
 * Return if the block info is available
 *
 * @param chip - pointer to chip handle
 *
 * @retval  non-zero, block info is available
 * @retval  0, block info is not available
 */
static inline int mnand_is_block_info_available(mnand_chip *chip)
{
    assert(chip != NULL);

    return (chip->num_blocks > 0);
}

/**
 * Return mnand storage type
 *
 * @param chip - pointer to chip handle
 *
 * @retval  storage type(enum) of the mnand device
 */
static inline MNAND_STORAGE_TYPE mnand_get_storage_type(mnand_chip *chip)
{
    assert(chip != NULL);

    return (chip->storage_type);
}

/* prototypes for wrapper functions */

/**
 * Return the status of BKOPS_EN bit
 *
 * @param chip - pointer to chip handle
 * @param enabled - store the BKOPS_EN bit
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_get_bkops_en(mnand_chip *chip, uint8_t *enabled);

/**
 * Return the status of BKOPS
 *
 * @param chip - pointer to chip handle
 * @param stat - store the status of BKOPS
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_get_bkops_status(mnand_chip *chip, uint8_t *stat);

/**
 * set the BKOPS_EN bit in ext_csd
 * Note: BNKOPS_EN is a OTP, once programmed can't change the value.
 *
 * @param chip - pointer to chip handle
 * @param enable - Value to set in BKOPS_EN bit
 *                Set 1 to BKOPS_EN, indicates host driver will issue bkops req
 *                and 0 indicate host driver will not issue bkops request.
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_set_bkops_en(mnand_chip *chip, uint8_t enable);

/**
 * Mark the start of BKOPS
 *
 * @param chip - pointer to chip handle
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_set_bkops_start(mnand_chip *chip);

/**
 * Enable power-off notification
 *
 * @param chip - pointer to chip handle
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_enable_power_off_notification(mnand_chip *chip);

/**
 * Send power-off notification
 *
 * @param chip - pointer to chip handle
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_send_power_off_notification(mnand_chip *chip);

/**
 * "Opening" the mNAND chip. This function needs to be called first to
 * establish mNAND health related content. Identification of mNAND will
 * be performed. Various health related data structures will be allocated,
 * initialized, and set up.
 *
 * mnand_open and mnand_close to be used in pair.
 *
 * @param devname - pointer to device name (e.g. /dev/emmc0)
 * @param chip - the mNAND chip handle to be initialized
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_open(const char *devname, mnand_chip *chip);

/**
 * "Closing" (aka releasing) the mNAND chip. This function needs to be
 * called last to free up health related resource and various data
 * structures.
 *
 * mnand_open and mnand_close to be used in pair.
 *
 * @param chip - the mNAND chip handle to be used
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_close(mnand_chip *chip);

/**
 * "Binding" the mNAND chip. This function needs to be called first to
 * establish mNAND health related content. Identification of mNAND will
 * be performed. Various health related data structures will be allocated,
 * initialized, and set up.
 *
 * mnand_bind and mnand_unbind to be used in pair.
 *
 * Note: this function uses existed file descriptor (unlike mnand_open,
 *       which takes a pathname for device node).
 *
 * @param fd - file descriptor to mNAND device
 * @param chip - the mNAND chip handle to be initialized
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_bind(int fd, mnand_chip *chip);

/**
 * "Unbinding" (aka releasing) the mNAND chip. This function needs to be
 * called last to free up health related resource and various data
 * structures.
 *
 * mnand_bind and mnand_unbind to be used in pair.
 *
 * Note: file descriptor used in "bind" will NOT be closed.
 *
 * @param chip - the mNAND chip handle to be used
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_unbind(mnand_chip *chip);

/**
 * Updating the CID field inside mNAND chip handle.
 *
 * @param chip - the mNAND chip handle to be updated
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_update_cid(mnand_chip *chip);

/**
 * Updating the CSD field inside mNAND chip handle.
 *
 * @param chip - the mNAND chip handle to be updated
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_update_csd(mnand_chip *chip);

/**
 * Updating the EXT CSD field inside mNAND chip handle.
 *
 * @param chip - the mNAND chip handle to be updated
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_update_xcsd(mnand_chip *chip);

/**
 * Updating the card status field inside mNAND chip handle.
 *
 * @param chip - the mNAND chip handle to be updated
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_update_card_status(mnand_chip *chip);

/**
 * Updating the list of block info inside mNAND chip handle.
 *
 * @param chip - the mNAND chip handle to be updated
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_update_block_info(mnand_chip *chip);

/**
 * Updating the summary info inside mNAND chip handle.
 *
 * @param chip - the mNAND chip handle to be updated
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_update_summary_info(mnand_chip *chip);

/**
 * Checking EOL status.
 *
 * @param chip - the mNAND chip handle to be updated
 * @param eol_status - EOL status (MNAND_EOL_DETECTED = EOL set)
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_check_eol_status(mnand_chip *chip,
    MNAND_EOL_STATUS *eol_status);

/**
 * Send "refresh" command to the mNAND chip. The purpose is to force data
 * rotation to improve the data retention.
 *
 * @param chip - the mNAND chip handle to be used
 * @param block_type - which region (SLC/MLC) to be used
 * @param num_blocks - #blocks to be refreshed
 * @prarm progress - refresh progress
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_send_refresh(mnand_chip *chip, uint8_t block_type,
    int num_blocks, uint32_t rfsh_unit_time_us,
    mnand_refresh_progress *progress);

/**
 * Extract age info from the mNAND chip.
 *
 * @param chip - the mNAND chip handle to be used
 * @param block_type - which region (SLC/MLC) to be used
 * @param total_age - the sum of the ages from the blocks
 * @prarm blk_count - #blocks in specified region
 * @param avg_age - average age of the blocks,
 *                  can be used when total_age/blk_counot not available
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_extract_age_info(mnand_chip *chip, int block_type,
    uint64_t *total_age, int *blk_count, double *avg_age);

/**
 * Extract refresh progress from the mNAND chip.
 *
 * @param chip - the mNAND chip handle to be used
 * @param rfsh_progress - current refresh progress (in percentage)
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_get_refresh_progress(mnand_chip *chip,
    double *rfsh_progress);

/**
 * Extract spare count info from the mNAND chip.
 *
 * @param chip - the mNAND chip handle to be used
 * @param life_time_info - information on life time
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_extract_life_time_info(mnand_chip *chip,
    mnand_life_time_info *life_time_info);

/**
 * Enable/Disable cache on mnand
 *
 * @param chip - the mNAND chip handle to be used
 * @param enable - Enables/Disable state for Read and Write cache
 *                 Bit 0 - Read Cache, Bit 1 - Write Cache
 *                 Bit value 1 - Cache enabled, Bit value 0 - Cache Disabled
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_set_cache_en(mnand_chip *chip, uint8_t enable);

/**
 * Flush cache on mnand
 *
 * @param chip - the mNAND chip handle to be used
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_flush_cache(mnand_chip *chip);

/**
 * Reading data sectors into given buffer (pointed by virtual address).
 *
 *
 * @param chip - the mNAND chip handle to be used
 * @param start_sector - the starting sector to be read
 * @param sector_cnt - the number of sectors to be read
 * @param virtaddr - buffer address (in virtual address form)
 *
 * @retval  number of sectors read
 * @retval  -1: problem with operation
 */
int mnand_read_sectors_virt(mnand_chip *chip, unsigned int start_sector,
    unsigned int sector_cnt, void *virtaddr);

/**
 * Reading data sectors into given buffer (pointed by physical address).
 *
 *
 * @param chip - the mNAND chip handle to be used
 * @param start_sector - the starting sector to be read
 * @param sector_cnt - the number of sectors to be read
 * @param addr - buffer address (in physical address form)
 *
 * @retval  number of sectors read
 * @retval  -1: problem with operation
 */
int mnand_read_sectors_phys(mnand_chip *chip, unsigned int start_sector,
    unsigned int sector_cnt, off_t physaddr);

/**
 * Trimming the whole device.
 *
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_trim_all(mnand_chip *chip);

/**
 * Determine if eMMC cache is enabled
 *
 * @param chip - pointer to chip handle
 * @param enabled - store the Read and Write cache enabled status
 *                  Bit 0 - Read Cache, Bit 1 - Write Cache
 *                  Bit value 1 - Cache enabled, Bit value 0 - Cache Disabled
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_get_cache_en(mnand_chip *chip, uint8_t *enabled);

/**
 * Send mnand switch cmd to mmc to set EXT CSD register.(eMMC only)
 *
 * @param chip - the mNAND chip handle to be used
 * @param index - which EXT CSD register to be set
 * @param value - the value of EXT CSD register to be set
 * @param errmsg_prepend - error msg to be displayed on failure
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_switch_cmd(mnand_chip *chip, uint8_t index,
    uint8_t value, char *errmsg_prepend);

/**
 * Extract mnand geometry info.
 *
 * @param chip - the mNAND chip handle to be used
 * @param geo_info - mnand geometry info
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_get_geo_info(mnand_chip *chip, mnand_geo_info *geo_info);

/**
 * get mnand lu info (UFS only)
 *
 * @param chip - the mNAND chip handle to be used
 * @param lun  - logical unit number for which lu info is needed
 * @param lu_info - pointer for the lu_info data to be updated
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_get_lu_info(mnand_chip *chip, uint32_t lun,
    mnand_lu_info *lu_info);

/**
 * get mnand current lun (UFS only)
 *
 * @param chip - the mNAND chip handle to be used
 * @param lun  -  Pointer to logical unit number to be updated
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_get_current_lun(mnand_chip *chip, uint32_t *lun);

/**
 * Send mnand combo query request (UFS only)
 *
 * @param chip - the mNAND chip handle to be used
 * @param query_req - Pointer to ufs query request info
 * @param num_cmds  - number of query requests in this combo request
 * @param need_cq_empty - Flag to specify if this request needs cq to be empty
 * @param ret_on_err - Flag to specify to return on error or continue furthur
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_send_combo_query_req(mnand_chip *chip,
    mnand_ufs_query_req *query_req, uint8_t num_cmds, uint8_t need_cq_empty,
    uint8_t ret_on_error);

/**
 * Test IOCTL MULTI CMD
 * Currently under this test:
 * Reads the XCSD status, toggles the cache status, reads the XCSD status again
 * toggles the cache status again, and updates the XCSD register.
 *
 * @param chip - the mNAND chip handle to be used
 *
 * @retval  MNAND_OK, success
 * @retval  else problem with operation
 */
MNAND_STATUS mnand_test_multi_cmd(mnand_chip *chip);

#if defined(__cplusplus)
}
#endif  /* __cplusplus */

#endif /* __NV_MNAND_H__ */
