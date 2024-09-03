/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <unistd.h>
#include "cdi_max96712_nv.h"
#include "cdi_max96712_priv_nv.h"

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
#include "cdi_max96712_pg_setting_nv.h"
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

#include "os_common.h"
#include "sipl_error.h"
#include "sipl_util.h"
#include "cdd_nv_error.h"

#define MAX_REGS_ARRAY_SIZE (uint32_t)50U

/**
 * @brief Returns a bit mask of link shifted by val
 * @param[in] val number of bit shifts
 * @param[in] link value to be shifted
 * @return bit mask
 */
static inline uint8_t MAX96712_LINK_SHIFT(uint8_t val, uint8_t link)
{
    _Static_assert(MAX96712_MAX_NUM_LINK == 4, "expecting bit shift by 0..3");
    return  ((val & (uint8_t) 0x1FU) << ((link) % ((uint8_t) MAX96712_MAX_NUM_LINK)));
}

/**
 * @brief Returns a one bit value of pipeline shifted by pipeline
 * @param[in] pipeline number of bit shifts
 * @return bit mask
 */
static inline uint8_t MAX96712_PIPELINE_SHIFT(uint8_t pipeline)
{
    _Static_assert(MAX96712_NUM_VIDEO_PIPELINES == 8, "expecting bit shift by 0..7");
    return (bit8((pipeline) % ((uint8_t) MAX96712_NUM_VIDEO_PIPELINES)));
}

/**
 * @brief Helper function for testing if specified GMSL link is set
 * - returns:
 *  - (((1U << ((linkNum) & 0xFU)) & (uint8_t)(((uint32_t)(linkVar)) & 0xFU)) != 0U)
 *  .
 * @param[in]    linkVar   link variable to test
 * @param[in]    linkNum   link number to test
 * @returns true of link is set, otherwise false
 */
static inline bool
MAX96712_IS_GMSL_LINK_SET(LinkMAX96712 const linkVar, uint8_t linkNum) {
    return (((1U << ((linkNum) & 0xFU)) & (uint8_t)(((uint32_t)(linkVar)) & 0xFU)) != 0U);
}

/**
 * @brief Help function for testing if specified GMSL mask is set
 * - returns
 *  - (((1U << ((linkNum) & 0xFU)) & (uint8_t)(((uint32_t)(linkMask)) & 0xFU)) != 0U)
 *  .
 * @param[in]    linkMask   link mask to test
 * @param[in]    linkNum    link number to test
 * @returns true of link is set, otherwise false
 */
static inline bool
MAX96712_IS_GMSL_LINKMASK_SET(uint8_t linkMask, uint8_t linkNum) {
    return (((1U << ((linkNum) & 0xFU)) & (uint8_t)(((uint32_t)(linkMask)) & 0xFU)) != 0U);
}

/**
 * @brief Help function for testing if multiple links are set
 * - returns
 *  - ((((uint32_t)link) & (((uint32_t)link) - 1U)) != 0U)
 *  .
 * @param[in]    link   link mask to test
 * @returns true of multiple links are set, otherwise false
 */
static inline bool
MAX96712_IS_MULTIPLE_GMSL_LINK_SET(LinkMAX96712 const link) {
    return ((((uint32_t)link) & (((uint32_t)link) - 1U)) != 0U);
}

#if !NV_IS_SAFETY
#define MAX96712_REPLICATION(SRC, DST)              ((((uint8_t)DST) << 5U) | (((uint8_t)SRC) << 3U))
#endif
/** Defined maximum number of error status codes
 * currently deser error buffer supports (up to) 24 global, plus (up to)
 * 16*8 pipeline, plus (up to) 16*4 link error status */
#define MAX96712_MAX_ERROR_STATUS_COUNT       255U

/** defines max number of PHY links */
#define MAX96712_MAX_NUM_PHY                  4U

/** defines deserializer oscilator frequency in MHz */
#define MAX96712_OSC_MHZ                      25U

/** defines device id for MAX96712 deserializer */
#define MAX96712_DEV_ID                       0xA0U

/** defines device id for MAX96722 deserializer */
#define MAX96722_DEV_ID                       0xA1U

/** defines number of bytes read/writen in one operation */
#define MAX96712_NUM_DATA_BYTES               1U

/** define size of deserializer register address */
#define MAX96712_NUM_ADDR_BYTES               2U

/** defines number of deserializer GPIOs */
#define MAX96712_MAX_PHYSICAL_MFP_NUM         16U

#if !NV_IS_SAFETY
#define REG_WRITE_BUFFER_BYTES                MAX96712_NUM_DATA_BYTES
#endif

#define REG5           0x005U
#define REG_GPIO1_A    0x303U
#define REG_GPIO1_B    0x304U
#define REG_GPIO3_A    0x309U
#define REG_GPIO3_B    0x30AU
#define REG_GPIO10_A   0x320U
#define REG_GPIO10_B   0x321U

#define REG_INTR2     0x25U
#define REG_INTR3     0x26U
#define REG_INTR8     0x2BU
#define REG_INTR9     0x2CU

#define REG_CNT0      0x35U
#define REG_CNT1      0x36U
#define REG_CNT2      0x37U
#define REG_CNT3      0x38U
#define REG_CNT4      0x39U
#define REG_CNT5      0x3AU
#define REG_CNT6      0x3BU
#define REG_CNT7      0x3CU

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
typedef enum {
    CDI_MAX96712_PG_PCLK_150MHX,
    CDI_MAX96712_PG_PCLK_375MHX,
    CDI_MAX96712_PG_PCLK_NUM,
} PGPCLKMAX96712;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/** enumerates type of I2C command modes */
typedef enum {
    REG_READ_MODE,
    REG_WRITE_MODE,
    REG_READ_MOD_WRITE_MODE,
} RegBitFieldAccessMode;

/** contains mapping from revision number to revision enumerator */
typedef struct {
    /** revision name */
    RevisionMAX96712 revId;

    /** revision enumerator */
    uint32_t revVal;
} MAX96712SupportedRevisions;

/** holds parameters of an I2C command accessing deserializer registers */
typedef struct {
    /** i2c register address */
    uint16_t regAddr;
    /** last meaningfull bit in data */
    uint8_t msbPos;
    /** first meaningfull bit in data */
    uint8_t lsbPos;
    /** expect long delay if true , defaults to false */
    bool longDelay;
} RegBitFieldProp;

/**
 * @brief Contains parameters of all I2C commands accessing deserializer registers
 * - initialized by
 *  - static const RegBitFieldProp regBitFieldProps[REG_FIELD_MAX] = {
 *   - [REG_FIELD_GMSL1_LOCK_A]        = {.regAddr = 0x0BCBU, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_GMSL1_LOCK_B]        = {.regAddr = 0x0CCBU, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_GMSL1_LOCK_C]        = {.regAddr = 0x0DCBU, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_GMSL1_LOCK_D]        = {.regAddr = 0x0ECBU, .msbPos = 0U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_GMSL1_DET_ERR_A]     = {.regAddr = 0x0B15U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_GMSL1_DET_ERR_B]     = {.regAddr = 0x0C15U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_GMSL1_DET_ERR_C]     = {.regAddr = 0x0D15U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_GMSL1_DET_ERR_D]     = {.regAddr = 0x0E15U, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_GMSL1_VIDEO_LOCK_A]  = {.regAddr = 0x0BCBU, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_GMSL1_VIDEO_LOCK_B]  = {.regAddr = 0x0CCBU, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_GMSL1_VIDEO_LOCK_C]  = {.regAddr = 0x0DCBU, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_GMSL1_VIDEO_LOCK_D]  = {.regAddr = 0x0ECBU, .msbPos = 6U, .lsbPos = 6U},
 *   -
 *   - [REG_FIELD_GMSL1_CONFIG_LOCK_A] = {.regAddr = 0x0BCBU, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_GMSL1_CONFIG_LOCK_B] = {.regAddr = 0x0CCBU, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_GMSL1_CONFIG_LOCK_C] = {.regAddr = 0x0DCBU, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_GMSL1_CONFIG_LOCK_D] = {.regAddr = 0x0ECBU, .msbPos = 7U, .lsbPos = 7U},
 *   -
 *   - [REG_FIELD_GMSL2_LOCK_A]        = {.regAddr = 0x001AU, .msbPos = 3U, .lsbPos = 3U},
 *   - [REG_FIELD_GMSL2_LOCK_B]        = {.regAddr = 0x000AU, .msbPos = 3U, .lsbPos = 3U},
 *   - [REG_FIELD_GMSL2_LOCK_C]        = {.regAddr = 0x000BU, .msbPos = 3U, .lsbPos = 3U},
 *   - [REG_FIELD_GMSL2_LOCK_D]        = {.regAddr = 0x000CU, .msbPos = 3U, .lsbPos = 3U},
 *   -
 *   - [REG_FIELD_GMSL2_DEC_ERR_A]     = {.regAddr = 0x0035U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_GMSL2_DEC_ERR_B]     = {.regAddr = 0x0036U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_GMSL2_DEC_ERR_C]     = {.regAddr = 0x0037U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_GMSL2_DEC_ERR_D]     = {.regAddr = 0x0038U, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_GMSL2_IDLE_ERR_A]         = {.regAddr = 0x0039U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_GMSL2_IDLE_ERR_B]         = {.regAddr = 0x003AU, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_GMSL2_IDLE_ERR_C]         = {.regAddr = 0x003BU, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_GMSL2_IDLE_ERR_D]         = {.regAddr = 0x003CU, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_VIDEO_LOCK_PIPE_0]        = {.regAddr = 0x01DCU, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_LOCK_PIPE_1]        = {.regAddr = 0x01FCU, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_LOCK_PIPE_2]        = {.regAddr = 0x021CU, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_LOCK_PIPE_3]        = {.regAddr = 0x023CU, .msbPos = 0U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_DIS_REM_CC_A]             = {.regAddr = 0x0003U, .msbPos = 1U, .lsbPos = 0U},
 *   - [REG_FIELD_DIS_REM_CC_B]             = {.regAddr = 0x0003U, .msbPos = 3U, .lsbPos = 2U},
 *   - [REG_FIELD_DIS_REM_CC_C]             = {.regAddr = 0x0003U, .msbPos = 5U, .lsbPos = 4U},
 *   - [REG_FIELD_DIS_REM_CC_D]             = {.regAddr = 0x0003U, .msbPos = 7U, .lsbPos = 6U},
 *   -
 *   - [REG_FIELD_SEC_XOVER_SEL_PHY_A]      = {.regAddr = 0x0007U, .msbPos = 4U, .lsbPos = 4U},
 *   - [REG_FIELD_SEC_XOVER_SEL_PHY_B]      = {.regAddr = 0x0007U, .msbPos = 5U, .lsbPos = 5U},
 *   - [REG_FIELD_SEC_XOVER_SEL_PHY_C]      = {.regAddr = 0x0007U, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_SEC_XOVER_SEL_PHY_D]      = {.regAddr = 0x0007U, .msbPos = 7U, .lsbPos = 7U},
 *   -
 *   - [REG_FIELD_LINK_EN_A]                = {.regAddr = 0x0006U, .msbPos = 0U, .lsbPos = 0U, .longDelay = true},
 *   - [REG_FIELD_LINK_EN_B]                = {.regAddr = 0x0006U, .msbPos = 1U, .lsbPos = 1U, .longDelay = true},
 *   - [REG_FIELD_LINK_EN_C]                = {.regAddr = 0x0006U, .msbPos = 2U, .lsbPos = 2U, .longDelay = true},
 *   - [REG_FIELD_LINK_EN_D]                = {.regAddr = 0x0006U, .msbPos = 3U, .lsbPos = 3U, .longDelay = true},
 *   -
 *   - [REG_FIELD_LINK_GMSL2_A]             = {.regAddr = 0x0006U, .msbPos = 4U, .lsbPos = 4U, .longDelay = true},
 *   - [REG_FIELD_LINK_GMSL2_B]             = {.regAddr = 0x0006U, .msbPos = 5U, .lsbPos = 5U, .longDelay = true},
 *   - [REG_FIELD_LINK_GMSL2_C]             = {.regAddr = 0x0006U, .msbPos = 6U, .lsbPos = 6U, .longDelay = true},
 *   - [REG_FIELD_LINK_GMSL2_D]             = {.regAddr = 0x0006U, .msbPos = 7U, .lsbPos = 7U, .longDelay = true},
 *   -
 *   - [REG_FIELD_RX_RATE_PHY_A]            = {.regAddr = 0x0010U, .msbPos = 1U, .lsbPos = 0U},
 *   - [REG_FIELD_RX_RATE_PHY_B]            = {.regAddr = 0x0010U, .msbPos = 5U, .lsbPos = 4U},
 *   - [REG_FIELD_RX_RATE_PHY_C]            = {.regAddr = 0x0011U, .msbPos = 1U, .lsbPos = 0U},
 *   - [REG_FIELD_RX_RATE_PHY_D]            = {.regAddr = 0x0011U, .msbPos = 5U, .lsbPos = 4U},
 *   -
 *   - [REG_FIELD_SOFT_BPP_0]               = {.regAddr = 0x040BU, .msbPos = 7U, .lsbPos = 3U},
 *   - [REG_FIELD_SOFT_BPP_1]               = {.regAddr = 0x0411U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_SOFT_BPP_2_L]             = {.regAddr = 0x0412U, .msbPos = 1U, .lsbPos = 0U},
 *   - [REG_FIELD_SOFT_BPP_3]               = {.regAddr = 0x0412U, .msbPos = 6U, .lsbPos = 2U},
 *   - [REG_FIELD_SOFT_BPP_2_H]             = {.regAddr = 0x0411U, .msbPos = 7U, .lsbPos = 5U},
 *   - [REG_FIELD_SOFT_BPP_4]               = {.regAddr = 0x042BU, .msbPos = 7U, .lsbPos = 3U},
 *   - [REG_FIELD_SOFT_BPP_5]               = {.regAddr = 0x0431U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_SOFT_BPP_6_H]             = {.regAddr = 0x0431U, .msbPos = 7U, .lsbPos = 5U},
 *   - [REG_FIELD_SOFT_BPP_6_L]             = {.regAddr = 0x0432U, .msbPos = 1U, .lsbPos = 0U},
 *   - [REG_FIELD_SOFT_BPP_7]               = {.regAddr = 0x0432U, .msbPos = 6U, .lsbPos = 2U},
 *   -
 *   - [REG_FIELD_SOFT_DT_0]                = {.regAddr = 0x040EU, .msbPos = 5U, .lsbPos = 0U},
 *   - [REG_FIELD_SOFT_DT_1_L]              = {.regAddr = 0x040FU, .msbPos = 3U, .lsbPos = 0U},
 *   - [REG_FIELD_SOFT_DT_2_L]              = {.regAddr = 0x0410U, .msbPos = 1U, .lsbPos = 0U},
 *   - [REG_FIELD_SOFT_DT_3]                = {.regAddr = 0x0410U, .msbPos = 7U, .lsbPos = 2U},
 *   - [REG_FIELD_SOFT_DT_1_H]              = {.regAddr = 0x040EU, .msbPos = 7U, .lsbPos = 6U},
 *   - [REG_FIELD_SOFT_DT_2_H]              = {.regAddr = 0x040FU, .msbPos = 7U, .lsbPos = 4U},
 *   - [REG_FIELD_SOFT_DT_4]                = {.regAddr = 0x042EU, .msbPos = 5U, .lsbPos = 0U},
 *   - [REG_FIELD_SOFT_DT_5_H]              = {.regAddr = 0x042EU, .msbPos = 7U, .lsbPos = 6U},
 *   - [REG_FIELD_SOFT_DT_5_L]              = {.regAddr = 0x042FU, .msbPos = 3U, .lsbPos = 0U},
 *   - [REG_FIELD_SOFT_DT_6_H]              = {.regAddr = 0x042FU, .msbPos = 7U, .lsbPos = 4U},
 *   - [REG_FIELD_SOFT_DT_6_L]              = {.regAddr = 0x0430U, .msbPos = 1U, .lsbPos = 0U},
 *   - [REG_FIELD_SOFT_DT_7]                = {.regAddr = 0x0430U, .msbPos = 7U, .lsbPos = 2U},
 *   -
 *   - [REG_FIELD_SOFT_OVR_0_EN]            = {.regAddr = 0x0415U, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_SOFT_OVR_1_EN]            = {.regAddr = 0x0415U, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_SOFT_OVR_2_EN]            = {.regAddr = 0x0418U, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_SOFT_OVR_3_EN]            = {.regAddr = 0x0418U, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_SOFT_OVR_4_EN]            = {.regAddr = 0x041BU, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_SOFT_OVR_5_EN]            = {.regAddr = 0x041BU, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_SOFT_OVR_6_EN]            = {.regAddr = 0x041DU, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_SOFT_OVR_7_EN]            = {.regAddr = 0x041DU, .msbPos = 7U, .lsbPos = 7U},
 *   -
 *   - [REG_FIELD_CC_CRC_ERRCNT_A]          = {.regAddr = 0x0B19U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_CC_CRC_ERRCNT_B]          = {.regAddr = 0x0C19U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_CC_CRC_ERRCNT_C]          = {.regAddr = 0x0D19U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_CC_CRC_ERRCNT_D]          = {.regAddr = 0x0E19U, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_I2C_FWDCCEN_PHY_A]        = {.regAddr = 0x0B04U, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_I2C_FWDCCEN_PHY_B]        = {.regAddr = 0x0C04U, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_I2C_FWDCCEN_PHY_C]        = {.regAddr = 0x0D04U, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_I2C_FWDCCEN_PHY_D]        = {.regAddr = 0x0E04U, .msbPos = 0U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_I2C_REVCCEN_PHY_A]        = {.regAddr = 0x0B04U, .msbPos = 1U, .lsbPos = 1U},
 *   - [REG_FIELD_I2C_REVCCEN_PHY_B]        = {.regAddr = 0x0C04U, .msbPos = 1U, .lsbPos = 1U},
 *   - [REG_FIELD_I2C_REVCCEN_PHY_C]        = {.regAddr = 0x0D04U, .msbPos = 1U, .lsbPos = 1U},
 *   - [REG_FIELD_I2C_REVCCEN_PHY_D]        = {.regAddr = 0x0E04U, .msbPos = 1U, .lsbPos = 1U},
 *   -
 *   - [REG_FIELD_I2C_PORT_GMSL1_PHY_A]     = {.regAddr = 0x0B04U, .msbPos = 3U, .lsbPos = 3U},
 *   - [REG_FIELD_I2C_PORT_GMSL1_PHY_B]     = {.regAddr = 0x0C04U, .msbPos = 3U, .lsbPos = 3U},
 *   - [REG_FIELD_I2C_PORT_GMSL1_PHY_C]     = {.regAddr = 0x0D04U, .msbPos = 3U, .lsbPos = 3U},
 *   - [REG_FIELD_I2C_PORT_GMSL1_PHY_D]     = {.regAddr = 0x0E04U, .msbPos = 3U, .lsbPos = 3U},
 *   -
 *   - [REG_FIELD_DE_EN_PHY_A]              = {.regAddr = 0x0B0FU, .msbPos = 3U, .lsbPos = 3U},
 *   - [REG_FIELD_DE_EN_PHY_B]              = {.regAddr = 0x0C0FU, .msbPos = 3U, .lsbPos = 3U},
 *   - [REG_FIELD_DE_EN_PHY_C]              = {.regAddr = 0x0D0FU, .msbPos = 3U, .lsbPos = 3U},
 *   - [REG_FIELD_DE_EN_PHY_D]              = {.regAddr = 0x0E0FU, .msbPos = 3U, .lsbPos = 3U},
 *   -
 *   - [REG_FIELD_DE_PRBS_TYPE_PHY_A]       = {.regAddr = 0x0B0FU, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_DE_PRBS_TYPE_PHY_B]       = {.regAddr = 0x0C0FU, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_DE_PRBS_TYPE_PHY_C]       = {.regAddr = 0x0D0FU, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_DE_PRBS_TYPE_PHY_D]       = {.regAddr = 0x0E0FU, .msbPos = 0U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_PKTCCEN_LINK_A]           = {.regAddr = 0x0B08U, .msbPos = 2U, .lsbPos = 2U},
 *   - [REG_FIELD_PKTCCEN_LINK_B]           = {.regAddr = 0x0C08U, .msbPos = 2U, .lsbPos = 2U},
 *   - [REG_FIELD_PKTCCEN_LINK_C]           = {.regAddr = 0x0D08U, .msbPos = 2U, .lsbPos = 2U},
 *   - [REG_FIELD_PKTCCEN_LINK_D]           = {.regAddr = 0x0E08U, .msbPos = 2U, .lsbPos = 2U},
 *   -
 *   - [REG_FIELD_ALT_MEM_MAP12_PHY0]       = {.regAddr = 0x0933U, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_ALT_MEM_MAP12_PHY1]       = {.regAddr = 0x0973U, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_ALT_MEM_MAP12_PHY2]       = {.regAddr = 0x09B3U, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_ALT_MEM_MAP12_PHY3]       = {.regAddr = 0x09F3U, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_ALT_MEM_MAP8_PHY0]        = {.regAddr = 0x0933U, .msbPos = 1U, .lsbPos = 1U},
 *   - [REG_FIELD_ALT_MEM_MAP8_PHY1]        = {.regAddr = 0x0973U, .msbPos = 1U, .lsbPos = 1U},
 *   - [REG_FIELD_ALT_MEM_MAP8_PHY2]        = {.regAddr = 0x09B3U, .msbPos = 1U, .lsbPos = 1U},
 *   - [REG_FIELD_ALT_MEM_MAP8_PHY3]        = {.regAddr = 0x09F3U, .msbPos = 1U, .lsbPos = 1U},
 *   - [REG_FIELD_ALT_MEM_MAP10_PHY0]       = {.regAddr = 0x0933U, .msbPos = 2U, .lsbPos = 2U},
 *   - [REG_FIELD_ALT_MEM_MAP10_PHY1]       = {.regAddr = 0x0973U, .msbPos = 2U, .lsbPos = 2U},
 *   - [REG_FIELD_ALT_MEM_MAP10_PHY2]       = {.regAddr = 0x09B3U, .msbPos = 2U, .lsbPos = 2U},
 *   - [REG_FIELD_ALT_MEM_MAP10_PHY3]       = {.regAddr = 0x09F3U, .msbPos = 2U, .lsbPos = 2U},
 *   - [REG_FIELD_ALT2_MEM_MAP8_PHY0]       = {.regAddr = 0x0933U, .msbPos = 4U, .lsbPos = 4U},
 *   - [REG_FIELD_ALT2_MEM_MAP8_PHY1]       = {.regAddr = 0x0973U, .msbPos = 4U, .lsbPos = 4U},
 *   - [REG_FIELD_ALT2_MEM_MAP8_PHY2]       = {.regAddr = 0x09B3U, .msbPos = 4U, .lsbPos = 4U},
 *   - [REG_FIELD_ALT2_MEM_MAP8_PHY3]       = {.regAddr = 0x09F3U, .msbPos = 4U, .lsbPos = 4U},
 *   -
 *   - [REG_FIELD_BPP8DBL_0]                = {.regAddr = 0x0414U, .msbPos = 4U, .lsbPos = 4U},
 *   - [REG_FIELD_BPP8DBL_1]                = {.regAddr = 0x0414U, .msbPos = 5U, .lsbPos = 5U},
 *   - [REG_FIELD_BPP8DBL_2]                = {.regAddr = 0x0414U, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_BPP8DBL_3]                = {.regAddr = 0x0414U, .msbPos = 7U, .lsbPos = 7U},
 *   -
 *   - [REG_FIELD_BPP8DBL_4]                = {.regAddr = 0x0434U, .msbPos = 4U, .lsbPos = 4U},
 *   - [REG_FIELD_BPP8DBL_5]                = {.regAddr = 0x0434U, .msbPos = 5U, .lsbPos = 5U},
 *   - [REG_FIELD_BPP8DBL_6]                = {.regAddr = 0x0434U, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_BPP8DBL_7]                = {.regAddr = 0x0434U, .msbPos = 7U, .lsbPos = 7U},
 *   -
 *   - [REG_FIELD_BPP8DBL_MODE_0]           = {.regAddr = 0x0417U, .msbPos = 4U, .lsbPos = 4U},
 *   - [REG_FIELD_BPP8DBL_MODE_1]           = {.regAddr = 0x0417U, .msbPos = 5U, .lsbPos = 5U},
 *   - [REG_FIELD_BPP8DBL_MODE_2]           = {.regAddr = 0x0417U, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_BPP8DBL_MODE_3]           = {.regAddr = 0x0417U, .msbPos = 7U, .lsbPos = 7U},
 *   -
 *   - [REG_FIELD_BPP8DBL_MODE_4]           = {.regAddr = 0x0437U, .msbPos = 4U, .lsbPos = 4U},
 *   - [REG_FIELD_BPP8DBL_MODE_5]           = {.regAddr = 0x0437U, .msbPos = 5U, .lsbPos = 5U},
 *   - [REG_FIELD_BPP8DBL_MODE_6]           = {.regAddr = 0x0437U, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_BPP8DBL_MODE_7]           = {.regAddr = 0x0437U, .msbPos = 7U, .lsbPos = 7U},
 *   -
 *   - [REG_FIELD_AEQ_PHY_A]                = {.regAddr = 0x0B14U, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_AEQ_PHY_B]                = {.regAddr = 0x0C14U, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_AEQ_PHY_C]                = {.regAddr = 0x0D14U, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_AEQ_PHY_D]                = {.regAddr = 0x0E14U, .msbPos = 7U, .lsbPos = 7U},
 *   -
 *   - [REG_FIELD_PERIODIC_AEQ_PHY_A]       = {.regAddr = 0x0B14U, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_PERIODIC_AEQ_PHY_B]       = {.regAddr = 0x0C14U, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_PERIODIC_AEQ_PHY_C]       = {.regAddr = 0x0D14U, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_PERIODIC_AEQ_PHY_D]       = {.regAddr = 0x0E14U, .msbPos = 6U, .lsbPos = 6U},
 *   -
 *   - [REG_FIELD_EOM_PER_THR_PHY_A]        = {.regAddr = 0x0B14U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_EOM_PER_THR_PHY_B]        = {.regAddr = 0x0C14U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_EOM_PER_THR_PHY_C]        = {.regAddr = 0x0D14U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_EOM_PER_THR_PHY_D]        = {.regAddr = 0x0E14U, .msbPos = 4U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_VIDEO_PIPE_SEL_0]         = {.regAddr = 0x00F0U, .msbPos = 3U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_PIPE_SEL_1]         = {.regAddr = 0x00F0U, .msbPos = 7U, .lsbPos = 4U},
 *   - [REG_FIELD_VIDEO_PIPE_SEL_2]         = {.regAddr = 0x00F1U, .msbPos = 3U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_PIPE_SEL_3]         = {.regAddr = 0x00F1U, .msbPos = 7U, .lsbPos = 4U},
 *   - [REG_FIELD_VIDEO_PIPE_SEL_4]         = {.regAddr = 0x00F2U, .msbPos = 3U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_PIPE_SEL_5]         = {.regAddr = 0x00F2U, .msbPos = 7U, .lsbPos = 4U},
 *   - [REG_FIELD_VIDEO_PIPE_SEL_6]         = {.regAddr = 0x00F3U, .msbPos = 3U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_PIPE_SEL_7]         = {.regAddr = 0x00F3U, .msbPos = 7U, .lsbPos = 4U},
 *   -
 *   - [REG_FIELD_VIDEO_PIPE_EN_0]          = {.regAddr = 0x00F4U, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_PIPE_EN_1]          = {.regAddr = 0x00F4U, .msbPos = 1U, .lsbPos = 1U},
 *   - [REG_FIELD_VIDEO_PIPE_EN_2]          = {.regAddr = 0x00F4U, .msbPos = 2U, .lsbPos = 2U},
 *   - [REG_FIELD_VIDEO_PIPE_EN_3]          = {.regAddr = 0x00F4U, .msbPos = 3U, .lsbPos = 3U},
 *   - [REG_FIELD_VIDEO_PIPE_EN_4]          = {.regAddr = 0x00F4U, .msbPos = 4U, .lsbPos = 4U},
 *   - [REG_FIELD_VIDEO_PIPE_EN_5]          = {.regAddr = 0x00F4U, .msbPos = 5U, .lsbPos = 5U},
 *   - [REG_FIELD_VIDEO_PIPE_EN_6]          = {.regAddr = 0x00F4U, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_VIDEO_PIPE_EN_7]          = {.regAddr = 0x00F4U, .msbPos = 7U, .lsbPos = 7U},
 *   -
 *   - [REG_FIELD_PATGEN_CLK_SRC_PIPE_0]    = {.regAddr = 0x01DCU, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_PATGEN_CLK_SRC_PIPE_1]    = {.regAddr = 0x01FCU, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_PATGEN_CLK_SRC_PIPE_2]    = {.regAddr = 0x021CU, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_PATGEN_CLK_SRC_PIPE_3]    = {.regAddr = 0x023CU, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_PATGEN_CLK_SRC_PIPE_4]    = {.regAddr = 0x025CU, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_PATGEN_CLK_SRC_PIPE_5]    = {.regAddr = 0x027CU, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_PATGEN_CLK_SRC_PIPE_6]    = {.regAddr = 0x029CU, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_PATGEN_CLK_SRC_PIPE_7]    = {.regAddr = 0x02BCU, .msbPos = 7U, .lsbPos = 7U},
 *   -
 *   - [REG_FIELD_MIPI_OUT_CFG]             = {.regAddr = 0x08A0U, .msbPos = 5U, .lsbPos = 0U},
 *   - [REG_FIELD_T_T3_PREBEGIN]            = {.regAddr = 0x08ADU, .msbPos = 5U, .lsbPos = 0U},
 *   - [REG_FIELD_T_T3_POST_PREP]           = {.regAddr = 0x08AEU, .msbPos = 6U, .lsbPos = 0U},
 *   - [REG_FIELD_DEV_REV]                  = {.regAddr = 0x004CU, .msbPos = 3U, .lsbPos = 0U},
 *   - [REG_FIELD_DEV_ID]                   = {.regAddr = 0x000DU, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_RESET_ONESHOT]            = {.regAddr = 0x0018U, .msbPos = 3U, .lsbPos = 0U},
 *   - [REG_FIELD_BACKTOP_EN]               = {.regAddr = 0x0400U, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_PHY_STANDBY]              = {.regAddr = 0x08A2U, .msbPos = 7U, .lsbPos = 4U},
 *   - [REG_FIELD_FORCE_CSI_OUT_EN]         = {.regAddr = 0x08A0U, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_ERRB]                     = {.regAddr = 0x001AU, .msbPos = 2U, .lsbPos = 2U},
 *   - [REG_FIELD_OVERFLOW_FIRST4]          = {.regAddr = 0x040AU, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_OVERFLOW_LAST4]           = {.regAddr = 0x042AU, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_OSN_COEFFICIENT_0]        = {.regAddr = 0x142EU, .msbPos = 5U, .lsbPos = 0U},
 *   - [REG_FIELD_OSN_COEFFICIENT_1]        = {.regAddr = 0x152EU, .msbPos = 5U, .lsbPos = 0U},
 *   - [REG_FIELD_OSN_COEFFICIENT_2]        = {.regAddr = 0x162EU, .msbPos = 5U, .lsbPos = 0U},
 *   - [REG_FIELD_OSN_COEFFICIENT_3]        = {.regAddr = 0x172EU, .msbPos = 5U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_ENABLE_OSN_0]             = {.regAddr = 0x1432U, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_ENABLE_OSN_1]             = {.regAddr = 0x1532U, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_ENABLE_OSN_2]             = {.regAddr = 0x1632U, .msbPos = 6U, .lsbPos = 6U},
 *   - [REG_FIELD_ENABLE_OSN_3]             = {.regAddr = 0x1732U, .msbPos = 6U, .lsbPos = 6U},
 *   -
 *   - [REG_FIELD_OSN_COEFF_MANUAL_SEED_0]  = {.regAddr = 0x1457U, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_OSN_COEFF_MANUAL_SEED_1]  = {.regAddr = 0x1557U, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_OSN_COEFF_MANUAL_SEED_2]  = {.regAddr = 0x1657U, .msbPos = 0U, .lsbPos = 0U},
 *   - [REG_FIELD_OSN_COEFF_MANUAL_SEED_3]  = {.regAddr = 0x1757U, .msbPos = 0U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_SET_TX_AMP_0]             = {.regAddr = 0x1495U, .msbPos = 5U, .lsbPos = 0U},
 *   - [REG_FIELD_SET_TX_AMP_1]             = {.regAddr = 0x1595U, .msbPos = 5U, .lsbPos = 0U},
 *   - [REG_FIELD_SET_TX_AMP_2]             = {.regAddr = 0x1695U, .msbPos = 5U, .lsbPos = 0U},
 *   - [REG_FIELD_SET_TX_AMP_3]             = {.regAddr = 0x1795U, .msbPos = 5U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_SLV_TO_P0_A]              = {.regAddr = 0x0640U, .msbPos = 2U, .lsbPos = 0U},
 *   - [REG_FIELD_SLV_TO_P0_B]              = {.regAddr = 0x0650U, .msbPos = 2U, .lsbPos = 0U},
 *   - [REG_FIELD_SLV_TO_P0_C]              = {.regAddr = 0x0660U, .msbPos = 2U, .lsbPos = 0U},
 *   - [REG_FIELD_SLV_TO_P0_D]              = {.regAddr = 0x0670U, .msbPos = 2U, .lsbPos = 0U},
 *   - [REG_FIELD_SLV_TO_P1_A]              = {.regAddr = 0x0680U, .msbPos = 2U, .lsbPos = 0U},
 *   - [REG_FIELD_SLV_TO_P1_B]              = {.regAddr = 0x0690U, .msbPos = 2U, .lsbPos = 0U},
 *   - [REG_FIELD_SLV_TO_P1_C]              = {.regAddr = 0x06A0U, .msbPos = 2U, .lsbPos = 0U},
 *   - [REG_FIELD_SLV_TO_P1_D]              = {.regAddr = 0x06B0U, .msbPos = 2U, .lsbPos = 0U},
 *   - [REG_FIELD_SLV_TO_P2_A]              = {.regAddr = 0x0688U, .msbPos = 2U, .lsbPos = 0U},
 *   - [REG_FIELD_SLV_TO_P2_B]              = {.regAddr = 0x0698U, .msbPos = 2U, .lsbPos = 0U},
 *   - [REG_FIELD_SLV_TO_P2_C]              = {.regAddr = 0x06A8U, .msbPos = 2U, .lsbPos = 0U},
 *   - [REG_FIELD_SLV_TO_P2_D]              = {.regAddr = 0x06B8U, .msbPos = 2U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_REM_ERR_FLAG]             = {.regAddr = 0x002AU, .msbPos = 1U, .lsbPos = 1U},
 *   -
 *   - [REG_FIELD_CTRL3]                    = {.regAddr = 0x001AU, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_INTR4]                    = {.regAddr = 0x0027U, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_INTR5]                    = {.regAddr = 0x0028U, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_INTR6]                    = {.regAddr = 0x0029U, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_INTR7]                    = {.regAddr = 0x002AU, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_INTR8]                    = {.regAddr = 0x002BU, .msbPos = 3U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_INTR9]                    = {.regAddr = 0x002CU, .msbPos = 3U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_INTR10]                   = {.regAddr = 0x002DU, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_INTR11]                   = {.regAddr = 0x002EU, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_INTR12_MEM_ERR_OEN]       = {.regAddr = 0x002FU, .msbPos = 6U, .lsbPos = 6U},
 *   -
 *   - [REG_FIELD_VID_PXL_CRC_ERR_INT]      = {.regAddr = 0x0045U, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_FSYNC_22]                 = {.regAddr = 0x04B6U, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_BACKTOP25]                = {.regAddr = 0x0438U, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_VIDEO_RX8_PIPE_0]         = {.regAddr = 0x0108U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_RX8_PIPE_1]         = {.regAddr = 0x011AU, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_RX8_PIPE_2]         = {.regAddr = 0x012CU, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_RX8_PIPE_3]         = {.regAddr = 0x013EU, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_RX8_PIPE_4]         = {.regAddr = 0x0150U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_RX8_PIPE_5]         = {.regAddr = 0x0168U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_RX8_PIPE_6]         = {.regAddr = 0x017AU, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_RX8_PIPE_7]         = {.regAddr = 0x018CU, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_VIDEO_MASKED_OEN]         = {.regAddr = 0x0049U, .msbPos = 5U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_VIDEO_MASKED_FLAG]        = {.regAddr = 0x004AU, .msbPos = 5U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_PWR0]                     = {.regAddr = 0x0012U, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_PWR_STATUS_FLAG]          = {.regAddr = 0x0047U, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_ENABLE_LOCK]              = {.regAddr = 0x0005U, .msbPos = 7U, .lsbPos = 7U},
 *   - [REG_FIELD_ENABLE_ERRB]              = {.regAddr = 0x0005U, .msbPos = 6U, .lsbPos = 6U},
 *   -
 *   - [REG_FIELD_CRUSSC_MODE_0]            = {.regAddr = 0x1445U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_CRUSSC_MODE_1]            = {.regAddr = 0x1545U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_CRUSSC_MODE_2]            = {.regAddr = 0x1645U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_CRUSSC_MODE_3]            = {.regAddr = 0x1745U, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_VIDEO_RX0_PIPE_0]         = {.regAddr = 0x0100U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_RX0_PIPE_1]         = {.regAddr = 0x0112U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_RX0_PIPE_2]         = {.regAddr = 0x0124U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_RX0_PIPE_3]         = {.regAddr = 0x0136U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_RX0_PIPE_4]         = {.regAddr = 0x0148U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_RX0_PIPE_5]         = {.regAddr = 0x0160U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_RX0_PIPE_6]         = {.regAddr = 0x0172U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_VIDEO_RX0_PIPE_7]         = {.regAddr = 0x0184U, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_CFGH_VIDEO_CRC0]          = {.regAddr = 0x0060U, .msbPos = 7U, .lsbPos = 0U},
 *   - [REG_FIELD_CFGH_VIDEO_CRC1]          = {.regAddr = 0x0061U, .msbPos = 7U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_GPIO_00_A_A]              = {.regAddr = 0x0300U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_01_A_A]              = {.regAddr = 0x0303U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_02_A_A]              = {.regAddr = 0x0306U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_03_A_A]              = {.regAddr = 0x0309U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_04_A_A]              = {.regAddr = 0x030CU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_05_A_A]              = {.regAddr = 0x0310U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_06_A_A]              = {.regAddr = 0x0313U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_07_A_A]              = {.regAddr = 0x0316U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_08_A_A]              = {.regAddr = 0x0319U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_09_A_A]              = {.regAddr = 0x031CU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_10_A_A]              = {.regAddr = 0x0320U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_11_A_A]              = {.regAddr = 0x0323U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_12_A_A]              = {.regAddr = 0x0326U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_13_A_A]              = {.regAddr = 0x0329U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_14_A_A]              = {.regAddr = 0x032CU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_15_A_A]              = {.regAddr = 0x0330U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_16_A_A]              = {.regAddr = 0x0333U, .msbPos = 4U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_GPIO_00_A_B]              = {.regAddr = 0x0301U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_01_A_B]              = {.regAddr = 0x0304U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_02_A_B]              = {.regAddr = 0x0307U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_03_A_B]              = {.regAddr = 0x030AU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_04_A_B]              = {.regAddr = 0x030DU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_05_A_B]              = {.regAddr = 0x0311U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_06_A_B]              = {.regAddr = 0x0314U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_07_A_B]              = {.regAddr = 0x0317U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_08_A_B]              = {.regAddr = 0x031AU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_09_A_B]              = {.regAddr = 0x031DU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_10_A_B]              = {.regAddr = 0x0321U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_11_A_B]              = {.regAddr = 0x0324U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_12_A_B]              = {.regAddr = 0x0327U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_13_A_B]              = {.regAddr = 0x032AU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_14_A_B]              = {.regAddr = 0x032DU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_15_A_B]              = {.regAddr = 0x0331U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_16_A_B]              = {.regAddr = 0x0334U, .msbPos = 4U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_GPIO_00_B_B]              = {.regAddr = 0x0337U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_01_B_B]              = {.regAddr = 0x033AU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_02_B_B]              = {.regAddr = 0x033DU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_03_B_B]              = {.regAddr = 0x0341U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_04_B_B]              = {.regAddr = 0x0344U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_05_B_B]              = {.regAddr = 0x0347U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_06_B_B]              = {.regAddr = 0x034AU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_07_B_B]              = {.regAddr = 0x034DU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_08_B_B]              = {.regAddr = 0x0351U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_09_B_B]              = {.regAddr = 0x0354U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_10_B_B]              = {.regAddr = 0x0357U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_11_B_B]              = {.regAddr = 0x035AU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_12_B_B]              = {.regAddr = 0x035DU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_13_B_B]              = {.regAddr = 0x0361U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_14_B_B]              = {.regAddr = 0x0364U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_15_B_B]              = {.regAddr = 0x0367U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_16_B_B]              = {.regAddr = 0x036AU, .msbPos = 4U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_GPIO_00_C_B]              = {.regAddr = 0x036DU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_01_C_B]              = {.regAddr = 0x0371U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_02_C_B]              = {.regAddr = 0x0374U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_03_C_B]              = {.regAddr = 0x0377U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_04_C_B]              = {.regAddr = 0x037AU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_05_C_B]              = {.regAddr = 0x037DU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_06_C_B]              = {.regAddr = 0x0381U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_07_C_B]              = {.regAddr = 0x0384U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_08_C_B]              = {.regAddr = 0x0387U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_09_C_B]              = {.regAddr = 0x038AU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_10_C_B]              = {.regAddr = 0x038DU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_11_C_B]              = {.regAddr = 0x0391U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_12_C_B]              = {.regAddr = 0x0394U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_13_C_B]              = {.regAddr = 0x0397U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_14_C_B]              = {.regAddr = 0x039AU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_15_C_B]              = {.regAddr = 0x039DU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_16_C_B]              = {.regAddr = 0x03A1U, .msbPos = 4U, .lsbPos = 0U},
 *   -
 *   - [REG_FIELD_GPIO_00_D_B]              = {.regAddr = 0x03A4U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_01_D_B]              = {.regAddr = 0x03A7U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_02_D_B]              = {.regAddr = 0x03AAU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_03_D_B]              = {.regAddr = 0x03ADU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_04_D_B]              = {.regAddr = 0x03B1U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_05_D_B]              = {.regAddr = 0x03B4U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_06_D_B]              = {.regAddr = 0x03B7U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_07_D_B]              = {.regAddr = 0x03BAU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_08_D_B]              = {.regAddr = 0x03BDU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_09_D_B]              = {.regAddr = 0x03C1U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_10_D_B]              = {.regAddr = 0x03C4U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_11_D_B]              = {.regAddr = 0x03C7U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_12_D_B]              = {.regAddr = 0x03CAU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_13_D_B]              = {.regAddr = 0x03CDU, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_14_D_B]              = {.regAddr = 0x03D1U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_15_D_B]              = {.regAddr = 0x03D4U, .msbPos = 4U, .lsbPos = 0U},
 *   - [REG_FIELD_GPIO_16_D_B]              = {.regAddr = 0x03D7U, .msbPos = 4U, .lsbPos = 0U},
 *   .
 *  - }
 *  .
 */
static const RegBitFieldProp regBitFieldProps[REG_FIELD_MAX] = {
    [REG_FIELD_GMSL1_LOCK_A]        = {.regAddr = 0x0BCBU, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_GMSL1_LOCK_B]        = {.regAddr = 0x0CCBU, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_GMSL1_LOCK_C]        = {.regAddr = 0x0DCBU, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_GMSL1_LOCK_D]        = {.regAddr = 0x0ECBU, .msbPos = 0U, .lsbPos = 0U},

    [REG_FIELD_GMSL1_DET_ERR_A]     = {.regAddr = 0x0B15U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_GMSL1_DET_ERR_B]     = {.regAddr = 0x0C15U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_GMSL1_DET_ERR_C]     = {.regAddr = 0x0D15U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_GMSL1_DET_ERR_D]     = {.regAddr = 0x0E15U, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_GMSL1_VIDEO_LOCK_A]  = {.regAddr = 0x0BCBU, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_GMSL1_VIDEO_LOCK_B]  = {.regAddr = 0x0CCBU, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_GMSL1_VIDEO_LOCK_C]  = {.regAddr = 0x0DCBU, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_GMSL1_VIDEO_LOCK_D]  = {.regAddr = 0x0ECBU, .msbPos = 6U, .lsbPos = 6U},

    [REG_FIELD_GMSL1_CONFIG_LOCK_A] = {.regAddr = 0x0BCBU, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_GMSL1_CONFIG_LOCK_B] = {.regAddr = 0x0CCBU, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_GMSL1_CONFIG_LOCK_C] = {.regAddr = 0x0DCBU, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_GMSL1_CONFIG_LOCK_D] = {.regAddr = 0x0ECBU, .msbPos = 7U, .lsbPos = 7U},

    [REG_FIELD_GMSL2_LOCK_A]        = {.regAddr = 0x001AU, .msbPos = 3U, .lsbPos = 3U},
    [REG_FIELD_GMSL2_LOCK_B]        = {.regAddr = 0x000AU, .msbPos = 3U, .lsbPos = 3U},
    [REG_FIELD_GMSL2_LOCK_C]        = {.regAddr = 0x000BU, .msbPos = 3U, .lsbPos = 3U},
    [REG_FIELD_GMSL2_LOCK_D]        = {.regAddr = 0x000CU, .msbPos = 3U, .lsbPos = 3U},

    [REG_FIELD_GMSL2_DEC_ERR_A]     = {.regAddr = 0x0035U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_GMSL2_DEC_ERR_B]     = {.regAddr = 0x0036U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_GMSL2_DEC_ERR_C]     = {.regAddr = 0x0037U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_GMSL2_DEC_ERR_D]     = {.regAddr = 0x0038U, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_GMSL2_IDLE_ERR_A]         = {.regAddr = 0x0039U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_GMSL2_IDLE_ERR_B]         = {.regAddr = 0x003AU, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_GMSL2_IDLE_ERR_C]         = {.regAddr = 0x003BU, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_GMSL2_IDLE_ERR_D]         = {.regAddr = 0x003CU, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_VIDEO_LOCK_PIPE_0]        = {.regAddr = 0x01DCU, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_LOCK_PIPE_1]        = {.regAddr = 0x01FCU, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_LOCK_PIPE_2]        = {.regAddr = 0x021CU, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_LOCK_PIPE_3]        = {.regAddr = 0x023CU, .msbPos = 0U, .lsbPos = 0U},

    [REG_FIELD_DIS_REM_CC_A]             = {.regAddr = 0x0003U, .msbPos = 1U, .lsbPos = 0U},
    [REG_FIELD_DIS_REM_CC_B]             = {.regAddr = 0x0003U, .msbPos = 3U, .lsbPos = 2U},
    [REG_FIELD_DIS_REM_CC_C]             = {.regAddr = 0x0003U, .msbPos = 5U, .lsbPos = 4U},
    [REG_FIELD_DIS_REM_CC_D]             = {.regAddr = 0x0003U, .msbPos = 7U, .lsbPos = 6U},

    [REG_FIELD_SEC_XOVER_SEL_PHY_A]      = {.regAddr = 0x0007U, .msbPos = 4U, .lsbPos = 4U},
    [REG_FIELD_SEC_XOVER_SEL_PHY_B]      = {.regAddr = 0x0007U, .msbPos = 5U, .lsbPos = 5U},
    [REG_FIELD_SEC_XOVER_SEL_PHY_C]      = {.regAddr = 0x0007U, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_SEC_XOVER_SEL_PHY_D]      = {.regAddr = 0x0007U, .msbPos = 7U, .lsbPos = 7U},

    [REG_FIELD_LINK_EN_A]                = {.regAddr = 0x0006U, .msbPos = 0U, .lsbPos = 0U, .longDelay = true},
    [REG_FIELD_LINK_EN_B]                = {.regAddr = 0x0006U, .msbPos = 1U, .lsbPos = 1U, .longDelay = true},
    [REG_FIELD_LINK_EN_C]                = {.regAddr = 0x0006U, .msbPos = 2U, .lsbPos = 2U, .longDelay = true},
    [REG_FIELD_LINK_EN_D]                = {.regAddr = 0x0006U, .msbPos = 3U, .lsbPos = 3U, .longDelay = true},

    [REG_FIELD_LINK_GMSL2_A]             = {.regAddr = 0x0006U, .msbPos = 4U, .lsbPos = 4U, .longDelay = true},
    [REG_FIELD_LINK_GMSL2_B]             = {.regAddr = 0x0006U, .msbPos = 5U, .lsbPos = 5U, .longDelay = true},
    [REG_FIELD_LINK_GMSL2_C]             = {.regAddr = 0x0006U, .msbPos = 6U, .lsbPos = 6U, .longDelay = true},
    [REG_FIELD_LINK_GMSL2_D]             = {.regAddr = 0x0006U, .msbPos = 7U, .lsbPos = 7U, .longDelay = true},

    [REG_FIELD_RX_RATE_PHY_A]            = {.regAddr = 0x0010U, .msbPos = 1U, .lsbPos = 0U},
    [REG_FIELD_RX_RATE_PHY_B]            = {.regAddr = 0x0010U, .msbPos = 5U, .lsbPos = 4U},
    [REG_FIELD_RX_RATE_PHY_C]            = {.regAddr = 0x0011U, .msbPos = 1U, .lsbPos = 0U},
    [REG_FIELD_RX_RATE_PHY_D]            = {.regAddr = 0x0011U, .msbPos = 5U, .lsbPos = 4U},

    [REG_FIELD_SOFT_BPP_0]               = {.regAddr = 0x040BU, .msbPos = 7U, .lsbPos = 3U},
    [REG_FIELD_SOFT_BPP_1]               = {.regAddr = 0x0411U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_SOFT_BPP_2_L]             = {.regAddr = 0x0412U, .msbPos = 1U, .lsbPos = 0U},
    [REG_FIELD_SOFT_BPP_3]               = {.regAddr = 0x0412U, .msbPos = 6U, .lsbPos = 2U},
    [REG_FIELD_SOFT_BPP_2_H]             = {.regAddr = 0x0411U, .msbPos = 7U, .lsbPos = 5U},
    [REG_FIELD_SOFT_BPP_4]               = {.regAddr = 0x042BU, .msbPos = 7U, .lsbPos = 3U},
    [REG_FIELD_SOFT_BPP_5]               = {.regAddr = 0x0431U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_SOFT_BPP_6_H]             = {.regAddr = 0x0431U, .msbPos = 7U, .lsbPos = 5U},
    [REG_FIELD_SOFT_BPP_6_L]             = {.regAddr = 0x0432U, .msbPos = 1U, .lsbPos = 0U},
    [REG_FIELD_SOFT_BPP_7]               = {.regAddr = 0x0432U, .msbPos = 6U, .lsbPos = 2U},

    [REG_FIELD_SOFT_DT_0]                = {.regAddr = 0x040EU, .msbPos = 5U, .lsbPos = 0U},
    [REG_FIELD_SOFT_DT_1_L]              = {.regAddr = 0x040FU, .msbPos = 3U, .lsbPos = 0U},
    [REG_FIELD_SOFT_DT_2_L]              = {.regAddr = 0x0410U, .msbPos = 1U, .lsbPos = 0U},
    [REG_FIELD_SOFT_DT_3]                = {.regAddr = 0x0410U, .msbPos = 7U, .lsbPos = 2U},
    [REG_FIELD_SOFT_DT_1_H]              = {.regAddr = 0x040EU, .msbPos = 7U, .lsbPos = 6U},
    [REG_FIELD_SOFT_DT_2_H]              = {.regAddr = 0x040FU, .msbPos = 7U, .lsbPos = 4U},
    [REG_FIELD_SOFT_DT_4]                = {.regAddr = 0x042EU, .msbPos = 5U, .lsbPos = 0U},
    [REG_FIELD_SOFT_DT_5_H]              = {.regAddr = 0x042EU, .msbPos = 7U, .lsbPos = 6U},
    [REG_FIELD_SOFT_DT_5_L]              = {.regAddr = 0x042FU, .msbPos = 3U, .lsbPos = 0U},
    [REG_FIELD_SOFT_DT_6_H]              = {.regAddr = 0x042FU, .msbPos = 7U, .lsbPos = 4U},
    [REG_FIELD_SOFT_DT_6_L]              = {.regAddr = 0x0430U, .msbPos = 1U, .lsbPos = 0U},
    [REG_FIELD_SOFT_DT_7]                = {.regAddr = 0x0430U, .msbPos = 7U, .lsbPos = 2U},


    [REG_FIELD_SOFT_OVR_0_EN]            = {.regAddr = 0x0415U, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_SOFT_OVR_1_EN]            = {.regAddr = 0x0415U, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_SOFT_OVR_2_EN]            = {.regAddr = 0x0418U, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_SOFT_OVR_3_EN]            = {.regAddr = 0x0418U, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_SOFT_OVR_4_EN]            = {.regAddr = 0x041BU, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_SOFT_OVR_5_EN]            = {.regAddr = 0x041BU, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_SOFT_OVR_6_EN]            = {.regAddr = 0x041DU, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_SOFT_OVR_7_EN]            = {.regAddr = 0x041DU, .msbPos = 7U, .lsbPos = 7U},

    [REG_FIELD_CC_CRC_ERRCNT_A]          = {.regAddr = 0x0B19U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_CC_CRC_ERRCNT_B]          = {.regAddr = 0x0C19U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_CC_CRC_ERRCNT_C]          = {.regAddr = 0x0D19U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_CC_CRC_ERRCNT_D]          = {.regAddr = 0x0E19U, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_I2C_FWDCCEN_PHY_A]        = {.regAddr = 0x0B04U, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_I2C_FWDCCEN_PHY_B]        = {.regAddr = 0x0C04U, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_I2C_FWDCCEN_PHY_C]        = {.regAddr = 0x0D04U, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_I2C_FWDCCEN_PHY_D]        = {.regAddr = 0x0E04U, .msbPos = 0U, .lsbPos = 0U},

    [REG_FIELD_I2C_REVCCEN_PHY_A]        = {.regAddr = 0x0B04U, .msbPos = 1U, .lsbPos = 1U},
    [REG_FIELD_I2C_REVCCEN_PHY_B]        = {.regAddr = 0x0C04U, .msbPos = 1U, .lsbPos = 1U},
    [REG_FIELD_I2C_REVCCEN_PHY_C]        = {.regAddr = 0x0D04U, .msbPos = 1U, .lsbPos = 1U},
    [REG_FIELD_I2C_REVCCEN_PHY_D]        = {.regAddr = 0x0E04U, .msbPos = 1U, .lsbPos = 1U},

    [REG_FIELD_I2C_PORT_GMSL1_PHY_A]     = {.regAddr = 0x0B04U, .msbPos = 3U, .lsbPos = 3U},
    [REG_FIELD_I2C_PORT_GMSL1_PHY_B]     = {.regAddr = 0x0C04U, .msbPos = 3U, .lsbPos = 3U},
    [REG_FIELD_I2C_PORT_GMSL1_PHY_C]     = {.regAddr = 0x0D04U, .msbPos = 3U, .lsbPos = 3U},
    [REG_FIELD_I2C_PORT_GMSL1_PHY_D]     = {.regAddr = 0x0E04U, .msbPos = 3U, .lsbPos = 3U},

    [REG_FIELD_DE_EN_PHY_A]              = {.regAddr = 0x0B0FU, .msbPos = 3U, .lsbPos = 3U},
    [REG_FIELD_DE_EN_PHY_B]              = {.regAddr = 0x0C0FU, .msbPos = 3U, .lsbPos = 3U},
    [REG_FIELD_DE_EN_PHY_C]              = {.regAddr = 0x0D0FU, .msbPos = 3U, .lsbPos = 3U},
    [REG_FIELD_DE_EN_PHY_D]              = {.regAddr = 0x0E0FU, .msbPos = 3U, .lsbPos = 3U},

    [REG_FIELD_DE_PRBS_TYPE_PHY_A]       = {.regAddr = 0x0B0FU, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_DE_PRBS_TYPE_PHY_B]       = {.regAddr = 0x0C0FU, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_DE_PRBS_TYPE_PHY_C]       = {.regAddr = 0x0D0FU, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_DE_PRBS_TYPE_PHY_D]       = {.regAddr = 0x0E0FU, .msbPos = 0U, .lsbPos = 0U},

    [REG_FIELD_PKTCCEN_LINK_A]           = {.regAddr = 0x0B08U, .msbPos = 2U, .lsbPos = 2U},
    [REG_FIELD_PKTCCEN_LINK_B]           = {.regAddr = 0x0C08U, .msbPos = 2U, .lsbPos = 2U},
    [REG_FIELD_PKTCCEN_LINK_C]           = {.regAddr = 0x0D08U, .msbPos = 2U, .lsbPos = 2U},
    [REG_FIELD_PKTCCEN_LINK_D]           = {.regAddr = 0x0E08U, .msbPos = 2U, .lsbPos = 2U},

    [REG_FIELD_ALT_MEM_MAP12_PHY0]       = {.regAddr = 0x0933U, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_ALT_MEM_MAP12_PHY1]       = {.regAddr = 0x0973U, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_ALT_MEM_MAP12_PHY2]       = {.regAddr = 0x09B3U, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_ALT_MEM_MAP12_PHY3]       = {.regAddr = 0x09F3U, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_ALT_MEM_MAP8_PHY0]        = {.regAddr = 0x0933U, .msbPos = 1U, .lsbPos = 1U},
    [REG_FIELD_ALT_MEM_MAP8_PHY1]        = {.regAddr = 0x0973U, .msbPos = 1U, .lsbPos = 1U},
    [REG_FIELD_ALT_MEM_MAP8_PHY2]        = {.regAddr = 0x09B3U, .msbPos = 1U, .lsbPos = 1U},
    [REG_FIELD_ALT_MEM_MAP8_PHY3]        = {.regAddr = 0x09F3U, .msbPos = 1U, .lsbPos = 1U},
    [REG_FIELD_ALT_MEM_MAP10_PHY0]       = {.regAddr = 0x0933U, .msbPos = 2U, .lsbPos = 2U},
    [REG_FIELD_ALT_MEM_MAP10_PHY1]       = {.regAddr = 0x0973U, .msbPos = 2U, .lsbPos = 2U},
    [REG_FIELD_ALT_MEM_MAP10_PHY2]       = {.regAddr = 0x09B3U, .msbPos = 2U, .lsbPos = 2U},
    [REG_FIELD_ALT_MEM_MAP10_PHY3]       = {.regAddr = 0x09F3U, .msbPos = 2U, .lsbPos = 2U},
    [REG_FIELD_ALT2_MEM_MAP8_PHY0]       = {.regAddr = 0x0933U, .msbPos = 4U, .lsbPos = 4U},
    [REG_FIELD_ALT2_MEM_MAP8_PHY1]       = {.regAddr = 0x0973U, .msbPos = 4U, .lsbPos = 4U},
    [REG_FIELD_ALT2_MEM_MAP8_PHY2]       = {.regAddr = 0x09B3U, .msbPos = 4U, .lsbPos = 4U},
    [REG_FIELD_ALT2_MEM_MAP8_PHY3]       = {.regAddr = 0x09F3U, .msbPos = 4U, .lsbPos = 4U},

    [REG_FIELD_BPP8DBL_0]                = {.regAddr = 0x0414U, .msbPos = 4U, .lsbPos = 4U},
    [REG_FIELD_BPP8DBL_1]                = {.regAddr = 0x0414U, .msbPos = 5U, .lsbPos = 5U},
    [REG_FIELD_BPP8DBL_2]                = {.regAddr = 0x0414U, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_BPP8DBL_3]                = {.regAddr = 0x0414U, .msbPos = 7U, .lsbPos = 7U},

    [REG_FIELD_BPP8DBL_4]                = {.regAddr = 0x0434U, .msbPos = 4U, .lsbPos = 4U},
    [REG_FIELD_BPP8DBL_5]                = {.regAddr = 0x0434U, .msbPos = 5U, .lsbPos = 5U},
    [REG_FIELD_BPP8DBL_6]                = {.regAddr = 0x0434U, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_BPP8DBL_7]                = {.regAddr = 0x0434U, .msbPos = 7U, .lsbPos = 7U},

    [REG_FIELD_BPP8DBL_MODE_0]           = {.regAddr = 0x0417U, .msbPos = 4U, .lsbPos = 4U},
    [REG_FIELD_BPP8DBL_MODE_1]           = {.regAddr = 0x0417U, .msbPos = 5U, .lsbPos = 5U},
    [REG_FIELD_BPP8DBL_MODE_2]           = {.regAddr = 0x0417U, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_BPP8DBL_MODE_3]           = {.regAddr = 0x0417U, .msbPos = 7U, .lsbPos = 7U},

    [REG_FIELD_BPP8DBL_MODE_4]           = {.regAddr = 0x0437U, .msbPos = 4U, .lsbPos = 4U},
    [REG_FIELD_BPP8DBL_MODE_5]           = {.regAddr = 0x0437U, .msbPos = 5U, .lsbPos = 5U},
    [REG_FIELD_BPP8DBL_MODE_6]           = {.regAddr = 0x0437U, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_BPP8DBL_MODE_7]           = {.regAddr = 0x0437U, .msbPos = 7U, .lsbPos = 7U},

    [REG_FIELD_AEQ_PHY_A]                = {.regAddr = 0x0B14U, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_AEQ_PHY_B]                = {.regAddr = 0x0C14U, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_AEQ_PHY_C]                = {.regAddr = 0x0D14U, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_AEQ_PHY_D]                = {.regAddr = 0x0E14U, .msbPos = 7U, .lsbPos = 7U},

    [REG_FIELD_PERIODIC_AEQ_PHY_A]       = {.regAddr = 0x0B14U, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_PERIODIC_AEQ_PHY_B]       = {.regAddr = 0x0C14U, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_PERIODIC_AEQ_PHY_C]       = {.regAddr = 0x0D14U, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_PERIODIC_AEQ_PHY_D]       = {.regAddr = 0x0E14U, .msbPos = 6U, .lsbPos = 6U},

    [REG_FIELD_EOM_PER_THR_PHY_A]        = {.regAddr = 0x0B14U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_EOM_PER_THR_PHY_B]        = {.regAddr = 0x0C14U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_EOM_PER_THR_PHY_C]        = {.regAddr = 0x0D14U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_EOM_PER_THR_PHY_D]        = {.regAddr = 0x0E14U, .msbPos = 4U, .lsbPos = 0U},

    [REG_FIELD_VIDEO_PIPE_SEL_0]         = {.regAddr = 0x00F0U, .msbPos = 3U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_PIPE_SEL_1]         = {.regAddr = 0x00F0U, .msbPos = 7U, .lsbPos = 4U},
    [REG_FIELD_VIDEO_PIPE_SEL_2]         = {.regAddr = 0x00F1U, .msbPos = 3U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_PIPE_SEL_3]         = {.regAddr = 0x00F1U, .msbPos = 7U, .lsbPos = 4U},
    [REG_FIELD_VIDEO_PIPE_SEL_4]         = {.regAddr = 0x00F2U, .msbPos = 3U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_PIPE_SEL_5]         = {.regAddr = 0x00F2U, .msbPos = 7U, .lsbPos = 4U},
    [REG_FIELD_VIDEO_PIPE_SEL_6]         = {.regAddr = 0x00F3U, .msbPos = 3U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_PIPE_SEL_7]         = {.regAddr = 0x00F3U, .msbPos = 7U, .lsbPos = 4U},

    [REG_FIELD_VIDEO_PIPE_EN_0]          = {.regAddr = 0x00F4U, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_PIPE_EN_1]          = {.regAddr = 0x00F4U, .msbPos = 1U, .lsbPos = 1U},
    [REG_FIELD_VIDEO_PIPE_EN_2]          = {.regAddr = 0x00F4U, .msbPos = 2U, .lsbPos = 2U},
    [REG_FIELD_VIDEO_PIPE_EN_3]          = {.regAddr = 0x00F4U, .msbPos = 3U, .lsbPos = 3U},
    [REG_FIELD_VIDEO_PIPE_EN_4]          = {.regAddr = 0x00F4U, .msbPos = 4U, .lsbPos = 4U},
    [REG_FIELD_VIDEO_PIPE_EN_5]          = {.regAddr = 0x00F4U, .msbPos = 5U, .lsbPos = 5U},
    [REG_FIELD_VIDEO_PIPE_EN_6]          = {.regAddr = 0x00F4U, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_VIDEO_PIPE_EN_7]          = {.regAddr = 0x00F4U, .msbPos = 7U, .lsbPos = 7U},

    [REG_FIELD_PATGEN_CLK_SRC_PIPE_0]    = {.regAddr = 0x01DCU, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_PATGEN_CLK_SRC_PIPE_1]    = {.regAddr = 0x01FCU, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_PATGEN_CLK_SRC_PIPE_2]    = {.regAddr = 0x021CU, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_PATGEN_CLK_SRC_PIPE_3]    = {.regAddr = 0x023CU, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_PATGEN_CLK_SRC_PIPE_4]    = {.regAddr = 0x025CU, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_PATGEN_CLK_SRC_PIPE_5]    = {.regAddr = 0x027CU, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_PATGEN_CLK_SRC_PIPE_6]    = {.regAddr = 0x029CU, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_PATGEN_CLK_SRC_PIPE_7]    = {.regAddr = 0x02BCU, .msbPos = 7U, .lsbPos = 7U},

    [REG_FIELD_MIPI_OUT_CFG]             = {.regAddr = 0x08A0U, .msbPos = 5U, .lsbPos = 0U},
    [REG_FIELD_T_T3_PREBEGIN]            = {.regAddr = 0x08ADU, .msbPos = 5U, .lsbPos = 0U},
    [REG_FIELD_T_T3_POST_PREP]           = {.regAddr = 0x08AEU, .msbPos = 6U, .lsbPos = 0U},
    [REG_FIELD_DEV_REV]                  = {.regAddr = 0x004CU, .msbPos = 3U, .lsbPos = 0U},
    [REG_FIELD_DEV_ID]                   = {.regAddr = 0x000DU, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_RESET_ONESHOT]            = {.regAddr = 0x0018U, .msbPos = 3U, .lsbPos = 0U},
    [REG_FIELD_BACKTOP_EN]               = {.regAddr = 0x0400U, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_PHY_STANDBY]              = {.regAddr = 0x08A2U, .msbPos = 7U, .lsbPos = 4U},
    [REG_FIELD_FORCE_CSI_OUT_EN]         = {.regAddr = 0x08A0U, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_ERRB]                     = {.regAddr = 0x001AU, .msbPos = 2U, .lsbPos = 2U},
    [REG_FIELD_OVERFLOW_FIRST4]          = {.regAddr = 0x040AU, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_OVERFLOW_LAST4]           = {.regAddr = 0x042AU, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_OSN_COEFFICIENT_0]        = {.regAddr = 0x142EU, .msbPos = 5U, .lsbPos = 0U},
    [REG_FIELD_OSN_COEFFICIENT_1]        = {.regAddr = 0x152EU, .msbPos = 5U, .lsbPos = 0U},
    [REG_FIELD_OSN_COEFFICIENT_2]        = {.regAddr = 0x162EU, .msbPos = 5U, .lsbPos = 0U},
    [REG_FIELD_OSN_COEFFICIENT_3]        = {.regAddr = 0x172EU, .msbPos = 5U, .lsbPos = 0U},

    [REG_FIELD_ENABLE_OSN_0]             = {.regAddr = 0x1432U, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_ENABLE_OSN_1]             = {.regAddr = 0x1532U, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_ENABLE_OSN_2]             = {.regAddr = 0x1632U, .msbPos = 6U, .lsbPos = 6U},
    [REG_FIELD_ENABLE_OSN_3]             = {.regAddr = 0x1732U, .msbPos = 6U, .lsbPos = 6U},

    [REG_FIELD_OSN_COEFF_MANUAL_SEED_0]  = {.regAddr = 0x1457U, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_OSN_COEFF_MANUAL_SEED_1]  = {.regAddr = 0x1557U, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_OSN_COEFF_MANUAL_SEED_2]  = {.regAddr = 0x1657U, .msbPos = 0U, .lsbPos = 0U},
    [REG_FIELD_OSN_COEFF_MANUAL_SEED_3]  = {.regAddr = 0x1757U, .msbPos = 0U, .lsbPos = 0U},

    [REG_FIELD_SET_TX_AMP_0]             = {.regAddr = 0x1495U, .msbPos = 5U, .lsbPos = 0U},
    [REG_FIELD_SET_TX_AMP_1]             = {.regAddr = 0x1595U, .msbPos = 5U, .lsbPos = 0U},
    [REG_FIELD_SET_TX_AMP_2]             = {.regAddr = 0x1695U, .msbPos = 5U, .lsbPos = 0U},
    [REG_FIELD_SET_TX_AMP_3]             = {.regAddr = 0x1795U, .msbPos = 5U, .lsbPos = 0U},

    [REG_FIELD_SLV_TO_P0_A]              = {.regAddr = 0x0640U, .msbPos = 2U, .lsbPos = 0U},
    [REG_FIELD_SLV_TO_P0_B]              = {.regAddr = 0x0650U, .msbPos = 2U, .lsbPos = 0U},
    [REG_FIELD_SLV_TO_P0_C]              = {.regAddr = 0x0660U, .msbPos = 2U, .lsbPos = 0U},
    [REG_FIELD_SLV_TO_P0_D]              = {.regAddr = 0x0670U, .msbPos = 2U, .lsbPos = 0U},
    [REG_FIELD_SLV_TO_P1_A]              = {.regAddr = 0x0680U, .msbPos = 2U, .lsbPos = 0U},
    [REG_FIELD_SLV_TO_P1_B]              = {.regAddr = 0x0690U, .msbPos = 2U, .lsbPos = 0U},
    [REG_FIELD_SLV_TO_P1_C]              = {.regAddr = 0x06A0U, .msbPos = 2U, .lsbPos = 0U},
    [REG_FIELD_SLV_TO_P1_D]              = {.regAddr = 0x06B0U, .msbPos = 2U, .lsbPos = 0U},
    [REG_FIELD_SLV_TO_P2_A]              = {.regAddr = 0x0688U, .msbPos = 2U, .lsbPos = 0U},
    [REG_FIELD_SLV_TO_P2_B]              = {.regAddr = 0x0698U, .msbPos = 2U, .lsbPos = 0U},
    [REG_FIELD_SLV_TO_P2_C]              = {.regAddr = 0x06A8U, .msbPos = 2U, .lsbPos = 0U},
    [REG_FIELD_SLV_TO_P2_D]              = {.regAddr = 0x06B8U, .msbPos = 2U, .lsbPos = 0U},

    [REG_FIELD_REM_ERR_FLAG]             = {.regAddr = 0x002AU, .msbPos = 1U, .lsbPos = 1U},

    [REG_FIELD_CTRL3]                    = {.regAddr = 0x001AU, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_INTR4]                    = {.regAddr = 0x0027U, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_INTR5]                    = {.regAddr = 0x0028U, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_INTR6]                    = {.regAddr = 0x0029U, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_INTR7]                    = {.regAddr = 0x002AU, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_INTR8]                    = {.regAddr = 0x002BU, .msbPos = 3U, .lsbPos = 0U},

    [REG_FIELD_INTR9]                    = {.regAddr = 0x002CU, .msbPos = 3U, .lsbPos = 0U},

    [REG_FIELD_INTR10]                   = {.regAddr = 0x002DU, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_INTR11]                   = {.regAddr = 0x002EU, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_INTR12_MEM_ERR_OEN]       = {.regAddr = 0x002FU, .msbPos = 6U, .lsbPos = 6U},

    [REG_FIELD_VID_PXL_CRC_ERR_INT]      = {.regAddr = 0x0045U, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_FSYNC_22]                 = {.regAddr = 0x04B6U, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_BACKTOP25]                = {.regAddr = 0x0438U, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_VIDEO_RX8_PIPE_0]         = {.regAddr = 0x0108U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_RX8_PIPE_1]         = {.regAddr = 0x011AU, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_RX8_PIPE_2]         = {.regAddr = 0x012CU, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_RX8_PIPE_3]         = {.regAddr = 0x013EU, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_RX8_PIPE_4]         = {.regAddr = 0x0150U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_RX8_PIPE_5]         = {.regAddr = 0x0168U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_RX8_PIPE_6]         = {.regAddr = 0x017AU, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_RX8_PIPE_7]         = {.regAddr = 0x018CU, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_VIDEO_MASKED_OEN]         = {.regAddr = 0x0049U, .msbPos = 5U, .lsbPos = 0U},

    [REG_FIELD_VIDEO_MASKED_FLAG]        = {.regAddr = 0x004AU, .msbPos = 5U, .lsbPos = 0U},

    [REG_FIELD_PWR0]                     = {.regAddr = 0x0012U, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_PWR_STATUS_FLAG]          = {.regAddr = 0x0047U, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_ENABLE_LOCK]              = {.regAddr = 0x0005U, .msbPos = 7U, .lsbPos = 7U},
    [REG_FIELD_ENABLE_ERRB]              = {.regAddr = 0x0005U, .msbPos = 6U, .lsbPos = 6U},

    [REG_FIELD_CRUSSC_MODE_0]            = {.regAddr = 0x1445U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_CRUSSC_MODE_1]            = {.regAddr = 0x1545U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_CRUSSC_MODE_2]            = {.regAddr = 0x1645U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_CRUSSC_MODE_3]            = {.regAddr = 0x1745U, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_VIDEO_RX0_PIPE_0]         = {.regAddr = 0x0100U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_RX0_PIPE_1]         = {.regAddr = 0x0112U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_RX0_PIPE_2]         = {.regAddr = 0x0124U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_RX0_PIPE_3]         = {.regAddr = 0x0136U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_RX0_PIPE_4]         = {.regAddr = 0x0148U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_RX0_PIPE_5]         = {.regAddr = 0x0160U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_RX0_PIPE_6]         = {.regAddr = 0x0172U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_VIDEO_RX0_PIPE_7]         = {.regAddr = 0x0184U, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_CFGH_VIDEO_CRC0]          = {.regAddr = 0x0060U, .msbPos = 7U, .lsbPos = 0U},
    [REG_FIELD_CFGH_VIDEO_CRC1]          = {.regAddr = 0x0061U, .msbPos = 7U, .lsbPos = 0U},

    [REG_FIELD_GPIO_00_A_A]              = {.regAddr = 0x0300U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_01_A_A]              = {.regAddr = 0x0303U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_02_A_A]              = {.regAddr = 0x0306U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_03_A_A]              = {.regAddr = 0x0309U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_04_A_A]              = {.regAddr = 0x030CU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_05_A_A]              = {.regAddr = 0x0310U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_06_A_A]              = {.regAddr = 0x0313U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_07_A_A]              = {.regAddr = 0x0316U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_08_A_A]              = {.regAddr = 0x0319U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_09_A_A]              = {.regAddr = 0x031CU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_10_A_A]              = {.regAddr = 0x0320U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_11_A_A]              = {.regAddr = 0x0323U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_12_A_A]              = {.regAddr = 0x0326U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_13_A_A]              = {.regAddr = 0x0329U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_14_A_A]              = {.regAddr = 0x032CU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_15_A_A]              = {.regAddr = 0x0330U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_16_A_A]              = {.regAddr = 0x0333U, .msbPos = 4U, .lsbPos = 0U},

    [REG_FIELD_GPIO_00_A_B]              = {.regAddr = 0x0301U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_01_A_B]              = {.regAddr = 0x0304U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_02_A_B]              = {.regAddr = 0x0307U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_03_A_B]              = {.regAddr = 0x030AU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_04_A_B]              = {.regAddr = 0x030DU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_05_A_B]              = {.regAddr = 0x0311U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_06_A_B]              = {.regAddr = 0x0314U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_07_A_B]              = {.regAddr = 0x0317U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_08_A_B]              = {.regAddr = 0x031AU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_09_A_B]              = {.regAddr = 0x031DU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_10_A_B]              = {.regAddr = 0x0321U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_11_A_B]              = {.regAddr = 0x0324U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_12_A_B]              = {.regAddr = 0x0327U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_13_A_B]              = {.regAddr = 0x032AU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_14_A_B]              = {.regAddr = 0x032DU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_15_A_B]              = {.regAddr = 0x0331U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_16_A_B]              = {.regAddr = 0x0334U, .msbPos = 4U, .lsbPos = 0U},

    [REG_FIELD_GPIO_00_B_B]              = {.regAddr = 0x0337U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_01_B_B]              = {.regAddr = 0x033AU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_02_B_B]              = {.regAddr = 0x033DU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_03_B_B]              = {.regAddr = 0x0341U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_04_B_B]              = {.regAddr = 0x0344U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_05_B_B]              = {.regAddr = 0x0347U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_06_B_B]              = {.regAddr = 0x034AU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_07_B_B]              = {.regAddr = 0x034DU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_08_B_B]              = {.regAddr = 0x0351U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_09_B_B]              = {.regAddr = 0x0354U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_10_B_B]              = {.regAddr = 0x0357U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_11_B_B]              = {.regAddr = 0x035AU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_12_B_B]              = {.regAddr = 0x035DU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_13_B_B]              = {.regAddr = 0x0361U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_14_B_B]              = {.regAddr = 0x0364U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_15_B_B]              = {.regAddr = 0x0367U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_16_B_B]              = {.regAddr = 0x036AU, .msbPos = 4U, .lsbPos = 0U},

    [REG_FIELD_GPIO_00_C_B]              = {.regAddr = 0x036DU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_01_C_B]              = {.regAddr = 0x0371U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_02_C_B]              = {.regAddr = 0x0374U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_03_C_B]              = {.regAddr = 0x0377U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_04_C_B]              = {.regAddr = 0x037AU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_05_C_B]              = {.regAddr = 0x037DU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_06_C_B]              = {.regAddr = 0x0381U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_07_C_B]              = {.regAddr = 0x0384U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_08_C_B]              = {.regAddr = 0x0387U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_09_C_B]              = {.regAddr = 0x038AU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_10_C_B]              = {.regAddr = 0x038DU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_11_C_B]              = {.regAddr = 0x0391U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_12_C_B]              = {.regAddr = 0x0394U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_13_C_B]              = {.regAddr = 0x0397U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_14_C_B]              = {.regAddr = 0x039AU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_15_C_B]              = {.regAddr = 0x039DU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_16_C_B]              = {.regAddr = 0x03A1U, .msbPos = 4U, .lsbPos = 0U},

    [REG_FIELD_GPIO_00_D_B]              = {.regAddr = 0x03A4U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_01_D_B]              = {.regAddr = 0x03A7U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_02_D_B]              = {.regAddr = 0x03AAU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_03_D_B]              = {.regAddr = 0x03ADU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_04_D_B]              = {.regAddr = 0x03B1U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_05_D_B]              = {.regAddr = 0x03B4U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_06_D_B]              = {.regAddr = 0x03B7U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_07_D_B]              = {.regAddr = 0x03BAU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_08_D_B]              = {.regAddr = 0x03BDU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_09_D_B]              = {.regAddr = 0x03C1U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_10_D_B]              = {.regAddr = 0x03C4U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_11_D_B]              = {.regAddr = 0x03C7U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_12_D_B]              = {.regAddr = 0x03CAU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_13_D_B]              = {.regAddr = 0x03CDU, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_14_D_B]              = {.regAddr = 0x03D1U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_15_D_B]              = {.regAddr = 0x03D4U, .msbPos = 4U, .lsbPos = 0U},
    [REG_FIELD_GPIO_16_D_B]              = {.regAddr = 0x03D7U, .msbPos = 4U, .lsbPos = 0U},
};

/**
 * @brief Returns the masks needed for read-verify and write-verify
 *
 * @param[in]  address  register address
 * @param[out] regMasks structure to store the read/write-verify masks
 *
 * @retval NVMEDIA_STATUS_OK if mask is found
 * @retval NVMEDIA_STATUS_ERROR if mask is not found
 *
 * Mask selection logic per bit:
 *  - Bit Nature         | ReadMask | WriteMask
 *  - Reserved           |    0     |     0
 *  - Undef              |    0     |     0
 *  - Read + Write       |    1     |     1
 *  - ReadOnly           |    1     |     0
 *  - WriteOnly          |    0     |     0
 *  - COR (ClearOnRead)  |    0     |     0
 *  - COW (ClearOnWrite) |    0     |     0
 *  - Read + COW         |    1     |     0
 *  - COR + Write        |    0     |     0
 *  - COR + COW          |    0     |     0
 */
static inline NvMediaStatus GetRegReadWriteMasks(
    uint16_t const address, MAX96712RegMask * const regMasks)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t rMask = 0x00U;
    uint8_t wMask = 0x00U;
    switch (address)
    {
        case 0x0003U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0005U: rMask = 0xE0U; wMask = 0xE0U; break;
        case 0x0006U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0007U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0009U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x000AU: rMask = 0x08U; wMask = 0x00U; break;
        case 0x000BU: rMask = 0x08U; wMask = 0x00U; break;
        case 0x000CU: rMask = 0x08U; wMask = 0x00U; break;
        case 0x000DU: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x0010U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0011U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0012U: rMask = 0xFFU; wMask = 0x00U; break;
        /* 0x0013: has RESET_ALL bit will reset all the registers so, updated
         * rMask = 0x40U and wMask = 0x00U */
        case 0x0013U: rMask = 0x40U; wMask = 0x00U; break;
        case 0x0018U: rMask = 0xFFU; wMask = 0xF0U; break;
        case 0x001AU: rMask = 0x0EU; wMask = 0x00U; break;
        case 0x0023U: rMask = 0x0FU; wMask = 0x0FU; break;
        case 0x0025U: rMask = 0x0FU; wMask = 0x0FU; break;
        case 0x0026U: rMask = 0x0FU; wMask = 0x00U; break;
        case 0x0027U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0028U: rMask = 0xFCU; wMask = 0x00U; break;
        case 0x0029U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x002AU: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x002BU: rMask = 0x0FU; wMask = 0x0FU; break;
        case 0x002CU: rMask = 0x0FU; wMask = 0x00U; break;
        case 0x002DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x002EU: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x002FU: rMask = 0xDFU; wMask = 0xDFU; break;
        case 0x0030U: rMask = 0x9FU; wMask = 0x9FU; break;
        case 0x0031U: rMask = 0x9FU; wMask = 0x9FU; break;
        case 0x0032U: rMask = 0x9FU; wMask = 0x9FU; break;
        case 0x0033U: rMask = 0x9FU; wMask = 0x9FU; break;
        case 0x0035U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0036U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0037U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0038U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0039U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x003AU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x003BU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x003CU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0044U: rMask = 0xCFU; wMask = 0xCFU; break;
        case 0x0045U: rMask = 0xCFU; wMask = 0x00U; break;
        case 0x0046U: rMask = 0xA0U; wMask = 0xA0U; break;
        case 0x0047U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0049U: rMask = 0x3FU; wMask = 0x3FU; break;
        case 0x004AU: rMask = 0x20U; wMask = 0x00U; break;
        case 0x004CU: rMask = 0x0FU; wMask = 0x00U; break;
        case 0x0060U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0061U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0070U: rMask = 0xCCU; wMask = 0xCCU; break;
        case 0x0074U: rMask = 0xCCU; wMask = 0xCCU; break;
        case 0x0078U: rMask = 0xCCU; wMask = 0xCCU; break;
        case 0x007CU: rMask = 0xCCU; wMask = 0xCCU; break;
        case 0x00A0U: rMask = 0xCCU; wMask = 0xCCU; break;
        case 0x00A6U: rMask = 0x03U; wMask = 0x03U; break;
        case 0x00A7U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x00A8U: rMask = 0xCCU; wMask = 0xCCU; break;
        case 0x00AEU: rMask = 0x03U; wMask = 0x03U; break;
        case 0x00AFU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x00B0U: rMask = 0xCCU; wMask = 0xCCU; break;
        case 0x00B6U: rMask = 0x03U; wMask = 0x03U; break;
        case 0x00B7U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x00B8U: rMask = 0xCCU; wMask = 0xCCU; break;
        case 0x00BEU: rMask = 0x03U; wMask = 0x03U; break;
        case 0x00BFU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x00F0U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x00F1U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x00F2U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x00F3U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x00F4U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0100U: rMask = 0x13U; wMask = 0x13U; break;
        case 0x0106U: rMask = 0x08U; wMask = 0x08U; break;
        case 0x0108U: rMask = 0x60U; wMask = 0x00U; break;
        case 0x0112U: rMask = 0x13U; wMask = 0x13U; break;
        case 0x0118U: rMask = 0x08U; wMask = 0x08U; break;
        case 0x011AU: rMask = 0x60U; wMask = 0x00U; break;
        case 0x0124U: rMask = 0x13U; wMask = 0x13U; break;
        case 0x012AU: rMask = 0x08U; wMask = 0x08U; break;
        case 0x012CU: rMask = 0x60U; wMask = 0x00U; break;
        case 0x0136U: rMask = 0x13U; wMask = 0x13U; break;
        case 0x013CU: rMask = 0x08U; wMask = 0x08U; break;
        case 0x013EU: rMask = 0x60U; wMask = 0x00U; break;
        case 0x0148U: rMask = 0x13U; wMask = 0x13U; break;
        case 0x014EU: rMask = 0x08U; wMask = 0x08U; break;
        case 0x0150U: rMask = 0x60U; wMask = 0x00U; break;
        case 0x0160U: rMask = 0x13U; wMask = 0x13U; break;
        case 0x0166U: rMask = 0x08U; wMask = 0x08U; break;
        case 0x0168U: rMask = 0x60U; wMask = 0x00U; break;
        case 0x0172U: rMask = 0x13U; wMask = 0x13U; break;
        case 0x0178U: rMask = 0x08U; wMask = 0x08U; break;
        case 0x017AU: rMask = 0x60U; wMask = 0x00U; break;
        case 0x0184U: rMask = 0x13U; wMask = 0x13U; break;
        case 0x018AU: rMask = 0x08U; wMask = 0x08U; break;
        case 0x018CU: rMask = 0x60U; wMask = 0x00U; break;
        case 0x01DBU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x01DCU: rMask = 0x9FU; wMask = 0x9EU; break;
        case 0x01DDU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x01FBU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x01FCU: rMask = 0x9FU; wMask = 0x9EU; break;
        case 0x01FDU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x021BU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x021CU: rMask = 0x9FU; wMask = 0x9EU; break;
        case 0x021DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x023BU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x023CU: rMask = 0x9FU; wMask = 0x9EU; break;
        case 0x023DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x025BU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x025CU: rMask = 0x9FU; wMask = 0x9EU; break;
        case 0x025DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x027BU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x027CU: rMask = 0x9FU; wMask = 0x9EU; break;
        case 0x027DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x029BU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x029CU: rMask = 0x9FU; wMask = 0x9EU; break;
        case 0x029DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x02BBU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x02BCU: rMask = 0x9FU; wMask = 0x9EU; break;
        case 0x02BDU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0303U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x0304U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0306U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x0307U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0309U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x030AU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x030CU: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x030DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0310U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x0311U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0313U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x0314U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0316U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x0317U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0319U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x031AU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x031CU: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x031DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0320U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x0321U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0323U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x0324U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0326U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x0327U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0329U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x032AU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x032CU: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x032DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0330U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x0331U: rMask = 0xDFU; wMask = 0xDFU; break;
        case 0x0333U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x0334U: rMask = 0xDFU; wMask = 0xDFU; break;
        case 0x0337U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x033AU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x033DU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0341U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0344U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0347U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x034AU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x034DU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0351U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0354U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0357U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x035AU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x035DU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0361U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0364U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0367U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x036AU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x036DU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0371U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0374U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0377U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x037AU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x037DU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0381U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0384U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0387U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x038AU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x038DU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0391U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0394U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0397U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x039AU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x039DU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03A1U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03A4U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03A7U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03AAU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03ADU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03B1U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03B4U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03B7U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03BAU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03BDU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03C1U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03C4U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03C7U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03CAU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03CDU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03D1U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03D4U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x03D7U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0400U: rMask = 0xFDU; wMask = 0x0DU; break;
        case 0x040AU: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x040BU: rMask = 0xFAU; wMask = 0xFAU; break;
        case 0x040EU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x040FU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0410U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0411U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0412U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0414U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0415U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0417U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0418U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x041BU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x041DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x041EU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x042AU: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x042BU: rMask = 0xF8U; wMask = 0xF8U; break;
        case 0x042EU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x042FU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0430U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0431U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0432U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0434U: rMask = 0xF0U; wMask = 0xF0U; break;
        case 0x0437U: rMask = 0xF0U; wMask = 0xF0U; break;
        case 0x0438U: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x04A0U: rMask = 0x3FU; wMask = 0x3FU; break;
        case 0x04A2U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x04A5U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x04A6U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x04A7U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x04A8U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x04A9U: rMask = 0x1FU; wMask = 0x1FU; break;
        case 0x04AAU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x04ABU: rMask = 0x9FU; wMask = 0x9FU; break;
        case 0x04AFU: rMask = 0xDFU; wMask = 0xDFU; break;
        case 0x04B1U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x04B6U: rMask = 0x7FU; wMask = 0x00U; break;
        case 0x0500U: rMask = 0xCFU; wMask = 0xCFU; break;
        case 0x0503U: rMask = 0x07U; wMask = 0x07U; break;
        case 0x0506U: rMask = 0x03U; wMask = 0x03U; break;
        case 0x0507U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0510U: rMask = 0xCFU; wMask = 0xCFU; break;
        case 0x0513U: rMask = 0x07U; wMask = 0x07U; break;
        case 0x0516U: rMask = 0x03U; wMask = 0x03U; break;
        case 0x0517U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0520U: rMask = 0xCFU; wMask = 0xCFU; break;
        case 0x0523U: rMask = 0x07U; wMask = 0x07U; break;
        case 0x0526U: rMask = 0x03U; wMask = 0x03U; break;
        case 0x0527U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0530U: rMask = 0xCFU; wMask = 0xCFU; break;
        case 0x0533U: rMask = 0x07U; wMask = 0x07U; break;
        case 0x0536U: rMask = 0x03U; wMask = 0x03U; break;
        case 0x0537U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0560U: rMask = 0xCFU; wMask = 0xCFU; break;
        case 0x0566U: rMask = 0x03U; wMask = 0x03U; break;
        case 0x0567U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0570U: rMask = 0xCFU; wMask = 0xCFU; break;
        case 0x0576U: rMask = 0x03U; wMask = 0x03U; break;
        case 0x0577U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0580U: rMask = 0xCFU; wMask = 0xCFU; break;
        case 0x0586U: rMask = 0x03U; wMask = 0x03U; break;
        case 0x0587U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0590U: rMask = 0xCFU; wMask = 0xCFU; break;
        case 0x0596U: rMask = 0x03U; wMask = 0x03U; break;
        case 0x0597U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x05A0U: rMask = 0xCFU; wMask = 0xCFU; break;
        case 0x05A6U: rMask = 0x03U; wMask = 0x03U; break;
        case 0x05A7U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x05B0U: rMask = 0xCFU; wMask = 0xCFU; break;
        case 0x05B6U: rMask = 0x03U; wMask = 0x03U; break;
        case 0x05B7U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x05C0U: rMask = 0xCFU; wMask = 0xCFU; break;
        case 0x05C6U: rMask = 0x03U; wMask = 0x03U; break;
        case 0x05C7U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x05D0U: rMask = 0xCFU; wMask = 0xCFU; break;
        case 0x05D6U: rMask = 0x03U; wMask = 0x03U; break;
        case 0x05D7U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0640U: rMask = 0x37U; wMask = 0x37U; break;
        case 0x0650U: rMask = 0x37U; wMask = 0x37U; break;
        case 0x0660U: rMask = 0x37U; wMask = 0x37U; break;
        case 0x0670U: rMask = 0x37U; wMask = 0x37U; break;
        case 0x0680U: rMask = 0x37U; wMask = 0x37U; break;
        case 0x0688U: rMask = 0x37U; wMask = 0x37U; break;
        case 0x0690U: rMask = 0x37U; wMask = 0x37U; break;
        case 0x0698U: rMask = 0x37U; wMask = 0x37U; break;
        case 0x06A0U: rMask = 0x37U; wMask = 0x37U; break;
        case 0x06A8U: rMask = 0x37U; wMask = 0x37U; break;
        case 0x06B0U: rMask = 0x37U; wMask = 0x37U; break;
        case 0x06B8U: rMask = 0x37U; wMask = 0x37U; break;
        /* As per datasheet, must set bits [7:0] = 0x10 in reserved
         * register 0x6C2. All bits are write/read */
        case 0x06C2U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x06DFU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x08A0U: rMask = 0xFDU; wMask = 0xFDU; break;
        case 0x08A2U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x08A3U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x08A4U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x08A9U: rMask = 0xF8U; wMask = 0xF8U; break;
        case 0x08AAU: rMask = 0xF8U; wMask = 0xF8U; break;
        case 0x08ADU: rMask = 0x3FU; wMask = 0x3FU; break;
        case 0x08AEU: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0903U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x090AU: rMask = 0xF7U; wMask = 0xF7U; break;
        case 0x090BU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x090DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x090EU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x090FU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0910U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0911U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0912U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0913U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0914U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x092DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0933U: rMask = 0x1FU; wMask = 0x1FU; break;
        case 0x0943U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x094AU: rMask = 0xF7U; wMask = 0xF7U; break;
        case 0x094BU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x094DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x094EU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x094FU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0950U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0951U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0952U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0953U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0954U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x096DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0973U: rMask = 0x1FU; wMask = 0x1FU; break;
        case 0x0983U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x098AU: rMask = 0xF7U; wMask = 0xF7U; break;
        case 0x098BU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x098DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x098EU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x098FU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0990U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0991U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0992U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0993U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0994U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x09ADU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x09B3U: rMask = 0x1FU; wMask = 0x1FU; break;
        case 0x09C3U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x09CAU: rMask = 0xF7U; wMask = 0xF7U; break;
        case 0x09CBU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x09CDU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x09CEU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x09CFU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x09D0U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x09D1U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x09D2U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x09D3U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x09D4U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x09EDU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x09F3U: rMask = 0x1FU; wMask = 0x1FU; break;
        case 0x0A0BU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A0DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A0EU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A0FU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A10U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A11U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A12U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A2DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A4BU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A4DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A4EU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A4FU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A50U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A51U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A52U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A6DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A8BU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A8DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A8EU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A8FU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A90U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A91U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0A92U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0AADU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0ACBU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0ACDU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0ACEU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0ACFU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0AD0U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0AD1U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0AD2U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0AEDU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0B04U: rMask = 0x39U; wMask = 0x39U; break;
        case 0x0B06U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0B07U: rMask = 0xEDU; wMask = 0xEDU; break;
        case 0x0B08U: rMask = 0xF7U; wMask = 0xF7U; break;
        case 0x0B0DU: rMask = 0x84U; wMask = 0x84U; break;
        case 0x0B0FU: rMask = 0x79U; wMask = 0x79U; break;
        case 0x0B14U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0B15U: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x0B19U: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x0BCBU: rMask = 0x01U; wMask = 0x00U; break;
        case 0x0C04U: rMask = 0x39U; wMask = 0x39U; break;
        case 0x0C06U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0C07U: rMask = 0xEDU; wMask = 0xEDU; break;
        case 0x0C08U: rMask = 0xF7U; wMask = 0xF7U; break;
        case 0x0C0DU: rMask = 0x84U; wMask = 0x84U; break;
        case 0x0C0FU: rMask = 0x79U; wMask = 0x79U; break;
        case 0x0C14U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0C15U: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x0C19U: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x0CCBU: rMask = 0x01U; wMask = 0x00U; break;
        case 0x0D04U: rMask = 0x39U; wMask = 0x39U; break;
        case 0x0D06U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0D07U: rMask = 0xEDU; wMask = 0xEDU; break;
        case 0x0D08U: rMask = 0xF7U; wMask = 0xF7U; break;
        case 0x0D0DU: rMask = 0x84U; wMask = 0x84U; break;
        case 0x0D0FU: rMask = 0x79U; wMask = 0x79U; break;
        case 0x0D14U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0D15U: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x0D19U: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x0DCBU: rMask = 0x01U; wMask = 0x00U; break;
        case 0x0E04U: rMask = 0x39U; wMask = 0x39U; break;
        case 0x0E06U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0E07U: rMask = 0xEDU; wMask = 0xEDU; break;
        case 0x0E08U: rMask = 0xF7U; wMask = 0xF7U; break;
        case 0x0E0DU: rMask = 0x84U; wMask = 0x84U; break;
        case 0x0E0FU: rMask = 0x79U; wMask = 0x79U; break;
        case 0x0E14U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0E15U: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x0E19U: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x0ECBU: rMask = 0x01U; wMask = 0x00U; break;
        case 0x1050U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1051U: rMask = 0xB1U; wMask = 0xB1U; break;
        case 0x1052U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1053U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1054U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1055U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1056U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1057U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1058U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1059U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x105AU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x105BU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x105CU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x105DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x105EU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x105FU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1060U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1061U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1062U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1063U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1064U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1065U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1066U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1067U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1068U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1069U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x106AU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x106BU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x106CU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x106DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x106EU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x106FU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1070U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1071U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1072U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1073U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1074U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1075U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1076U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x11D0U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x11D1U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x11D2U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x11E0U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x11E1U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x11E2U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x11E3U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x11E4U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x11E5U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x11E6U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x11E7U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x11E8U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x11E9U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x11EAU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x11EBU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x11ECU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x1250U: rMask = 0xFFU; wMask = 0xFCU; break;
        /* These below registers are not documented in datasheet
         * - 0x142E, 0x152E, 0x162E, 0x172E
         * but they are configured only in DES Rev2 and Rev3.
         * DES Rev5 is being used on P3710 and P3663. So, putting mask 0x00
         */
        case 0x142EU: rMask = 0x00U; wMask = 0x00U; break;
        /* These below registers are not documented in datasheet
         * - 0x1432, 0x1532, 0x1632, 0x1732
         * but they are configured only in DES Rev2 and Rev3.
         * DES Rev5 is being used on P3710 and P3663. So, putting mask 0x00
         */
        case 0x1432U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x143EU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x143FU: rMask = 0xFFU; wMask = 0xFFU; break;
        /* As per datasheet, "Set bits [2:0] = 3'b000 in registers 0x1445, 0x1545,
         * 0x1645, 0x1745"
         * [7:6] and [2:0] are write/read */
        case 0x1445U: rMask = 0xC7U; wMask = 0xC7U; break;
        /* These below registers are not documented in datasheet
         * - 0x1457, 0x1557, 0x1657, 0x1757
         * but they are configured only in DES Rev2 and Rev3.
         * DES Rev5 is being used on P3710 and P3663. So, putting mask 0x00
         */
        case 0x1457U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x1458U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x1459U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x1495U: rMask = 0x80U; wMask = 0x80U; break;
        case 0x14C4U: rMask = 0x04U; wMask = 0x04U; break;
        case 0x14C5U: rMask = 0xFFU; wMask = 0xFFU; break;
        /* As per datasheet, "Set bits [7:0] = 0x03 in registers 0x14D1, 0x15D1,
         * 0x16D1, 0x17D1."
         * All bits write/read */
        case 0x14D1U: rMask = 0xFFU; wMask = 0xFFU; break;
        /* These below registers are not documented in datasheet
         * - 0x142E, 0x152E 0x162E, 0x172E
         * but they are configured only in DES Rev2 and Rev3.
         * DES Rev5 is being used on P3710 and P3663. So, putting mask 0x00
         */
        case 0x152EU: rMask = 0x00U; wMask = 0x00U; break;
        /* These below registers are not documented in datasheet
         * - 0x1432, 0x1532, 0x1632, 0x1732
         * but they are configured only in DES Rev2 and Rev3.
         * DES Rev5 is being used on P3710 and P3663. So, putting mask 0x00
         */
        case 0x1532U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x153EU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x153FU: rMask = 0xFFU; wMask = 0xFFU; break;
        /* As per datasheet, "Set bits [2:0] = 3'b000 in registers 0x1445, 0x1545,
         * 0x1645, 0x1745"
         * [7:6] and [2:0] are write/read */
        case 0x1545U: rMask = 0xC7U; wMask = 0xC7U; break;
        /* These below registers are not documented in datasheet
         * 0x1457, 0x1557, 0x1657, 0x1757
         * but they are configured only in DES Rev2 and Rev3.
         * DES Rev5 is being used on P3710 and P3663. So, putting mask 0x00
         */
        case 0x1557U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x1558U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x1559U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x1595U: rMask = 0x80U; wMask = 0x80U; break;
        case 0x15C4U: rMask = 0x04U; wMask = 0x04U; break;
        case 0x15C5U: rMask = 0xFFU; wMask = 0xFFU; break;
        /* As per datasheet, "Set bits [7:0] = 0x03 in registers 0x14D1, 0x15D1,
         * 0x16D1, 0x17D1."
         * All bits write/read */
        case 0x15D1U: rMask = 0xFFU; wMask = 0xFFU; break;
        /* These below registers are not documented in datasheet
         * - 0x142E, 0x152E, 0x162E, 0x172E
         * but they are configured only in DES Rev2 and Rev3.
         * DES Rev5 is being used on P3710 and P3663. So, putting mask 0x00
         */
        case 0x162EU: rMask = 0x00U; wMask = 0x00U; break;
        /* These below registers are not documented in datasheet
         * - 0x1432, 0x1532, 0x1632, 0x1732
         * but they are configured only in DES Rev2 and Rev3.
         * DES Rev5 is being used on P3710 and P3663. So, putting mask 0x00
         */
        case 0x1632U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x163EU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x163FU: rMask = 0xFFU; wMask = 0xFFU; break;
        /* As per datasheet, "Set bits [2:0] = 3'b000 in registers 0x1445, 0x1545,
         * 0x1645, 0x1745"
         * [7:6] and [2:0] are write/read */
        case 0x1645U: rMask = 0xC7U; wMask = 0xC7U; break;
        /* These below registers are not documented in datasheet
         * - 0x1457, 0x1557, 0x1657, 0x1757
         * but they are configured only in DES Rev2 and Rev3.
         * DES Rev5 is being used on P3710 and P3663. So, putting mask 0x00
         */
        case 0x1657U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x1658U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x1659U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x1695U: rMask = 0x80U; wMask = 0x80U; break;
        case 0x16C4U: rMask = 0x04U; wMask = 0x04U; break;
        case 0x16C5U: rMask = 0xFFU; wMask = 0xFFU; break;
        /* As per datasheet, "Set bits [7:0] = 0x03 in registers 0x14D1, 0x15D1,
         * 0x16D1, 0x17D1."
         * All bits write/read */
        case 0x16D1U: rMask = 0xFFU; wMask = 0xFFU; break;
        /* These below registers are not documented in datasheet
         * - 0x142E, 0x152E, 0x162E, 0x172E
         * but they are configured only in DES Rev2 and Rev3.
         * DES Rev5 is being used on P3710 and P3663. So, putting mask 0x00
         */
        case 0x172EU: rMask = 0x00U; wMask = 0x00U; break;
        /* These below registers are not documented in datasheet
         * - 0x1432, 0x1532, 0x1632, 0x1732
         * but they are configured only in DES Rev2 and Rev3.
         * DES Rev5 is being used on P3710 and P3663. So, putting mask 0x00
         */
        case 0x1732U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x173EU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x173FU: rMask = 0xFFU; wMask = 0xFFU; break;
        /* As per datasheet, "Set bits [2:0] = 3'b000 in registers 0x1445, 0x1545,
         * 0x1645, 0x1745"
         * [7:6] and [2:0] are write/read */
        case 0x1745U: rMask = 0xC7U; wMask = 0xC7U; break;
        /* These below registers are not documented in datasheet
         * - 0x1457, 0x1557, 0x1657, 0x1757
         * but they are configured only in DES Rev2 and Rev3.
         * DES Rev5 is being used on P3710 and P3663. So, putting mask 0x00
         */
        case 0x1757U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x1758U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x1759U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x1795U: rMask = 0x80U; wMask = 0x80U; break;
        case 0x17C4U: rMask = 0x04U; wMask = 0x04U; break;
        case 0x17C5U: rMask = 0xFFU; wMask = 0xFFU; break;
        /* As per datasheet, "Set bits [7:0] = 0x03 in registers 0x14D1, 0x15D1,
         * 0x16D1, 0x17D1."
         * All bits write/read */
        case 0x17D1U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1C00U: rMask = 0x01U; wMask = 0x01U; break;
        case 0x1D00U: rMask = 0x01U; wMask = 0x01U; break;
        case 0x1E00U: rMask = 0x01U; wMask = 0x01U; break;
        case 0x1F00U: rMask = 0x01U; wMask = 0x01U; break;

        default:
            SIPL_LOG_ERR_STR_HEX_UINT("Missing RegMasks for register : ", address);
            status = NVMEDIA_STATUS_ERROR;
            break;
    }

    regMasks->maskRead = rMask;
    regMasks->maskWrite = wMask;

    return status;
}

/**
 * @brief Returns device driver handle from handle
 *
 * - Verifies handle is not NULL - returns NULL if verification fails
 * - returns device driver handle if not NULL
 * @param[in] handle  DEVBLK handle pointer to be verified
 * @return device driver handle, or NULL if fails */
static DriverHandleMAX96712 *
getHandlePrivMAX96712(DevBlkCDIDevice const* handle)
{
    DriverHandleMAX96712 *ret = NULL;

    if (NULL != handle) {
       /* coverity[misra_c_2012_rule_11_5_violation] : intentional TID-1417 */
        ret = (DriverHandleMAX96712 *)handle->deviceDriverHandle;
    }
    return ret;
}

/**
 * @brief verify the data passed by the user by reading again.
 *
 * @param[in]  i2cProgrammer  I2C Programmer handle
 * @param[in]  address  Register address
 * @param[in]  data  Expected register data
 * @param[in]  mask  Register Mask
 *
 * @retval NVMEDIA_STATUS_OK if verification was succesful.
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
static inline NvMediaStatus MAX96712ReadVerify(
    DevBlkCDII2CPgmr const i2cProgrammer,
    uint16_t const address,
    uint8_t const data,
    uint8_t const mask)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    if (mask != 0x00U) {
        uint8_t readData = 0U;
        status = DevBlkCDII2CPgmrReadUint8(i2cProgrammer, address, &readData);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("DevBlkCDII2CPgmrReadUint8 failed with ", (uint32_t)status);
        } else {
            readData ^= data;
            readData &= mask;
            if (readData > 0U) {
                SIPL_LOG_ERR_STR_HEX_UINT("Masked verification failed for reg", address);
                status = NVMEDIA_STATUS_ERROR;
            }
        }
    }
    return status;
}

/**
 * @brief Write Uint8 data and readback to verify
 *
 * @param[in]  handle  CDI Device Block Handle
 * @param[in]  address  Register address
 * @param[in]  data  Data to be written
 *
 * @retval NVMEDIA_STATUS_OK if succesful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if invalid parameter
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
static inline NvMediaStatus MAX96712WriteUint8Verify(
    DevBlkCDIDevice const * const handle,
    uint16_t const address,
    uint8_t const data)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const * const drvHandle = getHandlePrivMAX96712(handle);
    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer, address, data);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("DevBlkCDII2CPgmrWriteUint8 failed with ", (uint32_t)status);
        } else {
            MAX96712RegMask regMasks;
            status = GetRegReadWriteMasks(address, &regMasks);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT("GetRegReadWriteMasks failed with ", (uint32_t)status);
            } else {
                status = MAX96712ReadVerify(
                    drvHandle->i2cProgrammer, address, data, regMasks.maskWrite);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR_HEX_UINT("MAX96712ReadVerify failed with ", (uint32_t)status);
                }
            }
        }
    }
    return status;
}

/**
 * @brief Write an array of Uint8 data and readback to verify
 *
 * @param[in]  handle  CDI Device Block Handle
 * @param[in]  regList  List of register address and data
 *
 * @retval NVMEDIA_STATUS_OK if succesful.
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
static inline NvMediaStatus MAX96712WriteArrayVerify(
    DevBlkCDIDevice const * const handle,
    DevBlkCDII2CRegList const *regList)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, regList);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_HEX_UINT("DevBlkCDII2CPgmrWriteArray failed with ", (uint32_t)status);
    } else {
        if (regList->numRegs >= MAX_REGS_ARRAY_SIZE) {
            SIPL_LOG_ERR_STR("Invalid array size");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
        } else {
            DevBlkCDII2CReg readRegsList[MAX_REGS_ARRAY_SIZE] = {0};
            DevBlkCDII2CRegListWritable readVerifyRegList = {
                .regs = readRegsList,
                .numRegs = regList->numRegs
            };

            /* Copy register address to DevBlkCDII2CRegListWritable reg list */
            for (uint32_t i = 0; i < regList->numRegs; i++) {
                readVerifyRegList.regs[i].address = regList->regs[i].address;
                readVerifyRegList.regs[i].delayUsec = 0;
            }

            status = DevBlkCDII2CPgmrReadArray(drvHandle->i2cProgrammer, &readVerifyRegList);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT("DevBlkCDII2CPgmrReadArray failed with ", (uint32_t)status);
            } else {
                for (uint32_t i = 0; i < regList->numRegs; i++) {
                    MAX96712RegMask regMasks = {0};
                    /* Get the mask for registers */
                    status = GetRegReadWriteMasks(regList->regs[i].address, &regMasks);
                    if (status != NVMEDIA_STATUS_OK) {
                        SIPL_LOG_ERR_STR_HEX_UINT("Failed to get the mask for address ",
                                            (uint32_t)regList->regs[i].address);
                        break;
                    } else {
                        const uint8_t mask = regMasks.maskWrite;
                        /* Compare only if mask is not 0x00 */
                        if (mask != 0x00U) {
                            readVerifyRegList.regs[i].data ^= regList->regs[i].data;
                            readVerifyRegList.regs[i].data &= mask;
                            if (readVerifyRegList.regs[i].data > 0U) {
                                SIPL_LOG_ERR_STR_HEX_UINT("Failed to write verify for address ",
                                                        (uint32_t)regList->regs[i].address);
                                status = NVMEDIA_STATUS_ERROR;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    return status;
}

/**
 * @brief Read Uint8 data and readback to verify
 *
 * @param[in]  handle  CDI Device Block Handle
 * @param[in]  address  Register address
 * @param[out] data  Buffer to store read data
 *
 * @retval NVMEDIA_STATUS_OK if succesful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if invalid paramater
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
static inline NvMediaStatus MAX96712ReadUint8Verify(
    DevBlkCDIDevice const * const handle,
    uint16_t const address,
    uint8_t *data)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const * const drvHandle = getHandlePrivMAX96712(handle);
    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, address, data);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("DevBlkCDII2CPgmrReadUint8 failed with ", (uint32_t)status);
        } else {
            MAX96712RegMask regMasks;

            status = GetRegReadWriteMasks(address, &regMasks);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT("GetRegReadWriteMasks failed with ", (uint32_t)status);
            } else {
                status = MAX96712ReadVerify(
                    drvHandle->i2cProgrammer, address, *data, regMasks.maskRead);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR_HEX_UINT("MAX96712ReadVerify failed with ", (uint32_t)status);
                }
            }
        }
    }
    return status;
}

/**
 * @brief Read an array of Uint8 data and readback to verify
 *
 * @param[in]  handle  CDI Device Block Handle
 * @param[out] regList  List of register address and writable data
 *
 * @retval NVMEDIA_STATUS_OK if succesful.
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
static inline NvMediaStatus MAX96712ReadArrayVerify(
    DevBlkCDIDevice const * const handle,
    DevBlkCDII2CRegListWritable const *regList)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const * const drvHandle = getHandlePrivMAX96712(handle);

    status = DevBlkCDII2CPgmrReadArray(drvHandle->i2cProgrammer, regList);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_HEX_UINT("DevBlkCDII2CPgmrWriteArray failed with ", (uint32_t)status);
    } else {
        if (regList->numRegs >= MAX_REGS_ARRAY_SIZE) {
            SIPL_LOG_ERR_STR("Invalid array size");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
        } else {
            DevBlkCDII2CReg readRegsList[MAX_REGS_ARRAY_SIZE] = {0};
            DevBlkCDII2CRegListWritable readVerifyRegList = {
                .regs = readRegsList,
                .numRegs = regList->numRegs
            };

            /* Copy register address to DevBlkCDII2CRegListWritable reg list */
            for (uint32_t i = 0; i < regList->numRegs; i++) {
                readVerifyRegList.regs[i].address = regList->regs[i].address;
                readVerifyRegList.regs[i].delayUsec = 0;
            }

            status = DevBlkCDII2CPgmrReadArray(drvHandle->i2cProgrammer, &readVerifyRegList);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT("DevBlkCDII2CPgmrReadArray failed with ", (uint32_t)status);
            } else {
                for (uint32_t i = 0; i < regList->numRegs; i++) {
                    MAX96712RegMask regMasks = {0};
                    /* Get the mask for registers */
                    status = GetRegReadWriteMasks(regList->regs[i].address, &regMasks);
                    if (status != NVMEDIA_STATUS_OK) {
                        SIPL_LOG_ERR_STR_HEX_UINT("Failed to get the mask for address ",
                                            (uint32_t)regList->regs[i].address);
                        break;
                    } else {
                        const uint8_t mask = regMasks.maskRead;
                        /* Compare only if mask is not 0x00 */
                        if (mask != 0x00U) {
                            readVerifyRegList.regs[i].data ^= regList->regs[i].data;
                            readVerifyRegList.regs[i].data &= mask;
                            if (readVerifyRegList.regs[i].data > 0U) {
                                SIPL_LOG_ERR_STR_HEX_UINT("Failed to read verify for address ",
                                                        (uint32_t)regList->regs[i].address);
                                status = NVMEDIA_STATUS_ERROR;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    return status;
}

NvMediaStatus
MAX96712ReadRegisterVerify(
    DevBlkCDIDevice const* handle,
    uint16_t registerAddr,
    uint8_t *regData)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);

    if ((NULL == drvHandle) || (NULL == regData)) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = MAX96712ReadUint8Verify(handle,
                                         registerAddr,
                                         regData);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_AND_UINT("MAX96712: Register I2C read failed with status",
                                            registerAddr, (uint32_t)status);
        }
    }

    return status;
}

/**
 * @brief Checks whether all register fields belong to the same register.
 *
 * - Verifies  that all RegBitFieldQMAX96712 bits in regBit array
 * - refers to same register, by comparing their register address
 * @param[in] regBit array of RegBitFieldQMAX96712 bits
 * @param[in] numFields number of bit field in array
 * @return boolean true if all in same register, false, otherwise */
static bool
IsSingleRegister(
    RegBitFieldQMAX96712 const *regBit,
    uint8_t const numFields)
{
    bool status = true;
    RegBitFieldProp const *regBitProp = NULL;
    uint16_t regAddr = 0U;
    uint8_t i;

    regBitProp = &regBitFieldProps[regBit->name[0]];
    regAddr = regBitProp->regAddr;

    for (i = 0U; i < numFields; i++) {
        regBitProp = &regBitFieldProps[regBit->name[i]];
        if (regBitProp->regAddr != regAddr) {
            status = false;
            break;
        }
    }

    return status;
}

/**
 * @brief Driver handle's regBitFieldQ.numRegBitFieldArgs
 * - Verifies handle is not NULL
 * - clears regBitFieldQ.numRegBitFieldArgs
 * @param[in] handle  DEVBLK handle pointer to be verified */
static void
ClearRegFieldQ(
    DevBlkCDIDevice const* handle)
{
    DriverHandleMAX96712 *drvHandle = getHandlePrivMAX96712(handle);

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
    } else {
        PrintLogMsg(LOG_LEVEL_DEBUG, "MAX96712: Clearing RegFieldQ\n");
        drvHandle->regBitFieldQ.numRegBitFieldArgs = 0U;
    }
}

/**
 * @brief Queues name/value for register operation
 *
 * - Verifies handle is not NULL
 * - Verifies driver handle is not NULL
 * - Vedrifies that queue array is not full
 * - Verifies name to be a valid name
 * - if all verification pass
 *  - saves name/value in next record and increment record count
 *  .
 * @param[in] handle  DEVBLK handle pointer to be verified
 * @param[in] name name of parameter
 * @param[in] val
 * @return NVMEDIA_STATUS_OK id sucessfull operation, otherwise, an error code */
static NvMediaStatus
AddToRegFieldQ(
    DevBlkCDIDevice const* handle,
    RegBitField name,
    uint8_t val)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 *drvHandle = getHandlePrivMAX96712(handle);

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        uint8_t index = drvHandle->regBitFieldQ.numRegBitFieldArgs;

        if (index == MAX96712_REG_MAX_FIELDS_PER_REG) {
            SIPL_LOG_ERR_STR_UINT("MAX96712: RegFieldQ full. Failed to add",
                                  (uint32_t)name);
            status = NVMEDIA_STATUS_ERROR;
        } else if (name >= REG_FIELD_MAX) {
            SIPL_LOG_ERR_STR_UINT("MAX96712: RegFieldQ name over max. Failed to add",
                                  (uint32_t)name);
            status = NVMEDIA_STATUS_ERROR;
        } else {
            PrintLogMsg(LOG_LEVEL_DEBUG, "MAX96712: Adding regField = %u, val = %u "
                                         "to index %u in RegFieldQ\n",
                        name,
                        val,
                        index);

            drvHandle->regBitFieldQ.name[index] = name;
            drvHandle->regBitFieldQ.val[index] = val;
            if (index <= (uint8_t)254U) {
                drvHandle->regBitFieldQ.numRegBitFieldArgs = index + (uint8_t)1U;
            } else {
                status = NVMEDIA_STATUS_BAD_PARAMETER;
                SIPL_LOG_ERR_STR_UINT("MAX96712: RegFieldQ Bad index",
                                      (uint32_t)index);
            }
        }
    }
    return status;
}

/**
 * @brief Reads specified deserializer register 10 times
 *
 * - performs a loop of reading deserializer register value by
 *  - status = DevBlkCDII2CPgmrReadUint8(
 *   - drvHandle->i2cProgrammer,
 *   - regAddr,
 *   - regData);
 *   .
 *  .
 * @param[in] drvHandle  driver handle pointer
 * @param[in] regAddr    register address
 * @param[in] regData    location for storing read data
 * @return NVMEDIA_STATUS_OK id sucessfull operation, otherwise, an error code */
static NvMediaStatus
sRegReadRegNTimes(
    DevBlkCDIDevice const *handle,
    uint16_t regAddr,
    uint8_t *regData)
{
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    uint8_t loop = 0U;

    for (loop = 0U; loop < 10U; loop++) {
        status = MAX96712ReadUint8Verify(handle, regAddr, regData);
        if (status == NVMEDIA_STATUS_OK) {
            break;
        }
        /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
        (void)nvsleep(10);
    }

    return status;
}

/**
 * @brief Writes specified deserializer register 10 times
 *
 * - performs a loop of writing deserializer register value by
 *  - status = DevBlkCDII2CPgmrWriteUint8(
 *   - drvHandle->i2cProgrammer,
 *   - regAddr,
 *   - regData)
 *   .
 *  .
 * @param[in] drvHandle  driver handle pointer
 * @param[in] regAddr    register address
 * @param[in] regData    data to be written
 * @return NVMEDIA_STATUS_OK id sucessfull operation, otherwise, an error code */
static NvMediaStatus
sRegWriteRegNTimes(
    DevBlkCDIDevice const *handle,
    uint16_t regAddr,
    uint8_t regData)
{
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    uint8_t loop = 0U;

    for (loop = 0U; loop < 10U; loop++) {
        status = MAX96712WriteUint8Verify(handle, regAddr, regData);
        if (status == NVMEDIA_STATUS_OK) {
            break;
        }
        /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
        (void)nvsleep(10);
    }

    return status;
}

/**
 * @brief Utility function for calling  short or long sleep api
 * - calls sleep function according to regBitProp->longDelay
 *   - long:   nvsleep(LONG_DELAY_VALUE)
 *   - short: nvsleep(SHORT_DELAY_VALUE)
 *   .
 * @param[in] regBitProp contains boolean field longDelay
 */
static void
sRegAccessDelay(
    RegBitFieldProp const *regBitProp)
{
    if (regBitProp->longDelay) {
        /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
        (void)nvsleep(5000);
    } else {
        /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
        (void)nvsleep(20);
    }
}

/**
 * @brief  Access register fields belong to a single register.
 *
 * - It performs WR and RDWR modes.
 * - This function performs all the set of queued bit fields request.
 * - for each,
 *   - verifies bit position validity
 *   - performs I2C register access based on operation mode it:
 *     - REG_READ_MODE: Register is read and specified field vals are unpacked into
 *       regBitFieldArg array.
 *     - REG_WRITE_MODE: Specified field vals from regBitFieldArg array are packed and
 *       written to register.
 *     - REG_READ_MOD_WRITE_MODE: Register is read, specified field vals in
 *       regBitFieldArg are modified
 *       and written to register
 *     .
 *   .
 * @param[in] handle  DEVBLK handle pointer
 * @param[in] mode operation mode
 * @return NVMEDIA_STATUS_OK if operation was sucessfull, otherwise error code. */
static NvMediaStatus
sSingleRegAccessRegFieldQ(
    DevBlkCDIDevice const *handle,
    RegBitFieldAccessMode mode)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    RegBitFieldQMAX96712 const* regBit = &(drvHandle->regBitFieldQ);
    uint8_t numFields = regBit->numRegBitFieldArgs;
    uint16_t regAddr = 0U;
    RegBitFieldProp const *regBitProp = NULL;
    uint8_t fieldMask = 0U;
    uint8_t regData = 0U;
    uint8_t i = 0U;
#if !NV_IS_SAFETY
    bool exitFlag = false;
#endif
    uint32_t shiftValue = 0U;
    uint32_t shiftValue1 = 0U;
    uint32_t msbPosTmp = 0U;

    regBitProp = &regBitFieldProps[regBit->name[0]];
    regAddr = regBitProp->regAddr;

    /* Check if msbPos and lsbPos are valid. */
    for (i = 0U; i < numFields; i++) {
        regBitProp = &regBitFieldProps[regBit->name[i]];
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
        if (regBitProp->lsbPos > regBitProp->msbPos) {
            SIPL_LOG_ERR_STR("MAX96712: Bad parameter");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            exitFlag = true;
            break;
        }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
    }

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    if (!exitFlag) {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
        if (mode == REG_READ_MOD_WRITE_MODE) {
            status = sRegReadRegNTimes(handle, regAddr, &regData);
        }

        if(status == NVMEDIA_STATUS_OK) {
            for (i = 0U; i < numFields; i++) {
                regBitProp = &regBitFieldProps[regBit->name[i]];
                msbPosTmp = (uint32_t)(regBitProp->msbPos);
                msbPosTmp += 1U;
                shiftValue = leftBitsShift(1U,(msbPosTmp));
                shiftValue1 = leftBitsShift(1U, (uint32_t)(regBitProp->lsbPos));
                if(shiftValue >= shiftValue1) {
                    fieldMask = toUint8FromUint32(shiftValue - shiftValue1);
                }

                /* Pack fieldVals for write*/
                regData &= ~fieldMask;
                regData |= (toUint8FromUint32(leftBitsShift(regBit->val[i],regBitProp->lsbPos)) &
                            fieldMask);
            }

            status = sRegWriteRegNTimes(handle, regAddr, regData);

            sRegAccessDelay(regBitProp);

            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_AND_UINT("MAX96712: Register I2C read failed with status",
                                              regAddr, (uint32_t)status);
            }
        }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

    return status;
}

/**
 * @brief Utility function that verifies all regBit entries
 * - Verifies that all entries access same register and
 *  - mode is either REG_WRITE_MODE or REG_READ_MOD_WRITE_MODE
 *  .
 * @param[in] mode       access mode
 * @param[in] regBit     array of RegBitFieldQMAX96712 records
 * @param[in] numFields  number of entries in regBit
 * @return true of is single op of wr/rdwr, otherwise, false */
static bool
isSingleRegisterAccess(
    RegBitFieldAccessMode mode,
    RegBitFieldQMAX96712 const* regBit,
    uint8_t numFields)
{
    return (IsSingleRegister(regBit, numFields) &&
            ((mode == REG_WRITE_MODE) ||
            (mode == REG_READ_MOD_WRITE_MODE)));
}

/**
 * @brief  Access register fields belong to a single register.
 * - extract uint8_t numFields = regBit->numRegBitFieldArgs
 * - if (numFields != 0) and isSingleRegisterAccess(mode, regBit, numFields) is true
 *  - execute register access by
 *   - status = sSingleRegAccessRegFieldQ(handle, mode)
 *   .
 *  .
 * - else
 *  - if (numFields != 0)
 *   - This function performs all the set of queued bit fields request.
 *   - for each field,
 *    - extract regAddr = regBitFieldProps[regBit->name[i]].regAddr
 *    - extract regBitProp = &regBitFieldProps[regBit->name[i]]
 *    - Note: verify all the supplied fields belongs to same register addr.
 *    - verify  msbPos and lsbPos are valid, by
 *     - ((regAddr == regBitProp->regAddr) &&
 *      - (regBitProp->lsbPos <= regBitProp->msbPos))
 *      .
 *     .
 *    - if ((mode == REG_READ_MODE) or (mode == REG_READ_MOD_WRITE_MODE)))
 *     - read the values from the register set, and delay by
 *      - status = sRegReadRegNTimes(drvHandle, regAddr, &regData)
 *      - sRegAccessDelay(regBitProp)
 *      .
 *     .
 *    - extract the following by
 *     - regBitProp = &regBitFieldProps[regBit->name[i]]
 *     - msbPosTmp = (uint32_t)(regBitProp->msbPos)
 *     - msbPosTmp += 1U
 *     - shiftValue = leftBitsShift(1U,msbPosTmp)
 *     - shiftValue1 = leftBitsShift(1U,regBitProp->lsbPos)
 *     .
 *    - if (mode == REG_READ_MODE)
 *     -  Unpack fieldVals by
 *      - regBit->val[i] = (((regData & fieldMask) >> (regBitProp->lsbPos &
 *                         ((uint8_t)0xFFU))) & (uint8_t)0xFFU)
 *      .
 *    - else
 *     - Pack fieldVals for write by
 *      - regData &= ~fieldMask
 *      - shiftValue = leftBitsShift(regBit->val[i],regBitProp->lsbPos)
 *      - regData |= ((toUint8FromUint32(shiftValue)) & fieldMask)
 *      - Write to device N times
 *      - status = sRegWriteRegNTimes(drvHandle, regAddr, regData)
 *      - sRegAccessDelay(regBitProp)
 *      .
 *    .
 *   .
 *  .
 * @param[in] handle  DEVBLK handle
 * @param[in] mode operation mode
 * @return NVMEDIA_STATUS_OK if operation was sucessfull, otherwise error code. */
static NvMediaStatus
sAccessRegFieldQ(
    DevBlkCDIDevice const* handle,
    RegBitFieldAccessMode mode)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 *drvHandle = getHandlePrivMAX96712(handle);
    RegBitFieldQMAX96712 *regBit = &(drvHandle->regBitFieldQ);
    uint8_t numFields = regBit->numRegBitFieldArgs;
    uint16_t regAddr =  0U;
    RegBitFieldProp const *regBitProp = NULL;
    uint8_t fieldMask = 0U;
    uint8_t regData = 0U;
    uint8_t i = 0U;
    uint32_t shiftValue = 0U;
    uint32_t shiftValue1 = 0U;
    uint32_t msbPosTmp = 0U;

    if (numFields == 0U) {
        PrintLogMsg(LOG_LEVEL_DEBUG, "MAX96712: Skipping sAccessRegFieldQ\n");
    }

    /*
     * use sSingleRegAccessRegFieldQ() if all register fields belong to
     * a single register
     */
    if ((numFields != 0U) && isSingleRegisterAccess(mode, regBit, numFields)) {
        status = sSingleRegAccessRegFieldQ(handle, mode);
    } else if ((numFields != 0U)) {
        /* Check if all the supplied fields belongs to same register addr.
        * Check if msbPos and lsbPos are valid. */
        for (i = 0U; i < numFields; i++) {
            regAddr = regBitFieldProps[regBit->name[i]].regAddr;
            regBitProp = &regBitFieldProps[regBit->name[i]];
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
            if ((regAddr != regBitProp->regAddr) ||
                (regBitProp->lsbPos > regBitProp->msbPos)) {
                SIPL_LOG_ERR_STR("MAX96712: Bad parameter");
                status = NVMEDIA_STATUS_BAD_PARAMETER;
                break;
            }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

            if ((mode == REG_READ_MODE) || (mode == REG_READ_MOD_WRITE_MODE)) {
                status = sRegReadRegNTimes(handle, regAddr, &regData);
                sRegAccessDelay(regBitProp);
            }

            if (status == NVMEDIA_STATUS_OK) {
                regBitProp = &regBitFieldProps[regBit->name[i]];
                msbPosTmp = (uint32_t)(regBitProp->msbPos);
                msbPosTmp += 1U;
                shiftValue = leftBitsShift(1U,msbPosTmp);
                shiftValue1 = leftBitsShift(1U,regBitProp->lsbPos);

                if(shiftValue >= shiftValue1) {
                    fieldMask = toUint8FromUint32(shiftValue - shiftValue1);

                }

                if (mode == REG_READ_MODE) {
                    /* Unpack fieldVals */
                    regBit->val[i] = (((regData & fieldMask) >> (regBitProp->lsbPos &
                                      ((uint8_t)0xFFU))) & (uint8_t)0xFFU);
                } else {
                    /* Pack fieldVals for write */
                    regData &= ~fieldMask;
                    shiftValue = leftBitsShift(regBit->val[i],regBitProp->lsbPos);
                    regData |= ((toUint8FromUint32(shiftValue)) & fieldMask);
                    status = sRegWriteRegNTimes(handle, regAddr, regData);
                    sRegAccessDelay(regBitProp);
                }
            }
            if (status != NVMEDIA_STATUS_OK) {
                break;
            }
        }
    } else {
            /*do nothing */
    }

    return status;
}

/**
 * @brief Helper function for calling AddToRegFieldQ
 * - verify no previous error occured
 * - add to queue by
 *  - status = AddToRegFieldQ(handle, name, val)
 *  .
 * @param[in] instatus previous status to be checked. Noting done if this is != NVMEDIA_STATUS_OK
 * @param[in] handle   DEVBLK handle
 * @param[in] name     name of parameter
 * @param[in] val
 * @return NVMEDIA_STATUS_OK if operation was sucessfull, otherwise error code. */
static NvMediaStatus
Max96712AddOneRegField(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle,
    RegBitField name,
    uint8_t val)
{
    NvMediaStatus status = instatus;

    if (status == NVMEDIA_STATUS_OK) {
        status = AddToRegFieldQ(handle,
                                (name),
                                (val));
    }
    return status;
}

/**
 * @brief Helper function for calling sAccessRegFieldQ
 * - verify no previous error occured
 * - access register by
 *  - status = sAccessRegFieldQ(handle, mode)
 *  .
 *
 * @param[in] instatus previous status to be checked. Noting done if this is != NVMEDIA_STATUS_OK
 * @param[in] handle   DEVBLK handle
 * @param[in] mode operation mode
 * @return NVMEDIA_STATUS_OK if operation was sucessfull, otherwise error code. */
static NvMediaStatus
Max96712AccessRegField(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle,
    RegBitFieldAccessMode mode)
{
    NvMediaStatus status = instatus;

    if (status == NVMEDIA_STATUS_OK) {
        status = sAccessRegFieldQ(handle,
                                  (mode));
    }
    return status;
}

/**
 * @brief Helper function for sending one field regField
 * - verify no previous error occured
 * - calls ClearRegFieldQ to clear queue
 * - calls sAccessRegFieldQ to queue single regField
 * - calls Max96712AccessRegField to perform I2c command
 * @param[in] instatus previous status to be checked. Noting done if this is != NVMEDIA_STATUS_OK
 * @param[in] handle   DEVBLK handle
 * @param[in] name  name of bitField
 * @param[in] val value to write
 * @param[in] mode operation mode
 * @return NVMEDIA_STATUS_OK if operation was sucessfull, otherwise error code. */
static NvMediaStatus
Max96712AccessOneRegField(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle,
    RegBitField name,
    uint8_t val,
    RegBitFieldAccessMode mode)
{
    NvMediaStatus status = instatus;

    if (status == NVMEDIA_STATUS_OK) {
        ClearRegFieldQ(handle);
        status = Max96712AddOneRegField(status, handle, name, val);
        if (status == NVMEDIA_STATUS_OK) {
            status = Max96712AccessRegField(status, handle, mode);
        }
    }
    return status;
}

/**
 * @brief Helper function for adding one field regField w/ reg offset
 * - verify no previous error occured
 * - Verifies name + offset is within range of REG_FIELD_MAX - fails if out of range by
 *  - if (((uint32_t)name + (uint32_t)offset) < (uint32_t)REG_FIELD_MAX)
 *  .
 * - calls Max96712AddOneRegField to queue single regField
 * @param[in] instatus previous status to be checked. Noting done if this is != NVMEDIA_STATUS_OK
 * @param[in] handle   DEVBLK handle
 * @param[in] name  base bitfield name
 * @param[in] offset  offset from reg name of bitField
 * @param[in] val value for bitField
 * @return NVMEDIA_STATUS_OK if operation was sucessfull, otherwise error code. */
static NvMediaStatus
Max96712AddRegFieldOffset(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle,
    RegBitField name,
    uint8_t offset,
    uint8_t val)
{
    NvMediaStatus status = instatus;

    if (status == NVMEDIA_STATUS_OK) {
        if (((uint32_t)name + (uint32_t)offset) < (uint32_t)REG_FIELD_MAX) {
            RegBitField temp = (uint32_t)name + (uint32_t)offset;
            status = Max96712AddOneRegField(status, handle, temp, val);
        } else {
            SIPL_LOG_ERR_STR("MAX96712: unexpected offset");
        }
    }
    return status;
}

/**
 * @brief Helper function for accessing one field regField w/ reg offset
 * - verify no previous error occured
 * - Verifies name + offset is within range of REG_FIELD_MAX - fails if out of range by
 *  - if (((uint32_t)name + (uint32_t)offset) < (uint32_t)REG_FIELD_MAX)
 *  .
 * - calls Max96712AccessOneRegField to queue single regField
 * @param[in] instatus previous status to be checked. Noting done if this is != NVMEDIA_STATUS_OK
 * @param[in] handle   DEVBLK handle
 * @param[in] name  base bitfield name
 * @param[in] offset  offset from reg name of bitField
 * @param[in] val value for bitField
 * @param[in] mode operation mode
 * @return NVMEDIA_STATUS_OK if operation was sucessfull, otherwise error code. */
static NvMediaStatus
Max96712AccessOneRegFieldOffset(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle,
    RegBitField name,
    uint8_t offset,
    uint8_t val,
    RegBitFieldAccessMode mode)
{
    NvMediaStatus status = instatus;

    if (status == NVMEDIA_STATUS_OK) {
        if (((uint32_t)name + (uint32_t)offset) < (uint32_t)REG_FIELD_MAX) {
            RegBitField temp = (uint32_t)name + (uint32_t)offset;
            status = Max96712AccessOneRegField(status, handle, temp, val, mode);
        } else {
            SIPL_LOG_ERR_STR("MAX96712: unexpected offset");
        }
    }
    return status;
}

/**
 * @brief Adds a Pipeline error and regFieldVal info to pipeline array
 *
 * - sets next entry in array with error and regFieldVal by
 *  - errorStatus->pipelineFailureType[pipelineNum][*pipelineErrorCount] = (error)
 *  - errorStatus->pipelineRegVal[pipelineNum][*pipelineErrorCount] = regFieldVal
 *  - errorStatus->pipeline |= (uint8_t)((1U << (pipelineNum & 0xFU)) & 0xFFU)
 *  .
 * - increment errorStatus->count ifnot maxed out by
 *  - if (errorStatus->count < MAX96712_MAX_ERROR_STATUS_COUNT)
 *   - errorStatus->count++
 *   .
 *  -
 * - increment *pipelineErrorCount if last == false, and not overflowing the array by
 *  - if ((last== false) && (*pipelineErrorCount < MAX96712_MAX_PIPELINE_ERROR_NUM))
 *   - (*pipelineErrorCount)++
 *   .
 *  .
 * @param[in] errorStatus pointer to error array ErrorStatusMAX96712
 * @param[out] pipelineErrorCount pointer to pipeline error counter
 * @param[in] regFieldVal value with error
 * @param[in] pipelineNum pipeberline num
 * @param[in] error  error id PipelineFailureTypeMAX96712
 * @param[in] last true if this is the last error record
 * @return NVMEDIA_STATUS_OK if operation was sucessfull, otherwise error code. */
static void
Max96712UpdatePipelineError(
    ErrorStatusMAX96712 *errorStatus,
    uint8_t *pipelineErrorCount,
    uint8_t regFieldVal,
    uint8_t pipelineNum,
    PipelineFailureTypeMAX96712 error,
    bool last)
{
    errorStatus->pipelineFailureType[pipelineNum][*pipelineErrorCount] = (error);
    errorStatus->pipelineRegVal[pipelineNum][*pipelineErrorCount] = regFieldVal;
    errorStatus->pipeline |= (uint8_t)((1U << (pipelineNum & 0xFU)) & 0xFFU);

    if (errorStatus->count < MAX96712_MAX_ERROR_STATUS_COUNT) {
        errorStatus->count++;
    } else {
        SIPL_LOG_ERR_STR("MAX96712: error count over max");
    }

    if ((last== false) && (*pipelineErrorCount < MAX96712_MAX_PIPELINE_ERROR_NUM)) {
        (*pipelineErrorCount)++;
    }
}

/**
 * @brief Adds a link error and regFieldVal info to link array
 * - sets next entry in array with error and regFieldVal by
 *  - errorStatus->linkFailureType[linkNum][*linkErrorCount] = (error)
 *  - errorStatus->linkRegVal[linkNum][*linkErrorCount] = regFieldVal
 *  - errorStatus->link |= (uint8_t)((1U << (linkNum & 0xFU)) & 0xFFU)
 *  .
 * - increment errorStatus->count if not maxed out by
 *  - if (errorStatus->count < MAX96712_MAX_ERROR_STATUS_COUNT)
 *   - errorStatus->count++
 *   .
 *  -
 * - increment *linkErrorCount if last == false, and not overflowing the array by
 *  - if ((last == false) && (*linkErrorCount < MAX96712_MAX_LINK_BASED_ERROR_NUM))
 *   - (*linkErrorCount)++
 *   .
 *  .
 * @param[in] errorStatus pointer to error array ErrorStatusMAX96712
 * @param[out] linkErrorCount pointer to link error counter
 * @param[in] regFieldVal value with error
 * @param[in] linkNum link number
 * @param[in] error  error id LinkFailureTypeMAX96712
 * @param[in] last true if this is the last error record
 * @return NVMEDIA_STATUS_OK if operation was sucessfull, otherwise error code. */

static void
Max96712UpdateLinkError(
    ErrorStatusMAX96712 *errorStatus,
    uint8_t *linkErrorCount,
    uint8_t regFieldVal,
    uint8_t linkNum,
    LinkFailureTypeMAX96712 error,
    bool last)
{
    errorStatus->linkFailureType[linkNum][*linkErrorCount] = (error);
    errorStatus->linkRegVal[linkNum][*linkErrorCount] = regFieldVal;
    errorStatus->link |= (uint8_t)((1U << (linkNum & 0xFU)) & 0xFFU);

    if (errorStatus->count < MAX96712_MAX_ERROR_STATUS_COUNT) {
        errorStatus->count++;
    } else {
        SIPL_LOG_ERR_STR("MAX96712: error count over max");
    }

    if ((last == false) && (*linkErrorCount < MAX96712_MAX_LINK_BASED_ERROR_NUM)) {
        (*linkErrorCount)++;
    }
}

/**
 * @brief Adds a global error and regFieldVal info to global array
 * - sets next entry in array with error and regFieldVal by
 *  - errorStatus->globalFailureType[*globalErrorCount] = (error)
 *  - errorStatus->globalRegVal[*globalErrorCount] = regFieldVal
 *  .
 * - increment errorStatus->count if not maxed out by
 *  - if (errorStatus->count < MAX96712_MAX_ERROR_STATUS_COUNT)
 *   - errorStatus->count++
 *   .
 *  -
 * - increment *globalErrorCount if last == false, and not overflowing the array by
 *  - if ((last == false) && (*linkErrorCount < MAX96712_MAX_GLOBAL_ERROR_NUM))
 *   - (*globalErrorCount)++
 *   .
 *  .
 * @param[in] errorStatus pointer to error array ErrorStatusMAX96712
 * @param[out] globalErrorCount pointer to link error counter
 * @param[in] regFieldVal value with error
 * @param[in] error  error id GlobalFailureTypeMAX96712
 * @param[in] last true if this is the last error record
 * @return NVMEDIA_STATUS_OK if operation was sucessfull, otherwise error code. */
static void
Max96712UpdateGlobalError(
    ErrorStatusMAX96712 *errorStatus,
    uint8_t *globalErrorCount,
    uint8_t regFieldVal,
    GlobalFailureTypeMAX96712 error,
    bool last)
{
    errorStatus->globalFailureType[*globalErrorCount] = (error);
    errorStatus->globalRegVal[*globalErrorCount] = regFieldVal;

    if (errorStatus->count < MAX96712_MAX_ERROR_STATUS_COUNT) {
        errorStatus->count++;
    } else {
        SIPL_LOG_ERR_STR("MAX96712: error count over max");
    }

    if ((last == false) && (*globalErrorCount < MAX96712_MAX_GLOBAL_ERROR_NUM)) {
        (*globalErrorCount)++;
    }
}

/**
 * @brief Verifies that mode is GMSK2 mode
 * - verify by
 *  - if ((mode == CDI_MAX96712_GMSL2_MODE_6GBPS) ||
 *   - (mode == CDI_MAX96712_GMSL2_MODE_3GBPS))
 *   .
 *  .
 * @param[in] mode GMSL mode
 * @return true if mode is GMSL2 mode, otherwise false */
static bool
IsGMSL2Mode(GMSLModeMAX96712 const mode)
{
    bool status = false;

    if ((mode == CDI_MAX96712_GMSL2_MODE_6GBPS) ||
        (mode == CDI_MAX96712_GMSL2_MODE_3GBPS)) {
        status = true;
    }
    return status;
}

/**
 * @brief Reads value of specified index into bitReg queue
 * - extracts device driver handle from handle
 * - Verifies handle and device driver handle are not NULL
 * - verify index is within array range by
 *  - (index < drvHandle->regBitFieldQ.numRegBitFieldArgs)
 *  .
 * - reads value from queue by
 *  - val = drvHandle->regBitFieldQ.val[index]
 *  .
 * @param[in] handle   DEVBLK handle
 * @param[in] index  index into bitReg array
 * @return value at index, or 1 if invalid index, or 0 if handles are NULL */
static uint8_t
ReadFromRegFieldQ(
    DevBlkCDIDevice const* handle,
    uint8_t index)
{
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    uint8_t val = 0U;

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        val = (uint8_t)NVMEDIA_STATUS_BAD_PARAMETER;
    }else if (index >= drvHandle->regBitFieldQ.numRegBitFieldArgs) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: Bad parameter. Invalid index",
                              index);
    } else {
        val = drvHandle->regBitFieldQ.val[index];

        PrintLogMsg(LOG_LEVEL_DEBUG, "MAX96712: Read index %u from RegFieldQ. "
                                     "Val = %u\n",
                    index,
                    val);
    }
    return val;
}

/**
 * @brief Returns link bit mask for specified link
 *
 * - Verifies val is not NULL
 * - if link is CDI_MAX96712_LINK_NONE -    return 0x00
 * - elif link is CDI_MAX96712_LINK_0 -     return 0x01
 * - elif link is CDI_MAX96712_LINK_1 -     return 0x02
 * - elif link is CDI_MAX96712_LINK_2 -     return 0x04
 * - elif link is CDI_MAX96712_LINK_3 -     return 0x08
 * - elif link is CDI_MAX96712_LINK_0_1_3 - return 0xb
 * - elif link is CDI_MAX96712_LINK_ALL -   return 0x0f
 * - else return error "MAX96712: Bad parameter. Invalid link"
 * @param[in] link ling id
 * @param[out] val pointor for storing bit mask value
 * @return NVMEDIA_STATUS_OK if done, otherwise error code */
static NvMediaStatus
GetMAX96712LinkVal(LinkMAX96712 const link, uint8_t *val)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (val == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter. NULL ptr");
    } else {
        switch (link) {
            case CDI_MAX96712_LINK_NONE:
                *val = 0U;
                break;
            case CDI_MAX96712_LINK_0:
                *val = 1U;
                break;
            case CDI_MAX96712_LINK_1:
                *val = 2U;
                break;
            case CDI_MAX96712_LINK_2:
                *val = 4U;
                break;
            case CDI_MAX96712_LINK_3:
                *val = 8U;
                break;
            case CDI_MAX96712_LINK_0_1_3:
                *val = 11U;
                break;
            case CDI_MAX96712_LINK_ALL:
                *val = 15U;
                break;
            default:
                *val = 0U;
                status = NVMEDIA_STATUS_BAD_PARAMETER;
                SIPL_LOG_ERR_STR_UINT("MAX96712: Bad parameter. Invalid link",
                                      (uint32_t)link);
                break;
        }
    }

    return status;
}

/**
 * @brief Checks every locked link, that deserializer is in locked state
 * - extracts device driver handle from handle
 * - verifies not previous error occured
 * - For every link
 *  - if link bit is set in context linkHasBeenLocked
 *   - read deserializer REG_FIELD_GMSL2_LOCK_A register by
 *    - status = Max96712AccessOneRegFieldOffset(
 *     - status, handle,
 *     - REG_FIELD_GMSL2_LOCK_A,
 *     - linkIndex,
 *     - 0U, REG_READ_MODE)
 *     .
 *    . get read value and check if it is set by
 *     - if (ReadFromRegFieldQ(handle, 0U) == 1U)
 *     .
 *   - if it is set
 *    - set link bit as active in context activeLinkMask by
 *     - drvHandle->ctx.activeLinkMask |= MAX96712_LINK_SHIFT(0x1U, linkIndex)
 *     .
 *    .
 *   - else
 *    - clearlink bit as off in context linkHasBeenLocked by
 *     - drvHandle->ctx.linkHasBeenLocked[linkIndex] = false
 *    .
 *   .
 *  .
 * @param[in] instatus previous status to be checked. Noting done if this is != NVMEDIA_STATUS_OK
 * @param[in] handle   DEVBLK handle
 * @return NVMEDIA_STATUS_OK if no error occur. otherwise, error code */
static NvMediaStatus
CheckAgainstActiveLink(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = instatus;
    DriverHandleMAX96712* drvHandle = getHandlePrivMAX96712(handle);
    drvHandle->ctx.activeLinkMask = 0U;
    uint8_t linkIndex = 0U;

    if (status == NVMEDIA_STATUS_OK) {
        for (linkIndex = 0U; linkIndex < MAX96712_MAX_NUM_LINK; linkIndex++) {
            if (drvHandle->ctx.linkHasBeenLocked[linkIndex]) {
                status = Max96712AccessOneRegFieldOffset(status, handle,
                                                         REG_FIELD_GMSL2_LOCK_A,
                                                         linkIndex,
                                                         0U, REG_READ_MODE);

                if (status == NVMEDIA_STATUS_OK) {
                    if (ReadFromRegFieldQ(handle, 0U) == 1U)  {
                        drvHandle->ctx.activeLinkMask |= MAX96712_LINK_SHIFT(0x1U, linkIndex);
                    } else {
                        drvHandle->ctx.linkHasBeenLocked[linkIndex] = false;
                    }
                } else {
                    PrintLogMsg(LOG_LEVEL_NONE,"MAX96712: Link %u: GMSL2 link is no longer "
                                "accessible!\n", linkIndex);
                    status = NVMEDIA_STATUS_ERROR;
                }
            }
        }

        PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: mask of active link is "
                                    "0x%x\n", drvHandle->ctx.activeLinkMask);
    }

    return status;
}

NvMediaStatus
MAX96712OneShotReset(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if ((link > CDI_MAX96712_LINK_NONE) && (link <= CDI_MAX96712_LINK_ALL))
    {
        status = Max96712AccessOneRegField(status, handle, REG_FIELD_RESET_ONESHOT, (uint8_t)link,
            REG_WRITE_MODE);
        if (status == NVMEDIA_STATUS_OK) {
            /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
            (void)nvsleep(100000);
        }
    } else {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }
    return status;
}

/**
 * @brief Enables specified links
 *
 * - Verifies that parameter size matches size of link
 * - clears bitReg queue by
 *  - ClearRegFieldQ(handle)
 *  .
 * - for each link
 *  - enables/disables each link based on its mask in "link"
 *   - by adding setting to bitReg queue, by
 *    - status = Max96712AddRegFieldOffset(
 *     - status, handle, REG_FIELD_LINK_EN_A, i,
 *     - (uint8_t)(MAX96712_IS_GMSL_LINK_SET(link, i) ?
 *      - 1U : 0U))
 *      .
 *    .
 *  .
 * - write entry to deserializer REG_FIELD_LINK_EN_A register by
 *  - status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE)
 *  .
 * @param[in] handle DEVBLK handle
 * @param[in] link   bit mask of links to enable/disable
 * @param[in] parameterSize size in bytes of register to be written
 * @param[out] isValidSize set to true if parameterSize was valid, false otherwise
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
EnableSpecificLinks(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link,
    size_t parameterSize,
    bool *isValidSize,
    bool checkLink)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 *drvHandle = getHandlePrivMAX96712(handle);

    if ((handle == NULL) || (isValidSize == NULL) || (drvHandle == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (parameterSize != sizeof(link)) {
       *isValidSize = false;
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        uint8_t i = 0U;
        *isValidSize = true;
        drvHandle->ctx.linkMask = link;

        /* Disable the link lock error report to avoid the false alarm */
        ClearRegFieldQ(handle);
        status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_ENABLE_LOCK, 0U, 0U);
        status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE);

        ClearRegFieldQ(handle);
        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_LINK_EN_A, i,
                                               (uint8_t)(MAX96712_IS_GMSL_LINK_SET(link, i) ?
                                               1U : 0U));
        }
        status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE);

        /* Make sure the link is locked properly before enabling the link lock signal */
        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                if (status == NVMEDIA_STATUS_OK) {
                    if (drvHandle->ctx.gmslMode[i] == CDI_MAX96712_GMSL1_MODE) {
                        /* HIM mode is not enabled yet so the link lock will not be set
                         * Instead use sleep function */
                        /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
                        (void)nvsleep(100000);
                    } else if (checkLink) {
                        status = MAX96712CheckLink(handle,
                                     link,
                                     CDI_MAX96712_LINK_LOCK_GMSL2,
                                     true);
                    }
                }
            }
        }

        /* Enable the link lock error report */
        if (drvHandle->ctx.linkMask != CDI_MAX96712_LINK_NONE) {
            ClearRegFieldQ(handle);
            status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_ENABLE_LOCK, 0U, 1U);
            status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE);
        }
    }
    return status;
}

/**
 * @brief Sets up all links mode
 *
 * - Verifies not previous failure occured
 * - Clears bitReg queue by
 *  - ClearRegFieldQ(handle)
 *  .
 * - for all links
 *  - if (MAX96712_IS_GMSL_LINK_SET(link, i))
 *   - Adds mode settings for REG_FIELD_LINK_GMSL2_A based on context specified link mode by
 *    - status = Max96712AddRegFieldOffset(
 *     - status, handle, REG_FIELD_LINK_GMSL2_A, i,
 *     - (uint8_t)((drvHandle->ctx.gmslMode[i] == CDI_MAX96712_GMSL1_MODE) ?
 *      - 0U : 1U))
 *      .
 *     .
 *    .
 *   .
 *  .
 * - write queue to deserializer's register by
 *  - status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE)
 *  .
 * - Clears bitReg queue by
 *  - ClearRegFieldQ(handle)
 *  .
 * - for all links
 *  - if (MAX96712_IS_GMSL_LINK_SET(link, i))
 *   - Note:
 *    - CDI_MAX96712_GMSL1_MODE     : 1
 *    - CDI_MAX96712_GMSL2_MODE_6GBPS : 2
 *    - CDI_MAX96712_GMSL2_MODE_3GBPS : 1
 *    .
 *   - Adds mode settings for REG_FIELD_RX_RATE_PHY_A based on context specified GMSL mode by
 *    - status = Max96712AddRegFieldOffset(
 *     - status, handle, REG_FIELD_RX_RATE_PHY_A, i,
 *     - (uint8_t)((drvHandle->ctx.gmslMode[i] == CDI_MAX96712_GMSL2_MODE_6GBPS) ?
 *      - 2U : 1U))
 *      .
 *     .
 *    .
 *   .
 *  .
 * - write queue to deserializer's register by
 *  - status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE)
 *  .
 * @param[in] instatus previous status to be checked. Noting done if this is != NVMEDIA_STATUS_OK
 * @param[in] handle DEVBLK handle
 * @param[in] link   bit mask of links to setup
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
SetLinkMode(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link)
{
    NvMediaStatus status = instatus;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    uint8_t i = 0U;

    if (status == NVMEDIA_STATUS_OK) {
        /* Set GMSL mode */
        ClearRegFieldQ(handle);
        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                PrintLogMsg(LOG_LEVEL_DEBUG, "SetLinkMode: Setting Link for %d\n", i);
                status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_LINK_GMSL2_A, i,
                                                   (uint8_t)((drvHandle->ctx.gmslMode[i] ==
                                                              CDI_MAX96712_GMSL1_MODE) ? 0U : 1U));
            }
        }
        status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE);

        /* Set Link speed */
        ClearRegFieldQ(handle);
        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                PrintLogMsg(LOG_LEVEL_DEBUG, "SetLinkMode: Link is set and now "
                                             "setting the speed for %i\n", i);
                /*CDI_MAX96712_GMSL1_MODE     : 1
                CDI_MAX96712_GMSL2_MODE_6GBPS : 2
                CDI_MAX96712_GMSL2_MODE_3GBPS : 1*/
                status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_RX_RATE_PHY_A, i,
                                                   (uint8_t)((drvHandle->ctx.gmslMode[i] ==
                                                    CDI_MAX96712_GMSL2_MODE_6GBPS) ? 2U : 1U));
            }
        }
        status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE);
    }

    return status;
}

/**
 * @brief Enables period AEQ for specified links
 *
 * - Verifies that parameter size matches size of link
 * - for each link
 *  - if (MAX96712_IS_GMSL_LINK_SET(link, i))
 *   - clears bitReg queue by
 *    - ClearRegFieldQ(handle)
 *    .
 *   - enables REG_FIELD_AEQ_PHY_A by adding setting to bitReg queue by
 *    - status = Max96712AddRegFieldOffset(
 *     - status, handle, REG_FIELD_AEQ_PHY_A, i, 1U)
 *     .
 *    .
 *   - enables REG_FIELD_PERIODIC_AEQ_PHY_A by adding setting to bitReg queue by
 *    - status = Max96712AddRegFieldOffset(
 *     - status, handle, REG_FIELD_PERIODIC_AEQ_PHY_A, i, 1U)
 *     .
 *    .
 * 0  - enables REG_FIELD_EOM_PER_THR_PHY_A by adding setting to bitReg queue by
 *    - status = Max96712AddRegFieldOffset(
 *     - status, handle, REG_FIELD_EOM_PER_THR_PHY_A, i, 0x10U)
 *     .
 *    .
 *   - write entry to deserializer REG_FIELD_LINK_EN_A register by
 *    - status = Max96712AccessRegField(status, handle, REG_WRITE_MODE)
 *    .
 *   - delay for 10 msec by
 *    - (void)nvsleep(10000)
 *    .
 *   .
 *  .
 *
 * @param[in] handle DEVBLK handle
 * @param[in] link   bit mask of links to process
 * @param[in] parameterSize size in bytes of register to be written
 * @param[out] isValidSize set to true if parameterSize was valid, false otherwise
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
EnablePeriodicAEQ(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link,
    size_t parameterSize,
    bool *isValidSize)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    if (parameterSize != sizeof(link)) {
       *isValidSize = false;
       status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
       uint8_t i = 0U;
       *isValidSize = true;
       for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
           if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
               ClearRegFieldQ(handle);
               status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_AEQ_PHY_A, i,
                                      1U);
               status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_PERIODIC_AEQ_PHY_A, i,
                                      1U);
               status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_EOM_PER_THR_PHY_A, i,
                                      0x10U);
               status = Max96712AccessRegField(status, handle, REG_WRITE_MODE);

               PrintLogMsg(LOG_LEVEL_NONE,"MAX96712: Enable periodic AEQ on Link %d\n", i);
               /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
               (void)nvsleep(10000);
           }
       }
   }
   return status;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
static NvMediaStatus
SetDefaultGMSL1HIMEnabled(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link,
    uint8_t step)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        DevBlkCDII2CReg max96712_defaults_HIM_step0_regs[] = {
            /* GMSL1 - Turn on HIM */
            {0x0B06U, 0xEFU},
            /* GMSL1 - Enable reverse channel cfg and turn on local I2C ack */
            {0x0B0DU, 0x81U},
        };
        DevBlkCDII2CRegList max96712_defaults_HIM_step0 = {
            .regs = max96712_defaults_HIM_step0_regs,
            .numRegs = (uint32_t)(sizeof(max96712_defaults_HIM_step0_regs) /
                                sizeof(max96712_defaults_HIM_step0_regs[0])),
        };
        DevBlkCDII2CReg max96712_defaults_HIM_step1_regs[] = {
            /* GMSL1 - Turn off HIM */
            {0x0B06U, 0x6FU},
            /* GMSL1 - Enable manual override of reverse channel pulse
             * length. */
            {0x14C5U, 0xAAU},
            /* GMSL1 - Enable manual override of reverse channel rise fall
             * time setting */
            {0x14C4U, 0x80U},
            /* GMSL1 - Tx amplitude manual override */
            {0x1495U, 0xC8U},
        };
        DevBlkCDII2CRegList max96712_defaults_HIM_step1 = {
            .regs = max96712_defaults_HIM_step1_regs,
            .numRegs = (uint32_t)(sizeof(max96712_defaults_HIM_step1_regs) /
                                sizeof(max96712_defaults_HIM_step1_regs[0])),
        };
        DevBlkCDII2CReg max96712_defaults_HIM_step2_regs[] = {
            /* Enable HIM */
            {0x0B06U, 0xEFU},
            /* Manual override of reverse channel pulse length */
            {0x14C5U, 0x40U},
            /* Manual override of reverse channel rise fall time setting */
            {0x14C4U, 0x40U},
            /* TxAmp manual override */
            {0x1495U, 0x69U},
        };
        DevBlkCDII2CRegList max96712_defaults_HIM_step2 = {
            .regs = max96712_defaults_HIM_step2_regs,
            .numRegs = (uint32_t)(sizeof(max96712_defaults_HIM_step2_regs) /
                                sizeof(max96712_defaults_HIM_step2_regs[0])),
        };
        DevBlkCDII2CRegList const* stepHIM = NULL;
        uint16_t i = 0U;

        if (step > 2U) {
            SIPL_LOG_ERR_STR("MAX96712: Bad parameter. Step must be either 0, 1 or 2.");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
        } else {
            for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
                if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                    /* Update register offset */
                    if (step == 0U) {
                        max96712_defaults_HIM_step0_regs[0].address += (i << 8U);
                        max96712_defaults_HIM_step0_regs[1].address += (i << 8U);
                    } else if (step == 1U) {
                        max96712_defaults_HIM_step1_regs[0].address += (i << 8U);
                        max96712_defaults_HIM_step1_regs[1].address += (i << 8U);
                        max96712_defaults_HIM_step1_regs[2].address += (i << 8U);
                        max96712_defaults_HIM_step1_regs[3].address += (i << 8U);
                    } else {
                        max96712_defaults_HIM_step2_regs[0].address += (i << 8U);
                        max96712_defaults_HIM_step2_regs[1].address += (i << 8U);
                        max96712_defaults_HIM_step2_regs[2].address += (i << 8U);
                        max96712_defaults_HIM_step2_regs[3].address += (i << 8U);
                    }

                    stepHIM = (step == 0U) ? &max96712_defaults_HIM_step0 :
                              ((step == 1U) ? &max96712_defaults_HIM_step1 :
                                              &max96712_defaults_HIM_step2);

                    status = MAX96712WriteArrayVerify(handle, stepHIM);
                    if (status != NVMEDIA_STATUS_OK) {
                        break;
                    }
                }
            }
        }
    }
    return status;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
static NvMediaStatus
EnablePacketBasedControlChannel(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link,
    bool enable,
    size_t parameterSize,
    bool *isValidSize)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    DevBlkCDII2CReg ctrlChannelReg = {0x0B08U, 0x25U};
    uint8_t i = 0U;
    if ((NULL == drvHandle) || (parameterSize != sizeof(link))) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        *isValidSize = false;
    } else {
        *isValidSize = true;
        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                ctrlChannelReg.address += ((uint16_t)i) << 8U;

                if (!enable) {
                    ctrlChannelReg.data = 0x21U;
                }
                status = MAX96712WriteUint8Verify(
                                         handle,
                                         ctrlChannelReg.address,
                                         toUint8FromUint16(ctrlChannelReg.data));
                if (status != NVMEDIA_STATUS_OK) {
                    break;
                }
                /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
                (void)nvsleep(10000);
            }
        }
    }
    return status;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/**
 * @brief Writes 0x01 to REG_FIELD_BPP8DBL_0 register
 *
 * - Verifies not previous failure occured
 * - Clears bitReg queue by
 *  - ClearRegFieldQ(handle)
 *  .
 * - for each link
 *  - if (MAX96712_IS_GMSL_LINK_SET(link, i))
 *   - Adds data 0x01  settings for REG_FIELD_BPP8DBL_0 by
 *    - status = Max96712AddRegFieldOffset(
 *     - status, handle, REG_FIELD_BPP8DBL_0, i, 1U)
 *     .
 *    .
 *   -
 *  - write queue to deserializer's register by
 *   - status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE)
 *   .
 *  .
 *
 * @param[in] instatus previous status to be checked. Noting done if this is != NVMEDIA_STATUS_OK
 * @param[in] handle DEVBLK handle
 * @param[in] link   bit mask of links to setup
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
Max96712WriteRegFieldBpp8Dbl0(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link)
{
    NvMediaStatus status = instatus;
    uint8_t i = 0U;

    if (status == NVMEDIA_STATUS_OK) {
        ClearRegFieldQ(handle);
        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_BPP8DBL_0, i,
                                                   1U);
            }
        }
        status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE);
    }
    return status;
}

/**
 * @brief Writes 0x01 to REG_FIELD_BPP8DBL_MODE_0 register
 * - Verifies not previous failure occured
 * - Clears bitReg queue by
 *  - ClearRegFieldQ(handle)
 *  .
 * - for each link
 *  - if (MAX96712_IS_GMSL_LINK_SET(link, i))
 *   - Adds data 0x01  settings for REG_FIELD_BPP8DBL_MODE_0 by
 *    - status = Max96712AddRegFieldOffset(
 *     - status, handle, REG_FIELD_BPP8DBL_MODE_0, i, 1U)
 *     .
 *    .
 *   .
 *  .
 * - write queue to deserializer's register by
 *  - status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE)
 *  .
 * @param[in] instatus previous status to be checked. Noting done if this is != NVMEDIA_STATUS_OK
 * @param[in] handle DEVBLK handle
 * @param[in] link   bit mask of links to setup
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
Max96712WriteRegFieldBpp8DblMode0(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link)
{
    NvMediaStatus status = instatus;
    uint8_t i = 0U;

    if (status == NVMEDIA_STATUS_OK) {
        ClearRegFieldQ(handle);
        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_BPP8DBL_MODE_0, i,
                                                   1U);
            }
        }
        status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE);
    }
    return status;
}

/**
 * @brief Writes 0x01 to REG_FIELD_BPP8DBL_4 register
 *
 * - Verifies not previous failure occured
 * - Clears bitReg queue by
 *  - ClearRegFieldQ(handle)
 *  .
 * - for each link
 *  - if (MAX96712_IS_GMSL_LINK_SET(link, i))
 *   - Adds data 0x01  settings for REG_FIELD_BPP8DBL_4 by
 *    - status = Max96712AddRegFieldOffset(
 *     - status, handle, REG_FIELD_BPP8DBL_4, i, 1U)
 *     .
 *    .
 *   .
 * - write queue to deserializer's register by
 *  - status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE)
 *  .
 * @param[in] instatus previous status to be checked. Noting done if this is != NVMEDIA_STATUS_OK
 * @param[in] handle DEVBLK handle
 * @param[in] link   bit mask of links to setup
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
Max96712WriteRegFieldBpp8Dbl4(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link)
{
    NvMediaStatus status = instatus;
    uint8_t i = 0U;

    if (status == NVMEDIA_STATUS_OK) {
        ClearRegFieldQ(handle);
        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_BPP8DBL_4, i,
                                                   1U);
            }
        }
        status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE);
    }
    return status;
}

/**
 * @brief Writes 0x01 to REG_FIELD_BPP8DBL_MODE_4 register
 *
 * - Verifies not previous failure occured
 * - Clears bitReg queue by
 *  - ClearRegFieldQ(handle)
 *  .
 * - for each link
 *  - if (MAX96712_IS_GMSL_LINK_SET(link, i))
 *   - Adds data 0x01  settings for REG_FIELD_BPP8DBL_MODE_4 by
 *    - status = Max96712AddRegFieldOffset(
 *     - status, handle, REG_FIELD_BPP8DBL_MODE_4, i, 1U)
 *     .
 *    .
 *   .
 * - write queue to deserializer's register by
 *  - status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE)
 *  .
 * @param[in] instatus previous status to be checked. Noting done if this is != NVMEDIA_STATUS_OK
 * @param[in] handle DEVBLK handle
 * @param[in] link   bit mask of links to setup
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
Max96712WriteRegFieldBpp8DblMode4(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link)
{
    NvMediaStatus status = instatus;
    uint8_t i = 0U;

    if (status == NVMEDIA_STATUS_OK) {
        ClearRegFieldQ(handle);
        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_BPP8DBL_MODE_4, i,
                                                   1U);
            }
        }
        status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE);
    }
    return status;
}

/**
 * @brief Sets up alternative Memory Map
 *
 * - Verifies not previous failure occured
 * - extract tx port from context to be used as an offset for setting bitReg queue entries by
 *  - uint8_t txPort = (uint8_t)(((uint32_t)drvHandle->ctx.txPort) & 0xFFU)
 *  .
 * - Clears bitReg queue by
 *  - ClearRegFieldQ(handle)
 *  .
 * - if (dataType == CDI_MAX96712_DATA_TYPE_RAW12) and (isSharedPipeline is true)
 *  - if embDataType is true
 *   - Note: In cases where EMB and Pix data share the same pipeline enable ALT2_MEM_MAP8
 *   - sets REG_FIELD_ALT2_MEM_MAP8_PHY0[tx port] to 0x01 by
 *    - status = Max96712AddRegFieldOffset(
 *     - status, handle, REG_FIELD_ALT2_MEM_MAP8_PHY0, txPort, 1U)
 *     .
 *    .
 *   .
 *  - else
 *   - sets REG_FIELD_ALT_MEM_MAP12_PHY0[tx port] to 0x01 by
 *    -
 *   .
 *  .
 * - else
 *  - if dataType == CDI_MAX96712_DATA_TYPE_RAW12
 *   - sets REG_FIELD_ALT_MEM_MAP12_PHY0[tx port] to 0x01 by
 *    - status = Max96712AddRegFieldOffset(
 *     - status, handle, REG_FIELD_ALT_MEM_MAP12_PHY0, txPort, 1U)
 *     .
 *   .
 *  - elif  dataType == CDI_MAX96712_DATA_TYPE_RAW10
 *   - sets REG_FIELD_ALT_MEM_MAP10_PHY0[tx port] to 0x01 by
 *    - status = Max96712AddRegFieldOffset(
 *     - status, handle, REG_FIELD_ALT_MEM_MAP10_PHY0, txPort, 1U)
 *     .
 *    .
 *   .
 *  - elif dataType == CDI_MAX96712_DATA_TYPE_RAW8
 *   - sets REG_FIELD_ALT_MEM_MAP8_PHY0[tx port] to 0x01 by
 *    - status = Max96712AddRegFieldOffset(
 *     - status, handle, REG_FIELD_ALT_MEM_MAP8_PHY0, txPort, 1U)
 *     .
 *    .
 *   .
 *  - if embDataType is true
 *   - sets REG_FIELD_ALT_MEM_MAP8_PHY0[tx port] to 0x01 by
 *    - status = Max96712AddRegFieldOffset(
 *     - status, handle, REG_FIELD_ALT_MEM_MAP8_PHY0, txPort, 1U)
 *     .
 *    .
 *   .
 *  .
 * - write queue to deserializer's register
 * @param[in] instatus previous status to be checked. Noting done if this is != NVMEDIA_STATUS_OK
 * @param[in] handle DEVBLK handle
 * @param[in] dataType data type
 * @param[in] embDataType embedded data type if true
 * @param[in] isSharedPipeline  set true if shared pipeline
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
Max96712WriteRegFieldAltMemMap(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle,
    DataTypeMAX96712 dataType,
    bool const embDataType)
{
    NvMediaStatus status = instatus;

    for (uint8_t txPort = 0U; txPort < MAX96712_MAX_NUM_PHY; txPort++) {
        if (status == NVMEDIA_STATUS_OK) {
            ClearRegFieldQ(handle);
            if (dataType == CDI_MAX96712_DATA_TYPE_RAW12) {
                if (embDataType) {
                    /* In cases where EMB and Pix data share
                     * the same pipeline enable ALT2_MEM_MAP8 */
                    status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_ALT2_MEM_MAP8_PHY0,
                                                       txPort, 1U);
                } else {
                    status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_ALT_MEM_MAP12_PHY0,
                                                       txPort, 1U);
                }
            } else if (dataType == CDI_MAX96712_DATA_TYPE_RAW10) {
                if (embDataType) {
                    /* In cases where EMB and Pix data share
                     * the same pipeline enable ALT2_MEM_MAP8 */
                    status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_ALT2_MEM_MAP8_PHY0,
                                                       txPort, 1U);
                } else {
                    status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_ALT_MEM_MAP10_PHY0,
                                                       txPort, 1U);
                }
            } else if (dataType == CDI_MAX96712_DATA_TYPE_RAW8) {
                if (embDataType) {
                    /* In cases where EMB and Pix data share
                     * the same pipeline enable ALT2_MEM_MAP8 */
                    status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_ALT2_MEM_MAP8_PHY0,
                                                       txPort, 1U);
                } else {
                    status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_ALT_MEM_MAP8_PHY0,
                                                       txPort, 1U);
                }
            }

            status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE);
        }
    }

    return status;
}

/**
 * @brief Sets up SOMETHING ??????? packet detector
 *
 * - Verifies not previous failure occured
 * - initialize DevBlkCDII2CReg structure with
 *  - DevBlkCDII2CReg disPktDetectorRegs[] = {
 *   - Note: VIDEO_RX0 SEQ_MISS_EN = 1, DIS_PKT_DET = 0
 *   - {0x0100U, 0x12U},
 *   - Note: VIDEO_RX6 LIM_HEART = 1 : DisableVIDEO_RX6 LIM_HEART = 1 : Disable
 *   - 0x0106U, 0x0AU},
 *   .
 *  - }
 *  .
 * - initialize wrapper for disPktDetectorRegs by
 *  - DevBlkCDII2CRegList disPktDetectorArr = {
 *   - .regs = disPktDetectorRegs,
 *   - .numRegs = (uint32_t)(sizeof(disPktDetectorRegs) /
 *    - sizeof(disPktDetectorRegs[0])),
 *    .
 *   .
 *  - }
 * - for all links
 *  - if MAX96712_IS_GMSL_LINK_SET in link mask
 *   - set disPktDetectorRegs[0].address =
        ((disPktDetectorRegs[0].address & 0xFF00U) + (0x12U * i));
 *   - set disPktDetectorRegs[1].address =
         ((disPktDetectorRegs[1].address & 0xFF00U) + 0x06U + (0x12U * i));
 *   - if isSharedPipeline true
 *    - Note: VIDEO_RX0 SEQ_MISS_EN = 1, DIS_PKT_DET = 0
 *    - set data to be disPktDetectorRegs[0].data = 0x23U
 *    .
 *   - else
 *    - Note: Video only :  DIS_PKT_DET = 0, SEQ_MISS_EN = 0
 *    - set data to be disPktDetectorRegs[0].data = 0x22U
 *    .
 *   - note: writes value to register
 *   - write value to link specific register by
 *    - status = DevBlkCDII2CPgmrWriteArray(
 *     - drvHandle->i2cProgrammer, &disPktDetectorArr)
 *     .
 *    .
 *   - set disPktDetectorRegs[0].address =
 *    - (disPktDetectorRegs[0].address & 0xFF00U) +
 *     - (0x12U * (i + 4U)) + ((i != 0U) ? 0x6U : 0U)
 *     .
 *    .
 *   - Note: EMB only :  DIS_PKT_DET = 0, SEQ_MISS_EN = 1
 *    - disPktDetectorRegs[0].data = 0x12U
 *    .
 *   - write value to link specific register by
 *    -  status = DevBlkCDII2CPgmrWriteUint8(
 *     - drvHandle->i2cProgrammer,
 *     - disPktDetectorRegs[0].address,
 *     - (uint8_t)disPktDetectorRegs[0].data)
 *     .
 *    .
 *   .
 *  .
 *
 * @param[in] instatus previous status to be checked. Noting done if this is != NVMEDIA_STATUS_OK
 * @param[in] handle DEVBLK handle
 * @param[in] link   bit mask of links to setup
 * @param[in] isSharedPipeline boolean true if shared pipeline, otherwise false
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
Max96712WriteRegFieldDisPktDetector(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link,
    bool const embDataType)
{
    NvMediaStatus status = instatus;

    /* VIDEO_RX0 SEQ_MISS_EN = 1, LINE_CRC_EN = 1, DIS_PKT_DET = 0 (SM11) */
    DevBlkCDII2CReg disPktDetectorRegs[] = {
        {0x0100U, 0x32U},
        {0x0112U, 0x32U},
        {0x0124U, 0x32U},
        {0x0136U, 0x32U},
        {0x0148U, 0x32U},
        {0x0160U, 0x32U},
        {0x0172U, 0x32U},
        {0x0184U, 0x32U},
    };

    /* VIDEO_RX6 LIM_HEART = 0 : Enable */
    DevBlkCDII2CReg disLIMHeartRegs[] = {
        {0x0106U, 0x02U},
        {0x0118U, 0x02U},
        {0x012AU, 0x02U},
        {0x013CU, 0x02U},
        {0x014EU, 0x02U},
        {0x0166U, 0x02U},
        {0x0178U, 0x02U},
        {0x018AU, 0x02U},
    };

    uint8_t i = 0U;

    if (status == NVMEDIA_STATUS_OK) {
       for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
           if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
               if (embDataType == true) {
                   disPktDetectorRegs[i].data = 0x23U;
               }
               status =  MAX96712WriteUint8Verify(handle,
                            disPktDetectorRegs[i].address,
                            toUint8FromUint16(disPktDetectorRegs[i].data));
               if (status == NVMEDIA_STATUS_OK) {
                   status =  MAX96712WriteUint8Verify(handle,
                                disPktDetectorRegs[i + 4U].address,
                                toUint8FromUint16(disPktDetectorRegs[i + 4U].data));
               }

               if ((status == NVMEDIA_STATUS_OK) && (embDataType == false)) {
                   status =  MAX96712WriteUint8Verify(handle,
                                disLIMHeartRegs[i].address,
                                toUint8FromUint16(disLIMHeartRegs[i].data));
                   if (status == NVMEDIA_STATUS_OK) {
                       status =  MAX96712WriteUint8Verify(handle,
                                    disLIMHeartRegs[i + 4U].address,
                                    toUint8FromUint16(disLIMHeartRegs[i + 4U].data));
                   }
               }
               /* Common check for both the DevBlkCDII2CPgmrWriteUint8 calls,
                  fresh check, not "else" */
               if (status != NVMEDIA_STATUS_OK) {
                   break;
               }
           }
       }
    }
    return status;
}

/**
 * @brief Enables double pixel mode
 *
 * - if data type is CDI_MAX96712_DATA_TYPE_RAW8
 *  - call Max96712WriteRegFieldBpp8Dbl0
 *  - call Max96712WriteRegFieldBpp8DblMode0
 *  .
 * - call Max96712WriteRegFieldBpp8Dbl4
 * - call Max96712WriteRegFieldBpp8DblMode4
 * - call Max96712WriteRegFieldAltMemMap
 * @param[in] handle dev BLK handle
 * @param[in] link link id
 * @param[in] dataType data type
 * @param[in] embDataType embedded data type
 * @param[in] isSharedPipeline boolean true if shared pipe line
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
EnableDoublePixelMode(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link,
    DataTypeMAX96712 dataType,
    bool const embDataType,
    bool isSharedPipeline)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    /* Only touch BPP8DBL_0* for RAW8 format, it seems to negatively impact other formats */
    if(dataType == CDI_MAX96712_DATA_TYPE_RAW8) {
        status = Max96712WriteRegFieldBpp8Dbl0(status, handle, link);
        status = Max96712WriteRegFieldBpp8DblMode0(status, handle, link);
        status = Max96712WriteRegFieldBpp8Dbl4(status, handle, link);
        status = Max96712WriteRegFieldBpp8DblMode4(status, handle, link);
    }

    status = Max96712WriteRegFieldAltMemMap(status,
                                            handle,
                                            dataType,
                                            embDataType);

    status = Max96712WriteRegFieldDisPktDetector(status, handle, link, embDataType);

    return status;
}

/**
 * @brief Configures TX amp timing
 * Bug 2182451: The below errors were observed intermittently in GMSL2 6Gbps link speed.
 * To resolve it, adjust the Tx amplitude and timing parameters
 * CSI error(short or long line) is seen Decoding error is seen on the deserializer
 * Link margin becomes bad
 *
 * - Note: Configure the set of commands (address, data, delay) to be sent to deserializer:
 * - Set DevBlkCDII2CReg adjTxAmpAndTimingArrRegs[5] = {
 *  - Note: vth1 : Error channel power-down then delay 1ms
 *  - {0x1458U, 0x28U, 0x2701U},
 *  - Note: vth0 : + 104 * 4.7mV = 488.8 mV  then delay 1ms
 *  - {0x1459U, 0x68U, 0x2701U},
 *  - Note: Error channel phase secondary timing adjustment  then delay 1ms
 *  - {0x143EU, 0xB3U, 0x2701U},
 *  - Note: Error channel phase primary timing adjustment  then delay 1ms
 *  - {0x143FU, 0x72U, 0x2701U},
 *  - Note: Reverse channel Tx amplitude to 180 mV  then delay 1ms
 *  - {0x1495U, 0xD2U, 0x2701U},
 *  .
 * - }
 * - set wrapper structure for adjTxAmpAndTimingArrRegs by
 *  - DevBlkCDII2CRegList adjTxAmpAndTimingArr = {
 *   - .regs = adjTxAmpAndTimingArrRegs,
 *   - .numRegs = (uint32_t)(sizeof(adjTxAmpAndTimingArrRegs) /
 *    - sizeof(adjTxAmpAndTimingArrRegs[0])),
 *    .
 *   .
 *  - }
 *  .
 * - for each link
 *  - if MAX96712_IS_GMSL_LINK_SET in link mask
 *   - Note: calculate address for specific link to be set
 *   - for (j = 0U; j < adjTxAmpAndTimingArr.numRegs; j++) {
 *    - adjTxAmpAndTimingArrRegs[j].address += ((uint16_t)i) << 8U;
 *    .
 *   - }
 *   .
 *  - write set of commands to deserializer calling DevBlkCDII2CPgmrWriteArray by
 *   -  status = DevBlkCDII2CPgmrWriteArray(
 *    - drvHandle->i2cProgrammer,
 *    - &adjTxAmpAndTimingArr)
 *    .
 *   .
 *  .
 * @param[in] handle DEVBLK handle
 * @param[in] link   bit mask of links to setup
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
ConfigTxAmpTiming(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    GMSLModeMAX96712 gmslMode = CDI_MAX96712_GMSL_MODE_INVALID;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    RevisionMAX96712 rev = CDI_MAX96712_REV_INVALID;
    DevBlkCDII2CReg adjTxAmpAndTimingArrRegs[] = {
        /* vth1 : Error channel power-down then delay 1ms*/
        {0x1458U, 0x28U, 0x2701U},
        /* vth0 : + 104 * 4.7mV = 488.8 mV  then delay 1ms*/
        {0x1459U, 0x68U, 0x2701U},
        /* Error channel phase secondary timing adjustment then delay 1ms*/
        {0x143EU, 0xB3U, 0x2701U},
        /* Error channel phase primary timing adjustment then delay 1ms*/
        {0x143FU, 0x72U, 0x2701U},
        /* Reverse channel Tx amplitude to 180 mV  then delay 1ms*/
        {0x1495U, 0xD2U, 0x2701U},
    };
    DevBlkCDII2CRegList adjTxAmpAndTimingArr = {
        .regs = adjTxAmpAndTimingArrRegs,
        .numRegs = (uint32_t)(sizeof(adjTxAmpAndTimingArrRegs) /
                              sizeof(adjTxAmpAndTimingArrRegs[0])),
    };
    uint8_t i = 0U;

    rev = drvHandle->ctx.revision;
    for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
            gmslMode = drvHandle->ctx.gmslMode[i];
            if (!IsGMSL2Mode(gmslMode)) {
                PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: Link %d: Tx amplitude is only required "
                                            "in GMSL2 mode\n", i);
                continue;
            }
 /* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

            adjTxAmpAndTimingArrRegs[0].address += ((uint16_t)i) << 8U;
            adjTxAmpAndTimingArrRegs[1].address += ((uint16_t)i) << 8U;
            adjTxAmpAndTimingArrRegs[2].address += ((uint16_t)i) << 8U;
            adjTxAmpAndTimingArrRegs[3].address += ((uint16_t)i) << 8U;
            adjTxAmpAndTimingArrRegs[4].address += ((uint16_t)i) << 8U;
            status =  MAX96712WriteArrayVerify(handle, &adjTxAmpAndTimingArr);

            if (status != NVMEDIA_STATUS_OK) {
                PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: Link %d: Failed to updte Tx amplitude\n",
                            i);
                break;
            }
            (void)rev;
            PrintLogMsg(LOG_LEVEL_NONE,"MAX96712 Rev %d: Link %d: ", rev, i);
            PrintLogMsg(LOG_LEVEL_NONE,"Tx amplitude 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x\n",
                        adjTxAmpAndTimingArrRegs[0].data,
                        adjTxAmpAndTimingArrRegs[1].data,
                        adjTxAmpAndTimingArrRegs[2].data,
                        adjTxAmpAndTimingArrRegs[3].data,
                        adjTxAmpAndTimingArrRegs[4].data);
        }
    }
    return status;
}

/**
 * @brief Returns data type value & bpp value for datatype
 *
 * - returns 0x8U / 0x2AU  for CDI_MAX96712_DATA_TYPE_RAW8
 * - returns 0xAU / 0x2BU  for CDI_MAX96712_DATA_TYPE_RAW10
 * - returns 0xCU / 0x2CU  for CDI_MAX96712_DATA_TYPE_RAW12
 * - returns 0x10U / 0x2EU for CDI_MAX96712_DATA_TYPE_RAW16
 * - returns 0x18U / 0x24U for CDI_MAX96712_DATA_TYPE_RGB
 * - returns 0x10U / 0x1EU for CDI_MAX96712_DATA_TYPE
 * - returns an error if none of the above NVMEDIA_STATUS_BAD_PARAMETER
 * @param[in] dataType requested data type settings
 * @param[out] dataTypeVal  data type value for dataType
 * @param[out] bpp bpp value for dataType
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
GetDataTypeValAndBpp(
       DataTypeMAX96712 dataType,
       uint8_t *dataTypeVal,
       uint8_t *bpp)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    switch (dataType) {
        case CDI_MAX96712_DATA_TYPE_RAW8:
            *bpp = 0x8U;          /* 8 bits per pixel */
            *dataTypeVal = 0x2AU;
            break;
        case CDI_MAX96712_DATA_TYPE_RAW10:
            *bpp = 0xAU;          /* 10 bits per pixel */
            *dataTypeVal = 0x2BU;
            break;
        case CDI_MAX96712_DATA_TYPE_RAW12:
            *bpp = 0xCU;          /* 12 bits per pixel */
            *dataTypeVal = 0x2CU;
            break;
        case CDI_MAX96712_DATA_TYPE_RAW16:
            *bpp = 0x10U;         /* 16 bits per pixel */
            *dataTypeVal = 0x2EU;
            break;
        case CDI_MAX96712_DATA_TYPE_RGB:
            *bpp = 0x18U;         /* 24 bits per pixel */
            *dataTypeVal = 0x24U;
            break;
        case CDI_MAX96712_DATA_TYPE_YUV_8:
            *bpp = 0x10U;         /* 16 bits per pixel */
            *dataTypeVal = 0x1EU;
            break;
        case CDI_MAX96712_DATA_TYPE_YUV_10:
            *bpp = 0x14U;         /* 20 bits per pixel */
            *dataTypeVal = 0x1FU;
            break;
        default:
            SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Invalid data type");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }
    return status;
}

/**
 * @brief Overrides settings per linkPipelineMap[i].dataType for data format & bpp
 *
 * - verifies that paramSize is the size of pipelink_sz - fails if not same
 * - for each link
 *  - if  (MAX96712_IS_GMSL_LINK_SET is set in link mask) &&
 *   - (linkPipelineMap[i].isDTOverride is set)
 *    - call GetDataTypeValAndBpp to get settings (dataFormat, bpp) by
 *     - status = GetDataTypeValAndBpp(linkPipelineMap[i].dataType, &dataFormat, &bpp)
 *     .
 *    - read/mod/write bpp value to REG_FIELD_SOFT_BPP_0 register by
 *     - status = Max96712AccessOneRegFieldOffset(
 *      - status, handle, REG_FIELD_SOFT_BPP_0, i, bpp, REG_READ_MOD_WRITE_MODE)
 *      .
 *     .
 *    - read/mod/write  dataFormat to REG_FIELD_SOFT_DT_0 register by
 *     - status = Max96712AccessOneRegFieldOffset(
 *      - status, handle, REG_FIELD_SOFT_DT_0, i,
 *      - dataFormat, REG_READ_MOD_WRITE_MODE)
 *      .
 *     .
 *    - if link index == 1
 *     - write (dataFormat >> 4U)       to REG_FIELD_SOFT_DT_1_H register by
 *      - status = Max96712AccessOneRegField(
 *       - status, handle, REG_FIELD_SOFT_DT_1_H,
 *       - (dataFormat >> 4U), REG_READ_MOD_WRITE_MODE)
 *       .
 *      .
 *     .
 *    - elif link index == 2
 *     - write (bpp >> 2U)              to REG_FIELD_SOFT_BPP_2_H register by
 *      - status = Max96712AccessOneRegField(
 *       - status, handle, REG_FIELD_SOFT_BPP_2_H,
 *       - (bpp >> 2U), REG_READ_MOD_WRITE_MODE)
 *       .
 *      .
 *     - write (dataFormat >> 2U)       to REG_FIELD_SOFT_DT_2_H register by
 *      - status = Max96712AccessOneRegField(
 *       - (dataFormat >> 2U), REG_READ_MOD_WRITE_MODE)
 *       .
 *     .
 *    .
 *   - write 0x01 to REG_FIELD_SOFT_OVR_0_EN register by
 *    - status = Max96712AccessOneRegFieldOffset(
 *     - status, handle, REG_FIELD_SOFT_OVR_0_EN
 *     - i, 1U, REG_READ_MOD_WRITE_MODE)
 *     .
 *    .
 *   .
 *  - if tpg is Enabled in context, and pipeline[link + 4] is Enabled
 *   - if link index == 0
 *    - write bpp                  to REG_FIELD_SOFT_BPP_4 register by
 *     - status = Max96712AccessOneRegField(
 *      - status, handle, REG_FIELD_SOFT_BPP_4, bpp, REG_READ_MOD_WRITE_MODE)
 *      .
 *     .
 *    - write dataFormat           to REG_FIELD_SOFT_DT_4 register by
 *     - status = Max96712AccessOneRegField(
 *      - status, handle, REG_FIELD_SOFT_DT_4,
 *      - dataFormat, REG_READ_MOD_WRITE_MODE)
 *      .
 *     .
 *    .
 *   - elif link index == 1
 *    - write bpp                  to REG_FIELD_SOFT_BPP_5 register by
 *     - status = Max96712AccessOneRegField(
 *      - status, handle, REG_FIELD_SOFT_BPP_5, bpp, REG_READ_MOD_WRITE_MODE)
 *      .
 *     .
 *    - write (dataFormat >> 4U)   to REG_FIELD_SOFT_DT_5_H register by
 *     - status = Max96712AccessOneRegField(
 *      - status, handle, REG_FIELD_SOFT_DT_5_H,
 *      - (dataFormat >> 4U),  REG_READ_MOD_WRITE_MODE)
 *      .
 *     .
 *    - write (dataFormat & 0x0fU) to REG_FIELD_SOFT_DT_5_L register by
 *     - status = Max96712AccessOneRegField(
 *      - status, handle, REG_FIELD_SOFT_DT_5_H,
 *      - (dataFormat >> 4U), REG_READ_MOD_WRITE_MODE)
 *      .
 *     .
 *    .
 *   - else if link index == 2
 *    - write (bpp >> 2U)          to REG_FIELD_SOFT_BPP_6_H register by
 *     - status = Max96712AccessOneRegField(
 *      - status, handle, REG_FIELD_SOFT_BPP_6_H,
 *      - (bpp >> 2U), REG_READ_MOD_WRITE_MODE)
 *      .
 *     .
 *    - write (bpp & 3U)           to REG_FIELD_SOFT_BPP_6_L register by
 *     - status = Max96712AccessOneRegField(
 *      - status, handle, REG_FIELD_SOFT_BPP_6_L,
 *      - (bpp & 0x3U), REG_READ_MOD_WRITE_MODE)
 *      .
 *     .
 *    - write (dataFormat >> 2U)   to REG_FIELD_SOFT_DT_6_H register by
 *     - status = Max96712AccessOneRegField(
 *      - status, handle, REG_FIELD_SOFT_DT_6_H,
 *      - (dataFormat >> 2U), REG_READ_MOD_WRITE_MODE)
 *      .
 *     .
 *    - write (dataFormat & 0x03U) to REG_FIELD_SOFT_DT_6_L register by
 *     - status = Max96712AccessOneRegField(
 *      - status, handle, REG_FIELD_SOFT_DT_6_L,
 *      - (dataFormat & 0x3U), REG_READ_MOD_WRITE_MODE)
 *      .
 *     .
 *    .
 *   - else
 *    - write bpp                  to REG_FIELD_SOFT_BPP_7 register by
 *     - status = Max96712AccessOneRegField(
 *      - status, handle, REG_FIELD_SOFT_BPP_7, bpp, REG_READ_MOD_WRITE_MODE)
 *      .
 *     .
 *    - write dataFormat           to REG_FIELD_SOFT_DT_7 register by
 *     - status = Max96712AccessOneRegField(
 *      - status, handle, REG_FIELD_SOFT_DT_7,
 *      - dataFormat, REG_READ_MOD_WRITE_MODE)
 *      .
 *     .
 *    .
 *   - write 0x00                     to REG_FIELD_SOFT_OVR_0_EN register by
 *    - status = Max96712AccessOneRegFieldOffset(
 *     - status, handle, REG_FIELD_SOFT_OVR_0_EN,
 *     - i, 0U, REG_READ_MOD_WRITE_MODE)
 *     .
 *    .
 *   - write 0x01                     to REG_FIELD_SOFT_OVR_4_EN register by
 *    - status = Max96712AccessOneRegFieldOffset(
 *     - status, handle, REG_FIELD_SOFT_OVR_4_EN,
 *     - i, 1U, REG_READ_MOD_WRITE_MODE)
 *     .
 *    .
 *   .
 *  .
 * @param[in] handle DEVBLK handle
 * @param[in] link   bit mask of links to setup
 * @param[in] linkPipelineMap  pipeline setting for links
 * @param[in] paramSize parameter structure size in bytes
 * @param[in] pipelink_sz expected parameter structure size
 * @param[out] isValidSize set true if size matchaches expected size
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
OverrideDataType(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link,
    LinkPipelineMapMAX96712 const* linkPipelineMap,
    size_t paramSize,
    size_t pipelink_sz,
    bool *isValidSize)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    if (pipelink_sz != paramSize) {
        *isValidSize = false;
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
        uint8_t bpp = 0U;
        uint8_t dataFormat = 0U;
        uint8_t i = 0U;

        /* Override is enabled only for pipes 0-3 */
        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
            if (MAX96712_IS_GMSL_LINK_SET(link, i) &&
                linkPipelineMap[i].isDTOverride) {

                status = GetDataTypeValAndBpp(linkPipelineMap[i].dataType, &dataFormat, &bpp);

                status = Max96712AccessOneRegFieldOffset(status, handle, REG_FIELD_SOFT_BPP_0,
                                                         i, bpp,
                                                         REG_READ_MOD_WRITE_MODE);

                status = Max96712AccessOneRegFieldOffset(status, handle, REG_FIELD_SOFT_DT_0, i,
                                                         dataFormat,
                                                         REG_READ_MOD_WRITE_MODE);


                if (i == 1U) {
                    status = Max96712AccessOneRegField(status, handle, REG_FIELD_SOFT_DT_1_H,
                                                       (dataFormat >> 4U),
                                                       REG_READ_MOD_WRITE_MODE);
                } else if (i == 2U) {
                    status = Max96712AccessOneRegField(status, handle, REG_FIELD_SOFT_BPP_2_H,
                                                       (bpp >> 2U),
                                                       REG_READ_MOD_WRITE_MODE);

                    status = Max96712AccessOneRegField(status, handle, REG_FIELD_SOFT_DT_2_H,
                                                       (dataFormat >> 2U),
                                                       REG_READ_MOD_WRITE_MODE);
                } else {
                    /* do nothing */
                }

                status = Max96712AccessOneRegFieldOffset(status, handle, REG_FIELD_SOFT_OVR_0_EN,
                                                         i,
                                                         1U,
                                                         REG_READ_MOD_WRITE_MODE);

                if (drvHandle->ctx.tpgEnabled &&
                    ((drvHandle->ctx.pipelineEnabled & (0x10U << i)) != 0U)) {
                    /* Override BPP, DT for the pipeline 4 ~ 7 */
                    if (i == 0U) {
                        status = Max96712AccessOneRegField(status, handle, REG_FIELD_SOFT_BPP_4,
                                                           bpp,
                                                           REG_READ_MOD_WRITE_MODE);

                        status = Max96712AccessOneRegField(status, handle, REG_FIELD_SOFT_DT_4,
                                                           dataFormat,
                                                           REG_READ_MOD_WRITE_MODE);
                    } else if (i == 1U) {
                        status = Max96712AccessOneRegField(status, handle, REG_FIELD_SOFT_BPP_5,
                                                           bpp,
                                                           REG_READ_MOD_WRITE_MODE);

                        status = Max96712AccessOneRegField(status, handle, REG_FIELD_SOFT_DT_5_H,
                                                           (dataFormat >> 4U),
                                                           REG_READ_MOD_WRITE_MODE);

                        status = Max96712AccessOneRegField(status, handle, REG_FIELD_SOFT_DT_5_L,
                                                           (dataFormat & 0xFU),
                                                           REG_READ_MOD_WRITE_MODE);
                    } else if (i == 2U) {
                        status = Max96712AccessOneRegField(status, handle, REG_FIELD_SOFT_BPP_6_H,
                                                           (bpp >> 2U),
                                                           REG_READ_MOD_WRITE_MODE);

                        status = Max96712AccessOneRegField(status, handle, REG_FIELD_SOFT_BPP_6_L,
                                                           (bpp & 0x3U),
                                                           REG_READ_MOD_WRITE_MODE);

                        status = Max96712AccessOneRegField(status, handle, REG_FIELD_SOFT_DT_6_H,
                                                           (dataFormat >> 2U),
                                                           REG_READ_MOD_WRITE_MODE);

                        status = Max96712AccessOneRegField(status, handle, REG_FIELD_SOFT_DT_6_L,
                                                           (dataFormat & 0x3U),
                                                           REG_READ_MOD_WRITE_MODE);
                    } else {
                        status = Max96712AccessOneRegField(status, handle, REG_FIELD_SOFT_BPP_7,
                                                           bpp,
                                                           REG_READ_MOD_WRITE_MODE);

                        status = Max96712AccessOneRegField(status, handle, REG_FIELD_SOFT_DT_7,
                                                           dataFormat,
                                                           REG_READ_MOD_WRITE_MODE);
                    }

                    status = Max96712AccessOneRegFieldOffset(status, handle,
                                                             REG_FIELD_SOFT_OVR_0_EN,
                                                             i,
                                                             0U,
                                                             REG_READ_MOD_WRITE_MODE);
                    status = Max96712AccessOneRegFieldOffset(status, handle,
                                                             REG_FIELD_SOFT_OVR_4_EN,
                                                             i,
                                                             1U,
                                                             REG_READ_MOD_WRITE_MODE);
                }
                if (status != NVMEDIA_STATUS_OK) {
                    break;
                }
            }
        }
    }
    return status;
}

/**
 * @brief Selects Video pipeline
 *
 * - For each link
 *  - if (MAX96712_IS_GMSL_LINK_SET(link, i))
 *   - if it is configured as a single pipeline in linkPipelineMap
 *    - Note: in case of single pipe Z from ser, select that for pipe in deser
 *    - write ((4U * link index) + 2U) to REG_FIELD_VIDEO_PIPE_SEL_0 register by
 *     - status = Max96712AccessOneRegFieldOffset(
 *      - status, handle, REG_FIELD_VIDEO_PIPE_SEL_0,
 *      - i, (4U * i) + 2U, REG_READ_MOD_WRITE_MODE)
 *      .
 *     .
 *    .
 *   - else
 *    - write (4U * link index) to REG_FIELD_VIDEO_PIPE_SEL_0 register by
 *     - status = Max96712AccessOneRegFieldOffset(
 *      - status, handle, REG_FIELD_VIDEO_PIPE_SEL_0,
 *      - i, 4U * i, REG_READ_MOD_WRITE_MODE)
 *      .
 *     .
 *    - if pipeline set ot be embedded data type in linkPipelineMap
 *     - write ((4U * link index) + 1U) to REG_FIELD_VIDEO_PIPE_SEL_4 register by
 *      - status = Max96712AccessOneRegFieldOffset(
 *       - status, handle, REG_FIELD_VIDEO_PIPE_SEL_4,
 *       - i, (4U * i) + 1U, REG_READ_MOD_WRITE_MODE)
 *       .
 *      .
 *     .
 *    .
 *   .
 *  .
 * @param[in] handle DEVBLK handle
 * @param[in] link   bit mask of links to setup
 * @param[in] linkPipelineMap  pipeline setting for links
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
VideoPipelineSel(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link,
    LinkPipelineMapMAX96712 const* linkPipelineMap)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    uint8_t i = 0U;

    for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            if (IsGMSL2Mode(drvHandle->ctx.gmslMode[i])) {
                /* pipe Z from ser is connected to the pipe in deser
                 * Two different pipelines receives the data from the same camera
                 */
                status = Max96712AccessOneRegFieldOffset(status, handle, REG_FIELD_VIDEO_PIPE_SEL_0,
                                                         i,
                                                         (4U * i) + 2U,
                                                         REG_READ_MOD_WRITE_MODE);

                status = Max96712AccessOneRegFieldOffset(status, handle, REG_FIELD_VIDEO_PIPE_SEL_4,
                                                         i,
                                                         (4U * i) + 2U,
                                                         REG_READ_MOD_WRITE_MODE);
            }
        }
    }

    return status;
}

/**
 * @brief Enables Video pipeline
 *
 * - Clears bitReg queue by
 *  - ClearRegFieldQ(handle)
 *  .
 * - set uint8_t pipelineEnabled = 0U
 * - For each link
 *  - if (MAX96712_IS_GMSL_LINK_SET(link, i))
 *   - add register REG_FIELD_VIDEO_PIPE_EN_0 to queue by
 *    - status = Max96712AddRegFieldOffset(
 *     - status, handle, REG_FIELD_VIDEO_PIPE_EN_0, i, 1U)
 *     .
 *   . set pipelineEnabled |= (uint8_t)((1U << i) & 0x0FU)
 *   - if (linkPipelineMap[i].isEmbDataType && !linkPipelineMap[i].isSinglePipeline)
 *    - set pipelineEnabled |= (uint8_t)((0x10U << i) & 0xFFU)
 *    - add register REG_FIELD_VIDEO_PIPE_EN_4 to queue by
 *     - status = Max96712AddRegFieldOffset(
 *      - status, handle, REG_FIELD_VIDEO_PIPE_EN_4, i, 1U)
 *      .
 *     .
 *    .
 *   .
 *  .
 * - write commands from queue to deserializer registers by
 *  - status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE)
 *  .
 * @param[in] handle DEVBLK handle
 * @param[in] link   bit mask of links to setup
 * @param[in] linkPipelineMap  pipeline setting for links
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
VideoPipelineEnable(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link,
    LinkPipelineMapMAX96712 const* linkPipelineMap)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712  *drvHandle = getHandlePrivMAX96712(handle);
    uint8_t pipelineEnabled = 0U;
    uint8_t i = 0U;

    /* Enable Pipelines from 0 to 3 */
    ClearRegFieldQ(handle);
    for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_VIDEO_PIPE_EN_0, i, 1U);
            pipelineEnabled |= (uint8_t)((1U << i) & 0x0FU);
        }
    }
#if !NV_IS_SAFETY || SAFETY_DBG_OV
    /* Enable Pipelines from 4 to 7 */
    if (drvHandle->ctx.cfgPipeCopy == CDI_MAX96712_CFG_PIPE_COPY_NONE) {
        status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_VIDEO_PIPE_EN_4, i, 1U);
    } else if (drvHandle->ctx.cfgPipeCopy < CDI_MAX96712_CFG_PIPE_COPY_INVALID) {
        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_VIDEO_PIPE_EN_4, i,
                                                   1U);
                pipelineEnabled |= (uint8_t)((0x10U << i) & 0x0FU);
            }
        }
    }
#endif
    status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE);
    if (status == NVMEDIA_STATUS_OK) {
        drvHandle->ctx.pipelineEnabled |= pipelineEnabled;
    }

    return status;
}

/**
 * @brief Updates TX port settings in mappingRaw->regs and mappingEmbRegs->regs
 *
 * - if it is a single pipeline
 *  - if it is an embedded data type
 *  - Enable 4 mappings FS, FE, PIX, EMB
 *    - mappingRawRegs[0].data = 0x0FU
 *    .
 *  - Map all 4 to controller specified by txPort
 *   - mappingRawRegs[1].data =
 *    - ((txPort << 6U) | (txPort << 4U) |
 *    - (txPort << 2U) | txPort) & 0xFFU
 *    .
 *   .
 *  - else
 *   - Enable 3 mappings FS, FE, PIX
 *    - mappingRawRegs[0].data = 0x07U
 *    .
 *   - Map all 4 to controller specified by txPort by
 *    - mappingRawRegs[1].data =
 *     - ((txPort << 4U) | (txPort << 2U) | txPort) & 0xFFU
 *     .
 *    .
 *   .
 *  - else
 *   - mappingRawRegs[1].data = ((txPort << 4U) | (txPort << 2U) | txPort) & 0xFFU
 *   - mappingEmbRegs[1].data = ((txPort << 4U) | (txPort << 2U) | txPort) & 0xFFU.
 *   .
 *
 * @param isSinglePipeline set if it is a single pipeline
 * @param[in] isEmbDataType set if fit is an embedded data type
 * @param[in] txPort TX port number
 * @param[out] mappingRawRegs pointer to DevBlkCDII2CReg for update
 * @param[out] mappingEmbRegs pointer to DevBlkCDII2CReg for update
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static void
UpdateTxPort(
    bool isEmbDataType,
    uint16_t txPort,
    DevBlkCDII2CReg *mappingRawRegs,
    DevBlkCDII2CReg *mappingEmbRegs)
{
    /* Update Tx Port */
   if (isEmbDataType) {
       /*Enable 4 mappings FS, FE, PIX, EMB */
       mappingRawRegs[0].data = 0x0FU;
       /* Map all 4 to controller specified by txPort */
       mappingRawRegs[1].data = ((txPort << 6U) | (txPort << 4U) |
                                 (txPort << 2U) | txPort) & 0xFFU;
   } else {
       /*Enable 3 mappings FS, FE, PIX */
       mappingRawRegs[0].data = 0x07U;
       /* Map all 4 to controller specified by txPort */
       mappingRawRegs[1].data = ((txPort << 4U) | (txPort << 2U) | txPort) & 0xFFU;
   }
}
#if !NV_IS_SAFETY || SAFETY_DBG_OV
/**
 * @brief Helper function for getting Deserializer's tx port
 * - If camera recorder config is 1
 *  - Tx port is 2
 * - else if camera recorder config is between 2 and 4
 *  - if the pipeline is 4 and 5, then txport is 2
 *  - else if the pipeline is 6 and 7, then txport is 3
 *  .
 * @param[in] handle   DEVBLK handle
 * @param[in] pipeline   the pipline number
 * @return Tx Port number */
static uint16_t
GetTxPort(
    DevBlkCDIDevice const* handle,
    uint8_t pipeline)
{
    DriverHandleMAX96712 *drvHandle = getHandlePrivMAX96712(handle);
    uint16_t txPort = 2U;

    if (drvHandle->ctx.cfgPipeCopy == CDI_MAX96712_CFG_PIPE_COPY_MODE_1) {
        txPort = 2U; /* for x2 and x4 */
    } else if (drvHandle->ctx.cfgPipeCopy == CDI_MAX96712_CFG_PIPE_COPY_MODE_2) {
        switch (pipeline) {
            case 4:
            case 5:
                txPort = 2U;
                break;
            case 6:
            case 7:
                txPort = 3U;
                break;
            default:
                txPort = 2U;
                break;
        }
    }

    return txPort;
}

/**
 * @brief Updates Pipeline Map Tx port for Port B
 *
 * - if the pipeline is not bewteen 4 and 7
 *  - return NVMEDIA_STATUS_BAD_PARAMETER
 *  .
 * .
 *
 * - if it is an embedded data type
 *  - Enable 4 mappings FS, FE, PIX, EMB
 *    - mappingRawRegs[0].data = 0x0FU
 *    .
 *  - Map all 4 to controller specified by txPort
 *   - mappingRawRegs[1].data =
 *    - ((txPort << 6U) | (txPort << 4U) |
 *    - (txPort << 2U) | txPort) & 0xFFU
 *    .
 *   .
 *  - else
 *   - Enable 3 mappings FS, FE, PIX
 *    - mappingRawRegs[0].data = 0x07U
 *    .
 *   - Map all 3 to controller specified by txPort by
 *    - mappingRawRegs[1].data =
 *     - ((txPort << 4U) | (txPort << 2U) | txPort) & 0xFFU
 *     .
 *    .
 *   .
 *
 * @param[in] pipeline to indicate the pipeline number
 * @param[in] isEmbDataType set if fit is an embedded data type
 * @param[out] mappingRawRegs pointer to DevBlkCDII2CReg for update
 * @param[out] mappingEmbRegs pointer to DevBlkCDII2CReg for update
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
UpdateTxPortB(
    DevBlkCDIDevice const* handle,
    uint8_t pipeline,
    bool isEmbDataType,
    DevBlkCDII2CReg *mappingRawRegs,
    DevBlkCDII2CReg *mappingEmbRegs)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 *drvHandle = getHandlePrivMAX96712(handle);
    uint16_t txPort = 0U;

    if ((pipeline < 4U) || (pipeline > 7U)) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (drvHandle->ctx.cfgPipeCopy == CDI_MAX96712_CFG_PIPE_COPY_MODE_1) {
        txPort = 2U; /* for x2 and x4 */
    } else if (drvHandle->ctx.cfgPipeCopy == CDI_MAX96712_CFG_PIPE_COPY_MODE_2) {
        txPort = GetTxPort(handle, pipeline);
    }

    /* Update Tx Port */
    if (isEmbDataType) {
        /*Enable 4 mappings FS, FE, PIX, EMB */
        mappingRawRegs[0].data = 0x0FU;
        /* Map all 4 to controller specified by txPort */
        mappingRawRegs[1].data = ((txPort << 6U) | (txPort << 4U) |
                                  (txPort << 2U) | txPort) & 0xFFU;
    } else {
        /*Enable 3 mappings FS, FE, PIX */
        mappingRawRegs[0].data = 0x07U;
        /* Map all 4 to controller specified by txPort */
        mappingRawRegs[1].data = ((txPort << 4U) | (txPort << 2U) | txPort) & 0xFFU;
    }

    return status;
}
#endif
/**
 * @brief Updates Pipeline Map
 *
 * - Update the data type and VC number for FS, FE, Pixel data
 * - Update the register address for FS, FE, Pixel data
 *
 * - if it is an embedded data type
 *  - Update the VC number for EMB
 *  - Update the register address for EMB
 *
 * @param[in] pipeline to indicate the pipeline number
 * @param[in] isEmbDataType set if fit is an embedded data type
 * @param[in] dataTypeVal the data type value for the pixel data
 * @param[in] vcID the VC number
 * @param[out] mappingRawRegs pointer to DevBlkCDII2CReg for update
 * @param[out] mappingEmbRegs pointer to DevBlkCDII2CReg for update */
static void UpdatePipelineMap(
    uint8_t pipeline,
    bool isEmbDataType,
    uint8_t dataTypeVal,
    uint16_t vcID,
    DevBlkCDII2CReg *mappingRawRegs,
    DevBlkCDII2CReg *mappingEmbRegs)
{
    /* Update the data type, VC number */
    mappingRawRegs[2].data = dataTypeVal;
    mappingRawRegs[3].data = ((vcID << 6U) | dataTypeVal) & 0xFFU;
    mappingRawRegs[5].data = (vcID << 6U) & 0xFFU;
    mappingRawRegs[7].data = ((vcID << 6U) | 0x01) & 0xFFU;

    /* Update the offset */
    mappingRawRegs[0].address = (0x090BU + pipeline * 0x40U) & 0xFFFFU;
    mappingRawRegs[1].address = (0x092DU + pipeline * 0x40U) & 0xFFFFU;
    mappingRawRegs[2].address = (0x090DU + pipeline * 0x40U) & 0xFFFFU;
    mappingRawRegs[3].address = (0x090EU + pipeline * 0x40U) & 0xFFFFU;
    mappingRawRegs[4].address = (0x090FU + pipeline * 0x40U) & 0xFFFFU;
    mappingRawRegs[5].address = (0x0910U + pipeline * 0x40U) & 0xFFFFU;
    mappingRawRegs[6].address = (0x0911U + pipeline * 0x40U) & 0xFFFFU;
    mappingRawRegs[7].address = (0x0912U + pipeline * 0x40U) & 0xFFFFU;

    if (isEmbDataType) {
        /* Update the VC number */
        mappingEmbRegs[1].data = ((vcID << 6U) | 0x12U) & 0xFFU;

        /* Update the offset */
        mappingEmbRegs[0].address = (0x0913U + pipeline * 0x40U) & 0xFFFFU;
        mappingEmbRegs[1].address = (0x0914U + pipeline * 0x40U) & 0xFFFFU;
    }
}

/**
 * @brief Returns index of TX port based on enum TxPortMAX96712 port
 *
 * - verifies "val" is not null
 * - returns the following values:
 *  - for port CDI_MAX96712_TXPORT_PHY_C: 0U
 *  - for port CDI_MAX96712_TXPORT_PHY_D: 1U
 *  - for port CDI_MAX96712_TXPORT_PHY_E: 2U
 *  - for port CDI_MAX96712_TXPORT_PHY_F: 3U
 *
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
GetTxPortMAX96712Val(TxPortMAX96712 const port, uint16_t *val)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (val == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter. NULL ptr");
    } else {
        switch (port) {
            case CDI_MAX96712_TXPORT_PHY_C:
                *val = 0U;
                break;
            case CDI_MAX96712_TXPORT_PHY_D:
                *val = 1U;
                break;
            case CDI_MAX96712_TXPORT_PHY_E:
                *val = 2U;
                break;
            case CDI_MAX96712_TXPORT_PHY_F:
                *val = 3U;
                break;
            default:
                *val = 0U;
                status = NVMEDIA_STATUS_BAD_PARAMETER;
                SIPL_LOG_ERR_STR_UINT("MAX96712: Bad parameter. Invalid link",
                                      (uint32_t)port);
                break;
        }
    }

    return status;
}

/**
 * @brief Prepares the setup for pipeline mapping
 *
 * - verifies that paramSize is the size of pipelink_sz - fails
 * - if not same
 *  - set return value isValidSize = false
 *  .
 * - else
 *  - set return value ValidSize = true
 *  .
 * - extracts device driver handle from handle
 * - Sets up the mapping for raw
 *  - DevBlkCDII2CReg mappingRawRegs[8] = {
 *   - Note: Send RAW12 FS and FE from X to Controller 1
 *   - {0x090BU, 0x07U},
 *   - {0x092DU, 0x00U},
 *   - Note: For the following MSB 2 bits = VC, LSB 6 bits = DT
 *   - {0x090DU, 0x2CU},
 *   - {0x090EU, 0x2CU},
 *   - {0x090FU, 0x00U},
 *   - {0x0910U, 0x00U},
 *   - {0x0911U, 0x01U},
 *   - {0x0912U, 0x01U},
 *   .
 *  - }
 *  .
 * - setup wrapper structure for mappingRawRegs by
 *  -  DevBlkCDII2CRegList mappingRaw = {
 *   - .regs = mappingRawRegs,
 *   - .numRegs = (uint32_t)(sizeof(mappingRawRegs) /
 *    -  sizeof(mappingRawRegs[0])),
 *    .
 *   .
 *  - };
 *  .

 * - Sets up the mapping for embedded
 *  - DevBlkCDII2CReg mappingEmbRegs[8] = {
 *   - Note: Send EMB8 from Y to Controller 1 with VC unchanged
 *   - {0x0A0BU, 0x07U},
 *   - {0x0A2DU, 0x00U},
 *   - Note: For the following MSB 2 bits = VC, LSB 6 bits = DT
 *   - {0x0A0DU, 0x12U},
 *   - {0x0A0EU, 0x12U},
 *   - {0x0A0FU, 0x00U},
 *   - {0x0A10U, 0x00U},
 *   - {0x0A11U, 0x01U},
 *   - {0x0A12U, 0x01U},
 *   .
 *  - }
 * - setup wrapper structure for mappingEmbRegs by
 *  - DevBlkCDII2CRegList mappingEmb = {
 *   - .regs = mappingEmbRegs,
 *   - .numRegs = (uint32_t)(sizeof(mappingEmbRegs) /
 *    - sizeof(mappingEmbRegs[0])),
 *    .
 *   .
 *  - };
 *  .
 * - sets up mapping for embewdded pipe Z
 *  - DevBlkCDII2CReg mappingEmbPipeZRegs[2] = {
 *   - Send EMB data from pipe Z to controller 1
 *   - {0x0913U, 0x12U},
 *   - {0x0914U, 0x12U},
 *   .
 *  - }
 *  .
 * - setup wrapper structure for mappingEmbPipeZRegs by
 *  - DevBlkCDII2CRegList mappingEmbPipeZ = {
 *   - .regs = mappingEmbPipeZRegs,
 *   - .numRegs = (uint32_t)(sizeof(mappingEmbPipeZRegs) /
 *    - sizeof(mappingEmbPipeZRegs[0])),
 *    .
 *   .
 *  - }
 *  .
 * - extract txPort from context by
 *  - status = GetTxPortMAX96712Val(drvHandle->ctx.txPort,&txPort)
 *  .
 * - for each link
 *  - if MAX96712_IS_GMSL_LINK_SET for this link
 *   - extract
 *    - isEmbDataType = linkPipelineMap[i].isEmbDataType
 *    - vcID = linkPipelineMap[i].vcID
 *    .
 *   - call UpdateTxPort to setup the txPorts by
 *    - UpdateTxPort(
 *     - linkPipelineMap[i].isSinglePipeline,
 *     - linkPipelineMap[i].isEmbDataType,
 *     - txPort, mappingRawRegs, mappingEmbRegs)
 *     .
 *    .
 *   - if isEmbDataType is true and isDTOverride is true then
 *    - Error: Emb data type is valid for GMSL2 only
 *    .
 *   - call GetDataTypeValAndBpp to extract dataTypeVal and bpp by
 *    - status = GetDataTypeValAndBpp(linkPipelineMap[i].dataType, &dataTypeVal, &bpp)
 *    .
 *   - extract vcID from linkPipelineMap[i].vcID
 *   -  update pipeline settings by calling
 *    - SetPipelineMapUpdateOffset(linkPipelineMap[i].isSinglePipeline,
 *     - isEmbDataType,
 *     - dataTypeVal,
 *     - vcID,
 *     - mappingRawRegs,
 *     - mappingEmbRegs,
 *     - mappingEmbPipeZRegs)
 *     .
 *    .
 *   - write setting in mappingRaw to deserializer registers by
 *    - status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &mappingRaw)
 *    .
 *   - if isEmbDataType is true
 *    - if isSinglePipeline
 *     - call DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &mappingEmb)
 *     .
 *    - else
 *     - call DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &mappingEmbPipeZ);
 *     .
 *    .
 *   .
 *  - call SetPipelineMapUpdateRegAddrForNextLink to calcumate and set address
 *   - in mappingEmb, mappingEmbPipeZ, and mappingRawRegs nodes for next link by
 *    - SetPipelineMapUpdateRegAddrForNextLink(status,
 *     -  mappingRawRegs, mappingEmbRegs,  mappingEmbPipeZRegs)
 *     .
 *    .
 *   .
 *  .
 * - call VideoPipelineSel  to select pileline by
 *  - status = VideoPipelineSel(handle, link, linkPipelineMap)
 *  .
 * - call VideoPipelineEnable to enable pipeline by
 *  - status = VideoPipelineEnable(handle, link, linkPipelineMap)
 *  .
 * @param[in] handle DEVBLK handle
 * @param[in] link   bit mask of links to setup
 * @param[in] linkPipelineMap  pipeline setting for links
 * @param[in] paramSize parameter structure size in bytes
 * @param[in] pipelink_sz expected parameter structure size
 * @param[out] isValidSize set true if size matchaches expected size
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
SetPipelineMap(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link,
    LinkPipelineMapMAX96712 const* linkPipelineMap,
    size_t paramSize,
    size_t pipelink_sz,
    bool *isValidSize)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (pipelink_sz != paramSize) {
        *isValidSize = false;
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
        uint16_t txPort;
        /* Each camera's data is processed per link in the separate pipeline individually.
         * Camera on the link 0 --> Pipeline 0
         * Camera on the link 1 --> Pipeline 1
         * Camera on the link 2 --> Pipeline 2
         * Camera on the link 3 --> Pipeline 3
         * The same data is also processed in the different pipeline
         * Camera on the link 0 --> Pipeline 4
         * Camera on the link 1 --> Pipeline 5
         * Camera on the link 2 --> Pipeline 6
         * Camera on the link 3 --> Pipeline 7
         */
        DevBlkCDII2CReg mappingRawRegs[] = {
            /* Send Pixel data, FS and FE to Controller 0 */
            {0x090BU, 0x07U},
            {0x092DU, 0x00U},
            /* For the following MSB 2 bits = VC, LSB 6 bits = DT */
            {0x090DU, 0x2CU},
            {0x090EU, 0x2CU},
            {0x090FU, 0x00U},
            {0x0910U, 0x00U},
            {0x0911U, 0x01U},
            {0x0912U, 0x01U},
        };
        DevBlkCDII2CRegList mappingRaw = {
            .regs = mappingRawRegs,
            .numRegs = (uint32_t)(sizeof(mappingRawRegs) /
                    sizeof(mappingRawRegs[0])),
        };
        DevBlkCDII2CReg mappingEmbRegs[] = {
            /* For the following MSB 2 bits = VC, LSB 6 bits = DT */
            {0x0913U, 0x12U},
            {0x0914U, 0x12U},
        };
        DevBlkCDII2CRegList mappingEmb = {
            .regs = mappingEmbRegs,
            .numRegs = (uint32_t)(sizeof(mappingEmbRegs) /
                    sizeof(mappingEmbRegs[0])),
        };

        bool isEmbDataType = false;
        uint16_t vcID = 0U;
        uint8_t dataTypeVal = 0U;
        uint8_t i = 0U;

        *isValidSize = true;
        status = GetTxPortMAX96712Val(drvHandle->ctx.txPort,&txPort);
        if (status == NVMEDIA_STATUS_OK) {
            for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
                /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
                if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                    isEmbDataType = linkPipelineMap[i].isEmbDataType;

                    vcID = linkPipelineMap[i].vcID;


                    UpdateTxPort(linkPipelineMap[i].isEmbDataType,
                                 txPort,
                                 mappingRawRegs,
                                 mappingEmbRegs);

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
                    if (isEmbDataType && !IsGMSL2Mode(drvHandle->ctx.gmslMode[i])) {
                        SIPL_LOG_ERR_STR("MAX96712: Emb data type is valid for GMSL2 only");
                        status = NVMEDIA_STATUS_ERROR;
                    }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
                    if (status == NVMEDIA_STATUS_OK) {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
                        if (isEmbDataType && linkPipelineMap[i].isDTOverride) {
                            SIPL_LOG_ERR_STR("MAX96712: Emb data type is not supported "
                                             "with dt override enabled");
                            status = NVMEDIA_STATUS_ERROR;
                        }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
                    }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

                    if (status == NVMEDIA_STATUS_OK) {
                        uint8_t bpp = 0U;
                        status = GetDataTypeValAndBpp(linkPipelineMap[i].dataType,
                                                      &dataTypeVal, &bpp);
                    }

                    /* pipeline 0 ~ 3 */
                    if (status == NVMEDIA_STATUS_OK) {
                        UpdatePipelineMap(i,
                                          isEmbDataType,
                                          dataTypeVal,
                                          vcID,
                                          mappingRawRegs,
                                          mappingEmbRegs);
                        status = MAX96712WriteArrayVerify(handle, &mappingRaw);
                    }

                    if (status == NVMEDIA_STATUS_OK) {
                        if (isEmbDataType) {
                            status = MAX96712WriteArrayVerify(handle, &mappingEmb);
                        }
                    }
#if !NV_IS_SAFETY || SAFETY_DBG_OV
                    if (status == NVMEDIA_STATUS_OK) {
                        status = UpdateTxPortB(handle,
                                               i + 4U,
                                               linkPipelineMap[i].isEmbDataType,
                                               mappingRawRegs,
                                               mappingEmbRegs);
                    }

                    /* pipeline 4 ~ 7 */
                    if (status == NVMEDIA_STATUS_OK) {
                        UpdatePipelineMap(i + 4U,
                                          isEmbDataType,
                                          dataTypeVal,
                                          vcID,
                                          mappingRawRegs,
                                          mappingEmbRegs);
                        status = MAX96712WriteArrayVerify(handle, &mappingRaw);
                    }
#endif
                    if (status == NVMEDIA_STATUS_OK) {
                        if (isEmbDataType) {
                            status = MAX96712WriteArrayVerify(handle, &mappingEmb);
                        }
                    }
                }
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            status = VideoPipelineSel(handle,
                    link,
                    linkPipelineMap);

            if (status == NVMEDIA_STATUS_OK) {
                status = VideoPipelineEnable(handle,
                                             link,
                                             linkPipelineMap);
            }
        }
    }
    return status;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
static NvMediaStatus
SetPipelineMapTPG(
    DevBlkCDIDevice const* handle,
    uint8_t linkIndex,
    LinkPipelineMapMAX96712 const* linkPipelineMap)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    uint16_t txPort = 0U;
    bool PGGen0 = true;
    uint16_t vcID = 0U;
    uint8_t dataTypeVal = 0U;
    uint8_t i = 0U;
    uint8_t bpp = 0U;
    uint32_t data = 0U;
    /* Two pipelines are one set to process raw12 and emb */
    DevBlkCDII2CReg mappingRawRegs[] = {
        /* Send RAW12 FS and FE from X to Controller 1 */
        {0x090BU, 0x07U},
        {0x092DU, 0x00U},
        /* For the following MSB 2 bits = VC, LSB 6 bits = DT */
        {0x090DU, 0x2CU},
        {0x090EU, 0x2CU},
        {0x090FU, 0x00U},
        {0x0910U, 0x00U},
        {0x0911U, 0x01U},
        {0x0912U, 0x01U},
    };
    DevBlkCDII2CRegList mappingRaw = {
        .regs = mappingRawRegs,
        .numRegs = (uint32_t)(sizeof(mappingRawRegs) /
                              sizeof(mappingRawRegs[0])),
    };

    if ((linkPipelineMap == NULL) || (NULL == drvHandle)) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {

        txPort = (uint16_t)drvHandle->ctx.txPort;
        vcID = linkPipelineMap[linkIndex].vcID;

        /* Update Tx Port */
        data = (txPort << 4U) | (txPort << 2U) | txPort;
        mappingRawRegs[1].data = toUint16FromUint32(data);

        status = GetDataTypeValAndBpp(linkPipelineMap[linkIndex].dataType, &dataTypeVal, &bpp);
        if (status == NVMEDIA_STATUS_OK) {

            if ((drvHandle->ctx.pipelineEnabled & MAX96712_LINK_SHIFT(0x1U, linkIndex)) != 0U) {
                PGGen0 = true;
            } else if ((drvHandle->ctx.pipelineEnabled & MAX96712_LINK_SHIFT(0x10U, linkIndex)) != 0U) {
                PGGen0 = false;
            } else {
                SIPL_LOG_ERR_STR_UINT("MAX96712: No pipeline enabled for the link",
                                      linkIndex);
                SIPL_LOG_ERR_STR("Please make sure if CDI_WRITE_PARAM_CMD_MAX96712_SET_PG calling");
                SIPL_LOG_ERR_STR("before CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING_TPG");
                status = NVMEDIA_STATUS_BAD_PARAMETER;
            }

            if (status == NVMEDIA_STATUS_OK) {
                /* update offset */
                for (i = 0U; i < 8U; i++) {
                    mappingRawRegs[i].address += (((uint16_t)linkIndex * 0x40U) +
                                     ((PGGen0 == false) ? 0x100U : 0x0U));
                }

                /* 4 mapping for the pixel data */
                mappingRawRegs[2].data = dataTypeVal;
                mappingRawRegs[3].data = (vcID << 6U) | dataTypeVal;
                /* Change FS packet's DT to reserved for RAW pipeline */
                mappingRawRegs[5].data = (vcID << 6U) | 0U;
                /* Change FE packet's DT to reserved for RAW pipeline */
                mappingRawRegs[7].data = (vcID << 6U) | 1U;

                status = MAX96712WriteArrayVerify(handle, &mappingRaw);

            }
        }
    }
    return status;
}

static NvMediaStatus
ConfigPGSettings(
    DevBlkCDIDevice const* handle,
    uint32_t width,
    uint32_t height,
    float_t frameRate,
    uint8_t linkIndex)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 *drvHandle = getHandlePrivMAX96712(handle);
    DevBlkCDII2CRegList const* PGarray = NULL;
    DevBlkCDII2CReg *regsPG = NULL;
    uint8_t i = 0U, j = 0U;
    uint16_t pipelinectx = 1U;
    PGPCLKMAX96712 pclk = CDI_MAX96712_PG_PCLK_150MHX;
    PGModeMAX96712 pgMode = CDI_MAX96712_PG_NUM;

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        drvHandle->ctx.tpgEnabled = true;

        if ((width == 1920U) && (height == 1236U) && (frameRate == 30.0f)) {
            pgMode = CDI_MAX96712_PG_1920_1236_30FPS;
            PGarray = GetConfigPGArrCmd(pgMode);
            regsPG = PGArr1920x1236_30FPS_PATGEN0;
            pclk = CDI_MAX96712_PG_PCLK_150MHX;
        } else if ((width == 1920U) && (height == 1236U) && (frameRate == 60.0f)) {
            pgMode = CDI_MAX96712_PG_1920_1236_60FPS;
            PGarray = GetConfigPGArrCmd(pgMode);
            regsPG = PGArr1920x1236_60FPS_PATGEN0;
            pclk = CDI_MAX96712_PG_PCLK_375MHX;
        } else if ((width == 3848U) && (height == 2168U) && (frameRate == 30.0f)) {
            pgMode = CDI_MAX96712_PG_3848_2168_30FPS;
            PGarray = GetConfigPGArrCmd(pgMode);
            regsPG = PGArr3848x2168_30FPS_PATGEN0;
            pclk = CDI_MAX96712_PG_PCLK_375MHX;
        } else if ((width == 2880U) && (height == 1860U) && (frameRate == 30.0f)) {
            pgMode = CDI_MAX96712_PG_2880_1860_30FPS;
            PGarray = GetConfigPGArrCmd(pgMode);
            regsPG = PGArr2880x1860_30FPS_PATGEN0;
            pclk = CDI_MAX96712_PG_PCLK_375MHX;
        } else {
            status = NVMEDIA_STATUS_BAD_PARAMETER;
        }

        if (status == NVMEDIA_STATUS_OK) {
            bool exitFlag = false;
            for (i = 0U; i < MAX96712_MAX_NUM_PG; i++) {
                if (drvHandle->ctx.pgMode[i] == pgMode) {
                    exitFlag = true;
                }
                if ((!exitFlag) && (drvHandle->ctx.pgMode[i] == CDI_MAX96712_PG_NUM)) {
                    drvHandle->ctx.pgMode[i] = pgMode;
                    exitFlag = true;
                }
                if (exitFlag) {
                    break;
                }
            }

            drvHandle->ctx.pipelineEnabled |= (uint8_t)(MAX96712_LINK_SHIFT(pipelinectx,
                                                                            linkIndex) << (i * 4U));

            if (i == 1U) { /* For 2nd PG, need to update the register offset */
                /* PG setting */
                for (j = 0U; j < 38U; j++) {
                    regsPG[j].address += 0x30U;
                }
            }

            status = MAX96712WriteArrayVerify(handle, PGarray);
            if (status == NVMEDIA_STATUS_OK) {

                if (pclk == CDI_MAX96712_PG_PCLK_150MHX) {
                    status = MAX96712WriteArrayVerify(handle, GetConfigPGPCLK150MHZ(i));
                    if (status == NVMEDIA_STATUS_OK) {
                        status = Max96712AccessOneRegField(status, handle,
                                            REG_FIELD_FORCE_CSI_OUT_EN,
                                            1U,
                                            REG_READ_MOD_WRITE_MODE);
                    }
                }
            }
        }
    }

    return status;
}

static NvMediaStatus
MapUnusedPipe(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 *drvHandle = getHandlePrivMAX96712(handle);
    drvHandle->ctx.tpgEnabled = true;
    uint16_t i = 0U, j = 0U;
    DevBlkCDII2CReg mappingRawRegs[] = {
        /* Send RAW12 FS and FE from X to Controller 1 */
        {0x090BU, 0x07U},
        {0x092DU, 0x3FU},
        /* For the following MSB 2 bits = VC, LSB 6 bits = DT */
        {0x090DU, 0x24U},
        {0x090EU, 0x3FU},
        {0x090FU, 0x00U},
        {0x0910U, 0x02U},
        {0x0911U, 0x01U},
        {0x0912U, 0x03U},
    };
    DevBlkCDII2CRegList mappingRaw = {
        .regs = mappingRawRegs,
        .numRegs = (uint32_t)(sizeof(mappingRawRegs) /
                              sizeof(mappingRawRegs[0])),
    };

    /* When enabling TPG on Max96712, 1st TPG output is going to pipeline 0 ~ 3,
     * 2nd TPG output is going to pipeline 4 ~ 7.
     * And pipeline 0/4 is going to controller 0, pipeline 1/5 is going to controller 1
     * pipeline 2/6 is going to controller 2, pipeline 3/7 is going to controller 3 by default.
     * Since there is no way to disable TPG and TPG is behind the pipeline,
     * undesired pipeline output has to be mapped to unused controller.
     */
    for (i = 0U; i < MAX96712_NUM_VIDEO_PIPELINES; i++) {
        if ((drvHandle->ctx.pipelineEnabled & (0x1U << i)) == 0U) {
            if (drvHandle->ctx.mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2) {
                mappingRawRegs[1].data = 0x3FU; /* controller 1 */
            } else if (drvHandle->ctx.mipiOutMode ==
                       CDI_MAX96712_MIPI_OUT_2x4) {
                /* 2x4 mode*/
                mappingRawRegs[1].data = 0x3FU; /* controller 0 */
            } else {
                status = NVMEDIA_STATUS_BAD_PARAMETER;
            }

            if (status == NVMEDIA_STATUS_OK) {
                status = MAX96712WriteArrayVerify(handle, &mappingRaw);
            }

            if (status != NVMEDIA_STATUS_OK) {
                break;
            }
        }

        for (j = 0U; j < 8U; j++) {
            mappingRawRegs[j].address += (uint16_t)0x40U;
        }
    }

    return status;
}

static NvMediaStatus
EnablePG(
    DevBlkCDIDevice const* handle)
{
    DriverHandleMAX96712 *drvHandle = getHandlePrivMAX96712(handle);
    DevBlkCDII2CReg enablePGArrCmdReg = {0x1050U, 0xF3U};
    drvHandle->ctx.tpgEnabled = true;

    if ((drvHandle->ctx.pipelineEnabled & 0xF0U) != 0U) {
        enablePGArrCmdReg.address += 0x30U;
    }

    return MAX96712WriteUint8Verify(handle,
                                    enablePGArrCmdReg.address,
                                    (uint8_t)enablePGArrCmdReg.data);
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/**
 * @brief Sets TX source id
 *
 * - verifies that paramSize is the size of LinkMAX96712
 * - if not same
 *  - set returned value  isValidSize = false
 * else
 *  - set returned value  isValidSize = true
 *  - verifies MAX96712_IS_MULTIPLE_GMSL_LINK_SET is false
 *  - extracts device driver handle from handle
 *  - initialize DevBlkCDII2CReg txSrcIdReg = {0x0503U, 0x00U}
 *  - if MAX96712_IS_MULTIPLE_GMSL_LINK_SET in link mask
 *   - Error: Bad param: Multiple links specified
 *   .
 *  - else
 *   - for each link
 *    - if MAX96712_IS_GMSL_LINK_SET for link
 *     -calculate
 *      - txSrcIdReg.address += ((uint16_t)i) << 4U
 *      - Note: 0 - link 0, 1 - link 1, so on
 *      - txSrcIdReg.data = i;
 *      .
 *     - write settings to device by
 *      - status = DevBlkCDII2CPgmrWriteUint8(
 *       - drvHandle->i2cProgrammer,
 *       - txSrcIdReg.address,
 *       - (uint8_t)txSrcIdReg.data)
 *       .
 *      .
 *     .
 *    .
 *   .
 *  .
 * @param[in] handle DEVBLK handle
 * @param[in] link   bit mask of links to process
 * @param[in] parameterSize size in bytes of register to be written
 * @param[out] isValidSize set to true if parameterSize was valid, false otherwise
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
SetTxSRCId(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link,
    size_t parameterSize,
    bool *isValidSize)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    if(parameterSize != sizeof(link)) {
        *isValidSize = false;
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        DevBlkCDII2CReg txSrcIdReg = {0x0503U, 0x00U};
        uint8_t i = 0U;

        *isValidSize = true;
        if (MAX96712_IS_MULTIPLE_GMSL_LINK_SET(link)) {
            SIPL_LOG_ERR_STR("MAX96712: Bad param: Multiple links specified");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
        } else {
            for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
                if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                    txSrcIdReg.address += ((uint16_t)i) << 4U;
                    txSrcIdReg.data = i; /* 0 - link 0, 1 - link 1, so on */
                    status = MAX96712WriteUint8Verify(handle,
                                                      txSrcIdReg.address,
                                                      (uint8_t)txSrcIdReg.data);
                    break;
                }
            }
        }
    }

    return status;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
static NvMediaStatus
DisableAutoAck(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    DevBlkCDII2CReg autoAckReg = {0x0B0DU, 0x00U};
    uint16_t i = 0U;

    for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, (uint8_t)(i & 0xFFU))) {
            autoAckReg.address += (i << 8U); /* update the address */

            status = MAX96712WriteUint8Verify(handle,
                                              autoAckReg.address,
                                              (uint8_t)autoAckReg.data);

            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_UINT("MAX96712: failed to disableAutoAck for link ", i);
                break;
            }
            /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
            (void)nvsleep(25000);
        }
    }

    return status;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/**
 * @brief Enable ERRB
 *
 * - Clears bitReg queue by
 *  - ClearRegFieldQ(handle)
 *  .
 * - based on enable param, enables/disables REG_FIELD_ENABLE_ERRB register by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_ENABLE_ERRB,
 *   - (uint8_t)(enable ? 1U : 0U),
 *   - REG_READ_MOD_WRITE_MODE)
 *   .
 *  .
 * - read REG_FIELD_ENABLE_ERRB register back by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_ENABLE_ERRB, 0x00U, REG_READ_MODE)
 *   .
 *  .
 * - get value off queue by
 *  - reg_read = ReadFromRegFieldQ(handle, 0U)
 *  .
 * - verify data written correctly by
 *  - if (reg_read == bit_check)
 *  .
 * @param[in] handle DEVBLK handle
 * @param[in] enable if true enable, if false disable ERRB
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
EnableERRB(
        DevBlkCDIDevice const* handle,
        bool enable)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    ClearRegFieldQ(handle);
    status = Max96712AccessOneRegField(status, handle, REG_FIELD_ENABLE_ERRB,
            (uint8_t)(enable ? 1U : 0U),
            REG_READ_MOD_WRITE_MODE);
    /* Validate HW has ERRB bit set */
    if (status == NVMEDIA_STATUS_OK) {
        uint8_t reg_read = 0U;
        uint8_t bit_check = (uint8_t)(enable ? 1U : 0U);
        status = Max96712AccessOneRegField(status, handle, REG_FIELD_ENABLE_ERRB, 0x00U,
                REG_READ_MODE);
        if (status == NVMEDIA_STATUS_OK) {
            reg_read = ReadFromRegFieldQ(handle, 0U);
            if (reg_read != bit_check) {
                SIPL_LOG_ERR_STR("MAX96712: ERRB register write and read back mismatch");
                status = NVMEDIA_STATUS_ERROR;
            }
        }
    }
    return status;
}

/**
 * @brief enable LOCK
 *
 * - Clears bitReg queue by
 *  - ClearRegFieldQ(handle)
 *  .
 * - based on enable param, enables/disables REG_FIELD_ENABLE_LOCK register by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_ENABLE_LOCK,
 *   - (uint8_t)(enable ? 1U : 0U),
 *   - REG_READ_MOD_WRITE_MODE)
 *   .
 *  .
 * - read REG_FIELD_ENABLE_LOCK register back by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_ENABLE_LOCK, 0x00U, REG_READ_MODE)
 *   .
 *  .
 * - get value off queue by
 *  - reg_read = ReadFromRegFieldQ(handle, 0U)
 *  .
 * - verify data written correctly by
 *  - if (reg_read == bit_check)
 *  .
 * @param[in] handle DEVBLK handle
 * @param[in] enable if true enable, if false disable LOCK
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
EnableLOCK(
        DevBlkCDIDevice const *const handle,
        bool const enable)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t bit_check = 0U;
    if(enable) {
        bit_check = 1U;
    }

    ClearRegFieldQ(handle);
    status = Max96712AccessOneRegField(status, handle, REG_FIELD_ENABLE_LOCK,
             bit_check, REG_READ_MOD_WRITE_MODE);
    /* Validate HW has LOCK bit set */
    if (status == NVMEDIA_STATUS_OK) {
        status = Max96712AccessOneRegField(status, handle, REG_FIELD_ENABLE_LOCK
                 , 0x00U, REG_READ_MODE);
        if (status == NVMEDIA_STATUS_OK) {
            uint8_t reg_read = 0U;
            reg_read = ReadFromRegFieldQ(handle, (uint8_t){0U});
            if (reg_read != bit_check) {
                SIPL_LOG_ERR_STR("MAX96712: LOCK register write and read back mismatch");
                status = NVMEDIA_STATUS_ERROR;
            }
        }
    }
    return status;
}

/**
 * @brief Enables/disables CSI out
 *
 * - enables or disables REG_FIELD_BACKTOP_EN based on "enable" param by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_BACKTOP_EN,
 *   - (uint8_t)(enable ? 1U : 0U), REG_READ_MOD_WRITE_MODE)
 *  .
 * .
 * @param[in] handle DEVBLK handle
 * @param[in] enable if true enable, if false disable ERRB
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
EnableCSIOut(
    DevBlkCDIDevice const* handle,
    bool enable)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = Max96712AccessOneRegField(status, handle, REG_FIELD_BACKTOP_EN,
                                       (uint8_t)(enable ? 1U : 0U),
                                       REG_READ_MOD_WRITE_MODE);

    return status;
}

/**
 * @brief Will trigger deskew
 *
 * - extracts device driver handle from handle
 * - initialize DevBlkCDII2CReg deskewReg = {0x0903U, 0x00U}
 *
 * - Note: Trigger the initial deskew patterns two times
 *  -  to make sure Rx device recevies the patterns
 *  .
 * - repeat two times
 *  - for each PHY index
 *   - update reg address to
 *    -  deskewReg.address =
 *     - (deskewReg.address & 0xFF00U) +
 *     - (0x40U * phy_num) + 0x03U
 *     .
 *    .
 *   - read register content from deserializer by
 *    - status = DevBlkCDII2CPgmrReadUint8(
 *     - drvHandle->i2cProgrammer, deskewReg.address, &temp)
 *     .
 *    .
 *   - toggle bit (1U << 5U)
 *   - write it back to deserlalizer by
 *    - status = DevBlkCDII2CPgmrWriteUint8(
 *     - drvHandle->i2cProgrammer,
 *     - deskewReg.address, toUint8FromUint16(deskewReg.data))
 *     .
 *    .
 *   .
 *  - call to nvsleep(10000) before second trial
 *  .
 * @param[in] handle DEVBLK handle
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
TriggerDeskew(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t i = 0U;
    uint8_t phy_num = 0U;
    uint8_t temp;
    DevBlkCDII2CReg deskewReg = {0x0903U, 0x00U};

    /* Trigger the initial deskew patterns two times
     * to make sure Rx device recevies the patterns */
    for (i = 0U; i < 2U; i++) {
        for (phy_num = 0U; phy_num < MAX96712_MAX_NUM_PHY; phy_num++) {
            /* Update the register offset */
            deskewReg.address = (deskewReg.address & 0xFF00U) +
                                (0x40U * phy_num) + 0x03U;
            status = MAX96712ReadUint8Verify(handle,
                                             deskewReg.address,
                                             &temp);
            if (status == NVMEDIA_STATUS_OK) {
                deskewReg.data = temp;
                deskewReg.data ^= (1U << 5U);
                status = MAX96712WriteUint8Verify(handle,
                                                  deskewReg.address,
                                                  toUint8FromUint16(deskewReg.data));
            }
            if (status != NVMEDIA_STATUS_OK) {
                break;
            }
        }
        if (status != NVMEDIA_STATUS_OK) {
            break;
        }
        if ((i == 0U)) {
            /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
            (void)nvsleep(10000);
        }
    }

    return status;
}

/**
 * @brief Enables extra SMs
 *
 * - verifies not previous failue occured
 * - clear bitReg queue by
 *  - ClearRegFieldQ(handle)
 *  .
 * - read/mod/write 0x00U to REG_FIELD_INTR4 register by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_INTR4, 0x00U,
 *   - REG_READ_MOD_WRITE_MODE)
 *   .
 *  .
 * - clear bitReg queue by
 *  - ClearRegFieldQ(handle)
 *  .
 * - read/mod/write 0xFDU to REG_FIELD_INTR6 register by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_INTR6, 0xFDU,
 *   - REG_READ_MOD_WRITE_MODE
 *   .
 *  .
 * - clear bitReg queue by
 *  - ClearRegFieldQ(handle)
 *  .
 * - read/mod/write 0x0FU to REG_FIELD_INTR10 register by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_INTR10, 0x0FU
 *   - REG_READ_MOD_WRITE_MODE)
 *   .
 *  .
 * - clear bitReg queue by
 *  - ClearRegFieldQ(handle)
 *  .
 * - read/mod/write 0x01U to REG_FIELD_INTR12_MEM_ERR_OEN register by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_INTR12_MEM_ERR_OEN, 0x1U,
 *   - REG_READ_MOD_WRITE_MODE)
 *   .
 *  .
 * - read/mod/write 0x3FU to REG_FIELD_VIDEO_MASKED_OEN register by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_VIDEO_MASKED_OEN,
 *   - 0x3FU, REG_READ_MOD_WRITE_MODE)
 *   .
 *  .
 * - read/mod/write 0xFFU to REG_FIELD_CFGH_VIDEO_CRC0 register by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_CFGH_VIDEO_CRC0,
 *   - 0xFFU, REG_READ_MOD_WRITE_MODE)
 *   .
 *  .
 * - read/mod/write 0xFFU to REG_FIELD_CFGH_VIDEO_CRC1 register by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_CFGH_VIDEO_CRC1,
 *   - 0xFFU, REG_READ_MOD_WRITE_MODE)
 *   .
 *  .
 * .
 * @param[in] instatus previous status to be checked. Noting done if this is != NVMEDIA_STATUS_OK
 * @param[in] handle DEVBLK handle
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
EnableExtraSMs(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = instatus;

    if (status == NVMEDIA_STATUS_OK) {
        ClearRegFieldQ(handle);
        /* Disabling: SM2, SM13, SM14 and SM22 */
        status = Max96712AccessOneRegField(status, handle, REG_FIELD_INTR4, 0x00U,
                                           REG_READ_MOD_WRITE_MODE);

        /* Enabling: SM4, SM9 and SM20
         * Disabling: SM40 */
        ClearRegFieldQ(handle);
        status = Max96712AccessOneRegField(status, handle, REG_FIELD_INTR6, 0xFEU,
                                           REG_READ_MOD_WRITE_MODE);

        /* Enabling: SM3, SM5, SM35 */
        ClearRegFieldQ(handle);
        status = Max96712AccessOneRegField(status, handle, REG_FIELD_INTR10, 0x0FU,
                                           REG_READ_MOD_WRITE_MODE);

        ClearRegFieldQ(handle);
        status = Max96712AccessOneRegField(status, handle, REG_FIELD_INTR12_MEM_ERR_OEN,
                                           0x1U, REG_READ_MOD_WRITE_MODE);

        /* Enabling: SM15 */
        ClearRegFieldQ(handle);
        status = Max96712AccessOneRegField(status, handle, REG_FIELD_VIDEO_MASKED_OEN,
                                           0x3FU,
                                           REG_READ_MOD_WRITE_MODE);

        /* Enabling: SM4 */
        ClearRegFieldQ(handle);
        status = Max96712AccessOneRegField(status, handle, REG_FIELD_CFGH_VIDEO_CRC0,
                                           0xFFU,
                                           REG_READ_MOD_WRITE_MODE);

        /* Enabling: SM4 */
        ClearRegFieldQ(handle);
        status = Max96712AccessOneRegField(status, handle, REG_FIELD_CFGH_VIDEO_CRC1,
                                           0xFFU,
                                           REG_READ_MOD_WRITE_MODE);
    }

    return status;
}

/**
 * @brief Sets I2C slave time output
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - extract i2cPort from device driver handle context by
 *  - I2CPortMAX96712 i2cPort = drvHandle->ctx.i2cPort
 *  .
 * - verifies not previous failue occured
 * - for each link
 *  - if i2c port == CDI_MAX96712_I2CPORT_0
 *   - Note: using 16 ms timeout. This value is less than I2C_INTREG_SLV_0_TO
 *   - read/mod/write 0x5U to REG_FIELD_SLV_TO_P0_A + link offset register by
 *    - status = Max96712AccessOneRegFieldOffset(
 *     - status, handle, REG_FIELD_SLV_TO_P0_A,
 *     - i, 0x5U, REG_READ_MOD_WRITE_MODE)
 *     .
 *    .
 *   .
 *  - elif i2c port == CDI_MAX96712_I2CPORT_1
 *   - Note: 16 ms timeout. This value is less than I2C_INTREG_SLV_1_TO
 *   - read/mod/write 0x5U to REG_FIELD_SLV_TO_P1_A + link offset register by
 *    - status = Max96712AccessOneRegFieldOffset(
 *     - status, handle,
 *     - REG_FIELD_SLV_TO_P1_A, i, 0x5U, REG_READ_MOD_WRITE_MODE)
 *     .
 *    .
 *   .
 *  - else
 *   - error exit "MAX96712: I2c port not set"
 *   .
 *  .
 * @param[in] instatus previous status to be checked. Noting done if this is != NVMEDIA_STATUS_OK
 * @param[in] handle DEVBLK handle
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
SetI2CSlaveTimeOutput(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = instatus;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    I2CPortMAX96712 i2cPort = drvHandle->ctx.i2cPort;
    uint8_t i = 0U;

    if (status == NVMEDIA_STATUS_OK) {
        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            /* Update I2C slave timeout */
            if (i2cPort == CDI_MAX96712_I2CPORT_0) {
                status = Max96712AccessOneRegFieldOffset(status, handle,
                                                         REG_FIELD_SLV_TO_P0_A,
                                                         i,
                                                         0x5U,  /* 16 ms timeout. This value is
                                                                 * less than I2C_INTREG_SLV_0_TO */
                                                         REG_READ_MOD_WRITE_MODE);
            } else if (i2cPort == CDI_MAX96712_I2CPORT_1)  {
                status = Max96712AccessOneRegFieldOffset(status, handle,
                                                         REG_FIELD_SLV_TO_P1_A,
                                                         i,
                                                         0x5U, /* 16 ms timeout. This value is
                                                                * less than I2C_INTREG_SLV_1_TO */
                                                         REG_READ_MOD_WRITE_MODE);
            } else {
                SIPL_LOG_ERR_STR("MAX96712: I2c port not set");
                status = NVMEDIA_STATUS_ERROR;
            }
        }
    }

    return status;
}

/**
 * @brief Verifies registers and register groups for SMs are enabled.
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - read ARQ1 registers value by
 *  - status = DevBlkCDII2CPgmrReadArray(drvHandle->i2cProgrammer,
 *   - &readARQ1RegData);
 *   .
 *  - for each value read
 *   - if ARQ1Regs[index].data & 0x2U !- 0x2U
 *    - log error
 *    .
 *   .
 *  .
 * - read VID RX 0/1/2/3/4/5/6/7 registers value by
 *  - status = DevBlkCDII2CPgmrReadArray(drvHandle->i2cProgrammer,
 *   - &readVIDEO_RX0RegData);
 *   .
 *  - for each value read
 *   - if VIDEO_RX0Regs[index].data & 0x2U !- 0x2U
 *    - log error
 *    .
 *   .
 *  .
 * - read TX/RX_CRC_EN bit values by
 *  - status = DevBlkCDII2CPgmrReadArray(drvHandle->i2cProgrammer,
 *   - &readTX_RX_CRC_ENRegData);
 *   .
 *  - for each value read
 *   - if TX_RX_CRC_ENRegs[index].data & 0xC0U !- 0xC0U
 *    - log error
 *    .
 *   .
 *  .
 * - read VID_PXL_CRC_ERR_OEN register value by
 *  - status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
 *   - VID_PXL_CRC_ERR_OEN, &data);
 *  - if data & 0x0FU !- 0x0FU
 *   - log error
 *   .
 *  .
 * - read DEC_ERR_OEN register value by
 *  - status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
 *   - DEC_ERR_OEN, &data);
 *  - if data & 0x0FU !- 0x0FU
 *   - log error
 *   .
 *  .
 * - read IDLE_ERR_OEN register value by
 *  - status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
 *   - IDLE_ERR_OEN, &data);
 *  - if data & 0x0FU !- 0x00U
 *   - log error
 *   .
 *  .
 * - read REG5 register by
 *  - status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
 *   - REG5, &data);
 *  - if data & 0xE0U !- 0xC0U
 *   - log error
 *   .
 *  .
 * @param[in] handle DEVBLK handle
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
VerifyRegGroupSMs(DevBlkCDIDevice const *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t data = 0U;

    /* MAX_RT_ERR_OEN in ARQ1 registers for CFGL_A/B/C/D GPIO, CFGC_A/B/C/D GPIO CC_0/1/2 */
    DevBlkCDII2CReg ARQ1Regs[] = {
        {0x0A6U, 0x0U},
        {0x0AEU, 0x0U},
        {0x0B6U, 0x0U},
        {0x0BEU, 0x0U},
        {0x506U, 0x0U},
        {0x516U, 0x0U},
        {0x526U, 0x0U},
        {0x536U, 0x0U},
        {0x566U, 0x0U},
        {0x576U, 0x0U},
        {0x586U, 0x0U},
        {0x596U, 0x0U},
        {0x5A6U, 0x0U},
        {0x5B6U, 0x0U},
        {0x5C6U, 0x0U},
        {0x5D6U, 0x0U}
    };
    DevBlkCDII2CRegListWritable const readARQ1RegData = {
        .regs = ARQ1Regs,
        .numRegs = (uint32_t)(sizeof(ARQ1Regs) /
                              sizeof(ARQ1Regs[0])),
    };

    /* LINE_CRC_EN and SEQ_MISS_EN in VID_RX 0/1/2/3/4/5/6/7 registers */
    DevBlkCDII2CReg VIDEO_RX0Regs[] = {
        {0x100U, 0x0U},
        {0x112U, 0x0U},
        {0x124U, 0x0U},
        {0x136U, 0x0U},
        {0x148U, 0x0U},
        {0x160U, 0x0U},
        {0x172U, 0x0U},
        {0x184U, 0x0U}
    };
    DevBlkCDII2CRegListWritable const readVIDEO_RX0RegData = {
        .regs = VIDEO_RX0Regs,
        .numRegs = (uint32_t)(sizeof(VIDEO_RX0Regs) /
                              sizeof(VIDEO_RX0Regs[0])),
    };

    /* TX/RX_CRC_EN bits in CFGI_A/B/C/D INFOFR, CFGL_A/B/C/D GPIO CFGC_A/B/C/D CC_0/1/2 */
    DevBlkCDII2CReg TX_RX_CRC_EN_Regs[] = {
        {0x070U, 0x0U},
        {0x074U, 0x0U},
        {0x078U, 0x0U},
        {0x07CU, 0x0U},
        {0x0A0U, 0x0U},
        {0x0A8U, 0x0U},
        {0x0B0U, 0x0U},
        {0x0B8U, 0x0U},
        {0x500U, 0x0U},
        {0x510U, 0x0U},
        {0x520U, 0x0U},
        {0x530U, 0x0U},
        {0x560U, 0x0U},
        {0x570U, 0x0U},
        {0x580U, 0x0U},
        {0x590U, 0x0U},
        {0x5A0U, 0x0U},
        {0x5B0U, 0x0U},
        {0x5C0U, 0x0U},
        {0x5D0U, 0x0U}
    };
    DevBlkCDII2CRegListWritable const readTX_RX_CRC_EN_RegData = {
        .regs = TX_RX_CRC_EN_Regs,
        .numRegs = (uint32_t)(sizeof(TX_RX_CRC_EN_Regs) /
                              sizeof(TX_RX_CRC_EN_Regs[0])),
    };

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712ReadArrayVerify(handle, &readARQ1RegData);
        if (status == NVMEDIA_STATUS_OK) {
            for (uint32_t index= 0U; index < readARQ1RegData.numRegs; index++) {
                if ((ARQ1Regs[index].data & 0x2U) != 0x2U) {
                    SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: MAX_RT_ERR_OEN is not enabled at addr:",
                                              (uint32_t)ARQ1Regs[index].address);
                    status = NVMEDIA_STATUS_ERROR;
                }
            }
            if (status == NVMEDIA_STATUS_OK) {
                PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: SM5: Verified MAX_RT_ERR_OEN "
                                            "registers enabled.\n");
            }
        }
    }
    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712ReadArrayVerify(handle, &readVIDEO_RX0RegData);
        if (status == NVMEDIA_STATUS_OK) {
            for (uint32_t index = 0U; index < readVIDEO_RX0RegData.numRegs; \
                ++index) {
                if ((VIDEO_RX0Regs[index].data & 0x2U) != 0x2U) {
                    SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: LINE_CRC_EN is not enabled at addr:",
                                              (uint32_t)VIDEO_RX0Regs[index].address);
                    status = NVMEDIA_STATUS_ERROR;
                }
                if ((VIDEO_RX0Regs[index].data & 0x10U) != 0x10U) {
                    SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: SEQ_MISS_EN is not enabled at addr:",
                                              (uint32_t)VIDEO_RX0Regs[index].address);
                    status = NVMEDIA_STATUS_ERROR;
                }
            }
            if (status == NVMEDIA_STATUS_OK) {
                PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: SM4: Verified VIDEO_RX0 "
                                            "registers enabled.\n");
            }
        }
    }
    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712ReadArrayVerify(handle, &readTX_RX_CRC_EN_RegData);
        if (status == NVMEDIA_STATUS_OK) {
            for (uint32_t index = 0U; index < readTX_RX_CRC_EN_RegData.numRegs; \
                ++index) {
                if ((TX_RX_CRC_EN_Regs[index].data & 0xC0U) != 0xC0U) {
                    SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: RX/TX_CRC_EN is not enabled at addr:",
                                              (uint32_t)TX_RX_CRC_EN_Regs[index].address);
                    status = NVMEDIA_STATUS_ERROR;
                }
            }
            if (status == NVMEDIA_STATUS_OK) {
                PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: SM3: Verified RX/TX_CRC_EN "
                                            "registers enabled.\n");
            }
        }
    }
    if (status == NVMEDIA_STATUS_OK) {
        /* Register address for VID_PXL_CRC_ERR_OEN_A/B/C/D and
           MEM_ECC_ERR2/1_OEN bits*/
        uint16_t const VID_PXL_CRC_ERR_OEN = 0x044U;
        status = MAX96712ReadUint8Verify(handle, VID_PXL_CRC_ERR_OEN, &data);
        if (status == NVMEDIA_STATUS_OK) {
            if ((data & 0xCFU) != 0x8FU) {
                SIPL_LOG_ERR_STR("MAX96712: Bits in VID_PXL_CRC_ERR_OEN not as expected.\n");
                status = NVMEDIA_STATUS_ERROR;
            }
            else {
                PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: SM4,41: Verified VID_PXL_CRC_ERR_OEN"
                                            " and MEM_ECC_ERR2 bits are enabled.\n");
            }
        }
    }
    if (status == NVMEDIA_STATUS_OK) {
        /* Register address for VDDCMP_INT_OEN and VDDBAD_INT_OEN bits */
        uint16_t const PWR_STATUS_OEN = 0x046U;
        status = MAX96712ReadUint8Verify(handle, PWR_STATUS_OEN, &data);
        if (status == NVMEDIA_STATUS_OK) {
            if ((data & 0xA0U) != 0xA0U) {
                SIPL_LOG_ERR_STR("MAX96712: PWR_STATUS_OEN bits are not enabled.\n");
                status = NVMEDIA_STATUS_ERROR;
            } else {
                PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: SM15: Verified VDDCMP_INT_OEN and "
                                            "VDDBAD_INT_OEN bits are enabled.\n");
            }
        }
    }
    if (status == NVMEDIA_STATUS_OK) {
        /* Verify AUTO_ERR_RST_EN is '0' so that DEC_ERR and IDLE_ERR
           registers are not automatically reset after the errors occur. */
        /* Register address for AUTO_ERR_RST_EN bit */
        uint16_t const AUTO_ERR_RST_EN = 0x023U;
        status = MAX96712ReadUint8Verify(handle, AUTO_ERR_RST_EN, &data);
        if (status == NVMEDIA_STATUS_OK) {
            if ((data & 0x08U) != 0U) {
                SIPL_LOG_ERR_STR("MAX96712: AUTO_ERR_RST_EN is enabled.\n");
                status = NVMEDIA_STATUS_ERROR;
            } else {
                PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: SM20: Verified AUTO_ERR_RST_EN "
                                            "register enabled.\n");
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            /* Register address for DEC_ERR_OEN_A/B/C/D bits */
            status = MAX96712ReadUint8Verify(handle, REG_INTR2, &data);
            if (status == NVMEDIA_STATUS_OK) {
                if ((data & 0x0FU) != 0x0FU) {
                    SIPL_LOG_ERR_STR("MAX96712: DEC_ERR_OEN is not enabled.\n");
                    status = NVMEDIA_STATUS_ERROR;
                } else {
                    PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: SM20: Verified DEC_ERR_OEN "
                                                "register enabled.\n");
                }
            }
        }
    }
    if (status == NVMEDIA_STATUS_OK) {
        /* Register address for IDLE_ERR_OEN_A/B/C/D and
           PKT_CNT_OEN_A/B/C/D bits */
        status = MAX96712ReadUint8Verify(handle, REG_INTR8, &data);
        if (status == NVMEDIA_STATUS_OK) {
            if ((data & 0x0FU) != 0U) {
                SIPL_LOG_ERR_STR("MAX96712: IDLE_ERR_OEN is enabled.\n");
                status = NVMEDIA_STATUS_ERROR;
            } else if ((data & 0xF0U) != 0U) {
                SIPL_LOG_ERR_STR("MAX96712: PKT_CNT_OEN is enabled.\n");
                status = NVMEDIA_STATUS_ERROR;
            } else {
                PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: SM19,20: Verified IDLE_ERR_OEN and "
                                            "PKT_CNT_OEN bits disabled.\n");
            }
        }
    }
    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712ReadUint8Verify(handle, REG5, &data);
        if (status == NVMEDIA_STATUS_OK) {
            /* Mask to check for LOCK_EN, ERRB_EN and LOCK_CFG bits */
            if ((data & 0xE0U) != 0xC0U) {
                SIPL_LOG_ERR_STR("MAX96712: REG5 is not configured as expected.\n");
                status = NVMEDIA_STATUS_ERROR;
            } else {
                PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: LOCK_EN, ERRB_EN, LOCK_CFG"
                                            "set as expected.\n");
            }
        }
    }

    return status;
}

/**
 * @brief Verifies SMs are enabled
 *
 * - reads REG_FIELD_INTR4 register value by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_INTR4, 0x00U,
 *   - REG_READ_MODE)
 *   .
 *  .
 * - fetch reg_read from bitReg queue by
 *  - reg_read = ReadFromRegFieldQ(handle, 0U)
 *  .
 * - if value !- 0x00U
 *  - exit error: NVMEDIA_STATUS_ERROR
 *  .
 * - reads REG_FIELD_INTR6 register value by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_INTR6, 0x00U, REG_READ_MODE)
 *   .
 *  .
 * - fetch reg_read from bitReg queue by
 *  - reg_read = ReadFromRegFieldQ(handle, 0U)
 *  .
 * - if reg_read !- 0xFDU
 *  - exit error: NVMEDIA_STATUS_ERROR
 *  .
 * - reads REG_FIELD_INTR10 register value by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_INTR10, 0x00U, REG_READ_MODE)
 *   .
 *  .
 * - fetch value from bitReg queue by
 * - reg_read = ReadFromRegFieldQ(handle, 0U)
 * - if reg_read !- 0x0FU
 *  - exit error: NVMEDIA_STATUS_ERROR
 *  .
 * - reads REG_FIELD_INTR12_MEM_ERR_OEN register value
 * - fetch value from bitReg queue
 *  - reg_read = ReadFromRegFieldQ(handle, 0U)
 *  .
 * - if value !- 0x01U
 *  - exit error: NVMEDIA_STATUS_ERROR
 *  .
 * - reads REG_FIELD_VIDEO_MASKED_OEN register value by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_VIDEO_MASKED_OEN, 0x00U, REG_READ_MODE)
 *   .
 *  .
 * - fetch value from bitReg queue
 * - if value !- 0x3FU
 *  -  exit error NVMEDIA_STATUS_ERROR
 *  .
 * - reads REG_FIELD_CFGH_VIDEO_CRC0 register value by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_CFGH_VIDEO_CRC0, 0x0U, REG_READ_MODE)
 *   .
 *  .
 * - fetch value from bitReg queue
 * - if value !- 0xFFU
 *  -  exit error NVMEDIA_STATUS_ERROR
 *  .
 * - reads REG_FIELD_CFGH_VIDEO_CRC1 register value by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_CFGH_VIDEO_CRC1, 0x0U, REG_READ_MODE)
 *   .
 *  .
 * - fetch value from bitReg queue
 * - if value !- 0xFFU
 *  -  exit error NVMEDIA_STATUS_ERROR
 *  .
 *  - read registers associated with SMs by
 *   - status = VerifyRegGroupSM(handle)
 *  .
 * @param[in] handle DEVBLK handle
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
VerifySMsEnabled(DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    uint8_t reg_read = 0U;
    status = Max96712AccessOneRegField(status, handle, REG_FIELD_INTR4, 0x00U,
            REG_READ_MODE);
    reg_read = ReadFromRegFieldQ(handle, 0U);
    if (reg_read != 0x00U) {
        status = NVMEDIA_STATUS_ERROR;
    } else {
        PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: SM2,13,14: INTR4 value verified. "
                                    "Value: %x\n", reg_read);
        status = Max96712AccessOneRegField(status, handle, REG_FIELD_INTR6, 0x00U,
                REG_READ_MODE);

        reg_read = ReadFromRegFieldQ(handle, 0U);
        if (reg_read != 0xFEU) {
            status = NVMEDIA_STATUS_ERROR;
        } else {
            PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: SM1,4,9,34,40: INTR6 value verified. "
                                        "Value: %x\n", reg_read);
            status = Max96712AccessOneRegField(status, handle, REG_FIELD_INTR10, 0x00U,
                    REG_READ_MODE);
            reg_read = ReadFromRegFieldQ(handle, 0U);
            if (reg_read != 0x0FU) {
                status = NVMEDIA_STATUS_ERROR;
            } else {
                PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: SM3,5,35: INTR10 value verified. "
                                            "Value: %x\n", reg_read);
                status = Max96712AccessOneRegField(status, handle, REG_FIELD_INTR12_MEM_ERR_OEN,
                                                   0x0U, REG_READ_MODE);
                reg_read = ReadFromRegFieldQ(handle, 0U);
                if (reg_read != 0x01U) {
                    status = NVMEDIA_STATUS_ERROR;
                } else {
                    PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: SM22: INTR12 MEM_ERR_OEN value verified."
                                                " Value: %x\n", reg_read);
                    status = Max96712AccessOneRegField(status, handle, REG_FIELD_VIDEO_MASKED_OEN,
                                                       0x00U,
                                                       REG_READ_MODE);
                    reg_read = ReadFromRegFieldQ(handle, 0U);
                    if (reg_read != 0x3FU) {
                        status = NVMEDIA_STATUS_ERROR;
                    } else {
                        PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: SM15: VIDEO_MASKED_OEN value "
                                                    "verified. Value: %x\n", reg_read);
                        status = Max96712AccessOneRegField(status, handle,
                                                           REG_FIELD_CFGH_VIDEO_CRC0,
                                                           0x0U, REG_READ_MODE);
                        reg_read = ReadFromRegFieldQ(handle, 0U);
                        if (reg_read != 0xFFU) {
                            status = NVMEDIA_STATUS_ERROR;
                        } else {
                            status = Max96712AccessOneRegField(status, handle,
                                                               REG_FIELD_CFGH_VIDEO_CRC1,
                                                               0x0U, REG_READ_MODE);
                            reg_read = ReadFromRegFieldQ(handle, 0U);
                            if (reg_read != 0xFFU) {
                                status = NVMEDIA_STATUS_ERROR;
                            } else {
                                status = VerifyRegGroupSMs(handle);
                            }
                        }
                    }
                }
            }
        }
    }
    return status;
}

/**
 * @brief Sets the I2C port
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - extract i2cPort from device driver handle context by
 *  - I2CPortMAX96712 i2cPort = drvHandle->ctx.i2cPort
 *  .
 * - extract linkMask from device driver handle context by
 *  - uint8_t linkMask = drvHandle->ctx.linkMask
 *  .
 * - verifies not previous failue occured
 * - for each link
 *  - if MAX96712_IS_GMSL_LINKMASK_SET for this link
 *   - if IsGMSL2Mode for this link
 *    - if (drvHandle->ctx.gmslMode[i] == CDI_MAX96712_GMSL1_MODE)
 *     - set REG_FIELD_I2C_PORT_GMSL1_PHY_A register by
 *      - status = Max96712AccessOneRegFieldOffset(
 *       - status, handle, REG_FIELD_I2C_PORT_GMSL1_PHY_A,
 *       - i, (uint8_t)((i2cPort == CDI_MAX96712_I2CPORT_0) ? 0U : 1U),
 *       - REG_READ_MOD_WRITE_MODE)
 *       .
 *      .
 *     .
 *    - elif (IsGMSL2Mode(drvHandle->ctx.gmslMode[i]))
 *     - Disable connection from both port 0/1  by
 *      - status = Max96712AccessOneRegFieldOffset(
 *       - status, handle, REG_FIELD_DIS_REM_CC_A,
 *       - i, 0x3U, REG_READ_MOD_WRITE_MODE)
 *       .
 *      .
 *     - Select port 0 or 1 over the link BY
 *      - status = Max96712AccessOneRegFieldOffset(
 *       - status, handle, REG_FIELD_SEC_XOVER_SEL_PHY_A,
 *       - i, (uint8_t)((i2cPort == CDI_MAX96712_I2CPORT_0) ? 0U : 1U),
 *       - REG_READ_MOD_WRITE_MODE)
 *       .
 *      .
 *     - Enable connection from port 0 or 1 BY
 *      - status = Max96712AccessOneRegFieldOffset(
 *       - status, handle, REG_FIELD_DIS_REM_CC_A,
 *       - i, (uint8_t)((i2cPort == CDI_MAX96712_I2CPORT_0) ? 2U : 1U),
 *       - REG_READ_MOD_WRITE_MODE)
 *       .
 *      .
 *     - set slave timeout setting bt
 *      -  status = SetI2CSlaveTimeOutput(status, handle)
 *      .
 *     .
 *    - else
 *     - error exit "MAX96712: Unexpected GMSL mode, I2c port not set"
 *     .
 *    .
 *   .
 *  .
 * @param[in] instatus previous status to be checked. Noting done if this is != NVMEDIA_STATUS_OK
 * @param[in] handle DEVBLK handle
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
SetI2CPort(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = instatus;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    I2CPortMAX96712 i2cPort = drvHandle->ctx.i2cPort;
    uint8_t linkMask = drvHandle->ctx.linkMask;
    uint8_t i = 0U;

    if (status == NVMEDIA_STATUS_OK) {
        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            if (MAX96712_IS_GMSL_LINKMASK_SET(linkMask, i)) {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
                if (drvHandle->ctx.gmslMode[i] == CDI_MAX96712_GMSL1_MODE) {
                    status = Max96712AccessOneRegFieldOffset(status, handle,
                                                             REG_FIELD_I2C_PORT_GMSL1_PHY_A,
                                                             i,
                                                             (uint8_t)((i2cPort ==
                                                             CDI_MAX96712_I2CPORT_0) ? 0U : 1U),
                                                             REG_READ_MOD_WRITE_MODE);
                } else
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
                if (IsGMSL2Mode(drvHandle->ctx.gmslMode[i])) {
                    /* Disable connection from both port 0/1 */
                    status = Max96712AccessOneRegFieldOffset(status, handle, REG_FIELD_DIS_REM_CC_A,
                                                             i,
                                                             0x3U,
                                                             REG_READ_MOD_WRITE_MODE);

                    /* Select port 0 or 1 over the link */
                    status = Max96712AccessOneRegFieldOffset(status, handle,
                                                             REG_FIELD_SEC_XOVER_SEL_PHY_A,
                                                             i,
                                                             (uint8_t)((i2cPort ==
                                                             CDI_MAX96712_I2CPORT_0) ? 0U : 1U),
                                                             REG_READ_MOD_WRITE_MODE);

                    /* Enable connection from port 0 or 1 */
                    status = Max96712AccessOneRegFieldOffset(status, handle,
                                                             REG_FIELD_DIS_REM_CC_A,
                                                             i,
                                                             (uint8_t)((i2cPort ==
                                                             CDI_MAX96712_I2CPORT_0) ? 2U : 1U),
                                                             REG_READ_MOD_WRITE_MODE);

                    status = SetI2CSlaveTimeOutput(status, handle);
                } else {
                    SIPL_LOG_ERR_STR("MAX96712: Unexpected GMSL mode, I2c port not set");
                    status = NVMEDIA_STATUS_ERROR;
                }
            }
        }
    }

    return status;
}

/**
 * @brief  Sets FSYNC Mode External
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - initialize
 *  - DevBlkCDII2CReg const setGpio2Mode[] = {{0x0306U, 0x83U}}
 *  .
 * - initialize
 *  - DevBlkCDII2CReg const setGMSL2PerLinkExtFsyncModeRegs[MAX96712_MAX_NUM_LINK] = {
 *   -    {0x0307U, 0xA0U | (uint32_t)CDI_MAX96712_GPIO_2},
 *   -    {0x033DU, 0x20U | (uint32_t)CDI_MAX96712_GPIO_2},
 *   -    {0x0374U, 0x20U | (uint32_t)CDI_MAX96712_GPIO_2},
 *   -    {0x03AAU, 0x20U | (uint32_t)CDI_MAX96712_GPIO_2},
 *   .
 *  - }
 *  .
 * - for each link
 *  - if (MAX96712_IS_GMSL_LINK_SET(link, i))
 *   - write setExtFsyncModeReg content to deserializer by
 *    - status = DevBlkCDII2CPgmrWriteUint8(
 *      - drvHandle->i2cProgrammer, setGpio2Mode[0].address,
 *      - toUint8FromUint16(setGpio2Mode[0].data))
 *      .
 *   - set
 *    - enableGpiGpoReg.data = 0x65U;
 *    - enableGpiGpoReg.address += ((uint16_t)link_index) << 8U;
 *    .
 *   - write enableGpiGpoReg  content by
 *    - status = DevBlkCDII2CPgmrWriteUint8(
 *     - drvHandle->i2cProgrammer, enableGpiGpoReg.address,
 *     - (uint8_t)enableGpiGpoReg.data)
 *     .
 *    .
 *   - nvsleep(10000)
 * @param[in] handle DEVBLK handle
 * @param[in] link   bit mask of links to process
 * @param[in] gmslMode GMSL mode
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
static NvMediaStatus
SetFSYNCMode_External(
    DevBlkCDIDevice const *handle,
    LinkMAX96712 const link,
    GMSLModeMAX96712 gmslMode)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 *drvHandle = getHandlePrivMAX96712(handle);

    DevBlkCDII2CReg const setGpio2Mode[] = {{0x0306U, 0x83U}};
    DevBlkCDII2CReg const setGMSL2PerLinkExtFsyncModeRegs[4] = {
        {0x0307U, 0xA0U | (uint32_t)CDI_MAX96712_GPIO_2},
        {0x033DU, 0x20U | (uint32_t)CDI_MAX96712_GPIO_2},
        {0x0374U, 0x20U | (uint32_t)CDI_MAX96712_GPIO_2},
        {0x03AAU, 0x20U | (uint32_t)CDI_MAX96712_GPIO_2},
    };
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    DevBlkCDII2CReg setExtFsyncModeReg = {0x04A0U, 0x08U};
    DevBlkCDII2CReg enableGpiGpoReg = {0x0B08U, 0x00U};
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
    uint8_t i = 0U;

        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
                if (gmslMode == CDI_MAX96712_GMSL1_MODE) {
                    status = MAX96712WriteUint8Verify(handle,
                                       setExtFsyncModeReg.address,
                                       (uint8_t)setExtFsyncModeReg.data);
                    if (status == NVMEDIA_STATUS_OK) {
                         enableGpiGpoReg.data = 0x65U;
                         enableGpiGpoReg.address += ((uint16_t)i) << 8U;

                         status = MAX96712WriteUint8Verify(handle,
                                                enableGpiGpoReg.address,
                                                (uint8_t)enableGpiGpoReg.data);
                        if (status == NVMEDIA_STATUS_OK) {
                            /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
                            (void)nvsleep(10000);
                        }
                    }
                } else
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
                {
                    status = MAX96712WriteUint8Verify(handle,
                                         setGpio2Mode[0].address,
                                         toUint8FromUint16(setGpio2Mode[0].data));
                    if (status == NVMEDIA_STATUS_OK) {
                        status = MAX96712WriteUint8Verify(handle,
                                    setGMSL2PerLinkExtFsyncModeRegs[i].address,
                                    (uint8_t)(setGMSL2PerLinkExtFsyncModeRegs[i].data
                                    & 0xFFU));
                    }
                }

                if (status != NVMEDIA_STATUS_OK) {
                    break;
                } else {
                    /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
                    (void)nvsleep(10000);
                    drvHandle->ctx.FSyncMode[i] = CDI_MAX96712_FSYNC_EXTERNAL;
                    drvHandle->ctx.FsyncExtGpioNum[i] = CDI_MAX96712_GPIO_2;
                }
            }
        }
    /* NOP: just to make sure misra_cpp_2008_rule_2_7_violation is suppressed */
    (void) gmslMode;

    return status;
}

/**
 * @brief Check Valid FPS
 *
 * - if ((fps == 0U) || ((manualFSyncFPS != 0U) && (manualFSyncFPS != fps)))
 *  - Error: NVMEDIA_STATUS_NOT_SUPPORTED
 *  .
 * - else
     - return  NVMEDIA_STATUS_OK;
 *
 * @param[in] fps
 * @param[in] manualFSyncFPS
 * @return NVMEDIA_STATUS_OK condition is false, otherwise an error code  */
static NvMediaStatus
CheckValidFPS(
    uint32_t fps,
    uint32_t manualFSyncFPS)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if ((fps == 0U) || ((manualFSyncFPS != 0U) && (manualFSyncFPS != fps))) {
        status = NVMEDIA_STATUS_NOT_SUPPORTED;
    }

    return status;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
static NvMediaStatus
Max96712EnableGpioRegsForGMSL1Manual(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DevBlkCDII2CReg enableGpiGpoReg = {0x0B08U, 0x35U};
    uint16_t i = 0U;

    if (((link <= CDI_MAX96712_LINK_3) && !(MAX96712_IS_MULTIPLE_GMSL_LINK_SET(link))) ||
        (link == CDI_MAX96712_LINK_ALL)) {

        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
           enableGpiGpoReg.address += (uint16_t)i << (uint16_t)8U;

           status = MAX96712WriteUint8Verify(handle,
                                             enableGpiGpoReg.address,
                                             (uint8_t)enableGpiGpoReg.data);
           if (status != NVMEDIA_STATUS_OK) {
              break;
           }

           /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
           (void)nvsleep(10000);
        }
    }

    return status;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/**
 * @brief Set FSYNC Mode_OSCManual
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - initialize
 *  - DevBlkCDII2CReg fsyncPeriodRegs[3] = {
 *   - Note: Calculate FRSYNC period H. don't move position
 *   -   {0x04A7U, 0x00U},
 *   - Note: Calculate FRSYNC period M. don't move position
 *   -   {0x04A6U, 0x00U},
 *   - Note: Calculate FRSYNC period L. don't move position
 *   -   {0x04A5U, 0x00U},
 *   .
 *  -  }
 *  .
 * - initialized wrapper for fsyncPeriodRegs
 *  - DevBlkCDII2CRegList fsyncPeriod = {
 *   - .regs = fsyncPeriodRegs,
 *   - .numRegs = (uint32_t)(sizeof(fsyncPeriodRegs) /
 *    - sizeof(fsyncPeriodRegs[0])),
 *    .
 *   .
 *  - }
 *  .
 * - initialize
 *  - DevBlkCDII2CReg setOSCManualFsyncModeRegs[] = {
 *   - Note: Set FSYNC to GMSL1 type then delay 10ms
 *   - {0x04AFU, 0x40U, 0x2710U},
 *   - Note: Set FSYNC to manual mode then delay 10ms
 *   - {0x04A0U, 0x00U, 0x2710U},
 *   - Note: Turn off auto master link selection then delay 10ms
 *   - {0x04A2U, 0x00U, 0x2710U},
 *   - Note: Disable overlap window then delay 10ms
 *   - {0x04AAU, 0x00U, 0x2710U},
 *   - Note: Disable overlap window then delay 10ms
 *   - {0x04ABU, 0x00U, 0x2710U},
 *   - Note: Disable overlap window then delay 10ms
 *   - {0x04A8U, 0x00U, 0x2710U},
 *   - Note: Disable overlap window then delay 10ms
 *   - {0x04A9U, 0x00U, 0x2710U},
 *   .
 *  - }
 *  .
 * - initialize wrapper for
 *  - DevBlkCDII2CRegList setOSCManualFsyncMode = {
 *   - .regs = setOSCManualFsyncModeRegs,
 *   - .numRegs = (uint32_t)(sizeof(setOSCManualFsyncModeRegs) /
 *    - sizeof(setOSCManualFsyncModeRegs[0])),
 *    .
 *   .
 *  - }
 *  .
 * - initialize
 *  - Note:    GPIO ID 20
 *  - DevBlkCDII2CReg setTxIDIntReg = {
 *   -  0x04B1, (uint16_t)((uint16_t)CDI_MAX96712_GPIO_20 << 3U)
 *   .
 *  - }
 *  .
 * - call CheckValidFPS to verify params by
 *  - verify  (CheckValidFPS(fps, drvHandle->ctx.manualFSyncFPS) == NVMEDIA_STATUS_OK)
 * - if link in IsGMSL2Mode (IsGMSL2Mode(gmslMode))
 *  -  modify to Set FSYNC to GMSL2 type
 *   - setOSCManualFsyncModeRegs[0].data |= ((uint16_t)1U << (uint16_t)7U);
 *   .
 *  .
 * - write setOSCManualFsyncMode to deserializer by
 *  - status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &setOSCManualFsyncMode)
 *  .
 * - if link in IsGMSL2Mode (IsGMSL2Mode(gmslMode))
 *  - write setTxIDIntReg to deserializer by
 *   - status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
 *    - setTxIDIntReg.address, (uint8_t)setTxIDIntReg.data)
 *    .
 *   .
 *  .
 * - calculate frsync high period
 *  - fsyncPeriodRegs[0].data =
 *   - (uint16_t)(((MAX96712_OSC_MHZ * 1000U * 1000U / fps) >> 16U) & 0xFFU);
 *   .
 *  .
 * - calculate frsync middle period
 *  - fsyncPeriodRegs[1].data =
 *   - (uint16_t)(((MAX96712_OSC_MHZ * 1000U * 1000U / fps) >> 8U) & 0xFFU);
 *   .
 *  .
 * - calculate frsync low period
 *  - fsyncPeriodRegs[2].data =
 *   - (uint16_t)((MAX96712_OSC_MHZ * 1000U * 1000U / fps) & 0xFFU);
 *   .
 *  .
 * - write fsyncPeriodRegs to deserializer by
 *  - status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &fsyncPeriod)
 *  .
 * - update drvHandle->ctx.manualFSyncFPS = (uint16_t)fps
 *
 * @param[in] handle DEVBLK handle
 * @param[in] fps frames per seconds
 * @param[in] link  link mask
 * @param[in] gmslMode GMSL mode
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code  */
static NvMediaStatus
SetFSYNCMode_OSCManual(
    DevBlkCDIDevice const* handle,
    uint32_t fps,
    LinkMAX96712 const link,
    GMSLModeMAX96712 gmslMode)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 *drvHandle = getHandlePrivMAX96712(handle);
    DevBlkCDII2CReg fsyncPeriodRegs[] = {
        {0x04A7U, 0x00U}, /* Calculate FRSYNC period H. don't move position */
        {0x04A6U, 0x00U}, /* Calculate FRSYNC period M. don't move position */
        {0x04A5U, 0x00U}, /* Calculate FRSYNC period L. don't move position */
    };
    DevBlkCDII2CRegList fsyncPeriod = {
        .regs = fsyncPeriodRegs,
        .numRegs = (uint32_t)(sizeof(fsyncPeriodRegs) /
                              sizeof(fsyncPeriodRegs[0])),
    };

    DevBlkCDII2CReg setOSCManualFsyncModeRegs[] = {
        {0x04AFU, 0x40U, 0x2710U}, /* Set FSYNC to GMSL1 type then delay 10ms*/
        {0x04A0U, 0x04U, 0x2710U}, /* Set FSYNC to manual mode output to GPIO then delay 10ms*/
        {0x04A2U, 0x00U, 0x2710U}, /* Turn off auto master link selection then delay 10ms*/
        {0x04AAU, 0x00U, 0x2710U}, /* Disable overlap window then delay 10ms*/
        {0x04ABU, 0x00U, 0x2710U}, /* Disable overlap window then delay 10ms*/
        {0x04A8U, 0x00U, 0x2710U}, /* Disable error threshold then delay 10ms*/
        {0x04A9U, 0x00U, 0x2710U}, /* Disable error threshold then delay 10ms*/
    };
    DevBlkCDII2CRegList setOSCManualFsyncMode = {
        .regs = setOSCManualFsyncModeRegs,
        .numRegs = (uint32_t)(sizeof(setOSCManualFsyncModeRegs) /
                              sizeof(setOSCManualFsyncModeRegs[0])),
    };
    DevBlkCDII2CReg setTxIDIntReg = {0x04B1, (uint16_t)(
                      (uint16_t)CDI_MAX96712_GPIO_20 << 3U)}; /* GPIO ID 20 */
    if (CheckValidFPS(fps, drvHandle->ctx.manualFSyncFPS) != NVMEDIA_STATUS_OK) {
          SIPL_LOG_ERR_STR("MAX96712: Incorrect fsync frequencies requested");
          status = NVMEDIA_STATUS_NOT_SUPPORTED;
    }

    /* MAXIM doesn't recommend to use auto or semi-auto mode for the safety concern.
     * If the master link is lost, the frame sync will be lost for other links in both modes.
     * Instead the manual mode with OSC in MAX96712 is recommended.
    */

    if ((status == NVMEDIA_STATUS_OK) && IsGMSL2Mode(gmslMode)) {
       /* Set FSYNC to GMSL2 type */
       setOSCManualFsyncModeRegs[0].data |= ((uint16_t)1U << (uint16_t)7U);
    }


    if ((status == NVMEDIA_STATUS_OK)) {
        status = MAX96712WriteArrayVerify(handle, &setOSCManualFsyncMode);
    }

    if ((status == NVMEDIA_STATUS_OK) && (IsGMSL2Mode(gmslMode))) {
       status = MAX96712WriteUint8Verify(handle,
                                         setTxIDIntReg.address,
                                         (uint8_t)setTxIDIntReg.data);
    }

    if ((status == NVMEDIA_STATUS_OK)) {
        /* calculate frsync high period */
        fsyncPeriodRegs[0].data = (uint16_t)(((MAX96712_OSC_MHZ * 1000U * 1000U
                                                  / fps) >> 16U) & 0xFFU);
        /* calculate frsync middle period */
        fsyncPeriodRegs[1].data = (uint16_t)(((MAX96712_OSC_MHZ * 1000U * 1000U
                                                  / fps) >> 8U) & 0xFFU);
        /* calculate frsync low period */
        fsyncPeriodRegs[2].data = (uint16_t)((MAX96712_OSC_MHZ * 1000U * 1000U
                                                        / fps) & 0xFFU);

        status = MAX96712WriteArrayVerify(handle, &fsyncPeriod);
    }

    if ((status == NVMEDIA_STATUS_OK)) {
        drvHandle->ctx.manualFSyncFPS = (uint16_t)fps;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
        if (gmslMode == CDI_MAX96712_GMSL1_MODE) {
            status = Max96712EnableGpioRegsForGMSL1Manual(handle, link);
        }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
    }

    /* NOP: just to make sure misra_cpp_2008_rule_2_7_violation is suppressed */
    (void) link;

    return status;
}

/**
 * @brief Sets FSYNC Mode Manual
 *
 * - extracts device driver handle from handle
 * - initializes
 *  - DevBlkCDII2CReg fsyncPeriodRegs[3] = {
 *   - {0x04A7U, 0x00U}, Calculate FRSYNC period H. don't move position
 *   - {0x04A6U, 0x00U}, Calculate FRSYNC period M. don't move position
 *   - {0x04A5U, 0x00U}, Calculate FRSYNC period L. don't move position
 *   .
 *  - }
 *  .
 * - initialize wrapper for fsyncPeriodRegs
 *  - DevBlkCDII2CRegList fsyncPeriod = {
 *   - .regs = fsyncPeriodRegs,
 *   - .numRegs = (uint32_t)(sizeof(fsyncPeriodRegs) /
 *    - sizeof(fsyncPeriodRegs[0])),
 *    .
 *   .
 *   - }
 *   .
 *  .
 * - initializes
 *  - DevBlkCDII2CReg setManualFsyncModeRegs[7] = {
 *   - {0x04A2U, 0xE0U}, video link for fsync generation
 *   - {0x04AAU, 0x00U}, Disable overlap window
 *   - {0x04ABU, 0x00U}, Disable overlap window
 *   - {0x04A8U, 0x00U}, Disable error threshold
 *   - {0x04A9U, 0x00U}, Disable error threshold
 *   - {0x04AFU, 0x1FU}, Set FSYNC to GMSL1 type
 *   - {0x04A0U, 0x10U}, Set FSYNC to manual mode
 *  - }
 *  .
 * - initialize wrapper for setManualFsyncModeRegs
 *  - DevBlkCDII2CRegList setManualFsyncMode = {
 *   - .regs = setManualFsyncModeRegs,
 *   - .numRegs = (uint32_t)(sizeof(setManualFsyncModeRegs) /
 *   - sizeof(setManualFsyncModeRegs[0])),
 *   .
 *  - }
 *  .
 * - verify (CheckValidFPS(fps, drvHandle->ctx.manualFSyncFPS) == NVMEDIA_STATUS_OK)
 * - if (gmslMode == CDI_MAX96712_GMSL1_MODE)
 *  - calculate frsync high/middle/low periods
 *   - fsyncPeriodRegs[0].data = (uint8_t)(((pclk / fps) >> 16U) & 0xFFU)
 *   - fsyncPeriodRegs[1].data = (uint8_t)(((pclk / fps) >> 8U) & 0xFFU)
 *   - fsyncPeriodRegs[2].data = (uint8_t)((pclk / fps) & 0xFFU)
 *   .
 *  .
 * -else
 *  - fsyncPeriodRegs[0].data = 0x25U;
 *  - fsyncPeriodRegs[1].data = 0x4CU;
 *  - fsyncPeriodRegs[2].data = 0x9CU;
 *  .
 * - write fsyncPeriodRegs to deserializer by
 *  - status = DevBlkCDII2CPgmrWriteArray(
 *   - drvHandle->i2cProgrammer, &fsyncPeriod)
 *   .
 *  .
 * - if IsGMSL2Mode(gmslMode)
 *  - setManualFsyncModeRegs[6].data = 0x90U;
 *  .
 * - write setManualFsyncModeRegs to deserializer by
 *  - status = DevBlkCDII2CPgmrWriteArray(
 *   - drvHandle->i2cProgrammer, &setManualFsyncMode)
 *   .
 *  .
 * - nvsleep(10000)
 * - update drvHandle->ctx.manualFSyncFPS = (uint16_t)fps
 * @param[in] handle DEVBLK handle
 * @param[in] pclk pixel clock
 * @param[in] fps frame per second
 * @param[in] link link masks
 * @param[in] gmslMode GMSL Mode
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
SetFSYNCMode_Manual(
    DevBlkCDIDevice const* handle,
    uint32_t pclk,
    uint32_t fps,
    LinkMAX96712 const link,
    GMSLModeMAX96712 gmslMode)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 *drvHandle = getHandlePrivMAX96712(handle);
    DevBlkCDII2CReg fsyncPeriodRegs[] = {
        {0x04A7U, 0x00U}, /* Calculate FRSYNC period H. don't move position */
        {0x04A6U, 0x00U}, /* Calculate FRSYNC period M. don't move position */
        {0x04A5U, 0x00U}, /* Calculate FRSYNC period L. don't move position */
    };
    DevBlkCDII2CRegList fsyncPeriod = {
        .regs = fsyncPeriodRegs,
        .numRegs = (uint32_t)(sizeof(fsyncPeriodRegs) /
                              sizeof(fsyncPeriodRegs[0])),
    };
    DevBlkCDII2CReg setManualFsyncModeRegs[] = {
        {0x04A2U, 0xE0U}, /* video link for fsync generation */
        {0x04AAU, 0x00U}, /* Disable overlap window */
        {0x04ABU, 0x00U}, /* Disable overlap window */
        {0x04A8U, 0x00U}, /* Disable error threshold */
        {0x04A9U, 0x00U}, /* Disable error threshold */
        {0x04AFU, 0x1FU}, /* Set FSYNC to GMSL1 type */
        {0x04A0U, 0x10U}, /* Set FSYNC to manual mode */
    };
    DevBlkCDII2CRegList setManualFsyncMode = {
        .regs = setManualFsyncModeRegs,
        .numRegs = (uint32_t)(sizeof(setManualFsyncModeRegs) /
                              sizeof(setManualFsyncModeRegs[0])),
    };

    if (CheckValidFPS(fps, drvHandle->ctx.manualFSyncFPS) != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX96712: Incorrect fsync frequencies requested");
        status = NVMEDIA_STATUS_NOT_SUPPORTED;
    } else {
        if (gmslMode == CDI_MAX96712_GMSL1_MODE) {
            /* calculate frsync high/middle/low periods */
            fsyncPeriodRegs[0].data = (uint8_t)(((pclk / fps) >> 16U) & 0xFFU);
            fsyncPeriodRegs[1].data = (uint8_t)(((pclk / fps) >> 8U) & 0xFFU);
            fsyncPeriodRegs[2].data = (uint8_t)((pclk / fps) & 0xFFU);
        } else {
            fsyncPeriodRegs[0].data = 0x25U;
            fsyncPeriodRegs[1].data = 0x4CU;
            fsyncPeriodRegs[2].data = 0x9CU;
        }

        status = MAX96712WriteArrayVerify(handle, &fsyncPeriod);
        if (status == NVMEDIA_STATUS_OK) {
            if (IsGMSL2Mode(gmslMode)) {
                setManualFsyncModeRegs[6].data = 0x90U;
            }

            status = MAX96712WriteArrayVerify(handle, &setManualFsyncMode);
            if (status == NVMEDIA_STATUS_OK) {
                /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
                (void)nvsleep(10000);

                drvHandle->ctx.manualFSyncFPS = (uint16_t)fps;
            }
        }
    }

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    if ((status == NVMEDIA_STATUS_OK) && (gmslMode == CDI_MAX96712_GMSL1_MODE)) {
        status = Max96712EnableGpioRegsForGMSL1Manual(handle, link);
    }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

    /* NOP: just to make sure misra_cpp_2008_rule_2_7_violation is suppressed */
    (void) link;

    return status;
}

/**
 * @brief Sets FSYNC Mode
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 * - initializes
 *  - const DevBlkCDII2CReg setAutoFsyncModeRegs[7] = {
 *   - {0x04A2U, 0xE1U}, video link for fsync generation
 *   - {0x04AAU, 0x00U}, Disable overlap window
 *   - {0x04ABU, 0x00U}, Disable overlap window
 *   - {0x04A8U, 0x00U}, Disable error threshold
 *   - {0x04A9U, 0x00U}, Disable error threshold
 *   - {0x04B1U, 0x78U}, GPIO ID setup to output FSYNC. For Auto mode, select ID=0xF
 *   - {0x04A0U, 0x12U}, Set FSYNC to auto mode
 *   .
 *  - }
 *  .
 * - initialize wrapper for setAutoFsyncModeRegs
 *  - DevBlkCDII2CRegList setAutoFsyncMode = {
 *   - .regs = setAutoFsyncModeRegs,
 *   - .numRegs = (uint32_t)(sizeof(setAutoFsyncModeRegs) /
 *    - sizeof(setAutoFsyncModeRegs[0])),
 *    .
 *   .
 *  - }
 *  .
 * - set return value isValidSize = true
 * - if (((link <= CDI_MAX96712_LINK_3) &&
 *  - !(MAX96712_IS_MULTIPLE_GMSL_LINK_SET(link))) ||
 *  - (link == CDI_MAX96712_LINK_ALL))
 *  - for each link
 *   - if (MAX96712_IS_GMSL_LINK_SET(link, i))
 *    - extract gmslMode = drvHandle->ctx.gmslMode[i]
 *    - break
 *    .
 *   .
 *  .
 * - else
 *  - exit error NVMEDIA_STATUS_BAD_PARAMETER
 *  .
 * - if FSyncMode == CDI_MAX96712_FSYNC_MANUAL
 *  - call SetFSYNCMode_Manual(handle, pclk, fps, link, gmslMode)
 *  .
 * - elif FSyncMode == CDI_MAX96712_FSYNC_OSC_MANUAL
 *  - call SetFSYNCMode_OSCManual(handle, fps, link, gmslMode)
 *  .
 * - elif FSyncMode == CDI_MAX96712_FSYNC_EXTERNAL
 *  - call SetFSYNCMode_External(handle, link, gmslMode)
 *  .
 * - elif FSyncMode == CDI_MAX96712_FSYNC_AUTO  :
 *  - call DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &setAutoFsyncMode)
 *  .
 * - else
 *  - error NVMEDIA_STATUS_BAD_PARAMETER
 *  .
 *
 * @param[in] handle DEVBLK handle
 * @param[in] FSyncMode frame sync mode
 * @param[in] pclk pixel clock
 * @param[in] fps frame per second
 * @param[in] link link mask
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
SetFSYNCMode(
    DevBlkCDIDevice const *handle,
    FSyncModeMAX96712 FSyncMode,
    uint32_t pclk,
    uint32_t fps,
    LinkMAX96712 const link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    GMSLModeMAX96712 gmslMode = CDI_MAX96712_GMSL_MODE_INVALID;

    DevBlkCDII2CReg const setAutoFsyncModeRegs[] = {
        {0x04A2U, 0xE1U}, /* video link for fsync generation */
        {0x04AAU, 0x00U}, /* Disable overlap window */
        {0x04ABU, 0x00U}, /* Disable overlap window */
        {0x04A8U, 0x00U}, /* Disable error threshold */
        {0x04A9U, 0x00U}, /* Disable error threshold */
        {0x04B1U, 0x78U}, /* GPIO ID setup to output FSYNC. For Auto mode, select ID=0xF */
        {0x04A0U, 0x12U}, /* Set FSYNC to auto mode */
    };
    DevBlkCDII2CRegList setAutoFsyncMode = {
        .regs = setAutoFsyncModeRegs,
        .numRegs = (uint32_t)(sizeof(setAutoFsyncModeRegs) /
                              sizeof(setAutoFsyncModeRegs[0])),
    };

    uint8_t i = 0U;

    /* Verify ONLY ONE link or ALL links */
    for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)){
            gmslMode = drvHandle->ctx.gmslMode[i];
            break;
        }
    }

    if (status == NVMEDIA_STATUS_OK) {
        switch (FSyncMode) {
        case CDI_MAX96712_FSYNC_MANUAL:
            status = SetFSYNCMode_Manual(handle, pclk, fps, link, gmslMode);
            break;
        case CDI_MAX96712_FSYNC_OSC_MANUAL:
            status = SetFSYNCMode_OSCManual(handle, fps, link, gmslMode);
            break;
        case CDI_MAX96712_FSYNC_EXTERNAL:
            status = SetFSYNCMode_External(handle, link, gmslMode);
            break;
        case CDI_MAX96712_FSYNC_AUTO:
            status = MAX96712WriteArrayVerify(handle, &setAutoFsyncMode);
            break;
        default:
            SIPL_LOG_ERR_STR("MAX96712: Invalid param: FSyncMode");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
        }
    }

    return status;
}

/**
 * @brief  Enable GPIO Rx
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - verify parameterSize == sizeof(gpioIndex)
 *  - return isValidSize = true  if test fails
 *  .
 * - Note: ; pull-up 1M ohm, GPIO source en for GMSL2
 * - initialize DevBlkCDII2CReg setGPIOMode = {0x0300U, 0x1CU}
 * - get mask for link index by
 *  - status = GetMAX96712LinkVal(linkIndex, &linkPort)
 *  .
 * - set setGPIOMode.address += ((uint16_t)gpioIndex * 3U)
 * - Note:    0x30F, 0x31F, 0x32F are not used
 *   - GPIO index offset applied here
 *   .
 *  - set  setGPIOMode.address +=
 *   -  if (setGPIOMode.address & 0xFFU) > 0x2EU)
 *    -    0x03
 *    .
 *   - elif setGPIOMode.address & 0xFFU) > 0x1EU)
 *    - 0x02
 *    .
 *   - elif (setGPIOMode.address & 0xFFU) > 0xEU)
 *      - 0x01
 *      .
 *   - else
 *    - 0x00
 *    .
 *   .
 *  - read register into data by
 *   - status = DevBlkCDII2CPgmrReadUint8(
 *    - drvHandle->i2cProgrammer,
 *    - setGPIOMode.address,
 *    -  &data)
 *    .
 *   .
 *  - Note: check for Link port 0
 *  - if ((linkPort & 0x01U) == 1U)
 *   - Note:  Set GPIO_RX_EN
 *   - data |= 0x4U
 *   - Note: Unset GPIO_TX_EN, GPIO_OUT_DIS
 *   - data &= ~(uint8_t)0x3U
 *   - write data to setGPIOMode register by
 *    - status = DevBlkCDII2CPgmrWriteUint8(
 *     - drvHandle->i2cProgrammer,
 *      - setGPIOMode.address
 *      - data)
 *      .
 *    .
 *   .
 *  - elif ((linkPort & 0x0EU) > 1U)
 *   - Note: Link port 1/2/3
 *   - Note: Link0 Unset GPIO_TX_EN, GPIO_OUT_DIS
 *   - data &= ~(uint8_t)0x7U;
 *   - write data to setGPIOMode register by
 *    - status = DevBlkCDII2CPgmrWriteUint8(
 *     - drvHandle->i2cProgrammer,
 *      - setGPIOMode.address,
 *      - data)
 *      .
 *     .
 *   - Note: Set GPIO_RX_EN
 *   - data |= 0x20U
 *   - Note: set GPIO_RX_ID
 *   - data |= (gpioIndex & 0x1FU);
 *
 *   - ApplyLinkIndexOffsetForNonZeroPort(&setGPIOMode, linkPort)
 *   - Note: write data to setGPIOMode register by
 *     - status = DevBlkCDII2CPgmrWriteUint8(
 *      - drvHandle->i2cProgrammer,
 *       - setGPIOMode.address,
 *       - data)
 *       .
 *      .
 *     .
 *   .
 *  - else
 *   - Error:  link Index is out of bound  NO ERROR RETURNED
 *   .
 *  .
 *
 * @param[in] handle Dev Blk handle
 * @param[in] gpioIndex GPIO index
 * @param[in] linkIndex link index
 * @param[in] parameterSize  size of buffer for parameter
 * @param[out] isValidSize return true if valid parameter size, otherwise set to false
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
UpdateGPIOTxID(
    DevBlkCDIDevice const *handle,
    uint32_t const linkIndex,
    uint32_t const gpioIndex,
    uint32_t const gpioTxID)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (gpioIndex <= CDI_MAX96712_GPIO_8) {
        DevBlkCDII2CReg setGPIOA = {.address = 0x0300U, .data = 0x00U};
        DevBlkCDII2CReg setGPIOB = {.address = 0x0301U, .data = 0x00U};
        uint8_t data = 0U;

        /* Update the group GPIO index */
        switch (linkIndex) {
            case 0U:
                setGPIOA.address = regBitFieldProps[(uint32_t)REG_FIELD_GPIO_00_A_A +
                                                              gpioIndex].regAddr;
                setGPIOB.address = regBitFieldProps[(uint32_t)REG_FIELD_GPIO_00_A_B +
                                                              gpioIndex].regAddr;
                break;
            case 1U:
                setGPIOA.address = regBitFieldProps[(uint32_t)REG_FIELD_GPIO_00_A_A +
                                                              gpioIndex].regAddr;
                setGPIOB.address = regBitFieldProps[(uint32_t)REG_FIELD_GPIO_00_B_B +
                                                              gpioIndex].regAddr;
                break;
            case 2U:
                setGPIOA.address = regBitFieldProps[(uint32_t)REG_FIELD_GPIO_00_A_A +
                                                              gpioIndex].regAddr;
                setGPIOB.address = regBitFieldProps[(uint32_t)REG_FIELD_GPIO_00_C_B +
                                                              gpioIndex].regAddr;
                break;
            case 3U:
                setGPIOA.address = regBitFieldProps[(uint32_t)REG_FIELD_GPIO_00_A_A +
                                                              gpioIndex].regAddr;
                setGPIOB.address = regBitFieldProps[(uint32_t)REG_FIELD_GPIO_00_D_B +
                                                              gpioIndex].regAddr;
                break;
            default:
                status = NVMEDIA_STATUS_BAD_PARAMETER;
                break;
        }

        if (status == NVMEDIA_STATUS_OK) {
            status = MAX96712ReadUint8Verify(handle, setGPIOB.address, &data);
            if (status == NVMEDIA_STATUS_OK) {
                /* Clear the GPIO Tx ID */
                data &= (uint8_t)0xE0U;
                /* GPIO_TX_EN_C, Update the GPIO Tx ID */
                data |= (uint8_t)(1U << 5U) | (uint8_t)(gpioTxID & 0x1FU);
                status = MAX96712WriteUint8Verify(handle, setGPIOB.address, data);
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            status = MAX96712ReadUint8Verify(handle, setGPIOA.address, &data);
            if (status == NVMEDIA_STATUS_OK) {
                data |= (uint8_t)0x2U; /* Enable GPIO Tx function */
                status = MAX96712WriteUint8Verify(handle, setGPIOA.address, data);
            }
        }
    }

    return status;
}

/**
 * @brief Read control channel CRC Error from deserializer
 *
 * - verify parameterSizePassed equal to parameterSizeCalculated
 * - for each link
 *  - if (MAX96712_IS_GMSL_LINK_SET(link, i))
 *   - read REG_FIELD_CC_CRC_ERRCNT_A register value by
 *    - status = Max96712AccessOneRegFieldOffset(
 *     - status, handle, REG_FIELD_CC_CRC_ERRCNT_A, i,
 *     - 0U, REG_READ_MODE)
 *     .
 *    .
 *   - return it from bitReg queue to caller by
 *    - *errVal = ReadFromRegFieldQ(handle, 0U)
 *    .
 *   .
 *  .
 * @param[in] handle DEVBLK handle
 * @param[in] parameterSizePassed actual buffer size
 * @param[in] parameterSizeCalculated calculated buffer size
 * @param[in] link link masks
 * @param[out] errVal pointer for storing error value
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
SetHeteroFSYNCMode(
    DevBlkCDIDevice const *handle,
    uint32_t const *gpioNum)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    uint8_t linkMask = drvHandle->ctx.linkMask;

    for (uint32_t i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(linkMask, i)) {
            if (drvHandle->ctx.FSyncMode[i] == CDI_MAX96712_FSYNC_EXTERNAL) {
                if (gpioNum[i] != CDI_MAX96712_GPIO_2) {
                    /* Update GPIO Tx ID */
                    if (status == NVMEDIA_STATUS_OK) {
                        status = UpdateGPIOTxID(handle, i, CDI_MAX96712_GPIO_2, gpioNum[i]);
                    }
                    if (status == NVMEDIA_STATUS_OK) {
                        status = UpdateGPIOTxID(handle, i, gpioNum[i], CDI_MAX96712_GPIO_2);
                    }
                }
            }
        }
    }

    return status;
}

static NvMediaStatus
ReadCtrlChnlCRCErr(
    DevBlkCDIDevice const* handle,
    size_t parameterSizePassed,
    size_t parameterSizeCalculated,
    LinkMAX96712 const link,
    uint8_t *errVal)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t i = 0U;

    if (parameterSizePassed != parameterSizeCalculated) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (status == NVMEDIA_STATUS_OK) {
        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                status = Max96712AccessOneRegFieldOffset(status, handle, REG_FIELD_CC_CRC_ERRCNT_A,
                                                         i,
                                                         0U,
                                                         REG_READ_MODE);
                *errVal = ReadFromRegFieldQ(handle, 0U);
            }
        }
    }

    return status;
}

/**
 * @brief Get Enabled Links
 * - get MAX96712 Deseriailizer driver handle
 * - read value of register 0x0006[REG6] into link by
 *  - DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, (uint16_t)0x0006U, link)
 *  .
 * - or the link value with 0xFU
 *
 * @param[in] handle DEVBLK handle
 * @param[out] link uset link masks to be updated
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetEnabledLinks(
    DevBlkCDIDevice const* handle,
    uint8_t *link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = MAX96712ReadUint8Verify(handle, (uint16_t)0x0006U, link);

    *link = *link & 0xFU;

    return status;
}

/**
 * @brief Clears ERRB register
 * - verify parameterSizePassed equal to parameterSizeCalculated
 * - clear bitReg queue by
 *  - ClearRegFieldQ(handle)
 *  .
 * - add register REG_FIELD_ERRB to queue by
 *  - status = Max96712AddOneRegField(status, handle, REG_FIELD_ERRB, 0U)
 *  .
 * - reads REG_FIELD_ERRB register from deserializer by
 *  - status = Max96712AccessRegField(status, handle, REG_READ_MODE)
 *  .
 * - get data from bitReg queue by
 *  - ReadFromRegFieldQ(handle, 0U)
 *  .
 * - if value is 0x01
 *  - Note: log that ERRB is set
 *  - read register to clear error by
 *   - status = MAX96712GetErrorStatus(handle,
 *   - (uint32_t)sizeof(errorStatus), &errorStatus)
 *   .
 *  .
 * @param[in] handle DEVBLK handle
 * @param[in] parameterSizePassed actual buffer size
 * @param[in] parameterSizeCalculated calculated buffer size
 * @param[out] link    unused
 * @param[out] errVal  unused
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
ClearErrb(
    DevBlkCDIDevice const* handle,
    size_t parameterSizePassed,
    size_t parameterSizeCalculated,
    LinkMAX96712 const* link,
    uint8_t const* errVal)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    ErrorStatusMAX96712 errorStatus;

    if (parameterSizePassed != parameterSizeCalculated) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (status == NVMEDIA_STATUS_OK) {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
        DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
        if (drvHandle->ctx.tpgEnabled == true) {
            status = NVMEDIA_STATUS_OK;
        } else
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
        {
            ClearRegFieldQ(handle);
            status = Max96712AddOneRegField(status, handle, REG_FIELD_ERRB, 0U);
            status = Max96712AccessRegField(status, handle, REG_READ_MODE);
            if (ReadFromRegFieldQ(handle, 0U) == 1U) {
                SIPL_LOG_ERR_STR("MAX96712 ERRB was Set");
                status = MAX96712GetErrorStatus(handle,
                                                (uint32_t)sizeof(errorStatus),
                                                &errorStatus);
            }
        }
    }

    /* NOP: just to make sure misra_cpp_2008_rule_2_7_violation is suppressed */
    (void) link;
    (void) errVal;

    return status;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
static NvMediaStatus
UpdateReplicationForPhyC(
    MipiOutModeMAX96712 mipiOutMode,
    DevBlkCDII2CReg *dataRegs,
    RevisionMAX96712 revision)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2) {
        dataRegs[0].data =
        /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
            MAX96712_REPLICATION(CDI_MAX96712_TXPORT_PHY_C,
                                 CDI_MAX96712_TXPORT_PHY_E);
        dataRegs[1].data =
        /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
            MAX96712_REPLICATION(CDI_MAX96712_TXPORT_PHY_C,
                                 CDI_MAX96712_TXPORT_PHY_F);
        if (revision < CDI_MAX96712_REV_4) {
            /* 3rd I2C port connected to 3rd Xavier is enabled by default only in MAX96712 Rev D(4)
             * For other revisions, the replication from PHY C to PHY F is enabled by the master
             */
            dataRegs[1].data |= (uint8_t)(1U << 7U);
        }
    } else if (mipiOutMode == CDI_MAX96712_MIPI_OUT_2x4) {
        dataRegs[0].data =
        /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
            MAX96712_REPLICATION(CDI_MAX96712_TXPORT_PHY_D,
                                 CDI_MAX96712_TXPORT_PHY_E);
    } else {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }
    return status;
}

static NvMediaStatus
UpdateReplicationForPhyD(
    MipiOutModeMAX96712 mipiOutMode,
    DevBlkCDII2CReg *dataRegs)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if ((mipiOutMode == CDI_MAX96712_MIPI_OUT_2x4) || (mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2)) {
        dataRegs[0].data =
        /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
            MAX96712_REPLICATION(CDI_MAX96712_TXPORT_PHY_D,
                                 CDI_MAX96712_TXPORT_PHY_E);
    } else {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }
    return status;
}

static NvMediaStatus
UpdateReplicationForPhyE(
    MipiOutModeMAX96712 mipiOutMode,
    DevBlkCDII2CReg *dataRegs)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (mipiOutMode == CDI_MAX96712_MIPI_OUT_2x4) {
        dataRegs[0].data =
        /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
            MAX96712_REPLICATION(CDI_MAX96712_TXPORT_PHY_E,
                                 CDI_MAX96712_TXPORT_PHY_D);
    } else if (mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2) {
        dataRegs[0].data =
        /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
            MAX96712_REPLICATION(CDI_MAX96712_TXPORT_PHY_E,
                                 CDI_MAX96712_TXPORT_PHY_C);
    } else {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }
    return status;
}

static void
UpdateReplicationDefault(
    DevBlkCDII2CReg *dataRegs)
{
    dataRegs[0].data =
    /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
    MAX96712_REPLICATION(CDI_MAX96712_TXPORT_PHY_C,
                         CDI_MAX96712_TXPORT_PHY_E);
}

static NvMediaStatus
EnableReplicationPassiveEnabled(
    DevBlkCDIDevice const* handle,
    bool enable,
    TxPortMAX96712 port,
    DevBlkCDII2CReg *dataRegs,
    uint32_t numRegs,
    RevisionMAX96712 revision)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t temp;
    uint8_t i;

    DevBlkCDII2CRegList data = {
        .regs = dataRegs,
        /* bugfix:(uint32_t)(sizeof(dataRegs) / sizeof(dataRegs[0])), */
        .numRegs = numRegs,
    };

    if ((port == CDI_MAX96712_TXPORT_PHY_F) && (revision < CDI_MAX96712_REV_4)) {
        PrintLogMsg(LOG_LEVEL_INFO, "The replication mode is already enabled\n");
        status = NVMEDIA_STATUS_OK;
    } else {
        for (i = 0U; i < 2U; i++) {
            status = MAX96712ReadUint8Verify(handle, dataRegs[i].address, &temp);
            if (status != NVMEDIA_STATUS_OK) {
                break;
            }
            dataRegs[i].data = temp;
            if (((dataRegs[i].data >> 5U) & 3U) == (uint8_t)port) {
                /* if the destination is same as port */
                if (enable) {
                    dataRegs[i].data |= (uint8_t)(1U << 7U);
                    /* Enable the replication */
                } else {
                    dataRegs[i].data &= ~(uint8_t)(1U << 7U);
                    /* Disable the replication */
                }
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            status = MAX96712WriteArrayVerify(handle, &data);
        }
    }

    return status;
}

static NvMediaStatus
EnableReplication(
    DevBlkCDIDevice const* handle,
    bool enable)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    TxPortMAX96712 port = CDI_MAX96712_TXPORT_PHY_C;
    MipiOutModeMAX96712 mipiOutMode = CDI_MAX96712_MIPI_OUT_INVALID;
    PHYModeMAX96712 phyMode = CDI_MAX96712_PHY_MODE_INVALID;
    RevisionMAX96712 revision = CDI_MAX96712_REV_INVALID;
    NvMediaBool passiveEnabled;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    DevBlkCDII2CReg dataRegs[] = {
        {0x08A9U, 0U},  /* For the replication from Tegra A to Tegra B */
        {0x08AAU, 0U},  /* For the replication from Tegra A to Tegra C */
    };
    DevBlkCDII2CRegList data = {
        .regs = dataRegs,
        .numRegs = (uint32_t)(sizeof(dataRegs) / sizeof(dataRegs[0])),
    };

    port = drvHandle->ctx.txPort;
    mipiOutMode = drvHandle->ctx.mipiOutMode;
    revision = drvHandle->ctx.revision;
    phyMode = drvHandle->ctx.phyMode;
    passiveEnabled = drvHandle->ctx.passiveEnabled;

    /* Replication is not supported on revision 1 in CPHY mode */
    if ((revision == CDI_MAX96712_REV_1) &&
        (phyMode == CDI_MAX96712_PHY_MODE_CPHY)) {
        SIPL_LOG_ERR_2STR("MAX96712: Replication in CPHY mode is supported only ",
                          "on platforms with MAX96712 revision 2 or higher.");
        status = NVMEDIA_STATUS_NOT_SUPPORTED;
    }

    /* Update the replication but do NOT enable the replication */
    if ((status == NVMEDIA_STATUS_OK) && ((!passiveEnabled) && enable)) {
        switch (port) {
            case CDI_MAX96712_TXPORT_PHY_C :
                status = UpdateReplicationForPhyC(mipiOutMode, dataRegs, revision);
                break;
            case CDI_MAX96712_TXPORT_PHY_D :
                status = UpdateReplicationForPhyD(mipiOutMode, dataRegs);
                break;
            case CDI_MAX96712_TXPORT_PHY_E :
                status = UpdateReplicationForPhyE(mipiOutMode, dataRegs);
                break;
            default :
                UpdateReplicationDefault(dataRegs);
                break;
        }

        if (status == NVMEDIA_STATUS_OK) {
            status = MAX96712WriteArrayVerify(handle, &data);
        }
    } else if ((status == NVMEDIA_STATUS_OK) && (passiveEnabled)) {
        /* Enable or disable the replication */
        status = EnableReplicationPassiveEnabled(handle, enable, port, dataRegs,
                                                 data.numRegs, revision);
    } else {
        /* do nothing */
    }

    return status;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/**
 * @brief Helper function to configure MIPI output
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - extracts mipiOutMode from device driver context by
 *  - MipiOutModeMAX96712 mipiOutMode = drvHandle->ctx.mipiOutMode
 *  .
 * - initialize settings for Mapping data lanes Port A
 * - initialize DevBlkCDII2CReg mipiOutputReg =  {
 *  - 0x08A3U, CDI_MAX96712_MIPI_OUT_4x2) ? 0x44U : 0xE4U
 *  .
 * - }
 * .
 * - write settings to deserializer
 * - initialize settings for Mapping data lanes Port B
 *  - initialize mipiOutputReg.address = 0x08A4U
 *  - initialize mipiOutputReg.data =
 *   - (mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2) ? 0x44U : 0xE4U
 *   .
 *  -
 * - write settings to deserializer by
 *  - status = DevBlkCDII2CPgmrWriteUint8(
 *   - drvHandle->i2cProgrammer,
 *   - mipiOutputReg.address, (uint8_t)mipiOutputReg.data)
 *   .
 *  .
 * - setup Mapping data lanes Port B
 *  - mipiOutputReg.address = 0x08A4U
 *  - mipiOutputReg.data = (mipiOutMode ==  CDI_MAX96712_MIPI_OUT_4x2) ? 0x44U : 0xE4U
 *  .
 * - write to register by
 *  - status = DevBlkCDII2CPgmrWriteUint8(
 *   - drvHandle->i2cProgrammer,
 *   - mipiOutputReg.address, (uint8_t)mipiOutputReg.data)
 *   .
 *  .
 * - Note: Configure Lane count for Port B in x4 or E,F in x2 an config phyMode
 * - for (i = 0U; i < 2U; i++)
 *  - set  mipiOutputReg.address = (mipiOutputReg.address & 0xFF00U) +  0x8AU + (i * 0x40U)
 *  - write to register by
 *   - status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
 *    - mipiOutputReg.address, (uint8_t)mipiOutputReg.data)
 *    .
 *   .
 *
 * - initialize settings for deactive DPLL
 *  - set mipiOutputReg.data = 0xF4U
 *  - for each link
 *   - initialize mipiOutputReg.address = (0x1CU + link_index) << 8U
 *   - write mipiOutputReg settings to deserializer by
 *    - status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
 *     - mipiOutputReg.address, (uint8_t)mipiOutputReg.data)
 *     .
 *    .
 *   .
 *  .
 * - initialize settings for Set MIPI speed
 *  - initialize mipiOutputReg.address = 0x0415U
 *  .
 * - for each link
 *  - set mipiOutputReg.address =
 *   - (mipiOutputReg.address & 0xFF00U) + 0x15U + (link_index * 0x3U)
 *   .
 *  - read settings from deserializer register address  -> temp by
 *   - status = DevBlkCDII2CPgmrReadUint8(
 *    - drvHandle->i2cProgrammer, mipiOutputReg.address, &temp)
 *    .
 *   .
 *  .
 *  - set mipiOutputReg.data = temp
 *  - mipiOutputReg.data &= 0x00C0U
 *  - mipiOutputReg.data |= ((1U << 5U) | (uint16_t)mipiSpeed);
 *  - write settings to deserializer by
 *   -
 *  .
 * - initialize settings for active DPLL
 * - initialize mipiOutputReg.data = 0xF5U
 * - for each link
 *  - initialize mipiOutputReg.address = (0x1CU + link_index) << 8U
 *  - write settings to deserializer
 *   - status = DevBlkCDII2CPgmrWriteUint8(
 *    - drvHandle->i2cProgrammer, mipiOutputReg.address,
 *    - (uint8_t)(mipiOutputReg.data & 0xFFU))
 *    .
 *   .
 *  .
 * - Note: Deskew is enabled if MIPI speed is faster than or equal to 1.5GHz
 * - if ((phyMode == CDI_MAX96712_PHY_MODE_DPHY) && (mipiSpeed >= 15U))
 *  - initialize  mipiOutputReg.address = 0x0903U;
 *  - Note: enable the initial deskew with 8 * 32K UI
 *  - initialize mipiOutputReg.data = 0x97U
 *  - for each link
 *   - set mipiOutputReg.address =
 *    - (mipiOutputReg.address & 0xff00U) +
 *    - ((mipiOutputReg.address + 0x40U) & 0xffU)
 *    .
 *   - write settings to deserializer by
 *    - status = DevBlkCDII2CPgmrWriteUint8(
 *     - drvHandle->i2cProgrammer,
 *      - mipiOutputReg.address, (uint8_t)mipiOutputReg.data)
 *      .
 *     .
 *    .
 *   .
 *  .
 * @param[in] handle DEVBLK handle
 * @param[in] mipiSpeed MIPI speed
 * @param[in] phyMode PHY mode
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
ConfigureMIPIOutputHelper(
        DevBlkCDIDevice const* handle,
        uint8_t mipiSpeed,
        PHYModeMAX96712 phyMode)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    MipiOutModeMAX96712 mipiOutMode = drvHandle->ctx.mipiOutMode;
    DevBlkCDII2CReg mipiOutputReg = {0x08A2U, 0x00U};
    uint16_t i = 0U;
    uint8_t temp;

    /* Mapping data lanes Port A */
    mipiOutputReg.address = 0x08A3U;
    mipiOutputReg.data = (mipiOutMode ==
            CDI_MAX96712_MIPI_OUT_4x2) ? 0x44U : 0xE4U;

    status = MAX96712WriteUint8Verify(handle,
                                      mipiOutputReg.address,
                                      (uint8_t)mipiOutputReg.data);

    if (status == NVMEDIA_STATUS_OK) {
        /* Mapping data lanes Port B */
        mipiOutputReg.address = 0x08A4U;
#if !NV_IS_SAFETY || SAFETY_DBG_OV
        mipiOutputReg.data =
            (drvHandle->ctx.cfgPipeCopy == CDI_MAX96712_CFG_PIPE_COPY_MODE_2) ? 0x44U : 0xE4U;
#else
        mipiOutputReg.data = 0xE4U;
#endif
        status = MAX96712WriteUint8Verify(handle,
                                          mipiOutputReg.address,
                                          (uint8_t)mipiOutputReg.data);

        if (status == NVMEDIA_STATUS_OK) {
            if (status == NVMEDIA_STATUS_OK) {
                /* deactivate DPLL */
                mipiOutputReg.data = 0xF4U;

                for (i = 0U; i < MAX96712_MAX_NUM_PHY; i++) {
                    mipiOutputReg.address = lshift16(0x1CU + i, 8U);

                    status = MAX96712WriteUint8Verify(handle,
                                                      mipiOutputReg.address,
                                                      (uint8_t)(mipiOutputReg.data));
                    if (status != NVMEDIA_STATUS_OK) {
                        break;
                    }
                }
                if (status == NVMEDIA_STATUS_OK) {
                    /* Set MIPI speed */
                    mipiOutputReg.address = 0x0415U;
                    for (i = 0U; i < MAX96712_MAX_NUM_PHY; i++) {
                        mipiOutputReg.address =
                            (mipiOutputReg.address & 0xFF00U) +
                            0x15U +
                            (i * 0x3U);
                        status = MAX96712ReadUint8Verify(handle,
                                                         mipiOutputReg.address,
                                                         &temp);
                        if (status == NVMEDIA_STATUS_OK) {
                            mipiOutputReg.data = temp;

                            mipiOutputReg.data &= 0x00C0U;
                            mipiOutputReg.data |= ((1U << 5U) | (uint16_t)mipiSpeed);
                            status = MAX96712WriteUint8Verify(
                                                handle,
                                                mipiOutputReg.address,
                                                (uint8_t)(mipiOutputReg.data & 0xFFU));
                        }

                        if (status != NVMEDIA_STATUS_OK) {
                            break;
                        }
                    }

                    if (status == NVMEDIA_STATUS_OK) {
                        /* activate DPLL */
                        mipiOutputReg.data = 0xF5U;

                        for (i = 0U; i < MAX96712_MAX_NUM_PHY; i++) {
                            mipiOutputReg.address = lshift16(0x1CU + i, 8U);

                            status = MAX96712WriteUint8Verify(handle,
                                                              mipiOutputReg.address,
                                                              (uint8_t)(mipiOutputReg.data));
                            if (status != NVMEDIA_STATUS_OK) {
                                break;
                            }
                        }

                        if (status == NVMEDIA_STATUS_OK) {
                            /* Deskew is enabled if MIPI speed is faster than or equal to 1.5GHz */
                            if ((phyMode == CDI_MAX96712_PHY_MODE_DPHY) && (mipiSpeed >= 15U)) {
                                mipiOutputReg.address = 0x0903U;
                                /* enable the initial deskew with 8 * 32K UI */
                                mipiOutputReg.data = 0x97U;
                                for (i = 0U; i < MAX96712_MAX_NUM_PHY; i++) {
                                    mipiOutputReg.address = (mipiOutputReg.address & 0xff00U) +
                                        ((mipiOutputReg.address + 0x40U) & 0xffU);
                                    status = MAX96712WriteUint8Verify(handle,
                                                        mipiOutputReg.address,
                                                        (uint8_t)mipiOutputReg.data);
                                    if (status != NVMEDIA_STATUS_OK) {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return status;

}

/**
 * @brief Configures MIPI output
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - extracts mipiOutMode from device driver context by
 *  - MipiOutModeMAX96712 mipiOutMode = drvHandle->ctx.mipiOutMode
 *  .
 * - calculate mipiOutModeVal =
 *  - (mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2) ?
 *   - (1U << 0U) : (1U << 2U)
 *   .
 *  .
 * - initialize DevBlkCDII2CReg mipiOutputReg = {0x08A2U, 0x00U}
 * - if ((phyMode != CDI_MAX96712_PHY_MODE_DPHY) and
 *  - (phyMode != CDI_MAX96712_PHY_MODE_CPHY))
 *   - Error:  MAX96712: Invalid MIPI output port
 *   .
 *  .
 * - if ((mipiSpeed < 1U) or (mipiSpeed > 25U))
 *  - Error:  MAX96712: Invalid parameters - exit
 *  .
 * - Note: Force DPHY0 clk enabled not needed for Rev 1
 * - if ((drvHandle->ctx.revision != CDI_MAX96712_REV_1) and
 *  - (phyMode == CDI_MAX96712_PHY_MODE_DPHY))
 *   - calculate mipiOutModeVal = mipiOutModeVal | (1U << 5U)
 *   .
 *  .
 * - read/mod/write settings to REG_FIELD_MIPI_OUT_CFG register by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_MIPI_OUT_CFG, mipiOutModeVal,
 *   - REG_READ_MOD_WRITE_MODE)
 *   .
 *  .
 * - Note: Set prebegin phase, post length and prepare for CPHY mode
 * - This is a requirement for CPHY periodic calibration
 * - if (phyMode == CDI_MAX96712_PHY_MODE_CPHY)
 *  - if (mipiSpeed == 17U)
 *   - Note: TODO : This is a temporal solution to support the previous platform
 *    - This will be updated once CPHY calibration logic in RCE updated
 *     - t3_prebegin = (63 + 1) * 7 = 448 UI
 *     - Bit[6:2] = t3_post = (31 + 1) * 7 = 224 UI
 *     - Bit[1:0] = t3_prepare = 86.7ns
 *     .
 *    .
 *   -  initialize prebegin = 0x3FU
 *   - initialize post = 0x7FU
 *   .
 *  -  else
 *   - Note: t3_prebegin = (19 + 1) * 7 = 140 UI
 *    - Bit[6:2] = t3_post = (31 + 1) * 7 = 224 UI
 *    - Bit[1:0] = t3_prepare = 40ns
 *    .
 *   - initialize prebegin = 0x13U
 *   - initialize post = 0x7CU
 *   .
 *  .
 * - read/mod/write prebegin to REG_FIELD_T_T3_PREBEGIN register by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_T_T3_PREBEGIN, prebegin,
 *   - REG_READ_MOD_WRITE_MODE)
 *   .
 *  .
 * - read/mod/write post to REG_FIELD_T_T3_POST_PREP register by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_T_T3_POST_PREP, post,
 *   - REG_READ_MOD_WRITE_MODE)
 *   .
 *  .
 * - Put all Phys in standby mode by
 *  - set mipiOutputReg.address = 0x08A2U
 *  - Note: Bug 200383247 : t_lpx 106.7 ns
 *  - set  mipiOutputReg.data = 0xF4U
 *  - write to register by
 *   - status = DevBlkCDII2CPgmrWriteUint8(
 *    - drvHandle->i2cProgrammer, mipiOutputReg.address,
 *    - (uint8_t)mipiOutputReg.data)
 *    .
 *   .
 *  .
 * - Note: Set CSI2 lane count per Phy
 *  - Configure Lane count for Port A in x4 or C,D in x2 and config phyMode by
 *  .
 *  -  mipiOutputReg.address = 0x09A2U
 *  -  mipiOutputReg.data = (mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2) ? 0x40U : 0xC0U
 *  - mipiOutputReg.data |= (phyMode == CDI_MAX96712_PHY_MODE_CPHY) ? (1U << 5U) : 0U
 *  - for (i = 0U; i < 2U; i++)
 *   -set  mipiOutputReg.address = (mipiOutputReg.address & 0xFF00U) + 0x0AU + (i * 0x40U)
 *   - write to device by
 *    - status = DevBlkCDII2CPgmrWriteUint8(
 *     - drvHandle->i2cProgrammer, mipiOutputReg.address, (uint8_t)mipiOutputReg.data)
 *     .
 *    .
 *   .
 *  .
 * - setup GPIO by
 *  - status = ConfigureMIPIOutputHelper(handle, mipiSpeed, phyMode)
 *  .
 * @param[in] handle DEVBLK handle
 * @param[in] mipiSpeed MIPI speed
 * @param[in] phyMode PHY mode
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
ConfigureMIPIOutput(
        DevBlkCDIDevice const* handle,
        uint8_t mipiSpeed,
        PHYModeMAX96712 phyMode)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    MipiOutModeMAX96712 mipiOutMode = drvHandle->ctx.mipiOutMode;
	uint8_t mipiOutModeVal = (mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2) ? (1U << 0U) :
                             ((mipiOutMode == CDI_MAX96712_MIPI_OUT_2x4) ? (1U << 2U) : (1U << 3U));
    DevBlkCDII2CReg mipiOutputReg = {0x08A2U, 0x00U};
    uint16_t i = 0U;
    uint8_t prebegin = 0U, post = 0U;

    if ((phyMode != CDI_MAX96712_PHY_MODE_DPHY) &&
          (phyMode != CDI_MAX96712_PHY_MODE_CPHY)) {
        SIPL_LOG_ERR_STR("MAX96712: Invalid MIPI output port");
        status =  NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if ((mipiSpeed < 1U) || (mipiSpeed > 25U)) {
        status =  NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if ( status == NVMEDIA_STATUS_OK){
        /* Force DPHY0 clk enabled not needed for Rev 1 */
        if ((drvHandle->ctx.revision != CDI_MAX96712_REV_1) &&
                (phyMode == CDI_MAX96712_PHY_MODE_DPHY)) {
            mipiOutModeVal = mipiOutModeVal | (1U << 5U);
        }

        status = Max96712AccessOneRegField(status, handle, REG_FIELD_MIPI_OUT_CFG, mipiOutModeVal,
                REG_READ_MOD_WRITE_MODE);

        /* Set prebegin phase, post length and prepare for CPHY mode
         * This is a requirement for CPHY periodic calibration */
        if (phyMode == CDI_MAX96712_PHY_MODE_CPHY) {
            if (mipiSpeed == 17U) {
                /* TODO : This is a temporal solution to support the previous platform
                 * This will be updated once CPHY calibration logic in RCE updated
                 */
                /* t3_prebegin = (63 + 1) * 7 = 448 UI
                 * Bit[6:2] = t3_post = (31 + 1) * 7 = 224 UI
                 * Bit[1:0] = t3_prepare = 86.7ns
                 */
                prebegin = 0x3FU;
                post = 0x7FU;
            } else {
                /* t3_prebegin = (19 + 1) * 7 = 140 UI
                 * Bit[6:2] = t3_post = (31 + 1) * 7 = 224 UI
                 * Bit[1:0] = t3_prepare = 40ns
                 */
                prebegin = 0x13U;
                post = 0x7CU;
            }
            status = Max96712AccessOneRegField(status, handle, REG_FIELD_T_T3_PREBEGIN, prebegin,
                    REG_READ_MOD_WRITE_MODE);
            status = Max96712AccessOneRegField(status, handle, REG_FIELD_T_T3_POST_PREP, post,
                    REG_READ_MOD_WRITE_MODE);
        }

        /* Put all Phys in standby mode */
        mipiOutputReg.address = 0x08A2U;
        mipiOutputReg.data = 0xF4U; /* Bug 200383247 : t_lpx 106.7 ns */
        status = MAX96712WriteUint8Verify(handle,
                                          mipiOutputReg.address,
                                          (uint8_t)mipiOutputReg.data);

        if (status == NVMEDIA_STATUS_OK) {
            /* Set CSI2 lane count per Phy */
            /* Configure Lane count for Port A in x4 */
            mipiOutputReg.address = 0x090AU;
            mipiOutputReg.data = (mipiOutMode ==
                    CDI_MAX96712_MIPI_OUT_4x2) ? 0x40U : 0xC0U;
            mipiOutputReg.data |= (phyMode == CDI_MAX96712_PHY_MODE_CPHY) ? (1U << 5U) : 0U;
            for (i = 0U; i < 2U; i++) {
                mipiOutputReg.address = 0x090AU + (i * 0x40U);
                status = MAX96712WriteUint8Verify(handle,
                                                  mipiOutputReg.address,
                                                  (uint8_t)mipiOutputReg.data);
                if (status != NVMEDIA_STATUS_OK) {
                    break;
                }
            }

            /* Configure Lane count for Port B */
            mipiOutputReg.address = 0x090AU;
#if !NV_IS_SAFETY || SAFETY_DBG_OV
            mipiOutputReg.data =
                (drvHandle->ctx.cfgPipeCopy == CDI_MAX96712_CFG_PIPE_COPY_MODE_2) ? 0x40U : 0xC0U;
#else
            mipiOutputReg.data = 0xC0U;
#endif
            mipiOutputReg.data |= (phyMode == CDI_MAX96712_PHY_MODE_CPHY) ? (1U << 5U) : 0U;
            for (i = 2U; i < 4U; i++) {
                mipiOutputReg.address = 0x090AU + (i * 0x40U);
                status = MAX96712WriteUint8Verify(handle,
                                                  mipiOutputReg.address,
                                                  toUint8FromUint16(mipiOutputReg.data));
                if (status != NVMEDIA_STATUS_OK) {
                    break;
                }
            }

            if (status == NVMEDIA_STATUS_OK) {
                status = ConfigureMIPIOutputHelper(
                          handle,
                          mipiSpeed,
                          phyMode);
            }
        }
    }

    return status;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
static NvMediaStatus
DisableDE(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t i = 0U;

    for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            ClearRegFieldQ(handle);

            status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_DE_EN_PHY_A, i,
                                               0U);
            status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_DE_PRBS_TYPE_PHY_A, i,
                                               1U);

            status = Max96712AccessRegField(status, handle, REG_WRITE_MODE);
        }
    }

    return status;
}

static NvMediaStatus
SetDBL(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link,
    bool enable)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DevBlkCDII2CReg dblReg = {0x0B07, 0x8C};
    uint16_t i = 0U;

    if (enable == false) {
        dblReg.data = 0x0U;
    }

    for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            dblReg.address += (i << 8U);
            status = MAX96712WriteUint8Verify(handle,
                                              dblReg.address,
                                              (uint8_t)dblReg.data);
            if (status != NVMEDIA_STATUS_OK) {
                break;
            }
            /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
            (void)nvsleep(5000);
        }
    }
    return status;
}

static NvMediaStatus
ControlForwardChannels(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link,
    bool enable)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    I2CPortMAX96712 i2cPort = drvHandle->ctx.i2cPort;
    uint8_t i = 0U;
    uint8_t data = 0U;

    for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            /* WAR Bug 2411206 and 200653773:
               Sometimes when reading the 0x0B04/0x0C04/0x0D04/0x0E04 registers,
               0x00 is returned, regardless of what was written to the register.
               To get around this issue, program the register with i2c write command
               directly, instead of using read-modify-write method with access field
               queue.
             */

            /* Set forward control channel bit if enabled */
            if (enable) {
                data |= 0x1U;
            }

            /* Always set reverse control channel bit to 1 */
                data |= 0x2U;

            /* Set I2C/UART port bit for Port 1 */
            if (i2cPort == CDI_MAX96712_I2CPORT_1) {
                data |= 0x8U;
            }

            status =
                MAX96712WriteUint8Verify(handle,
                        regBitFieldProps[(uint8_t)REG_FIELD_I2C_FWDCCEN_PHY_A + i].regAddr,
                        data);
            /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
            (void)nvsleep(10000);
        }
    }

    return status;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

NvMediaStatus
MAX96712CheckLink(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link,
    LinkLockTypeMAX96712 const linkType,
    bool display)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 * drvHandle = getHandlePrivMAX96712(handle);
    GMSLModeMAX96712 gmslMode = CDI_MAX96712_GMSL_MODE_INVALID;
    uint8_t i = 0U, linkIndex = 0U, success = 0U;
    bool exitFlag = false;

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status =  NVMEDIA_STATUS_BAD_PARAMETER;
        exitFlag = true;
    }

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    if (drvHandle->ctx.tpgEnabled == true) {
        status =  NVMEDIA_STATUS_OK;
        exitFlag = true;
    }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

    if ( exitFlag == false) {
        for (linkIndex = 0U; linkIndex < MAX96712_MAX_NUM_LINK; linkIndex++) {
            if (MAX96712_IS_GMSL_LINK_SET(link, linkIndex)) {
                gmslMode = drvHandle->ctx.gmslMode[linkIndex];
                /* Check lock for each link */
                switch (linkType) {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
                    case CDI_MAX96712_LINK_LOCK_GMSL1_CONFIG:
                        if (gmslMode != CDI_MAX96712_GMSL1_MODE) {
                            SIPL_LOG_ERR_STR_UINT("MAX96712: Link : Config link lock is only "
                                                  "valid in GMSL1 mode",
                                                  linkIndex);
                            status = NVMEDIA_STATUS_ERROR;
                            break;
                        }

                        /* Check for GMSL1 Link Lock.*/
                        ClearRegFieldQ(handle);
                        status = Max96712AddRegFieldOffset(status, handle,
                                                           REG_FIELD_GMSL1_LOCK_A, linkIndex,
                                                           0U);
                        status = Max96712AddRegFieldOffset(status, handle,
                                                           REG_FIELD_GMSL1_CONFIG_LOCK_A, linkIndex,
                                                           0U);
                        /* From Max96712 programming guide V1.1, typical link rebuilding time is 25
                         * ~ 100ms. check the link lock in 100ms periodically every 10ms */
                        for (i = 0U; i < 50U; i++) {
                            status = Max96712AccessRegField(status, handle, REG_READ_MODE);
                            if (status == NVMEDIA_STATUS_OK) {
                                if ((ReadFromRegFieldQ(handle, 0U) == 1U) ||
                                    (ReadFromRegFieldQ(handle, 1U) == 1U))  {
                                    PrintLogMsg(LOG_LEVEL_DEBUG, "MAX96712: Link %u: GMSL1 config "
                                                                 "link lock after %u ms\n",
                                                linkIndex, (i * 10U));
                                    success = 1U;
                                    break;
                                }
                                /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
                                (void)nvsleep(10000);
                            }
                        }

                        if (success == 1U) {
                            success = 0U;
                        } else {
                            if (display) {

                                SIPL_LOG_ERR_STR_2UINT("MAX96712: Link: GMSL1 config link lock not detected",

                                                        linkIndex, i);

                            }
                            status =  NVMEDIA_STATUS_ERROR;
                        }

                        break;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
                    case CDI_MAX96712_LINK_LOCK_GMSL2:
                        if (!IsGMSL2Mode(gmslMode)) {
                            SIPL_LOG_ERR_STR_2UINT("MAX96712: Link: GMSL2 link lock is only valid in GMSL2 mode, mode",
                                                    linkIndex, (uint32_t)gmslMode);
                            status =  NVMEDIA_STATUS_ERROR;
                            break;
                        }

                        /* Only register 0x001A is available on MAX96712 Rev 1 to check
                            * link lock in GMSL2 mode*/
                        if ((drvHandle->ctx.revision == CDI_MAX96712_REV_1) &&
                                                        (linkIndex > 0U)) {
                            PrintLogMsg(LOG_LEVEL_DEBUG, "%s: GMSL2 link lock for link %u is not "
                                                         "available on MAX96712 Rev 1\n",
                                        linkIndex);
                            status = NVMEDIA_STATUS_OK;
                            break;
                        }

                        /* From Max96712 programming guide V1.1, typical link
                            * rebuilding time is 25 ~ 100ms
                            * check the link lock in 100ms periodically
                            * TODO : Intermittently the link lock takes more than 100ms.
                            * Check it with MAXIM */
                        for (i = 0U; i < 50U; i++) {
                            status = Max96712AccessOneRegFieldOffset(status, handle,
                                                    REG_FIELD_GMSL2_LOCK_A,
                                                    linkIndex,
                                                    0U, REG_READ_MODE);

                            if (status == NVMEDIA_STATUS_OK) {
                                if (ReadFromRegFieldQ(handle, 0U) == 1U)  {
                                    PrintLogMsg(LOG_LEVEL_DEBUG, "MAX96712: Link %u: GMSL2 link "
                                                                 "lock after %u ms\n",
                                                linkIndex, (i * 10U));
                                    success = 1U;
                                    drvHandle->ctx.linkHasBeenLocked[linkIndex] = true;
                                    break;
                                }
                                /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
                                (void)nvsleep(10000);
                            }
                        }
                        if (success == 1U) {
                            if (i > 10U) {
                                PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: GMSL2 Link "
                                                            "time %d\n", i * 10U);
                            }
                            success = 0U;
                        } else {
                            if (display) {
                                SIPL_LOG_ERR_STR_UINT("MAX96712: Link : GMSL2 link lock not detected",
                                                        linkIndex);
                            }
                            if (status == NVMEDIA_STATUS_OK) {
                                status = NVMEDIA_STATUS_TIMED_OUT;
                            }
                            break;
                        }
                        break;
                    default:
                        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Invalid link type");
                        status =  NVMEDIA_STATUS_BAD_PARAMETER;
                        break;
                }
            }
        }
    }
    return status;
}

/**
 * @brief  Verifies read/write operations on deserializer register are possible
 *
 * - initialize uint8_t const valuesToWrite[2] = {0x55U, 0xAAU}
 * - write settings to deserializer by
 *  - status = Max96712AccessOneRegField(status, handle, reg,
 *   -  0U, REG_READ_MODE)
 *   .
 *  .
 * - read deserializer register value from bitReg queue into regFieldValToRestore
 *  - regFieldVal = ReadFromRegFieldQ(handle, 0U)
 *  - regFieldValToRestore = regFieldVal
 *  .
 * - set RegBitField reg = REG_FIELD_INTR10
 * - for index = 0 , index <= size of valuesToWrite, index++
 *  - if (index < size of valuesToWrite)
 *   - set valueToWrite = valuesToWrite[index]
 *   .
 *  - else
 *   - set valueToWrite = regFieldValToRestore
 *   .
 *  - set  expectedReadBackValue = valueToWrite
 *  - write valueToWrite to reg by
 *   - status = Max96712AccessOneRegField(
 *    - status, handle, reg, valueToWrite,
 *    - REG_WRITE_MODE);
 *    .
 *   .
 *  - read back register value by
 *   - status = Max96712AccessOneRegField(
 *    - status, handle, reg, 0U, REG_READ_MODE)
 *    .
 *   - regFieldVal = ReadFromRegFieldQ(handle, 0U)
 *   .
 *  - verify regFieldVal contains expectedReadBackValue value
 *   - exit error if no match
 *   .
 *  .
 * @param[in] handle DEVBLK handle
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712VerifyI2cReadWrite(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint32_t i = 0U;
    uint8_t regFieldVal = 0U;
    uint8_t regFieldValToRestore = 0U;
    uint8_t valueToWrite = 0U;
    uint8_t const valuesToWrite[2] = {0x55U, 0xAAU};
    uint8_t expectedReadBackValue = 0U;

    /* INTR10[7:0] has MAX_RT_OEN_[A|B|C|D] in bit 0-3, and RT_CNT_OEN_[A|B|C|D] in bit 4-7. */
    RegBitField reg = REG_FIELD_INTR10;
    status = Max96712AccessOneRegField(status, handle, reg,
                                 0U,
                                 REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);

    regFieldValToRestore = regFieldVal;

    for (i = 0U; i <= sizeof(valuesToWrite); i++) {
        if (i < sizeof(valuesToWrite)) {
            valueToWrite = valuesToWrite[i];
        } else {
            valueToWrite = regFieldValToRestore;
        }
        expectedReadBackValue = valueToWrite;
        status = Max96712AccessOneRegField(status, handle, reg,
                                     valueToWrite,
                                     REG_WRITE_MODE);
        status = Max96712AccessOneRegField(status, handle, reg,
                                     0U,
                                     REG_READ_MODE);
        regFieldVal = ReadFromRegFieldQ(handle, 0U);
        if (regFieldVal != expectedReadBackValue) {
            status = NVMEDIA_STATUS_ERROR;
            break;
        }
    }

    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX96712: i2c integrity check failed!");
    }

    return status;
}

NvMediaStatus
MAX96712CheckPresence(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 *drvHandle = getHandlePrivMAX96712(handle);
    RevisionMAX96712 revision = CDI_MAX96712_REV_INVALID;
    uint8_t revisionVal = 0U;
    /* These values must include all of values in the RevisionMAX96712 enum */
    static const MAX96712SupportedRevisions supportedRevisionsMAX96712[] = {
        {CDI_MAX96712_REV_1, 0x1U},
        {CDI_MAX96712_REV_2, 0x2U},
        {CDI_MAX96712_REV_3, 0x3U},
        {CDI_MAX96712_REV_4, 0x4U},
        {CDI_MAX96712_REV_5, 0x5U},
    };
    uint32_t numRev = (uint32_t)(sizeof(supportedRevisionsMAX96712) /
                                 sizeof(supportedRevisionsMAX96712[0]));
    uint8_t devID = 0U;
    uint32_t i = 0U;

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if(status == NVMEDIA_STATUS_OK) {
        /* Check device ID */
        status = Max96712AccessOneRegField(status, handle, REG_FIELD_DEV_ID,
                                     0U,
                                     REG_READ_MODE);
        devID = ReadFromRegFieldQ(handle, 0U);
        if ((devID != MAX96712_DEV_ID) && (devID != MAX96722_DEV_ID)) {
            SIPL_LOG_ERR_STR_2UINT("MAX96712: Device ID mismatch. Expected:, Readval:",
                                   (uint32_t)MAX96712_DEV_ID, devID);
            SIPL_LOG_ERR_STR_2UINT("MAX96722: Device ID mismatch. Expected:, Readval:",
                                   (uint32_t)MAX96722_DEV_ID, devID);
            status = NVMEDIA_STATUS_ERROR;
        }
    }

    if(status == NVMEDIA_STATUS_OK) {
        /* Check revision ID */
        status = Max96712AccessOneRegField(status, handle, REG_FIELD_DEV_REV,
                                     0U,
                                     REG_READ_MODE);

        if (status == NVMEDIA_STATUS_OK) {
            revisionVal = ReadFromRegFieldQ(handle, 0U);

            status = NVMEDIA_STATUS_NOT_SUPPORTED;
            for (i = 0U; i < numRev; i++) {
                if (revisionVal == supportedRevisionsMAX96712[i].revVal) {
                    revision = supportedRevisionsMAX96712[i].revId;
                    PrintLogMsg(LOG_LEVEL_NONE,"MAX96712: Revision %u detected\n", revision);

                    if (revision == CDI_MAX96712_REV_1) {
                        PrintLogMsg(LOG_LEVEL_NONE,"MAX96712: Warning: MAX96712 revision 1 "
                                    "detected. All features may not be supported\n"
                                    "Please use a platform with MAX96712 revision 2 or "
                                    "higher for full support\n");
                        PrintLogMsg(LOG_LEVEL_NONE,"And the below error can be observed"
                                    "  - FE_FRAME_ID_FAULT on CSIMUX_FRAME : Frame IDs are "
                                    "mismatched between FS and FE packets\n");
                    }
                    drvHandle->ctx.revision = revision;
                    status = NVMEDIA_STATUS_OK;
                    break;
                }
            }

            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_UINT("MAX96712: Unsupported MAX96712 revision detected! "
                                      "Supported revisions are: ",
                                      revisionVal);
                for (i = 0U; i < numRev; i++) {
                    PrintLogMsg(LOG_LEVEL_NONE,"MAX96712: Revision %u\n",
                                supportedRevisionsMAX96712[i].revVal);
                }
            } else {
                status = MAX96712VerifyI2cReadWrite(handle);
            }
        }
    }

    return status;
}

/**
 * @brief Verifies i2cPort is a valid value:
 * - compares i2cPort to either
 *  - CDI_MAX96712_I2CPORT_0
 *  - CDI_MAX96712_I2CPORT_1
 *  .
 * - returns error if no match
 * @param[in] i2cPort value to be validated
 * @return NVMEDIA_STATUS_OK if valid, otherwise an error code */
static bool
isValidI2cPort(
    I2CPortMAX96712 const i2cPort)
{
    bool status = false;

    if ((i2cPort == CDI_MAX96712_I2CPORT_0) ||
        (i2cPort == CDI_MAX96712_I2CPORT_1)) {
       status = true;
    }
    return status;
}

/**
 * @brief Verifies txPort is a valid value
 *
 * - compares txPort to either
 *  - CDI_MAX96712_TXPORT_PHY_C
 *  - CDI_MAX96712_TXPORT_PHY_D
 *  - CDI_MAX96712_TXPORT_PHY_E
 *  - CDI_MAX96712_TXPORT_PHY_F
 *  .
 * - returns error if no match
 * @param[in] txPort value to be validated
 * @return NVMEDIA_STATUS_OK if valid, otherwise an error code */
static bool
isValidTxPort(
    TxPortMAX96712 const txPort)
{
    bool status = false;

    if ((txPort == CDI_MAX96712_TXPORT_PHY_C) ||
        (txPort == CDI_MAX96712_TXPORT_PHY_D) ||
        (txPort == CDI_MAX96712_TXPORT_PHY_E) ||
        (txPort == CDI_MAX96712_TXPORT_PHY_F)) {
       status = true;
    }
    return status;
}

/**
 * @brief Verifies mipiOutMode is a valid value:
 *
 * - compares txPort to either
 *  - CDI_MAX96712_MIPI_OUT_4x2
 *  - CDI_MAX96712_MIPI_OUT_2x4
 *  .
 * - returns error if no match
 * @param[in] mipiOutMode value to be validated
 * @return NVMEDIA_STATUS_OK if valid, otherwise an error code */
static bool
isValidMipiOutMode(
    MipiOutModeMAX96712 const mipiOutMode)
{
    bool status = false;

    if ((mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2) ||
        (mipiOutMode == CDI_MAX96712_MIPI_OUT_2x4) ||
        (mipiOutMode == CDI_MAX96712_MIPI_OUT_4a_2x2)) {
       status = true;
    }
    return status;
}

/**
 * @brief Verify validity of context setting
 *
 * - verify context' i2cPort by
 *  - isValidI2cPort(ctx->i2cPort)
 *  .
 * - verify context' txPort by
 *  - isValidTxPort(ctx->txPort)
 *  .
 * - verify context' mipiOutMode by
 *  - isValidMipiOutMode(ctx->mipiOutMode)
 *  .
 * - returns true if ok, false otherwiae
 * @param[in] ctx context to b verified
 * @return NVMEDIA_STATUS_OK if valid, otherwise an error code */
static bool
isValidPortsCfg(
    ContextMAX96712 const* ctx)
{
    bool status = true;

    if (!isValidI2cPort(ctx->i2cPort)) {
        SIPL_LOG_ERR_STR("MAX96712: Invalid I2c port");
        status = false;
    } else if (!isValidTxPort(ctx->txPort)) {
        SIPL_LOG_ERR_STR("MAX96712: Invalid Tx port");
        status = false;
    } else if (!isValidMipiOutMode(ctx->mipiOutMode)) {
        SIPL_LOG_ERR_STR("MAX96712: Invalid MIPI output port");
        status = false;
    } else {
        /* Do nothing */
    }
    return status;
}

/**
 * @brief Verifies context gmslMode is invalid
 *
 * - returns true if
 *  - (!IsGMSL2Mode(ctx->gmslMode[linkIndex]) and
 *  - (ctx->gmslMode[linkIndex] != CDI_MAX96712_GMSL_MODE_UNUSED))
 * @param[in] ctx context to be verified
 * @param[in] linkIndex index into gmslMode array in context
 * @return true of invalid setting */
static bool
isGMSLModeInvalid(ContextMAX96712 const* ctx, uint8_t linkIndex)
{
    bool status = false;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    status = ((ctx->gmslMode[linkIndex] != CDI_MAX96712_GMSL1_MODE) &&
            !IsGMSL2Mode(ctx->gmslMode[linkIndex]) &&
            (ctx->gmslMode[linkIndex] != CDI_MAX96712_GMSL_MODE_UNUSED));
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#else
    status = (!IsGMSL2Mode(ctx->gmslMode[linkIndex]) &&
            (ctx->gmslMode[linkIndex] != CDI_MAX96712_GMSL_MODE_UNUSED));
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
    return status;
}

/**
 * @brief Initializes deserializer device driver
 *
 * - converts the type client context to DriverHandleMAX96712 by
 *  - ContextMAX96712 const* ctx = (ContextMAX96712 const*) clientContext;
 *  .
 * - verifies handle and clientContext are not NULL
 * - Check GMSL Mode in supplied context by
 *  - for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++)
 *   - if(isGMSLModeInvalid(ctx, i))
 *    - exit error NVMEDIA_STATUS_BAD_PARAMETER "MAX96712: Invalid GMSL mode"
 *    .
 *   .
 *  .
 * - check i2cPort in supplied context by
 *  - isValidPortsCfg(ctx)
 *  .
 * - Allocates memory for device driver handle by
 *  - drvHandle = calloc(1, sizeof(DriverHandleMAX96712))
 *  .
 * - verify memory was allocated (drvHandle is not NULL)
 * - copy clientContext into device driver handle->ctx
 * - store drvHandle in handle->deviceDriverHandle
 * - set drvHandle->ctx.revision = CDI_MAX96712_REV_INVALID
 * - set drvHandle->ctx.manualFSyncFPS = 0U
 * - Create the I2C programmer for register read/write by
 *  -    drvHandle->i2cProgrammer =
 *   - DevBlkCDII2CPgmrCreate(handle,
 *   - MAX96712_NUM_ADDR_BYTES,
 *   - MAX96712_NUM_DATA_BYTES)
 *   .
 *  .
 *  - verify i2cProgrammer is not NULL
 *
 * @param[in] handle DEVBLK handle
 * @param[in] clientContext defined context setting
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
DriverCreateMAX96712(
    DevBlkCDIDevice *handle,
    void const* clientContext)
{
    DriverHandleMAX96712 *drvHandle = NULL;
   /* coverity[misra_c_2012_rule_11_5_violation] : intentional TID-1417 */
    ContextMAX96712 const* ctx = (ContextMAX96712 const*) clientContext;
    uint8_t i = 0U;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if ((NULL == handle) || (clientContext == NULL)) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        /* Check supplied context */
        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            if(isGMSLModeInvalid(ctx, i)) {
                SIPL_LOG_ERR_STR("MAX96712: Invalid GMSL mode");
                status = NVMEDIA_STATUS_BAD_PARAMETER;
                break;
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            if (!isValidPortsCfg(ctx)) {
                SIPL_LOG_ERR_STR("MAX96712: Invalid Ports Cfg");
                status = NVMEDIA_STATUS_BAD_PARAMETER;
            } else {
                /* coverity[misra_c_2012_rule_11_5_violation] : intentional TID-1417 */
                /* coverity[misra_c_2012_rule_21_3_violation] : intentional TID-1493 */
                drvHandle = calloc(1, sizeof(DriverHandleMAX96712));
                if (drvHandle == NULL) {
                    SIPL_LOG_ERR_STR("MAX96712: Memory allocation for context failed");
                    status =  NVMEDIA_STATUS_OUT_OF_MEMORY;
                } else {
                    (void)memmove(&drvHandle->ctx, ctx, sizeof(ContextMAX96712));
                    drvHandle->ctx.revision = CDI_MAX96712_REV_INVALID;
                    drvHandle->ctx.manualFSyncFPS = 0U;
                    handle->deviceDriverHandle = (void *)drvHandle;

                    /* Create the I2C programmer for register read/write */
                    drvHandle->i2cProgrammer =
                        DevBlkCDII2CPgmrCreate(handle,
                                                MAX96712_NUM_ADDR_BYTES,
                                                MAX96712_NUM_DATA_BYTES);
                    if (drvHandle->i2cProgrammer == NULL) {
                        SIPL_LOG_ERR_STR("Failed to initialize the I2C programmer");
                        /* coverity[misra_c_2012_rule_21_3_violation] : intentional TID-1493 */
                        free(drvHandle);
                        status = NVMEDIA_STATUS_ERROR;
                    }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
                    else {
                        for (i = 0U; i < MAX96712_MAX_NUM_PG; i++) {
                            drvHandle->ctx.pgMode[i] = CDI_MAX96712_PG_NUM;
                        }
                    }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
                }
            }
        }
    }
    return status;
}

/**
 * @brief Destroys device driver instance
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - verifies device driver handle is not null
 * - Destroy the I2C programmer by
 *  - calling DevBlkCDII2CPgmrDestroy(drvHandle->i2cProgrammer)
 *  .
 * - free device driver handle memory by
 *  - free(handle->deviceDriverHandle)
 *  .
 * - sets handle->deviceDriverHandle to NULL
 * @param[in] handle DEVBLK handle
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
DriverDestroyMAX96712(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        /* Destroy the I2C programmer */
        DevBlkCDII2CPgmrDestroy(drvHandle->i2cProgrammer);

        if (handle->deviceDriverHandle != NULL) {
            /* coverity[misra_c_2012_rule_21_3_violation] : intentional TID-1493 */
            free(handle->deviceDriverHandle);
            handle->deviceDriverHandle = NULL;
        }
    }
    return status;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
#define MAX96712_REG_MAX_ADDRESS              0x1F03U

NvMediaStatus
MAX96712DumpRegisters(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    uint8_t data = 0U;
    uint32_t i = 0U;

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        for (i = 0U; i <= MAX96712_REG_MAX_ADDRESS; i++) {
            /* Note: skipped for Read Verify since it dump all the registers.
             */
            status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                               ((i / 256U) << 8U) | (i % 256U),
                                               &data);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_AND_UINT("MAX96712: Register I2C read failed with status",
                                              i, (uint32_t)status);
                break;
            }

            PrintLogMsg(LOG_LEVEL_NONE,"Max96712: 0x%04X%02X - 0x%02X\n",
                        (i / 256U), (i % 256U), data);
        }
    }

    return status;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/**
 * @brief Gets Global Error Status Ctrl3
 *
 * - Note:ctrl3 (R0x1A)
 *  -    ctrl3[3]: LOCKED (only for Link A)
 *  -    ctrl3[2]: ERROR
 *  -    ctrl3[1]: CMU_LOCKED
 *  -    rest bits are reserved.
 *  .
 * - reads REG_FIELD_CTRL3 register to regFieldVal by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_CTRL3, 0U, REG_READ_MODE)
 *   . regFieldVal = ReadFromRegFieldQ(handle, 0U)
 *  -
 * - if ((regFieldVal & 0x02U) == 0U)
 *  -    Note: global CMU locked status , report error by
 *   -        Max96712UpdateGlobalError(
 *    -          errorStatus, globalErrorCount,
 *    -          regFieldVal,
 *    -          CDI_MAX96712_GLOBAL_CMU_UNLOCK_ERR,
 *    -          false)
 *    .
 *   .
 *  .
 * - if ((regFieldVal & 0x04U) != 0U)
 *  - Note: global ERRB status , report error by
 *   -       Max96712UpdateGlobalError(
 *    -          errorStatus,
 *    -          globalErrorCount,
 *    -          regFieldVal,
 *    -          CDI_MAX96712_GLOBAL_ERR,
 *    -          true)
 *    .
 *   .
 *  .
 * - Note: The reporting will update the errorStatus & globalErrorCount for the caller
 *
 * - Note: this function is called LAST in a series of status collection calls
 *  - That is why the last reported error in this function is marked as "last"
 *
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[out] globalErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetGlobalErrorStatusCtrl3(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t *globalErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    /* ctrl3 (R0x1A)
    * ctrl3[3]: LOCKED (only for Link A)
    * ctrl3[2]: ERROR
    * ctrl3[1]: CMU_LOCKED
    * rest bits are reserved. */
    status = Max96712AccessOneRegField(status, handle, REG_FIELD_CTRL3,
                                 0U,
                                 REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);

    if ((regFieldVal & 0x2U) == 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: global CMU locked status",
                              regFieldVal);
        Max96712UpdateGlobalError(errorStatus, globalErrorCount, regFieldVal,
                                  CDI_MAX96712_GLOBAL_CMU_UNLOCK_ERR, false);
    }

    if ((regFieldVal & 0x4U) != 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: global ERRB status", regFieldVal);
        Max96712UpdateGlobalError(errorStatus, globalErrorCount, regFieldVal,
                                  CDI_MAX96712_GLOBAL_ERR, true);
    }

    return status;
}

/**
 * @brief Get Global Error Status Line Fault Err
 *
 * - Note:intr5 (R0x28)
 *  -    intr5[2]: LFLT_INT
 *  -    (rest bits are for RTTN_CRC_INT, WM, EOM link based errors)
 *  .
 * - reads REG_FIELD_INTR5 register from deserializer by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_INTR5, 0U, REG_READ_MODE)
 *   .
 *  - regFieldVal = ReadFromRegFieldQ(handle, 0U)
 *  .
 * - if ((regFieldVal & 0x04U) != 0U)
 *  - checks for global line fault error in bit 2
 *  - report error by
 *   - Max96712UpdateGlobalError(
 *    - errorStatus,
 *    - globalErrorCount,
 *    - regFieldVal,
 *    - CDI_MAX96712_GLOBAL_LINE_FAULT,
 *    - false)
 *    .
 *   .
 *  .
 * Note: The reporting will update the errorStatus & globalErrorCount for the caller
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[out] globalErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetGlobalErrorStatusLineFaultErr(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t *globalErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    /* intr5 (R0x28)
    * intr5[2]: LFLT_INT
    * (rest bits are for RTTN_CRC_INT, WM, EOM link based errors) */
    status = Max96712AccessOneRegField(status, handle, REG_FIELD_INTR5,
                                 0U,
                                 REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    if ((regFieldVal & 0x4U) != 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: global line fault error in bit 2:",
                              regFieldVal);
        Max96712UpdateGlobalError(errorStatus, globalErrorCount, regFieldVal,
                                  CDI_MAX96712_GLOBAL_LINE_FAULT, false);
    }

    return status;
}

/**
 * @brief Get Global Error Status Intr7
 *
 * - note:intr7 (R0x2A)
 *  - intr7[3]: LCRC_ERR_FLAG
 *  - intr7[2]: VPRBS_ERR_FLAG
 *  - intr7[1]: REM_ERR_FLAG
 *  - intr7[0]: FSYNC_ERR_FLAG
 *  - (rest bits are for G1 link based errors,
 *  - note we use R0xBCB than R0xB etc in later code)
 *  .
 * - read REG_FIELD_INTR7 register into regFieldVal by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_INTR7,  0U, REG_READ_MODE)
 *   .
 *  - regFieldVal = ReadFromRegFieldQ(handle, 0U)
 *  .
 * - if ((regFieldVal & 0x08U) != 0U)
 *  - error : global video line crc error in bit 3, report by
 *   - Max96712UpdateGlobalError(
 *    - errorStatus,
 *    - globalErrorCount,
 *    - regFieldVal,
 *    - CDI_MAX96712_GLOBAL_VID_LINE_CRC,
 *    - false)
 *    .
 *   .
 *  .
 * - if ((regFieldVal & 0x4U) != 0U)
 *  - error : global video PRBS error in bit 2, report by
 *   - Max96712UpdateGlobalError(
 *    - errorStatus,
 *    - globalErrorCount,
 *    - regFieldVal,
 *    - CDI_MAX96712_GLOBAL_VID_PRBS,
 *    - false)
 *    .
 *   .
 *  .
 * - if ((regFieldVal & 0x2U) != 0U)
 *  - error : global remote side error in bit 1, report by
 *   -  Max96712UpdateGlobalError(
 *    - errorStatus,
 *    - globalErrorCount,
 *    - regFieldVal,
 *    - CDI_MAX96712_GLOBAL_REMOTE_SIDE,
 *    - false)
 *    .
 *   .
 *  .
 * - if ((regFieldVal & 0x1U) != 0U)
 *  - error : global frame sync error in bit 0, report by
 *   - Max96712UpdateGlobalError(
 *    - errorStatus,
 *    - globalErrorCount,
 *    - regFieldVal,
 *    - CDI_MAX96712_GLOBAL_FRAME_SYNC,
 *    - false)
 *    .
 *   .
 *  .
 * .
 *
 * Note: The reporting will update the errorStatus & globalErrorCount for the caller
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[out] globalErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetGlobalErrorStatusIntr7(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t *globalErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    /* intr7 (R0x2A)
    * intr7[3]: LCRC_ERR_FLAG
    * intr7[2]: VPRBS_ERR_FLAG
    * intr7[1]: REM_ERR_FLAG
    * intr7[0]: FSYNC_ERR_FLAG
    * (rerst bits are for G1 link based errors, note we use R0xBCB than R0xB etc in later code) */
    status = Max96712AccessOneRegField(status, handle, REG_FIELD_INTR7,
                                 0U,
                                 REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    if ((regFieldVal & 0x8U) != 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: global video line crc error in bit 3:",
                              regFieldVal);
        Max96712UpdateGlobalError(errorStatus, globalErrorCount, regFieldVal,
                                  CDI_MAX96712_GLOBAL_VID_LINE_CRC, false);
    }
    if ((regFieldVal & 0x4U)!= 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: global video PRBS error in bit 2:",
                              regFieldVal);
        Max96712UpdateGlobalError(errorStatus, globalErrorCount, regFieldVal,
                                  CDI_MAX96712_GLOBAL_VID_PRBS, false);
    }
    if ((regFieldVal & 0x2U)!= 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: global remote side error in bit 1:",
                              regFieldVal);
        Max96712UpdateGlobalError(errorStatus, globalErrorCount, regFieldVal,
                                  CDI_MAX96712_GLOBAL_REMOTE_SIDE, false);
    }
    if ((regFieldVal & 0x1U)!= 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: global frame sync error in bit 0:",
                              regFieldVal);
        Max96712UpdateGlobalError(errorStatus, globalErrorCount, regFieldVal,
                                  CDI_MAX96712_GLOBAL_FRAME_SYNC, false);
    }

    return status;
}

/**
 * @brief Get Global Error Status Vid Pxl Crc Err Int
 *
 * - Note:vid_pxl_crc_err_int (R0x45)
 *  - vid_pxl_crc_err_int[7]: mem ecc 2
 *  - vid_pxl_crc_err_int[6]: mem ecc 1
 *  - (rest bits are for video pixel crc link based errors)
 *  .
 * - reads REG_FIELD_VID_PXL_CRC_ERR_INT register into regFieldVal by
 *  . status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_VID_PXL_CRC_ERR_INT, 0U, REG_READ_MODE)
 *   .
 *  - regFieldVal = ReadFromRegFieldQ(handle, 0U)
 *  .
 * - if ((regFieldVal & 0x80U) != 0U)
 *  - error : global mem ecc 2 error in bit 7, report by
 *   - Max96712UpdateGlobalError(
 *    - errorStatus,
 *    - globalErrorCount,
 *    - regFieldVal,
 *    - CDI_MAX96712_GLOBAL_MEM_ECC2,
 *    - false)
 *  - clear error by:
 *   - DevBlkCDII2CPgmrWriteUint8(
 *          drvHandle->i2cProgrammer
 *          MAX96712_REG_MEM_ECC0,
 *          data)
 *    .
 *   .
 *  .
 * - if ((regFieldVal & 0x40U) != 0U)
 *  - error : global mem ecc error in bit 6, report by
 *   - Max96712UpdateGlobalError(
 *    - errorStatus,
 *    - globalErrorCount,
 *    - regFieldVal,
 *    - CDI_MAX96712_GLOBAL_MEM_ECC1,
 *    - false)
 *  - clear error by:
 *   - DevBlkCDII2CPgmrWriteUint8(
 *          drvHandle->i2cProgrammer
 *          MAX96712_REG_MEM_ECC0,
 *          data)
 *    .
 *   .
 *  .
 * - Note: The reporting will update the errorStatus & globalErrorCount for the caller
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[out] globalErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetGlobalErrorStatusVidPxlCrcErrInt(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t *globalErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    /* vid_pxl_crc_err_int (R0x45)
    * vid_pxl_crc_err_int[7]: MEM_ECC_ERR2_INT
    * vid_pxl_crc_err_int[6]: MEM_ECC_ERR1_INT
    * (rest bits are for video pixel crc link based errors) */
    status = Max96712AccessOneRegField(status, handle, REG_FIELD_VID_PXL_CRC_ERR_INT,
                                 0U,
                                 REG_READ_MODE);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: Failed to read "
            "REG_FIELD_VID_PXL_CRC_ERR_INT", (uint32_t)status);
            goto end;
    }
    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    if ((regFieldVal & 0x80U) != 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: MEM_ECC_ERR2_INT bit is set",
                              regFieldVal);
        Max96712UpdateGlobalError(errorStatus, globalErrorCount, regFieldVal,
                                  CDI_MAX96712_GLOBAL_MEM_ECC2, false);
        /* Clear MEM_ECC_ERR2_INT flag */
        uint8_t clear_data = 0x2U;
        status = MAX96712WriteUint8Verify(handle,
                                          MAX96712_REG_MEM_ECC0,
                                          clear_data);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_UINT("MAX96712: Failed to clear 2-bit "
                "Uncorrectable Memory ECC Errors", (uint32_t)status);
            goto end;
        }
    }
    if ((regFieldVal & 0x40U) != 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: MEM_ECC_ERR1_INT bit is set",
                              regFieldVal);
        Max96712UpdateGlobalError(errorStatus, globalErrorCount, regFieldVal,
                                  CDI_MAX96712_GLOBAL_MEM_ECC1, false);
        /* Clear MEM_ECC_ERR1_INT flag */
        uint8_t clear_data = 0x1U;
        status = MAX96712WriteUint8Verify(handle,
                                          MAX96712_REG_MEM_ECC0,
                                          clear_data);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_UINT("MAX96712: Failed to clear 1-bit Correctable "
                "Memory ECC Errors", (uint32_t)status);
        }
    }
end:
    return status;
}

/**
 * @brief Get Global Error Status Fsync22
 *
 * - Note:fsync_22 (R0x4B6)
 *  - fsync_22[7]: FSYNC_LOSS_OF_LOCK
 *  - fsync_22[6]: FSYNC_LOCKED
 *  - rest 6 bits are for FRM_DIFF_H, currently not to report
 *  .
 * - reads REG_FIELD_FSYNC_22 register into regFieldVal by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_FSYNC_22, 0U, REG_READ_MODE)
 *   .
 *  - regFieldVal = ReadFromRegFieldQ(handle, 0U)
 *  .
 * - if ((regFieldVal & 0x80U) != 0U)
 *  - error : global fsync sync loss error in bit 7, report by
 *   - Max96712UpdateGlobalError(
 *    - errorStatus,
 *    - globalErrorCount,
 *    - regFieldVal,
 *    - CDI_MAX96712_GLOBAL_FSYNC_SYNC_LOSS,
 *    - false)
 *    .
 *   .
 *  .
 * - if ((regFieldVal & 0x40U) != 0U)
 *  - error : global fsync status in bit 6, report by
 *   - Max96712UpdateGlobalError(
 *    - errorStatus,
 *    - globalErrorCount,
 *    - regFieldVal,
 *    - CDI_MAX96712_GLOBAL_FSYNC_STATUS,
 *    - false)
 *    .
 *   .
 *  .
 * .
 * Note: The reporting will update the errorStatus & globalErrorCount for the caller
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[out] globalErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetGlobalErrorStatusFsync22(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t *globalErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    /* fsync_22 (R0x4B6)
    * fsync_22[7]: FSYNC_LOSS_OF_LOCK
    * fsync_22[6]: FSYNC_LOCKED
    * (rest 6 bits are for FRM_DIFF_H, currently not to report) */
    status = Max96712AccessOneRegField(status, handle, REG_FIELD_FSYNC_22,
                                 0U,
                                 REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    if ((regFieldVal & 0x80U) != 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: global fsync sync loss error in bit 7:",
                              regFieldVal);
        Max96712UpdateGlobalError(errorStatus, globalErrorCount, regFieldVal,
                                  CDI_MAX96712_GLOBAL_FSYNC_SYNC_LOSS, false);
    }
    if ((regFieldVal & 0x40U) != 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: global fsync status in bit 6:",
                              regFieldVal);
        Max96712UpdateGlobalError(errorStatus, globalErrorCount, regFieldVal,
                                  CDI_MAX96712_GLOBAL_FSYNC_STATUS, false);
    }

    return status;
}

/**
 * @brief Get Global Error Status Video Masked Flag
 *
 * - Note:VIDEO_MASKED_FLAG (R0x04A)
 *  - VIDEO_MASKED_FLAG[5]: CMP_VTERM_STATUS
 *  - VIDEO_MASKED_FLAG[4]: VDD_OV_FLAG
 *  - rest 6 bits are for FRM_DIFF_H, currently not to report
 *  .
 * - reads REG_FIELD_VIDEO_MASKED_FLAG register into regFieldVal by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_VIDEO_MASKED_FLAG,  0U, REG_READ_MODE)
 *   .
 *  - regFieldVal = ReadFromRegFieldQ(handle, 0U)
 *  .
 * - if ((regFieldVal & 0x20U) == 0U)
 *  - error : global fsync sync loss error in bit 7, report by
 *   - Max96712UpdateGlobalError(
 *    - errorStatus,
 *    - globalErrorCount,
 *    - regFieldVal,
 *    - CDI_MAX96712_GLOBAL_CMP_VTERM_STATUS,
 *    - false)
 *    .
 *   .
 *  .
 * - if ((regFieldVal & 0x10U) != 0U)
 *  - error : global fsync status in bit 6, report by
 *   - Max96712UpdateGlobalError(
 *    - errorStatus,
 *    - globalErrorCount,
 *    - regFieldVal,
 *    - CDI_MAX96712_GLOBAL_VDD_OV_FLAG,
 *    - false)
 *    .
 *   .
 *  .
 * - Note: The reporting will update the errorStatus & globalErrorCount for the caller
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[out] globalErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetGlobalErrorStatusVidMaskedFlag(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t *globalErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    /* VIDEO_MASKED_FLAG (R0x04A)
    * VIDEO_MASKED_FLAG[5]: CMP_VTERM_STATUS
    * VIDEO_MASKED_FLAG[4]: VDD_OV_FLAG */
    status = Max96712AccessOneRegField(status, handle, REG_FIELD_VIDEO_MASKED_FLAG,
                                 0U,
                                 REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    if ((regFieldVal & 0x20U) == 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: Vterm is latched low and less than 1v, "
                              "in video masked reg bit 5:",
                              regFieldVal);
        Max96712UpdateGlobalError(errorStatus, globalErrorCount, regFieldVal,
                                  CDI_MAX96712_GLOBAL_CMP_VTERM_STATUS, false);
    }
    if ((regFieldVal & 0x10U) != 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: Vdd_sw overvoltage condition detected, "
                              "in video masked reg bit 4:",
                              regFieldVal);
        Max96712UpdateGlobalError(errorStatus, globalErrorCount, regFieldVal,
                                  CDI_MAX96712_GLOBAL_VDD_OV_FLAG, false);
    }

    return status;
}

/**
 * @brief Get Global Error Status Pwr0
 *
 * - Note:PWR0 (R0x012)
 *  - PWR0[7:5]: VDDBAD_STATUS with bits 5 and bit 6 are effectively used.
 *  - PWR0[4:0]: CMP_STATUS, with bit 0,1,2 are for Vdd18/Vddio/Vdd_sw
 *   - undervoltage latch low indicator
 *   .
 *  .
 * - reads REG_FIELD_PWR0 register into regFieldVal by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_PWR0, 0U, REG_READ_MODE)
 *   .
 *  - regFieldVal = ReadFromRegFieldQ(handle, 0U)
 *  .
 * - if ((regFieldVal & 0x60U) == 0x60U)
 *  - error : Vdd_sw less than 0.82v is observed since last read, report by
 *   - Max96712UpdateGlobalError(
 *    - errorStatus,
 *    - globalErrorCount,
 *    - regFieldVal,
 *    - CDI_MAX96712_GLOBAL_VDDBAD_STATUS,
 *    - false)
 *    .
 *   .
 *  .
 * - if ((regFieldVal & 0x04U) == 0U)
 *  - error : Vdd_sw (1.0v) is latched low (undervoltage), report by
 *   - Max96712UpdateGlobalError(
 *    - errorStatus,
 *    - globalErrorCount,
 *    - regFieldVal,
 *    - CDI_MAX96712_GLOBAL_VDDSW_UV,
 *    - false)
 *    .
 *   .
 *  .
 * - if ((regFieldVal & 0x02U) == 0U)
 *  - error : Vddio (1.8v) is latched low (undervoltage), report by
 *   - Max96712UpdateGlobalError(
 *    - errorStatus,
 *    - globalErrorCount,
 *    - regFieldVal,
 *    - CDI_MAX96712_GLOBAL_VDDIO_UV,
 *    - false)
 *    .
 *   .
 *  .
 * - if ((regFieldVal & 0x01U) == 0U)
 *  - error : Vdd 1.8v is latched low (undervoltage), report by
 *   - Max96712UpdateGlobalError(
 *    - errorStatus,
 *    - globalErrorCount,
 *    - regFieldVal,
 *    - CDI_MAX96712_GLOBAL_VDD18_UV,
 *    - false)
 *
 * - If any of the above errors are detected
 *  - reads REG_FIELD_PWR0 register into regFieldVal
 *   - if (regFieldVal == 0x0U)
 *    - reads REG_FIELD_PWR_STATUS_FLAG register to clear
 *     - PWR_STATUS_FLAG (VDDBAD_INT_FLAG and VDDCMP_INT_FLAG)
 *
 * Note: The reporting will update the errorStatus & globalErrorCount for the caller
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[out] globalErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetGlobalErrorStatusPwr0(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t *globalErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    /* PWR0 (R0x012)
    * PWR0[7:5]: VDDBAD_STATUS with bits 5 and bit 6 are effectively used.
    * PWR0[4:0]: CMP_STATUS, with bit 0,1,2 are for Vdd18/Vddio/Vdd_sw undervoltage
    * latch low indicator */
    bool readAgainPwr0 = false;
    status = Max96712AccessOneRegField(status, handle, REG_FIELD_PWR0,
                                0U,
                                REG_READ_MODE);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: Failed to read REG_FIELD_PWR0",
                                (uint32_t)status);
        goto end;
    }
    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    if ((regFieldVal & 0x60U) == 0x60U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: Vdd_sw less than 0.82v is observed since last read:",
                              regFieldVal);
        Max96712UpdateGlobalError(errorStatus, globalErrorCount, regFieldVal,
                                  CDI_MAX96712_GLOBAL_VDDBAD_STATUS, false);
        readAgainPwr0 = true;
    }
    if ((regFieldVal & 0x4U) == 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: Vdd_sw (1.0v) is latched low (undervoltage), reg value:",
                              regFieldVal);
        Max96712UpdateGlobalError(errorStatus, globalErrorCount, regFieldVal,
                                  CDI_MAX96712_GLOBAL_VDDSW_UV, false);
        readAgainPwr0 = true;
    }
    if ((regFieldVal & 0x2U) == 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: Vddio (1.8v) is latched low (undervoltage), reg value:",
                              regFieldVal);
        Max96712UpdateGlobalError(errorStatus, globalErrorCount, regFieldVal,
                                  CDI_MAX96712_GLOBAL_VDDIO_UV, false);
        readAgainPwr0 = true;
    }
    if ((regFieldVal & 0x1U) == 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: Vdd 1.8v is latched low (undervoltage), reg value:",
                              regFieldVal);
        Max96712UpdateGlobalError(errorStatus, globalErrorCount, regFieldVal,
                                  CDI_MAX96712_GLOBAL_VDD18_UV, false);
        readAgainPwr0 = true;
    }
    if (readAgainPwr0) {
        status = Max96712AccessOneRegField(status, handle, REG_FIELD_PWR0,
                                     0U,
                                     REG_READ_MODE);
        regFieldVal = ReadFromRegFieldQ(handle, 0U);
        if (regFieldVal == 0x0U) {
            PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: all undervoltage latches "
                                        "in PWR0 are cleared\n");

            /* further read clear PWR_STATUS_FLAG (VDDBAD_INT_FLAG and VDDCMP_INT_FLAG) */
            status = Max96712AccessOneRegField(status, handle, REG_FIELD_PWR_STATUS_FLAG,
                                         0U,
                                         REG_READ_MODE);
            regFieldVal = ReadFromRegFieldQ(handle, 0U);
            PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: read clear "
                                        "PWR_STATUS_FLAG (%u)\n", regFieldVal);
        } else {
            /* PWR0 are not cleared, and PWR_STATUS_FLAG will still flag ERRB */
            SIPL_LOG_ERR_STR("MAX96712: not all undervoltage latches are cleared!");
        }
    }
end:
    return status;
}

/**
 * @brief Get all Global Error Status
 *
 * - collects all error status by
 *  - status = MAX96712GetGlobalErrorStatusLineFaultErr(
 *   - handle, errorStatus, &globalErrorCount)
 *   .
 *  - status = MAX96712GetGlobalErrorStatusIntr7(
 *   - handle, errorStatus, &globalErrorCount)
 *   .
 *  - status = MAX96712GetGlobalErrorStatusVidPxlCrcErrInt(
 *   - handle, errorStatus, &globalErrorCount)
 *   .
 *  - status = MAX96712GetGlobalErrorStatusFsync22(
 *   - handle, errorStatus, &globalErrorCount)
 *   .
 *  - status = MAX96712GetGlobalErrorStatusVidMaskedFlag(
 *   - handle,  errorStatus, &globalErrorCount)
 *   .
 *  - status = MAX96712GetGlobalErrorStatusPwr0(
 *   - handle, errorStatus, &globalErrorCount)
 *   .
 *  - status = MAX96712GetGlobalErrorStatusCtrl3(
 *   - handle, errorStatus, &globalErrorCount)
 *   .
 *  .
 * - Note: this MAX96712GetGlobalErrorStatusCtrl3 function is called LAST in the
 *  -series of status collection calls.
 *  - That is why the last reported error in MAX96712GetGlobalErrorStatusCtrl3 is marked as "last"
 * .
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetGlobalErrorStatus(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t globalErrorCount = 0U;

    status = MAX96712GetGlobalErrorStatusLineFaultErr(handle,
                                                      errorStatus,
                                                      &globalErrorCount);

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712GetGlobalErrorStatusIntr7(handle,
                                                   errorStatus,
                                                   &globalErrorCount);
    }

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712GetGlobalErrorStatusVidPxlCrcErrInt(handle,
                                                             errorStatus,
                                                             &globalErrorCount);
    }

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712GetGlobalErrorStatusFsync22(handle,
                                                     errorStatus,
                                                     &globalErrorCount);
    }

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712GetGlobalErrorStatusVidMaskedFlag(handle,
                                                           errorStatus,
                                                           &globalErrorCount);
    }

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712GetGlobalErrorStatusPwr0(handle,
                                                  errorStatus,
                                                  &globalErrorCount);
    }

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712GetGlobalErrorStatusCtrl3(handle,
                                                   errorStatus,
                                                   &globalErrorCount);
    }

    return status;
}

/**
 * @brief Get Pipeline Error Status Overflow
 * set uint8_t pipelnNo = pipelineNum  % MAX96712_NUM_VIDEO_PIPELINES
 * - initialize reporting of error to CDI_MAX96712_PIPELINE_ERROR_INVALID, by
 *  -    placing value in errorStatus->pipelineFailureType[pipelnNo][*pipelineErrorCount]
 *  .
 * - if pipelnNo < = 3
 *  - set lshift = pipelnNo
 *  - read REG_FIELD_OVERFLOW_FIRST4 register into bitReg queue by
 *   - status = Max96712AccessOneRegField(
 *    - status, handle, REG_FIELD_OVERFLOW_FIRST4, 0U, REG_READ_MODE)
 *    .
 *   .
 *  .
 * - else
 *  - set lshift = pipelnNo - 4U
 *  - read REG_FIELD_OVERFLOW_LAST4 register into bitReg queue by
 *   - status = Max96712AccessOneRegField(
 *    - status, handle, REG_FIELD_OVERFLOW_LAST4, 0U, REG_READ_MODE)
 *    .
 *   .
 *  .
 * - extract read value from queue by
 *  - regFieldVal = ReadFromRegFieldQ(handle, 0U)
 *  .
 * - Note: line memory overflow bits are at BACKTOP11 register's bit[3:0]
 * - if bit set in regFieldVal for (1 << leftShift)
 *  - error: pipeline lmo overflow error, report by
 *   - Max96712UpdatePipelineError(
 *    - errorStatus, pipelineErrorCount, regFieldVal, pipelnNo,
 *    - CDI_MAX96712_PIPELINE_LMO_OVERFLOW_ERR, false)
 *    .
 *   .
 *  .
 * - Note: cmd overflow bits are at BACKTOP11 register's bit[7:4]
 * - if bit set in regFieldVal for (1 << (leftShift + 4))
 *  - error: pipeline lmo overflow error, report by
 *   - Max96712UpdatePipelineError(
 *    - errorStatus, pipelineErrorCount, regFieldVal, pipelnNo,
 *    - CDI_MAX96712_PIPELINE_CMD_OVERFLOW_ERR, false)
 *    .
 *   .
 *  .
 *
 * Note: The reporting will update the errorStatus & globalErrorCount for the caller
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[in] pipelineNum  pipeline number
 * @param[out] pipelineErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetPipelineErrorStatusOverflow(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t pipelineNum,
    uint8_t *pipelineErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;
    uint8_t lshift = 0U;
    uint8_t pipelnNo = pipelineNum;

    pipelnNo = pipelnNo % MAX96712_NUM_VIDEO_PIPELINES;

    /* overflow */
    errorStatus->pipelineFailureType[pipelnNo][*pipelineErrorCount] =
                                  CDI_MAX96712_PIPELINE_ERROR_INVALID;
    if (pipelnNo <= 3U) {
        /* Overflow error check on video pipeline 0-3 */
        lshift = pipelnNo;
        status = Max96712AccessOneRegField(status, handle, REG_FIELD_OVERFLOW_FIRST4,
                                     0U,
                                     REG_READ_MODE);
    } else { /* pipelineNum >= 4U && pipelineNum <= 7U) */
        /* Overflow error check on video pieline 4-7 */
        lshift = pipelnNo - 4U;
        status = Max96712AccessOneRegField(status, handle, REG_FIELD_OVERFLOW_LAST4,
                                     0U,
                                     REG_READ_MODE);
    }

    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    /* line memory overflow bits are at BACKTOP11 register's bit[3:0] */
    if ((regFieldVal & (uint8_t)((1U << lshift) & 0xFFU)) != 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: pipeline lmo overflow error",
                              pipelnNo);
        Max96712UpdatePipelineError(errorStatus, pipelineErrorCount, regFieldVal, pipelnNo,
                                    CDI_MAX96712_PIPELINE_LMO_OVERFLOW_ERR, false);
    }
    /* cmd overflow bits are at BACKTOP11 register's bit[7:4] */
    if (((uint8_t)(regFieldVal >> 4U) & (uint8_t)((1U << lshift) & 0xFFU)) != 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: pipeline cmd overflow error",
                              pipelnNo);
        Max96712UpdatePipelineError(errorStatus, pipelineErrorCount, regFieldVal, pipelnNo,
                                    CDI_MAX96712_PIPELINE_CMD_OVERFLOW_ERR, false);
    }

    return status;
}

/**
 * @brief Get Pipeline Error Status Vid Unlock
 *
 * - initialize reporting of error to CDI_MAX96712_PIPELINE_ERROR_INVALID, by
 *  - placing value in errorStatus->pipelineFailureType[pipelineNum][*pipelineErrorCount]
 *  .
 * - read REG_FIELD_PATGEN_CLK_SRC_PIPE_0 register into bitReg queue by
 *  - status = Max96712AccessOneRegFieldOffset(
 *   - status, handle, REG_FIELD_PATGEN_CLK_SRC_PIPE_0,
 *   - pipelineNum, 0U, REG_READ_MODE)
 *   .
 *  - regFieldVal = ReadFromRegFieldQ(handle, 0U)
 *  .
 * - if (regFieldVal & 0x1U) == 0U)
 *  - error: video unlock on pipeline, reported by
 *   - Max96712UpdatePipelineError(
 *    - errorStatus, pipelineErrorCount, regFieldVal,
 *    - pipelineNum, CDI_MAX96712_PIPELINE_PGEN_VID_UNLOCK_ERR, false)
 *    .
 *   .
 *  .
 *
 * Note: The reporting will update the errorStatus & globalErrorCount for the caller
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[in] pipelineNum  pipeline number
 * @param[out] pipelineErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetPipelineErrorStatusVidUnlock(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t pipelineNum,
    uint8_t *pipelineErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    /* pipe pattern generator video lock status, register 0x1DC etc's bit 0,
     * defined by 8 contiguous enums. */
    errorStatus->pipelineFailureType[pipelineNum][*pipelineErrorCount] =
                                                  CDI_MAX96712_PIPELINE_ERROR_INVALID;
    status = Max96712AccessOneRegFieldOffset(status, handle, REG_FIELD_PATGEN_CLK_SRC_PIPE_0,
                                        pipelineNum,
                                        0U,
                                        REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    if ((regFieldVal & 0x1U) == 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: video unlock on pipeline",
                              pipelineNum);
        Max96712UpdatePipelineError(errorStatus, pipelineErrorCount, regFieldVal, pipelineNum,
                                    CDI_MAX96712_PIPELINE_PGEN_VID_UNLOCK_ERR, false);
    }

    return status;
}

/**
 * @brief Get Pipeline Error Status Memory Error
 *
 * - initialize reporting of error to CDI_MAX96712_PIPELINE_ERROR_INVALID, by
 *  - placing value in errorStatus->pipelineFailureType[pipelineNum][*pipelineErrorCount]
 *  .
 * - read REG_FIELD_BACKTOP25 register into bitReg queue by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_BACKTOP25, 0U, REG_READ_MODE)
 *   .
 *  - regFieldVal = ReadFromRegFieldQ(handle, 0U)
 *  .
 * - if ((regFieldVal & MAX96712_PIPELINE_SHIFT(pipelineNum)) != 0U)
 *  - error: pipeline line mem err, reported by
 *   - Max96712UpdatePipelineError(errorStatus, pipelineErrorCount,
 *    - regFieldVal, pipelineNum, CDI_MAX96712_PIPELINE_MEM_ERR, false)
 *    .
 *   .
 *  .
 * - Note: The reporting will update the errorStatus & globalErrorCount for the caller
 *
 * - Note: this function is called LAST in a series of status collection calls
 *
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[in] pipelineNum  pipeline number
 * @param[out] pipelineErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetPipelineErrorStatusMemErr(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t pipelineNum,
    uint8_t *pipelineErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;
    if ((handle == NULL) || (errorStatus == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if ((pipelineErrorCount == NULL) || (pipelineNum >= MAX96712_NUM_VIDEO_PIPELINES) ||
       (*pipelineErrorCount >= MAX96712_MAX_PIPELINE_ERROR_NUM)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* mem_err */
    errorStatus->pipelineFailureType[pipelineNum][*pipelineErrorCount] = \
                                     CDI_MAX96712_PIPELINE_ERROR_INVALID;

    status = Max96712AccessOneRegField(status, handle, REG_FIELD_BACKTOP25,
                                0U,
                                REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    if ((regFieldVal & MAX96712_PIPELINE_SHIFT(pipelineNum)) != 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: pipeline line mem err",
                              pipelineNum);
        Max96712UpdatePipelineError(errorStatus, pipelineErrorCount, regFieldVal, pipelineNum, \
                                                        CDI_MAX96712_PIPELINE_MEM_ERR, false);
    }

    return status;
}

/**
 * @brief Get Pipeline Error Status Line CRC Error
 *
 * - initialize reporting of error to CDI_MAX96712_PIPELINE_ERROR_INVALID, by
 *  -  placing value in errorStatus->pipelineFailureType[pipelineNum][*pipelineErrorCount]
 *  .
 * - read REG_FIELD_VIDEO_RX0_PIPE_0 register into bitReg queue by
 *  - status = Max96712AccessOneRegFieldOffset(
 *   - status, handle, REG_FIELD_VIDEO_RX0_PIPE_0, pipelineNum, 0U, REG_READ_MODE)
 *   .
 *  - regFieldVal = ReadFromRegFieldQ(handle, 0U)
 *  .
 * - if ((regFieldVal & 0x80U) != 0U)
 *  - error: pipeline line crc error, reported by
 *   - Max96712UpdatePipelineError(errorStatus, pipelineErrorCount,
 *    - regFieldVal, pipelineNum, CDI_MAX96712_PIPELINE_LCRC_ERR, false)
 *    .
 *   .
 *  .
 * - Note: The reporting will update the errorStatus & globalErrorCount for the caller
 *
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[in] pipelineNum  pipeline number
 * @param[out] pipelineErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetPipelineErrorStatusLcrcErr(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t pipelineNum,
    uint8_t *pipelineErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    /* video line crc error status, register 0x100 etc's bit 7, defined by 8 contiguous enums. */
    errorStatus->pipelineFailureType[pipelineNum][*pipelineErrorCount] =
                                                 CDI_MAX96712_PIPELINE_ERROR_INVALID;
    status = Max96712AccessOneRegFieldOffset(status, handle,
                                             REG_FIELD_VIDEO_RX0_PIPE_0, pipelineNum,
                                             0U, REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    if ((regFieldVal & 0x80U) != 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: video line crc error",
                              pipelineNum);
        Max96712UpdatePipelineError(errorStatus, pipelineErrorCount, regFieldVal, pipelineNum,
                                    CDI_MAX96712_PIPELINE_LCRC_ERR, false);
    }

    return status;
}

/**
 * @brief Get Pipeline Error Status Video Sequence Error
 *
 * - initialize reporting of error to CDI_MAX96712_PIPELINE_ERROR_INVALID, by
 *  -  placing value in errorStatus->pipelineFailureType[pipelineNum][*pipelineErrorCount]
 *  .
 * - read REG_FIELD_VIDEO_RX8_PIPE_0 register into bitReg queue by
 *  - status = Max96712AccessOneRegFieldOffset(
 *   - status, handle, REG_FIELD_VIDEO_RX8_PIPE_0, pipelineNum, 0U, REG_READ_MODE)
 *   .
 *  - regFieldVal = ReadFromRegFieldQ(handle, 0U)
 *  .
 * - if ((regFieldVal & 0x10U) != 0U)
 *  - error: pipeline video sequence error, reported by
 *   - Max96712UpdatePipelineError(errorStatus, pipelineErrorCount,
 *    - regFieldVal, pipelineNum, CDI_MAX96712_PIPELINE_VID_SEQ_ERR, true)
 *    .
 *   .
 *  .
 * - Note: The reporting will update the errorStatus & globalErrorCount for the caller
 *
 * - Note: this function is called LAST in a series of status collection calls
 * - That is why the last reported error in this function is marked as "last"
 *
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[in] pipelineNum  pipeline number
 * @param[out] pipelineErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetPipelineErrorStatusVidSeqErr(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t pipelineNum,
    uint8_t *pipelineErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    /* video sequence error status, register 0x108 etc's bit 4, defined by 8 contiguous enums. */
    errorStatus->pipelineFailureType[pipelineNum][*pipelineErrorCount] =
                                                  CDI_MAX96712_PIPELINE_ERROR_INVALID;
    status = Max96712AccessOneRegFieldOffset(status, handle,REG_FIELD_VIDEO_RX8_PIPE_0,
                                             pipelineNum,
                                             0U,
                                             REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    if ((regFieldVal & 0x10U) != 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: pipeline video sequence error",
                              pipelineNum);
        Max96712UpdatePipelineError(errorStatus, pipelineErrorCount, regFieldVal, pipelineNum, \
                                                     CDI_MAX96712_PIPELINE_VID_SEQ_ERR, true);
    }

    return status;
}

/**
 * @brief Get pipeline Video Sequence error information for custom API
 *
 * @param handle DevBlk handle
 * @param customErrInfo pointer to store error information. 8-bit value
 *                      will hold per-pipeline VID_SEQ_ERR bit value.
 * @return NvMediaStatus
 */
NvMediaStatus
MAX96712GetVidSeqError(
    DevBlkCDIDevice const* handle,
    uint8_t *customErrInfo)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);

    if ((NULL ==  drvHandle) || (NULL == customErrInfo)) {
        SIPL_LOG_ERR_STR("MAX96712GetVidSeqErr: Bad parameter.\n");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        *customErrInfo = 0U;

        DevBlkCDII2CReg VIDEO_RX8Regs[] = {
            {0x0108U, 0x0U},
            {0x011AU, 0x0U},
            {0x012CU, 0x0U},
            {0x013EU, 0x0U},
            {0x0150U, 0x0U},
            {0x0168U, 0x0U},
            {0x017AU, 0x0U},
            {0x018CU, 0x0U}
        };
        DevBlkCDII2CRegListWritable const readVIDEO_RX8RegData = {
            .regs = VIDEO_RX8Regs,
            .numRegs = (uint32_t)(sizeof(VIDEO_RX8Regs) /
                                  sizeof(VIDEO_RX8Regs[0])),
        };

        status = MAX96712ReadArrayVerify(handle, &readVIDEO_RX8RegData);
        if (status == NVMEDIA_STATUS_OK) {
            for (uint32_t pipelineNum = 0U; pipelineNum < readVIDEO_RX8RegData.numRegs; \
                 pipelineNum++) {
                    uint8_t bit = ((VIDEO_RX8Regs[pipelineNum].data & 0x10U) > 0U) ? \
                                   1U : 0U;
                    *customErrInfo |= (uint8_t)(bit << pipelineNum);
            }
        }
    }

    return status;
}

/**
 * @brief Get all pipeline Error Status
 *
 * - collects all error status by calling:
 * - for each pipeline
 *  - status = MAX96712GetPipelineErrorStatusOverflow(
 *   - handle, errorStatus, pipelineNum, &pipelineErrorCount)
 *   .
 *  - status = MAX96712GetPipelineErrorStatusVidUnlock(
 *   - handle,  errorStatus,  pipelineNum, &pipelineErrorCount)
 *   .
 *  - status = MAX96712GetPipelineErrorStatusMemErr(
 *   - handle, errorStatus, pipelineNum, &pipelineErrorCount)
 *   .
 *  - status = MAX96712GetPipelineErrorStatusVidSeqErr(
 *   - handle, errorStatus,  pipelineNum, &pipelineErrorCount)
 *   .
 *  .
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetPipelineErrorStatus(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t pipelineErrorCount = 0U;
    uint8_t pipelineNum = 0U;

    for (pipelineNum = 0U; pipelineNum < MAX96712_NUM_VIDEO_PIPELINES; pipelineNum++) {
        pipelineErrorCount = 0U;

        if (status == NVMEDIA_STATUS_OK) {
            status = MAX96712GetPipelineErrorStatusOverflow(handle,
                                                            errorStatus,
                                                            pipelineNum,
                                                            &pipelineErrorCount);
        }

        if (status == NVMEDIA_STATUS_OK) {
            status = MAX96712GetPipelineErrorStatusVidUnlock(handle,
                                                             errorStatus,
                                                             pipelineNum,
                                                             &pipelineErrorCount);
        }

        if (status == NVMEDIA_STATUS_OK) {
            status = MAX96712GetPipelineErrorStatusMemErr(handle,
                                                          errorStatus,
                                                          pipelineNum,
                                                          &pipelineErrorCount);
        }

        if (status == NVMEDIA_STATUS_OK) {
            status = MAX96712GetPipelineErrorStatusLcrcErr(handle,
                                                           errorStatus,
                                                           pipelineNum,
                                                           &pipelineErrorCount);
        }

        if (status == NVMEDIA_STATUS_OK) {
            status = MAX96712GetPipelineErrorStatusVidSeqErr(handle,
                                                             errorStatus,
                                                             pipelineNum,
                                                             &pipelineErrorCount);
        }
    }

    return status;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
static NvMediaStatus
MAX96712GetLinkErrorStatusGMSL1Lock(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t linkNum,
    uint8_t *linkErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    status = Max96712AccessOneRegFieldOffset(status, handle, REG_FIELD_GMSL1_LOCK_A,
                                             linkNum,
                                             0U,
                                             REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    if (regFieldVal != 1U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: Link GMSL1 link unlocked",
                              linkNum);
        Max96712UpdateLinkError(errorStatus, linkErrorCount, regFieldVal,
                                linkNum, CDI_MAX96712_GMSL1_LINK_UNLOCK_ERR, false);
    }

    return status;
}

static NvMediaStatus
MAX96712GetLinkErrorStatusGMSL1DecErr(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t linkNum,
    uint8_t *linkErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    status = Max96712AccessOneRegFieldOffset(status, handle, REG_FIELD_GMSL1_DET_ERR_A,
                                             linkNum,
                                             0U,
                                             REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    if ((regFieldVal != 0U) &&
        (errorStatus->linkFailureType[linkNum][*linkErrorCount] ==
                                               CDI_MAX96712_GMSL_LINK_ERROR_INVALID)) {
        SIPL_LOG_ERR_STR_2UINT("MAX96712: Link: GMSL1 decoding error",
                               linkNum, regFieldVal);
        Max96712UpdateLinkError(errorStatus, linkErrorCount, regFieldVal,
                                linkNum, CDI_MAX96712_GMSL1_LINK_DET_ERR, false);
    }

    return status;
}

/* MAX96712 Get Link Error Status GMSL1 Pkt CC */
static NvMediaStatus
MAX96712GetLinkErrorStatusGMSL1PktCC(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t linkNum,
    uint8_t *linkErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    status = Max96712AccessOneRegFieldOffset(status, handle, REG_FIELD_CC_CRC_ERRCNT_A,
                                             linkNum,
                                             0U,
                                             REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    if ((regFieldVal != 0U) &&
        (errorStatus->linkFailureType[linkNum][*linkErrorCount] ==
                                      CDI_MAX96712_GMSL_LINK_ERROR_INVALID)) {
        SIPL_LOG_ERR_STR_2UINT("MAX96712: Link GMSL1 PKTCC CRC",
                               linkNum, regFieldVal);
        Max96712UpdateLinkError(errorStatus, linkErrorCount, regFieldVal,
                                linkNum, CDI_MAX96712_GMSL1_LINK_PKTCC_CRC_ERR, true);
    }

    return status;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/**
 * @brief Get Link Error Status GMSL2 Lock
 *
 * - read REG_FIELD_GMSL2_LOCK_A register into bitReg queue
 * - if (regFieldVal == 0U)
 *  - error: pipeline video sequence error, reported by
 *   - Max96712UpdateLinkError(errorStatus, linkErrorCount,
 *    - regFieldVal, linkNum, CDI_MAX96712_GMSL2_LINK_UNLOCK_ERR, false)
 *    .
 *   .
 * - Note: The reporting will update the errorStatus & globalErrorCount for the caller
 *
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[in] linkNum  llink number
 * @param[out] linkErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetLinkErrorStatusGMSL2Lock(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t linkNum,
    uint8_t *linkErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    /* link lock err */
    status = Max96712AccessOneRegFieldOffset(status, handle, REG_FIELD_GMSL2_LOCK_A,
                                             linkNum,
                                             0U,
                                             REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    if (regFieldVal == 0U) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: Link GMSL2 link unlocked",
                              linkNum);
        Max96712UpdateLinkError(errorStatus, linkErrorCount, regFieldVal,
                                linkNum, CDI_MAX96712_GMSL2_LINK_UNLOCK_ERR, false);
    }

    return status;
}

/**
 * @brief Get Link GMSL2 Decoding Error
 *
 * - read REG_FIELD_GMSL2_DEC_ERR_A register into bitReg queue
 * - if  ((regFieldVal != 0U) and previous error was CDI_MAX96712_GMSL_LINK_ERROR_INVALID
 *  - error: Link GMSL2 decoding error, reported by
 *   - Max96712UpdateLinkError(errorStatus, linkErrorCount,
 *    - regFieldVal, linkNum, CDI_MAX96712_GMSL2_LINK_DEC_ERR, false)
 *   .
 *  .
 * - Note: The reporting will update the errorStatus & globalErrorCount for the caller
 *
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[in] linkNum  llink number
 * @param[out] linkErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetLinkErrorStatusGMSL2DecErr(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t linkNum,
    uint8_t *linkErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    /* dec err */
    status = Max96712AccessOneRegFieldOffset(status, handle, REG_FIELD_GMSL2_DEC_ERR_A,
                                             linkNum,
                                             0U,
                                             REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    if ((regFieldVal != 0U) &&
        (errorStatus->linkFailureType[linkNum][*linkErrorCount] ==
                                               CDI_MAX96712_GMSL_LINK_ERROR_INVALID)) {
        SIPL_LOG_ERR_STR_2UINT("MAX96712: Link GMSL2 decoding error",
                               linkNum, regFieldVal);
        Max96712UpdateLinkError(errorStatus, linkErrorCount, regFieldVal,
                                linkNum, CDI_MAX96712_GMSL2_LINK_DEC_ERR, false);
    }

    return status;
}

/**
 * @brief Get Link GMSL2 Idle Error
 *
 * - read REG_FIELD_GMSL2_IDLE_ERR_A register into bitReg queue
 * - if  ((regFieldVal != 0U) and previous error was CDI_MAX96712_GMSL_LINK_ERROR_INVALID
 *   - error: Link GMSL2 idle error, reported by:
 *   .
 *  - Max96712UpdateLinkError(errorStatus, linkErrorCount,
 *   - regFieldVal, linkNum, CDI_MAX96712_GMSL2_LINK_IDLE_ERR, false)
 *   .
 *  .
 * - Note: The reporting will update the errorStatus & globalErrorCount for the caller
 *
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[in] linkNum  llink number
 * @param[out] linkErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetLinkErrorStatusGMSL2IdleErr(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t linkNum,
    uint8_t *linkErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    /* idle err */
    status = Max96712AccessOneRegFieldOffset(status, handle, REG_FIELD_GMSL2_IDLE_ERR_A,
                                             linkNum,
                                             0U,
                                             REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    if ((regFieldVal != 0U) &&
        (errorStatus->linkFailureType[linkNum][*linkErrorCount] ==
                                               CDI_MAX96712_GMSL_LINK_ERROR_INVALID)) {
        SIPL_LOG_ERR_STR_2UINT("MAX96712: Link GMSL2 idle error",
                               linkNum, regFieldVal);
        Max96712UpdateLinkError(errorStatus, linkErrorCount, regFieldVal,
                                linkNum, CDI_MAX96712_GMSL2_LINK_IDLE_ERR, false);
    }

    return status;
}

/**
 * @brief Get Link GMSL2 EOM Error
 *
 * - read REG_FIELD_INTR5 register into bitReg queue
 * - if  (((regFieldVal & (0x1U << (linkNum + 4U))) != 0U) and
 *  - previous error was CDI_MAX96712_GMSL_LINK_ERROR_INVALID
 *  - error: Link eye open monitor error, reported by
 *   - Max96712UpdateLinkError(errorStatus, linkErrorCount,
 *    - regFieldVal, linkNum, CDI_MAX96712_GMSL2_LINK_EOM_ERR, false)
 *    .
 *   .
 * - Note: The reporting will update the errorStatus & globalErrorCount for the caller
 *
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[in] linkNum  llink number
 * @param[out] linkErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetLinkErrorStatusGMSL2EOMErr(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t linkNum,
    uint8_t *linkErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    /* EOM error (intr5, bit[7:4]) */
    status = Max96712AccessOneRegField(status, handle, REG_FIELD_INTR5,
                                    0U,
                                    REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);

    linkNum = linkNum % MAX96712_MAX_NUM_LINK;
    if (((regFieldVal & (0x1U << (linkNum + 4U))) != 0U) &&
        (errorStatus->linkFailureType[linkNum][*linkErrorCount] ==
                                               CDI_MAX96712_GMSL_LINK_ERROR_INVALID)) {
        SIPL_LOG_ERR_STR_2UINT("MAX96712: Link eye open monitor error",
                                linkNum, regFieldVal);
        Max96712UpdateLinkError(errorStatus, linkErrorCount, regFieldVal,
                                linkNum, CDI_MAX96712_GMSL2_LINK_EOM_ERR, false);
    }

    return status;
}

/**
 * @brief Get Link GMSL2 ARQ Error
 *
 * - read REG_FIELD_INTR11 register into bitReg queue
 * - if  ((regFieldVal & (1U << (linkNum + 4U))) != 0U)
 *  - error: Link combined ARQ transmission error, reported by
 *   - Max96712UpdateLinkError(errorStatus, linkErrorCount,
 *    - regFieldVal, linkNum, CDI_MAX96712_GMSL2_LINK_ARQ_RETRANS_ERR, false)
 *    .
 *   .
 *  .
 * - if  ((regFieldVal & MAX96712_LINK_SHIFT(1U, linkNum)) != 0U)
 *  - error: Link combined ARQ transmission error, reported by
 *   - Max96712UpdateLinkError(errorStatus, linkErrorCount,
 *    - regFieldVal, linkNum, CDI_MAX96712_GMSL2_LINK_MAX_RETRANS_ERR, false)
 *    .
 *   .
 *  .
 * - Note: The reporting will update the errorStatus & globalErrorCount for the caller
 *
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[in] linkNum  llink number
 * @param[out] linkErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetLinkErrorStatusGMSL2ARQErr(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t linkNum,
    uint8_t *linkErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    /* ARQ errors (intr11) */
    status = Max96712AccessOneRegField(status, handle, REG_FIELD_INTR11,
                                 0U,
                                 REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);
    if ((regFieldVal != 0U) &&
        (errorStatus->linkFailureType[linkNum][*linkErrorCount] ==
                                              CDI_MAX96712_GMSL_LINK_ERROR_INVALID)) {
        if ((regFieldVal & (1U << ((linkNum % MAX96712_MAX_NUM_LINK) + 4U))) != 0U) {
            SIPL_LOG_ERR_STR_2UINT("MAX96712: Link combined ARQ transmission error",
                                   linkNum, regFieldVal);
            Max96712UpdateLinkError(errorStatus, linkErrorCount, regFieldVal,
                                    linkNum, CDI_MAX96712_GMSL2_LINK_ARQ_RETRANS_ERR, false);
        }
        if ((regFieldVal & MAX96712_LINK_SHIFT(1U, linkNum)) != 0U) {
            SIPL_LOG_ERR_STR_2UINT("MAX96712: Link combined ARQ max transmission error",
                                   linkNum, regFieldVal);
            Max96712UpdateLinkError(errorStatus, linkErrorCount, regFieldVal,
                                    linkNum, CDI_MAX96712_GMSL2_LINK_MAX_RETRANS_ERR, false);
        }
    }

    return status;
}

/**
 * @brief Get Link GMSL2 Video Pixel Crc Error
 *
 * - read REG_FIELD_VID_PXL_CRC_ERR_INT register into bitReg queue
 * - if ((regFieldVal & 0x0fU) != 0U)
 *  - if ((regFieldVal & (uint8_t)(MAX96712_LINK_SHIFT(0x1U, linkNum) & 0xFFU)) != 0)
 *   - error: link video pixel crc count: error, reported by
 *    - Max96712UpdateLinkError(errorStatus, linkErrorCount,
 *     - regFieldVal, linkNum, CDI_MAX96712_GMSL2_LINK_VIDEO_PXL_CRC_ERR, true)
 *     .
 *    .
 *   .
 *  .
 * - Note: The reporting will update the errorStatus & globalErrorCount for the caller
 *
 * - Note: this function is called LAST in a series of status collection calls
 * - That is why the last reported error in this function is marked as "last"
 *
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[in] linkNum  llink number
 * @param[out] linkErrorCount pointer for storing the error count for the caller
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetLinkErrorStatusGMSL2VidPixCrcErr(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t linkNum,
    uint8_t *linkErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regFieldVal = 0U;

    /* vid_pxl_crc_err_int (R0x45[[3:0]) */
    status = Max96712AccessOneRegField(status, handle, REG_FIELD_VID_PXL_CRC_ERR_INT,
                                       0U,
                                       REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0U);

    if (linkNum >= MAX96712_MAX_NUM_LINK) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if ((regFieldVal & 0x0fU) != 0U) {
         if ((regFieldVal & (uint8_t)(MAX96712_LINK_SHIFT(0x1U, linkNum) & 0xFFU)) != 0U) {
             SIPL_LOG_ERR_STR_2UINT("MAX96712: Link video pixel crc count: error",
                                    linkNum, regFieldVal);
             Max96712UpdateLinkError(errorStatus, linkErrorCount, regFieldVal, linkNum,
                                     CDI_MAX96712_GMSL2_LINK_VIDEO_PXL_CRC_ERR, true);
         }
    }

    return status;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
static NvMediaStatus
MAX96712GetLinkErrorStatusGMSL1(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t linkNum,
    uint8_t *linkErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712GetLinkErrorStatusGMSL1Lock(handle,
                                                     errorStatus,
                                                     linkNum,
                                                     linkErrorCount);
    }

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712GetLinkErrorStatusGMSL1DecErr(handle,
                                                       errorStatus,
                                                       linkNum,
                                                       linkErrorCount);
    }

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712GetLinkErrorStatusGMSL1PktCC(handle,
                                                      errorStatus,
                                                      linkNum,
                                                      linkErrorCount);
    }

    return status;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/**
 * @brief Get all Link Error Status GMSL2 for specific link
 *
 * - collects all error status by calling:
 *  - status = MAX96712GetLinkErrorStatusGMSL2Lock(
 *   - handle, errorStatus, linkNum, linkErrorCount)
 *   .
 *  - status = MAX96712GetLinkErrorStatusGMSL2DecErr(
 *   - handle, errorStatus, linkNum, linkErrorCount)
 *   .
 *  - status = MAX96712GetLinkErrorStatusGMSL2IdleErr(
 *   - handle, errorStatus, linkNum, linkErrorCount)
 *   .
 *  - status = MAX96712GetLinkErrorStatusGMSL2EOMErr(
 *   - handle, errorStatus, linkNum, linkErrorCount)
 *   .
 *  - status = MAX96712GetLinkErrorStatusGMSL2ARQErr(
 *   - handle, errorStatus, linkNum, linkErrorCount)
 *   .
 *  - status = MAX96712GetLinkErrorStatusGMSL2VidPixCrcErr(
 *   - handle, errorStatus, linkNum, linkErrorCount)
 *   .
 *  .
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @param[in] linkNum link number
 * @param[out] linkErrorCount pointer for storing updated error counter value
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetLinkErrorStatusGMSL2(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus,
    uint8_t linkNum,
    uint8_t *linkErrorCount)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = MAX96712GetLinkErrorStatusGMSL2Lock(handle,
                                                 errorStatus,
                                                 linkNum,
                                                 linkErrorCount);

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712GetLinkErrorStatusGMSL2DecErr(handle,
                                                       errorStatus,
                                                       linkNum,
                                                       linkErrorCount);
    }

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712GetLinkErrorStatusGMSL2IdleErr(handle,
                                                        errorStatus,
                                                        linkNum,
                                                        linkErrorCount);
    }

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712GetLinkErrorStatusGMSL2EOMErr(handle,
                                                       errorStatus,
                                                       linkNum,
                                                       linkErrorCount);
    }

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712GetLinkErrorStatusGMSL2ARQErr(handle,
                                                       errorStatus,
                                                       linkNum,
                                                       linkErrorCount);
    }

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712GetLinkErrorStatusGMSL2VidPixCrcErr(handle,
                                                             errorStatus,
                                                             linkNum,
                                                             linkErrorCount);
    }
    return status;
}

/**
 * @brief Read Registers after getting Link Error Status for all links
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - initialize
 *  - DevBlkCDII2CReg MaxRTErr0  = {0x0A7U, 0x0U};
 *  - DevBlkCDII2CReg MaxRTErr1  = {0x0AFU, 0x0U},
 *  - DevBlkCDII2CReg MaxRTErr2  = {0x0B7U, 0x0U},
 *  - DevBlkCDII2CReg MaxRTErr3  = {0x0BFU, 0x0U},
 *  - DevBlkCDII2CReg MaxRTErr4  = {0x507U, 0x0U},
 *  - DevBlkCDII2CReg MaxRTErr5  = {0x517U, 0x0U},
 *  - DevBlkCDII2CReg MaxRTErr6  = {0x527U, 0x0U},
 *  - DevBlkCDII2CReg MaxRTErr7  = {0x537U, 0x0U},
 *  - DevBlkCDII2CReg MaxRTErr8  = {0x567U, 0x0U},
 *  - DevBlkCDII2CReg MaxRTErr9  = {0x577U, 0x0U},
 *  - DevBlkCDII2CReg MaxRTErr10 = {0x587U, 0x0U},
 *  - DevBlkCDII2CReg MaxRTErr11 = {0x597U, 0x0U},
 *  - DevBlkCDII2CReg MaxRTErr12 = {0x5A7U, 0x0U},
 *  - DevBlkCDII2CReg MaxRTErr13 = {0x5B7U, 0x0U},
 *  - DevBlkCDII2CReg MaxRTErr14 = {0x5C7U, 0x0U},
 *  - DevBlkCDII2CReg MaxRTErr15 = {0x5D7U, 0x0U},
 *
 * - initialize
 *  - DevBlkCDII2CReg VidPxlCrc0  = {0x11D0U, 0x0U};
 *  - DevBlkCDII2CReg VidPxlCrc1  = {0x11D1U, 0x0U},
 *  - DevBlkCDII2CReg VidPxlCrc2  = {0x11D2U, 0x0U},
 *  - DevBlkCDII2CReg VidPxlCrc3  = {0x11D0U, 0x0U},
 *  - DevBlkCDII2CReg VidPxlCrc4  = {0x11E1U, 0x0U},
 *  - DevBlkCDII2CReg VidPxlCrc5  = {0x11E2U, 0x0U},
 *  - DevBlkCDII2CReg VidPxlCrc6  = {0x11E3U, 0x0U},
 *  - DevBlkCDII2CReg VidPxlCrc7  = {0x11E4U, 0x0U},
 *  - DevBlkCDII2CReg VidPxlCrc8  = {0x11E5U, 0x0U},
 *  - DevBlkCDII2CReg VidPxlCrc9  = {0x11E6U, 0x0U},
 *  - DevBlkCDII2CReg VidPxlCrc10 = {0x11E7U, 0x0U},
 *  - DevBlkCDII2CReg VidPxlCrc11 = {0x11E8U, 0x0U},
 *  - DevBlkCDII2CReg VidPxlCrc12 = {0x11E9U, 0x0U},
 *  - DevBlkCDII2CReg VidPxlCrc13 = {0x11EAU, 0x0U},
 *  - DevBlkCDII2CReg VidPxlCrc14 = {0x11EBU, 0x0U},
 *  - DevBlkCDII2CReg VidPxlCrc15 = {0x11ECU, 0x0U},
 *  .
 * - if not previous error
 *  - read register into MaxRTErr0
 *  - read register into MaxRTErr1
 *  - read register into MaxRTErr2
 *  - read register into MaxRTErr3
 *  - read register into MaxRTErr4
 *  - read register into MaxRTErr5
 *  - read register into MaxRTErr6
 *  - read register into MaxRTErr7
 *  - read register into MaxRTErr8
 *  - read register into MaxRTErr9
 *  - read register into MaxRTErr10
 *  - read register into MaxRTErr11
 *  - read register into MaxRTErr12
 *  - read register into MaxRTErr13
 *  - read register into MaxRTErr14
 *  - read register into MaxRTErr15
 *
 * - if not previous error
 *  - read register into VidPxlCrc0
 *  - read register into VidPxlCrc1
 *  - read register into VidPxlCrc2
 *  - read register into VidPxlCrc3
 *  - read register into VidPxlCrc4
 *  - read register into VidPxlCrc5
 *  - read register into VidPxlCrc6
 *  - read register into VidPxlCrc7
 *  - read register into VidPxlCrc8
 *  - read register into VidPxlCrc9
 *  - read register into VidPxlCrc10
 *  - read register into VidPxlCrc11
 *  - read register into VidPxlCrc12
 *  - read register into VidPxlCrc13
 *  - read register into VidPxlCrc14
 *  - read register into VidPxlCrc15
 *  .
 *
 * @param[in] instatus previous error code
 * @param[in] handle DEVBLK handle
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
ReadRegistersPostGetLinkErrorStatus(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = instatus;

    /* MAX_RT_ERR (and RT_CNT) in ARQ2 registers. These are clear
       on read registers. */
    DevBlkCDII2CReg MaxRTErrRegs[] = {
        {0x0A7U, 0x0U},
        {0x0AFU, 0x0U},
        {0x0B7U, 0x0U},
        {0x0BFU, 0x0U},
        {0x507U, 0x0U},
        {0x517U, 0x0U},
        {0x527U, 0x0U},
        {0x537U, 0x0U},
        {0x567U, 0x0U},
        {0x577U, 0x0U},
        {0x587U, 0x0U},
        {0x597U, 0x0U},
        {0x5A7U, 0x0U},
        {0x5B7U, 0x0U},
        {0x5C7U, 0x0U},
        {0x5D7U, 0x0U},
    };
    DevBlkCDII2CRegListWritable const readMaxRTErrData = {
        .regs = MaxRTErrRegs,
        .numRegs = (uint32_t)(sizeof(MaxRTErrRegs) /
                              sizeof(MaxRTErrRegs[0])),
    };

    /* VID_PXL_CRC_ERR_A(/B/C/D)X(Y/Z/U). These are clear on read registers*/
    DevBlkCDII2CReg VidPxlCrcRegs[] = {
        {0x11D0U, 0x0U},
        {0x11D1U, 0x0U},
        {0x11D2U, 0x0U},
        {0x11E0U, 0x0U},
        {0x11E1U, 0x0U},
        {0x11E2U, 0x0U},
        {0x11E3U, 0x0U},
        {0x11E4U, 0x0U},
        {0x11E5U, 0x0U},
        {0x11E6U, 0x0U},
        {0x11E7U, 0x0U},
        {0x11E8U, 0x0U},
        {0x11E9U, 0x0U},
        {0x11EAU, 0x0U},
        {0x11EBU, 0x0U},
        {0x11ECU, 0x0U},
    };
    DevBlkCDII2CRegListWritable const readVidPxlCrcErrData = {
        .regs = VidPxlCrcRegs,
        .numRegs = (uint32_t)(sizeof(VidPxlCrcRegs) /
                              sizeof(VidPxlCrcRegs[0])),
    };

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712ReadArrayVerify(handle, &readMaxRTErrData);
        if (status == NVMEDIA_STATUS_OK) {
            for (uint32_t index= 0U; index < readMaxRTErrData.numRegs; index++) {
                MaxRTErrRegs[index].data &= 0xffU;
                if (MaxRTErrRegs[index].data != 0x0U) {
                    SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: MaxRTErrRegs data value is non-zero at"
                                              "addr", (uint32_t)MaxRTErrRegs[index].address);
                }
            }
        }
    }
    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712ReadArrayVerify(handle, &readVidPxlCrcErrData);
        if (status == NVMEDIA_STATUS_OK) {
            for (uint32_t index= 0U; index < readVidPxlCrcErrData.numRegs; index++) {
                VidPxlCrcRegs[index].data &= 0xffU;
                if (VidPxlCrcRegs[index].data != 0x0U) {
                    SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: VidPxlCrcRegs data value is non-zero at"
                                              "addr", (uint32_t)VidPxlCrcRegs[index].address);
                }
            }
        }
    }

    return status;
}

/**
 * @brief Get Link Error Status for all links
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - for all links
 *  - set errorStatus->linkFailureType[linkNum][linkErrorCount] =
 *                                              CDI_MAX96712_GMSL_LINK_ERROR_INVALID
 *  - if (drvHandle->ctx.gmslMode[linkNum] != CDI_MAX96712_GMSL_MODE_UNUSED)
 *   - if (IsGMSL2Mode(drvHandle->ctx.gmslMode[linkNum]))
 *    - report all errors reported by
 *     - MAX96712GetLinkErrorStatusGMSL2(handle, errorStatus, linkNum, &linkErrorCount)
 *     .
 *    .
 *   - else
 *    - error NVMEDIA_STATUS_ERROR
 *    .
 *   .
 *  .
 * @param[in] handle DEVBLK handle
 * @param[out] errorStatus pointer to the error structure for information to be updated
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712GetLinkErrorStatus(
    DevBlkCDIDevice const* handle,
    ErrorStatusMAX96712 *errorStatus)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    uint8_t linkNum = 0U;
    uint8_t linkErrorCount = 0U;

    for (linkNum = 0U; linkNum < MAX96712_MAX_NUM_LINK; linkNum++) {
        linkErrorCount = 0U;
        errorStatus->linkFailureType[linkNum][linkErrorCount] =
                                              CDI_MAX96712_GMSL_LINK_ERROR_INVALID;
        if (drvHandle->ctx.gmslMode[linkNum] != CDI_MAX96712_GMSL_MODE_UNUSED) {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
            /* GMSL1/GMSL2 link based errors to be reported */
            if (drvHandle->ctx.gmslMode[linkNum] == CDI_MAX96712_GMSL1_MODE) {
                status = MAX96712GetLinkErrorStatusGMSL1(handle, errorStatus,
                                                         linkNum, &linkErrorCount);
            } else
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
            if (IsGMSL2Mode(drvHandle->ctx.gmslMode[linkNum])) {
                status = MAX96712GetLinkErrorStatusGMSL2(handle, errorStatus,
                                                         linkNum, &linkErrorCount);
            } else {
                status = NVMEDIA_STATUS_ERROR;
                break;
            }
        }
    }

    status = ReadRegistersPostGetLinkErrorStatus(status, handle);

    return status;
}

NvMediaStatus
MAX96712GetErrorStatus(
    DevBlkCDIDevice const* handle,
    uint32_t parameterSize,
    void *parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    /* coverity[misra_c_2012_rule_11_5_violation] : intentional TID-1417 */
    ErrorStatusMAX96712 *errorStatus = (ErrorStatusMAX96712 *) parameter;

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr for drvHandle");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (status == NVMEDIA_STATUS_OK) {
        if (NULL == parameter) {
            SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr for parameter");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
        }
    }

    if (status == NVMEDIA_STATUS_OK) {
        if (parameterSize != sizeof(ErrorStatusMAX96712)) {
            SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Incorrect param size");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
        } else {
            (void)memset(errorStatus, (int32_t)0U, sizeof(ErrorStatusMAX96712));

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
            /* MAX96712_REG_GMSL1_LINK_A read back as 0 without
             * this delay when any link is powered down */
            /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
            (void)nvsleep(5000);
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
        }
    }

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712GetPipelineErrorStatus(handle, errorStatus);
    }

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712GetLinkErrorStatus(handle, errorStatus);
    }

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712GetGlobalErrorStatus(handle, errorStatus);
    }

    return status;
}

NvMediaStatus
MAX96712GetSerializerErrorStatus(DevBlkCDIDevice const* handle,
                                 bool * isSerError)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = Max96712AccessOneRegField(status, handle, REG_FIELD_REM_ERR_FLAG,
                                     0U,
                                     REG_READ_MODE);
        if (ReadFromRegFieldQ(handle, 0U) == 1U) {
            *isSerError = true;
        }
    }

    return status;
}

NvMediaStatus
MAX96712ReadParameters(
    DevBlkCDIDevice const* handle,
    ReadParametersCmdMAX96712 parameterType,
    size_t parameterSize,
    void *parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    /* coverity[misra_c_2012_rule_11_5_violation] : intentional TID-1417 */
    ReadParametersParamMAX96712 *param = (ReadParametersParamMAX96712 *)parameter;

    if ((NULL == drvHandle) || (NULL == parameter)) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {

        /* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
        switch (parameterType) {
            case CDI_READ_PARAM_CMD_MAX96712_REV_ID:
                if (parameterSize == sizeof(param->revision)) {
                    param->revision = drvHandle->ctx.revision;
                    status = NVMEDIA_STATUS_OK;
                } else {
                    status = NVMEDIA_STATUS_BAD_PARAMETER;
                }
                break;
            case CDI_READ_PARAM_CMD_MAX96712_CONTROL_CHANNEL_CRC_ERROR:
                status = ReadCtrlChnlCRCErr(handle,
                                            parameterSize,
                                            sizeof(param->ErrorStatus),
                                            param->ErrorStatus.link,
                                            &param->ErrorStatus.errVal);
                break;
            case CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS:
                if (parameterSize == sizeof(param->link)) {
                    status = MAX96712GetEnabledLinks(handle,
                                                        (uint8_t *)&param->link);
                } else {
                    status = NVMEDIA_STATUS_BAD_PARAMETER;
                }
                break;
            case CDI_READ_PARAM_CMD_MAX96712_ERRB:
                status = ClearErrb(handle,
                                    parameterSize,
                                    sizeof(param->ErrorStatus),
                                    &param->ErrorStatus.link,
                                    &param->ErrorStatus.errVal);
                break;
            default:
                SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Invalid command");
                status = NVMEDIA_STATUS_BAD_PARAMETER;
                break;
        }
    }

    return status;
}

NvMediaStatus
MAX96712ReadRegister(
    DevBlkCDIDevice const* handle,
    uint16_t registerNum,
    uint8_t *dataByte)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);

    if ((NULL == drvHandle) || (NULL == dataByte)) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {

        status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                            registerNum,
                                            dataByte);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_AND_UINT("MAX96712: Register I2C read failed with status",
                                            registerNum, (uint32_t)status);
        }
    }

    return status;
}

/**
 * @brief Applies GMSL2 Link Adaptation
 *
 * - for each link
 *  -    if MAX96712_IS_GMSL_LINK_SET for link in link mask
 *   -       if !IsGMSL2Mode of context' gmslMode[link_index]
 *    -          continue
 *    .
 *   -       read/mod/write 0x00 to REG_FIELD_ENABLE_OSN_0 register
 *   -       read/mod/write 0x01 to REG_FIELD_OSN_COEFF_MANUAL_SEED_0 register
 *   -       nvsleep(10000)
 *   -       repeat 100 times
 *    -          read REG_FIELD_OSN_COEFFICIENT_0 regisster
 *    -          if value read == 0x31
 *    -              break
 *    .
 *   .
 *  .
 * - Note: no habdling of timeout ?????
 *
 * @param[in] handle DEVBLK handle
 * @param[in] link link mask
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
GMSL2LinkAdaptation(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    GMSLModeMAX96712 gmslMode = CDI_MAX96712_GMSL_MODE_INVALID;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    RevisionMAX96712 rev = drvHandle->ctx.revision;
    uint8_t regVal = 0U, i = 0U, loop = 0U;

    for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            gmslMode = drvHandle->ctx.gmslMode[i];

            if (!IsGMSL2Mode(gmslMode)) {
                PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: Link %d: adaptation is required only "
                                            "in GMSL2 mode\n", i);
                continue;
            }

            /* Disable OSN */
            status = Max96712AccessOneRegFieldOffset(status, handle, REG_FIELD_ENABLE_OSN_0,
                                                     i,
                                                     0U,
                                                     REG_READ_MOD_WRITE_MODE);

            /* Reseed and set to default value 31 */
            status = Max96712AccessOneRegFieldOffset(status, handle,
                                                     REG_FIELD_OSN_COEFF_MANUAL_SEED_0,
                                                     i,
                                                     1U,
                                                     REG_READ_MOD_WRITE_MODE);

            /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
            (void)nvsleep(10000);

            for (loop = 0U; loop < 100U; loop++) {
                /* Read back OSN value */
                status = Max96712AccessOneRegFieldOffset(status, handle,
                                                         REG_FIELD_OSN_COEFFICIENT_0,
                                                         i, 0U, REG_READ_MODE);
                regVal = ReadFromRegFieldQ(handle, 0U);
                if (regVal == 31U) {
                    break;
                }
                /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
                (void)nvsleep(1000);
            }
            (void)rev;
            PrintLogMsg(LOG_LEVEL_NONE,"MAX96712 Rev %d manual adaptation on the link %d (%d)\n",
                        rev,
                        i,
                        regVal);
        }
    }

    return status;
}

/**
 * @brief Enable Memory ECCs
 *
 * - initialize DevBlkCDII2CReg memECCReg = {0x0044U, 0x0FU}
 * - if not previous error
 *  -    if enable2bitReport is true
 *   -        memECCReg.data |= (uint8_t)(1U << 7U)
 *   .
 *  -   if enable1bitReport is true
 *   -       memECCReg.data |= (uint8_t)(1U << 6U)
 *   .
 *  - write value to register
 *
 * @param[in] instatus previous error code
 * @param[in] handle DEVBLK handle
 * @param[in] enable2bitReport - true if needs to enable bit 7
 * @param[in] enable1bitReport - true if needs to enable bit 6
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
EnableMemoryECC(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle,
    bool enable2bitReport,
    bool enable1bitReport)
{
    NvMediaStatus status = instatus;
    DevBlkCDII2CReg memECCReg = {0x0044U, 0x0FU};

    if (status == NVMEDIA_STATUS_OK) {
        if (enable2bitReport) {
            memECCReg.data |= (uint8_t)(1U << 7U);
        }
        if (enable1bitReport) {
            memECCReg.data |= (uint8_t)(1U << 6U);
        }
        status = MAX96712WriteUint8Verify(handle,
                                          memECCReg.address,
                                          (uint8_t)memECCReg.data);
    }
    return status;
}

/**
 * @brief Sets up CRUSSC Modes
 *
 * - initialize
 *  - DevBlkCDII2CReg CRUSSCMode0 = {0x1445U, 0x0U};
 *  - DevBlkCDII2CReg CRUSSCMode1 = {0x1545U, 0x0U};
 *  - DevBlkCDII2CReg CRUSSCMode2 = {0x1645U, 0x0U};
 *  - DevBlkCDII2CReg CRUSSCMode3 = {0x1745U, 0x0U};
 *  .
 * - if not previous error
 *  - write CRUSSCMode0 to register
 *  - write CRUSSCMode1 to register
 *  - write CRUSSCMode2 to register
 *  - write CRUSSCMode3 to register
 *
 * @param[in] instatus previous error code
 * @param[in] handle DEVBLK handle
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
SetCRUSSCModes(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = instatus;

    DevBlkCDII2CReg const CRUSSCModeRegs[] = {
        {0x1445U, 0x0U},
        {0x1545U, 0x0U},
        {0x1645U, 0x0U},
        {0x1745U, 0x0U},
    };
    DevBlkCDII2CRegList CRUSSCModeArr = {
        .regs = CRUSSCModeRegs,
        .numRegs = (uint32_t)(sizeof(CRUSSCModeRegs) /
                              sizeof(CRUSSCModeRegs[0])),
    };

    if (status == NVMEDIA_STATUS_OK) {
        status = MAX96712WriteArrayVerify(handle, &CRUSSCModeArr);
    }

    return status;
}

/**
 * @brief Enable Individual Reset
 *
 * - initialize DevBlkCDII2CReg individualReset = {0x06DFU, 0x7FU};
 * - if no previous error
 *  - write individualReset to register
 *
 * @param[in] instatus previous error code
 * @param[in] handle DEVBLK handle
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
EnableIndividualReset(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = instatus;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    DevBlkCDII2CReg individualReset = {0x06DFU, 0x7FU};

    if (status == NVMEDIA_STATUS_OK) {
        if (drvHandle->ctx.revision >= CDI_MAX96712_REV_3) {
            status = MAX96712WriteUint8Verify(handle,
                                              individualReset.address,
                                              (uint8_t)individualReset.data);
        }
    }
    return status;
}

/**
 * @brief Read Register Field Csi Pll Lock Before Force Pll Lock
 *
 * - initialize CSIPllLockReg = {0x0400U, 0x00U};
 * - set mipiOutMode to context' mipiOutMode
 * - if no previous error
 *  - repeat 20 times
 *   - read CSIPllLockReg to register
 *   - data value is returned to caller
 *   - if  read op failed ||
 *    - (((mipiOutMode == CDI_MAX96712_MIPI_OUT_2x4) &&
 *     - ((*data & 0xF0U) == 0x60U)) ||
 *     .
 *    - ((mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2) &&
 *     - ((*data & 0xF0U) == 0xF0U))))
 *     .
 *    - break loop
 *    .
 *   - nvsleep(10000)
 * @param[in] instatus previous error code
 * @param[in] drvHandle driver handle
 * @param[out] data device driver handle
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
Max96712ReadRegFieldCsiPllLockBeforeForcePllLock(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle,
    uint8_t *data)
{
    NvMediaStatus status = instatus;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);

    MipiOutModeMAX96712 mipiOutMode = drvHandle->ctx.mipiOutMode;
    DevBlkCDII2CReg CSIPllLockReg = {0x0400U, 0x00U};
    uint16_t i = 0U;

        for (i = 0U; i < 20U; i++) {
            status = MAX96712ReadUint8Verify(handle,
                                             CSIPllLockReg.address,
                                             data);
            if ((status != NVMEDIA_STATUS_OK) ||
                (((mipiOutMode == CDI_MAX96712_MIPI_OUT_2x4) &&
                  ((*data & 0xF0U) == 0x60U)) ||
                  ((mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2) &&
                  ((*data & 0xF0U) == 0xF0U)))) {
                    break;
                }
            /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
            (void)nvsleep(10000);
        }

    return status;
}

/**
 * @brief Check CSI PLL Lock
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - check lock by
 *  - status = Max96712ReadRegFieldCsiPllLockBeforeForcePllLock(
 *   - status, drvHandle, &data)
 *   .
 *  .
 * @param[in] handle Dev Blk handle
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
CheckCSIPLLLock(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t data = 0U;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    NvMediaBool passiveEnabled = drvHandle->ctx.passiveEnabled;
    if (!passiveEnabled)
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
    {
        data = 0U;
        status = Max96712ReadRegFieldCsiPllLockBeforeForcePllLock(status, handle, &data);
    }

    return status;
}

/**
 * @brief GMSL2 PHY Optimization RevB RevC
 *
 * - status = ConfigTxAmpTiming(handle, link)
 * - status = GMSL2LinkAdaptation(handle, link)
 *
 * @param[in] handle Dev Blk handle
 * @param[in] link link mask
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
GMSL2PHYOptimizationRevBRevC(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = ConfigTxAmpTiming(handle, link);
    if (status == NVMEDIA_STATUS_OK) {
        status = GMSL2LinkAdaptation(handle, link);
    }

    return status;
}

/**
 * @brief GMSL2 PHY Optimization RevE
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - Note : Increase CMU regulator output voltage (bit 4)
 * - initialize DevBlkCDII2CReg increaseCMUOutVoltageReg = {0x06C2U, 0x10U}
 * - Note: Set VgaHiGain_Init_6G (bit 1) and VgaHiGain_Init_3G (bit 0)
 * - initialize DevBlkCDII2CReg vgaHiGain_InitReg = {0x14D1U, 0x03U}
 * - write increaseCMUOutVoltageReg to deserializer register by
 *  - status = DevBlkCDII2CPgmrWriteUint8(
 *   - drvHandle->i2cProgrammer, increaseCMUOutVoltageReg.address,
 *   -(uint8_t)increaseCMUOutVoltageReg.data)
 *   .
 *  .
 * - for each link
 *  - if ((MAX96712_IS_GMSL_LINK_SET(link, i)) and
 *   - (IsGMSL2Mode(drvHandle->ctx.gmslMode[i]))
 *    - calculate vgaHiGain_InitReg.address = (uint16_t)(0x14D1U + (link_index * 0x100U))
 *    - write vgaHiGain_InitReg to deserializer register by
 *     - status = DevBlkCDII2CPgmrWriteUint8(
 *      - drvHandle->i2cProgrammer, vgaHiGain_InitReg.address,
 *      - (uint8_t)vgaHiGain_InitReg.data)
 *      .
 *     .
 *    .
 *   .
 *  .
 *
 * @param[in] handle Dev Blk handle
 * @param[in] link link mask
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
GMSL2PHYOptimizationRevE(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    uint8_t i = 0U;
    /* Increase CMU regulator output voltage (bit 4) */
    DevBlkCDII2CReg increaseCMUOutVoltageReg = {0x06C2U, 0x10U};
    /* Set VgaHiGain_Init_6G (bit 1) and VgaHiGain_Init_3G (bit 0) */
    DevBlkCDII2CReg vgaHiGain_InitReg = {0x14D1U, 0x03U};

    status = MAX96712WriteUint8Verify(handle,
                                      increaseCMUOutVoltageReg.address,
                                      (uint8_t)increaseCMUOutVoltageReg.data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX96712: Failed to increase CMU output voltage");
    } else {
        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            if ((MAX96712_IS_GMSL_LINK_SET(link, i)) &&
                    (IsGMSL2Mode(drvHandle->ctx.gmslMode[i]))) {
                vgaHiGain_InitReg.address = (uint16_t)(0x14D1U + (i * 0x100U));
                status = MAX96712WriteUint8Verify(handle,
                                                  vgaHiGain_InitReg.address,
                                                  (uint8_t)vgaHiGain_InitReg.data);
                if (status != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR_UINT("MAX96712: Link: Failed to set VgaHighGain_Init",
                            i);
                    break;
                }
                PrintLogMsg(LOG_LEVEL_NONE,"MAX96712 Link %d: PHY optimization was enabled\n", i);
            }
        }
    }

    return status;
}

/**
 * @brief GMSL2 PHY Optimization
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - extract uint8_t linkMask = drvHandle->ctx.linkMask
 * - if no previous error
 *  - if context revision is CDI_MAX96712_REV_2 or CDI_MAX96712_REV_3
 *   - for each link
 *    - if (MAX96712_IS_GMSL_LINKMASK_SET(linkMask, i))
 *     - set LinkMAX96712 linkVar = (1U << i) & 0xFFU;
 *     - call GMSL2PHYOptimizationRevBRevC(handle, linkVar);
 *      - break, exit error if failed
 *      .
 *     .
 *    .
 *   .
 *  - elif context revision is CDI_MAX96712_REV_5
 *   - for each link
 *    - if (MAX96712_IS_GMSL_LINKMASK_SET for link index in mask) and
 *     - (IsGMSL2Mode context' gmslMode[lin index])
 *     - set  LinkMAX96712 linkVar = (1U << i) & 0xFFU;
 *     - call GMSL2PHYOptimizationRevE(handle, linkVar)
 *      - break, exit error if failed
 *      .
 *     .
 *    .
 *   .
 *  - else
 *   - Error: GMSL2PHYOptimization with a version that need no extra programming
 *   - NO ERROR !!! ???
 *   .
 *  .
 *
 * @param[in] instatus previous error code
 * @param[in] handle dev BLK handle
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
GMSL2PHYOptimization(
    NvMediaStatus instatus,
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = instatus;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    uint8_t i = 0U;
    uint8_t linkMask = drvHandle->ctx.linkMask;

    if (status == NVMEDIA_STATUS_OK) {
        /* If any link is configured in GMSL2 mode, execute the link adaptation */
        switch (drvHandle->ctx.revision) {
            case CDI_MAX96712_REV_2:
            case CDI_MAX96712_REV_3:
                for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
                    if (MAX96712_IS_GMSL_LINKMASK_SET(linkMask, i)) {
                        LinkMAX96712 linkVar = (1U << i) & 0xFFU;
                        status = GMSL2PHYOptimizationRevBRevC(handle, linkVar);
                        if (status != NVMEDIA_STATUS_OK) {
                            break;
                        }
                    }
                }
                break;
            case CDI_MAX96712_REV_5:
                for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
                    if ((MAX96712_IS_GMSL_LINKMASK_SET(linkMask, i)) &&
                            (IsGMSL2Mode(drvHandle->ctx.gmslMode[i]))) {
                        LinkMAX96712 linkVar = (1U << i) & 0xFFU;
                        status = GMSL2PHYOptimizationRevE(handle, linkVar);
                        if (status != NVMEDIA_STATUS_OK) {
                            break;
                        }
                    }
                }
                break;
            default:
                PrintLogMsg(LOG_LEVEL_NONE,"MAX96712: GMSL2PHYOptimization with a version "
                            "that need no extra programming\n");
                break;
        }
    }

    return status;
}

/**
 * @brief  Enable GPIO Rx
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - verify parameterSize == sizeof(gpioIndex)
 *  - return isValidSize = true  if test fails
 *  .
 * - Note: ; pull-up 1M ohm, GPIO source en for GMSL2
 * - initialize DevBlkCDII2CReg setGPIOMode = {0x0300U, 0x1CU}
 * - get mask for link index by
 *  - status = GetMAX96712LinkVal(linkIndex, &linkPort)
 *  .
 * - set setGPIOMode.address += ((uint16_t)gpioIndex * 3U)
 * - Note:    0x30F, 0x31F, 0x32F are not used
 *   - GPIO index offset applied here
 *   .
 *  - set  setGPIOMode.address +=
 *   -  if (setGPIOMode.address & 0xFFU) > 0x2EU)
 *    -    0x03
 *    .
 *   - elif setGPIOMode.address & 0xFFU) > 0x1EU)
 *    - 0x02
 *    .
 *   - elif (setGPIOMode.address & 0xFFU) > 0xEU)
 *      - 0x01
 *      .
 *   - else
 *    - 0x00
 *    .
 *   .
 *  - read register into data by
 *   - status = DevBlkCDII2CPgmrReadUint8(
 *    - drvHandle->i2cProgrammer,
 *    - setGPIOMode.address,
 *    -  &data)
 *    .
 *   .
 *  - Note: check for Link port 0
 *  - if ((linkPort & 0x01U) == 1U)
 *   - Note:  Set GPIO_RX_EN
 *   - data |= 0x4U
 *   - Note: Unset GPIO_TX_EN, GPIO_OUT_DIS
 *   - data &= ~(uint8_t)0x3U
 *   - write data to setGPIOMode register by
 *    - status = DevBlkCDII2CPgmrWriteUint8(
 *     - drvHandle->i2cProgrammer,
 *      - setGPIOMode.address
 *      - data)
 *      .
 *    .
 *   .
 *  - elif ((linkPort & 0x0EU) > 1U)
 *   - Note: Link port 1/2/3
 *   - Note: Link0 Unset GPIO_TX_EN, GPIO_OUT_DIS
 *   - data &= ~(uint8_t)0x7U;
 *   - write data to setGPIOMode register by
 *    - status = DevBlkCDII2CPgmrWriteUint8(
 *     - drvHandle->i2cProgrammer,
 *      - setGPIOMode.address,
 *      - data)
 *      .
 *     .
 *   - Note: Set GPIO_RX_EN
 *   - data |= 0x20U
 *   - Note: set GPIO_RX_ID
 *   - data |= (gpioIndex & 0x1FU);
 *
 *   - ApplyLinkIndexOffsetForNonZeroPort(&setGPIOMode, linkPort)
 *   - Note: write data to setGPIOMode register by
 *     - status = DevBlkCDII2CPgmrWriteUint8(
 *      - drvHandle->i2cProgrammer,
 *       - setGPIOMode.address,
 *       - data)
 *       .
 *      .
 *     .
 *   .
 *  - else
 *   - Error:  link Index is out of bound  NO ERROR RETURNED
 *   .
 *  .
 *
 * @param[in] handle Dev Blk handle
 * @param[in] gpioIndex GPIO index
 * @param[in] linkIndex link index
 * @param[in] parameterSize  size of buffer for parameter
 * @param[out] isValidSize return true if valid parameter size, otherwise set to false
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
EnableGPIORx(
    DevBlkCDIDevice const* handle,
    uint8_t gpioIndex,
    LinkMAX96712 const linkIndex,
    size_t parameterSize,
    bool *isValidSize)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    if (parameterSize != sizeof(gpioIndex)) {
        *isValidSize = false;
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        /* pull-up 1M ohm, GPIO source en for GMSL2 */
        DevBlkCDII2CReg setGPIOMode = {0x0300U, 0x1CU};
        uint8_t data = 0U;

        if (status == NVMEDIA_STATUS_OK) {
            *isValidSize = true;
            setGPIOMode.address += ((uint16_t)gpioIndex * 3U);
            status = MAX96712ReadUint8Verify(handle, setGPIOMode.address, &data);
            if (status == NVMEDIA_STATUS_OK) {
                data |= 0x4U; /* Set GPIO_RX_EN */
                data &= 0xFCU; /* Unset GPIO_TX_EN, GPIO_OUT_DIS */
                status = MAX96712WriteUint8Verify(handle, setGPIOMode.address, data);
            }
        }
    }
    return status;
}

/**
 * @brief Unset errb Rx
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - verify pipelink_sz == pipelink_sz
 * - if not match
 *  - set isValidSize to false, error exit
 *  .
 * - set isValidSize to true
 * - for each link_index
 *  - get linkVal by  WHY ???
 *   - status = GetMAX96712LinkVal(link, &linkVal)
 *   .
 *  - if bit[link loop index] is set in link
 *   - write value  0x1FU  to register address )(0x30U + link_index) by
 *    - status = DevBlkCDII2CPgmrWriteUint8(
 *     - drvHandle->i2cProgrammer,
 *     - (uint16_t)(0x30U + i),
 *     - (uint8_t)(0x1FU))
 *     .
 *    .
 *   .
 *  .
 *
 * @param[in] handle Dev Blk handle
 * @param[in] link link mask
 * @param[in] paramSize  size of buffer for parameter
 * @param[in] pipelink_sz expected parameter size
 * @param[out] isValidSize return true if valid parameter size, otherwise set to false
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712UnsetErrbRx(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link,
    size_t paramSize,
    size_t pipelink_sz,
    bool *isValidSize)
{

    NvMediaStatus status = NVMEDIA_STATUS_OK;
    if (pipelink_sz != paramSize) {
        *isValidSize = false;
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        *isValidSize = true;

        for (uint16_t i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            uint8_t linkVal = 0U;
            status = GetMAX96712LinkVal(link, &linkVal);
            if(((1U << i) & (uint8_t)link) != (uint8_t)0U) {
                if (status == NVMEDIA_STATUS_OK) {
                    status = MAX96712WriteUint8Verify(handle,
                                                      (uint16_t)(0x30U + i),
                                                      (uint8_t)(0x1FU));
                }
                if (status != NVMEDIA_STATUS_OK) {
                    break;
                }
            }
        }
    }

    return status;
}

/**
 * @brief Set errb Rx
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - verify paramSize == expectedParamSize
 *  - if not match, set isValidSize to false, error exit
 * - set isValidSize to true
 * - for each link_index
 *  - get linkVal by  WHY ???
 *   - status = GetMAX96712LinkVal(link, &linkVal)
 *   .
 *  - if bit[link loop index] is set in link
 *   - write value  0x9FU  to register address )(0x30U + link_index) by
 *    - status = DevBlkCDII2CPgmrWriteUint8(
 *     - drvHandle->i2cProgrammer,
 *     - (uint16_t)(0x30U + i),
 *     - (uint8_t)(0x9FU))
 *    .
 *   .
 *  .
 * @param[in] handle Dev Blk handle
 * @param[in] link link mask
 * @param[in] paramSize  size of buffer for parameter
 * @param[in] expectedParamSize expected parameter size
 * @param[out] isValidSize return true if valid parameter size, otherwise set to false
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712SetErrbRx(
    DevBlkCDIDevice const* handle,
    LinkMAX96712 const link,
    size_t paramSize,
    size_t expectedParamSize,
    bool *isValidSize)
{

    NvMediaStatus status = NVMEDIA_STATUS_OK;
    if (expectedParamSize != paramSize) {
        *isValidSize = false;
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        *isValidSize = true;
        DriverHandleMAX96712 * drvHandle = getHandlePrivMAX96712(handle);

        for (uint16_t i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            uint8_t linkVal = 0U;
            status = GetMAX96712LinkVal(link, &linkVal);
            if(((1U << i) & (uint8_t)link) != (uint8_t)0U) {
                if (status == NVMEDIA_STATUS_OK) {
                    status = MAX96712WriteUint8Verify(handle,
                                                      (uint16_t)(0x30U + i),
                                                      (uint8_t)(0x9FU));
                }
                if (status != NVMEDIA_STATUS_OK) {
                    break;
                } else {
                    drvHandle->ctx.errbRxEnabled |= (1U << i);
                }
            }
        }
    }

    return status;
}

/**
 * @brief MAX96712 Disable Link
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - verify (pipelink_sz == paramSize) and (link < MAX96712_MAX_NUM_LINK)
 *  - if not match, set isValidSize to false, error exit
 *  .
 * - set isValidSize to true
 * -  Disable Video Pipeline
 *  - read register 0xF4U from device by
 *   - status = DevBlkCDII2CPgmrReadUint8(
 *    - drvHandle->i2cProgrammer, (uint16_t)(0xF4U), &regVal)
 *    .
 *   .
 *  - set  regVal &= (uint8_t)(~MAX96712_LINK_SHIFT(0x11U, link) & 0xFFU)
 *  - write to device by
 *   - status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
 *    - (uint16_t)(0xF4U), regVal)
 *    .
 *   .
 *  - read register 0x30U from device by
 *   - status = DevBlkCDII2CPgmrReadUint8(
 *    - drvHandle->i2cProgrammer,
 *    - (uint16_t)(0x30U) + (uint16_t)link, &regVal)
 *    .
 *   .
 *  - set regVal &= 0x7FU
 *  - write to device by
 *   - status = DevBlkCDII2CPgmrWriteUint8(
 *    - drvHandle->i2cProgrammer,
 *    - (uint16_t)(0x30U) + (uint16_t)link, regVal)
 *    .
 *   .
 *  .
 * @param[in] handle Dev Blk handle
 * @param[in] link link mask
 * @param[in] paramSize  size of buffer for parameter
 * @param[in] pipelink_sz expected parameter size
 * @param[out] isValidSize return true if valid parameter size, otherwise set to false
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712DisableLink(
    DevBlkCDIDevice const* handle,
    uint8_t link,
    size_t paramSize,
    size_t pipelink_sz,
    bool *isValidSize)
{
    NvMediaStatus status;

    if ((pipelink_sz != paramSize) || (link >= MAX96712_MAX_NUM_LINK)) {
        *isValidSize = false;
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        *isValidSize = true;
        uint8_t regVal = 0U;

        /* Disable Video Pipeline */
        status = MAX96712ReadUint8Verify(handle, (uint16_t)(0xF4U), &regVal);
        if (status == NVMEDIA_STATUS_OK ) {
            regVal &= (uint8_t)(MAX96712_LINK_SHIFT(0x11U, link) ^ 0xFFU);
            status = MAX96712WriteUint8Verify(handle, (uint16_t)(0xF4U), regVal);
            if (status == NVMEDIA_STATUS_OK ) {
                /* Disable ERRB Rx */
                regVal = 0U;
                status = MAX96712ReadUint8Verify(handle,
                                                 (uint16_t)(0x30U) + (uint16_t)link,
                                                 &regVal);
                if (status == NVMEDIA_STATUS_OK ) {
                    regVal &= 0x7FU;
                    status = MAX96712WriteUint8Verify(handle,
                                                     (uint16_t)(0x30U) + (uint16_t)link,
                                                     regVal);
                }
            }
        }
    }

    return status;
}

/**
 * @brief Restore Link
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - verify paramSize ==  pipelink_sz
 *  - if not match, set isValidSize to false, error exit
 *  .
 * - set isValidSize to true
 * - read register address (0xF4U) by
 *  - status = DevBlkCDII2CPgmrReadUint8(
 *   - drvHandle->i2cProgrammer,
 *   - (uint16_t)(0xF4U),
 *   - &regVal)
 *   .
 *  .
 * - turn on the bits (0x11 << link) if set in context pipelineEnabled in read value by
 *  - regVal |= (drvHandle->ctx.pipelineEnabled & MAX96712_LINK_SHIFT(0x11U, link))
 *  .
 * - write register address (0xF4U) with updated value by
 *  - status = DevBlkCDII2CPgmrWriteUint8(
 *   - drvHandle->i2cProgrammer,
 *   - (uint16_t)(0xF4U),
 *   - regVal)
 *   .
 *  .
 * - Note:  Restore ERRB Rx Enable bit
 * - read register address (0x30U + link) by
 *  - status = DevBlkCDII2CPgmrReadUint8(
 *   - drvHandle->i2cProgrammer,
 *   - (uint16_t)(0x30U) + (uint16_t)link,
 *   - &regVal)
 *   .
 *  .
 * - if bits position (1U << link) is set in context errbRxEnabled by
 *  - if ((drvHandle->ctx.errbRxEnabled & (uint8_t)(MAX96712_LINK_SHIFT(1U, link) & 0x0FU)) != 0U)
 *   -  turn on bit 0x80U in return value
 *   .
 *  .
 * - write updated value to register address (0x30U + link) by
 *  - status = DevBlkCDII2CPgmrWriteUint8(
 *   - drvHandle->i2cProgrammer,
 *   - (uint16_t)(0x30U) + (uint16_t)link,
 *   - regVal)
 *   .
 *  .
 *
 * @param[in] handle Dev Blk handle
 * @param[in] link link number
 * @param[in] paramSize  size of buffer for parameter
 * @param[in] pipelink_sz expected parameter size
 * @param[out] isValidSize return true if valid parameter size, otherwise set to false
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712RestoreLink(
    DevBlkCDIDevice const* handle,
    uint8_t link,
    size_t paramSize,
    size_t pipelink_sz,
    bool *isValidSize)
{
    NvMediaStatus status;

    if ((pipelink_sz != paramSize) || (link >= MAX96712_MAX_NUM_LINK)) {
        *isValidSize = false;
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        *isValidSize = true;
        uint8_t regVal = 0U;
        DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);

       /* Restore Video Pipeline Enable bits*/
       status = MAX96712ReadUint8Verify(handle, (uint16_t)(0xF4U), &regVal);
       if (status == NVMEDIA_STATUS_OK ) {
           regVal |= (drvHandle->ctx.pipelineEnabled & MAX96712_LINK_SHIFT(0x11U, link));
           status = MAX96712WriteUint8Verify(handle, (uint16_t)(0xF4U), regVal);
           if (status == NVMEDIA_STATUS_OK ) {
               /* Restore ERRB Rx Enable bit*/
               regVal = 0U;
               status = MAX96712ReadUint8Verify(handle,
                                                (uint16_t)(0x30U) + (uint16_t)link,
                                                &regVal);
               if (status == NVMEDIA_STATUS_OK ) {
                   if ((drvHandle->ctx.errbRxEnabled &
                       (uint8_t)(MAX96712_LINK_SHIFT(1U, link) &
                       0x0FU)) != 0U) {
                        regVal |= 0x80U;
                   }
                   status = MAX96712WriteUint8Verify(handle,
                                                     (uint16_t)(0x30U) + (uint16_t)link,
                                                     regVal);
               }
           }
        }
    }

    return status;
}

/**
 * @brief Set Errb Rx Id
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - verify paramSize == expectedParamSize
 * - if not match
 *  -  set isValidSize to false, error out
 *  .
 * - set isValidSize to true
 * - verify gpioIndex is valid <= MAX96712_MAX_PHYSICAL_MFP_NUM
 * - set uint16_t intr6Address = (uint16_t)0x0029U;
 * - if link == CDI_MAX96712_LINK_0
 *  - set errRxIdAddress = (uint16_t)0x0030U
 *  .
 * - elif ink == CDI_MAX96712_LINK_1
 *  - set errRxIdAddress = (uint16_t)0x0031U
 *  .
 * - elif ink == CDI_MAX96712_LINK_2
 *  - set errRxIdAddress = (uint16_t)0x0032U
 *  .
 * - elif ink == CDI_MAX96712_LINK_3
 *  - set errRxIdAddress = (uint16_t)0x0033U
 *  .
 * - else
 *  - error: invalid link mask - error exit
 *  .
 * - read content of errRxIdAddress register into data by
 *  - status = DevBlkCDII2CPgmrReadUint8(
 *   - drvHandle->i2cProgrammer,
 *   - intr6Address,
 *   - &data)
 *   .
 *  .
 * - Note: Set REM_ERR_OEN
 * - set data |= 2U
 * - write data to errRxIdAddress register by
 *  - status = DevBlkCDII2CPgmrWriteUint8(
 *   - drvHandle->i2cProgrammer,
 *   - intr6Address,
 *   -  data)
 *   .
 *  .
 * - Note: Set ERR_RX_EN and ERR_RX_ID
 * - set data = MAX96712_MAX_PHYSICAL_MFP_NUM
 * - write data to errRxIdAddress register by
 *  - status = DevBlkCDII2CPgmrWriteUint8(
 *   - drvHandle->i2cProgrammer,
 *   - intr6Address,
 *   -  data)
 *   .
 *  .
 * - set drvHandle->ctx.errbRxEnabled |= (uint8_t)link
 *
 * @param[in] handle Dev Blk handle
 * @param[in] gpioIndex  GPIO index
 * @param[in] link  link id
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712SetErrbRxId(
    DevBlkCDIDevice const* handle,
    uint8_t const gpioIndex,
    LinkMAX96712 const link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 * drvHandle = getHandlePrivMAX96712(handle);
    uint16_t intr6Address = (uint16_t)0x0029U;
    uint16_t errRxIdAddress = (uint16_t)0U;
    uint8_t data = 0U;

    if (gpioIndex > (uint8_t)MAX96712_MAX_PHYSICAL_MFP_NUM) {
        SIPL_LOG_ERR_STR("GPIO out of range");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {

        switch (link) {
            case CDI_MAX96712_LINK_0:
                errRxIdAddress = (uint16_t)0x0030U;
                break;
            case CDI_MAX96712_LINK_1:
                errRxIdAddress = (uint16_t)0x0031U;
                break;
            case CDI_MAX96712_LINK_2:
                errRxIdAddress = (uint16_t)0x0032U;
                break;
            case CDI_MAX96712_LINK_3:
                errRxIdAddress = (uint16_t)0x0033U;
                break;
            default:
                errRxIdAddress = (uint16_t)0U;
                status = NVMEDIA_STATUS_BAD_PARAMETER;
                break;
        }

        if (errRxIdAddress != (uint16_t)0U) {
            status = MAX96712ReadUint8Verify(handle, intr6Address, &data);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT("read reg: failed", intr6Address);
            } else {
                data |= 2U; /* Set REM_ERR_OEN */
                status = MAX96712WriteUint8Verify(handle, intr6Address, data);
                if (status != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR_HEX_UINT("write reg: failed",
                                              intr6Address);
                } else {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#ifdef NVMEDIA_QNX
                    /* Set ERR_RX_EN and ERR_RX_ID */
                    data = ((1U << 7U) | gpioIndex);
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#else
                    data = MAX96712_MAX_PHYSICAL_MFP_NUM;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
                    status = MAX96712WriteUint8Verify(handle, errRxIdAddress, data);
                    if (status != NVMEDIA_STATUS_OK) {
                        SIPL_LOG_ERR_STR_HEX_UINT("write reg: failed",
                                                  errRxIdAddress);
                    } else {
                        drvHandle->ctx.errbRxEnabled |= (uint8_t)link;
                    }
                }
            }
        }
    }

    return status;
}

/**
 * @brief  Reset All registers
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - initialize DevBlkCDII2CReg resetAllReg = {0x0013U, 0x40U}
 * - write resetAllReg to deserializer register by
 *  - status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
 *   - resetAllReg.address, (uint8_t)resetAllReg.data)
 *   .
 *  .
 * - nvsleep(100000)
 * - set resetAllReg.data = 0U
 * - write resetAllReg to deserializer register by
 *  - status =  DevBlkCDII2CPgmrWriteUint8(
 *   - drvHandle->i2cProgrammer,
 *   - resetAllReg.address, (uint8_t)resetAllReg.data)
 *   .
 *  .
 * @param[in] handle Dev Blk handle
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
ResetAll(DevBlkCDIDevice const* handle)
{
    DevBlkCDII2CReg resetAllReg = {0x0013U, 0x40U};
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = MAX96712WriteUint8Verify(handle,
                                      resetAllReg.address,
                                      (uint8_t)resetAllReg.data);
    if (status == NVMEDIA_STATUS_OK) {
        /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
        (void)nvsleep(100000);
        resetAllReg.data = 0U;

        status = MAX96712WriteUint8Verify(handle,
                                          resetAllReg.address,
                                          (uint8_t)resetAllReg.data);
    }

    return status;
}

#ifdef NVMEDIA_QNX
/* Execute verification steps as per MAX96712 SM Implementation Guide for SM36 TC_1 */
static NvMediaStatus
MAX96712VerifyGPIOOpenDetection(
    DevBlkCDIDevice const* handle,
    uint16_t const *gpioAList,
    uint16_t const *gpioBList,
    uint8_t const numList)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    for (uint8_t index = 0; index < numList; index++) {
        uint8_t writeGPIOA = 0U, writeGPIOB = 0U, readGPIOAPullUp = 0U, readGPIOAPullDown = 0U;

        /* For storing initial values of GPIO_A and GPIO_B and restoring them once
         * SM36 is verified */
        DevBlkCDII2CReg GPIOs[] = {
            {gpioAList[index], 0x00U},
            {gpioBList[index], 0x00U},
        };
        DevBlkCDII2CRegListWritable ReadGPIORegList = {
            .regs = GPIOs,
            .numRegs = (uint32_t)(sizeof(GPIOs) /
                    sizeof(GPIOs[0]))
        };
        status = MAX96712ReadArrayVerify(handle, &ReadGPIORegList);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIOOpenDetection: Failed to read GPIO_A and GPIO_B");
            break;
        }

        /* GPIO initialization.
         * GPIO_A: bit 7 = 1; bit 0 = 1.
         * GPIO_B: bit 5 = 0. */
        writeGPIOA = 0x81 | GPIOs[0].data;
        status = MAX96712WriteUint8Verify(handle, gpioAList[index], writeGPIOA);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIOOpenDetection: Failed to write GPIO_A");
            break;
        }
        writeGPIOB = GPIOs[1].data & ~(1 << 5);
        status = MAX96712WriteUint8Verify(handle, gpioBList[index], writeGPIOB);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIOOpenDetection: Failed to write GPIO_B");
            break;
        }

        /* Turn on 1Mohm pullup.
         * GPIO_B: bit[7:6] = 2'b01 */
        writeGPIOB = writeGPIOB | (1 << 6);
        writeGPIOB = writeGPIOB & ~(1 << 7);
        status = MAX96712WriteUint8Verify(handle, gpioBList[index], writeGPIOB);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIOOpenDetection: Failed to write GPIO_B");
            break;
        }

        /* Reading GPIO_A */
        status = MAX96712ReadUint8Verify(handle, gpioAList[index], &readGPIOAPullUp);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIOOpenDetection: Failed to read GPIO_B");
            break;
        }

        /* Turn on 1Mohm pulldown.
         * GPIO_B: bit[7:6] = 2'b10 */
        writeGPIOB = writeGPIOB | (1 <<7);
        writeGPIOB = writeGPIOB & ~(1 <<6);
        status = MAX96712WriteUint8Verify(handle, gpioBList[index], writeGPIOB);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIOOpenDetection: Failed to write GPIO_B");
            break;
        }

        status = MAX96712ReadUint8Verify(handle, gpioAList[index], &readGPIOAPullDown);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIOOpenDetection: Failed to read GPIO_A");
            break;
        }

        /* Compare GPIO_IN (bit[3]) of GPIO_A register after pull up and pull down */
        if (((readGPIOAPullUp >> 3) & 1U) != ((readGPIOAPullDown >> 3) & 1U)) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIOOpenDetection: Bit mismatch occurred between"
                             "the pullup and pulldown checks");
            status = NVMEDIA_STATUS_ERROR;
            break;
        }

        DevBlkCDII2CRegList WriteGPIORegList = {
            .regs = GPIOs,
            .numRegs = (uint32_t)(sizeof(GPIOs) /
                    sizeof(GPIOs[0]))
        };
        /* Restore the init values of GPIO_A and GPIO_B */
        status = MAX96712WriteArrayVerify(handle, &WriteGPIORegList);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIOOpenDetection: Failed to write GPIO_A and GPIO_B");
            break;
        }
    }

    if (status == NVMEDIA_STATUS_OK) {
        PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: SM36: MAX96712VerifyGPIOOpenDetection"
                                    " is successfully executed\n");
    }

    return status;
}

/* Execute verification steps as per MAX96712 SM Implementation Guide for SM26 TC_1 */
static NvMediaStatus
MAX96712VerifyGPIOReadBackStatus(
    DevBlkCDIDevice const* handle,
    uint16_t const *gpioAList,
    uint8_t numList)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    for (uint8_t index = 0; index < numList; index++) {
        uint8_t readGPIOA, initVal;
        uint8_t gpio3_b_val = 0U;

        /* Get the initial value of GPIO_A so as to restore it once SM26 is verified */
        status = MAX96712ReadUint8Verify(handle, gpioAList[index], &initVal);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIOReadBackStatus: Failed to read GPIO_A");
            break;
        }

        /* MFP3 pin needs to be configured as OD for P3663 board */
        if (gpioAList[index] == REG_GPIO3_A) {
            /* For saving the original settings of REG_GPIO3_B */
            status = MAX96712ReadUint8Verify(handle, REG_GPIO3_B, &gpio3_b_val);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR("MAX96712VerifyGPIOOpenDetection: Failed to read GPIO_B"
                    "for MFP3");
                break;
            }

            /* Write GPIO_B to set MFP3 in OD configuration.
             * open-drain output, GPIO_TX_ID of 3 (default)
             */
            status = MAX96712WriteUint8Verify(handle, REG_GPIO3_B, 0x03);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR("MAX96712VerifyGPIOReadBackStatus: Failed to write GPIO_B"
                    "for MFP3");
                break;
            }
        }

        /* Drive 4th bit of GPIO_A high, GPIO_OUT = 1 */
        status = MAX96712WriteUint8Verify(handle, gpioAList[index], 0x10);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIOReadBackStatus: Failed to write GPIO_A");
            break;
        }

        /* Read back GPIO_A */
        status = MAX96712ReadUint8Verify(handle, gpioAList[index], &readGPIOA);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIOReadBackStatus: Failed to read GPIO_A");
            break;
        } else {
            /* Verify GPIO_IN is 1 */
            if (!((readGPIOA >> 3) & 1U)) {
                SIPL_LOG_ERR_STR_HEX_UINT("MAX96712VerifyGPIOReadBackStatus: GPIO_IN is"
                                          "erroneously unset, MFP: ", gpioAList[index]);
                status = NVMEDIA_STATUS_ERROR;
                break;
            }
        }

        /* Drive 4th bit of GPIO_A low, GPIO_OUT=0 */
        status = MAX96712WriteUint8Verify(handle, gpioAList[index], 0x00);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIOReadBackStatus: Failed to write GPIO_A");
            break;
        }

        /* Read back GPIO_A */
        status = MAX96712ReadUint8Verify(handle, gpioAList[index], &readGPIOA);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIOReadBackStatus: Failed to read GPIO_A");
            break;
        } else {
            /* Verify GPIO_IN is 0 */
            if (((readGPIOA >> 3) & 1U) != 0U) {
                SIPL_LOG_ERR_STR_HEX_UINT("MAX96712VerifyGPIOReadBackStatus: GPIO_IN is"
                                          "erroneously set,  MFP: ", gpioAList[index]);
                status = NVMEDIA_STATUS_ERROR;
                break;
            }
        }

        /* Restore the REG_GPIO3_B */
        if (gpioAList[index] == REG_GPIO3_A) {
            /* Restore the init values of REG_GPIO3_B */
            status = MAX96712WriteUint8Verify(handle, REG_GPIO3_B, gpio3_b_val);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR("MAX96712VerifyGPIOReadBackStatus: Failed to write GPIO_B"
                    "for MFP3");
                break;
            }
        }

        /* Write initial value back to GPIO_A */
        status = MAX96712WriteUint8Verify(handle, gpioAList[index], initVal);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIOReadBackStatus: Failed to read GPIO_A");
            break;
        }
    }

    if (status == NVMEDIA_STATUS_OK) {
        PrintLogMsg(LOG_LEVEL_INFO, "MAX96712: SM26: MAX96712VerifyGPIOReadBackStatus"
                                    " is successfully executed\n");
    }

    return status;
}

/* Verify the MFP/GPIO pins 1, 3 and 10 used on P3663 and P3710 platforms using
 * MAX96712 SMs (SM26 and SM36) */
static NvMediaStatus
MAX96712VerifyGPIO(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96712VerifyGPIO: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        /* GPIO1, GPIO3, and GPIO10 */
        const uint16_t gpios_a[3] = {REG_GPIO1_A, REG_GPIO3_A, REG_GPIO10_A};
        const uint16_t gpios_b[3] = {REG_GPIO1_B, REG_GPIO3_B, REG_GPIO10_B};

        /* By default MFP1 is LOCK, MFP3 is ERRB, and MFP10 is GPIO
         * Disable ERRB and LOCK output to MFP3 and MFP1 respectively. */
        status = MAX96712WriteUint8Verify(handle, REG5, 0x00);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIO: Failed to disable ERRB and LOCK output to GPIO");
            goto end;
        }

        /* Verify SM26 (GPIO readback status) for GPIO1, GPIO3, and GPIO10 */
        status = MAX96712VerifyGPIOReadBackStatus(handle, gpios_a, 3U);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIO: Failed to verify GPIO Read back status");
            goto end;
        }

        /* Verify SM36 (GPIO open detection) for GPIO1, GPIO3, and GPIO10 */
        status = MAX96712VerifyGPIOOpenDetection(handle, gpios_a, gpios_b, 3U);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIO: Failed to verify GPIO Open Detection");
            goto end;
        }

        /* Re-enable ERRB and LOCK output to MFP3 and MFP1 respectively  */
        status = MAX96712WriteUint8Verify(handle, REG5, 0xC0);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyGPIO: Failed to enable ERRB and LOCK output to GPIO");
            goto end;
        }
    }

end:
    return status;
}
#endif

NvMediaStatus
MAX96712SetDefaults(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    uint8_t i = 0U;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }
    else {
        if (drvHandle->ctx.defaultResetAll == true ) {
            status = ResetAll(handle);
        }

        if (drvHandle->ctx.revision == CDI_MAX96712_REV_3) {
            /* Bug 2446492: Disable 2-bit ECC error reporting as spurious ECC errors are
            * intermittently observed on Rev C of MAX96712
            * Disable reporting 2-bit ECC errors to ERRB
            */
            status = EnableMemoryECC(status, handle, false, false);
        } else if (drvHandle->ctx.revision >= CDI_MAX96712_REV_4) {
            /* Enabling only 2-bit ECC errors to ERRB and disabling 1-bit
               ECC errors. */
            status = EnableMemoryECC(status, handle, true, false);
        } else {
            /* do nothing */
        }

        status = EnableIndividualReset(status, handle);

        LinkMAX96712 linkMask = drvHandle->ctx.linkMask;
        PrintLogMsg(LOG_LEVEL_INFO, "Calling CheckAgainstActiveLink before SetLinkMod, "
                                    "linkMask = %x\n", (uint32_t)linkMask);
        status = CheckAgainstActiveLink(status, handle);
        status = SetLinkMode(status, handle, linkMask);

        if (drvHandle->ctx.revision == CDI_MAX96712_REV_5) {
            status = SetCRUSSCModes(status, handle);
        }

        PrintLogMsg(LOG_LEVEL_INFO, "Calling CheckAgainstActiveLink before "
                                    "GMSL2PHYOptimization, linkMask = %x\n",
                    (uint32_t)linkMask);
        status = CheckAgainstActiveLink(status, handle);
        status = GMSL2PHYOptimization(status, handle);

        /* Default mode is GMSL2, 6Gbps
        * one shot reset is required for GMSL1 mode & GMSL2
        */
        if (status == NVMEDIA_STATUS_OK) {
            PrintLogMsg(LOG_LEVEL_INFO, "Calling CheckAgainstActiveLink before "
                                        "MAX96712OneShotReset, linkMask = %x\n",
                        (uint32_t)linkMask);
            status = CheckAgainstActiveLink(status, handle);
            if (status == NVMEDIA_STATUS_OK) {
                status = MAX96712OneShotReset(handle, linkMask);
            }
        }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
            if ((status == NVMEDIA_STATUS_OK)) {
                for (i = 0; i < MAX96712_MAX_NUM_LINK; i++) {
                    if (MAX96712_IS_GMSL_LINK_SET(drvHandle->ctx.linkMask, i)) {
                        if (drvHandle->ctx.gmslMode[i] == CDI_MAX96712_GMSL1_MODE) {
                            /* HIM mode is not enabled yet so the link lock will not be set
                            * Instead use sleep function */
                            /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
                            (void)nvsleep(100000);
                        }
                    }
                }
            }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

        status = SetI2CPort(status, handle);

#ifdef NVMEDIA_QNX
        if (status == NVMEDIA_STATUS_OK) {
            status = MAX96712VerifyGPIO(handle);
        }
#endif

        status = EnableExtraSMs(status, handle);
        if (status == NVMEDIA_STATUS_OK) {
            status = VerifySMsEnabled(handle);
        }

        /* Disable all Pipelines */
        ClearRegFieldQ(handle);
        for (uint8_t i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_VIDEO_PIPE_EN_0, i, 0U);
            status = Max96712AddRegFieldOffset(status, handle, REG_FIELD_VIDEO_PIPE_EN_4, i, 0U);
        }

        status = Max96712AccessRegField(status, handle, REG_READ_MOD_WRITE_MODE);
    }

    return status;
}

/**
 * @brief Set Device Config Group 1
 *
 * - if switchSetDeviceCfg == CDI_CONFIG_MAX96712_ENABLE_BACKTOP
 *  - status = EnableBackTop(handle, true)
 *  .
 * - elif switchSetDeviceCfg == CDI_CONFIG_MAX96712_DISABLE_BACKTOP
 *  - status = EnableBackTop(handle, false)
 *  .
 * - elif switchSetDeviceCfg == CDI_CONFIG_MAX96712_TRIGGER_DESKEW
 *  - status = TriggerDeskew(handle)
 *  .
 * - elif switchSetDeviceCfg == CDI_CONFIG_MAX96712_CHECK_CSIPLL_LOCK
 *  - status = CheckCSIPLLLock(handle)
 *  .
 * - elif switchSetDeviceCfg == CDI_CONFIG_MAX96712_ENABLE_ERRB
 *  - status = EnableERRB(handle, true)
 *  .
 * - elif switchSetDeviceCfg == CDI_CONFIG_MAX96712_DISABLE_ERRB
 *  - status = EnableERRB(handle, false)
 *  .
 * * - elif switchSetDeviceCfg == CDI_CONFIG_MAX96712_ENABLE_LOCK
 *  - status = EnableLOCK(handle, true)
 *  .
* - elif switchSetDeviceCfg == CDI_CONFIG_MAX96712_DISABLE_LOCK
 *  - status = EnableLOCK(handle, false)
 *  .
 * - else
 *  - Error: Bad parameter: Invalid command
 *
 * @param[in] handle Dev Blk handle
 * @param[in] switchSetDeviceCfg selects configuration to setup
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712SetDeviceConfigGroup1(
    DevBlkCDIDevice const* handle,
    ConfigSetsMAX96712 switchSetDeviceCfg)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    switch (switchSetDeviceCfg) {
        case CDI_CONFIG_MAX96712_ENABLE_CSI_OUT:
            status = EnableCSIOut(handle,
                                  true);
            break;
        case CDI_CONFIG_MAX96712_DISABLE_CSI_OUT:
            status = EnableCSIOut(handle,
                                  false);
            break;
        case CDI_CONFIG_MAX96712_TRIGGER_DESKEW:
            status = TriggerDeskew(handle);
            break;
        case CDI_CONFIG_MAX96712_CHECK_CSIPLL_LOCK:
            status = CheckCSIPLLLock(handle);
            break;
        case CDI_CONFIG_MAX96712_ENABLE_ERRB:
            status = EnableERRB(handle, true);
            break;
        case CDI_CONFIG_MAX96712_DISABLE_ERRB:
            status = EnableERRB(handle, false);
            break;
        case CDI_CONFIG_MAX96712_ENABLE_LOCK:
            status = EnableLOCK(handle, true);
            break;
        case CDI_CONFIG_MAX96712_DISABLE_LOCK:
            status = EnableLOCK(handle, false);
            break;
        default:
            SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Invalid command");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }

    return status;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
static NvMediaStatus
MAX96712SetDeviceConfigGroup2(
    DevBlkCDIDevice const* handle,
    ConfigSetsMAX96712 switchSetDeviceCfg)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    switch (switchSetDeviceCfg) {
        case CDI_CONFIG_MAX96712_ENABLE_PG:
            status = EnablePG(handle);
            break;
        case CDI_CONFIG_MAX96712_MAP_UNUSED_PIPE:
            status = MapUnusedPipe(handle);
            break;
        case CDI_CONFIG_MAX96712_ENABLE_REPLICATION:
            status = EnableReplication(handle, true);
            break;
        case CDI_CONFIG_MAX96712_DISABLE_REPLICATION:
            status = EnableReplication(handle, false);
            break;
        default:
            SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Invalid command");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }

    return status;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/**
 * @brief  Set all Device Config
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - verifies drvHandle is not NULL
 * - verifies enumeratedDeviceConfig is valid
 * - status = MAX96712SetDeviceConfigGroup1(handle, enumeratedDeviceConfig)
 *
 * @param[in] handle Dev Blk handle
 * @param[in] enumeratedDeviceConfig device configuration to apply
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
NvMediaStatus
MAX96712SetDeviceConfig(
    DevBlkCDIDevice const* handle,
    ConfigSetsMAX96712 enumeratedDeviceConfig)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else if ((enumeratedDeviceConfig > CDI_CONFIG_MAX96712_INVALID)
        && (enumeratedDeviceConfig <= CDI_CONFIG_MAX96712_DISABLE_LOCK)) {
        status = MAX96712SetDeviceConfigGroup1(handle, enumeratedDeviceConfig);
    }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    else if ((enumeratedDeviceConfig > CDI_CONFIG_MAX96712_DISABLE_LOCK) &&
             (enumeratedDeviceConfig < CDI_CONFIG_MAX96712_NUM)) {
        status = MAX96712SetDeviceConfigGroup2(handle, enumeratedDeviceConfig);
    }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
    else {
        SIPL_LOG_ERR_STR_UINT("MAX96712: Bad parameter: invalid dev config",
                             (uint32_t)enumeratedDeviceConfig);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return status;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
NvMediaStatus
MAX96712WriteRegister(
    DevBlkCDIDevice const* handle,
    uint32_t deviceIndex,
    uint16_t registerNum,
    uint32_t dataLength,
    uint8_t const* dataBuff)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    uint32_t i = 0U;
    if ((NULL == drvHandle) || (NULL == dataBuff)) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else if (dataLength > (uint32_t)REG_WRITE_BUFFER_BYTES) {
        SIPL_LOG_ERR_STR("MAX96712: Insufficient buffering");
        status = NVMEDIA_STATUS_INSUFFICIENT_BUFFERING;
    } else {

        for (i = 0; i < dataLength; i++) {
            /* Note: Not used in DES _nv driver and IMX728 and IMX623 CDD files.
             */
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                registerNum,
                                                dataBuff[i]);

            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_AND_UINT("MAX96712: Register I2C write failed with status",
                                               registerNum, (uint32_t)status);
                break;
            }
        }
    }

    /* NOP: just to make sure misra_cpp_2008_rule_2_7_violation is suppressed */
    (void) deviceIndex;

    return status;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/**
 * @brief Write Parameters Group 1
 *
 * - if parameterType == CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING
 *  - status = SetPipelineMap(
 *   - handle,
 *   - param->PipelineMapping.link,
 *   - param->PipelineMapping.linkPipelineMap,
 *   - parameterSize, sizeof(param->PipelineMapping),
 *   - isValidSize)
 *   .
 *  .
 * - elif parameterType == CDI_WRITE_PARAM_CMD_MAX96712_OVERRIDE_DATATYPE
 *  - status = OverrideDataType(
 *   - handle,
 *   - param->PipelineMapping.link,
 *   -  param->PipelineMapping.linkPipelineMap,
 *   -  parameterSize, sizeof(param->PipelineMapping),
 *   - isValidSize)
 *   .
 *  .
 * - elif parameterType == CDI_WRITE_PARAM_CMD_MAX96712_SET_FSYNC
 *  - verify parameterSize == sizeof(param->FSyncSettings))
 *   - if not
 *    - set *isValidSize = false , error exit
 *    .
 *   - else
 *    - set set *isValidSize = true
 *    .
 *   .
 *  - status = SetFSYNCMode(
 *   - handle,
 *   - param->FSyncSettings.FSyncMode,
 *   - param->FSyncSettings.pclk,
 *   - param->FSyncSettings.fps,
 *   - param->FSyncSettings.link)
 *   .
 *  .
 * - elif parameterType == CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS
 *  - status = CheckAgainstActiveLink(status, handle, param->link)
 *  - status = EnableSpecificLinks(
 *   - handle, param->link, parameterSize, isValidSize)
 *   .
 *  .
 * - elif parameterType == CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_DOUBLE_PIXEL_MODE
 *  - verify (parameterSize == sizeof(param->DoublePixelMode))
 *   - if not
 *    - set *isValidSize = false , error exit
 *    .
 *   - else
 *    - set set *isValidSize = true
 *    .
 *   .
 *  - status = EnableDoublePixelMode(
 *   - handle,
 *   - param->DoublePixelMode.link,
 *   - param->DoublePixelMode.dataType,
 *   - param->DoublePixelMode.embDataType,
 *   - param->DoublePixelMode.isSharedPipeline)
 *   .
 *  .
 * - elif  parameterType == CDI_WRITE_PARAM_CMD_MAX96712_SET_MIPI
 *  - verify  (parameterSize == sizeof(param->MipiSettings))
 *   - if not
 *    - set *isValidSize = false , error exit
 *    .
 *   - else
 *    - set set *isValidSize = true
 *    .
 *   .
 *  - status = ConfigureMIPIOutput(
 *   - handle,
 *   - param->MipiSettings.mipiSpeed,
 *   - param->MipiSettings.phyMode)
 *   .
 *  .
 * - elif  parameterType == CDI_WRITE_PARAM_CMD_MAX96712_SET_TX_SRC_ID
 *  - status = CheckAgainstActiveLink(status, handle, param->link)
 *  - call status = SetTxSRCId(
 *   - handle, param->link, parameterSize, isValidSize)
 *   .
 *  .
 * - elif  parameterType == CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PERIODIC_AEQ
 *  - status = CheckAgainstActiveLink(status, handle, param->link)
 *  . status = EnablePeriodicAEQ(
 *   - handle, param->link, parameterSize, isValidSize)
 *   .
 *  .
 * - elif  parameterType == CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_GPIO_RX
 *  - status = CheckAgainstActiveLink(status, handle, param->link)
 *  - status = EnableGPIORx(
 *   - handle, param->gpioIndex, param->link,
 *   - parameterSize, isValidSize)
 *   .
 *  .
 * - elif  parameterType == CDI_WRITE_PARAM_CMD_MAX96712_UNSET_ERRB_RX
 *  - status = CheckAgainstActiveLink(status, handle, param->GpioErrbSetting.link)
 *  - status = MAX96712UnsetErrbRx(
 *   - handle, param->GpioErrbSetting.link,
 *   - parameterSize, sizeof(param->GpioErrbSetting),isValidSize)
 *   .
 *  .
 * - elif  parameterType == CDI_WRITE_PARAM_CMD_MAX96712_SET_ERRB_RX
 *  - status = MAX96712SetErrbRx(
 *   - handle, param->GpioErrbSetting.link, parameterSize,
 *   - sizeof(param->GpioErrbSetting), isValidSize)
 *   .
 *  .
 * - elif  parameterType == CDI_WRITE_PARAM_CMD_MAX96712_SET_ERRB_RX_ID
 *  - verify (parameterSize == sizeof(param->GpioErrbSetting))
 *   - if not
 *    - set *isValidSize = false , error exit
 *    .
 *   - else
 *    - set set *isValidSize = true
 *    .
 *   .
 *  - status = CheckAgainstActiveLink(status, handle, param->GpioErrbSetting.link)
 *  - status = MAX96712SetErrbRxId(
 *   - handle, param->GpioErrbSetting.gpioIndex, param->GpioErrbSetting.link)
 *   .
 *  .
 * - else
 *  - error: ad parameter: Invalid command
 * @param[in] handle dev BLK handle
 * @param[in] parameterType type of parameter to apply
 * @param[in] parameterSize user specified parameter record size
 * @param[in] parameter void pointer to user's parameter buffer
 * @param[out] isValidSize returned value set to tru if parameter size matches user specified size
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
static NvMediaStatus
MAX96712WriteParametersGroup1(
    DevBlkCDIDevice const* handle,
    WriteParametersCmdMAX96712 const parameterType,
    size_t parameterSize,
    void const* parameter,
    bool* isValidSize)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    /* coverity[misra_c_2012_rule_11_5_violation] : intentional TID-1417 */
    WriteParametersParamMAX96712 const* param = (WriteParametersParamMAX96712 const*)parameter;

    switch (parameterType) {
        case CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING:
            status = SetPipelineMap(handle,
                                    param->PipelineMapping.link,
                                    param->PipelineMapping.linkPipelineMap,
                                    parameterSize, sizeof(param->PipelineMapping),
                                    isValidSize);
            break;
       case CDI_WRITE_PARAM_CMD_MAX96712_OVERRIDE_DATATYPE:
            status = OverrideDataType(handle,
                                      param->PipelineMapping.link,
                                      param->PipelineMapping.linkPipelineMap,
                                      parameterSize, sizeof(param->PipelineMapping),
                                      isValidSize);

            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_SET_FSYNC:
            if (parameterSize == sizeof(param->FSyncSettings)) {
                *isValidSize = true;
                status = SetFSYNCMode(handle,
                                      param->FSyncSettings.FSyncMode,
                                      param->FSyncSettings.pclk,
                                      param->FSyncSettings.fps,
                                      param->FSyncSettings.link);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_SET_HETERO_FRAME_SYNC:
            if (parameterSize == sizeof(param->HeteroFSyncSettings)) {
                *isValidSize = true;
                status = SetHeteroFSYNCMode(handle,
                                            param->HeteroFSyncSettings.gpioNum);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS:
            PrintLogMsg(LOG_LEVEL_INFO, "Calling CheckAgainstActiveLink before "
                                        "EnableSpecificLinks, linkMask = %x\n",
                        (uint32_t)param->link);
            status = CheckAgainstActiveLink(status, handle);
            if (status == NVMEDIA_STATUS_OK) {
                status = EnableSpecificLinks(handle,
                                             param->link,
                                             parameterSize,
                                             isValidSize,
                                             true);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS_NO_CHECK:
            PrintLogMsg(LOG_LEVEL_INFO, "Calling CheckAgainstActiveLink before "
                                        "EnableSpecificLinks, linkMask = %x\n",
                        (uint32_t)param->link);
            status = CheckAgainstActiveLink(status, handle);
            if (status == NVMEDIA_STATUS_OK) {
                status = EnableSpecificLinks(handle,
                                             param->link,
                                             parameterSize,
                                             isValidSize,
                                             false);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_DOUBLE_PIXEL_MODE:
            if (parameterSize == sizeof(param->DoublePixelMode)) {
                *isValidSize = true;
                PrintLogMsg(LOG_LEVEL_INFO, "Calling CheckAgainstActiveLink before "
                                            "EnableDoublePixelMod, linkMask = %x\n",
                            (uint32_t)param->DoublePixelMode.link);
                status = CheckAgainstActiveLink(status, handle);
                if (status == NVMEDIA_STATUS_OK) {
                    status = EnableDoublePixelMode(handle,
                                                   param->DoublePixelMode.link,
                                                   param->DoublePixelMode.dataType,
                                                   param->DoublePixelMode.embDataType,
                                                   param->DoublePixelMode.isSharedPipeline);
                }
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_SET_MIPI:
            if (parameterSize == sizeof(param->MipiSettings)) {
                *isValidSize = true;
                status = ConfigureMIPIOutput(handle,
                                             param->MipiSettings.mipiSpeed,
                                             param->MipiSettings.phyMode);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_SET_TX_SRC_ID:
            PrintLogMsg(LOG_LEVEL_INFO, "Calling CheckAgainstActiveLink before "
                                        "SetTxSRCId, linkMask = %x\n",
                        (uint32_t)param->link);
            status = CheckAgainstActiveLink(status, handle);
            if (status == NVMEDIA_STATUS_OK) {
                status = SetTxSRCId(handle,
                                    param->link,
                                    parameterSize,
                                    isValidSize);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PERIODIC_AEQ:
            PrintLogMsg(LOG_LEVEL_INFO, "Calling CheckAgainstActiveLink "
                                        "before EnablePeriodicAEQ, linkMask = %x\n",
                        (uint32_t)param->link);
            status = CheckAgainstActiveLink(status, handle);
            if (status == NVMEDIA_STATUS_OK) {
                status = EnablePeriodicAEQ(handle,
                                           param->link,
                                           parameterSize,
                                           isValidSize);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_GPIO_RX:
            PrintLogMsg(LOG_LEVEL_INFO, "Calling CheckAgainstActiveLink before "
                                        "EnableGPIORx, linkMask = %x\n",
                        (uint32_t)param->link);
            status = CheckAgainstActiveLink(status, handle);
            if (status == NVMEDIA_STATUS_OK) {
                status = EnableGPIORx(handle,
                                      param->gpioIndex,
                                      param->link,
                                      parameterSize,
                                      isValidSize);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_UNSET_ERRB_RX:
            PrintLogMsg(LOG_LEVEL_INFO, "Calling CheckAgainstActiveLink before "
                                        "MAX96712UnsetErrbRx, linkMask = %x\n",
                        (uint32_t)param->GpioErrbSetting.link);
            status = CheckAgainstActiveLink(status, handle);
            if (status == NVMEDIA_STATUS_OK) {
                status = MAX96712UnsetErrbRx(handle,
                                             param->GpioErrbSetting.link,
                                             parameterSize,
                                             sizeof(param->GpioErrbSetting),
                                             isValidSize);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_SET_ERRB_RX:
                status = MAX96712SetErrbRx(handle,
                                           param->GpioErrbSetting.link,
                                           parameterSize,
                                           sizeof(param->GpioErrbSetting),
                                           isValidSize);
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_LINK:
            status = MAX96712DisableLink(handle,
                                         param->linkIndex,
                                         parameterSize,
                                         sizeof(uint8_t),
                                         isValidSize);
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_RESTORE_LINK:
            status = MAX96712RestoreLink(handle,
                                         param->linkIndex,
                                         parameterSize,
                                         sizeof(uint8_t),
                                         isValidSize);
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_SET_ERRB_RX_ID:
            if (parameterSize == sizeof(param->GpioErrbSetting)) {
                *isValidSize = true;
                PrintLogMsg(LOG_LEVEL_INFO, "Calling CheckAgainstActiveLink before "
                                            "MAX96712SetErrbRxId, linkMask = %x\n",
                            (uint32_t)param->GpioErrbSetting.link);
                status = CheckAgainstActiveLink(status, handle);
                if (status == NVMEDIA_STATUS_OK) {
                    status = MAX96712SetErrbRxId(handle,
                                                 param->GpioErrbSetting.gpioIndex,
                                                 param->GpioErrbSetting.link);
                }
            }
            break;
        default:
            SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Invalid command");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }

    return status;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
static NvMediaStatus
MAX96712WriteParametersGroup2(
    DevBlkCDIDevice const* handle,
    WriteParametersCmdMAX96712 const parameterType,
    size_t parameterSize,
    void const* parameter,
    bool* isValidSize)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    /* coverity[misra_c_2012_rule_11_5_violation] : intentional TID-1417 */
    WriteParametersParamMAX96712 const* param = (WriteParametersParamMAX96712 const*)parameter;

    switch (parameterType) {
        case CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING_TPG:
            if (parameterSize == sizeof(param->PipelineMapping)) {
                *isValidSize = true;
                status = SetPipelineMapTPG(handle,
                                           param->PipelineMappingTPG.linkIndex,
                                           param->PipelineMappingTPG.linkPipelineMap);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_PACKET_BASED_CONTROL_CHANNEL:
            PrintLogMsg(LOG_LEVEL_INFO, "Calling CheckAgainstActiveLink before "
                                        "DisablePacketBasedControlChannel, linkMask = %x\n",
                        (uint32_t)param->link);
            status = CheckAgainstActiveLink(status, handle);
            if (status == NVMEDIA_STATUS_OK) {
                status = EnablePacketBasedControlChannel(handle,
                                                         param->link,
                                                         false,
                                                         parameterSize,
                                                         isValidSize);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PACKET_BASED_CONTROL_CHANNEL:
            PrintLogMsg(LOG_LEVEL_INFO, "Calling CheckAgainstActiveLink before "
                                        "EnablePacketBasedControlChannel, linkMask = %x\n",
                        (uint32_t)param->link);
            status = CheckAgainstActiveLink(status, handle);
            if (status == NVMEDIA_STATUS_OK) {
                status = EnablePacketBasedControlChannel(handle,
                                                         param->link,
                                                         true,
                                                         parameterSize,
                                                         isValidSize);

            }
            break;
         case CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_FORWARD_CHANNELS:
            if (parameterSize == sizeof(param->link)) {
                *isValidSize = true;
                status = ControlForwardChannels(handle,
                                                param->link,
                                                false);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_FORWARD_CHANNELS:
            if (parameterSize == sizeof(param->link)) {
                *isValidSize = true;
                status = ControlForwardChannels(handle,
                                                param->link,
                                                true);
            }
            break;
       case CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_DE:
            if (parameterSize == sizeof(param->link)) {
                *isValidSize = true;
                status = DisableDE(handle,
                                   param->link);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_SET_DEFAULT_GMSL1_HIM_ENABLED:
            if (parameterSize == sizeof(param->GMSL1HIMEnabled)) {
                *isValidSize = true;
                PrintLogMsg(LOG_LEVEL_INFO, "Calling CheckAgainstActiveLink before "
                                            "SetDefaultGMSL1HIMEnabled, linkMask = %x\n",
                            (uint32_t)param->GMSL1HIMEnabled.link);
                status = CheckAgainstActiveLink(status, handle);
                if (status == NVMEDIA_STATUS_OK) {
                    status = SetDefaultGMSL1HIMEnabled(handle,
                                                       param->GMSL1HIMEnabled.link,
                                                       param->GMSL1HIMEnabled.step);
                }
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_SET_DBL:
            if (parameterSize == sizeof(param->link)) {
                *isValidSize = true;
                status = SetDBL(handle,
                                param->link,
                                true);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_UNSET_DBL:
            if (parameterSize == sizeof(param->link)) {
                *isValidSize = true;
                status = SetDBL(handle,
                                param->link,
                                false);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_AUTO_ACK:
            if (parameterSize == sizeof(param->link)) {
                *isValidSize = true;
                status = DisableAutoAck(handle,
                                        param->link);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_SET_PG:
            if (parameterSize == sizeof(param->SetPGSetting)) {
                *isValidSize = true;
                status = ConfigPGSettings(handle,
                                          param->SetPGSetting.width,
                                          param->SetPGSetting.height,
                                          param->SetPGSetting.frameRate,
                                          param->SetPGSetting.linkIndex);
            }
            break;
        default:
            SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Invalid command");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }

    return status;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/**
 * @brief Write Parameters
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - verifies drvHandle and parameters are not NULL
 * - if ((parameterType > CDI_WRITE_PARAM_CMD_MAX96712_INVALID) &&
 *  - (parameterType <= CDI_WRITE_PARAM_CMD_MAX96712_SET_ERRB_RX_ID))
 *  - status = MAX96712WriteParametersGroup1(
 *   - handle, parameterType, parameterSize, parameter, &isValidSize)
 *   .
 *  .
 * - else
 *  - Error:  Bad parameter: invalid write param cmd
 *
 * @param[in] handle Dev Blk handle
 * @param[in] parameterType selects parameter to setup
 * @param[in] parameterSize user provided parameter size
 * @param[in] parameter pointer to parameter structure
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
NvMediaStatus
MAX96712WriteParameters(
    DevBlkCDIDevice const* handle,
    WriteParametersCmdMAX96712 parameterType,
    size_t parameterSize,
    void const* parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
        /* coverity[misra_c_2012_rule_11_5_violation] : intentional TID-1417 */
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    bool isValidSize = false;

    if ((NULL == drvHandle) || (NULL == parameter)) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        if ((parameterType > CDI_WRITE_PARAM_CMD_MAX96712_INVALID) &&
            (parameterType <= CDI_WRITE_PARAM_CMD_MAX96712_SET_ERRB_RX_ID)) {
            status = MAX96712WriteParametersGroup1(handle,
                                                   parameterType,
                                                   parameterSize,
                                                   parameter,
                                                   &isValidSize);
        }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
        else if ((parameterType > CDI_WRITE_PARAM_CMD_MAX96712_SET_ERRB_RX_ID) &&
                 (parameterType < CDI_WRITE_PARAM_CMD_MAX96712_NUM)) {
                 status = MAX96712WriteParametersGroup2(handle,
                                                        parameterType,
                                                        parameterSize,
                                                        parameter,
                                                        &isValidSize);
        }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
        else {
            SIPL_LOG_ERR_STR_UINT("MAX96712: Bad parameter: invalid write param cmd",
                                 (uint32_t)parameterType);
            status = NVMEDIA_STATUS_BAD_PARAMETER;
        }

        if ((status == NVMEDIA_STATUS_OK) && (!isValidSize)) {
            SIPL_LOG_ERR_STR_UINT("MAX96712: Bad parameter: Invalid param size",
                                 (uint32_t)parameterType);
            status = NVMEDIA_STATUS_BAD_PARAMETER;
        }
    }

    return status;
}

/**
 * @brief Returns link mask from link index
 *
 * @param[in] linkNum link index
 * @return  if linkNum == 0 return CDI_MAX96712_LINK_0
 *          elif linkNum == 1 return CDI_MAX96712_LINK_1
 *          elif linkNum == 2 return CDI_MAX96712_LINK_2
 *          elif linkNum == 3 return CDI_MAX96712_LINK_3
 *          else
 *              return   CDI_MAX96712_LINK_NONE
 */
LinkMAX96712
GetMAX96712Link(uint8_t linkNum)
{
    LinkMAX96712 statusLink = CDI_MAX96712_LINK_NONE;

    switch (linkNum) {
        case 0:
            statusLink = CDI_MAX96712_LINK_0;
            break;
        case 1:
            statusLink = CDI_MAX96712_LINK_1;
            break;
        case 2:
            statusLink = CDI_MAX96712_LINK_2;
            break;
        case 3:
            statusLink = CDI_MAX96712_LINK_3;
            break;
        default:
            statusLink = CDI_MAX96712_LINK_NONE;
            break;
    }
    return statusLink;
}

NvMediaStatus
MAX96712PRBSChecker(
    DevBlkCDIDevice const* handle,
    bool enable)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);

    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712PRBSChecker: Bad parameter");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        uint16_t vPRBSRegs[8] = {0x1DC, 0x1FC, 0x21C, 0x23C, 0x25C, 0x27C, 0x29C, 0x2BC};
        uint8_t videoPipeLineEnabled = 0;

        /* Read 0xF4 to check how many video pipelines are enabled */
        status =  MAX96712ReadUint8Verify(handle, 0xF4, &videoPipeLineEnabled);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("MAX96712PRBSChecker: Failed to read");
            goto end;
        }

        if (enable == true) {
            /* Enable PRBS checker for all enabled video pipelines */
            for (uint8_t numPipe = 0; numPipe < MAX96712_NUM_VIDEO_PIPELINES; numPipe++) {
                if (((videoPipeLineEnabled >> numPipe) & 1U) != 0U) {
                     /* Write 0x10 to enable PRBS checker as 7th bit is for PRBS checker */
                     status = MAX96712WriteUint8Verify(handle,
                                                       vPRBSRegs[numPipe],
                                                       0x10);
                     if (NVMEDIA_STATUS_OK != status) {
                         SIPL_LOG_ERR_STR("MAX96712PRBSChecker: Failed to write");
                         goto end;
                     }
                }
            }
        } else {
            /* Disable PRBS checker for all enabled video pipelines */
            for (uint8_t numPipe = 0; numPipe < MAX96712_NUM_VIDEO_PIPELINES; numPipe++) {
                if ((videoPipeLineEnabled >> numPipe) & 1U) {
                    /* Write 0x00 to disable PRBS checker as 7th bit is for PRBS checker */
                    status = MAX96712WriteUint8Verify(handle,
                                                      vPRBSRegs[numPipe],
                                                      0x00);
                    if (NVMEDIA_STATUS_OK != status) {
                        SIPL_LOG_ERR_STR("MAX96712PRBSChecker: Failed to write");
                        goto end;
                    }
                }
            }
        }
    }
end:
    return status;
}

NvMediaStatus
MAX96712VerifyVPRBSErrs(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);

    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712VerifyVPRBSErrs: Bad parameter");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        uint8_t videoPipeLineEnabled = 0;
        bool videoLock = false;
        uint16_t vPRBSRegs[MAX96712_NUM_VIDEO_PIPELINES] = {0x1DC, 0x1FC, 0x21C, 0x23C,
                                                            0x25C, 0x27C, 0x29C, 0x2BC};
        uint16_t PRBSErr[MAX96712_NUM_VIDEO_PIPELINES] = {0x1DB, 0x1FB, 0x21B,0x23B,
                                                          0x25B, 0x27B, 0x29B, 0x2BB};

        /* Read 0xF4 to check how many video pipelines are enabled */
        status = MAX96712ReadUint8Verify(handle, 0xF4, &videoPipeLineEnabled);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("MAX96712VerifyVPRBSErrs: Failed to read VIDEO_PIPE_EN register");
            goto end;
        }

        /* Check VPRBS regs for enabled video pipeline */
        for (uint8_t numPipe = 0; numPipe < MAX96712_NUM_VIDEO_PIPELINES; numPipe++) {
            uint8_t vPRBSRegData;
            if (((videoPipeLineEnabled >> numPipe) & 1U) != 0U) {
                status = MAX96712ReadUint8Verify(handle,
                                                 vPRBSRegs[numPipe],
                                                 &vPRBSRegData);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR("MAX96712VerifyVPRBSErrs: Failed to read VPRBS registers");
                    goto end;
                }

                if (((vPRBSRegData >> 0) & 1) == 1)
                   videoLock = true;
            }
        }

        /* Check if any one VPRBS regs has VIDEO_LOCK = 1 */
        if (!videoLock) {
            status  = NVMEDIA_STATUS_ERROR;
            SIPL_LOG_ERR_STR("MAX96712VerifyVPRBSErrs: VIDEO_LOCK failed");
            goto end;
        }

        /* Check VPRBS_ERR regs for enabled video pipeline */
        for (uint8_t numPipe = 0; numPipe < 8; numPipe++) {
            uint8_t data;
            if ((videoPipeLineEnabled >> numPipe) & 1U) {
                status = MAX96712ReadUint8Verify(handle,
                                                 PRBSErr[numPipe],
                                                 &data);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR("MAX96712VerifyVPRBSErrs: Failed to read VPRBS_ERR regs");
                    goto end;
                }
                /* Check it does not have any VPRBS_ERR */
                if (data != 0U) {
                    status = NVMEDIA_STATUS_ERROR;
                    SIPL_LOG_ERR_STR("MAX96712VerifyVPRBSErrs: VPRBS_ERR found");
                    goto end;
                }
            }
        }
    }

end:
    return status;
}

NvMediaStatus
MAX96712EnableDECErrToERRB(
    DevBlkCDIDevice const* handle,
    bool enable)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);

    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712EnableDECErrorToERRB: Bad parameter");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        if (enable) {
            /* Enable DEC_ERR to ERRB */
            status = MAX96712WriteUint8Verify(handle, REG_INTR2, 0x0f);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR("MAX96712EnableDECErrorToERRB: Failed to write REG_INTR2");
            }
        } else {
            status = MAX96712WriteUint8Verify(handle, REG_INTR2, 0x00);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR("MAX96712EnableDECErrorToERRB: Failed to write REF_INTR2");
            }
        }
    }
    return status;
}

NvMediaStatus
MAX96712VerifyERRGDiagnosticErrors(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);

    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712VerifyERRGDiagnosticErrors: Bad parameter");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        uint8_t dec_err, idle_err;
        uint16_t dec_err_regs[MAX96712_MAX_NUM_LINK] = {REG_CNT0, REG_CNT1, REG_CNT2, REG_CNT3};
        uint16_t idle_err_regs[MAX96712_MAX_NUM_LINK] = {REG_CNT4, REG_CNT5, REG_CNT6, REG_CNT7};

        DevBlkCDII2CReg ErrFlagRegs[] = {
            /* DEC_ERR Flag */
            {REG_INTR3, 0x00},
            /* IDLE_ERR Flag */
            {REG_INTR9, 0x00}
        };
        DevBlkCDII2CRegListWritable ErrFlagData = {
            .regs = ErrFlagRegs,
            .numRegs = (uint32_t)(sizeof(ErrFlagRegs) /
                                sizeof(ErrFlagRegs[0])),
        };

        status = MAX96712ReadArrayVerify(handle, &ErrFlagData);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96712VerifyERRGDiagnosticErrors: Failed to read error flags");
            return status;
        }

        /* Verify error flags are asserted */
        if((ErrFlagRegs[0].data == 0U) || (ErrFlagRegs[1].data == 0U)) {
            SIPL_LOG_ERR_STR("MAX96712VerifyERRGDiagnosticErrors: Error flags are not asserted");
            status = NVMEDIA_STATUS_ERROR;
            return status;
        }

        /* Read registers to clear the errors */
        for (uint8_t i = 0; i < MAX96712_MAX_NUM_LINK; i++) {
            if ((((uint32_t)ErrFlagRegs[0].data >> i) & 1U) == 1U) {
                status = MAX96712ReadUint8Verify(handle, dec_err_regs[i], &dec_err);
                if (status != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR_HEX_UINT("MAX96712VerifyERRGDiagnosticErrors: Failed to read"
                                              "DEC_ERR register: ", dec_err_regs[i]);
                    break;
                }
            }
            if ((((uint32_t)ErrFlagRegs[1].data >> i) & 1U) == 1U) {
                status = MAX96712ReadUint8Verify(handle, idle_err_regs[i], &idle_err);
                if (status != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR_HEX_UINT("MAX96712VerifyERRGDiagnosticErrors: Failed to read"
                                              "IDLE_ERR register: ", idle_err_regs[i]);
                    break;
                }
            }
        }
    }

    return status;
}

DevBlkCDIDeviceDriver *
GetMAX96712NewDriver(void)
{
    /** true if structure is initialized, otherwise false */
    static bool deviceDriverMAX96712_initialized = false;
    static DevBlkCDIDeviceDriver deviceDriverMAX96712;

    if (!deviceDriverMAX96712_initialized) {
        deviceDriverMAX96712.deviceName    = "Maxim 96712 Deserializer";
        deviceDriverMAX96712.regLength     = (int32_t)MAX96712_NUM_ADDR_BYTES;
        deviceDriverMAX96712.dataLength    = (int32_t)MAX96712_NUM_DATA_BYTES;
        deviceDriverMAX96712.DriverCreate  = DriverCreateMAX96712;
        deviceDriverMAX96712.DriverDestroy = DriverDestroyMAX96712;
        deviceDriverMAX96712_initialized = true;
    }

    return &deviceDriverMAX96712;
}

NvMediaStatus
IsLinkLock(
    DevBlkCDIDevice const* handle,
    uint8_t const linkIndex,
    bool *bLocked)
{

    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    NvMediaStatus nvstatus = NVMEDIA_STATUS_OK;

    if ((drvHandle == NULL) || (linkIndex >= MAX96712_MAX_NUM_LINK)
        || (bLocked == NULL)) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: IsLinkLock");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    uint8_t lockStatus = 0U;
    nvstatus = MAX96712ReadUint8Verify(handle,
                                       (uint16_t)regBitFieldProps
                                       [REG_FIELD_GMSL2_LOCK_A + linkIndex].regAddr,
                                       &lockStatus);

    /* bLocked = true means no error */
    if (nvstatus == NVMEDIA_STATUS_OK) {
        *bLocked = false;
        if ((lockStatus & 0x8U) != 0U) {
            *bLocked = true;
        }
    }

    return nvstatus;
}

NvMediaStatus
IsErrbSet(
    DevBlkCDIDevice const* handle,
    bool *bSet)
{
    DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
    NvMediaStatus nvstatus = NVMEDIA_STATUS_OK;

    if ((drvHandle == NULL) || (bSet == NULL)) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: IsErrbSet");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    uint8_t errbStatus = 0;
    nvstatus = MAX96712ReadUint8Verify(handle,
                                       (uint16_t)regBitFieldProps
                                       [REG_FIELD_ERRB].regAddr,
                                       &errbStatus);

    /* bSet = true means error */
    if (nvstatus == NVMEDIA_STATUS_OK) {
        *bSet = false;
        if (errbStatus & 0x4U) {
            *bSet = true;
        }
    }

    return nvstatus;
}
