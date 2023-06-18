/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef AR0820_CUST1_CUSTOMDATA_H
#define AR0820_CUST1_CUSTOMDATA_H

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
#include <cstdbool>
#include <cmath>
#include <cstdint>
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#else
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif

/** Maxinum number of voltage data */
#define MAXNUM_VOLTAGE    (  18U  )
/** Maxinum number of raw PDI data */
#define MAX_PDI_REGNUM    ( 400U  )
#define NUM_ACKS    ( 5 )
#define ROI_STATS_BYTE_NUM ( 18U )

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
namespace nvsipl {
extern "C" {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif

/** @brief the data array of PDI data */
typedef struct {
    /** read out defective pixel list */
    uint16_t pdi[MAX_PDI_REGNUM];
} AR0820PDIList;

/** @brief the parsed Embedded data (customized information) */
typedef struct {
    /** indicate voltage error */
    uint16_t volterr;
    /** indicate temperature error */
    uint16_t temperr;
    /** indicate stats error */
    uint16_t statserr;
    /** current voltage value array */
    float_t  currvolt[MAXNUM_VOLTAGE];
    /** maximum voltage value array */
    float_t  maxvolt[MAXNUM_VOLTAGE];
    /** minimum voltage value array */
    float_t  minvolt[MAXNUM_VOLTAGE];
    /** current temperature */
    float_t  currtemp;
    /** expected roi1 stats */
    uint8_t expect_roiStats[ROI_STATS_BYTE_NUM];
    /** current roi1 stats */
    uint8_t roiStats[ROI_STATS_BYTE_NUM];
} AR0820CustomEmbeddedData;

/**
 * @brief Describes the custom MAX9295A serializer error information related to
 *        ERRB pin and needed by the application.
 */
typedef struct {
    /** Indicates the value of \b INTR5 interrupt register in the MAX9295A serializer. */
    uint8_t  intr5Reg;
    /** Indicates the value of \b INTR7 interrupt register in the MAX9295A serializer. */
    uint8_t  intr7Reg;
    /** Indicates the number of Link A decoding errors detected in the MAX9295A serializer. */
    uint8_t  decErrA;
    /** Indicates the number of idle-word errors detected in the MAX9295A serializer. */
    uint8_t  idleErr;
    /** Indicates max re-transmission error in main control channel path.
     *  bit7 indicates MAX_RT_ERROR flag, bits[6:0] indicates retransmission count. */
    uint8_t  maxRTErrCC;
    /** Indicates max re-transmission error in GPIO path.
     *  bit7 indicates MAX_RT_ERROR flag, bits[6:0] indicates retransmission count. */
    uint8_t  maxRTErrGPIO;
    /** Indicates max re-transmission error in passthrough channel 1 I2CX path.
     *  bit7 indicates MAX_RT_ERROR flag, bits[6:0] indicates retransmission count. */
    uint8_t  maxRTErrI2CX;
    /** Indicates max re-transmission error in passthrough channel 2 I2CY path.
     *  bit7 indicates MAX_RT_ERROR flag, bits[6:0] indicates retransmission count. */
    uint8_t  maxRTErrI2CY;
    /** Indicates MIPI PHY1 LP error register value \b phy1_lp_err in
     *  the MAX9295A serializer */
    uint8_t  phy1LpErr;
    /** Indicates MIPI PHY1 high speed error register value \b phy1_hs_err in
     *  the MAX9295A serializer */
    uint8_t  phy1HsErr;
    /** Indicates MIPI PHY2 LP error register value \b phy2_lp_err in
     *  the MAX9295A serializer */
    uint8_t  phy2LpErr;
    /** Indicates MIPI PHY2 high speed error register value \b phy2_hs_err in
     *  the MAX9295A serializer */
    uint8_t  phy2HsErr;
    /** Indicates CSI2 Ctrl1 status register value in \b ctrl1_csi_err_l bits[7:0] in
     *  the MAX9295A serializer */
    uint8_t ctrl1CSIErrL;
    /** Indicates CSI2 Ctrl1 status register value in \b ctrl1_csi_err_h bits[2:0] in
     *  the MAX9295A serializer */
    uint8_t ctrl1CSIErrH;
    /** INTR3 interrupt register in MAX9295A serializer */
    uint8_t  intr3Reg;
    /** CTRL3 status    register in MAX9295A serializer */
    uint8_t  ctrl3Reg;
    /** MFP5 (SYS_CHECK) status GPIO5 register bit3 in MAX9295A serializer */
    uint8_t  gpio5SysCheck;
 } MAX9295AErrbErrInfo;

/** the number of PMIC startup ACK registers */
#define NUM_ACKS    ( 5 )
/**
 * @brief Describes the custom TPS650332 PMIC startup ACK information needed by the application.
 */
typedef struct {
    /** The array of register index and value pair to hold ACKs information
     *  DEV_ERR_ACK_1
     *  DEV_ERR_ACK_2
     *  ABIST_BULK1_2_ACK
     *  ABIST_LDO_BUCK3_ACK
     *  ABIST_SYSTEM_ACK **/
    struct {
        uint8_t index;
        uint8_t value;
    } reg[NUM_ACKS];
} TPS650332CustomStartupAck;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
}
} /* namespace nvsipl */
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif

#endif //AR0820_CUST1_CUSTOMDATA_H
