/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * All information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef IMX623_CUSTOMDATA_H
#define IMX623_CUSTOMDATA_H

#ifdef __cplusplus
#include <cstdbool>
#else
#include <stdbool.h>
#endif

#ifdef __cplusplus
namespace nvsipl {
extern "C" {
#endif

typedef uint64_t errorInfoTypeIMX623;

/**
 * @brief Describes the Custom Embedded Data needed by the application.
 */
typedef struct {
    /** Bit field indicating the error flags set, indexed by @ref IMX623CustomErrType
     * enum
     * Valid range : [0, UINT64_MAX].
     */
    errorInfoTypeIMX623 customEmbeddedDataError;
} IMX623CustomEmbeddedData;

/**
 * @brief Describes the IMX623 Safety mechanism error types.
 */
typedef enum {
    /** Clock Monitor SM error */
    IMX623_CLKMON_ERROR = 0U,
    /** Sequence Monitor SM error */
    IMX623_SEQMON_ERROR,
    /** Control Flow Monitor SM error */
    IMX623_CTRLFLOWMON_ERROR,
    /** Sync Monitor SM error */
    IMX623_SYNCMON_ERROR,
    /** Outside threshold range fault for temperature sensor 1
     * of Dual Temperature Sensor SM */
    IMX623_DUALTEMP1_ERROR,
    /** Outside threshold range fault for temperature sensor 2
     * of Dual Temperature Sensor SM */
    IMX623_DUALTEMP2_ERROR,
    /** Detection result of any faults in temperature sensor 1 or 2
     * of Dual Temperature Sensor SM */
    IMX623_DUALTEMP3_ERROR,
    /** 1-bit data errors detected by SRAM ECC SM */
    IMX623_SRAMECC1_ERROR,
    /** 2-bit data errors detected by SRAM ECC SM */
    IMX623_SRAMECC2_ERROR,
    /** 1-bit address errors detected by SRAM ECC SM */
    IMX623_SRAMECC3_ERROR,
    /** OTP CRC SM result of any data errors in the system area */
    IMX623_OTPCRC1_ERROR,
    /** OTP CRC SM result of any data errors in the SPSHD offset area */
    IMX623_OTPCRC2_ERROR,
    /** OTP CRC SM result of any data errors in the SPSHD gain area */
    IMX623_OTPCRC3_ERROR,
    /** OTP CRC SM result of any data errors in the STC area */
    IMX623_OTPCRC4_ERROR,
    /** OTP CRC SM result of any data errors in the user BIST area */
    IMX623_OTPCRC5_ERROR,
    /** OTP CRC SM result of any data errors in the user-specified area */
    IMX623_OTPCRC6_ERROR,
    /** OTP CRC SM result of any data errors in the security area */
    IMX623_OTPCRC7_ERROR,
    /** 1-bit data errors detected by ROM ECC SM */
    IMX623_ROMECC1_ERROR,
    /** 2-bit data errors detected by ROM ECC SM */
    IMX623_ROMECC2_ERROR,
    /** 1-bit address errors detected by ROM ECC SM */
    IMX623_ROMECC3_ERROR,
    /** Logic Built-In-Self-Test SM error */
    IMX623_LBIST_ERROR,
    /** Memory Built-In-Self-Test SM error */
    IMX623_MBIST_ERROR,
    /** Analog Build-In-Self-Test SM error */
    IMX623_ABIST_ERROR,
    /** Data Path Test SM error */
    IMX623_DATAPATH_ERROR,
    /** Register Monitor SM error */
    IMX623_REGMON_ERROR,
    /** Internal Bus Monitor SM error */
    IMX623_INTBUSMON_ERROR,
    /** Row Column ID Check SM error */
    IMX623_ROWCOLIDCHECK_ERROR,
    /** Voltage abnormalities detected by Power Monitor
     * SM(Power Monitor 0 (VDDH)) */
    IMX623_POWERMON1_ERROR,
    /** Voltage abnormalities detected by Power Monitor
     * SM(Power Monitor 1 (VDDM)) */
    IMX623_POWERMON2_ERROR,
    /** voltage abnormalities detected by Power Monitor
     * SM(Power Monitor 2 (VDDL)) */
    IMX623_POWERMON3_ERROR,
    /** Dual Lockstep SM error */
    IMX623_DUALLOCKSTEP_ERROR,
    /** Watch Dog Timer SM error */
    IMX623_WDT_ERROR,
    /** 1-bit errors detected by Flash Check(FW) SM */
    IMX623_FLASHCHECK1_ERROR,
    /** 2-bit errors detected by Flash Check(FW) SM */
    IMX623_FLASHCHECK2_ERROR,
    /** DMAC Monitor SM error */
    IMX623_DMACMON_ERROR,
    /** Power Switch Montior SM error */
    IMX623_POWSWITCHMON_ERROR,
    /** CCC Peeling Check SM error */
    IMX623_CCCPEELCHECK_ERROR,
    /** Memory Protection SM error */
    IMX623_MEMPROTECT_ERROR,
    /** OTP CRC SM result of any data errors in the authentication area */
    IMX623_OTPCRC8_ERROR,
    /** OTP CRC SM result of any data errors in the
     * sensor's system configuration settings area */
    IMX623_OTPCRC9_ERROR,
    /** OTP CRC SM result of any data errors in the
     * sensor adjustment results area */
    IMX623_OTPCRC10_ERROR,
    /** FSync Monitor SM error */
    IMX623_FSYNCMON_ERROR,
    /** Flip Information Error - Triggered when Flip information  (horizontal and vertical)
     *  provided by frame's front embedded data does not match the requested flip settings. */
    IMX623_FLIP_ERROR,
    /** Max/Invalid SM error */
    IMX623_MAX_ERROR,
} IMX623CustomErrType;

/**
 * @brief Describes the Custom Error Information needed by the application.
 */
typedef struct {
    /** Bit field indicating the error flags set, indexed by @ref IMX623CustomErrType
     * enum
     * Valid range : [0, UINT64_MAX].
     */
    errorInfoTypeIMX623 errorInfo;
} IMX623CustomErrInfo;

#ifdef __cplusplus
} /** end of extern C */
} /** end of namespace nvsipl */
#endif

#endif /** IMX623_CUSTOMDATA_H */
