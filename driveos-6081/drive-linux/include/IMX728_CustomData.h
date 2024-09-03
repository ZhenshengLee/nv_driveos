/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * All information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef IMX728_CUSTOMDATA_H
#define IMX728_CUSTOMDATA_H

#ifdef __cplusplus
#include <cstdbool>
#else
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif


typedef uint64_t errorInfoTypeIMX728;

/**
 * @brief Describes the Custom Embedded Data needed by the application.
 */
typedef struct {
    /** Bit field indicating the error flags set, indexed by
     *  IMX728CustomErrType enum */
    errorInfoTypeIMX728 customEmbeddedDataError;
} IMX728CustomEmbeddedData;

/**
 * @brief Describes the IMX728 Safety mechanism error types.
 */
typedef enum {
        /** Clock Monitor SM error */
        IMX728_CLKMON_ERROR = 0U,
        /** Sequence Monitor SM error */
        IMX728_SEQMON_ERROR = 1U,
        /** Control Flow Monitor SM  error */
        IMX728_CTRLFLOWMON_ERROR = 2U,
        /** Sync Monitor SM  error */
        IMX728_SYNCMON_ERROR = 3U,
        /** Outside threshold range fault for temperature sensor 1 of
         *  DT sensor SM */
        IMX728_DUALTEMP1_ERROR = 4U,
        /** Outside threshold range fault for temperature sensor 2 of
         *  DT sensor SM */
        IMX728_DUALTEMP2_ERROR = 5U,
        /** Outside threshold range fault for temperature sensor 1 or
         *  2 of DT sensor SM */
        IMX728_DUALTEMP3_ERROR = 6U,
        /** 1-bit data errors detected by SRAM ECC SM */
        IMX728_SRAMECC1_ERROR = 7U,
        /** 2-bit data errors detected by SRAM ECC SM */
        IMX728_SRAMECC2_ERROR = 8U,
        /** 1-bit address errors detected by SRAM ECC SM */
        IMX728_SRAMECC3_ERROR = 9U,
        /** OTP CRC SM result of any data errors in the system area */
        IMX728_OTPCRC1_ERROR = 10U,
        /** OTP CRC SM result of any data errors in the authentication
         *  system area */
        IMX728_OTPCRC2_ERROR = 11U,
        /** OTP CRC SM result of any data errors in the user STC
         *  area */
        IMX728_OTPCRC3_ERROR = 12U,
        /** OTP CRC SM result of any data errors in the user SPS
         *  gain area */
        IMX728_OTPCRC4_ERROR = 13U,
        /** OTP CRC SM result of any data errors in the user
         *  BIST area */
        IMX728_OTPCRC5_ERROR = 14U,
        /** OTP CRC SM result of any data errors in the user-specified
         *  area */
        IMX728_OTPCRC6_ERROR = 15U,
        /** OTP CRC SM result of any data errors in the security area */
        IMX728_OTPCRC7_ERROR = 16U,
        /*  1-bit data errors detected by ROM ECC SM */
        IMX728_ROMECC1_ERROR = 17U,
        /** 2-bit data errors detected by ROM ECC SM */
        IMX728_ROMECC2_ERROR = 18U,
        /** 1-bit address errors detected by ROM ECC SM */
        IMX728_ROMECC3_ERROR = 19U,
        /** Logic BIST (Built-In-Self-Test) SM error */
        IMX728_LBIST_ERROR = 20U,
        /** Memory BIST SM error */
        IMX728_MBIST_ERROR = 21U,
        /** Analog BIST SM error for all internal tests (1-11) combined */
        IMX728_ABIST_ERROR = 22U,
        /** Data Path Test SM error */
        IMX728_DATAPATH_ERROR = 23U,
        /** Register Monitor SM error */
        IMX728_REGMON_ERROR = 24U,
        /** Internal Bus Monitor SM error */
        IMX728_INTBUSMON_ERROR = 25U,
        /** Row Column ID Check SM error */
        IMX728_ROWCOLIDCHECK_ERROR = 26U,
        /** Voltage abnormalities detected by Power Monitor (0)
         *  SM for VDDH */
        IMX728_POWERMON1_ERROR = 27U,
        /** Voltage abnormalities detected by Power Monitor (1)
         *  SM for VDDH */
        IMX728_POWERMON2_ERROR = 28U,
        /** Voltage abnormalities detected by Power Monitor (2)
         *  SM for VDDH */
        IMX728_POWERMON3_ERROR = 29U,
        /** Voltage abnormalities detected by Power Monitor (3)
         *  SM for VDDH */
        IMX728_POWERMON4_ERROR = 30U,
        /** Voltage abnormalities detected by Power Monitor (4)
         *  SM for VDDH */
        IMX728_POWERMON5_ERROR = 31U,
        /** Any faults detected by Dual Lock Step SM */
        IMX728_DUALLOCKSTEP_ERROR = 32U,
        /** Watch Dog Timer SM error */
        IMX728_WDT_ERROR = 33U,
        /** 1-bit errors detected by Flash Check(FW) SM */
        IMX728_FLASHCHECK1_ERROR = 34U,
        /** 2-bit errors detected by Flash Check(FW) SM */
        IMX728_FLASHCHECK2_ERROR = 35U,
        /** DMAC Monitor SM error */
        IMX728_DMACMON_ERROR = 36U,
        /** Power Switch Montior SM error */
        IMX728_POWSWITCHMON_ERROR = 37U,
        /** CCC Peeling Check SM error */
        IMX728_CCCPEELCHECK_ERROR = 38U,
        /** Memory Protection SM error */
        IMX728_MEMPROTECT_ERROR = 39U,
        /** Application Lock SM error */
        IMX728_APPLOCK_ERROR = 40U,
        /** Data Range Check SM error */
        IMX728_DATARANGECHK_ERROR = 41U,
        /** FSync Monitor SM Error */
        IMX728_FSYNCMON1_ERROR = 42U,
        /** FSync Monitor SM Range Check Error */
        IMX728_FSYNCMON2_ERROR = 43U,
        /** FSync Monitor SM Timeout Error */
        IMX728_FSYNCMON3_ERROR = 44U,
        /** Flip Information Error - Triggered when Flip information (horizontal and vertical)
         *  provided by frame's front embedded data does not match the requested flip settings. */
        IMX728_FLIP_ERROR = 45U,
        /** Max/Invalid SM error = 46;
         * Keep this enum at last, change value to max; */
        IMX728_MAX_ERROR = 46U,
} IMX728CustomErrType;

/**
 * @brief Describes the Custom Error Information needed
 *        by the application.
 */
typedef struct {
    /** Bit field indicating the error flags set, indexed
     *  by IMX728CustomErrType enum. */
    errorInfoTypeIMX728 errorInfo;
} IMX728CustomErrInfo;

#ifdef __cplusplus
} /** end of extern C */
#endif

#endif /** IMX728_CUSTOMDATA_H */


