/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef TPS650332_CUSTOMDATA_H
#define TPS650332_CUSTOMDATA_H

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
#include <cstdint>
#include <cassert>
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#else
// required for C, defines static_assert()
#include <assert.h>
#include <stdint.h>
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
namespace nvsipl {
extern "C" {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif

/* Hold error information for TPS650332 */
typedef uint64_t errorInfoTypeTPS650332;

/**
 * @brief error type for TPS650332
 */
typedef enum {
    /** nINT error */
    TPS650332_NINT_ERROR = 0U,
    /** I2C CRC error */
    TPS650332_I2C_CRC_ERROR,
    /** I2C Memory address error */
    TPS650332_I2C_MEMORY_ADDR_ERROR,
    /** OVD1 error */
    TPS650332_OVD1_ERROR,
    /** UVD1 error */
    TPS650332_UVD1_ERROR,
    /** OCP1 error */
    TPS650332_OCP1_ERROR,
    /** OTD1 error */
    TPS650332_OTD1_ERROR,
    /** OCP2 error */
    TPS650332_OCP2_ERROR,
    /** OTD2 error */
    TPS650332_OTD2_ERROR,
    /** OVD3 error */
    TPS650332_OVD3_ERROR,
    /** UVD3 error */
    TPS650332_UVD3_ERROR,
    /** OCP3 error */
    TPS650332_OCP3_ERROR,
    /** OTD3 error */
    TPS650332_OTD3_ERROR,
    /** OVD4 error */
    TPS650332_OVD4_ERROR,
    /** UVD4 error */
    TPS650332_UVD4_ERROR,
    /** OCP4 error */
    TPS650332_OCP4_ERROR,
    /** OTD4 error */
    TPS650332_OTD4_ERROR,
    /** ABIST_BUCK1_UV_ACK error */
    TPS650332_ABIST_BUCK1_UV_ERROR,
    /** ABIST_BUCK1_OV_ACK erorr */
    TPS650332_ABIST_BUCK1_OV_ERROR,
    /** ABIST_BUCK1_OVP_ACK error */
    TPS650332_ABIST_BUCK1_OVP_ERROR,
    /** ABIST_BUCK2_UV_ACK error */
    TPS650332_ABIST_BUCK2_UV_ERROR,
    /** ABIST_BUCK2_OV_ACK erorr */
    TPS650332_ABIST_BUCK2_OV_ERROR,
    /** ABIST_BUCK2_OVP_ACK error */
    TPS650332_ABIST_BUCK2_OVP_ERROR,
    /** ABIST_BUCK3_UV_ACK error */
    TPS650332_ABIST_BUCK3_UV_ERROR,
    /** ABIST_BUCK3_OV_ACK erorr */
    TPS650332_ABIST_BUCK3_OV_ERROR,
    /** ABIST_BUCK3_OVP_ACK error */
    TPS650332_ABIST_BUCK3_OVP_ERROR,
    /** ABIST_LDO_UV_ACK error */
    TPS650332_ABIST_LDO_UV_ERROR,
    /** ABIST_LDO_OV_ACK erorr */
    TPS650332_ABIST_LDO_OV_ERROR,
    /** ABIST_LDO_OVP_ACK error */
    TPS650332_ABIST_LDO_OVP_ERROR,
    /** ABIST_GNDLOSSSS_ERR_ACK error */
    TPS650332_ABIST_GNDLOSS_ERROR,
    /** ABIST_SYSTEM_TSHUT_ACK error */
    TPS650332_ABIST_SYSTEM_TSHUT_ERROR,
    /** CFG_REG_CRC_ERR_ACK error */
    TPS650332_CFG_REG_CRC_ERROR,
} TPS650332CustomErrType;

static_assert(sizeof(TPS650332CustomErrType) != 0U, "Required API"); // Squelch Rule 2.3

/**
 * @brief Holds custom error type for TPS650332.
 */
typedef struct {
    /** Bit field indicating the error flags set, indexed by TPS650332CustomErrType
     * enum */
    errorInfoTypeTPS650332 errorInfo;
} TPS650332CustomErrInfo;

static_assert(sizeof(TPS650332CustomErrInfo) != 0U, "Required API"); // Squelch Rule 2.3

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
} /* end of extern C*/
} /* end of namespace nvsipl */
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif

#endif /* end of TPS650332_CUSTOMDATA_H */