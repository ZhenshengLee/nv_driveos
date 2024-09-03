/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef MAX96717F_CUSTOMDATA_H
#define MAX96717F_CUSTOMDATA_H

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

#define MAX96717F_MFP_STATUS_ERR  0x00U
#define MAX96717F_MFP_STATUS_OK   0x01U

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
namespace nvsipl {
extern "C" {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif

typedef enum {
    /* MAX96717F SM1 Bist for serial link */
    SM01_BIST_FOR_SERIAL_LINK,
    /* MAX96717F SM2 Line fault detection on serializer output */
    SM02_LINE_FAULT_DETECTION_ON_SERIALIZER_OUTPUT,
    /* MAX96717F SM3 CRC for control channel */
    SM03_CRC_FOR_CONTROL_CHANNEL,
    /* MAX96717F SM4 CRC for forward channel */
    SM04_CRC_FOR_FORWARD_CHANNEL,
    /* MAX96717F SM5 Error correction through retransmission for control channel */
    SM05_ERROR_CORRECTION_THROUGH_RETRANSMISSION,
    /* MAX96717F SM6 Lock indicator for forward channel */
    SM06_LOCK_INDICATOR_OF_FORWARD_CHANNEL,
    /* MAX96717F SM7 ERRB indicator for forward channel */
    SM07_ERRB_INDICATOR_OF_FORWARD_CHANNEL,
    /* MAX96717F SM8 Lock indicator for reverse channel */
    SM08_LOCK_INDICATOR_OF_REVERSE_CHANNEL,
    /* MAX96717F SM9 ERRB indicator for reverse channel */
    SM09_ERRB_INDICATOR_OF_REVERSE_CHANNEL,
    /* MAX96717F SM10 Error generator */
    SM10_ERROR_GENERATOR,
    /* MAX96717F SM11 Video sequence number */
    SM11_VIDEO_SEQUENCE_NUMBER,
    /* MAX96717F SM12 Acknowledge of UART or I2C */
    SM12_ACKNOWLEDGE_OF_UART_I2C,
    /* MAX96717F SM14 Eye opening monitor */
    SM14_EYE_OPENING_MONITOR,
    /* MAX96717F SM15 Voltage measurement and monitoring */
    SM15_VOLTAGE_MESUREMENT_AND_MONITORING,
    /* MAX96717F SM16 Evaluate CSI-2 ECC/CRC */
    SM16_EVALUATE_CSI2_ECC_CRC,
    /* MAX96717F SM17 FIFO Overflow detection */
    SM17_FIFO_OVERFLOW_DETECTION,
    /* MAX96717F SM18 Logic BIST */
    SM18_LOGIC_BIST,
    /* MAX96717F SM19 Memory BIST */
    SM19_MEMORY_BIST,
    /* MAX96717F SM20 End to end CRC */
    SM20_END_TO_END_CRC_DURING_VIDEO_TRANSMISSION,
    /* MAX96717F SM21 CRC on UART and I2C transactions */
    SM21_CRC_ON_I2C_UART_TRANSACTIONS,
    /* MAX96717F SM22 Message counter */
    SM22_MESSAGE_COUNTER,
    /* MAX96717F SM23 GPO Readback */
    SM23_GPO_READBACK,
    /* MAX96717F SM24 Overtemperature warning */
    SM24_OVERTEMPERATURE_WARNING,
    /* MAX96717F SM25 ADC BIST*/
    SM25_ADC_BIST,
    /* MAX96717F SM26 Memory ECC */
    SM26_MEMORY_ECC,
    /* MAX96717F SM27 Configuration register CRC */
    SM27_CONFIGURATION_REGISTER_CRC,
    /* MAX96717F SM28 Self test of temperature sensor */
    SM28_SELF_TEST_OF_TEMPERATURE_SENSOR,
    /* MAX96717F SM29 CRC of non-volatile memory */
    SM29_CRC_OF_NONVOLATILE_MEMORY,
    /* MAX96717F SM30 Self clearing unlock register */
    SM30_SELF_CLEARING_UNLOCK_REGISTER,
    /* MAX96717F SM31 DPLL lock indicator*/
    SM31_DPLL_LOCK_INDICATOR,
    /* MAX96717F SM32 Video detect */
    SM32_VIDEO_DETECT,
    /* MAX96717F SM33 9b/10b illegal symbol check */
    SM33_ILLEGAL_SYMBOL_CHECKING_RUNNING_DISPARITY,
    /* MAX96717F SM34 Packet counter for GPIO, I2C, SPI and UART */
    SM34_PACKET_COUNTER_FOR_GPIO_I2C_SPI_UART,
    /* MAX96717F SM35 I2C Timeout verification on I2C */
    SM35_I2C_TIMEOUT_VERIFICATION_ON_I2C,
    /* MAX96717F SM36 Info frame sent periodically */
    SM36_INFO_FRAME_IS_SENT_PERIODICALLY,
    /* MAX96717F SM37 CRC on configuration data for data retention memory */
    SM37_CRC_ON_CONFIGURATION_DATA,
    /* MAX96717F SM40 GPIO open detection */
    SM40_GPIO_OPEN_DETECTION,
    /* MAX96717F Max SM number */
    SM_LIST_MAX_NUM,
//coverity[misra_c_2012_rule_2_3_violation] # False positive
} MAX96717FSMErrIndex;

static_assert(sizeof(MAX96717FSMErrIndex) != 0U, "Required API"); // Squelch Rule 2.3

/**
 * @brief Describes the custom error information for OVERFLOW bit,
 *        TUN_FIFO_OVERFLOW bit and PCLKDET bit.
 */
typedef struct {
    /**
     * @brief Holds the status of OVERFLOW bit. If set to 1U, the Video transmit
     *        FIFO has overflowed. No overflow if set to 0U.
     */
    uint8_t overflow_status;
    /**
     * @brief Holds the status of PCLKDET bit. If set to 1U, the Video Pixel Clock
     *        has been detected. No clock detection if set to 0U.
     */
    uint8_t pclkdet_status;
    /**
     * @brief Holds the status of TUN_FIFO_OVERFLOW bit. If set to 1U, the Tunnel
     *        FIFO has overflowed. No overflow if set to 0U.
     */
    uint8_t tun_fifo_overflow_status;
} MAX96717FVideoStatus;

static_assert(sizeof(MAX96717FVideoStatus) != 0U, "Required API"); // Squelch Rule 2.3

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
} /* end of extern C*/
} /* end of namespace nvsipl */
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif

#endif /* end of MAX96717F_CUSTOMDATA_H */