/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */
#ifndef CDI_MAX96717F_H
#define CDI_MAX96717F_H

#include "MAX96717F_CustomData.h"
#include "devblk_cdi_i2c.h"

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
#include <cstdbool>
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#else
#include <stdbool.h>
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif

#define SER_MFP0        (0U)    /* MFP0 pin index */
#define SER_MFP3        (3U)    /* MFP3 pin index */
#define SER_MFP4        (4U)    /* MFP4 pin index */
#define SER_MFP5        (5U)    /* MFP5 pin index */
#define SER_MFP6        (6U)    /* MFP6 pin index */
#define SER_MFP7        (7U)    /* MFP7 pin index */
#define SER_MFP8        (8U)    /* MFP8 pin index */
#define SER_MFP10       (10U)   /* MFP10 pin index */
#define REG_MFP5        0x02CDU /* GPIO_A register for MFP5 */
#define REG_MFP6        0x02D0U /* GPIO_A register for MFP6 */
#define REG_MFP7        0x02D3U /* GPIO_A register for MFP7 */
#define REG_REGCRC8     0x1D40U /* Skip Register 0 LSB (from GMSL2/3 Register CRC App Note) */
#define REG_I2C_4       0x0044U /* I2C_4 */
#define REG_I2C_5       0x0045U /* I2C_5 */
#define REG_VIDEO_TX0   0x0110U /* VIDEO_TX0 */
#define REG_VIDEO_TX2   0x0112U /* VIDEO_TX2 */
#define REG_VTX1        0x024FU /* VTX1 */
#define REG_GPIO0_A     0x02BEU /* GPIO_A register for MFP0 */
#define REG_GPIO8_C     0x02D8U /* GPIO_C register for MFP8 */
#define REG_REGADCBIST0 0x1D28U /* REGADCBIST0 */

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
namespace nvsipl {
extern "C" {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif

typedef enum {
    /* This type must be contiguous and start from 0 */
    CDI_WRITE_PARAM_CMD_MAX96717F_INVALID,
    CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_A,
    CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B,
    CDI_WRITE_PARAM_CMD_MAX96717F_SET_DEVICE_ADDRESS,
    CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT,
    CDI_WRITE_PARAM_CMD_MAX96717F_SET_FSYNC_GPIO,
    CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_REF_CLOCK,
    CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_VIDEO_PIPELINES,
    CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_PHY,
    CDI_WRITE_PARAM_CMD_MAX96717F_GENERATE_CLOCK,
    CDI_WRITE_PARAM_CMD_MAX96717F_SET_DEBUG_REGS,
    CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_PCLK_PIO_SLEW,
    CDI_WRITE_PARAM_CMD_MAX96717F_GPIO_FORWARD,
    CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_SM_REPORTS,
    CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_SM_REPORTS_SKIP_REFGEN_CHECK,
    CDI_WRITE_PARAM_CMD_MAX96717F_UNSET_ERRB_TX,
    CDI_WRITE_PARAM_CMD_MAX96717F_SET_ERRB_TX,
    CDI_WRITE_PARAM_CMD_MAX96717F_SET_ERRB_TX_ID,
    CDI_WRITE_PARAM_CMD_MAX96717F_ASSERT_ERRB,
    CDI_WRITE_PARAM_CMD_MAX96717F_NUM,
} WriteParametersCmdMAX96717F;

typedef enum {
    CDI_MAX96717F_DATA_TYPE_INVALID,
    CDI_MAX96717F_DATA_TYPE_RAW8,
    CDI_MAX96717F_DATA_TYPE_RAW10,
    CDI_MAX96717F_DATA_TYPE_RAW12
} DataTypeMAX96717F;

typedef enum {
    CDI_MAX96717F_GPIO_TYPE_INVALID,
    CDI_MAX96717F_GPIO_TYPE_MFP0,
    CDI_MAX96717F_GPIO_TYPE_MFP1,
    CDI_MAX96717F_GPIO_TYPE_MFP2,
    CDI_MAX96717F_GPIO_TYPE_MFP3,
    CDI_MAX96717F_GPIO_TYPE_MFP4,
    CDI_MAX96717F_GPIO_TYPE_MFP5,
    CDI_MAX96717F_GPIO_TYPE_MFP6,
    CDI_MAX96717F_GPIO_TYPE_MFP7,
    CDI_MAX96717F_GPIO_TYPE_MFP8,
    CDI_MAX96717F_GPIO_TYPE_NUM,
} GPIOTypeMAX96717F;

typedef struct {
    uint8_t phy0_d0;
    uint8_t phy0_d1;
    uint8_t phy1_d0;
    uint8_t phy1_d1;
    uint8_t phy2_d0;
    uint8_t phy2_d1;
    uint8_t phy3_d0;
    uint8_t phy3_d1;
    bool enableMapping;
} phyMapMAX96717F;

typedef struct {
    uint8_t phy1_d0;               /* lane0pol */
    uint8_t phy1_d1;               /* lane1pol */
    uint8_t phy1_clk;              /* clk1pol */
    uint8_t phy2_d0;               /* lane2pol */
    uint8_t phy2_d1;               /* lane3pol */
    uint8_t phy2_clk;              /* clk2pol */
    bool setPolarity;
} phyPolarityMAX96717F;

typedef struct {
    struct {
        uint8_t source;               /* 7 bit I2C address */
        uint8_t destination;          /* 7 bit I2C address */
    } Translator;
    struct {
        uint8_t address;              /* 7 bit I2C address */
    } DeviceAddress;
    struct {
        GPIOTypeMAX96717F gpioInd;    /* Must be 0-8 for MFP0-MFP8 pins */
        bool level;                   /* level = true to set logic high */
    } GPIOOutp;
    struct {
        GPIOTypeMAX96717F gpioInd;    /* Must be 0-8 for MFP0-MFP8 pins */
        uint8_t rxID;                 /* GPIO Rx ID. Must match with deserializer val */
    } FSyncGPIO;
    struct {
        GPIOTypeMAX96717F gpioInd;    /* Must be 0-8 for MFP0-MFP8 pins */
        bool enableRClk;              /* Enable RCLK output on PCLKOUT pin */
    } RefClkGPIO;
    struct {
        DataTypeMAX96717F dataType;   /* Sensor data type for pixel data */
        bool embDataType;             /* Set to true if emb data has emb data type */
    } ConfigVideoPipeline;
    struct {
        phyMapMAX96717F mapping;
        phyPolarityMAX96717F polarity;
        uint8_t numDataLanes;
    } ConfigPhy;
    struct {
        uint8_t freq;                 /* Generate Clock Rate in Mhz */
    } ClockRate;
    struct {
        uint8_t srcGpio;              /* Serializer GPIO number as the input */
        uint8_t dstGpio;              /* Destination GPIO number as the output */
    } GPIOForward;
    struct {
        uint8_t dstGpio;            /* Destination GPIO number at deserializer */
    } GPIOErrb;
    bool assertERRB;                /* Set to true to assert ERRB */
} ReadWriteParamsMAX96717F;

/**
 * @brief defines the MAX96717F SMs information data.
 */
typedef struct {
    /** the number SM name. */
    const char_t*        name;
    /** SM status register address */
    uint16_t           addr;
    /** Indicate the SM status bit */
    uint8_t            bit;
    /** Indicate SM index */
    uint8_t      index;
} MAX96717FSMErrInfoRegSet;

/**
 * @brief defines the MAX96717F Clear SM information data.
 */
typedef struct {
    /** SM status register value */
    uint8_t         statusVal;
    /** SM clear flag register address */
    uint16_t        clrAddr;
    /** SM clear bit */
    uint8_t         clrBit;
    /** Read or Write clear permission */
    bool            writeClr;
    /** clear bit2 valid flag*/
    bool            clrBit2Valid;
    /** SM clear bit 2 */
    uint8_t         clrBit2;
    /** SM index */
    uint8_t         index;
} SMErrClearRegSet;

/**
 * @brief SM result report data
 */
typedef struct {
    /** SM index */
    uint8_t index;
    /** Status register value for reporting */
    uint8_t regVal;
} SMErrInfoResult;

/**
 * @brief Describes the MAX96717F serializer error information captured by the driver.
 */
typedef struct {
#if FUSA_CDD_NV
    /** Indicates the value of \b REFGEN_UNLOCKED bit at 0x001B[7] interrupt
     *  register in MAX96717F serializer. */
    uint8_t  refgen_unlocked_flag;
    /** Indicates the value of \b IDLE_ERR_FLAG bit at 0x001B[2] interrupt
     *  register in MAX96717F serializer. */
    uint8_t  idle_err_flag;
    /** Indicates the value of \b DEC_ERR_FLAG_A bit at 0x001B[0] interrupt
     *  register in MAX96717F serializer. */
    uint8_t  dec_err_flag_a;
    /** Indicates the value of \b VREG_OV_FLAG bit at 0x001D[7] interrupt register
     *  in the MAX96717F serializer. */
    uint8_t  vreg_ov_flag;
    /** Indicates the value of \b EOM_ERR_FLAG bit at 0x001D[6] interrupt register
     *  in the MAX96717F serializer. */
    uint8_t  eom_err_flag;
    /** Indicates the value of \b VDD_OV_FLAG bit at 0x001D[5] interrupt register
     *  in the MAX96717F serializer. */
    uint8_t  vdd_ov_flag;
    /** Indicates the value of \b VDD18_OV_FLAG bit at 0x001D[4] interrupt register
     *  in the MAX96717F serializer. */
    uint8_t  vdd18_ov_flag;
    /** Indicates the value of \b MAX_RT_FLAG bit at 0x001D[3] interrupt register
     *  in the MAX96717F serializer. */
    uint8_t  max_rt_flag;
    /** Indicates the value of \b RT_CNT_FLAG bit at 0x001D[2] interrupt register
     * in teh MAX96717F serializer. */
    uint8_t  rt_cnt_flag;
    /** Indicates the value of \b VDDCMP_INT_FLAG bit at 0x001F[7] interrupt register
     *  in the MAX96717F serializer. */
    uint8_t  vddcmp_int_flag;
    /** Indicates the value of \b PORZ_INT_FLAG bit at 0x001F[6] interrupt register
     *  in the MAX96717F serializer. */
    uint8_t  porz_int_flag;
    /** Indicates the value of \b VDDBAD_INT_FLAG bit at 0x001F[5] interrupt register
     *  in the MAX96717F serializer. */
    uint8_t  vddbad_int_flag;
    /** Indicates the value of \b EFUSE_CRC_ERR_FLAG bit at 0x001F[4] interrupt register
     *  in the MAX96717F serializer. */
    uint8_t  efuse_crc_err_flag;
    /** Indicates the value of \b ADC_INT_FLAG bit at 0x001F[2] interrupt register
     *  in the MAX96717F serializer. */
    uint8_t  adc_int_flag;
    /** Indicates the value of \b MIPI_ERR_FLAG bit at 0x001F[0] interrupt register
     *  in the MAX96717F serializer. */
    uint8_t  mipi_err_flag;
#else
    /** Indicates the value of \b INTR3 interrupt register in the MAX96717F serializer. */
    uint8_t  intr3Reg;
    /** Indicates the value of \b INTR5 interrupt register in the MAX96717F serializer. */
    uint8_t  intr5Reg;
    /** Indicates the value of \b INTR7 interrupt register in the MAX96717F serializer. */
    uint8_t  intr7Reg;
#endif
    /** Indicates max retransmission error in serializer path.
     *  bit7 indicates MAX_RT_ERROR flag, bits[6:0] indicates retransmission count.
     *  In FUSA_CDD_NV code, this indicates only the MAX_RT_ERR flag at 0x008F[7], which is
     *  ARQ2 register for CC */
    uint8_t  maxRTErrSer;
    /** Indicates max retransmission error in GPIO path.
     *  bit7 indicates MAX_RT_ERROR flag, bits[6:0] indicates retransmission count.
     *  In FUSA_CDD_NV code, this indicates only the MAX_RT_ERR flag at 0x0097[7], which is
     *  ARQ2 register for GPIO */
    uint8_t  maxRTErrGPIO;
    /** Indicates max retransmission error in I2CX path.
     *  bit7 indicates MAX_RT_ERROR flag, bits[6:0] indicates retransmission count.
     *  In FUSA_CDD_NV code, this indicates only the MAX_RT_ERR flag at 0x00A7[7], which is
     *  ARQ2 register for I2C_X */
    uint8_t  maxRTErrI2CX;
    /** Indicates max retransmission error in I2CY path.
     *  bit7 indicates MAX_RT_ERROR flag, bits[6:0] indicates retransmission count.
     *  In FUSA_CDD_NV code, this indicates only the MAX_RT_ERR flag at 0x00AF[7], which is
     *  ARQ2 register for I2C_Y */
    uint8_t  maxRTErrI2CY;
    /** Indicates MIPI PHY1 high speed error register value \b phy1_hs_err in
     *  the MAX96717F serializer */
    uint8_t  phy1Err;
    /** Indicates MIPI PHY2 high speed error register value \b phy2_hs_err in
     *  the MAX96717F serializer */
    uint8_t  phy2Err;
    /** Indicates CSI2 Ctrl1 status register value in \b ctrl1_csi_err_l,h bits[10:0] in
     *  the MAX96717F serializer */
    uint16_t ctrl1CSIErr;
    /* Indicates the number of idle word errors detected in register 0x0024[7:0] */
    uint8_t  idleErr;
    /* Indicates the number of decoding errors detected in register 0x0022[7:0] */
    uint8_t  decodeErr;
    /* Indicates each ADC interrupt flag */
    uint8_t  adcINTR0;
    /* Indicates each ADC channel hi limit monitor flag */
    uint8_t  adcINTR1;
    /* Indicates each ADC channel lo limit monitor flag */
    uint8_t  adcINTR2;
#if FUSA_CDD_NV
    /** Indicates the value of \b MEM_ECC_ERR2_INT bit at 0x1D13[5] register in the
     *  MAX96717F serializer. */
    uint8_t  mem_ecc_err2_int_flag;
    /** Indicates the value of \b MEM_ECC_ERR1_INT bit at 0x1D13[4] register in the
     *  MAX96717F serializer. */
    uint8_t  mem_ecc_err1_int_flag;
    /** Indicates the value of \b REG_CRC_ERR_FLAG bit at 0x1D13[0] register in the
     *  MAX96717F serializer. */
    uint8_t  reg_crc_err_flag;
    /** Indicates the value of \b TMON_ERR_IF bit in  the 0x0513[1] register in the
     *  MAX96717F serializer. */
    uint8_t  tmon_err_int_flag;
    /** Indicates the value of \b REFGEN_LOCKED bit at 0x03F0[7] registrer in the
     *  MAX96717F serializer. */
    uint8_t  refgen_locked_flag;
    /** Indicates the value of \b TUN_FIFO_OVERFLOW bit at 0x0380[0] register in the
     *  MAX96717F serializer. */
    uint8_t tun_fifo_overflow_flag;
    /** Indicates the value of \b LOCKED bit at 0x0013[3] register in the MAX96717F
     *  serializer. */
    uint8_t locked_flag;
#endif
//coverity[misra_c_2012_rule_2_3_violation] # False positive
} MAX96717FErrorInfo;

DevBlkCDIDeviceDriver *GetMAX96717FDriver(void);

NvMediaStatus
MAX96717FCheckPresence(
    DevBlkCDIDevice const* handle);

NvMediaStatus
MAX96717FSetDefaults(
    DevBlkCDIDevice const* handle);

NvMediaStatus
MAX96717FReadRegisters(
    DevBlkCDIDevice const* handle,
    uint16_t registerNum,
    uint16_t dataLength,
    uint8_t *dataBuff);

/**
 * @brief Read-verify a block of registers
 *
 * @param[in]  handle Device Handle for DevBlkCDIDevice
 * @param[in]  registerNum Start register number
 * @param[in]  dataLength Number of registers to be read
 * @param[out] dataBuff Buffer to store read data
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
NvMediaStatus
MAX96717FReadRegistersVerify(
    DevBlkCDIDevice const * const handle,
    uint16_t const registerNum,
    uint16_t const dataLength,
    uint8_t * const dataBuff);

NvMediaStatus
MAX96717FWriteRegisters(
    DevBlkCDIDevice const* handle,
    uint16_t registerNum,
    uint16_t dataLength,
    uint8_t const* dataBuff);

/**
 * @brief Write-verify a block of registers
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice
 * @param[in] registerNum Start register number
 * @param[in] dataLength Number of registers to be written
 * @param[in] dataBuff Buffer containing data
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
NvMediaStatus
MAX96717FWriteRegistersVerify(
    DevBlkCDIDevice const * const handle,
    uint16_t const registerNum,
    uint16_t const dataLength,
    uint8_t const * const dataBuff);

NvMediaStatus
MAX96717FWriteParameters(
    DevBlkCDIDevice const* handle,
    WriteParametersCmdMAX96717F parameterType,
    size_t parameterSize,
    void const* parameter);

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
NvMediaStatus
MAX96717FDumpRegisters(
    DevBlkCDIDevice const* handle);
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif /* !NV_IS_SAFETY */

/**
 * @brief Get MAX96717F's Eye-Open-Monitor SM value
 *
 * @param[in] handle   Device Handle for DevBlkCDIDevice
 * @param[out] eomValue EOM value.
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_BAD_PARAMETER Either handle or params is NULL
 * @retval NVMEDIA_STATUS_ERROR any other error.
 */
NvMediaStatus
MAX96717FGetEOMValue(
    DevBlkCDIDevice const* handle,
    uint8_t         *eomValue);

/**
 * @brief Get MAX96717F's temperature data
 *
 * MAX96717F has two temperature sensors.
 * This function returns two temperature sensor data.
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice
 * @param[out] tmon1 temperature sensor 1 reading in celsius.
 * @param[out] tmon2 temperature sensor 2 reading in celsius.
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_BAD_PARAMETER Either handle or params is NULL
 * @retval NVMEDIA_STATUS_ERROR any other error.
 */
NvMediaStatus
MAX96717FGetTemperature(
    DevBlkCDIDevice const* handle,
    float_t *tmon1,
    float_t *tmon2);

/**
 * @brief Check the SM status register-bit
 *
 * @param[in] handle    Device Handle for DevBlkCDIDevice.
 * @param[in] errIndex  Which SM needs to check the status.
 * @param[out] store    stored the SM status register value.
 *
 * @retval NVMEDIA_STATUS_OK    if checking to SM status is successful
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if one or more params was NULL or invalid.
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
NvMediaStatus
MAX96717FGetSMStatus(
    DevBlkCDIDevice const *handle,
    uint8_t const errIndex,
    uint8_t *store);

/**
 * @brief Writes the SM register
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice.
 * @param[in] index  Which SM needs to write the register.
 * @param[in] onOff  set to true or false.
 *
 * @retval NVMEDIA_STATUS_OK if the register setting is succeed.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if handle is NULL.
 * @retval NVMEDIA_STATUS_ERROR on any other error.
 */
NvMediaStatus
MAX96717FSetSMStatus(
    DevBlkCDIDevice const *handle,
    uint8_t const index,
    bool const onOff);

/**
 * @brief Read the safety mechanism error data.
 *
 * @param[in]  handle  Device Handle for DevBlkCDIDevice.
 * @param[out] errInfo Error information pointer.
 *
 * @retval NVMEDIA_STATUS_OK Call successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER One of the input param is NULL.
 * @retval NVMEDIA_STATUS_ERROR Any other error.
 */
NvMediaStatus
MAX96717FReadErrorData(
    DevBlkCDIDevice const* handle,
    MAX96717FErrorInfo *const errInfo);

/**
 * @brief Checking all SM status
 *
 * This function reports all SM fault status
 * if ERRB triggered by a fault.
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice.
 * @param[out] regSize valid buffer size
 * @param[out] buffer Payload buffer pointer
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_BAD_PARAMETER either handle or params is NULL
 * @retval NVMEDIA_STAUTS_ERROR on any other error
 */
NvMediaStatus
MAX96717FChkSMStatus(
    DevBlkCDIDevice const *handle,
    size_t *regSize,
    uint8_t * const buffer);

/**
 * @brief Gets the status of MFP pin of MAX96717F.
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice.
 * @param[in] regAddr MFP register address.
 * @param[out] state State of the MFP pin
 *             (range: MAX96717F_MFP_STATUS_OK - MAX96717F_MFP_STATUS_ERR).
 *
 * @retval NVMEDIA_STATUS_OK Success.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER Parameters are NULL.
 */
NvMediaStatus
MAX96717FGetMFPState(
    DevBlkCDIDevice const *handle,
    uint16_t regAddr,
    uint8_t *state);

/**
 * @brief Configure internal PCLK and enable pattern generator
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice.
 * @param[in] enable enable or disable config.
 *
 * @retval NVMEDIA_STATUS_OK Success.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER handle is NULL.
 * @retval NVMEDIA_STATUS_ERROR on any other error.
 */
NvMediaStatus
MAX96717FConfigPatternGenerator(
    DevBlkCDIDevice const *handle,
    bool enable);

/**
 * @brief Enable/Disable VPRBS generator
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice.
 * @param[in] enable enable or disable flag
 *
 * @retval NVMEDIA_STATUS_OK Success.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER handle is NULL.
 * @retval NVMEDIA_STATUS_ERROR on any other error.
 */
NvMediaStatus
MAX96717FVPRBSGenerator(
    DevBlkCDIDevice const *handle,
    bool enable);

/**
 * @brief Verify if PCLKDET is true.
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice.
 *
 * @retval NVMEDIA_STATUS_OK Success.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER handle is NULL.
 * @retval NVMEDIA_STATUS_ERROR on any other error.
 */
NvMediaStatus
MAX96717FVerifyPCLKDET(
    DevBlkCDIDevice const *handle);

/**
 * @brief Get OVERFLOW and PCLKDET bit status.
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice
 * @param[out] overflow_status Holds OVERFLOW bit status[valid range: 0-1]
 *             1U: Video transmit FIFO has overflowed
 *             0U: Video transmit FIFO has not overflowed
 * @param[out] pclkdet_status Holds PCLKDET bit status[valid range: 0-1]
 *             1U: Video Pixel Clock detected
 *             0U: Video Pixel Clock not detected
 * @param[out] tun_fifo_overflow_status Holds TUN_FIFO_OVERFLOW bit
 *             status[valid range: 0-1]
 *             1U: Tunnel FIFO overflow detected
 *             0U: Tunnel FIFO overflow not detected
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_BAD_PARAMETER handle or flagStatus is NULL
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
NvMediaStatus
MAX96717FGetVideoStatus(
    DevBlkCDIDevice const *handle,
    uint8_t *overflow_status,
    uint8_t *pclkdet_status,
    uint8_t *tun_fifo_overflow_status);

/**
 * @brief Enable/Disable ERRG
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice
 * @param[in] enableERRG true is enable / false is disable
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
NvMediaStatus
MAX96717FERRG(
    DevBlkCDIDevice const *handle,
    bool enableERRG);

/**
 * @brief Check if ERRB is set
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice
 * @param[out] errbStatus ERRB status [0,1]
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
NvMediaStatus
MAX96717FIsErrbSet(
    DevBlkCDIDevice const * const handle,
    uint8_t* const errbStatus);

/**
 * @brief Write an array of register values to serilaizer
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice
 * @param[in] regs register values to write to device
 * @param[in] length number of registers
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_ERROR and error
 */
NvMediaStatus
MAX96717FArrayWrite(
    DevBlkCDIDevice const * const handle,
    DevBlkCDII2CReg const * const regs,
    uint32_t const length);

/**
 * @brief Write-verify an array of register values to serilaizer
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice
 * @param[in] regs register values to write to device
 * @param[in] length number of registers
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
NvMediaStatus
MAX96717FArrayWriteVerify(
    DevBlkCDIDevice const * const handle,
    DevBlkCDII2CReg const * const regs,
    uint32_t const length);

/**
 * @brief Poll bit in specified register and bit position
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice
 * @param[in] addr address to find bit
 * @param[in] bit bit position to check
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_ERROR and error
 */
NvMediaStatus
MAX96717FPollBit(
    DevBlkCDIDevice const *handle,
    uint16_t addr,
    uint8_t bit);

/**
 * @brief Check if GPIO x is valid
 *
 * @param[in] gpioInd GPIO Index
 *
 * @retval NVMEDIA_STATUS_OK if GPIO index is valid for the MAX96717(F) Serializer
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if GPIO index is not valid
 */
NvMediaStatus
MAX96717FIsGPIOValid(
    uint8_t const gpioInd);

/**
 * @brief Check if GPIO x is set
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice
 * @param[in] gpioInd GPIO Index
 * @param[out] gpioStatus GPIO status [0,1]
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
NvMediaStatus
MAX96717FIsGPIOSet(
    DevBlkCDIDevice const * const handle,
    uint8_t const gpioInd,
    uint8_t* const gpioStatus);

/**
 * @brief Setup address translation B
 *
 * @param[in] serCDI Device Handle for DevBlkCDIDevice
 * @param[in] src source address for i2c device
 * @param[in] dst destination address for i2c device
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
NvMediaStatus
MAX96717FSetupAddressTranslationsB(
    DevBlkCDIDevice const* const serCDI,
    uint8_t const src,
    uint8_t const dst);

#ifdef NVMEDIA_QNX
/**
 * @brief Verify GPO Readback Status
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice
 * @param[in] gpioListSM23 List of GPIO_A register addresses
 * @param[in] numListSM23 Length of the above list
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
NvMediaStatus
MAX96717FVerifyGPIOReadBackStatus(
    DevBlkCDIDevice const * const handle,
    uint8_t const * const gpioListSM23,
    uint8_t const numListSM23);

/**
 * @brief Verify GPIO Open Detection
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice
 * @param[in] gpioListSM40 List of GPIO_A register addresses
 * @param[in] numListSM40 Length of the above list
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
NvMediaStatus
MAX96717FVerifyGPIOOpenDetection(
    DevBlkCDIDevice const * const handle,
    uint8_t const * const gpioListSM40,
    uint8_t const numListSM40);
#endif

/**
 * @brief Disable Serializer register CRC reporting to ERRB
 *
 * @param[in] handle Device handle for DevBlkCDIDevice
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
NvMediaStatus
MAX96717FDisableRegCRC(
    DevBlkCDIDevice const *const handle);

/**
 * @brief Re-enable configuration for MAX96717(F) Serializer register CRC
 *
 * @param[in] handle Device handle for DevBlkCDIDevice
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
NvMediaStatus
MAX96717FReEnableRegCRC(
    DevBlkCDIDevice const *const handle);

/**
 * @brief Set configuration for MAX96717(F) Serializer register CRC
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
NvMediaStatus
MAX96717FSetRegCRC(
    DevBlkCDIDevice const * const handle);

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
} /* extern "C" */
} /* namespace nvsipl */
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentionalTID-1433 */
#endif /* CDI_MAX96717F_H */
