/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>

#include "sipl_error.h"
#include "devblk_cdi_i2c.h"
#include "cdi_max96717f.h"
#include "os_common.h"
#include "sipl_util.h"
#include "cdd_nv_error.h"

#define MAX96717F_NUM_ADDR_BYTES       2U
#define MAX96717F_NUM_DATA_BYTES       1U

#define INCK_24MHZ                    (24U)
#define INCK_25MHZ                    (25U)
#define UINT8_MAX_SHIFT                7U
#define DPLL_LOCK_UNLOCKED_BIT         7U
#define MFP_INPUT_STATUS_BIT           3U
#define MAX96717F_LOCK_ENABLE_BIT      7U
#define MAX96717F_ERRB_ENABLE_BIT      6U
#define MAX96717F_LOCK_INDICATOR_BIT   3U
#define MAX96717F_ERRB_INDICATOR_BIT   2U
#define MAX_LOCK_CHECK_RETRY_CNT       25U
#define MAX_RETRY_COUNT                10U
#define UNLOCK_STATUS_SLEEP_US         4000 /* 4ms */
#define ADC_DONE_IF_SLEEP_US           3000 /* 3ms */
#define INT_DONE_IF_SLEEP_US           3000 /* 3ms */
/* Must be set with DPLL_LOCK_CHECK_RETRY_SLEEP_US as overall
 * timeout applied for DPLL LOCK will depend on their product*/
#define DPLL_LOCK_CHECK_RETRY_CNT      5U
#define DPLL_LOCK_CHECK_RETRY_SLEEP_US 1000 /* 1ms */
#define MAX_ADC_CHANNEL_NUM            3U
#define MAX_READ_WRITE_ARRAY_SIZE      (50U)
#define MAX_VALID_REG                  (0x1D5F) /* Refer to GetRegReadWriteMasks() */

#define REG_DEV_ID_ADDR                0x0DU
#define MAX96717F_DEV_ID_C8            0xC8U /* C8 received from query MAX96717 (T32) --> 0xBF */
#define MAX96717_DEV_ID_BF             0xBFU
#define REG_DEV_REV_ADDR               0x0EU
#define REG_ADCBIST13                  0x1D3BU
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
#define MAX96717F_REG_MAX_ADDRESS      0x1D3DU
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif /* !NV_IS_SAFETY */
#define MAX96717F_NUM_REGS_PER_GPIO_PIN 3U

#define MAX96717F_LINK_LOCKED       0x01U
#define MAX96717F_LINK_UNLOCKED     0x00U
#define MAX96717F_ERRB_ASSERTED     0x01U
#define MAX96717F_ERRB_NOT_ASSERTED 0x00U

#define REG_REG0        0x0000U
#define REG_REG5        0x0005U
#define REG_CTRL3       0x0013U
#define REG_VIDEO_TX0   0x0110U
#define REG_REF_VTG0    0x03F0U
#define REG_INTR3       0x001BU
#define REG_INTR5       0x001DU
#define REG_INTR7       0x001FU
#define REG_INTR8       0x0020U
#define REG_CTRL1_CSI_L 0x0343U
#define REG_CTRL1_CSI_H 0x0344U
#define REG_PHY1_HS_ERR 0x033CU
#define REG_PHY2_HS_ERR 0x033EU
#define REG_EXT8        0x0380U
#define REG_ARQ2_SER    0x008FU
#define REG_ARQ2_GPIO   0x0097U
#define REG_ARQ2_I2C_X  0x00A7U
#define REG_ARQ2_I2C_Y  0x00AFU
#define REG_FS_INTR1    0x1D13U
#define REG_ADC_DATA0   0x0508U
#define REG_ADC_INTR0   0x0510U
#define REG_ADC_INTR1   0x0511U
#define REG_ADC_INTR2   0x0512U
#define REG_ADC_INTR3   0x0513U
#define REG_ADC_CTRL4   0x053EU
#define REG_POST0       0x1D20U
#define REG_DEC_ERROR   0x0022U
#define REG_IDLE_ERROR  0x0024U
#define REG_EOM_RLMS4   0x1404U
#define REG_EOM_RLMS5   0x1405U
#define REG_EOM_RLMS7   0x1407U
#define REG_EOM_RLMSA4  0x14A4U
#define REG_CNT3        0x0025U
#define REG_VTX29       0x026BU
#define REG_TX1         0x0029U
#define REG_TX2         0x002AU
#define REG_CFGV_TX0    0x0058U
#define REG_REGCRC0     0x1D00U
#define REG_REGCRC1     0x1D01U
#define REG_FS_INTR0    0x1D12U

#if FUSA_CDD_NV
#define INTR3_REFGEN_UNLOCKED_MASK      0x80U   /* represents 0x001B[7] */
#define INTR3_IDLE_ERR_FLAG_MASK        0x04U   /* represents 0x001B[2] */
#define INTR3_DEC_ERR_FLAG_A_MASK       0x01U   /* represents 0x001B[0] */
#define INTR5_VREG_OV_FLAG_MASK         0x80U   /* represents 0x001D[7] */
#define INTR5_EOM_ERR_FLAG_A_MASK       0x40U   /* represents 0x001D[6] */
#define INTR5_VDD_OV_FLAG_MASK          0x20U   /* represents 0x001D[5] */
#define INTR5_VDD18_OV_FLAG_MASK        0x10U   /* represents 0x001D[4] */
#define INTR5_MAX_RT_FLAG_MASK          0x08U   /* represents 0x001D[3] */
#define INTR5_RT_CNT_FLAG_MASK          0x04U   /* represents 0x001D[2] */
#define INTR7_VDDCMP_INT_FLAG_MASK      0x80U   /* represents 0x001F[7] */
#define INTR7_PORZ_INT_FLAG_MASK        0x40U   /* represents 0x001F[6] */
#define INTR7_VDDBAD_INT_FLAG_MASK      0x20U   /* represents 0x001F[5] */
#define INTR7_EFUSE_CRC_ERR_MASK        0x10U   /* represents 0x001F[4] */
#define INTR7_ADC_INTR_FLAG_MASK        0x04U   /* represents 0x001F[2] */
#define INTR7_MIPI_ERR_FLAG_MASK        0x01U   /* represents 0x001F[0] */
#define FS_INTR1_MEM_ECC_ERR2_INT_MASK  0x20U   /* represents 0x1D13[5] */
#define FS_INTR1_MEM_ECC_ERR1_INT_MASK  0x10U   /* represents 0x1D13[4] */
#define FS_INTR1_REG_CRC_ERR_FLAG_MASK  0x01U   /* represents 0x1D13[0] */
#define TMON_ERR_IF_FLAG_MASK           0X02U   /* represents 0x0513[1] */
#define REFGEN_LOCKED_FLAG_MASK         0x80U   /* represents 0x03F0[7] */
#define TUN_FIFO_OVERFLOW_FLAG_MASK     0x01U   /* represents 0x0380[0] */
#define LOCKED_FLAG_MASK                0X08U   /* represents 0x0013[3] */
#define MAX_RT_ERR_ARQ2_FLAG_MASK       0x80U   /* represents (0x008F, 0x0097,
                                                 * 0x00A7, 0x00AF)[7] */
#endif

/* calculate the GPIO Register_A address based on the GPIO index.
 * The macro relies on the GPIO register addresses being contiguous in the
 * register map.
 */
#define GET_GPIO_A_ADDR(gpioInd) ((uint32_t)(( \
    (uint32_t)REG_GPIO0_A + (uint32_t)((uint32_t)(gpioInd) * 3U)) & 0xFFFFU))

typedef enum {
    CDI_MAX96717F_INVALID_REV,
    CDI_MAX96717F_REV_2,
    CDI_MAX96717F_REV_4,
} RevisionMAX96717F;

typedef struct {
    RevisionMAX96717F revId;
    uint32_t revVal;
} RevMAX96717F;

typedef struct {
    RevisionMAX96717F   revisionID;
    DevBlkCDII2CPgmr    i2cProgrammer;
} DriverHandleMAX96717F;

/**
 * @brief Unset ERRB TX to disable error reporting
 *
 * @param[in] handle CDI handle to the serializer.
 * @param[out] NVMEDIA error status.
 *
 * @retval NVMEDIA_STATUS_OK if successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if invalid parameter passed
 * @retval NVMEDIA_STATUS_ERROR in case of failure.
 */

static NvMediaStatus UnsetErrbTx(DevBlkCDIDevice const *handle);

/**
 * @brief Set ERRB TX to enable error reporting
 *
 * @param[in] handle CDI handle to the serializer.
 * @param[out] NVMEDIA error status.
 *
 * @retval NVMEDIA_STATUS_OK if successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if invalid parameter passed
 * @retval NVMEDIA_STATUS_ERROR in case of failure.
 */

static NvMediaStatus SetErrbTx(DevBlkCDIDevice const *handle);

/**
 * @brief Set ERRB TX to enable error reporting with given ID
 *
 * @param[in] handle CDI handle to the serializer.
 * @param[in] destination gpio i.e. deserializer gpio.
 * @param[out] NVMEDIA error status.
 *
 * @retval NVMEDIA_STATUS_OK if successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if invalid parameter passed
 * @retval NVMEDIA_STATUS_ERROR in case of failure.
 */

static NvMediaStatus SetErrbTxId(DevBlkCDIDevice const *handle, uint8_t dstGpio);

/**
 * @brief check specific bit is set or clear
 *
 * @param[in] var   current value
 * @param[in] shift which bit is set or clear Range [0:7]
 *
 * @return if a bit is set return to positive number
 *         if a bit is clear return to zero.
 */
static inline uint8_t GetBit(uint8_t var, uint8_t shift);

/**
 * @brief Sets a bit in given value at given position.
 *
 * @param[in] val Bit value to be set.
 * @param[in] bit Bit shift value.
 *
 * @retval 8bit bit set value returned.
 */
static inline uint8_t SetBit(uint8_t val, uint8_t bit);

/**
 * @brief Clear a bit in given value at given position.
 *
 * @param[in] val Bit value to be clear.
 * @param[in] bit Bit shift value.
 *
 * @retval 8bit bit clear value returned.
 */
static inline uint8_t ClearBit(uint8_t val, uint8_t bit);

/**
 * @brief Sets the register bit to the specified value
 *
 * @param[in] handle  CDI handle to the serializer.
 * @param[in] addr Which Register address to modify
 * @param[in] bit Register bit to modify.
 * @param[in] val Bit value to be set.
 *
 * @retval NVMEDIA_STATUS_OK if succesful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if i2cProgrammer handle is null
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
static NvMediaStatus SetRegBitMAX96717F(
    DevBlkCDIDevice const * const handle,
    uint16_t addr,
    uint8_t bit,
    bool val);

/**
 * @brief Return the MAX96717FSMErrInfoRegSet pointer.
 *
 * find which MAX96717FSMErrInfoRegSet is matched with index.
 *
 * @param[in] index Which SMErrIndex needs to get the pointer.
 *
 * @returns NULL Wrong index or didn't support SM (Failure).
 * @returns non-NULL if finding index matched MAX96717FSMErrInfoRegSet.
 */
static MAX96717FSMErrInfoRegSet const *GetRegSetMAX96717F(
    uint8_t const index);

/**
 * @brief Checks for Safety Mechanism error bits and fills
 *        relevant information.
 *
 * @param[in]  handle  CDI handle to the serializer.
 * @param[out] errInfo Error information structure to be filled.
 *
 * @retval NVMEDIA_STATUS_OK if successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if invalid parameter passed
 * @retval NVMEDIA_STATUS_ERROR in case of failure.
 */
static NvMediaStatus GetSMErrInfoMAX96717F(
    DevBlkCDIDevice const* handle,
    MAX96717FErrorInfo *const errInfo);

/**
 * @brief Gets the MIPI error information.
 *
 * @param[in] handle CDI device handle to the sensor.
 * @param[out] smErrInfo SM error info structure to be filled.
 *
 * @return NVMEDIA_STATUS_OK Call successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER Invalid param passed.
 * @retval NVMEDIA_STATUS_ERROR Get MIPI error info call failed.
 */
static NvMediaStatus GetMIPIErrInfo(
    DevBlkCDIDevice const* handle,
    MAX96717FErrorInfo *const smErrInfo);

/**
 * @brief Gets the ARQ error information.
 *
 * @param[in] handle CDI device handle to the sensor.
 * @param[out] smErrInfo SM error info structure to be filled.
 *
 * @return NVMEDIA_STATUS_OK Call successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER Invalid param passed.
 * @retval NVMEDIA_STATUS_ERROR Get ARQ error info call failed.
 */
static NvMediaStatus GetARQErrInfo(
    DevBlkCDIDevice const* handle,
    MAX96717FErrorInfo *const smErrInfo);

/**
 * @brief Enables Safety Mechanism error reporting via ERRB pin
 *
 * @param[in] handle CDI handle to the serializer.
 * @param[out] errInfo Error information structure to be filled.
 *
 * @retval NVMEDIA_STATUS_OK if successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if invalid parameter passed
 * @retval NVMEDIA_STATUS_ERROR in case of failure.
 */
static NvMediaStatus EnableSMReports(
    DevBlkCDIDevice const *handle,
    bool skipRefgenCheck
);

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if NVMEDIA_QNX
/**
 * @brief temperature sensor status of MAX96717F is changed to readable.
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice.
 *
 * @retval NVMEDIA_STATUS_OK Register setting is success
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
static NvMediaStatus
SetTemperatureMAX96717F(
    DevBlkCDIDevice const *handle);
#endif

/**
 * @brief Clear fault report status bit.
 *
 * A SM had different register between status and clear.
 * This fuction cleared the SM report status.
 *
 * @param[in] handle CDI handle to the serializer.
 * @param[in] index SM index.
 * @param[in] regVal SM status reg value.
 *
 * @retval NVMEDIA_STATUS_OK Success.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER handle is NULL
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
static NvMediaStatus
ClrSMStatusMAX96717F(
    DevBlkCDIDevice const * const handle,
    uint8_t index,
    uint8_t regVal);

/**
 * @brief Assert/De-assert ERRB
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice
 * @param[in] trigger true is assert / false is de-assert
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
static NvMediaStatus
AssertErrbMAX96717F(
    DevBlkCDIDevice const *handle,
    bool trigger);

/**
 * @brief Convert temperature from Kelvin to Celsius
 *
 * Dividing 10-bit ADC value by 2 to get Kelvin value
 * Subtracting 273 to convert it to Celsius
 *
 * @param[in] kelvinVal Kelvin value
 *
 * @return float Celsius value
 */
static inline float_t KelvinToCelsius(uint16_t kelvinVal){
    float_t retVal = 0.0F;
    retVal = (float_t)((float_t)kelvinVal - 273.15F);

    return retVal;
}

/**
 * @brief Runs the manual Eye-Open-Monitor to get the EOM value.
 *
 * @param[in]  handle  CDI handle to the serializer.
 * @param[out] errInfo Error information structure to be filled.
 *
 * @retval NVMEDIA_STATUS_OK if successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if invalid parameter passed
 * @retval NVMEDIA_STATUS_ERROR in case of failure.
 */
static NvMediaStatus GetEOMVal(DevBlkCDIDevice const* handle,
                               uint8_t *const eomVal);

/**
 * @Resets MIPI receiver to prepare for video stream
 *
 * @param[in] handle CDI Device Block Handle.
 *
 * @retval NVMEDIA_STATUS_OK Write was successful
 * @retval NVMEDIA_STATUS_BAD_PARAMETER input parameter pointer is NULL
 * @retval NVMEDIA_STATUS_ERROR Write was failure
 */
static NvMediaStatus MIPI_Reset(DevBlkCDIDevice const* handle);

/**
 * @brief Check if a single bit is set or not
 *
 * @param[in] val Register value
 * @param[in] mask Bit mask to check for bit. Only for single bit masks.
 *
 * @return uint8_t 1U if bit is set or 0U if bit is not not set
 */
static inline uint8_t IsBitSet(uint8_t val, uint8_t mask) {
    uint8_t const bit_set = 1U, bit_unset = 0U;
    return ((val & mask) == mask) ? bit_set : bit_unset;
}

static inline uint8_t GetBit(uint8_t var, uint8_t shift) {
    uint8_t retVal = 0U;
    if (UINT8_MAX_SHIFT >= shift) {
        retVal = var & bit8(shift);
    }

    return retVal;
}

static DriverHandleMAX96717F *getHandlePrivMAX96717F(
    DevBlkCDIDevice const *handle)
{
    DriverHandleMAX96717F *ret = NULL;

    if (handle != NULL) {
        /* coverity[misra_c_2012_rule_11_5_violation] : intentional TID-1417 */
        ret = (DriverHandleMAX96717F *)handle->deviceDriverHandle;
    }
    return ret;
}

static inline uint8_t SetBit(uint8_t val, uint8_t bit) {
    uint8_t retVal = 0U;
    if (UINT8_MAX_SHIFT >= bit) {
        retVal = val | bit8(bit);
    }

    return retVal;
}

static inline uint8_t ClearBit(uint8_t val, uint8_t bit) {
    uint8_t retVal = 0U;
    if (UINT8_MAX_SHIFT >= bit) {
        retVal = val & ~bit8(bit);
    }

    return retVal;
}

/**
 * @brief Contains the masks used to perform read-verify and write-verify.
 * Masks are needed to avoid verification of bits which don't have a persistent state,
 * eg. Clear on Read bits will change state as soon as first read is done over them,
 * thus we can't perform read-verify over these bits as it would fail as the bits have
 * changed states.
 *
 * maskRead : this is used to perform read-verify
 * maskWrite : this is used to perform write-verify
 * If a register doesn't have any bit which can change states then the default mask i.e 0xFF can
 * be applied to verify all bits.
 */
typedef struct {
    /** Mask for read-verify */
    uint8_t      maskRead;
    /** Mask for write-verify */
    uint8_t      maskWrite;
} MAX96717FRegMask;

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
 */
static inline NvMediaStatus GetRegReadWriteMasks(
    uint16_t const address, MAX96717FRegMask * const regMasks)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t rMask = 0x00U;
    uint8_t wMask = 0x00U;
    switch (address)
    {
        /* Skip Verify for 0x0000U as I2C read fails */
        case 0x0000U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0003U: rMask = 0x37U; wMask = 0x37U; break;
        case 0x0005U: rMask = 0xF3U; wMask = 0xF3U; break;
        case 0x0006U: rMask = 0x30U; wMask = 0x30U; break;
        case 0x000DU: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x000EU: rMask = 0x0FU; wMask = 0x00U; break;
        case 0x0010U: rMask = 0xE8U; wMask = 0x48U; break;
        case 0x0013U: rMask = 0x0EU; wMask = 0x00U; break;
        case 0x001AU: rMask = 0xADU; wMask = 0xADU; break;
        case 0x001BU: rMask = 0xADU; wMask = 0x00U; break;
        case 0x001CU: rMask = 0xFEU; wMask = 0xFEU; break;
        case 0x001DU: rMask = 0x0EU; wMask = 0x00U; break;
        case 0x001EU: rMask = 0xFDU; wMask = 0xFDU; break;
        case 0x001FU: rMask = 0x1DU; wMask = 0x00U; break;
        case 0x0020U: rMask = 0x9FU; wMask = 0x9FU; break;
        case 0x0022U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0024U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0025U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0029U: rMask = 0x9BU; wMask = 0x9BU; break;
        case 0x002AU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x002CU: rMask = 0xCFU; wMask = 0xCFU; break;
        case 0x0042U: rMask = 0xFEU; wMask = 0xFEU; break;
        case 0x0043U: rMask = 0xFEU; wMask = 0xFEU; break;
        case 0x0044U: rMask = 0xFEU; wMask = 0xFEU; break;
        case 0x0045U: rMask = 0xFEU; wMask = 0xFEU; break;
        case 0x0058U: rMask = 0x80U; wMask = 0x80U; break;
        case 0x0078U: rMask = 0xC0U; wMask = 0xC0U; break;
        case 0x008EU: rMask = 0x03U; wMask = 0x03U; break; /* similar to ARQ1 */
        case 0x008FU: rMask = 0x00U; wMask = 0x00U; break; /* similar to ARQ2 */
        case 0x0090U: rMask = 0xC0U; wMask = 0xC0U; break;
        case 0x0096U: rMask = 0x03U; wMask = 0x03U; break;
        case 0x0097U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x00A0U: rMask = 0xC0U; wMask = 0xC0U; break;
        case 0x00A6U: rMask = 0x03U; wMask = 0x03U; break;
        case 0x00A7U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x00A8U: rMask = 0xC0U; wMask = 0xC0U; break;
        case 0x00AEU: rMask = 0x03U; wMask = 0x03U; break;
        case 0x00AFU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0110U: rMask = 0xFCU; wMask = 0xFCU; break;
        case 0x0111U: rMask = 0x3FU; wMask = 0x3FU; break;
        case 0x0112U: rMask = 0x84U; wMask = 0x04U; break;
        case 0x024FU: rMask = 0x2FU; wMask = 0x0FU; break;
        case 0x026BU: rMask = 0xA7U; wMask = 0x87U; break;
        case 0x02BEU: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x02BFU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x02C0U: rMask = 0x9FU; wMask = 0x9FU; break;
        case 0x02C1U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x02C2U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x02C3U: rMask = 0x9FU; wMask = 0x9FU; break;
        case 0x02C4U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x02C5U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x02C6U: rMask = 0x9FU; wMask = 0x9FU; break;
        case 0x02C7U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x02C8U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x02C9U: rMask = 0x9FU; wMask = 0x9FU; break;
        case 0x02CAU: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x02CBU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x02CCU: rMask = 0x9FU; wMask = 0x9FU; break;
        case 0x02CDU: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x02CEU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x02CFU: rMask = 0x9FU; wMask = 0x9FU; break;
        case 0x02D0U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x02D1U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x02D2U: rMask = 0x9FU; wMask = 0x9FU; break;
        case 0x02D3U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x02D4U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x02D5U: rMask = 0x9FU; wMask = 0x9FU; break;
        case 0x02D6U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x02D7U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x02D8U: rMask = 0x9FU; wMask = 0x9FU; break;
        case 0x02D9U: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x02DAU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x02DBU: rMask = 0x9FU; wMask = 0x9FU; break;
        case 0x02DCU: rMask = 0xBFU; wMask = 0xB7U; break;
        case 0x02DDU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x02DEU: rMask = 0x9FU; wMask = 0x9FU; break;
        case 0x0308U: rMask = 0x60U; wMask = 0x60U; break;
        case 0x0311U: rMask = 0x40U; wMask = 0x40U; break;
        case 0x0312U: rMask = 0x04U; wMask = 0x04U; break;
        case 0x0313U: rMask = 0x44U; wMask = 0x44U; break;
        case 0x0318U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x0319U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x031EU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0330U: rMask = 0x68U; wMask = 0x68U; break;
        case 0x0331U: rMask = 0xF0U; wMask = 0xF0U; break;
        case 0x0332U: rMask = 0xF0U; wMask = 0xF0U; break;
        case 0x0333U: rMask = 0x0FU; wMask = 0x0FU; break;
        case 0x0334U: rMask = 0x70U; wMask = 0x70U; break;
        case 0x0335U: rMask = 0x07U; wMask = 0x07U; break;
        case 0x033CU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x033EU: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0343U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0344U: rMask = 0x00U; wMask = 0x00U; break;
        case 0x0380U: rMask = 0x01U; wMask = 0x00U; break;
        case 0x0383U: rMask = 0x80U; wMask = 0x80U; break;
        case 0x03F0U: rMask = 0xFBU; wMask = 0x7BU; break;
        case 0x03F1U: rMask = 0xBFU; wMask = 0xBFU; break;
        case 0x03F4U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x03F5U: rMask = 0x0FU; wMask = 0x0FU; break;
        case 0x0500U: rMask = 0x9FU; wMask = 0x9EU; break;
        case 0x0501U: rMask = 0xFEU; wMask = 0xFEU; break;
        case 0x0502U: rMask = 0x0FU; wMask = 0x0FU; break;
        case 0x0508U: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x0509U: rMask = 0x03U; wMask = 0x00U; break;
        case 0x050CU: rMask = 0xEFU; wMask = 0xEFU; break;
        case 0x050DU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x050EU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x050FU: rMask = 0x7AU; wMask = 0x7AU; break;
        case 0x0510U: rMask = 0X00U; wMask = 0x00U; break;
        case 0x0511U: rMask = 0X00U; wMask = 0x00U; break;
        case 0x0512U: rMask = 0X00U; wMask = 0x00U; break;
        case 0x0513U: rMask = 0X02U; wMask = 0x00U; break; /* confirmed that b1 is not COR */
        case 0x0514U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0515U: rMask = 0xF3U; wMask = 0xF3U; break;
        case 0x0516U: rMask = 0x3FU; wMask = 0x3FU; break;
        case 0x0517U: rMask = 0x3FU; wMask = 0x3FU; break;
        case 0x0518U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0519U: rMask = 0xF3U; wMask = 0xF3U; break;
        case 0x051AU: rMask = 0x3FU; wMask = 0x3FU; break;
        case 0x051BU: rMask = 0x3FU; wMask = 0x3FU; break;
        case 0x051CU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x051DU: rMask = 0xF3U; wMask = 0xF3U; break;
        case 0x051EU: rMask = 0x3FU; wMask = 0x3FU; break;
        case 0x051FU: rMask = 0x3FU; wMask = 0x3FU; break;
        case 0x0524U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x0525U: rMask = 0xF3U; wMask = 0xF3U; break;
        case 0x0526U: rMask = 0x3FU; wMask = 0x3FU; break;
        case 0x0527U: rMask = 0x3FU; wMask = 0x3FU; break;
        case 0x0534U: rMask = 0x01U; wMask = 0x01U; break;
        case 0x0535U: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0x0536U: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0x0537U: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0x053EU: rMask = 0x07U; wMask = 0x07U; break;
        case 0x056FU: rMask = 0x3FU; wMask = 0x3FU; break;
        case 0x0570U: rMask = 0x3CU; wMask = 0x3CU; break;
        case 0x1404U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1405U: rMask = 0x7FU; wMask = 0x7FU; break;
        case 0x1407U: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x14A4U: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0x14CEU: rMask = 0x19U; wMask = 0x19U; break;
        case 0x1A03U: rMask = 0x97U; wMask = 0x97U; break;
        case 0x1A07U: rMask = 0xFCU; wMask = 0xFCU; break;
        case 0x1A08U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1A09U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1A0AU: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1A0BU: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0x1D00U: rMask = 0x1FU; wMask = 0x1EU; break;
        case 0x1D01U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1D09U: rMask = 0x03U; wMask = 0x00U; break;
        case 0x1D12U: rMask = 0xF1U; wMask = 0xF1U; break;
        case 0x1D13U: rMask = 0xF1U; wMask = 0x00U; break;
        case 0x1D14U: rMask = 0x03U; wMask = 0x00U; break;
        case 0x1D20U: rMask = 0xE0U; wMask = 0x00U; break;
        case 0x1D28U: rMask = 0x95U; wMask = 0x90U; break;
        case 0x1D37U: rMask = 0xFFU; wMask = 0xFFU; break;
        case 0x1D3BU: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x1D3CU: rMask = 0xC3U; wMask = 0x00U; break;
        case 0x1D3DU: rMask = 0xFFU; wMask = 0x00U; break;
        case 0x1D40U: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0x1D41U: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0x1D42U: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0x1D43U: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0X1D44U: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0X1D45U: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0X1D46U: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0X1D47U: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0X1D48U: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0X1D49U: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0X1D4AU: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0X1D4BU: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0X1D4CU: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0X1D4DU: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0X1D4EU: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0X1D4FU: rMask = 0x00U; wMask = 0x00U; break; /* Undocumented in datasheet */
        case 0x1D5FU: rMask = 0x07U; wMask = 0x03U; break;
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
static inline NvMediaStatus ReadVerifyMAX96717F(
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
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
static inline NvMediaStatus WriteUint8VerifyMAX96717F(
    DevBlkCDIDevice const * const handle,
    uint16_t const address,
    uint8_t const data)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96717F const * const drvHandle = getHandlePrivMAX96717F(handle);
    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer, address, data);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("DevBlkCDII2CPgmrWriteUint8 failed with ", (uint32_t)status);
        } else {
            MAX96717FRegMask regMasks;
            status = GetRegReadWriteMasks(address, &regMasks);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT("GetRegReadWriteMasks failed with ", (uint32_t)status);
            } else {
                status = ReadVerifyMAX96717F(
                    drvHandle->i2cProgrammer, address, data, regMasks.maskWrite);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR_HEX_UINT("ReadVerifyMAX96717F failed with ", (uint32_t)status);
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
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
static inline NvMediaStatus ReadUint8VerifyMAX96717F(
    DevBlkCDIDevice const * const handle,
    uint16_t const address,
    uint8_t *data)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96717F const * const drvHandle = getHandlePrivMAX96717F(handle);
    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, address, data);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("DevBlkCDII2CPgmrReadUint8 failed with ", (uint32_t)status);
        } else {
            MAX96717FRegMask regMasks;
            status = GetRegReadWriteMasks(address, &regMasks);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT("GetRegReadWriteMasks failed with ", (uint32_t)status);
            } else {
                status = ReadVerifyMAX96717F(
                    drvHandle->i2cProgrammer, address, *data, regMasks.maskRead);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR_HEX_UINT("ReadVerifyMAX96717F failed with ", (uint32_t)status);
                }
            }
        }
    }

    return status;
}

/**
 * @brief verify the readback data from ArrayRead
 *
 * @param[in]  data1  Array of expected Value
 * @param[in]  data2  Array of readback Value
 * @param[in]  dataLength  Size of above arrays
 * @param[in]  useWriteMask  Use right mask to verify readback value
 *
 * @retval NVMEDIA_STATUS_OK if verification was succesful.
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
static inline NvMediaStatus VerifyArrayReadVal(
    DevBlkCDII2CReg const * data1,
    DevBlkCDII2CReg const * data2,
    uint16_t const dataLength,
    bool const useWriteMask)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    for (uint32_t i = 0; i < dataLength; i++) {
        MAX96717FRegMask regMasks;
        status = GetRegReadWriteMasks(data1[i].address, &regMasks);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT(
                "GetRegReadWriteMasks failed with ", (uint32_t)status);
            break;
        } else {
            uint8_t d1 = (uint8_t)(data1[i].data & 0xFFU);
            uint8_t d2 = (uint8_t)(data2[i].data & 0xFFU);
            uint8_t readData = d1 ^ d2;
            readData &= useWriteMask ? regMasks.maskWrite : regMasks.maskRead;
            if (readData > 0U) {
                if (useWriteMask) {
                    SIPL_LOG_ERR_STR_HEX_UINT(
                        "Masked Write-verification failed for reg", data1[i].address);
                } else {
                    SIPL_LOG_ERR_STR_HEX_UINT(
                        "Masked Read-verification failed for reg", data1[i].address);
                }
                status = NVMEDIA_STATUS_ERROR;
                break;
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
static inline NvMediaStatus WriteArrayVerifyMAX96717F(
    DevBlkCDIDevice const * const handle,
    DevBlkCDII2CRegList const *regList)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96717F const * const drvHandle = getHandlePrivMAX96717F(handle);
    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else if ((regList->numRegs == 0U) || (regList->numRegs > MAX_READ_WRITE_ARRAY_SIZE)) {
        SIPL_LOG_ERR_STR("Bad parameter: Invalid array size passed");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        uint32_t i = 0;
        while (i < regList->numRegs) {
            uint16_t addrArray[MAX_READ_WRITE_ARRAY_SIZE];
            uint16_t subArraySize = 0;
            DevBlkCDII2CReg tReadRegs[MAX_READ_WRITE_ARRAY_SIZE];
            DevBlkCDII2CReg tWriteRegs[MAX_READ_WRITE_ARRAY_SIZE];

            /**
             * Construct sub-arrays for CDI WriteArray without containing repeated addresses
             * so as to perform the readback verification successfully.
             */
            while (i < regList->numRegs) {
                bool exitFlag = false;
                for (uint32_t j = 0; j < subArraySize; j++) {
                    if (regList->regs[i].address == addrArray[j]) {
                        exitFlag = true;
                    }
                }
                if (exitFlag) {
                    break;
                }
                addrArray[subArraySize] = regList->regs[i].address;
                tReadRegs[subArraySize].address = regList->regs[i].address;
                tWriteRegs[subArraySize] = regList->regs[i];
                subArraySize++;
                i++;
            }

            /* Create Placeholder RegLists for Array read and write */
            DevBlkCDII2CRegList writeRegList = {
                .regs = tWriteRegs,
                .numRegs = subArraySize,
            };
            DevBlkCDII2CRegListWritable const readRegList = {
                .regs = tReadRegs,
                .numRegs = subArraySize
            };
            /* Perform sub-array-wise write-verify operations. */
            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &writeRegList);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "DevBlkCDII2CPgmrWriteArray failed with ", (uint32_t)status);
                goto done;
            } else {
                status = DevBlkCDII2CPgmrReadArray(drvHandle->i2cProgrammer, &readRegList);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR_HEX_UINT(
                        "DevBlkCDII2CPgmrReadArray failed with ", (uint32_t)status);
                    goto done;
                } else {
                    /* Verify the value written */
                    status = VerifyArrayReadVal(&tWriteRegs[0], &tReadRegs[0], subArraySize, true);
                    if (NVMEDIA_STATUS_OK != status) {
                        SIPL_LOG_ERR_STR_HEX_UINT(
                            "VerifyArrayReadVal failed with ", (uint32_t)status);
                        goto done;
                    }
                }
            }
        }
    }

done:
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
static inline NvMediaStatus ReadArrayVerifyMAX96717F(
    DevBlkCDIDevice const * const handle,
    DevBlkCDII2CRegListWritable const *regList)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96717F const * const drvHandle = getHandlePrivMAX96717F(handle);
    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else if ((regList->numRegs == 0U) || (regList->numRegs > MAX_READ_WRITE_ARRAY_SIZE)) {
        SIPL_LOG_ERR_STR("Bad parameter: Invalid array size passed");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = DevBlkCDII2CPgmrReadArray(drvHandle->i2cProgrammer, regList);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT(
                "DevBlkCDII2CPgmrWriteArray failed with ", (uint32_t)status);
        } else {
            DevBlkCDII2CReg tReadRegs[MAX_READ_WRITE_ARRAY_SIZE];
            for (uint32_t i = 0; i < regList->numRegs; i++) {
                tReadRegs[i].address = regList->regs[i].address;
            }
            /* Create Placeholder RegLists for Array read */
            DevBlkCDII2CRegListWritable const readRegList = {
                .regs = tReadRegs,
                .numRegs = regList->numRegs
            };
            status = DevBlkCDII2CPgmrReadArray(drvHandle->i2cProgrammer, &readRegList);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "DevBlkCDII2CPgmrReadArray failed with ", (uint32_t)status);
            } else {
                /* Verify the value Read */
                status = VerifyArrayReadVal(
                    regList->regs, &tReadRegs[0], (uint16_t)regList->numRegs, false);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR_HEX_UINT("VerifyArrayReadVal failed with ", (uint32_t)status);
                }
            }
        }
    }

    return status;
}

/**
 * @brief verify the readback data from BlockRead
 *
 * @param[in]  data1  Array of expected Value
 * @param[in]  data2  Array of readback Value
 * @param[in]  registerNum  Starting register number
 * @param[in]  dataLength  Size of above arrays
 * @param[in]  useWriteMask  Use right mask to verify readback value
 *
 * @retval NVMEDIA_STATUS_OK if verification was succesful.
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
static inline NvMediaStatus VerifyBlockReadVal(
    uint8_t const * data1,
    uint8_t const * data2,
    uint16_t const registerNum,
    uint16_t const dataLength,
    bool const useWriteMask)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    for (uint16_t i = 0; i < dataLength; i++) {
        MAX96717FRegMask regMasks;
        uint16_t address = registerNum + i;
        status = GetRegReadWriteMasks(address, &regMasks);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT(
                "GetRegReadWriteMasks failed with ", (uint32_t)status);
            break;
        } else {
            uint8_t readData = data1[i] ^ data2[i];
            readData &= useWriteMask ? regMasks.maskWrite : regMasks.maskRead;
            if (readData > 0U) {
                if (useWriteMask) {
                    SIPL_LOG_ERR_STR_HEX_UINT(
                        "Masked Write-verification failed for reg", (uint32_t)address);
                } else {
                    SIPL_LOG_ERR_STR_HEX_UINT(
                        "Masked Read-verification failed for reg", (uint32_t)address);
                }
                status = NVMEDIA_STATUS_ERROR;
                break;
            }
        }
    }

    return status;
}

/**
 * @brief Write a block of Uint8 data and readback to verify
 *
 * @param[in]  handle  CDI Device Block Handle
 * @param[in]  registerNum Start register number
 * @param[in]  dataLength Number of registers to be written
 * @param[in]  dataBuff Buffer containing data
 *
 * @retval NVMEDIA_STATUS_OK if succesful.
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
static inline NvMediaStatus WriteBlockVerifyMAX96717F(
    DevBlkCDIDevice const * const handle,
    uint16_t const registerNum,
    uint16_t const dataLength,
    uint8_t const * const dataBuff)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96717F const * const drvHandle = getHandlePrivMAX96717F(handle);
    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else if (((dataLength - (uint16_t)1U) > ((uint16_t)MAX_VALID_REG - registerNum))
        || (dataLength == 0U) || (dataLength > MAX_READ_WRITE_ARRAY_SIZE)) {
        SIPL_LOG_ERR_STR("Bad parameter: Invalid data length passed");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = DevBlkCDII2CPgmrWriteBlock(
            drvHandle->i2cProgrammer, registerNum, dataBuff, dataLength);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("DevBlkCDII2CPgmrWriteBlock failed with ", (uint32_t)status);
        } else {
            uint8_t readDataBuff[MAX_READ_WRITE_ARRAY_SIZE];
            status = DevBlkCDII2CPgmrReadBlock(
                drvHandle->i2cProgrammer, registerNum, dataLength, &readDataBuff[0]);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "DevBlkCDII2CPgmrReadBlock failed with ", (uint32_t)status);
            } else {
                /* Verify the value written */
                status = VerifyBlockReadVal(
                    dataBuff, &readDataBuff[0], registerNum, dataLength, true);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR_HEX_UINT("VerifyBlockReadVal failed with ", (uint32_t)status);
                }
            }
        }
    }

    return status;
}

/**
 * @brief Read a block of Uint8 data and readback to verify
 *
 * @param[in]  handle  CDI Device Block Handle
 * @param[in]  registerNum Start register number
 * @param[in]  dataLength Number of registers to be written
 * @param[out] dataBuff Buffer to store read data
 *
 * @retval NVMEDIA_STATUS_OK if succesful.
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
static inline NvMediaStatus ReadBlockVerifyMAX96717F(
    DevBlkCDIDevice const * const handle,
    uint16_t const registerNum,
    uint16_t const dataLength,
    uint8_t * const dataBuff)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96717F const * const drvHandle = getHandlePrivMAX96717F(handle);
    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else if (((dataLength - (uint16_t)1U) > ((uint16_t)MAX_VALID_REG - registerNum))
        || (dataLength == 0U) || (dataLength > MAX_READ_WRITE_ARRAY_SIZE)) {
        SIPL_LOG_ERR_STR("Bad parameter: Invalid data length passed");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = DevBlkCDII2CPgmrReadBlock(
            drvHandle->i2cProgrammer, registerNum, dataLength, dataBuff);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("DevBlkCDII2CPgmrWriteBlock failed with ", (uint32_t)status);
        } else {
            uint8_t readDataBuff[MAX_READ_WRITE_ARRAY_SIZE];
            status = DevBlkCDII2CPgmrReadBlock(
                drvHandle->i2cProgrammer, registerNum, dataLength, &readDataBuff[0]);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "DevBlkCDII2CPgmrReadBlock failed with ", (uint32_t)status);
            } else {
                /* Verify the value Read */
                status = VerifyBlockReadVal(
                    dataBuff, &readDataBuff[0], registerNum, dataLength, false);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR_HEX_UINT("VerifyBlockReadVal failed with ", (uint32_t)status);
                }
            }
        }
    }

    return status;
}

static NvMediaStatus
SetDeviceAddressMAX96717F(
    DevBlkCDIDevice const* handle,
    uint8_t address)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (address >= (0x80U)) { /* Check for 7 bit I2C address */
        SIPL_LOG_ERR_STR_HEX_UINT(
            "MAX96717F: Bad parameter: Address is greater than 0x80", address);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        uint8_t const data = lshift8(address, 1U);
        status = WriteUint8VerifyMAX96717F(handle, REG_REG0, data);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("max96717f_setAddr is failed", (uint32_t)status);
        }
    }
    return status;
}

/**
 * @brief Check I2C integrity status
 *
 * @param[in] drvHandle
 * @param[in] parameterType Type of the address tanslator to program
 * @param[in] source Source address.
 * @param[in] addr Register address.
 *
 * @retval NVMEDIA_STATUS_OK on successful completion
 * @retval NVMEDIA_STATUS_ERROR  on any other error
 */
static NvMediaStatus
I2CIntegrityChkMAX96717F(
    DevBlkCDIDevice const * const handle,
    uint32_t parameterType,
    uint8_t source,
    uint16_t addr)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (parameterType == (uint32_t)CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_A) {
        uint8_t data = 0U;
        status = ReadUint8VerifyMAX96717F(handle, addr, &data);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("Failed to read translator A register");
        } else {
            if (data != lshift16(source, 1U)) {
                SIPL_LOG_ERR_STR("Serializer I2C Data integrity checking failed");
                status = NVMEDIA_STATUS_ERROR;
            }
        }
    }

    return status;
}

static NvMediaStatus
SetTranslatorMAX96717F(
    DevBlkCDIDevice const* handle,
    uint32_t parameterType,
    uint8_t source,
    uint8_t destination)
{
    DevBlkCDII2CReg setTranslatorRegs[] = {
        {0x0042U, 0x00U},
        {0x0043U, 0x00U},
    };

    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if ((source >= 0x80U) || (destination >= 0x80U)) { /* Check for 7 bit I2C address*/
        SIPL_LOG_ERR_STR_2HEX("MAX96717F: Bad parameter: Source address,"
                              "destination address must be less than 0x80",
                              source, destination);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {

        if (parameterType == (uint32_t)CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B) {
            uint32_t tmpAddress = 0U;
            tmpAddress = saturatingAddUint32((uint32_t)setTranslatorRegs[0].address, (uint32_t)2U);
            setTranslatorRegs[0].address = (uint16_t)(tmpAddress & 0xFFFFU);

            tmpAddress = saturatingAddUint32((uint32_t) setTranslatorRegs[1].address, (uint32_t)2U);
            setTranslatorRegs[1].address = (uint16_t)(tmpAddress & 0xFFFFU);
        }

        setTranslatorRegs[0].data = lshift16(source, 1U);
        setTranslatorRegs[1].data = lshift16(destination, 1U);

        DevBlkCDII2CRegList max96717_setTranslator = {
            .regs = setTranslatorRegs,
            .numRegs = I2C_ARRAY_SIZE(setTranslatorRegs),
        };

        status = WriteArrayVerifyMAX96717F(handle, &max96717_setTranslator);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("WriteArrayVerifyMAX96717F failed", (uint32_t)status);
        } else {
            /* I2C Data integrity checking */
            status = I2CIntegrityChkMAX96717F(
                handle, parameterType, source, setTranslatorRegs[0].address);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR("I2C Integrity Check failed");
            }
        }
    }
    return status;
}

static NvMediaStatus
ConfigPipelinesMAX96717F(
    DevBlkCDIDevice const* handle,
    DataTypeMAX96717F dataType,
    bool embDataType)
{
    DevBlkCDII2CReg const mappingPixel12EMBRegs[] = {
        {0x0308U, 0x64U},
        {0x0311U, 0x40U},
        {0x0312U, 0x04U},
        {0x0110U, 0x60U},
        {0x031EU, 0x2CU},
        {0x0111U, 0x50U},
        {0x0318U, 0x6CU},
        {0x0319U, 0x52U},
        {0x02D5U, 0x07U},
        {0x02D8U, 0x08U},
    };
    DevBlkCDII2CRegList mappingPixel12EMB = {
        .regs = mappingPixel12EMBRegs,
        .numRegs = I2C_ARRAY_SIZE(mappingPixel12EMBRegs),
    };

    DevBlkCDII2CReg mappingPixel12Regs[] = {
        {0x0383U, 0x00U}, /* Disable tunneling mode */
        {0x0318U, 0x6CU}, /* RAW12 to pipe Z */
        {0x0313U, 0x40U}, /* Double 12-bit data on pipe Z */
        {0x031EU, 0x38U}, /* Pipe Z BPP = 24 */
        {0x0112U, 0x0AU}, /* Do not limit heartbeat in pipe Z */
    };
    DevBlkCDII2CRegList mappingPixel12 = {
        .regs = mappingPixel12Regs,
        .numRegs = I2C_ARRAY_SIZE(mappingPixel12Regs),
    };

    DevBlkCDII2CReg const mappingPixel10Regs[] = {
        {0x0308U, 0x64U},
        {0x0311U, 0x40U},
        {0x0312U, 0x04U},
        {0x0313U, 0x04U},
        {0x0110U, 0x60U},
        {0x031EU, 0x34U},
        {0x0318U, 0x6BU}
    };
    DevBlkCDII2CRegList mappingPixel10 = {
        .regs = mappingPixel10Regs,
        .numRegs = I2C_ARRAY_SIZE(mappingPixel10Regs),
    };

    DevBlkCDII2CReg const mappingPixel8Regs[] = {
        {0x0308U, 0x64U},
        {0x0311U, 0x40U},
        {0x0312U, 0x04U},
        {0x031EU, 0x30U},
        {0x0318U, 0x6AU},
        {0x0319U, 0x52U},
        {0x0383U, 0x00U}, /* Disable tunneling */
    };
    DevBlkCDII2CRegList mappingPixel8 = {
        .regs = mappingPixel8Regs,
        .numRegs = I2C_ARRAY_SIZE(mappingPixel8Regs),
    };

    NvMediaStatus status = NVMEDIA_STATUS_OK;


    /* for not used parameter */
    (void)embDataType;
    if (dataType == CDI_MAX96717F_DATA_TYPE_RAW8) {
        status = WriteArrayVerifyMAX96717F(handle, &mappingPixel8);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT(
                "WriteArrayVerifyMAX96717F failed with ", (uint32_t)status);
        }
    } else if (dataType == CDI_MAX96717F_DATA_TYPE_RAW10) {
        status = WriteArrayVerifyMAX96717F(handle, &mappingPixel10);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT(
                "WriteArrayVerifyMAX96717F failed with ", (uint32_t)status);
        }
    } else if (dataType == CDI_MAX96717F_DATA_TYPE_RAW12) {
        if (embDataType) {
            status = WriteArrayVerifyMAX96717F(handle, &mappingPixel12EMB);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "WriteArrayVerifyMAX96717F failed with ", (uint32_t)status);
            }
        } else {
            status = WriteArrayVerifyMAX96717F(handle, &mappingPixel12);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "WriteArrayVerifyMAX96717F failed with ", (uint32_t)status);
            }
        }
    } else {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return status;
}

static NvMediaStatus
MIPI_Reset(
    DevBlkCDIDevice const *handle)
{
    DriverHandleMAX96717F const* drvHandle = getHandlePrivMAX96717F(handle);
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        if(drvHandle->revisionID == CDI_MAX96717F_REV_4) {
            status = SetRegBitMAX96717F(handle, 0x330U, 3U, true);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT("SetRegBit failed", (uint32_t)status);
            } else {
                status = SetRegBitMAX96717F(handle, 0x330U, 3U, false);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR_HEX_UINT("SetRegBit failed", (uint32_t)status);
                }
            }
        }
    }

    return status;
}


static NvMediaStatus
ConfigPhyMap(
    DevBlkCDIDevice const* handle,
    phyMapMAX96717F const* mapping,
    phyPolarityMAX96717F const* polarity,
    uint8_t numDataLanes)
{
    DevBlkCDII2CReg phyMapRegs[] = {
        {0x0330U, 0x00U},
        {0x0331U, 0x00U},
        {0x0332U, 0xEEU},
        {0x0333U, 0xE4U},
    };
    DevBlkCDII2CRegList phyMap = {
        .regs = phyMapRegs,
        .numRegs = I2C_ARRAY_SIZE(phyMapRegs),
    };

    DevBlkCDII2CReg phyPolarityRegs[] = {
        {0x0334U, 0x00U},
        {0x0335U, 0x00U}
    };
    DevBlkCDII2CRegList phyPolarity = {
        .regs = phyPolarityRegs,
        .numRegs = I2C_ARRAY_SIZE(phyPolarityRegs),
    };

    uint8_t regVal = 0U;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (NULL == mapping) {
        SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else if ((numDataLanes != 2U) && (numDataLanes != 4U)) {
        SIPL_LOG_ERR_STR_UINT("MAX96717F: numDataLanes Valid vals are 2 and 4",
                              numDataLanes);
        status = NVMEDIA_STATUS_NOT_SUPPORTED;
    } else if ((numDataLanes == 2U) && (mapping->enableMapping)) {
        SIPL_LOG_ERR_STR("MAX96717F: Lane swapping is supported only in 4 lane mode");
        status = NVMEDIA_STATUS_NOT_SUPPORTED;
    } else {
        status = ReadUint8VerifyMAX96717F(handle, phyMapRegs[1].address, &regVal);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_2HEX(
                "RegRead failed addr: data:", (uint32_t)(phyMapRegs[1].address), regVal);
        } else {
            /* data lanes indexing starts at 0 (0 = 1 lane, 1 = 2 lanes, etc) */
            uint8_t numLanes = numDataLanes - 1U;
            /* clear the data lanes settings for Port B */
            regVal &= (uint8_t)(~0x30U & 0xFFU);
            /* Set num data lanes for Port B */
            regVal |= lshift8(numLanes, 4U);
            phyMapRegs[1].data = regVal;

           if (mapping->enableMapping) {
                regVal = lshift8(mapping->phy1_d1, 6U) |
                            lshift8(mapping->phy1_d0, 4U) |
                            lshift8(mapping->phy0_d1, 2U) |
                            mapping->phy0_d0;
                phyMapRegs[2].data = regVal;

                regVal = lshift8(mapping->phy3_d1, 6U) |
                            lshift8(mapping->phy3_d0, 4U) |
                            lshift8(mapping->phy2_d1, 2U) |
                            mapping->phy2_d0;
                phyMapRegs[3].data = regVal;
            }

            if (polarity->setPolarity) {
                phyPolarityRegs[0].data = lshift16((uint16_t)polarity->phy1_clk, 6U) |
                                            lshift16((uint16_t)polarity->phy1_d1, 5U) |
                                            lshift16((uint16_t)polarity->phy1_d0, 4U);
                phyPolarityRegs[1].data = lshift16((uint16_t)polarity->phy2_clk, 2U) |
                                            lshift16((uint16_t)polarity->phy2_d1, 1U) |
                                            (uint16_t)polarity->phy2_d0;
                status = WriteArrayVerifyMAX96717F(handle, &phyPolarity);
                if (status != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR_INT("MAX96717F: Set phy polarity write failed", (int32_t)status);
                    return status;
                }
            }
            status = WriteArrayVerifyMAX96717F(handle, &phyMap);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT("WriteArrayVerifyMAX96717F failed", (uint32_t)status);
            }
        }
    }
    return status;
}

/**
 * @brief Check DPLL Lock status.
 *
 * @param[in] handle  CDI handle to the serializer.
 * @param[out] lockStatus DPLL lock status. Must be initialized to false by the caller.
 *                        If DPLL is locked, set to true.
 *
 * @retval NVMEDIA_STATUS_OK if successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if invalid parameter passed.
 * @retval NVMEDIA_STATUS_ERR in case of failure.
 */
static NvMediaStatus
IsDPLLLock(
    DevBlkCDIDevice const * const handle,
    bool *lockStatus)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regVal = 0U;
    bool exitFlag = false;
    uint32_t cnt = 0;

     if(lockStatus == NULL) {
        SIPL_LOG_ERR_STR("IsDPLLLock: Bad parameter passed.");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        /* Check DPLL Lock status here */
        do {
            status = ReadUint8VerifyMAX96717F(handle, REG_REF_VTG0, &regVal);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT("ReadUint8VerifyMAX96717F failed", (uint32_t)status);
                exitFlag = true;
            } else {
                if (GetBit(regVal, DPLL_LOCK_UNLOCKED_BIT) > 0U) {
                    *lockStatus = true;
                    exitFlag = true;
                } else {
                    /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
                    (void)nvsleep(DPLL_LOCK_CHECK_RETRY_SLEEP_US);
                }
            }
        } while(((cnt++)<DPLL_LOCK_CHECK_RETRY_CNT) && !exitFlag);
    }

    return status;
}

static NvMediaStatus
GenerateClock(
    DevBlkCDIDevice const* handle,
    uint8_t freq)
{
    DevBlkCDII2CReg const genClockRegs_25MHz[] = {
        {0x03F1U, 0x05U},
        {0x03F0U, 0x12U},
        {0x03F4U, 0x0AU},
        {0x03F5U, 0x07U},
        {0x03F0U, 0x10U},
        {0x1A03U, 0x12U},
        {0x1A07U, 0x04U},
        {0x1A08U, 0x3DU},
        {0x1A09U, 0x40U},
        {0x1A0AU, 0xC0U},
        {0x1A0BU, 0x7FU},
        {0x03F0U, 0x11U},
    };
    DevBlkCDII2CRegList genClock_25MHz = {
        .regs = genClockRegs_25MHz,
        .numRegs = I2C_ARRAY_SIZE(genClockRegs_25MHz),
    };

    DevBlkCDII2CReg genClockRegs_24MHz[] = {
        {0x0003U, 0x03U},
        {0x0006U, 0xB0U},
        {0x03F0U, 0x59U},
        {0x0570U, 0x0CU},
    };
    DevBlkCDII2CRegList genClock_24MHz = {
        .regs = genClockRegs_24MHz,
        .numRegs = I2C_ARRAY_SIZE(genClockRegs_24MHz),
    };

    NvMediaStatus status = NVMEDIA_STATUS_OK;

    PrintLogMsg(LOG_LEVEL_DEBUG, "MAX96717F: Generate Clock\n");
    if ((INCK_24MHZ != freq) && (INCK_25MHZ != freq)) {
        SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Not supported clock rate");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        if (INCK_24MHZ == freq) {
            status = WriteArrayVerifyMAX96717F(handle, &genClock_24MHz);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "GenerateClock: GetClock 24MHz failed :", (uint32_t)status);
            }
        } else {
            status = WriteArrayVerifyMAX96717F(handle, &genClock_25MHz);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "GenerateClock: GetClock 25MHz failed :", (uint32_t)status);
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            /* Check DPLL Lock status here */
            bool lockStatus = false;
            status = IsDPLLLock(handle, &lockStatus);
            if (status == NVMEDIA_STATUS_OK) {
                if (!lockStatus) {
                    status = NVMEDIA_STATUS_ERROR;
                }
            } else {
                SIPL_LOG_ERR_STR("GenerateClock: DPLL Lock status check failed");
            }
        }
    }
    return status;
}

static NvMediaStatus
EnablePClkPIOSlew(
    DevBlkCDIDevice const* handle)
{
    /* Clear all reset bits and maintain default values for
     * the rest, including the Reserved bits */
    DevBlkCDII2CReg const setLinkMapsRegs[] = {
        {0x0010U, 0x11U},
    };
    DevBlkCDII2CRegList setLinkMaps = {
        .regs = setLinkMapsRegs,
        .numRegs = I2C_ARRAY_SIZE(setLinkMapsRegs),
    };

    DevBlkCDII2CReg const pioPClkSlewRegs[] = {
        {0x0570U, 0x00U},
        {0x03F1U, 0x89U},
        {0x056FU, 0x00U},
    };
    DevBlkCDII2CRegList pioPClkSlew = {
        .regs = pioPClkSlewRegs,
        .numRegs = I2C_ARRAY_SIZE(pioPClkSlewRegs),
    };

    DevBlkCDII2CReg const pipeZRaw12Regs[] = {
        {0x0318U, 0x6CU},
    };
    DevBlkCDII2CRegList pipeZRaw12 = {
        .regs = pipeZRaw12Regs,
        .numRegs = I2C_ARRAY_SIZE(pipeZRaw12Regs),
    };

    /* Enable links before mapping the PHYs */
    NvMediaStatus status = WriteArrayVerifyMAX96717F(handle, &setLinkMaps);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_HEX_UINT("Error configuring PHY Links failed", (uint32_t)status);
    } else {
        status = WriteArrayVerifyMAX96717F(handle, &pioPClkSlew);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT(
                "Error configuration pioPClkSle failed", (uint32_t)status);
        } else {
            status = WriteArrayVerifyMAX96717F(handle, &pipeZRaw12);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "Error pipZRaw12 configuration failed", (uint32_t)status);
            }
        }
    }
    return status;
}

static NvMediaStatus
SetGPIOOutputMAX96717F(
    DevBlkCDIDevice const* handle,
    GPIOTypeMAX96717F gpio,
    bool level)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if ((gpio == CDI_MAX96717F_GPIO_TYPE_INVALID) || (gpio >= CDI_MAX96717F_GPIO_TYPE_NUM)) {
        SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Invalid GPIO pin");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        uint16_t addrGPIOA = (uint16_t)GET_GPIO_A_ADDR((uint32_t)gpio - 1U);
        uint8_t data = 0x80U;

        if (level) {
            data |= bit8(4U);
        }

        status = WriteUint8VerifyMAX96717F(handle, addrGPIOA, data);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("MAX96717F: Failed to write to GPIO_A register: ", addrGPIOA);
        }
    }
    return status;
}

static NvMediaStatus
SetFsyncGPIOMAX96717F(
    DevBlkCDIDevice const* handle,
    GPIOTypeMAX96717F gpio,
    uint8_t rxID)
{
    DevBlkCDII2CReg setGPIOModeRegs[] = {
        {0x02BEU, 0x84U},
        {0x02BFU, 0xA0U},
        {0x02C0U, 0x40U},
    };
    DevBlkCDII2CRegList setGPIOMode = {
        .regs = setGPIOModeRegs,
        .numRegs = I2C_ARRAY_SIZE(setGPIOModeRegs),
    };

    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if ((gpio == CDI_MAX96717F_GPIO_TYPE_INVALID) || (gpio >= CDI_MAX96717F_GPIO_TYPE_NUM)) {
        SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Invalid GPIO pin");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        /* unused rxId */
        (void)rxID;
        /* Note: GPIO ID value i.e. "gpio" is already increased by 1 because
        * of the way "GPIOTypeMAX96717F" enumerates MFPs. So account for it by
        * subtracting 1 below while calculating the right register address. */
        uint32_t const GPIOIndex = ((uint32_t) gpio - 1U) * MAX96717F_NUM_REGS_PER_GPIO_PIN;
        uint32_t tmpAddr = 0U;
        tmpAddr = saturatingAddUint32(setGPIOModeRegs[0].address,
                                      GPIOIndex);
        setGPIOModeRegs[0].address = (uint16_t)(tmpAddr & 0xFFFFU);

        tmpAddr = saturatingAddUint32(setGPIOModeRegs[1].address,
                                      GPIOIndex);
        setGPIOModeRegs[1].address = (uint16_t)(tmpAddr & 0xFFFFU);
        setGPIOModeRegs[1].data = (uint16_t)((setGPIOModeRegs[1].data | rxID) & 0xFFFFU);

        tmpAddr = saturatingAddUint32(setGPIOModeRegs[2].address,
                                      GPIOIndex);
        setGPIOModeRegs[2].address = (uint16_t)(tmpAddr & 0xFFFFU);
        setGPIOModeRegs[2].data = (uint16_t)((setGPIOModeRegs[2].data | rxID) & 0xFFFFU);

        PrintLogMsg(LOG_LEVEL_DEBUG, "WriteArray 0x%04x 0x%x\n",
                    setGPIOModeRegs[0].address,
                    setGPIOModeRegs[0].data);
        status = WriteArrayVerifyMAX96717F(handle, &setGPIOMode);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("WriteArrayVerifyMAX96717F failed", (uint32_t)status);
        }

        /* sleep 10ms */
        /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
        (void)nvsleep(10000);
    }

    return status;
}

static NvMediaStatus
EnableRefClockMAX96717F(
    DevBlkCDIDevice const* handle,
    GPIOTypeMAX96717F gpio,
    bool enableRClk)
{
    DevBlkCDII2CReg enablePCLKOutRegs[] = {
        {0x03F1U, 0x00U},
    };
    DevBlkCDII2CRegList enablePCLKOut = {
        .regs = enablePCLKOutRegs,
        .numRegs = I2C_ARRAY_SIZE(enablePCLKOutRegs)
    };

    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if ((gpio == CDI_MAX96717F_GPIO_TYPE_INVALID) || (gpio >= CDI_MAX96717F_GPIO_TYPE_NUM)) {
        SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Invalid GPIO pin");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        if (enableRClk) {
            enablePCLKOutRegs[0].data |= 0x80U;
        }
        enablePCLKOutRegs[0].data |= lshift16((uint16_t)gpio - 1U, 1U) | 0x1U;
        status = WriteArrayVerifyMAX96717F(handle, &enablePCLKOut);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("WriteArrayVerifyMAX96717F failed", (uint32_t)status);
        }
    }
    return status;
}

static NvMediaStatus
ForwardGPIOMAX96717F(
    DevBlkCDIDevice const* handle,
    uint8_t srcGpio,
    uint8_t dstGpio)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint16_t addrGPIOA = (uint16_t)GET_GPIO_A_ADDR(srcGpio);
    uint16_t addrGPIOB = addrGPIOA + (uint16_t)1U;
    uint8_t regVal = 0U;

    if ((srcGpio > 10U ) || (dstGpio > 31U)) {
         SIPL_LOG_ERR_STR_2UINT("MAX96717f: Bad parameter srcGpio, dstGpio", srcGpio, dstGpio);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = ReadUint8VerifyMAX96717F(handle, addrGPIOA, &regVal);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("ReadUint8VerifyMAX96717F failed", (uint32_t)status);
        } else {
            regVal |= 0x3U; /* Set GPIO_TX_EN, GPIO_OUT_DIS */
            regVal &= ~bit8(2U); /* Unset GPIO_RX_EN */

            status = WriteUint8VerifyMAX96717F(handle, addrGPIOA, regVal);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT("WriteUint8VerifyMAX96717F failed", (uint32_t)status);
            } else {
                status = ReadUint8VerifyMAX96717F(handle, addrGPIOB, &regVal);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR_HEX_UINT("ReadUint8VerifyMAX96717F failed", (uint32_t)status);
                } else {
                    regVal &= 0xE0U;
                    /* MFP3 of the deserializer is ERRB signal
                     * In this case, use 31 for TX ID of srcGpio of the serilaizer
                     * 31 is ERR_TX_ID
                     */
                    if (dstGpio == 3U) {
                        regVal |= (31U & 0x1FU);
                    } else {
                        regVal |= (dstGpio & 0x1FU);
                    }

                    status = WriteUint8VerifyMAX96717F(handle, addrGPIOB, regVal);
                    if (status != NVMEDIA_STATUS_OK) {
                        SIPL_LOG_ERR_STR_HEX_UINT(
                            "WriteUint8VerifyMAX96717F failed", (uint32_t)status);
                    }
                }
            }
        }
    }
    return status;
}

static NvMediaStatus
GetRevIdMAX96717F(
    DevBlkCDIDevice  const* handle,
    uint8_t devID,
    RevisionMAX96717F *rev)
{
    /* These values must include all of values in the RevisionMAX96717F enum */
    static const RevMAX96717F supportedRevisionsMAX96717F[] = {
        {CDI_MAX96717F_REV_2, 2U},
        {CDI_MAX96717F_REV_4, 4U},
    };

    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint32_t numRev = I2C_ARRAY_SIZE(supportedRevisionsMAX96717F);
    uint8_t rvsn = 0U;
    uint32_t i = 0U;

    status = ReadUint8VerifyMAX96717F(handle, REG_DEV_REV_ADDR, &rvsn);
    if (status == NVMEDIA_STATUS_OK) {
        rvsn &= 0x0FU;
        for (i = 0U; i < numRev; i++) {
            if (rvsn == supportedRevisionsMAX96717F[i].revVal) {
                *rev = supportedRevisionsMAX96717F[i].revId;
                PrintLogMsg(LOG_LEVEL_NONE,"%s: Revision %u detected!\n",
                            (devID == MAX96717F_DEV_ID_C8) ? "MAX96717F" : "MAX96717", rvsn);
                status = NVMEDIA_STATUS_OK;
                break;
            }
        }
        if(i == numRev) {
            SIPL_LOG_ERR_STR_UINT("MAX96717F: Unsupported MAX96717F revision detected!"
                                  "Supported revisions are:", rvsn);
            for (i = 0u; i < numRev; i++) {
                PrintLogMsg(LOG_LEVEL_NONE,"MAX96717F: Revision %u\n",
                            supportedRevisionsMAX96717F[i].revVal);
            }
        }
    }
    return status;
}

static NvMediaStatus
DriverCreateMAX96717F(
    DevBlkCDIDevice *handle,
    void const* clientContext)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96717F *drvHandle = NULL;

    if (NULL == handle) {
        SIPL_LOG_ERR_STR("MAX96717f: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else if (clientContext != NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Context must not be supplied");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        /* allocation MAX96717F drvHandle */
        /* coverity[misra_c_2012_rule_11_5_violation] : intentional TID-1417 */
        /* coverity[misra_c_2012_rule_21.3_violation] : intentional TID-1493 */
        drvHandle = calloc(1, sizeof(*drvHandle));
        if (NULL == drvHandle) {
            status =  NVMEDIA_STATUS_OUT_OF_MEMORY;
            SIPL_LOG_ERR_STR_HEX_UINT("memory allocation failed",
                                      (uint32_t)status);
        } else {
            handle->deviceDriverHandle = (void *)drvHandle;

            /* Create the I2C programmer for register read/write */
            drvHandle->i2cProgrammer = DevBlkCDII2CPgmrCreate(handle,
                                            MAX96717F_NUM_ADDR_BYTES,
                                            MAX96717F_NUM_DATA_BYTES);

            if (drvHandle->i2cProgrammer == NULL) {
                SIPL_LOG_ERR_STR("Failed to initialize the I2C Programmer");
                status = NVMEDIA_STATUS_ERROR;
            }
        }
    }
    return status;
}

static NvMediaStatus
DriverDestroyMAX96717F(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96717F const* drvHandle = getHandlePrivMAX96717F(handle);

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96717f: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        /* Destroy the I2C Programmer */
        DevBlkCDII2CPgmrDestroy(drvHandle->i2cProgrammer);
        /* coverity[misra_c_2012_rule_21.3_violation] : intentional TID-1493 */
        free(handle->deviceDriverHandle);
        handle->deviceDriverHandle = NULL;
    }
    return status;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#ifdef NVMEDIA_QNX
/**
 * @brief configure to ADC Power-on satus
 *
 * @param[in] handle CDI handle to the serializer.
 *
 * @retval NVMEDIA_STATUS_OK success to power-on status.
 * @retval NVMEDIA_STATUS_ERROR in case of failure.
 */
static NvMediaStatus
ADCPoweron(
    DevBlkCDIDevice const *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t adcStatus = 0U;

    DevBlkCDII2CReg const ADCTestRegs1[] = {
        {0x0500U, 0x00U, 1000U}, /* ADC power off */
        {0x001EU, 0xF5U},
        {0x0501U, 0x08U},
        {0x050CU, 0x03U},
        {0x0500U, 0x1EU}, /* ADC Power on */
    };

    DevBlkCDII2CRegList ADCTest1 = {
        .regs = ADCTestRegs1,
        .numRegs = I2C_ARRAY_SIZE(ADCTestRegs1)
    };

    status = WriteArrayVerifyMAX96717F(handle, &ADCTest1);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR("ADCBIST power-on failed");
    } else {
        for(uint8_t i=0U; i < MAX_RETRY_COUNT; i++) {
            status = ReadUint8VerifyMAX96717F(handle, REG_ADC_INTR0, &adcStatus);
            if (NVMEDIA_STATUS_OK == status) {
                if (GetBit(adcStatus, 1) > 0U) {
                    PrintLogMsg(LOG_LEVEL_INFO, "adc_ref_ready_if is set\n");
                    break;
                } else {
                    PrintLogMsg(LOG_LEVEL_INFO, "adc_ref_read_if is not set\n");
                    /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
                    (void)nvsleep(ADC_DONE_IF_SLEEP_US);
                }
            } else {
                SIPL_LOG_ERR_STR("Read ADC_INTR0 failed");
            }
        }
    }

    return status;
}

static inline bool
waitForADCDoneIf(
    DevBlkCDIDevice const *handle)
{
    NvMediaStatus status;
    bool adcDoneIf = false;
    for (uint8_t i=0U; i < 10U; i++) {
        uint8_t adcStatus = 0U;
        status = ReadUint8VerifyMAX96717F(handle, REG_ADC_INTR0, &adcStatus);
        if (NVMEDIA_STATUS_OK == status) {
            if (GetBit(adcStatus, 0U) > 0U) {
                PrintLogMsg(LOG_LEVEL_INFO, "adc_done_if is done\n");
                adcDoneIf = true;
                break;
            } else {
                /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
                (void)nvsleep(ADC_DONE_IF_SLEEP_US);
            }
        }
    }
    return adcDoneIf;
}

/**
 * @brief Set to ADC-BIST run status.
 *
 * @param[in] handle CDI handle to the serializer.
 * @param[in] adcChannel adc Channel index(range : 0 - 2)
 *
 * @retval NVMEDIA_STATUS_OK success
 * @retval NVMEDIA_STATUS_BAD_PARAMETER adcChannel value out of range.
 * @retval NVMEDIA_STATUS_ERROR in case of failure.
 */
static NvMediaStatus
ADCBISTConfig(
    DevBlkCDIDevice const *handle,
    uint8_t adcChannel)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t rRegVal[2] = {0U};

    DevBlkCDII2CReg ADCTestRegs2[] = {
        {0x053EU, 0x01U}, /* ADC channel */
        {0x0502U, 0x01U},
        {0x0501U, 0x08U},
        {0x0509U, 0x80U},
        {0x1D28U, 0x10U},
        {0x1D37U, 0xFEU},
        {0x0500U, 0x1FU},
    };

    DevBlkCDII2CRegList ADCTest2 = {
        .regs = ADCTestRegs2,
        .numRegs = I2C_ARRAY_SIZE(ADCTestRegs2)
    };

    if (adcChannel > 0U) {
        uint8_t data_t = 0U;
        /* Set ADC channel */
        data_t = SetBit(0U, adcChannel);
        ADCTestRegs2[0].data = (uint16_t)data_t;

        data_t = (uint8_t)(ADCTestRegs2[2].data & 0xFFU);
        data_t = SetBit(data_t, ((adcChannel+3U) & 0xFFU));
        ADCTestRegs2[2].data = (uint16_t)data_t;
        data_t = (uint8_t)(((adcChannel*2U) -1U) & 0xFFU);

        if (saturatingSubtractUint32((uint32_t)ADCTestRegs2[5].data,
                                     (uint32_t)data_t) < (uint32_t)UINT8_MAX) {
            ADCTestRegs2[5].data =
            (uint16_t)(saturatingSubtractUint32((uint32_t)ADCTestRegs2[5].data,
                                                (uint32_t)data_t) & 0xFFFFU);
        } else {
            SIPL_LOG_ERR_STR("Register value is out of range");
            status = NVMEDIA_STATUS_ERROR;
        }
        PrintLogMsg(LOG_LEVEL_DEBUG, "0x%x: ADCTestRegs2[0].data = 0x%x\n",
                    ADCTestRegs2[0].address,
                    ADCTestRegs2[0].data);
        PrintLogMsg(LOG_LEVEL_DEBUG, "0x%x: ADCTestRegs2[2].data = 0x%x\n",
                    ADCTestRegs2[2].address,
                    ADCTestRegs2[2].data);
        PrintLogMsg(LOG_LEVEL_DEBUG, "0x%x: ADCTestRegs2[5].data = 0x%x\n",
                    ADCTestRegs2[5].address,
                    ADCTestRegs2[5].data);
    }

    if (NVMEDIA_STATUS_OK == status) {
        /* Writing ADC Setting2 */
        status = WriteArrayVerifyMAX96717F(handle, &ADCTest2);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("WriteArrayVerifyMAX96717F failed", (uint32_t)status);
        } else {
            bool adcDoneIf = false;
            /* waiting until adc_done_if = 1 */
            adcDoneIf = waitForADCDoneIf(handle);

            if (adcDoneIf == true) {
                /* Read 0x508 and 0x509 */
                status = MAX96717FReadRegistersVerify(handle, REG_ADC_DATA0, 2U, &rRegVal[0]);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR("ADC_DATA read failed");
                } else {
                    uint16_t adcData = 0U;

                    adcData |= (uint16_t)(rRegVal[0]);
                    adcData |= lshift16(rRegVal[1], 8U) & 0x03FFU;
                    if (adcData > 0x000FU) {
                        SIPL_LOG_ERR_STR_HEX_UINT("Should need within 15codes 0x000", adcData);
                        status = NVMEDIA_STATUS_ERROR;
                    } else {
                        PrintLogMsg(LOG_LEVEL_DEBUG, "ADC DATA : 0x%x\n", adcData);
                    }
                }
            } else {
                SIPL_LOG_ERR_STR("adc_done_if is not set");
                status = NVMEDIA_STATUS_ERROR;
            }
        }
    }

    return status;
}

/**
 * @brief 3rd ADC-BIST configuration.
 *
 * @param[in] handle CDI handle to the serializer.
 * @param[in] adcChannel adc Channel index(range : 0 - 2)
 *
 * @retval NVMEDIA_STATUS_OK success
 * @retval NVMEDIA_STATUS_BAD_PARAMETER adcChannel value out of index.
 * @retval NVMEDIA_STATUS_ERROR in case of failure.
 */
static NvMediaStatus
ADCSBISTStep3(
    DevBlkCDIDevice const *handle,
    uint8_t adcChannel)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t rRegVal[2] = {0U};

    DevBlkCDII2CReg ADCTestRegs3[] = {
        {0x1D37U, 0x01U},
        {0x0500U, 0x1FU},
    };

    DevBlkCDII2CRegList ADCTest3 = {
        .regs = ADCTestRegs3,
        .numRegs = I2C_ARRAY_SIZE(ADCTestRegs3)
    };

    if (adcChannel > 0U) {
        uint16_t data_t = 0U;
        /* Move Next ADC channel */
        data_t = SetBit(0U, adcChannel);
        ADCTestRegs3[0].data = (uint16_t)data_t;
        PrintLogMsg(LOG_LEVEL_DEBUG, "0x%x: ADCTestRegs3[0].data = 0x%x\n",
                    ADCTestRegs3[0].address,
                    ADCTestRegs3[0].data);
    }

    /* Writing 3rd list */
    status = WriteArrayVerifyMAX96717F(handle, &ADCTest3);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_HEX_UINT("WriteArrayVerify failed", (uint32_t)status);
    } else {
        uint16_t adcData = 0U;
        bool adcDoneIf = false;

        /* waiting until adc_done_if = 1 */
        adcDoneIf = waitForADCDoneIf(handle);

        if (adcDoneIf == true) {
            /* Read 0x508 and 0x509 */
            status = MAX96717FReadRegistersVerify(handle, REG_ADC_DATA0, 2U, &rRegVal[0]);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR("ADC_DATA0 read failed");
            } else {
                adcData |= lshift16(rRegVal[1], 8U);
                adcData |= (uint16_t)(rRegVal[0]);
                if (adcData < 0x03F0U) {
                    SIPL_LOG_ERR_STR_HEX_UINT("Should need within 15codes 0x3FF",
                                                adcData);
                    status = NVMEDIA_STATUS_ERROR;
                } else {
                    /* Final Setting */
                    uint8_t wRegVal = 0x00U;
                    status = WriteUint8VerifyMAX96717F(handle, REG_REGADCBIST0, wRegVal);
                    if (NVMEDIA_STATUS_OK != status) {
                        SIPL_LOG_ERR_STR("Reg Write failed");
                    }
                }
            }
        } else {
            SIPL_LOG_ERR_STR("adc_done_if is not set");
            status = NVMEDIA_STATUS_ERROR;
        }
    }

    return status;
}

/**
 * @brief Run ADC-BIST
 *
 * @param[in] handle CDI handle to the serializer.
 *
 * @retval NVMEDIA_STATUS_OK success
 * @retval NVMEDIA_STATUS_ERROR in case of failure.
 */
static NvMediaStatus
RunADCBIST(
    DevBlkCDIDevice const *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    for (uint8_t i = 0U; i < MAX_ADC_CHANNEL_NUM; i++) {
        /* ADC Power on */
        status = ADCPoweron(handle);

        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("ADCPower on failed");
        } else {
            /* ADC Configuration */
            status = ADCBISTConfig(handle, i);

            if (NVMEDIA_STATUS_OK != status){
                SIPL_LOG_ERR_STR("ADCBIST Config failed");
            } else {
                /* ADC-BIST step3 */
                status = ADCSBISTStep3(handle, i);

                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR("ADC-BIST step3 failed");
                }
            }
        }
    }

    return status;
}

/**
 * @brief Check ADC-BIST result status.
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice.
 *
 * @retval NVMEDIA_STAUTS_OK ADC-BIST passed.
 * @retval NVMEDIA_STATUS_ERROR ADC-BIST is reported to failed
 *                              or any system error.
 */
static NvMediaStatus
CheckADCBISTStatus(
    DevBlkCDIDevice const *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regVal = 0U;

    status = RunADCBIST(handle);
    if (NVMEDIA_STATUS_OK != status) {
    } else {
        status = ReadUint8VerifyMAX96717F(handle, REG_ADC_INTR0, &regVal);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("Reg ADC_INTR0 read failed");
        } else {
            /* Check adc_hi_limit_if */
            if (GetBit(regVal, 2U) == 0U) {
                PrintLogMsg(LOG_LEVEL_INFO, "adc_hi_limit_if is not set\n");
                /* Check adc_lo_limit_if */
                if (GetBit(regVal, 3U) == 0U) {
                    PrintLogMsg(LOG_LEVEL_INFO, "adc_lo_limit_if is not set");

                    /* Checking ADC_INT_FLAG */
                    status = ReadUint8VerifyMAX96717F(handle, REG_INTR7, &regVal);
                    if (NVMEDIA_STATUS_OK != status) {
                        SIPL_LOG_ERR_STR("ADC-BIST ADC_INT_FLAG read failed");
                    } else {
                        if (GetBit(regVal, 2U) == 0U) {
                            PrintLogMsg(LOG_LEVEL_INFO, "No ADC interrupt\n");
                        } else {
                            SIPL_LOG_ERR_STR("ADC interrupt set");
                            status = NVMEDIA_STATUS_ERROR;
                        }
                    }
                } else {
                    SIPL_LOG_ERR_STR("adc_lo_limit_if is set");
                    status = NVMEDIA_STATUS_ERROR;
                }
            } else {
                SIPL_LOG_ERR_STR("adc_hi_limit_if is set");
                status = NVMEDIA_STATUS_ERROR;
            }
        }
    }

    return status;
}

/**
 * @brief Check both LBIST and MBIST status
 *
 * POST_LBIST_PASSED and POST_MBIST_PASSED should be high
 * or this is an error. And POST_DONE for completion.
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice.
 *
 * @retval NVMEDIA_STATUS_OK both LBIST and MBIST passed.
 * @retval NVMEDIA_STATUS_ERROR Either LBIST / MBIST is failed or any system error
 */
static NvMediaStatus
CheckBISTStatus(
    DevBlkCDIDevice const* handle)
{
    uint8_t regVal = 0U;
    NvMediaStatus status = ReadUint8VerifyMAX96717F(handle, REG_POST0, &regVal);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_HEX_UINT("Read REG_POST0 failed", (uint32_t)status);
    } else {
        /* SM18 and 19 ask an error to be returned immediately if POST_DONE
         * is not already set. No need for additional timeout. */
        if (GetBit(regVal, 7U) > 0U) {
            PrintLogMsg(LOG_LEVEL_INFO, "POST has run\n");
            if (GetBit(regVal, 5U) > 0U) {
                PrintLogMsg(LOG_LEVEL_INFO, "LBIST is done\n");
            } else {
                status = NVMEDIA_STATUS_ERROR;
            }
            if (GetBit(regVal, 6U) > 0U) {
                PrintLogMsg(LOG_LEVEL_INFO, "MBIST is done\n");
            } else {
                status = NVMEDIA_STATUS_ERROR;
            }
        } else {
            PrintLogMsg(LOG_LEVEL_INFO, "POST has not run\n");
            status = NVMEDIA_STATUS_ERROR;
        }
    }

    return status;
}

/**
 * @brief Check Link Lock and ERRB status in forward channel.
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice.
 * @param[out] lock Lock status is true, unLock status is false.
 * @param[out] errb ERRB asserted is true, not asserted is false.
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUAS_ERROR on any other error
 */
static NvMediaStatus
ChkLockErrbStatusMAX96717F(
    DevBlkCDIDevice const *handle,
    bool *lock,
    bool *errb)
{
    uint8_t regVal = 0U;
    NvMediaStatus status = ReadUint8VerifyMAX96717F(handle, REG_CTRL3, &regVal);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96717FChkLockStatus: Regiser read fail", (uint32_t)status);
    } else {
        /* Check Link Lock indicator */
        if (GetBit(regVal, MAX96717F_LOCK_INDICATOR_BIT) == 0x08U) {
            *lock = (bool)MAX96717F_LINK_LOCKED;
        } else {
            *lock = (bool)MAX96717F_LINK_UNLOCKED;
        }

        /* ERRB indicator */
        if (GetBit(regVal, MAX96717F_ERRB_INDICATOR_BIT) == 0x04U) {
            *errb = (bool)MAX96717F_ERRB_ASSERTED;
        } else {
            *errb = (bool)MAX96717F_ERRB_NOT_ASSERTED;
        }
    }
    return status;
}

/**
 * @brief Run ADC-BIST diagonal test
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice.
 * @param[out] lockStatus PLL lock status.
 * @param[out] errbStatus ERRB status.
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUAS_ERROR on any other error
 */
static NvMediaStatus
RunADCBISTMAX96717F(
    DevBlkCDIDevice const* handle,
    bool *lockStatus,
    bool *errbStatus)
{
    /* LBIST & MBIST check here */
    NvMediaStatus status = CheckBISTStatus(handle);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR("BIST check is failed");
    } else {
        /* Check ADC-BIST status */
        status = CheckADCBISTStatus(handle);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("ADC-BIST is failed");
        } else {
            bool tmpLockStats = false;
            bool tmpERRBStats = false;

            /* Check Link Lock status */
            for (uint8_t i=0U; i < MAX_LOCK_CHECK_RETRY_CNT; i++) {
                status = ChkLockErrbStatusMAX96717F(handle, &tmpLockStats, &tmpERRBStats);
                if (NVMEDIA_STATUS_OK == status) {
                    if (tmpLockStats == true) {
                        *lockStatus = tmpLockStats;
                        *errbStatus = tmpERRBStats;
                    } else {
                        /* 4ms */
                        /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
                        (void)nvsleep(UNLOCK_STATUS_SLEEP_US);
                        status = NVMEDIA_STATUS_ERROR;
                    }
                }

                if ((tmpLockStats == true)) {
                    break;
                }
            }
        }
    }

    /* Writing 0x53E to 0x0 for MFP6 to GPIO */
    if (NVMEDIA_STATUS_OK == status) {
        uint8_t regVal = 0x0U;
        status = WriteUint8VerifyMAX96717F(handle, REG_ADC_CTRL4, regVal);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("REG_ADC_CTRL4 writing fail");
        }
    }

    return status;
}
#endif

static NvMediaStatus
EnableRXTXCRCMAX96717F(
    DevBlkCDIDevice const *handle
)
{
    /* Ensure TX/RX_CRC_EN bits are set by default in
       the CFG registers for INFOFR, GPIO and I2C */
    DevBlkCDII2CReg TX_CRC_EN_Regs[] = {
        {0x0078U, 0x00U},
        {0x0090U, 0x00U},
        {0x00A0U, 0x00U},
        {0x00A8U, 0x00U},
    };
    DevBlkCDII2CRegListWritable const readTX_CRC_EN_Regs = {
        .regs = TX_CRC_EN_Regs,
        .numRegs = I2C_ARRAY_SIZE(TX_CRC_EN_Regs)
    };
    NvMediaStatus status = ReadArrayVerifyMAX96717F(handle, &readTX_CRC_EN_Regs);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_HEX_UINT("ReadArrayVerifyMAX96717F failed", (uint32_t)status);
    } else {
        for (uint8_t i = 0U; i < readTX_CRC_EN_Regs.numRegs; i++) {
            if ((TX_CRC_EN_Regs[i].data & 0xC0U) != 0xC0U) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "MAX96717F: TX/RX_CRC_EN is not enabled at addr:",
                    (uint32_t)TX_CRC_EN_Regs[i].address
                );
                status = NVMEDIA_STATUS_ERROR;
                break;
            }
        }
    }
    if (status == NVMEDIA_STATUS_OK) {
        /* Set TX_CRC_EN in the CFG register for Video
         * Setting bits 4 and 5 as they are reserved with reset value 1 */
        uint8_t tx0_val = 0xB0U;
        status = WriteUint8VerifyMAX96717F(handle, REG_CFGV_TX0, tx0_val);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96717FSetDefaults: CFGV VIDEO_Z TX0 write failed");
        }
    }
    return status;
}

static NvMediaStatus
VerifyReg5MAX96717F(
    DevBlkCDIDevice const* handle)
{
    uint8_t regVal = 0U;
    NvMediaStatus status = ReadUint8VerifyMAX96717F(handle, REG_REG5, &regVal);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96717F: Failed to read register:", (uint32_t)REG_REG5);
    } else {
        /* SMs 6,7,8,9: Verify that LOCK and ERRB indicator output pins are disabled
        * (only Deserializer counterparts are used).*/
        if (((regVal & bit8(MAX96717F_LOCK_ENABLE_BIT)) > 0U) ||
            ((regVal & bit8(MAX96717F_ERRB_ENABLE_BIT)) > 0U)) {
            SIPL_LOG_ERR_STR("MAX96717F REG5 bits set.");
            status = NVMEDIA_STATUS_ERROR;
        }
    }

    return status;
}

NvMediaStatus
MAX96717FSetDefaults(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = EnableRXTXCRCMAX96717F(handle);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96717F: EnableRXTXCRCMAX96717F failed.");
        } else {
            status = VerifyReg5MAX96717F(handle);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR("MAX96717F: VerifyReg5MAX96717F failed.");
            } else {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#ifdef NVMEDIA_QNX
                bool lockStatus = false;
                bool errbStatus = false;

                status = RunADCBISTMAX96717F(handle, &lockStatus, &errbStatus);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_3STR("MAX96717FSetDefaults: Lock and ERRB",
                                    ((lockStatus) ? "True" : "False"),
                                    ((errbStatus) ? "True" : "False"));
                } else {
                    /* ADC configure for temperature sensor */
                    status = SetTemperatureMAX96717F(handle);
                    if (NVMEDIA_STATUS_OK != status) {
                        SIPL_LOG_ERR_STR_HEX_UINT("MAX96717F: SetTemperatureMAX96717F failed",
                                                (uint32_t)status);
                    } else {
#endif
                        uint8_t rvsn = 0U;
                        status = ReadUint8VerifyMAX96717F(handle, REG_DEV_REV_ADDR, &rvsn);
                        if (status == NVMEDIA_STATUS_OK) {
                            rvsn &= 0x0FU;
                            if (rvsn == 4U) {
                                DriverHandleMAX96717F *drvHandle = getHandlePrivMAX96717F(handle);
                                drvHandle->revisionID = CDI_MAX96717F_REV_4;
                                /*Enable negative output by writing enminus_man = 1 and
                                enminus_reg = 1*/
                                uint8_t wRegVal = 0x19U;
                                status = WriteUint8VerifyMAX96717F(handle, 0x14CEU, wRegVal);
                                if (NVMEDIA_STATUS_OK != status) {
                                    SIPL_LOG_ERR_STR_HEX_UINT("Negative output register settings "
                                                            "failed", (uint32_t)status);
                                }
                            }
                            if (status == NVMEDIA_STATUS_OK) {
                                status = MIPI_Reset(handle);
                            }
                        }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#ifdef NVMEDIA_QNX
                    }
                }
#endif
            }
        }
    }
    return status;
}

NvMediaStatus
MAX96717FReadRegisters(
    DevBlkCDIDevice const* handle,
    uint16_t registerNum,
    uint16_t dataLength,
    uint8_t *dataBuff)
{
    DriverHandleMAX96717F const* drvHandle = getHandlePrivMAX96717F(handle);
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96717FReadRegisters: Bad parameter \n");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else if (NULL == dataBuff) {
        SIPL_LOG_ERR_STR("MAX96717FReadRegisters: invalid dataBuff\n");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else if (((dataLength - (uint16_t)1U) > ((uint16_t)MAX_VALID_REG - registerNum))
        || (dataLength == 0U)) {
        SIPL_LOG_ERR_STR("Bad parameter: Invalid data length passed");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = DevBlkCDII2CPgmrReadBlock(
            drvHandle->i2cProgrammer, registerNum, dataLength, dataBuff);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_2HEX("MAX96717FReadRegisters: MAX96717F read register failed: ",
                registerNum, registerNum + dataLength);
        }
    }
    return status;
}

NvMediaStatus
MAX96717FReadRegistersVerify(
    DevBlkCDIDevice const * const handle,
    uint16_t const registerNum,
    uint16_t const dataLength,
    uint8_t * const dataBuff)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if(NULL == dataBuff) {
        SIPL_LOG_ERR_STR("MAX96717FReadRegistersVerify: invalid dataBuff\n");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = ReadBlockVerifyMAX96717F(handle, registerNum, dataLength, dataBuff);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("ReadBlockVerifyMAX96717F failed", (uint32_t)status);
        }
    }

    return status;
}

NvMediaStatus
MAX96717FWriteRegisters(
    DevBlkCDIDevice const* handle,
    uint16_t registerNum,
    uint16_t dataLength,
    uint8_t const* dataBuff)
{
    DriverHandleMAX96717F const* drvHandle  = getHandlePrivMAX96717F(handle);
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if ((NULL == drvHandle) || (NULL == dataBuff)) {
        SIPL_LOG_ERR_STR("MAX96717FWriteRegisters: Bad parameter \n");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else if (((dataLength - (uint16_t)1U) > ((uint16_t)MAX_VALID_REG - registerNum))
        || (dataLength == 0U)) {
        SIPL_LOG_ERR_STR("Bad parameter: Invalid data length passed");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = DevBlkCDII2CPgmrWriteBlock(
            drvHandle->i2cProgrammer, registerNum, dataBuff, dataLength);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_2HEX("MAX96717FWriteRegisters: MAX96717F write register failed: ",
                registerNum, registerNum + dataLength);
        }
    }

    return status;
}

NvMediaStatus
MAX96717FWriteRegistersVerify(
    DevBlkCDIDevice const * const handle,
    uint16_t const registerNum,
    uint16_t const dataLength,
    uint8_t const * const dataBuff)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if(NULL == dataBuff) {
        SIPL_LOG_ERR_STR("MAX96717FWriteRegistersVerify: invalid dataBuff\n");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = WriteBlockVerifyMAX96717F(handle, registerNum, dataLength, dataBuff);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("ReadBlockVerifyMAX96717F failed", (uint32_t)status);
        }
    }

    return status;
}

/**
 * @brief Sets group A parameters of the 96717F module
 *
 * @param[in] handle CDI Device Block Handle.
 * @param[in] parameterType Type of the parameter to be set
 * @param[in] parameterSize Size of the parameter to be set
 * @param[in] parameter Parameter pointer.
 *
 * @retval NVMEDIA_STATUS_OK    Write was successful
 * @retval NVMEDIA_STATUS_BAD_PARAMETER input parameter pointer is NULL
 * @retval NVMEDIA_STATUS_ERROR Write was failure.
 */
static NvMediaStatus
WriteParametersAMAX96717F(
    DevBlkCDIDevice const* handle,
    WriteParametersCmdMAX96717F parameterType,
    size_t parameterSize,
    void const* parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    /* coverity[misra_c_2012_rule_11_5_violation] : intentional TID-1417 */
    ReadWriteParamsMAX96717F const* param = (ReadWriteParamsMAX96717F const*) parameter;

    switch ((WriteParametersCmdMAX96717F)parameterType) {
        case CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_A:
        case CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B:
            if (parameterSize == sizeof(param->Translator)) {
                status = SetTranslatorMAX96717F(handle,
                                                (uint32_t)parameterType,
                                                param->Translator.source,
                                                param->Translator.destination);
            } else {
                status = NVMEDIA_STATUS_BAD_PARAMETER;
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_SET_DEVICE_ADDRESS:
            if (parameterSize == sizeof(param->DeviceAddress)) {
                status = SetDeviceAddressMAX96717F(handle,
                                                   param->DeviceAddress.address);
            }  else {
                status = NVMEDIA_STATUS_BAD_PARAMETER;
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT:
            if (parameterSize == sizeof(param->GPIOOutp)) {
                status = SetGPIOOutputMAX96717F(handle,
                                                param->GPIOOutp.gpioInd,
                                                param->GPIOOutp.level);
            }  else {
                status = NVMEDIA_STATUS_BAD_PARAMETER;
            }
            break;
        default:
            SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Invalid command");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }

    return status;
}

/**
 * @brief Sets group B parameters of the 96717F module
 *
 * @param[in] handle CDI Device Block Handle.
 * @param[in] parameterType Type of the parameter to be set
 * @param[in] parameterSize Size of the parameter to be set
 * @param[in] parameter Parameter pointer.
 *
 * @retval NVMEDIA_STATUS_OK    Write was successful
 * @retval NVMEDIA_STATUS_BAD_PARAMETER input parameter pointer is NULL
 * @retval NVMEDIA_STATUS_ERROR Write was failure.
 */
static NvMediaStatus
WriteParametersBMAX96717F(
    DevBlkCDIDevice const* handle,
    WriteParametersCmdMAX96717F parameterType,
    size_t parameterSize,
    void const* parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    /* coverity[misra_c_2012_rule_11_5_violation] : intentional TID-1417 */
    ReadWriteParamsMAX96717F const* param = (ReadWriteParamsMAX96717F const*) parameter;

    switch ((WriteParametersCmdMAX96717F)parameterType) {
        case CDI_WRITE_PARAM_CMD_MAX96717F_SET_FSYNC_GPIO:
            if (parameterSize == sizeof(param->FSyncGPIO)) {
                status = SetFsyncGPIOMAX96717F(handle,
                                               param->FSyncGPIO.gpioInd,
                                               param->FSyncGPIO.rxID);
            } else {
                status = NVMEDIA_STATUS_BAD_PARAMETER;
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_REF_CLOCK:
            if (parameterSize == sizeof(param->RefClkGPIO)) {
                status = EnableRefClockMAX96717F(handle,
                                                 param->RefClkGPIO.gpioInd,
                                                 param->RefClkGPIO.enableRClk);
            } else {
                status = NVMEDIA_STATUS_BAD_PARAMETER;
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_VIDEO_PIPELINES:
            if (parameterSize == sizeof(param->ConfigVideoPipeline)) {
                status = ConfigPipelinesMAX96717F(
                                handle,
                                param->ConfigVideoPipeline.dataType,
                                param->ConfigVideoPipeline.embDataType);
            } else {
                status = NVMEDIA_STATUS_BAD_PARAMETER;
            }
            break;
        default:
            SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Invalid command");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }

    return status;
}

/**
 * @brief Sets group C parameters of the 96717F module
 *
 * @param[in] handle CDI Device Block Handle.
 * @param[in] parameterType Type of the parameter to be set
 * @param[in] parameterSize Size of the parameter to be set
 * @param[in] parameter Parameter pointer.
 *
 * @retval NVMEDIA_STATUS_OK    Write was successful
 * @retval NVMEDIA_STATUS_BAD_PARAMETER input parameter pointer is NULL
 * @retval NVMEDIA_STATUS_ERROR Write was failure.
 */
static NvMediaStatus
WriteParametersCMAX96717F(
    DevBlkCDIDevice const* handle,
    WriteParametersCmdMAX96717F parameterType,
    size_t parameterSize,
    void const* parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    /* coverity[misra_c_2012_rule_11_5_violation] : intentional TID-1417 */
    ReadWriteParamsMAX96717F const* param = (ReadWriteParamsMAX96717F const*) parameter;

    switch ((WriteParametersCmdMAX96717F)parameterType) {
        case CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_PHY:
            if (parameterSize == sizeof(param->ConfigPhy)) {
                status = ConfigPhyMap(handle,
                                      &param->ConfigPhy.mapping,
                                      &param->ConfigPhy.polarity,
                                      param->ConfigPhy.numDataLanes);
            } else {
                status = NVMEDIA_STATUS_BAD_PARAMETER;
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_GENERATE_CLOCK:
            if (parameterSize == sizeof(param->ClockRate)) {
                status = GenerateClock(handle,
                                       param->ClockRate.freq);
            } else {
                status = NVMEDIA_STATUS_BAD_PARAMETER;
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_SET_DEBUG_REGS:
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_PCLK_PIO_SLEW:
            status = EnablePClkPIOSlew(handle);
            break;
        default:
            SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Invalid command");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }

    return status;
}

/**
 * @brief Sets group D parameters of the 96717F module
 *
 * @param[in] handle CDI Device Block Handle.
 * @param[in] parameterType Type of the parameter to be set
 * @param[in] parameterSize Size of the parameter to be set
 * @param[in] parameter Parameter pointer.
 *
 * @retval NVMEDIA_STATUS_OK    Write was successful
 * @retval NVMEDIA_STATUS_BAD_PARAMETER input parameter pointer is NULL
 * @retval NVMEDIA_STATUS_ERROR Write was failure.
 */
static NvMediaStatus
WriteParametersDMAX96717F(
    DevBlkCDIDevice const* handle,
    WriteParametersCmdMAX96717F parameterType,
    size_t parameterSize,
    void const* parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    /* coverity[misra_c_2012_rule_11_5_violation] : intentional TID-1417 */
    ReadWriteParamsMAX96717F const* param = (ReadWriteParamsMAX96717F const*) parameter;

    switch ((WriteParametersCmdMAX96717F)parameterType) {
        case CDI_WRITE_PARAM_CMD_MAX96717F_GPIO_FORWARD:
            if (parameterSize == sizeof(param->GPIOForward)) {
                status = ForwardGPIOMAX96717F(handle,
                                              param->GPIOForward.srcGpio,
                                              param->GPIOForward.dstGpio);
            } else {
                status = NVMEDIA_STATUS_BAD_PARAMETER;
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_SM_REPORTS:
            status = EnableSMReports(handle, false);
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_SM_REPORTS_SKIP_REFGEN_CHECK:
            status = EnableSMReports(handle, true);
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_UNSET_ERRB_TX:
            status = UnsetErrbTx(handle);
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_SET_ERRB_TX:
            status = SetErrbTx(handle);
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_SET_ERRB_TX_ID:
            status = SetErrbTxId(handle,param->GPIOErrb.dstGpio);
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_ASSERT_ERRB:
            if (parameterSize == sizeof(param->assertERRB)) {
                status = AssertErrbMAX96717F(handle,
                                             param->assertERRB);
            } else {
                status = NVMEDIA_STATUS_BAD_PARAMETER;
            }
            break;
        default:
            SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Invalid command");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }

    return status;
}

static NvMediaStatus
AssertErrbMAX96717F(
    DevBlkCDIDevice const *handle,
    bool trigger)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    DevBlkCDII2CReg const assertErrbRegs[] = {
        {0x001CU, 0xFEU}, /* set PKT_CNT_OEN */
        {0x002CU, 0x0EU}, /* PKT_CNT_SEL to all GMSL packets */
    };

    DevBlkCDII2CRegList assertErrb = {
        .regs = assertErrbRegs,
        .numRegs = I2C_ARRAY_SIZE(assertErrbRegs)
    };

    if (trigger == true) {
        status = WriteArrayVerifyMAX96717F(handle, &assertErrb);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("WriteArrayVerifyMAX96717F failed", (uint32_t)status);
        }
    } else {
        /* unset PKTCNT_OEN & clear PKT_CNT_SEL */
        uint8_t pkt_cnt = 0U;
        status = ReadUint8VerifyMAX96717F(handle, REG_CNT3, &pkt_cnt);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("ReadUint8VerifyMAX96717F failed", (uint32_t)status);
        } else {
            status = WriteUint8VerifyMAX96717F(handle, assertErrbRegs[0].address, 0x08U);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT("WriteUint8VerifyMAX96717F failed", (uint32_t)status);
            }
        }
    }

    return status;
}

NvMediaStatus
MAX96717FWriteParameters(
    DevBlkCDIDevice const* handle,
    WriteParametersCmdMAX96717F parameterType,
    size_t parameterSize,
    void const* parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if ((NULL == handle) || (parameter == NULL)) {
        SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        PrintLogMsg(LOG_LEVEL_DEBUG, "MAX96717F: paramType %d\n",
                    (uint32_t)parameterType);
        if (CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT >= parameterType) {
            status = WriteParametersAMAX96717F(handle,
                                               parameterType,
                                               parameterSize,
                                               parameter);
        } else if (CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_VIDEO_PIPELINES >= parameterType) {
            status = WriteParametersBMAX96717F(handle,
                                               parameterType,
                                               parameterSize,
                                               parameter);
        } else if (CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_PCLK_PIO_SLEW >= parameterType) {
            status = WriteParametersCMAX96717F(handle,
                                               parameterType,
                                               parameterSize,
                                               parameter);
        } else if (CDI_WRITE_PARAM_CMD_MAX96717F_ASSERT_ERRB >= parameterType) {
            status = WriteParametersDMAX96717F(handle,
                                               parameterType,
                                               parameterSize,
                                               parameter);
        } else {
            /* code */
            status = NVMEDIA_STATUS_BAD_PARAMETER;
        }

        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_UINT("MAX96717F: paramType failed", (uint32_t)parameterType);
        }
    }

    return status;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
NvMediaStatus
MAX96717FDumpRegisters(
    DevBlkCDIDevice const* handle)
{
    uint16_t address = 0U;
    uint8_t regVal = 0U;
    DriverHandleMAX96717F const* drvHandle = getHandlePrivMAX96717F(handle);
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96717f: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        for (uint32_t i = 0U; i <= MAX96717F_REG_MAX_ADDRESS; i++) {
            address = (uint16_t)(lshift16(i / 256U, 8U) | (i%256U));
            /* Skip ReadVerify as this reads all Regs */
            status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, address, &regVal);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_AND_UINT("MAX96717F: Register I2C read failed with status",
                                              i, (uint32_t)status);
                break;
            }
            SIPL_LOG_ERR_STR_2HEX("Max96717F: address: data:", address, regVal);
        }
    }
    return status;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif /* !NV_IS_SAFETY */

NvMediaStatus
MAX96717FCheckPresence(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    RevisionMAX96717F rev = CDI_MAX96717F_INVALID_REV;
    uint8_t devID = 0U;

    if (NULL == handle) {
        SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = ReadUint8VerifyMAX96717F(handle, REG_DEV_ID_ADDR, &devID);
        if (status == NVMEDIA_STATUS_OK) {
            if ((devID != MAX96717F_DEV_ID_C8) &&
                (devID != MAX96717_DEV_ID_BF)) {
                SIPL_LOG_ERR_STR_UINT("MAX96717F: Not supported devID val:", devID);
            } else {
                status = GetRevIdMAX96717F(handle, devID, &rev);
            }
        }
    }
    return status;
}

NvMediaStatus
MAX96717FGetEOMValue(
    DevBlkCDIDevice const* handle,
    uint8_t *eomValue)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (eomValue ==  NULL) {
        SIPL_LOG_ERR_STR("Input parameter is NULL");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = GetEOMVal(handle, eomValue);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("GetEOMValue failed", (uint32_t)status);
        }
    }

    return status;
}

NvMediaStatus
MAX96717FGetTemperature(
    DevBlkCDIDevice const* handle,
    float_t *tmon1,
    float_t *tmon2)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t temperatureEst[3] = {0U};
    uint16_t tmp = 0U;

    if ((tmon1 == NULL) || (tmon2 == NULL)) {
        SIPL_LOG_ERR_STR("MAX96717FGetTemperature: Bad parameter tmon1, tmon2 passed");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        /* Read ADC registers */
        status = MAX96717FReadRegistersVerify(handle, REG_ADCBIST13, 3U, temperatureEst);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("Read register REG_ADCBIST13 & REG_ADCBIST14 failed",
                                      (uint32_t)status);

        } else {
            /* Combine the values from 2 registers to 10-bit ADC value*/
            /* TMON1 calculation */
            tmp = (uint16_t)((uint32_t)temperatureEst[1] & 0xC0U);
            tmp = lshift16(tmp, 2U);
            tmp |= temperatureEst[0];

            tmp = rshift16(tmp, 1U);
            *tmon1 = KelvinToCelsius(tmp);

            /* TMON2 calculation */
            tmp = (uint16_t)((uint32_t)temperatureEst[1] & 0x03U);
            tmp = lshift16(tmp, 8U);
            tmp |= temperatureEst[2];

            tmp = rshift16(tmp, 1U);
            *tmon2 = KelvinToCelsius(tmp);
        }
    }

    return status;
}

static NvMediaStatus GetRegBitMAX96717F(
    DevBlkCDIDevice const * const handle,
    uint16_t addr,
    uint8_t bit,
    uint8_t * const val)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regVal = 0U;

    if (NULL == val) {
        SIPL_LOG_ERR_STR("MAX96717f: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = ReadUint8VerifyMAX96717F(handle, addr, &regVal);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("ReadUint8VerifyMAX96717F failed", (uint32_t)status);
        } else {
            if (bit <= UINT8_MAX_SHIFT ) {
                *val = GetBit(regVal, bit);
            } else {
                /* Bit & operation if bit value is bigger
                * UINT8_MAX_SHIFT */
                *val = regVal & bit;
            }
        }
    }

    return status;
}

static NvMediaStatus SetRegBitMAX96717F(
    DevBlkCDIDevice const * const handle,
    uint16_t addr,
    uint8_t bit,
    bool val)
{
    uint8_t regVal = 0U;
    NvMediaStatus status = ReadUint8VerifyMAX96717F(handle, addr, &regVal);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_HEX_UINT("ReadUint8VerifyMAX96717F failed", (uint32_t)status);
    } else {
        if (val == true) { /* set to 1 */
            regVal = SetBit(regVal, bit);
        } else {        /* set to 0 */
            regVal = ClearBit(regVal, bit);
        }

        status = WriteUint8VerifyMAX96717F(handle, addr, regVal);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("ReadUint8VerifyMAX96717F failed", (uint32_t)status);
        }
    }

    return status;
}

static MAX96717FSMErrInfoRegSet const *GetRegSetMAX96717F(
    uint8_t const index)
{
    const uint8_t smlistmax = (uint8_t)SM_LIST_MAX_NUM;
    MAX96717FSMErrInfoRegSet const *retVal = NULL;
    static const MAX96717FSMErrInfoRegSet sm_err_info_regset_max96717f[] = {
        /* Checking lock indicator for forward channel(SM06) */
        {.name = "Forward Lock indicator", .addr = REG_CTRL3,
         .bit = 3U, .index = (uint8_t)SM06_LOCK_INDICATOR_OF_FORWARD_CHANNEL},
        /* Checking ERRB indicator for forward channel(SM09) */
        {.name = "Forward ERRB indicator", .addr = REG_CTRL3,
         .bit = 2U, .index = (uint8_t)SM07_ERRB_INDICATOR_OF_FORWARD_CHANNEL},
        /* Checking lock indicator for reverse channel(SM08) */
        {.name = "Reverse Lock indicator", .addr = REG_CTRL3,
         .bit = 3U, .index = (uint8_t)SM08_LOCK_INDICATOR_OF_REVERSE_CHANNEL},
        /* Checking ERRB indicator for reverse channel(SM09) */
        {.name = "Reverse ERRB indicator", .addr = REG_CTRL3,
         .bit = 2U, .index = (uint8_t)SM09_ERRB_INDICATOR_OF_REVERSE_CHANNEL},
        /* Checking TUN_FIFO_OVERFLOW bit in EXT8 register(SM17) */
        {.name = "FIFO Overflow detection", .addr = REG_EXT8,
         .bit = 0U, .index = (uint8_t)SM17_FIFO_OVERFLOW_DETECTION},
        /* Checking for LBIST done(SM18) */
        {.name = "Logic BIST", .addr = REG_POST0,
         .bit = 0xA0U, .index = (uint8_t)SM18_LOGIC_BIST},
        /* Checking for MBIST done(SM19) */
        {.name = "Memory BIST", .addr = REG_POST0,
         .bit = 0xC0U, .index = (uint8_t)SM19_MEMORY_BIST},
#if !FUSA_CDD_NV
        /* These SMs are unused as per the latest degradation sheet shared by
           Valeo, and hence have been disabled in DOS 6.0. */
        {.name = "CRC on I2C transaction", .addr = REG_FS_INTR1,
         .bit = 6U, .index = (uint8_t)SM21_CRC_ON_I2C_UART_TRANSACTIONS},
        {.name = "I2C MSG CNT", .addr = REG_FS_INTR1,
         .bit = 7U, .index = (uint8_t)SM22_MESSAGE_COUNTER},
#endif
        /* Checking for MEM_ECC_ERR2_INT and MEM_ECC_ERR1_INT bits in FS_INTR1
         * register(SM26) */
        {.name = "MEM_ECC_ERR", .addr = REG_FS_INTR1,
         .bit = 0x30U, .index = (uint8_t)SM26_MEMORY_ECC},
        /* Checking for REG_CRC_ERR_FLAG bit in FS_INTR1 register(SM27) */
        {.name = "CONFIG_REG_CRC", .addr = REG_FS_INTR1,
         .bit = 0U, .index = (uint8_t)SM27_CONFIGURATION_REGISTER_CRC},
        /* Checking for ADC_INT_FLAG bit in INTR7 register(SM28) */
        {.name = "Self test of temperature", .addr = REG_INTR7,
         .bit = 2U, .index = (uint8_t)SM28_SELF_TEST_OF_TEMPERATURE_SENSOR},
        /* Checking EFUSE_CRC_ERR bit in INTR7 register(SM29) */
        {.name = "CRC of non-volatile memory", .addr = REG_INTR7,
         .bit = 4U, .index = (uint8_t)SM29_CRC_OF_NONVOLATILE_MEMORY},
#if !FUSA_CDD_NV
        {.name = "SELF_CLEARING_UNLOCK", .addr = REG_REG0,
         .bit = 0U, .index = (uint8_t)SM30_SELF_CLEARING_UNLOCK_REGISTER},
#endif
        /* Checking REFGEN_LOCKED bit in REF_VTG0 register(SM31) */
        {.name = "DPLL_LOCK_INDICATOR", .addr = REG_REF_VTG0,
         .bit = 7U, .index = (uint8_t)SM31_DPLL_LOCK_INDICATOR},
        /* Checking for PCLK_DET bit in VIDEO_TX2 register(SM32) */
        {.name = "VIDEO_DETECT", .addr = REG_VIDEO_TX2,
         .bit = 7U, .index = (uint8_t)SM32_VIDEO_DETECT},
    };

    if (index < smlistmax) {
        uint8_t i = 0U;
        uint8_t arrSize = (uint8_t)(I2C_ARRAY_SIZE(sm_err_info_regset_max96717f) & 0xFFU);

        /* Searching the SMRegSet list */
        for (i = 0U; i < arrSize; i++) {
            uint8_t list = (uint8_t)sm_err_info_regset_max96717f[i].index;
            if (list == index) {
                retVal = &sm_err_info_regset_max96717f[i];
                break;
            }
        }

        /* If can't find SMRegSet, RegSet set to NULL */
        if (i == arrSize) {
            PrintLogMsg(LOG_LEVEL_INFO, "%s: Can't find index(%d) SM RegSet\n",
                        __func__, index);
            retVal = NULL;
        }
    }
    return retVal;
}

NvMediaStatus
MAX96717FGetSMStatus(
    DevBlkCDIDevice const *handle,
    uint8_t const errIndex,
    uint8_t *store)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (NULL == store) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        MAX96717FSMErrInfoRegSet const *regSet = GetRegSetMAX96717F(errIndex);
        if (NULL == regSet) {
            SIPL_LOG_ERR_STR("RegSet is not available");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
        } else {
            uint8_t regVal = 0U;
            status = GetRegBitMAX96717F(handle, regSet->addr, regSet->bit, &regVal);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT("GetRegBit failed",
                                          (uint32_t)status);
            } else {
                *store = regVal;
            }
        }
    }
    return status;
}

NvMediaStatus
MAX96717FSetSMStatus(
    DevBlkCDIDevice const *handle,
    uint8_t const index,
    bool const onOff)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    MAX96717FSMErrInfoRegSet const *regSet = GetRegSetMAX96717F(index);

    if (NULL == regSet) {
        SIPL_LOG_ERR_STR("RegSet is not available");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = SetRegBitMAX96717F(handle, regSet->addr, regSet->bit, onOff);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("SetRegBit failed",
                                        (uint32_t)status);
        }
    }

    return status;
}

NvMediaStatus
MAX96717FGetVideoStatus(
    DevBlkCDIDevice const* handle,
    uint8_t *overflow_status,
    uint8_t *pclkdet_status,
    uint8_t *tun_fifo_overflow_status
)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if ((overflow_status == NULL) || (pclkdet_status == NULL) ||
        (tun_fifo_overflow_status == NULL)) {
        SIPL_LOG_ERR_STR("MAX96717FGetVideoStatus: Bad parameter\n");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        *overflow_status = 0U;
        *pclkdet_status = 0U;
        *tun_fifo_overflow_status = 0U;
        uint8_t data = 0U;

        status = ReadUint8VerifyMAX96717F(handle, REG_VIDEO_TX2, &data);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("Failed to read register", (uint32_t)REG_VIDEO_TX2);
        } else {
            /* Check the OVERFLOW bit (b5) */
            if ((data & 0x20U) > 0U) {
                *overflow_status = 1U;
            }
            /* Check the PCLKDET bit (b7) */
            if ((data & 0x80U) > 0U) {
                *pclkdet_status = 1U;
            }

            status = ReadUint8VerifyMAX96717F(handle, REG_EXT8, &data);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT("Failed to read register", (uint32_t)REG_EXT8);
            } else {
                /* Check the TUN_FIFO_OVERFLOW bit (b0) */
                if ((data & 0x01U) > 0U) {
                    *tun_fifo_overflow_status = 1U;
                }
            }
        }
    }

    return status;
}

NvMediaStatus
MAX96717FReadErrorData(
    DevBlkCDIDevice const* handle,
    MAX96717FErrorInfo *const errInfo)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    (void)memset(errInfo, 0, sizeof(MAX96717FErrorInfo));
    /* Read Whole SM Error register */
    status = GetSMErrInfoMAX96717F(handle, errInfo);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR("Error check failed");
    }

    return status;
}

static NvMediaStatus
    UnsetErrbTx(DevBlkCDIDevice const *handle)
{
    uint8_t wRegVal = 0x1FU;
    NvMediaStatus status = WriteUint8VerifyMAX96717F(handle, REG_INTR8, wRegVal);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR("Reg Write failed");
    }

    return status;
}

static NvMediaStatus
    SetErrbTx(DevBlkCDIDevice const *handle)
{
    uint8_t wRegVal = 0x9FU;
    NvMediaStatus status = WriteUint8VerifyMAX96717F(handle, REG_INTR8, wRegVal);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR("Reg Write failed");
    }

    return status;
}

static NvMediaStatus
    SetErrbTxId(DevBlkCDIDevice const *handle, uint8_t dstGpio)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (dstGpio > (uint8_t)31U) {
        SIPL_LOG_ERR_STR_UINT("MAX96717f: Bad parameter dstGpio",
                              dstGpio);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        uint8_t wRegVal = 0x80U | dstGpio;
        status = WriteUint8VerifyMAX96717F(handle, REG_INTR8, wRegVal);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("Reg Write failed");
        }
    }

    return status;
}

static NvMediaStatus EnableSMReports(
    DevBlkCDIDevice const *handle,
    bool skipRefgenCheck)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regVal = 0U;

    static const DevBlkCDII2CReg max96717f_err_report_enabling_regs[] = {
        {0x001CU, 0x00B8U}, /* Enable MAX_RT and OV/UV error related SM */
#if FUSA_CDD_NV
        /* Disable INTR2 (0x001A) SMs
            * SM2: Line-fault detection [bit 3]
            * SM7: Remote side error status [bit 5]
            * SM33, 36: Idle and decode errors [bits 2, 0]
            *
            * Enable INTR2 (0x001A) SMs
            * SM31: reference generation PLL lock [bit 7]
            */
        {0x001AU, 0x80U},

        /* Enable INTR6 (0x001E) SMs
            * SM16: MIPI RX errors [bit 0]
            * SM15: OV/UV related errors [bits 5 - 7]
            * SM29: EFUSE CRC errors [bit 4]
            *
            * Disable INTR (0x001E) SMs
            * SM37: retention memory CRC errors [bit 3]
            * SM24, 25, 28: ADC related SMs [bit 2]
            *
            * Setting bit 1 as it is reserved with reset value 1
            */
        {0x001EU, 0xF3U},
#else
        /* Enable REM_ERR_OEN and IDLE/DECODE error SM reporting. */
        {0x001AU, 0x25U},
        /* Enable MIPI and OV/UV error related SM
            * Enable RTTN_CRC_ERR_OEN */
        {0x001EU, 0xFDU},
#endif
        /* Setting bits 4,5 and 6 as they are reserved with reset value 1 */
        {0x008EU, 0x72U}, /* Enable MAX_RT error reporting for SER */
        {0x0096U, 0x72U}, /* Enable MAX_RT error reporting for GPIO */
        {0x00A6U, 0x72U}, /* Enable MAX_RT error reporting for I2C PT X */
        {0x00AEU, 0x72U}, /* Enable MAX_RT error reporting for I2C PT Y */
#if FUSA_CDD_NV
        /* Enable FS_INTR0 (0x1D12) SMs:
            * SM26: ECC 2-bit uncorrectable errors [bit 5]
            *
            * Disable FS_INTR0 (0x1D12) SMs:
            * SM22: I2C/UART message counter errors [bit 7]
            * SM21: I2C/UART CRC errors [bit 6]
            */
        {0x1D12U, 0x20U},
#else
        /* Enable cfg reg CRC[bit 0] & mem ECC error reporting[bits 4,5]
            * Enable I2C_UART_CRC_ERR_OEN[bit 6] and I2C_UART_MSGCNTR_ERR_ONE[bit 7] */
        {0x1D12U, 0xF1U},
#endif
    };

    static const DevBlkCDII2CRegList max96717f_enable_err_reporting = {
        .regs = max96717f_err_report_enabling_regs,
        .numRegs = I2C_ARRAY_SIZE(max96717f_err_report_enabling_regs)
    };

    status = WriteArrayVerifyMAX96717F(handle, &max96717f_enable_err_reporting);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_HEX_UINT("WriteArrayVerifyMAX96717F failed", (uint32_t)status);
    }

    if (NVMEDIA_STATUS_OK == status) {
        /* Read Idle error count to clear the existing count */
        status = ReadUint8VerifyMAX96717F(handle, REG_IDLE_ERROR, &regVal);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("REG_IDLE_ERROR read failed");
        } else {
            PrintLogMsg(LOG_LEVEL_INFO, "IDLE Error Count is %u\n", (uint32_t)regVal);
        }
    }

    if (NVMEDIA_STATUS_OK == status) {
        /* Read Decode error count to clear the existing count */
        status = ReadUint8VerifyMAX96717F(handle, REG_DEC_ERROR, &regVal);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("REG_DEC_ERROR read failed");
        } else {
            PrintLogMsg(LOG_LEVEL_INFO, "Decode Error Count is %u\n", (uint32_t)regVal);
        }
    }

    /* If INTR3 is set, it indicates an error state on the serializer side */
    /* and hence it needs to be reported back to the application */
    if (NVMEDIA_STATUS_OK == status) {
        status = ReadUint8VerifyMAX96717F(handle, REG_INTR3, &regVal);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("INTR3 read failed");
        } else {
            if (skipRefgenCheck) {
                regVal &= 0x7FU;
            }
            if (regVal != 0x00U) {
                SIPL_LOG_ERR_STR_HEX_UINT("INTR3 reported error", (uint32_t)regVal);

                /* Call below for reading Idle and Decode error counts
                 * are to obtain additional debugging information.
                 * The primary error that needs to be reported back to
                 * application correspons to INTR3. Hence any error status of
                 * Idle and Decode error count reads is not reported back to
                 * application
                 */
                status = ReadUint8VerifyMAX96717F(handle, REG_IDLE_ERROR, &regVal);
                if (NVMEDIA_STATUS_OK == status) {
                    SIPL_LOG_ERR_STR_HEX_UINT("IDLE Error Count is ", (uint32_t)regVal);
                    /* Read Decode error count */
                    status = ReadUint8VerifyMAX96717F(handle, REG_DEC_ERROR, &regVal);
                    if (NVMEDIA_STATUS_OK == status) {
                        SIPL_LOG_ERR_STR_HEX_UINT("Decode Error Count is ", (uint32_t)regVal);
                        status = NVMEDIA_STATUS_ERROR;
                    }
                }
            } else {
                PrintLogMsg(LOG_LEVEL_INFO, "No error reported\n");
            }
        }
    }

    /* Checking for EFUSE CRC errors in INTR7, at start-up. */
    if (NVMEDIA_STATUS_OK == status) {
        status = ReadUint8VerifyMAX96717F(handle, REG_INTR7, &regVal);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("Failed to read register", (uint32_t)REG_INTR7);
        } else {
            /* Check the OVERFLOW bit (b4) */
            if ((regVal & 0x10U) > 0U) {
                SIPL_LOG_ERR_STR("MAX96717F: EFUSE_CRC_ERR flag is set.\n");
                status = NVMEDIA_STATUS_ERROR;
            }
        }
    }

    return status;
}

/**
 * @brief Setup up the pre-requisite registers for manual EOM.
 *
 * @param[in] handle  CDI handle to the serializer.
 *
 * @return NVMEDIA_STATUS_OK Call successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER Invalid param passed.
 * @retval NVMEDIA_STATUS_ERROR Call failed.
 */
static NvMediaStatus SetupManualEOM(DevBlkCDIDevice const * const handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    /* AEQ to 0b0*/
    uint8_t aeqRegVal = 0U;
    status = ReadUint8VerifyMAX96717F(handle, REG_EOM_RLMSA4, &aeqRegVal);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_UINT("Serializer AEQ register read failed", (uint32_t)status);
    } else {
        aeqRegVal &= 0xC0U;
        status = WriteUint8VerifyMAX96717F(handle, REG_EOM_RLMSA4, aeqRegVal);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("Serializer write failed");
        } else {
            /* EOM_PER_MODE to 0b0*/
            status = SetRegBitMAX96717F(handle, REG_EOM_RLMS4, 1U, false);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR("Serializer write to set EOM_PER_MODE to 0 failed");
            } else {
                /* EOM_EN to 0b0*/
                status = SetRegBitMAX96717F(handle, REG_EOM_RLMS4, 0U, false);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR("Serializer write to set EOM_EN to 0 failed");
                } else {
                    /* EOM_EN to 0b1*/
                    status = SetRegBitMAX96717F(handle, REG_EOM_RLMS4, 0U, true);
                    if (NVMEDIA_STATUS_OK != status) {
                        SIPL_LOG_ERR_STR("Serializer write for EOM_EN to 0b1 failed");
                    }
                }
            }
        }
    }

    return status;
}

/**
 * @brief Polls for the manual EOM to be done and get the EOM value.
 *
 * @param[in] handle  CDI handle to the serializer.
 * @param[out] eomVal EOM value to be filled.
 *
 * @return NVMEDIA_STATUS_OK Call successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER Invalid param passed.
 * @retval NVMEDIA_STATUS_ERROR Call failed.
 */
static NvMediaStatus PollEOMDone(
    DevBlkCDIDevice const * const handle,
    uint8_t *const     eomVal)
{
    uint8_t i = 0U;
    bool exitFlag = false;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    while (i <= 6U) {
        status = ReadUint8VerifyMAX96717F(handle, REG_EOM_RLMS7, eomVal);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("Serializer write failed");
            exitFlag = true;
        } else {
            if (GetBit(*eomVal, 7U) > 0U) {
                exitFlag = true;
            } else {
                i++;
                /* Sleep for 1000ms */
                /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
                (void)nvsleep(1000000);
            }
        }
        if (exitFlag) {
            break;
        }
    }

    if (i > 6U) {
        status = NVMEDIA_STATUS_ERROR;
        SIPL_LOG_ERR_STR("Error, reached max retries");
    }

    return status;
}

/**
 * @brief Gets the EOM value after running a manual EOM routine.
 *
 * @param[in] handle  CDI handle to the serializer.
 * @param[out] eomVal EOM value to be filled.
 *
 * @return NVMEDIA_STATUS_OK Call successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER Invalid param passed.
 * @retval NVMEDIA_STATUS_ERROR Call failed.
 */
static NvMediaStatus GetEOMVal(
    DevBlkCDIDevice const* handle,
    uint8_t *const eomVal)
{
    NvMediaStatus status = SetupManualEOM(handle);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("Setup of manual EOM failed");
    } else {
        /* Trigger manual EOM */
        status = SetRegBitMAX96717F(handle, REG_EOM_RLMS5, 7U, true);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("Serializer read failed");
        } else {
            uint8_t bit = 0U;
            status = GetRegBitMAX96717F(handle, REG_EOM_RLMS7, 7U, &bit);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR("GetRegBit  failed");
            } else {
                status = PollEOMDone(handle, eomVal);
            }
        }
    }

    return status;
}

/**
 * @brief Gets the Idle/Decode error information.
 *
 * @param[in] handle CDI device handle to the sensor.
 * @param[out] smErrInfo SM error info structure to be filled.
 *
 * @return NVMEDIA_STATUS_OK if successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if invalid param passed.
 * @retval NVMEDIA_STATUS_ERROR in case of failure.
 */
static NvMediaStatus GetIdleDecErrInfo(
    DevBlkCDIDevice const* handle,
    MAX96717FErrorInfo *const smErrInfo)
{
    NvMediaStatus status = ReadUint8VerifyMAX96717F(handle, REG_DEC_ERROR, &smErrInfo->decodeErr);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR("Serializer read decode error failed");
    } else {
        status = ReadUint8VerifyMAX96717F(handle, REG_IDLE_ERROR, &smErrInfo->idleErr);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("Serializer read idle error failed");
        }
    }

    return status;
}

/**
 * @brief Gets the MIPI error information.
 *
 * @param[in] handle CDI device handle to the sensor.
 * @param[out] smErrInfo SM error info structure to be filled.
 *
 * @return NVMEDIA_STATUS_OK if successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if invalid param passed.
 * @retval NVMEDIA_STATUS_ERROR in case of failure.
 */
static NvMediaStatus GetMIPIErrInfo(
    DevBlkCDIDevice const* handle,
    MAX96717FErrorInfo *const smErrInfo)
{
    NvMediaStatus status = ReadUint8VerifyMAX96717F(handle, REG_PHY1_HS_ERR, &smErrInfo->phy1Err);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR("Serializer read failed");
    } else {
        status = ReadUint8VerifyMAX96717F(handle, REG_PHY2_HS_ERR, &smErrInfo->phy2Err);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("Serializer read failed");
        } else {
            uint8_t data = 0U;
            status = ReadUint8VerifyMAX96717F(handle, REG_CTRL1_CSI_H, &data);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR("Serializer read failed");
            } else {
                smErrInfo->ctrl1CSIErr = (uint16_t)data;
                status = ReadUint8VerifyMAX96717F(handle, REG_CTRL1_CSI_L, &data);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR_HEX_UINT("ReadUint8VerifyMAX96717F failed", (uint32_t)status);
                } else {
                    smErrInfo->ctrl1CSIErr = lshift16(smErrInfo->ctrl1CSIErr, 8U);
                    smErrInfo->ctrl1CSIErr |= ((uint16_t)data);
                }
            }
        }
    }

    return status;
}

static NvMediaStatus GetARQErrInfo(
    DevBlkCDIDevice const* handle,
    MAX96717FErrorInfo *const smErrInfo)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t data = 0U;
    /* Check the MAX_RT_ERR bit (b7) of SER ARQ2 */
    status = GetRegBitMAX96717F(handle, REG_ARQ2_SER, 7U, &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("Serializer read failed");
    } else {
#if FUSA_CDD_NV
        smErrInfo->maxRTErrSer = IsBitSet(data, MAX_RT_ERR_ARQ2_FLAG_MASK);
#else
        smErrInfo->maxRTErrSer = data;
#endif
        /* Check the MAX_RT_ERR bit (b7) of GPIO ARQ2 */
        status = GetRegBitMAX96717F(handle, REG_ARQ2_GPIO, 7U, &data);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("Serializer read failed");
        } else {
#if FUSA_CDD_NV
            smErrInfo->maxRTErrGPIO = IsBitSet(data, MAX_RT_ERR_ARQ2_FLAG_MASK);
#else
            smErrInfo->maxRTErrGPIO = data;
#endif
            /* Check the MAX_RT_ERR bit (b7) of I2C_X ARQ2 */
            status = GetRegBitMAX96717F(handle, REG_ARQ2_I2C_X, 7U, &data);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR("Serializer read failed");
            } else {
#if FUSA_CDD_NV
                smErrInfo->maxRTErrI2CX = IsBitSet(data, MAX_RT_ERR_ARQ2_FLAG_MASK);
#else
                smErrInfo->maxRTErrI2CX = data;
#endif
                /* Check the MAX_RT_ERR bit (b7) of I2C_Y ARQ2 */
                status = GetRegBitMAX96717F(handle, REG_ARQ2_I2C_Y, 7U, &data);
                if (status != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR("Serializer read failed");
                } else {
#if FUSA_CDD_NV
                    smErrInfo->maxRTErrI2CY = IsBitSet(data, MAX_RT_ERR_ARQ2_FLAG_MASK);
#else
                    smErrInfo->maxRTErrI2CY = data;
#endif
                }
            }
        }
    }

    return status;
}

/**
 * @brief Get each ADC channel hi/lo error information
 *
 * @param[in] handle CDI device handle to the serializer
 * @param[out] smErrInf SM error info structure to be filled.
 *
 * @retval NVMEDIA_STATUS_OK success.
 * @retval NVMEDIA_STATUS_ERROR in case of failure.
 */
static NvMediaStatus GetADCchErrInfo(
    DevBlkCDIDevice const *handle,
    MAX96717FErrorInfo *const smErrInfo)
{
    NvMediaStatus status = ReadUint8VerifyMAX96717F(handle, REG_ADC_INTR0, &smErrInfo->adcINTR0);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR("Serializer read adc_intr0 failed");
    } else {
        status = ReadUint8VerifyMAX96717F( handle, REG_ADC_INTR1, &smErrInfo->adcINTR1);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("Serializer read adc_hi_if failed");
        } else {
            status = ReadUint8VerifyMAX96717F(handle, REG_ADC_INTR2, &smErrInfo->adcINTR2);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR("Serializer read adc_lo_if failed");
            } else {
#if FUSA_CDD_NV
                uint8_t regVal = 0U;
                status = ReadUint8VerifyMAX96717F(handle, REG_ADC_INTR3, &regVal);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR("Serializer read ADC_INTR3 register failed");
                } else {
                    smErrInfo->tmon_err_int_flag = IsBitSet(regVal, TMON_ERR_IF_FLAG_MASK);
                }
#endif
            }
        }
    }

    return status;
}

NvMediaStatus
MAX96717FChkSMStatus(
    DevBlkCDIDevice const* handle,
    size_t *regSize,
    uint8_t * const buffer)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if ((NULL == buffer) || (NULL == regSize)) {
        SIPL_LOG_ERR_STR("MAX96717FChkSMStatus: Handle is bad parameter");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        /* Check all SM status */
        uint8_t bufIdx = 0U;
        SMErrInfoResult result[SM_LIST_MAX_NUM];
        (void)memset(result, 0, sizeof(result));

        for (uint8_t i = (uint8_t)SM01_BIST_FOR_SERIAL_LINK;
             i <= (uint8_t)SM40_GPIO_OPEN_DETECTION; i++) {
            MAX96717FSMErrInfoRegSet const *regSet = GetRegSetMAX96717F(i);
            /* Not implement or not support SM */
            if (NULL == regSet) {
                PrintLogMsg(LOG_LEVEL_INFO, "MAX96717FChkSMStatus: %d SM didn't support\n", i);
            } else {
                uint8_t regVal = 0U;
                status = MAX96717FGetSMStatus(handle,
                                              i,
                                              &regVal);
                if (NVMEDIA_STATUS_OK != status) {
                    PrintLogMsg(LOG_LEVEL_INFO, "MAX96717FChkSMStatus: %d SM GetSMStatus failed\n",
                                i);
                    break;
                } else {
                    result[bufIdx].index = i;
                    result[bufIdx].regVal = regVal;
                    status = ClrSMStatusMAX96717F(handle, i, regVal);
                    if ((uint16_t)UINT8_MAX >
                            toUint8FromUint16(saturatingAddUint16((uint16_t)bufIdx, 1U))) {
                        bufIdx += (uint8_t)1U;
                    } else {
                        SIPL_LOG_ERR_STR("buffer index overflow");
                        status = NVMEDIA_STATUS_ERROR;
                    }
                }
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            *regSize = sizeof(SMErrInfoResult)*bufIdx;
            (void)fusa_memcpy(buffer, result, *regSize);
        } else {
            SIPL_LOG_ERR_STR("MAX96717FChkSMStatus is failed");
        }
    }

    return status;
}

#if FUSA_CDD_NV
/**
 * @brief Fill the error info structure according to error bits for INTR3 register
 *
 * @param[in] regVal  register value of INTR3
 * @param[in] errInfo MAX96717FErrorInfo structure
 *
 * @retval NVMEDIA_STATUS_OK success.
 * @retval NVMEDIA_STATUS_ERROR in case of failure.
 */
static void
GetINTR3ErrInfo(
    uint8_t regVal,
    MAX96717FErrorInfo *const errInfo
)
{
    errInfo->refgen_unlocked_flag = IsBitSet(regVal, INTR3_REFGEN_UNLOCKED_MASK);
    errInfo->idle_err_flag        = IsBitSet(regVal, INTR3_IDLE_ERR_FLAG_MASK);
    errInfo->dec_err_flag_a       = IsBitSet(regVal, INTR3_DEC_ERR_FLAG_A_MASK);
}

/**
 * @brief Fill the error info structure according to error bits for INTR5 register
 *
 * @param[in] regVal  register value of INTR5
 * @param[in] errInfo MAX96717FErrorInfo structure
 *
 * @retval NVMEDIA_STATUS_OK success.
 * @retval NVMEDIA_STATUS_ERROR in case of failure.
 */
static void
GetINTR5ErrInfo(
    uint8_t regVal,
    MAX96717FErrorInfo *const errInfo
)
{
    errInfo->vreg_ov_flag  = IsBitSet(regVal, INTR5_VREG_OV_FLAG_MASK);
    errInfo->eom_err_flag  = IsBitSet(regVal, INTR5_EOM_ERR_FLAG_A_MASK);
    errInfo->vdd_ov_flag   = IsBitSet(regVal, INTR5_VDD_OV_FLAG_MASK);
    errInfo->vdd18_ov_flag = IsBitSet(regVal, INTR5_VDD18_OV_FLAG_MASK);
    errInfo->max_rt_flag   = IsBitSet(regVal, INTR5_MAX_RT_FLAG_MASK);
    errInfo->rt_cnt_flag   = IsBitSet(regVal, INTR5_RT_CNT_FLAG_MASK);
}

/**
 * @brief Fill the error info structure according to error bits for INTR7 register
 *
 * @param[in] regVal  register value of INTR7
 * @param[in] errInfo MAX96717FErrorInfo structure
 *
 * @retval NVMEDIA_STATUS_OK success.
 * @retval NVMEDIA_STATUS_ERROR in case of failure.
 */
static void
GetINTR7ErrInfo(
    uint8_t regVal,
    MAX96717FErrorInfo *const errInfo
)
{
    errInfo->vddcmp_int_flag    = IsBitSet(regVal, INTR7_VDDCMP_INT_FLAG_MASK);
    errInfo->porz_int_flag      = IsBitSet(regVal, INTR7_PORZ_INT_FLAG_MASK);
    errInfo->vddbad_int_flag    = IsBitSet(regVal, INTR7_VDDBAD_INT_FLAG_MASK);
    errInfo->efuse_crc_err_flag = IsBitSet(regVal, INTR7_EFUSE_CRC_ERR_MASK);
    errInfo->adc_int_flag       = IsBitSet(regVal, INTR7_ADC_INTR_FLAG_MASK);
    errInfo->mipi_err_flag      = IsBitSet(regVal, INTR7_MIPI_ERR_FLAG_MASK);
}

/**
 * @brief Fill the remaining error info structure bits
 *
 * @param[in] handle  CDI handle to the serializer.
 * @param[in] errInfo MAX96717FErrorInfo structure
 *
 * @retval NVMEDIA_STATUS_OK success.
 * @retval NVMEDIA_STATUS_ERROR in case of failure.
 */
static NvMediaStatus
GetMiscErrorInfo(
    DevBlkCDIDevice const * const handle,
    MAX96717FErrorInfo *const errInfo
)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regVal = 0U;

    status = ReadUint8VerifyMAX96717F(handle, REG_FS_INTR1, &regVal);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_HEX_UINT("Serializer read failed for register",
                                    (uint32_t)REG_FS_INTR1);
    } else {
        errInfo->mem_ecc_err2_int_flag = IsBitSet(regVal, FS_INTR1_MEM_ECC_ERR2_INT_MASK);
        errInfo->mem_ecc_err1_int_flag = IsBitSet(regVal, FS_INTR1_MEM_ECC_ERR1_INT_MASK);
        errInfo->reg_crc_err_flag      = IsBitSet(regVal, FS_INTR1_REG_CRC_ERR_FLAG_MASK);

        status = ReadUint8VerifyMAX96717F(handle, REG_REF_VTG0, &regVal);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("Serializer read failed for register",
                                      (uint32_t)REG_REF_VTG0);
        } else {
            errInfo->refgen_locked_flag = IsBitSet(regVal, REFGEN_LOCKED_FLAG_MASK);

            status = ReadUint8VerifyMAX96717F(handle, REG_CTRL3, &regVal);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT("Serializer read failed for register",
                                          (uint32_t)REG_CTRL3);
            } else {
                errInfo->locked_flag = IsBitSet(regVal, LOCKED_FLAG_MASK);

                status = ReadUint8VerifyMAX96717F(handle, REG_EXT8, &regVal);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR_HEX_UINT("Serializer read failed for register",
                                              (uint32_t)REG_EXT8);
                } else {
                    errInfo->tun_fifo_overflow_flag = IsBitSet(regVal,
                                                               TUN_FIFO_OVERFLOW_FLAG_MASK);
                }
            }
        }
    }

    return status;
}
#endif

static NvMediaStatus GetSMErrInfoMAX96717F(
    DevBlkCDIDevice const* handle,
    MAX96717FErrorInfo *const errInfo)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t regVal = 0U;

    if (NULL == errInfo) {
        SIPL_LOG_ERR_STR("Bad input parameter");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        /* Get interrupt error mask */
        /* Reading INTR3 register to get status of REFGEN_UNLOCKED bit(SM31) */
        status = ReadUint8VerifyMAX96717F(handle, REG_INTR3, &regVal);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("Serializer read failed:", (uint32_t)REG_INTR3);
        } else {
#if FUSA_CDD_NV
            GetINTR3ErrInfo(regVal, errInfo);
#else
            errInfo->intr3Reg = regVal;
#endif

            status = ReadUint8VerifyMAX96717F(handle, REG_INTR5, &regVal);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT("Serializer read failed:", (uint32_t)REG_INTR5);
            } else {
#if FUSA_CDD_NV
                GetINTR5ErrInfo(regVal, errInfo);
#else
                errInfo->intr5Reg = regVal;
#endif

                /* Reading INTR7 register to get status of MIPI_ERR_FLAG(SM16)
                 * and EFUSE_CRC_ERR(SM29) bit */
                status = ReadUint8VerifyMAX96717F(handle, REG_INTR7, &regVal);
                if (NVMEDIA_STATUS_OK != status) {
                    SIPL_LOG_ERR_STR_HEX_UINT("Serializer read failed:", (uint32_t)REG_INTR7);
                } else {
#if FUSA_CDD_NV
                    GetINTR7ErrInfo(regVal, errInfo);

                    status = GetMiscErrorInfo(handle, errInfo);
                    if (NVMEDIA_STATUS_OK != status) {
                        SIPL_LOG_ERR_STR("MAX96717F: FillMAX96717FErrorInfo failed.")
                    } else {
#else
                        errInfo->intr7Reg = regVal;
#endif
                        status = GetMIPIErrInfo(handle, errInfo);
                        if (NVMEDIA_STATUS_OK != status) {
                            SIPL_LOG_ERR_STR("Serializer read failed");
                        } else {
                            status = GetARQErrInfo(handle, errInfo);
                            if (NVMEDIA_STATUS_OK != status) {
                                SIPL_LOG_ERR_STR("Serializer read failed");
                            } else {
                                status = GetIdleDecErrInfo(handle, errInfo);
                                if (NVMEDIA_STATUS_OK != status) {
                                    SIPL_LOG_ERR_STR("Serializer read failed");
                                } else {
                                    status = GetADCchErrInfo(handle, errInfo);
                                    if (NVMEDIA_STATUS_OK != status) {
                                        SIPL_LOG_ERR_STR("Serializer read failed");
                                    }
                                }
                            }
                        }
#if FUSA_CDD_NV
                    }
#endif
                }
            }
        }
    }

    return status;
}

static NvMediaStatus
ClrSMStatusInternalMAX96717F(
    DevBlkCDIDevice const * const handle,
    SMErrClearRegSet const* clrRegSet,
    uint8_t regVal
)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    /* Reported Error status */
    if ((clrRegSet->statusVal == regVal) && (clrRegSet->writeClr)) {
        /* Write clear case */
        uint8_t clearRegVal = 0U;
        status = ReadUint8VerifyMAX96717F(handle, clrRegSet->clrAddr, &clearRegVal);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("ClrSMStatusInternalMAX96717F: i2c read failed");
        } else {
            clearRegVal = SetBit(clearRegVal, clrRegSet->clrBit);
            if (clrRegSet->clrBit2Valid) {
                clearRegVal = SetBit(clearRegVal, clrRegSet->clrBit2);
            }

            status = WriteUint8VerifyMAX96717F(handle, clrRegSet->clrAddr, clearRegVal);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR("ClrSMStatusInternalMAX96717F: i2c write fail");
            }
        }
    } else if ((clrRegSet->statusVal == regVal) && !(clrRegSet->writeClr)) {
        /* Read Reg Cleard */
        uint8_t regRead;
        status = ReadUint8VerifyMAX96717F(handle, clrRegSet->clrAddr, &regRead);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("ClrSMStatusInternalMAX96717F: read clear fail");
        }
    } else {
        PrintLogMsg(LOG_LEVEL_INFO, "ClrSMStatusInternalMAX96717F: SM is not Error Status\n");
    }

    return status;
}

static NvMediaStatus
ClrSMStatusMAX96717F(
    DevBlkCDIDevice const * const handle,
    uint8_t index,
    uint8_t regVal)
{
    const uint8_t smlistmax = (uint8_t)SM_LIST_MAX_NUM;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    const SMErrClearRegSet clrRegSet[] = {
        {.statusVal=0x01U, .clrAddr=0x0112U, .clrBit=5U,
         .writeClr=false, .index=(uint8_t)SM17_FIFO_OVERFLOW_DETECTION,
         .clrBit2Valid = false, .clrBit2 = 0xA5U},
#if !FUSA_CDD_NV
        {.statusVal=0x40U, .clrAddr=0x1D09U, .clrBit=0U,
         .writeClr=true, .index=(uint8_t)SM21_CRC_ON_I2C_UART_TRANSACTIONS,
         .clrBit2Valid = false, .clrBit2 = 0xA5U},
        {.statusVal=0x80U, .clrAddr=0x1D09U, .clrBit=1U,
         .writeClr=true, .index=(uint8_t)SM22_MESSAGE_COUNTER,
         .clrBit2Valid = false, .clrBit2 = 0xA5U},
#endif
        {.statusVal=0x30U, .clrAddr=0x1D14U, .clrBit=0U,
         .writeClr=true, .index=(uint8_t)SM26_MEMORY_ECC,
         .clrBit2Valid = true, .clrBit2 = 1U},
        {.statusVal=0x01U, .clrAddr=0x1D00U, .clrBit=0U,
         .writeClr=true, .index=(uint8_t)SM27_CONFIGURATION_REGISTER_CRC,
         .clrBit2Valid = false, .clrBit2 = 0xA5U},
        {.statusVal=0x04U, .clrAddr=0x0513U, .clrBit=1U,
         .writeClr=false, .index=(uint8_t)SM28_SELF_TEST_OF_TEMPERATURE_SENSOR,
         .clrBit2Valid = false, .clrBit2 = 0xA5U},
        /* Write clear to bit RESET_EFUSE_CRC_ERR in CC_RTTN_ERR(0X1D5F)
         * to clear SM29 */
        {.statusVal=0x10U, .clrAddr=0x1D5FU, .clrBit=2U,
         .writeClr=true, .index=(uint8_t)SM29_CRC_OF_NONVOLATILE_MEMORY,
         .clrBit2Valid = false, .clrBit2 = 0xA5U},
    };

    if (index < smlistmax) {
        uint8_t arrSize = (uint8_t)(I2C_ARRAY_SIZE(clrRegSet) & 0xFFU);

        for (uint8_t i=0U; i < arrSize; i++) {
            uint8_t tmpIndex = clrRegSet[i].index;

            if (index == tmpIndex) {
                status = ClrSMStatusInternalMAX96717F(handle, &clrRegSet[i], regVal);
                if (status != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR("ClrSMStatusInternalMAX96717F is failed");
                    break;
                }
            } else {
                PrintLogMsg(LOG_LEVEL_INFO, "SM didn't need to clear\n");
            }
        }
    } else {
        SIPL_LOG_ERR_STR_HEX_UINT("%d index out of range", (uint32_t)index);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* If clear was successful, check if ERROR bit is 0 */
    if (NVMEDIA_STATUS_OK == status) {
        uint8_t errBit = 0U;
        status = GetRegBitMAX96717F(handle, REG_CTRL3, 2U, &errBit);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("MAX96717FReadRegisters: Regiser read fail",
                                    (uint32_t)status);
        } else {
            if (errBit > 0U) {
                status = NVMEDIA_STATUS_ERROR;
                SIPL_LOG_ERR_STR("Clear unsuccessful!");
            }
        }
    }


    return status;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#ifdef NVMEDIA_QNX
static NvMediaStatus
SetTemperatureMAX96717F(
    DevBlkCDIDevice const* handle)
{
    DevBlkCDII2CReg const temperatureRegs[] = {
        {0x0535U, 0x80U},
        {0x0500U, 0x00U},
        {0x0501U, 0x00U},
        {0x0502U, 0x00U},
        {0x0501U, 0xB8U}, /* 0x08 value is Enable ADC clock, select internal reference */
        {0x050CU, 0x83U},
        {0x0500U, 0x1EU}, /* Power up ADC, input buffer, reference buffer and charge pump */
        {0x050CU, 0xFFU},
        {0x1D28U, 0x01U}, /* Trigger temperature reading */
    };
    DevBlkCDII2CRegList temperature = {
        .regs = temperatureRegs,
        .numRegs = I2C_ARRAY_SIZE(temperatureRegs)
    };

    NvMediaStatus status = WriteArrayVerifyMAX96717F(handle, &temperature);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_HEX_UINT(
            "SetTemperatureMAX96717F: WriteArrayVerifyMAX96717F failed", (uint32_t)status);
    } else {
        bool adcCaldoneIf = false;
        /* waiting until adc_done_if = 1 */
        for (uint8_t i=0U; i < MAX_RETRY_COUNT; i++) {
            uint8_t adcStatus = 0U;
            status = ReadUint8VerifyMAX96717F(handle, REG_ADC_INTR0, &adcStatus);
            if (NVMEDIA_STATUS_OK == status) {
                if (GetBit(adcStatus, 7U) > 0U) {
                    PrintLogMsg(LOG_LEVEL_INFO, "adc_calDone_if is done\n");
                    adcCaldoneIf = true;
                    break;
                } else {
                    /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
                    (void)nvsleep(ADC_DONE_IF_SLEEP_US);
                }
            } else {
                SIPL_LOG_ERR_STR("SetTemperatureMAX96717F: Failed to read");
                break;
            }
        }

        if (adcCaldoneIf == false) {
            SIPL_LOG_ERR_STR("adc_calDone_if is not set");
        }
    }

    return status;
}
#endif

NvMediaStatus
MAX96717FGetMFPState(
    DevBlkCDIDevice const* handle,
    uint16_t regAddr,
    uint8_t *state
)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (NULL == state) {
        SIPL_LOG_ERR_STR("MAX96717FGetMFPState: Bad parameter");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        uint8_t regVal = 0U;

        status = ReadUint8VerifyMAX96717F(handle, regAddr, &regVal);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("ReadUint8VerifyMAX96717F failed", (uint32_t)status);
        } else {
            if (GetBit(regVal, MFP_INPUT_STATUS_BIT) > 0U) {
                *state = MAX96717F_MFP_STATUS_OK;
            } else {
                *state = MAX96717F_MFP_STATUS_ERR;
            }
        }
    }
    return status;
}

NvMediaStatus
MAX96717FConfigPatternGenerator(
    DevBlkCDIDevice const *handle, bool enable)
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;

    if (enable) {
        DevBlkCDII2CReg ConfigRegs[] = {
            /* Disable auto bpp */
            {REG_VIDEO_TX0, 0x60U},
            /* Internal PCLK generation 75MHz */
            {REG_VTX1, 0x0AU},
            /* Enable pattern generator */
            {REG_VTX29, 0x01U}
        };
        DevBlkCDII2CRegList ConfigRegList = {
            .regs = ConfigRegs,
            .numRegs = I2C_ARRAY_SIZE(ConfigRegs)
        };
        nvmStatus = WriteArrayVerifyMAX96717F(handle, &ConfigRegList);
    } else {
        DevBlkCDII2CReg ConfigRegs[] = {
            /* Reset auto bpp */
            {REG_VIDEO_TX0, 0x68U},
            /* Reset Internal PCLK generation */
            {REG_VTX1, 0x01U},
        };
        DevBlkCDII2CRegList ConfigRegList = {
            .regs = ConfigRegs,
            .numRegs = I2C_ARRAY_SIZE(ConfigRegs)
        };
        nvmStatus = WriteArrayVerifyMAX96717F(handle, &ConfigRegList);
    }

    return nvmStatus;
}

NvMediaStatus
MAX96717FVPRBSGenerator(
    DevBlkCDIDevice const *handle,
    bool enable)
{
    uint8_t data = 0x00U;
    if(enable) {
        data = 0x81U;
    }

    NvMediaStatus const nvmStatus = WriteUint8VerifyMAX96717F(handle, REG_VTX29, data);
    if (NVMEDIA_STATUS_OK != nvmStatus) {
        SIPL_LOG_ERR_STR_HEX_UINT("WriteUint8VerifyMAX96717F failed", (uint32_t)nvmStatus);
    }

    return nvmStatus;
}

NvMediaStatus
MAX96717FVerifyPCLKDET(
    DevBlkCDIDevice const *handle)
{
    uint8_t data;
    /* Verify serializer has PCLKDET = 1 */
    NvMediaStatus nvmStatus = ReadUint8VerifyMAX96717F(handle, REG_VIDEO_TX2, &data);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX96717FVerifyPCLKDET: Failed to read");
    } else {
        if ((rshift8(data, 7) & 1U) == 0U) {
            nvmStatus = NVMEDIA_STATUS_ERROR;
            SIPL_LOG_ERR_STR("MAX96717FVerifyPCLKDET: PCLKDET is 0");
        }
    }

    return nvmStatus;
}

NvMediaStatus
MAX96717FERRG(
    DevBlkCDIDevice const *handle,
    bool enableERRG)
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    if (enableERRG) {
        DevBlkCDII2CReg ERRGRegs[] = {
            /* Config ERRG */
            {REG_TX2, 0x80U},
            /* Enable ERRG */
            {REG_TX1, 0x10U}
        };
        DevBlkCDII2CRegList ERRGRegList = {
            .regs = ERRGRegs,
            .numRegs = I2C_ARRAY_SIZE(ERRGRegs)
        };

        nvmStatus = WriteArrayVerifyMAX96717F(handle, &ERRGRegList);
        if (NVMEDIA_STATUS_OK != nvmStatus) {
            SIPL_LOG_ERR_STR_HEX_UINT("WriteArrayVerifyMAX96717F failed", (uint32_t)nvmStatus);
        }
    } else {
        /* Disable ERRG */
        nvmStatus = WriteUint8VerifyMAX96717F(handle, REG_TX1, 0x00U);
        if (NVMEDIA_STATUS_OK != nvmStatus) {
            SIPL_LOG_ERR_STR_HEX_UINT("WriteUint8VerifyMAX96717F failed", (uint32_t)nvmStatus);
        }
    }

    return nvmStatus;
}

NvMediaStatus
MAX96717FIsErrbSet(
    DevBlkCDIDevice const * const handle,
    uint8_t* const errbStatus)
{

    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;

    if (errbStatus == NULL) {
        SIPL_LOG_ERR_STR("Bad input parameter");
        nvmStatus = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        uint8_t data;
        nvmStatus = ReadUint8VerifyMAX96717F(handle, (uint16_t)REG_CTRL3, &data);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX96717F read error status failed");
        } else {
            *errbStatus = ((data & 0x4U) != 0U)? 0x01U: 0x00U;
        }
    }

    return nvmStatus;
}

NvMediaStatus
MAX96717FIsGPIOValid(
    uint8_t const gpioInd)
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;

    /**
     * check gpioInd against SER_MFP10 which is the last
     * valid GPIO index for MAX96717(F) Serializer
     */
    if (gpioInd > SER_MFP10) {
        SIPL_LOG_ERR_STR("Invalid GPIO Index");
        nvmStatus = NVMEDIA_STATUS_BAD_PARAMETER;
    }
    return nvmStatus;
}

NvMediaStatus
MAX96717FIsGPIOSet(
    DevBlkCDIDevice const * const handle,
    uint8_t const gpioInd,
    uint8_t* const gpioStatus)
{
    NvMediaStatus nvmStatus = MAX96717FIsGPIOValid(gpioInd);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("Invalid GPIO Index");
    } else {
        if (gpioStatus == NULL) {
            SIPL_LOG_ERR_STR("Bad input parameter");
            nvmStatus = NVMEDIA_STATUS_BAD_PARAMETER;
        } else {
            uint8_t data;
            nvmStatus = ReadUint8VerifyMAX96717F(handle, (uint16_t)GET_GPIO_A_ADDR(gpioInd), &data);
            if (nvmStatus != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("MAX96717F read error status failed", (int32_t)nvmStatus);
            } else {
                *gpioStatus = ((data & 0x8U) != 0U)? 0x01U: 0x00U;
            }
        }
    }

    return nvmStatus;
}

/**
 * Setup address translations B
 */
NvMediaStatus MAX96717FSetupAddressTranslationsB(
    DevBlkCDIDevice const* const serCDI,
    uint8_t const src,
    uint8_t const dst)
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    ReadWriteParamsMAX96717F paramsMAX96717F = {};

    paramsMAX96717F.Translator.source = src;
    paramsMAX96717F.Translator.destination = dst;
    PrintLogMsg(LOG_LEVEL_INFO,"Translate device addr %x to %x\n",
                paramsMAX96717F.Translator.source, paramsMAX96717F.Translator.destination);

    nvmStatus = MAX96717FWriteParameters(
                            serCDI,
                            CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B,
                            sizeof(paramsMAX96717F.Translator),
                            &paramsMAX96717F);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B failed"
            "with NvMedia error:", (int32_t)(nvmStatus));
    }

    return nvmStatus;
}

#ifdef NVMEDIA_QNX
/* Execute verification steps adopted from the MAX96717 Serializer
 * Implementation Guide for its SM23. */
NvMediaStatus
MAX96717FVerifyGPIOReadBackStatus(
    DevBlkCDIDevice const * const handle,
    uint8_t const * const gpioListSM23,
    uint8_t const numListSM23)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (NULL == gpioListSM23) {
        SIPL_LOG_ERR_STR(
            "MAX96717FVerifyGPIOReadBackStatus: Bad parameter - Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (status == NVMEDIA_STATUS_OK) {
        bool lockStatus;
        bool errbStatus;
        /* Ensure Link is Locked and ERRB is Not Asserted */
        status = ChkLockErrbStatusMAX96717F(handle, &lockStatus, &errbStatus);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR(
                "MAX96717FVerifyGPIOReadBackStatus: Call to "
                "ChkLockErrbStatusMAX96717F failed");
        } else {
            if (lockStatus == (bool)MAX96717F_LINK_UNLOCKED) {
                SIPL_LOG_ERR_STR(
                    "MAX96717FVerifyGPIOReadBackStatus: Link is not Locked");
                status = NVMEDIA_STATUS_ERROR;
            }
            if (errbStatus == (bool)MAX96717F_ERRB_ASSERTED) {
                SIPL_LOG_ERR_STR(
                    "MAX96717FVerifyGPIOReadBackStatus: ERRB is Asserted");
                status = NVMEDIA_STATUS_ERROR;
            }
        }
    }

    for (uint8_t index = 0U; index < numListSM23; index++) {
        uint16_t addrGPIOA;
        uint8_t readGPIOA = 0U;
        uint8_t initValA = 0U;

        if (status == NVMEDIA_STATUS_OK) {
            addrGPIOA = (uint16_t)GET_GPIO_A_ADDR(gpioListSM23[index]);

            /* Get the initial value of GPIO_A so as to restore it once SM23 is verified */
            status = ReadUint8VerifyMAX96717F(handle, addrGPIOA, &initValA);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "MAX96717FVerifyGPIOReadBackStatus: Failed to read GPIO_A "
                    "at address ", (uint32_t)addrGPIOA);
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            /* Drive 4th bit of GPIO_A high, GPIO_OUT = 1
             * Drive 0th bit of GPIO_A low, GPIO_OUT_DIS = 0
             * GPIO_A = 0x10 */
            status = WriteUint8VerifyMAX96717F(handle, addrGPIOA, 0x10U);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "MAX96717FVerifyGPIOReadBackStatus: Failed to write GPIO_A "
                    "at address ", (uint32_t)addrGPIOA);
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            /* Read GPIO_A */
            status = ReadUint8VerifyMAX96717F(handle, addrGPIOA, &readGPIOA);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "MAX96717FVerifyGPIOReadBackStatus: Failed to read GPIO_A "
                    "at address ", (uint32_t)addrGPIOA);
            } else {
                /* Verify GPIO_IN is set HIGH */
                if ((rshift8(readGPIOA, 3U) & (uint8_t)1U) == (uint8_t)0U) {
                    SIPL_LOG_ERR_STR_HEX_UINT(
                        "MAX96717FVerifyGPIOReadBackStatus: GPIO_IN is"
                        "erroneously unset, MFP: ", (uint32_t)addrGPIOA);
                    status = NVMEDIA_STATUS_ERROR;
                }
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            /* Drive 4th bit of GPIO_A low, GPIO_OUT=0
             * Drive 0th bit of GPIO_A low, GPIO_OUT_DIS = 0
             * GPIO_A = 0x00*/
            status = WriteUint8VerifyMAX96717F(handle, addrGPIOA, 0x00U);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "MAX96717FVerifyGPIOReadBackStatus: Failed to write GPIO_A "
                    "at address ", (uint32_t)addrGPIOA);
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            /* Read GPIO_A */
            status = ReadUint8VerifyMAX96717F(handle, addrGPIOA, &readGPIOA);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "MAX96717FVerifyGPIOReadBackStatus: Failed to read GPIO_A "
                    "at address ", (uint32_t)addrGPIOA);
            } else {
                /* Verify GPIO_IN is set LOW */
                if ((rshift8(readGPIOA, 3U) & (uint8_t)1U) != (uint8_t)0U) {
                    SIPL_LOG_ERR_STR_HEX_UINT(
                        "MAX96717FVerifyGPIOReadBackStatus: GPIO_IN is"
                        "erroneously set,  MFP: ", (uint32_t)addrGPIOA);
                    status = NVMEDIA_STATUS_ERROR;
                }
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            /* Write initial value back to GPIO_A */
            status = WriteUint8VerifyMAX96717F(handle, addrGPIOA, initValA);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "MAX96717FVerifyGPIOReadBackStatus: Failed to read GPIO_A "
                    "at address ", (uint32_t)addrGPIOA);
            }
        }

        if (status != NVMEDIA_STATUS_OK) {
            break;
        }
    }

    return status;
}

/* Execute verification steps adopted from the MAX96717 Serializer
 * Implementation Guide for its SM40.
 *
 * This test is meant to be run only on GPIO pins that are supposed to be
 * externally driven (and hence not open-circuited). The criterion for the test
 * to pass is that we should not be able to successfully effect both the
 * pull-up and pull-down states (HIGH and LOW) of the GPIO pin tested, which
 * means the values read in the two cases should not be different - only an
 * open pin will toggle freely even with a weak pull-up/pull-down resistor. We
 * fail the test if the pin states read are different.
 */
NvMediaStatus
MAX96717FVerifyGPIOOpenDetection(
    DevBlkCDIDevice const * const handle,
    uint8_t const * const gpioListSM40,
    uint8_t const numListSM40)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (NULL == gpioListSM40) {
        SIPL_LOG_ERR_STR(
            "MAX96717FVerifyGPIOOpenDetection: Bad parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (status == NVMEDIA_STATUS_OK) {
        bool lockStatus;
        bool errbStatus;
        /* Ensure Link is Locked and ERRB is Not Asserted */
        status = ChkLockErrbStatusMAX96717F(handle, &lockStatus, &errbStatus);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR(
                "MAX96717FVerifyGPIOOpenDetection: Call to "
                "ChkLockErrbStatusMAX96717F failed");
        } else {
            if (lockStatus == (bool)MAX96717F_LINK_UNLOCKED) {
                SIPL_LOG_ERR_STR(
                    "MAX96717FVerifyGPIOOpenDetection: Link is not Locked");
                status = NVMEDIA_STATUS_ERROR;
            }
            if (errbStatus == (bool)MAX96717F_ERRB_ASSERTED) {
                SIPL_LOG_ERR_STR(
                    "MAX96717FVerifyGPIOOpenDetection: ERRB is Asserted");
                status = NVMEDIA_STATUS_ERROR;
            }
        }
    }

    for (uint8_t index = 0U; index < numListSM40; index++) {
        uint16_t addrGPIOA;
        uint16_t addrGPIOB;
        uint8_t readGPIOAPullUp = 0U;
        uint8_t readGPIOAPullDown = 0U;
        DevBlkCDII2CReg initVals[2];

        if (status == NVMEDIA_STATUS_OK) {
            addrGPIOA = (uint16_t)GET_GPIO_A_ADDR(gpioListSM40[index]);
            addrGPIOB = addrGPIOA + (uint16_t)1U;

            /* For storing initial values of GPIO_A and GPIO_B and restoring
            * them once SM40 is verified */
            initVals[0].address = addrGPIOA;
            initVals[0].data = 0U;
            initVals[0].delayUsec = 0U;
            initVals[1].address = addrGPIOB;
            initVals[1].data = 0U;
            initVals[1].delayUsec = 0U;

            /* Read GPIO_A and GPIO_B to be able to restore them */
            DevBlkCDII2CRegListWritable const ReadGPIORegList = {
                .regs = initVals,
                .numRegs = I2C_ARRAY_SIZE(initVals)
            };
            status = ReadArrayVerifyMAX96717F(handle, &ReadGPIORegList);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_2HEX(
                    "MAX96717FVerifyGPIOOpenDetection: Failed to read GPIO_A "
                    "and GPIO_B at address ",
                    (uint32_t)addrGPIOA, (uint32_t)addrGPIOB);
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            /* Configure GPIO_A register with settings common to the
             * Pull-Up and Pull-Down tests
             * GPIO_OUT_DIS = 1 and RES_CFG = 1
             * GPIO_A = 0x81 */
            status = WriteUint8VerifyMAX96717F(handle, addrGPIOA, 0x81U);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "MAX96717FVerifyGPIOOpenDetection: Failed to write GPIO_A "
                    "at address ", (uint32_t)addrGPIOA);
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            /* Pull-Up Test
             * GPIO_B: Set PULL_UPDN_SEL to 0b01, selecting pullup buffer configuration
             * GPIO_B: Set OUT_TYPE to 1, selecting push-pull output type
             * GPIO_B = 0x60*/
            status = WriteUint8VerifyMAX96717F(handle, addrGPIOB, 0x60U);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "MAX96717FVerifyGPIOOpenDetection: Failed to write GPIO_B "
                    "at address ", (uint32_t)addrGPIOB);
            }
        }
        if (status == NVMEDIA_STATUS_OK) {
            /* Reading GPIO_A */
            status = ReadUint8VerifyMAX96717F(handle, addrGPIOA, &readGPIOAPullUp);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "MAX96717FVerifyGPIOOpenDetection: Failed to read GPIO_A "
                    "at address ", (uint32_t)addrGPIOA);
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            /* Pull-Down Test
             * GPIO_B: Set PULL_UPDN_SEL to 0b10, selecting pulldown buffer configuration
             * GPIO_B: Set OUT_TYPE to 1, selecting push-pull output type
             * GPIO_B = 0xA0 */
            status = WriteUint8VerifyMAX96717F(handle, addrGPIOB, 0xA0U);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "MAX96717FVerifyGPIOOpenDetection: Failed to write GPIO_B "
                    "at address ", (uint32_t)addrGPIOB);
            }
        }
        if (status == NVMEDIA_STATUS_OK) {
            /* Reading GPIO_A */
            status = ReadUint8VerifyMAX96717F(handle, addrGPIOA, &readGPIOAPullDown);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "MAX96717FVerifyGPIOOpenDetection: Failed to read GPIO_A "
                    "at address ", (uint32_t)addrGPIOA);
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            /* Verify if the GPIO_IN(bit 3) of GPIO_A remains unchanged between
             * PullUp and PullDown test, if it changes return ERROR */
            if ((rshift8(readGPIOAPullUp, 3U) & (uint8_t)1U) !=
                (rshift8(readGPIOAPullDown, 3U) & (uint8_t)1U)) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                        "MAX96717FVerifyGPIOOpenDetection: Bit Mismatched while"
                        " verifing PullUp/Down test at address ",
                        (uint32_t)addrGPIOA);
                status = NVMEDIA_STATUS_ERROR;
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            DevBlkCDII2CRegList const WriteGPIORegList = {
                .regs = initVals,
                .numRegs = I2C_ARRAY_SIZE(initVals)
            };
            /* Restore the init values of GPIO_A and GPIO_B */
            status = WriteArrayVerifyMAX96717F(handle, &WriteGPIORegList);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_2HEX(
                    "MAX96717FVerifyGPIOOpenDetection: Failed to write GPIO_A "
                    "and GPIO_B at address ",
                    (uint32_t)addrGPIOA, (uint32_t)addrGPIOB);
            }
        }

        if (status != NVMEDIA_STATUS_OK) {
            break;
        }
    }

    return status;
}
#endif

/* Set configuration register CRC (Basic CRC only) as per the GMSL2/3 Register CRC App Note. */
NvMediaStatus
MAX96717FSetRegCRC(DevBlkCDIDevice const * const handle)
{
    /* Setting REG_CRC_ERR_OEN bit */
    NvMediaStatus status = SetRegBitMAX96717F(handle, REG_FS_INTR0, 0U, true);
    if (NVMEDIA_STATUS_OK == status) {
        /* Setting I2C_WR_COMPUTE bit */
        status = SetRegBitMAX96717F(handle, REG_REGCRC0, 3U, true);
        if (NVMEDIA_STATUS_OK == status) {
            /* Setting CRC_PERIOD[7] bit (corresponds to a CRC period of ~250ms) */
            status = SetRegBitMAX96717F(handle, REG_REGCRC1, 7U, true);
            if (NVMEDIA_STATUS_OK == status) {
                /* Setting PERIODIC_COMPUTE bit */
                status = SetRegBitMAX96717F(handle, REG_REGCRC0, 2U, true);
                if (NVMEDIA_STATUS_OK == status) {
                    /* Setting CHECK_CRC bit */
                    status = SetRegBitMAX96717F(handle, REG_REGCRC0, 1U, true);
                }
            }
        }
    }

    return status;
}

NvMediaStatus
MAX96717FDisableRegCRC(
    DevBlkCDIDevice const * const handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96717F const * const drvHandle = getHandlePrivMAX96717F(handle);

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96717FDisableRegCRC: Bad parameter\n");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        status = SetRegBitMAX96717F(drvHandle->i2cProgrammer,
                                    REG_FS_INTR0,
                                    0U,
                                    false);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR("Failed to disable REG_CRC_OEN\n");
        }
    }

    return status;
}

NvMediaStatus
MAX96717FReEnableRegCRC(
    DevBlkCDIDevice const * const handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX96717F const * const drvHandle = getHandlePrivMAX96717F(handle);

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX96717FResetRegCRC: Bad parameter\n");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        uint8_t const crc_reset_val = 1U;
        status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                            REG_REGCRC0,
                                            crc_reset_val);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("MAX96717FResetRegCRC: Write failed"
                                      "for register: ", toUint32FromUint16(REG_REGCRC0));
        } else {
            status = MAX96717FSetRegCRC(handle);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR("MAX96717FResetRegCRC: MAX96717FSetRegCRC failed\n");
            }
        }
    }

    return status;
}

NvMediaStatus
MAX96717FPollBit(
    DevBlkCDIDevice const *handle,
    uint16_t addr,
    uint8_t bit)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    for (uint8_t i = 0U; i < 10U; i++) {
        uint8_t intStatus = 0U;
        status = ReadUint8VerifyMAX96717F(handle, addr, &intStatus);
        if (NVMEDIA_STATUS_OK == status) {
            if (GetBit(intStatus,bit) > 0U) {
                break;
            } else {
                status = NVMEDIA_STATUS_ERROR;
                /* coverity[misra_c_2012_directive_4_9_violation] : intentional TID-1427 */
                (void)nvsleep(INT_DONE_IF_SLEEP_US);
            }
        } else {
            break;
        }
    }
    return status;
}

NvMediaStatus
MAX96717FArrayWrite(
    DevBlkCDIDevice const * const handle,
    DevBlkCDII2CReg const * const regs,
    uint32_t const length)
{
    DriverHandleMAX96717F const * const drvHandle = getHandlePrivMAX96717F(handle);

    DevBlkCDII2CRegList regMap = {
        .regs = regs,
        .numRegs = length,
    };

    NvMediaStatus const status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &regMap);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_HEX_UINT(
            "ArrayWriteMAX96717F: DevBlkCDII2CPgmrWriteArray failed", (uint32_t)status);
    }
    return status;
}

NvMediaStatus
MAX96717FArrayWriteVerify(
    DevBlkCDIDevice const * const handle,
    DevBlkCDII2CReg const * const regs,
    uint32_t const length)
{
    DevBlkCDII2CRegList const regMap = {
        .regs = regs,
        .numRegs = length,
    };

    NvMediaStatus const status = WriteArrayVerifyMAX96717F(handle, &regMap);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_HEX_UINT(
            "ArrayWriteMAX96717F: WriteArrayVerifyMAX96717F failed", (uint32_t)status);
    }
    return status;
}

DevBlkCDIDeviceDriver *
GetMAX96717FDriver(
    void)
{
    static DevBlkCDIDeviceDriver deviceDriverMAX96717F = {
        .deviceName = "Maxim 96717F Serializer",
        .regLength = (int32_t)MAX96717F_NUM_ADDR_BYTES,
        .dataLength = (int32_t)MAX96717F_NUM_DATA_BYTES,
        .DriverCreate = DriverCreateMAX96717F,
        .DriverDestroy = DriverDestroyMAX96717F,
    };
    return &deviceDriverMAX96717F;
}
