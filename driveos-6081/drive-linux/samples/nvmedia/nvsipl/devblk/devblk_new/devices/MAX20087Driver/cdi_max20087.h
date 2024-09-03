/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CDI_MAX20087_H
#define CDI_MAX20087_H

#include "devblk_cdi.h"
#ifdef __cplusplus
extern "C" {
#endif

// Sleep times
#define SLEEP_TIME              (2000)
#define BUG_3807625_SLEEP_TIME  (450U)
#define SLEEP_PERIOD_MS         (10U)
#define SLEEP_10MS_AS_US        (10000)

#define VERIFY_TRUE             (true)
#define VERIFY_FALSE            (false)
#define DISABLE_UV_MASK         (true)
#define REGISTER_ADDRESS_LENGTH (1U)
#define REGISTER_DATA_LENGTH    (1U)
#define NUM_STAT2_REG           (2U)
#define NUM_LINKS_PER_STAT2     (2U)
#define NUM_ADC_REG             (4U)

#define ADDR_REG_MASK           (0U)
#define ADDR_REG_CONFIG         (1U)
#define ADDR_REG_ID             (2U)
#define ADDR_REG_STAT1          (3U)
#define ADDR_REG_STAT2_1        (4U)
#define ADDR_REG_STAT2_2        (5U)
#define ADDR_REG_ADC1           (6U)
#define ADDR_REG_ADC2           (7U)
#define ADDR_REG_ADC3           (8U)
#define ADDR_REG_ADC4           (9U)
#define REG_STAT1_MASK_INP_ERR  (0x0FU)
#define LSB_4_MASK              (0x0FU)
#define MSB_4_MASK              (0xF0U)
#define MASK_STAT1_ISET         (0x20U)

// Mask register bits
#define UV_BIT                  (0U)
#define ACCM_BIT                (6U)
#define OVTST_BIT               (7U)

// Config register bits
#define CLR_BIT                 (4U)
#define ENC_BIT                 (5U)
#define MUX_BIT0                (6U)
#define MUX_BIT1                (7U)
#define ALL_OUT_LINKS_ENABLED   (0x0FU)
#define REG_CONFIG_MASK_EN1     (0x01U)
#define REG_CONFIG_MASK_EN2     (0x02U)
#define REG_CONFIG_MASK_EN3     (0x04U)
#define REG_CONFIG_MASK_EN4     (0x08U)

// ID register bits
#define MAX20087_DEV_ID 0x20U
#define MAX20087_REV_ID 0x1U

// Voltage related
#define DATA_VALUES_ARR_LENGTH  (5U)
#define DATA_INDEX_VOUT         (0U)
#define DATA_INDEX_VIN          (1U)
#define DATA_INDEX_VDD          (2U)
#define DATA_INDEX_VISET        (3U)
#define DATA_INDEX_CURRENT      (4U)
#define REGISTER_DEVICE_INDEX   (1U)
#define CURRENT_MULTIPLIER      (3U)
#define VOLTAGE_MULTIPLIER      (70U)
#define VIN_MULTIPLIER          (70U)
#define VDD_MULTIPLIER          (25U)
#define VISET_MULTIPLIER        (5U)
#define ADC_LENGTH              (4U)
/**
 * nominal current in nano-ampere to avoid float
*/
#define NOMINAL_CURRENT         (12500U)    // expressed in nano-ampere
#define MARGIN_FACTOR           (100U)
#define VISET_MARGIN_FACTOR     (1000U)
#define NOMINAL_CURRENT_VARIANCE    (600U)    // expressed in nano-ampere
#define STAT1_OVDD_BIT          (1U)
#define STAT1_OVIN_BIT          (3U)
#define STAT2_UV_MASK           (0x11U)
#define STAT2_OV_MASK           (0x22U)

// Interrupt Mask related
#define INTERRUPT_MASKED_STATE      (1U)
#define INTERRUPT_RESTORED_STATE    (2U)

// Commonly used expressions
#define MIN_RANGE(expected, margin, factor) \
    ((expected) - (((expected) * (margin)) / (factor)))
#define MAX_RANGE(expected, margin, factor) \
    ((expected) + (((expected) * (margin)) / (factor)))
#define CHECK_RANGE(in, exp, margin, factor) \
    (((in) < MIN_RANGE((exp), (margin), (factor))) || \
     ((in) > MAX_RANGE((exp), (margin), (factor))))

DevBlkCDIDeviceDriver*
GetMAX20087Driver(
    void);

NvMediaStatus
MAX20087Init(
    DevBlkCDIRootDevice* const cdiRootDev,
    DevBlkCDIDevice const* handle,
    int32_t const csiPort);

NvMediaStatus
MAX20087CheckPresence(
    DevBlkCDIDevice const* handle);

/**
 * Use power switch to power on/off the camera modules.
 *
 * @param[in] handle                Handle to the power device
 * @param[in]  linkIndex            The data link index;
 *                                  Valid range: [0, (Maximum Links Supported per Deserializer - 1)]
 * @param[in]  enable               True to turn on power, False to turn off power
 * @param[in] stateInterruptMask    Interrupt mask state
 *                                  (INTERRUPT_MASK_INITED_STATE = 0) - Interrupts not masked
 *                                  (INTERRUPT_GLOBAL_MASKED_STATE = 1) - Interrupts masked
                                    (INTERRUPT_MASK_RESTORED_STATE = 2) - Interrupts restored
 * @param[out] savedInterruptMask   Stores updates of mask register (enable/disable bits) when
                                    interrupts are in masked state.
                                    Used when restoring mask register.
 *
 * @retval      NVMEDIA_STATUS_OK On Success
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER when incorrect parameters are passed.
 * @retval      NVMEDIA_STATUS_ERROR error status propagated
 */
NvMediaStatus
MAX20087SetLinkPower(
    DevBlkCDIDevice const* handle,
    uint8_t const linkIndex,
    bool const enable,
    uint8_t const stateInterruptMask,
    uint8_t *savedInterruptMask);

NvMediaStatus
MAX20087ReadRegister(
    DevBlkCDIDevice const* handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff);

NvMediaStatus
MAX20087WriteRegister(
    DevBlkCDIDevice const* handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t const* dataBuff);

/**
 * Mask or restore mask of power switch interrupts
 *
 * @param[in] enableGlobalMask true if all power switch interrupts are to
 *                                  be masked after saving the original
 *                                  interrupt mask.
 *                             false if the saved power switch interrupt
 *                                  mask is to be restored.
 * @param[in,out] savedInterruptMask if enableGlobleMask is true then
 *                                  savedInterruptMask is output parameter and
 *                                  hold the current interrupt mask.
 *                                  if enableGlobleMask is false then
 *                                  savedInterruptMask is input parameter.
 *
 * @retval       NVSIPL_STATUS_OK On Success
 * @retval      (SIPLStatus) error status propagated
 */
NvMediaStatus
MAX20087MaskRestoreGlobalInterrupt(
    DevBlkCDIDevice const* handle,
    uint8_t * savedInterruptMask,
    const bool enableGlobalMask);

#ifdef __cplusplus
}
#endif
#endif /* CDI_MAX20087_H */
