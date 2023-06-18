/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef MAX96712_FUSA_NV_CUSTOMINTERFACE_HPP
#define MAX96712_FUSA_NV_CUSTOMINTERFACE_HPP

#include "INvSIPLDeviceInterfaceProvider.hpp"
#include "NvSIPLCommon.hpp"

namespace nvsipl {

const UUID MAX96712_FUSA_NV_CUSTOM_INTERFACE_ID(0xF9FE70B0U, 0x289FU, 0x4240U, 0xA055U,
                                              0xDFU, 0x66U, 0xC7U, 0xC8U, 0x64U, 0xB5U);

/**
 * @brief Describes the custom error information for CSI PLL lock.
 *        Bits 3-0 represent CSI PLL lock status 3-0. If bit is set, PLL is
 *        locked.
 */
typedef struct  {
    uint8_t csi_pll_lock_status;
} DeserializerCSIPLLLockInfo;

/**
 * @brief Describes the custom error information for line memory overflow and
 *        command buffer overflow for deserializer. Bits 7-0 represent pipe 7-0
 *        overflow. If bit is set, overflow has occurred.
 */
typedef struct {
    uint8_t cmd_buffer_overflow_info;
    uint8_t lm_overflow_info;
} DeserializerOverflowErrInfo;

/**
 * @brief Describes the custom error information for RT_CNT_FLAG and MAX_RT_FLAG
 *        bits in INTR11 register. Bits 0 to 3 represent RT_CNT_FLAG_A/B/C/D and
 *        MAX_RT_FLAG_A/B/C/D for both variables. If bit is set, corresponding
 *        flag in register is set.
 */
typedef struct {
    uint8_t rt_cnt_flag_info;
    uint8_t max_rt_flag_info;
} DeserializerRTFlagsInfo;

class MAX96712FusaNvCustomInterface : public Interface
{
public:
    /**
     * @brief Function to get Interface ID that's unique to this class
     */
    static const UUID& getClassInterfaceID() {
        return MAX96712_FUSA_NV_CUSTOM_INTERFACE_ID;
    }

    /**
     * @brief Used for a confirmatory test by the app to ensure typecasted
     * pointer indeed points to the right object
     */
    const UUID& getInstanceInterfaceID() {
        return MAX96712_FUSA_NV_CUSTOM_INTERFACE_ID;
    }

    /**
     * @brief Get Deserializer overflow error information. Checks for line
     *        memory overflow and cmd buffer overflow.
     *
     * @param[in] customErrInfo Structure to hold the error information.
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT In case of bad parameter.
     * @retval NVSIPL_STATUS_ERROR Any system error.
     */
    virtual SIPLStatus GetOverflowError(
        DeserializerOverflowErrInfo *const customErrInfo
    ) const = 0;

    /**
     * @brief Get Deserializer CSI PLL Lock status.
     *
     * @param[in] customErrInfo Structure to hold the error information.
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT In case of bad parameter.
     * @retval NVSIPL_STATUS_ERROR Any system error.
     */
    virtual SIPLStatus GetCSIPLLLockStatus(
        DeserializerCSIPLLLockInfo *const customErrInfo
    ) const = 0;

    /**
     * @brief Get Deserializer RT_CNT_FLAG and MAX_RT_FLAG status.
     *
     * @param[in] customErrInfo Structure to hold the error information.
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT In case of bad parameter.
     * @retval NVSIPL_STATUS_ERROR Any system error.
     */
    virtual SIPLStatus GetRTFlagsStatus(
        DeserializerRTFlagsInfo *const customErrInfo
    ) const = 0;

    /**
     * @brief Set Heterogenous frame synchronization
     *
     * Select the multiplexer's input source and the GPIO number of the deserializer
     * to select the input frame sync source per link when the external synchronization
     * is used
     * This function is applicable only if the platform has the HW supports
     *
     * @param[input] muxSel  the multiplexer's input selection
     * @param[input] gpioNum Array to store the gpio number per link
     *
     * @retval NVSIPL_STATUS_OK Success
     * @retval NVSIPL_STATUS_BAD_ARGUMENT gpioNum is NULL
     * @retval NVSIPL_STATUS_ERROR any system error.
     */
    virtual SIPLStatus SetHeteroFrameSync(
        uint8_t const muxSel,
        uint32_t const gpioNum[MAX_CAMERAMODULES_PER_BLOCK]
    ) const = 0;

    /**
     * @brief Get the camera group ID
     *
     * Return the camera group ID
     *
     * @param[input] grpId a Group Index
     *
     * @retval NVSIPL_STATUS_OK Success
     * @retval NVSIPL_STATUS_BAD_ARGUMENT grpId is NULL
     */
    virtual SIPLStatus GetGroupID(
        uint32_t &grpId
    ) const = 0;

#if !NV_IS_SAFETY
    /**
     * @brief Get Deserializer register data
     *
     * @param[in] registerNum register address.
     * @param[in] dataByte data byte to hold the register byte read.
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT In case of bad parameter.
     * @retval NVSIPL_STATUS_ERROR Any system error.
     */
    virtual SIPLStatus getDeserData(
        uint16_t registerNum,
        uint8_t *dataByte
    ) const = 0;
#endif

protected:
    ~MAX96712FusaNvCustomInterface() = default;
};

} // namespace nvsipl
#endif // MAX96712_FUSA_NV_CUSTOMINTERFACE_HPP
