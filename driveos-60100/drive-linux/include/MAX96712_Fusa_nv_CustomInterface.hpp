/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited
 */

#ifndef MAX96712_FUSA_NV_CUSTOMINTERFACE_HPP
#define MAX96712_FUSA_NV_CUSTOMINTERFACE_HPP

#include "INvSIPLDeviceInterfaceProvider.hpp"
#include "NvSIPLCommon.hpp"

namespace nvsipl {

/**
 * @brief It's unique to MAX96712 custom interface class and will be used to
 *        uniquely identify this interface. The client can use this ID to
 *        validate the correct interface before use.
 */
constexpr UUID MAX96712_FUSA_NV_CUSTOM_INTERFACE_ID(0xF9FE70B0U, 0x289FU, 0x4240U, 0xA055U,
                                              0xDFU, 0x66U, 0xC7U, 0xC8U, 0x64U, 0xB5U);

/**
 * Structure to hold deserializer's custom error information for CSI phase-locked
 * loop (PLL) lock.
 */
typedef struct  {
    /** CSI PLL lock status. For each bit, 0: PLL not locked, 1: PLL locked.*/
    uint8_t csi_pll_lock_status;
} DeserializerCSIPLLLockInfo;

/**
 * Structure to hold deserializer's custom error information for command buffer overflow
 * and line memory overflow.
 */
typedef struct {
    /**Command buffer overflow information. For each bit, 0: No overflow, 1: Overflow.*/
    uint8_t cmd_buffer_overflow_info;
    /**Line memory overflow information. For each bit, 0: No overflow, 1: Overflow.*/
    uint8_t lm_overflow_info;
} DeserializerOverflowErrInfo;

/**
 * Structure to hold deserializer's custom error information for Combined ARQ Retransmission
 * Event Flag (RT_CNT_FLAG) and Combined ARQ Maximum Retransmission Limit Error Flag
 * (MAX_RT_FLAG) in INTR11 register.
 */
typedef struct {
    /**
     * Holds the error information for Combined ARQ Retransmission Event Flag.
     * For each bit, 0: None of the selected channels have done at least one ARQ retransmission.
     * 1: One or more of the selected channels has done at least one ARQ retransmission.
     */
    uint8_t rt_cnt_flag_info;
    /**
     * Holds the error information for Combined ARQ Maximum Retransmission Limit Error Flag.
     * For each bit, 0: None of the selected channels have reached the maximum retry limit.
     * 1: One or more of the selected channels has reached the maximum retry limit.
     */
    uint8_t max_rt_flag_info;
} DeserializerRTFlagsInfo;

/**
 * Structure to hold deserializer's custom error information for video RX sequence error detected
 * (VID_SEQ_ERR) in VIDEO_RX8 registers.
 */
typedef struct {
    /**
     * Holds the error information for Video Rx sequence error detected.
     * For each bit, 0: No video Rx sequence error detected.
     * 1: Video Rx sequence error detected.
     */
    uint8_t vid_seq_err_info;
} DeserializerVidSeqErrInfo;

/**
 * @brief The class of custom interfaces in deserializer MAX96712.
 */
class MAX96712FusaNvCustomInterface : public Interface
{
public:
    /**
     * @brief Function to get interface ID that's unique to this class.
     *
     * This function is used by application to get deserializer's
     * custom interface ID. Application can use this ID to get
     * the interface of the class.
     *
     * @pre None.
     *
     * @retval MAX96712_FUSA_NV_CUSTOM_INTERFACE_ID Custom interface id for MAX96712.
     *                                              Valid range :
     *                                              [ @ref MAX96712_FUSA_NV_CUSTOM_INTERFACE_ID].
     */
    static const UUID& getClassInterfaceID() {
        return MAX96712_FUSA_NV_CUSTOM_INTERFACE_ID;
    }

    /**
     * @brief Function to get interface ID that's unique to this instance.
     *
     * This function is used by application to get deserializer's custom interface ID
     * and ensure typecasted pointer indeed points to the right object.
     * Application can use this ID to get the interface of the instance.
     *
     * @pre None.
     *
     * @retval MAX96712_FUSA_NV_CUSTOM_INTERFACE_ID Custom interface id for MAX96712.
     *                                              Valid range :
     *                                              [ @ref MAX96712_FUSA_NV_CUSTOM_INTERFACE_ID].
     */
    const UUID& getInstanceInterfaceID() override {
        return MAX96712_FUSA_NV_CUSTOM_INTERFACE_ID;
    }

    /**
     * @brief Get deserializer overflow error information. Checks for line
     *        memory overflow and cmd buffer overflow.
     *
     * @pre A valid deserializer object created with @ref CNvMDeserializer_Create().
     *
     * @param[out] customErrInfo    Pointer to structure @ref DeserializerOverflowErrInfo
     *                              to hold the error information.
     *                              Valid range: [non-NULL].
     *
     * @retval NVSIPL_STATUS_OK             Deserializer overflow error information was got
     *                                      successfully.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT   Input parameter validation failed.
     * @retval (SIPLStatus)                 Any errors from dependencies.
     */
    virtual SIPLStatus GetOverflowError(
        DeserializerOverflowErrInfo *const customErrInfo
    ) const = 0;

    /**
     * @brief Get deserializer CSI PLL Lock status.
     *
     * @pre A valid deserializer object created with @ref CNvMDeserializer_Create().
     *
     * @param[out] customErrInfo    Pointer to structure @ref DeserializerCSIPLLLockInfo
     *                              to hold the error information.
     *                              Valid range: [non-NULL].
     *
     * @retval NVSIPL_STATUS_OK             Deserializer CSI PLL Lock status was got successfully.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT   Input parameter validation failed.
     * @retval (SIPLStatus)                 Any errors from dependencies.
     */
    virtual SIPLStatus GetCSIPLLLockStatus(
        DeserializerCSIPLLLockInfo *const customErrInfo
    ) const = 0;

    /**
     * @brief Get deserializer RT_CNT_FLAG and MAX_RT_FLAG status.
     *
     * @pre A valid deserializer object created with @ref CNvMDeserializer_Create().
     *
     * @param[out] customErrInfo    Pointer to structure @ref DeserializerRTFlagsInfo
     *                              to hold the error information.
     *                              Valid range: [non-NULL].
     *
     * @retval NVSIPL_STATUS_OK             Deserializer RT_CNT_FLAG and MAX_RT_FLAG status was got
     *                                      successfully.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT   Input parameter validation failed.
     * @retval (SIPLStatus)                 Any errors from dependencies.
     */
    virtual SIPLStatus GetRTFlagsStatus(
        DeserializerRTFlagsInfo *const customErrInfo
    ) const = 0;

    /**
     * @brief Get deserializer VID_SEQ_ERR bits status.
     *
     * @pre A valid deserializer object created with @ref CNvMDeserializer_Create().
     *
     * @param[out] customErrInfo    Pointer to structure @ref DeserializerVidSeqErrInfo
     *                              to hold the error information.
     *                              Valid range: [non-NULL].
     *
     * @retval NVSIPL_STATUS_OK             Deserializer VID_SEQ_ERR bits status was got
     *                                      successfully.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT   Input parameter validation failed.
     * @retval (SIPLStatus)                 Any errors from dependencies.
     */
    virtual SIPLStatus GetVidSeqErrStatus(
        DeserializerVidSeqErrInfo *const customErrInfo
    ) const = 0;
#if !NV_IS_SAFETY
    /**
     * @brief Set Heterogenous frame synchronization.
     *
     * This function is used to choose the multiplexer's input source and the GPIO number
     * of the deserializer to select the input frame sync source per link when the external
     * synchronization is used.
     *
     * @pre This function is applicable only if the platform has the HW support.
     *
     * @param[input] muxSel     The multiplexer's input selection.
     *                          Valid range: [0, 3].
     * @param[input] gpioNum    Array to store the gpio number per link.
     *                          Valid range: [0, 16].
     *
     * @retval NVSIPL_STATUS_OK             Heterogenous frame synchronization was set successfully.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT   Input parameter validation failed.
     * @retval (SIPLStatus)                 Any errors from dependencies.
     */
    virtual SIPLStatus SetHeteroFrameSync(
        uint8_t const muxSel,
        uint32_t const gpioNum[MAX_CAMERAMODULES_PER_BLOCK]
    ) const = 0;

    /**
     * @brief Get the camera group ID.
     *
     * This function is used to get the camera group ID.
     *
     * @pre A valid deserializer object created with @ref CNvMDeserializer_Create().
     *
     * @param[input] grpId      Group Index.
     *                          Valid range: [0, 3].
     *
     * @retval NVSIPL_STATUS_OK             Camera group ID was got successfully.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT   Input parameter validation failed.
     */
    virtual SIPLStatus GetGroupID(
        uint32_t &grpId
    ) const = 0;
#endif
protected:

    /** @brief Default Constructor */
    MAX96712FusaNvCustomInterface() = default;

    /** @brief Prevent the Copy constructor */
    MAX96712FusaNvCustomInterface(MAX96712FusaNvCustomInterface const &) = delete;

    /** @brief Prevent the Move constructor */
    MAX96712FusaNvCustomInterface(MAX96712FusaNvCustomInterface &&) = delete;

    /** @brief Prevent default copy assignment operator */
    MAX96712FusaNvCustomInterface& operator = (MAX96712FusaNvCustomInterface const &)& = delete;

    /** @brief Prevent default move assignment operator */
    MAX96712FusaNvCustomInterface& operator = (MAX96712FusaNvCustomInterface &&)& = delete;

    /** @brief Default destructor */
    ~MAX96712FusaNvCustomInterface() override = default;
};

} // namespace nvsipl
#endif // MAX96712_FUSA_NV_CUSTOMINTERFACE_HPP
