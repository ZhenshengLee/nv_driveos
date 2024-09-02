/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef IMX728_CUSTOMINTERFACE_HPP
#define IMX728_CUSTOMINTERFACE_HPP

#include "INvSIPLDeviceInterfaceProvider.hpp"
#include "IMX728_CustomData.h"
#include "MAX96717F_CustomData.h"

namespace nvsipl {

/**
 * @brief This is version 4 UUID obtained using https://www.uuidgenerator.net/
 *        It's unique to IMX728 custom interface class and will be used to
 *        uniquely identify this interface. The client can use this ID to
 *        validate the correct interface before use.
 */
constexpr UUID IMX728_CUSTOM_INTERFACE_ID(0xBCE64356U, 0x1505U, 0x11EDU, 0x861DU,
                                    0x02U, 0x42U, 0xACU, 0x12U, 0x00U, 0x02U);

/**
 * @brief The class of custom interfaces in Sensor IMX728.
 */
class IMX728_CustomInterface : public Interface {
public:

   /**
    * @brief Get Sensor IMX728 class custom interface ID.
    *
    * This function is used by application to get sensor's
    * custom interface ID. Application can use this ID to get
    * the interface of the class.
    *
    * @pre None.
    *
    * @retval IMX728_CUSTOM_INTERFACE_ID    Custom interface id for IMX728.
    *                                       Valid range :
    *                                       [ @ref IMX728_CUSTOM_INTERFACE_ID].
    */
    static const UUID& getClassInterfaceID() {
        return IMX728_CUSTOM_INTERFACE_ID;
    }
    /**
    * @brief Get Sensor IMX728 instance custom interface ID.
    *
    * This function is used by application to get sensor's
    * custom interface ID. Application can use this ID to get
    * the interface of the instance.
    *
    * @pre None.
    *
    * @retval IMX728_CUSTOM_INTERFACE_ID    Custom interface id for IMX728.
    *                                       Valid range :
    *                                       [ @ref IMX728_CUSTOM_INTERFACE_ID].
    */
    const UUID& getInstanceInterfaceID() override {
        return IMX728_CUSTOM_INTERFACE_ID;
    }

    /**
    * @brief Set heater on or off.
    *
    * This function is used by application to turn heater on or off.
    * This API is virtual function and will be overridden by other implementation.
    *
    * @pre Serializer and PMIC must be initialized.
    *
    * @param[in] setState   Bool value representing on or off for the heater.
    *                       True represents setting heater on.
    *                       Valid range : [true, false].
    *
    * @retval NVSIPL_STATUS_OK              If successfully able to set heater state.
    * @retval NVSIPL_STATUS_NOT_SUPPORTED   Not a heater supported module.
    * @retval (SIPLStatus)                  Any errors from dependencies.
    *
    */
    virtual SIPLStatus SetHeaterState(bool const setState) const = 0;

    /**
    * @brief Parse the custom embedded data from the passed raw data.
    *
    * This API is virtual function and will be overridden by other implementation.
    *
    * @pre None.
    *
    * @param[in] embeddedBufTop         Buffer pointer to top chunk of embedded data.
    *                                   Valid range : [non-NULL].
    * @param[in] embeddedBufTopSize     Size of top chunk of embedded data.
    *                                   Valid range : [1, UINT32_MAX].
    * @param[out] customEmbData         Pointer to struct @ref IMX728CustomEmbeddedData.
    *                                   If embedded data is present, data from the parse
    *                                   written to @ref IMX728CustomEmbeddedData.
    *                                   Valid range : [non-NULL].
    *
    * @retval NVSIPL_STATUS_OK              If successfully able to parse the custom.
    *                                       embedded  data from raw data.
    * @retval NVSIPL_STATUS_BAD_ARGUMENT    Input parameter validation failed.
    * @retval (SIPLStatus)                  Any errors from dependencies.
    */
    virtual SIPLStatus ParseCustomEmbeddedData(
        uint8_t const * const embeddedBufTop,
        uint32_t const embeddedBufTopSize,
        IMX728CustomEmbeddedData * const customEmbData) const = 0;

    /**
    * @brief Get serializer's video status.
    *
    * This function is used to get serializer's video status by reading OVERFLOW bit,
    * TUN_FIFO_OVERFLOW bit and PCLK detected (PCLKDET) bit status which related to the
    * health of the video transmitted from the serializer.
    * This API is virtual function and will be overridden by other implementation.
    *
    * @pre The serializer should be in streaming state for video input.
    *
    * @param[out] customErrInfo     Pointer to struct @ref MAX96717FVideoStatus which
    *                               contains status of serializer OVERFLOW bit,
    *                               TUN_FIFO_OVERFLOW bit and PCLKDET bit.
    *                               Valid range : [non-NULL].
    *
    * @retval NVSIPL_STATUS_OK              If the video transmitted from the
    *                                       serializer is healthy.
    * @retval NVSIPL_STATUS_BAD_ARGUMENT    Input parameter validation failed.
    * @retval (SIPLStatus)                  Any errors from dependencies.
    */
    virtual SIPLStatus GetSerializerVideoStatus(
        MAX96717FVideoStatus *const customErrInfo) const = 0;

    /**
    * @brief Read ISET(Current-Limit Setting) value from Power Switch (PS)
    * Status register.
    *
    * This function reads the ISET value for the link specified by linkIndex
    * and stores it in the data pointer. ISET value of 0 means no error.
    * When ISET pin is open or shorted, ISET value of 1 is reported.
    * This API is virtual function and will be overridden by other implementation.
    *
    * @pre Power Switch must be powered on using GPIO control.
    *
    * @param[out] dataBuff      Buffer pointer to output (ISET) value from
    *                           Power Switch (PS) status register.
    *                           Valid range : [non-NULL].
    *
    * @retval NVSIPL_STATUS_OK              If successfully able to read ISET value.
    * @retval NVSIPL_STATUS_BAD_ARGUMENT    Input parameter validation failed.
    * @retval (SIPLStatus)                  Any errors from dependencies.
   */
    virtual SIPLStatus PSCheckIsetComparator(uint8_t* const dataBuff) const = 0;

    /**
    * @brief Reads the voltage and current values for a Power Switch
    * link/channel.
    *
    * This function reads input/output voltages and output channel current
    * for a given Power Switch link/channel.
    * This API is virtual function and will be overridden by other implementation.
    *
    * @pre Power Switch must be powered on using GPIO control.
    *
    * @param[in]  dataBuff_size     Value is fixed to 5. Calling function
    *                               needs to allocate space for the 5
    *                               values specified below in dataBuff.
    *                               Valid range : [5]
    * @param[out] dataBuff          pointer to the array Voltage/Current values
    *                               for specified linkIndex.
    *                               Index 0: Output Voltage.
    *                               Index 1: Vin.
    *                               Index 2: Vdd.
    *                               Index 3: VISET.
    *                               Index 4: Output Channel Current.
    *                               Valid range : [non-NULL].
    *
    * @retval NVSIPL_STATUS_OK              If successfully able to read Input/Output
    *                                       voltages and Current values.
    * @retval NVSIPL_STATUS_BAD_ARGUMENT    Input parameter validation failed.
    * @retval (SIPLStatus)                  Any errors from dependencies.
    */
    virtual SIPLStatus PSReadVoltageAndCurrentValues(
        uint8_t const dataBuff_size, uint16_t* const dataBuff) const = 0;

    /**
    * @brief Verify Serializer GPIO Readback Status.
    *
    * This function is used to perform run-time GPIO readback verification test
    * on applicable GPIO pins.
    * This API is virtual function and will be overridden by other implementation.
    *
    * @pre A valid serializer object must be created and powered on.
    *
    * @retval NVSIPL_STATUS_OK              If Serializer GPIO Readback Status works normally.
    * @retval NVSIPL_STATUS_BAD_ARGUMENT    Input parameter validation failed.
    * @retval (SIPLStatus)                  Any errors from dependencies.
    */
    virtual SIPLStatus VerifySerGPIOReadBackStatus(void) const = 0;

    /**
    * @brief Execute BD868B2 PMIC FBIST Runtime
    *
    * This API is virtual function and will be overridden by other implementation.
    *
    * @pre Serializer and BD868B2 must be powered on.
    *
    * @retval NVSIPL_STATUS_OK              If successfully able to execute BD868B2 PMIC FBIST.
    * @retval (SIPLStatus)                  Any errors from dependencies.
    */
    virtual SIPLStatus BD868B2ExecuteFBIST(void) const = 0;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !(NV_IS_SAFETY)
    /**
    * @brief Write to the EEPROM module.
    *
    * @param[in] address    Location to write to EEPROM.
    *                       Valid range : [256, 511].
    * @param[in] length     Length of buffer to write into EEPROM.
    *                       Valid range : [1, 256].
    * @param[in] buffer     Pointer to the array of values to write to EEPROM.
    *                       Valid range : [non-NULL].
    *
    * @retval NVSIPL_STATUS_OK              If successfully able to write data to EEPROM.
    * @retval NVSIPL_STATUS_BAD_ARGUMENT    Input parameter validation failed.
    * @retval NVSIPL_STATUS_NOT_SUPPORTED   EEPROM handle invalid.
    * @retval (SIPLStatus)                  Any errors from dependencies.
    */
    virtual SIPLStatus WriteEEPROMData(
        uint16_t const address, uint32_t const length, uint8_t * const buffer) = 0;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif // !(NV_IS_SAFETY)

protected:

    /** @brief Default Constructor of class IMX728_CustomInterface */
    IMX728_CustomInterface() = default;

    /** @brief Prevent the Copy constructor of class IMX728_CustomInterface */
    IMX728_CustomInterface(IMX728_CustomInterface const &) = delete;

    /** @brief Prevent the Move constructor of class IMX728_CustomInterface */
    IMX728_CustomInterface(IMX728_CustomInterface &&) = delete;

    /** @brief Prevent default copy assignment operator of class IMX728_CustomInterface */
    IMX728_CustomInterface& operator = (IMX728_CustomInterface const &)& = delete;

    /** @brief Prevent default move assignment operator of class IMX728_CustomInterface */
    IMX728_CustomInterface& operator = (IMX728_CustomInterface &&)& = delete;

    /**
    * @brief Default destructor of class IMX728_CustomInterface.
    */
    ~IMX728_CustomInterface() override = default;
};
}

#endif //IMX728_CUSTOMINTERFACE_HPP
