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

#ifndef IMX623_CUSTOMINTERFACE_HPP
#define IMX623_CUSTOMINTERFACE_HPP

#include "INvSIPLDeviceInterfaceProvider.hpp"
#include "IMX623_CustomData.h"
#include "MAX96717F_CustomData.h"
#include "TPS650332_CustomData.h"

namespace nvsipl {

/**
 * @brief This is version 4 UUID obtained using https://www.uuidgenerator.net/
 *        It's unique to IMX623 custom interface class and will be used to
 *        uniquely identify this interface. The client can use this ID to
 *        validate the correct interface before use.
 */
constexpr UUID IMX623_CUSTOM_INTERFACE_ID(0x90FB9A0DU, 0xA3AAU, 0x4C1BU, 0xA3B0U,
                                    0xA1U, 0xE0U, 0xE9U, 0xBFU, 0x4FU, 0x7DU);

/**
 * @brief The class of custom interfaces in Sensor IMX623.
 */
class IMX623_CustomInterface : public Interface {
public:
    /**
    * @brief Get Sensor IMX623 class custom interface ID.
    *
    * This function is used by application to get sensor's
    * custom interface ID. Application can use this ID to get
    * the interface of the class.
    *
    * @pre None.
    *
    * @retval IMX623_CUSTOM_INTERFACE_ID    Custom interface id for IMX623.
    *                                       Valid range :
    *                                       [ @ref IMX623_CUSTOM_INTERFACE_ID].
    */
    static const UUID& getClassInterfaceID() {
        return IMX623_CUSTOM_INTERFACE_ID;
    }
    /**
    * @brief Get Sensor IMX623 instance custom interface ID.
    *
    * This function is used by application to get sensor's
    * custom interface ID. Application can use this ID to get
    * the interface of the instance.
    *
    * @pre None.
    *
    * @retval IMX623_CUSTOM_INTERFACE_ID    Custom interface id for IMX623.
    *                                       Valid range :
    *                                       [ @ref IMX623_CUSTOM_INTERFACE_ID].
    */
    const UUID& getInstanceInterfaceID() override {
        return IMX623_CUSTOM_INTERFACE_ID;
    }

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
    * @param[out] customEmbData         Pointer to struct @ref IMX623CustomEmbeddedData.
    *                                   If embedded data is present, data from the parse
    *                                   written to @ref IMX623CustomEmbeddedData.
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
        IMX623CustomEmbeddedData * const customEmbData) const = 0;

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
    * @retval NVSIPL_STATUS_OK             If successfully able to read ISET value.
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

    /**
    * @brief Execute TPS650332 PMIC Analog Built In Self Test (ABIST) Runtime.
    *
    * This API is virtual function and will be overridden by other implementation.
    *
    * @pre Serializer and TPS650332 must be powered on.
    *
    * @param[out] hasError      Pointer to bool value. The bool value will be Set true
    *                           if any error is detected by the ABIST.
    *                           Valid range : [non-NULL].
    *
    * @retval NVSIPL_STATUS_OK          If successfully able to execute
    *                                   TPS650332 PMIC ABIST Runtime.
    * @retval NVSIPL_STATUS_TIMED_OUT   If ABIST is not completed.
    * @retval (SIPLStatus)              Any errors from dependencies.
    */
    virtual SIPLStatus TPS650332ExecuteABIST(bool* const hasError) const = 0;

    /**
    * @brief Get error info for TPS650332 PMIC.
    *
    * This function is used to get information about error bits in the relevant ACK registers,
    * which are set in case of interrupt faults.
    * This API is virtual function and will be overridden by other implementation.
    *
    * @pre Serializer and TPS650332 must be powered on.
    *
    * @param[out] customErrInfo     Pointer to struct @ref TPS650332CustomErrInfo which
    *                               contains error information filled by driver,
    *                               bit field indicating the error flags set is
    *                               indexed by enum @ref TPS650332CustomErrType.
    *                               Valid range : [non-NULL].
    *
    * @retval NVSIPL_STATUS_OK      If successfully able to get error info for TPS650332 PMIC.
    * @retval (SIPLStatus)          Any errors from dependencies.
    */
    virtual SIPLStatus TPS650332GetErrorInfo(TPS650332CustomErrInfo *const customErrInfo) const = 0;

    /**
    * @brief Get fault status of nINT pin of TPS650332 PMIC
    *
    * This function is used to get error status of nINT pin by reading register of TPS650332 via
    * I2C programmer.
    * This API is virtual function and will be overridden by other implementation..
    *
    * @pre Serializer and TPS650332 must be powered on.
    *
    * @param[out] hasError      Point to bool value. The bool value will be set true if any error
    *                           is detected by the fault monitoring mechanism for nINT pin.
    *                           The bool value will be set false if nINT pin has no error.
    *                           Valid range : [non-NULL].
    *
    * @retval NVSIPL_STATUS_OK      If successfully able to get fault status of nINT pin of
    *                               TPS650332 PMIC.
    * @retval (SIPLStatus)          Any errors from dependencies.
    */
    virtual SIPLStatus TPS650332GetnINTErrorStatus(bool* const hasError) const = 0;

#if !NV_IS_SAFETY
    /**
     * @brief Perform I2C CRC integrity check
     *
     * This function is used to verify the I2C communication by comparing calculated CRC with the
     * expected CRC from sensor.
     *
     * @pre A valid camera module object must be created and powered on.
     *
     * @param[in] handle                    Pointer to structure @ref DevBlkCDIDevice.
     *                                      Valid range: [non-NULL].
     *
     * @retval NVSIPL_STATUS_OK             If successfully able to check IMX623's I2C CRC
     *                                      integrity.
     * @retval NVSIPL_STATUS_ERROR          I2C CRC integrity check failed or any
     * @retval                              errors from dependencies.
     */
    virtual SIPLStatus IMX623I2CCRCIntegrityCheck(void) const = 0;

    /**
     * @brief Perform I2C CMAC integrity check
     *
     * This function is used to verify the I2C communication by comparing calculated CMAC with the
     * expected CMAC from sensor.
     *
     * @pre A valid camera module object must be created and powered on.
     *
     * @param[in] handle                    Pointer to structure @ref DevBlkCDIDevice.
     *                                      Valid range: [non-NULL].
     *
     * @retval NVSIPL_STATUS_OK             If successfully able to check IMX623's I2C CMAC
     *                                      integrity.
     * @retval NVSIPL_STATUS_ERROR          I2C CMAC integrity check failed or any
     * @retval                              errors from dependencies.
     */
    virtual SIPLStatus IMX623I2CCMACIntegrityCheck(void) const = 0;
#endif //!NV_IS_SAFETY

protected:

    /** @brief Default Constructor of class IMX623_CustomInterface */
    IMX623_CustomInterface() = default;

    /** @brief Prevent the Copy constructor of class IMX623_CustomInterface */
    IMX623_CustomInterface(IMX623_CustomInterface const &) = delete;

    /** @brief Prevent the Move constructor of class IMX623_CustomInterface */
    IMX623_CustomInterface(IMX623_CustomInterface &&) = delete;

    /** @brief Prevent default copy assignment operator of class IMX623_CustomInterface */
    IMX623_CustomInterface& operator = (IMX623_CustomInterface const &)& = delete;

    /** @brief Prevent default move assignment operator of class IMX623_CustomInterface */
    IMX623_CustomInterface& operator = (IMX623_CustomInterface &&)& = delete;

    /**
    * @brief Default destructor of class IMX623_CustomInterface.
    */
    ~IMX623_CustomInterface() override = default;
};
}

#endif
