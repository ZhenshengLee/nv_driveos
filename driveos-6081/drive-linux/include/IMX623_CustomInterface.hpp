/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * All information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
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
const UUID IMX623_CUSTOM_INTERFACE_ID(0x90FB9A0DU, 0xA3AAU, 0x4C1BU, 0xA3B0U,
                                    0xA1U, 0xE0U, 0xE9U, 0xBFU, 0x4FU, 0x7DU);

class IMX623_CustomInterface : public Interface {
public:
    static const UUID& getClassInterfaceID() {
        return IMX623_CUSTOM_INTERFACE_ID;
    }
    virtual const UUID& getInstanceInterfaceID() override {
        return IMX623_CUSTOM_INTERFACE_ID;
    }

    /**
     * @brief Parse the Custom Embedded Data from the passed raw data.
     *
     * @param[in] embeddedBufTop Buffer pointer to top chunk of embedded data.
     *            Valid range : [non-NULL]
     * @param[in] embeddedBufTopSize Size of top chunk of embedded data.
     * @param[in] embeddedBufBottom Buffer pointer to bottom chunk of embedded
     *            data. Valid range : [non-NULL]
     * @param[in] embeddedBufBottomSize Size of bottom chunk of embedded data.
     * @param[out] customEmbData Custom Embedded data from IMX623.
                   Valid range : [non-NULL]
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT Argument validation failed.
     * @retval NVSIPL_STATUS_ERROR Any system error occurs.
     */
    virtual SIPLStatus ParseCustomEmbeddedData(
        uint8_t const * const embeddedBufTop,
        uint32_t const embeddedBufTopSize,
        uint8_t const * const embeddedBufBottom,
        uint32_t const embeddedBufBottomSize,
        IMX623CustomEmbeddedData * const customEmbData) const = 0;

    /**
     * @brief Get Serializer OVERFLOW bit, TUN_FIFO_OVERFLOW bit
     *        and PCLKDET bit status related to the health of the
     *        video transmitted from the Serializer.
     *
     * @param[in] customErrInfo Structure to hold the error information.
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT In case of bad parameter.
     * @retval NVSIPL_STATUS_ERROR Any system error
     */
    virtual SIPLStatus GetSerializerVideoStatus(
        MAX96717FVideoStatus *const customErrInfo) const = 0;

    /**
     * @brief Read ISET value from the Power Switch (PS) Status register.
     *          ISET value of 0 means no error.
     *          ISET value of 1 is reported on error when ISET Pin is open or
     *          shorted. In this case, default current limit of 600 mA is
     *          applied which may be more than the desired limit.
     *
     * @param[out] dataBuff output (ISET) value from Power Switch (PS) status
     *                      register
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT Argument validation failed.
     * @retval NVSIPL_STATUS_ERROR Any system error occurs.
    */
    virtual SIPLStatus PSCheckIsetComparator(uint8_t* const dataBuff) const = 0;

    /**
     * @brief Read Input/Output voltages and Current values being delivered
     *  by Power Switch (PS) on the output channel
     *
     * @param[in]  dataBuff_size Value is fixed to 5. Calling
     *             function needs to allocate space for the 5 values
     *             specified below in dataBuff.
     *             Valid range : [5]
     * @param[out] dataBuff pointer to the array Voltage/Current values
     *             for specified linkIndex.
     *             Index 0: Output Voltage
     *             Index 1: Vin
     *             Index 2: Vdd
     *             Index 3: VISET
     *             Index 4: Output Channel Current
     *             Valid range : [non-NULL]
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT Argument validation failed.
     * @retval NVSIPL_STATUS_ERROR Any system error occurs.
     */
    virtual SIPLStatus PSReadVoltageAndCurrentValues(
        uint8_t const dataBuff_size, uint16_t* const dataBuff) const = 0;

    /**
     * @brief Verify Serializer GPIO Readback Status
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT Argument validation failed.
     * @retval NVSIPL_STATUS_ERROR Any system error occurs.
     */
    virtual SIPLStatus VerifySerGPIOReadBackStatus(void) const = 0;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !(NV_IS_SAFETY)
    /**
     * @brief Write to the EEPROM module
     *
     * @param[in] address Location to write to EEPROM.
     *            Valid range : [256, 511]
     * @param[in] length Length of buffer to write into EEPROM.
     *            Valid range : [1, 256]
     * @param[in] buffer Pointer to the array of values to write to EEPROM
     *            Valid range : [non-NULL]
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT Argument validation failed.
     * @retval NVSIPL_STATUS_NOT_SUPPORTED EEPROM not available.
     * @retval NVSIPL_STATUS_ERROR Any system error occurs.
     */
    virtual SIPLStatus WriteEEPROMData(
        uint16_t const address, uint32_t const length, uint8_t * const buffer) = 0;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif // !(NV_IS_SAFETY)

    /**
     * @brief Execute TPS650332 PMIC ABIST Runtime
     *
     * @param[out] hasError Set true if any error is detected by the ABIST
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_TIMED_OUT If ABIST is not completed.
     * @retval NVSIPL_STATUS_ERROR Any system error occurs.
     */
    virtual SIPLStatus TPS650332ExecuteABIST(bool* const hasError) const = 0;

    /**
     * @brief Get error info for TPS650332 PMIC
     *
     * @param[out] customErrInfo Error Information filled by driver
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_ERROR Any system error occurs.
     */
    virtual SIPLStatus TPS650332GetErrorInfo(TPS650332CustomErrInfo *const customErrInfo) const = 0;

    /*
     * @brief Get fault status of nINT pin of TPS650332 PMIC
     *
     * @param[out] hasError Set true if any error is detected by the fault
     *                               monitoring mechanism for nINT pin.
     *                      Set false if nINT pin has no error.
     *
     * @retval NVSIPL_STATUS_OK Success.
     * @retval NVSIPL_STATUS_ERROR Any system error occurs.
     */
    virtual SIPLStatus TPS650332GetnINTErrorStatus(bool* const hasError) const = 0;

protected:
    virtual ~IMX623_CustomInterface(void) = default;
};
}

#endif
