/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef AR0820_CUST1_CUSTOMINTERFACE_HPP
#define AR0820_CUST1_CUSTOMINTERFACE_HPP
#include <cstdint>
#include <cmath>
#include "INvSIPLDeviceInterfaceProvider.hpp"
#include "NvSIPLCommon.hpp"
#include "AR0820_CUST1_CustomData.h"

namespace nvsipl
{
// This is version 1 UUID obtained using https://www.uuidgenerator.net/
UUID const AR0820_CUST1_MAX9295_CUSTOM_INTERFACE_ID{0xB549DD36U, 0x3A02U, 0x11EBU, 0xADC1U,
                                                     0x02U, 0x42U, 0xACU, 0x12U, 0x00U, 0x02U};

/**
 * @brief The class of custom interfaces in Serializer max9295
 */
class AR0820_CUST1_MAX9295_CustomInterface : public Interface
{
public:
    /**
     * @brief Get Serializer max9295 class custom interface ID
     *
     * This function is used by application to get serializer
     * custom interface ID. Application can use this ID to get
     * the instance of the class
     *
     * @retval UUID AR0820_CUST1_MAX9295_CUSTOM_INTERFACE_ID
     */
    static const UUID& getClassInterfaceID(void) {return AR0820_CUST1_MAX9295_CUSTOM_INTERFACE_ID;}

    /**
     * @brief Get Serializer max9295 instance custom interface ID
     *
     * This function is used by application to get serializer
     * custom interface ID. Application can use this ID to get
     * the instance of the interface
     *
     * @retval UUID AR0820_CUST1_MAX9295_CUSTOM_INTERFACE_ID
     */
    /* coverity[misra_cpp_2008_rule_0_1_10_violation] : intentional TID-1966 */
    const UUID& getInstanceInterfaceID(void) override {
                            return AR0820_CUST1_MAX9295_CUSTOM_INTERFACE_ID;}

    /**
     * @brief Detect MAX9295A serializer video pipe FIFO overflow.
     *
     * This fuction is used by the application to detect MAX9295A serializer
     * video pipe FIFO overflow. The caller of this function needs to allocate
     * the memory.
     *
     * @param[out] status video pipe FIFO overflow status from MAX9295A.
     *
     * @retval NVSIPL_STATUS_OK             Success
     * @retval NVSIPL_STATUS_BAD_ARGUMENT   Argument validation failed
     * @retval NVSIPL_STATUS_ERROR          Get FIFO overflow info call failed.
     */
    virtual SIPLStatus DetectFifoOverflow(uint16_t &status) const = 0;

    /**
     * @brief Get MAX9295A Serializer Temperature data
     *
     * This function is to get MAX9295A Serializer temperature from
     * ADC register.
     *
     * @param[out] tmon temperature sensor reading in celsius.

     *
     * @retval NVSIPL_STATUS_OK Success
     * @retval NVSIPL_STATUS_BAD_ARGUMENT upSerializer is NULL
     * @retval NVSIPL_STATUS_ERROR any system error.
     */
    virtual SIPLStatus GetSerializerTemperature(float_t &tmon) const = 0;

    /**
     * @brief Detect MAX9295A Serializer PCLK
     *
     * This function is to get MAX9295A Serializer PLCK from
     * register.
     *
     * @param[out] status: video transmit PCLK status .
     *                     bit0: PCLKDET for VID_TX X
     *                     bit1: PCLKDET for VID_TX Y
     *                     bit2: PCLKDET for VID_TX Z
     *                     bit3: PCLKDET for VID_TX U
     *
     * @retval NVSIPL_STATUS_OK Success
     * @retval NVSIPL_STATUS_BAD_ARGUMENT upSerializer is NULL
     * @retval NVSIPL_STATUS_ERROR any system error.
     */
    virtual SIPLStatus DetectVideoPCLK(uint16_t &status) const = 0;

    /**
     * @brief Assert MAX9295A ERRB.
     *
     * This function is used by the application to assert serializer MAX9295A
     * ERRB before streaming to guarantee the functionality of ERRB
     *
     * @retval NVSIPL_STATUS_OK             Success
     * @retval NVSIPL_STATUS_ERROR          Failed to assert serializer ERRB
     */
     virtual SIPLStatus AssertMax9295Errb(void) const = 0;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY

    virtual SIPLStatus setDeserData(uint32_t const deviceIndex, uint32_t const registerNum, uint32_t const dataLength, uint8_t *const dataBuff) const = 0;
    virtual SIPLStatus getDeserData(uint32_t const deviceIndex, uint32_t const registerNum, uint32_t const dataLength, uint8_t *const dataBuff) const = 0;
    virtual SIPLStatus readRegisterMax9295 (uint16_t const registerAddr, uint16_t const dataLength, uint8_t *const dataBuff) const = 0;
    virtual SIPLStatus writeRegisterMax9295(uint16_t const registerAddr, uint16_t const dataLength, uint8_t *const dataBuff) const = 0;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
protected:
    virtual ~AR0820_CUST1_MAX9295_CustomInterface(void) = default;
};

UUID const AR0820_CUST1_CUSTOM_INTERFACE_ID{0x4FC721B6U, 0x3481U, 0x11EBU, 0xADC1U,
                                             0x02U, 0x42U, 0xACU, 0x12U, 0x00U, 0x02U};

/**
 * @brief The class of custom interfaces in Sensor AR0820
 */
class AR0820_CUST1_CustomInterface : public Interface
{
public:
    /**
     * @brief Get Sensor AR0820 class custom interface ID
     *
     * This function is used by application to get sensor
     * custom interface ID. Application can use this ID to get
     * the instance of the class
     *
     * @retval UUID AR0820_CUST1_CUSTOM_INTERFACE_ID
     */
    static const UUID& getClassInterfaceID(void) {return AR0820_CUST1_CUSTOM_INTERFACE_ID;}
    /**
     * @brief Get Sensor AR0820 instance custom interface ID
     *
     * This function is used by application to get sensor
     * custom interface ID. Application can use this ID to get
     * the instance of the interface
     *
     * @retval UUID AR0820_CUST1_CUSTOM_INTERFACE_ID
     */
    /* coverity[misra_cpp_2008_rule_0_1_10_violation] : intentional TID-1966 */
    const UUID& getInstanceInterfaceID(void) override {return AR0820_CUST1_CUSTOM_INTERFACE_ID;}

    /**
     * @brief Get sensor startup error status.
     *
     * This function is used by the application to retrieve the detailed
     * sensor startup error information. The caller of this function
     * needs to allocate the memory.
     *
     * @param[out] errorstatus The value of register REG_ASIL_STARTUP_STATUS_00
     *
     * @retval NVSIPL_STATUS_OK             Success
     * @retval NVSIPL_STATUS_BAD_ARGUMENT   Argument validation failed
     * @retval NVSIPL_STATUS_ERROR          I2C access failed
     */
    virtual SIPLStatus GetPowerUpSMErr(uint16_t * const errorstatus) const = 0;

    /**
     * @brief Get sensor defective pixel list
     *
     * This function is used by the application to get ar0820 defective pixel
     * data which is stored in OTPM wintin the senor. The caller of this function
     * needs to allocate the memory. Also this function can ONLY be called before
     * the streaming.
     *
     * @param[out] pdiraw Hold the defective pixel raw data
     *                    Application needs to parse the data to extract pixel list.
     * @param[out] pdicount the number of defective pixels
     *
     * @retval NVSIPL_STATUS_OK             Success
     * @retval NVSIPL_STATUS_BAD_ARGUMENT   Argument validation failed
     * @retval NVSIPL_STATUS_ERROR          I2C access failed
     */
    virtual SIPLStatus GetPDIData(AR0820PDIList * const pdiraw,
                                   uint16_t * const pdicount) const = 0;

    /**
     * @brief Parse the Custom Embedded Data from the passed raw embedded data.
     *
     * This function is used by the application to retrieve the Custom Embedded
     * Data from the passed raw embedded data. The caller of this function
     * needs to allocate the memory.
     *
     * @param[in] embeddedBufTop Buffer pointer to top chunk of embedded data.
     * @param[in] embeddedBufTopSize Size of top chunk of embedded data.
     * @param[in] embeddedBufBottom Buffer pointer to bottom chunk of embedded data.
     * @param[in] embeddedBufBottomSize Size of bottom chunk of embedded data.
     * @param[out] embeddedData Custom Embedded data from AR0820.
     *
     * @retval NVSIPL_STATUS_OK             Success
     * @retval NVSIPL_STATUS_BAD_ARGUMENT   Argument validation failed
     */
    virtual SIPLStatus ParseCustomEmbeddedData(
        uint8_t const * const embeddedBufTop,
        const uint32_t embeddedBufTopSize,
        uint8_t const * const embeddedBufBottom,
        const uint32_t embeddedBufBottomSize,
        AR0820CustomEmbeddedData * const embeddedData) const = 0;

    /**
     * @brief Get TPS650332 Startup ACK status.
     *
     * This fuction is used by the application to read PMIC TPS650332 startup
     * ACK status. The caller of this function needs to allocate
     * the memory.
     *
     * @param[out] ack Registers Index and Value.
     *
     * @retval NVSIPL_STATUS_OK             Success
     * @retval NVSIPL_STATUS_BAD_ARGUMENT   Argument validation failed
     * @retval NVSIPL_STATUS_ERROR          Device access failed.
     */
     virtual SIPLStatus GetPMICStartupACK(
         TPS650332CustomStartupAck * const ack) const = 0;

    /**
     * @brief Set Imager Orientation.
     *
     * This function is used by the application to change imager orientation
     *
     * @param[in] hMirror Horizontal Mirror.
     * @param[in] vFlip   Vertical flip.
     * @param[out] cfa    Color format after rotation.
     *
     * @retval NVSIPL_STATUS_OK             Success
     * @retval NVSIPL_STATUS_BAD_ARGUMENT   Argument validation failed
     * @retval NVSIPL_STATUS_NOT_SUPPORTED  Invalid in current driver state
     * @retval NVSIPL_STATUS_ERROR          Device access failed.
     */
     virtual SIPLStatus SetImageOrientation(
         const uint8_t hMirror,
         const uint8_t vFlip,
         uint32_t * const cfa) const = 0;

    /**
     * @brief Assert Imager SYS_CHECK.
     *
     * This function is used by the application to assert imager SYS_CHECK
     * before streaming to guarantee the functionality of SYS_CHECK
     *
     * @retval NVSIPL_STATUS_OK             Success
     * @retval NVSIPL_STATUS_ERROR          Failed to assert SYS_CHECK
     */
     virtual SIPLStatus AssertImagerSysCheck(void) const = 0;

    /**
     * @brief Get Sensor temperature.
     *
     * This function is used by the application to get sensor temperature
     *
     * @param[out] temp : sensor temperature
     *
     * @retval NVSIPL_STATUS_OK             Success
     * @retval NVSIPL_STATUS_BAD_ARGUMENT   Argument validation failed
     * @retval NVSIPL_STATUS_ERROR          Device access failed.
     */
     virtual SIPLStatus GetSensorTemperature(
         float_t * const temp) const = 0;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY

    virtual SIPLStatus readRegisterAR820 (uint16_t const registerAddr, uint16_t dataBuff[]) const = 0;
    virtual SIPLStatus writeRegisterAR820(uint16_t const registerAddr, uint16_t const dataBuff[]) const = 0;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

protected:
    virtual ~AR0820_CUST1_CustomInterface(void) = default;
};

}
#endif //AR0820_CUST1_CUSTOMINTERFACE_HPP
