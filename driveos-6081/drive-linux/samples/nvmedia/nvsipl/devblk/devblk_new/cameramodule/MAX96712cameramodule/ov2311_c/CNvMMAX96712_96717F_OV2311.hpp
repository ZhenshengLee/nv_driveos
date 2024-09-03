/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2021-2022, OmniVision Technologies.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _CNVMMAX96712_96717F_OV2311_HPP_
#define _CNVMMAX96712_96717F_OV2311_HPP_

#include "cameramodule/MAX96712cameramodule/CNvMMAX96712_96717FCameraModule.hpp"
#include "OV2311_C_CustomInterface.hpp"

namespace nvsipl
{

class CNvMMAX96712_96717F_OV2311 : public CNvMMAX96712_96717FCameraModule,
public OV2311_MAX96717F_CustomInterface
{
public:
    virtual SIPLStatus GetErrorInfo(std::uint8_t * const buffer,
                            const std::size_t bufferSize,
                            std::size_t &size) const
    {
        size = 0U;
        LOG_WARN("GetErrorInfo not implemented for sensor");
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus
    GetErrorSize(size_t & sensorErrorSize) const
    {
        sensorErrorSize = 0U;
        LOG_WARN("GetErrorSize not implemented for sensor\n");
        return NVSIPL_STATUS_OK;
    };

protected:
    virtual SIPLStatus SetConfigModule(SensorInfo const* const sensorInfo, CNvMDevice::DeviceParams const* const params) override;

    virtual SIPLStatus DetectModule() override;

    virtual SIPLStatus InitModule() const override;

    virtual SIPLStatus PostInitModule() override;

    virtual SIPLStatus StartModule() override;

    virtual SIPLStatus StopModule() override;

    virtual SIPLStatus DeinitModule() override;

    virtual DevBlkCDIDeviceDriver *GetCDIDeviceDriver() const override;

    virtual DevBlkCDIDeviceDriver *GetCDIDeviceDriver(ModuleComponent const component) const override;

    virtual SIPLStatus SetSensorConnectionProperty(
        CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty
            * const sensorConnectionProperty) const noexcept override;

    virtual std::unique_ptr<CNvMDevice::DriverContext> GetCDIDeviceContext() const override;

    virtual Interface* GetInterface(const UUID &interfaceId) override;

    virtual SIPLStatus SetCustomValue(std::uint32_t const valueToSet) override;

    virtual SIPLStatus GetCustomValue(std::uint32_t * const valueToGet) override;

    virtual uint16_t GetPowerOnDelayMs() override;

    virtual uint16_t GetPowerOffDelayMs() noexcept override;

private:
     /**
      * @brief Setup address translation in the serializer.
      *
      * @param[in] serCDI Serializer CDI handle.
      * @param[in] src Source I2C Address
      * @param[in] dst Destination I2C Address
      *
      * @retval NVSIPL_STATUS_OK Successful completion.
      * @retval NVSIPL_STATUS_BAD_ARGUMENT CDI Device handle is null.
      * @retval NVSIPL_STATUS_ERROR Unable to setup address translation.
      */
     SIPLStatus SetupAddressTranslationsB(DevBlkCDIDevice const* const serCDI,
                                          uint8_t const src,
                                          uint8_t const dst) const;

    /**
     * @brief Set the serializer address translation to PMIC
     *
     * @retval NVSIPL_STATUS_OK EEPROM Call was successful
     * @retval NVSIPL_STATUS_BAD_ARGUMENT Invalid src and dst i2c address passed
     * @retval NVSIPL_STATUS_ERROR Unable to set the address translator.
     */
    SIPLStatus SetTrans2VCSEL();
    int m_ConfigIndex = 0;
    static const uint16_t POWER_ON_DELAY_MS  = 200U;
    static const uint16_t POWER_OFF_DELAY_MS = 100U;

    SIPLStatus InitSimulatorAndPassive() const;
};

} // end of namespace

#endif /* _CNVMMAX96712_96717F_OV2311_HPP_ */
