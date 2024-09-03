/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CNVMPMIC_HPP
#define CNVMPMIC_HPP

#include "NvSIPLPlatformCfg.hpp"
#include "devblk_cdi.h"
#include "CNvMDevice.hpp"
#include "utils/utils.hpp"

namespace nvsipl
{

/**
 * The CNvMPMIC class encapsulates PMIC control and configuration information.
 */
class CNvMPMIC : public CNvMDevice
{
public:
    /**
    * Default constructor
    */
    CNvMPMIC() = default;

    /**
    * Default Destructor
    */
    virtual ~CNvMPMIC() = default;


    /**
     * Sets the configuration for the PMIC object based on the params
     * Updates the object's state information to indicate that configuration has been set
     *
     * If simulator mode is not enabled and passive mode is not enabled,
     * this will save the PMIC's I2C address and register it with the address manager.
     *
     * - verify params is NOT nullptr
     * - set m_oDeviceParams = *params
     * - if (NOT m_oDeviceParams.bEnableSimulator) and ( NOT m_oDeviceParams.bPassive)
     *  - set m_nativeI2CAddr =  eepromInfo->i2cAddress
     *  - register I2C address by
     *   - m_oDeviceParams.pI2CAddrMgr->RegisterNativeI2CAddr(m_nativeI2CAddr)
     *   .
     *  .
     * - set m_oDeviceParams.bUseCDIv2API = NVMEDIA_TRUE
     * - set m_eState = CDI_DEVICE_CONFIG_SET
     *
     * @param[in] i2cAddress        PMIC I2C address
     * @param[in] params            Device information used to register I2C address
     * @retval                      NVSIPL_STATUS_OK on completion
     */
    virtual SIPLStatus SetConfig(uint8_t const i2cAddress, DeviceParams const *const params);

    /**
     * Sets the object's driver handle to the provided driver
     *
     * - verify driver is NOT nullptr
     * - set m_pCDIDriver = driver
     *
     * @param[in] driver            DevBlkCDIDeviceDriver ptr to store
     * @retval                      NVSIPL_STATUS_OK on successful completion
     * @retval                      NVSIPL_STATUS_BAD_ARGUMENT on error
     */
    SIPLStatus SetDriverHandle(DevBlkCDIDeviceDriver * const driver) {
        SIPLStatus status = NVSIPL_STATUS_BAD_ARGUMENT;
        if (driver != nullptr) {
            m_pCDIDriver = driver;
            status = NVSIPL_STATUS_OK;
        }

        return status;
    }

    /** @brief stub for unsupported function
     * @return NVSIPL_STATUS_OK */
    SIPLStatus DoInit() override
    {
        return NVSIPL_STATUS_OK;
    };

    /** @brief stub for unsupported function
     * @return NVSIPL_STATUS_OK */
    SIPLStatus DoStart() override
    {
        return NVSIPL_STATUS_OK;
    };

    SIPLStatus DoStop() override
    {
        return NVSIPL_STATUS_OK;
    };

};

} // end of namespace nvsipl
#endif //CNVMPMIC_HPP

