/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CNVMVCSEL_HPP
#define CNVMVCSEL_HPP

#include "NvSIPLPlatformCfg.hpp"
#include "devblk_cdi.h"
#include "CNvMDevice.hpp"
#include "utils/utils.hpp"

namespace nvsipl
{

/**
 * The CNvMVCSEL class encapsulates VCSEL control and configuration information.
 */
class CNvMVCSEL : public CNvMDevice
{
public:
    /**
    * Default constructor
    */
    CNvMVCSEL() = default;

    /**
    * Default Destructor
    */
    virtual ~CNvMVCSEL() = default;

    /**
     * Sets the configuration for the VCSEL object based on the params
     * Updates the object's state information to indicate that configuration has been set
     *
     * If simulator mode is not enabled and passive mode is not enabled,
     * this will save the VCSEL's I2C address and register it with the address manager.
     *
     * @param[in] i2cAddress        VCSEL I2C address
     * @param[in] params            Device information used to register I2C address
     * @retval                      NVSIPL_STATUS_OK on completion
     */
    virtual SIPLStatus SetConfig(uint8_t const i2cAddress, const DeviceParams *const params);

    /**
     * Sets the object's driver handle to the provided driver
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

    virtual SIPLStatus DoInit() override
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus DoStart() override
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus DoStop() override
    {
        return NVSIPL_STATUS_OK;
    };
};

} // end of namespace nvsipl
#endif //CNVMVCSEL_HPP
