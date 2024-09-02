/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _OV2311_NONFUSA_CUSTOMINTERFACE_HPP_
#define _OV2311_NONFUSA_CUSTOMINTERFACE_HPP_

#include "INvSIPLDeviceInterfaceProvider.hpp"
#include "NvSIPLCommon.hpp"

namespace nvsipl
{
// This is version 1 UUID obtained using https://www.uuidgenerator.net/
// This will be used to uniquely identify this interface
// The client can use this ID to validate the correct interface before use
const UUID OV2311_NONFUSA_CUSTOM_INTERFACE_ID(0xca09c89aU, 0x255bU, 0x11ecU, 0x9621U,
                                              0x02U, 0x42U, 0xacU, 0x13U, 0x00U, 0x02);

class OV2311NonFuSaCustomInterface : public Interface
{
public:
    static const UUID& getClassInterfaceID() {
        return OV2311_NONFUSA_CUSTOM_INTERFACE_ID;
    }

    // Used for a confirmatory test by the app to ensure typecasted pointer
    // indeed points to the right object
    const UUID& getInstanceInterfaceID() {
        return OV2311_NONFUSA_CUSTOM_INTERFACE_ID;
    }

    // Sample "set value" API
    // This type of custom API can be used to set a register value, etc.
    // The exact behavior can depend on the supported functionality of the sensor.
    virtual SIPLStatus SetCustomValue(uint32_t const valueToSet) = 0;

    // Sample "get value" API
    // This type of custom API can be used to read a register value,
    // parse custom embedded data, etc.
    // The exact behavior can depend on the supported functionality of the sensor.
    virtual SIPLStatus GetCustomValue(uint32_t * const valueToGet) = 0;

    /**
     * @brief Check the module availability
     *
     * Turn the module power on, enable the link and check the link lock
     * If the link lock is detected, keep the module power on and return NVSIPL_STATUS_OK
     * If the link lock is not detected, turn the module power off and return NVSIPL_STATUS_ERROR
     *
     * @retval NVSIPL_STATUS_OK           Success
     * @retval NVSIPL_STATUS_BAD_ARGUMENT Invalid input parameter
     * @retval NVSIPL_STATUS_ERROR        Any other error.
     */
    virtual SIPLStatus CheckModuleStatus() = 0;

protected:
    ~OV2311NonFuSaCustomInterface() = default;
};

} // end of namespace nvsipl
#endif // _OV2311_NONFUSA_CUSTOMINTERFACE_HPP_
