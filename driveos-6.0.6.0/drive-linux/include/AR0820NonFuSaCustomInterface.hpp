/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _AR0820_NONFUSA_CUSTOMINTERFACE_HPP_
#define _AR0820_NONFUSA_CUSTOMINTERFACE_HPP_

#include "INvSIPLDeviceInterfaceProvider.hpp"
#include "NvSIPLCommon.hpp"

namespace nvsipl
{
// This is version 1 UUID obtained using https://www.uuidgenerator.net/
// This will be used to uniquely identify this interface
// The client can use this ID to validate the correct interface before use
//024ea870-a3f3-4b4c-bbb9-dfba64955a1b
const UUID AR0820_NONFUSA_CUSTOM_INTERFACE_ID(0x024ea870U, 0xa3f3U, 0x4b4cU, 0xbbb9U,
                                              0xdfU, 0xbaU, 0x64U, 0x95U, 0x5AU, 0x1BU);


class AR0820NonFuSaCustomInterface : public Interface
{
public:
    static const UUID& getClassInterfaceID() {
        return AR0820_NONFUSA_CUSTOM_INTERFACE_ID;
    }

    // Used for a confirmatory test by the app to ensure typecasted pointer
    // indeed points to the right object
    const UUID& getInstanceInterfaceID() {
        return AR0820_NONFUSA_CUSTOM_INTERFACE_ID;
    }

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
    ~AR0820NonFuSaCustomInterface() = default;
};

} // end of namespace nvsipl
#endif // _AR0820_NONFUSA_CUSTOMINTERFACE_HPP_
