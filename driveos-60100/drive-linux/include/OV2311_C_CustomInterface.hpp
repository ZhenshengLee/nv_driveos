/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _OV2311_MAX96717F_CUSTOMINTERFACE_HPP_
#define _OV2311_MAX96717F_CUSTOMINTERFACE_HPP_

#include "INvSIPLDeviceInterfaceProvider.hpp"
#include "NvSIPLCommon.hpp"

namespace nvsipl
{
// This is version 1 UUID obtained using https://www.uuidgenerator.net/
// This will be used to uniquely identify this interface
// The client can use this ID to validate the correct interface before use
const UUID OV2311_MAX96717F_CUSTOM_INTERFACE_ID(0x7e5309ccU, 0xaa2eU, 0x11ecU, 0xb909U,
                                              0x02U, 0x42U, 0xacU, 0x12U, 0x00U, 0x02U);

class OV2311_MAX96717F_CustomInterface : public Interface
{
public:
    static const UUID& getClassInterfaceID() {
        return OV2311_MAX96717F_CUSTOM_INTERFACE_ID;
    }

    // Used for a confirmatory test by the app to ensure typecasted pointer
    // indeed points to the right object
    const UUID& getInstanceInterfaceID() {
        return OV2311_MAX96717F_CUSTOM_INTERFACE_ID;
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

protected:
    ~OV2311_MAX96717F_CustomInterface() = default;
};

} // end of namespace nvsipl
#endif // _OV2311_MAX96717F_CUSTOMINTERFACE_HPP_
