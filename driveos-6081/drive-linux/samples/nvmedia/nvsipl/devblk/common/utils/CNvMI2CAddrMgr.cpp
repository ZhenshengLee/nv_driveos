/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "CNvMI2CAddrMgr.hpp"
#include "sipl_error.h"
#include <vector>
#include <algorithm>

namespace nvsipl
{

CNvMI2CAddrMgr::CNvMI2CAddrMgr()
{
    /* These addresses are reserved by I2C spec */
    reservedI2CAddr.push_back(0x0U); // General call/START byte
    reservedI2CAddr.push_back(0x1U); // CBUS address
    reservedI2CAddr.push_back(0x2U); // Reserved for different bus format
    reservedI2CAddr.push_back(0x3U); // Reserved for future purposes
    reservedI2CAddr.push_back(0x4U); // Hs-mode master code
    reservedI2CAddr.push_back(0x5U); // Hs-mode master code
    reservedI2CAddr.push_back(0x6U); // Hs-mode master code
    reservedI2CAddr.push_back(0x7U); // Hs-mode master code
    reservedI2CAddr.push_back(0x78U); // 10-bit slave addressing
    reservedI2CAddr.push_back(0x79U); // 10-bit slave addressing
    reservedI2CAddr.push_back(0x7AU); // 10-bit slave addressing
    reservedI2CAddr.push_back(0x7BU); // 10-bit slave addressing
    reservedI2CAddr.push_back(0x7CU); // Reserved for future purposes
    reservedI2CAddr.push_back(0x7DU); // Reserved for future purposes
    reservedI2CAddr.push_back(0x7EU); // Reserved for future purposes
    reservedI2CAddr.push_back(0x7FU); // Reserved for future purposes

    /* Additional reserved address can be added here */
    reservedI2CAddr.push_back(0x58U); //Device at 0x58 on i2c bus 2 of E3550.

    /// @todo Add reservedI2CAddr depending on module/platform (right now we assume E3550 only).
}

void CNvMI2CAddrMgr::RegisterNativeI2CAddr(uint8_t const nativeI2CAddr)
{
    if(std::find(reservedI2CAddr.begin(), reservedI2CAddr.end(), nativeI2CAddr) == reservedI2CAddr.end()) {
        reservedI2CAddr.push_back(nativeI2CAddr);
    } else {
        LOG_INFO("Native I2C address already registered\n");
    }
}

uint8_t CNvMI2CAddrMgr::GenerateI2CAddr(uint8_t const nativeI2CAddr)
{
    uint8_t generatedI2CAddr = nativeI2CAddr;

    if (reservedI2CAddr.size() == 128U) {
        SIPL_LOG_ERR_STR_INT("No available I2C address while registering", static_cast<int32_t>(nativeI2CAddr));
        generatedI2CAddr = 0U;
    } else {
        while (std::find(reservedI2CAddr.begin(), reservedI2CAddr.end(), generatedI2CAddr) != reservedI2CAddr.end()) {
            // Address is present in the table
            if (generatedI2CAddr == 127U) {
                generatedI2CAddr = 0U;
            } else {
                generatedI2CAddr++;
            }
        }

        reservedI2CAddr.push_back(generatedI2CAddr);
        LOG_INFO("Generated I2C address 0x%x for native I2C address 0x%x\n", generatedI2CAddr, nativeI2CAddr);
    }

    return generatedI2CAddr;
}

} // end of namespace nvsipl
