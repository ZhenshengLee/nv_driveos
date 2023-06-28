/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CNVMI2CADDRMGR_HPP
#define CNVMI2CADDRMGR_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include "utils/utils.hpp"

/**
 * @file
 * @brief <b> NVIDIA Device Block: I2C Address Manager </b>
 *
 * This file contains the interface for the I2C Address Manager.
 */

namespace nvsipl
{


/**
 * @defgroup NvSIPLDevBlk_I2C DevBlock I2C Management
 * @ingroup NvSIPLDevBlk_API
 *
 * @brief Management of I2C busses.
 *
 * @{
 */

/**
 * @brief Class which manages allocation of I2C addresses on a given I2C bus.
 */
class CNvMI2CAddrMgr
{
public:
    /**
     * @brief Constructor
     *
     * Initializes internal state.
     * Reserves several address ranges which have special meaning:
     *    - 0x00 - 0x07
     *    - 0x78 - 0x7f
     *
     * See [The I2C Bus Specification] for further information.
     *
     * Pre-registers any I2C addresses known to be used on the current
     * platform's camera I2C busses, but which are not themselves camera
     * devices. Camera device drivers should register their I2C devices using
     * RegisterNativeI2CAddr() rather than the list in this constructor.
     */
    CNvMI2CAddrMgr();

    /**
     * @brief Prevent CNvMI2CAddrMgr from being copy constructed.
     */
    CNvMI2CAddrMgr(const CNvMI2CAddrMgr &) = delete;

    /**
     * @brief Prevent CNvMI2CAddrMgr from being copy assigned.
     */
    CNvMI2CAddrMgr& operator=(const CNvMI2CAddrMgr &) = delete;

    /**
     * @brief Destructor
     *
     * Cleans up any internal allocations and state.
     */
    ~CNvMI2CAddrMgr() {};

    /**
     * @brief Registers an I2C address.
     *
     * Registers an I2C address with the manager so it cannot be allocated as a
     * virtual address.
     *
     * @param[in] nativeI2CAddr Address to reserve.
     */
    void RegisterNativeI2CAddr(uint8_t const nativeI2CAddr);

    /**
     * @brief Generate a fresh I2C address based on the native device address.
     *
     * All native device addresses on the bus managed by this manager must be
     * registered via RegisterNativeI2CAddr() before GenerateI2CAddr() is called.
     *
     * The generated address is marked as registered.
     *
     * @param[in] nativeI2CAddr The native address of the I2C device.
     *
     * @retval      0 when no additional address are available. (Failure)
     * @retval      ADDRESS a 7-bit I2C address.
     *              The returned address will not conflict with any address
     *              previously registed, including those implicitly registered
     *              by the constructor.
     */
    uint8_t GenerateI2CAddr(uint8_t const nativeI2CAddr);

private:
    /**
     * @brief List of reserved I2C addresses.
     *
     * Each element of this list represents a reserved 7-bit I2C bus address.
     */
    std::vector<uint8_t> reservedI2CAddr;
};

/** @} */ // End NvSIPLDevBlk_I2C

} // end of namespace nvsipl
#endif //CNVMI2CADDRMGR_HPP
