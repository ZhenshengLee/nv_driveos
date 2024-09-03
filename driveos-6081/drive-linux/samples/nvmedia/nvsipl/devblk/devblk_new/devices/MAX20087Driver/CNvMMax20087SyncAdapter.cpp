/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <mutex>

#include "CNvMMax20087SyncAdapter.hpp"
#include "CNvMMax20087.hpp"
#include "cdi_max20087.h"
#include "sipl_error.h"

namespace Impl {

using namespace nvsipl;

/**
 * Sets the configuration for the camera power device object based on the
 * params. Updates the object's state information to indicate that
 * configuration has been set.
*/
SIPLStatus
CNvMMax20087SyncAdapter::SetConfig(uint8_t i2cAddress,
    DeviceParams const * const params)
{
    std::lock_guard<std::mutex> lock(m_lock);
    if (m_driverConfigSet) {
        LOG_INFO("Sync Adapter Configuration Already Set\n");
        return NVSIPL_STATUS_OK;
    }

    LOG_INFO("Configure Sync Adapter\n");
    SIPLStatus ret = m_driver.SetConfig(i2cAddress, params);
    if (ret == NVSIPL_STATUS_OK) {
        m_driverConfigSet = true;
    }

    return ret;
}

// Check to see if a power control device is supported
SIPLStatus
CNvMMax20087SyncAdapter::isSupported()
{
    std::lock_guard<std::mutex> lock(m_lock);
    return m_driver.isSupported();
}

// Use power control device to power on/off the camera modules.
SIPLStatus
CNvMMax20087SyncAdapter::PowerControlSetUnitPower(DevBlkCDIDevice* cdiDev, uint8_t const linkIndex, bool const enable)
{
    std::lock_guard<std::mutex> lock(m_lock);
    if (cdiDev == nullptr) {
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    return m_driver.PowerControlSetUnitPower(cdiDev, linkIndex, enable);
}


// Creates a new power control CDI device.
SIPLStatus
CNvMMax20087SyncAdapter::CreatePowerDevice(
    DevBlkCDIRootDevice* const cdiRootDev,
    uint8_t const linkIndex)
{
    std::lock_guard<std::mutex> lock(m_lock);
    if (cdiRootDev == nullptr) {
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    if (m_driverIsCreated) {
        LOG_INFO("Sync Adapter Driver Already Created\n");
        return NVSIPL_STATUS_OK;
    }

    LOG_INFO("Create Sync Adapter Driver\n");
    SIPLStatus ret = m_driver.CreatePowerDevice(cdiRootDev, linkIndex);
    if (ret == NVSIPL_STATUS_OK) {
        m_driverIsCreated = true;
    }

    return ret;
}

// Retrieves the CDI device handle for the power device.
DevBlkCDIDevice*
CNvMMax20087SyncAdapter::GetDeviceHandle()
{
    std::lock_guard<std::mutex> lock(m_lock);
    return m_driver.GetDeviceHandle();
}

// Check the device presence  for the power device.
SIPLStatus
CNvMMax20087SyncAdapter::CheckPresence(DevBlkCDIRootDevice* const cdiRootDev,
                                       DevBlkCDIDevice* const cdiDev)
{
    std::lock_guard<std::mutex> lock(m_lock);
    if (cdiDev == nullptr) {
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    return m_driver.CheckPresence(cdiRootDev, cdiDev);
}

// Initializes the power device object.
SIPLStatus
CNvMMax20087SyncAdapter::InitPowerDevice(DevBlkCDIRootDevice* const cdiRootDev,
                            DevBlkCDIDevice* const cdiDev, uint8_t const linkIndex,
                            int32_t const csiPort)
{
    std::lock_guard<std::mutex> lock(m_lock);
    if (cdiDev == nullptr) {
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    if (m_driverIsInitialized) {
        LOG_INFO("Sync Adapter Driver Already Initialized\n");
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus const ret { m_driver.InitPowerDevice(cdiRootDev, cdiDev,
                                                    linkIndex, csiPort) };
    if (ret == NVSIPL_STATUS_OK) {
        m_driverIsInitialized = true;
    }

    return ret;
}

// Read from a PowerSwitch Register.
SIPLStatus
CNvMMax20087SyncAdapter::ReadRegister(DevBlkCDIDevice const * const handle,
                         uint8_t const linkIndex, uint32_t const registerNum,
                         uint32_t const dataLength, uint8_t * const dataBuff)
{
    std::lock_guard<std::mutex> lock(m_lock);

    SIPLStatus status = NVSIPL_STATUS_OK;
    if ((handle == nullptr) || (dataBuff == nullptr)) {
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    } else {
        status = m_driver.ReadRegister(handle, linkIndex, registerNum,
                              dataLength, dataBuff);
    }

    return status;
}

// Write to a PowerSwitch Register.
SIPLStatus
CNvMMax20087SyncAdapter::WriteRegister(DevBlkCDIDevice const * const handle,
                         uint8_t const linkIndex, uint32_t const registerNum,
                         uint32_t const dataLength,
                         uint8_t const * const dataBuff)
{
    std::lock_guard<std::mutex> lock(m_lock);

    SIPLStatus status {NVSIPL_STATUS_OK};
    if ((handle == nullptr) || (dataBuff == nullptr)) {
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    } else {
        status = m_driver.WriteRegister(handle, linkIndex, registerNum,
                              dataLength, dataBuff);
    }

    return status;
}

/* Mask or restore mask of interrupts */
SIPLStatus
CNvMMax20087SyncAdapter::MaskRestoreInterrupt(const bool enableGlobalMask)
{
    return m_driver.MaskRestoreInterrupt(enableGlobalMask);
}

/**
 * Sets the configuration for the camera power device object based on the
 * params. Updates the object's state information to indicate that
 * configuration has been set.
*/
SIPLStatus
CNvMMax20087AccessToken::SetConfig(uint8_t i2cAddress,
    DeviceParams const * const params)
{
    if (params == nullptr) {
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }
    return m_adapter->SetConfig(i2cAddress, params);
}

// Check to see if a power control device is supported
SIPLStatus
CNvMMax20087AccessToken::isSupported()
{
    return m_adapter->isSupported();
}

// Use power control device to power on/off the camera modules.
SIPLStatus
CNvMMax20087AccessToken::PowerControlSetUnitPower(DevBlkCDIDevice* cdiDev, uint8_t const linkIndex, bool const enable)
{
    if (cdiDev == nullptr) {
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    if (linkIndex != m_linkIdx) {
        SIPL_LOG_ERR_STR("Access Token Detected Illegal Access!");
        return NVSIPL_STATUS_ERROR;
    }

    return m_adapter->PowerControlSetUnitPower(cdiDev, linkIndex, enable);
}

// Creates a new power control CDI device.
SIPLStatus
CNvMMax20087AccessToken::CreatePowerDevice(
    DevBlkCDIRootDevice* const cdiRootDev, uint8_t const linkIndex)
{
    if (cdiRootDev == nullptr) {
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    if (linkIndex != m_linkIdx) {
        SIPL_LOG_ERR_STR("Access Token Detected Illegal Access!");
        return NVSIPL_STATUS_ERROR;
    }
    return m_adapter->CreatePowerDevice(cdiRootDev, linkIndex);
}

//  Retrieves the CDI device handle for the power device.
DevBlkCDIDevice*
CNvMMax20087AccessToken::GetDeviceHandle()
{
    return m_adapter->GetDeviceHandle();
}

// Check the device presence  for the power device.
SIPLStatus
CNvMMax20087AccessToken::CheckPresence(DevBlkCDIRootDevice* const cdiRootDev,
                                       DevBlkCDIDevice* const cdiDev)
{
    return m_adapter->CheckPresence(cdiRootDev, cdiDev);
}

// Initializes the power device object.
SIPLStatus
CNvMMax20087AccessToken::InitPowerDevice(DevBlkCDIRootDevice* const cdiRootDev,
                            DevBlkCDIDevice* const cdiDev, uint8_t const linkIndex,
                            int32_t const csiPort)
{
    if (linkIndex != m_linkIdx) {
        SIPL_LOG_ERR_STR("Access Token Detected Illegal Access!");
        return NVSIPL_STATUS_ERROR;
    }
    return m_adapter->InitPowerDevice(cdiRootDev, cdiDev, linkIndex, csiPort);
}

// Read from a PowerSwitch Register.
SIPLStatus
CNvMMax20087AccessToken::ReadRegister(DevBlkCDIDevice const * const handle,
                         uint8_t const linkIndex, uint32_t const registerNum,
                         uint32_t const dataLength, uint8_t * const dataBuff)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    if (linkIndex != m_linkIdx) {
        SIPL_LOG_ERR_STR("Access Token Detected Illegal Access!");
        status = NVSIPL_STATUS_ERROR;
    } else if ((handle == nullptr) || (dataBuff == nullptr)) {
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    } else {
        status = m_adapter->ReadRegister(handle, linkIndex, registerNum,
                            dataLength, dataBuff);
    }
    return status;
}

// Write to a PowerSwitch Register.
SIPLStatus
CNvMMax20087AccessToken::WriteRegister(DevBlkCDIDevice const * const handle,
                         uint8_t const linkIndex, uint32_t const registerNum,
                         uint32_t const dataLength,
                         uint8_t const * const dataBuff)
{
    SIPLStatus status {NVSIPL_STATUS_OK};
    if (linkIndex != m_linkIdx) {
        SIPL_LOG_ERR_STR("Access Token Detected Illegal Access!");
        status = NVSIPL_STATUS_ERROR;
    } else if ((handle == nullptr) || (dataBuff == nullptr)) {
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    } else {
        status = m_adapter->WriteRegister(handle, linkIndex, registerNum,
                            dataLength, dataBuff);
    }
    return status;
}

/* Mask or restore mask of interrupts */
SIPLStatus
CNvMMax20087AccessToken::MaskRestoreInterrupt(const bool enableGlobalMask)
{
    return m_adapter->MaskRestoreInterrupt(enableGlobalMask);
}

} // end of namespace Impl
