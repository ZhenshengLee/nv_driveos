/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CNVMMAX20087_SYNC_ADAPTER_HPP
#define CNVMMAX20087_SYNC_ADAPTER_HPP

#include <mutex>

#include "CNvMMax20087.hpp"
#include "CampwrIF/CNvMCampwr.hpp"

namespace Impl {

using namespace nvsipl;

/**
 * \brief  A stateful & owning class of a CNvMMax20087 that adapts the
 * CNvMCampwr interface to a synchronized CNvMMax20087 interface.
 *
 * It's purpose is to let the CNvMMax20087 focus on being a driver.
 */
class CNvMMax20087SyncAdapter : public CNvMCampwr
{
  public:
    CNvMMax20087SyncAdapter() = default;

    virtual SIPLStatus SetConfig(uint8_t i2cAddress, const DeviceParams *const params) override;
    virtual SIPLStatus isSupported() override;
    virtual SIPLStatus PowerControlSetUnitPower(DevBlkCDIDevice* cdiDev, uint8_t const linkIndex, bool const enable) override;
    virtual SIPLStatus CreatePowerDevice(DevBlkCDIRootDevice* const cdiRoot, const uint8_t linkIndex) override;
    virtual DevBlkCDIDevice* GetDeviceHandle() override;
    virtual SIPLStatus InitPowerDevice(DevBlkCDIRootDevice* const cdiRootDev,
                        DevBlkCDIDevice* const cdiDev, uint8_t const linkIndex) override;
    virtual SIPLStatus ReadRegister(DevBlkCDIDevice const * const handle,
                       uint8_t const linkIndex, uint32_t const registerNum,
                       uint32_t const dataLength, uint8_t * const dataBuff)
                       override;
    virtual SIPLStatus WriteRegister(DevBlkCDIDevice const * const handle,
                       uint8_t const linkIndex, uint32_t const registerNum,
                       uint32_t const dataLength, uint8_t * const dataBuff)
                       override;

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

  private:
    CNvMMax20087 m_driver;
    std::mutex m_lock;

    bool m_driverConfigSet{ false };
    bool m_driverIsCreated{ false };
    bool m_driverIsInitialized{ false };
};

/**
 * \brief  A unique access token which allows the consumer to fully exercise
 * CNvMCampwr interface while being restricted to the link requested.
 */
class CNvMMax20087AccessToken : public CNvMCampwr
{
  public:
    CNvMMax20087AccessToken(CNvMMax20087SyncAdapter* adapter, int linkIdx)
      : m_adapter(adapter)
      , m_linkIdx(linkIdx)
    {}

    virtual SIPLStatus SetConfig(uint8_t i2cAddress, const DeviceParams *const params) override;
    virtual SIPLStatus isSupported() override;
    virtual SIPLStatus PowerControlSetUnitPower(DevBlkCDIDevice* cdiDev, uint8_t const linkIndex, bool const enable) override;
    virtual SIPLStatus CreatePowerDevice(DevBlkCDIRootDevice* const cdiRoot, const uint8_t linkIndex) override;
    virtual DevBlkCDIDevice* GetDeviceHandle() override;
    virtual SIPLStatus InitPowerDevice(DevBlkCDIRootDevice* const cdiRootDev,
                        DevBlkCDIDevice* const cdiDev, uint8_t const linkIndex) override;
    virtual SIPLStatus ReadRegister(DevBlkCDIDevice const * const handle,
                       uint8_t const linkIndex, uint32_t const registerNum,
                       uint32_t const dataLength, uint8_t * const dataBuff)
                       override;
    virtual SIPLStatus WriteRegister(DevBlkCDIDevice const * const handle,
                       uint8_t const linkIndex, uint32_t const registerNum,
                       uint32_t const dataLength, uint8_t * const dataBuff)
                       override;


    virtual SIPLStatus DoInit()
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus DoStart()
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus DoStop()
    {
        return NVSIPL_STATUS_OK;
    };

  private:
    CNvMMax20087SyncAdapter* m_adapter;
    uint8_t m_linkIdx;
};

} // end of namespace Impl
#endif // CNVMMAX20087_SYNC_ADAPTER_HPP
