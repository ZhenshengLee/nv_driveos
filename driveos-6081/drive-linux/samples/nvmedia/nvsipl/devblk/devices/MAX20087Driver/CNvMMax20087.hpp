/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CNVMMAX20087_HPP
#define CNVMMAX20087_HPP

#include "CampwrIF/CNvMCampwr.hpp"
#include "utils/utils.hpp"
#include <devblk_cdi.h>

namespace nvsipl
{

/*! Serializer */
class CNvMMax20087 final : public CNvMCampwr
{
public:
    SIPLStatus SetConfig(uint8_t i2cAddress, const DeviceParams *const params) override;

    SIPLStatus GetErrorSize(size_t & errorSize) override;

    SIPLStatus GetErrorInfo(std::uint8_t * const buffer,
                                      const std::size_t bufferSize,
                                      std::size_t &size) override;


    SIPLStatus isSupported() override;

    SIPLStatus PowerControlSetUnitPower(DevBlkCDIDevice* cdiDev, uint8_t const linkIndex, bool const enable) override;

    SIPLStatus CreatePowerDevice(DevBlkCDIRootDevice* const cdiRootDev, const uint8_t linkIndex) override;

    DevBlkCDIDevice* GetDeviceHandle() override;

    SIPLStatus InitPowerDevice(DevBlkCDIRootDevice* const cdiRootDev,
               DevBlkCDIDevice* const cdiDev, uint8_t const linkIndex,
               int32_t const csiPort) override;

    SIPLStatus ReadRegister(DevBlkCDIDevice const * const handle,
                uint8_t const linkIndex, uint32_t const registerNum,
                uint32_t const dataLength, uint8_t * const dataBuff) override;

    SIPLStatus WriteRegister(DevBlkCDIDevice const * const handle,
                uint8_t const linkIndex, uint32_t const registerNum,
                uint32_t const dataLength,
                uint8_t const * const dataBuff) override;

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

    /* Mask or restore mask of interrupts */
    SIPLStatus MaskRestoreInterrupt(const bool enableGlobalMask) override;

};

} // end of namespace nvsipl
#endif // CNVMMAX20087_HPP

