/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMCampwr.hpp"

namespace nvsipl
{

SIPLStatus CNvMCampwr::SetConfig(uint8_t i2cAddress, const DeviceParams *const params)
{
    m_oDeviceParams = *params;

    if ((!m_oDeviceParams.bEnableSimulator) and (!m_oDeviceParams.bPassive)) {
        m_nativeI2CAddr = i2cAddress;
        m_oDeviceParams.pI2CAddrMgr->RegisterNativeI2CAddr(m_nativeI2CAddr);
    }
#if (USE_CDAC == 0)
    m_oDeviceParams.bUseCDIv2API = NVMEDIA_FALSE;
#else
    m_oDeviceParams.bUseCDIv2API = NVMEDIA_TRUE;
#endif

    m_eState = CDI_DEVICE_CONFIG_SET;

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMCampwr::GetErrorSize(size_t & errorSize)
{
    LOG_WARN("Detailed error information is not supported for this serializer in CNvMCampwr::GetErrorInfo\n");
    errorSize = 0U;
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

SIPLStatus CNvMCampwr::GetErrorInfo(std::uint8_t * const buffer,
                                        const std::size_t bufferSize,
                                        std::size_t &size)
{
    LOG_WARN("Detailed error information is not supported for this serializer in CNvMCampwr::GetErrorInfo\n");

    return NVSIPL_STATUS_NOT_SUPPORTED;
}

SIPLStatus CNvMCampwr::isSupported()
{
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

SIPLStatus CNvMCampwr::PowerControlSetUnitPower(DevBlkCDIDevice* cdiDev, uint8_t const linkIndex, bool const enable)
{
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

SIPLStatus CNvMCampwr::CreatePowerDevice(DevBlkCDIRootDevice* const cdiRoot, const uint8_t linkIndex)
{
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

DevBlkCDIDevice* CNvMCampwr::GetDeviceHandle()
{
    return nullptr;
}

SIPLStatus CNvMCampwr::CheckPresence(DevBlkCDIRootDevice* const cdiRootDev,
                                     DevBlkCDIDevice* const cdiDev)
{
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

SIPLStatus CNvMCampwr::InitPowerDevice(DevBlkCDIRootDevice* const cdiRootDev,
                        DevBlkCDIDevice* const cdiDev, uint8_t const linkIndex)
{
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

SIPLStatus
CNvMCampwr::ReadRegister(DevBlkCDIDevice const * const handle, uint8_t const
            linkIndex, uint32_t const registerNum, uint32_t const dataLength,
            uint8_t * const dataBuff)
{
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

SIPLStatus
CNvMCampwr::WriteRegister(DevBlkCDIDevice const * const handle, uint8_t
            const linkIndex, uint32_t const registerNum, uint32_t const
            dataLength, uint8_t * const dataBuff)
{
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

} // end of nvsipl