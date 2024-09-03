/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMPMIC.hpp"

namespace nvsipl
{

SIPLStatus CNvMPMIC::SetConfig(uint8_t const i2cAddress, DeviceParams const *const params)
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    if (params == nullptr) {
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    } else {
        m_oDeviceParams = *params;

        if ((!m_oDeviceParams.bEnableSimulator) and (!m_oDeviceParams.bPassive)) {
            m_nativeI2CAddr = i2cAddress;
            m_oDeviceParams.pI2CAddrMgr->RegisterNativeI2CAddr(m_nativeI2CAddr);
        }

        /*
        * Always use CDI version 2 API (CDAC) for QNX and not for Linux.
        */
#ifdef NVMEDIA_QNX
        m_oDeviceParams.bUseCDIv2API = NVMEDIA_TRUE;
#else
        m_oDeviceParams.bUseCDIv2API = NVMEDIA_FALSE;
#endif

        m_eState = DeviceState::CDI_DEVICE_CONFIG_SET;
    }

    return status;
}

} // end of nvsipl
