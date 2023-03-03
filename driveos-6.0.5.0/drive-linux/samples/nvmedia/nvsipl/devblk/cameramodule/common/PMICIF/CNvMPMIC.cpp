/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMPMIC.hpp"

namespace nvsipl
{

SIPLStatus CNvMPMIC::SetConfig(uint8_t const i2cAddress, const DeviceParams *const params)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    if (params == nullptr) {
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    } else {
        m_oDeviceParams = *params;

        if ((!m_oDeviceParams.bEnableSimulator) and (!m_oDeviceParams.bPassive)) {
            m_nativeI2CAddr = i2cAddress;
            m_oDeviceParams.pI2CAddrMgr->RegisterNativeI2CAddr(m_nativeI2CAddr);
        }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
        m_oDeviceParams.bUseCDIv2API = NVMEDIA_FALSE;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#else
        m_oDeviceParams.bUseCDIv2API = NVMEDIA_TRUE;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

        m_eState = CDI_DEVICE_CONFIG_SET;
    }

    return status;
}

} // end of nvsipl
