/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMSerializer.hpp"

namespace nvsipl
{

SIPLStatus CNvMSerializer::SetConfig(SerInfo const *const serializerInfo, DeviceParams *const params)
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    if ((serializerInfo == nullptr) || (params == nullptr)) {
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    } else {
        m_oDeviceParams = *params;

        if ((!m_oDeviceParams.bEnableSimulator) and (!m_oDeviceParams.bPassive)) {
            m_nativeI2CAddr =  serializerInfo->i2cAddress;
            m_oDeviceParams.pI2CAddrMgr->RegisterNativeI2CAddr(m_nativeI2CAddr);
        }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
#if (USE_CDAC == 0)
        m_oDeviceParams.bUseCDIv2API = NVMEDIA_FALSE;
#else
        m_oDeviceParams.bUseCDIv2API = serializerInfo->useCDIv2API;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#else
        m_oDeviceParams.bUseCDIv2API = NVMEDIA_TRUE;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

    }

    return status;
}

SIPLStatus CNvMSerializer::GetErrorSize(size_t & errorSize) const
{
    LOG_WARN("Detailed error information is not supported for this serializer in CNvMSerializer::GetErrorInfo\n");
    errorSize = 0U;
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

SIPLStatus CNvMSerializer::GetErrorInfo(std::uint8_t * const buffer,
                                        std::size_t const bufferSize,
                                        std::size_t &size) const
{
    static_cast<void>(buffer);
    static_cast<void>(bufferSize);
    static_cast<void>(size);
    LOG_WARN("Detailed error information is not supported for this serializer in CNvMSerializer::GetErrorInfo\n");

    return NVSIPL_STATUS_NOT_SUPPORTED;
}

} // end of nvsipl
