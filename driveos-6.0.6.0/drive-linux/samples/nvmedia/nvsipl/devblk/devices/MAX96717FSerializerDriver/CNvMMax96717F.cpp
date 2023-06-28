/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */
#include "CNvMMax96717F.hpp"
#include "sipl_error.h"

extern "C" {
#include "cdi_max96717f.h"
}

namespace nvsipl
{
SIPLStatus CNvMMax96717F::SetConfig(const SerInfo *serializerInfo, DeviceParams *params)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    if (m_eState != CREATED) {
        SIPL_LOG_ERR_STR("MAX96717F: CDI invalid state\n");
        return NVSIPL_STATUS_INVALID_STATE;
    }

    status = CNvMSerializer::SetConfig(serializerInfo, params);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96717F: CNvMSerializer::SetConfig failed", (int32_t)status);
        return status;
    }

    // Get CDI Driver
    m_pCDIDriver = GetMAX96717FDriver();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96717F: GetMAX96717FDriver() in CNvMMax96705::SetConfig failed", (int32_t)status);
        return status;
    }

    m_eState = CDI_DEVICE_CONFIG_SET;
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMax96717F::GetErrorInfo(std::uint8_t * const buffer,
                                       const std::size_t bufferSize,
                                       std::size_t &size) const
{
    size = 0U;
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMax96717F::GetErrorSize(size_t & errorSize) const
{
#if USE_MOCK_ERRORS
    errorSize = 1;
#else
    errorSize = sizeof(uint8_t); //8bits containing a set of error value
#endif
    return NVSIPL_STATUS_OK;
}

} // end of nvsipl
