/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMMax96759.hpp"
#include "sipl_error.h"

extern "C" {
#include "cdi_max96759.h"
}

namespace nvsipl
{

SIPLStatus CNvMMax96759::SetConfig(const SerInfo *serializerInfo, DeviceParams *params)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    if (m_eState != DeviceState::CREATED) {
        SIPL_LOG_ERR_STR("MAX96759: CDI invalid state\n");
        return NVSIPL_STATUS_INVALID_STATE;
    }

    status = CNvMSerializer::SetConfig(serializerInfo, params);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96759: CNvMSerializer::SetConfig failed with SIPL error", (int32_t)status);
        return status;
    }

    /*! Get CDI Driver */
    m_pCDIDriver = GetMAX96759Driver();
    if (m_pCDIDriver == nullptr) {
        SIPL_LOG_ERR_STR("MAX96759: GetMAX96759Driver() failed");
        return NVSIPL_STATUS_ERROR;
    }

    m_eState = DeviceState::CDI_DEVICE_CONFIG_SET;
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMax96759::GetErrorSize(size_t & errorSize) const
{
    errorSize = 0;
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMax96759::GetErrorInfo(std::uint8_t * const buffer,
                                     const std::size_t bufferSize,
                                     std::size_t &size) const
{
    size = 0U;
    return NVSIPL_STATUS_OK;
}

} // end of nvsipl
