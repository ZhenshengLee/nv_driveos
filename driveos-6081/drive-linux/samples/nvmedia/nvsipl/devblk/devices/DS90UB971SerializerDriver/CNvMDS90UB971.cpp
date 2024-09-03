/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMDS90UB971.hpp"
#include "sipl_error.h"

extern "C" {
#include "cdi_ds90ub971.h"
}

namespace nvsipl
{

SIPLStatus CNvMDS90UB971::SetConfig(const SerInfo *serializerInfo, DeviceParams *params)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    if (m_eState != DeviceState::CREATED) {
        SIPL_LOG_ERR_STR("DS90UB971: CDI invalid state");
        return NVSIPL_STATUS_INVALID_STATE;
    }

    status = CNvMSerializer::SetConfig(serializerInfo, params);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB971: CNvMSerializer::SetConfig failed with SIPL error", (int32_t)status);
        return status;
    }

    /*! Get CDI Driver */
    m_pCDIDriver = GetDS90UB971Driver();
    if (m_pCDIDriver == nullptr) {
        SIPL_LOG_ERR_STR("DS90UB971: GetDS90UB971Driver() failed");
        return NVSIPL_STATUS_ERROR;
    }

    m_eState = DeviceState::CDI_DEVICE_CONFIG_SET;
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMDS90UB971::GetErrorSize(size_t & errorSize) const
{
    errorSize = 0;
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMDS90UB971::GetErrorInfo(std::uint8_t * const buffer,
                                     const std::size_t bufferSize,
                                     std::size_t &size) const
{
    size = 0U;
    return NVSIPL_STATUS_OK;
}

} // end of nvsipl
