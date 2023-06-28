/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMMax9295.hpp"
#include "sipl_error.h"

extern "C" {
#include "cdi_max9295.h"
}

namespace nvsipl
{

SIPLStatus CNvMMax9295::SetConfig(const SerInfo *serializerInfo, DeviceParams *params)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    if (m_eState != CREATED) {
        SIPL_LOG_ERR_STR("MAX9295: CDI invalid state");
        return NVSIPL_STATUS_INVALID_STATE;
    }

    status = CNvMSerializer::SetConfig(serializerInfo, params);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: CNvMSerializer::SetConfig failed with SIPL error", (int32_t)status);
        return status;
    }

    /*! Get CDI Driver */
    m_pCDIDriver = GetMAX9295Driver();
    if (m_pCDIDriver == nullptr) {
        SIPL_LOG_ERR_STR("MAX9295: GetMAX9295Driver() failed");
        return NVSIPL_STATUS_ERROR;
    }

    m_eState = CDI_DEVICE_CONFIG_SET;
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMax9295::GetErrorSize(size_t & errorSize) const
{
#if USE_MOCK_ERRORS
    errorSize = 1;
#else
    errorSize = sizeof(uint8_t); // 8 bits containing a set of error values
#endif
    return NVSIPL_STATUS_OK;
}


SIPLStatus CNvMMax9295::GetErrorInfo(std::uint8_t * const buffer,
                                     const std::size_t bufferSize,
                                     std::size_t &size) const
{
    SIPLStatus status = NVSIPL_STATUS_OK;
#if USE_MOCK_ERRORS
    size = 1U;
#else
    NvMediaStatus nvmStatus;
    uint8_t errorStatus = 0U; // 8 bits containing a set of error values

    if (buffer == nullptr) {
        SIPL_LOG_ERR_STR("MAX9295: Buffer provided to GetErrorInfo is null");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    if (bufferSize < sizeof(errorStatus)) {
        SIPL_LOG_ERR_STR("MAX9295: Buffer provided to GetErrorInfo too small");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    // Read error register
    // In power-over-coax mode (which we use), we use the multi function pins for line faults
    // Faults in serial-link lines are reported on the ERRB output pin and stored in register 0x1B (LFLT_INT)
    // 0x1B contains ||PHY_INT_A|REM_ERR_FLAG||LFLT_INIT|IDLE_ERR_FLAG||DEC_ERR_FLAG_A
    nvmStatus = MAX9295ReadErrorStatus(m_upCDIDevice.get(), sizeof(errorStatus), &errorStatus);

    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: MAX9295ReadErrorStatus failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    memcpy(buffer, &errorStatus, sizeof(errorStatus));
    size = sizeof(errorStatus);
#endif /* not USE_MOCK_ERRORS */

    return status;
}

} // end of nvsipl
