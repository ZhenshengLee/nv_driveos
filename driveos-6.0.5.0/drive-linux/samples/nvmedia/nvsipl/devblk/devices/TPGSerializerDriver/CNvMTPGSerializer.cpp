/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMTPGSerializer.hpp"
#include "sipl_error.h"

extern "C" {
#include "cdi_tpgserializer.h"
}

namespace nvsipl
{

SIPLStatus CNvMTPGSerializer::SetConfig(const SerInfo *serializerInfo, DeviceParams *params)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    if (m_eState != CREATED) {
        SIPL_LOG_ERR_STR("TPGSerializer: CDI invalid state");
        return NVSIPL_STATUS_INVALID_STATE;
    }

    status = CNvMSerializer::SetConfig(serializerInfo, params);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("TPGSerializer: CNvMTPGSerializer::SetConfig failed with SIPL error", (int32_t)status);
        return status;
    }

    /*! Get CDI Driver */
    m_pCDIDriver = GetTPGSerializerDriver();
    if (m_pCDIDriver == nullptr) {
        SIPL_LOG_ERR_STR("TPGSerializer: GetTPGSerializerDriver() failed");
        return NVSIPL_STATUS_ERROR;
    }

    m_eState = CDI_DEVICE_CONFIG_SET;
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMTPGSerializer::GetErrorSize(size_t & errorSize) const
{
    /*  Need to retrieve maximum error size expected to be returned by the driver.
        This will be used by the client to allocate buffers for error detail retrieval
    */
    errorSize = TPGSerializerReadErrorSize();
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMTPGSerializer::GetErrorInfo(std::uint8_t * const buffer,
                                     const std::size_t bufferSize,
                                     std::size_t &size) const
{
     /* this function retrieve the detailed error information for this device */
    if (buffer == nullptr) {
        SIPL_LOG_ERR_STR("TPGSerializer: Provided buffer is NULL");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }
    else if (bufferSize < (sizeof(uint16_t) * static_cast<uint16_t>(TPGSERIALIZER_STATUS_MAX_ERR))) {
        //! assuming TPGSERIAL_STATUS_MAX_ERR is the size of the error group of AR0233
        //! check whether the size of the buffer provided matches the expected size
        SIPL_LOG_ERR_STR("TPGSerializer: Provided buffer is incorrect size in GetErrorInfo");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }
    else {
        // Read registers and copy over info
        NvMediaStatus const mediaStatus = TPGSerializerReadErrorData(bufferSize, buffer);
        if (mediaStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("TPGSerializer: failed with NvMedia error: %d\n", mediaStatus);
        }
        size = (sizeof(uint16_t)* static_cast<uint32_t>(TPGSERIALIZER_STATUS_MAX_ERR));
        return ConvertNvMediaStatus(mediaStatus);
    }
}

} // end of nvsipl
