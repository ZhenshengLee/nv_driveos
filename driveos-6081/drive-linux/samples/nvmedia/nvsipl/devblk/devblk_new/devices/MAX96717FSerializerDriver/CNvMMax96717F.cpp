/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */
#include "CNvMMax96717F.hpp"
#include "sipl_error.h"
#include "sipl_util.h"
#include "cdi_max96717f.h"

namespace nvsipl
{
SIPLStatus CNvMMax96717F::SetConfig(SerInfo const* const serializerInfo,
                                    DeviceParams *const params)
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    if (m_eState != DeviceState::CREATED) {
        SIPL_LOG_ERR_STR("CDI invalid state");
        status = NVSIPL_STATUS_INVALID_STATE;
    } else {
        status = CNvMSerializer::SetConfig(serializerInfo, params);
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("CNvMSerializer::SetConfig fialed",
                                      static_cast<uint32_t>(status));
        } else {
            /* Get CDI Driver */
            m_pCDIDriver = GetMAX96717FDriver();
            if (m_pCDIDriver == nullptr) {
                SIPL_LOG_ERR_STR("GetMAX96717FDriver() in CNvMMax96705::SetConfig failed");
                status = NVSIPL_STATUS_ERROR;
            } else {
                m_eState = DeviceState::CDI_DEVICE_CONFIG_SET;
            }
        }
    }

    return status;
}

SIPLStatus CNvMMax96717F::GetErrorInfo(std::uint8_t *const buffer,
                                       std::size_t const bufferSize,
                                       std::size_t &size) const
{
    NvMediaStatus mediaStatus {NVMEDIA_STATUS_OK};

    if ((m_upCDIDevice == nullptr) || (buffer == nullptr)) {
        SIPL_LOG_ERR_STR("Input parameter invalid");
        mediaStatus = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        MAX96717FErrorInfo errInfo;

        /* Read serializer register error information */
        mediaStatus = MAX96717FReadErrorData(m_upCDIDevice.get(),
                                             &errInfo);
        if (NVMEDIA_STATUS_OK != mediaStatus) {
            SIPL_LOG_ERR_STR_INT("MAX96717FReadErrorData failed",
                                 static_cast<int32_t>(mediaStatus));
        } else {
            uint8_t tmpBuf[sizeof(MAX96717FErrorInfo) + \
                                  (sizeof(SMErrInfoResult) * static_cast<size_t> (SM_LIST_MAX_NUM))] = {0U};
            size_t regSize {0U};
            uint8_t smBuf[sizeof(SMErrInfoResult) * \
                                static_cast<size_t> (SM_LIST_MAX_NUM)];

            fusa_memcpy(static_cast<void*>(&tmpBuf[0]), static_cast<void*>(&errInfo), sizeof(MAX96717FErrorInfo));

            mediaStatus = MAX96717FChkSMStatus(m_upCDIDevice.get(),
                                               &regSize,
                                               &smBuf[0U]);
            if (NVMEDIA_STATUS_OK != mediaStatus) {
                SIPL_LOG_ERR_STR("CNvMMax96717F::GetErrorInfo: Failed get Error Status");
            } else {
                size_t tmpSize = sizeof(MAX96717FErrorInfo);
                if (bufferSize < tmpSize) {
                    LOG_INFO("GetErrorInfo: tmpSize(0x%x) is outof buffer size(0x%x)\n", \
                                                                    tmpSize, bufferSize);
                    mediaStatus = NVMEDIA_STATUS_OUT_OF_MEMORY;
                } else {
                    fusa_memcpy(static_cast<void*>(&tmpBuf[tmpSize]), static_cast<void*>(&smBuf[0]), regSize);

                    if ((static_cast<size_t>(SIZE_MAX) - tmpSize) < regSize) {
                        LOG_INFO("GetErrorInfo: result is over buffer size(0x%x)\n", \
                                                            tmpSize+regSize);
                        mediaStatus = NVMEDIA_STATUS_OUT_OF_MEMORY;
                    } else {
                        tmpSize += regSize;
                        fusa_memcpy(buffer, static_cast<void*>(&tmpBuf[0U]), tmpSize);
                        size = tmpSize;
                    }
                }
            }
        }
    }

    return ConvertNvMediaStatus(mediaStatus);
}

SIPLStatus CNvMMax96717F::GetErrorSize(size_t & errorSize) const
{
    errorSize = sizeof(MAX96717FErrorInfo) + \
                (sizeof(SMErrInfoResult) * static_cast<size_t> (SM_LIST_MAX_NUM));
    return NVSIPL_STATUS_OK;
}

} /* end of nvsipl */
