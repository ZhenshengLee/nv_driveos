/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef SIPL_ERROR_CONVERTER_HPP
#define SIPL_ERROR_CONVERTER_HPP

#include "NvSIPLCommon.hpp"
#include "nvmedia_core.h"

namespace nvsipl
{

/**
 * @brief Convert NvMediaStatus codes to SIPLStatus return values.
 *
 * The mapping of NvMediaStatus to SIPLStatus is as follows:
 * NVMEDIA_STATUS_OK            -> NVSIPL_STATUS_OK
 * NVMEDIA_STATUS_BAD_PARAMETER -> NVSIPL_STATUS_BAD_ARGUMENT
 * NVMEDIA_STATUS_OUT_OF_MEMORY -> NVSIPL_STATUS_OUT_OF_MEMORY
 * NVMEDIA_STATUS_TIMED_OUT     -> NVSIPL_STATUS_TIMED_OUT
 * NVMEDIA_STATUS_NOT_SUPPORTED -> NVSIPL_STATUS_NOT_SUPPORTED
 * Other                        -> NVSIPL_STATUS_ERROR
 *
 * @param[in] nvmStatus The NvMediaStatus value to convert.
 *
 * @retval SIPLStatus value after conversion
 */
inline SIPLStatus ConvertNvMediaStatus(NvMediaStatus const nvmStatus) noexcept
{
    SIPLStatus status;

    switch (nvmStatus) {
    case NVMEDIA_STATUS_OK:
        {
            status = NVSIPL_STATUS_OK;
            break;
        }
    case NVMEDIA_STATUS_BAD_PARAMETER:
        {
            status = NVSIPL_STATUS_BAD_ARGUMENT;
            break;
        }
    case NVMEDIA_STATUS_OUT_OF_MEMORY:
        {
            status = NVSIPL_STATUS_OUT_OF_MEMORY;
            break;
        }
    case NVMEDIA_STATUS_TIMED_OUT:
        {
            status = NVSIPL_STATUS_TIMED_OUT;
            break;
        }
    case NVMEDIA_STATUS_NOT_SUPPORTED:
        {
            status = NVSIPL_STATUS_NOT_SUPPORTED;
            break;
        }
    default:
        {
            status = NVSIPL_STATUS_ERROR;
            break;
        }
    }
    return status;
}


} // end of namespace nvsipl

#endif // SIPL_ERROR_CONVERTER_HPP

