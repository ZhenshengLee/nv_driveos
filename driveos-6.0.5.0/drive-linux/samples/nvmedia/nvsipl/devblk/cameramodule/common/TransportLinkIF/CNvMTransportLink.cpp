/* * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMTransportLink.hpp"

namespace nvsipl
{

SIPLStatus CNvMTransportLink::SetConfig(LinkParams const &params) {
    SIPLStatus status = NVSIPL_STATUS_OK;

    if ((params.pSerCDIDevice == nullptr) or (params.pDeserCDIDevice == nullptr)) {
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    } else {
        m_oLinkParams = params;
    }
    return status;
}

} /* end of namespace nvsipl */