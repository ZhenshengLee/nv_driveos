/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CNvMMAX96717F_HPP
#define CNvMMAX96717F_HPP

#include "SerializerIF/CNvMSerializer.hpp"

namespace nvsipl
{

/*! Serializer */
class CNvMMax96717F final : public CNvMSerializer
{
public:
    virtual SIPLStatus SetConfig(SerInfo const *const serializerInfo,
                                 DeviceParams *const params) override;

    virtual SIPLStatus GetErrorSize(size_t & errorSize) const override;

    virtual SIPLStatus GetErrorInfo(std::uint8_t * const buffer,
                                    std::size_t const bufferSize,
                                    std::size_t &size) const override;
    virtual SIPLStatus DoInit()
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus DoStart()
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus DoStop()
    {
        return NVSIPL_STATUS_OK;
    };
};

} /* end of namespace nvsipl */
#endif /* CNvMMAX96717F_HPP */
