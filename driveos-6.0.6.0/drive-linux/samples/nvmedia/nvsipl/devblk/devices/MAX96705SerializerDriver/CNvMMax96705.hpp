/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CNvMMAX96705_HPP_
#define _CNvMMAX96705_HPP_

#include "SerializerIF/CNvMSerializer.hpp"
#include "utils/utils.hpp"

namespace nvsipl
{

/*! Serializer */
class CNvMMax96705 final : public CNvMSerializer
{
public:
    SIPLStatus SetConfig(const SerInfo *serializerInfo, DeviceParams *params) override;

    SIPLStatus GetErrorSize(size_t & errorSize) const override;

    SIPLStatus GetErrorInfo(std::uint8_t * const buffer,
                                      const std::size_t bufferSize,
                                      std::size_t &size) const override;
private:
    virtual SIPLStatus DoInit() override
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus DoStart() override
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus DoStop() override
    {
        return NVSIPL_STATUS_OK;
    };
};

} // end of namespace nvsipl
#endif //_CNvMMAX96705_HPP_
