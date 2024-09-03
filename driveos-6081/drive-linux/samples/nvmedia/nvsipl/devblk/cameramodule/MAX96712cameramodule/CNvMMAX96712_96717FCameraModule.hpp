/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _CNVMMAX96712_96717FCAMERAMODULE_HPP_
#define _CNVMMAX96712_96717FCAMERAMODULE_HPP_

#include "cameramodule/MAX96712cameramodule/CNvMMAX96712CameraModule.hpp"

namespace nvsipl
{

class CNvMMAX96712_96717FCameraModule : public CNvMMAX96712CameraModule
{
    std::unique_ptr<CNvMSerializer> CreateNewSerializer() override final;

    std::unique_ptr<CNvMTransportLink> CreateNewTransportLink() override final;

    CNvMDeserializer::LinkMode GetLinkMode() override;

public:
    Interface* GetInterface(const UUID &interfaceId) override;

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

} // end of namespace

#endif /* _CNVMMAX96712_96717FCAMERAMODULE_HPP_ */
