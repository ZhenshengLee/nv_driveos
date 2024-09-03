/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CNVMMAX96712_96717FCAMERAMODULE_HPP
#define CNVMMAX96712_96717FCAMERAMODULE_HPP

#include "cameramodule/MAX96712cameramodule/CNvMMAX96712CameraModule.hpp"

namespace nvsipl
{

class CNvMMAX96712_96717FCameraModule : public CNvMMAX96712CameraModule
{
private:
    virtual std::unique_ptr<CNvMSerializer> CreateNewSerializer() const override final;

    virtual std::unique_ptr<CNvMTransportLink> CreateNewTransportLink() const override final;

    virtual CNvMDeserializer::LinkMode GetLinkMode() override;

public:
    virtual Interface* GetInterface(UUID const &interfaceId) override;

    virtual SIPLStatus GetInterruptStatus(
        uint32_t const gpioIdx,
        IInterruptNotify &intrNotifier) noexcept override;
};

} /* end of namespace */

#endif /* CNVMMAX96712_96717FCAMERAMODULE_HPP */
