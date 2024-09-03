/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CNVMMAX96712_96717CAMERAMODULE_HPP
#define CNVMMAX96712_96717CAMERAMODULE_HPP

#include "cameramodule/MAX96712cameramodule/CNvMMAX96712CameraModule.hpp"

namespace nvsipl
{

class CNvMMAX96712_96717CameraModule : public CNvMMAX96712CameraModule
{
private:
    std::unique_ptr<CNvMSerializer> CreateNewSerializer() const final;

    std::unique_ptr<CNvMTransportLink> CreateNewTransportLink() const final;

    CNvMDeserializer::LinkMode GetLinkMode() noexcept override;

public:
    virtual Interface* GetInterface(const UUID &interfaceId) = 0;
};

} /* end of namespace */

#endif /* CNVMMAX96712_96717CAMERAMODULE_HPP */
