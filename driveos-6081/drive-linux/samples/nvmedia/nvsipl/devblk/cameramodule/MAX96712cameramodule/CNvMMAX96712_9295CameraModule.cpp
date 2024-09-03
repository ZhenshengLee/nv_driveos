/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "MAX96712cameramodule/CNvMMAX96712_9295CameraModule.hpp"
#include "MAX96712cameramodule/CNvMTransportLink_Max96712_9295.hpp"
#include "MAX9295ASerializerDriver/CNvMMax9295.hpp"

namespace nvsipl
{

std::unique_ptr<CNvMSerializer> CNvMMAX96712_9295CameraModule::CreateNewSerializer() {
    return std::unique_ptr<CNvMSerializer>(new CNvMMax9295());
}

std::unique_ptr<CNvMTransportLink> CNvMMAX96712_9295CameraModule::CreateNewTransportLink() {
    return std::unique_ptr<CNvMTransportLink>(new CNvMTransportLink_Max96712_9295());
}

CNvMDeserializer::LinkMode CNvMMAX96712_9295CameraModule::GetLinkMode() {
    return CNvMDeserializer::LinkMode::LINK_MODE_GMSL2_6GBPS;
}

Interface* CNvMMAX96712_9295CameraModule::GetInterface(const UUID &interfaceId) {
    return nullptr;
}

} // end of namespace
