/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "cameramodule/MAX96712cameramodule/CNvMMAX96712_96759CameraModule.hpp"
#include "cameramodule/MAX96712cameramodule/CNvMTransportLink_Max96712_96759.hpp"
#include "devices/MAX96759SerializerDriver/CNvMMax96759.hpp"

namespace nvsipl
{

std::unique_ptr<CNvMSerializer> CNvMMAX96712_96759CameraModule::CreateNewSerializer() {
    return std::unique_ptr<CNvMSerializer>(new CNvMMax96759());
}

std::unique_ptr<CNvMTransportLink> CNvMMAX96712_96759CameraModule::CreateNewTransportLink() {
    return std::unique_ptr<CNvMTransportLink>(new CNvMTransportLink_Max96712_96759());
}

CNvMDeserializer::LinkMode CNvMMAX96712_96759CameraModule::GetLinkMode() {
    return CNvMDeserializer::LinkMode::LINK_MODE_GMSL2_6GBPS;
}

} // end of namespace
