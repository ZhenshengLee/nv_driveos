/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "MAX96712cameramodule/CNvMMAX96712_96717FCameraModule.hpp"
#include "MAX96712cameramodule/CNvMTransportLink_Max96712_96717.hpp"
#include "MAX96717FSerializerDriver/CNvMMax96717F.hpp"

namespace nvsipl
{

std::unique_ptr<CNvMSerializer> CNvMMAX96712_96717FCameraModule::CreateNewSerializer() {
    return std::unique_ptr<CNvMSerializer>(new CNvMMax96717F());
}

std::unique_ptr<CNvMTransportLink> CNvMMAX96712_96717FCameraModule::CreateNewTransportLink() {
    return std::unique_ptr<CNvMTransportLink>(new CNvMTransportLink_Max96712_96717());
}

CNvMDeserializer::LinkMode CNvMMAX96712_96717FCameraModule::GetLinkMode() {
    return CNvMDeserializer::LinkMode::LINK_MODE_GMSL2_3GBPS;
}

Interface* CNvMMAX96712_96717FCameraModule::GetInterface(const UUID &interfaceId) {
    return nullptr;
}

} // end of namespace
