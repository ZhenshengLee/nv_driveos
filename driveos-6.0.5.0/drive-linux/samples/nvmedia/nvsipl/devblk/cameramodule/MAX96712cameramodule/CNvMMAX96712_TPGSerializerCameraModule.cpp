/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "MAX96712cameramodule/CNvMMAX96712_TPGSerializerCameraModule.hpp"
#include "MAX96712cameramodule/CNvMTransportLink_Max96712_TPGSerializer.hpp"
#include "TPGSerializerDriver/CNvMTPGSerializer.hpp"

namespace nvsipl
{

std::unique_ptr<CNvMSerializer> CNvMMAX96712_TPGSerializerCameraModule::CreateNewSerializer() {
    return std::unique_ptr<CNvMSerializer>(new CNvMTPGSerializer());
}

std::unique_ptr<CNvMTransportLink> CNvMMAX96712_TPGSerializerCameraModule::CreateNewTransportLink() {
    return std::unique_ptr<CNvMTransportLink>(new CNvMTransportLink_Max96712_TPGSerializer());
}

CNvMDeserializer::LinkMode CNvMMAX96712_TPGSerializerCameraModule::GetLinkMode() {
    return CNvMDeserializer::LinkMode::LINK_MODE_NONE;
}

std::string CNvMMAX96712_TPGSerializerCameraModule::GetSupportedDeserailizer()
{
    return "MAX96712";
}

} // end of namespace
