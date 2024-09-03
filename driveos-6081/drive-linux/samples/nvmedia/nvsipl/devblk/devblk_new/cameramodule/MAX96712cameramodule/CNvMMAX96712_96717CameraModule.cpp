/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "MAX96712cameramodule/CNvMMAX96712_96717CameraModule.hpp"
#include "MAX96712cameramodule/CNvMTransportLink_Max96712_96717F.hpp"
#include "MAX96717FSerializerDriver/CNvMMax96717F.hpp"

namespace nvsipl
{

std::unique_ptr<CNvMSerializer> CNvMMAX96712_96717CameraModule::CreateNewSerializer() const {
    /* coverity[misra_c_2012_rule_21.3_violation] # TID-1493 */
    return std::unique_ptr<CNvMSerializer>(new CNvMMax96717F());
}

std::unique_ptr<CNvMTransportLink> CNvMMAX96712_96717CameraModule::CreateNewTransportLink() const {
    /* coverity[misra_c_2012_rule_21.3_violation] # TID-1493 */
    return std::unique_ptr<CNvMTransportLink>(new CNvMTransportLink_Max96712_96717F());
}

CNvMDeserializer::LinkMode CNvMMAX96712_96717CameraModule::GetLinkMode() noexcept {
    return CNvMDeserializer::LinkMode::LINK_MODE_GMSL2_6GBPS;
}

} // end of namespace
