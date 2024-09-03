/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "MAX96712cameramodule/CNvMMAX96712_96717FCameraModule.hpp"
#include "MAX96712cameramodule/CNvMTransportLink_Max96712_96717F.hpp"
#include "MAX96717FSerializerDriver/CNvMMax96717F.hpp"

namespace nvsipl
{

std::unique_ptr<CNvMSerializer> CNvMMAX96712_96717FCameraModule::CreateNewSerializer() const {
    /* coverity[misra_c_2012_rule_21.3_violation] # TID-1493 */
    return std::unique_ptr<CNvMSerializer>(new CNvMMax96717F());
}

std::unique_ptr<CNvMTransportLink> CNvMMAX96712_96717FCameraModule::CreateNewTransportLink() const {
    /* coverity[misra_c_2012_rule_21.3_violation] # TID-1493 */
    return std::unique_ptr<CNvMTransportLink>(new CNvMTransportLink_Max96712_96717F());
}

CNvMDeserializer::LinkMode CNvMMAX96712_96717FCameraModule::GetLinkMode() {
    return CNvMDeserializer::LinkMode::LINK_MODE_GMSL2_3GBPS;
}

Interface* CNvMMAX96712_96717FCameraModule::GetInterface(UUID const &interfaceId) {
    static_cast<void>(interfaceId);
    return nullptr;
}

SIPLStatus CNvMMAX96712_96717FCameraModule::GetInterruptStatus(
    uint32_t const gpioIdx,
    IInterruptNotify &intrNotifier) noexcept
{
    static_cast<void>(gpioIdx);
    static_cast<void>(intrNotifier);
    return NVSIPL_STATUS_NOT_SUPPORTED;
}


} // end of namespace
