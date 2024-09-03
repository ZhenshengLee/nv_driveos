/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "cameramodule/DS90UB9724cameramodule/CNvMDS90UB9724_971CameraModule.hpp"
#include "cameramodule/DS90UB9724cameramodule/CNvMTransportLink_DS90UB9724_971.hpp"
#include "devices/DS90UB971SerializerDriver/CNvMDS90UB971.hpp"

namespace nvsipl
{

std::unique_ptr<CNvMSerializer> CNvMDS90UB9724_971CameraModule::CreateNewSerializer() {
    return std::unique_ptr<CNvMSerializer>(new CNvMDS90UB971());
}

std::unique_ptr<CNvMTransportLink> CNvMDS90UB9724_971CameraModule::CreateNewTransportLink() {
    return std::unique_ptr<CNvMTransportLink>(new CNvMTransportLink_DS90UB9724_971());
}

CNvMDeserializer::LinkMode CNvMDS90UB9724_971CameraModule::GetLinkMode() {
    return CNvMDeserializer::LinkMode::LINK_MODE_FPDLINK4_SYNC;
}

} // end of namespace
