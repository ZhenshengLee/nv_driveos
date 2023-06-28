/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _CNVMDS90UB9724_971CAMERAMODULE_HPP_
#define _CNVMDS90UB9724_971CAMERAMODULE_HPP_

#include "cameramodule/DS90UB9724cameramodule/CNvMDS90UB9724CameraModule.hpp"

namespace nvsipl
{

class CNvMDS90UB9724_971CameraModule : public CNvMDS90UB9724CameraModule
{
    std::unique_ptr<CNvMSerializer> CreateNewSerializer() override final;

    std::unique_ptr<CNvMTransportLink> CreateNewTransportLink() override final;

    CNvMDeserializer::LinkMode GetLinkMode() override;
};


} // end of namespace

#endif /* _CNVMDS90UB9724_971CAMERAMODULE_HPP_ */
