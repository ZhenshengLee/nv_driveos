/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CIMAGEMANAGER_HPP
#define CIMAGEMANAGER_HPP

// Standard header files
#include <memory>

// SIPL header files
#include "NvSIPLCommon.hpp"
#include "NvSIPLCamera.hpp"

// Sample application header files
#include "CUtils.hpp"

// Other NVIDIA header files
#include "nvscibuf.h"
#include "nvmedia_core.h"

using namespace nvsipl;

#define MAX_NUM_IMAGE_OUTPUTS (4U)

class CImageManager
{
public:
    //! Initialize: allocates image groups and images and registers them with SIPL.
    SIPLStatus Init(INvSIPLCamera *siplCamera,
                    const NvSIPLPipelineConfiguration &pipelineCfg,
                    NvSciBufModule& sciBufModule,
                    bool disp);

    //! Deinitialize: deallocates image groups and images.
    void Deinit();

    //! Destructor: calls Deinit.
    ~CImageManager();

    SIPLStatus Allocate(uint32_t sensorId);
    SIPLStatus Register(uint32_t sensorId);
    SIPLStatus GetBuffers(uint32_t uSensorId, INvSIPLClient::ConsumerDesc::OutputType outputType, std::vector<NvSciBufObj> &buffers);

private:
    typedef struct {
        bool enable;
        size_t size;
        INvSIPLClient::ConsumerDesc::OutputType outputType;
        std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> attrList;
        std::vector<NvSciBufObj> sciBufObjs;
    } ImagePool;

    //! Allocates buffers to be used for either capture or processing.
    SIPLStatus AllocateBuffers(ImagePool &imagePool);

    bool m_disp;
    INvSIPLCamera *m_siplCamera = nullptr;
    NvSIPLPipelineConfiguration m_pipelineCfg;
    NvSciBufModule m_sciBufModule = nullptr;
    ImagePool m_imagePools[MAX_NUM_SENSORS][MAX_NUM_IMAGE_OUTPUTS];
};

#endif // CIMAGEMANAGER_HPP
