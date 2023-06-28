/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "CImageManager.hpp"

// Number of images (buffers) to be allocated and registered for the capture output
static constexpr size_t CAPTURE_IMAGE_POOL_SIZE {6U};
// Number of images (buffers) to be allocated and registered for the ISP0 and ISP1 outputs
static constexpr size_t ISP_IMAGE_POOL_SIZE {4U};

SIPLStatus CImageManager::AllocateBuffers(ImagePool &imagePool)
{
    NvSciError err = NvSciError_Success;
    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> reconciledAttrList;
    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> conflictAttrList;

    reconciledAttrList.reset(new NvSciBufAttrList());
    conflictAttrList.reset(new NvSciBufAttrList());

    err = NvSciBufAttrListReconcile(imagePool.attrList.get(),
                                    1U,
                                    reconciledAttrList.get(),
                                    conflictAttrList.get());
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListReconcile");

    imagePool.sciBufObjs.resize(imagePool.size);
    for (size_t i = 0U; i < imagePool.size; i++) {
        err = NvSciBufObjAlloc(*reconciledAttrList, &(imagePool.sciBufObjs[i]));
        CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufObjAlloc");
        CHK_PTR_AND_RETURN(imagePool.sciBufObjs[i], "NvSciBufObjAlloc");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CImageManager::Allocate(uint32_t sensorId)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    for (uint32_t i = 0U; i < MAX_NUM_IMAGE_OUTPUTS; i++) {
        if (m_imagePools[sensorId][i].enable) {
            m_imagePools[sensorId][i].attrList.reset(new NvSciBufAttrList());
            NvSciError err = NvSciBufAttrListCreate(m_sciBufModule, m_imagePools[sensorId][i].attrList.get());
            CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListCreate");

            NvSciBufType bufType = NvSciBufType_Image;
            NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_Readonly;
            bool isCpuAcccessReq = false;
            bool isCpuCacheEnabled = false;
            NvSciBufAttrKeyValuePair attrKvp[] = {
                { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
                { NvSciBufGeneralAttrKey_RequiredPerm, &accessPerm, sizeof(accessPerm) },
                { NvSciBufGeneralAttrKey_NeedCpuAccess, &isCpuAcccessReq, sizeof(isCpuAcccessReq) },
                { NvSciBufGeneralAttrKey_EnableCpuCache, &isCpuCacheEnabled, sizeof(isCpuCacheEnabled) }
            };
            size_t uNumAttrs = (m_imagePools[sensorId][i].outputType
                == INvSIPLClient::ConsumerDesc::OutputType::ICP) ? 2U : 4U;
            err = NvSciBufAttrListSetAttrs(*m_imagePools[sensorId][i].attrList, attrKvp, uNumAttrs);
            CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListSetAttrs");

            status = m_siplCamera->GetImageAttributes(sensorId,
                                                      m_imagePools[sensorId][i].outputType,
                                                      *m_imagePools[sensorId][i].attrList);
            CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::GetImageAttributes()");
            switch (m_imagePools[sensorId][i].outputType) {
                case INvSIPLClient::ConsumerDesc::OutputType::ICP:
                case INvSIPLClient::ConsumerDesc::OutputType::ISP0:
                case INvSIPLClient::ConsumerDesc::OutputType::ISP1:
                case INvSIPLClient::ConsumerDesc::OutputType::ISP2:
                    status = AllocateBuffers(m_imagePools[sensorId][i]);
                    CHK_STATUS_AND_RETURN(status, "CImageManager::AllocateBuffers()");
                    break;
                default:
                    LOG_ERR("Unexpected output type\n");
                    return NVSIPL_STATUS_ERROR;
            }
        }
    }

    return status;
}

SIPLStatus CImageManager::Register(uint32_t sensorId)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    for (uint32_t i = 0U; i < MAX_NUM_IMAGE_OUTPUTS; i++) {
        if (m_imagePools[sensorId][i].enable) {
            switch (m_imagePools[sensorId][i].outputType) {
                case INvSIPLClient::ConsumerDesc::OutputType::ICP:
                case INvSIPLClient::ConsumerDesc::OutputType::ISP0:
                case INvSIPLClient::ConsumerDesc::OutputType::ISP1:
                case INvSIPLClient::ConsumerDesc::OutputType::ISP2:
                    status = m_siplCamera->RegisterImages(sensorId,
                                                          m_imagePools[sensorId][i].outputType,
                                                          m_imagePools[sensorId][i].sciBufObjs);
                    CHK_STATUS_AND_RETURN(status, "INvSIPLCamera::RegisterImages()");
                    break;
                default:
                    LOG_ERR("Unexpected output type\n");
                    return NVSIPL_STATUS_ERROR;
            }
        }
    }

    return status;
}

SIPLStatus CImageManager::Init(INvSIPLCamera *siplCamera,
                               const NvSIPLPipelineConfiguration &pipelineCfg,
                               NvSciBufModule& sciBufModule)
{
    m_siplCamera = siplCamera;
    m_pipelineCfg = pipelineCfg;
    m_sciBufModule = sciBufModule;

    for (auto uSensorId = 0U; uSensorId < MAX_NUM_SENSORS; uSensorId++) {
        m_imagePools[uSensorId][0].enable = pipelineCfg.captureOutputRequested;
        m_imagePools[uSensorId][0].outputType = INvSIPLClient::ConsumerDesc::OutputType::ICP;
        m_imagePools[uSensorId][0].size = CAPTURE_IMAGE_POOL_SIZE;
        m_imagePools[uSensorId][1].enable = pipelineCfg.isp0OutputRequested;
        m_imagePools[uSensorId][1].outputType = INvSIPLClient::ConsumerDesc::OutputType::ISP0;
        m_imagePools[uSensorId][1].size = ISP_IMAGE_POOL_SIZE;
        m_imagePools[uSensorId][2].enable = pipelineCfg.isp1OutputRequested;
        m_imagePools[uSensorId][2].outputType = INvSIPLClient::ConsumerDesc::OutputType::ISP1;
        m_imagePools[uSensorId][2].size = ISP_IMAGE_POOL_SIZE;
        m_imagePools[uSensorId][3].enable = pipelineCfg.isp2OutputRequested;
        m_imagePools[uSensorId][3].outputType = INvSIPLClient::ConsumerDesc::OutputType::ISP2;
        m_imagePools[uSensorId][3].size = ISP_IMAGE_POOL_SIZE;
    }

    return NVSIPL_STATUS_OK;
}

void CImageManager::Deinit()
{
    for (auto uSensorId = 0U; uSensorId < MAX_NUM_SENSORS; uSensorId++) {
        for (uint32_t i = 0U; i < MAX_NUM_IMAGE_OUTPUTS; i++) {
            if (m_imagePools[uSensorId][i].enable) {
                for (uint32_t j = 0U; j < m_imagePools[uSensorId][i].sciBufObjs.size(); j++) {
                    if (m_imagePools[uSensorId][i].sciBufObjs[j] == nullptr) {
                        LOG_WARN("Attempt to free null NvSciBufObj\n");
                        continue;
                    }
                    NvSciBufObjFree(m_imagePools[uSensorId][i].sciBufObjs[j]);
                }
                // Swap sciBufObjs vector with an equivalent empty vector to force deallocation
                std::vector<NvSciBufObj>().swap(m_imagePools[uSensorId][i].sciBufObjs);
            }
        }
    }
}

SIPLStatus CImageManager::GetBuffers(uint32_t uSensorId, INvSIPLClient::ConsumerDesc::OutputType outputType, std::vector<NvSciBufObj> &buffers)
{
    if (m_imagePools[uSensorId][(uint32_t)outputType].enable) {
        buffers = m_imagePools[uSensorId][(uint32_t)outputType].sciBufObjs;
        return NVSIPL_STATUS_OK;
    } else {
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }
}

CImageManager::~CImageManager()
{
    Deinit();
}
