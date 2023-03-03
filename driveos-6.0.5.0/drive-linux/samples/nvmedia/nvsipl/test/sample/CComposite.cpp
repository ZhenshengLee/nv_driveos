/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <iostream>
#include <chrono>
#include <pthread.h>

#include "CComposite.hpp"

SIPLStatus CComposite::Init(NvSciBufModule& bufModule, CImageManager* pImageManager, std::vector<uint32_t>& vSensorIds)
{
    m_bRunning = false;

    m_vSensorIds = vSensorIds;
    m_pImageManager = pImageManager;
    NvMedia2D *p2dHandle = nullptr;
    NvMediaStatus nvmStatus = NvMedia2DCreate(&p2dHandle, nullptr);
    m_up2DDevice.reset(p2dHandle);
    CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DCreate");
    CHK_PTR_AND_RETURN(m_up2DDevice, "NvMedia2DCreate");

    m_upWfdConsumer.reset(new CWFDConsumer());
    CHK_PTR_AND_RETURN(m_upWfdConsumer, "WFD consumer creation");

    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> bufAttrList2d;
    bufAttrList2d.reset(new NvSciBufAttrList());
    NvSciError sciErr = NvSciBufAttrListCreate(bufModule, bufAttrList2d.get());
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate");

    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrKeyValuePair attrKvp = {
        NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType),
    };
    sciErr = NvSciBufAttrListSetAttrs(*bufAttrList2d, &attrKvp, 1U);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    nvmStatus = NvMedia2DFillNvSciBufAttrList(m_up2DDevice.get(), *bufAttrList2d);
    CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DFillNvSciBufAttrList");

    SIPLStatus status = m_upWfdConsumer->Init(bufModule, std::move(bufAttrList2d));
    CHK_STATUS_AND_RETURN(status, "CWFDConsumer::Init()");

    m_pDstBuffer = m_upWfdConsumer->GetBuffer();

    status = RegisterImages();
    CHK_STATUS_AND_RETURN(status, "RegisterImages");

    status = ComputeInputRects();
    CHK_STATUS_AND_RETURN(status, "ComputeInputRects");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::FillNvSciSyncAttrList(NvSciSyncAttrList &attrList, NvMediaNvSciSyncClientType clientType)
{
    NvMediaStatus nvmStatus = NvMedia2DFillNvSciSyncAttrList(m_up2DDevice.get(), attrList, clientType);
    CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DFillNvSciSyncAttrList");
    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::RegisterNvSciSyncObj(uint32_t uSensorId,
                                            NvMediaNvSciSyncObjType syncObjType,
                                            std::unique_ptr<NvSciSyncObj, CloseNvSciSyncObj> syncObj)
{
    NvMediaStatus nvmStatus = NvMedia2DRegisterNvSciSyncObj(m_up2DDevice.get(), syncObjType, *syncObj);
    CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DRegisterNvSciSyncObj");
    if (syncObjType == NVMEDIA_EOFSYNCOBJ) {
        m_syncObjs[uSensorId] = std::move(syncObj);
    }
    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::UnregisterNvSciSyncObjs(void)
{
    SIPLStatus ret = NVSIPL_STATUS_OK;
    for (uint32_t sensorId : m_vSensorIds) {
        NvMediaStatus nvmStatus = NvMedia2DUnregisterNvSciSyncObj(m_up2DDevice.get(),
                                                                  *m_syncObjs[sensorId]);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            LOG_ERR("NvMedia2DUnregisterNvSciSyncObj failed, status: %u\n", nvmStatus);
            ret = NVSIPL_STATUS_ERROR;
        }
    }
    return ret;
}

SIPLStatus CComposite::RegisterImages(void)
{
    // Register source images with 2D
    for (uint32_t sensorId : m_vSensorIds) {
        std::vector<NvSciBufObj> vSrcBufs;
        SIPLStatus status = m_pImageManager->GetBuffers(sensorId, INvSIPLClient::ConsumerDesc::OutputType::ISP0, vSrcBufs);
        CHK_STATUS_AND_RETURN(status, "CImageManager::GetBuffers");
        for (NvSciBufObj &srcBuf: vSrcBufs) {
            if (srcBuf) {
                NvMediaStatus nvmStatus = NvMedia2DRegisterNvSciBufObj(m_up2DDevice.get(), srcBuf);
                CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DRegisterNvSciBufObj");
            }
        }
    }
    // Register destination image with 2D
    if (m_pDstBuffer) {
        NvMediaStatus nvmStatus = NvMedia2DRegisterNvSciBufObj(m_up2DDevice.get(), m_pDstBuffer);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DRegisterNvSciBufObj");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::UnregisterImages(void)
{
    // Unregister source images with 2D
    for (uint32_t sensorId : m_vSensorIds) {
        std::vector<NvSciBufObj> vSrcBufs;
        SIPLStatus status = m_pImageManager->GetBuffers(sensorId, INvSIPLClient::ConsumerDesc::OutputType::ISP0, vSrcBufs);
        CHK_STATUS_AND_RETURN(status, "CImageManager::GetBuffers");
        for (NvSciBufObj &srcBuf:  vSrcBufs) {
            if (srcBuf) {
                NvMediaStatus nvmStatus = NvMedia2DUnregisterNvSciBufObj(m_up2DDevice.get(), srcBuf);
                CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DUnregisterNvSciBufObj");
            }
        }
    }
    // Unregister destination image with 2D
    if (m_pDstBuffer) {
        NvMediaStatus nvmStatus = NvMedia2DUnregisterNvSciBufObj(m_up2DDevice.get(), m_pDstBuffer);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DUnregisterNvSciBufObj");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::ComputeInputRects(void)
{
    // Set up the destination rectangles
    uint16_t xStep = DisplayImageWidth;
    uint16_t yStep = DisplayImageHeight;
    uint16_t countPerLine = 1U;
    uint32_t cameraCount = m_vSensorIds.size();
    if (cameraCount > 1U) {
        if (cameraCount <= 4U) {
           countPerLine = 2U;
        } else {
            countPerLine = 4U;
        }
    }
    xStep = DisplayImageWidth / countPerLine;
    yStep = DisplayImageHeight / countPerLine;
    for (auto i = 0U; i < cameraCount; i++) {
        auto rowIndex = i / countPerLine;
        auto colIndex = i % countPerLine;
        uint16_t startx = colIndex * xStep;
        uint16_t starty = rowIndex * yStep;
        uint16_t endx = startx + xStep;
        uint16_t endy = starty + yStep;
        m_oInputRects[i] = { startx, starty, endx, endy };
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::Post(uint32_t sensorId, INvSIPLClient::INvSIPLNvMBuffer *pBuffer)
{
    LOG_DBG("CComposite::Post\n");

    if (!m_bRunning) {
        // Composite is not ready to accept buffers
        LOG_WARN("Compositor is not ready to accept buffers\n");
        return NVSIPL_STATUS_OK;
    }
    if (sensorId >= MAX_NUM_SENSORS) {
        LOG_ERR("%s: sensorId: %u is invalid!\n", __func__, sensorId);
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    // Add buffer to queue
    pBuffer->AddRef();
    std::unique_lock<std::mutex> lck(m_vInputQueueMutex[sensorId]);
    m_vInputQueue[sensorId].push(pBuffer);

    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::Start()
{
    // Start the composite thread
    LOG_INFO("Starting compositor thread\n");
    m_upthread.reset(new std::thread(&CComposite::ThreadFunc, this));
    if (m_upthread == nullptr) {
        LOG_ERR("Failed to create compositor thread\n");
        return NVSIPL_STATUS_OUT_OF_MEMORY;
    }
    LOG_INFO("Created compositor thread: ID:%u\n", m_upthread->get_id());

    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::Stop()
{
    // Signal thread to stop
    m_bRunning = false;

    // Wait for the thread
    if (m_upthread != nullptr) {
        LOG_INFO("Waiting to join compositor thread: ID:%u\n", m_upthread->get_id());
        m_upthread->join();
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CComposite::Deinit()
{
    // Dequeue and release all input buffers
    for (auto uSensorId : m_vSensorIds) {
        std::unique_lock<std::mutex> lck(m_vInputQueueMutex[uSensorId]);
        while(!m_vInputQueue[uSensorId].empty()) {
            auto pBuffer = m_vInputQueue[uSensorId].front();
            pBuffer->Release();
            m_vInputQueue[uSensorId].pop();
        }
    }
    SIPLStatus ret = NVSIPL_STATUS_OK;
    SIPLStatus status = UnregisterNvSciSyncObjs();
    if (status != NVSIPL_STATUS_OK) {
        LOG_ERR("UnregisterNvSciSyncObjs failed, status: %u\n", status);
        ret = status;
    }
    status = UnregisterImages();
    if (status != NVSIPL_STATUS_OK) {
        LOG_ERR("UnregisterImages failed, status: %u\n", status);
        ret = status;
    }

    return ret;
}

bool CComposite::CheckInputQueues(void)
{
    bool anyReady = false;
    for (auto uSensorId : m_vSensorIds) {
        std::unique_lock<std::mutex> lck(m_vInputQueueMutex[uSensorId]);
        if (!m_vInputQueue[uSensorId].empty()) {
            anyReady = true;
            break;
        }
    }
    return anyReady;
}

struct ReleaseNvMBuffer {
    void operator()(INvSIPLClient::INvSIPLNvMBuffer *pNvMBuffer) const {
        if (pNvMBuffer != nullptr) {
            SIPLStatus status = pNvMBuffer->Release();
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("Buffer release failed\n");
            }
        }
    }
};

void CComposite::ThreadFunc(void)
{
    pthread_setname_np(pthread_self(), "CComposite");

    m_bRunning = true;

    uint32_t sleepTimeMs = uint32_t(1000 / (60.0f)); // assume refresh rate is 60 fps

    while (m_bRunning) {
        // Check for input readiness
        bool anyReady = CheckInputQueues();
        if (!anyReady) {
            LOG_INFO("Compositor does not have any inputs available yet\n");

            // Sleep for refresh rate
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepTimeMs));
            continue;
        }
        // Get full buffer from input queue and composite on to output image
        for (auto uSensorId : m_vSensorIds) {
            std::unique_ptr<INvSIPLClient::INvSIPLNvMBuffer, ReleaseNvMBuffer> upNvMBuffer;
            {
                std::unique_lock<std::mutex> lck(m_vInputQueueMutex[uSensorId]);
                if (m_vInputQueue[uSensorId].empty()) {
                    LOG_INFO("Compositor: Input queue for source %u is empty\n", uSensorId);
                    continue;
                }
                upNvMBuffer.reset(m_vInputQueue[uSensorId].front());
                m_vInputQueue[uSensorId].pop();
            }

            LOG_INFO("Compositor: Input queue for source %u is ready\n", uSensorId);

            /* Acquire an empty NvMedia 2D parameters object */
            NvMedia2DComposeParameters params;
            NvMediaStatus nvmStatus = NvMedia2DGetComposeParameters(m_up2DDevice.get(), &params);
            CHK_NVMSTATUS_AND_EXIT(nvmStatus, "NvMedia2DGetComposeParameters");

            /* Set the NvSciSyncObj to use for end-of-frame synchronization */
            nvmStatus = NvMedia2DSetNvSciSyncObjforEOF(m_up2DDevice.get(), params, *m_syncObjs[uSensorId]);
            CHK_NVMSTATUS_AND_EXIT(nvmStatus, "NvMedia2DSetNvSciSyncObjforEOF");

            /* Set the source layer parameters for layer zero */
            nvmStatus = NvMedia2DSetSrcNvSciBufObj(m_up2DDevice.get(), params, 0U, upNvMBuffer->GetNvSciBufImage());
            CHK_NVMSTATUS_AND_EXIT(nvmStatus, "NvMedia2DSetSrcNvSciBufObj");

            nvmStatus = NvMedia2DSetSrcGeometry(m_up2DDevice.get(),
                                                params,
                                                0U,
                                                NULL,
                                                &m_oInputRects[uSensorId],
                                                NVMEDIA_2D_TRANSFORM_NONE);
            CHK_NVMSTATUS_AND_EXIT(nvmStatus, "NvMedia2DSetSrcGeometry");

            /* Set the destination surface */
            nvmStatus = NvMedia2DSetDstNvSciBufObj(m_up2DDevice.get(), params, m_pDstBuffer);
            CHK_NVMSTATUS_AND_EXIT(nvmStatus, "NvMedia2DSetDstNvSciBufObj");

            /* Get the post-fence from the SIPL buffer */
            NvSciSyncFence fenceSipl = NvSciSyncFenceInitializer;
            SIPLStatus status = upNvMBuffer->GetEOFNvSciSyncFence(&fenceSipl);

            /* Set the pre-fence for the compose operation */
            nvmStatus = NvMedia2DInsertPreNvSciSyncFence(m_up2DDevice.get(), params, &fenceSipl);
            CHK_NVMSTATUS_AND_EXIT(nvmStatus, "NvMedia2DInsertPreNvSciSyncFence");
            NvSciSyncFenceClear(&fenceSipl);

            /* Submit the compose operation */
            NvMedia2DComposeResult composeResult;
            nvmStatus = NvMedia2DCompose(m_up2DDevice.get(), params, &composeResult);
            CHK_NVMSTATUS_AND_EXIT(nvmStatus, "NvMedia2DCompose");

            /* Get the end-of-frame fence for the compose operation */
            NvSciSyncFence fence2d = NvSciSyncFenceInitializer;
            nvmStatus = NvMedia2DGetEOFNvSciSyncFence(m_up2DDevice.get(), &composeResult, &fence2d);
            CHK_NVMSTATUS_AND_EXIT(nvmStatus, "NvMedia2DGetEOFNvSciSyncFence");

            /* Add end-of-frame fence as pre-fence to SIPL buffer */
            status = upNvMBuffer->AddNvSciSyncPrefence(fence2d);
            CHK_STATUS_AND_EXIT(status, "AddNvSciSyncPrefence");
            NvSciSyncFenceClear(&fence2d);

            /* Automatically release buffer in upNvMBuffer destructor */
        }

        // Send composited output image to display
        auto status = m_upWfdConsumer->WfdFlip();
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("Compositor: CWFDConsumer::WfdFlip() failed!. status: %u\n", status);
            return;
        }
    } // while ()

    return;
}

