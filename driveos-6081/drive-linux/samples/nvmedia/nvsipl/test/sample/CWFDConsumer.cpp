/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "CWFDConsumer.hpp"

static const uint32_t ImagePlaneCount = 1U;

CWFDConsumer::CNvSciBufResources::CNvSciBufResources() : m_bufModule(NULL),
                                         m_conflictList(NULL), m_reconciledList(NULL)
{
}

CWFDConsumer::CNvSciBufResources::~CNvSciBufResources()
{
    LOG_DBG("%s enter\n", __func__);

    for (uint32_t i = 0U; i < RECON_NUM; i++) {
        if (m_unreconciledList[i]) {
            NvSciBufAttrListFree(m_unreconciledList[i]);
        }
    }

    if (m_reconciledList) {
        NvSciBufAttrListFree(m_reconciledList);
    }

    if (m_conflictList) {
        NvSciBufAttrListFree(m_conflictList);
    }

    LOG_DBG("%s exit\n", __func__);
}

SIPLStatus CWFDConsumer::CNvSciBufResources::InitBufRes(NvSciBufModule& bufModule,
                                                        std::vector<NvSciBufObj> &vBufObj,
                                                        std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> attrList,
                                                        uint32_t uWidth,
                                                        uint32_t uHeight)
{
    // Default buffer attributes
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
    NvSciBufAttrValColorFmt bufColorFormat = NvSciColor_A8B8G8R8;
    NvSciBufAttrValColorStd bufColorStd = NvSciColorStd_SRGB;
    NvSciBufAttrValImageScanType bufScanType = NvSciBufScan_ProgressiveType;
    NvSciBufAttrValAccessPerm bufPerm = NvSciBufAccessPerm_ReadWrite;

    bool needCpuAccessFlag = false;
    NvSciBufAttrKeyValuePair bufAttrs[] = {
        {
            NvSciBufGeneralAttrKey_RequiredPerm,
            &bufPerm,
            sizeof(bufPerm)
        },
        {
            NvSciBufGeneralAttrKey_Types,
            &bufType,
            sizeof(bufType)
        },
        {
            NvSciBufGeneralAttrKey_NeedCpuAccess,
            &needCpuAccessFlag,
            sizeof(needCpuAccessFlag)
        },
        {
            NvSciBufImageAttrKey_Layout,
            &layout,
            sizeof(layout)
        },
        {
            NvSciBufImageAttrKey_PlaneCount,
            &ImagePlaneCount,
            sizeof(ImagePlaneCount)
        },
        {
            NvSciBufImageAttrKey_PlaneColorFormat,
            &bufColorFormat,
            sizeof(bufColorFormat)
        },
        {
            NvSciBufImageAttrKey_PlaneColorStd,
            &bufColorStd,
            sizeof(bufColorStd)
        },
        {
            NvSciBufImageAttrKey_PlaneWidth,
            &uWidth,
            sizeof(uWidth),
        },
        {
            NvSciBufImageAttrKey_PlaneHeight,
            &uHeight,
            sizeof(uHeight),
        },
        {
            NvSciBufImageAttrKey_ScanType,
            &bufScanType,
            sizeof(bufScanType)
        },
    };

    m_bufModule = bufModule;
    for (uint32_t i = 0U; i < RECON_NUM; i++) {
        m_unreconciledList[i] = NULL;
        auto sciErr = NvSciBufAttrListCreate(m_bufModule, &m_unreconciledList[i]);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate");
    }

    WFDErrorCode wfdErr = wfdNvSciBufSetDisplayAttributesNVX(&m_unreconciledList[0]);
    CHK_WFDSTATUS_AND_RETURN(wfdErr, "wfdNvSciBufSetDisplayAttributesNVX");

    auto sciErr = NvSciBufAttrListSetAttrs(m_unreconciledList[0], bufAttrs,
                                   sizeof(bufAttrs)/sizeof(NvSciBufAttrKeyValuePair));
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    /* Create RGBA PL image with CPU access */
    NvSciBufType bufType2d = NvSciBufType_Image;
    bool imgCpuAccess = true;
    uint32_t planeCount = 1U;
    NvSciBufAttrValColorFmt planeColorFmt = NvSciColor_A8B8G8R8;
    NvSciBufAttrValColorStd planeColorStd = NvSciColorStd_SRGB;
    NvSciBufAttrValImageLayoutType imgLayout = NvSciBufImage_PitchLinearType;
    uint64_t topPadding = 0U;
    uint64_t bottomPadding = 0U;
    uint64_t leftPadding = 0U;
    uint64_t rightPadding = 0U;
    uint32_t planeWidth = uWidth;
    uint32_t planeHeight = uHeight;
    uint32_t planeBaseAddrAlign = 256U;
    NvSciBufAttrKeyValuePair setAttrs[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType2d, sizeof(NvSciBufType) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &imgCpuAccess, sizeof(bool) },
        { NvSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(uint32_t) },
        { NvSciBufImageAttrKey_PlaneColorFormat, &planeColorFmt, sizeof(NvSciBufAttrValColorFmt) },
        { NvSciBufImageAttrKey_PlaneColorStd, &planeColorStd, sizeof(NvSciBufAttrValColorStd) },
        { NvSciBufImageAttrKey_Layout, &imgLayout, sizeof(NvSciBufAttrValImageLayoutType) },
        { NvSciBufImageAttrKey_TopPadding, &topPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_BottomPadding, &bottomPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_LeftPadding, &leftPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_RightPadding, &rightPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_PlaneWidth, &planeWidth, sizeof(uint32_t) },
        { NvSciBufImageAttrKey_PlaneHeight, &planeHeight, sizeof(uint32_t)  },
        { NvSciBufImageAttrKey_PlaneBaseAddrAlign, &planeBaseAddrAlign, sizeof(uint32_t) }
    };
    size_t length = sizeof(setAttrs) / sizeof(NvSciBufAttrKeyValuePair);
    sciErr = NvSciBufAttrListSetAttrs(m_unreconciledList[1], setAttrs, length);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    m_unreconciledList[2] = *attrList;

    sciErr = NvSciBufAttrListReconcile(m_unreconciledList,
                                       RECON_NUM,
                                       &m_reconciledList,
                                       &m_conflictList);
    // m_unreconciledList[2] will be freed automatically by attrList
    // Set to null to avoid segmentation fault on attempted double free in ~CNvSciBufResources
    m_unreconciledList[2] = NULL;
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListReconcile");

    for (auto i = 0U; i < BUFFER_NUM; i++) {
        NvSciBufObj bufObj;
        sciErr = NvSciBufObjAlloc(m_reconciledList, &bufObj);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjAlloc");
        CHK_PTR_AND_RETURN(bufObj, "NvSciBufObjAlloc");
        vBufObj.push_back(bufObj);
    }

    return NVSIPL_STATUS_OK;
}

void CWFDConsumer::GetBuffers(std::vector<NvSciBufObj>& vBuffers)
{
    vBuffers = m_vBufObj;
    return;
}

CWFDConsumer::CWFDConsumer():
            m_wfdDevice(WFD_INVALID_HANDLE),
            m_wfdPort(WFD_INVALID_HANDLE),
            m_wfdPipeline(WFD_INVALID_HANDLE),
            m_inited(false)
{
}

SIPLStatus CWFDConsumer::Init(NvSciBufModule &bufModule,
                              std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> attrList,
                              uint32_t &uWidth,
                              uint32_t &uHeight)
{
    auto status = InitWFD(uWidth, uHeight);
    CHK_STATUS_AND_RETURN(status, "InitWFD");

    status = m_nvSciBufRes.InitBufRes(bufModule, m_vBufObj, std::move(attrList), uWidth, uHeight);
    CHK_STATUS_AND_RETURN(status, "CNvSciBufResources::InitBufRes");

    status = RegisterBuffers(uWidth, uHeight);
    CHK_STATUS_AND_RETURN(status, "RegisterBuffers");

    m_inited = true;
    return NVSIPL_STATUS_OK;
}

SIPLStatus CWFDConsumer::InitWFD(uint32_t &uWidth,
                                 uint32_t &uHeight)
{
    WFDint wfdPortId = WFD_INVALID_PORT_ID;
    WFDPortMode wfdPortMode = WFD_INVALID_HANDLE;

    m_wfdDevice = wfdCreateDevice(WFD_DEFAULT_DEVICE_ID, NULL);
    if (!m_wfdDevice) {
        LOG_ERR("wfdCreateDevice failed\n");
        return NVSIPL_STATUS_RESOURCE_ERROR;
    }

    auto wfdNumPorts = wfdEnumeratePorts(m_wfdDevice, &wfdPortId, 1, NULL);
    if (!wfdNumPorts) {
        GET_WFDERROR_AND_RETURN(m_wfdDevice);
    }
    m_wfdPort = wfdCreatePort(m_wfdDevice, wfdPortId, NULL);
    if (!m_wfdPort) {
        GET_WFDERROR_AND_RETURN(m_wfdDevice);
    }
    auto wfdNumModes = wfdGetPortModes(m_wfdDevice, m_wfdPort, &wfdPortMode, 1);
    if (!wfdNumModes) {
        GET_WFDERROR_AND_RETURN(m_wfdDevice);
    }

    // Get the preferred resolution of the display panel attached to the port
    WFDint displayRes[2] = {0};
    wfdGetPortAttribiv(m_wfdDevice, m_wfdPort, WFD_PORT_NATIVE_RESOLUTION, 2, displayRes);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    if ((displayRes[0] == 0) || (displayRes[1] == 0)) {
        LOG_ERR("Failed to get the resolution of the display\n");
        return NVSIPL_STATUS_ERROR;
    }

    wfdSetPortMode(m_wfdDevice, m_wfdPort, wfdPortMode);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    wfdDeviceCommit(m_wfdDevice, WFD_COMMIT_ENTIRE_PORT, m_wfdPort);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    // Enumerate the pipelines
    WFDint wfdBindablePipeId;
    WFDint wfdNumPipelines = wfdEnumeratePipelines(m_wfdDevice, &wfdBindablePipeId, 1, nullptr);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);
    if (wfdNumPipelines <= 0) {
        LOG_ERR("No pipeline was found\n");
        return NVSIPL_STATUS_ERROR;
    }

    m_wfdPipeline = wfdCreatePipeline(m_wfdDevice, wfdBindablePipeId, NULL);
    if (!m_wfdPipeline) {
        GET_WFDERROR_AND_RETURN(m_wfdDevice);
    }
    wfdBindPipelineToPort(m_wfdDevice, m_wfdPort, m_wfdPipeline);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    LOG_DBG("%s: pipeline is created and bound successfully\n", __func__);

    wfdDeviceCommit(m_wfdDevice, WFD_COMMIT_ENTIRE_PORT, m_wfdPort);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);
    LOG_DBG("%s: wfdBindPipelineToPort success\n", __func__);

    uWidth = displayRes[0];
    uHeight = displayRes[1];

    return NVSIPL_STATUS_OK;
}

SIPLStatus CWFDConsumer::RegisterBuffers(uint32_t uWidth,
                                         uint32_t uHeight)
{
    m_vBufSourcePair.clear();

    WFDint wfdRect[4] {0, 0, (WFDint)uWidth, (WFDint)uHeight};

    for (NvSciBufObj& bufObj : m_vBufObj) {
        WFDSource wfdSource = wfdCreateSourceFromNvSciBufNVX(m_wfdDevice, m_wfdPipeline, &bufObj);
        if (!wfdSource) {
            GET_WFDERROR_AND_RETURN(m_wfdDevice);
        }
        LOG_DBG("%s: Create WFD source success\n", __func__);

        wfdBindSourceToPipeline(m_wfdDevice, m_wfdPipeline, wfdSource, WFD_TRANSITION_AT_VSYNC, NULL);
        GET_WFDERROR_AND_RETURN(m_wfdDevice);

        wfdSetPipelineAttribiv(m_wfdDevice, m_wfdPipeline, WFD_PIPELINE_SOURCE_RECTANGLE, 4, wfdRect);
        GET_WFDERROR_AND_RETURN(m_wfdDevice);

        wfdSetPipelineAttribiv(m_wfdDevice, m_wfdPipeline, WFD_PIPELINE_DESTINATION_RECTANGLE, 4, wfdRect);
        GET_WFDERROR_AND_RETURN(m_wfdDevice);

        m_vBufSourcePair.push_back(std::make_pair(bufObj, wfdSource));
    }

    return NVSIPL_STATUS_OK;
}

CWFDConsumer::~CWFDConsumer()
{
    LOG_DBG("%s enter\n", __func__);

    m_inited = false;

    NullFlip();

    for (auto &bufSourcePair : m_vBufSourcePair) {
        if (bufSourcePair.second != WFD_INVALID_HANDLE) {
            wfdDestroySource(m_wfdDevice, bufSourcePair.second);
        }

        if (bufSourcePair.first) {
            NvSciBufObjFree(bufSourcePair.first);
        }
    }

    if (m_wfdPipeline) {
        wfdDestroyPipeline(m_wfdDevice, m_wfdPipeline);
    }

    if (m_wfdPort) {
        wfdDestroyPort(m_wfdDevice, m_wfdPort);
    }

    if (m_wfdDevice) {
        wfdDestroyDevice(m_wfdDevice);
    }

    LOG_DBG("%s exit\n", __func__);
}

SIPLStatus CWFDConsumer::NullFlip(void)
{
    if (m_wfdDevice && m_wfdPipeline) {
        // Perform a null flip
        wfdBindSourceToPipeline(m_wfdDevice, m_wfdPipeline, (WFDSource)0, WFD_TRANSITION_AT_VSYNC, NULL);
        GET_WFDERROR_AND_RETURN(m_wfdDevice);

        wfdDeviceCommit(m_wfdDevice, WFD_COMMIT_PIPELINE, m_wfdPipeline);
        GET_WFDERROR_AND_RETURN(m_wfdDevice);
    }
    return NVSIPL_STATUS_OK;
}

SIPLStatus CWFDConsumer::WfdFlip(NvSciBufObj& bufObj)
{
    if (!m_inited) {
        LOG_ERR("Post before init completed.\n");
        return NVSIPL_STATUS_INVALID_STATE;
    }

    for (auto &bufSourcePair : m_vBufSourcePair) {
        if (bufSourcePair.first == bufObj) {
            wfdBindSourceToPipeline(m_wfdDevice, m_wfdPipeline, bufSourcePair.second, WFD_TRANSITION_AT_VSYNC, NULL);
            GET_WFDERROR_AND_RETURN(m_wfdDevice);

            wfdDeviceCommit(m_wfdDevice, WFD_COMMIT_PIPELINE, m_wfdPipeline);
            GET_WFDERROR_AND_RETURN(m_wfdDevice);
        }
    }

    return NVSIPL_STATUS_OK;
}
