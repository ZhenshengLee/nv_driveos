/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
                                                        NvSciBufObj &bufObj,
                                                        std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> attrList)
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
            &DisplayImageWidth,
            sizeof(DisplayImageWidth),
        },
        {
            NvSciBufImageAttrKey_PlaneHeight,
            &DisplayImageHeight,
            sizeof(DisplayImageHeight),
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
    uint32_t planeWidth = DisplayImageWidth;
    uint32_t planeHeight = DisplayImageHeight;
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

    sciErr = NvSciBufObjAlloc(m_reconciledList, &bufObj);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjAlloc");

    return NVSIPL_STATUS_OK;
}

NvSciBufObj & CWFDConsumer::GetBuffer(void)
{
    return m_bufObj;
}

CWFDConsumer::CWFDConsumer():
            m_wfdDevice(WFD_INVALID_HANDLE),
            m_wfdPort(WFD_INVALID_HANDLE),
            m_wfdPipeline(WFD_INVALID_HANDLE),
            m_wfdSource(WFD_INVALID_HANDLE),
            m_inited(false)
{
}

SIPLStatus CWFDConsumer::Init(NvSciBufModule &bufModule,
                              std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> attrList)
{
    auto status = m_nvSciBufRes.InitBufRes(bufModule, m_bufObj, std::move(attrList));
    CHK_STATUS_AND_RETURN(status, "CNvSciBufResources::InitBufRes");

    status = InitWFD();
    CHK_STATUS_AND_RETURN(status, "InitWFD");

    m_inited = true;
    return NVSIPL_STATUS_OK;
}

SIPLStatus CWFDConsumer::InitWFD()
{
    WFDint wfdPortId = WFD_INVALID_PORT_ID;
    WFDPortMode wfdPortMode = WFD_INVALID_HANDLE;
    WFDint wfdSrcRect[4] {0, 0, DisplayImageWidth, DisplayImageHeight};

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

    wfdSetPortMode(m_wfdDevice, m_wfdPort, wfdPortMode);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    wfdDeviceCommit(m_wfdDevice, WFD_COMMIT_ENTIRE_PORT, m_wfdPort);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    //Get the number of bindable pipeline IDs for a port
    auto wfdNumPipelines = wfdGetPortAttribi(m_wfdDevice, m_wfdPort, WFD_PORT_PIPELINE_ID_COUNT);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    //Populate pipeline IDs into m_wfdPipelines
    std::vector<WFDint> wfdBindablePipeIds(wfdNumPipelines);
    wfdGetPortAttribiv(m_wfdDevice, m_wfdPort, WFD_PORT_BINDABLE_PIPELINE_IDS, wfdNumPipelines, wfdBindablePipeIds.data());
    GET_WFDERROR_AND_RETURN(m_wfdDevice);
    if (wfdNumPipelines <= 0) {
        LOG_ERR("InitWFD, no pipeline is found.");
        return NVSIPL_STATUS_ERROR;
    }

    m_wfdPipeline = wfdCreatePipeline(m_wfdDevice, wfdBindablePipeIds[0], NULL);
    if (!m_wfdPipeline) {
        GET_WFDERROR_AND_RETURN(m_wfdDevice);
    }
    wfdBindPipelineToPort(m_wfdDevice, m_wfdPort, m_wfdPipeline);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    LOG_DBG("%s: pipeline is created and bound successfully\n", __func__);

    wfdDeviceCommit(m_wfdDevice, WFD_COMMIT_ENTIRE_PORT, m_wfdPort);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);
    LOG_DBG("%s: wfdBindPipelineToPort success\n", __func__);

    m_wfdSource = wfdCreateSourceFromNvSciBufNVX(m_wfdDevice, m_wfdPipeline, &m_bufObj);
    if (!m_wfdSource) {
        GET_WFDERROR_AND_RETURN(m_wfdDevice);
    }
    LOG_DBG("%s: Create wfd source success\n", __func__);

    wfdBindSourceToPipeline(m_wfdDevice, m_wfdPipeline, m_wfdSource, WFD_TRANSITION_AT_VSYNC, NULL);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    wfdSetPipelineAttribiv(m_wfdDevice, m_wfdPipeline, WFD_PIPELINE_SOURCE_RECTANGLE, 4, wfdSrcRect);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    wfdSetPipelineAttribiv(m_wfdDevice, m_wfdPipeline, WFD_PIPELINE_DESTINATION_RECTANGLE, 4, wfdSrcRect);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    return NVSIPL_STATUS_OK;
}

CWFDConsumer::~CWFDConsumer()
{
    LOG_DBG("%s enter\n", __func__);

    m_inited = false;

    if (m_bufObj) {
        NvSciBufObjFree(m_bufObj);
    }

    if (m_wfdDevice && m_wfdPipeline) {
        // Perform a null flip
        wfdBindSourceToPipeline(m_wfdDevice, m_wfdPipeline, (WFDSource)0, WFD_TRANSITION_AT_VSYNC, NULL);
        wfdDeviceCommit(m_wfdDevice, WFD_COMMIT_PIPELINE, m_wfdPipeline);
    }

    if (m_wfdSource != WFD_INVALID_HANDLE) {
        wfdDestroySource(m_wfdDevice, m_wfdSource);
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

SIPLStatus CWFDConsumer::WfdFlip(void)
{
    if (!m_inited) {
        LOG_ERR("Post before init completed.\n");
        return NVSIPL_STATUS_INVALID_STATE;
    }
    wfdBindSourceToPipeline(m_wfdDevice, m_wfdPipeline, m_wfdSource, WFD_TRANSITION_IMMEDIATE, NULL);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    wfdDeviceCommit(m_wfdDevice, WFD_COMMIT_PIPELINE, m_wfdPipeline);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    wfdBindSourceToPipeline(m_wfdDevice, m_wfdPipeline, m_wfdSource, WFD_TRANSITION_AT_VSYNC, NULL);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    return NVSIPL_STATUS_OK;
}
