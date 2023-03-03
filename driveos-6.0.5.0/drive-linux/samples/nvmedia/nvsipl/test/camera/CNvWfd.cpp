/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "CNvWfd.hpp"

static const uint32_t ImagePlaneCount = 1U;

CNvWfdResources::CNvSciBufResources::CNvSciBufResources() :
    m_conflictList(NULL),
    m_reconciledList(NULL)
{
}

CNvWfdResources::CNvSciBufResources::~CNvSciBufResources()
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

SIPLStatus CNvWfdResources::CNvSciBufResources::InitBufRes(NvSciBufModule &bufModule,
                                                           NvSciBufObj &bufObj,
                                                           NvSciBufAttrList &bufAttrList,
                                                           uint32_t uWidth,
                                                           uint32_t uHeight)
{
    // Default buffer attributes
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
    NvSciBufAttrValColorFmt bufColorFormat = NvSciColor_A8B8G8R8;
    NvSciBufAttrValColorStd bufColorStd = NvSciColorStd_SRGB;
    uint32_t bufWidth = uWidth;
    uint32_t bufHeight = uHeight;
    NvSciBufAttrValImageScanType bufScanType = NvSciBufScan_ProgressiveType;
    NvSciBufAttrValAccessPerm bufPerm = NvSciBufAccessPerm_ReadWrite;
    bool needCpuAccessFlag = false;

    NvSciBufAttrKeyValuePair bufAttrs[] = {
        { NvSciBufGeneralAttrKey_RequiredPerm, &bufPerm, sizeof(bufPerm) },
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &needCpuAccessFlag, sizeof(needCpuAccessFlag) },
        { NvSciBufImageAttrKey_Layout, &layout, sizeof(layout) },
        { NvSciBufImageAttrKey_PlaneCount, &ImagePlaneCount, sizeof(ImagePlaneCount) },
        { NvSciBufImageAttrKey_PlaneColorFormat, &bufColorFormat, sizeof(bufColorFormat) },
        { NvSciBufImageAttrKey_PlaneColorStd, &bufColorStd, sizeof(bufColorStd) },
        { NvSciBufImageAttrKey_PlaneWidth, &bufWidth, sizeof(bufWidth), },
        { NvSciBufImageAttrKey_PlaneHeight, &bufHeight, sizeof(bufHeight), },
        { NvSciBufImageAttrKey_ScanType, &bufScanType, sizeof(bufScanType) }
    };

    for (uint32_t i = 0U; i < RECON_NUM; i++) {
        m_unreconciledList[i] = NULL;
        NvSciError sciErr = NvSciBufAttrListCreate(bufModule, &m_unreconciledList[i]);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate");
    }

    WFDErrorCode wfdErr = wfdNvSciBufSetDisplayAttributesNVX(&m_unreconciledList[0]);
    CHK_WFDSTATUS_AND_RETURN(wfdErr, "wfdNvSciBufSetDisplayAttributesNVX");

    NvSciError sciErr = NvSciBufAttrListSetAttrs(m_unreconciledList[0],
                                                 bufAttrs,
                                                 (sizeof(bufAttrs) / sizeof(bufAttrs[0])));
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    // Create RGBA PL image with CPU access
    NvSciBufType bufType2d = NvSciBufType_Image;
    bool imgCpuAccess = true;
    uint32_t planeCount = 1U;
    NvSciBufAttrValColorFmt planeColorFmt = NvSciColor_A8B8G8R8;
    NvSciBufAttrValColorStd planeColorStd = NvSciColorStd_SRGB;
    NvSciBufAttrValImageLayoutType imgLayout = NvSciBufImage_PitchLinearType;
    uint64_t zeroPadding = 0U;
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
        { NvSciBufImageAttrKey_TopPadding, &zeroPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_BottomPadding, &zeroPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_LeftPadding, &zeroPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_RightPadding, &zeroPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_PlaneWidth, &planeWidth, sizeof(uint32_t) },
        { NvSciBufImageAttrKey_PlaneHeight, &planeHeight, sizeof(uint32_t)  },
        { NvSciBufImageAttrKey_PlaneBaseAddrAlign, &planeBaseAddrAlign, sizeof(uint32_t) }
    };
    size_t length = sizeof(setAttrs) / sizeof(NvSciBufAttrKeyValuePair);
    sciErr = NvSciBufAttrListSetAttrs(m_unreconciledList[1], setAttrs, length);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    m_unreconciledList[2] = bufAttrList;

    sciErr = NvSciBufAttrListReconcile(m_unreconciledList,
                                       RECON_NUM,
                                       &m_reconciledList,
                                       &m_conflictList);
    /*
     * bufAttrList will be freed by the caller so set m_unreconciledList[2] to null to avoid a
     * segmentation fault on the attempted double free in ~CNvSciBufResources
     */
    m_unreconciledList[2] = NULL;
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListReconcile");

    sciErr = NvSciBufObjAlloc(m_reconciledList, &bufObj);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjAlloc");

    return NVSIPL_STATUS_OK;
}

NvSciBufObj & CNvWfdResources::GetBuffer()
{
    return m_bufObj;
}

CNvWfdResources::CNvWfdResources(WFDDevice wfdDevice, WFDint wfdPortId) :
    m_wfdDevice(wfdDevice),
    m_wfdPortId(wfdPortId),
    m_wfdPort(WFD_INVALID_HANDLE),
    m_wfdPipeline(WFD_INVALID_HANDLE),
    m_wfdSource(WFD_INVALID_HANDLE),
    m_inited(false)
{
}

SIPLStatus CNvWfdResources::Init(uint32_t &uWidth,
                                 uint32_t &uHeight,
                                 NvSciBufModule &bufModule,
                                 NvSciBufAttrList &bufAttrList)
{
    WFDPortMode wfdPortMode = WFD_INVALID_HANDLE;

    if ((!m_wfdDevice) || (m_wfdPortId == WFD_INVALID_PORT_ID)) {
        LOG_ERR("Not ready for WFD initialization\n");
        return NVSIPL_STATUS_INVALID_STATE;
    }

    m_wfdPort = wfdCreatePort(m_wfdDevice, m_wfdPortId, NULL);
    if (!m_wfdPort) {
        GET_WFDERROR_AND_RETURN(m_wfdDevice);
    }
    WFDint wfdNumModes = wfdGetPortModes(m_wfdDevice, m_wfdPort, &wfdPortMode, 1);
    if (!wfdNumModes) {
        GET_WFDERROR_AND_RETURN(m_wfdDevice);
    }

    // Get the resolution of the display attached to the port
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

    // Get the number of bindable pipeline IDs for the port
    WFDint wfdNumPipelines = wfdGetPortAttribi(m_wfdDevice, m_wfdPort, WFD_PORT_PIPELINE_ID_COUNT);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    // Populate pipeline IDs into local variable
    std::vector<WFDint> wfdBindablePipeIds(wfdNumPipelines);
    wfdGetPortAttribiv(m_wfdDevice, m_wfdPort, WFD_PORT_BINDABLE_PIPELINE_IDS, wfdNumPipelines, wfdBindablePipeIds.data());
    GET_WFDERROR_AND_RETURN(m_wfdDevice);
    if (wfdNumPipelines <= 0) {
        LOG_ERR("No pipeline was found\n");
        return NVSIPL_STATUS_ERROR;
    }

    m_wfdPipeline = wfdCreatePipeline(m_wfdDevice, wfdBindablePipeIds[0], NULL);
    if (!m_wfdPipeline) {
        GET_WFDERROR_AND_RETURN(m_wfdDevice);
    }
    wfdBindPipelineToPort(m_wfdDevice, m_wfdPort, m_wfdPipeline);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    LOG_DBG("%s: pipeline was created and bound successfully\n", __func__);

    wfdDeviceCommit(m_wfdDevice, WFD_COMMIT_ENTIRE_PORT, m_wfdPort);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    LOG_DBG("%s: wfdBindPipelineToPort success\n", __func__);

    // Allocate buffer
    SIPLStatus status = m_nvSciBufRes.InitBufRes(bufModule,
                                                 m_bufObj,
                                                 bufAttrList,
                                                 displayRes[0],
                                                 displayRes[1]);
    CHK_STATUS_AND_RETURN(status, "InitBufRes");
    CHK_PTR_AND_RETURN(m_bufObj, "InitBufRes");

    m_wfdSource = wfdCreateSourceFromNvSciBufNVX(m_wfdDevice, m_wfdPipeline, &m_bufObj);
    if (!m_wfdSource) {
        GET_WFDERROR_AND_RETURN(m_wfdDevice);
    }

    LOG_DBG("%s: Create WFD source success\n", __func__);

    wfdBindSourceToPipeline(m_wfdDevice, m_wfdPipeline, m_wfdSource, WFD_TRANSITION_AT_VSYNC, NULL);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    WFDint wfdRect[4] {0, 0, displayRes[0], displayRes[1]};

    wfdSetPipelineAttribiv(m_wfdDevice, m_wfdPipeline, WFD_PIPELINE_SOURCE_RECTANGLE, 4, wfdRect);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    wfdSetPipelineAttribiv(m_wfdDevice, m_wfdPipeline, WFD_PIPELINE_DESTINATION_RECTANGLE, 4, wfdRect);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    uWidth = displayRes[0];
    uHeight = displayRes[1];
    m_inited = true;

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvWfdResources::WfdFlip()
{
    if (!m_inited) {
        LOG_ERR("Attempting to post before initialization is complete\n");
        return NVSIPL_STATUS_INVALID_STATE;
    }

    wfdBindSourceToPipeline(m_wfdDevice, m_wfdPipeline, m_wfdSource, WFD_TRANSITION_AT_VSYNC, NULL);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    wfdDeviceCommit(m_wfdDevice, WFD_COMMIT_PIPELINE, m_wfdPipeline);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);

    return NVSIPL_STATUS_OK;
}

CNvWfdResources::~CNvWfdResources()
{
    LOG_DBG("%s enter\n", __func__);

    m_inited = false;

    if (m_wfdDevice && m_wfdPipeline) {
        // Perform a null flip
        wfdBindSourceToPipeline(m_wfdDevice, m_wfdPipeline, (WFDSource)0, WFD_TRANSITION_AT_VSYNC, NULL);
        wfdDeviceCommit(m_wfdDevice, WFD_COMMIT_PIPELINE, m_wfdPipeline);
    }

    if (m_wfdSource != WFD_INVALID_HANDLE) {
        wfdDestroySource(m_wfdDevice, m_wfdSource);
    }

    if (m_bufObj) {
        NvSciBufObjFree(m_bufObj);
    }

    if (m_wfdPipeline) {
        wfdDestroyPipeline(m_wfdDevice, m_wfdPipeline);
    }

    if (m_wfdPort) {
        wfdDestroyPort(m_wfdDevice, m_wfdPort);
    }

    LOG_DBG("%s exit\n", __func__);
}

CNvWfdResourcesCommon::~CNvWfdResourcesCommon()
{
    LOG_DBG("%s enter\n", __func__);

    for (uint32_t i = 0U; i < MAX_SUPPORTED_DISPLAYS; i++) {
        if (m_nvWfd[i] != NULL) {
            delete m_nvWfd[i];
            m_nvWfd[i] = NULL;
        }
    }

    if (m_wfdDevice) {
        wfdDestroyDevice(m_wfdDevice);
    }

    LOG_DBG("%s exit\n", __func__);
}

SIPLStatus CNvWfdResourcesCommon::Init(uint32_t uNumDisplays)
{
    m_wfdDevice = wfdCreateDevice(WFD_DEFAULT_DEVICE_ID, NULL);
    if (!m_wfdDevice) {
        LOG_ERR("wfdCreateDevice failed\n");
        return NVSIPL_STATUS_RESOURCE_ERROR;
    }

    WFDint wfdNumPorts = wfdEnumeratePorts(m_wfdDevice, m_wfdPortIds, uNumDisplays, NULL);
    GET_WFDERROR_AND_RETURN(m_wfdDevice);
    if (wfdNumPorts != ((WFDint)uNumDisplays)) {
        LOG_ERR("Could not find the requested number of ports: %d != %d\n",
                wfdNumPorts,
                uNumDisplays);
        return NVSIPL_STATUS_RESOURCE_ERROR;
    }

    for (uint32_t dispId = 0U; dispId < uNumDisplays; dispId++) {
        m_nvWfd[dispId] = new CNvWfdResources(m_wfdDevice, m_wfdPortIds[dispId]);
        if (m_nvWfd[dispId] == NULL) {
            LOG_ERR("CNvWfdResources creation failed\n");
            return NVSIPL_STATUS_OUT_OF_MEMORY;
        }
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvWfdResourcesCommon::GetDisplayInterface(uint32_t uDispId,
                                                      IDisplayInterface * &pDispIf)
{
    if (uDispId > MAX_SUPPORTED_DISPLAYS) {
        LOG_ERR("Display ID is invalid\n");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }
    if (m_nvWfd[uDispId] == NULL) {
        LOG_ERR("Display interface is not ready to be set\n");
        return NVSIPL_STATUS_INVALID_STATE;
    }
    pDispIf = static_cast<IDisplayInterface *>(m_nvWfd[uDispId]);

    return NVSIPL_STATUS_OK;
}
