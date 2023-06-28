/* Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "ofa_class.h"

NvMOfa::NvMOfa(uint32_t width,
               uint32_t height,
               uint32_t profile)
{
    m_ofaHandle = NULL;
    m_inpwidth = width;
    m_inpheight = height;
    m_profile = profile;
    m_initialized = false;
}

NvMOfa::~NvMOfa ()
{
    m_ofaHandle = NULL;
}

bool NvMOfa::createNvMOfa ()
{
    m_ofaHandle = NvMediaIOFACreate();
    if (m_ofaHandle == NULL)
    {
        cerr << "NvMedia OFA handle is NULL";
        return false;
    }
    return true;
}

void NvMOfa::destroyNvMOfa()
{
    NvMediaIOFADestroy(m_ofaHandle);
}

bool NvMOfa::initNvMOfa(NvMediaIofaInitParams ofaInitParams,
                        uint8_t maxInputBuffer)
{
    NvMediaStatus status;

    status = NvMediaIOFAInit(m_ofaHandle, &ofaInitParams, maxInputBuffer);
    if (status != NVMEDIA_STATUS_OK)
    {
    	cerr << "NvMediaIOFAInit function failed \n";
        return false;
    }
    m_initialized = true;
    return true;
}

bool NvMOfa::insertPreFence(const NvSciSyncFence* preFence)
{
    NvMediaStatus status;

    status = NvMediaIOFAInsertPreNvSciSyncFence(m_ofaHandle, preFence);
    if (status != NVMEDIA_STATUS_OK)
    {
    	cerr << "NvMediaIOFAInsertPreNvSciSyncFence function failed \n";
        return false;
    }

    return true;
}

bool NvMOfa::printProfileData ()
{
    NvMediaIofaProfileData ProfData = {};
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = NvMediaIOFAGetProfileData(m_ofaHandle, &ProfData);
    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "PrintProfileData: NvMediaIOFAGetProfileData is failed \n";
        return false;
    }
    if (ProfData.validProfData == true)
    {
        cout << "PrintProfileData: Time in us sw time " << ProfData.swTimeInUS <<" hw time " <<  ProfData.hwTimeInUS << " sync wait time " << ProfData.syncWaitTimeInUS << "\n";
    }
    return true;
}

bool NvMOfaStereo::getSGMStereoConfigParams(SGMStereoParams &ofaSGMStereoParams)
{
    NvMediaStatus status;
    NvMediaIofaSGMParams nvMediaSGMParams{};
    status = NvMediaIOFAGetSGMConfigParams(m_ofaHandle, &nvMediaSGMParams);

    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "NvMediaIOFAGetSGMConfigParams failed \n";
        return false;
    }
    ofaSGMStereoParams.penalty1   = nvMediaSGMParams.penalty1[0U];
    ofaSGMStereoParams.penalty2   = nvMediaSGMParams.penalty2[0U];
    ofaSGMStereoParams.adaptiveP2 = nvMediaSGMParams.adaptiveP2[0U];
    ofaSGMStereoParams.alphaLog2  = nvMediaSGMParams.alphaLog2[0U];
    ofaSGMStereoParams.enableDiag = nvMediaSGMParams.enableDiag[0U];
    ofaSGMStereoParams.numPasses  = nvMediaSGMParams.numPasses[0U];

    return true;
}

bool NvMOfaStereo::setSGMStereoConfigParams(SGMStereoParams &ofaSGMStereoParams)
{
    NvMediaStatus status;
    NvMediaIofaSGMParams pSGMParams{};
    pSGMParams.penalty1[0U]   = ofaSGMStereoParams.penalty1;
    pSGMParams.penalty2[0U]   = ofaSGMStereoParams.penalty2;
    pSGMParams.adaptiveP2[0U] = ofaSGMStereoParams.adaptiveP2;
    pSGMParams.alphaLog2[0U]  = ofaSGMStereoParams.alphaLog2;
    pSGMParams.enableDiag[0U] = ofaSGMStereoParams.enableDiag;
    pSGMParams.numPasses[0U]  = ofaSGMStereoParams.numPasses;
    status = NvMediaIOFASetSGMConfigParams(m_ofaHandle, &pSGMParams);

    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "NvMediaIOFASetSGMConfigParams failed \n";
        return false;
    }

    return true;
}

bool NvMOfa::registerSurfaceToNvMOfa(NvSciBufObj *image)
{
    NvMediaStatus status;

    if (!m_initialized)
    {
        return false;
    }
    status = NvMediaIOFARegisterNvSciBufObj(m_ofaHandle, *image);
    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "NvMediaIOFARegisterNvSciBufObj failed \n";
        return true;
    }

    return true;
}

bool NvMOfa::unRegisterSurfaceToNvMOfa(NvSciBufObj *image)
{
    NvMediaStatus status;

    status = NvMediaIOFAUnregisterNvSciBufObj(m_ofaHandle, *image);
    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "NvMediaIOFAUnregisterNvSciBufObj failed \n";
        return false;
    }
    return true;
}

bool NvMOfa::registerSciSyncToNvMOfa(NvSciSyncObj syncObj, NvMediaNvSciSyncObjType type)
{
    NvMediaStatus status;
    if (!m_initialized)
    {
        cerr << "NvMediaIOFARegisterNvSciSyncObj failed \n";
        return false;
    }
    status = NvMediaIOFARegisterNvSciSyncObj(m_ofaHandle, type, syncObj);
    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "NvMediaIOFARegisterNvSciSyncObj failed \n";
        return true;
    }
    return true;
}

bool NvMOfa::unRegisterSciSyncToNvMOfa(NvSciSyncObj syncObj)
{
    NvMediaStatus status;
    status = NvMediaIOFAUnregisterNvSciSyncObj(m_ofaHandle, syncObj);
    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "NvMediaIOFAUnregisterNvSciSyncObj failed " << status << "\n";
        return false;
    }
    return true;
}
bool NvMOfa::setSciSyncObjToNvMOfa(NvSciSyncObj syncObj)
{
    NvMediaStatus status;
    status = NvMediaIOFASetNvSciSyncObjforEOF(m_ofaHandle, syncObj);
    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "NvMediaIOFASetNvSciSyncObjforEOF failed \n";
        return false;
    }
    return true;
}
bool NvMOfa::getEOFSciSyncFence(NvSciSyncObj syncObj, NvSciSyncFence *preFence)
{
    NvMediaStatus status;
    status = NvMediaIOFAGetEOFNvSciSyncFence(m_ofaHandle, syncObj, preFence);
    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "NvMediaIOFAGetEOFNvSciSyncFence failed \n";
        return false;
    }
    return true;
}

NvMOfaStereo::NvMOfaStereo(uint32_t width,
                           uint32_t height,
                           uint32_t gridsize,
                           uint16_t ndisp,
                           uint32_t profile,
                           uint32_t preset)
    : NvMOfa(width, height, profile)
{
    m_gridSize = gridsize;
    m_ndisp = ndisp;
    m_outWidth = (m_inpwidth + (1 << m_gridSize) -1 ) >> m_gridSize;
    m_outHeight = (m_inpheight + (1 << m_gridSize) -1 ) >> m_gridSize;
    m_preset = preset;
}

bool NvMOfaStereo::initNvMOfaStereo(uint8_t maxInputBuffer)
{
    NvMediaIofaInitParams ofaInitParams{};

    ofaInitParams.ofaMode      = NVMEDIA_IOFA_MODE_STEREO;
    ofaInitParams.ofaPydLevel  = 1;
    ofaInitParams.width[0]     = m_inpwidth;
    ofaInitParams.height[0]    = m_inpheight;
    ofaInitParams.gridSize[0]  = static_cast <NvMediaIofaGridSize>(m_gridSize);
    ofaInitParams.outWidth[0]  = m_outWidth;
    ofaInitParams.outHeight[0] = m_outHeight;
    ofaInitParams.preset       = static_cast <NvMediaIofaPreset>(m_preset);
    if (m_ndisp == 128U)
    {
        ofaInitParams.dispRange = NVMEDIA_IOFA_DISPARITY_RANGE_128;
    }
    else
    {
        ofaInitParams.dispRange = NVMEDIA_IOFA_DISPARITY_RANGE_256;
    }
    ofaInitParams.profiling = static_cast <NvMediaIofaProfileMode>(m_profile);

    if (!initNvMOfa(ofaInitParams, maxInputBuffer))
    {
        return false;
    }
    m_initialized = true;
    return true;
}

bool NvMOfaStereo::getOutSurfaceParams(uint16_t& outWidth,
                                       uint16_t& outHeight,
                                       uint8_t&  gridSizeLog2X,
                                       uint8_t&  gridSizeLog2Y)
{
    if (!m_initialized)
    {
        return false;
    }

    outWidth = m_outWidth;
    outHeight = m_outHeight;
    gridSizeLog2X = m_gridSize;
    gridSizeLog2Y = m_gridSize;

    return true;
}

bool NvMOfaStereo::processSurfaceNvMOfa(NvSciBufObj *inputImage,
                                        NvSciBufObj *refImage,
                                        NvSciBufObj *outSurface,
                                        NvSciBufObj *costSurface,
                                        bool  RLcheck)
{
    NvMediaIofaProcessParams ofaProcessParams;
    NvMediaIofaBufArray surfArray;
    NvMediaStatus status;

    if ((inputImage == NULL) || (refImage == NULL) || (outSurface == NULL) || (costSurface == NULL))
    {
        cerr << "NULL surface for input or output\n";
        return false;
    }

    if (!m_initialized)
    {
        return false;
    }
    memset(&ofaProcessParams, 0, sizeof(NvMediaIofaProcessParams));
    memset(&surfArray, 0, sizeof(NvMediaIofaBufArray));

    surfArray.inputSurface[0] = *inputImage;
    surfArray.refSurface[0] = *refImage;
    surfArray.outSurface[0] = *outSurface;
    surfArray.costSurface[0] = *costSurface;
    ofaProcessParams.rightDispMap = RLcheck;

    status = NvMediaIOFAProcessFrame(m_ofaHandle, &surfArray, &ofaProcessParams, NULL, NULL);
    if (status != NVMEDIA_STATUS_OK)
    {
        return false;
    }

    return true;
}

NvMOfaFlow::NvMOfaFlow(uint32_t width,
                       uint32_t height,
                       ChromaFormat chromaFormat,
                       uint32_t* gridsize,
                       uint16_t pydSGMMode,
                       uint32_t profile,
                       uint32_t preset)
    : NvMOfa(width, height, profile)
{
    uint8_t k;

    m_chromaFormat = chromaFormat;
    m_gridSize[0] = gridsize[0];
    m_preset = preset;

    for (k = 1; k < NVMEDIA_IOFA_MAX_PYD_LEVEL; k++)
    {
        if (pydSGMMode == 1)
        {
            if (gridsize[k] > 1)
            {
                m_gridSize[k] = gridsize[k]-1;
            }
            else
            {
                m_gridSize[k] = gridsize[k];
            }
        }
        else
        {
           m_gridSize[k] = gridsize[k];
    	}
    }
    m_pydSGMMode = pydSGMMode;
}

bool NvMOfaFlow::initNvMOfaFlow(uint8_t maxInputBuffer)
{
    NvMediaIofaInitParams ofaInitParams{};
    uint8_t k;

    decidePydLevelAndRes();

    ofaInitParams.ofaMode = NVMEDIA_IOFA_MODE_PYDOF;
    ofaInitParams.ofaPydLevel = m_pydLevel;
    for (k = 0; k < m_pydLevel; k++)
    {
        ofaInitParams.gridSize[k]  = static_cast <NvMediaIofaGridSize>(m_gridSize[k]);
        ofaInitParams.width[k]     = m_width[k];
        ofaInitParams.height[k]    = m_height[k];
        ofaInitParams.outWidth[k]  = m_outWidth[k];
        ofaInitParams.outHeight[k] = m_outHeight[k];
        ofaInitParams.preset       = static_cast <NvMediaIofaPreset>(m_preset);
    }
    ofaInitParams.pydMode = static_cast <NvMediaIofaPydMode>(m_pydSGMMode);
    ofaInitParams.profiling = static_cast <NvMediaIofaProfileMode>(m_profile);

    if (!initNvMOfa(ofaInitParams, maxInputBuffer))
    {
        cerr << "initNvMOfa function failed \n";
        return false;
    }
    m_initialized = true;
    return true;
}

bool NvMOfaFlow::decidePydLevelAndRes()
{
    NvMediaIofaCapability Capability;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t pydLevel = 0, i;
    uint16_t width = m_inpwidth;
    uint16_t height = m_inpheight;

    status = NvMediaIOFAGetCapability(m_ofaHandle, NVMEDIA_IOFA_MODE_PYDOF, &Capability);
    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "NvMediaIOFAGetCapability function failed \n";
        return false;
    }

    if (m_chromaFormat == YUV_420)
    {
        if (((m_inpwidth % 2) != 0)  || ((m_inpheight % 2) != 0))
        {
            cerr << "input width and height must be even for YUV420 input \n";
            return false;
        }
    }
    else if (m_chromaFormat == YUV_422)
    {
        if (((m_inpwidth % 2) != 0))
        {
            cerr << "input width must be even for YUV422 input \n";
            return false;
        }
    }

    while (pydLevel < NVMEDIA_IOFA_MAX_PYD_LEVEL)
    {
        if ((width < Capability.minWidth) || (height < Capability.minHeight))
        {
            break;
        }
        m_width[pydLevel] = width;
        m_height[pydLevel] = height;
        width = (width + 1U) >> 1U;
        height = (height + 1U) >> 1U;
        if ((m_chromaFormat == YUV_420) || (m_chromaFormat == YUV_422))
        {
            width = (width + 1) & 0xFFFE;
        }
        if (m_chromaFormat == YUV_420)
        {
            height = (height + 1) & 0xFFFE;
        }
        pydLevel++;
    }
    if (pydLevel == 0U)
    {
        cerr << "main: Pyramid level 0 is not supported \n";
        return false;
    }
    m_pydLevel = pydLevel;
    for (i = 0; i < pydLevel; i++)
    {
        m_outWidth[i] = (m_width[i] + (1U << m_gridSize[i]) - 1) >> m_gridSize[i];
        m_outHeight[i] = (m_height[i] + (1U << m_gridSize[i]) -1) >> m_gridSize[i];
    }

    return true;
}

bool NvMOfaFlow::getInputSurfaceParams(uint16_t **width, uint16_t **height)
{
    if (!m_initialized)
    {
        return false;
    }
    *width = m_width;
    *height = m_height;
    return true;
}

bool NvMOfaFlow::getOutSurfaceParams(uint16_t **outWidth,
                                     uint16_t **outHeight,
                                     uint32_t  **gridSizeLog2X,
                                     uint32_t  **gridSizeLog2Y)
{
    if (!m_initialized)
    {
        return false;
    }

    *outWidth = m_outWidth;
    *outHeight = m_outHeight;
    *gridSizeLog2X = m_gridSize;
    *gridSizeLog2Y = m_gridSize;

    return true;
}

bool NvMOfaFlow::processSurfaceNvMOfa(ImagePyramid *inputImage,
                                      ImagePyramid *refImage,
                                      ImagePyramid *outSurface,
                                      ImagePyramid *costSurface,
                                      ImagePyramid *hintSurface,
                                      uint8_t currPydLevel)
{
    NvMediaIofaProcessParams ofaProcessParams;
    NvMediaIofaBufArray surfArray;
    NvMediaStatus status;
    uint8_t i;

    if ((inputImage == NULL) || (refImage == NULL) || (outSurface == NULL) || (costSurface == NULL))
    {
        cerr << "NULL surface for input or output\n";
        return false;
    }

    if (!m_initialized)
    {
        cerr << "OFA Class not initialized\n";
        return false;
    }
    memset(&ofaProcessParams, 0, sizeof(NvMediaIofaProcessParams));
    memset(&surfArray, 0, sizeof(NvMediaIofaBufArray));
    for (i = 0; i < m_pydLevel; i++)
    {
        if (inputImage->getImageBuffer(i) != NULL)
        {
           surfArray.inputSurface[i] = *(inputImage->getImageBuffer(i))->getHandle();
        }
        else
        {
            cerr << "Input image is NULL\n";
            return false;
        }
        if (refImage->getImageBuffer(i) != NULL)
        {
            surfArray.refSurface[i] = *(refImage->getImageBuffer(i))->getHandle();
        }
        else
        {
            cerr << "Reference image is NULL\n";
            return false;
        }
        if (outSurface->getImageBuffer(i) != NULL)
        {
            surfArray.outSurface[i] = *(outSurface->getImageBuffer(i))->getHandle();
        }
        else
        {
            cerr << "Output image is NULL\n";
            return false;
        }
        if (costSurface->getImageBuffer(i) != NULL)
        {
            surfArray.costSurface[i] = *(costSurface->getImageBuffer(i))->getHandle();
        }
        else
        {
            cerr << "Cost image is NULL\n";
            return false;
        }
        if ((m_pydSGMMode == (uint16_t)NVMEDIA_IOFA_PYD_LEVEL_MODE) &&
            (i < (m_pydLevel - 1U)))
        {
            if (hintSurface == NULL)
            {
                cerr << "hint Pyramid is NULL\n";
                return false;
            }
            if (hintSurface->getImageBuffer(i) != NULL)
            {
                surfArray.pydHintSurface[i] = *(hintSurface->getImageBuffer(i))->getHandle();
            }
            else
            {
                cerr << "Pyramid Hint image is NULL\n";
                return false;
            }
        }
        else
        {
            surfArray.pydHintSurface[i] = NULL;
        }
    }
    if (m_pydSGMMode == (uint16_t)NVMEDIA_IOFA_PYD_LEVEL_MODE)
    {
        ofaProcessParams.pydHintParams.pydHintMagnitudeScale2x = true;
        ofaProcessParams.currentPydLevel = currPydLevel;
        if (currPydLevel < (m_pydLevel - 1U))
        {
            uint16_t wh = hintSurface->getImageBuffer(currPydLevel)->getWidth();
            uint16_t wo = outSurface->getImageBuffer(currPydLevel)->getWidth();

            if (wh == (wo + 1)/2)
            {
                ofaProcessParams.pydHintParams.pydHintWidth2x = true;
            }
            else if (wh == wo)
            {
                ofaProcessParams.pydHintParams.pydHintWidth2x = false;
            }
            else
            {
                cerr << "Hint Surface width not matching with output surface \n";
                return false;
            }

            uint16_t hh = hintSurface->getImageBuffer(currPydLevel)->getHeight();
            uint16_t ho = outSurface->getImageBuffer(currPydLevel)->getHeight();
            if (hh == (ho + 1)/2)
            {
                ofaProcessParams.pydHintParams.pydHintHeight2x = true;
            }
            else if (hh == ho)
            {
                ofaProcessParams.pydHintParams.pydHintHeight2x = false;
            }
            else
            {
                cerr << "Hint Surface height not matching with output surface \n";
                return false;
            }
        }
    }
    status = NvMediaIOFAProcessFrame(m_ofaHandle, &surfArray, &ofaProcessParams, NULL, NULL);
    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "NvMediaIOFAProcessFrame failed\n";
        return false;
    }

    return true;
}

bool NvMOfaFlow::getSGMFlowConfigParams(SGMFlowParams &ofaSGMFlowParams)
{
    NvMediaStatus status;
    NvMediaIofaSGMParams nvMediaSGMParams{};
    uint8_t i;
    status = NvMediaIOFAGetSGMConfigParams(m_ofaHandle, &nvMediaSGMParams);
    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "NvMediaIOFAGetSGMConfigParams failed\n";
        return false;
    }
    for (i = 0U; i < NVMEDIA_IOFA_MAX_PYD_LEVEL; i++)
    {
        ofaSGMFlowParams.penalty1[i]   = nvMediaSGMParams.penalty1[i];
        ofaSGMFlowParams.penalty2[i]   = nvMediaSGMParams.penalty2[i];
        ofaSGMFlowParams.adaptiveP2[i] = nvMediaSGMParams.adaptiveP2[i];
        ofaSGMFlowParams.alphaLog2[i]  = nvMediaSGMParams.alphaLog2[i];
        ofaSGMFlowParams.enableDiag[i] = nvMediaSGMParams.enableDiag[i];
        ofaSGMFlowParams.numPasses[i]  = nvMediaSGMParams.numPasses[i];
    }
    return true;
}

bool NvMOfaFlow::setSGMFlowConfigParams(SGMFlowParams &ofaSGMFlowParams)
{
    NvMediaStatus status;
    NvMediaIofaSGMParams pSGMParams{};
    uint8_t i;
    for (i = 0U; i < NVMEDIA_IOFA_MAX_PYD_LEVEL; i++)
    {
        pSGMParams.penalty1[i]   = ofaSGMFlowParams.penalty1[i];
        pSGMParams.penalty2[i]   = ofaSGMFlowParams.penalty2[i];
        pSGMParams.adaptiveP2[i] = ofaSGMFlowParams.adaptiveP2[i];
        pSGMParams.alphaLog2[i]  = ofaSGMFlowParams.alphaLog2[i];
        pSGMParams.enableDiag[i] = ofaSGMFlowParams.enableDiag[i];
        pSGMParams.numPasses[i]  = ofaSGMFlowParams.numPasses[i];
    }
    status = NvMediaIOFASetSGMConfigParams(m_ofaHandle, &pSGMParams);

    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "NvMediaIOFASetSGMConfigParams failed\n";
        return false;
    }

    return true;
}



