/* Copyright (c) 2022 - 2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef OFA_CLASS_H
#define OFA_CLASS_H

#include <iostream>
#include <cstring>

#include "nvmedia_iofa.h"
#include "common_defs.h"
#include "image_buffer.h"
#include "nvscibuf.h"
#include "image_pyramid.h"

using namespace std;

typedef struct
{
    uint8_t penalty1;
    uint8_t penalty2;
    bool    adaptiveP2;
    uint8_t alphaLog2;
    bool    enableDiag;
    uint8_t numPasses;
} SGMStereoParams;

typedef struct
{
    uint8_t penalty1[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint8_t penalty2[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    bool    adaptiveP2[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint8_t alphaLog2[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    bool    enableDiag[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint8_t numPasses[NVMEDIA_IOFA_MAX_PYD_LEVEL];
} SGMFlowParams;


class NvMOfa
{
protected:
    NvMediaIofa *m_ofaHandle;
    uint16_t    m_inpwidth;
    uint16_t    m_inpheight;
    bool        m_initialized;

public:
    NvMOfa(uint32_t width, uint32_t height);
    ~NvMOfa();
    bool createNvMOfa();
    void destroyNvMOfa();
    bool initNvMOfa(NvMediaIofaInitParams ofaInitParams,
                    uint8_t maxInputBuffer);
    bool registerSurfaceToNvMOfa(NvSciBufObj *image);
    bool unRegisterSurfaceToNvMOfa(NvSciBufObj *image);
    bool registerSciSyncToNvMOfa(NvSciSyncObj syncObj, NvMediaNvSciSyncObjType type);
    bool unRegisterSciSyncToNvMOfa(NvSciSyncObj syncObj);
    bool setSciSyncObjToNvMOfa(NvSciSyncObj syncObj);
    bool getEOFSciSyncFence(NvSciSyncObj syncObj, NvSciSyncFence *preFence);
    bool insertPreFence(const NvSciSyncFence* preFence);
};

class NvMOfaStereo : public NvMOfa
{
private:
    uint16_t m_ndisp;
    uint32_t m_gridSize;
    uint16_t m_outWidth;
    uint16_t m_outHeight;
    uint32_t m_preset;

public:
    NvMOfaStereo(uint32_t width,
                 uint32_t height,
                 uint32_t gridsize,
                 uint16_t ndisp,
                 uint32_t preset);
    ~NvMOfaStereo(){};
    bool initNvMOfaStereo(uint8_t maxInputBuffer);
    bool getOutSurfaceParams(uint16_t& outWidth,
                             uint16_t& outHeight,
                             uint8_t&  gridSizeLog2X,
                             uint8_t&  gridSizeLog2Y);

    bool processSurfaceNvMOfa(NvSciBufObj *inputImage,
                              NvSciBufObj *refImage,
                              NvSciBufObj *outSurface,
                              NvSciBufObj *costSurface,
                              bool RLcheck = false);
    bool getSGMStereoConfigParams(SGMStereoParams &SGMStereoParams);
    bool setSGMStereoConfigParams(SGMStereoParams &SGMStereoParams);
};

class NvMOfaFlow : public NvMOfa
{
private:
    uint16_t     m_width[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint16_t     m_height[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint16_t     m_outWidth[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint16_t     m_outHeight[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint8_t      m_pydLevel;
    uint16_t     m_pydSGMMode;
    uint32_t     m_gridSize[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    ChromaFormat m_chromaFormat;
    uint32_t m_preset;

public:

    NvMOfaFlow(uint32_t width,
               uint32_t height,
               ChromaFormat chromaFormat,
               uint32_t gridsize[NVMEDIA_IOFA_MAX_PYD_LEVEL],
               uint16_t pydSGMMode,
               uint32_t preset);

    uint8_t getNumOfPydLevels() { return m_pydLevel; }

    bool initNvMOfaFlow(uint8_t maxInputBuffer);

    bool getInputSurfaceParams(uint16_t **width, uint16_t **height);

    bool getOutSurfaceParams(uint16_t **outWidth,
                             uint16_t **outHeight,
                             uint32_t  **gridSizeLog2X,
                             uint32_t  **gridSizeLog2Y);

    bool processSurfaceNvMOfa(ImagePyramid *inputImage,
                              ImagePyramid *refImage,
                              ImagePyramid *outSurface,
                              ImagePyramid *costSurface,
                              ImagePyramid *hintSurface = NULL,
                              uint8_t currPydLevel = 0);

    bool decidePydLevelAndRes();
    bool getSGMFlowConfigParams(SGMFlowParams &SGMFlowParams);
    bool setSGMFlowConfigParams(SGMFlowParams &SGMFlowParams);
};
#endif
