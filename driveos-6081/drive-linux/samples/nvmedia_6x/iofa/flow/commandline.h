/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef FLOW_CMDLINE_H
#define FLOW_CMDLINE_H

#include "image_ofa_utils.h"
#include "log_utils.h"
#include "misc_utils.h"
#include "nvmedia_iofa.h"
#include <unistd.h>

#define NVMEDIA_IOFA_BUFFERING (8U)
#define FILE_NAME_SIZE         (256U)

typedef struct {
    char        *crcFilename;
    bool crcGenMode;
    bool crcCheckMode;
} CRCOptions;

typedef enum {
    YUV_400     = 0,
    YUV_420     = 1,
    YUV_422     = 2,
    YUV_444     = 3,
    NONE_CF     = 5
} ChromaFormatOFA;

typedef struct {
    NvSciBufObj            inputFrame[NVMEDIA_IOFA_BUFFERING][2][NVMEDIA_IOFA_MAX_PYD_LEVEL];
    NvSciBufObj            inputFrameDup[NVMEDIA_IOFA_BUFFERING][2][NVMEDIA_IOFA_MAX_PYD_LEVEL];
    NvSciBufObj            outputSurface[NVMEDIA_IOFA_BUFFERING][NVMEDIA_IOFA_MAX_PYD_LEVEL];
    NvSciBufObj            costSurface[NVMEDIA_IOFA_BUFFERING][NVMEDIA_IOFA_MAX_PYD_LEVEL];
    NvSciSyncFence         preFence[NVMEDIA_IOFA_BUFFERING];
    NvSciSyncFence         eofFence[NVMEDIA_IOFA_BUFFERING];
    uint64_t               startTime[NVMEDIA_IOFA_BUFFERING];
    FILE                   *FlowCrcFile;
    FILE                   *costCrcFile;
    uint32_t               frameRead;
    uint32_t               frameProcessed;
    uint32_t               frameQueued;
    uint32_t               processIdx;
    uint32_t               queueIdx;
    uint32_t               readIdx;
    uint32_t               numFrames;
    uint16_t               width[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint16_t               height[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint16_t               outWidth[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint16_t               outHeight[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint8_t                pydLevel;
}  FlowTestCtx;

typedef enum
{
    SGMPARAM_P1_OVERRIDE         = (1<<0),
    SGMPARAM_P2_OVERRIDE         = (1<<1),
    SGMPARAM_DIAGONAL_OVERRIDE   = (1<<2),
    SGMPARAM_ADAPTIVEP2_OVERRIDE = (1<<3),
    SGMPARAM_NUMPASSES_OVERRIDE  = (1<<4),
    SGMPARAM_ALPHA_OVERRIDE      = (1<<5),
} SGMPARAM_OVERRIDE;

typedef struct
{
    uint32_t        logLevel;
    bool            version;
    ChromaFormatOFA chromaFormat;
    ChromaFormatOFA pydChromaFormat;
    uint8_t         pydLevel;
    uint32_t        width;
    uint32_t        height;
    char            *inputFilename;
    char            *outputFilename;
    char            *costFilename;
    char            *epipolarFilename;
    uint32_t        numFrames;
    CRCOptions      flowCrcoption;
    CRCOptions      costCrcoption;
    uint8_t         gridsize[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint32_t        bitDepth;
    uint8_t         roiMode;
    char            *roiFilename;
    uint8_t         disableCostOut;
    uint8_t         pydSGMMode;
    uint8_t         inputBuffering;
    uint32_t        profile;
    uint32_t        timeout;
    uint8_t         p1[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint8_t         p2[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint8_t         alpha[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint8_t         diagonalMode[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint8_t         adaptiveP2[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint8_t         numPasses[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint32_t        overrideParam;
    bool            backwardRef;
    uint32_t        frameIntervalInMS;
    bool            noopMode;
    bool            preFetchInput;
    bool            earlyFenceExpiry;
    bool            playfair;
    char            *profileStatsFilePath;
    double          initLat;
    double          submitLat;
    double          execLat;
    bool            profileTestEnable;
    uint32_t        preset;
    uint8_t         flowmode;
    uint8_t         multiEpi;
    uint16_t        ndisp;
    uint16_t        epiSearchRange;
    bool            enableVMSuspend;
} TestArgs;

void PrintUsage(void);
int  ParseArgs(int argc, char **argv, TestArgs *args);

#endif // FLOW_CMDLINE_H
