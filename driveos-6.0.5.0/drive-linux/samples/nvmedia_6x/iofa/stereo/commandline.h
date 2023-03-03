/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef STEREO_CMDLINE_H
#define STEREO_CMDLINE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "log_utils.h"
#include "misc_utils.h"
#include "nvmedia_iofa.h"

#define NVMEDIA_IOFA_BUFFERING (8U)
#define FILE_NAME_SIZE         (256U)

typedef struct {
    char *crcFilename;
    bool  crcGenMode;
    bool  crcCheckMode;
} CRCOptions;

typedef enum {
    YUV_400     = 0,
    YUV_420     = 1,
    YUV_422     = 2,
    YUV_444     = 3,
    NONE_CF     = 5
} ChromaFormatStereo;

typedef struct {
    NvSciBufObj            inputFrame[NVMEDIA_IOFA_BUFFERING][2];
    NvSciBufObj            inputFrameDup[NVMEDIA_IOFA_BUFFERING][2];
    NvSciBufObj            outputSurface[NVMEDIA_IOFA_BUFFERING];
    NvSciBufObj            costSurface[NVMEDIA_IOFA_BUFFERING];
    NvSciSyncFence         preFence[NVMEDIA_IOFA_BUFFERING];
    NvSciSyncFence         eofFence[NVMEDIA_IOFA_BUFFERING];
    uint64_t               startTime[NVMEDIA_IOFA_BUFFERING];
    FILE                   *stereoCrcFile;
    FILE                   *costCrcFile;
    uint32_t               frameRead;
    uint32_t               frameProcessed;
    uint32_t               frameQueued;
    uint32_t               processIdx;
    uint32_t               queueIdx;
    uint32_t               readIdx;
    uint32_t               numFrames;
    uint32_t               width;
    uint32_t               height;
    uint32_t               outWidth;
    uint32_t               outHeight;
    NvMediaIofaProfileData *profDataArray;
}  StereoTestCtx;

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
    uint32_t           logLevel;
    bool               version;
    ChromaFormatStereo chromaFormat;
    uint32_t           width;
    uint32_t           height;
    char               *leftFilename;
    char               *rightFilename;
    char               *outputFilename;
    char               *costFilename;
    uint32_t           inputFormat;
    uint32_t           numFrames;
    CRCOptions         stereoCrcoption;
    CRCOptions         costCrcoption;
    uint8_t            gridsize;
    uint32_t           bitDepth;
    uint8_t            roiMode;
    char               *roiFilename;
    uint8_t            rlSearch;
    uint16_t           ndisp;
    uint8_t            inputBuffering;
    uint32_t           timeout;
    uint8_t            p1;
    uint8_t            p2;
    uint8_t            numPasses;
    uint8_t            alpha;
    uint8_t            diagonalMode;
    uint8_t            adaptiveP2;
    uint32_t           overrideParam;
    uint32_t           profile;
    uint32_t           frameIntervalInMS;
    uint32_t           preset;
    bool               noopMode;
    bool               preFetchInput;
    bool               earlyFenceExpiry;
    bool               playfair;
    char               *profileStatsFilePath;
} TestArgs;

void PrintUsage(void);
int  ParseArgs(int argc, char **argv, TestArgs *args);

#endif // STEREO_CMDLINE_H
