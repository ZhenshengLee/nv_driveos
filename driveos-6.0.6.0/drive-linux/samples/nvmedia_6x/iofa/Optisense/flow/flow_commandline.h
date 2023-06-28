/* Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef FLOW_COMMANDLINE_H
#define FLOW_COMMANDLINE_H

#include <iostream>
#include "common_defs.h"
#include "nvmedia_iofa.h"

typedef struct
{
    string inputFilename;
    string refFilename;
    string outputFilename;
    string costFilename;
    string flowCRCGenFilename;
    string flowCRCChkFilename;
    uint16_t width;
    uint16_t height;
    uint32_t bitdepth;
    uint32_t gridsize[NVMEDIA_IOFA_MAX_PYD_LEVEL] = {0};
    uint32_t median;
    ChromaFormat chromaFormat;
    uint32_t upsample;
    uint32_t profile;
    uint16_t p1[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint16_t p2[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint16_t adaptiveP2[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint16_t alpha[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint16_t DiagonalMode[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint16_t numPasses[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint16_t nframes;
    uint16_t overrideParam;
    uint16_t do_bwd;
    uint16_t fbCheck;
    uint16_t fbCheckThr;
    uint16_t pydSGMMode;
    uint32_t preset;
} FlowTestArgs;

void PrintUsage(void);
int  ParseArgs(int argc, char **argv, FlowTestArgs *args);

#endif
