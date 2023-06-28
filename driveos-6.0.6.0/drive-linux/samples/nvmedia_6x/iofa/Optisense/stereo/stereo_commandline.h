/* Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef STEREO_COMMANDLINE_H
#define STEREO_COMMANDLINE_H

#include <iostream>
#include <string.h>
#include "common_defs.h"

typedef struct
{
    string inputFilename;
    string refFilename;
    string outputFilename;
    string costFilename;
    string stereoCRCGenFilename;
    string stereoCRCChkFilename;
    uint16_t width;
    uint16_t height;
    uint32_t bitdepth;
    uint32_t estimationType;
    uint16_t ndisp;
    uint32_t gridsize;
    ChromaFormat chromaFormat;
    uint32_t median;
    uint32_t upsample;
    uint32_t profile;
    uint16_t p1;
    uint16_t p2;
    uint16_t adaptiveP2;
    uint16_t alpha;
    uint16_t DiagonalMode;
    uint16_t numPasses;
    uint16_t lrCheck;
    uint16_t lrCheckThr;
    uint16_t nframes;
    uint32_t preset;
    uint16_t do_RL;
} TestArgs;

void PrintUsage(void);
int  ParseArgs(int argc, char **argv, TestArgs *args);

#endif
