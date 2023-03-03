/*
 * Copyright (c) 2013-2021 NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _NVMEDIA_TEST_CMD_LINE_H_
#define _NVMEDIA_TEST_CMD_LINE_H_

/*
 * Macro Variable to enable QNX secific code
 */
#ifdef __QNXNTO__
#ifndef NVMEDIA_QNX
#define NVMEDIA_QNX
#endif
#endif

#include <stdbool.h>

#include "nvmedia_parser.h"

typedef struct _TestArgs {
    int                      logLevel;
    NvMediaVideoCodec        eCodec;
    char                    *filename;
    int64_t                  fileSize;
    char                    *OutputYUVFilename;
    int                      loop;
    float                    aspectRatio;
    double                   frameTimeUSec;
    int                      numFramesToDecode;
    int                      deinterlace;
    int                      deinterlaceAlgo;
    bool                     inverceTelecine;
    bool                     showDecodeTimimg;
    int                      displayId;
    bool                     displayEnabled;
    uint8_t                  displayDeviceEnabled;
    bool                     positionSpecifiedFlag;
    NvMediaRect              position;
    unsigned int             windowId;
    unsigned int             depth;
    unsigned int             filterQuality;
    unsigned int             instanceId;
    uint8_t                  checkCRC;
    uint8_t                  generateCRC;
    uint8_t                  cropCRC;
    char                    *crcFilePath;
    uint8_t                  decProfiling;
    uint8_t                  setAnnexBStream;
    unsigned char            av1annexBStream;
    uint8_t                  setOperatingPoint;
    uint8_t                  av1OperatingPoint;
    uint8_t                  setOutputAllLayers;
    uint8_t                  av1OutputAllLayers;
    uint8_t                  setMaxRes;
    uint8_t                  enableMaxRes;
    bool                     alternateCreateAPI;
} TestArgs;

//  PrintUsage
//
//    PrintUsage()  Prints video demo application usage options

void PrintUsage(void);

//  ParseArgs
//
//    ParseArgs()  Parsing command line arguments
//
//  Arguments:
//
//   argc
//      (in) Number of tokens in the command line
//
//   argv
//      (in) Command line tokens
//
//   args
//      (out) Pointer to test arguments structure

int  ParseArgs(int argc, char **argv, TestArgs *args);

#endif /* _NVMEDIA_TEST_CMD_LINE_H_ */
