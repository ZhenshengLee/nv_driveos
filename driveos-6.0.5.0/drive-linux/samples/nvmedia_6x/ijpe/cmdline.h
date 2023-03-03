/*
 * Copyright (c) 2014-2020, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _NVMEDIA_JPEG_TEST_CMD_LINE_H_
#define _NVMEDIA_JPEG_TEST_CMD_LINE_H_

#include <stdbool.h>
#include "scibuf_utils.h"
#include "nvmedia_common_encode_decode.h"

#define FILE_NAME_SIZE                  256

typedef struct {
    char        *crcFilename;
    bool         crcGenMode;
    bool         crcCheckMode;
} CRCOptions;

typedef struct _TestArgs {
    char                        *infile;
    char                        *outfile;
    char                        *huffFileName;
    char                        *quantFileName;
    unsigned int                inputWidth;
    unsigned int                inputHeight;
    ChromaFormat                inputSurfType;

    unsigned int                maxOutputBuffering;
    unsigned char               quality;

    CRCOptions                  crcoption;
    int                         logLevel;
    bool                        huffTable;
    bool                        quantTable;
    NvMediaJPEGInstanceId       instanceId;
} TestArgs;

void PrintUsage(void);
int  ParseArgs(int argc, char **argv, TestArgs *args);

#endif /* _NVMEDIA_JPEG_TEST_CMD_LINE_H_ */
