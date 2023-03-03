/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _NVMEDIA_JPEG_TEST_CMD_LINE_H_
#define _NVMEDIA_JPEG_TEST_CMD_LINE_H_

#include <stdbool.h>
#include "scibuf_utils.h"
#include "nvmedia_common_encode_decode.h"

#define FILE_NAME_SIZE            256
#define MAX_JPEG_BITSTREAM_BYTES  16384*16384
#define MAX_JPEG_DECODE_WIDTH     16384
#define MAX_JPEG_DECODE_HEIGHT    16384

typedef struct {
    char        *crcFilename;
    bool crcGenMode;
    bool crcCheckMode;
} CRCOptions;

typedef struct _TestArgs {
    char                        *infile;
    char                        *outfile;
    unsigned int                frameNum;
    unsigned int                outputWidth;
    unsigned int                outputHeight;
    bool                        isYUV;
    ChromaFormat                chromaFormat;
    bool                        bMonochrome;

    unsigned int                maxBitstreamBytes;
    unsigned int                maxWidth;
    unsigned int                maxHeight;
    unsigned char               downscaleLog2;
    bool                        supportPartialAccel;

    CRCOptions                  crcoption;
    bool                        cropCRC;
    int                         logLevel;
    NvMediaJPEGInstanceId       instanceId;
} TestArgs;

void PrintUsage(void);
int  ParseArgs(int argc, char **argv, TestArgs *args);

#endif /* _NVMEDIA_JPEG_TEST_CMD_LINE_H_ */
