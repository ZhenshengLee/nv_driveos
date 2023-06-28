/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _NVMEDIA_TEST_CMD_LINE_H_
#define _NVMEDIA_TEST_CMD_LINE_H_

/* Include header containing NvMediaLDC declarations */
#include "nvmedia_ldc.h"
#include "nvmedia_ldc_util.h"

/* Maximum length of the path including file name */
#define FILE_NAME_SIZE 1024

/* TestArgs contains all arguments and variables required to run the LDC test */
typedef struct _TestArgs {
    int logLevel;

    /* A major version of compatible Library */
    uint8_t versionMajor;
    /* A minor version of compatible Library */
    uint8_t versionMinor;

    /* Number of frames to process */
    uint32_t numFrames;

    /* Enable different modes */
    /* Enable geometric transformation */
    NvMediaBool enableGeotrans;
    /* Enable warp map generating */
    NvMediaBool generateWarpMap;
    /* Read warp map from the file */
    NvMediaBool applyWarpMap;
    /* Write xSobel */
    NvMediaBool writeXSobel;
    /* Write downsampled xSobel */
    NvMediaBool writeXSobelDS;
    /* Rotate TNR parameters while processing frames */
    NvMediaBool updateTnrParams;

    /* Source and Destination dimensions */
    uint16_t srcWidth;
    uint16_t srcHeight;
    NvMediaRect srcRect;
    uint16_t dstWidth;
    uint16_t dstHeight;
    NvMediaRect dstRect;

    /* Parameters allocation attributes
     * To enable and allocate memory for TNR and Mask Map
     */
    NvMediaLdcParametersAttributes paramsAttrs;

    /* LDC parameters */
    NvMediaLdcFilter filter;
    NvMediaLdcLensDistortion lensDistortion;
    NvMediaLdcCameraIntrinsic Kin;
    NvMediaLdcCameraIntrinsic Kout;
    NvMediaLdcCameraExtrinsic X;
    NvMediaLdcIptParameters iptParams;
    NvMediaLdcMaskMapParameters maskMapParams;
    NvMediaLdcTnrParameters tnrParams;

    uint32_t numControlPoints;

    /* Input and output files */
    char warpMapFile[FILE_NAME_SIZE];
    char bitMaskFile[FILE_NAME_SIZE];
    char inFile[FILE_NAME_SIZE];
    char outFile[FILE_NAME_SIZE];
    char xSobelFile[FILE_NAME_SIZE];
    char xSobelDSFile[FILE_NAME_SIZE];
} TestArgs;

/* PrintUsage()  Prints application usage options */
void PrintUsage(void);

/* ParseArgs()  Parses command line arguments.
 * Also parses any configuration files supplied in the command line arguments.
* Arguments:
* argc
*    (in) Number of tokens in the command line
* argv
*    (in) Command line tokens
* args
*    (out) Pointer to test arguments structure
*/
int  ParseArgs(int argc, char **argv, TestArgs *args);

#endif /* _NVMEDIA_TEST_CMD_LINE_H_ */
