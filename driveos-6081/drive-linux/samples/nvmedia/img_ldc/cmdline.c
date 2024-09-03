/* Copyright (c) 2017-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

/* Standard headers */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Nvidia headers */
#include "cmdline.h"
#include "log_utils.h"
#include "config_parser.h"

/* see cmdline.h for details */
void PrintUsage()
{
    LOG_MSG("nvmimg_ldc is NvMedia Lens Distortion Correction test program\n");
    LOG_MSG("Usage: nvmimg_ldc [options]\n");
    LOG_MSG("Options:\n");
    LOG_MSG("-h           Prints usage\n");
    LOG_MSG("-v  [level]  Verbosity Level = 0(Err), 1(Warn), 2(Info), 3(Debug)\n");
    LOG_MSG("-cf [config] LDC config file. Path length limited to %u chars\n", FILE_NAME_SIZE);
}

SectionMap sectionsMap[] = {
    {SECTION_NONE, "", 0, 0} /* Has to be the last item - specifies the end of array */
};

/* see cmdline.h for details */
int ParseArgs(int argc, char *argv[], TestArgs *args)
{
    NvMediaBool bLastArg = NVMEDIA_FALSE;
    NvMediaBool bDataAvailable = NVMEDIA_FALSE;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    char *filename = NULL;
    int i;

    NvMediaLdcParametersAttributes *paramsAttrs = &(args->paramsAttrs);
#if !NV_IS_SAFETY
    NvMediaLdcLensDistortion *distortion = &(args->lensDistortion);
    NvMediaLdcCameraIntrinsic *Kin = &(args->Kin);
    NvMediaLdcCameraIntrinsic *Kout = &(args->Kout);
    NvMediaLdcCameraExtrinsic *X = &(args->X);
#endif

    NvMediaLdcIptParameters *ipt = &(args->iptParams);
    NvMediaLdcRegionParameters *region = &(ipt->regionParams);
    NvMediaLdcMaskMapParameters *maskMap = &(args->maskMapParams);
    NvMediaLdcTnrParameters *tnr = &(args->tnrParams);

    /* ConfigParamsMap
     * See nvmedia_ldc.h and sample config file(s) for details.
     */
    ConfigParamsMap paramsMap[] = {

        /*ParamName,             &args->variableName,          paramType,     D, LimitType,   Mn, Mx, CharSize,       p2C, section   */
        {"versionMajor",         &args->versionMajor,          TYPE_UCHAR,    0, LIMITS_BOTH, 3,  3,  0,              0, SECTION_NONE},
        {"versionMinor",         &args->versionMinor,          TYPE_UCHAR,    0, LIMITS_NONE, 0,  0,  0,              0, SECTION_NONE},

        /*ParamName,             &args->variableName,          paramType,     D, LimitType,   Mn, Mx, CharSize,       p2C, section   */

        {"enableGeotrans",       &args->enableGeotrans,        TYPE_UINT,     0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"enableMaskMap",        &paramsAttrs->enableMaskMap,  TYPE_UCHAR,    0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"enableTnr",            &paramsAttrs->enableTnr,      TYPE_UCHAR,    0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"generateWarpMap",      &args->generateWarpMap,       TYPE_UINT,     0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"applyWarpMap",         &args->applyWarpMap,          TYPE_UINT,     0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"writeXSobel",          &args->writeXSobel,           TYPE_UINT,     0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"writeXSobelDS",        &args->writeXSobelDS,         TYPE_UINT,     0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"updateTnrParams",      &args->updateTnrParams,       TYPE_UINT,     0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},

        /*src*/
        {"srcWidth",             &args->srcWidth,              TYPE_USHORT,   0, LIMITS_MIN,  0,  0,  0,              0, SECTION_NONE},
        {"srcHeight",            &args->srcHeight,             TYPE_USHORT,   0, LIMITS_MIN,  0,  0,  0,              0, SECTION_NONE},
        /*srcRect*/
        {"srcRectx0",            &args->srcRect.x0,            TYPE_USHORT,   0, LIMITS_MIN,  0,  0,  0,              0, SECTION_NONE},
        {"srcRecty0",            &args->srcRect.y0,            TYPE_USHORT,   0, LIMITS_MIN,  0,  0,  0,              0, SECTION_NONE},
        {"srcRectx1",            &args->srcRect.x1,            TYPE_USHORT,   0, LIMITS_MIN,  0,  0,  0,              0, SECTION_NONE},
        {"srcRecty1",            &args->srcRect.y1,            TYPE_USHORT,   0, LIMITS_MIN,  0,  0,  0,              0, SECTION_NONE},
        /*dst*/
        {"dstWidth",             &args->dstWidth,              TYPE_USHORT,   0, LIMITS_MIN,  0,  0,  0,              0, SECTION_NONE},
        {"dstHeight",            &args->dstHeight,             TYPE_USHORT,   0, LIMITS_MIN,  0,  0,  0,              0, SECTION_NONE},
        /*dstRect*/
        {"dstRectx0",            &args->dstRect.x0,            TYPE_USHORT,   0, LIMITS_MIN,  0,  0,  0,              0, SECTION_NONE},
        {"dstRecty0",            &args->dstRect.y0,            TYPE_USHORT,   0, LIMITS_MIN,  0,  0,  0,              0, SECTION_NONE},
        {"dstRectx1",            &args->dstRect.x1,            TYPE_USHORT,   0, LIMITS_MIN,  0,  0,  0,              0, SECTION_NONE},
        {"dstRecty1",            &args->dstRect.y1,            TYPE_USHORT,   0, LIMITS_MIN,  0,  0,  0,              0, SECTION_NONE},

        /*NvMediaLdcFilter*/
        {"filter",               &args->filter,                TYPE_UINT,     0, LIMITS_BOTH, 0,  2,  0,              0, SECTION_NONE},

#if !NV_IS_SAFETY
        /*NvMediaLdcLensDistortion*/
        {"model",                &distortion->model,           TYPE_UINT,     0, LIMITS_BOTH, 0,  4,  0,              0, SECTION_NONE},
        {"k1",                   &distortion->k1,              TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"k2",                   &distortion->k2,              TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"p1",                   &distortion->p1,              TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"p2",                   &distortion->p2,              TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"k3",                   &distortion->k3,              TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"k4",                   &distortion->k4,              TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"k5",                   &distortion->k5,              TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"k6",                   &distortion->k6,              TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},

        /*NvMediaLdcCameraIntrinsic*/
        {"fx",                   &Kin->matrixCoeffs[0][0],     TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"fy",                   &Kin->matrixCoeffs[1][1],     TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"cx",                   &Kin->matrixCoeffs[0][2],     TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"cy",                   &Kin->matrixCoeffs[1][2],     TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},

        /*NvMediaLdcCameraExtrinsic*/
        /*Rotation Matrix*/
        {"R00",                  &X->R[0][0],                  TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"R01",                  &X->R[0][1],                  TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"R02",                  &X->R[0][2],                  TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"R10",                  &X->R[1][0],                  TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"R11",                  &X->R[1][1],                  TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"R12",                  &X->R[1][2],                  TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"R20",                  &X->R[2][0],                  TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"R21",                  &X->R[2][1],                  TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"R22",                  &X->R[2][2],                  TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        /*Translation vector*/
        {"T0",                   &X->T[0],                     TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"T1",                   &X->T[1],                     TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"T2",                   &X->T[2],                     TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},

        /*NvMediaLdcCameraIntrinsic*/
        {"targetKfx",            &Kout->matrixCoeffs[0][0],    TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"targetKfy",            &Kout->matrixCoeffs[1][1],    TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"targetKcx",            &Kout->matrixCoeffs[0][2],    TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"targetKcy",            &Kout->matrixCoeffs[1][2],    TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
#endif

        /*ParamName,             &args->variableName,          paramType,     D, LimitType,   Mn, Mx, CharSize,       p2C, section   */

        /*Perspective Matrix*/
        {"ptMatrix00",           &ipt->matrixCoeffs[0][0],     TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"ptMatrix01",           &ipt->matrixCoeffs[0][1],     TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"ptMatrix02",           &ipt->matrixCoeffs[0][2],     TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"ptMatrix10",           &ipt->matrixCoeffs[1][0],     TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"ptMatrix11",           &ipt->matrixCoeffs[1][1],     TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"ptMatrix12",           &ipt->matrixCoeffs[1][2],     TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"ptMatrix20",           &ipt->matrixCoeffs[2][0],     TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"ptMatrix21",           &ipt->matrixCoeffs[2][1],     TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"ptMatrix22",           &ipt->matrixCoeffs[2][2],     TYPE_FLOAT,    0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},

        /*ParamName,             &args->variableName,          paramType,     D, LimitType,   Mn, Mx, CharSize,       p2C, section   */

        /*Region Config*/
        {"numHorRegion",         &region->numRegionsX,         TYPE_UCHAR,    0, LIMITS_BOTH, 1,  4, 0,               0, SECTION_NONE},
        {"numVerRegion",         &region->numRegionsY,         TYPE_UCHAR,    0, LIMITS_BOTH, 1,  4, 0,               0, SECTION_NONE},
        {"horRegionWidth0",      &region->regionWidth[0],      TYPE_USHORT,   0, LIMITS_MIN,  0,  1, 0,               0, SECTION_NONE},
        {"horRegionWidth1",      &region->regionWidth[1],      TYPE_USHORT,   0, LIMITS_MIN,  0,  1, 0,               0, SECTION_NONE},
        {"horRegionWidth2",      &region->regionWidth[2],      TYPE_USHORT,   0, LIMITS_MIN,  0,  1, 0,               0, SECTION_NONE},
        {"horRegionWidth3",      &region->regionWidth[3],      TYPE_USHORT,   0, LIMITS_MIN,  0,  1, 0,               0, SECTION_NONE},
        {"verRegionHeight0",     &region->regionHeight[0],     TYPE_USHORT,   0, LIMITS_MIN,  0,  1, 0,               0, SECTION_NONE},
        {"verRegionHeight1",     &region->regionHeight[1],     TYPE_USHORT,   0, LIMITS_MIN,  0,  1, 0,               0, SECTION_NONE},
        {"verRegionHeight2",     &region->regionHeight[2],     TYPE_USHORT,   0, LIMITS_MIN,  0,  1, 0,               0, SECTION_NONE},
        {"verRegionHeight3",     &region->regionHeight[3],     TYPE_USHORT,   0, LIMITS_MIN,  0,  1, 0,               0, SECTION_NONE},
        {"log2horSpace0",        &region->controlPointXSpacingLog2[0], TYPE_USHORT,   0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"log2horSpace1",        &region->controlPointXSpacingLog2[1], TYPE_USHORT,   0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"log2horSpace2",        &region->controlPointXSpacingLog2[2], TYPE_USHORT,   0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"log2horSpace3",        &region->controlPointXSpacingLog2[3], TYPE_USHORT,   0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"log2verSpace0",        &region->controlPointYSpacingLog2[0], TYPE_USHORT,   0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"log2verSpace1",        &region->controlPointYSpacingLog2[1], TYPE_USHORT,   0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"log2verSpace2",        &region->controlPointYSpacingLog2[2], TYPE_USHORT,   0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"log2verSpace3",        &region->controlPointYSpacingLog2[3], TYPE_USHORT,   0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},

        /*ParamName,             &args->variableName,          paramType,     D, LimitType,   Mn, Mx, CharSize,       p2C, section   */

        {"bitMaskFile",          &args->bitMaskFile,           TYPE_CHAR_ARR, 0, LIMITS_NONE, 0,  0, FILE_NAME_SIZE,  0, SECTION_NONE},

        /*NvMediaLdcMaskMapParameters*/
        {"bitMaskWidth",         &maskMap->width,              TYPE_USHORT,   0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"bitMaskHeight",        &maskMap->height,             TYPE_USHORT,   0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"maskedPixelFillColor", &maskMap->useMaskColor,       TYPE_UINT,     0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"maskY",                &maskMap->maskColorY,         TYPE_FLOAT,    0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"maskU",                &maskMap->maskColorU,         TYPE_FLOAT,    0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"maskV",                &maskMap->maskColorV,         TYPE_FLOAT,    0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},

        /*TNR*/
        {"spatialSigmaLuma",     &tnr->spatialSigmaLuma,       TYPE_USHORT,   0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"spatialSigmaChroma",   &tnr->spatialSigmaChroma,     TYPE_USHORT,   0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"rangeSigmaLuma",       &tnr->rangeSigmaLuma,         TYPE_USHORT,   0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"rangeSigmaChroma",     &tnr->rangeSigmaChroma,       TYPE_USHORT,   0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},
        {"sadMultiplier",        &tnr->sadMultiplier,          TYPE_FLOAT,    0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"sadWeightLuma",        &tnr->sadWeightLuma,          TYPE_FLOAT,    0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"alphaSmoothEnable",    &tnr->alphaSmoothEnable,      TYPE_UINT,     0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"alphaIncreaseCap",     &tnr->alphaIncreaseCap,       TYPE_FLOAT,    0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"alphaScaleIIR",        &tnr->alphaScaleIIR,          TYPE_FLOAT,    0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"alphaMaxLuma",         &tnr->alphaMaxLuma,           TYPE_FLOAT,    0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"alphaMinLuma",         &tnr->alphaMinLuma,           TYPE_FLOAT,    0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"alphaMaxChroma",       &tnr->alphaMaxChroma,         TYPE_FLOAT,    0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"alphaMinChroma",       &tnr->alphaMinChroma,         TYPE_FLOAT,    0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"betaX1",               &tnr->betaX1,                 TYPE_FLOAT,    0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"betaX2",               &tnr->betaX2,                 TYPE_FLOAT,    0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"minBeta",              &tnr->minBeta,                TYPE_FLOAT,    0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},
        {"maxBeta",              &tnr->maxBeta,                TYPE_FLOAT,    0, LIMITS_BOTH, 0,  1, 0,               0, SECTION_NONE},

        /*NvMediaLdcChecksumMode*/
        {"checksumMode",         &args->checksumMode,          TYPE_UINT,     0, LIMITS_BOTH, 0,  1,  0,              0, SECTION_NONE},

        /*ParamName,             &args->variableName,          paramType,     D, LimitType,   Mn, Mx, CharSize,       p2C, section   */

        /*WarpMap*/
        {"warpMapFile",          &args->warpMapFile,           TYPE_CHAR_ARR, 0, LIMITS_NONE, 0,  0, FILE_NAME_SIZE,  0, SECTION_NONE},

        {"numControlPoints",     &args->numControlPoints,      TYPE_UINT,     0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE},

        /*ParamName,             &args->variableName,          paramType,     D, LimitType,   Mn, Mx, CharSize,       p2C, section   */

        {"numFrames",            &args->numFrames,             TYPE_UINT,     1, LIMITS_MIN,  1,  0,  0,              0, SECTION_NONE},

        {"inputFile",            &args->inFile,                TYPE_CHAR_ARR, 0, LIMITS_NONE, 0,  0,  FILE_NAME_SIZE, 0, SECTION_NONE},
        {"outputFile",           &args->outFile,               TYPE_CHAR_ARR, 0, LIMITS_NONE, 0,  0,  FILE_NAME_SIZE, 0, SECTION_NONE},
        {"xSobelFile",           &args->xSobelFile,            TYPE_CHAR_ARR, 0, LIMITS_NONE, 0,  0,  FILE_NAME_SIZE, 0, SECTION_NONE},
        {"xSobelDSFile",         &args->xSobelDSFile,          TYPE_CHAR_ARR, 0, LIMITS_NONE, 0,  0,  FILE_NAME_SIZE, 0, SECTION_NONE},

         /*End of the array */
        {NULL,                   NULL,                         TYPE_UINT,     0, LIMITS_NONE, 0,  0, 0,               0, SECTION_NONE}
    };

    for (i = 1; i < argc; i++) {
        /* check if this is the last argument*/
        bLastArg = ((argc - i) == 1);

        /* check if there is data available to be parsed following the option*/
        bDataAvailable = (!bLastArg) && !(argv[i+1][0] == '-');

        if (argv[i][0] == '-') {
            if (strcmp(&argv[i][1], "h") == 0) {
                return -1;
            }
            else if (strcmp(&argv[i][1], "v") == 0) {
                if (bDataAvailable) {
                    args->logLevel = atoi(argv[++i]);
                    if (args->logLevel < LEVEL_ERR || args->logLevel > LEVEL_DBG) {
                        LOG_ERR("ParseArgs: Invalid logging level chosen (%d).\n", args->logLevel);
                        LOG_ERR("           default logging level is LEVEL_ERR \n");
                        args->logLevel = LEVEL_ERR;
                    }
                }
                SetLogLevel(args->logLevel);
            }
            else if (strcmp(&argv[i][1], "cf") == 0) {
                /* Init Parser Map*/
                LOG_INFO("ParseArgs: Initializing Parser Params Map\n");
                status = ConfigParser_InitParamsMap(paramsMap);
                if (status != NVMEDIA_STATUS_OK) {
                    LOG_ERR("ERR: ConfigParser_InitParamsMap failed! status:%x\n", status);
                    return -1;
                }

                filename = argv[++i];
                if (!filename) {
                    LOG_ERR("ERR: Invalid config file name\n");
                    return -1;
                }

                LOG_INFO("ParseArgs: Parsing config file %s\n", filename);
                status = ConfigParser_ParseFile(paramsMap, 1, sectionsMap, filename);
                if (status != NVMEDIA_STATUS_OK) {
                    LOG_ERR("ERR: Failed to parse config file. status:%x\n", status);
                    return -1;
                }
            }
            else
            {
                LOG_ERR("ERR: option %c is not supported and ignored.\n", argv[i][1]);
            }
        }
    }

    LOG_INFO("ParseArgs: Validating params from config file\n");
    status = ConfigParser_ValidateParams(paramsMap, sectionsMap);
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("ERR: Some of the params in config file are invalid.\n");
        /* Ignore the failure and let library handle any exceptions */
    }

    if (args->logLevel > LEVEL_ERR) {
        LOG_INFO("ParseArgs: Printing params from config file\n");
        ConfigParser_DisplayParams(paramsMap, sectionsMap);
    }

    return 0;
}

