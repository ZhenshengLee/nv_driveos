/* Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "commandline.h"

void PrintUsage()
{
    LOG_MSG("nvmedia_ofa_flow_6x is opticalFlow estimation test program on OFA engine \n");
    LOG_MSG("Usage: nvm_iofa_flow_sci [options] \n");
    LOG_MSG("Required Arguments: \n");
    LOG_MSG("-f   [input file]     Input file name (must be YV12 or IYUV format) \n");
    LOG_MSG("-res [WxH]            Input file resolution (eg. 800x480) \n");
    LOG_MSG("Optional Arguments: \n");
    LOG_MSG("-h                    Prints usage \n");
    LOG_MSG("-version              Get NvMediaIOFA Major/Minor version \n");
    LOG_MSG("-chromaFormat         Chroma format IDC of Input YUV. Default is 1\n");
    LOG_MSG("                      0: 400\n");
    LOG_MSG("                      1: 420 (default)\n");
    LOG_MSG("                      2: 422\n");
    LOG_MSG("                      3: 444\n");
    LOG_MSG("-pydChromaFormat      Chroma format IDC of Input Pyramid except bottom layer (input frame). Default is 0\n");
    LOG_MSG("                      Bottom layer pyramid use chromaFormat option\n");
    LOG_MSG("                      0: 400 (default)\n");
    LOG_MSG("                      1: 420\n");
    LOG_MSG("                      2: 422\n");
    LOG_MSG("                      3: 444\n");
    LOG_MSG("-o   [output file]    Output file name \n");
    LOG_MSG("-co  [cost file]      Cost output file name \n");
    LOG_MSG("-frames [n]           Number of frames used for opticalflow estimation. Default: all frames \n");
    LOG_MSG("-flowcrcgen [txt]     Generate FLOW CRC values \n");
    LOG_MSG("-flowcrcchk [txt]     Check FLOW CRC values \n");
    LOG_MSG("-costcrcgen [txt]     Generate COST CRC values \n");
    LOG_MSG("-costcrcchk [txt]     Check COST CRC values \n");
    LOG_MSG("-gridSize             Grid size which ofa will consider for processing. \n");
    LOG_MSG("                      0: Grid size 1x1 \n");
    LOG_MSG("                      1: Grid Size 2x2 \n");
    LOG_MSG("                      2: Grid Size 4x4 \n");
    LOG_MSG("                      3: Grid Size 8x8 \n");
    LOG_MSG("                      Different values for each pyramid level can be entered \n");
    LOG_MSG("-v   [level]          Logging Level = 0(Errors), 1(Warnings), 2(Info), 3(Debug)\n");
    LOG_MSG("-bitDepth             Bitdepth for input: valid arguments 8/10/12/16, Default 8 \n");
    LOG_MSG("-roiMode              Enable ROI mode \n");
    LOG_MSG("                      0: Disable ROI mode (default) \n");
    LOG_MSG("                      1: Enable ROI mode \n");
    LOG_MSG("-roiFile              file containing frame number, total ROI number and coordinates of ROIs \n");
    LOG_MSG("-inputBuffering       input buffering valid range 1 to 8, Default 5 \n");
    LOG_MSG("-timeout              timeout value to wait for OFA operation to finish. default - 1000ms\n");
    LOG_MSG("-backwardRef          use backward referencing for flow\n");
    LOG_MSG("-pass                 num of passes valid range 1 to 3\n");
    LOG_MSG("-profile              enable profiling support\n");
    LOG_MSG("                      0: disable profiling\n");
    LOG_MSG("                      1: enable profiling\n");
    LOG_MSG("                      2: enable profiling with sync/blocking API mode \n");
    LOG_MSG("                      3: enable profiling with nvplayfair API\n");
    LOG_MSG("-preset               ofa preset used for flow processing\n");
    LOG_MSG("                      0: high quality preset\n");
    LOG_MSG("                      1: high performance preset\n");
    LOG_MSG("-frameIntervalInMS    time interval between two input frames pairs in ms\n");
    LOG_MSG("-p1                   penalty1 value in cost function. Different values for each pyramid level can be entered.\n");
    LOG_MSG("-p2                   penalty2 value in cost function. Different values for each pyramid level can be entered \n");
    LOG_MSG("-diag                 Enable diagonal path search in SGM: 0/1. Different values for each pyramid level can be entered \n");
    LOG_MSG("-adaptiveP2           Enable adaptive P2.Different values for each pyramid level can be entered \n");
    LOG_MSG("-alpha                alpha log value for adaptive P2.valid range 0 to 3.Different values for each pyramid level can be entered \n");
    LOG_MSG("-noop                 skip OFA HW processing and signal frame done \n");
    LOG_MSG("-prefence             enable prefence support \n");
    LOG_MSG("-profileStatsFilePath path to dump playfair profile stats file \n");
}

int ParseArgs(int argc, char *argv[], TestArgs *args)
{
    bool bLastArg = false;
    bool bDataAvailable = false;
    bool bHasInputFileName = false;
    bool bHasroiFileName = false;
    uint16_t i, j, k;

    //init flowCrcoption
    args->flowCrcoption.crcGenMode = false;
    args->flowCrcoption.crcCheckMode = false;
    //init costCrcoption
    args->costCrcoption.crcGenMode = false;
    args->costCrcoption.crcCheckMode = false;

    args->disableCostOut = 1U;
    args->chromaFormat = 1U;
    args->pydChromaFormat = 0U;
    args->version = true;

    args->bitDepth = 8;
    args->roiMode = 0;
    args->inputBuffering = 1;
    args->timeout = 1000;
    args->overrideParam = 0;
    args->numFrames = 10000;
    args->pydLevel = NVMEDIA_IOFA_MAX_PYD_LEVEL;

    args->preFetchInput = true;
    args->earlyFenceExpiry = true;

    for (i = 1; i < argc; i++)
    {
        // check if this is the last argument
        bLastArg = ((argc - i) == 1);

        // check if there is data available to be parsed following the option
        bDataAvailable = (!bLastArg) && !(argv[i+1][0] == '-');

        if (argv[i][0] == '-')
        {
            if (strcmp(&argv[i][1], "res") == 0)
            {
                if (!bDataAvailable || sscanf(argv[++i], "%dx%d", &args->width, &args->height) != 2)
                {
                    LOG_ERR("ERR: -res must be followed by resolution (e.g. 800x400) \n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "f") == 0)
            {
                if (bDataAvailable)
                {
                    args->inputFilename = argv[++i];
                    bHasInputFileName = true;
                }
                else
                {
                    LOG_ERR("ERR: -f must be followed by input file name \n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "o") == 0)
            {
                if (bDataAvailable)
                {
                    args->outputFilename = argv[++i];
                }
                else
                {
                    LOG_ERR("ERR: -o must be followed by output file name \n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "co") == 0)
            {
                if (bDataAvailable)
                {
                    args->costFilename = argv[++i];
                    args->disableCostOut = 0U;
                }
                else
                {
                    LOG_ERR("ERR: -co must be followed by cost file name \n");
                    return -1;
                }
            }
            else if(strcmp(&argv[i][1], "version") == 0)
            {
               args->version = true;
            }
            else if (strcmp(&argv[i][1], "inputBuffering") == 0)
            {
                if (bDataAvailable)
                {
                    args->inputBuffering = atoi(argv[++i]);
                }
                else
                {
                    LOG_ERR("ERR: -inputBuffering must be followed by integer \n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "chromaFormat") == 0)
            {
                if (bDataAvailable)
                {
                    args->chromaFormat = atoi(argv[++i]);
                }
                else
                {
                    LOG_ERR("ParseArgs: -chromaFormat must be followed by value \n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "pydChromaFormat") == 0)
            {
                if (bDataAvailable)
                {
                    args->pydChromaFormat = atoi(argv[++i]);
                }
                else
                {
                    LOG_ERR("ParseArgs: -pydChromaFormat must be followed by value \n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "timeout") == 0)
            {
                if (bDataAvailable)
                {
                    args->timeout = atoi(argv[++i]);
                }
                else
                {
                    LOG_ERR("ParseArgs: -timeout must be followed by timeout value in millisec \n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "h") == 0)
            {
                return -1;
            }
            else if (strcmp(&argv[i][1], "v") == 0)
            {
                if (bDataAvailable)
                {
                    args->logLevel = atoi(argv[++i]);
                    if (args->logLevel < LEVEL_ERR || args->logLevel > LEVEL_DBG)
                    {
                        LOG_ERR("ParseArgs: Invalid logging level chosen (%d). ", args->logLevel);
                        LOG_ERR("           Setting logging level to LEVEL_ERR (0)\n");
                    }
                }
                else
                {
                    args->logLevel = LEVEL_DBG; // Max logging level
                }
                SetLogLevel(args->logLevel);
            }
            else if (strcmp(&argv[i][1], "frames") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%d", &args->numFrames))
                {
                    LOG_ERR("ERR: -frames must be followed by frame count \n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "gridSize") == 0)
            {
                if (!bDataAvailable)
                {
                    LOG_ERR("ERR: -gridSize must be followed by ofa gridSize (0/1/2/3) \n");
                    return -1;
                }
                else
                {
                    j = 0;
                    while ((argv[i+1] != NULL) && (argv[i+1][0] != '-') && (j < NVMEDIA_IOFA_MAX_PYD_LEVEL))
                    {
                        sscanf(argv[++i], "%hhd", &args->gridsize[j++]);
                    }
                    if (j)  // if only 1 argument is provided for gridsize put same arg for all level
                    {
                        for (k = j; k < NVMEDIA_IOFA_MAX_PYD_LEVEL; k++)
                        {
                            args->gridsize[k] = args->gridsize[j-1];
                        }
                    }
                }
            }
            else if (strcmp(&argv[i][1], "bitDepth") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%d", &args->bitDepth))
                {
                    LOG_ERR("ERR: -bitDepth must be followed by bitdepth (8/10/12/16) \n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "roiMode") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%hhd", &args->roiMode))
                {
                    LOG_ERR("ERR: -roiMode should be followed by 0 or 1. \n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "roiFile") == 0)
            {
                if (bDataAvailable)
                {
                    args->roiFilename = argv[++i];
                    bHasroiFileName = true;
                }
                else
                {
                    LOG_ERR("ERR: -roiFile must be followed by roi file name \n");
                    return -1;
                }
            }
            else if(strcmp(&argv[i][1], "flowcrcchk") == 0)
            {
                // crc check
                if(bDataAvailable)
                {
                    args->flowCrcoption.crcCheckMode = true;
                    args->flowCrcoption.crcFilename = argv[++i];
                }
                else
                {
                    LOG_ERR("ParseArgs: -crcchk must be followed by crc file name \n");
                    return -1;
                }
            }
            else if(strcmp(&argv[i][1], "flowcrcgen") == 0)
            {
                // crc generate
                if(bDataAvailable)
                {
                    args->flowCrcoption.crcGenMode = true;
                    args->flowCrcoption.crcFilename = argv[++i];
                }
                else
                {
                    LOG_ERR("ParseArgs: -crcgen must be followed by crc file name \n");
                    return -1;
                }
            }
            else if(strcmp(&argv[i][1], "costcrcchk") == 0)
            {
                // crc check
                if(bDataAvailable)
                {
                    args->costCrcoption.crcCheckMode = true;
                    args->costCrcoption.crcFilename = argv[++i];
                }
                else
                {
                    LOG_ERR("ParseArgs: -crcchk must be followed by crc file name \n");
                    return -1;
                }
            }
            else if(strcmp(&argv[i][1], "costcrcgen") == 0)
            {
                // crc generate
                if(bDataAvailable)
                {
                    args->costCrcoption.crcGenMode = true;
                    args->costCrcoption.crcFilename = argv[++i];
                }
                else
                {
                    LOG_ERR("ParseArgs: -crcgen must be followed by crc file name \n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "p1") == 0)
            {
                if (!bDataAvailable)
                {
                    LOG_ERR("ERR: -p1 must be followed by penalty 1 \n");
                    return -1;
                }
                else
                {
                    j = 0;
                    while ((argv[i+1] != NULL) && (argv[i+1][0] != '-') && (j < NVMEDIA_IOFA_MAX_PYD_LEVEL))
                    {
                        sscanf(argv[++i], "%hhd", &args->p1[j++]);
                    }
                    if (j == 1U)  // if only 1 argument is provided for p1 put same arg for all level
                    {
                        for (j = 1; j < NVMEDIA_IOFA_MAX_PYD_LEVEL; j++)
                        {
                            args->p1[j] = args->p1[0];
                        }
                    }
                }
                args->overrideParam |= SGMPARAM_P1_OVERRIDE;
            }
            else if (strcmp(&argv[i][1], "p2") == 0)
            {
                if (!bDataAvailable)
                {
                    LOG_ERR("ERR: -p2 must be followed by penalty 2\n");
                    return -1;
                }
                else
                {
                    j = 0;
                    while ((argv[i+1] != NULL) && (argv[i+1][0] != '-') && (j < NVMEDIA_IOFA_MAX_PYD_LEVEL))
                    {
                        sscanf(argv[++i], "%hhd", &args->p2[j++]);
                    }
                    if (j == 1U)  // if only 1 argument is provided for p2 put same arg for all level
                    {
                        for (j = 1; j < NVMEDIA_IOFA_MAX_PYD_LEVEL; j++)
                        {
                            args->p2[j] = args->p2[0];
                        }
                    }
                }
                args->overrideParam |= SGMPARAM_P2_OVERRIDE;
            }
            else if (strcmp(&argv[i][1], "adaptiveP2") == 0)
            {
                if (!bDataAvailable)
                {
                    LOG_ERR("ERR: -adaptiveP2 must be followed 0 or 1 \n");
                    return -1;
                }
                else
                {
                    j = 0;
                    while ((argv[i+1] != NULL) && (argv[i+1][0] != '-') && (j < NVMEDIA_IOFA_MAX_PYD_LEVEL))
                    {
                        sscanf(argv[++i], "%hhd", &args->adaptiveP2[j++]);
                    }
                    if (j == 1U)  // if only 1 argument is provided for adaptiveP2 put same arg for all level
                    {
                        for (j = 1; j < NVMEDIA_IOFA_MAX_PYD_LEVEL; j++)
                        {
                            args->adaptiveP2[j] = args->adaptiveP2[0];
                        }
                    }
                }
                args->overrideParam |= SGMPARAM_ADAPTIVEP2_OVERRIDE;
            }
            else if (strcmp(&argv[i][1], "alpha") == 0)
            {
                if (!bDataAvailable)
                {
                    LOG_ERR("ERR: -alpha must be followed alpha value \n");
                    return -1;
                }
                else
                {
                    j = 0;
                    while ((argv[i+1] != NULL) && (argv[i+1][0] != '-') && (j < NVMEDIA_IOFA_MAX_PYD_LEVEL))
                    {
                        sscanf(argv[++i], "%hhd", &args->alpha[j++]);
                    }
                    if (j == 1U)  // if only 1 argument is provided for alpha put same arg for all level
                    {
                        for (j = 1; j < NVMEDIA_IOFA_MAX_PYD_LEVEL; j++)
                        {
                            args->alpha[j] = args->alpha[0];
                        }
                    }
                }
                args->overrideParam |= SGMPARAM_ALPHA_OVERRIDE;
            }
            else if (strcmp(&argv[i][1], "d") == 0)
            {
                if (!bDataAvailable)
                {
                    LOG_ERR("ERR: -d must be followed 0 to disable diagonal or 1 to enable daigonal search \n");
                    return -1;
                }
                else
                {
                    j = 0;
                    while ((argv[i+1] != NULL) && (argv[i+1][0] != '-') && (j < NVMEDIA_IOFA_MAX_PYD_LEVEL))
                    {
                        sscanf(argv[++i], "%hhd", &args->diagonalMode[j++]);
                    }
                    if (j == 1U)  // if only 1 argument is provided for d put same arg for all level
                    {
                        for (j = 1; j < NVMEDIA_IOFA_MAX_PYD_LEVEL; j++)
                        {
                            args->diagonalMode[j] = args->diagonalMode[0];
                        }
                    }
                }
                args->overrideParam |= SGMPARAM_DIAGONAL_OVERRIDE;
            }
            else if (strcmp(&argv[i][1], "p") == 0)
            {
                if (!bDataAvailable)
                {
                    LOG_ERR("ERR: -p must be followed by number of passes (1/2/3) \n");
                    return -1;
                }
                else
                {
                    j = 0;
                    while ((argv[i+1] != NULL) && (argv[i+1][0] != '-') && (j < NVMEDIA_IOFA_MAX_PYD_LEVEL))
                    {
                        sscanf(argv[++i], "%hhd", &args->numPasses[j++]);
                    }
                    if (j == 1U)  // if only 1 argument is provided for numPasses put same arg for all level
                    {
                        for (j = 1; j < NVMEDIA_IOFA_MAX_PYD_LEVEL; j++)
                        {
                            args->numPasses[j] = args->numPasses[0];
                        }
                    }
                }
                args->overrideParam |= SGMPARAM_NUMPASSES_OVERRIDE;
            }
            else if(strcmp(&argv[i][1], "backwardRef") == 0)
            {
               args->backwardRef = true;
            }
            else if (strcmp(&argv[i][1], "pydLevel") == 0)
            {
                if (bDataAvailable)
                {
                    args->pydLevel = atoi(argv[++i]);
                }
                else
                {
                    LOG_ERR("ParseArgs: -pydLevel must be followed by number of pyramid level \n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "frameIntervalInMS") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%d", &args->frameIntervalInMS))
                {
                    LOG_ERR("ERR: -frameIntervalInMS must be followed by time intervale between input frames\n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "profile") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%d", &args->profile))
                {
                    LOG_ERR("ERR: -profile must be followed by profile value\n");
                    return -1;
                }
            }
	    else if (strcmp(&argv[i][1], "preset") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%d", &args->preset))
                {
                    LOG_ERR("ERR: -preset must be followed by ofa preset\n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "noop") == 0)
            {
                args->noopMode = true;
            }
            else if (strcmp(&argv[i][1], "prefence") == 0)
            {
                args->preFetchInput = false;
                args->earlyFenceExpiry = false;
            }
            else if (strcmp(&argv[i][1], "profileStatsFilePath") == 0)
            {
                if (bDataAvailable)
                {
                    args->profileStatsFilePath = argv[++i];
                }
                else
                {
                    LOG_ERR("ERR: -profileStatsFilePath must be followed by filepath \n");
                    return -1;
                }
            }
            else
            {
                LOG_ERR("ERR: option %s is not supported and returning \n", &argv[i][1]);
                return -1;
            }
        }
    }

    if(!args->width || !args->height)
    {
        LOG_ERR("ERR: Input resolution must be set \n");
        return -1;
    }
    if(!bHasInputFileName)
    {
        LOG_ERR("ERR: No Input file name was given\n");
        return -1;
    }
    if ((args->inputBuffering == 0) || (args->inputBuffering > NVMEDIA_IOFA_BUFFERING))
    {
        LOG_ERR("ERR: inputBuffering parameter is not within valid range 1 to 8\n");
        return -1;
    }
    if ((args->pydLevel > NVMEDIA_IOFA_MAX_PYD_LEVEL) || (args->pydLevel == 0))
    {
        LOG_ERR("ERR: pyd level parameter is not within valid range 1 to 5\n");
        return -1;
    }
    if((args->roiMode == 1) && (!bHasroiFileName))
    {
        LOG_ERR("ERR: No ROI file name is given\n");
        return -1;
    }

    // Disable output and cost file write, CRC check and generation
    // when frame interval time is specified. This will help to reduce other overheads.
    if (args->frameIntervalInMS != 0)
    {
        args->costCrcoption.crcGenMode     = false;
        args->costCrcoption.crcCheckMode   = false;
        args->flowCrcoption.crcGenMode   = false;
        args->flowCrcoption.crcCheckMode = false;
        args->outputFilename = NULL;
        args->costFilename   = NULL;
    }
    if (args->profile == 3)
    {
        args->profile = 0;
        args->inputBuffering = 1;
        args->playfair = true;
        args->preFetchInput = true;
        args->earlyFenceExpiry = false;
    }

    return 0;
}

