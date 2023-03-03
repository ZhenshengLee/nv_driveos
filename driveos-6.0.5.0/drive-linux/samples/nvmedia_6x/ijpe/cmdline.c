/*
 * Copyright (c) 2014-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>

#include "cmdline.h"
#include "config_parser.h"
#include "log_utils.h"
#include "scibuf_utils.h"

void PrintUsage()
{
    LOG_MSG("Usage: nvm_ijpe_sci [options]\n");
    LOG_MSG("Options:\n");
    LOG_MSG("-h                         Prints usage\n");
    LOG_MSG("-f [file]                  Input image file. \n");
    LOG_MSG("                           Input file should be in YUV420 with UV order\n");
    LOG_MSG("-fr [WxH]                  Input file resolution\n");
    LOG_MSG("-q  [value]                JPEG Encoder quality (1..100)\n");
    LOG_MSG("-of [file]                 Output JPEG file. \n");
    LOG_MSG("-crcgen [crcs.txt]         Generate CRC values\n");
    LOG_MSG("-crcchk [crcs.txt]         Check CRC values\n");
    LOG_MSG("-v  [level]                Logging Level = 0(Errors), 1(Warnings), 2(Info), 3(Debug)\n");
    LOG_MSG("-HuffTable   [file]        input cfg file with huffman table\n");
    LOG_MSG("-QuantTable  [file]        input cfg file with quant table\n");
    LOG_MSG("-hwid [0/1/2]]             NVJPG HW instance id, 0-NVJPG0, 1-NVJPG1, 2-Auto mode\n");
}

int ParseArgs(int argc, char **argv, TestArgs *args)
{
    int i;
    int bLastArg = 0;
    int bDataAvailable = 0;
    // Defaults
    args->maxOutputBuffering = 5;
    args->quality = 50;
    //init crcoption
    args->crcoption.crcGenMode = false;
    args->crcoption.crcCheckMode = false;

    if((argc == 2 && (strcasecmp(argv[1], "-h") == 0)) || argc < 2) {
        PrintUsage();
        exit(0);
    }

    for(i = 0; i < argc; i++) {
        bLastArg = ((argc - i) == 1);

        // check if there is data available to be parsed following the option
        bDataAvailable = (!bLastArg) && !(argv[i+1][0] == '-');

        if(argv[i][0] == '-') {
            if(strcmp(&argv[i][1], "h") == 0) {
                PrintUsage();
                exit(0);
            } else if(strcmp(&argv[i][1], "v") == 0) {
                args->logLevel = LEVEL_DBG;
                if(bDataAvailable) {
                    args->logLevel = atoi(argv[++i]);
                    if(args->logLevel < LEVEL_ERR || args->logLevel > LEVEL_DBG) {
                        LOG_INFO("MainParseArgs: Invalid logging level chosen (%d). ", args->logLevel);
                        LOG_INFO("           Setting logging level to LEVEL_ERR (0)\n");
                        args->logLevel = LEVEL_ERR;
                    }
                }
                SetLogLevel(args->logLevel);
            } else if(strcmp(&argv[i][1], "f") == 0) {
                // Input file name
                if(bDataAvailable) {
                    args->infile = argv[++i];
                } else {
                    LOG_ERR("ParseArgs: -f must be followed by input file name\n");
                    return 0;
                }
            } else if(strcmp(&argv[i][1], "of") == 0) {
                // Output file name
                if(bDataAvailable) {
                    args->outfile = argv[++i];
                } else {
                    LOG_ERR("ParseArgs: -of must be followed by output file name\n");
                    return 0;
                }
            } else if(strcmp(&argv[i][1], "fr") == 0) {
                if(bDataAvailable) {
                    if((sscanf(argv[++i], "%ux%u", &args->inputWidth, &args->inputHeight) != 2)) {
                        LOG_ERR("ParseArgs: Bad output resolution: %s\n", argv[i]);
                        return 0;
                    }
                } else {
                    LOG_ERR("ParseArgs: -fr must be followed by resolution\n");
                    return 0;
                }
            } else if(strcmp(&argv[i][1], "q") == 0) {
                if(bDataAvailable) {
                    args->quality = atoi(argv[++i]);
                    if(args->quality < 1 || args->quality > 100) {
                        LOG_ERR("MainParseArgs: Invalid quality (%d). ", args->quality);
                        LOG_ERR("               Quality level should be in [1..100]\n");
                        return 0;
                    }
                } else {
                    LOG_ERR("ParseArgs: -q must be followed by quality [1..100]\n");
                    return 0;
                }
            } else if(strcmp(&argv[i][1], "crcchk") == 0) {
                // crc check
                if(bDataAvailable) {
                    args->crcoption.crcCheckMode = true;
                    args->crcoption.crcFilename = argv[++i];
                } else {
                    LOG_ERR("ParseArgs: -crcchk must be followed by crc file name\n");
                    return 0;
                }
            } else if(strcmp(&argv[i][1], "crcgen") == 0) {
                // crc generate
                if(bDataAvailable) {
                    args->crcoption.crcGenMode = true;
                    args->crcoption.crcFilename = argv[++i];
                } else {
                    LOG_ERR("ParseArgs: -crcgen must be followed by crc file name\n");
                    return 0;
                }
            } else if(strcmp(&argv[i][1], "HuffTable") == 0) {
                // Input file name
                if(bDataAvailable) {
                    args->huffTable = true;
                    args->huffFileName = argv[++i];
                } else {
                    LOG_ERR("ParseArgs: -f must be followed by huffman table cfg file name\n");
                    return 0;
                }
            } else if(strcmp(&argv[i][1], "QuantTable") == 0) {
                // Input file name
                if(bDataAvailable) {
                    args->quantTable = true;
                    args->quantFileName = argv[++i];
                } else {
                    LOG_ERR("ParseArgs: -f must be followed by quant table cfg file name\n");
                    return 0;
                }
            } else if(strcmp(&argv[i][1], "hwid") == 0) {
                // crc generate
                if(bDataAvailable) {
                    unsigned int hwid = atoi(argv[++i]);
                    if (hwid == 0U) {
                        args->instanceId = NVMEDIA_JPEG_INSTANCE_0;
                    }
                    else if (hwid == 1U) {
                        args->instanceId = NVMEDIA_JPEG_INSTANCE_1;
                    }
                    else if (hwid == 2U) {
                        args->instanceId = NVMEDIA_JPEG_INSTANCE_AUTO;
                    }
                    else {
                        args->instanceId = NVMEDIA_JPEG_INSTANCE_0;
                        LOG_ERR("ParseArgs: Supported NVJPG HW instance ids are 0/1/2, forcing default id 0...\n");
                    }
                } else {
                    LOG_ERR("ParseArgs: -hwid must be followed by NVJPG HW instance id 0/1/2\n");
                    return 0;
                }
            }
        }

    }

    if (!args->infile || !args->outfile || !args->inputWidth || !args->inputHeight) {
        LOG_ERR("ParseArgs: command line not complete\n");
        PrintUsage();
        return 0;
    }

    //print command line
    LOG_MSG("command: ");
    for(i = 0; i<argc; i++)
    {
        LOG_MSG("%s ",argv[i]);
    }
    LOG_MSG("\n");

    return 1;
}
