/*
 * Copyright (c) 2013-2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "cmdline.h"
#include "log_utils.h"
#include "misc_utils.h"

void PrintUsage()
{
    LOG_MSG("Usage: nvm_ide_sci [options]\n");
    LOG_MSG("Options:\n");
    LOG_MSG("-h                Prints usage\n");
    LOG_MSG("-c  [codec type]  Numeric codec type  = 1(MPEG), 2(MPEG4), 3(VC1), 4(H264) 5(VP8) 6(H265) 7(VP9) 8(AV1 - supported on T234 and higher chips only )\n");
    LOG_MSG("-c  [codec type]  Text codec type  = mpeg, mpeg4, vc1, h264, vp8, h265, vp9, av1\n");
    LOG_MSG("-f  [input file]  Input file name\n");
    LOG_MSG("-t                Show decode timing info (do not use with -prof)\n");
    LOG_MSG("-r  [value]       Frame rate\n");
    LOG_MSG("-operating_point <0/1/2/3> AV1Specific option : Option to choose the operating point in case of AV1 SVC stream\n");
    LOG_MSG("-output_all_layers 0/1> AV1Specific option : Option to dump all layers or only Highest resolution Enhancement layer frames in case of AV1 SVC stream.\n");
    LOG_MSG("-annexBStream <0/1> AV1Specific option : Option to specify AnnexB kind of stream \n");
    LOG_MSG("-set_max_res <0/1>:  set as 1 for setting max resolution supported for decoder create \n");
    LOG_MSG("-n  [frames]      Number of frames to decode\n");
    LOG_MSG("-l  [loops]       Number of loops of playback\n");
    LOG_MSG("                  -1 for infinite loops of playback (default: 1)\n");
    LOG_MSG("-s  [output file] Output YUV File name to save\n");
    LOG_MSG("-crc       [gen/chk][CRC file]  Enable CRC checks (chk) or generate CRC file (gen)\n");
    LOG_MSG("-crcpath   [new path]           New Path for crc picture\n");
    LOG_MSG("-cropcrc                        CRC will be calculated on actual resolution\n");
    LOG_MSG("-a  [value]       Aspect ratio\n");
    LOG_MSG("-prof             Enable per frame profiling at decoder\n");
    LOG_MSG("-id [instance id] Decoder instance Id. 0(Instance 0), 1(Instance 1)\n");
    LOG_MSG("-alternateCreateAPI        Use alternate create API of IDP element \n");
    LOG_MSG("-v  [level]       Logging Level = 0(Errors), 1(Warnings), 2(Info), 3(Debug)\n");
}

int ParseArgs(int argc, char **argv, TestArgs *args)
{
    bool bLastArg = false;
    bool bDataAvailable = false;
    bool bHasCodecType = false;
    bool bHasFileName = false;
    int  i;
    FILE *file;

    /* app defaults */
    args->loop = 1;
    args->aspectRatio = 0.0;
    args->depth = 1;
    args->windowId = 1;
    args->instanceId = 0;
    args->checkCRC = false;
    args->generateCRC = false;
    args->cropCRC = false;
    args->crcFilePath = '\0';
    args->decProfiling = false;

    SetLogLevel(LEVEL_ERR); // Default logging level

    for (i = 1; i < argc; i++) {
        // check if this is the last argument
        bLastArg = ((argc - i) == 1);

        // check if there is data available to be parsed following the option
        bDataAvailable = (!bLastArg) && !(argv[i+1][0] == '-');

        if (argv[i][0] == '-') {
            if (strcmp(&argv[i][1], "h") == 0) {
                PrintUsage();
                return 1;
            }
            else if (strcmp(&argv[i][1], "c") == 0) {
                if (bDataAvailable) {
                    struct {
                        char *name;
                        NvMediaVideoCodec codec;
                    } codecs[] = {
                        { "mpeg", NVMEDIA_VIDEO_CODEC_MPEG2 },
                        { "mpeg4", NVMEDIA_VIDEO_CODEC_MPEG4 },
                        { "vc1", NVMEDIA_VIDEO_CODEC_VC1 },
                        { "h264", NVMEDIA_VIDEO_CODEC_H264 },
                        { "vp8", NVMEDIA_VIDEO_CODEC_VP8 },
                        { "h265", NVMEDIA_VIDEO_CODEC_HEVC },
                        { "vp9", NVMEDIA_VIDEO_CODEC_VP9 },
                        { "av1", NVMEDIA_VIDEO_CODEC_AV1 },
                        { NULL, NVMEDIA_VIDEO_CODEC_H264 }
                    };
                    char *arg = argv[++i];
                    if (*arg >= '1' && *arg <= '8') {
                        args->eCodec = codecs[atoi(arg) - 1].codec;
                        bHasCodecType = true;
                    } else {
                        int j;
                        for(j = 0; codecs[j].name; j++) {
                            if (!strcasecmp(arg, codecs[j].name)) {
                                args->eCodec = codecs[j].codec;
                                bHasCodecType = true;
                                break;
                            }
                        }
                        if (!bHasCodecType) {
                            LOG_ERR("ParseArgs: -c must be followed by codec type\n");
                            return -1;
                        }
                    }
                } else {
                    LOG_ERR("ParseArgs: -c must be followed by codec type\n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "f") == 0) {
                if (bDataAvailable) {
                    args->filename = argv[++i];
                    bHasFileName = true;
                    struct stat st;
                    file = fopen(args->filename, "rb");
                    if (!file) {
                        LOG_ERR("ParseArgs: failed to open stream %s\n", args->filename);
                        return -1;
                    }

                    if (stat(args->filename, &st) == -1) {
                        fclose(file);
                        LOG_ERR("ParseArgs: failed to stat stream %s\n", args->filename);
                        return -1;
                    }
                    args->fileSize = st.st_size;
                    fclose(file);
                } else {
                    LOG_ERR("ParseArgs: -f must be followed by file name\n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "t") == 0) {
                args->showDecodeTimimg = true;
            }
            else if (strcmp(&argv[i][1], "v") == 0) {
                if (bDataAvailable) {
                    char *arg = argv[++i];
                    args->logLevel = atoi(arg);
                    if (args->logLevel < LEVEL_ERR || args->logLevel > LEVEL_DBG) {
                        LOG_ERR("ParseArgs: Invalid logging level chosen (%d). ", args->logLevel);
                        LOG_ERR("           Setting logging level to LEVEL_ERR (0)\n");
                    }
                } else {
                    args->logLevel = LEVEL_DBG; // Max logging level
                }
                SetLogLevel((enum LogLevel)args->logLevel);
            }
            else if (strcmp(&argv[i][1], "r") == 0) {
                if (bDataAvailable) {
                    float framerate;
                    if (sscanf(argv[++i], "%f", &framerate)) {
                        args->frameTimeUSec = 1000000.0 / framerate;
                    } else {
                        LOG_ERR("ParseArgs: Invalid frame rate encountered (%s)\n", argv[i]);
                    }
                } else {
                    LOG_ERR("ParseArgs: -r must be followed by frame rate\n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "n") == 0) {
                if (bDataAvailable) {
                    int decodeCount;
                    if (sscanf(argv[++i], "%d", &decodeCount)) {
                        args->numFramesToDecode = decodeCount;
                    } else {
                        LOG_ERR("ParseArgs: -n must be followed by decode frame count\n");
                    }
                } else {
                    LOG_ERR("ParseArgs: -n must be followed by frame count\n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "s") == 0) {
                if (bDataAvailable) {
                    args->OutputYUVFilename = argv[++i];
                } else {
                    LOG_ERR("ParseArgs: -s must be followed by file name\n");
                    return -1;
                }
            }
            else if(strcmp(&argv[i][1], "cropcrc") == 0) {
                args->cropCRC = true;
            }
            else if(strcmp(&argv[i][1], "alternateCreateAPI") == 0) {
                args->alternateCreateAPI = true;
            }
            else if(strcmp(&argv[i][1], "annexBStream") == 0) {
                args->setAnnexBStream = true;
                args->av1annexBStream = (uint8_t)atoi(argv[++i]);
            }
            else if(strcmp(&argv[i][1], "operating_point") == 0) {
                args->setOperatingPoint = true;
                args->av1OperatingPoint = (uint8_t)atoi(argv[++i]);
            }
            else if(strcmp(&argv[i][1], "output_all_layers") == 0) {
                args->setOutputAllLayers = args->setOperatingPoint;    //Set this parameter only if setOperatingPoint = 1
                args->av1OutputAllLayers = (uint8_t)atoi(argv[++i]);
            }
            else if(strcmp(&argv[i][1], "set_max_res") == 0) {
                args->setMaxRes = true;
                args->enableMaxRes = (uint8_t)atoi(argv[++i]);
            }
            else if(strcmp(&argv[i][1], "crc") == 0) {
                if(bDataAvailable) {
                    ++i;
                    if (!strcasecmp(argv[i], "chk"))
                        args->checkCRC = true;
                    else if (!strcasecmp(argv[i], "gen"))
                        args->generateCRC = true;
                    else {
                        LOG_ERR("ParseArgs: -crc must be followed by gen/chk. "
                                "Instead %s was encountered.\n",
                            &argv[i][0]);
                        return -1;
                    }
                } else {
                    LOG_ERR("ParseArgs: -crc must be followed by gen/chk.\n");
                    return -1;
                }
            }
            else if(strcmp(&argv[i][1], "crcpath") == 0) {
                if(bDataAvailable) {
                    args->crcFilePath = argv[++i];
                } else {
                    printf("Error: -crcpath must be followed by crc file path.\n");
                    return 0;
                }
            }
            else if(strcmp(&argv[i][1], "prof") == 0) {
                args->decProfiling = true;
            }
            else if (strcmp(&argv[i][1], "l") == 0) {
                if (argv[i+1]) {
                    int loop;
                    if (sscanf(argv[++i], "%d", &loop) && loop >= -1 && loop != 0) {
                        args->loop = loop;
                    } else {
                        LOG_ERR("ParseArgs: Invalid loop count encountered (%s)\n", argv[i]);
                    }
                } else {
                    LOG_ERR("ParseArgs: -l must be followed by loop count\n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "a") == 0) {
                if (bDataAvailable) {
                    float aspectRatio;
                    if (sscanf(argv[++i], "%f", &aspectRatio) && aspectRatio > 0.0) {
                        args->aspectRatio = aspectRatio;
                    } else {
                        LOG_ERR("ParseArgs: Invalid aspect ratio encountered (%s)\n", argv[i]);
                    }
                } else {
                    LOG_ERR("ParseArgs: -a must be followed by aspect ratio\n");
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "id") == 0) {
                if (bDataAvailable) {
                    char *arg = argv[++i];
                    args->instanceId = atoi(arg);
                } else {
                    LOG_ERR("ParseArgs: -i must be followed by instance id\n");
                    return -1;
                }
                if (args->instanceId >= 2) {
                    LOG_ERR("ParseArgs: Bad instance ID: %d. Valid values are [0-1]. ", args->instanceId);
                    LOG_ERR("           Using default instance ID 0\n");
                    args->instanceId = 0;
                }
            }
            else {
                LOG_ERR("ParseArgs: option %c is not supported anymore\n", argv[i][1]);
            }
        }
    }

    if (!bHasCodecType || !bHasFileName) {
        return -1;
    }

    return 0;
}
