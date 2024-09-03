/* Copyright (c) 2022 - 2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "stereo_commandline.h"

using namespace std;

void PrintUsage()
{
    printf("stereosense is disparity  estimation pipeline \n");
    printf("Usage: stereosense [options] \n");
    printf("Options: \n");
    printf("-h                   Prints usage \n");
    printf("-input  [input file] Input file name (must be png or yuv format) \n");
    printf("-ref                 Reference input image \n");
    printf("-width               Width of the input \n");
    printf("-height              Height of the input \n");
    printf("-chromaFormat        Chroma format IDC of Input YUV. Default is 0\n");
    printf("                     0: 400 (default)\n");
    printf("                     1: 420\n");
    printf("                     2: 422\n");
    printf("                     3: 444\n");
    printf("-o   [output file]   Output file name \n");
    printf("-co  [cost file]     Cost output file name \n");
    printf("-nframes [n]         Number of frames used for stereo estimation. Default: one frames \n");
    printf("-stereocrcgen [txt]  Generate Stereo CRC values \n");
    printf("-stereocrcchk [txt]  Check Stereo CRC values \n");
    printf("-etype               Estimation type 0: Use default SGM presets non-zero: use config presets \n");
    printf("-ndisp               Number of disparities\n");
    printf("                     128 disparities\n");
    printf("                     256 disparities\n");
    printf("-gridSize            Grid size which ofa will consider for processing. \n");
    printf("                     0: Grid size 1x1 \n");
    printf("                     1: Grid Size 2x2 \n");
    printf("                     2: Grid Size 4x4 \n");
    printf("                     3: Grid Size 8x8 \n");
    printf("                     Different values for each pyramid level can be entered \n");
    printf("-bitDepth            Bitdepth for input: valid arguments 8/10/12/16, Default 8 \n");
    printf("-lrCheck             Do LR check\n");
    printf("-lrCheckThr          LR check threshold\n");
    printf("-p1                  penalty1 value in cost function. Different values for each pyramid level can be entered \n");
    printf("-p2                  penalty2 value in cost function. Different values for each pyramid level can be entered \n");
    printf("-diag                Enable diagonal path search in SGM: 0/1. Different values for each pyramid level can be entered \n");
    printf("-adaptiveP2          Enable adaptive P2.Different values for each pyramid level can be entered \n");
    printf("-pass                num of passes. Different values for each pyramid level can be entered \n");
    printf("-alpha               alpha log value for adaptive P2.valid range 0 to 3.Different values for each pyramid level can be entered \n");
    printf("-median              1: Enable median filtering 0: Disable median filtering (Default) \n");
    printf("-upsample            1: Enable upsampling 0: Disable upsampling (Default) \n");
    printf("-preset              ofa preset used for stereo processing\n");
    printf("                     0: high quality preset\n");
    printf("                     1: high performance preset\n");
    printf("-do_RL               0: only generate left disparity map 1: generate right disparity map as well\n");
}

int ParseArgs(int argc, char *argv[], TestArgs *args)
{
    bool bLastArg = false;
    bool bDataAvailable = false;
    bool bHasInputFileName = false;
    bool blr_threshold = false;
    uint16_t i;

    args->chromaFormat =  static_cast<ChromaFormat> (0);
    args->bitdepth = 8;
    args->nframes = 1;
    args->lrCheckThr = 3;

    for (i = 1; i < argc; i++)
    {
        // check if this is the last argument
        bLastArg = ((argc - i) == 1);

        // check if there is data available to be parsed following the option
        bDataAvailable = (!bLastArg) && !(argv[i+1][0] == '-');

        if (argv[i][0] == '-')
        {
            if (strcmp(&argv[i][1], "input") == 0)
            {
                if (bDataAvailable)
                {
                    args->inputFilename = argv[++i];
                    bHasInputFileName = true;
                }
                else
                {
                    cerr << "ERR: -input must be followed by input file name \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "ref") == 0)
            {
                if (bDataAvailable)
                {
                    args->refFilename = argv[++i];
                    bHasInputFileName = true;
                }
                else
                {
                    cerr << "ERR: -ref must be followed by reference file name \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "width") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%hd", &args->width))
                {
                    cerr << "ERR: -width must be followed by width of input \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "height") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%hd", &args->height))
                {
                    cerr << "ERR: -height must be followed by width of input \n";
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
                    cerr << "ERR: -o must be followed by output file name \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "co") == 0)
            {
                if (bDataAvailable)
                {
                    args->costFilename = argv[++i];
                }
                else
                {
                    cerr << "ERR: -co must be followed by cost file name \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "chromaFormat") == 0)
            {
                if (bDataAvailable)
                {
                    args->chromaFormat =static_cast<ChromaFormat> (atoi(argv[++i]));
                }
                else
                {
                    cerr << "ERR: -chromaFormat must be followed by value \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "nframes") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%hd", &args->nframes))
                {
                    cerr << "ERR: -nframes must be followed by frame count \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "etype") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%d", &args->estimationType))
                {
                    cerr << "ERR: -etype must be followed by estimation type \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "gridSize") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%d", &args->gridsize))
                {
                    cerr << "ERR: gridSize  must be followed by ofa GridLog2 \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "ndisp") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%hd", &args->ndisp))
                {
                    cerr << "ERR: -ndisp must be followed by number of disparities 128/256 \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "bitDepth") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%d", &args->bitdepth))
                {
                    cerr << "ERR: -bitDepth must be followed by bitdepth (8/10/12/16) \n";
                    return -1;
                }
            }
            else if(strcmp(&argv[i][1], "stereocrcchk") == 0)
            {
                // crc check
                if (bDataAvailable)
                {
                   args->stereoCRCChkFilename = argv[++i];
                }
                else
                {
                    cerr << "ERR: -crcchk must be followed by crc file name \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "stereocrcgen") == 0)
            {
                // crc generate
                if (bDataAvailable)
                {
                    args->stereoCRCGenFilename = argv[++i];
                }
                else
                {
                    cerr << "ERR: -crcgen must be followed by crc file name \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "p1") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%hd", &args->p1))
                {
                    cerr << "ERR: -p1 must be followed by penalty 1 \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "p2") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%hd", &args->p2))
                {
                    cerr << "ERR: -p2 must be followed by penalty 2\n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "adaptiveP2") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%hd", &args->adaptiveP2))
                {
                    cerr << "ERR: -adaptiveP2 must be followed 0 or 1 \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "alpha") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%hd", &args->alpha))
                {
                    cerr << "ERR: -alpha must be followed alpha value \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "diag") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%hd", &args->DiagonalMode))
                {
                    cerr << "ERR: -d must be followed 0 to disable diagonal or 1 to enable daigonal search \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "pass") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%hd", &args->numPasses))
                {
                    cerr << "ERR: -p must be followed by number of passes (1/2/3) \n";
                    return -1;
                }
            }
            else if(strcmp(&argv[i][1], "do_RL") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%hd", &args->do_RL))
                {
                    cerr << "ERR: -do_RL must be followed by 0: generate left disparity only 1: Generate right disparity as well \n";
                    return -1;
                }
            }
            else if(strcmp(&argv[i][1], "lrCheck") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%hd", &args->lrCheck))
                {
                    cerr << "ERR: -lrCheck must be followed by 0: do left to right check 1: Disable \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "lrCheckThr") == 0)
            {
                blr_threshold = true;
                if (!bDataAvailable || !sscanf(argv[++i], "%hd", &args->lrCheckThr))
                {
                    cerr << "ERR: -lrCheckThr must be followed by threshold used by lrcheck \n";
                    return -1;
                }
            }
            else if(strcmp(&argv[i][1], "median") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%d", &args->median))
                {
                    cerr << "ERR: -median must be followed by 0: Disable median filtering 1: square window 2: cross window \n";
                    return -1;
                }
            }
            else if(strcmp(&argv[i][1], "upsample") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%d", &args->upsample))
                {
                    cerr << "ERR: -upsample must be followed by 0: Disable upsampling 1: Enable upsampling \n";
                    return -1;
                }
            }
            else if (strcmp(&argv[i][1], "preset") == 0)
            {
                if (!bDataAvailable || !sscanf(argv[++i], "%d", &args->preset))
                {
                    cerr << "ERR: -preset must be followed by ofa preset\n";
                    return -1;
                }
            }
            else
            {
                cerr << "ERR: option "<< &argv[i][1] << " is not supported and returning \n";
                return -1;
            }
        }
    }

    if (!bHasInputFileName)
    {
        cerr << "ERR: No Input file name was given\n";
        return -1;
    }
    if (args->lrCheck==1 && !blr_threshold)
    {
        clog << "No threshold given for left-right consistency check, will be using default value 3\n";
    }
    if(!args->width || !args->height)
    {
        cerr << "ERR: Input resolution must be set \n";
        return -1;
    }
    if (args->lrCheck ==1 && args->upsample==0)
    {
        cerr << "ERR: Upsample must be set to 1 to use consistency check \n";
        return -1;
    }

    return 0;
}
