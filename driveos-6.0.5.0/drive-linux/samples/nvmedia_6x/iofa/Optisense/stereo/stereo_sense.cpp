/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include "common_defs.h"
#include "image_reader.h"
#include "ofa_class.h"
#include "image_buffer.h"
#include "file_writer.h"
#include "cmn_functions.h"
#include "crc_class.h"
#include "median.h"
#include "upsample.h"
#include "lrcheck.h"
#include "sync_impl.h"
#include "stereo_commandline.h"
ImageReader *create_image_reader(TestArgs *args, uint32_t frame_num, bool *isPNG);

using namespace std;

int main(int argc, char **argv)
{
    bool isPNG;
    Buffer *image1 = NULL, *image2 = NULL;
    Buffer *ofaOutSurface[2] = {}, *costSurface[2] = {};
    Buffer *outSurface[2] = {};
    Buffer *medianSurface[2] = {};
    Buffer *usSurface[2] = {};
    Buffer *LRSurface = NULL;
    NvMOfaStereo *pNvMOfa = NULL;
    UpSample *upSample = NULL;
    ConstCheck *LRcheck = NULL;
    uint16_t outWidth, outHeight;
    uint8_t  gridSizeLog2X, gridSizeLog2Y;
    ImageReader *p_im_read_1 = NULL, *p_im_read_2 = NULL;
    FileWriter *outFile[3] = {}, *costFile = NULL;
    TestArgs st_args = {};
    CRCGen *pOutCRCGen = NULL;
    CRCCmp *pOutCRCCmp = NULL;
    SGMStereoParams SGMStereoParams{};
    bool RLdirection = false;
    uint32_t i, DispPass;
    MedianFilter *median = NULL;
    uint16_t frames=0;
    NvSciSyncImpl *pSync_CS, *pSync_OS;
    NvSciSyncFence preFence = NvSciSyncFenceInitializer;
    NvSciSyncFence postFence = NvSciSyncFenceInitializer;

    if (ParseArgs(argc, argv, &st_args))
    {
        PrintUsage();
        return -1;
    }

    if (st_args.lrCheck == 1)
    {
        DispPass=2;
    }
    else
    {
        DispPass=1;
    }
    if (st_args.estimationType != 0)
    {
        SGMStereoParams.penalty1   = st_args.p1;
        SGMStereoParams.penalty2   = st_args.p2;
        if (st_args.adaptiveP2 != 0)
        {
            SGMStereoParams.adaptiveP2 = true;
        }
        else
        {
            SGMStereoParams.adaptiveP2 = false;
        }
        SGMStereoParams.alphaLog2  = st_args.alpha;
        if (st_args.DiagonalMode != 0)
        {
            SGMStereoParams.enableDiag = true;
        }
        else
        {
            SGMStereoParams.enableDiag = false;
        }
        SGMStereoParams.numPasses  = st_args.numPasses;
    }
    if (!st_args.outputFilename.empty())
    {
        outFile[0] =  new FileWriter(st_args.outputFilename);
        if (!outFile[0]->initialize())
        {
            cerr << "outFile[0]->initialize failed\n";
            goto exit;
        }
        if (st_args.lrCheck == 1)
        {
            int pos = st_args.outputFilename.find(".");
            string out1 = st_args.outputFilename.substr(0, pos) + "_R.bin";
            outFile[1] =  new FileWriter(out1);
            if (!outFile[1]->initialize())
            {
                cerr << "outFile[1]->initialize failed\n";
                goto exit;
            }
            string out2 = st_args.outputFilename.substr(0, pos) + "_LR_check.bin";
            outFile[2] =  new FileWriter(out2);
            if (!outFile[2]->initialize())
            {
                cerr << "outFile[2]->initialize failed\n";
                goto exit;
            }
        }
    }
    if (!st_args.costFilename.empty())
    {
        costFile = new FileWriter(st_args.costFilename);
        if (!costFile->initialize())
        {
            cerr << "costFile->initialize failed\n";
            goto exit;
        }
    }
    p_im_read_1 = create_image_reader(&st_args, 1U, &isPNG);
    if (p_im_read_1 == NULL)
    {
        cerr << "create_image_reader failed\n";
        goto exit;
    }
    p_im_read_2 = create_image_reader(&st_args, 0U, &isPNG);
    if (p_im_read_2 == NULL)
    {
        cerr << "create_image_reader failed\n";
        goto exit;
    }
    if (!p_im_read_1->initialize())
    {
        cerr << "p_im_read_1->initialize failed\n";
        goto exit;
    }
    if (!p_im_read_2->initialize())
    {
        cerr << "p_im_read_1->initialize failed\n";
        goto exit;
    }
    if (!st_args.stereoCRCChkFilename.empty())
    {
        pOutCRCCmp = new CRCCmp(st_args.stereoCRCChkFilename, 0xEDB88320L);
        if (!pOutCRCCmp)
        {
            cerr << "pOutCRCCmp is failed\n";
            goto exit;
        }
        pOutCRCCmp->fileOpen();
    }
    else if (!st_args.stereoCRCGenFilename.empty())
    {
        pOutCRCGen = new CRCGen(st_args.stereoCRCGenFilename, 0xEDB88320L);
        if (!pOutCRCGen)
        {
            cerr << "pOutCRCGen is failed\n";
            goto exit;
        }
        pOutCRCGen->fileOpen();
    }

    image1 = new ImageBuffer(st_args.bitdepth, st_args.chromaFormat, p_im_read_1->getWidth(), p_im_read_1->getHeight());
    if (image1 == NULL)
    {
        cerr << "ImageBuffer1 class creation failed\n";
        goto exit;
    }
    image2 = new ImageBuffer(st_args.bitdepth, st_args.chromaFormat, p_im_read_1->getWidth(), p_im_read_1->getHeight());
    if (image2 == NULL)
    {
        cerr << "ImageBuffer2 class creation failed\n";
        goto exit;
    }
    if (image1->createBuffer() != NVMEDIA_STATUS_OK)
    {
        cerr << "image1 createBuffer failed\n";
        goto exit;
    }
    if (image2->createBuffer() != NVMEDIA_STATUS_OK)
    {
        cerr << "image2 createBuffer failed\n";
        goto exit;
    }

    pSync_OS = new NvSciSyncImpl(WAITER_CPU, SIGNALER_OFA);
    if (pSync_OS == NULL)
    {
        cout << "NvSciSyncImpl class creation failed\n";
        goto exit;
    }

    pSync_CS = new NvSciSyncImpl(WAITER_OFA, SIGNALER_CPU);
    if (pSync_CS == NULL)
    {
        cout << "NvSciSyncImpl class creation failed\n";
        goto exit;
    }

    cout << "Input buffers creation successful\n";

    pNvMOfa = new NvMOfaStereo(p_im_read_1->getWidth(),
                               p_im_read_1->getHeight(),
                               st_args.gridsize,
                               st_args.ndisp,
                               st_args.profile,
                               st_args.preset);
    if (pNvMOfa == NULL)
    {
        cerr << "NvMOfa class creation failed\n";
        goto exit;
    }
    if (!pNvMOfa->createNvMOfa())
    {
        cerr << "NvMOfa createNvMOfa api failed\n";
        goto exit;
    }
    cout << "NvMOfa creation successful\n";
    if (!pNvMOfa->initNvMOfaStereo(2))
    {
        cerr << "NvMOfa init api failed\n";
        goto exit;
    }
    cout << "NvMOfa initialization successful\n";
    if (st_args.estimationType != 0)
    {
        if (!pNvMOfa->setSGMStereoConfigParams(SGMStereoParams))
        {
            cerr << "NvMOfa set sgm params api failed\n";
            goto exit;
        }
        cout << "NvMOfa set params successful\n";

        if (!pNvMOfa->getSGMStereoConfigParams(SGMStereoParams))
        {
            cerr << "NvMOfa get sgm params api failed\n";
            goto exit;
        }
        cout << "NvMOfa get params successful\n";
    }
    if (!pNvMOfa->getOutSurfaceParams(outWidth, outHeight, gridSizeLog2X, gridSizeLog2Y))
    {
        cerr << "NvMOfa getOutSurfaceParams api failed\n";
        goto exit;
    }
    if (pSync_CS->alloc() != NVMEDIA_STATUS_OK)
    {
        cerr << "sciSync alloc failed\n";
        goto exit;
    }
    if (pSync_OS->alloc() != NVMEDIA_STATUS_OK)
    {
        cerr << "sciSync alloc failed\n";
        goto exit;
    }
    for (i = 0; i < DispPass; i++)
    {
        ofaOutSurface[i] = new ImageBuffer(16, A16, outWidth, outHeight);
        if (ofaOutSurface[i] == NULL)
        {
            cerr << "ofaOutSurface class creation failed\n";
            goto exit;
        }
        if (ofaOutSurface[i]->createBuffer() != NVMEDIA_STATUS_OK)
        {
            cerr << "outSurface createBuffer failed\n";
            goto exit;
        }
        costSurface[i] = new ImageBuffer(8, A8, outWidth, outHeight);
        if (costSurface[i] == NULL)
        {
            cerr << "CostSurface class creation failed\n";
            goto exit;
        }

        if (costSurface[i]->createBuffer() != NVMEDIA_STATUS_OK)
        {
            cerr << "CostSurface createBuffer failed\n";
            goto exit;
        }

        if (st_args.median == 1 || st_args.median == 2)
        {
            medianSurface[i] = new ImageBuffer(16, A16, outWidth, outHeight);
            if (medianSurface[i] == NULL)
            {
                cerr << "medianSurface class creation failed\n";
                goto exit;
            }
            if (medianSurface[i]->createBuffer() != NVMEDIA_STATUS_OK)
            {
                cerr << "outSurface createBuffer failed\n";
                goto exit;
            }
        }
        if (st_args.upsample == 1)
        {
            usSurface[i] = new ImageBuffer(16, A16, p_im_read_1->getWidth(), p_im_read_1->getHeight());
            if (usSurface[i] == NULL)
            {
                cerr << "usSurface class creation failed\n";
                goto exit;
            }
            if (usSurface[i]->createBuffer() != NVMEDIA_STATUS_OK)
            {
                cerr << "usSurface createBuffer failed\n";
                goto exit;
            }
        }
        if (i == 1)
        {
            LRSurface = new ImageBuffer(16, A16, p_im_read_1->getWidth(), p_im_read_1->getHeight());
            if (LRSurface == NULL)
            {
                cerr << "LRSurface class creation failed\n";
                goto exit;
            }
            if (LRSurface->createBuffer() != NVMEDIA_STATUS_OK)
            {
                cerr << "OutSurface createBuffer failed\n";
                goto exit;
            }

            LRcheck = new LRCheckStereo(p_im_read_1->getWidth(), p_im_read_1->getHeight());
            if (LRcheck == NULL)
            {
                cerr << "LRCheckStereo class creation failed\n";
                goto exit;
            }
            else
            {
                if (!LRcheck->initialize())
                {
                    cerr << "initialize method of upsample class failed\n";
                    goto exit;
                }
            }
        }
    }
    if (st_args.median == 1 || st_args.median == 2)
    {
        median = new StereoMedianFilterCPU(st_args.median, 3, outWidth, outHeight);
        if (median == NULL)
        {
            cerr << "StereoMedianFilterCPU class creation failed\n";
            goto exit;
        }
        else
        {
            if (!median->initialize())
            {
                cerr << "initialize method of median class failed\n";
                goto exit;
           }
        }
    }
    if (st_args.upsample == 1)
    {
        upSample = new NNUpSampleStereo(outWidth, outHeight, p_im_read_1->getWidth(), p_im_read_1->getHeight(), gridSizeLog2X);
        if (upSample == NULL)
        {
            cerr << "NNUpSampleStereo class creation failed\n";
            goto exit;
        }
        else
        {
            if (!upSample->initialize())
            {
                cerr << "initialize method of upsample class failed\n";
                goto exit;
            }
        }
    } 
    if (!RegisterSurfaces(pNvMOfa, image1, image2, ofaOutSurface[0], costSurface[0]))
    {
        cerr << "RegisterSurfaces LR failed\n";
        goto exit;
    }
    cout << "Surface registration sucessful\n";
    if (st_args.lrCheck == 1)
    {
        if (!RegisterSurfaces(pNvMOfa, ofaOutSurface[1], costSurface[1]))
        {
            cerr << "RegisterSurfaces RL failed\n";
            goto exit;
        }
        cout << "Surface registration successful\n";
    }
    if (!pNvMOfa->registerSciSyncToNvMOfa(pSync_OS->getSyncObj(), NVMEDIA_EOFSYNCOBJ))
    {
        cerr << "registerSciSyncToNvMOfa API failed \n";
        goto exit;
    }
    if (!pNvMOfa->registerSciSyncToNvMOfa(pSync_CS->getSyncObj(), NVMEDIA_PRESYNCOBJ))
    {
        cerr << "registerSciSyncToNvMOfa API failed \n";
        goto exit;
    }

    while(frames < st_args.nframes)
    {
        if (!p_im_read_1->read_file())
        {
            cerr << "p_im_read_1 read file failed\n";
            goto exit;
        }
        if (!p_im_read_2->read_file())
        {
            cerr << "p_im_read_2 read file failed\n";
            goto exit;
        }
        cout<< "Processing frame  " << frames << "\n";
        // image1->writeBuffer(reinterpret_cast<uint8_t *>(p_im_read_1->current_item()), isPNG);
        // image2->writeBuffer(reinterpret_cast<uint8_t *>(p_im_read_2->current_item()), isPNG);

        for (i = 0; i < DispPass; i++)
        {
            if (!pNvMOfa->setSciSyncObjToNvMOfa(pSync_OS->getSyncObj()))
            {
                cerr << "NvMOfa setSciSyncObjToNvMOfa api failed\n";
                goto exit;
            }
            if (!pSync_CS->generateFence(&preFence))
            {
                cerr << "generateFence method failed\n";
                goto exit;
            }
            if(i == 0)
            {
                RLdirection = false;
            }
	    else
	    {
                RLdirection = true;
            }

            if (!pNvMOfa->insertPreFence(&preFence))
            {
                cerr << "inseret prefence method failed\n";
                goto exit;
            }

            if (!pNvMOfa->processSurfaceNvMOfa(image1->getHandle(),
                                               image2->getHandle(),
                                               ofaOutSurface[i]->getHandle(),
                                               costSurface[i]->getHandle(),
                                               RLdirection))
            {
                cerr << "NvMOfa processSurfaceNvMOfa api failed\n";
                goto exit;
            }
            if (!pNvMOfa->getEOFSciSyncFence(pSync_OS->getSyncObj() , &postFence))
            {
                cerr << "NvMOfa getEOFSciSyncFence api failed\n";
                goto exit;
            }
            if (i ==  0)
            {
                image1->writeBuffer(reinterpret_cast<uint8_t *>(p_im_read_1->current_item()), isPNG);
                image2->writeBuffer(reinterpret_cast<uint8_t *>(p_im_read_2->current_item()), isPNG);
            }

            if (!pSync_CS->signalSyncObj())
            {
                cerr << " method failed\n";
                goto exit;
            }
            NvSciSyncFenceClear(&preFence);
            cout << "NvMOfa process frame successful\n";
            cout << "Waiting on post fence\n";
            if (!pSync_OS->checkOpDone(&postFence))
            {
                cerr << "SciSyncImpl checkOpDone api failed\n";
                goto exit;
            }
            if (st_args.profile != 0)
            {
                if (!pNvMOfa->printProfileData())
                {
                    cerr << "NvMOfa printProfileData api failed\n";
                    goto exit;
                }
            }

            outSurface[i] = ofaOutSurface[i];
            if (median != NULL)
            {
                cout <<  "median filter processing\n";
                if (!median->process(ofaOutSurface[i], medianSurface[i]))
                {
                    cerr << "median process filter failed\n";
                    goto exit;
                }
                outSurface[i] = medianSurface[i];
            }
            if (upSample != NULL)
            {
                if (!upSample->process(outSurface[i], usSurface[i]))
                {
                    cerr << "upsample process failed\n";
                    goto exit;
                }
                outSurface[i] = usSurface[i];
            }
            if (outFile[i] != NULL)
            {
                if (!outFile[i]->writeSurface(outSurface[i]))
                {
                    cerr << "writeSurface failed for output surface\n";
                    goto exit;
                }
            }
        }

        if (LRcheck != NULL)
        {
            cout <<  "LR check processing using threshold " << st_args.lrCheckThr << "\n";
            if (!LRcheck->process(outSurface[0], outSurface[1], LRSurface, st_args.lrCheckThr))
            {
                cerr << "LRcheck process failed\n";
                goto exit;
            }
            if (outFile[2] != NULL)
            {
                if (!outFile[2]->writeSurface(LRSurface))
                {
                    cerr << "writeSurface failed for output surface\n";
                    goto exit;
                }
            }
        }
        for (i = 0; i < DispPass; i++)
        {
            if (pOutCRCGen != NULL)
            {
                if (!pOutCRCGen->CalculateCRC(outSurface[i]))
                {
                    cerr << "CalculateCRC failed \n";
                    goto exit;
                }
                if (!pOutCRCGen->fileWrite())
                {
                    cerr << "fileWrite failed \n";
                    goto exit;
                }
            }
            if (pOutCRCCmp != NULL)
            {
                if (!pOutCRCCmp->CalculateCRC(outSurface[i]))
                {
                    cerr << "CalculateCRC failed \n";
                    goto exit;
                }
                if (!pOutCRCCmp->CmpCRC())
                {
                    cerr << "CmpCRC failed \n";
                    goto exit;
                }
                else
                {
                    cout << "/** CmpCRC comparison passed **/ \n";
                }
            }
        }
        if (costFile != NULL)
        {
            if (!costFile->writeSurface(costSurface[0]))
            {
                cerr << "Writing cost bin file failed\n";
                goto exit;
            }
        }

        cout << "Output surface write successful for Frame Num " << frames << "\n";
        frames++;
    }
    cout << "NvMOfa Stereo Estimation finish!\n";
exit:
    if (pNvMOfa != NULL)
    {
        UnRegisterSurfaces(pNvMOfa, image1, image2, ofaOutSurface[0], costSurface[0]);
        UnRegisterSurfaces(pNvMOfa, ofaOutSurface[1], costSurface[1]);
        if (pSync_OS != NULL)
        {
            pNvMOfa->unRegisterSciSyncToNvMOfa(pSync_OS->getSyncObj());
            delete pSync_OS;
        }
        if (pSync_CS != NULL)
        {
            pNvMOfa->unRegisterSciSyncToNvMOfa(pSync_CS->getSyncObj());
            delete pSync_CS;
        }
        pNvMOfa->destroyNvMOfa();
        delete pNvMOfa;
    }
    if (median != NULL)
    {
        median->release();
        delete median;
    }
    if (upSample != NULL)
    {
        upSample->release();
        delete upSample;
    }
    if (LRcheck != NULL)
    {
        LRcheck->release();
        delete LRcheck;
    }
    for (i = 0; i < DispPass; i++)
    {
        if (costSurface[i] != NULL)
        {
            costSurface[i]->destroyBuffer();
            delete costSurface[i];
        }
        if (ofaOutSurface[i] != NULL)
        {
            ofaOutSurface[i]->destroyBuffer();
            delete ofaOutSurface[i];
        }
        if (medianSurface[i] != NULL)
        {
            medianSurface[i]->destroyBuffer();
            delete medianSurface[i];
        }
        if (usSurface[i] != NULL)
        {
            usSurface[i]->destroyBuffer();
            delete usSurface[i];
        }
    }
    for (i = 0; i < 3; i++)
    {
        if (outFile[i] != NULL)
        {
            delete outFile[i];
        }
    }
    if (LRSurface != NULL)
    {
        LRSurface->destroyBuffer();
        delete LRSurface;
    }
    if (image1 != NULL)
    {
        image1->destroyBuffer();
        delete image1;
    }
    if (image2 != NULL)
    {
        image2->destroyBuffer();
        delete image2;
    }
    if (pOutCRCGen != NULL)
    {
        pOutCRCGen->fileClose();
        delete pOutCRCCmp;
    }
    if (pOutCRCCmp != NULL)
    {
        pOutCRCCmp->fileClose();
        delete pOutCRCGen;
    }
    if (p_im_read_1 != NULL)
    {
        delete p_im_read_1;
    }
    if (p_im_read_2 != NULL)
    {
        delete p_im_read_2;
    }
    if (costFile != NULL)
    {
        delete costFile;
    }

    return 0;
}

ImageReader *create_image_reader(TestArgs *args, uint32_t frame_num, bool *isPNG)
{
    string filename;
    string ext;
    ImageReader *p_im_read = NULL;

    ext = GET_EXTENSION(args->inputFilename);
    cout << "extension: " << ext << endl;
    if ((ext == "png") || (ext =="PNG"))
    {
#if PNG_SUPPORT
        filename = (frame_num == 1U) ? args->inputFilename : args->refFilename;
        p_im_read = new PNGReader(0, 0, args->bitdepth, YUV_400, filename);
        *isPNG = true;
#else
	cerr << "PNG is not supported\n";
#endif

    }
    else  if ((ext == "yuv") || (ext == "yuv444") || (ext == "yuv400") || (ext == "yuv420") || (ext == "yuv422"))
    {
        filename = (frame_num == 1U) ? args->inputFilename : args->refFilename;
        p_im_read = new YUVReader(args->height, args->width, args->bitdepth, args->chromaFormat, filename, 0);
        *isPNG = false;
    }
    else
    {
        cerr << "Only YUV and PNG extension is supported\n";
    }

    return p_im_read;
}

