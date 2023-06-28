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
#include "flow_commandline.h"
#include "sync_impl.h"

ImageReader *create_flow_image_reader(FlowTestArgs *args, uint32_t frame_num, bool *isPNG);

using namespace std;

int main(int argc, char **argv)
{
    bool isPNG;
    ImagePyramid *ofaOutSurface[2] = {}, *costSurface[2] = {};
    Buffer *outSurface[2] = {};
    ImagePyramid *image1 = NULL, *image2 = NULL;
    Buffer *medianSurface[2] = {};
    Buffer *usSurface[2] = {};
    Buffer *FBSurface = NULL;
    Buffer *CrcSurface = NULL;
    NvMOfaFlow *pNvMOfa = NULL;
    uint16_t *inWidth, *inHeight;
    uint16_t *outWidth, *outHeight;
    uint32_t  *gridSizeLog2X, *gridSizeLog2Y;
    ImageReader *p_im_read_1 = NULL, *p_im_read_2 = NULL;
    FileWriter *outFile[3] = {}, *costFile = NULL;
    FlowTestArgs of_args = {};
    CRCGen *pOutCRCGen = NULL;
    CRCCmp *pOutCRCCmp = NULL;
    MedianFilter *median = NULL;
    UpSample *upSample = NULL;
    ConstCheck *fbcheck = NULL;
    uint16_t frames = 0;
    uint32_t i, FlowPass;
    SGMFlowParams SGMFlowParams{};
    NvSciSyncImpl *pSync;
    NvSciSyncFence preFence = NvSciSyncFenceInitializer;

    if (ParseArgs(argc, argv, &of_args))
    {
          PrintUsage();
          return -1;
    }
    if (of_args.fbCheck == 1)
    {
        FlowPass=2;
    }
    else
    {
        FlowPass=1;
    }
    if (of_args.overrideParam)
    {
        for (uint8_t i = 0U; i < NVMEDIA_IOFA_MAX_PYD_LEVEL; i++)
        {
            SGMFlowParams.penalty1[i] = of_args.p1[i];
            SGMFlowParams.penalty2[i] = of_args.p2[i];
            if (of_args.adaptiveP2[i] != 0)
            {
                SGMFlowParams.adaptiveP2[i] = true;
            }
            else
            {
                SGMFlowParams.adaptiveP2[i] = false;
            }
            SGMFlowParams.alphaLog2[i] = of_args.alpha[i];
            if (of_args.DiagonalMode[i] != 0)
            {
                SGMFlowParams.enableDiag[i] = true;
            }
            else
            {
                SGMFlowParams.enableDiag[i] = false;
            }
            SGMFlowParams.numPasses[i] = of_args.numPasses[i];
        }
    }

    if (!of_args.outputFilename.empty())
    {
        outFile[0] =  new FileWriter(of_args.outputFilename);
        if (!outFile[0]->initialize())
        {
            cerr << "outFile->initialize failed\n";
            goto exit;
        }
        if (of_args.fbCheck == 1)
        {
            int pos = of_args.outputFilename.find(".");
            string out1 = of_args.outputFilename.substr(0, pos) + "_bwd.bin";
            outFile[1] =  new FileWriter(out1);
            if (!outFile[1]->initialize())
            {
                cerr << "outFile[1]->initialize failed\n";
                goto exit;
            }
            string out2 = of_args.outputFilename.substr(0, pos) + "_FB_check.bin";
            outFile[2] =  new FileWriter(out2);
            if (!outFile[2]->initialize())
            {
                cerr << "outFile[2]->initialize failed\n";
                goto exit;
            }
        }
    }
    if (!of_args.costFilename.empty())
    {
        costFile = new FileWriter(of_args.costFilename);
        if (!costFile->initialize())
        {
            cerr << "costFile->initialize failed\n";
            goto exit;
        }
    }
    p_im_read_1 = create_flow_image_reader(&of_args, 1U^of_args.do_bwd, &isPNG);
    if (p_im_read_1 == NULL)
    {
        cerr << "create_image_reader failed\n";
        goto exit;
    }
    p_im_read_2 = create_flow_image_reader(&of_args, 0U^of_args.do_bwd, &isPNG);
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

    if (!of_args.flowCRCChkFilename.empty())
    {
        pOutCRCCmp = new CRCCmp(of_args.flowCRCChkFilename, 0xEDB88320L);
        if (!pOutCRCCmp)
        {
            cerr << "pOutCRCCmp is failed\n";
            goto exit;
        }
        pOutCRCCmp->fileOpen();
    }
    else if (!of_args.flowCRCGenFilename.empty())
    {
        pOutCRCGen = new CRCGen(of_args.flowCRCGenFilename, 0xEDB88320L);
        if (!pOutCRCGen)
        {
            cerr << "pOutCRCGen is failed\n";
            goto exit;
        }
        pOutCRCGen->fileOpen();
    }

    pSync = new NvSciSyncImpl(WAITER_CPU, SIGNALER_OFA);
    if (pSync == NULL)
    {
        cout << "NvSciSyncImpl class creation failed\n";
        goto exit;
    }

    pNvMOfa = new NvMOfaFlow(p_im_read_1->getWidth(),
                             p_im_read_1->getHeight(),
                             of_args.chromaFormat,
                             of_args.gridsize,
                             of_args.pydSGMMode,
                             of_args.profile,
                             of_args.preset);
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
    if (!pNvMOfa->initNvMOfaFlow(2))
    {
        cerr << "NvMOfa init api failed\n";
        goto exit;
    }
    cout << "NvMOfa initialization successful\n";

    if (!pNvMOfa->getInputSurfaceParams(&inWidth, &inHeight))
    {
        cerr << "NvMOfa getInputSurfaceParams api failed\n";
        goto exit;
    }
    if (pSync->alloc() != NVMEDIA_STATUS_OK)
    {
        cout << "sciSync alloc failed\n";
        goto exit;
    }

    image1 = new ImagePyramid(of_args.bitdepth, of_args.chromaFormat, inWidth, inHeight, pNvMOfa->getNumOfPydLevels());
    if (image1 == NULL)
    {
        cerr << "ImagePramid class creation failed\n";
        goto exit;
    }
    image2 = new ImagePyramid(of_args.bitdepth, of_args.chromaFormat, inWidth, inHeight, pNvMOfa->getNumOfPydLevels());
    if (image2 == NULL)
    {
        cerr << "ImagePyramid class creation failed\n";
        goto exit;
    }
    if (image1->createPyramid() != NVMEDIA_STATUS_OK)
    {
        cerr << "image1 createPyramid failed\n";
        goto exit;
    }
    if (image2->createPyramid() != NVMEDIA_STATUS_OK)
    {
        cerr << "image2 createPyramid failed\n";
        goto exit;
    }

    if (!pNvMOfa->getOutSurfaceParams(&outWidth, &outHeight, &gridSizeLog2X, &gridSizeLog2Y))
    {
        cerr << "NvMOfa getOutSurfaceParams api failed\n";
        goto exit;
    }
    for (i = 0; i < FlowPass; i++)
    {
        ofaOutSurface[i] = new ImagePyramid(16U, RG16, outWidth, outHeight, pNvMOfa->getNumOfPydLevels());
        if (ofaOutSurface[i] == NULL)
        {
            cerr << "ofaOutSurface class creation failed\n";
            goto exit;
        }
        if (ofaOutSurface[i]->createPyramid() != NVMEDIA_STATUS_OK)
        {
            cerr << "ofaOutSurface createPyramid failed\n";
            goto exit;
        }

        costSurface[i] = new ImagePyramid(8U, A8, outWidth, outHeight, pNvMOfa->getNumOfPydLevels());
        if (costSurface[i] == NULL)
        {
            cerr << "CostSurface class creation failed\n";
            goto exit;
        }
        if (costSurface[i]->createPyramid() != NVMEDIA_STATUS_OK)
        {
            cerr << "CostSurface createPyramid failed\n";
            goto exit;
        }
        cout << "Input and Output Pyramid creation successful\n";
        if (of_args.median == 1 || of_args.median == 2)
        {
            medianSurface[i] = new ImageBuffer(16U, RG16, outWidth[0], outHeight[0]);
            if (medianSurface[i] == NULL)
            {
                cerr << "medianSurface class creation failed\n";
                goto exit;
            }
            if (medianSurface[i]->createBuffer() != NVMEDIA_STATUS_OK)
            {
                cerr << "medianSurface createBuffer failed\n";
                goto exit;
            }
        }
    
        if (of_args.upsample == 1)
        {
            usSurface[i] = new ImageBuffer(16U, RG16, p_im_read_1->getWidth(), p_im_read_1->getHeight());
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
            FBSurface = new ImageBuffer(16U, RG16, p_im_read_1->getWidth(), p_im_read_1->getHeight());
            if (FBSurface == NULL)
            {
                cerr << "FBSurface class creation failed\n";
                goto exit;
            }
            if (FBSurface->createBuffer() != NVMEDIA_STATUS_OK)
            {
                cerr << "FBSurface createBuffer failed\n";
                goto exit;
            }

            fbcheck = new FBCheckFlow(p_im_read_1->getWidth(), p_im_read_1->getHeight());
            if (fbcheck == NULL)
            {
                cerr << "FBCheckFlow class creation failed\n";
                goto exit;
            }
            else
            {
                if (!fbcheck->initialize())
                {
                    cerr << "initialize method of FB check class failed\n";
                    goto exit;
                }
            }
        }
    }
    if (of_args.median == 1 || of_args.median == 2)
    {
        median = new OFMedianFilterCPU(of_args.median , 3, outWidth[0], outHeight[0]);
        if (median == NULL)
        {
            cerr << "OFMedianFilter class creation failed\n";
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
    if (of_args.upsample == 1)
    {
        upSample = new NNUpSampleFlow(outWidth[0], outHeight[0], p_im_read_1->getWidth(), p_im_read_1->getHeight(), gridSizeLog2X[0]);
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

    if (!RegisterSurfaces(pNvMOfa, image2, image1, ofaOutSurface[0], costSurface[0]))
    {
        cerr << "RegisterSurfaces failed\n";
        goto exit;
    }
    if (of_args.fbCheck == 1)
    {
        if (!RegisterSurfaces(pNvMOfa, ofaOutSurface[1], costSurface[1]))
        {
            cerr << "RegisterSurfaces FB failed\n";
            goto exit;
        }
        cout << "Surface registration successful\n";
    }

    if (!pNvMOfa->registerSciSyncToNvMOfa(pSync->getSyncObj(), NVMEDIA_EOFSYNCOBJ))
    {
        cout << "registerSciSyncToNvMOfa API failed \n";
        goto exit;
    }
    cout << "Surface registration successful\n";

    if (of_args.overrideParam)
    {
        if (!pNvMOfa->setSGMFlowConfigParams(SGMFlowParams))
        {
            cerr << "NvMOfa set sgm params api failed\n";
            goto exit;
        }
        if (!pNvMOfa->getSGMFlowConfigParams(SGMFlowParams))
        {
            cerr << "NvMOfa get sgm params api failed\n";
            goto exit;
        }
        cout << "NvMOfa set and get params successful\n";
    }

    while(frames < of_args.nframes)
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
        cout << "Reading the input image files successful\n";
        if (image1->writePyramid(reinterpret_cast<uint8_t *>(p_im_read_1->current_item()), isPNG) != NVMEDIA_STATUS_OK)
        {
            cerr << "write pyramid failed \n";
            goto exit;
        }
        if (image2->writePyramid(reinterpret_cast<uint8_t *>(p_im_read_2->current_item()), isPNG) != NVMEDIA_STATUS_OK)
        {
            cerr << "write pyramid failed \n";
            goto exit;
        }

        for (i = 0; i < FlowPass; i++)
        {
            if (!pNvMOfa->setSciSyncObjToNvMOfa(pSync->getSyncObj()))
            {
                cerr << "NvMOfa setSciSyncObjToNvMOfa api failed\n";
                goto exit;
            }
            if (i == 0)
            {
                if (!pNvMOfa->processSurfaceNvMOfa(image2, image1, ofaOutSurface[i], costSurface[i]))
                {
                    cerr << "NvMOfa processSurfaceNvMOfa api failed\n";
                    goto exit;
                }
            }
            else
            {
            if (!pNvMOfa->processSurfaceNvMOfa(image1, image2, ofaOutSurface[i], costSurface[i]))
                {
                    cerr << "NvMOfa processSurfaceNvMOfa api failed\n";
                    goto exit;
                }
            }

            if (!pNvMOfa->getEOFSciSyncFence(pSync->getSyncObj() , &preFence))
            {
                cerr << "NvMOfa getEOFSciSyncFence api failed\n";
                goto exit;
            }

            cout << "NvMOfa process frame successful\n";
            cout << "Waiting on post fence\n";

            if (!pSync->checkOpDone(&preFence))
            {
                cerr << "SciSyncImpl checkOpDone api failed\n";
                goto exit;
            }
            if (of_args.profile != 0)
            {
                if (!pNvMOfa->printProfileData())
                {
                cerr << "NvMOfa printProfileData api failed\n";
                goto exit;
                }
            }

            outSurface[i] = ofaOutSurface[i]->getImageBuffer(0);
            if (median != NULL)
            {
                cout <<  "median filter processing\n";
                if (!median->process(ofaOutSurface[i]->getImageBuffer(0), medianSurface[i]))
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

            if (outFile[i])
            {
                if (!outFile[i]->writeSurface(outSurface[i]))
                {
                    cerr << "writeSurface failed for output surface\n";
                    goto exit;
                }
            }
        }
        if (fbcheck != NULL)
        {
            cout <<  "FB check processing using threshold " << of_args.fbCheckThr << "\n";
            if (!fbcheck->process(outSurface[0], outSurface[1], FBSurface, of_args.fbCheckThr))
            {
                cerr << "FBcheck process failed\n";
                goto exit;
            }
            if (outFile[2] != NULL)
            {
                if (!outFile[2]->writeSurface(FBSurface))
                {
                    cerr << "writeSurface failed for output surface\n";
                    goto exit;
                }
            }
        }

        if (pOutCRCGen)
        {
            if (fbcheck != NULL)
            {
                CrcSurface = FBSurface;
            }
            else
            {
                CrcSurface = outSurface[0];
            }
            if (!pOutCRCGen->CalculateCRC(CrcSurface))
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
        if (pOutCRCCmp)
        {
            if (fbcheck != NULL)
            {
                CrcSurface = FBSurface;
            }
            else
            {
                CrcSurface = outSurface[0];
            }
            if (!pOutCRCCmp->CalculateCRC(CrcSurface))
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
               cout << "/** CmpCRC comparison passed **/\n";
            }
        }

        if (costFile)
        {
            if (!costFile->writeSurface(costSurface[0]->getImageBuffer(0)))
            {
                cerr << "Writing cost bin file failed\n";
                goto exit;
            }
        }

        cout << "Output surface write successful for frame Num " <<frames <<"\n";
        frames++;
    }
    cout << "NvMOfa Optical Flow Estimation finish!\n";

exit:
    if (pNvMOfa != NULL)
    {
        UnRegisterSurfaces(pNvMOfa, image1, image2, ofaOutSurface[0], costSurface[0]);
        UnRegisterSurfaces(pNvMOfa, ofaOutSurface[1], costSurface[1]);
        if (pSync != NULL)
        {
            pNvMOfa->unRegisterSciSyncToNvMOfa(pSync->getSyncObj());
            delete pSync;
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
    if (fbcheck != NULL)
    {
        fbcheck->release();
        delete fbcheck;
    }
    for (i = 0; i < FlowPass; i++)
    {

        if (costSurface[i] != NULL)
        {
            costSurface[i]->destroyPyramid();
            delete costSurface[i];
        }
        if (ofaOutSurface[i] != NULL)
        {
            ofaOutSurface[i]->destroyPyramid();
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
    if (FBSurface != NULL)
    {
        FBSurface->destroyBuffer();
        delete FBSurface;
    }
    for (i = 0; i < 3; i++)
    {
        if (outFile[i] != NULL)
        {
            delete outFile[i];
        }
    }
    if (image1 != NULL)
    {
        image1->destroyPyramid();
        delete image1;
    }
    if (image2 != NULL)
    {
        image2->destroyPyramid();
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

ImageReader *create_flow_image_reader(FlowTestArgs *args, uint32_t frame_num, bool *isPNG)
{
    string filename;
    string ext;
    ImageReader *p_im_read = NULL;

    ext = GET_EXTENSION(args->inputFilename);
    cout << "extension: " << ext << endl;
    if ((ext == "png") || (ext =="PNG"))
    {
#if PNG_SUPPORT
        filename = (frame_num == 1U) ? args->refFilename : args->inputFilename;
        p_im_read = new PNGReader(0, 0, args->bitdepth, YUV_400, filename);
        *isPNG = true;
#else
        cerr << "PNG is not supported\n";
#endif

    }
    else  if ((ext == "yuv") || (ext == "yuv444") || (ext == "yuv400") || (ext == "yuv420") || (ext == "yuv422"))
    {
        p_im_read = new YUVReader(args->height, args->width, args->bitdepth, args->chromaFormat, args->inputFilename, frame_num);
        *isPNG = false;
    }
    else
    {
        cerr << "Only YUV and PNG extension is supported\n";
    }

    return p_im_read;
}
