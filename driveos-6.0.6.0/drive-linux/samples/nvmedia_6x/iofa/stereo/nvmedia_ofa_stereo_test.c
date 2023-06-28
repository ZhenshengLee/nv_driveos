/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "commandline.h"
#include "scibuf_utils.h"

#ifdef ENABLE_PLAYFAIR
#include "nvplayfair.h"
#endif

#define NVSCISYNC_ATTR_FREE(nvscisyncAttrList)    \
    if( nvscisyncAttrList != NULL) {              \
        NvSciSyncAttrListFree(nvscisyncAttrList); \
        nvscisyncAttrList = NULL ;                \
    }

#define NVMTEST_CHECK(condition, code, message, ...)    \
    if( !(condition)) {                                 \
        LOG_MSG(message, ## __VA_ARGS__);               \
        code ;                                          \
    }

NvSciSyncObj syncObj_nvm_cpu = NULL;
NvSciSyncObj syncObj_cpu_nvm = NULL;
NvSciSyncCpuWaitContext cpu_wait_context = NULL;

static NvMediaStatus
nvm_signaler_cpu_waiter_init(NvMediaIofa *ofa, NvSciSyncModule module)
{
    NvSciSyncAttrList nvm_signalerAttrList = NULL;
    NvSciSyncAttrList cpu_waiterAttrList = NULL;
    NvSciSyncAttrList unreconciledList[2] = {NULL};
    NvSciSyncAttrList reconciledList = NULL;
    NvSciSyncAttrList newConflictList = NULL;
    NvSciSyncAttrKeyValuePair keyValue[2] = {0};
    bool cpuWaiter = true;
    NvSciError err;
    NvMediaStatus nvmstatus = NVMEDIA_STATUS_OK;

    /* Alloc CPU wait context */
    err = NvSciSyncCpuWaitContextAlloc(module, &cpu_wait_context);
    NVMTEST_CHECK(err == NvSciError_Success,
                  nvmstatus = NVMEDIA_STATUS_ERROR; goto fail,
                  "NvSciSyncCpuWaitContextAlloc failed\n");

    /* Create NvSciSyncAttrList for signaler*/
    err = NvSciSyncAttrListCreate(module, &nvm_signalerAttrList);
    NVMTEST_CHECK(err == NvSciError_Success,
                  nvmstatus = NVMEDIA_STATUS_ERROR; goto fail,
                  "NvSciSyncAttrListCreate failed\n");

     /* Fill  NvSciSyncAttrList nvm signaler*/
    nvmstatus = NvMediaIOFAFillNvSciSyncAttrList(ofa, nvm_signalerAttrList, NVMEDIA_SIGNALER);
    NVMTEST_CHECK( nvmstatus == NVMEDIA_STATUS_OK, goto fail,
                    "NvMediaIOFAFillNvSciSyncAttrList failed \n");

    /* Create NvSciSyncAttrList for waiter*/
    err = NvSciSyncAttrListCreate(module, &cpu_waiterAttrList);
    NVMTEST_CHECK(err == NvSciError_Success,
                  nvmstatus = NVMEDIA_STATUS_ERROR; goto fail,
                  "NvSciSyncAttrListCreate failed\n");
    /* Fill  NvSciSyncAttrList cpu waiter*/
    keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void *)&cpuWaiter;
    keyValue[0].len = sizeof(cpuWaiter);
    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
    keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*)&cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);

    err = NvSciSyncAttrListSetAttrs(cpu_waiterAttrList, keyValue, 2);
    NVMTEST_CHECK(err == NvSciError_Success,
                  nvmstatus = NVMEDIA_STATUS_ERROR; goto fail,
                  "NvSciSyncAttrListSetAttrs failed\n");

    unreconciledList[0] = nvm_signalerAttrList;
    unreconciledList[1] = cpu_waiterAttrList;

    /* Reconcile Signaler and Waiter NvSciSyncAttrList */
    err = NvSciSyncAttrListReconcile(unreconciledList, 2, &reconciledList,
            &newConflictList);
    NVMTEST_CHECK(err == NvSciError_Success,
                  nvmstatus = NVMEDIA_STATUS_ERROR; goto fail,
                  "NvSciSyncAttrListReconcile failed\n");

    /* Create NvSciSync object and get the syncObj */
    err = NvSciSyncObjAlloc(reconciledList, &syncObj_nvm_cpu);
    NVMTEST_CHECK(err == NvSciError_Success,
                  nvmstatus = NVMEDIA_STATUS_ERROR; goto fail,
                  "NvSciSyncObjAlloc failed\n");

fail:
    NVSCISYNC_ATTR_FREE(reconciledList);
    NVSCISYNC_ATTR_FREE(newConflictList);
    NVSCISYNC_ATTR_FREE(nvm_signalerAttrList);
    NVSCISYNC_ATTR_FREE(cpu_waiterAttrList);
    return nvmstatus;
}

static NvMediaStatus
cpu_signaler_nvm_waiter_init(NvMediaIofa *ofa, NvSciSyncModule module)
{
    NvSciSyncAttrList cpu_signalerAttrList = NULL;
    NvSciSyncAttrList nvm_waiterAttrList = NULL;
    NvSciSyncAttrList unreconciledList[2] = {NULL};
    NvSciSyncAttrList reconciledList = NULL;
    NvSciSyncAttrList newConflictList = NULL;
    NvSciError err;
    NvMediaStatus nvmstatus = NVMEDIA_STATUS_OK;
    NvSciSyncAttrKeyValuePair keyValue[2] = {0};
    bool cpuSignaler = true;

    /* Create NvSciSyncAttrList for signaler*/
    err = NvSciSyncAttrListCreate(module, &cpu_signalerAttrList);
    NVMTEST_CHECK(err == NvSciError_Success,
                  nvmstatus = NVMEDIA_STATUS_ERROR; goto fail,
                  "NvSciSyncAttrListCreate failed\n");

    /* Fill NvSciSyncAttrList for CPU signaler */
    keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void*) &cpuSignaler;
    keyValue[0].len = sizeof(cpuSignaler);
    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_SignalOnly;
    keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*)&cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);
    err = NvSciSyncAttrListSetAttrs(cpu_signalerAttrList, keyValue, 2);
    NVMTEST_CHECK(err == NvSciError_Success,
                  nvmstatus = NVMEDIA_STATUS_ERROR; goto fail,
                  "NvSciSyncAttrListSetAttrs failed\n");

    /* Create NvSciSyncAttrList for waiter*/
    err = NvSciSyncAttrListCreate(module, &nvm_waiterAttrList);
    NVMTEST_CHECK(err == NvSciError_Success,
                  nvmstatus = NVMEDIA_STATUS_ERROR; goto fail,
                  "NvSciSyncAttrListCreate failed\n");
    /* Fill NvSciSyncAttrList for nvmedia waiter */
    nvmstatus = NvMediaIOFAFillNvSciSyncAttrList(ofa, nvm_waiterAttrList, NVMEDIA_WAITER);
    NVMTEST_CHECK( nvmstatus == NVMEDIA_STATUS_OK, goto fail,
                    "NvMFillNvSciSyncAttrList failed \n");

    unreconciledList[0] = cpu_signalerAttrList;
    unreconciledList[1] = nvm_waiterAttrList;

    /* Reconcile Signaler and Waiter NvSciSyncAttrList */
    err = NvSciSyncAttrListReconcile(unreconciledList, 2, &reconciledList,
            &newConflictList);
    NVMTEST_CHECK(err == NvSciError_Success,
                  nvmstatus = NVMEDIA_STATUS_ERROR; goto fail,
                  "NvSciSyncAttrListReconcile failed\n");

    /* Create NvSciSync object and get the syncObj */
    err = NvSciSyncObjAlloc(reconciledList, &syncObj_cpu_nvm);
    NVMTEST_CHECK(err == NvSciError_Success,
                  nvmstatus = NVMEDIA_STATUS_ERROR; goto fail,
                  "NvSciSyncObjAlloc failed\n");

fail:
    /* Free Attribute list objects */
    NVSCISYNC_ATTR_FREE(reconciledList);
    NVSCISYNC_ATTR_FREE(newConflictList);
    NVSCISYNC_ATTR_FREE(cpu_signalerAttrList);
    NVSCISYNC_ATTR_FREE(nvm_waiterAttrList);

    return nvmstatus;
}

static uint32_t
GetFileSize (
    char *inputFilename
)
{
    FILE *inputFile;
    uint32_t fileLength;

    if (inputFilename == NULL)
    {
        LOG_ERR("GetFileSize: Null file name\n");
        return 0;
    }
    inputFile = fopen(inputFilename, "rb");
    if (!inputFile)
    {
        LOG_ERR("GetFileSize: file %s Open failed \n", inputFilename);
        return 0;
    }
    fseeko(inputFile, 0, SEEK_END);
    fileLength = ftell(inputFile);
    fclose(inputFile);

    return fileLength;
}

static uint32_t
GetImageSize (
    TestArgs *args
)
{
    uint32_t imageSize;

    imageSize = args->width*args->height;
    switch (args->chromaFormat)
    {
        case YUV_400:
            imageSize = imageSize * 1;
            break;
        case YUV_420:
            imageSize = imageSize * 3/2;
            break;
        case YUV_422:
            imageSize = imageSize * 2;
            break;
        case YUV_444:
            imageSize = imageSize * 3;
            break;
        default:
            imageSize = imageSize * 1;
            break;
    }
    if (args->bitDepth > 8)
    {
        imageSize = imageSize * 2;
    }

    return imageSize;
}

static uint32_t
GetNumFrames (
    TestArgs *args
)
{
    uint32_t leftFileSize, rightFileSize, frameSize, numFrames;

    leftFileSize = GetFileSize(args->leftFilename);
    rightFileSize = GetFileSize(args->rightFilename);
    frameSize = GetImageSize(args);
    if (frameSize == 0)
    {
        return 0;
    }
    if (leftFileSize < rightFileSize)
    {
        numFrames = leftFileSize/frameSize;
    }
    else
    {
        numFrames = rightFileSize/frameSize;
    }

    return numFrames;
}

static bool
GetFormatType(
    uint32_t           bitDepth,
    ChromaFormatStereo chromaFormat,
    ChromaFormat       *chromaTarget
)
{
    if(bitDepth == 8)
    {
        switch(chromaFormat)
        {
            case YUV_400:
                *chromaTarget = YUV400P_8bit;
                break;
            case YUV_420:
                *chromaTarget = YUV420SP_8bit;
                break;
            case YUV_422:
                *chromaTarget = YUV422SP_8bit;
                break;
            case YUV_444:
                *chromaTarget = YUV444SP_8bit;
                break;
            default:
                return false;
        }
    }
    else if(bitDepth == 10)
    {
        switch(chromaFormat)
        {
            case YUV_400:
                *chromaTarget = YUV400P_10bit;
                break;
            case YUV_420:
                *chromaTarget = YUV420SP_10bit;
                break;
            case YUV_422:
                *chromaTarget = YUV422SP_10bit;
                break;
            case YUV_444:
                *chromaTarget = YUV444SP_10bit;
                break;
            default:
                return false;
        }
    }
    else if(bitDepth == 12)
    {
        switch(chromaFormat)
        {
            case YUV_400:
                *chromaTarget = YUV400P_12bit;
                break;
            case YUV_420:
                *chromaTarget = YUV420SP_12bit;
                break;
            case YUV_422:
                *chromaTarget = YUV422SP_12bit;
                break;
            case YUV_444:
                *chromaTarget = YUV444SP_12bit;
                break;
            default:
                return false;
        }
    }
    else if(bitDepth == 16)
    {
        switch(chromaFormat)
        {
            case YUV_400:
                *chromaTarget = YUV400P_16bit;
                break;
            case YUV_420:
                *chromaTarget = YUV420SP_16bit;
                break;
            case YUV_422:
                *chromaTarget = YUV422SP_16bit;
                break;
            case YUV_444:
                *chromaTarget = YUV444SP_16bit;
                break;
            default:
                return false;
        }
    }
    else
    {
        return false;
    }
    return true;
}

static bool
GetFileType(
    uint32_t           bitDepth,
    ChromaFormatStereo chromaFormat,
    ChromaFormat       *chromaTarget)
{
    if(bitDepth == 8)
    {
        switch(chromaFormat)
        {
            case YUV_400:
                *chromaTarget = YUV400P_8bit;
                break;
            case YUV_420:
                *chromaTarget = YUV420P_8bit;
                break;
            case YUV_422:
                *chromaTarget = YUV422P_8bit;
                break;
            case YUV_444:
                *chromaTarget = YUV444P_8bit;
                break;
            default:
                return false;
        }

    }
    else if(bitDepth == 10)
    {
        switch(chromaFormat)
        {
            case YUV_400:
                *chromaTarget = YUV400P_10bit;
                break;
            case YUV_420:
                *chromaTarget = YUV420P_10bit;
                break;
            case YUV_422:
                *chromaTarget = YUV422P_10bit;
                break;
            case YUV_444:
                *chromaTarget = YUV444P_10bit;
                break;
            default:
                return false;
        }
    }
    else if(bitDepth == 12)
    {
        switch(chromaFormat)
        {
            case YUV_400:
                *chromaTarget = YUV400P_12bit;
                break;
            case YUV_420:
                *chromaTarget = YUV420P_12bit;
                break;
            case YUV_422:
                *chromaTarget = YUV422P_12bit;
                break;
            case YUV_444:
                *chromaTarget = YUV444P_12bit;
                break;
            default:
                return false;
        }
    }
    else if(bitDepth == 16)
    {
        switch(chromaFormat)
        {
            case YUV_400:
                *chromaTarget = YUV400P_16bit;
                break;
            case YUV_420:
                *chromaTarget = YUV420P_16bit;
                break;
            case YUV_422:
                *chromaTarget = YUV422P_16bit;
                break;
            case YUV_444:
                *chromaTarget = YUV444P_16bit;
                break;
            default:
                return false;
        }
    }
    else
    {
        return false;
    }
    return true;
}

static bool
allocateSurfaces (
    TestArgs      *args,
    NvMediaIofa    *testOFA,
    StereoTestCtx *ctx
)
{
    bool ret = true;
    NvSciError err;
    NvMediaStatus status;
    uint32_t i, j, planeCount;
    ChromaFormat InputFormat;
    NvSciBufModule bufModule;
    NvSciBufAttrList attributeList=NULL;
    NvSciBufAttrList outattributeList=NULL;
    NvSciBufAttrList costattributeList=NULL;
    NvSciBufAttrList reconciledAttributeList=NULL;
    NvSciBufAttrList outreconciledAttributeList=NULL;
    NvSciBufAttrList costreconciledAttributeList=NULL;
    NvSciBufAttrList conflictList=NULL, outconflictList=NULL, costconflictList=NULL; //This is to check if there is any error while conciling

    if (args->chromaFormat == YUV_400)
    {
        planeCount = 1;
    }
    else
    {
        planeCount = 2;
    }
    if (!GetFormatType(args->bitDepth, args->chromaFormat, &InputFormat))
    {
        LOG_ERR("allocateSurfaces: Incorrect Input format Type\n");
        return false;
    }
    //Opening NvSciBuf Module
    err = NvSciBufModuleOpen(&bufModule);
    if (err != NvSciError_Success)
    {
        LOG_ERR("allocateSurfaces: NvSciBufModuleOpen failed\n");
        return false;
    }

    err = NvSciBufAttrListCreate(bufModule, &attributeList);
    if (err != NvSciError_Success)
    {
        LOG_ERR("allocateSurfaces: NvSciBufAttrListCreate failed\n");
        return false;
    }

    status = NvMediaIOFAFillNvSciBufAttrList(attributeList);
    if (NVMEDIA_STATUS_OK != status)
    {
        LOG_ERR("allocateSurfaces: Failed to fill IOFA internal attributes for input surface %d\n", status);
        return false;
    }

    status = PopulateNvSciBufAttrList(InputFormat, ctx->width, ctx->height, true, NvSciBufImage_BlockLinearType, planeCount, NvSciBufAccessPerm_ReadWrite, 256U, NvSciColorStd_REC601_ER, NvSciBufScan_ProgressiveType, attributeList);
    if (NVMEDIA_STATUS_OK != status)
    {
        LOG_ERR("allocateSurfaces: Failed to fill IOFA external attributes for input surface\n");
        return false;
    }

    err =  NvSciBufAttrListReconcile(&attributeList, 1U, &reconciledAttributeList, &conflictList);
    if (err != NvSciError_Success)
    {
        LOG_ERR("allocateSurfaces: Reconciliation for input frame failed for input surface\n");
        return false;
    }

    for (i = 0; i < args->inputBuffering; i++)
    {
        for (j = 0; j < 2; j++)
        {
            err =  NvSciBufObjAlloc(reconciledAttributeList, &ctx->inputFrame[i][j]);
            if (err != NvSciError_Success)
            {
                LOG_ERR("allocateSurfaces: NvSciBuf Obj creation for input surface failed\n");
                return false;
            }
            err = NvSciBufObjDupWithReducePerm(ctx->inputFrame[i][j],
                                               NvSciBufAccessPerm_Readonly,
                                               &ctx->inputFrameDup[i][j]);
            if (err != NvSciError_Success)
            {
                LOG_ERR("allocateSurfaces: NvSciBufObjDupWithReducePerm for input surface failed\n");
                return false;
            }
        }
    }

    err = NvSciBufAttrListCreate(bufModule, &outattributeList);
    if (err != NvSciError_Success)
    {
        LOG_ERR("allocateSurfaces: NvSciBufAttrListCreate failed for out buffer\n");
        return false;
    }

    err = NvSciBufAttrListCreate(bufModule, &costattributeList);
    if (err != NvSciError_Success)
    {
        LOG_ERR("allocateSurfaces: NvSciBufAttrListCreate failed for cost buffer\n");
        return false;
    }

    status = NvMediaIOFAFillNvSciBufAttrList(outattributeList);
    if (NVMEDIA_STATUS_OK != status)
    {
        LOG_ERR("allocateSurfaces: Failed to fill IOFA internal attributes for output surface\n");
        return false;
    }

    status = NvMediaIOFAFillNvSciBufAttrList(costattributeList);
    if (NVMEDIA_STATUS_OK != status)
    {
        LOG_ERR("allocateSurfaces: Failed to fill IOFA internal attributes for cost surface\n");
        return false;
    }

    status = PopulateNvSciBufAttrList(A16, ctx->outWidth, ctx->outHeight, true, NvSciBufImage_BlockLinearType, 1U, NvSciBufAccessPerm_ReadWrite, 256U, NvSciColorStd_REC601_ER, NvSciBufScan_ProgressiveType, outattributeList);
    if (NVMEDIA_STATUS_OK != status)
    {
        LOG_ERR("allocateSurfaces: Failed to fill IOFA external attributes for output surface\n");
        return false;
    }

    status = PopulateNvSciBufAttrList(A8, ctx->outWidth, ctx->outHeight, true, NvSciBufImage_BlockLinearType, 1U, NvSciBufAccessPerm_ReadWrite, 256U, NvSciColorStd_REC601_ER, NvSciBufScan_ProgressiveType, costattributeList);
    if (NVMEDIA_STATUS_OK != status)
    {
        LOG_ERR("allocateSurfaces: Failed to fill IOFA external attributes for cost surface\n");
        return false;
    }

    err =  NvSciBufAttrListReconcile(&outattributeList, 1U, &outreconciledAttributeList, &outconflictList);
    if (err != NvSciError_Success)
    {
        LOG_ERR("allocateSurfaces: Reconciliation for output/cost frame failed\n");
        return false;
    }

    err =  NvSciBufAttrListReconcile(&costattributeList, 1U, &costreconciledAttributeList, &costconflictList);
    if (err != NvSciError_Success)
    {
        LOG_ERR("allocateSurfaces: Reconciliation for output/cost frame failed\n");
        return false;
    }

    for (i = 0; i < args->inputBuffering; i++)
    {
        // create output surface based on out format, output width and height returned by testOfa instance
        err =  NvSciBufObjAlloc(outreconciledAttributeList, &ctx->outputSurface[i]);
        if (err != NvSciError_Success)
        {
            LOG_ERR("allocateSurfaces: NvSciBuf Obj creation for output frame failed\n");
            return false;
        }

        // create cost surface based on cost format, output width and height returned by testOfa instance
        err =  NvSciBufObjAlloc(costreconciledAttributeList, &ctx->costSurface[i]);
        if (err != NvSciError_Success)
        {
            LOG_ERR("allocateSurfaces: NvSciBuf Obj creation for cost frame failed\n");
            return false;
        }
    }

    NvSciBufAttrListFree(attributeList);
    NvSciBufAttrListFree(outattributeList);
    NvSciBufAttrListFree(costattributeList);
    NvSciBufAttrListFree(reconciledAttributeList);
    NvSciBufAttrListFree(outreconciledAttributeList);
    NvSciBufAttrListFree(costreconciledAttributeList);
    NvSciBufAttrListFree(conflictList);
    NvSciBufAttrListFree(outconflictList);
    NvSciBufAttrListFree(costconflictList);
    if (NULL != bufModule)
    {
        NvSciBufModuleClose(bufModule);
    }

    return ret;
}

static bool
PrintVersion (
    void
)
{
    NvMediaVersion version;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = NvMediaIOFAGetVersion(&version);
    if (status != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("PrintVersion: NvMediaIOFAGetVersion failed\n");
        return false;
    }
    if ((version.major != NVMEDIA_IOFA_VERSION_MAJOR) || (version.minor != NVMEDIA_IOFA_VERSION_MINOR))
    {
        LOG_ERR("PrintVersion: NvMediaIOFAGetVersion version mismatch\n");
        return false;
    }
    LOG_INFO("PrintVersion: NvMediaIOFA Major version - %d \n", NVMEDIA_IOFA_VERSION_MAJOR);
    LOG_INFO("PrintVersion: NvMediaIOFA Minor version - %d \n", NVMEDIA_IOFA_VERSION_MINOR);

    return true;
}

static void
UnRegisterSurfaces (
    const NvMediaIofa   *testOFA,
    const StereoTestCtx *ctx,
    uint8_t             inputBuffering
)
{
    uint8_t i, j;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    LOG_DBG("UnRegisterSurfaces: UnRegister Input surfaces, output and cost surfaces \n");

    for (i = 0; i < inputBuffering; i++)
    {
        for (j = 0; j < 2; j++)
        {
            if (ctx->inputFrameDup[i][j] != NULL)
            {
                status = NvMediaIOFAUnregisterNvSciBufObj(testOFA, ctx->inputFrameDup[i][j]);
                if (status != NVMEDIA_STATUS_OK)
                {
                    LOG_ERR("UnRegisterSurfaces: NvMediaIOFAUnregisterNvSciBufObj input frame failed\n");
                }
            }
        }
        if (ctx->outputSurface[i] != NULL)
        {
            status = NvMediaIOFAUnregisterNvSciBufObj(testOFA, ctx->outputSurface[i]);
            if (status != NVMEDIA_STATUS_OK)
            {
                LOG_ERR("UnRegisterSurfaces: NvMediaIOFAUnregisterNvSciBufObj output frame failed\n");
            }
        }
        if (ctx->costSurface[i] != NULL)
        {
            status = NvMediaIOFAUnregisterNvSciBufObj(testOFA, ctx->costSurface[i]);
            if (status != NVMEDIA_STATUS_OK)
            {
                LOG_ERR("UnRegisterSurfaces: NvMediaIOFAUnregisterNvSciBufObj cost frame failed\n");
            }
        }
    }
}

static bool
ValidateAndOpenROIFile (
    TestArgs *args,
    FILE     **stereoROIFile
)
{
    if (args->roiMode == 1)
    {
        *stereoROIFile = fopen(args->roiFilename, "r");
        if (*stereoROIFile == NULL)
        {
            LOG_ERR("NVMEDIA_IOFA_MAIN: Cannot open stereo ROI file for reading\n");
            return false;
        }
    }
    return true;
}

static bool
ValidateAndOpenCRCFiles (
    TestArgs *args,
    FILE     **stereoCrcFile,
    FILE     **costCrcFile
)
{
    // CRC mode check
    if (args->stereoCrcoption.crcGenMode && args->stereoCrcoption.crcCheckMode)
    {
        LOG_ERR("ValidateAndOpenCRCFiles: FlowCrcGenMode and FlowCrcCheckMode cannot be enabled at the same time\n");
        return false;
    }
    if (args->costCrcoption.crcGenMode && args->costCrcoption.crcCheckMode)
    {
        LOG_ERR("ValidateAndOpenCRCFiles: costCrcGenMode and costCrcCheckMode cannot be enabled at the same time\n");
        return false;
    }

    // Open STEREO CRC file
    if (args->stereoCrcoption.crcGenMode)
    {
        *stereoCrcFile = fopen(args->stereoCrcoption.crcFilename, "wt");
        if (*stereoCrcFile == NULL)
        {
            LOG_ERR("ValidateAndOpenCRCFiles: Cannot open stereo Crc gen file for writing\n");
            return false;
        }
    }
    else if (args->stereoCrcoption.crcCheckMode)
    {
        *stereoCrcFile = fopen(args->stereoCrcoption.crcFilename, "rb");
        if (*stereoCrcFile == NULL)
        {
            LOG_ERR("ValidateAndOpenCRCFiles: Cannot open stereo Crc gen file for reading\n");
            return false;
        }
    }

    // Open COST CRC file
    if (args->costCrcoption.crcGenMode)
    {
        *costCrcFile = fopen(args->costCrcoption.crcFilename, "wt");
        if (*costCrcFile == NULL)
        {
            LOG_ERR("ValidateAndOpenCRCFiles: Cannot open cost Crc gen file for writing\n");
            return false;
        }
    }
    else if (args->costCrcoption.crcCheckMode)
    {
        *costCrcFile = fopen(args->costCrcoption.crcFilename, "rb");
        if (*costCrcFile == NULL)
        {
            LOG_ERR("ValidateAndOpenCRCFiles: Cannot open cost Crc gen file for reading\n");
            return false;
        }
    }

    return true;
}

static void
crcFileLog (
    TestArgs *args
)
{
    if (args->stereoCrcoption.crcGenMode)
    {
        LOG_MSG("\n crcFileLog: ***STEREO crc gold file %s has been generated***\n", args->stereoCrcoption.crcFilename);
    }
    else if (args->stereoCrcoption.crcCheckMode)
    {
        LOG_MSG("\n crcFileLog: ***STEREO crc checking with file %s is successful\n", args->stereoCrcoption.crcFilename);
    }

    if (args->costCrcoption.crcGenMode)
    {
        LOG_MSG("\n crcFileLog: ***COST crc gold file %s has been generated***\n", args->costCrcoption.crcFilename);
    }
    else if (args->costCrcoption.crcCheckMode)
    {
        LOG_MSG("\n crcFileLog: ***COST crc checking with file %s is successful\n", args->costCrcoption.crcFilename);
    }
    return;
}

static void OverrideSGMParam (
    const TestArgs      *args,
    NvMediaIofaSGMParams *pSGMParams
)
{
    if (args->overrideParam & SGMPARAM_P1_OVERRIDE)
    {
        pSGMParams->penalty1[0U] = args->p1;
    }
    if (args->overrideParam & SGMPARAM_P2_OVERRIDE)
    {
        pSGMParams->penalty2[0U] = args->p2;
    }
    if (args->overrideParam & SGMPARAM_DIAGONAL_OVERRIDE)
    {
        pSGMParams->enableDiag[0U] = (args->diagonalMode != 0) ? true:false;
    }
    if (args->overrideParam & SGMPARAM_ADAPTIVEP2_OVERRIDE)
    {
        pSGMParams->adaptiveP2[0U] = (args->adaptiveP2 != 0) ? true:false;
    }
    if (args->overrideParam & SGMPARAM_NUMPASSES_OVERRIDE)
    {
        pSGMParams->numPasses[0U] = args->numPasses;
    }
    if (args->overrideParam & SGMPARAM_ALPHA_OVERRIDE)
    {
        pSGMParams->alphaLog2[0U] = args->alpha;
    }

    return;
}

static bool
RegisterSurfaces (
    const NvMediaIofa   *testOFA,
    const StereoTestCtx *ctx,
    uint8_t             inputBuffering
)
{
    uint8_t i, j;
    NvMediaStatus stat;

    LOG_MSG("RegisterSurfaces: Register input frames, output and cost surfaces \n");
    for (i = 0; i < inputBuffering; i++)
    {
        for (j = 0; j < 2; j++)
        {
            stat = NvMediaIOFARegisterNvSciBufObj(testOFA, ctx->inputFrameDup[i][j]);
            if (stat != NVMEDIA_STATUS_OK)
            {
                LOG_ERR("RegisterSurfaces: Register input frame failed and error code is %d\n", stat );
                return false;
            }
        }

        if (NvMediaIOFARegisterNvSciBufObj(testOFA, ctx->outputSurface[i]) != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("RegisterSurfaces: Register Output surface failed\n");
            return false;
        }

        if (NvMediaIOFARegisterNvSciBufObj(testOFA, ctx->costSurface[i]) != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("RegisterSurfaces: Register cost surface failed\n");
            return false;
        }
    }
    return true;
}

static bool
ProcessOutputSurfaces (
    const TestArgs      *args,
    const StereoTestCtx *ctx,
    const NvMediaIofa    *testOFA
)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint32_t calcCrcCost = 0, calcCrcStereo = 0, refCrc = 0;

    // Write output surface
    if (args->outputFilename != NULL)
    {
        status = WriteOutput(args->outputFilename, ctx->outputSurface[ctx->processIdx], false,
                            ctx->frameProcessed ? true : false, NULL);
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("ProcessOutputSurfaces: WriteImage for output surface failed\n");
            return false;
        }
        LOG_DBG("ProcessOutputSurfaces: Writing output stereo for Frame completed\n");
    }
    // Write cost surface
    if (args->costFilename != NULL)
    {
        status = WriteOutput(args->costFilename, ctx->costSurface[ctx->processIdx], false,
                            ctx->frameProcessed ? true : false, NULL);
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("ProcessOutputSurfaces: WriteImage for cost failed\n");
            return false;
        }
        LOG_DBG("ProcessOutputSurfaces: Writing cost completed\n");
    }

    if (args->stereoCrcoption.crcGenMode)
    {
        calcCrcStereo = 0;
        status = GetNvSciBufObjCrcNoSrcRect(ctx->outputSurface[ctx->processIdx], false, &calcCrcStereo);
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("ProcessOutputSurfaces: GetImageCrc failed: %x\n", status);
            return false;
        }
        if (ctx->stereoCrcFile != NULL)
        {
            if (!fprintf(ctx->stereoCrcFile, "%08x\n", calcCrcStereo))
           {
                LOG_ERR("ProcessOutputSurfaces: Failed writing calculated STEREO CRC to file %s\n", ctx->stereoCrcFile);
                return false;
           }
        }
    }
    else if (args->stereoCrcoption.crcCheckMode)
    {
        calcCrcStereo = 0;
        status = GetNvSciBufObjCrcNoSrcRect(ctx->outputSurface[ctx->processIdx], false, &calcCrcStereo);
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("ProcessOutputSurfaces: GetImageCrc failed: %x\n", status);
            return false;
        }
        if (ctx->stereoCrcFile != NULL)
        {
            if (fscanf(ctx->stereoCrcFile, "%8x\n", &refCrc) == 1)
            {
                if (refCrc != calcCrcStereo)
                {
                    LOG_ERR("ProcessOutputSurfaces: Frame %d STEREO CRC 0x%x does not match with ref crc 0x%x\n", ctx->frameProcessed, calcCrcStereo, refCrc);
                    return false;
                }
            }
            else
            {
                LOG_ERR("ProcessOutputSurfaces: Failed checking STEREO CRC. Failed reading file %s\n", ctx->stereoCrcFile);
                return false;
            }
        }
        else
        {
            LOG_ERR("ProcessOutputSurfaces: STEREO CRC file pointer is NULL\n");
            return false;
        }
    }
    if (args->costCrcoption.crcGenMode)
    {
        calcCrcCost = 0;
        status = GetNvSciBufObjCrcNoSrcRect(ctx->costSurface[ctx->processIdx], false , &calcCrcCost);
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("ProcessOutputSurfaces: GetImageCrc failed: %x\n", status);
            return false;
        }
        if (ctx->costCrcFile != NULL)
        {
            if (!fprintf(ctx->costCrcFile, "%08x\n", calcCrcCost))
            {
                LOG_ERR("ProcessOutputSurfaces: Failed writing calculated COST CRC to file %s\n", ctx->costCrcFile);
                return false;
            }
        }
    }
    else if (args->costCrcoption.crcCheckMode)
    {
        calcCrcCost = 0;
        status = GetNvSciBufObjCrcNoSrcRect(ctx->costSurface[ctx->processIdx], false , &calcCrcCost);
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("ProcessOutputSurfaces: GetImageCrc failed: %x\n", status);
            return false;
        }
        if (ctx->costCrcFile != NULL)
        {
            if (fscanf(ctx->costCrcFile, "%8x\n", &refCrc) == 1)
            {
                if (refCrc != calcCrcCost)
                {
                    LOG_ERR("ProcessOutputSurfaces: Frame %d COST CRC does not match with ref crc 0x%x\n", ctx->frameProcessed, refCrc);
                    return false;
                }
            }
            else
            {
                LOG_ERR("ProcessOutputSurfaces: Failed checking COST CRC. Failed reading file %s\n", ctx->costCrcFile);
                return false;
            }
        }
        else
        {
            LOG_ERR("ProcessOutputSurfaces: COST CRC file pointer is NULL\n");
            return false;
        }
    }

    return true;
}

static bool
StoreProfileData (
    const NvMediaIofa   *testOFA,
    const StereoTestCtx *ctx
)
{
    NvMediaIofaProfileData ProfData = {};
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    memset(&ProfData, 0, sizeof(NvMediaIofaProfileData));
    status = NvMediaIOFAGetProfileData(testOFA, &ProfData);
    if (status != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("StoreProfileData: NvMediaIOFAGetProfileData is failed\n");
        return false;
    }
    if ((ProfData.validProfData == true) && (ctx->frameProcessed < ctx->numFrames))
    {
        ctx->profDataArray[ctx->frameProcessed] = ProfData;
    }

    return true;
}

static void
ProcessProfileData (
    const StereoTestCtx *ctx
)
{
    if (ctx->frameProcessed == 0)
    {
        LOG_ERR("ProcessProfileData: Number of frames processed are 0");
        return;
    }
    else
    {
        float avgSwTimeInMS = 0, avgHwTimeInMS = 0, avgSyncWaitTimeInMS = 0;
        for(uint32_t i = 0; i<ctx->frameProcessed; i++)
        {
           avgSwTimeInMS += (float)ctx->profDataArray[i].swTimeInUS/1000;
           avgHwTimeInMS += (float)ctx->profDataArray[i].hwTimeInUS/1000;
           avgSyncWaitTimeInMS += (float)ctx->profDataArray[i].syncWaitTimeInUS/1000;
        }
        avgSwTimeInMS = avgSwTimeInMS/ctx->frameProcessed;
        avgHwTimeInMS = avgHwTimeInMS/ctx->frameProcessed;
        avgSyncWaitTimeInMS = avgSyncWaitTimeInMS/ctx->frameProcessed;
        LOG_MSG("ProcessProfileData: avg sw time %f ms, avg hw time %f ms, avg sync wait time %f ms \n",
                 avgSwTimeInMS,
                 avgHwTimeInMS,
                 avgSyncWaitTimeInMS);
    }
    return;
}

static bool
ParseRoiData (
    NvMediaIofaROIParams *roiparamsdata,
    FILE *roifilename,
    const StereoTestCtx *ctx
)
{
    int roi_num;
    size_t max_roisize = 500;
    char frame_roi[max_roisize];
    uint32_t roidata[max_roisize];
    uint32_t r=0;
    roi_num = fscanf(roifilename, "%[^\n] ", frame_roi);
    if (roi_num == -1)
    {
        LOG_ERR("ParseRoiData: no roi data available for frame %d \n",ctx->frameQueued);
        return false;
    }
    else
    {
        char * roitoken = strtok(frame_roi, " ");
        while( roitoken != NULL && r < (uint32_t)max_roisize)
        {
            roidata[r++] = atoi(roitoken);
            roitoken = strtok(NULL, " ");
        }
    }
    uint32_t frm_num = roidata[0];
    if (frm_num != ctx->frameQueued)
    {
        LOG_ERR("ParseRoiData: no roi data available for frame %d \n",ctx->frameQueued);
        return false;
    }
    roiparamsdata->numOfROIs = roidata[1];
    if (r < 4*roiparamsdata->numOfROIs+2)
    {
        LOG_ERR("ParseRoiData: Incomplete roi data, expects startx, starty, endx, endy");
        return false;
    }
    else
    {
        for (uint32_t numroi=0; numroi<roiparamsdata->numOfROIs; numroi++)
        {
            roiparamsdata->rectROIParams[numroi].startX = (roidata[4U*numroi +2U] );
            roiparamsdata->rectROIParams[numroi].startY = (roidata[4U*numroi +3U] );
            roiparamsdata->rectROIParams[numroi].endX = (roidata[4U*numroi +4U] );
            roiparamsdata->rectROIParams[numroi].endY = (roidata[4U*numroi +5U] );
        }
    }
    return true;
}


int main(int argc, char *argv[])
{
    TestArgs args;
    StereoTestCtx ctx;
    ChromaFormat format;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvMediaIofa *testOFA = NULL;
    NvMediaIofaInitParams ofaInitParams;
    NvMediaIofaProcessParams ofaProcessParams;
    NvMediaIofaBufArray surfArray;
    NvMediaIofaROIParams roiParams;
    NvSciError err;
    int32_t status_main = 0;
    uint32_t i, j, timeout, numFrames;
    bool skipReadImage = false;
    NvSciSyncTaskStatus taskStatus;
    NvSciSyncModule syncModule = NULL;
#ifdef ENABLE_PLAYFAIR
    uint64_t startTime = 0, stopTime = 0;
    NvpPerfData_t submissionLatencies, executionLatencies, initLatencies;
    NvpRateLimitInfo_t rateLimitInfo;
    bool perfDataConstruct = false;
#endif

    memset(&args, 0, sizeof(TestArgs));
    memset(&ctx, 0, sizeof(StereoTestCtx));
    FILE *roifilename = NULL;

/** variable args.playfair generate performance results with pre-fence support
    when the variable is true, following sequence is followed
    insert pre-fence in OFA pipeline
    call processframe API
    get eof-fence for OFA operation
    read reference and input surface
    signal pre-fence expiry to indicate input and refence surface is ready for hw operation
    wait for eof-fence for OFA operation completion.

    This sequence mimic actual use-case where we form pipeline in the beginning and then
    trigger pipeline when input is ready.

    when false, input and reference frame are initialized before calling processframe API.
**/

    if (ParseArgs(argc, argv, &args))
    {
        PrintUsage();
        return -1;
    }

    if (args.version)
    {
        if (!PrintVersion())
        {
            status_main = -1;
            goto fail;
        }
    }

    if (!GetFileType(args.bitDepth, args.chromaFormat, &format))
    {
        status_main = -1;
        goto fail;
    }

    if (!ValidateAndOpenCRCFiles(&args, &ctx.stereoCrcFile, &ctx.costCrcFile))
    {
        status_main = -1;
        goto fail;
    }

    if (!ValidateAndOpenROIFile(&args, &roifilename))
    {
        status_main = -1;
        goto fail;
    }
    numFrames = GetNumFrames(&args);
    if (args.frameIntervalInMS != 0U && numFrames > 0)
    {
        ctx.numFrames = args.numFrames;
    }
    else
    {
        if (numFrames < args.numFrames)
        {
            ctx.numFrames = numFrames;
        }
        else
        {
            ctx.numFrames = args.numFrames;
        }
    }
    if (ctx.numFrames == 0U)
    {
        LOG_ERR("NVMEDIA_IOFA_MAIN: numFrames to process can not be 0\n");
        status_main = -1;
        goto fail;
    }
    if (args.profile)
    {
        ctx.profDataArray = malloc(sizeof(NvMediaIofaProfileData)*ctx.numFrames);
    }

#ifdef ENABLE_PLAYFAIR
    if (args.playfair)
    {
        char executionLatencyFileName[FILE_NAME_SIZE] = {0};
        char submissionLatencyFileName[FILE_NAME_SIZE] = {0};

        if (args.profileStatsFilePath)
        {
            strcpy(executionLatencyFileName, args.profileStatsFilePath);
            strcpy(submissionLatencyFileName, args.profileStatsFilePath);
        }
        NvpConstructPerfData(&initLatencies, 1, NULL);
        strcat(executionLatencyFileName, "/ExecutionLatency.csv");
        NvpConstructPerfData(&executionLatencies, ctx.numFrames, executionLatencyFileName);
        strcat(submissionLatencyFileName,"/SubmissionLatency.csv");
        NvpConstructPerfData(&submissionLatencies, ctx.numFrames, submissionLatencyFileName);
        perfDataConstruct = true;
    }
    if (args.frameIntervalInMS != 0U)
    {
        NvpRateLimitInit(&rateLimitInfo, 1000U/args.frameIntervalInMS);
    }
#endif

    LOG_DBG("NVMEDIA_IOFA_MAIN: Create NvMedia IOFA instance\n");
    testOFA = NvMediaIOFACreate();
    if (testOFA == NULL)
    {
        LOG_ERR("NVMEDIA_IOFA_MAIN: NvMediaIofaCreate function failed \n");
        status_main = -1;
        goto fail;
    }

    ctx.width = args.width;
    ctx.height = args.height;
    ctx.outWidth = (ctx.width + (1U << args.gridsize) - 1) >> args.gridsize;
    ctx.outHeight = (ctx.height + (1U << args.gridsize) -1) >> args.gridsize;

    memset(&ofaInitParams, 0U, sizeof(ofaInitParams));
    ofaInitParams.ofaMode = NVMEDIA_IOFA_MODE_STEREO;
    ofaInitParams.width[0] = ctx.width;
    ofaInitParams.height[0] = ctx.height;
    ofaInitParams.gridSize[0] = args.gridsize;
    ofaInitParams.outWidth[0] = ctx.outWidth;
    ofaInitParams.outHeight[0] = ctx.outHeight;

    if (args.ndisp == 256U)
    {
        ofaInitParams.dispRange = NVMEDIA_IOFA_DISPARITY_RANGE_256;
    }
    else
    {
        ofaInitParams.dispRange = NVMEDIA_IOFA_DISPARITY_RANGE_128;
    }
    ofaInitParams.profiling = args.profile;
    ofaInitParams.preset    = args.preset;

    LOG_DBG("NVMEDIA_IOFA_MAIN: Initialize NvMedia IOFA instance\n");
#ifdef ENABLE_PLAYFAIR
    if (args.playfair)
    {
        startTime = NvpGetTimeMark();
    }
#endif
    status = NvMediaIOFAInit(testOFA, &ofaInitParams, args.inputBuffering);
    if (status != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("NVMEDIA_IOFA_MAIN: NvMediaIOFAInit function failed \n");
        status_main = -1;
        goto fail;
    }
#ifdef ENABLE_PLAYFAIR
    if (args.playfair)
    {
        stopTime = NvpGetTimeMark();
        NvpRecordSample(&initLatencies, startTime, stopTime);
    }
#endif
    LOG_DBG("NVMEDIA_IOFA_MAIN: Create input, output and cost surfaces\n");
    if (!allocateSurfaces(&args, testOFA, &ctx))
    {
        LOG_ERR("NVMEDIA_IOFA_MAIN: allocateSurfaces failed\n");
        status_main = -1;
        goto fail;
    }

    LOG_DBG("NVMEDIA_IOFA_MAIN: Register input, output and cost surfaces\n");
    if (!RegisterSurfaces(testOFA, &ctx, args.inputBuffering))
    {
        LOG_ERR("NVMEDIA_IOFA_MAIN: RegisterSurfaces failed\n");
        status_main = -1;
        goto fail;
    }

    err = NvSciSyncModuleOpen(&syncModule);
    if (err != NvSciError_Success)
    {
        LOG_ERR("NVMEDIA_IOFA_MAIN: NvSciBufModuleOpen failed\n");
        return false;
    }

    status = nvm_signaler_cpu_waiter_init (testOFA, syncModule);
    if (status != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("NVMEDIA_IOFA_MAIN: SciSync Object creation failed\n");
        status_main = -1;
        goto fail;
    }
    status = cpu_signaler_nvm_waiter_init (testOFA, syncModule);
    if (status != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("NVMEDIA_IOFA_MAIN: SciSync Object creation failed\n");
        status_main = -1;
        goto fail;
    }

    status = NvMediaIOFARegisterNvSciSyncObj(testOFA, NVMEDIA_EOFSYNCOBJ, syncObj_nvm_cpu);
    if(status != NVMEDIA_STATUS_OK)
    {
       LOG_ERR("NVMEDIA_IOFA_MAIN: Failed to register sci sync eof obj attr list.\n");
       status_main = -1;
       goto fail;
    }
    status = NvMediaIOFARegisterNvSciSyncObj(testOFA, NVMEDIA_PRESYNCOBJ, syncObj_cpu_nvm);
    if(status != NVMEDIA_STATUS_OK)
    {
       LOG_ERR("NVMEDIA_IOFA_MAIN: Failed to register sci sync eof obj attr list.\n");
       status_main = -1;
       goto fail;
    }
    status = NvMediaIOFASetNvSciSyncObjforEOF(testOFA, syncObj_nvm_cpu);
    if(status != NVMEDIA_STATUS_OK)
    {
       LOG_ERR("NVMEDIA_IOFA_MAIN: Failed to set sci sync eof obj attr list.\n");
       status_main = -1;
       goto fail;
    }

    if (args.overrideParam)
    {
        NvMediaIofaSGMParams nvMediaSGMParams;
        status = NvMediaIOFAGetSGMConfigParams(testOFA, &nvMediaSGMParams);
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("NVMEDIA_IOFA_MAIN: NvMediaIOFAGetSGMConfigParams function failed \n");
            status_main = -1;
            goto fail;
        }
        OverrideSGMParam(&args, &nvMediaSGMParams);
        status = NvMediaIOFASetSGMConfigParams(testOFA, &nvMediaSGMParams);
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("NVMEDIA_IOFA_MAIN: NvMediaIOFASetSGMConfigParams function failed \n");
            status_main = -1;
            goto fail;
        }
    }

    while (true)
    {
        while (ctx.frameRead < (ctx.frameProcessed + args.inputBuffering) && (ctx.frameRead < ctx.numFrames))
        {
            err = NvSciSyncObjGenerateFence(syncObj_cpu_nvm, &ctx.preFence[ctx.readIdx]);
            if (err != NvSciError_Success)
            {
                LOG_ERR("NVMEDIA_IOFA_MAIN: NvSciSyncObjGenerateFence function failed \n");
                status_main = -1;
                goto fail;
            }
            LOG_DBG("NVMEDIA_IOFA_MAIN: Generate pre-fence for frame count %d \n", ctx.frameRead);
            if (args.preFetchInput)
            {
                if (!skipReadImage)
                {
                    status = ReadInput(args.leftFilename, ctx.frameRead, ctx.width, ctx.height,
                                       ctx.inputFrame[ctx.readIdx][0], format, 1, LSB_ALIGNED);
                    if (status != NVMEDIA_STATUS_OK)
                    {
                        LOG_ERR("NVMEDIA_IOFA_MAIN: Left ReadImage failed\n", status);
                        break;
                    }

                    status = ReadInput(args.rightFilename, ctx.frameRead, ctx.width, ctx.height,
                                       ctx.inputFrame[ctx.readIdx][1], format, 1, LSB_ALIGNED);
                    if (status != NVMEDIA_STATUS_OK)
                    {
                        LOG_ERR("NVMEDIA_IOFA_MAIN: Right ReadImage failed\n");
                        break;
                    }
                    if ((args.frameIntervalInMS != 0U) &&
                        (ctx.frameRead == (args.inputBuffering-1U)))
                    {
                        skipReadImage = true;
                    }
                }
                LOG_DBG("NVMEDIA_IOFA_MAIN: ReadInput done with frame count %d \n", ctx.frameRead);
            }
            if (args.earlyFenceExpiry)
            {
                err = NvSciSyncObjSignal(syncObj_cpu_nvm);
                if (err != NvSciError_Success)
                {
                   LOG_ERR("NVMEDIA_IOFA_MAIN: NvSciSyncObjGenerateFence function failed \n");
                   status_main = -1;
                   goto fail;
                }
                LOG_DBG("NVMEDIA_IOFA_MAIN: Signal Sync obj for frame count %d \n", ctx.frameRead);
            }
            ctx.frameRead++;
            ctx.readIdx = (ctx.readIdx+1) % args.inputBuffering;
        }

        if ((ctx.frameRead == ctx.frameProcessed) || (ctx.frameProcessed >= ctx.numFrames))
        {
            LOG_DBG("NVMEDIA_IOFA_MAIN: Test done with frameRead count %d framesProcessed count %d\n", ctx.frameRead, ctx.frameProcessed);
            break;
        }

       if ((ctx.frameQueued < ctx.frameProcessed + args.inputBuffering) &&
           (ctx.frameRead > ctx.frameQueued))
       {
            memset(&ofaProcessParams, 0, sizeof(NvMediaIofaProcessParams));
            memset(&surfArray, 0, sizeof(NvMediaIofaBufArray));
            memset(&roiParams, 0, sizeof(NvMediaIofaROIParams));

            surfArray.inputSurface[0]  = ctx.inputFrameDup[ctx.queueIdx][0];
            surfArray.refSurface[0]    = ctx.inputFrameDup[ctx.queueIdx][1];
            surfArray.outSurface[0]    = ctx.outputSurface[ctx.queueIdx];
            surfArray.costSurface[0]   = ctx.costSurface[ctx.queueIdx];

            if (args.roiMode == 1U)
            {
                status = ClearSurface(ctx.outWidth, ctx.outHeight,ctx.outputSurface[ctx.queueIdx], A16);
                if (status != NVMEDIA_STATUS_OK)
                {
                    LOG_ERR("NVMEDIA_IOFA_MAIN: Clear surface failed\n", status);
                    break;
                } 
                if (! ParseRoiData(&roiParams, roifilename, &ctx) )
                {
                    LOG_ERR("NVMEDIA_IOFA_MAIN: Parse roi failed\n");
                    break;
                }
            }
            ofaProcessParams.rightDispMap = args.rlSearch;
            ofaProcessParams.noopMode = args.noopMode;
#ifdef ENABLE_PLAYFAIR
            if (args.frameIntervalInMS != 0U)
            {
                if (ctx.frameQueued != 0U)
                {
                    NvpRateLimitWait(&rateLimitInfo);
                }
                else
                {
                    NvpMarkPeriodicExecStart(&rateLimitInfo);
                }
            }
            if (args.playfair)
            {
                startTime = NvpGetTimeMark();
            }
#endif
            status = NvMediaIOFAInsertPreNvSciSyncFence(testOFA, &ctx.preFence[ctx.queueIdx]);
            if (status != NVMEDIA_STATUS_OK)
            {
                LOG_ERR("NVMEDIA_IOFA_MAIN: NvMediaIOFAInsertPreNvSciSyncFence function failed \n");
                status_main = -1;
                goto fail;
            }
            status = NvMediaIOFAProcessFrame(testOFA, &surfArray, &ofaProcessParams, NULL, &roiParams);
            if (status != NVMEDIA_STATUS_OK)
            {
                LOG_ERR("NVMEDIA_IOFA_MAIN: NvMediaIOFAProcessFrame failed: %x\n", status);
                status_main = -1;
                goto fail;
            }
            status = NvMediaIOFAGetEOFNvSciSyncFence(testOFA, syncObj_nvm_cpu, &(ctx.eofFence[ctx.queueIdx]));
            if (status != NVMEDIA_STATUS_OK)
            {
                LOG_ERR("NVMEDIA_IOFA_MAIN: NvMediaIOFAGetEOFNvSciSyncFence failed: %x\n", status);
                goto fail;
            }
#ifdef ENABLE_PLAYFAIR
            if (args.playfair)
            {
                stopTime = NvpGetTimeMark();
                NvpRecordSample(&submissionLatencies, startTime, stopTime);
            }
#endif
            if (!args.preFetchInput)
            {
                if (!skipReadImage)
                {
                    status = ReadInput(args.leftFilename, ctx.frameQueued, ctx.width, ctx.height,
                                       ctx.inputFrame[ctx.queueIdx][0], format, 1, LSB_ALIGNED);
                    if (status != NVMEDIA_STATUS_OK)
                    {
                        LOG_ERR("NVMEDIA_IOFA_MAIN: Left ReadImage failed\n", status);
                        break;
                    }

                    status = ReadInput(args.rightFilename, ctx.frameQueued, ctx.width, ctx.height,
                                       ctx.inputFrame[ctx.queueIdx][1], format, 1, LSB_ALIGNED);
                    if (status != NVMEDIA_STATUS_OK)
                    {
                        LOG_ERR("NVMEDIA_IOFA_MAIN: Right ReadImage failed\n");
                        break;
                    }
                    if ((args.frameIntervalInMS != 0U) &&
                        (ctx.frameQueued == (args.inputBuffering-1U)))
                    {
                        skipReadImage = true;
                    }
                }
                LOG_DBG("NVMEDIA_IOFA_MAIN: ReadInput done with frame count %d \n", ctx.frameQueued);
            }

            if (!args.earlyFenceExpiry)
            {
                err = NvSciSyncObjSignal(syncObj_cpu_nvm);
                if (err != NvSciError_Success)
                {
                   LOG_ERR("NVMEDIA_IOFA_MAIN: NvSciSyncObjGenerateFence function failed \n");
                   status_main = -1;
                   goto fail;
                }
#ifdef ENABLE_PLAYFAIR
                if (args.playfair)
                {
                    ctx.startTime[ctx.queueIdx] = NvpGetTimeMark();
                }
#endif
                LOG_DBG("NVMEDIA_IOFA_MAIN: Signal Sync obj for frame count %d \n", ctx.frameQueued);
            }
            ctx.queueIdx = (ctx.queueIdx+1)%args.inputBuffering;
            ctx.frameQueued++;
        }

        if (ctx.frameProcessed < ctx.frameQueued)
        {
            if ((ctx.frameQueued < (ctx.frameProcessed + args.inputBuffering)) &&
                (ctx.frameQueued != ctx.frameRead))
            {
                timeout = 0;
            }
            else
            {
                timeout  = args.timeout;
            }

            LOG_DBG("NVMEDIA_IOFA_MAIN: NvSciSyncFenceWait start with timeout %dms\n", timeout);

            /* Wait for operations on the image to complete */
            err = NvSciSyncFenceWait(&(ctx.eofFence[ctx.processIdx]), cpu_wait_context, timeout*1000);
            if (err == NvSciError_Success)
            {
#ifdef ENABLE_PLAYFAIR
                if (args.playfair)
                {
                    stopTime = NvpGetTimeMark();
                    NvpRecordSample(&executionLatencies, ctx.startTime[ctx.processIdx], stopTime);
                }
#endif
                taskStatus.status = NvSciSyncTaskStatusOFA_Invalid;
                err = NvSciSyncFenceGetTaskStatus(&(ctx.eofFence[ctx.processIdx]), &taskStatus);
                if (err != NvSciError_Success)
                {
                    LOG_ERR("NVMEDIA_IOFA_MAIN: Error while retrieving taskStatus from fence\n");
                    status_main = -1;
                    goto fail;
                }
                if (taskStatus.status != NvSciSyncTaskStatusOFA_Success)
                {
                    LOG_ERR("NVMEDIA_IOFA_MAIN: TaskStatus indicate failure in task execution\n");
                    if (taskStatus.status == NvSciSyncTaskStatusOFA_Invalid)
                    {
                        LOG_ERR("NVMEDIA_IOFA_MAIN: Task status set with invalid value\n");
                        LOG_ERR("NVMEDIA_IOFA_MAIN: Driver should set status with valid task status\n");
                    }
                    else if (taskStatus.status == NvSciSyncTaskStatusOFA_Error_Timeout)
                    {
                        LOG_ERR("NVMEDIA_IOFA_MAIN: Task status set with timeout error\n");
                        LOG_ERR("NVMEDIA_IOFA_MAIN: Check timeout set in nvmedia ofa driver\n");
                    }
                    else if (taskStatus.status == NvSciSyncTaskStatusOFA_Execution_Start)
                    {
                        LOG_ERR("NVMEDIA_IOFA_MAIN: Task status set with execution start (task is not finished before fence expiry\n");
                        LOG_ERR("NVMEDIA_IOFA_MAIN: Check if this is case of mlock timeout\n");
                    }
                    else
                    {
                        LOG_ERR("NVMEDIA_IOFA_MAIN: Task status set with error %d\n", taskStatus.status);
                        LOG_ERR("NVMEDIA_IOFA_MAIN: Check ofa interface file to get more details about error\n");
                    }
                    status_main = -1;
                    goto fail;
                }

                NvSciSyncFenceClear(&(ctx.preFence[ctx.processIdx]));
                NvSciSyncFenceClear(&(ctx.eofFence[ctx.processIdx]));

                if ((args.profile != 0U) && !StoreProfileData(testOFA, &ctx))
                {
                    LOG_ERR("NVMEDIA_IOFA_MAIN: NvMediaIOFAGetProfileData is failed\n");
                    status_main = -1;
                    goto fail;
                }
                if (ProcessOutputSurfaces(&args, &ctx, testOFA) != true)
                {
                    LOG_ERR("NVMEDIA_IOFA_MAIN: ProcessOutputSurfaces is failed\n");
                    status_main = -1;
                    goto fail;
                }
                ctx.processIdx = (ctx.processIdx+1)%args.inputBuffering;
                ctx.frameProcessed++;

                LOG_INFO("NVMEDIA_IOFA_MAIN: Processing finished for frame with \n");
                LOG_INFO("NVMEDIA_IOFA_MAIN: frameRead count %d fameQueued count %d  framesProcessed count %d\n", ctx.frameRead, ctx.frameQueued, ctx.frameProcessed);
            }
            else if ((timeout != 0) || (err != NvSciError_Timeout))
            {
                status_main = -1;
                LOG_ERR("NVMEDIA_IOFA_MAIN: NvSciSyncFenceWait failed\n");
                if (err == NvSciError_Timeout)
                {
                    LOG_WARN("NVMEDIA_IOFA_MAIN: Check if App set timeout of %d ms is sufficient for HW operation\n", timeout);
                    LOG_WARN("NVMEDIA_IOFA_MAIN: increase timeout if it is not sufficient to perform HW operation\n");
                }
                goto fail;
            }
        }
    }

    LOG_DBG("NVMEDIA_IOFA_MAIN: OFA Process loop is over. Now releasing resoureces.\n");
    crcFileLog(&args);

fail:
#ifdef ENABLE_PLAYFAIR
    if (perfDataConstruct)
    {
        NvpStatus_t status;

        NvpPrintStats(&initLatencies, NULL, USEC, "Init Latency", false);
        NvpPrintStats(&submissionLatencies, NULL, USEC, "Submission Latency", false);
        NvpPrintStats(&executionLatencies, NULL, USEC, "Execution Latency", false);
        if (args.profileStatsFilePath)
        {
            status = NvpDumpData(&submissionLatencies);
            if (status != NVP_PASS)
            {
                LOG_ERR("NVMEDIA_IOFA_MAIN: NvpDumpData failed with Error %d\n", status);
            }
            status = NvpDumpData(&executionLatencies);
            if (status != NVP_PASS)
            {
                LOG_ERR("NVMEDIA_IOFA_MAIN: NvpDumpData failed with Error %d\n", status);
            }
        }
        NvpDestroyPerfData(&initLatencies);
        NvpDestroyPerfData(&submissionLatencies);
        NvpDestroyPerfData(&executionLatencies);
    }
#endif
    if (args.profile != 0U)
    {
        ProcessProfileData(&ctx);
        if (ctx.profDataArray != NULL)
        {
            free(ctx.profDataArray);
        }
    }

    if (syncObj_nvm_cpu)
    {
        status  = NvMediaIOFAUnregisterNvSciSyncObj(testOFA, syncObj_nvm_cpu);
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("NVMEDIA_IOFA_MAIN: NvMediaIOFAUnregisterNvSciSyncObj failed: %x\n", status);
        }
    }
    if (syncObj_cpu_nvm)
    {
        status  = NvMediaIOFAUnregisterNvSciSyncObj(testOFA, syncObj_cpu_nvm);
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("NVMEDIA_IOFA_MAIN: NvMediaIOFAUnregisterNvSciSyncObj failed: %x\n", status);
        }
    }

    UnRegisterSurfaces(testOFA, &ctx, args.inputBuffering);
    if (testOFA != NULL)
    {
        LOG_DBG("NVMEDIA_IOFA_MAIN: Destroying IOFA Device\n");
        NvMediaIOFADestroy(testOFA);
    }

    LOG_DBG("NVMEDIA_IOFA_MAIN: Destroying input, output and cost surface \n");
    if (syncObj_nvm_cpu)
    {
        NvSciSyncObjFree(syncObj_nvm_cpu);
    }
    if (syncObj_cpu_nvm)
    {
        NvSciSyncObjFree(syncObj_cpu_nvm);
    }
    for (i = 0; i < NVMEDIA_IOFA_BUFFERING; i++)
    {
        for (j = 0; j < 2; j++)
        {
            if (ctx.inputFrameDup[i][j] != NULL)
            {
                NvSciBufObjFree(ctx.inputFrameDup[i][j]);
            }
            if (ctx.inputFrame[i][j] != NULL)
            {
                NvSciBufObjFree(ctx.inputFrame[i][j]);
            }
        }
        if (ctx.outputSurface[i] != NULL)
        {
            NvSciBufObjFree(ctx.outputSurface[i]);
        }
        if (ctx.costSurface[i] != NULL)
        {
            NvSciBufObjFree(ctx.costSurface[i]);
        }
    }

    if (ctx.stereoCrcFile != NULL)
    {
        fclose(ctx.stereoCrcFile);
    }
    if (ctx.costCrcFile != NULL)
    {
        fclose(ctx.costCrcFile);
    }
    if (roifilename != NULL)
    {
        fclose(roifilename);
    }
    if (cpu_wait_context != NULL)
    {
        NvSciSyncCpuWaitContextFree(cpu_wait_context);
    }
    if (syncModule != NULL)
    {
        NvSciSyncModuleClose(syncModule);
    }

    LOG_MSG("NVMEDIA_IOFA_MAIN: total processed frames: %d \n", ctx.frameProcessed);
    LOG_MSG("NVMEDIA_IOFA_MAIN: total failures: %d \n", ((status_main == -1) || (ctx.frameProcessed == 0))? 1 : 0);

    return 0;
}
