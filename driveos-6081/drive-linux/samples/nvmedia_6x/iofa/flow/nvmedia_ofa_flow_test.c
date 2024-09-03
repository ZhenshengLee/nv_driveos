/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "commandline.h"
#include "scibuf_utils.h"
#if __QNXNTO__
#include "nvdvms_client.h"
#endif
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
    uint32_t fileSize, frameSize, numFrames;

    fileSize = GetFileSize(args->inputFilename);
    frameSize = GetImageSize(args);
    if (frameSize == 0)
    {
        return 0;
    }
    numFrames = fileSize/frameSize;
    if (numFrames < 2)
    {
        return 0;
    }

    return numFrames - 1U;
}

static bool
GetFileType(
    uint32_t        bitDepth,
    ChromaFormatOFA chromaFormat,
    ChromaFormat    *chromaTarget)
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
GetFormatType(
    uint32_t        bitDepth,
    ChromaFormatOFA chromaFormat,
    ChromaFormat    *chromaTarget
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
AllocateSurfaces (
    TestArgs      *args,
    NvMediaIofa   *testOFA,
    FlowTestCtx   *ctx
)
{
    bool ret = true;
    NvSciError err;
    NvMediaStatus status;
    uint32_t i, j, k, planeCount;
    ChromaFormat InputFormat;

    NvSciBufModule bufModule;
    NvSciBufAttrList attributeList[NVMEDIA_IOFA_MAX_PYD_LEVEL] = {NULL}; //This will be fixed later
    NvSciBufAttrList outattributeList[NVMEDIA_IOFA_MAX_PYD_LEVEL] = {NULL}; //This will be fixed later
    NvSciBufAttrList costattributeList[NVMEDIA_IOFA_MAX_PYD_LEVEL] = {NULL}; //This will be fixed later
    NvSciBufAttrList conflictList[NVMEDIA_IOFA_MAX_PYD_LEVEL] = {NULL}; //This is to check if there is any error while conciling
    NvSciBufAttrList outconflictList[NVMEDIA_IOFA_MAX_PYD_LEVEL] = {NULL}; //This is to check if there is any error while conciling
    NvSciBufAttrList costconflictList[NVMEDIA_IOFA_MAX_PYD_LEVEL] = {NULL}; //This is to check if there is any error while conciling
    NvSciBufAttrList reconciledAttributeList[NVMEDIA_IOFA_MAX_PYD_LEVEL] = {NULL}; //This is shared in public data structure
    NvSciBufAttrList outreconciledAttributeList[NVMEDIA_IOFA_MAX_PYD_LEVEL] = {NULL}; //This is shared in public data structure
    NvSciBufAttrList costreconciledAttributeList[NVMEDIA_IOFA_MAX_PYD_LEVEL] = {NULL}; //This is shared in public data structure

    //Opening NvSciBuf Module
    err = NvSciBufModuleOpen(&bufModule);
    if (err != NvSciError_Success)
    {
        LOG_ERR("AllocateSurfaces: NvSciSyncModuleOpen failed\n");
        return false;
    }

    for (i = 0; i < args->inputBuffering; i++)
    {
        for (j = 0; j < 2; j++)
        {
            for (k = 0; k < ctx->pydLevel; k++)
            {
                ChromaFormatOFA chromaFormat;
                chromaFormat = (k == 0) ? args->chromaFormat : args->pydChromaFormat;
                if (chromaFormat == YUV_400)
                {
                    planeCount = 1;
                }
                else
                {
                    planeCount = 2;
                }
                if (!GetFormatType(args->bitDepth, args->chromaFormat, &InputFormat))
                {
                    LOG_ERR("AllocateSurfaces: Incorrect Input format Type\n");
                    return false;
                }
                err = NvSciBufAttrListCreate(bufModule, &attributeList[k]);
                if (err != NvSciError_Success)
                {
                    LOG_ERR("AllocateSurfaces: NvSciBufAttrListCreate failed\n");
                    ret = false;
                    break;
                }
                status = NvMediaIOFAFillNvSciBufAttrList(attributeList[k]);
                if (NVMEDIA_STATUS_OK != status)
                {
                    LOG_ERR("AllocateSurfaces: Failed to fill IOFA internal attributes\n");
                    ret = false;
                    break;
                }
                status = PopulateNvSciBufAttrList(InputFormat, ctx->width[k], ctx->height[k], true, NvSciBufImage_BlockLinearType, planeCount, NvSciBufAccessPerm_Readonly, 256U, NvSciColorStd_REC601_ER, NvSciBufScan_ProgressiveType, attributeList[k]);
                if (NVMEDIA_STATUS_OK != status)
                {
                    LOG_ERR("AllocateSurfaces: Failed to fill IOFA external attributes\n");
                    ret = false;
                    break;
                }
                err =  NvSciBufAttrListReconcile(&attributeList[k], 1U, &reconciledAttributeList[k], &conflictList[k]);
                if (err != NvSciError_Success)
                {
                    LOG_ERR("AllocateSurfaces: Reconciliation for input frame failed\n");
                    ret = false;
                    break;
                }
                err =  NvSciBufObjAlloc(reconciledAttributeList[k], &ctx->inputFrame[i][j][k]);
                if (err != NvSciError_Success)
                {
                    LOG_ERR("AllocateSurfaces: NvSciBuf Obj creation for input frame failed\n");
                    ret = false;
                    break;
                }
                err = NvSciBufObjDupWithReducePerm(ctx->inputFrame[i][j][k],
                                                   NvSciBufAccessPerm_Readonly,
                                                   &ctx->inputFrameDup[i][j][k]);
                if (err != NvSciError_Success)
                {
                    LOG_ERR("AllocateSurfaces: NvSciBufObjDupWithReducePerm for input surface failed\n");
                    return false;
                }

                NvSciBufAttrListFree(attributeList[k]);
                NvSciBufAttrListFree(reconciledAttributeList[k]);
                NvSciBufAttrListFree(conflictList[k]);
            }
        }
    }

    for (i = 0; i < args->inputBuffering; i++)
    {
        for (k = 0; k < ctx->pydLevel; k++)
        {
            //For Output Surface
            LOG_DBG("NVMEDIA_IOFA_MAIN: Creating output surface with resolution %d x %d\n", ctx->outWidth[k], ctx->outHeight[k]);
            err = NvSciBufAttrListCreate(bufModule, &outattributeList[k]);
            if (err != NvSciError_Success)
            {
                LOG_ERR("AllocateSurfaces: NvSciBufAttrListCreate failed out buffer\n");
                ret = false;
                break;
            }
            status = NvMediaIOFAFillNvSciBufAttrList(outattributeList[k]);
            if (NVMEDIA_STATUS_OK != status)
            {
                LOG_ERR("AllocateSurfaces: Failed to fill IOFA internal attributes\n");
                ret = false;
                break;
            }
            status = PopulateNvSciBufAttrList(RG16, ctx->outWidth[k], ctx->outHeight[k], true, NvSciBufImage_BlockLinearType,  1U, NvSciBufAccessPerm_ReadWrite, 256U, NvSciColorStd_REC601_ER, NvSciBufScan_ProgressiveType ,outattributeList[k]);
            if (NVMEDIA_STATUS_OK != status)
            {
                LOG_ERR("AllocateSurfaces: Failed to fill IOFA external attributes\n");
                ret = false;
                break;
            }
            err =  NvSciBufAttrListReconcile(&outattributeList[k], 1U, &outreconciledAttributeList[k], &outconflictList[k]);
            if (err != NvSciError_Success)
            {
                LOG_ERR("AllocateSurfaces: Reconciliation for out frame failed\n");
                ret = false;
                break;
            }
            err =  NvSciBufObjAlloc(outreconciledAttributeList[k], &ctx->outputSurface[i][k]);
            if (err != NvSciError_Success)
            {
                LOG_ERR("AllocateSurfaces: NvSciBuf Obj creation for out frame failed\n");
                ret = false;
                break;
            }
            NvSciBufAttrListFree(outattributeList[k]);
            NvSciBufAttrListFree(outreconciledAttributeList[k]);
            NvSciBufAttrListFree(outconflictList[k]);
            //For Cost Surface
            err = NvSciBufAttrListCreate(bufModule, &costattributeList[k]);
            if (err != NvSciError_Success)
            {
                LOG_ERR("AllocateSurfaces: NvSciBufAttrListCreate failed for cost buffer\n");
                ret = false;
                break;
            }
            status = NvMediaIOFAFillNvSciBufAttrList(costattributeList[k]);
            if (NVMEDIA_STATUS_OK != status)
            {
                LOG_ERR("AllocateSurfaces: Failed to fill IOFA internal attributes\n");
                break;
            }
            status = PopulateNvSciBufAttrList(A8, ctx->outWidth[k], ctx->outHeight[k], true, NvSciBufImage_BlockLinearType,  1U, NvSciBufAccessPerm_ReadWrite, 256U, NvSciColorStd_REC601_ER, NvSciBufScan_ProgressiveType, costattributeList[k]);
            if (NVMEDIA_STATUS_OK != status)
            {
                LOG_ERR("AllocateSurfaces: Failed to fill IOFA external attributes\n");
                ret = false;
                break;
            }
            err =  NvSciBufAttrListReconcile(&costattributeList[k], 1U, &costreconciledAttributeList[k], &costconflictList[k]);
            if (err != NvSciError_Success)
            {
                LOG_ERR("AllocateSurfaces: Reconciliation for input frame failed\n");
                ret = false;
                break;
            }
            err =  NvSciBufObjAlloc(costreconciledAttributeList[k], &ctx->costSurface[i][k]);
            if (err != NvSciError_Success)
            {
                LOG_ERR("AllocateSurfaces: NvSciBuf Obj creation for input frame failed\n");
                ret = false;
                break;
            }
            NvSciBufAttrListFree(costattributeList[k]);
            NvSciBufAttrListFree(costreconciledAttributeList[k]);
            NvSciBufAttrListFree(costconflictList[k]);
        }
    }
    if (NULL != bufModule)
    {
        NvSciBufModuleClose(bufModule);
    }
    return ret;
}

static bool
ValidateAndOpenROIFile (
    TestArgs *args,
    FILE     **flowROIFile
)
{
    if (args->roiMode == 1)
    {
        *flowROIFile = fopen(args->roiFilename, "r");
        if (*flowROIFile == NULL)
        {
            LOG_ERR("NVMEDIA_IOFA_MAIN: Cannot open flow ROI file for reading\n");
            return false;
        }
    }
    return true;
}

static bool
ValidateAndOpenCRCFiles (
    TestArgs *args,
    FILE     **flowCrcFile,
    FILE     **costCrcFile
)
{
    // CRC mode check
    if (args->flowCrcoption.crcGenMode && args->flowCrcoption.crcCheckMode)
    {
        LOG_ERR("ValidateAndOpenCRCFiles: FlowCrcGenMode and FlowCrcCheckMode cannot be enabled at the same time\n");
        return false;
    }
    if (args->costCrcoption.crcGenMode && args->costCrcoption.crcCheckMode)
    {
        LOG_ERR("ValidateAndOpenCRCFiles: costCrcGenMode and costCrcCheckMode cannot be enabled at the same time\n");
        return false;
    }

    // Open FLOW CRC file
    if (args->flowCrcoption.crcGenMode)
    {
        *flowCrcFile = fopen(args->flowCrcoption.crcFilename, "wt");
        if (*flowCrcFile == NULL)
        {
            LOG_ERR("ValidateAndOpenCRCFiles: Cannot open flow Crc gen file for writing\n");
            return false;
        }
    }
    else if (args->flowCrcoption.crcCheckMode)
    {
        *flowCrcFile = fopen(args->flowCrcoption.crcFilename, "rb");
        if (*flowCrcFile == NULL)
        {
            LOG_ERR("ValidateAndOpenCRCFiles: Cannot open flow Crc gen file for reading\n");
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
        LOG_ERR("NVMEDIA_IOFA_MAIN: NvMediaIOFAGetVersion failed\n");
        return false;
    }
    if ((version.major != NVMEDIA_IOFA_VERSION_MAJOR) || (version.minor != NVMEDIA_IOFA_VERSION_MINOR))
    {
        LOG_ERR("NVMEDIA_IOFA_MAIN: NvMediaIOFAGetVersion version mismatch\n");
        return false;
    }
    LOG_INFO("NVMEDIA_IOFA_MAIN: NvMediaIOFA Major version - %d \n", NVMEDIA_IOFA_VERSION_MAJOR);
    LOG_INFO("NVMEDIA_IOFA_MAIN: NvMediaIOFA Minor version - %d \n", NVMEDIA_IOFA_VERSION_MINOR);

    return true;
}

static void
UnRegisterSurfaces (
    NvMediaIofa       *testOFA,
    const FlowTestCtx *ctx,
    uint8_t           inputBuffering
)
{
    uint8_t i, j, k;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    LOG_DBG("NVMEDIA_IOFA_MAIN: UnRegister Input surfaces, output and cost surfaces \n");

    for (i = 0; i < inputBuffering; i++)
    {
        for (j = 0; j < ctx->pydLevel; j++)
        {
            for (k = 0; k < 2; k++)
            {
                if (ctx->inputFrameDup[i][k][j] != NULL)
                {
                    status = NvMediaIOFAUnregisterNvSciBufObj(testOFA, ctx->inputFrameDup[i][k][j]);
                    if (status != NVMEDIA_STATUS_OK)
                    {
                        LOG_ERR("NVMEDIA_IOFA_MAIN: NvMediaIOFAUnregisterNvSciBufObj input frame failed\n");
                    }
                }
            }
            if (ctx->outputSurface[i][j] != NULL)
            {
                status = NvMediaIOFAUnregisterNvSciBufObj(testOFA, ctx->outputSurface[i][j]);
                if (status != NVMEDIA_STATUS_OK)
                {
                    LOG_ERR("NVMEDIA_IOFA_MAIN: NvMediaIOFAUnregisterNvSciBufObj output frame failed\n");
                }
            }
            if (ctx->costSurface[i][j] != NULL)
            {
                status = NvMediaIOFAUnregisterNvSciBufObj(testOFA, ctx->costSurface[i][j]);
                if (status != NVMEDIA_STATUS_OK)
                {
                    LOG_ERR("NVMEDIA_IOFA_MAIN: NvMediaIOFAUnregisterNvSciBufObj cost frame failed\n");
                }
            }
        }
    }
}

static bool
RegisterSurfaces (
    NvMediaIofa       *testOFA,
    const FlowTestCtx *ctx,
    uint8_t           inputBuffering
)
{
    uint8_t i, j, k;

    // Pinning the surfaces with NvMediaIOFARegisterNvSciBufObj API
    LOG_DBG("NVMEDIA_IOFA_MAIN: Register input frames, output and cost surfaces \n");
    for (i = 0;  i< inputBuffering; i++)
    {
        for (j = 0; j < ctx->pydLevel; j++)
        {
            for (k = 0; k < 2; k++)
            {
                if (NvMediaIOFARegisterNvSciBufObj((const NvMediaIofa *)testOFA,ctx->inputFrameDup[i][k][j]) != NVMEDIA_STATUS_OK)
                {
                    LOG_ERR("NVMEDIA_IOFA_MAIN: Register input frame failed\n");
                    return false;
                }
            }

            if (NvMediaIOFARegisterNvSciBufObj((const NvMediaIofa *)testOFA,
                                      ctx->outputSurface[i][j]) != NVMEDIA_STATUS_OK)
            {
                LOG_ERR("NVMEDIA_IOFA_MAIN: Register Output surface failed\n");
                return false;
            }

            if (NvMediaIOFARegisterNvSciBufObj((const NvMediaIofa *)testOFA,
                                     ctx->costSurface[i][j]) != NVMEDIA_STATUS_OK)
            {
                LOG_ERR("NVMEDIA_IOFA_MAIN: Register cost surface failed\n");
                return false;
            }
        }
    }
    return true;
}

static bool
ProcessOutputSurfaces (
    const TestArgs      *args,
    const FlowTestCtx   *ctx,
    const NvMediaIofa   *testOFA
)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint32_t calcCrcCost, calcCrcFlow, refCrc;

    // Write output surface
    if (args->outputFilename != NULL)
    {
        status = WriteOutput(args->outputFilename, ctx->outputSurface[ctx->processIdx][0], false,
                                (ctx->frameProcessed) ? true : false, NULL);
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("ProcessOutputSurfaces: WriteOutput for output failed\n");
            return false;
        }
        LOG_DBG("ProcessOutputSurfaces: Writing output flow completed\n");
    }
    // Write cost surface
    if (args->costFilename != NULL)
    {
        status = WriteOutput(args->costFilename, ctx->costSurface[ctx->processIdx][0], false,
                               (ctx->frameProcessed) ? true : false, NULL);
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("ProcessOutputSurfaces: WriteOutput for cost failed\n");
            return false;
        }
        LOG_DBG("ProcessOutputSurfaces: Writing cost completed\n");
    }
    if (args->flowCrcoption.crcGenMode)
    {
        calcCrcFlow = 0;
        status = GetNvSciBufObjCrcNoSrcRect(ctx->outputSurface[ctx->processIdx][0], false, &calcCrcFlow);
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("ProcessOutputSurfaces: GetImageCrc failed: %x\n", status);
            return false;
        }
        if (ctx->FlowCrcFile != NULL)
        {
            if (!fprintf(ctx->FlowCrcFile, "%08x\n", calcCrcFlow))
            {
                LOG_ERR("ProcessOutputSurfaces: Failed writing calculated FLOW CRC to file %s\n", ctx->FlowCrcFile);
                return false;
            }
        }
        LOG_DBG("ProcessOutputSurfaces: Writing crc completed = %08x\n",calcCrcFlow);
    }
    else if (args->flowCrcoption.crcCheckMode)
    {
        calcCrcFlow = 0;
        status = GetNvSciBufObjCrcNoSrcRect(ctx->outputSurface[ctx->processIdx][0], false, &calcCrcFlow);
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("ProcessOutputSurfaces: GetImageCrc failed: %x\n", status);
            return false;
        }
        if (ctx->FlowCrcFile != NULL)
        {
            if (fscanf(ctx->FlowCrcFile, "%8x\n", &refCrc) == 1)
            {
                if (refCrc != calcCrcFlow)
                {
                    LOG_ERR("ProcessOutputSurfaces: Frame %d Flow CRC 0x%x does not match with ref crc 0x%x\n", ctx->frameProcessed, calcCrcFlow, refCrc);
                    return false;
                }
            }
            else
            {
                LOG_ERR("ProcessOutputSurfaces: Failed checking FLOW CRC. Failed reading file %s\n", ctx->FlowCrcFile);
                return false;
            }
        }
        else
        {
            LOG_ERR("ProcessOutputSurfaces: FLOW CRC file pointer is NULL\n");
            return false;
        }
    }

    if (args->costCrcoption.crcGenMode)
    {
        calcCrcCost = 0;
        status = GetNvSciBufObjCrcNoSrcRect(ctx->costSurface[ctx->processIdx][0], false , &calcCrcCost);
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
        status = GetNvSciBufObjCrcNoSrcRect(ctx->costSurface[ctx->processIdx][0], false , &calcCrcCost);
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

static void
OverrideSGMParam (
    const TestArgs       *args,
    NvMediaIofaSGMParams *pSGMParams,
    const FlowTestCtx    *ctx
)
{
    uint8_t i;
    if (args->overrideParam & SGMPARAM_P1_OVERRIDE)
    {
        for (i = 0; i < ctx->pydLevel; i++)
        {
            pSGMParams->penalty1[i] = args->p1[i];
        }
    }
    if (args->overrideParam & SGMPARAM_P2_OVERRIDE)
    {
        for (i = 0; i < ctx->pydLevel; i++)
        {
            pSGMParams->penalty2[i] = args->p2[i];
        }
    }
    if (args->overrideParam & SGMPARAM_DIAGONAL_OVERRIDE)
    {
        for (i = 0; i < ctx->pydLevel; i++)
        {
            pSGMParams->enableDiag[i] = (args->diagonalMode[i] != 0) ? true:false;
        }
    }
    if (args->overrideParam & SGMPARAM_ADAPTIVEP2_OVERRIDE)
    {
        for (i = 0; i < ctx->pydLevel; i++)
        {
            pSGMParams->adaptiveP2[i] = (args->adaptiveP2[i] != 0) ? true:false;
        }
    }
    if (args->overrideParam & SGMPARAM_NUMPASSES_OVERRIDE)
    {
        for (i = 0; i < ctx->pydLevel; i++)
        {
            pSGMParams->numPasses[i] = args->numPasses[i];
        }
    }
    if (args->overrideParam & SGMPARAM_ALPHA_OVERRIDE)
    {
        for (i = 0; i < ctx->pydLevel; i++)
        {
            pSGMParams->alphaLog2[i] = args->alpha[i];
        }
    }

    return;
}


static void
crcFileLog (
    TestArgs *args
)
{
    if (args->flowCrcoption.crcGenMode)
    {
        LOG_MSG("\n crcFileLog: ***FLOW crc gold file %s has been generated***\n", args->flowCrcoption.crcFilename);
    }
    else if (args->flowCrcoption.crcCheckMode)
    {
        LOG_MSG("\n crcFileLog: ***FLOW crc checking with file %s is successful\n", args->flowCrcoption.crcFilename);
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

static bool
DecidePyramidLeveAndResolution (
    const TestArgs    *args,
    const NvMediaIofa *testOFA,
    FlowTestCtx       *ctx
)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvMediaIofaCapability Capability;
    uint8_t pydLevel = 0;
    uint16_t width = args->width;
    uint16_t height = args->height;
    uint8_t i;

    status = NvMediaIOFAGetCapability(testOFA, NVMEDIA_IOFA_MODE_PYDOF, &Capability);
    if (status != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("DecidePyramidLeveAndResolution: NvMediaIOFAGetCapability function failed \n");
        return false;
    }

    if (args->chromaFormat == YUV_420)
    {
        if (((width % 2) != 0)  || ((height % 2) != 0))
        {
            LOG_ERR("DecidePyramidLeveAndResolution: input width and height must be even for YUV420 input \n");
            return false;
        }
    }
    else if (args->chromaFormat == YUV_422)
    {
        if (((width % 2) != 0))
        {
            LOG_ERR("DecidePyramidLeveAndResolution: input width must be even for YUV422 input \n");
            return false;
        }
    }

    while (pydLevel < NVMEDIA_IOFA_MAX_PYD_LEVEL)
    {
        if ((width < Capability.minWidth) || (height < Capability.minHeight))
        {
            break;
        }
        ctx->width[pydLevel] = width;
        ctx->height[pydLevel] = height;
        width = (width + 1U) >> 1U;
        height = (height + 1U) >> 1U;
        if ((args->pydChromaFormat == YUV_420) || (args->pydChromaFormat == YUV_422))
        {
            width = (width + 1) & 0xFFFE;
        }
        if (args->pydChromaFormat == YUV_420)
        {
            height = (height + 1) & 0xFFFE;
        }
        pydLevel++;
    }
    if (args->pydLevel <= pydLevel)
    {
        ctx->pydLevel = args->pydLevel;
    }
    else
    {
        ctx->pydLevel = pydLevel;
    }
    if (ctx->pydLevel == 0U)
    {
        LOG_ERR("DecidePyramidLeveAndResolution: Pyramid level 0 is not supported \n");
        return false;
    }
    for (i = 0; i < ctx->pydLevel; i++)
    {
        ctx->outWidth[i] = (ctx->width[i] + (1U << args->gridsize[i]) - 1) >> args->gridsize[i];
        ctx->outHeight[i] = (ctx->height[i] + (1U << args->gridsize[i]) -1) >> args->gridsize[i];
    }

    return true;
}

static bool
ParseRoiData (
    NvMediaIofaROIParams *roiparamsdata,
    FILE                 *roifilename,
    const FlowTestCtx    *ctx
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
    roiparamsdata->numOfROIs = roidata[1] ;
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

static void
GetFileName (
    char     *epipolarFilename,
    uint32_t frameIdx,
    char     *epiFile
)
{
    char buf[20];
    sprintf(buf, "%06d", frameIdx);
    strcat(epiFile, buf);
    strcat(epiFile, "_epi.txt");
}


static bool
GetEpipolarInfo (
    NvMediaIofaEpipolarInfo *epipolarInfo,
    char                    *epipolarFilename
)
{
    uint8_t i, j;
    float epi_x, epi_y;
    FILE *epiInfoFile = NULL;
    if (strcmp(epipolarFilename, ""))
    {
        epiInfoFile = fopen(epipolarFilename, "rb");
        LOG_MSG("GetEpipolarInfo: reading epipolar info file %s\n", epipolarFilename);

        if (epiInfoFile != NULL)
        {
            for (i = 0; i < 3; i++)
            {
                for (j = 0; j < 3; j++) // Reading row-wise
                {
                    if (fscanf(epiInfoFile, "%f ", &(epipolarInfo->F_Matrix[i][j]))!=1) 
                    {
                        return false;
                    }
                }
            }
            for (i = 0; i < 3; i++)
            {
                for (j = 0; j < 3; j++) // Reading row-wise
                {
                    if (fscanf(epiInfoFile, "%f ", &(epipolarInfo->H_Matrix[i][j]))!=1)
                    {
                        return false;
                    }
                }
            }
            if(fscanf(epiInfoFile, "%f ", &epi_x)!=1)
            {
                return false;
            }
            if(fscanf(epiInfoFile, "%f ", &epi_y)!=1)
            {
                return false;
            }
            if(fscanf(epiInfoFile, "%hhd ", &epipolarInfo->direction)!=1)
            {
                return false;
            }
            epipolarInfo->epipole_x = (uint32_t)(epi_x * 8);
            epipolarInfo->epipole_y = (uint32_t)(epi_y * 8);
            fclose(epiInfoFile);
        }
        else
        {
            LOG_MSG("GetEpipolarInfo: Unable to open epipolar info file %s\n", epipolarFilename);
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    TestArgs args;
    FlowTestCtx ctx;
    NvMediaIofa *testOFA = NULL;
    NvMediaIofaInitParams ofaInitParams;
    NvMediaIofaProcessParams ofaProcessParams;
    NvMediaIofaBufArray surfArray;
    NvMediaIofaROIParams roiParams;
    uint32_t imageSize = 0;
    NvSciError err;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    int32_t status_main = 0, i;
    uint32_t j, k, timeout, numFrames;
    ChromaFormat inputFileChromaFormat;
    bool skipReadImage = false;
    NvSciSyncTaskStatus taskStatus;
    NvSciSyncModule syncModule = NULL;
#ifdef ENABLE_PLAYFAIR
    uint64_t startTime = 0, stopTime = 0, numOfLatenciesToBeAggregated = 0;
    NvpPerfData_t submissionLatencies, executionLatencies, initLatencies, initLatenciesPartData[2U];
    NvpPerfData_t *initLatenciesArray[2U];
    NvpRateLimitInfo_t rateLimitInfo;
    bool perfDataConstruct = false;
#endif
    NvMediaIofaEpipolarInfo epipolarInfo;
    memset(&args, 0, sizeof(TestArgs));
    memset(&ctx, 0, sizeof(FlowTestCtx));
    memset(&epipolarInfo, 0, sizeof(NvMediaIofaEpipolarInfo));
    FILE *roifilename = NULL;
    char epiFileName[100];

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

    if (!ValidateAndOpenCRCFiles(&args, &ctx.FlowCrcFile, &ctx.costCrcFile))
    {
        status_main = -1;
        goto fail;
    }
    if (!ValidateAndOpenROIFile(&args, &roifilename))
    {
        status_main = -1;
        goto fail;
    }
    if (!GetFileType(args.bitDepth, args.chromaFormat, &inputFileChromaFormat))
    {
        LOG_ERR("NVMEDIA_IOFA_MAIN: Incorrect Input format Type\n");
        status_main = -1;
        goto fail;
    }
    switch (args.chromaFormat)
    {
        case YUV_400:
            imageSize = (args.width * args.height);
            break;
        case YUV_420:
            imageSize = (args.width * args.height * 3 / 2);
            break;
        case YUV_422:
            imageSize = (args.width * args.height * 2);
            break;
        case YUV_444:
            imageSize = (args.width * args.height * 3);
            break;
        default:
            imageSize = (args.width * args.height);
            break;
    }
    if (args.bitDepth > 8)
    {
        imageSize *= 2;
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

#ifdef ENABLE_PLAYFAIR
    if (args.frameIntervalInMS != 0U)
    {
        NvpRateLimitInit(&rateLimitInfo, 1000U/args.frameIntervalInMS);
    }
    if (args.playfair)
    {
        char initLatencyFileName[FILE_NAME_SIZE] = {0};
        char executionLatencyFileName[FILE_NAME_SIZE] = {0};
        char submissionLatencyFileName[FILE_NAME_SIZE] = {0};

        if (args.profileStatsFilePath)
        {
            strcpy(initLatencyFileName, args.profileStatsFilePath);
            strcpy(executionLatencyFileName, args.profileStatsFilePath);
            strcpy(submissionLatencyFileName, args.profileStatsFilePath);
        }
        for (k = 0; k < 2U; k++)
        {
            NvpConstructPerfData(&initLatenciesPartData[k], 1, NULL);
        }
        strcat(initLatencyFileName, "/InitLatency.csv");
        NvpConstructPerfData(&initLatencies, 1, initLatencyFileName);
        strcat(executionLatencyFileName, "/ExecutionLatency.csv");
        NvpConstructPerfData(&executionLatencies, ctx.numFrames, executionLatencyFileName);
        strcat(submissionLatencyFileName,"/SubmissionLatency.csv");
        NvpConstructPerfData(&submissionLatencies, ctx.numFrames, submissionLatencyFileName);
        perfDataConstruct = true;
        startTime = NvpGetTimeMark();
    }
#endif
    testOFA = NvMediaIOFACreate();
    if (testOFA == NULL)
    {
        LOG_ERR("NVMEDIA_IOFA_MAIN: NvMediaIofaCreate function failed \n");
        status_main = -1;
        goto fail;
    }
#ifdef ENABLE_PLAYFAIR
    if (args.playfair)
    {
        stopTime = NvpGetTimeMark();
        NvpRecordSample(&initLatenciesPartData[numOfLatenciesToBeAggregated], startTime, stopTime);
        initLatenciesArray[numOfLatenciesToBeAggregated] = &initLatenciesPartData[numOfLatenciesToBeAggregated];
        numOfLatenciesToBeAggregated++;
    }
#endif

    LOG_INFO("NVMEDIA_IOFA_MAIN: Create OFA successful\n");

    if (!DecidePyramidLeveAndResolution(&args, testOFA, &ctx))
    {
        LOG_ERR("NVMEDIA_IOFA_MAIN: DecidePyramidLeveAndResolution function failed \n");
        status_main = -1;
        goto fail;
    }
    memset(&ofaInitParams, 0U, sizeof(ofaInitParams));
    ofaInitParams.ofaMode = args.flowmode ? NVMEDIA_IOFA_MODE_EPIOF : NVMEDIA_IOFA_MODE_PYDOF;
    ofaInitParams.preset    = args.preset;
    if (args.ndisp == 256U)
    {
        ofaInitParams.dispRange = NVMEDIA_IOFA_DISPARITY_RANGE_256;
    }
    else
    {
        ofaInitParams.dispRange = NVMEDIA_IOFA_DISPARITY_RANGE_128;
    }

    if (args.epiSearchRange == 256U)
    {
        ofaInitParams.epiSearchRange = NVMEDIA_IOFA_EPI_SEARCH_RANGE_256;
    }
    else
    {
        ofaInitParams.epiSearchRange = NVMEDIA_IOFA_EPI_SEARCH_RANGE_128;
    }

    ctx.pydLevel = 1U;
    if (ofaInitParams.ofaMode == NVMEDIA_IOFA_MODE_PYDOF)
    {
        ofaInitParams.pydMode = args.pydSGMMode;
        if (!DecidePyramidLeveAndResolution(&args, testOFA, &ctx))
        {
            LOG_ERR("NVMEDIA_IOFA_MAIN: DecidePyramidLeveAndResolution function failed \n");
            status_main = -1;
            goto fail;
        }
    }
    ofaInitParams.ofaPydLevel = ctx.pydLevel;
    for (k = 0; k < ctx.pydLevel; k++)
    {
       ofaInitParams.width[k] = ctx.width[k];
       ofaInitParams.height[k] = ctx.height[k];
       ofaInitParams.outWidth[k] = ctx.outWidth[k];
       ofaInitParams.outHeight[k] = ctx.outHeight[k];
       ofaInitParams.gridSize[k] = args.gridsize[k];
    }
    LOG_INFO("NVMEDIA_IOFA_MAIN: NvMediaIOFAInit Called with gridSize: %d \n", ((uint32_t)ofaInitParams.gridSize[0]));
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
    LOG_DBG("NVMEDIA_IOFA_MAIN: Create input, output and cost surfaces\n");
    if (!AllocateSurfaces(&args, testOFA, &ctx))
    {
        LOG_ERR("NVMEDIA_IOFA_MAIN: AllocateSurfaces failed\n");
        status_main = -1;
        goto fail;
    }
    if (!RegisterSurfaces(testOFA, &ctx, args.inputBuffering))
    {
        LOG_ERR("NVMEDIA_IOFA_MAIN: RegisterSurfaces failed\n");
        status_main = -1;
        goto fail;
    }
#ifdef ENABLE_PLAYFAIR
    if (args.playfair)
    {
        stopTime = NvpGetTimeMark();
        NvpRecordSample(&initLatenciesPartData[numOfLatenciesToBeAggregated], startTime, stopTime);
        initLatenciesArray[numOfLatenciesToBeAggregated] = &initLatenciesPartData[numOfLatenciesToBeAggregated];
        numOfLatenciesToBeAggregated++;
    }
#endif
    err = NvSciSyncModuleOpen(&syncModule);
    if (err != NvSciError_Success)
    {
        LOG_ERR("NVMEDIA_IOFA_MAIN: NvSciBufModuleOpen failed\n");
        status_main = -1;
        goto fail;
    }

    status = nvm_signaler_cpu_waiter_init (testOFA, syncModule);
    if (status != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("NVMEDIA_IOFA_MAIN: SciSync Object creation failed \n");
        status_main = -1;
        goto fail;
    }
    status = cpu_signaler_nvm_waiter_init (testOFA, syncModule);
    if (status != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("NVMEDIA_IOFA_MAIN: SciSync Object creation failed \n");
        status_main = -1;
        goto fail;
    }

    status = NvMediaIOFARegisterNvSciSyncObj(testOFA, NVMEDIA_EOFSYNCOBJ, syncObj_nvm_cpu);
    if (status != NVMEDIA_STATUS_OK)
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
    if(status != NVMEDIA_STATUS_OK) {
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
        OverrideSGMParam(&args, &nvMediaSGMParams, &ctx);
        status = NvMediaIOFASetSGMConfigParams(testOFA, &nvMediaSGMParams);
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("NVMEDIA_IOFA_MAIN: NvMediaIOFASetSGMConfigParams function failed \n");
            status_main = -1;
            goto fail;
        }
    }

#if __QNXNTO__
        if(args.enableVMSuspend)
        {
            nvdvms_set_vm_state(NVDVMS_SUSPEND);
        }
#endif

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
                    status = ReadImageOFA(args.inputFilename, ctx.frameRead, ctx.width[0],
                                          ctx.height[0], ctx.inputFrame[ctx.readIdx][0], 1,
                                          LSB_ALIGNED, ctx.pydLevel, args.bitDepth, inputFileChromaFormat);
                    if (status != NVMEDIA_STATUS_OK)
                    {
                        LOG_ERR("NVMEDIA_IOFA_MAIN: ReadImage failed\n");
                        break;
                    }
                    status = ReadImageOFA(args.inputFilename, ctx.frameRead +1, ctx.width[0],
                                          ctx.height[0], ctx.inputFrame[ctx.readIdx][1], 1,
                                          LSB_ALIGNED, ctx.pydLevel, args.bitDepth, inputFileChromaFormat);
                    if (status != NVMEDIA_STATUS_OK)
                    {
                        LOG_ERR("NVMEDIA_IOFA_MAIN: ReadImage failed\n");
                        break;
                    }
                    if ((args.frameIntervalInMS != 0U) &&
                        (ctx.frameRead == (args.inputBuffering-1U)))
                    {
                        skipReadImage = true;
                    }
                }
                LOG_DBG("NVMEDIA_IOFA_MAIN: ReadImageOFA done with frame count %d \n", ctx.frameRead);
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
            memset(&surfArray, 0, sizeof(NvMediaIofaBufArray));
            memset(&roiParams, 0, sizeof(NvMediaIofaROIParams));
            for (i = 0; i < ctx.pydLevel; i++)
            {
                if(args.backwardRef)
                {
                    surfArray.inputSurface[i] = ctx.inputFrameDup[ctx.queueIdx][1][i];
                    surfArray.refSurface[i] = ctx.inputFrameDup[ctx.queueIdx][0][i];
                }
                else
                {
                    surfArray.inputSurface[i] = ctx.inputFrameDup[ctx.queueIdx][0][i];
                    surfArray.refSurface[i] = ctx.inputFrameDup[ctx.queueIdx][1][i];
                }
                surfArray.outSurface[i] = ctx.outputSurface[ctx.queueIdx][i];
                surfArray.costSurface[i] = ctx.costSurface[ctx.queueIdx][i];
                if ((args.pydSGMMode == NVMEDIA_IOFA_PYD_LEVEL_MODE) && (i < ((int32_t)ctx.pydLevel - 1)))
                {
                    surfArray.pydHintSurface[i] = ctx.outputSurface[ctx.queueIdx][i+1U];
                }
                else
                {
                    surfArray.pydHintSurface[i] = NULL;
                }
            }

            if (args.roiMode == 1U)
            {
                status = ClearSurface(ctx.outWidth[0], ctx.outHeight[0],ctx.outputSurface[ctx.queueIdx][0], RG16);
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

            memset(&ofaProcessParams, 0, sizeof(NvMediaIofaProcessParams));
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
            if (ofaInitParams.ofaMode == NVMEDIA_IOFA_MODE_EPIOF)
            {
                memset(&epipolarInfo, 0, sizeof(NvMediaIofaEpipolarInfo));
                if (args.multiEpi == 1U)
                {
                    memset(epiFileName, 0, sizeof(epiFileName));
                    GetFileName(args.epipolarFilename, ctx.frameProcessed, epiFileName);
                    if (!GetEpipolarInfo(&epipolarInfo, epiFileName))
                    {
                        LOG_ERR("NVMEDIA_IOFA_MAIN: Get Epipolar info failed\n");
                        status_main = -1;
                        goto fail;
                    }
                }
                else
                {
                    if (!GetEpipolarInfo(&epipolarInfo, args.epipolarFilename))
                    {
                        LOG_ERR("NVMEDIA_IOFA_MAIN: Get Epipolar info failed\n");
                        status_main = -1;
                        goto fail;
                    }
                }
            }
            status = NvMediaIOFAProcessFrame(testOFA, &surfArray, &ofaProcessParams, &epipolarInfo, &roiParams);
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
            LOG_INFO("NVMEDIA_IOFA_MAIN: OFA successfully submitted\n");

            if (!args.preFetchInput)
            {
                if (!skipReadImage)
                {
                    status = ReadImageOFA(args.inputFilename, ctx.frameQueued, ctx.width[0],
                                          ctx.height[0], ctx.inputFrame[ctx.queueIdx][0], 1,
                                          LSB_ALIGNED, ctx.pydLevel, args.bitDepth, inputFileChromaFormat);
                    if (status != NVMEDIA_STATUS_OK)
                    {
                        LOG_ERR("NVMEDIA_IOFA_MAIN: ReadImage failed\n");
                        break;
                    }
                    status = ReadImageOFA(args.inputFilename, ctx.frameQueued +1, ctx.width[0],
                                          ctx.height[0], ctx.inputFrame[ctx.queueIdx][1], 1,
                                          LSB_ALIGNED, ctx.pydLevel, args.bitDepth, inputFileChromaFormat);
                    if (status != NVMEDIA_STATUS_OK)
                    {
                        LOG_ERR("NVMEDIA_IOFA_MAIN: ReadImage failed\n");
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
                    NvpRecordSample(&executionLatencies, startTime, stopTime);
                }
#endif
                taskStatus.status = NvSciSyncTaskStatusOFA_Invalid;
                err = NvSciSyncFenceGetTaskStatus(&(ctx.eofFence[ctx.processIdx]), &taskStatus);
                if (err != NvSciError_Success) {
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

                NvSciSyncFenceClear(&(ctx.eofFence[ctx.processIdx]));
                NvSciSyncFenceClear(&(ctx.preFence[ctx.processIdx]));
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
                    LOG_ERR("NVMEDIA_IOFA_MAIN: Check if App set timeout of %d ms is sufficient for HW operation\n", timeout);
                    LOG_ERR("NVMEDIA_IOFA_MAIN: increase timeout if it is not sufficient to perform HW operation\n");
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

        // Initialization Latency includes time taken for IOFA Create and initialization,
        // Buffer allocation and buffer registration
        NvpAggregatePerfData(&initLatencies, initLatenciesArray, numOfLatenciesToBeAggregated);
        if (args.profileTestEnable)
        {
            NvpPerfStats_t submisionLatStats, executionLatStats, initLatStats;
            status = NvpCalcStats(&initLatencies, &initLatStats, USEC);
            if (status != NVP_PASS)
            {
                LOG_ERR("NVMEDIA_IOFA_MAIN: NvpCalcStats failed with Error %d\n", status);
            }
            status = NvpCalcStats(&submissionLatencies, &submisionLatStats, USEC);
            if (status != NVP_PASS)
            {
                LOG_ERR("NVMEDIA_IOFA_MAIN: NvpCalcStats failed with Error %d\n", status);
            }
            status = NvpCalcStats(&executionLatencies, &executionLatStats, USEC);
            if (status != NVP_PASS)
            {
                LOG_ERR("NVMEDIA_IOFA_MAIN: NvpCalcStats failed with Error %d\n", status);
            }
            if (args.initLat < initLatStats.mean)
            {
                 LOG_ERR("NVMEDIA_IOFA_MAIN: Initialization latency fail, Reference: %f, Measured: %f\n", args.initLat, initLatStats.mean);
                 status_main = -1;
            }
            if (args.submitLat < submisionLatStats.mean)
            {
                 LOG_ERR("NVMEDIA_IOFA_MAIN: Submission latency fail, Reference: %f, Measured: %f\n", args.submitLat, submisionLatStats.mean);
                 status_main = -1;
            }
            if (args.execLat < executionLatStats.mean)
            {
                 LOG_ERR("NVMEDIA_IOFA_MAIN: Execution latency fail, Reference: %f, Measured: %f\n", args.execLat, executionLatStats.mean);
                 status_main = -1;
            }
        }

        NvpPrintStats(&initLatencies, NULL, USEC, "Init Latency", false);
        NvpPrintStats(&submissionLatencies, NULL, USEC, "Submission Latency", false);
        NvpPrintStats(&executionLatencies, NULL, USEC, "Execution Latency", false);
        if (args.profileStatsFilePath)
        {
            status = NvpDumpData(&initLatencies);
            if (status != NVP_PASS)
            {
                LOG_ERR("NVMEDIA_IOFA_MAIN: NvpDumpData failed with Error %d\n", status);
            }
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
        for (k = 0; k < 2U; k++)
        {
            NvpDestroyPerfData(&initLatenciesPartData[k]);
        }
        NvpDestroyPerfData(&initLatencies);
        NvpDestroyPerfData(&submissionLatencies);
        NvpDestroyPerfData(&executionLatencies);
    }
#endif

    if (syncObj_nvm_cpu)
    {
        status = NvMediaIOFAUnregisterNvSciSyncObj(testOFA, syncObj_nvm_cpu);
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
    LOG_DBG("NVMEDIA_IOFA_MAIN: Destroying image frames\n");
    if (syncObj_nvm_cpu)
    {
        NvSciSyncObjFree(syncObj_nvm_cpu);
    }
    if (syncObj_cpu_nvm)
    {
        NvSciSyncObjFree(syncObj_cpu_nvm);
    }

    for (i = 0; i < args.inputBuffering; i++)
    {
        for (j = 0; j < 2; j++)
        {
            for (k = 0; k < NVMEDIA_IOFA_MAX_PYD_LEVEL; k++)
            {
                if (ctx.inputFrameDup[i][j][k] != NULL)
                {
                    NvSciBufObjFree(ctx.inputFrameDup[i][j][k]);
                }
                if (ctx.inputFrame[i][j][k] != NULL)
                {
                    NvSciBufObjFree(ctx.inputFrame[i][j][k]);
                }
            }
        }
    }
    LOG_DBG("NVMEDIA_IOFA_MAIN: Destroying out and cost surfaces\n");
    for (i = 0; i < args.inputBuffering; i++)
    {
        for (k = 0; k < NVMEDIA_IOFA_MAX_PYD_LEVEL; k++)
        {
            if (ctx.outputSurface[i][k] != NULL)
            {
                NvSciBufObjFree(ctx.outputSurface[i][k]);
            }
            if (ctx.costSurface[i][k] != NULL)
            {
                NvSciBufObjFree(ctx.costSurface[i][k]);
            }
        }
    }

    if (ctx.FlowCrcFile != NULL)
    {
        fclose(ctx.FlowCrcFile);
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
    if ((status_main == -1) || (ctx.frameProcessed == 0))
    {
        LOG_MSG("NVMEDIA_IOFA_MAIN: total failures: 1 \n");
        return -1;
    }
    else
    {
        LOG_MSG("NVMEDIA_IOFA_MAIN: total failures: 0 \n");
        return 0;
    }
}
