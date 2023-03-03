/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>
#include <stdbool.h>

#include "nvscibuf.h"
#include "nvscisync.h"

#include "cmdline.h"
#include "config_parser.h"
#include "log_utils.h"
#include "misc_utils.h"
#include "scibuf_utils.h"
#include "nvmedia_ijpd.h"

static bool decodeStop = 0;

#define MAX_BITSTREAM_SIZE (10 * 1024 * 1024)
#define ALIGN_16(_x) ((_x + 15) & (~15))
/* ICEIL(a, b) Returns the ceiling of a divided by b. */
#define ICEIL(a,b)  (((a) + (b)  - 1) /  (b))

#define IJPD_APP_BASE_ADDR_ALIGN 256U

/* Data structure that ties the buffer and synchronization primitive together */
typedef struct {
    /* TODO: This is currently not being used */
    NvSciSyncFence  eofFence;
    NvSciBufObj     bufObj;
} NvMediaAppBuffer;

/* Signal Handler for SIGINT */
static void sigintHandler(int sig_num)
{
    LOG_MSG("\n Exiting decode process \n");
    decodeStop = 1;
}

static NvMediaStatus
CheckVersion(void)
{
    NvMediaVersion version;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    memset(&version, 0, sizeof(NvMediaVersion));
    status = NvMediaIJPDGetVersion(&version);
    if (status != NVMEDIA_STATUS_OK)
        return status;

    if((version.major != NVMEDIA_IJPD_VERSION_MAJOR) ||
       (version.minor != NVMEDIA_IJPD_VERSION_MINOR) ||
       (version.patch != NVMEDIA_IJPD_VERSION_PATCH)) {
        LOG_ERR("%s: Incompatible JPEG Decode version found \n", __func__);
        LOG_ERR("%s: Client version: %d.%d.%d\n", __func__,
            NVMEDIA_IJPD_VERSION_MAJOR, NVMEDIA_IJPD_VERSION_MINOR,
            NVMEDIA_IJPD_VERSION_PATCH);
        LOG_ERR("%s: Core version: %d.%d.%d\n", __func__,
            version.major, version.minor, version.patch);
        return NVMEDIA_STATUS_INCOMPATIBLE_VERSION;
    }

    return status;
}

static NvMediaStatus
sAllocEOFNvSciSyncObj(
    NvMediaIJPD     *ijpdCtx,
    NvSciSyncModule syncModule,
    NvSciSyncObj    *eofSyncObj
)
{
    NvSciSyncAttrList ijpdSignalerAttrList = NULL;
    NvSciSyncAttrList cpuWaiterAttrList = NULL;
    bool cpuWaiter = true;
    NvSciSyncAttrList syncUnreconciledList[2] = {NULL};
    NvSciSyncAttrList syncReconciledList = NULL;
    NvSciSyncAttrList syncNewConflictList = NULL;
    NvSciSyncAttrKeyValuePair keyValue[2] = {0};
    NvSciError err = NvSciError_Success;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    err = NvSciSyncAttrListCreate(syncModule, &ijpdSignalerAttrList);
    if(err != NvSciError_Success) {
        LOG_ERR("%s: Create signaler attr list failed. Error: %d \n", __func__, err);
        goto fail;
    }

    status = NvMediaIJPDFillNvSciSyncAttrList(ijpdCtx, ijpdSignalerAttrList,
            NVMEDIA_SIGNALER);
    if(status != NVMEDIA_STATUS_OK) {
       LOG_ERR("main: Failed to fill signaler attr list.\n");
       goto fail;
    }

    err = NvSciSyncAttrListCreate(syncModule, &cpuWaiterAttrList);
    if(err != NvSciError_Success) {
        LOG_ERR("%s: Create waiter attr list failed. Error: %d \n", __func__, err);
        status = NVMEDIA_STATUS_ERROR;
        goto fail;
    }

    /* Fill  NvSciSyncAttrList cpu waiter*/
    keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void *)&cpuWaiter;
    keyValue[0].len = sizeof(cpuWaiter);
    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
    keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*)&cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);

    err = NvSciSyncAttrListSetAttrs(cpuWaiterAttrList, keyValue, 2);
    if(err != NvSciError_Success) {
        LOG_ERR("NvSciSyncAttrListSetAttrs failed\n");
        status = NVMEDIA_STATUS_ERROR;
        goto fail;
    }

    syncUnreconciledList[0] = ijpdSignalerAttrList;
    syncUnreconciledList[1] = cpuWaiterAttrList;

    /* Reconcile Signaler and Waiter NvSciSyncAttrList */
    err = NvSciSyncAttrListReconcile(syncUnreconciledList, 2, &syncReconciledList,
            &syncNewConflictList);
    if(err != NvSciError_Success) {
        LOG_ERR("NvSciSyncAttrListSetAttrs failed\n");
        status = NVMEDIA_STATUS_ERROR;
        goto fail;
    }

    /* Create NvSciSync object and get the syncObj */
    err = NvSciSyncObjAlloc(syncReconciledList, eofSyncObj);
    if(err != NvSciError_Success) {
        LOG_ERR("NvSciSyncAttrListSetAttrs failed\n");
        status = NVMEDIA_STATUS_ERROR;
        goto fail;
    }

    status = NVMEDIA_STATUS_OK;

fail:

    if (NULL != syncReconciledList) {
        NvSciSyncAttrListFree(syncReconciledList);
    }
    if (NULL != syncUnreconciledList[0]) {
        NvSciSyncAttrListFree(syncUnreconciledList[0]);
    }
    if (NULL != syncUnreconciledList[1]) {
        NvSciSyncAttrListFree(syncUnreconciledList[1]);
    }
    if (NULL != syncNewConflictList) {
        NvSciSyncAttrListFree(syncNewConflictList);
    }

    return status;
}


int main(int argc, char *argv[])
{
    TestArgs args;
    FILE *crcFile = NULL, *streamFile = NULL;
    char inFileName[FILE_NAME_SIZE];
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvMediaIJPD *ijpdCtx = NULL;
    NVMEDIAJPEGDecInfo jpegInfo = {0};
    NvMediaRect dstRect = {0};
    NvMediaRect srcRect = {0};
    uint8_t planeCount = 0;
    bool nextFrameFlag = true;
    long fileLength = 0;
    NvMediaBitstreamBuffer bsBuffer = {0};
    uint32_t frameCounter = 0;//, calcCrc = 0;
    uint64_t startTime, endTime;
    double elapse = 0;
    bool testPass = false;
    NVMEDIAJPEGDecAttributes attributes = {0};
    NvSciSyncModule syncModule = NULL;
    NvSciBufModule bufModule = NULL;
    NvSciSyncCpuWaitContext cpuWaitContext = NULL;
    NvSciError err = NvSciError_Success;
    NvSciBufAttrList bufAttributeList = {0};
    NvMediaAppBuffer appBuffer = {0};
    NvSciBufAttrList bufConflictList = {0};
    NvSciBufAttrList bufReconciledList = {0};
    NvSciSyncObj eofSyncObj = {0};
    unsigned int scaleFactor = 1;

    signal(SIGINT, sigintHandler);
    signal(SIGTERM, sigintHandler);

    memset(&args, 0, sizeof(TestArgs));
    memset(&bsBuffer, 0, sizeof(NvMediaBitstreamBuffer));

    LOG_DBG("main: Parsing jpeg decode command\n");
    if(!ParseArgs(argc, argv, &args)) {
        LOG_ERR("main: Parsing arguments failed\n");
        return -1;
    }

    if(CheckVersion() != NVMEDIA_STATUS_OK) {
        return -1;
    }

    if(args.crcoption.crcGenMode && args.crcoption.crcCheckMode) {
        LOG_ERR("main: crcGenMode and crcCheckMode cannot be enabled at the same time\n");
        return -1;
    }

    err = NvSciBufModuleOpen(&bufModule);
    if(err != NvSciError_Success) {
        LOG_ERR("%s: NvSciBuffModuleOpen failed\n",__func__);
        goto fail;
    }

    err = NvSciSyncModuleOpen(&syncModule);
    if(err != NvSciError_Success) {
        LOG_ERR("%s: NvSciBuffModuleOpen failed\n",__func__);
        goto fail;
    }

    err = NvSciSyncCpuWaitContextAlloc(syncModule, &cpuWaitContext);
    if(err != NvSciError_Success) {
        LOG_ERR("%s: NvSciBuffModuleOpen failed\n",__func__);
        goto fail;
    }

    // Read JPEG stream, get stream info
    sprintf(inFileName, args.infile, frameCounter);
    streamFile = fopen(inFileName, "rb");
    if(!streamFile) {
        LOG_ERR("main: Error opening '%s' for reading, decode done!\n", inFileName);
        nextFrameFlag = false;
        goto fail;
    }
    fseek(streamFile, 0, SEEK_END);
    fileLength = ftell(streamFile);
    if(!fileLength) {
        LOG_ERR("main: Zero file length for file %s, len=%d\n", args.infile, (int)fileLength);
        fclose(streamFile);
        goto fail;
    }

    bsBuffer.bitstream = malloc(fileLength);
    if(!bsBuffer.bitstream) {
        LOG_ERR("main: Error allocating %d bytes\n", fileLength);
        goto fail;
    }
    bsBuffer.bitstreamBytes = fileLength;
    fseek(streamFile, 0, SEEK_SET);
    if(fread(bsBuffer.bitstream, fileLength, 1, streamFile) != 1) {
       LOG_ERR("main: Error read JPEG file %s for %d bytes\n", inFileName, fileLength);
       goto fail;
    }
    fclose(streamFile);

    status = NvMediaIJPDGetInfo(&jpegInfo, 1, &bsBuffer);
    free(bsBuffer.bitstream);
    bsBuffer.bitstream = NULL;
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("main: Can't get JPEG stream info.\n");
        goto fail;
    }

    LOG_DBG("main: NvMediaIJPDGetInfo: width=%d, height=%d, partialAccel=%d\n",
            jpegInfo.width, jpegInfo.height, jpegInfo.partialAccel);

    args.supportPartialAccel = jpegInfo.partialAccel;
    if(!args.outputWidth || !args.outputHeight) {
        args.outputWidth = jpegInfo.width;
        args.outputHeight = jpegInfo.height;
    }

    // align width and height to multiple of 16
    args.outputWidth = ALIGN_16(args.outputWidth);
    args.outputHeight = ALIGN_16(args.outputHeight);

    if (!args.isYUV) {
        dstRect.x0 = 0;
        dstRect.y0 = 0;
        dstRect.x1 = args.outputWidth;
        dstRect.y1 = args.outputHeight;
        planeCount = 1;
    } else if (args.bMonochrome) {
        /* Planar pitch-linear monochrome */
        planeCount = 1;
    } else {
        /* Planar pitch-linear formats */
        planeCount = 3;
    }

    err = NvSciBufAttrListCreate(bufModule, &bufAttributeList);
    if(err != NvSciError_Success) {
        LOG_ERR("%s: SciBufAttrListCreate failed. Error: %d \n", __func__, err);
        goto fail;
    }

    status = NvMediaIJPDFillNvSciBufAttrList(args.instanceId,
                                             bufAttributeList);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("main: Failed to populate IJPD internal attributes\n");
        goto fail;
    }

    status = PopulateNvSciBufAttrList(
            args.chromaFormat,
            args.outputWidth,
            args.outputHeight,
            true,                           /* needCpuAccess */
            NvSciBufImage_PitchLinearType,
            planeCount,
            NvSciBufAccessPerm_ReadWrite,
            IJPD_APP_BASE_ADDR_ALIGN,
            NvSciColorStd_REC709_ER,
            NvSciBufScan_ProgressiveType,
            bufAttributeList);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("main: Failed to populate attributes\n");
        goto fail;
    }

    err = NvSciBufAttrListReconcile(&bufAttributeList, 1U,
            &bufReconciledList, &bufConflictList);
    if (err != NvSciError_Success) {
        LOG_ERR("main: Reconciliation for input frame failed\n");
        goto fail;
    }

    err = NvSciBufObjAlloc(bufReconciledList, &appBuffer.bufObj);
    if (err != NvSciError_Success) {
        LOG_ERR("main: Allocation of input frame failed\n");
        goto fail;
    }

    /* Free the allocated lists */
    NvSciBufAttrListFree(bufReconciledList);
    NvSciBufAttrListFree(bufAttributeList);
    NvSciBufAttrListFree(bufConflictList);

    ijpdCtx = NvMediaIJPDCreate(args.maxWidth,
                                args.maxHeight,
                                args.maxBitstreamBytes,
                                args.supportPartialAccel,
                                args.instanceId);
    if(!ijpdCtx) {
        LOG_ERR("main: NvMediaIJPDCreate failed\n");
        goto fail;
    }

    LOG_DBG("main: NvMediaIJPDCreate, %p\n", ijpdCtx);

    LOG_DBG("main: Jpeg Resize\n");
    status = NvMediaIJPDResize(ijpdCtx, args.maxWidth, args.maxHeight, args.maxBitstreamBytes);
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("main: NvMediaIJPDResize failed\n");
        goto fail;
    }

    LOG_DBG("main: Jpeg Resize successful\n");
    LOG_DBG("main: Nvmedia set attributes - Alpha value as 0xFF\n");
    attributes.alphaValue = 0xFF;
    status = NvMediaIJPDSetAttributes(ijpdCtx, NVMEDIA_JPEG_DEC_ATTRIBUTE_ALPHA_VALUE, &attributes);
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("main: Cannot set attributes for alpha value\n");
        goto fail;
    }

    LOG_DBG("main: Nvmedia set attributes - Color standard\n");
    attributes.colorStandard = NVMEDIA_IJPD_COLOR_STANDARD_ITUR_BT_601;
    status = NvMediaIJPDSetAttributes(ijpdCtx, NVMEDIA_JPEG_DEC_ATTRIBUTE_COLOR_STANDARD, &attributes);
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("main: Cannot set attributes for ITU BT.601 color standard\n");
        goto fail;
    }

    attributes.colorStandard = NVMEDIA_IJPD_COLOR_STANDARD_ITUR_BT_709;
    status = NvMediaIJPDSetAttributes(ijpdCtx, NVMEDIA_JPEG_DEC_ATTRIBUTE_COLOR_STANDARD, &attributes);
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("main: Cannot set attributes for ITU BT.709 color standard\n");
        goto fail;
    }

    attributes.colorStandard = NVMEDIA_IJPD_COLOR_STANDARD_ITUR_BT_601_ER;
    status = NvMediaIJPDSetAttributes(ijpdCtx, NVMEDIA_JPEG_DEC_ATTRIBUTE_COLOR_STANDARD, &attributes);
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("main: Cannot set attributes for ITU BT.601 color standard extended \n");
        goto fail;
    }

    attributes.colorStandard = NVMEDIA_IJPD_COLOR_STANDARD_ITUR_BT_709_ER;
    status = NvMediaIJPDSetAttributes(ijpdCtx, NVMEDIA_JPEG_DEC_ATTRIBUTE_COLOR_STANDARD, &attributes);
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("main: Cannot set attributes for ITU BT.709 color standard extended\n");
        goto fail;
    }

    LOG_DBG("main: Nvmedia set attributes successful\n");

    status = sAllocEOFNvSciSyncObj(ijpdCtx, syncModule, &eofSyncObj);
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("main: NvMediaIJPDResize failed\n");
        goto fail;
    }

    LOG_DBG("main: Created EOF NvSciSyncObj\n");

    status = NvMediaIJPDRegisterNvSciSyncObj(ijpdCtx, NVMEDIA_EOFSYNCOBJ, eofSyncObj);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("main: Failed to register EOF NvSciSyncObj\n");
        goto fail;
    }

    LOG_DBG("main: Registered EOF NvSciSyncObj\n");

    status = NvMediaIJPDSetNvSciSyncObjforEOF(ijpdCtx, eofSyncObj);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("main: Failed to set EOF NvSciSyncObj\n");
        goto fail;
    }

    LOG_DBG("main: Set EOF NvSciSyncObj\n");

    if(args.crcoption.crcGenMode){
        crcFile = fopen(args.crcoption.crcFilename, "wt");
        if(!crcFile){
            LOG_ERR("main: Cannot open crc gen file for writing\n");
            goto fail;
        }
    } else if(args.crcoption.crcCheckMode){
        crcFile = fopen(args.crcoption.crcFilename, "rb");
        if(!crcFile){
            LOG_ERR("main: Cannot open crc gen file for reading\n");
            goto fail;
        }
    }

    while(nextFrameFlag && (decodeStop == 0u)) {
        // Read JPEG stream
        sprintf(inFileName, args.infile, frameCounter);
        streamFile = fopen(inFileName, "rb");
        if(!streamFile) {
            LOG_ERR("main: Error opening '%s' for reading, decode done!\n", inFileName);
            nextFrameFlag = false;
            goto done;
        }
        fseek(streamFile, 0, SEEK_END);
        fileLength = ftell(streamFile);
        if(!fileLength) {
           LOG_ERR("main: Zero file length for file %s, len=%d\n", args.infile, (int)fileLength);
           fclose(streamFile);
           goto fail;
        }

        bsBuffer.bitstream = malloc(fileLength);
        if(!bsBuffer.bitstream) {
           LOG_ERR("main: Error allocating %d bytes\n", fileLength);
           goto fail;
        }
        bsBuffer.bitstreamBytes = fileLength;
        fseek(streamFile, 0, SEEK_SET);
        if(fread(bsBuffer.bitstream, fileLength, 1, streamFile) != 1) {
           LOG_ERR("main: Error read JPEG file %s for %d bytes\n", inFileName, fileLength);
           goto fail;
        }
        fclose(streamFile);
        LOG_DBG("main: Read JPEG stream %d done\n", frameCounter);

        GetTimeMicroSec(&startTime);
        LOG_DBG("main: Decoding frame #%d\n", frameCounter);

        if (!args.isYUV) {
           status = NvMediaIJPDRender(ijpdCtx,
                                      appBuffer.bufObj,
                                      NULL,
                                      &dstRect,
                                      args.downscaleLog2,
                                      1,
                                      &bsBuffer,
                                      0,
                                      args.instanceId);
        } else {
           status = NvMediaIJPDRenderYUV(ijpdCtx,
                                         appBuffer.bufObj,
                                         args.downscaleLog2,
                                         1,
                                         &bsBuffer,
                                         0,
                                         args.instanceId);
        }
        if(status != NVMEDIA_STATUS_OK) {
            LOG_ERR("main: NvMediaIJPDRender(%s) failed: %x\n",
                    (args.isYUV)? "YUV":"RGBA", status);
            goto fail;
        }

        GetTimeMicroSec(&endTime);
        elapse += (double)(endTime - startTime) / 1000.0;

        status = NvMediaIJPDGetEOFNvSciSyncFence(ijpdCtx, eofSyncObj,
                                                 &appBuffer.eofFence);
        if(status != NVMEDIA_STATUS_OK) {
            LOG_ERR("main: NvMediaIEPGetEOFNvSciSyncFence failed: %x\n", status);
            goto fail;
        }

        switch(args.downscaleLog2) {
        case 1:
            scaleFactor = 2;
            break;
        case 2:
            scaleFactor = 4;
            break;
        case 3:
            scaleFactor = 8;
            break;
        default:
            scaleFactor = 1;
        }
      
        if(args.cropCRC) {
            srcRect.x0 = 0;
            srcRect.y0 = 0;
            if (args.downscaleLog2) {
                srcRect.x1 = ICEIL(jpegInfo.width, scaleFactor);
                srcRect.y1 = ICEIL(jpegInfo.height, scaleFactor);
            }
            else {
                srcRect.x1 = jpegInfo.width;
                srcRect.y1 = jpegInfo.height;
            }
            LOG_DBG("main: using cropcrc rect (%d, %d) x (%d, %d)", srcRect.x0, srcRect.y0, srcRect.x1, srcRect.y1);
        } else {
            srcRect.x0 = 0;
            srcRect.y0 = 0;
            srcRect.x1 = args.outputWidth;
            srcRect.y1 = args.outputHeight;
        }

        /* Wait for operations on the image to be complete */
        err = NvSciSyncFenceWait(&appBuffer.eofFence, cpuWaitContext, 1000*1000);
        if(err != NvSciError_Success) {
            LOG_ERR("NvSciSyncFenceWait failed: %u\n", err);
            goto fail;
        }

        /* Clear pre-fence */
        NvSciSyncFenceClear(&appBuffer.eofFence);

        /* Write the output now that it has been generated */
        status = WriteOutput(args.outfile,
                             appBuffer.bufObj,
                             true,
                             frameCounter ? true : false,
                             NULL);
        if(status != NVMEDIA_STATUS_OK) {
            LOG_ERR("main: WriteOutput failed: %x\n", status);
            goto fail;
        }

        free(bsBuffer.bitstream);
        bsBuffer.bitstream = NULL;

        if(args.crcoption.crcGenMode){
            uint32_t calcCrc = 0U;
            status = GetNvSciBufObjCrc(appBuffer.bufObj,
                                       &srcRect,
                                       args.bMonochrome,
                                       &calcCrc);
            if(status != NVMEDIA_STATUS_OK) {
                LOG_ERR("main: GetNvSciBufObjCrc failed: %x\n", status);
                goto fail;
            }

            if(!fprintf(crcFile, "%08x\n",calcCrc)) {
                LOG_ERR("main: Failed writing calculated CRC to file %s\n", crcFile);
                goto fail;
            }
        } else if(args.crcoption.crcCheckMode){
            uint32_t calcCrc = 0U;
            uint32_t refCrc = 0U;
            if(fscanf(crcFile, "%8x\n", &refCrc) == 1) {
                status = GetNvSciBufObjCrc(appBuffer.bufObj,
                                           &srcRect,
                                           args.bMonochrome,
                                           &calcCrc);
                if(status != NVMEDIA_STATUS_OK) {
                    LOG_ERR("main: GetNvSciBufObjCrc failed: %x\n", status);
                    goto fail;
                }

                if(refCrc != calcCrc){
                    LOG_ERR("main: Frame %d crc 0x%x does not match with ref crc 0x%x\n",
                            frameCounter, calcCrc, refCrc);
                    goto fail;
                }
            } else {
                LOG_ERR("main: Failed checking CRC. Failed reading file %s\n", crcFile);
                goto fail;
            }

        }
        // Next frame
        frameCounter++;

        if(frameCounter == args.frameNum) {
            nextFrameFlag = false;
        }
    }

done:
    //get decoding time info
    LOG_MSG("\nTotal Decoding time for %d frames: %.3f ms\n", frameCounter, elapse);
    LOG_MSG("Decoding time per frame %.4f ms \n", elapse / frameCounter);
    LOG_MSG("\nTotal decoded frames = %d\n", frameCounter);
    if (args.crcoption.crcGenMode){
        LOG_MSG("\n***crc gold file %s has been generated***\n", args.crcoption.crcFilename);
    } else if (args.crcoption.crcCheckMode){
        LOG_MSG("\n***crc checking with file %s is successful\n", args.crcoption.crcFilename);
    }
    LOG_MSG("\n***DECODING PROCESS ENDED SUCCESSFULY***\n");
    testPass = true;

fail:

    /* Clear any populated fences */
    NvSciSyncFenceClear(&appBuffer.eofFence);

    if (eofSyncObj) {
        /* Unregister NvSciSyncObj */
        status = NvMediaIJPDUnregisterNvSciSyncObj(ijpdCtx, eofSyncObj);
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("%s: Failed to unregister NvSciSyncObj\n", __func__);
        }

        /* Free NvSciSyncObj */
        NvSciSyncObjFree(eofSyncObj);
    }

    /* Free NvSciBufObj */
    if(appBuffer.bufObj) {
        NvSciBufObjFree(appBuffer.bufObj);
    }

    if (ijpdCtx) {
        NvMediaIJPDDestroy(ijpdCtx);
    }

    if (NULL != cpuWaitContext) {
        NvSciSyncCpuWaitContextFree(cpuWaitContext);
    }

    if (NULL != syncModule) {
        NvSciSyncModuleClose(syncModule);
    }

    if (NULL != bufModule) {
        NvSciBufModuleClose(bufModule);
    }

    if(crcFile) {
        fclose(crcFile);
    }

    if (bsBuffer.bitstream)
        free(bsBuffer.bitstream);

    if (testPass) {
        LOG_MSG("total failures: 0 \n");
        return 0;
    } else {
        LOG_MSG("total failures: 1 \n");
        return -1;
    }
}
