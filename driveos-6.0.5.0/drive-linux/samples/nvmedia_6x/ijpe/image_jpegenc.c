/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>

#include "nvscisync.h"
#include "nvscibuf.h"
#include "nvscierror.h"
#include "nvmedia_ijpe.h"
#include "cmdline.h"
#include "config_parser.h"
#include "log_utils.h"
#include "misc_utils.h"
#include "scibuf_utils.h"
#include "thread_utils.h"

static bool encodeStop = 0;

#define MAX_BITSTREAM_SIZE (256 * 1024 * 1024) // 256 MB
#define MAX_HUFFMAN_CODES        256
#define MAX_HUFFMAN_CODE_LENGTH  16
#define QUANTIZATION_TABLE_SIZE  64

#define IJPE_APP_BASE_ADDR_ALIGN 256U

/* Signal Handler for SIGINT */
static void sigintHandler(int sig_num)
{
    LOG_MSG("\n Exiting encode process \n");
    encodeStop = 1;
}

/* Data structure that ties the buffer and synchronization primitive together */
typedef struct {
    NvSciSyncFence  preFence;
    NvSciBufObj     bufObj;
} NvMediaAppBuffer;

static NvMediaStatus
CheckVersion(void)
{
    NvMediaVersion version;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    memset(&version, 0, sizeof(NvMediaVersion));
    status = NvMediaIJPEGetVersion(&version);
    if (status != NVMEDIA_STATUS_OK)
        return status;

    if((version.major != NVMEDIA_IJPE_VERSION_MAJOR) ||
       (version.minor != NVMEDIA_IJPE_VERSION_MINOR) ||
       (version.patch != NVMEDIA_IJPE_VERSION_PATCH)) {
        LOG_ERR("%s: Incompatible image version found \n", __func__);
        LOG_ERR("%s: Client version: %d.%d.%d\n", __func__,
            NVMEDIA_IJPE_VERSION_MAJOR, NVMEDIA_IJPE_VERSION_MINOR, NVMEDIA_IJPE_VERSION_PATCH);
        LOG_ERR("%s: Core version: %d.%d.%d\n", __func__,
            version.major, version.minor, version.patch);
        return NVMEDIA_STATUS_INCOMPATIBLE_VERSION;
    }

    return status;
}

/*******************************************************************************
*   Function Name: readInputTable
*   Purpose:
*       Search for the input string "pString" in the input file
*       and reads "numElements" number of values from that point
*       and store it in the location - pTable
*
*   Inputs:
*        fp          - input file
*        pString     - String to be searched
*        pTable      - Location to save read values
*        numElements - number of elements to be read
*
*   Return Data:
*       Status       - 1 on success and 0 on failure.
*
*******************************************************************************/
static uint32_t
readInputTable(FILE *fp, const char *pString, unsigned char *pTable, uint32_t numElements)
{
    char temp[256];
    uint32_t j;
    bool found = false;
    while ((fscanf(fp, "%s", temp)) != EOF) {
        if (strcmp(temp, pString) == 0){
            found = true;
            break;
        }
    }

    if (found == false) {
        LOG_ERR("Huffman/Quant table is not found\n");
        return 0;
    }

    if ((fscanf(fp, "%s %s", temp + 10, temp + 20)) == EOF) {
        LOG_ERR("Error in huffman Table entry format\n");
        return 0;
    }

    /* Reading  values*/
    for (j = 0; j < numElements; j++) {
        if ((fscanf(fp, "%hhu,", &pTable[j])) == EOF) {
            LOG_ERR("Error in huffman Table entry format\n");
            return 0;
        }
    }
    return 1;
}

/*******************************************************************************
*   Function Name: sumOfTable
*   Purpose:
*       Add first n elements of the input array and returns the sum
*
*   Inputs:
*        array       - Input array
*        numElements - number of elements to be added
*
*   Return Data:
*        sum       - sum of the elements
*
*******************************************************************************/
static uint32_t
sumOfTable(unsigned char* array, uint32_t numElements)
{
    uint32_t sum = 0; // initialize sum
    uint32_t i;

    // Iterate through all elements and add them to sum
    for (i = 0; i < numElements; i++) {
        sum += array[i];
    }

    return sum;
}

static uint32_t
readHuffmanTable(FILE *huffFile,
                 NvMediaJPHuffmanTableSpecfication *colorComponent,
                 const char *pStringBits,
                 const char *pStringVal)
{
    uint32_t numValues;

    /* reads number of codes with length 1-16 */
    if (!readInputTable(huffFile, pStringBits, colorComponent->length, MAX_HUFFMAN_CODE_LENGTH)) {
        return 0;
    }

    /* calculate total number of codes */
    numValues = sumOfTable(colorComponent->length, MAX_HUFFMAN_CODE_LENGTH);
    /* total number of codes should not exceed 256 */
    if (numValues > MAX_HUFFMAN_CODES) {
        LOG_ERR("Error in Huffman table\n");
        return 0;
    }

    colorComponent->values = calloc(MAX_HUFFMAN_CODES, sizeof(unsigned char));
    if (!colorComponent->values) {
        LOG_ERR("Failed to allocate memory for huffman code values\n");
        return 0;
    }
    /* get values for each code */
    return (readInputTable(huffFile, pStringVal, colorComponent->values, numValues));
}

static NvMediaStatus
sAllocEOFNvSciSyncObj(
    NvMediaIJPE     *ijpeCtx,
    NvSciSyncModule syncModule,
    NvSciSyncObj    *eofSyncObj
)
{
    NvSciSyncAttrList ijpeSignalerAttrList = NULL;
    NvSciSyncAttrList cpuWaiterAttrList = NULL;
    bool cpuWaiter = true;
    NvSciSyncAttrList syncUnreconciledList[2] = {NULL};
    NvSciSyncAttrList syncReconciledList = NULL;
    NvSciSyncAttrList syncNewConflictList = NULL;
    NvSciSyncAttrKeyValuePair keyValue[2] = {0};
    NvSciError err = NvSciError_Success;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    err = NvSciSyncAttrListCreate(syncModule, &ijpeSignalerAttrList);
    if(err != NvSciError_Success) {
        LOG_ERR("%s: Create signaler attr list failed. Error: %d \n", __func__, err);
        goto fail;
    }

    status = NvMediaIJPEFillNvSciSyncAttrList(ijpeCtx, ijpeSignalerAttrList,
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

    syncUnreconciledList[0] = ijpeSignalerAttrList;
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
    FILE *crcFile = NULL, *outputFile = NULL, *streamFile = NULL;
    FILE *huffFile = NULL, *quantFile = NULL;
    char outFileName[FILE_NAME_SIZE];
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvMediaAppBuffer *appBuffer = NULL;
    NvMediaIJPE *ijpeCtx = NULL;
    bool nextFrameFlag = true, encodeDoneFlag;
    long long totalBytes = 0;
    long fileLength;
    uint8_t *buffer = NULL;
    uint32_t framesNum = 0, frameCounter = 0, bytes, bytesAvailable = 0, calcCrc = 0;
    uint32_t imageSize = 0;
    uint64_t startTime, endTime1, endTime2;
    double encodeTime = 0;
    double getbitsTime = 0;
    NvMediaJPEncAttributes attr;
    uint32_t i;
    bool testPass = false;
    NvSciError err;
    NvSciBufAttrList bufAttributeList;
    NvSciBufModule bufModule = NULL;
    NvSciSyncObj eofSyncObj = {0};
    NvSciSyncModule syncModule = NULL;
    NvSciSyncCpuWaitContext cpuWaitContext = NULL;

    signal(SIGINT, sigintHandler);
    signal(SIGTERM, sigintHandler);

    memset(&args, 0, sizeof(TestArgs));

    LOG_DBG("main: Parsing jpeg encode command\n");
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

    imageSize = (args.inputWidth * args.inputHeight * 3) / 2;

    LOG_DBG("main: Encode start from frame %d, imageSize=%d\n", frameCounter, imageSize);

    streamFile = fopen(args.infile, "rb");
    if(!streamFile) {
        LOG_ERR("main: Error opening '%s' for reading\n", args.infile);
        goto fail;
    }
    fseek(streamFile, 0, SEEK_END);
    fileLength = ftell(streamFile);
    fclose(streamFile);
    if(!fileLength) {
       LOG_ERR("main: Zero file length for file %s, len=%d\n", args.infile, (int)fileLength);
       goto fail;
    }
    framesNum = fileLength / imageSize;

    err = NvSciBufAttrListCreate(bufModule, &bufAttributeList);
    if(err != NvSciError_Success) {
        LOG_ERR("%s: SciBufAttrListCreate failed. Error: %d \n", __func__, err);
        goto fail;
    }

    status = NvMediaIJPEFillNvSciBufAttrList(args.instanceId,
                                             bufAttributeList);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("main: Failed to populate IJPD internal attributes\n");
        goto fail;
    }

    status = PopulateNvSciBufAttrList(
            YUV420SP_8bit,
            args.inputWidth,
            args.inputHeight,
            true,                           /* needCpuAccess */
            NvSciBufImage_PitchLinearType,
            2,
            NvSciBufAccessPerm_ReadWrite,
            IJPE_APP_BASE_ADDR_ALIGN,
            NvSciColorStd_REC709_ER,
            NvSciBufScan_ProgressiveType,
            bufAttributeList);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("main: Failed to populate attributes\n");
        goto fail;
    }

    NvSciBufAttrList bufConflictList;
    NvSciBufAttrList bufReconciledList;
    err = NvSciBufAttrListReconcile(&bufAttributeList, 1U,
            &bufReconciledList, &bufConflictList);
    if (err != NvSciError_Success) {
        LOG_ERR("main: Reconciliation for input frame failed\n");
        goto fail;
    }

    appBuffer = malloc(sizeof(NvMediaAppBuffer));
    memset(appBuffer, 0x0, sizeof(NvMediaAppBuffer));
    err = NvSciBufObjAlloc(bufReconciledList, &appBuffer->bufObj);
    if (err != NvSciError_Success) {
        LOG_ERR("main: Reconciliation for input frame failed\n");
        goto fail;
    }

    ijpeCtx = NvMediaIJPECreate(bufReconciledList,               // inputFormat
                                args.maxOutputBuffering,          // maxOutputBuffering
                                MAX_BITSTREAM_SIZE,               // maxBitstreamBytes
                                args.instanceId);                 // HW instance id
    if(!ijpeCtx) {
        LOG_ERR("main: NvMediaIJPECreate failed\n");
        goto fail;
    }

    LOG_DBG("main: NvMediaIJPECreate, %p\n", ijpeCtx);

    /* The reconciled list is needed for later */
    NvSciBufAttrListFree(bufAttributeList);
    NvSciBufAttrListFree(bufConflictList);

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

    if (args.huffTable) {
        attr.lumaDC = malloc(sizeof(NvMediaJPHuffmanTableSpecfication));
        attr.chromaDC = malloc(sizeof(NvMediaJPHuffmanTableSpecfication));
        attr.lumaAC = malloc(sizeof(NvMediaJPHuffmanTableSpecfication));
        attr.chromaAC = malloc(sizeof(NvMediaJPHuffmanTableSpecfication));
        if ((!attr.lumaDC) || (!attr.chromaDC) || (!attr.lumaAC) || (!attr.chromaAC)) {
            LOG_ERR("Failed to allocate memory\n");
            goto fail;
        }

        huffFile = fopen(args.huffFileName, "r");
        if(!huffFile){
            LOG_ERR("main: Cannot open huffman-table file for reading\n");
            goto fail;
        }


        if (!(readHuffmanTable(huffFile,
                              attr.lumaDC,
                              "dc_luminance_bits[16]",
                              "dc_luminance_value"))) {
            LOG_ERR("Error in Huffman table\n");
            goto fail;
        }

        if (!(readHuffmanTable(huffFile,
                              attr.chromaDC,
                              "dc_chrominance_bits[16]",
                              "dc_chrominance_value"))) {
            LOG_ERR("Error in Huffman table\n");
            goto fail;
        }

        if (!(readHuffmanTable(huffFile,
                              attr.lumaAC,
                              "ac_luminance_bits[16]",
                              "ac_luminance_value"))) {
            LOG_ERR("Error in Huffman table\n");
            goto fail;
        }

        if (!(readHuffmanTable(huffFile,
                              attr.chromaAC,
                              "ac_chrominance_bits[16]",
                              "ac_chrominance_value"))) {
            LOG_ERR("Error in Huffman table\n");
            goto fail;
        }

        status = NvMediaIJPESetAttributes(ijpeCtx,
                                          NVMEDIA_IMAGE_JPEG_ATTRIBUTE_HUFFTABLE,
                                          &attr);
        if (status != NVMEDIA_STATUS_OK) {
           LOG_ERR("main: NvMediaIJPESetAttributes failed\n");
           goto fail;
        }
    }

    if (args.quantTable) {
        quantFile = fopen(args.quantFileName, "r");
        if (!quantFile){
            LOG_ERR("main: Cannot open quantization-table file for reading\n");
            goto fail;
        }

        if (!readInputTable(quantFile,
                            "luma_quant_tbl[64]",
                            attr.lumaQuant,
                            QUANTIZATION_TABLE_SIZE)) {
            LOG_ERR("Error in quantization table\n");
            goto fail;
        }

        if (!readInputTable(quantFile,
                            "chroma_quant_tbl[64]",
                            attr.chromaQuant,
                            QUANTIZATION_TABLE_SIZE)) {
            LOG_ERR("Error in quantization table\n");
            goto fail;
        }

        for (i = 0; i < QUANTIZATION_TABLE_SIZE; i++) {
            if ((attr.lumaQuant[i] == 0) || (attr.chromaQuant[i] == 0)) // invalid quant value
            {
                LOG_ERR("main: Invalid quant value\n");
                goto fail;
            }
        }

        status = NvMediaIJPESetAttributes(ijpeCtx,
                                          NVMEDIA_IMAGE_JPEG_ATTRIBUTE_QUANTTABLE,
                                          &attr);

        if (status != NVMEDIA_STATUS_OK) {
           LOG_ERR("main: NvMediaIJPESetAttributes failed\n");
           goto fail;
        }
    }

    status = sAllocEOFNvSciSyncObj(ijpeCtx, syncModule, &eofSyncObj);
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("main: NvMediaIJPEResize failed\n");
        goto fail;
    }

    LOG_DBG("main: Created EOF NvSciSyncObj\n");

    status = NvMediaIJPERegisterNvSciSyncObj(ijpeCtx, NVMEDIA_EOFSYNCOBJ, eofSyncObj);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("main: Failed to register EOF NvSciSyncObj\n");
        goto fail;
    }

    LOG_DBG("main: Registered EOF NvSciSyncObj\n");

    status = NvMediaIJPESetNvSciSyncObjforEOF(ijpeCtx, eofSyncObj);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("main: Failed to set EOF NvSciSyncObj\n");
        goto fail;
    }

    LOG_DBG("main: Set EOF NvSciSyncObj\n");

    while(nextFrameFlag && (encodeStop == 0u)) {
        // Read Frame
        LOG_DBG("main: Reading YUV frame %d from file %s to image surface location: %p. (W:%d, H:%d)\n",
                 frameCounter, args.infile, appBuffer->bufObj, args.inputWidth, args.inputHeight);
        status = ReadInput(args.infile,
                           frameCounter,
                           args.inputWidth,
                           args.inputHeight,
                           appBuffer->bufObj,
                           YUV420P_8bit,
                           1,                       //rawBytesPerPixel
                           MSB_ALIGNED);            //pixelAlignment
        if(status != NVMEDIA_STATUS_OK) {
           LOG_ERR("readYUVFile failed\n");
           goto fail;
        }
        LOG_DBG("main: ReadYUVFrame %d/%d done\n", frameCounter, framesNum-1);

        GetTimeMicroSec(&startTime);
        LOG_DBG("main: Encoding frame #%d\n", frameCounter);

        if (args.quantTable) {
            status = NvMediaIJPEFeedFrameQuant(ijpeCtx,
                                               appBuffer->bufObj,
                                               attr.lumaQuant,
                                               attr.chromaQuant,
                                               args.instanceId);
        }
        else {
            status = NvMediaIJPEFeedFrame(ijpeCtx,
                                          appBuffer->bufObj,
                                          args.quality,
                                          args.instanceId);
        }

        if(status != NVMEDIA_STATUS_OK) {
            LOG_ERR("main: NvMediaIJPEFeedFrameQuality failed: %x\n", status);
            goto fail;
        }

        encodeDoneFlag = false;
        while(!encodeDoneFlag) {
            bytesAvailable = 0;
            bytes = 0;
            status = NvMediaIJPEGetEOFNvSciSyncFence(ijpeCtx, eofSyncObj,
                                                 &appBuffer->preFence);
            if(status != NVMEDIA_STATUS_OK) {
                LOG_ERR("main: NvMediaIJPEGetEOFNvSciSyncFence failed: %x\n", status);
                goto fail;
            }

            /* Wait for operations on the image to be complete */
            err = NvSciSyncFenceWait(&appBuffer->preFence, cpuWaitContext, 1000*1000);
            if(err != NvSciError_Success) {
                LOG_ERR("NvSciSyncFenceWait failed: %u\n", err);
                goto fail;
            }

            /* Clear pre-fence */
            NvSciSyncFenceClear(&appBuffer->preFence);
            status = NvMediaIJPEBitsAvailable(ijpeCtx,
                                             &bytesAvailable,
                                             NVMEDIA_ENCODE_BLOCKING_TYPE_IF_PENDING,
                                             1000U); // 1sec
            switch(status) {
                case NVMEDIA_STATUS_OK:
                    // Encode Time
                    GetTimeMicroSec(&endTime1);
                    encodeTime += (double)(endTime1 - startTime) / 1000.0;

                    buffer = malloc(bytesAvailable);
                    if(!buffer) {
                        LOG_ERR("main: Error allocating %d bytes\n", bytesAvailable);
                        goto fail;
                    }
                    status = NvMediaIJPEGetBits(ijpeCtx, &bytes, buffer, 0);
                    if(status != NVMEDIA_STATUS_OK && status != NVMEDIA_STATUS_NONE_PENDING) {
                        LOG_ERR("main: Error getting encoded bits\n");
                        goto fail;
                    }

                    if(bytes != bytesAvailable) {
                        LOG_ERR("main: byte counts do not match %d vs. %d\n", bytesAvailable, bytes);
                        goto fail;
                    }

                    GetTimeMicroSec(&endTime2);
                    getbitsTime += (double)(endTime2 - endTime1) / 1000.0;

                    LOG_DBG("main: Opening output file\n");
                    sprintf(outFileName, args.outfile, frameCounter);
                    outputFile = fopen(outFileName, "w+");
                    if(!outputFile) {
                        LOG_ERR("main: Failed opening '%s' file for writing\n", args.outfile);
                        goto fail;
                    }

                    if(fwrite(buffer, bytesAvailable, 1, outputFile) != 1) {
                       LOG_ERR("main: Error writing %d bytes\n", bytesAvailable);
                       fclose(outputFile);
                       goto fail;
                    }
                    fclose(outputFile);

                    if(args.crcoption.crcGenMode){
                        //calculate CRC from buffer 'buffer'
                        calcCrc = 0;
                        calcCrc = CalculateBufferCRC(bytesAvailable, calcCrc, buffer);
                        if(!fprintf(crcFile, "%08x\n",calcCrc)) {
                            LOG_ERR("main: Failed writing calculated CRC to file %s\n", crcFile);
                            goto fail;
                        }
                    } else if(args.crcoption.crcCheckMode){
                        //calculate CRC from buffer 'buffer'
                        uint32_t refCrc;
                        calcCrc = 0;
                        calcCrc = CalculateBufferCRC(bytesAvailable, calcCrc, buffer);
                        if(fscanf(crcFile, "%8x\n", &refCrc) == 1) {
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

                    free(buffer);
                    buffer = NULL;

                    //Tracking the bitrate
                    totalBytes += bytesAvailable;

                    encodeDoneFlag = 1;
                    break;
                case NVMEDIA_STATUS_PENDING:
                    LOG_DBG("main: Status - pending\n");
                    break;
                case NVMEDIA_STATUS_NONE_PENDING:
                    LOG_ERR("main: No encoded data is pending\n");
                    goto fail;
                default:
                    LOG_ERR("main: Error occured\n");
                    goto fail;
            }
        }

        // Next frame
        frameCounter++;

        if(frameCounter == framesNum) {
            nextFrameFlag = false;
        }
    }

    //get encoding time info
    LOG_MSG("\nTotal Encoding time for %d frames: %.3f ms\n", frameCounter, encodeTime + getbitsTime);
    LOG_MSG("Encoding time per frame %.4f ms \n", encodeTime / frameCounter);
    LOG_MSG("Get bits time per frame %.4f ms \n", getbitsTime / frameCounter);
    //Get the bitrate info
    LOG_MSG("\nTotal encoded frames = %d, avg. bitrate=%d\n",
            frameCounter, (int)(totalBytes*8*30/frameCounter));
    if (args.crcoption.crcGenMode){
        LOG_MSG("\n***crc gold file %s has been generated***\n", args.crcoption.crcFilename);
    } else if (args.crcoption.crcCheckMode){
        LOG_MSG("\n***crc checking with file %s is successful\n", args.crcoption.crcFilename);
    }
    LOG_MSG("\n***ENCODING PROCESS ENDED SUCCESSFULY***\n");
    testPass = true;

fail:
    NvSciBufAttrListFree(bufReconciledList);
    NvSciSyncFenceClear(&appBuffer->preFence);

    if (eofSyncObj) {
        /* Unregister NvSciSyncObj */
        status = NvMediaIJPEUnregisterNvSciSyncObj(ijpeCtx, eofSyncObj);
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("%s: Failed to unregister NvSciSyncObj\n", __func__);
        }

        /* Free NvSciSyncObj */
        NvSciSyncObjFree(eofSyncObj);
    }

    /* Free NvSciBufObj */
    if(appBuffer->bufObj) {
        NvSciBufObjFree(appBuffer->bufObj);
    }

    if (ijpeCtx) {
        NvMediaIJPEDestroy(ijpeCtx);
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

    if (buffer)
        free(buffer);

    if (args.huffTable) {
        if (attr.lumaDC) {
            if (attr.lumaDC->values)
                free(attr.lumaDC->values);
            free(attr.lumaDC);
        }

        if (attr.chromaDC) {
            if (attr.chromaDC->values)
                free(attr.chromaDC->values);
            free(attr.chromaDC);
        }

        if (attr.lumaAC) {
            if (attr.lumaAC->values)
                free(attr.lumaAC->values);
            free(attr.lumaAC);
        }
        if (attr.chromaAC) {
            if (attr.chromaAC->values)
                free(attr.chromaAC->values);
            free(attr.chromaAC);
        }
    }

    if (testPass) {
        return 0;
    } else {
        return -1;
    }
}
