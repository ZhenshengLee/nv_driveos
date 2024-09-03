/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>

#include "nvscisync.h"
#include "nvscibuf.h"
#include "nvscierror.h"
#include "nvmedia_iep.h"
#if !NV_IS_SAFETY
#include "nvmedia_iep_input_extradata.h"
#include "nvmedia_iep_output_extradata.h"
#endif

#include "scibuf_utils.h"
#include "cmdline.h"
#include "config_parser.h"
#include "log_utils.h"
#include "misc_utils.h"
#include "thread_utils.h"
#if ENABLE_PROFILING
#include "nvplayfair.h"
#endif
#if __QNXNTO__
#include "nvdvms_client.h"
#endif

#define ALIGN_8(_x)   (((_x) + 7)   & (~7))
#define ALIGN_16(_x)  (((_x) + 15)  & (~15))
#define ALIGN_32(_x)  (((_x) + 31)  & (~31))
#define ALIGN_64(_x)  (((_x) + 63)  & (~63))
#define ALIGN_256(_x) (((_x) + 255) & (~255))

/* NvMediaIEP only supports input surface formats which have 2 planes */
#define IEP_APP_MAX_INPUT_PLANE_COUNT 2U
#define IEP_APP_BASE_ADDR_ALIGN 256U

#define NVMIMGENC_TEST_RESULT(condition, code, message1, message2)          \
    LOG_MSG("[TestResult] suite=\"iep_negative\" case=\"");                 \
    LOG_MSG(message1);                                                      \
    LOG_MSG("\" specs=\"");                                                 \
    LOG_MSG(message2);                                                      \
    if( !(condition)) {                                                     \
        LOG_MSG("\" result=\"failed\" \n");                                 \
        code ;                                                              \
    }                                                                       \
    else{                                                                   \
        LOG_MSG("\" result=\"passed\" \n");         }

typedef enum{
    CODEC_H264   = 0,
    CODEC_H265   = 1,
    CODEC_VP9    = 2,
    CODEC_AV1    = 3,
    CODEC_UKNOWN = 4
} Codec;

typedef enum{
    GOP_PATTERN_I    = 0,
    GOP_PATTERN_IPP  = 1,
    GOP_PATTERN_IBP  = 2,
    GOP_PATTERN_IBBP = 3
} GOPPattern;

typedef struct {
    uint8_t * dataBuff;
    int dataSize;
} EventDataRecorderBuffData;

typedef struct {
    uint8_t encOutThreadExit;
    uint8_t encError;
    uint8_t errorOccurred;
    uint32_t frameCounter;
    uint32_t framesNum;
    uint32_t minBitrate;
    uint32_t maxBitrate;
    int32_t eventDataRecorderRingSize;
    int32_t  startIdx;
    int32_t endIdx;
    long long totalBytes;
    TestArgs *args;
    NvMediaIEP *encoderCtx;
    FILE *outputFile;
#if !NV_IS_SAFETY
    uint8_t enableSliceEncode;
#endif
#if ENABLE_PROFILING
    NvpPerfData_t latencies;
#endif
    NvSemaphore *feedFrameDoneSema;
    NvSemaphore *outBufAvailableSema;
    EventDataRecorderBuffData *eventDataRecorderDataRing;
} EncOutThreadArgs;

/* Data structure that ties the buffer and synchronization primitive together */
typedef struct {
    NvSciSyncFence  preFence;
    NvSciSyncFence  eofFence;
    NvSciBufObj     bufObj;
    NvQueue         *queue;
} NvMediaAppBuffer;

#if !NV_IS_SAFETY
typedef struct {
    signed char  *qp_delta_buffer[IMAGE_BUFFERS_POOL_SIZE];
} RingEntry;
#endif

static bool encodeStop = 0;


NvMediaStatus allocAndRegisterBuffers(
  TestArgs *args,
  NvSciBufModule bufModule,
  uint16_t bufferWidth,
  uint16_t bufferHeight,
  ChromaFormat surfaceChromaFormat,
  NvSciBufAttrList *bufAttributeList,
  NvSciBufAttrList *bufConflictList,
  NvSciBufAttrList *bufReconciledList,
  NvQueue **inputQueue,
  uint32_t imagePoolSize);

NvMediaStatus deregisterAndFreeBuffers(
  TestArgs *args,
  NvMediaIEP *encoderCtx,
  NvQueue **inputQueue,
  uint8_t countNvSciBufObj,
  NvSciSyncCpuWaitContext *cpuWaitContext
);

static long
GetInputFileSizeBytes(char *inputFileName);

static uint32_t
GetInputFrameSizeBytes(
        ChromaFormat inputChromaFormat,
        uint32_t width,
        uint32_t height);

static NvMediaStatus
GetInputFormatAttributes(
        uint32_t               inputFileFormat,
        ChromaFormat           *inputChromaFormat,
        bool                   *uvOrdering,
        PixelAlignment         *alignment,
        NvSciBufSurfSampleType *subsampleType
);

static NvMediaStatus
MapInputFileChromaToSurfaceChroma (
    ChromaFormat        inputFileChromaFormat,
    ChromaFormat        *surfaceChromaFormat,
    NvSciBufSurfBPC     *bitdepth
);

#if !NV_IS_SAFETY
static uint32_t getH265QPDeltaBufferSize(uint32_t width, uint32_t height) {
    uint32_t log2_ctbsize = 5;
    uint32_t ctbsize = 1 << log2_ctbsize;
    uint32_t maxwidth = ALIGN_32(width);
    uint32_t maxheight = ALIGN_32(height);
    uint32_t frameWidthInCtb = (maxwidth + ctbsize - 1) >> log2_ctbsize;
    uint32_t frameHeightInCtb = (maxheight + ctbsize - 1) >> log2_ctbsize;
    uint32_t QPDeltaBufferWidth = ALIGN_256(frameWidthInCtb);
    uint32_t QPDeltaBufferHeight = frameHeightInCtb + 2;
    return QPDeltaBufferWidth * QPDeltaBufferHeight;
}
#endif

static void AddIVFPrefix(TestArgs *args, FILE *outputFile, uint32_t frameSize, uint32_t frameNum, uint32_t totalFrameNum) {
    uint8_t ivfPrefix[32] = {0};
    uint8_t buffer[12] = {0};
    uint32_t framesToBeEncoded = args->framesToBeEncoded;

    if(framesToBeEncoded == 0) {
        framesToBeEncoded = totalFrameNum;
    }

    if(frameNum == 0) {
        //IVF file prefix

        memcpy(ivfPrefix, "DKIF", 4);

        *(ivfPrefix + 4) = 0;
        *(ivfPrefix + 5) = 0;
        *(ivfPrefix + 6) = 32;
        *(ivfPrefix + 7) = 0;

        if(args->videoCodec == CODEC_VP9) {
            memcpy(ivfPrefix + 8, "VP90", 4);
        }
        else {
            memcpy(ivfPrefix + 8, "AV01", 4);
        }

        *(ivfPrefix + 12) = (args->configParams.encodeWidth & 0xFF);
        *(ivfPrefix + 13) = (args->configParams.encodeWidth >> 8) & 0xFF;
        *(ivfPrefix + 14) = (args->configParams.encodeHeight & 0xFF);
        *(ivfPrefix + 15) = (args->configParams.encodeHeight >> 8) & 0xFF;

        *(ivfPrefix + 16) = (args->configParams.frameRateNum & 0xFF);    // time base den
        *(ivfPrefix + 17) = (args->configParams.frameRateNum>>8) & 0xFF;
        *(ivfPrefix + 18) = (args->configParams.frameRateNum>>16) & 0xFF;
        *(ivfPrefix + 19) = (args->configParams.frameRateNum>>24);

        *(ivfPrefix + 20) = (args->configParams.frameRateDen & 0xFF);    // time base num
        *(ivfPrefix + 21) = (args->configParams.frameRateDen>>8) & 0xFF;
        *(ivfPrefix + 22) = (args->configParams.frameRateDen>>16) & 0xFF;
        *(ivfPrefix + 23) = (args->configParams.frameRateDen>>24);

        *(ivfPrefix + 24) = (framesToBeEncoded & 0xFF);
        *(ivfPrefix + 25) = (framesToBeEncoded>>8) & 0xFF;
        *(ivfPrefix + 26) = (framesToBeEncoded>>16) & 0xFF;
        *(ivfPrefix + 27) = (framesToBeEncoded>>24);

        *(ivfPrefix + 28) = 0;
        *(ivfPrefix + 29) = 0;
        *(ivfPrefix + 30) = 0;
        *(ivfPrefix + 31) = 0;

        fwrite(ivfPrefix, 32, 1, outputFile);
    }

    *(buffer + 0) = (frameSize & 0xFF);
    *(buffer + 1) = (frameSize>>8) & 0xFF;
    *(buffer + 2) = (frameSize>>16) & 0xFF;
    *(buffer + 3) = (frameSize>>24);

    *(buffer + 4) = (frameNum & 0xFF);;
    *(buffer + 5) = (frameNum>>8) & 0xFF;
    *(buffer + 6) = (frameNum>>16) & 0xFF;
    *(buffer + 7) = (frameNum>>24);

    *(buffer + 8)  = 0;
    *(buffer + 9)  = 0;
    *(buffer + 10) = 0;
    *(buffer + 11) = 0;

    fwrite(buffer, 12, 1, outputFile);
}

static void SetEncoderInitParamsH264(NvMediaEncodeInitializeParamsH264 *params, TestArgs *args)
{
    params->encodeHeight          = args->configParams.encodeHeight;
    params->encodeWidth           = args->configParams.encodeWidth;
    params->frameRateDen          = args->configParams.frameRateDen;
    params->frameRateNum          = args->configParams.frameRateNum;
    params->profile               = args->configParams.profile;
    params->level                 = args->configParams.level;
    params->maxNumRefFrames       = args->configParams.maxNumRefFrames;
    params->enableExternalMEHints = false; //Not support yet
    params->enableROIEncode = args->configParams.enableROIEncode;
    params->useBFramesAsRef       = args->configParams.useBFramesAsRef;
    params->enableAllIFrames      = args->configParams.enableAllIFrames;
    params->enableMemoryOptimization = args->configParams.enableMemoryOptimization;
}

static void SetEncoderInitParamsH265(NvMediaEncodeInitializeParamsH265 *params, TestArgs *args)
{
    params->encodeHeight          = args->configParams.encodeHeight;
    params->encodeWidth           = args->configParams.encodeWidth;
    params->frameRateDen          = args->configParams.frameRateDen;
    params->frameRateNum          = args->configParams.frameRateNum;
    params->profile               = args->configParams.profile;
    params->level                 = args->configParams.level;
    params->maxNumRefFrames       = args->configParams.maxNumRefFrames;
    params->enableROIEncode = args->configParams.enableROIEncode;
#if !NV_IS_SAFETY
    params->enableSliceEncode     = (bool)(args->configH265Params.features & NVMEDIA_ENCODE_CONFIG_H265_ENABLE_SLICE_LEVEL_OUTPUT);
#endif
    params->ampDisable            = args->configParams.ampDisable;
}

static void SetEncoderInitParamsVP9(NvMediaEncodeInitializeParamsVP9 *params, TestArgs *args)
{
    params->encodeHeight          = args->configParams.encodeHeight;
    params->encodeWidth           = args->configParams.encodeWidth;
    params->frameRateDen          = args->configParams.frameRateDen;
    params->frameRateNum          = args->configParams.frameRateNum;
    params->maxNumRefFrames       = args->configParams.maxNumRefFrames;
    // VP9 auto parameters of Orin
    params->numEpCores        = args->configParams.numEpCores;
    params->log2TileRows      = args->configParams.log2NumTilesInRow;
    params->log2TileCols      = args->configParams.log2NumTilesInCol;
    params->vp9SkipChroma     = args->configParams.vp9SkipChroma;
}

static void SetEncoderInitParamsAV1(NvMediaEncodeInitializeParamsAV1 *params, TestArgs *args)
{
    params->encodeHeight          = args->configParams.encodeHeight;
    params->encodeWidth           = args->configParams.encodeWidth;
    params->frameRateDen          = args->configParams.frameRateDen;
    params->frameRateNum          = args->configParams.frameRateNum;
    params->maxNumRefFrames       = args->configParams.maxNumRefFrames;
    params->enableSsimRdo         = args->configParams.enableSsimRdo;
    params->enableTileEncode      = args->configParams.enableTileEncode;
    params->log2NumTilesInRow     = args->configParams.log2NumTilesInRow;
    params->log2NumTilesInCol     = args->configParams.log2NumTilesInCol;
    params->frameRestorationType  = args->configParams.frameRestorationType;
    params->enableBiCompound      = args->configParams.enableBiCompound;
    params->enableUniCompound     = args->configParams.enableUniCompound;
    params->enableInternalHighBitDepth = args->enableInternalHighBitDepth;
    params->profile               = args->configParams.profile;
    params->level                 = args->configParams.level;
}

static int SetEncodeConfigRCParam(NvMediaEncodeRCParams *rcParams, TestArgs *args, unsigned int rcSectionIndex)
{
    static unsigned int preRCIdex = (unsigned int)-1;

    if (preRCIdex == rcSectionIndex)
        return -1;
    else
       preRCIdex = rcSectionIndex;

    LOG_DBG("SetEncodeConfigRCParam: rc section index: %d, prev index: %d\n", rcSectionIndex, preRCIdex);

    rcParams->rateControlMode = args->rcParamsCollection[rcSectionIndex].rcMode;
    rcParams->numBFrames      = (args->configParams.gopPattern < GOP_PATTERN_IBP) ? 0 : (args->configParams.gopPattern - 1);

    switch(rcParams->rateControlMode)
    {
        case NVMEDIA_ENCODE_PARAMS_RC_CBR:
             rcParams->params.cbr.averageBitRate  = args->rcParamsCollection[rcSectionIndex].averageBitRate;
             rcParams->params.cbr.vbvBufferSize   = args->rcParamsCollection[rcSectionIndex].vbvBufferSize;
             rcParams->params.cbr.vbvInitialDelay = args->rcParamsCollection[rcSectionIndex].vbvInitialDelay;
             break;
        case NVMEDIA_ENCODE_PARAMS_RC_CONSTQP:
             memcpy(&rcParams->params.const_qp.constQP,
                    &args->quantizationParamsCollection[args->rcParamsCollection[rcSectionIndex].rcConstQPSectionNum - 1],
                    sizeof(NvMediaEncodeQP));
             break;
        case NVMEDIA_ENCODE_PARAMS_RC_VBR:
             rcParams->params.vbr.averageBitRate = args->rcParamsCollection[rcSectionIndex].averageBitRate;
             rcParams->params.vbr.maxBitRate     = args->rcParamsCollection[rcSectionIndex].maxBitRate;
             rcParams->params.vbr.vbvBufferSize  = args->rcParamsCollection[rcSectionIndex].vbvBufferSize;
             rcParams->params.vbr.vbvInitialDelay= args->rcParamsCollection[rcSectionIndex].vbvInitialDelay;
             break;
        case NVMEDIA_ENCODE_PARAMS_RC_VBR_MINQP:
             rcParams->params.vbr_minqp.averageBitRate  = args->rcParamsCollection[rcSectionIndex].averageBitRate;
             rcParams->params.vbr_minqp.maxBitRate      = args->rcParamsCollection[rcSectionIndex].maxBitRate;
             rcParams->params.vbr_minqp.vbvBufferSize   = args->rcParamsCollection[rcSectionIndex].vbvBufferSize;
             rcParams->params.vbr_minqp.vbvInitialDelay = args->rcParamsCollection[rcSectionIndex].vbvInitialDelay;
             if (args->rcParamsCollection[rcSectionIndex].enableMinQP)
             {
                memcpy(&rcParams->params.vbr_minqp.minQP,
                       &args->quantizationParamsCollection[args->rcParamsCollection[rcSectionIndex].rcMinQPSectionNum - 1],
                       sizeof(NvMediaEncodeQP));
             }
             break;
        case NVMEDIA_ENCODE_PARAMS_RC_CBR_MINQP:
             rcParams->params.cbr_minqp.averageBitRate  = args->rcParamsCollection[rcSectionIndex].averageBitRate;
             rcParams->params.cbr_minqp.vbvBufferSize   = args->rcParamsCollection[rcSectionIndex].vbvBufferSize;
             rcParams->params.cbr_minqp.vbvInitialDelay = args->rcParamsCollection[rcSectionIndex].vbvInitialDelay;
             if (args->rcParamsCollection[rcSectionIndex].enableMinQP)
             {
                memcpy(&rcParams->params.cbr_minqp.minQP,
                       &args->quantizationParamsCollection[args->rcParamsCollection[rcSectionIndex].rcMinQPSectionNum - 1],
                       sizeof(NvMediaEncodeQP));
             }
             break;
        default:
             return -1;
    }
    return 0;
}

static void SetEncodePicParamsH264(NvMediaEncodePicParamsH264 *picParams, TestArgs *args, int framesDecoded, int picParamsIndex)
{
    unsigned int h264ParamsIndex, rcSectionIndex, i;
    NvMediaEncodeH264SEIPayload *seiPayload = picParams->seiPayloadArray;

    h264ParamsIndex = args->picParamsCollection[picParamsIndex].PicParamsSectionNum - 1;
    rcSectionIndex  = args->picParamsCollection[picParamsIndex].rcParamsSectionNum - 1;

    picParams->pictureType = args->picParamsCollection[picParamsIndex].pictureType;
    picParams->encodePicFlags = args->picParamsCollection[picParamsIndex].encodePicFlags;

    if(!SetEncodeConfigRCParam(&picParams->rcParams, args, rcSectionIndex)) {
        picParams->encodePicFlags |= NVMEDIA_ENCODE_PIC_FLAG_RATECONTROL_CHANGE;
        LOG_DBG("SetEncodePicParams: PicParamsIndex =%d, RC changed\n", h264ParamsIndex);
    }

    //ME hint stuff

    //Later!!!

    //SEI payload
    picParams->seiPayloadArrayCnt = args->picH264ParamsCollection[h264ParamsIndex].payloadArraySize;

    for(i = 0; i < picParams->seiPayloadArrayCnt; i++) {
        seiPayload->payloadSize = args->payloadsCollection[args->picH264ParamsCollection[h264ParamsIndex].payloadArrayIndexes[i] - '1'].payloadSize;
        seiPayload->payloadType = args->payloadsCollection[args->picH264ParamsCollection[h264ParamsIndex].payloadArrayIndexes[i] - '1'].payloadType;
        seiPayload->payload     = args->payloadsCollection[args->picH264ParamsCollection[h264ParamsIndex].payloadArrayIndexes[i] - '1'].payload;
        LOG_DBG("SetEncodePicParams: Payload %d, size=%d, type=%d, payload=%x%x%x%x\n",
                i, seiPayload->payloadSize, seiPayload->payloadType, seiPayload->payload[0],
                seiPayload->payload[1],seiPayload->payload[2],seiPayload->payload[3]);
        seiPayload ++;
    }
}

static void SetEncodePicParamsH265(NvMediaEncodePicParamsH265 *picParams, TestArgs *args, int framesDecoded, int picParamsIndex)
{
    unsigned int h265ParamsIndex, rcSectionIndex, i;
    NvMediaEncodeH265SEIPayload *seiPayload = picParams->seiPayloadArray;

    h265ParamsIndex = args->picParamsCollection[picParamsIndex].PicParamsSectionNum - 1;
    rcSectionIndex  = args->picParamsCollection[picParamsIndex].rcParamsSectionNum - 1;

    picParams->pictureType = args->picParamsCollection[picParamsIndex].pictureType;
    picParams->encodePicFlags = args->picParamsCollection[picParamsIndex].encodePicFlags;

    if(!SetEncodeConfigRCParam(&picParams->rcParams, args, rcSectionIndex)) {
        picParams->encodePicFlags |= NVMEDIA_ENCODE_PIC_FLAG_RATECONTROL_CHANGE;
        LOG_DBG("SetEncodePicParams: PicParamsIndex =%d, RC changed\n", h265ParamsIndex);
    }

    //ME hint stuff

    //Later!!!

    //SEI payload
    picParams->seiPayloadArrayCnt = args->picH265ParamsCollection[h265ParamsIndex].payloadArraySize;

    for(i = 0; i < picParams->seiPayloadArrayCnt; i++) {
        seiPayload->payloadSize = args->payloadsCollection[args->picH265ParamsCollection[h265ParamsIndex].payloadArrayIndexes[i] - '1'].payloadSize;
        seiPayload->payloadType = args->payloadsCollection[args->picH265ParamsCollection[h265ParamsIndex].payloadArrayIndexes[i] - '1'].payloadType;
        seiPayload->payload     = args->payloadsCollection[args->picH265ParamsCollection[h265ParamsIndex].payloadArrayIndexes[i] - '1'].payload;
        LOG_DBG("SetEncodePicParams: Payload %d, size=%d, type=%d, payload=%x%x%x%x\n",
                i, seiPayload->payloadSize, seiPayload->payloadType, seiPayload->payload[0],
                seiPayload->payload[1],seiPayload->payload[2],seiPayload->payload[3]);
        seiPayload ++;
    }
}

/* Signal Handler for SIGINT */
static void sigintHandler(int sig_num)
{
    LOG_MSG("\n Exiting encode process \n");
    encodeStop = 1;
}

static void SetEncodePicParamsVP9(NvMediaEncodePicParamsVP9 *picParams, TestArgs *args, int framesDecoded, int picParamsIndex)
{
    unsigned int vp9ParamsIndex, rcSectionIndex;

    vp9ParamsIndex = args->picParamsCollection[picParamsIndex].PicParamsSectionNum - 1;
    rcSectionIndex  = args->picParamsCollection[picParamsIndex].rcParamsSectionNum - 1;

    picParams->pictureType = args->picParamsCollection[picParamsIndex].pictureType;
    picParams->encodePicFlags = args->picParamsCollection[picParamsIndex].encodePicFlags;

    if(!SetEncodeConfigRCParam(&picParams->rcParams, args, rcSectionIndex)) {
        picParams->encodePicFlags |= NVMEDIA_ENCODE_PIC_FLAG_RATECONTROL_CHANGE;
        LOG_DBG("SetEncodePicParams: PicParamsIndex =%d, RC changed\n", vp9ParamsIndex);
    }

    picParams->nextBFrames = 0;
}

static void SetEncodePicParamsAV1(NvMediaEncodePicParamsAV1 *picParams, TestArgs *args, int framesDecoded, int picParamsIndex)
{
    unsigned int av1ParamsIndex, rcSectionIndex;

    av1ParamsIndex = args->picParamsCollection[picParamsIndex].PicParamsSectionNum - 1;
    rcSectionIndex  = args->picParamsCollection[picParamsIndex].rcParamsSectionNum - 1;

    picParams->pictureType = args->picParamsCollection[picParamsIndex].pictureType;
    picParams->encodePicFlags = args->picParamsCollection[picParamsIndex].encodePicFlags;

    if(!SetEncodeConfigRCParam(&picParams->rcParams, args, rcSectionIndex)) {
        picParams->encodePicFlags |= NVMEDIA_ENCODE_PIC_FLAG_RATECONTROL_CHANGE;
        LOG_DBG("SetEncodePicParams: PicParamsIndex =%d, RC changed\n", av1ParamsIndex);
    }

    picParams->nextBFrames = 0;
}

static NvMediaStatus
CheckVersion(TestArgs *args)
{
    NvMediaVersion version;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    memset(&version, 0, sizeof(NvMediaVersion));
    status = NvMediaIEPGetVersion(&version);
    if (status != NVMEDIA_STATUS_OK)
        return status;

    if((version.major != NVMEDIA_IEP_VERSION_MAJOR) ||
       (version.minor != NVMEDIA_IEP_VERSION_MINOR) ||
       (version.patch != NVMEDIA_IEP_VERSION_PATCH)) {
        LOG_ERR("%s: Incompatible IEP version found \n", __func__);
        LOG_ERR("%s: Client version: %d.%d.%d\n", __func__,
            NVMEDIA_IEP_VERSION_MAJOR, NVMEDIA_IEP_VERSION_MINOR,
            NVMEDIA_IEP_VERSION_PATCH);
        LOG_ERR("%s: Core version: %d.%d.%d\n", __func__,
            version.major, version.minor, version.patch);
        return NVMEDIA_STATUS_INCOMPATIBLE_VERSION;
    }
    if (args->version)
    {
        LOG_MSG("NvMediaIEP Major version- %d \n", version.major);
        LOG_MSG("NvMediaIEP Minor version- %d \n", version.minor);
    }

    return status;
}

#if !NV_IS_SAFETY
static void WriteExtradata(
        FILE *extradataFile,
        uint32_t frameNum,
        int videoCodec,
        NvMediaEncodeOutputExtradata *extradata)
{
    fprintf(extradataFile, "Frame: %u\n",
            frameNum);
    fprintf(extradataFile, "ulExtraDataSize: %u\n",
            extradata->ulExtraDataSize);
    fprintf(extradataFile, "ulHdrSize: %u\n",
            extradata->ulHdrSize);
    fprintf(extradataFile, "AvgQP: %d\n",
            extradata->AvgQP);
    fprintf(extradataFile, "ulFrameMinQP: %u\n",
            extradata->ulFrameMinQP);
    fprintf(extradataFile, "ulFrameMaxQP: %u\n",
            extradata->ulFrameMaxQP);
    fprintf(extradataFile, "bMVbufferdump: %d\n",
            extradata->bMVbufferdump);
    fprintf(extradataFile, "MVBufferDumpSize: %u\n",
            extradata->MVBufferDumpSize);
    fprintf(extradataFile, "MVBufferDumpStartOffset: %u\n",
            extradata->MVBufferDumpStartOffset);
    fprintf(extradataFile, "Codec: %d\n",
            extradata->codec);

    switch(extradata->codec) {
    case NVMEDIA_VIDEO_CODEC_H264:
        fprintf(extradataFile, "codecExData.h264Extradata.eFrameType: %u\n",
                extradata->codecExData.h264Extradata.eFrameType);
        fprintf(extradataFile, "codecExData.h264Extradata.bRefPic: %u\n",
                extradata->codecExData.h264Extradata.bRefPic);
        fprintf(extradataFile, "codecExData.h264Extradata.bIntraRefresh: %u\n",
                extradata->codecExData.h264Extradata.bIntraRefresh);
        fprintf(extradataFile, "codecExData.h264Extradata.uIntraMBCount: %u\n",
                extradata->codecExData.h264Extradata.uIntraMBCount);
        fprintf(extradataFile, "codecExData.h264Extradata.uInterMBCount: %u\n",
                extradata->codecExData.h264Extradata.uInterMBCount);
        break;
    case NVMEDIA_VIDEO_CODEC_HEVC:
        fprintf(extradataFile, "codecExData.h265Extradata.eFrameType: %u\n",
                extradata->codecExData.h265Extradata.eFrameType);
        fprintf(extradataFile, "codecExData.h265Extradata.bRefPic: %u\n",
                extradata->codecExData.h265Extradata.bRefPic);
        fprintf(extradataFile, "codecExData.h265Extradata.bIntraRefresh: %u\n",
                extradata->codecExData.h265Extradata.bIntraRefresh);
        fprintf(extradataFile, "codecExData.h265Extradata.uIntraCU32x32Count: %u\n",
                extradata->codecExData.h265Extradata.uIntraCU32x32Count);
        fprintf(extradataFile, "codecExData.h265Extradata.uInterCU32x32Count: %u\n",
                extradata->codecExData.h265Extradata.uInterCU32x32Count);
        fprintf(extradataFile, "codecExData.h265Extradata.uIntraCU16x16Count: %u\n",
                extradata->codecExData.h265Extradata.uIntraCU16x16Count);
        fprintf(extradataFile, "codecExData.h265Extradata.uInterCU16x16Count: %u\n",
                extradata->codecExData.h265Extradata.uInterCU16x16Count);
        fprintf(extradataFile, "codecExData.h265Extradata.uIntraCU8x8Count: %u\n",
                extradata->codecExData.h265Extradata.uIntraCU8x8Count);
        fprintf(extradataFile, "codecExData.h265Extradata.uInterCU8x8Count: %u\n",
                extradata->codecExData.h265Extradata.uInterCU8x8Count);
        break;
    default:
        break;
    }

    fprintf(extradataFile, "\n");
    return;
}
#endif

static uint32_t EncoderOutputThread(void * threadArgs)
{
    FILE *crcFile = NULL;
    bool encodeDoneFlag;
    uint32_t frameCounter = 0, bytesAvailable = 0, bytes = 0, calcCrc = 0, framesNum = 0;
    uint8_t *buffer = NULL;
    unsigned int localBitrate = 0;
    NvMediaStatus status;
    EncOutThreadArgs *pOutputThreadArgs = (EncOutThreadArgs *)threadArgs;
    NvMediaIEP *encoderCtx = pOutputThreadArgs->encoderCtx;
    TestArgs *args = pOutputThreadArgs->args;
    frameCounter = pOutputThreadArgs->frameCounter;
    framesNum = pOutputThreadArgs->framesNum;
    EventDataRecorderBuffData *eventDataRecorderDataRing = pOutputThreadArgs->eventDataRecorderDataRing;
    int32_t eventDataRecorderRingSize = pOutputThreadArgs->eventDataRecorderRingSize;
    uint32_t bitstreamSize = 0U;
#if !NV_IS_SAFETY
    uint32_t widthMB = 0U;
    uint32_t heightMB = 0U;
    uint32_t numMacroBlocks = 0U;
    uint32_t mvBufferSize = 0U;
    FILE *mvDataFile = NULL;
    FILE *extradataFile = NULL;
    FILE *fslFile = NULL;
    bool enableMVBufferDump = false;
    NvMediaEncodeOutputExtradata extradata = {0};
    NvMediaEncodeOutputExtradata *extradataPtr = NULL;
    uint32_t sliceCounter = 0;
#else
    void *extradataPtr = NULL;
#endif

#if !NV_IS_SAFETY
    /* Check if MV Buffer Dump needs to be enabled */
    switch(args->videoCodec) {
    case CODEC_H264:
        widthMB = ALIGN_16(args->configParams.encodeWidth)/16U;
        heightMB = ALIGN_16(args->configParams.encodeHeight)/16U;
        numMacroBlocks = widthMB * heightMB;

        if (args->configH264Params.features &
                NVMEDIA_ENCODE_CONFIG_H264_ENABLE_MV_BUFFER_DUMP_V2) {
            enableMVBufferDump = true;
        }
        if (args->configH264Params.features &
                NVMEDIA_ENCODE_CONFIG_H264_ENABLE_MV_BUFFER_DUMP) {
            LOG_ERR("EncoderOutputThread: Unsupported feature in test app: MV_BUFFER_DUMP V1\n");
            pOutputThreadArgs->errorOccurred = 1;
            goto done;
        }
        break;
    case CODEC_H265:
        widthMB = ALIGN_32(args->configParams.encodeWidth)/32U;
        heightMB = ALIGN_32(args->configParams.encodeHeight)/32U;
        numMacroBlocks = widthMB * heightMB;

        if (args->configH265Params.features &
                NVMEDIA_ENCODE_CONFIG_H265_ENABLE_MV_BUFFER_DUMP_V2) {
            enableMVBufferDump = true;
        }
        if (args->configH264Params.features &
                NVMEDIA_ENCODE_CONFIG_H265_ENABLE_MV_BUFFER_DUMP) {
            LOG_ERR("EncoderOutputThread: Unsupported feature in test app: MV_BUFFER_DUMP V1\n");
            pOutputThreadArgs->errorOccurred = 1;
            goto done;
        }
        break;
    default:
        enableMVBufferDump = false;
    }

    if (enableMVBufferDump) {
        if (0U != strlen(args->mvDataFileName)) {
            mvDataFile = fopen(args->mvDataFileName, "wb+");
            if(!mvDataFile){
                LOG_ERR("EncoderOutputThread: Cannot open MV Data File [%s] for writing\n",
                        args->mvDataFileName);
                pOutputThreadArgs->errorOccurred = 1;
                goto done;
            }
        }
    }

    if (args->enableExtradata) {
        if (0U != strlen(args->extradataFileName)) {
            extradataFile = fopen(args->extradataFileName, "w+");
            if(!extradataFile){
                LOG_ERR("EncoderOutputThread: Cannot open Extra Data File [%s] for writing\n",
                        args->extradataFileName);
                pOutputThreadArgs->errorOccurred = 1;
                goto done;
            }
        }
    }

    if (args->dumpFrameSizeLog) {
        if (0U != strlen(args->fslFileName)) {
            fslFile = fopen(args->fslFileName, "w+");
            if(!fslFile){
                LOG_ERR("EncoderOutputThread: Cannot open Frame size log File [%s] for writing\n",
                        args->fslFileName);
                pOutputThreadArgs->errorOccurred = 1;
                goto done;
            }
        }
    }
#endif

    if(args->crcoption.crcGenMode){
        crcFile = fopen(args->crcoption.crcFilename, "wt");
        if(!crcFile){
            LOG_ERR("%s: Cannot open crc gen file for writing\n", __func__);
            pOutputThreadArgs->errorOccurred = 1;
            goto done;
        }
    } else if(args->crcoption.crcCheckMode){
        crcFile = fopen(args->crcoption.crcFilename, "rb");
        if(!crcFile){
            LOG_ERR("%s: Cannot open crc gen file for reading\n", __func__);
            pOutputThreadArgs->errorOccurred = 1;
            goto done;
        }
    }

    while (1) {
        encodeDoneFlag = false;
        NvSemaphoreDecrement(pOutputThreadArgs->feedFrameDoneSema, NV_TIMEOUT_INFINITE);
        if (pOutputThreadArgs->encError) {
            pOutputThreadArgs->errorOccurred = 1;
            goto done;
        }
        while(!encodeDoneFlag) {
#if !NV_IS_SAFETY
            if(pOutputThreadArgs->enableSliceEncode){
                bool sliceEncodeDone = false;
                uint32_t frameBytes = 0;
                while(!sliceEncodeDone){ // inner loop for slice-level output
                    NvMediaBitstreamBuffer bitstreams = {0};
                    bytesAvailable = 0;
                    bytes = 0;

                    status = NvMediaIEPBitsAvailable(encoderCtx,
                            &bytesAvailable,
                            NVMEDIA_ENCODE_BLOCKING_TYPE_IF_PENDING,
                            NVMEDIA_ENCODE_TIMEOUT_INFINITE);

                    switch(status) {
                        case NVMEDIA_STATUS_OK:
                            bitstreamSize = bytesAvailable;
                            extradataPtr = &extradata;
                            memset(extradataPtr, 0x0, sizeof(NvMediaEncodeOutputExtradata));
                            extradataPtr->ulExtraDataSize = sizeof(NvMediaEncodeOutputExtradata);

                            buffer = malloc(bitstreamSize);
                            if(!buffer) {
                                LOG_ERR("EncoderOutputThread: Error allocating %d bytes\n", bitstreamSize);
                                pOutputThreadArgs->errorOccurred = 1;
                                goto done;
                            }
                            bitstreams.bitstream = buffer;
                            bitstreams.bitstreamSize = bitstreamSize;
                            memset(buffer, 0xE5, bitstreamSize);

                            status = NvMediaIEPGetBits(encoderCtx, &bytes, 1, &bitstreams, extradataPtr);
                                if(status != NVMEDIA_STATUS_OK && status != NVMEDIA_STATUS_NONE_PENDING) {
                                    LOG_ERR("EncoderOutputThread: Error getting encoded bits\n");
                                    pOutputThreadArgs->errorOccurred = 1;
                                    goto done;
                            }
                            if(fwrite(buffer, bytes, 1, pOutputThreadArgs->outputFile) != 1) {
                                LOG_ERR("EncoderOutputThread: Error writing %d bytes\n", bytesAvailable);
                                pOutputThreadArgs->errorOccurred = 1;
                                goto done;
                            }
                            sliceCounter += 1;
                            //Tracking the bitrate
                            frameBytes += bytes;

                            if(buffer) {
                                free(buffer);
                                buffer = NULL;
                            }

                            if(extradataPtr->bEndOfFrame){
                                sliceEncodeDone = true;
                            }
                            break;

                        case NVMEDIA_STATUS_PENDING:
                            LOG_DBG("EncoderOutputThread: Status - pending\n");
                            break;
                        case NVMEDIA_STATUS_NONE_PENDING:
                            LOG_DBG("EncoderOutputThread: No encoded data is pending\n");
                            goto done;
                        default:
                            LOG_ERR("EncoderOutputThread: Error occured\n");
                            pOutputThreadArgs->errorOccurred = 1;
                            goto done;
                    }
                }
                //Tracking the bitrate
                pOutputThreadArgs->totalBytes += frameBytes;
                if (frameCounter<=(30+args->startFrame))
                {
                    localBitrate += frameBytes;
                    if (frameCounter == (30+args->startFrame))
                    {
                        pOutputThreadArgs->maxBitrate = pOutputThreadArgs->minBitrate = localBitrate;
                    }
                } else {
                    localBitrate = (localBitrate*29/30 + frameBytes);
                    if (localBitrate > pOutputThreadArgs->maxBitrate)
                        pOutputThreadArgs->maxBitrate = localBitrate;
                    if (localBitrate < pOutputThreadArgs->minBitrate)
                        pOutputThreadArgs->minBitrate = localBitrate;
                }

                encodeDoneFlag = 1;
                frameCounter++;
                NvSemaphoreIncrement(pOutputThreadArgs->outBufAvailableSema);
                continue;
            }
#endif

            NvMediaBitstreamBuffer bitstreams = {0};
            bytesAvailable = 0;
            bytes = 0;
#if ENABLE_PROFILING
            uint64_t startBitsAvailableTimeMark = 0;
            uint64_t endBitsAvailableTimeMark = 0;
            if (args->profileEnable)
            {
                startBitsAvailableTimeMark = NvpGetTimeMark();
            }
#endif
            status = NvMediaIEPBitsAvailable(encoderCtx,
                                        &bytesAvailable,
                                        NVMEDIA_ENCODE_BLOCKING_TYPE_IF_PENDING,
                                        NVMEDIA_ENCODE_TIMEOUT_INFINITE);
            switch(status) {
                case NVMEDIA_STATUS_OK:
                bitstreamSize = bytesAvailable;

#if !NV_IS_SAFETY
                    /* If extradata needs to be enabled */
                    if (args->enableExtradata) {
                        extradataPtr = &extradata;
                        memset(extradataPtr, 0x0, sizeof(NvMediaEncodeOutputExtradata));

                        extradataPtr->ulExtraDataSize = sizeof(NvMediaEncodeOutputExtradata);
                    } else {
                        extradataPtr = NULL;
                    }

                    /* If MV Buffer Dump is enabled */
                    if (enableMVBufferDump) {
                        mvBufferSize = sizeof(NvMediaEncodeMVBufferHeader) +
                            ALIGN_256(numMacroBlocks * sizeof(NvMediaEncodeMVData));
                        bitstreamSize = ALIGN_8(bitstreamSize) + mvBufferSize;
                    }
                    // Dump frame size in bytes
                    if(args->dumpFrameSizeLog && (NULL != fslFile)){
                        if(args->dumpFslLevel == 0){
                            fprintf(fslFile, "%d\n", bytesAvailable);
                        } else if(args->dumpFslLevel == 1){
                            if(((frameCounter - 1) % args->configParams.gopLength) == 0)
                                fprintf(fslFile, "%d\n", bytesAvailable);
                        } else {
                            LOG_ERR("Frame Size Log level (-fsl ) set to invalid value \n");
                        }
                    }

#endif

                    buffer = malloc(bitstreamSize);
                    if(!buffer) {
                        LOG_ERR("EncoderOutputThread: Error allocating %d bytes\n", bitstreamSize);
                        pOutputThreadArgs->errorOccurred = 1;
                        goto done;
                    }
                    bitstreams.bitstream = buffer;
                    bitstreams.bitstreamSize = bitstreamSize;
                    memset(buffer, 0xE5, bitstreamSize);

                    status = NvMediaIEPGetBits(encoderCtx, &bytes, 1, &bitstreams, extradataPtr);
                        if(status != NVMEDIA_STATUS_OK && status != NVMEDIA_STATUS_NONE_PENDING) {
                            LOG_ERR("EncoderOutputThread: Error getting encoded bits\n");
                            pOutputThreadArgs->errorOccurred = 1;
                            goto done;
                    }
#if ENABLE_PROFILING
                    if (args->profileEnable)
                    {
                        endBitsAvailableTimeMark = NvpGetTimeMark();
                        NvpRecordSample(&pOutputThreadArgs->latencies, startBitsAvailableTimeMark, endBitsAvailableTimeMark);
                    }
#endif
#if !NV_IS_SAFETY
                    if (((args->videoCodec == CODEC_AV1) && ((args->configAV1Params.features & NVMEDIA_ENCODE_CONFIG_AV1_ENABLE_PROFILING) > 1)) ||
                        ((args->videoCodec == CODEC_VP9) && ((args->configVP9Params.features & NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_PROFILING) > 1)) ||
                        ((args->videoCodec == CODEC_H264) && ((args->configH264Params.features & NVMEDIA_ENCODE_CONFIG_H264_ENABLE_PROFILING) > 1)) ||
                        ((args->videoCodec == CODEC_H265) && ((args->configH265Params.features & NVMEDIA_ENCODE_CONFIG_H265_ENABLE_PROFILING) > 1)))
                    {
                        args->sumCycleCount += ((NvMediaEncodeOutputExtradata *)extradataPtr)->FrameStats.ulCycleCount;
                    }
#endif
                    /* Bytes returned will not include the size of the MV Buffer dump */
                    if(bytes != bytesAvailable) {
                        LOG_ERR("EncoderOutputThread: byte counts do not match %d vs. %d\n", bytesAvailable, bytes);
                        pOutputThreadArgs->errorOccurred = 1;
                        goto done;
                    }

                    if(args->crcoption.crcGenMode){
                        //calculate CRC from buffer 'buffer'
                        calcCrc = 0;
                        calcCrc = CalculateBufferCRC(bitstreamSize, calcCrc, buffer);
                        if(!fprintf(crcFile, "%08x\n",calcCrc))
                            LOG_ERR("EncoderOutputThread: Failed writing calculated CRC to file %s\n", crcFile);
                    } else if(args->crcoption.crcCheckMode){
                        //calculate CRC from buffer 'buffer'
                        uint32_t refCrc;
                        calcCrc = 0;
                        calcCrc = CalculateBufferCRC(bitstreamSize, calcCrc, buffer);
                        if (fscanf(crcFile, "%8x\n", &refCrc) == 1) {
                            if(refCrc != calcCrc){
                                LOG_ERR("EncoderOutputThread: Frame %d crc 0x%x does not match with ref crc 0x%x\n",
                                    frameCounter, calcCrc, refCrc);
                                pOutputThreadArgs->errorOccurred = 1;
                                goto done;
                            }
                        } else {
                            LOG_ERR("EncoderOutputThread: Failed checking CRC. Failed reading file %s\n", crcFile);
                        }
                    }

                    if((args->videoCodec == CODEC_VP9) || (args->videoCodec == CODEC_AV1)) {
                        AddIVFPrefix(args, pOutputThreadArgs->outputFile, bytesAvailable,
                        frameCounter-1, framesNum);
                    }

                    if (args->eventDataRecorderMode == true) {
                        int numFrames;
                        uint32_t i;

                        pOutputThreadArgs->endIdx ++;
                        if (pOutputThreadArgs->endIdx >= eventDataRecorderRingSize)
                            pOutputThreadArgs->endIdx = 0;
                        eventDataRecorderDataRing[pOutputThreadArgs->endIdx].dataBuff = buffer;
                        eventDataRecorderDataRing[pOutputThreadArgs->endIdx].dataSize = bytesAvailable;

                        if (pOutputThreadArgs->startIdx == -1)
                            pOutputThreadArgs->startIdx = 0;
                        numFrames = pOutputThreadArgs->endIdx - pOutputThreadArgs->startIdx + 1;
                        if (numFrames <= 0)
                            numFrames += eventDataRecorderRingSize;

                        if (numFrames == eventDataRecorderRingSize) {
                            //Release the first second frames
                            for (i=0; i<args->configH264Params.gopLength; i++) {
                                free (eventDataRecorderDataRing[pOutputThreadArgs->startIdx].dataBuff);
                                eventDataRecorderDataRing[pOutputThreadArgs->startIdx].dataBuff = NULL;
                                eventDataRecorderDataRing[pOutputThreadArgs->startIdx].dataSize = 0;
                                pOutputThreadArgs->startIdx++;
                                if (pOutputThreadArgs->startIdx >= eventDataRecorderRingSize)
                                    pOutputThreadArgs->startIdx = 0;
                            }
                        }
                        buffer = NULL;
                    } else {
                        if (!args->preFetchBuffer)
                        {
                            if(fwrite(buffer, bytesAvailable, 1, pOutputThreadArgs->outputFile) != 1) {
                                LOG_ERR("EncoderOutputThread: Error writing %d bytes\n", bytesAvailable);
                                pOutputThreadArgs->errorOccurred = 1;
                                goto done;
                            }
                        }
#if !NV_IS_SAFETY
                        /* Write MV Buffer Output if enabled */
                        if ((enableMVBufferDump) && (NULL != extradataPtr)) {
                            if ((0U == extradataPtr->MVBufferDumpSize) ||
                                    (0U == extradataPtr->MVBufferDumpStartOffset) ||
                                    (false == extradataPtr->bMVbufferdump)) {
                                LOG_ERR("EncoderOutputThread: MV Buffer data not written\n");
                                pOutputThreadArgs->errorOccurred = 1;
                                goto done;
                            }

                            /* Write MV Data to file */
                            if ((extradataPtr->bMVbufferdump) && (NULL != mvDataFile)) {
                                uint8_t *pSrcBuffer =
                                    buffer + extradataPtr->MVBufferDumpStartOffset;
                                NvMediaEncodeMVBufferHeader *pBufferHeader =
                                    (NvMediaEncodeMVBufferHeader*)pSrcBuffer;
                                if (pBufferHeader->MagicNum != MV_BUFFER_HEADER) {
                                    LOG_ERR("Error in dumping extradata\n");
                                    pOutputThreadArgs->errorOccurred = 1;
                                    goto done;
                                }

                                /* Write MV Buffer including Header */
                                if(1U != fwrite(pSrcBuffer,
                                            extradataPtr->MVBufferDumpSize, 1U, mvDataFile)) {
                                    LOG_ERR("EncoderOutputThread: Error writing %d bytes of MV Data\n",
                                            extradataPtr->MVBufferDumpSize);
                                    pOutputThreadArgs->errorOccurred = 1;
                                    goto done;
                                }
                            }
                        }
                        /* Write extradata to file */
                        if ((args->enableExtradata) && (NULL != extradataPtr) && (NULL != extradataFile)) {
                            WriteExtradata(extradataFile, frameCounter, args->videoCodec, extradataPtr);
                        }
#endif

                        if(buffer) {
                            free(buffer);
                            buffer = NULL;
                        }
                    }
                    //Tracking the bitrate
                    pOutputThreadArgs->totalBytes += bytesAvailable;
                    if (frameCounter<=(30+args->startFrame))
                        {
                        localBitrate += bytesAvailable;
                        if (frameCounter == (30+args->startFrame))
                        {
                            pOutputThreadArgs->maxBitrate = pOutputThreadArgs->minBitrate = localBitrate;
                        }
                    } else {
                        localBitrate = (localBitrate*29/30 + bytesAvailable);
                        if (localBitrate > pOutputThreadArgs->maxBitrate)
                            pOutputThreadArgs->maxBitrate = localBitrate;
                        if (localBitrate < pOutputThreadArgs->minBitrate)
                            pOutputThreadArgs->minBitrate = localBitrate;
                    }
                    encodeDoneFlag = 1;
                    frameCounter++;
                    NvSemaphoreIncrement(pOutputThreadArgs->outBufAvailableSema);
                    break;
                case NVMEDIA_STATUS_PENDING:
                    LOG_DBG("EncoderOutputThread: Status - pending\n");
                    break;
                case NVMEDIA_STATUS_NONE_PENDING:
                    LOG_DBG("EncoderOutputThread: No encoded data is pending\n");
                    goto done;
                default:
                    LOG_ERR("EncoderOutputThread: Error occured\n");
                    pOutputThreadArgs->errorOccurred = 1;
                    goto done;
            }
        }
    }
done:
    if(buffer) {
        free(buffer);
    }

#if !NV_IS_SAFETY
    if ((args->enableExtradata) && (NULL != extradataFile)) {
        fclose(extradataFile);
    }
    if ((args->dumpFrameSizeLog) && (NULL != fslFile)) {
        fclose(fslFile);
    }
    if ((enableMVBufferDump) && (NULL != mvDataFile)) {
        fclose(mvDataFile);
    }
#endif

    if(crcFile) {
        fclose(crcFile);
    }

#if !NV_IS_SAFETY
    if(pOutputThreadArgs->enableSliceEncode) {
        LOG_MSG("Slice count %u\n", sliceCounter);
    }
#endif
    // Unblock the EncoderOutputThread thread
    pOutputThreadArgs->encOutThreadExit = 1;
    NvSemaphoreIncrement(pOutputThreadArgs->outBufAvailableSema);
    return 0;
}

static bool PrefetchData(TestArgs *args,
    NvQueue *inputQueue,
    ChromaFormat inputFileChromaFormat,
    bool uvOrderFlag,
    uint32_t pixelAlignment)
{
    NvMediaAppBuffer *appBuffer = NULL;
    uint32_t YUVFrameNum = 0, numBframes, gopLength, idrPeriod, i;
    uint32_t frameNumInGop = 0, frameNumInIDRperiod = 0;
    NvMediaEncodePicType pictureType;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    for (i=1; i<=args->maxOutputBuffering; i++) {
        status = NvQueueGet(inputQueue,
                (void *)&appBuffer,
                100);
        if(status != NVMEDIA_STATUS_OK) {
            LOG_ERR("main: NvQueueGet failed\n");
            return false;
        }

        if (args->configParams.gopPattern == GOP_PATTERN_I) { //Ionly
            pictureType = NVMEDIA_ENCODE_PIC_TYPE_AUTOSELECT;
            YUVFrameNum = i - 1;
        } else if (args->configParams.gopPattern == GOP_PATTERN_IPP) { //IP
            if (pictureType != NVMEDIA_ENCODE_PIC_TYPE_P_INTRA_REFRESH)
                pictureType = NVMEDIA_ENCODE_PIC_TYPE_AUTOSELECT;
            YUVFrameNum = i - 1;
        } else {
            numBframes = args->configParams.gopPattern - 1;
            if (args->videoCodec == CODEC_H264) {
                gopLength  = args->configH264Params.gopLength;
                idrPeriod  = args->configH264Params.idrPeriod;
            }
            else if (args->videoCodec == CODEC_H265) {
                gopLength  = args->configH265Params.gopLength;
                idrPeriod  = args->configH265Params.idrPeriod;
            }
            else if (args->videoCodec == CODEC_AV1) {
                gopLength  = args->configAV1Params.gopLength;
                idrPeriod  = args->configAV1Params.idrPeriod;
            }
            else {
                gopLength  = args->configVP9Params.gopLength;
                idrPeriod  = args->configVP9Params.idrPeriod;
            }
            if (idrPeriod == 0)
                idrPeriod = gopLength;
            if (i == 1) {
                pictureType = NVMEDIA_ENCODE_PIC_TYPE_IDR;
                YUVFrameNum = 0;
            } else {
                YUVFrameNum = i - 1;
                if (frameNumInGop % gopLength == 0 || frameNumInGop % idrPeriod == 0) {
                    pictureType = NVMEDIA_ENCODE_PIC_TYPE_I;
                    frameNumInGop = 0;
                } else if ((frameNumInGop-1) % args->configParams.gopPattern == GOP_PATTERN_I) {
                    pictureType = NVMEDIA_ENCODE_PIC_TYPE_P;
                    if ((frameNumInGop+numBframes)>=((gopLength<idrPeriod)?gopLength:idrPeriod)) {
                        YUVFrameNum += ((gopLength<idrPeriod)?gopLength:idrPeriod) - frameNumInGop - 1;
                    } else {
                        YUVFrameNum += numBframes;
                    }
                } else {
                    YUVFrameNum --;
                    pictureType = NVMEDIA_ENCODE_PIC_TYPE_B;
                }

                if ((frameNumInIDRperiod >= idrPeriod) && (pictureType != NVMEDIA_ENCODE_PIC_TYPE_B) ) {
                    if (pictureType == NVMEDIA_ENCODE_PIC_TYPE_P)
                        YUVFrameNum = i - 1;
                    pictureType = NVMEDIA_ENCODE_PIC_TYPE_IDR;
                    frameNumInGop  = 0;
                    frameNumInIDRperiod = 0;
                }
            }
            frameNumInGop++;
            frameNumInIDRperiod++;
        }

        LOG_DBG("%s: Reading YUV frame %d from file %s to image surface "
                "location: %p. (W:%d, H:%d)\n", __func__,
                YUVFrameNum, args->infile, appBuffer->bufObj,
                args->configParams.encodeWidth, args->configParams.encodeHeight);

        /* Read a frame's worth of data from input file */
        status = ReadInput(args->infile,
                YUVFrameNum,
                args->configParams.encodeWidth,
                args->configParams.encodeHeight,
                appBuffer->bufObj,
                inputFileChromaFormat,
                uvOrderFlag,  //inputUVOrderFlag
                pixelAlignment);
        if(status != NVMEDIA_STATUS_OK) {
            LOG_ERR("%s: readYUVFile failed\n", __func__);
            return false; 
        }
        LOG_DBG("%s: ReadYUVFrame %d done\n", __func__, YUVFrameNum);

        // Put the image back in queue
        NvQueuePut(appBuffer->queue,
                (void *)&appBuffer,
                100);
    }

    return true;
}

NvMediaStatus allocAndRegisterBuffers(
  TestArgs *args,
  NvSciBufModule bufModule,
  uint16_t bufferWidth,
  uint16_t bufferHeight,
  ChromaFormat surfaceChromaFormat,
  NvSciBufAttrList *bufAttributeList,
  NvSciBufAttrList *bufConflictList,
  NvSciBufAttrList *bufReconciledList,
  NvQueue **inputQueue,
  uint32_t imagePoolSize)
{
    bool needCpuAccess = true;
    NvSciError err;
    err = NvSciBufAttrListCreate(bufModule, bufAttributeList);
    if(err != NvSciError_Success) {
        LOG_ERR("%s: SciBufAttrListCreate failed. Error: %d \n", __func__, err);
        return NVMEDIA_STATUS_INSUFFICIENT_BUFFERING;
    }

    NvMediaStatus status = NvMediaIEPFillNvSciBufAttrList(args->instanceId, *bufAttributeList);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("main: Failed to populate IEP internal attributes\n");
        return NVMEDIA_STATUS_INSUFFICIENT_BUFFERING;
    }
    status = PopulateNvSciBufAttrList(
                               surfaceChromaFormat,
                               bufferWidth,
                               bufferHeight,
                               needCpuAccess,
                               NvSciBufImage_BlockLinearType,
                               IEP_APP_MAX_INPUT_PLANE_COUNT,
                               NvSciBufAccessPerm_ReadWrite,
                               IEP_APP_BASE_ADDR_ALIGN,
                               NvSciColorStd_REC601_ER,
                               NvSciBufScan_ProgressiveType,
                               *bufAttributeList);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("main: Failed to populate attributes\n");
        return status;
    }
    err = NvSciBufAttrListReconcile(bufAttributeList, 1U,
             bufReconciledList, bufConflictList);
    if (err != NvSciError_Success) {
         LOG_ERR("main: Reconciliation for input frame failed\n");
         return status;
    }

    for (uint32_t i = 0; i < imagePoolSize; i++)
    {
        NvMediaAppBuffer *appBuffer = NULL;
        appBuffer = malloc(sizeof(NvMediaAppBuffer));
        memset(appBuffer, 0x0, sizeof(NvMediaAppBuffer));

        appBuffer->queue = *inputQueue;
        err = NvSciBufObjAlloc(*bufReconciledList, &appBuffer->bufObj);
        if (err != NvSciError_Success) {
            LOG_ERR("main: Allocation of input frame failed\n");
            return NVMEDIA_STATUS_INSUFFICIENT_BUFFERING;
        }

        status = NvQueuePut(*inputQueue,
                 (void *)&appBuffer, NV_TIMEOUT_INFINITE);
        if(status != NVMEDIA_STATUS_OK) {
             LOG_ERR("%s: NvQueuePut failed\n",__func__);
             return NVMEDIA_STATUS_INSUFFICIENT_BUFFERING;
        }
    }
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus deregisterAndFreeBuffers(
  TestArgs *args,
  NvMediaIEP *encoderCtx,
  NvQueue **inputQueue,
  uint8_t countNvSciBufObj,
  NvSciSyncCpuWaitContext *cpuWaitContext
)
{
    NvSciSyncTaskStatus taskStatus;
    NvMediaStatus status;
    NvSciError err;
    /* Clear all fences */

    for (uint8_t i = 0; i < countNvSciBufObj; i++)
    {
      NvMediaAppBuffer *appBuffer = NULL;
      status = NvQueueGet(*inputQueue,
                          (void *)&appBuffer,
                          100);
      if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("main: NvQueueGet failed\n");
        break;
      }

      /* Wait for operations on the image to be complete */
      err = NvSciSyncFenceWait(&appBuffer->eofFence, *cpuWaitContext, 1000*1000);
      if(err != NvSciError_Success) {
        LOG_ERR("NvSciSyncFenceWait failed\n");
      }
      taskStatus.status = NvSciSyncTaskStatus_Invalid;
      err = NvSciSyncFenceGetTaskStatus(&appBuffer->eofFence, &taskStatus);
      if (err != NvSciError_Success) {
        if(err != NvSciError_ClearedFence )
        {
            LOG_ERR("Error while retrieving taskStatus from fence\n");
            return NVMEDIA_STATUS_ERROR;
        }
      }
      else
      {
        if (taskStatus.status != NvSciSyncTaskStatus_Success)
        {
            LOG_ERR("TaskStatus shows failure\n");
            return NVMEDIA_STATUS_ERROR;
        }
      }

      NvSciSyncFenceClear(&appBuffer->eofFence);
      NvSciSyncFenceClear(&appBuffer->preFence);
      NvQueuePut(appBuffer->queue,
                 (void *)&appBuffer,
                 100);
    }

    /* Unregister NvSciBufObj */
    for (uint8_t i = 0; i < countNvSciBufObj; i++)
    {
      NvMediaAppBuffer *appBuffer = NULL;
      status = NvQueueGet(*inputQueue,
                          (void *)&appBuffer,
                          100);
      if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("main: NvQueueGet failed\n");
        break;
      }

      if ((!args->skipImageRegister) && (appBuffer->bufObj)) {
        status = NvMediaIEPUnregisterNvSciBufObj(
          (const NvMediaIEP *)encoderCtx, appBuffer->bufObj);
        if (status != NVMEDIA_STATUS_OK)
        {
          LOG_ERR("main: NvMediaIEPImageUnRegister failed\n");
        }
      }

      /* Free NvSciBufObj */
      if(appBuffer->bufObj) {
        NvSciBufObjFree(appBuffer->bufObj);
      }

      free(appBuffer);
    }
    return NVMEDIA_STATUS_OK;
}

int main(int argc, char *argv[])
{
    TestArgs args;
    FILE *outputFile = NULL;
    char outFileName[FILE_NAME_SIZE];
    ChromaFormat inputFileChromaFormat;
    ChromaFormat surfaceChromaFormat;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvMediaEncodeH264SEIPayload payloadArrH264[MAX_PAYLOAD_ARRAY_SIZE];
    NvMediaEncodeH265SEIPayload payloadArrH265[MAX_PAYLOAD_ARRAY_SIZE];
    NvMediaIEP *encoderCtx = NULL;
    bool nextFrameFlag = true;
    NvMediaIEPType encodeType = NVMEDIA_IMAGE_ENCODE_H264;
    NvSciError err;
    uint16_t bufferWidth, bufferHeight;
#if ENABLE_PROFILING
    uint64_t initStartTimeMark = 0;
    uint64_t initStopTimeMark = 0;
    NvpPerfData_t submissionLatencies;
    uint64_t feedFrameStartTimeMark = 0;
    uint64_t feedFrameStopTimeMark = 0;
    NvpRateLimitInfo_t rateLimitInfo;
    uint64_t numOfLatenciesToBeAggregated = 0;
    uint32_t k;
    NvpPerfData_t bufRegisterData[2*IMAGE_BUFFERS_POOL_SIZE_MAX + 2], totalInitPerfData;
    NvpPerfData_t *initLatenciesArray[2*IMAGE_BUFFERS_POOL_SIZE_MAX + 2];
#endif

    NvSciSyncModule syncModule = NULL;
    NvSciBufModule bufModule = NULL;
    NvSciSyncCpuWaitContext cpuWaitContext = NULL;

    union {
        NvMediaEncodeInitializeParamsH264 encoderInitParamsH264;
        NvMediaEncodeInitializeParamsH265 encoderInitParamsH265;
        NvMediaEncodeInitializeParamsVP9  encoderInitParamsVP9;
        NvMediaEncodeInitializeParamsAV1  encoderInitParamsAV1;
    } encoderInitParams;
    union {
        NvMediaEncodePicParamsH264 encodePicParamsH264;
        NvMediaEncodePicParamsH265 encodePicParamsH265;
        NvMediaEncodePicParamsVP9  encodePicParamsVP9;
        NvMediaEncodePicParamsAV1  encodePicParamsAV1;
    } encodePicParams;
    unsigned int currFrameParamsSectionIndex, currIdInIntervalPattern = (unsigned int)-1;
    long fileLength;
    uint32_t framesNum = 0, frameCounter = 1, totalFramesEncoded = 0;
    uint32_t imageSize = 0;
    int eventDataRecorderRingSize = 0;
    EventDataRecorderBuffData *eventDataRecorderDataRing = NULL;
    bool uvOrderFlag;
    uint32_t pixelAlignment;
    uint32_t imagePoolSize = 0;
    bool testPass = false;
    NvQueue *inputQueue = NULL;
    NvThread *outputThread = NULL;
    NvSemaphore *feedFrameDoneSema = NULL;
    NvSemaphore *outBufAvailableSema = NULL;
    EncOutThreadArgs encOutThreadArgs = {0};
    NvSciBufAttrList bufAttributeList;
    uint8_t countNvSciBufObj = 0;  // Keep a record  of  how many objects were allocated successfully
    NvSciBufSurfBPC bitdepth;
    NvSciBufSurfSampleType subsampleType;
#if !NV_IS_SAFETY
    // Parameters for resolution change
    FILE *dynResFile = NULL;
    bool resolutionChange = false;
    bool resolutionChangeDone = false;
    uint32_t framesNumChangedRes = 0, frameCounterDRC = 0;
    NvQueue *inputQueueChangedRes = NULL;
    // Parameters for frame type change
    FILE *frmTypeChangeFile = NULL;
    bool frmTypeChangeEnable = false;
    bool frmTypeChangeCurrentFrame = false;
    uint32_t frmTypeChangeCurrentPic = 0;
    uint32_t frmTypeChangeNextFrameNum = 0;
    uint32_t frmTypeChangeNextPic = 0;

    // Parameters for bitrate change
    FILE *dynBitrateFile = NULL;
    bool dynBitrateEnable = false; // whether DBC feature is enabled
    bool dynBitrateChangeCurrentFrame = false; // whether DBC should override current frame bitrate from the RC param setting
    uint32_t dynBitrateNextFrameNum = 0; // next frame num read from DBC input file
    // DBC specified values for current frame
    uint32_t dynBitrateCurrentAvgBitrate = 0;
    uint32_t dynBitrateCurrentVbvBufferSize = 0;
    // next values read from DBC input file
    uint32_t dynBitrateNextAvgBitrate = 0;
    uint32_t dynBitrateNextVbvBufferSize = 0;
    // Parameters for DFPS change
    FILE *dynFpsFile = NULL;
    bool dynFpsEnable = false; // whether DFPS feature is enabled
    bool dynFpsChangeCurrentFrame = false; // whether DFPS should override current frame rate
    uint32_t dynFpsNextFrameNum = 0; // next frame num read from DFPS input file
    // DFPS specified values for current frame
    uint32_t dynFpsCurrentFrameRateNum = 0;
    uint32_t dynFpsCurrentFrameRateDen = 0;
    // Next DFPS specified values from DFPS input file
    uint32_t dynFpsNextFrameRateNum = 0;
    uint32_t dynFpsNextFrameRateDen = 0;
    // Parameters for ROI
    FILE *ROIParamFile = NULL;

    NvMediaEncodeInputExtradata inExtraData;
    uint8_t countNvSciBufObjChangedRes = 0;
    RingEntry ringEntry;
    uint32_t QPDeltaBufferSize;
    memset(&inExtraData, 0, sizeof(NvMediaEncodeInputExtradata));
#endif
    signal(SIGINT, sigintHandler);
    signal(SIGTERM, sigintHandler);
    memset(&args,0,sizeof(TestArgs));
    args.configH264Params.h264VUIParameters = calloc(1, sizeof(NvMediaEncodeConfigH264VUIParams));
    args.configH265Params.h265VUIParameters = calloc(1, sizeof(NvMediaEncodeConfigH265VUIParams));

    LOG_DBG("main: Parsing command and reading encoding parameters from config file\n");
    if(ParseArgs(argc, argv, &args)) {
        LOG_ERR("main: Parsing arguments failed\n");
        goto fail;
    }

    if(CheckVersion(&args) != NVMEDIA_STATUS_OK) {
        goto fail;
    }

    if(args.crcoption.crcGenMode && args.crcoption.crcCheckMode) {
        LOG_ERR("main: crcGenMode and crcCheckMode cannot be enabled at the same time\n");
        goto fail;
    }

    if(args.videoCodec != CODEC_H264 && args.videoCodec != CODEC_H265 && args.videoCodec != CODEC_VP9 && args.videoCodec != CODEC_AV1) {
        LOG_ERR("main: H.264, H.265, VP9 and AV1 codec are currently supported by NvMedia image encoder\n");
        goto fail;
    }

    if (args.maxOutputBuffering == 0) {
        args.maxOutputBuffering = IMAGE_BUFFERS_POOL_SIZE;
        imagePoolSize = IMAGE_BUFFERS_POOL_SIZE;

#if ENABLE_PROFILING
        if (args.profileEnable) {
            args.maxOutputBuffering = IMAGE_BUFFERS_POOL_SIZE_MAX;
            imagePoolSize = IMAGE_BUFFERS_POOL_SIZE_MAX;
        }
#endif
    } else {
            imagePoolSize = args.maxOutputBuffering;
    }
    if((args.videoCodec == CODEC_VP9) || (args.videoCodec == CODEC_AV1)) {
        args.maxOutputBuffering = 1;
        imagePoolSize = 1;
#if ENABLE_PROFILING
        if (args.profileEnable) {
            LOG_ERR("Perf and Profile tests for VP9 and AV1 are unsupported\n");
            goto fail;
        }
#endif
    }

    status = NvQueueCreate(&inputQueue,
                imagePoolSize,
                sizeof(NvMediaAppBuffer *));
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvQueueCreate failed\n",__func__);
        goto fail;
    }

#if !NV_IS_SAFETY
    if (args.drcBufRealloc) {
        status = NvQueueCreate(&inputQueueChangedRes,
                    imagePoolSize,
                    sizeof(NvMediaAppBuffer *));
        if(status != NVMEDIA_STATUS_OK) {
            LOG_ERR("%s: NvQueueCreate failed\n",__func__);
            goto fail;
        }
    }
#endif

    status = NvSemaphoreCreate(&feedFrameDoneSema, 0, imagePoolSize + 1U);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvSemaphoreCreate failed\n",__func__);
        goto fail;
    }

    status = NvSemaphoreCreate(&outBufAvailableSema, imagePoolSize, imagePoolSize);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvSemaphoreCreate failed\n",__func__);
        goto fail;
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

    LOG_DBG("main: Opening output file\n");
    strcpy(outFileName, args.outfile);
    outputFile = fopen(outFileName, "w+");
    if(!outputFile) {
        LOG_ERR("main: Failed opening '%s' file for writing\n", args.outfile);
        goto fail;
    }
#if !NV_IS_SAFETY
    if (strlen(args.dynResFileName) != 0) {
        dynResFile = fopen(args.dynResFileName, "r");
        if (!dynResFile) {
            LOG_ERR("main: Failed opening '%s' file for writing\n", args.dynResFileName);
            goto fail;
        }
        if (fscanf(dynResFile, "%d %d %d", &args.dynResFrameNum, &args.dynResFrameWidth, &args.dynResFrameHeight) != 3) {
            args.dynResFrameNum = 0;
            args.dynResFrameWidth = args.configParams.encodeWidth;
            args.dynResFrameHeight = args.configParams.encodeHeight;
        }
    }
    if (strlen(args.qpDeltaFileBaseName) != 0) {
        char fileName[FILE_NAME_SIZE];
        if (snprintf(fileName, FILE_NAME_SIZE, "%s_00001.bin", args.qpDeltaFileBaseName) < 0) {
            goto fail;
        }
        FILE *fp = fopen(fileName, "r");
        if (fp == NULL) {
            LOG_ERR("main: Failed opening '%s' file for reading\n", fileName);
            goto fail;
        }
        fseek(fp, 0, SEEK_END);
        QPDeltaBufferSize = ftell(fp);
        if (QPDeltaBufferSize < getH265QPDeltaBufferSize(args.configParams.encodeWidth, args.configParams.encodeHeight)) {
            LOG_ERR("main: QPDeltaBufferSize not sufficient\n");
            goto fail;
        }
        fseek(fp, 0, SEEK_SET);
        for ( int i=0; i < IMAGE_BUFFERS_POOL_SIZE; i++) {
            ringEntry.qp_delta_buffer[i] = (signed char *)malloc(sizeof(char) * QPDeltaBufferSize);
            if (ringEntry.qp_delta_buffer[i] == NULL) {
                LOG_ERR("main: Failed to allocate memory for ringEntry.qp_delta_buffer[%d]\n", i);
                goto fail;
            }
        }
        fclose(fp);
    }
    if(strlen(args.frmTypeChangeFileName) != 0) {
        frmTypeChangeFile = fopen(args.frmTypeChangeFileName, "r");
        if (!frmTypeChangeFile) {
            LOG_ERR("main: Failed opening '%s' file for writing\n", args.frmTypeChangeFileName);
            goto fail;
        }
        if (fscanf(
                frmTypeChangeFile,
                "%d %d",
                &frmTypeChangeNextFrameNum,
                &frmTypeChangeNextPic
            ) != 2)
        {
            frmTypeChangeEnable = false;
            frmTypeChangeNextFrameNum = 0;
            frmTypeChangeNextPic = 0;
        }
        else
        {
            frmTypeChangeEnable = true;
        }
    }
    if (strlen(args.dynBitrateFileName) != 0) {
        dynBitrateFile = fopen(args.dynBitrateFileName, "r");
        if (!dynBitrateFile) {
            LOG_ERR("main: Failed opening '%s' file for writing\n", args.dynBitrateFileName);
            goto fail;
        }
        if (fscanf(
                dynBitrateFile,
                "%d %d %d",
                &dynBitrateNextFrameNum,
                &dynBitrateNextAvgBitrate,
                &dynBitrateNextVbvBufferSize
            ) != 3)
        {
            dynBitrateEnable = false;
            dynBitrateNextFrameNum = 0;
            dynBitrateNextAvgBitrate = 0;
            dynBitrateNextVbvBufferSize = 0;
        }
        else
        {
            dynBitrateEnable = true;
        }
    }
    if (strlen(args.dynFpsFileName) != 0) {
        dynFpsFile = fopen(args.dynFpsFileName, "r");
        if (!dynFpsFile) {
            LOG_ERR("main: Failed opening '%s' file for writing\n", args.dynFpsFileName);
            goto fail;
        }
        if (fscanf(
                dynFpsFile,
                "%d %d %d",
                &dynFpsNextFrameNum,
                &dynFpsNextFrameRateNum,
                &dynFpsNextFrameRateDen
            ) != 3)
        {
            dynFpsEnable = false;
            dynFpsNextFrameNum = 0;
            dynFpsNextFrameRateNum = 0;
            dynFpsNextFrameRateDen = 0;
        }
        else
        {
            dynFpsEnable = true;
        }
    }
    if (args.configParams.enableROIEncode) {
        if (0U != strlen(args.ROIParamFileName)) {
            ROIParamFile = fopen(args.ROIParamFileName, "r");
            if (!ROIParamFile) {
                LOG_ERR("main: Failed opening '%s' file for writing\n", args.ROIParamFileName);
                goto fail;
            }
        }
    }
#endif
    frameCounter += args.startFrame - 1;
    LOG_DBG("main: Encode start from frame %d \n", frameCounter);

    /* Map input surface format attributes */
    status = GetInputFormatAttributes(args.inputFileFormat,
            &inputFileChromaFormat, &uvOrderFlag, &pixelAlignment, &subsampleType);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("main: Unsupported input file format: %d\n", args.inputFileFormat);
        goto fail;
    }

    imageSize = GetInputFrameSizeBytes(inputFileChromaFormat,
            args.configParams.encodeWidth, args.configParams.encodeHeight);
    if (0 == imageSize) {
        LOG_ERR("main: Unsupported image dims: %ux%u for format: %d\n",
            args.configParams.encodeWidth, args.configParams.encodeHeight,
            inputFileChromaFormat);
        goto fail;
    }

    fileLength = GetInputFileSizeBytes(args.infile);
    if(0 == fileLength) {
        LOG_ERR("%s: Zero file length for file %s, len=%ld\n",
                __func__, args.infile, fileLength);
        goto fail;
    }

    /* Total number of frames available in the file */
    framesNum = fileLength / imageSize;
#if !NV_IS_SAFETY
    if((args.drcBufRealloc) && (strlen(args.infiledrc) != 0)) {
        imageSize = GetInputFrameSizeBytes(inputFileChromaFormat,
            args.dynResFrameWidth, args.dynResFrameHeight);
        fileLength = GetInputFileSizeBytes(args.infiledrc);
        framesNumChangedRes = fileLength / imageSize;
        framesNum += framesNumChangedRes;
    }
#endif
    LOG_DBG("Total frames in input file: %u\n", framesNum);

    /* Note: As NVENC only supports semi-planar formats, there is a need to
       create a surface in semi-planar format for a given planar input data */
    status = MapInputFileChromaToSurfaceChroma(
            inputFileChromaFormat, &surfaceChromaFormat, &bitdepth);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("main: Failed to populate surfaceChromaFormat\n");
        goto fail;
    }

    switch(args.videoCodec) {
    case CODEC_H264: // H264, 16 aglinment
        bufferWidth = ALIGN_16(args.configParams.encodeWidth);
        bufferHeight = ALIGN_16(args.configParams.encodeHeight);
        break;
    case CODEC_H265: // HEVC, 32 alignment
        bufferWidth = ALIGN_32(args.configParams.encodeWidth);
        bufferHeight = ALIGN_32(args.configParams.encodeHeight);

        break;
    case CODEC_VP9: // VP9, 64 alignment
        bufferWidth = ALIGN_64(args.configParams.encodeWidth);
        bufferHeight = ALIGN_64(args.configParams.encodeHeight);
        break;
    case CODEC_AV1: // TBD: Fix alignment for AV1
        bufferWidth = args.configParams.encodeWidth;
        bufferHeight = args.configParams.encodeHeight;
        break;
    default:
        LOG_ERR("main: unknown codec type \n");
        goto fail;
    }

#if ENABLE_PROFILING
    if (args.profileEnable) {
        char executionLatencyFileName[FILE_NAME_SIZE_EXTRA], submissionLatencyFileName[FILE_NAME_SIZE_EXTRA], initLatencyFileName[FILE_NAME_SIZE_EXTRA], initLatencyTempFileName[FILE_NAME_SIZE_EXTRA];
        sprintf(executionLatencyFileName, "%sExecutionLatency.csv",args.profileStatsFilePath);
        sprintf(submissionLatencyFileName, "%sSubmissionLatency.csv",args.profileStatsFilePath);
        sprintf(initLatencyFileName, "%sinitLatency.csv",args.profileStatsFilePath);
        for (k = 0; (k < 2 * IMAGE_BUFFERS_POOL_SIZE_MAX + 2); k++) {
            sprintf(initLatencyTempFileName, "%sinitLatency_%d.csv", args.profileStatsFilePath, k);
            NvpConstructPerfData(&bufRegisterData[k], 1, initLatencyTempFileName);
        }
        NvpConstructPerfData(&totalInitPerfData, 1, initLatencyFileName);
        NvpConstructPerfData(&encOutThreadArgs.latencies,args.loopCount * framesNum, executionLatencyFileName);
        NvpConstructPerfData(&submissionLatencies, args.loopCount * framesNum, submissionLatencyFileName);
        NvpRateLimitInit(&rateLimitInfo, args.limitFPS);
        initStartTimeMark = NvpGetTimeMark();
    }
#endif

    NvSciBufAttrList bufConflictList;
    NvSciBufAttrList bufReconciledList;
    status = allocAndRegisterBuffers(&args,
                                     bufModule,
                                     bufferWidth,
                                     bufferHeight,
                                     surfaceChromaFormat,
                                     &bufAttributeList,
                                     &bufConflictList,
                                     &bufReconciledList,
                                     &inputQueue,
                                     imagePoolSize);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("main: allocAndRegisterBuffers failed\n");
        goto fail;
    }
#if ENABLE_PROFILING
    if (args.profileEnable) {
        initStopTimeMark = NvpGetTimeMark();
        NvpRecordSample(&bufRegisterData[numOfLatenciesToBeAggregated], initStartTimeMark, initStopTimeMark);
        initLatenciesArray[numOfLatenciesToBeAggregated] = &bufRegisterData[numOfLatenciesToBeAggregated];
        numOfLatenciesToBeAggregated++;
    }
#endif
    /* The reconciled list is needed for later */
    NvSciBufAttrListFree(bufAttributeList);
    NvSciBufAttrListFree(bufConflictList);
#if !NV_IS_SAFETY
    if (args.drcBufRealloc) {
        NvSciBufAttrList bufReconciledListChangedRes;
        status = allocAndRegisterBuffers(&args,
                                         bufModule,
                                         args.dynResFrameWidth,
                                         args.dynResFrameHeight,
                                         surfaceChromaFormat,
                                         &bufAttributeList,
                                         &bufConflictList,
                                         &bufReconciledListChangedRes,
                                         &inputQueueChangedRes,
                                         imagePoolSize);
        if (NVMEDIA_STATUS_OK != status) {
            LOG_ERR("main: allocAndRegisterBuffers failed\n");
            goto fail;
        }
        /* The reconciled list is needed for later */
        NvSciBufAttrListFree(bufAttributeList);
        NvSciBufAttrListFree(bufConflictList);
    }
#endif

    if (args.eventDataRecorderMode == true) {
        if (args.configParams.frameRateDen == 0)
            args.configParams.frameRateDen = 1;
        if (args.configParams.frameRateNum == 0)
            args.configParams.frameRateNum = 30;
        if(args.videoCodec == CODEC_H264)
        {
            args.configH264Params.gopLength = args.configParams.frameRateNum/args.configParams.frameRateDen;
            args.configH264Params.idrPeriod = args.configH264Params.gopLength;
            args.configH264Params.repeatSPSPPS = NVMEDIA_ENCODE_SPSPPS_REPEAT_IDR_FRAMES;
            eventDataRecorderRingSize = args.configH264Params.gopLength * (args.eventDataRecorderRecordingTime + 1);
        } else {
            args.configH265Params.gopLength = args.configParams.frameRateNum/args.configParams.frameRateDen;
            args.configH265Params.idrPeriod = args.configH265Params.gopLength;
            args.configH265Params.repeatSPSPPS = NVMEDIA_ENCODE_SPSPPS_REPEAT_IDR_FRAMES;
            eventDataRecorderRingSize = args.configH265Params.gopLength * (args.eventDataRecorderRecordingTime + 1);
        }

        eventDataRecorderDataRing = calloc(sizeof(EventDataRecorderBuffData), eventDataRecorderRingSize);
        LOG_DBG("main: eventDataRecorder Mode is on, eventDataRecorderRingSize=%d\n", eventDataRecorderRingSize);
    }

    LOG_DBG("main: Setting encoder initialization params\n");

    switch (args.videoCodec){
    case CODEC_H264:
        encodeType = NVMEDIA_IMAGE_ENCODE_H264;
        memset(&encoderInitParams.encoderInitParamsH264, 0, sizeof(NvMediaEncodeInitializeParamsH264));
        SetEncoderInitParamsH264(&encoderInitParams.encoderInitParamsH264, &args);
        if (args.configParams.gopPattern == GOP_PATTERN_I) {
            args.configParams.gopLength = 1;   //Has to match
        }

        args.configH264Params.gopLength = args.configParams.gopLength;
        if (args.configParams.gopPattern == 0) {
            args.configH264Params.gopLength = 1;
        }

        LOG_DBG("main: Creating video encoder\n I-Frames only: %s, GOP Size: %u, "
                "Frames to encode (0 means all frames): %u\n",
                    args.configParams.gopPattern == GOP_PATTERN_I ? "Yes" : "No",
                    args.configH264Params.gopLength,
                    args.framesToBeEncoded);
        break;

    case CODEC_H265:
        encodeType = NVMEDIA_IMAGE_ENCODE_HEVC;
        memset(&encoderInitParams.encoderInitParamsH265, 0, sizeof(NvMediaEncodeInitializeParamsH265));
        SetEncoderInitParamsH265(&encoderInitParams.encoderInitParamsH265, &args);
        if (args.configParams.gopPattern == GOP_PATTERN_I) {
            args.configH265Params.gopLength = 1;   //Has to match
        }

        args.configH265Params.gopLength = args.configParams.gopLength;
        if (args.configParams.gopPattern == 0) {
            args.configH265Params.gopLength = 1;
        }

        LOG_DBG("main: Creating video encoder\n I-Frames only: %s, GOP Size: %u, "
                "Frames to encode (0 means all frames): %u\n",
                   args.configParams.gopPattern == GOP_PATTERN_I ? "Yes" : "No",
                   args.configH265Params.gopLength,
                   args.framesToBeEncoded);
        break;

    case CODEC_VP9:
        encodeType = NVMEDIA_IMAGE_ENCODE_VP9;
        memset(&encoderInitParams.encoderInitParamsVP9, 0, sizeof(NvMediaEncodeInitializeParamsVP9));
        SetEncoderInitParamsVP9(&encoderInitParams.encoderInitParamsVP9, &args);
        if (args.configParams.gopPattern == GOP_PATTERN_I) {
            args.configVP9Params.gopLength = 1;
        }

        args.configVP9Params.gopLength = args.configParams.gopLength;
        if (args.configParams.gopPattern == 0) {
            args.configVP9Params.gopLength = 1;
        }

        LOG_DBG("main: Creating video encoder\n Key-Frames only: %s, GOP Size: %u,"
                " Frames to encode (0 means all frames): %u\n",
                    args.configParams.gopPattern == GOP_PATTERN_I ? "Yes" : "No",
                    args.configVP9Params.gopLength,
                    args.framesToBeEncoded);
        break;

    case CODEC_AV1:
        encodeType = NVMEDIA_IMAGE_ENCODE_AV1;
        memset(&encoderInitParams.encoderInitParamsAV1, 0, sizeof(NvMediaEncodeInitializeParamsAV1));
        SetEncoderInitParamsAV1(&encoderInitParams.encoderInitParamsAV1, &args);

        args.configAV1Params.gopLength = args.configParams.gopLength;
        if (args.configParams.gopPattern == 0) {
            args.configAV1Params.gopLength = 1;
        }

        LOG_DBG("main: Creating video encoder\n Key-Frames only: %s, GOP Size: "
                "%u, Frames to encode (0 means all frames): %u\n",
                    args.configParams.gopPattern == GOP_PATTERN_I ? "Yes" : "No",
                    args.configAV1Params.gopLength,
                    args.framesToBeEncoded);
        break;

    default:
        LOG_ERR("main: unknown codec type \n");
        goto fail;
    }

#if ENABLE_PROFILING
    if (args.profileEnable) {
        initStartTimeMark = NvpGetTimeMark();
    }
#endif
#if !NV_IS_SAFETY
    if (args.alternateCreateAPI == 2)
    {
        encoderCtx = NvMediaIEPCreateCtx();
        if(!encoderCtx) {
            LOG_ERR("main: NvMediaIEPCreateCtx failed\n");
            goto fail;
        }

        status = NvMediaIEPInit(encoderCtx,
                                encodeType,               // codec
                                &encoderInitParams,       // init params
                                bufReconciledList,        // reconciled attr list
                                args.maxOutputBuffering,  // maxOutputBuffering
                                args.instanceId);
        LOG_DBG("main: Using NvMediaIEPCreateCtx API for creating IEP SW instance \n");
        if(status != NVMEDIA_STATUS_OK) {
            LOG_ERR("main: NvMediaIEPInit failed\n");
            goto fail;
        }
    }
    else
#endif
    {
        if (args.alternateCreateAPI == 1)
        {
            encoderCtx = NvMediaIEPCreateEx(encodeType,               // codec
                                             &encoderInitParams,       // init params
                                             subsampleType,            // Sub-sampling type
                                             bitdepth,                 // Bitdepth
                                             args.maxOutputBuffering,  // maxOutputBuffering
                                             args.instanceId);
            LOG_DBG("main: Using NvMediaIEPCreateEx API for creating IEP SW instance \n");
            if(!encoderCtx) {
                LOG_ERR("main: NvMediaIEPCreate failed\n");
                goto fail;
            }
        }
        else
        {
            encoderCtx = NvMediaIEPCreate(encodeType,               // codec
                                          &encoderInitParams,       // init params
                                          bufReconciledList,        // reconciled attr list
                                          args.maxOutputBuffering,  // maxOutputBuffering
                                          args.instanceId);
            LOG_DBG("main: Using NvMediaIEPCreate API for creating IEP SW instance \n");
            if(!encoderCtx) {
                LOG_ERR("main: NvMediaIEPCreate failed\n");
                goto fail;
            }
        }
    }
#if ENABLE_PROFILING
    if (args.profileEnable) {
        initStopTimeMark = NvpGetTimeMark();
        NvpRecordSample(&bufRegisterData[numOfLatenciesToBeAggregated], initStartTimeMark, initStopTimeMark);
        initLatenciesArray[numOfLatenciesToBeAggregated] = &bufRegisterData[numOfLatenciesToBeAggregated];
        numOfLatenciesToBeAggregated++;
    }
#endif

    LOG_DBG("main: NvMediaIEPCreate, %p\n", encoderCtx);

    if (!args.skipImageRegister) {
        for (uint32_t i=0; i<imagePoolSize; i++)
        {
            NvMediaAppBuffer *appBuffer = NULL;
            status = NvQueueGet(inputQueue,
                    (void *)&appBuffer,
                    100);
            if(status != NVMEDIA_STATUS_OK) {
                LOG_ERR("main: NvQueueGet failed\n");
                goto fail;
            }
#if ENABLE_PROFILING
            if (args.profileEnable) {
                initStartTimeMark = NvpGetTimeMark();
            }
#endif
            status = NvMediaIEPRegisterNvSciBufObj(encoderCtx, appBuffer->bufObj);
            if (status != NVMEDIA_STATUS_OK) {
                LOG_ERR("main: Register input image failed\n");
                goto fail;
            }
#if ENABLE_PROFILING
            if (args.profileEnable) {
                initStopTimeMark = NvpGetTimeMark();
                NvpRecordSample(&bufRegisterData[numOfLatenciesToBeAggregated], initStartTimeMark, initStopTimeMark);
                initLatenciesArray[numOfLatenciesToBeAggregated] = &bufRegisterData[numOfLatenciesToBeAggregated];
                numOfLatenciesToBeAggregated++;
            }
#endif

            NvQueuePut(inputQueue,
                    (void *)&appBuffer,
                    100);
            countNvSciBufObj++;
        }
#if !NV_IS_SAFETY
        if (args.drcBufRealloc) {
            for (uint32_t i=0; i<imagePoolSize; i++)
            {
                NvMediaAppBuffer *appBuffer = NULL;
                status = NvQueueGet(inputQueueChangedRes,
                        (void *)&appBuffer,
                        100);
                if(status != NVMEDIA_STATUS_OK) {
                    LOG_ERR("main: NvQueueGet failed\n");
                    goto fail;
                }
                status = NvMediaIEPRegisterNvSciBufObj(encoderCtx, appBuffer->bufObj);
                if (status != NVMEDIA_STATUS_OK) {
                    LOG_ERR("main: Register input image failed\n");
                    goto fail;
                }

                NvQueuePut(inputQueueChangedRes,
                        (void *)&appBuffer,
                        100);
                countNvSciBufObjChangedRes++;
            }
        }
#endif
    }

    NvSciSyncAttrList iepSignalerAttrList = NULL;
    NvSciSyncAttrList cpuWaiterAttrList = NULL;
    NvSciSyncAttrList iepWaiterAttrList = NULL;
    NvSciSyncAttrList cpuSignalerAttrList = NULL;
    err = NvSciSyncAttrListCreate(syncModule, &iepSignalerAttrList);
    if(err != NvSciError_Success) {
        LOG_ERR("%s: Create signaler attr list failed. Error: %d \n", __func__, err);
        goto fail;
    }

    err = NvSciSyncAttrListCreate(syncModule, &iepWaiterAttrList);
    if(err != NvSciError_Success) {
        LOG_ERR("%s: Create waiter attr list failed. Error: %d \n", __func__, err);
        goto fail;
    }

    status = NvMediaIEPFillNvSciSyncAttrList(encoderCtx, iepSignalerAttrList,
            NVMEDIA_SIGNALER);
    if(status != NVMEDIA_STATUS_OK) {
       LOG_ERR("main: Failed to fill signaler attr list.\n");
       goto fail;
    }

    status = NvMediaIEPFillNvSciSyncAttrList(encoderCtx, iepWaiterAttrList,
            NVMEDIA_WAITER);
    if(status != NVMEDIA_STATUS_OK) {
       LOG_ERR("main: Failed to fill waiter attr list.\n");
       goto fail;
    }

    err = NvSciSyncAttrListCreate(syncModule, &cpuWaiterAttrList);
    if(err != NvSciError_Success) {
        LOG_ERR("%s: Create waiter attr list failed. Error: %d \n", __func__, err);
        goto fail;
    }

    err = NvSciSyncAttrListCreate(syncModule, &cpuSignalerAttrList);
    if(err != NvSciError_Success) {
        LOG_ERR("%s: Create signaler attr list failed. Error: %d \n", __func__, err);
        goto fail;
    }

    // Set NvSciSyncAttrKey_NeedCpuAccess to true and NvSciSyncAttrKey_RequiredPerm to NvSciSyncAccessPerm_WaitOnly
    // TODO Move to proper location if required
    bool cpuWaiter = true;
    NvSciSyncAttrList syncUnreconciledList[2] = {NULL};
    NvSciSyncAttrList syncReconciledList = NULL;
    NvSciSyncObj eofSyncObj = NULL;
    NvSciSyncObj preSyncObj = NULL;
    NvSciSyncAttrList syncNewConflictList = NULL;
    NvSciSyncAttrKeyValuePair keyValue[2] = {0};
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
        goto fail;
    }

    syncUnreconciledList[0] = iepSignalerAttrList;
    syncUnreconciledList[1] = cpuWaiterAttrList;

    /* Reconcile Signaler and Waiter NvSciSyncAttrList */
    err = NvSciSyncAttrListReconcile(syncUnreconciledList, 2, &syncReconciledList,
            &syncNewConflictList);
    if(err != NvSciError_Success) {
        LOG_ERR("NvSciSyncAttrListSetAttrs failed\n");
        goto fail;
    }

    /* Create NvSciSync object and get the syncObj */
    err = NvSciSyncObjAlloc(syncReconciledList, &eofSyncObj);
    if(err != NvSciError_Success) {
        LOG_ERR("NvSciSyncAttrListSetAttrs failed\n");
        goto fail;
    }

    NvSciSyncAttrListFree(syncReconciledList);
    NvSciSyncAttrListFree(syncUnreconciledList[0]);
    NvSciSyncAttrListFree(syncUnreconciledList[1]);
    NvSciSyncAttrListFree(syncNewConflictList);

     /* Fill NvSciSyncAttrList for CPU signaler */
    bool cpuSignaler = true;
    keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void*) &cpuSignaler;
    keyValue[0].len = sizeof(cpuSignaler);
    cpuPerm = NvSciSyncAccessPerm_SignalOnly;
    keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*)&cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);
    err = NvSciSyncAttrListSetAttrs(cpuSignalerAttrList, keyValue, 2);
    if(err != NvSciError_Success) {
        LOG_ERR("NvSciSyncAttrListSetAttrs failed\n");
        goto fail;
    }

    syncUnreconciledList[0] = iepWaiterAttrList;
    syncUnreconciledList[1] = cpuSignalerAttrList;

    /* Reconcile Signaler and Waiter NvSciSyncAttrList */
    err = NvSciSyncAttrListReconcile(syncUnreconciledList, 2, &syncReconciledList,
            &syncNewConflictList);
    if(err != NvSciError_Success) {
        LOG_ERR("NvSciSyncAttrListSetAttrs failed\n");
        goto fail;
    }

    /* Create NvSciSync object and get the syncObj */
    err = NvSciSyncObjAlloc(syncReconciledList, &preSyncObj);
    if(err != NvSciError_Success) {
        LOG_ERR("NvSciSyncAttrListSetAttrs failed\n");
        goto fail;
    }

    NvSciSyncAttrListFree(syncReconciledList);
    NvSciSyncAttrListFree(syncUnreconciledList[0]);
    NvSciSyncAttrListFree(syncUnreconciledList[1]);
    NvSciSyncAttrListFree(syncNewConflictList);

    status = NvMediaIEPRegisterNvSciSyncObj(encoderCtx, NVMEDIA_EOFSYNCOBJ, eofSyncObj);
    if(status != NVMEDIA_STATUS_OK) {
       LOG_ERR("main: Failed to register sci sync eof obj attr list.\n");
       goto fail;
    }

    status = NvMediaIEPRegisterNvSciSyncObj(encoderCtx, NVMEDIA_PRESYNCOBJ, preSyncObj);
    if(status != NVMEDIA_STATUS_OK) {
       LOG_ERR("main: Failed to register sci sync pre obj attr list.\n");
       goto fail;
    }

    status = NvMediaIEPSetNvSciSyncObjforEOF(encoderCtx, eofSyncObj);
    if(status != NVMEDIA_STATUS_OK) {
       LOG_ERR("main: Failed to set sci sync eof obj attr list.\n");
       goto fail;
    }

    /* Configure encoder */
    if (args.videoCodec == CODEC_H264) {
        SetEncodeConfigRCParam(&args.configH264Params.rcParams, &args, args.rateControlSectionNum - 1);
        status = NvMediaIEPSetConfiguration(encoderCtx, &args.configH264Params);
    } else if (args.videoCodec == CODEC_H265) {
        SetEncodeConfigRCParam(&args.configH265Params.rcParams, &args, args.rateControlSectionNum - 1);
        status = NvMediaIEPSetConfiguration(encoderCtx, &args.configH265Params);
    }
    else if (args.videoCodec == CODEC_AV1) {
        SetEncodeConfigRCParam(&args.configAV1Params.rcParams, &args, args.rateControlSectionNum - 1);
        status = NvMediaIEPSetConfiguration(encoderCtx, &args.configAV1Params);
    }
    else {
        SetEncodeConfigRCParam(&args.configVP9Params.rcParams, &args, args.rateControlSectionNum - 1);
        status = NvMediaIEPSetConfiguration(encoderCtx, &args.configVP9Params);
    }

    if(status != NVMEDIA_STATUS_OK) {
       LOG_ERR("main: SetConfiguration failed\n");
       goto fail;
    }
    LOG_DBG("main: NvMediaIEPSetConfiguration done\n");

    // Read the input images in advance
    if (args.preFetchBuffer) {
        if (!PrefetchData(&args, inputQueue, inputFileChromaFormat, uvOrderFlag, pixelAlignment)) {
            LOG_ERR("main: Prefetching data for perf test is failed\n");
            goto fail;
        }
    }

    encOutThreadArgs.encoderCtx = encoderCtx;
    encOutThreadArgs.totalBytes = 0;
    encOutThreadArgs.args = &args;
    encOutThreadArgs.outputFile = outputFile;
    encOutThreadArgs.feedFrameDoneSema = feedFrameDoneSema;
    encOutThreadArgs.outBufAvailableSema = outBufAvailableSema;
    encOutThreadArgs.encOutThreadExit = 0;
    encOutThreadArgs.encError = 0;
    encOutThreadArgs.errorOccurred = 0;
    encOutThreadArgs.frameCounter = frameCounter;
    encOutThreadArgs.framesNum = framesNum;
    encOutThreadArgs.maxBitrate = 0;
    encOutThreadArgs.minBitrate = 0xFFFFFFFF;
    encOutThreadArgs.startIdx = -1;
    encOutThreadArgs.endIdx = -1;
    encOutThreadArgs.eventDataRecorderRingSize = eventDataRecorderRingSize;
    encOutThreadArgs.eventDataRecorderDataRing = eventDataRecorderDataRing;
#if !NV_IS_SAFETY
    encOutThreadArgs.enableSliceEncode = (args.videoCodec == CODEC_H265) && encoderInitParams.encoderInitParamsH265.enableSliceEncode;
#endif

    status = NvThreadCreate(&outputThread, EncoderOutputThread,
                (void *)&encOutThreadArgs, NV_THREAD_PRIORITY_NORMAL);

    if (status != NVMEDIA_STATUS_OK) {
       LOG_ERR("main: NvThreadCreate failed\n");
       goto fail;
    }

#if __QNXNTO__
        if(args.enableVMSuspend)
        {
            nvdvms_set_vm_state(NVDVMS_SUSPEND);
        }
#endif

    //clock begin
    while (nextFrameFlag && (encodeStop == 0u)) {
        NvMediaAppBuffer *appBuffer = NULL;
        static int numBframes = 0, gopLength = 0, frameNumInGop = 0, idrPeriod = 0, frameNumInIDRperiod = 0;
        unsigned int YUVFrameNum = 0;
        NvMediaEncodePicType pictureType;
        unsigned int nextBFrames = 0;

        currIdInIntervalPattern = (currIdInIntervalPattern + 1) < args.frameIntervalPatternLength ? currIdInIntervalPattern + 1 : 0;
        currFrameParamsSectionIndex = args.frameIntervalPattern[currIdInIntervalPattern] - '0' - 1;

#if !NV_IS_SAFETY
        if (frmTypeChangeEnable == true) {
            if (frmTypeChangeNextFrameNum == frameCounter) { // this frame has a frame type force
                frmTypeChangeCurrentFrame = true;
                frmTypeChangeCurrentPic = frmTypeChangeNextPic;
                // read the frame ID and frame pic type
                if (fscanf(
                        frmTypeChangeFile,
                        "%d %d",
                        &frmTypeChangeNextFrameNum,
                        &frmTypeChangeNextPic
                    ) != 2)
                {
                    frmTypeChangeNextFrameNum = 0;
                    frmTypeChangeNextPic = NVMEDIA_ENCODE_PIC_TYPE_AUTOSELECT;
                }
            }
        }
        if (dynBitrateEnable == true) {
            if (dynBitrateNextFrameNum == frameCounter) { // this frame has a bitrate change
                dynBitrateChangeCurrentFrame = true;
                dynBitrateCurrentAvgBitrate = dynBitrateNextAvgBitrate;
                dynBitrateCurrentVbvBufferSize = dynBitrateNextVbvBufferSize;
                // read the frame ID and value for the next bitrate change, if it exists
                if (fscanf(
                        dynBitrateFile,
                        "%d %d %d",
                        &dynBitrateNextFrameNum,
                        &dynBitrateNextAvgBitrate,
                        &dynBitrateNextVbvBufferSize
                    ) != 3)
                {
                    dynBitrateNextFrameNum = 0;
                    dynBitrateNextAvgBitrate = 0;
                    dynBitrateNextVbvBufferSize = 0;
                }
            }
        }
        if (dynFpsEnable == true) {
            if (dynFpsNextFrameNum == frameCounter) { // this frame has a FPS change
                dynFpsChangeCurrentFrame = true;
                dynFpsCurrentFrameRateNum = dynFpsNextFrameRateNum;
                dynFpsCurrentFrameRateDen = dynFpsNextFrameRateDen;
                // read the frame ID and value for the next bitrate change, if it exists
                if (fscanf(
                        dynFpsFile,
                        "%d %d %d",
                        &dynFpsNextFrameNum,
                        &dynFpsNextFrameRateNum,
                        &dynFpsNextFrameRateDen
                    ) != 3)
                {
                    dynFpsNextFrameNum = 0;
                    dynFpsNextFrameRateNum = 0;
                    dynFpsNextFrameRateDen = 0;
                }
            }
        }
#endif

        if (args.videoCodec == CODEC_H264) {
            memset(&encodePicParams.encodePicParamsH264, 0, sizeof(NvMediaEncodePicParamsH264));
            encodePicParams.encodePicParamsH264.seiPayloadArray = &payloadArrH264[0];
            SetEncodePicParamsH264(&encodePicParams.encodePicParamsH264, &args, frameCounter - 1, currFrameParamsSectionIndex);
            pictureType = encodePicParams.encodePicParamsH264.pictureType;
            gopLength  = args.configH264Params.gopLength;
            idrPeriod  = args.configH264Params.idrPeriod;
#if !NV_IS_SAFETY
            if (dynBitrateChangeCurrentFrame)
            {
                encodePicParams.encodePicParamsH264.rcParams.params.cbr.averageBitRate = dynBitrateCurrentAvgBitrate;
                encodePicParams.encodePicParamsH264.rcParams.params.cbr.vbvBufferSize = dynBitrateCurrentVbvBufferSize;
                args.configH264Params.rcParams.params.cbr.averageBitRate = dynBitrateCurrentAvgBitrate;
                args.configH264Params.rcParams.params.cbr.vbvBufferSize = dynBitrateCurrentVbvBufferSize;
                encodePicParams.encodePicParamsH264.encodePicFlags |= NVMEDIA_ENCODE_PIC_FLAG_RATECONTROL_CHANGE;
                dynBitrateChangeCurrentFrame = false;
            }
#endif
        } else if (args.videoCodec == CODEC_H265){
            memset(&encodePicParams.encodePicParamsH265, 0, sizeof(NvMediaEncodePicParamsH265));
            encodePicParams.encodePicParamsH265.seiPayloadArray = &payloadArrH265[0];
            SetEncodePicParamsH265(&encodePicParams.encodePicParamsH265, &args, frameCounter - 1, currFrameParamsSectionIndex);
            pictureType = encodePicParams.encodePicParamsH265.pictureType;
            gopLength  = args.configH265Params.gopLength;
            idrPeriod  = args.configH265Params.idrPeriod;
#if !NV_IS_SAFETY
            if (dynBitrateChangeCurrentFrame)
            {
                encodePicParams.encodePicParamsH265.rcParams.params.cbr.averageBitRate = dynBitrateCurrentAvgBitrate;
                encodePicParams.encodePicParamsH265.rcParams.params.cbr.vbvBufferSize = dynBitrateCurrentVbvBufferSize;
                args.configH265Params.rcParams.params.cbr.averageBitRate = dynBitrateCurrentAvgBitrate;
                args.configH265Params.rcParams.params.cbr.vbvBufferSize = dynBitrateCurrentVbvBufferSize;
                encodePicParams.encodePicParamsH265.encodePicFlags |= NVMEDIA_ENCODE_PIC_FLAG_RATECONTROL_CHANGE;
                dynBitrateChangeCurrentFrame = false;
            }
#endif
        }
        else if (args.videoCodec == CODEC_AV1) {
            memset(&encodePicParams.encodePicParamsAV1, 0, sizeof(NvMediaEncodePicParamsAV1));
            SetEncodePicParamsAV1(&encodePicParams.encodePicParamsAV1, &args, frameCounter - 1, currFrameParamsSectionIndex);
            pictureType = encodePicParams.encodePicParamsAV1.pictureType;
            gopLength  = args.configAV1Params.gopLength;
            idrPeriod  = args.configAV1Params.idrPeriod;
#if !NV_IS_SAFETY
            //Update the new FPS params in case of DFPS
            if (dynFpsChangeCurrentFrame)
            {
                encodePicParams.encodePicParamsAV1.frameRateNum = dynFpsCurrentFrameRateNum;
                encodePicParams.encodePicParamsAV1.frameRateDen = dynFpsCurrentFrameRateDen;
                dynFpsChangeCurrentFrame = false;
            }
            //Update the new bitrate params in case of DBC
            if (dynBitrateChangeCurrentFrame)
            {
                encodePicParams.encodePicParamsAV1.rcParams.params.cbr.averageBitRate = dynBitrateCurrentAvgBitrate;
                encodePicParams.encodePicParamsAV1.rcParams.params.cbr.vbvBufferSize = dynBitrateCurrentVbvBufferSize;
                args.configAV1Params.rcParams.params.cbr.averageBitRate = dynBitrateCurrentAvgBitrate;
                args.configAV1Params.rcParams.params.cbr.vbvBufferSize = dynBitrateCurrentVbvBufferSize;
                encodePicParams.encodePicParamsAV1.encodePicFlags |= NVMEDIA_ENCODE_PIC_FLAG_RATECONTROL_CHANGE;
                dynBitrateChangeCurrentFrame = false;
            }
#endif
        }
        else {
            memset(&encodePicParams.encodePicParamsVP9, 0, sizeof(NvMediaEncodePicParamsVP9));
            SetEncodePicParamsVP9(&encodePicParams.encodePicParamsVP9, &args, frameCounter - 1, currFrameParamsSectionIndex);
            pictureType = encodePicParams.encodePicParamsVP9.pictureType;
            gopLength  = args.configVP9Params.gopLength;
            idrPeriod  = args.configVP9Params.idrPeriod;
        }

        if (args.configParams.gopPattern == GOP_PATTERN_I) { //Ionly
            pictureType = NVMEDIA_ENCODE_PIC_TYPE_AUTOSELECT;
            YUVFrameNum = frameCounter - 1;
        } else if (args.configParams.gopPattern == GOP_PATTERN_IPP) { //IP
            if (pictureType != NVMEDIA_ENCODE_PIC_TYPE_P_INTRA_REFRESH) {
                pictureType = NVMEDIA_ENCODE_PIC_TYPE_AUTOSELECT;
            }
            YUVFrameNum = frameCounter - 1;
        } else {
            numBframes = args.configParams.gopPattern - 1;
            if (idrPeriod == 0) {
                idrPeriod = gopLength;
            }
            if (frameCounter == 1) {
                pictureType = NVMEDIA_ENCODE_PIC_TYPE_IDR;
                YUVFrameNum = 0;
            } else {
                YUVFrameNum = frameCounter - 1;
                if (frameNumInGop % gopLength == 0 || frameNumInGop % idrPeriod == 0) {
                    pictureType = NVMEDIA_ENCODE_PIC_TYPE_I;
                    frameNumInGop = 0;
                    LOG_DBG("main: pictureType I\n");
                } else if ((frameNumInGop-1) % args.configParams.gopPattern == GOP_PATTERN_I) {
                    pictureType = NVMEDIA_ENCODE_PIC_TYPE_P;
                    LOG_DBG("main: pictureType P\n");
                    if ((frameNumInGop+numBframes)>=((gopLength<idrPeriod)?gopLength:idrPeriod)) {
                        nextBFrames = ((gopLength<idrPeriod)?gopLength:idrPeriod) - frameNumInGop - 1;
                        YUVFrameNum += ((gopLength<idrPeriod)?gopLength:idrPeriod) - frameNumInGop - 1;
                    } else {
                        nextBFrames = numBframes;
                        YUVFrameNum += numBframes;
                    }
                    if (YUVFrameNum >= args.loopCount * framesNum) {
                        goto Done;
                    }
                } else {
                    YUVFrameNum --;
                    pictureType = NVMEDIA_ENCODE_PIC_TYPE_B;
                    LOG_DBG("main: pictureType B\n");
                }

                if ((frameNumInIDRperiod >= idrPeriod) && (pictureType != NVMEDIA_ENCODE_PIC_TYPE_B) ) {
                    if (pictureType == NVMEDIA_ENCODE_PIC_TYPE_P) {
                        YUVFrameNum = frameCounter - 1;
                    }
                    pictureType = NVMEDIA_ENCODE_PIC_TYPE_IDR;
                    nextBFrames = 0;
                    LOG_DBG("main: pictureType IDR\n");
                    frameNumInGop  = 0;
                    frameNumInIDRperiod = 0;
                }
            }
            frameNumInGop++;
            frameNumInIDRperiod++;
        }

#if !NV_IS_SAFETY
        if(frmTypeChangeCurrentFrame) {
            pictureType = frmTypeChangeCurrentPic;
            frameNumInGop = 0;
            frmTypeChangeCurrentFrame = false;
        }
        if (frameCounter == args.dynResFrameNum) {
            pictureType = NVMEDIA_ENCODE_PIC_TYPE_IDR;
            resolutionChange = true;
        }
#endif
        if (args.videoCodec == CODEC_H264) {
            encodePicParams.encodePicParamsH264.pictureType = pictureType;
            encodePicParams.encodePicParamsH264.nextBFrames = nextBFrames;
        } else if(args.videoCodec == CODEC_H265) {
            encodePicParams.encodePicParamsH265.pictureType = pictureType;
            encodePicParams.encodePicParamsH265.nextBFrames = nextBFrames;
        }
        else if(args.videoCodec == CODEC_AV1) {
            encodePicParams.encodePicParamsAV1.pictureType  = pictureType;
        }
        else {
            encodePicParams.encodePicParamsVP9.pictureType  = pictureType;
        }
#if !NV_IS_SAFETY
        if (resolutionChange) {
            if(args.drcBufRealloc) {
                args.configParams.encodeWidth = args.dynResFrameWidth;
                args.configParams.encodeHeight = args.dynResFrameHeight;
            }
            resolutionChangeDone = true;
        }
        if ((!args.drcBufRealloc) || (resolutionChangeDone != true)) {
#endif
          status = NvQueueGet(inputQueue,
                  (void *)&appBuffer,
                  100);
          if(status != NVMEDIA_STATUS_OK) {
              LOG_ERR("main: NvQueueGet failed\n");
              goto fail;
          }
#if !NV_IS_SAFETY
        } else {
          status = NvQueueGet(inputQueueChangedRes,
                  (void *)&appBuffer,
                  100);
          if(status != NVMEDIA_STATUS_OK) {
              LOG_ERR("main: NvQueueGet failed\n");
              goto fail;
          }
        }
#endif

        /* Wait for operations on the image to be complete */
        err = NvSciSyncFenceWait(&appBuffer->eofFence, cpuWaitContext, 1000*1000);
        if(err != NvSciError_Success) {
            LOG_ERR("NvSciSyncFenceWait failed\n");
            goto fail;
        }

        NvSciSyncFenceClear(&appBuffer->eofFence);
        NvSciSyncFenceClear(&appBuffer->preFence);

        err = NvSciSyncObjGenerateFence(preSyncObj, &appBuffer->preFence);
        if (err != NvSciError_Success)
        {
            LOG_ERR("NvSciSyncObjGenerateFence function failed \n");
            goto fail;
        }

        status = NvMediaIEPInsertPreNvSciSyncFence(encoderCtx, &appBuffer->preFence);
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("NvMediaIEPInsertPreNvSciSyncFence function failed \n");
            goto fail;
        }

        err = NvSciSyncObjSignal(preSyncObj);
        if (err != NvSciError_Success)
        {
            LOG_ERR("Signalling function failed \n");
            goto fail;
        }

        if (!args.preFetchBuffer) {
            LOG_DBG("main: Reading YUV frame %d from file %s to image surface "
                    "location: %p. (W:%d, H:%d)\n",
                    YUVFrameNum, args.infile, appBuffer->bufObj,
                    args.configParams.encodeWidth, args.configParams.encodeHeight);

            /* Read a frame's worth of data from input file */
#if !NV_IS_SAFETY
            if ((!args.drcBufRealloc) || (resolutionChangeDone != true)) {
#endif
                status = ReadInput( args.infile,
                    YUVFrameNum % framesNum,
                    args.configParams.encodeWidth,
                    args.configParams.encodeHeight,
                    appBuffer->bufObj,
                    inputFileChromaFormat,
                    uvOrderFlag,  //inputUVOrderFlag
                    pixelAlignment);
                if(status != NVMEDIA_STATUS_OK) {
                    LOG_ERR("readYUVFile failed\n");
                    NvQueuePut(inputQueue,
                            (void *)&appBuffer,
                            100);
                    goto fail;
                }
#if !NV_IS_SAFETY
            } else {
                status = ReadInput( args.infiledrc,
                    frameCounterDRC % framesNumChangedRes,
                    args.configParams.encodeWidth,
                    args.configParams.encodeHeight,
                    appBuffer->bufObj,
                    inputFileChromaFormat,
                    uvOrderFlag,  //inputUVOrderFlag
                    pixelAlignment);
                if(status != NVMEDIA_STATUS_OK) {
                    LOG_ERR("readYUVFile failed\n");
                    NvQueuePut(inputQueueChangedRes,
                            (void *)&appBuffer,
                            100);
                    goto fail;
                }
                frameCounterDRC++;
            }
#endif
            LOG_DBG("main: ReadYUVFrame %d done\n", YUVFrameNum);
        }

        /* The fence wait operation that precedes this ensures that the buffer
           can be written to and the semaphore ensures that the output associated
           with the buffer has been read */
        NvSemaphoreDecrement(outBufAvailableSema, NV_TIMEOUT_INFINITE);
        if (encOutThreadArgs.encOutThreadExit) {
            LOG_DBG("main: Thread exit requested\n");
            NvQueuePut(appBuffer->queue,
                    (void *)&appBuffer,
                    100);
            goto fail;
        }
#if !NV_IS_SAFETY
        if (resolutionChange == true) {
            NvMediaEncodeDRCConfig DRCConfig_params;
            DRCConfig_params.ulDRCWidth = args.dynResFrameWidth;
            DRCConfig_params.ulDRCHeight = args.dynResFrameHeight;
            NvMediaStatus nvmediaStatus = NvMediaIEPSetAttribute(encoderCtx,
                                                                 NvMediaEncSetAttr_DRCParams,
                                                                 sizeof(NvMediaEncodeDRCConfig),
                                                                 (void *)&DRCConfig_params);
            if (nvmediaStatus != NVMEDIA_STATUS_OK) {
                LOG_ERR("main: Failure while setting DRC Parameter");
                goto fail;
            }
            if (args.videoCodec == CODEC_H264) {
                args.configH264Params.repeatSPSPPS = NVMEDIA_ENCODE_SPSPPS_REPEAT_IDR_FRAMES;
                nvmediaStatus = NvMediaIEPSetConfiguration(encoderCtx, &args.configH264Params);
            } else if (args.videoCodec == CODEC_H265) {
                args.configH265Params.repeatSPSPPS = NVMEDIA_ENCODE_SPSPPS_REPEAT_IDR_FRAMES;
                nvmediaStatus = NvMediaIEPSetConfiguration(encoderCtx, &args.configH265Params);
            }
            if(status != NVMEDIA_STATUS_OK) {
                LOG_ERR("main: SetConfiguration failed\n");
                goto fail;
            }
            if (fscanf(dynResFile, "%d %d %d", &args.dynResFrameNum, &args.dynResFrameWidth, &args.dynResFrameHeight) != 3) {
                args.dynResFrameNum = 0;
                args.dynResFrameWidth = args.configParams.encodeWidth;
                args.dynResFrameHeight = args.configParams.encodeHeight;
            }
        }
        if (args.configParams.enableROIEncode) {
            if (0U != strlen(args.ROIParamFileName)) {
                uint32_t ROIIndex = 0;

                // Read ROI params from file
                fscanf(ROIParamFile, "%u", &inExtraData.ulNumROIRegions);
                while (ROIIndex < inExtraData.ulNumROIRegions)
                {
                    fscanf(ROIParamFile, "%d", &inExtraData.ROIParams[ROIIndex].lQPdelta);
                    fscanf(ROIParamFile, "%hu", &inExtraData.ROIParams[ROIIndex].roiRect.x0);
                    fscanf(ROIParamFile, "%hu", &inExtraData.ROIParams[ROIIndex].roiRect.y0);
                    fscanf(ROIParamFile, "%hu", &inExtraData.ROIParams[ROIIndex].roiRect.x1);
                    fscanf(ROIParamFile, "%hu", &inExtraData.ROIParams[ROIIndex].roiRect.y1);
                    if ((inExtraData.ROIParams[ROIIndex].roiRect.x0 > inExtraData.ROIParams[ROIIndex].roiRect.x1) ||
                        (inExtraData.ROIParams[ROIIndex].roiRect.y0 > inExtraData.ROIParams[ROIIndex].roiRect.y1) ||
                        ((unsigned)inExtraData.ROIParams[ROIIndex].roiRect.x1 > args.configParams.encodeWidth - 1) ||
                        ((unsigned)inExtraData.ROIParams[ROIIndex].roiRect.y1 > args.configParams.encodeHeight - 1))
                    {
                        LOG_ERR("main: Invalid parameters of ROI \n");
                        goto fail;
                    }
                    ROIIndex++;
                }
                inExtraData.bSEIforROIEnable = NVMEDIA_FALSE;
                inExtraData.EncodeParamsFlag |= NvMediaVideoEncFrame_ROIParams;
            }
        }
        if (strlen(args.qpDeltaFileBaseName) != 0) {
            char fileName[FILE_NAME_SIZE];
            if (snprintf(fileName, FILE_NAME_SIZE, "%s_%05d.bin", args.qpDeltaFileBaseName, frameCounter) < 0) {
                goto  fail;
            }
            FILE *fp = fopen(fileName, "r");
            if (fp == NULL) {
                LOG_ERR("main: Failed opening '%s' file for reading\n", fileName);
                goto fail;
            }
            fseek(fp, 0, SEEK_END);
            QPDeltaBufferSize = ftell(fp);
            if (QPDeltaBufferSize < getH265QPDeltaBufferSize(args.configParams.encodeWidth, args.configParams.encodeHeight)) {
                LOG_ERR("main: QPDeltaBufferSize not sufficient\n");
                goto fail;
            }
            fseek(fp, 0, SEEK_SET);
            inExtraData.QPDeltaBuffer = ringEntry.qp_delta_buffer[frameCounter % IMAGE_BUFFERS_POOL_SIZE];
            inExtraData.QPDeltaBufferSize = QPDeltaBufferSize;
            fread(inExtraData.QPDeltaBuffer, QPDeltaBufferSize, 1, fp);
            fclose(fp);
            inExtraData.EncodeParamsFlag |= NvMediaVideoEncFrame_QPDeltaBuffer;
        }
        inExtraData.ulExtraDataSize = sizeof(NvMediaEncodeInputExtradata);
        NvMediaIEPSetInputExtraData(encoderCtx, &inExtraData);
#endif

        LOG_DBG("main: Encoding frame #%d\n", frameCounter);
#if ENABLE_PROFILING
        if (args.profileEnable)
        {
            if (frameCounter - 1)
            {
                NvpRateLimitWait(&rateLimitInfo);
            }
            else
            {
                NvpMarkPeriodicExecStart(&rateLimitInfo);
            }
            feedFrameStartTimeMark = NvpGetTimeMark();
        }
#endif
        status = NvMediaIEPFeedFrame(encoderCtx,
                appBuffer->bufObj,
                &encodePicParams,
                args.instanceId);
        if(status != NVMEDIA_STATUS_OK) {
            LOG_ERR("main: NvMediaIEPFeedFrame failed: %x\n", status);
            NvQueuePut(appBuffer->queue,
                    (void *)&appBuffer,
                    100);
            goto fail;
        }

        status = NvMediaIEPGetEOFNvSciSyncFence(encoderCtx, eofSyncObj,
                &appBuffer->eofFence);
        if(status != NVMEDIA_STATUS_OK) {
            LOG_ERR("main: NvMediaIEPGetEOFNvSciSyncFence failed: %x\n", status);
            NvQueuePut(appBuffer->queue,
                    (void *)&appBuffer,
                    100);
            goto fail;
        }
#if ENABLE_PROFILING
        if (args.profileEnable)
        {
            feedFrameStopTimeMark = NvpGetTimeMark();
            NvpRecordSample(&submissionLatencies, feedFrameStartTimeMark, feedFrameStopTimeMark);
        }
#endif

        //Signal the encoder output thread
        NvSemaphoreIncrement(feedFrameDoneSema);

        // Put the image back in queue
        NvQueuePut(appBuffer->queue,
                (void *)&appBuffer,
                100);

#if ENABLE_PROFILING
        if (!args.profileEnable)
#endif
        {
            if (((frameCounter == 1)
#if !NV_IS_SAFETY
                  || (resolutionChange == true)
#endif
                  ) && ((args.videoCodec == CODEC_H264) || (args.videoCodec == CODEC_H265))) {
                NvMediaNalData nalData;

                // Get SPS
                status = NvMediaIEPGetAttribute(encoderCtx, NvMediaEncAttr_GetSPS, sizeof(NvMediaNalData), (void *)&nalData);
                if (status != NVMEDIA_STATUS_OK) {
                    LOG_ERR("main: error getting SPS using NvMediaIEPGetAttribute \n");
                }
                // Get PPS
                status = NvMediaIEPGetAttribute(encoderCtx, NvMediaEncAttr_GetPPS, sizeof(NvMediaNalData), (void *)&nalData);
                if (status != NVMEDIA_STATUS_OK) {
                    LOG_ERR("main: error getting PPS using NvMediaIEPGetAttribute \n");
                }
#if !NV_IS_SAFETY
                resolutionChange = false;
#endif
            }
        }

        // Next frame
        frameCounter++;
        if ((args.framesToBeEncoded && frameCounter == (args.framesToBeEncoded + 1))) {
            nextFrameFlag = false;
        } else {
            if(frameCounter == (args.loopCount * framesNum) + 1) {
                nextFrameFlag = false;
            }
        }
    }

Done:
    NvSemaphoreIncrement(feedFrameDoneSema);

    totalFramesEncoded = frameCounter-args.startFrame;

    if(outputThread != NULL) {
        status = NvThreadDestroy(outputThread);
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("%s: Failed to destroy encoder output thread\n", __func__);
        }
        outputThread = NULL;
    }
    if (encOutThreadArgs.errorOccurred == 1U) {
        goto fail;
    }

    if (args.eventDataRecorderMode == true) {
        int i, numFrames;
        int32_t endIdx = encOutThreadArgs.endIdx;
        int32_t startIdx = encOutThreadArgs.startIdx;
        numFrames = endIdx - startIdx + 1;
        if (numFrames <= 0)
            numFrames += eventDataRecorderRingSize;
        //store the bitstream
        LOG_MSG("main: eventDataRecorder mode, store %d frames (%f seconds) bitstream\n", numFrames, (float)numFrames/args.configH264Params.gopLength);
        for (i=0; i<numFrames; i++) {
            if (eventDataRecorderDataRing[startIdx].dataBuff) {
                if(fwrite(eventDataRecorderDataRing[startIdx].dataBuff, eventDataRecorderDataRing[startIdx].dataSize, 1, outputFile) != 1) {
                    LOG_ERR("main: EventDataRecorder Mode, Error writing %d bytes for the %d frame\n", eventDataRecorderDataRing[startIdx].dataSize, startIdx);
                }
                free (eventDataRecorderDataRing[startIdx].dataBuff);
                eventDataRecorderDataRing[startIdx].dataBuff = NULL;
                eventDataRecorderDataRing[startIdx].dataSize = 0;
                startIdx ++;
                if (startIdx >= eventDataRecorderRingSize)
                    startIdx = 0;
            } else {
                LOG_ERR("main: EventDataRecorder mode, Error in writing the %d frame, buffer is empty\n", startIdx);
            }
        }
        if(eventDataRecorderDataRing){
            free(eventDataRecorderDataRing);
            eventDataRecorderDataRing = NULL;
        }
    }
#if ENABLE_PROFILING
    if (args.profileEnable)
    {
        NvpStatus_t status;
        NvpPerfStats_t initLatStats;
        NvpAggregatePerfData(&totalInitPerfData, initLatenciesArray,
                numOfLatenciesToBeAggregated);
        // Initialization Latency includes time taken for IEP Create, Buffer allocation and buffer
        // registration. It however does not include time taken for NvSci Sync and Buf object as
        // they do not contribute significantly to the initialization latency.
        status = NvpCalcStats(&totalInitPerfData, &initLatStats, USEC);
        if (status != NVP_PASS)
        {
            LOG_ERR("main: NvpCalcStats failed with Error %d\n", status);
        }
        NvpPrintStats(&totalInitPerfData, &initLatStats, USEC, "Initialization Latency", false);
        status = NvpDumpData(&totalInitPerfData);
        if (status != NVP_PASS)
        {
            LOG_ERR("main: NvpDumpData failed with Error %d\n", status);
        }
        NvpPerfStats_t submisionLatStats;
        status = NvpCalcStats(&submissionLatencies, &submisionLatStats, USEC);
        if (status != NVP_PASS)
        {
            LOG_ERR("main: NvpCalcStats failed with Error %d\n", status);
        }
        NvpPrintStats(&submissionLatencies, &submisionLatStats, USEC, "Submission Latency", false);
        status = NvpDumpData(&submissionLatencies);
        if (status != NVP_PASS)
        {
            LOG_ERR("main: NvpDumpData failed with Error %d\n", status);
        }
        NvpPerfStats_t executionLatStats;
        status = NvpCalcStats(&encOutThreadArgs.latencies, &executionLatStats, USEC);
        if (status != NVP_PASS)
        {
            LOG_ERR("main: NvpCalcStats failed with Error %d\n", status);
        }
        NvpPrintStats(&encOutThreadArgs.latencies, &executionLatStats, USEC, "Execution Latency", false);
        LOG_MSG("\nEncode FPS achieved: %.4f\n", 1000000.0 / executionLatStats.mean);
        status = NvpDumpData(&encOutThreadArgs.latencies);
        if (status != NVP_PASS)
        {
            LOG_ERR("main: NvpDumpData failed with Error %d\n", status);
        }
        for (k = 0; k< 2*IMAGE_BUFFERS_POOL_SIZE_MAX + 2; k++) {
            NvpDestroyPerfData(&bufRegisterData[k]);
        }
        NvpDestroyPerfData(&totalInitPerfData);
        NvpDestroyPerfData(&submissionLatencies);
        NvpDestroyPerfData(&encOutThreadArgs.latencies);
        if (args.profileTestEnable)
        {
            if (args.initLat < initLatStats.mean)
            {
                 LOG_ERR("main: Initialization latency fail, Reference: %.4fus, Measured: %.4fus\n", args.initLat, initLatStats.mean);
                 goto fail;
            }
            if (args.submitLat < submisionLatStats.mean)
            {
                 LOG_ERR("main: Submission latency fail, Reference: %.4fus, Measured: %.4fus\n", args.submitLat, submisionLatStats.mean);
                 goto fail;
            }
            if (args.execLat < executionLatStats.mean)
            {
                 LOG_ERR("main: Execution latency fail, Reference: %.4fus, Measured: %.4fus\n", args.execLat, executionLatStats.mean);
                 goto fail;
            }
        }
    }
#endif
#if !NV_IS_SAFETY
    if (((args.videoCodec == CODEC_AV1) && ((args.configAV1Params.features & NVMEDIA_ENCODE_CONFIG_AV1_ENABLE_PROFILING) > 1)) ||
        ((args.videoCodec == CODEC_VP9) && ((args.configVP9Params.features & NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_PROFILING) > 1)) ||
        ((args.videoCodec == CODEC_H264) && ((args.configH264Params.features & NVMEDIA_ENCODE_CONFIG_H264_ENABLE_PROFILING) > 1)) ||
        ((args.videoCodec == CODEC_H265) && ((args.configH265Params.features & NVMEDIA_ENCODE_CONFIG_H265_ENABLE_PROFILING) > 1)))
    {
        LOG_MSG("\nEncode FPS achieved for 1 GHz clock: %.4f\n", 1000000000.0* totalFramesEncoded / args.sumCycleCount);
    }
#endif
    //Get the bitrate info
    LOG_MSG("\nTotal encoded frames = %d, avg. bitrate=%d, maxBitrate=%d, minBitrate=%d\n",
            totalFramesEncoded,
            (int)(encOutThreadArgs.totalBytes*8*30/totalFramesEncoded),
            encOutThreadArgs.maxBitrate*8, encOutThreadArgs.minBitrate*8);
    if (args.crcoption.crcGenMode){
        LOG_MSG("\n***crc gold file %s has been generated***\n", args.crcoption.crcFilename);
    } else if (args.crcoption.crcCheckMode){
        LOG_MSG("\n***crc checking with file %s is successful\n", args.crcoption.crcFilename);
    }

    LOG_MSG("\n***ENCODING PROCESS ENDED SUCCESSFULY***\n");
    testPass = true;

fail:
    if(testPass != true) {
        encOutThreadArgs.encError = 1;
        if(feedFrameDoneSema) {
            NvSemaphoreIncrement(feedFrameDoneSema);
        }
    }
    if(args.configH264Params.h264VUIParameters)
        free(args.configH264Params.h264VUIParameters);
    if(args.configH265Params.h265VUIParameters)
        free(args.configH265Params.h265VUIParameters);

    if(outputFile) {
        fclose(outputFile);
    }

    if (inputQueue) {
        deregisterAndFreeBuffers(&args,
                                 encoderCtx,
                                 &inputQueue,
                                 countNvSciBufObj,
                                 &cpuWaitContext);
        NvQueueDestroy(inputQueue);
    }
#if !NV_IS_SAFETY
    if (inputQueueChangedRes) {
        deregisterAndFreeBuffers(&args,
                                 encoderCtx,
                                 &inputQueueChangedRes,
                                 countNvSciBufObjChangedRes,
                                 &cpuWaitContext);
        NvQueueDestroy(inputQueueChangedRes);
    }
#endif
    /* Unregister NvSciSyncObj */
    if (eofSyncObj) {
        status = NvMediaIEPUnregisterNvSciSyncObj(encoderCtx, eofSyncObj);
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("%s: Failed to unregister EOF NvSciSyncObj: %u\n", __func__, status);
        }

        /* Free NvSciSyncObj */
        NvSciSyncObjFree(eofSyncObj);
    }

    if (preSyncObj) {
        status = NvMediaIEPUnregisterNvSciSyncObj(encoderCtx, preSyncObj);
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("%s: Failed to unregister PRE NvSciSyncObj: %u\n", __func__, status);
        }

        /* Free NvSciSyncObj */
        NvSciSyncObjFree(preSyncObj);
    }

#if !NV_IS_SAFETY
    if (strlen(args.qpDeltaFileBaseName) != 0) {
        for (int i = 0; i < IMAGE_BUFFERS_POOL_SIZE; i++) {
            if (ringEntry.qp_delta_buffer[i] != NULL) {
                free(ringEntry.qp_delta_buffer[i]);
            }
        }
    }
    if (ROIParamFile) {
        fclose(ROIParamFile);
    }
#endif

    if (NULL != cpuWaitContext) {
        NvSciSyncCpuWaitContextFree(cpuWaitContext);
    }

    if (NULL != syncModule) {
        NvSciSyncModuleClose(syncModule);
    }

    if (NULL != bufModule) {
        NvSciBufModuleClose(bufModule);
    }

    if(feedFrameDoneSema != NULL) {
        NvSemaphoreDestroy(feedFrameDoneSema);
    }

    if(outBufAvailableSema != NULL) {
        NvSemaphoreDestroy(outBufAvailableSema);
    }

    if(outputThread != NULL) {
        status = NvThreadDestroy(outputThread);
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("%s: Failed to destroy encoder output thread\n", __func__);
        }
    }

    if(encoderCtx) {
        NvMediaIEPDestroy(encoderCtx);
    }

    /* Free reconciled attribute list */
    NvSciBufAttrListFree(bufReconciledList);

    //In case of negative testing test fail is treated as PASS and vice-versa
    if (args.negativeTest) {
        testPass = !testPass;
    }

    if (!testPass) {
        LOG_MSG("total failures: 1 \n");
        return -1;
    }
    else {
        LOG_MSG("total failures: 0 \n");
        return 0;
    }
}

static NvMediaStatus
MapInputFileChromaToSurfaceChroma (
    ChromaFormat        inputFileChromaFormat,
    ChromaFormat        *surfaceChromaFormat,
    NvSciBufSurfBPC     *bitdepth
)
{
    switch (inputFileChromaFormat)
    {
        case YUV420P_8bit:
            *surfaceChromaFormat = YUV420SP_8bit;
            *bitdepth = NvSciSurfBPC_8;
            break;
        case YUV444P_8bit:
            *surfaceChromaFormat = YUV444SP_8bit;
            *bitdepth = NvSciSurfBPC_8;
            break;
        case YUV420P_10bit:
            *surfaceChromaFormat = YUV420SP_10bit;
            *bitdepth = NvSciSurfBPC_10;
            break;
        case YUV444P_10bit:
            *surfaceChromaFormat = YUV444SP_10bit;
            *bitdepth = NvSciSurfBPC_10;
            break;
        case YUV420P_12bit:
            *surfaceChromaFormat = YUV420SP_12bit;
            *bitdepth = NvSciSurfBPC_12;
            break;
        case YUV444P_12bit:
            *surfaceChromaFormat = YUV444SP_12bit;
            *bitdepth = NvSciSurfBPC_12;
            break;
        case YUV420P_16bit:
            *surfaceChromaFormat = YUV420SP_16bit;
            *bitdepth = NvSciSurfBPC_16;
            break;
        case YUV444P_16bit:
            *surfaceChromaFormat = YUV444SP_16bit;
            *bitdepth = NvSciSurfBPC_16;
            break;
        default:
            return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return NVMEDIA_STATUS_OK;
}

static long
GetInputFileSizeBytes(char *inputFileName)
{
    FILE *streamFile = NULL;
    long fileLength = 0;

    streamFile = fopen(inputFileName, "rb");
    if(!streamFile) {
        LOG_ERR("%s: Error opening '%s' for reading\n", __func__, inputFileName);
        return 0;
    }

    fseek(streamFile, 0, SEEK_END);
    fileLength = ftell(streamFile);
    fclose(streamFile);

    return fileLength;
}

static uint32_t
GetInputFrameSizeBytes(
        ChromaFormat inputChromaFormat,
        uint32_t width,
        uint32_t height)
{
    switch(inputChromaFormat)
    {
        case YUV420P_8bit:
        case YUV420SP_8bit:
            return width * height * 3 / 2;
        case YUV420P_10bit:
        case YUV420SP_10bit:
            return width * height * 3;
        case YUV444P_8bit:
        case YUV444SP_8bit:
            return width * height * 3;
        case YUV444P_10bit:
        case YUV444SP_10bit:
            return width * height * 6;
        case YUV420P_12bit:
        case YUV420SP_12bit:
            return width * height * 3;
        default:
            return 0;
    }
}

static NvMediaStatus
GetInputFormatAttributes(
        uint32_t               inputFileFormat,
        ChromaFormat           *inputChromaFormat,
        bool                   *uvOrdering,
        PixelAlignment         *alignment,
        NvSciBufSurfSampleType *subsampleType
)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    *alignment = LSB_ALIGNED;
    /* Chroma ordering of UV */
    *uvOrdering = true;

    /* Mapping of input file format to input surface chroma format */
    switch(inputFileFormat) {
        case 0:
            *inputChromaFormat = YUV420P_8bit;
            *subsampleType = NvSciSurfSampleType_420;
            break;
        case 1:
            *inputChromaFormat = YUV420P_8bit;
            *subsampleType = NvSciSurfSampleType_420;
            /* chroma component's ordering will be VU for YV12 input format */
            *uvOrdering = false;
            break;
        case 3:
            *inputChromaFormat = YUV444P_8bit;
            *subsampleType = NvSciSurfSampleType_444;
            break;
        case 4:
            *inputChromaFormat = YUV420P_10bit;
            *subsampleType = NvSciSurfSampleType_420;
            break;
        case 5:
            *inputChromaFormat = YUV444P_10bit;
            *subsampleType = NvSciSurfSampleType_444;
            break;
        case 6:
            *inputChromaFormat = YUV420P_10bit;
            *subsampleType = NvSciSurfSampleType_420;
            *alignment = MSB_ALIGNED;
            break;
        case 7:
            *inputChromaFormat = YUV444P_10bit;
            *subsampleType = NvSciSurfSampleType_444;
            *alignment = MSB_ALIGNED;
            break;
        case 8:
            *inputChromaFormat = YUV420P_12bit;
            *subsampleType = NvSciSurfSampleType_420;
            break;
        default:
            status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return status;
}
