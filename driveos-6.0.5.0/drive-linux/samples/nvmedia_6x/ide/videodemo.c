/*
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <ctype.h>
#include <signal.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include "cmdline.h"
#include "log_utils.h"
#include "misc_utils.h"
#include "nvmedia_core.h"
#include "nvmedia_parser.h"
#include "video_utils.h"
#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvscierror.h"
#include "scibuf_utils.h"
#include "nvmedia_ide.h"


/* Max number of decoder reference buffers */
#define MAX_DEC_REF_BUFFERS         (16)
/* Max number of buffers for display */
#define MAX_DISPLAY_BUFFERS         (4)
/* Max number of buffers between decoder and deinterlace */
#define MAX_DEC_DEINTER_BUFFERS     (MAX_DISPLAY_BUFFERS)
/* Total number of buffers for decoder to operate.*/
#define MAX_DEC_BUFFERS             (MAX_DEC_REF_BUFFERS + MAX_DEC_DEINTER_BUFFERS + 1)
#define READ_SIZE                   (32 * 1024)
// For VP8 ivf file parsing
#define IVF_FILE_HDR_SIZE           32
#define IVF_FRAME_HDR_SIZE          12

/* NvMediaIDE only supports input surface formats which have 2 planes */
#define IDE_APP_MAX_INPUT_PLANE_COUNT 2U
#define IDE_APP_BASE_ADDR_ALIGN 256U

#define MAX_FILE_PATH_LENGTH 256

typedef struct _VideoDemoTestCtx {
    NvMediaParser           *parser;
    NvMediaParserSeqInfo    nvsi;
    NvMediaParserParams     nvdp;
    NvMediaVideoCodec       eCodec;

    //  Stream params
    FILE                    *file;
    char                    *filename;
    bool                    bVC1SimpleMainProfile;
    char                    *OutputYUVFilename;
    int64_t                 fileSize;
    bool                    bRCVfile;

    // Decoder params
    int                      decodeWidth;
    int                      decodeHeight;
    int                      displayWidth;
    int                      displayHeight;
    NvMediaIDE               *decoder;
    int                      decodeCount;
    float                    totalDecodeTime;
    bool                     stopDecoding;
    bool                     showDecodeTimimg;
    int                      numFramesToDecode;
    int                      loop;

    // Picture buffer params
    int                      nBuffers;
    int                      nPicNum;
    int                      sumCompressedLen;
    FrameBuffer              RefFrame[MAX_DEC_BUFFERS];
    // Display params
    int                      lDispCounter;
    double                   frameTimeUSec;
    float                    aspectRatio;
    bool                     videoFullRangeFlag;
    int                      colorPrimaries;
    bool                     positionSpecifiedFlag;
    NvMediaRect              position;
    unsigned int             depth;
    int                      monitorWidth;
    int                      monitorHeight;
    unsigned int             filterQuality;
    NvMediaDecoderInstanceId instanceId;
    uint32_t                 bMonochrome;
    int                      FrameCount;

    // Crc params
    uint32_t                 checkCRC;
    uint32_t                 generateCRC;
    uint32_t                 cropCRC;
    FILE                     *fpCrc;
    uint32_t                refCrc;
    char                    crcFilePath[MAX_FILE_PATH_LENGTH];
    uint32_t                CRCResult;
    uint32_t                YUVSaveComplete;
    uint32_t                CRCGenComplete;
    // Decoder Profiling
    uint32_t                decProfiling;
    uint32_t                setAnnexBStream;
    uint8_t                 av1annexBStream;
    uint32_t                setOperatingPoint;
    uint8_t                 av1OperatingPoint;
    uint32_t                setOutputAllLayers;
    uint8_t                 av1OutputAllLayers;
    uint32_t                setMaxRes;
    uint8_t                 enableMaxRes;
    NvSciSyncModule         syncModule;
    NvSciBufModule          bufModule;
    NvSciSyncCpuWaitContext cpuWaitContext;
    NvSciBufAttrList        bufAttributeList;
    NvSciSyncAttrList       ideSignalerAttrList;
    NvSciSyncAttrList       cpuWaiterAttrList;
    NvSciSyncAttrList       ideWaiterAttrList;
    NvSciSyncObj            eofSyncObj;
    bool                    alternateCreateAPI;
} VideoDemoTestCtx;

static uint32_t signal_stop = 0;

#ifdef NVMEDIA_QNX
/* Used for calculating explicit delays to achieve correct FPS
 * g_timeBase : timestamp when first frame is posted
 * g_timeNow  : timestamp before posting subsequent frame
 */
uint64_t g_timeBase = UINT64_MAX, g_timeNow = UINT64_MAX;

/* Count of frames after which measured FPS value is printed */
#define FPS_DISPLAY_PERIOD 500

/* For measuring the frames per sec after every FPS_DISPLAY_PERIOD frames */
uint64_t g_tFPS;
#endif

int32_t       cbBeginSequence(void *ptr, const NvMediaParserSeqInfo *pnvsi);
NvMediaStatus cbDecodePicture(void *ptr, NvMediaParserPictureData *pd);
NvMediaStatus cbDisplayPicture(void *ptr, NvMediaRefSurface *p, int64_t llPts);
void          cbUnhandledNALU(void *ptr, const uint8_t *buf, int32_t size);
NvMediaStatus cbAllocPictureBuffer(void *ptr, NvMediaRefSurface **p);
void          cbRelease(void *ptr, NvMediaRefSurface *p);
void          cbAddRef(void *ptr, NvMediaRefSurface *p);
NvMediaStatus cbGetBackwardUpdates(void *ptr, NvMediaVP9BackwardUpdates *backwardUpdate);

int     Init(VideoDemoTestCtx *ctx, TestArgs *testArgs);
void    Deinit(VideoDemoTestCtx *parser);
int     Decode(VideoDemoTestCtx *parser);
void    Stats(VideoDemoTestCtx *parser);

int     StreamVC1SimpleProfile(VideoDemoTestCtx *ctx);

static char *Strcasestr(char *haystack, char *needle)
{
    char *haystack_temp, *needle_temp, *res;
    int pos;

    if(!haystack || !strlen(haystack) || !needle || !strlen(needle))
        return NULL;

    haystack_temp = malloc(strlen(haystack) + 1);
    if(!haystack_temp)
        return NULL;

    needle_temp = malloc(strlen(needle) + 1);
    if(!needle_temp) {
        free(haystack_temp);
        return NULL;
    }

    pos = 0;
    while(haystack[pos]) {
        haystack_temp[pos] = toupper(haystack[pos]);
        pos++;
    }
    haystack_temp[pos] = 0;

    pos = 0;
    while(needle[pos]) {
        needle_temp[pos] = toupper(needle[pos]);
        pos++;
    }
    needle_temp[pos] = 0;

    res = strstr(haystack_temp, needle_temp);
    res = res ? (res - haystack_temp) + haystack : NULL;

    free(haystack_temp);
    free(needle_temp);

    return res;
}

static NvMediaStatus UpdateNvMediaSurfacePictureInfoH264(VideoDemoTestCtx *ctx, NvMediaPictureInfoH264 *pictureInfo)
{
    uint32_t i;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    for (i = 0; i < 16; i++)
    {
        NvMediaReferenceFrameH264* dpb_out = &pictureInfo->referenceFrames[i];
        FrameBuffer* picbuf = (FrameBuffer *)dpb_out->surface;
        //populate  the prefences
        if (picbuf != NULL)
        {
            status = NvMediaIDEInsertPreNvSciSyncFence(ctx->decoder, &picbuf->preFence);
            if (status != NVMEDIA_STATUS_OK)
            {
                LOG_ERR("%s  %d  Call to NvMediaIDEInsertPreNvSciSyncFence failed with status = %d \n", __func__, __LINE__, status);
                return status;
            }
        }
        dpb_out->surface = picbuf ? (NvMediaRefSurface *)(picbuf->videoSurface) : NULL;
    }
    return status;
}

static NvMediaStatus UpdateNvMediaSurfacePictureInfoH265(VideoDemoTestCtx *ctx, NvMediaPictureInfoH265 *pictureInfo)
{
    uint32_t i;
    FrameBuffer* picbuf;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    for (i = 0; i < 16; i++)
    {
        picbuf = (FrameBuffer *)pictureInfo->RefPics[i];
        //populate  the prefences
        if (picbuf != NULL)
        {
            status = NvMediaIDEInsertPreNvSciSyncFence(ctx->decoder, &picbuf->preFence);
            if (status != NVMEDIA_STATUS_OK)
            {
                LOG_ERR("%s  %d  Call to NvMediaIDEInsertPreNvSciSyncFence failed with status = %d \n", __func__, __LINE__, status);
                return status;
            }
        }
        pictureInfo->RefPics[i] = picbuf ? (NvMediaRefSurface *)(picbuf->videoSurface) : NULL;
    }
    return status;
}

static NvMediaStatus UpdateNvMediaSurfacePictureInfoMPEG(VideoDemoTestCtx *ctx, NvMediaPictureInfoMPEG1Or2 *pictureInfo, NvSciBufObj pCurrPic)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    if (pictureInfo->forward_reference)
    {
        status = NvMediaIDEInsertPreNvSciSyncFence(ctx->decoder, &(((FrameBuffer *)pictureInfo->forward_reference)->preFence));
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("%s  %d  Call to NvMediaIDEInsertPreNvSciSyncFence failed with status = %d \n", __func__, __LINE__, status);
            return status;
        }
        pictureInfo->forward_reference = (NvMediaRefSurface *)(((FrameBuffer *)pictureInfo->forward_reference)->videoSurface);
    }
    else
        pictureInfo->forward_reference = (NvMediaRefSurface *)pCurrPic;

    if (pictureInfo->backward_reference)
    {
        status = NvMediaIDEInsertPreNvSciSyncFence(ctx->decoder, &(((FrameBuffer *)pictureInfo->backward_reference)->preFence));
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("%s  %d  Call to NvMediaIDEInsertPreNvSciSyncFence failed with status = %d \n", __func__, __LINE__, status);
            return status;
        }
        pictureInfo->backward_reference = (NvMediaRefSurface *)(((FrameBuffer *)pictureInfo->backward_reference)->videoSurface);
    }
    else
        pictureInfo->backward_reference = (NvMediaRefSurface *)pCurrPic;
    return status;
}

static NvMediaStatus UpdateNvMediaSurfacePictureInfoMPEG4(VideoDemoTestCtx *ctx, NvMediaPictureInfoMPEG4Part2 *pictureInfo, NvSciBufObj pCurrPic)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    if (pictureInfo->forward_reference)
    {
        status = NvMediaIDEInsertPreNvSciSyncFence(ctx->decoder, &(((FrameBuffer *)pictureInfo->forward_reference)->preFence));
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("%s  %d  Call to NvMediaIDEInsertPreNvSciSyncFence failed with status = %d \n", __func__, __LINE__, status);
            return status;
        }
        pictureInfo->forward_reference = (NvMediaRefSurface *)(((FrameBuffer *)pictureInfo->forward_reference)->videoSurface);
    }
    else
        pictureInfo->forward_reference = (NvMediaRefSurface *)pCurrPic;

    if (pictureInfo->backward_reference)
    {
        status = NvMediaIDEInsertPreNvSciSyncFence(ctx->decoder, &(((FrameBuffer *)pictureInfo->backward_reference)->preFence));
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("%s  %d  Call to NvMediaIDEInsertPreNvSciSyncFence failed with status = %d \n", __func__, __LINE__, status);
            return status;
        }
        pictureInfo->backward_reference = (NvMediaRefSurface *)(((FrameBuffer *)pictureInfo->backward_reference)->videoSurface);
    }
    else
        pictureInfo->backward_reference = (NvMediaRefSurface *)pCurrPic;
    return status;
}

static NvMediaStatus UpdateNvMediaSurfacePictureInfoVC1(VideoDemoTestCtx *ctx, NvMediaPictureInfoVC1 *pictureInfo)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    if (pictureInfo->forward_reference)
    {
        status = NvMediaIDEInsertPreNvSciSyncFence(ctx->decoder, &(((FrameBuffer *)pictureInfo->forward_reference)->preFence));
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("%s  %d  Call to NvMediaIDEInsertPreNvSciSyncFence failed with status = %d \n", __func__, __LINE__, status);
            return status;
        }
        pictureInfo->forward_reference = (NvMediaRefSurface *)(((FrameBuffer *)pictureInfo->forward_reference)->videoSurface);
    }

    if (pictureInfo->backward_reference)
    {
        status = NvMediaIDEInsertPreNvSciSyncFence(ctx->decoder, &(((FrameBuffer *)pictureInfo->backward_reference)->preFence));
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("%s  %d  Call to NvMediaIDEInsertPreNvSciSyncFence failed with status = %d \n", __func__, __LINE__, status);
            return status;
        }
        pictureInfo->backward_reference = (NvMediaRefSurface *)(((FrameBuffer *)pictureInfo->backward_reference)->videoSurface);
    }

    if (pictureInfo->range_mapped)
        pictureInfo->range_mapped = (NvMediaRefSurface *)(((FrameBuffer *)pictureInfo->range_mapped)->videoSurface);
    return status;
}

static NvMediaStatus UpdateNvMediaSurfacePictureInfoVP8(VideoDemoTestCtx *ctx, NvMediaPictureInfoVP8 *pictureInfo, NvSciBufObj pCurrPic)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    if (pictureInfo->LastReference)
    {
        status = NvMediaIDEInsertPreNvSciSyncFence(ctx->decoder, &(((FrameBuffer *)pictureInfo->LastReference)->preFence));
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("%s  %d  Call to NvMediaIDEInsertPreNvSciSyncFence failed with status = %d \n", __func__, __LINE__, status);
            return status;
        }
        pictureInfo->LastReference = (NvMediaRefSurface *)(((FrameBuffer *)pictureInfo->LastReference)->videoSurface);
    }
    else
        pictureInfo->LastReference = (NvMediaRefSurface *)pCurrPic;

    if (pictureInfo->GoldenReference)
    {
        status = NvMediaIDEInsertPreNvSciSyncFence(ctx->decoder, &(((FrameBuffer *)pictureInfo->GoldenReference)->preFence));
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("%s  %d  Call to NvMediaIDEInsertPreNvSciSyncFence failed with status = %d \n", __func__, __LINE__, status);
            return status;
        }
        pictureInfo->GoldenReference = (NvMediaRefSurface *)(((FrameBuffer *)pictureInfo->GoldenReference)->videoSurface);
    }
    else
        pictureInfo->GoldenReference = (NvMediaRefSurface *)pCurrPic;

    if (pictureInfo->AltReference)
    {
        status = NvMediaIDEInsertPreNvSciSyncFence(ctx->decoder, &(((FrameBuffer *)pictureInfo->AltReference)->preFence));
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("%s  %d  Call to NvMediaIDEInsertPreNvSciSyncFence failed with status = %d \n", __func__, __LINE__, status);
            return status;
        }
        pictureInfo->AltReference = (NvMediaRefSurface *)(((FrameBuffer *)pictureInfo->AltReference)->videoSurface);
    }
    else
        pictureInfo->AltReference = (NvMediaRefSurface *)pCurrPic;
    return status;
}

static NvMediaStatus UpdateNvMediaSurfacePictureInfoVP9(VideoDemoTestCtx *ctx, NvMediaPictureInfoVP9 *pictureInfo, FrameBuffer *pCurrPic)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    pCurrPic->width = pictureInfo->width;
    pCurrPic->height = pictureInfo->height;

    // First copy ref width/height parameters of pictureInfo then update pointers.
    pictureInfo->ref0_width = pictureInfo->LastReference ? (uint32_t)((FrameBuffer *)pictureInfo->LastReference)->width : pictureInfo->width;
    pictureInfo->ref0_height = pictureInfo->LastReference ? (uint32_t)((FrameBuffer *)pictureInfo->LastReference)->height : pictureInfo->height;
    pictureInfo->ref1_width = pictureInfo->GoldenReference ? (uint32_t)((FrameBuffer *)pictureInfo->GoldenReference)->width : pictureInfo->width;
    pictureInfo->ref1_height = pictureInfo->GoldenReference ? (uint32_t)((FrameBuffer *)pictureInfo->GoldenReference)->height : pictureInfo->height;
    pictureInfo->ref2_width = pictureInfo->AltReference ? (uint32_t)((FrameBuffer *)pictureInfo->AltReference)->width : pictureInfo->width;
    pictureInfo->ref2_height = pictureInfo->AltReference ? (uint32_t)((FrameBuffer *)pictureInfo->AltReference)->height : pictureInfo->height;

    if (pictureInfo->LastReference)
    {
        status = NvMediaIDEInsertPreNvSciSyncFence(ctx->decoder, &(((FrameBuffer *)pictureInfo->LastReference)->preFence));
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("%s  %d  Call to NvMediaIDEInsertPreNvSciSyncFence failed with status = %d \n", __func__, __LINE__, status);
            return status;
        }
        pictureInfo->LastReference = (NvMediaRefSurface *)(((FrameBuffer *)pictureInfo->LastReference)->videoSurface);
    }
    else
        pictureInfo->LastReference = (NvMediaRefSurface *)(pCurrPic->videoSurface);

    if (pictureInfo->GoldenReference)
    {
        status = NvMediaIDEInsertPreNvSciSyncFence(ctx->decoder, &(((FrameBuffer *)pictureInfo->GoldenReference)->preFence));
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("%s  %d  Call to NvMediaIDEInsertPreNvSciSyncFence failed with status = %d \n", __func__, __LINE__, status);
            return status;
        }
        pictureInfo->GoldenReference = (NvMediaRefSurface *)(((FrameBuffer *)pictureInfo->GoldenReference)->videoSurface);
    }
    else
        pictureInfo->GoldenReference = (NvMediaRefSurface *)(pCurrPic->videoSurface);

    if (pictureInfo->AltReference)
    {
        status = NvMediaIDEInsertPreNvSciSyncFence(ctx->decoder, &(((FrameBuffer *)pictureInfo->AltReference)->preFence));
        if (status != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("%s  %d  Call to NvMediaIDEInsertPreNvSciSyncFence failed with status = %d \n", __func__, __LINE__, status);
            return status;
        }
        pictureInfo->AltReference = (NvMediaRefSurface *)(((FrameBuffer *)pictureInfo->AltReference)->videoSurface);
    }
    else
        pictureInfo->AltReference = (NvMediaRefSurface *)(pCurrPic->videoSurface);
    return status;
}

static NvMediaStatus UpdateNvMediaSurfacePictureInfoAV1(VideoDemoTestCtx *ctx, NvMediaPictureInfoAV1 *pictureInfo)
{
//LastReference,Last2Reference,Last3Reference,GoldenReference,BwdReference,AltReference,Alt2Reference
    unsigned int i;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    FrameBuffer* picbuf;
    for (i = 0; i < 7; i++)
    {
        picbuf = (FrameBuffer *)pictureInfo->RefPics[i];
        if (picbuf != NULL)
        {
            status = NvMediaIDEInsertPreNvSciSyncFence(ctx->decoder, &picbuf->preFence);
            if (status != NVMEDIA_STATUS_OK)
            {
                LOG_ERR("%s  %d  Call to NvMediaIDEInsertPreNvSciSyncFence failed with status = %d \n", __func__, __LINE__, status);
                return status;
            }
        }
        pictureInfo->RefPics[i] = picbuf ? (NvMediaRefSurface *)(picbuf->videoSurface) : NULL;
    }

    if (pictureInfo->fgsPic){
        pictureInfo->fgsPic = (NvMediaRefSurface *)(((FrameBuffer *)pictureInfo->fgsPic)->videoSurface);
    }
    return status;
}

int32_t cbBeginSequence(void *ptr, const NvMediaParserSeqInfo *pnvsi)
{
    VideoDemoTestCtx *ctx = (VideoDemoTestCtx*)ptr;
    NvSciError err;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    ChromaFormat surfaceChromaFormat = YUV420SP_8bit;  //Initialize  to avoid Misra-C errors
    unsigned int flags = 0;
    bool surface12bit = false;
    char *codecList[] = {
        "H264",
        "VC1",
        "VC1_ADVANCED",
        "MPEG1",
        "MPEG2",
        "MPEG4",
        "",
        "VP8",
        "H265",
        "VP9",
        "H264_MVC",
        "",
        "AV1"
    };
    char *chroma[] = {
        "Monochrome",
        "4:2:0",
        "4:2:2",
        "4:4:4"
    };

    if(!pnvsi || !ctx) {
        LOG_ERR("cbBeginSequence: Invalid NvMediaParserSeqInfo or VideoDemoTestCtx\n");
        return -1;
    }

    uint32_t decodeBuffers = pnvsi->uDecodeBuffers;

    if (pnvsi->eCodec < 0) {
        LOG_ERR("BeginSequence: Invalid codec type: %d\n", pnvsi->eCodec);
        return 0;
    }

    if((pnvsi->eCodec == NVMEDIA_VIDEO_CODEC_HEVC) || (pnvsi->eCodec == NVMEDIA_VIDEO_CODEC_VP9)) {
        if((pnvsi->uCodedWidth < 180) || (pnvsi->uCodedHeight < 180)) {
            LOG_ERR("BeginSequence: (Width=%d, Height=%d) < (180, 180) NOT SUPPORTED for %s\n",
                    pnvsi->uCodedWidth, pnvsi->uCodedHeight,
                    ((pnvsi->eCodec == NVMEDIA_VIDEO_CODEC_HEVC)?"H265":"VP9"));
                return -1;
            }
    }
    if(pnvsi->eCodec == NVMEDIA_VIDEO_CODEC_AV1) {
        //As per NVDEC5.0_IAS.docx - Check for modification
        if((pnvsi->uCodedWidth < 128) || (pnvsi->uCodedHeight < 128)) {
            LOG_ERR("BeginSequence: (Width=%d, Height=%d) < (128, 128) NOT SUPPORTED for %s\n",
                    pnvsi->uCodedWidth, pnvsi->uCodedHeight,
                    ("AV1"));
                return -1;
            }
    }
    LOG_MSG("BeginSequence: %dx%d (disp: %dx%d) codec: %s decode buffers: %d aspect: %d:%d fps: %f chroma: %s\n",
        pnvsi->uCodedWidth, pnvsi->uCodedHeight, pnvsi->uDisplayWidth, pnvsi->uDisplayHeight,
        codecList[pnvsi->eCodec], pnvsi->uDecodeBuffers, pnvsi->uDARWidth, pnvsi->uDARHeight,
        pnvsi->fFrameRate, pnvsi->eChromaFormat > 3 ? "Invalid" : chroma[pnvsi->eChromaFormat]);

    if (!ctx->frameTimeUSec && pnvsi->fFrameRate >= 5.0 && pnvsi->fFrameRate <= 120.0) {
        ctx->frameTimeUSec = 1000000.0 / pnvsi->fFrameRate;
    }

    if (!ctx->aspectRatio && pnvsi->uDARWidth && pnvsi->uDARHeight) {
        double aspect = (float)pnvsi->uDARWidth / (float)pnvsi->uDARHeight;
        if (aspect > 0.3 && aspect < 3.0)
            ctx->aspectRatio = aspect;
    }


    // Check resolution change
    if (pnvsi->uCodedWidth != ctx->decodeWidth || pnvsi->uCodedHeight != ctx->decodeHeight) {
        NvMediaVideoCodec codec;
        uint32_t maxReferences;
        int i;

        LOG_INFO("BeginSequence: Resolution changed: Old:%dx%d New:%dx%d\n",
                  ctx->decodeWidth, ctx->decodeHeight, pnvsi->uCodedWidth, pnvsi->uCodedHeight);


        if (NULL != ctx->bufModule) {
            NvSciBufModuleClose(ctx->bufModule);
            ctx->bufModule = NULL;
        }
        if (NULL != ctx->syncModule) {
            NvSciSyncModuleClose(ctx->syncModule);
            ctx->syncModule = NULL;
        }
        if (NULL != ctx->cpuWaitContext) {
            NvSciSyncCpuWaitContextFree(ctx->cpuWaitContext);
            ctx->cpuWaitContext = NULL;
        }

        err = NvSciBufModuleOpen(&ctx->bufModule);
        if(err != NvSciError_Success) {
            LOG_ERR("%s: NvSciBuffModuleOpen failed\n",__func__);
            return 0;
        }

        err = NvSciSyncModuleOpen(&ctx->syncModule);
        if(err != NvSciError_Success) {
            LOG_ERR("%s: NvSciBuffModuleOpen failed\n",__func__);
            return 0;
        }

        err = NvSciSyncCpuWaitContextAlloc(ctx->syncModule, &ctx->cpuWaitContext);
        if(err != NvSciError_Success) {
            LOG_ERR("%s: NvSciBuffModuleOpen failed\n",__func__);
            return 0;
        }

        ctx->decodeWidth = pnvsi->uCodedWidth;
        ctx->decodeHeight = pnvsi->uCodedHeight;

        ctx->displayWidth = pnvsi->uDisplayWidth;
        ctx->displayHeight = pnvsi->uDisplayHeight;

        ctx->videoFullRangeFlag = pnvsi->eVideoFullRangeFlag;
        ctx->colorPrimaries = pnvsi->eColorPrimaries;

        if (ctx->decoder) {
            NvMediaIDEDestroy(ctx->decoder);
        }
        LOG_INFO("Create decoder: ");
        switch (pnvsi->eCodec) {
            case NVMEDIA_VIDEO_CODEC_MPEG1:
                codec = NVMEDIA_VIDEO_CODEC_MPEG1;
                LOG_INFO("NVMEDIA_VIDEO_CODEC_MPEG1");
                break;
            case NVMEDIA_VIDEO_CODEC_MPEG2:
                codec = NVMEDIA_VIDEO_CODEC_MPEG2;
                LOG_INFO("NVMEDIA_VIDEO_CODEC_MPEG2");
                break;
            case NVMEDIA_VIDEO_CODEC_MPEG4:
                codec = NVMEDIA_VIDEO_CODEC_MPEG4;
                LOG_INFO("NVMEDIA_VIDEO_CODEC_MPEG4");
                break;
            case NVMEDIA_VIDEO_CODEC_VC1:
                codec = NVMEDIA_VIDEO_CODEC_VC1;
                LOG_INFO("NVMEDIA_VIDEO_CODEC_VC1");
                break;
            case NVMEDIA_VIDEO_CODEC_VC1_ADVANCED:
                codec = NVMEDIA_VIDEO_CODEC_VC1_ADVANCED;
                LOG_INFO("NVMEDIA_VIDEO_CODEC_VC1_ADVANCED");
                break;
            case NVMEDIA_VIDEO_CODEC_H264:
                codec = NVMEDIA_VIDEO_CODEC_H264;
                LOG_INFO("NVMEDIA_VIDEO_CODEC_H264");
                if (pnvsi->uBitDepthLumaMinus8 || pnvsi->uBitDepthChromaMinus8) {
                    flags |= NVMEDIA_IDE_10BIT_DECODE;
                    if (pnvsi->uBitDepthLumaMinus8 > 2 || pnvsi->uBitDepthChromaMinus8 > 2) {
                        surface12bit = true;
                    }
                }
                break;
            case NVMEDIA_VIDEO_CODEC_VP8:
                codec = NVMEDIA_VIDEO_CODEC_VP8;
                LOG_INFO("NVMEDIA_VIDEO_CODEC_VP8");
                break;
            case NVMEDIA_VIDEO_CODEC_HEVC:
                codec = NVMEDIA_VIDEO_CODEC_HEVC;
                LOG_INFO("NVMEDIA_VIDEO_CODEC_HEVC");
                if (pnvsi->uBitDepthLumaMinus8 || pnvsi->uBitDepthChromaMinus8) {
                    flags |= NVMEDIA_IDE_10BIT_DECODE;
                    if (pnvsi->uBitDepthLumaMinus8 > 2 || pnvsi->uBitDepthChromaMinus8 > 2) {
                        surface12bit = true;
                    }
                }

                if(pnvsi->eColorPrimaries == NvMColorPrimaries_BT2020)
                    flags |= NVMEDIA_IDE_PIXEL_REC_2020;
                break;
            case NVMEDIA_VIDEO_CODEC_VP9:
                codec = NVMEDIA_VIDEO_CODEC_VP9;
                LOG_INFO("NVMEDIA_VIDEO_CODEC_VP9");
                if (pnvsi->uBitDepthLumaMinus8 || pnvsi->uBitDepthChromaMinus8) {
                    flags |= NVMEDIA_IDE_10BIT_DECODE;
                    if (pnvsi->uBitDepthLumaMinus8 > 2 || pnvsi->uBitDepthChromaMinus8 > 2) {
                        surface12bit = true;
                    }
                }
                break;
            case NVMEDIA_VIDEO_CODEC_AV1:
                codec = NVMEDIA_VIDEO_CODEC_AV1;
                LOG_INFO("NVMEDIA_VIDEO_CODEC_AV1");
                //flags
                if (pnvsi->uBitDepthLumaMinus8 || pnvsi->uBitDepthChromaMinus8) {
                    flags |= NVMEDIA_IDE_10BIT_DECODE;
                if (pnvsi->uBitDepthLumaMinus8 > 2 || pnvsi->uBitDepthChromaMinus8 > 2) {
                    surface12bit = true;
                    }
                }
                break;
            default:
                LOG_ERR("Invalid decoder type\n");
                return 0;
        }

        maxReferences = (decodeBuffers > 0) ? decodeBuffers - 1 : 0;
        maxReferences = (maxReferences > MAX_DEC_REF_BUFFERS) ? MAX_DEC_REF_BUFFERS : maxReferences;

        LOG_DBG(" Size: %dx%d maxReferences: %d\n", ctx->decodeWidth, ctx->decodeHeight,
            maxReferences);
        if (ctx->decProfiling == true) {
            flags |= NVMEDIA_IDE_PROFILING;
        }
        if (ctx->alternateCreateAPI == true) {
            ctx->decoder = NvMediaIDECreateCtx ();
            if (!ctx->decoder) {
                LOG_ERR("Unable to create decoder\n");
                return 0;
            }
            status = NvMediaIDEInit (ctx->decoder,
                                     codec,
                                     ctx->decodeWidth, // width
                                     ctx->decodeHeight, // height
                                     maxReferences, // maxReferences
                                     pnvsi->uMaxBitstreamSize, //maxBitstreamSize
                                     5, // inputBuffering
                                     flags, // decoder flags
                                     ctx->instanceId  // instance ID
                                     );
            if (status != NVMEDIA_STATUS_OK) {
                LOG_ERR("Unable to Init decoder\n");
                return 0;
            }
        }
        else
        {
            ctx->decoder =     NvMediaIDECreate(codec, // codec
                                               ctx->decodeWidth, // width
                                               ctx->decodeHeight, // height
                                               maxReferences, // maxReferences
                                               pnvsi->uMaxBitstreamSize, //maxBitstreamSize
                                               5, // inputBuffering
                                               flags, // decoder flags
                                               ctx->instanceId); // instance ID

            if (!ctx->decoder) {
                LOG_ERR("Unable to create decoder\n");
                return 0;
            }
        }


        for(i = 0; i < MAX_DEC_BUFFERS; i++) {
            if (ctx->RefFrame[i].videoSurface) {
                NvSciSyncFenceClear(&ctx->RefFrame[i].preFence);
                status = NvMediaIDEUnregisterNvSciBufObj(ctx->decoder, ctx->RefFrame[i].videoSurface);
                if (status != NVMEDIA_STATUS_OK)
                {
                     LOG_ERR("main:failed to uregister NvSciBufObj\n");
                }
                NvSciBufObjFree(ctx->RefFrame[i].videoSurface);
                ctx->RefFrame[i].videoSurface = NULL;
            }
        }

        if (ctx->eofSyncObj != NULL)
        {
            NvMediaIDEUnregisterNvSciSyncObj(ctx->decoder, ctx->eofSyncObj);
            NvSciSyncObjFree(ctx->eofSyncObj);
            ctx->eofSyncObj = NULL;
        }

        ctx->ideSignalerAttrList = NULL;
        ctx->cpuWaiterAttrList = NULL;
        ctx->ideWaiterAttrList = NULL;

        err = NvSciSyncAttrListCreate(ctx->syncModule, &ctx->ideSignalerAttrList);
        if(err != NvSciError_Success) {
              LOG_ERR("%s: Create signaler attr list failed. Error: %d \n", __func__, err);
              return 0;
        }

        status = NvMediaIDEFillNvSciSyncAttrList(ctx->decoder, ctx->ideSignalerAttrList,
                                                 NVMEDIA_SIGNALER);

        if(status != NVMEDIA_STATUS_OK) {
            LOG_ERR("main: Failed to fill signaler attr list.\n");
            return 0;
        }

        err = NvSciSyncAttrListCreate(ctx->syncModule, &ctx->cpuWaiterAttrList);
        if(err != NvSciError_Success) {
            LOG_ERR("%s: Create waiter attr list failed. Error: %d \n", __func__, err);
            return 0;
        }

        err = NvSciSyncAttrListCreate(ctx->syncModule, &ctx->ideWaiterAttrList);
        if(err != NvSciError_Success) {
              LOG_ERR("%s: Create signaler attr list failed. Error: %d \n", __func__, err);
              return 0;
        }

        status = NvMediaIDEFillNvSciSyncAttrList(ctx->decoder, ctx->ideWaiterAttrList,
                                                 NVMEDIA_WAITER);

        if(status != NVMEDIA_STATUS_OK) {
            LOG_ERR("main: Failed to fill signaler attr list.\n");
            return 0;
        }
        // Set NvSciSyncAttrKey_NeedCpuAccess to true and NvSciSyncAttrKey_RequiredPerm to NvSciSyncAccessPerm_WaitOnly
        // TODO Move to proper location if required
        bool cpuWaiter = true;
        NvSciSyncAttrList syncUnreconciledList[3] = {NULL};
        NvSciSyncAttrList syncReconciledList = NULL;
        ctx->eofSyncObj = NULL;
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

        err = NvSciSyncAttrListSetAttrs(ctx->cpuWaiterAttrList, keyValue, 2);
        if(err != NvSciError_Success) {
            LOG_ERR("NvSciSyncAttrListSetAttrs failed\n");
            return 0;
        }

        syncUnreconciledList[0] = ctx->ideSignalerAttrList;
        syncUnreconciledList[1] = ctx->cpuWaiterAttrList;
        syncUnreconciledList[2] = ctx->ideWaiterAttrList;
        /* Reconcile Signaler and Waiter NvSciSyncAttrList */
        err = NvSciSyncAttrListReconcile(syncUnreconciledList, 3, &syncReconciledList,
                                         &syncNewConflictList);
        if(err != NvSciError_Success) {
            LOG_ERR("NvSciSyncAttrListSetAttrs failed\n");
            return 0;
        }

        /* Create NvSciSync object and get the syncObj */
        err = NvSciSyncObjAlloc(syncReconciledList, &ctx->eofSyncObj);
        if(err != NvSciError_Success) {
            LOG_ERR("NvSciSyncAttrListSetAttrs failed\n");
            return 0;
        }

        NvSciSyncAttrListFree(syncReconciledList);
        NvSciSyncAttrListFree(syncUnreconciledList[0]);
        NvSciSyncAttrListFree(syncUnreconciledList[1]);
        NvSciSyncAttrListFree(syncUnreconciledList[2]);
        NvSciSyncAttrListFree(syncNewConflictList);

        status = NvMediaIDERegisterNvSciSyncObj(ctx->decoder, NVMEDIA_EOF_PRESYNCOBJ, ctx->eofSyncObj);
        if(status != NVMEDIA_STATUS_OK) {
           LOG_ERR("main: Failed to register sci sync eof obj attr list.\n");
           return 0;
        }

        status = NvMediaIDESetNvSciSyncObjforEOF(ctx->decoder, ctx->eofSyncObj);
        if(status != NVMEDIA_STATUS_OK) {
           LOG_ERR("main: Failed to set sci sync eof obj attr list.\n");
           return 0;
        }

        ctx->bMonochrome = false;

        memset(&ctx->RefFrame[0], 0, sizeof(FrameBuffer) * MAX_DEC_BUFFERS);
        NvSciBufAttrValColorStd  colorFormat = NvSciColorStd_REC601_ER;  // Initil  to avoid MISRA-C errors
        switch (pnvsi->eChromaFormat) {
            case 0: // Monochrome
                ctx->bMonochrome = true;
            case 1: // 4:2:0
                if(flags & NVMEDIA_IDE_10BIT_DECODE) {
                    if (ctx->videoFullRangeFlag)
                    {
                       colorFormat = NvSciColorStd_REC2020_ER;
                    }
                    else
                    {
                       colorFormat = NvSciColorStd_REC2020_SR;
                    }
                    if(surface12bit) {
                        LOG_INFO("BeginSequence: Chroma format: NvMediaSurfaceType YUV 420 12bit\n");
                        surfaceChromaFormat = YUV420SP_12bit;

                    } else {
                        LOG_INFO("BeginSequence: Chroma format: NvMediaSurfaceType YUV 420 10bit\n");
                        surfaceChromaFormat = YUV420SP_10bit;
                    }
                } else {
                    LOG_INFO("Chroma format: NvMediaSurfaceType YUV 420 8bit\n");
                    surfaceChromaFormat = YUV420SP_8bit;
                    if (ctx->videoFullRangeFlag)
                    {
                       colorFormat = NvSciColorStd_REC601_ER;
                    }
                    else
                    {
                       colorFormat = NvSciColorStd_REC601_SR;
                    }
                }
                break;
            case 2: // 4:2:2
                if(flags & NVMEDIA_IDE_10BIT_DECODE) {
                    if(surface12bit) {
                        LOG_INFO("BeginSequence: Chroma format: NvMediaSurfaceType YUV 422 12bit\n");
                    } else {
                        LOG_INFO("BeginSequence: Chroma format: NvMediaSurfaceType YUV 422 10bit\n");
                    }
                } else {
                    LOG_INFO("Chroma format: NvMediaSurfaceType YUV 422 8bit\n");
                }
                break;
            case 3: // 4:4:4
                if(flags & NVMEDIA_IDE_10BIT_DECODE) {
                    if (ctx->videoFullRangeFlag)
                    {
                        colorFormat = NvSciColorStd_REC2020_ER;
                    }
                    else
                    {
                       colorFormat = NvSciColorStd_REC2020_SR;
                    }
                    if(surface12bit) {
                        LOG_INFO("BeginSequence: Chroma format: NvMediaSurfaceType YUV 444 12bit\n");
                        surfaceChromaFormat = YUV444SP_12bit;
                    } else {
                        LOG_INFO("BeginSequence: Chroma format: NvMediaSurfaceType YUV 444 10bit\n");
                        surfaceChromaFormat = YUV444SP_10bit;
                    }
                } else {
                    LOG_INFO("Chroma format: NvMediaSurfaceType YUV 444 8bit\n");
                    surfaceChromaFormat = YUV444SP_8bit;
                    if (ctx->videoFullRangeFlag)
                    {
                       colorFormat = NvSciColorStd_REC601_ER;
                    }
                    else
                    {
                       colorFormat = NvSciColorStd_REC601_SR;
                    }
                }

                break;
            default:
                LOG_INFO("Invalid chroma format: %d\n", pnvsi->eChromaFormat);
                return 0;
        }
        ctx->nBuffers = decodeBuffers + MAX_DEC_DEINTER_BUFFERS;
        err = NvSciBufAttrListCreate(ctx->bufModule, &(ctx->bufAttributeList));
        if(err != NvSciError_Success) {
            LOG_ERR("%s: SciBufAttrListCreate failed. Error: %d \n", __func__, err);
            return false;
        }
        NvSciBufAttrValImageScanType scanType = NvSciBufScan_ProgressiveType;
        if (!pnvsi->bProgSeq)
        {
            scanType = NvSciBufScan_InterlaceType;
        }

        status = NvMediaIDEFillNvSciBufAttrList(ctx->instanceId, ctx->bufAttributeList);
        if (NVMEDIA_STATUS_OK != status) {
            LOG_ERR("main: Failed to populate IDE internal attributes\n");
            return false;
        }

        status = PopulateNvSciBufAttrList(
                 surfaceChromaFormat,
                 (pnvsi->uCodedWidth + 15) & ~15,
                 (pnvsi->uCodedHeight + 15) & ~15,
                 true,                           /* needCpuAccess */
                 NvSciBufImage_BlockLinearType,
                 IDE_APP_MAX_INPUT_PLANE_COUNT,
                 NvSciBufAccessPerm_ReadWrite,
                 IDE_APP_BASE_ADDR_ALIGN,
                 colorFormat ,
                 scanType,
                 ctx->bufAttributeList);
        if (NVMEDIA_STATUS_OK != status) {
            LOG_ERR("main: Failed to populate attributes\n");
            return false;
        }
        NvSciBufAttrList bufConflictList;
        NvSciBufAttrList bufReconciledList;
        err = NvSciBufAttrListReconcile(&ctx->bufAttributeList, 1U,
                                        &bufReconciledList, &bufConflictList);
        if (err != NvSciError_Success) {
            LOG_ERR("main: Reconciliation for input frame failed\n");
            return false;
        }
        /* Creates surfaces for decode
         */
        for (i = 0; i < ctx->nBuffers; i++) {
            err = NvSciBufObjAlloc(bufReconciledList, &ctx->RefFrame[i].videoSurface);
            status = NvMediaIDERegisterNvSciBufObj(ctx->decoder, ctx->RefFrame[i].videoSurface);
            if (status != NVMEDIA_STATUS_OK)
            {
                LOG_ERR("main:failed to register NvSciBufObj\n");
            }
            if (err != NvSciError_Success) {
                LOG_ERR("main: Reconciliation for input frame failed\n");
                return false;
            }
        }

        ctx->monitorWidth = ctx->displayWidth;
        ctx->monitorHeight = ctx->displayHeight;

        if(!ctx->monitorWidth || !ctx->monitorHeight) {
            LOG_ERR("cbBeginSequence: bad monitor resolution \n");
            return false;
        }
         /* The reconciled list is needed for later */
         NvSciBufAttrListFree(ctx->bufAttributeList);
         NvSciBufAttrListFree(bufConflictList);
         NvSciBufAttrListFree(bufReconciledList);

    } else {
        LOG_INFO("cbBeginSequence: No resolution change\n");
    }

    return decodeBuffers;
}

NvMediaStatus cbDecodePicture(void *ptr, NvMediaParserPictureData *pd)
{
    VideoDemoTestCtx *ctx = (VideoDemoTestCtx*)ptr;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    FrameBuffer *targetBuffer = NULL;
    uint64_t timeEnd, timeStart = 0;
    FrameBuffer *fgsBuffer = NULL;
    FrameBuffer *rangeMappedBuffer = NULL;
    NvMediaBitstreamBuffer bitStreamBuffer[1];
    NvSciError err;

    if(!pd || !ctx) {
        LOG_ERR("cbDecodePicture: Invalid NvMediaParserPictureData or VideoDemoTestCtx\n");
        return NVMEDIA_STATUS_ERROR;
    }

    if (pd->pCurrPic) {
        targetBuffer = (FrameBuffer *)pd->pCurrPic;
        GetTimeMicroSec(&timeStart);

        /*
         * Update NvMedia reference pointers from parser with corresponding
         * NvSciBufObj pointers in NvMediaPictureInfo structure
         * for each codec type.
         */
        switch (ctx->eCodec) {
            case NVMEDIA_VIDEO_CODEC_MPEG1:
            case NVMEDIA_VIDEO_CODEC_MPEG2:
                status = UpdateNvMediaSurfacePictureInfoMPEG(ctx, (NvMediaPictureInfoMPEG1Or2 *)&pd->CodecSpecificInfo.mpeg2,targetBuffer->videoSurface);
                break;
            case NVMEDIA_VIDEO_CODEC_MPEG4:
                status = UpdateNvMediaSurfacePictureInfoMPEG4(ctx, (NvMediaPictureInfoMPEG4Part2 *)&pd->CodecSpecificInfo.mpeg4,targetBuffer->videoSurface);
                break;
            case NVMEDIA_VIDEO_CODEC_VC1:
            case NVMEDIA_VIDEO_CODEC_VC1_ADVANCED:
            {
                rangeMappedBuffer = (FrameBuffer*)((NvMediaPictureInfoVC1 *)(&pd->CodecSpecificInfo.vc1)->range_mapped);
                status = UpdateNvMediaSurfacePictureInfoVC1(ctx, (NvMediaPictureInfoVC1 *)&pd->CodecSpecificInfo.vc1);
            }
                break;
            case NVMEDIA_VIDEO_CODEC_H264:
                status = UpdateNvMediaSurfacePictureInfoH264(ctx, (NvMediaPictureInfoH264 *)&pd->CodecSpecificInfo.h264);
                break;
            case NVMEDIA_VIDEO_CODEC_VP8:
                status = UpdateNvMediaSurfacePictureInfoVP8(ctx, (NvMediaPictureInfoVP8 *)&pd->CodecSpecificInfo.vp8,targetBuffer->videoSurface);
                break;
            case NVMEDIA_VIDEO_CODEC_HEVC:
                status = UpdateNvMediaSurfacePictureInfoH265(ctx, (NvMediaPictureInfoH265 *)&pd->CodecSpecificInfo.hevc);
                break;
            case NVMEDIA_VIDEO_CODEC_VP9:
                status = UpdateNvMediaSurfacePictureInfoVP9(ctx, (NvMediaPictureInfoVP9 *)&pd->CodecSpecificInfo.vp9, targetBuffer);
                break;
            case NVMEDIA_VIDEO_CODEC_AV1:
            {
                // Save original FrameBuffer pointer before overwriting with videoSurface
                fgsBuffer = (FrameBuffer*)((NvMediaPictureInfoAV1 *)(&pd->CodecSpecificInfo.av1)->fgsPic);
                status = UpdateNvMediaSurfacePictureInfoAV1(ctx, (NvMediaPictureInfoAV1 *)&pd->CodecSpecificInfo.av1);
            }
                break;
            default:
                LOG_ERR("cbDecodePicture: Invalid decoder type\n");
                return NVMEDIA_STATUS_ERROR;
        }
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("cbDecodePicture: Decode failed in UpdateNvMediaSurfacePictureInfo: %d\n", status);
            return NVMEDIA_STATUS_ERROR;
        }
        targetBuffer->frameNum = ctx->nPicNum;
        targetBuffer->topFieldFirstFlag = !!pd->top_field_first;        // Frame pictures only
        targetBuffer->progressiveFrameFlag = !!pd->progressive_frame;   // Frame is progressive
        bitStreamBuffer[0].bitstream = (uint8_t *)pd->pBitstreamData;
        bitStreamBuffer[0].bitstreamBytes = pd->uBitstreamDataLen;

        targetBuffer->lDARWidth = pd->uDARWidth;
        targetBuffer->lDARHeight = pd->uDARHeight;
        targetBuffer->displayLeftOffset = pd->uDisplayLeftOffset;
        targetBuffer->displayTopOffset = pd->uDisplayTopOffset;
        targetBuffer->displayWidth = pd->uDisplayWidth;
        targetBuffer->displayHeight = pd->uDisplayHeight;

        LOG_DBG("cbDecodePicture: %d Ptr: %p Surface: %p (stream ptr: %p size: %d)\n",
            ctx->nPicNum, targetBuffer, targetBuffer->videoSurface, pd->pBitstreamData, pd->uBitstreamDataLen);
        ctx->nPicNum++;

        if (targetBuffer->videoSurface) {
            status = NvMediaIDEDecoderRender(ctx->decoder,                                      // decoder
                                             targetBuffer->videoSurface,                        // target
                                             (NvMediaPictureInfo *)&pd->CodecSpecificInfo,      // pictureInfo
                                             NULL,                                              // encryptParams
                                             1,                                                 // numBitstreamBuffers
                                             &bitStreamBuffer[0],                               // bitstreams
                                             NULL,                                              // FrameStatsDump
                                             ctx->instanceId);                                  // instance ID

            if (status != NVMEDIA_STATUS_OK) {
                LOG_ERR("cbDecodePicture: Decode failed: %d\n", status);
                return NVMEDIA_STATUS_ERROR;
            }
            LOG_DBG("cbDecodePicture: Frame decode done\n");

            status = NvMediaIDEGetEOFNvSciSyncFence(ctx->decoder, ctx->eofSyncObj,
                                                    &targetBuffer->preFence);
            if(status != NVMEDIA_STATUS_OK) {
                LOG_ERR("main: NvMediaIEPGetEOFNvSciSyncFence failed: %x\n", status);
                return 0;
            }
            if ((ctx->eCodec == NVMEDIA_VIDEO_CODEC_VC1) || (ctx->eCodec == NVMEDIA_VIDEO_CODEC_VC1_ADVANCED))
            {
                if (rangeMappedBuffer)
                {
                      rangeMappedBuffer->lDARWidth = pd->uDARWidth;
                      rangeMappedBuffer->lDARHeight = pd->uDARHeight;
                      rangeMappedBuffer->displayLeftOffset = pd->uDisplayLeftOffset;
                      rangeMappedBuffer->displayTopOffset = pd->uDisplayTopOffset;
                      rangeMappedBuffer->displayWidth = pd->uDisplayWidth;
                      rangeMappedBuffer->displayHeight = pd->uDisplayHeight;
                      // copy fence to  the rangemapped surface
                      NvSciSyncFenceDup ((const NvSciSyncFence *)&(targetBuffer->preFence), &(rangeMappedBuffer->preFence));
                 }
            }
            else if (ctx->eCodec == NVMEDIA_VIDEO_CODEC_AV1)
            {
                 const NvMediaPictureInfoAV1 *av1 = &pd->CodecSpecificInfo.av1;
                 if (av1->film_grain_enable && av1->film_grain.apply_grain)
                 {
                      FrameBuffer *pFgsPic = fgsBuffer;
                      pFgsPic->lDARWidth = pd->uDARWidth;
                      pFgsPic->lDARHeight = pd->uDARHeight;
                      pFgsPic->displayLeftOffset = pd->uDisplayLeftOffset;
                      pFgsPic->displayTopOffset = pd->uDisplayTopOffset;
                      pFgsPic->displayWidth = pd->uDisplayWidth;
                      pFgsPic->displayHeight = pd->uDisplayHeight;
                      // copy fence to  the fgs surface
                      NvSciSyncFenceDup ((const NvSciSyncFence *)&(targetBuffer->preFence), &(pFgsPic->preFence));
                 }
            }
        } else {
            LOG_ERR("cbDecodePicture: Invalid target surface\n");
        }
        if (ctx->showDecodeTimimg) {
            // Wait for decode completion
            err = NvSciSyncFenceWait(&(targetBuffer->preFence), ctx->cpuWaitContext, 1000*1000);
            if(err != NvSciError_Success) {
                LOG_ERR("NvSciSyncFenceWait failed\n");
                return NVMEDIA_STATUS_ERROR;
            }
        }

        GetTimeMicroSec(&timeEnd);
        ctx->totalDecodeTime += (timeEnd - timeStart) / 1000.0;
        if (ctx->showDecodeTimimg) {
            LOG_DBG("cbDecodePicture: %03d %lld us\n", ctx->decodeCount, timeEnd - timeStart);
        }

        ctx->decodeCount++;
        if (ctx->numFramesToDecode && ctx->numFramesToDecode == ctx->decodeCount) {
            LOG_DBG("cbDecodePicture: Requested number of frames read (%d). Setting stop decoding flag to TRUE\n", ctx->numFramesToDecode);
            ctx->stopDecoding = true;
        }

        ctx->sumCompressedLen += pd->uBitstreamDataLen;
    } else {
        LOG_ERR("cbDecodePicture: No valid frame\n");
        return NVMEDIA_STATUS_ERROR;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus cbDisplayPicture(void *ptr, NvMediaRefSurface *p, int64_t llPts)
{
    VideoDemoTestCtx *ctx = (VideoDemoTestCtx*)ptr;
    NvMediaStatus status;
    FrameBuffer* buffer = (FrameBuffer*)p;
    NvSciError err;
    bool CRCCheck = false;
    uint32_t calculatedCrc = 0;
    uint32_t refCrc = 0;
    NvMediaRect srcRect;

    if(!ctx) {
        LOG_ERR("Display: Invalid VideoDemoTestCtx\n");
        return NVMEDIA_STATUS_ERROR;
    }

    if(buffer) {
        if(ctx->cropCRC)
        {
            srcRect.x0 = buffer->displayLeftOffset;
            srcRect.y0 = buffer->displayTopOffset;
            srcRect.x1 = buffer->displayWidth;
            srcRect.y1 = buffer->displayHeight;
        }
        else
        {
            srcRect.x0 = 0;
            srcRect.y0 = 0;
            srcRect.x1 = ctx->decodeWidth;
            srcRect.y1 = ctx->decodeHeight;
        }
        /* Wait for operations on the image to be complete */
        err = NvSciSyncFenceWait(&buffer->preFence, ctx->cpuWaitContext, 1000*1000);
        if(err != NvSciError_Success) {
            LOG_ERR("NvSciSyncFenceWait failed\n");
            return NVMEDIA_STATUS_ERROR;
        }

        if (ctx->OutputYUVFilename) {
            {
                status = WriteOutput(ctx->OutputYUVFilename,
                                    buffer->videoSurface,
                                    true,
                                    (ctx->FrameCount == 0) ? false : true,
                                    &srcRect);
                if (status != NVMEDIA_STATUS_OK) {
                    LOG_ERR("cbDecodePicture: Write frame to file failed: %d\n", status);
                    ctx->YUVSaveComplete = false;
                }
                LOG_INFO("cbDecodePicture: Saving YUV frame %d to file\n", ctx->FrameCount);
                ctx->YUVSaveComplete = true;
            }
        }

        if(ctx->generateCRC) {
            if(ctx->fpCrc) {
                calculatedCrc = 0;
                status = GetNvSciBufObjCrc(buffer->videoSurface,
                                           &srcRect,
                                           (bool)ctx->bMonochrome,
                                           &calculatedCrc);


                if(status != NVMEDIA_STATUS_OK)
                {
                    LOG_ERR("cbDisplayPicture: GetSurfaceCrc failed for frame %d\n", ctx->FrameCount);
                    ctx->CRCGenComplete = false;
                }

                LOG_INFO("cbDisplayPicture: Saving CRC %8x for frame %d to file\n", calculatedCrc, ctx->FrameCount);
                ctx->CRCGenComplete = true;
                if(!fprintf(ctx->fpCrc, "%08x\n",calculatedCrc))
                {
                    LOG_ERR("cbDisplayPicture: Failed writing calculated CRC for frame %d to file\n", ctx->FrameCount);
                    ctx->CRCGenComplete = false;
                }
            }
        } else if(ctx->checkCRC) {
            if(ctx->fpCrc) {
                if (fscanf(ctx->fpCrc, "%8x\n", &refCrc) == 1) {
                    LOG_DBG("cbDisplayPicture: Checking CRC.... Expected: %8x\n", refCrc);
                        status = CheckSurfaceCrc(buffer->videoSurface,
                                                 &srcRect,
                                                 (bool)ctx->bMonochrome,
                                                 refCrc,
                                                 &CRCCheck);

                    if(status != NVMEDIA_STATUS_OK) {
                        LOG_ERR("cbDisplayPicture: CheckSurfaceCrc failed\n");
                    }
                    if(!CRCCheck) {
                        LOG_ERR("cbDisplayPicture: CRC check for frame %d : Fail\n", ctx->FrameCount);
                        ctx->CRCResult = false;
                        return NVMEDIA_STATUS_ERROR;
                    } else {
                        LOG_INFO("cbDisplayPicture: CRC check for frame %d : Pass\n", ctx->FrameCount);
                        ctx->CRCResult = true;
                    }
                }
            } else {
                LOG_WARN("cbDisplayPicture: CRC check couldn't be checked. Failed reading file.\n");
            }
        }
        ctx->FrameCount++;
    } else {
        LOG_ERR("Display: Invalid buffer\n");
        return NVMEDIA_STATUS_ERROR;
    }

    return NVMEDIA_STATUS_OK;
}

void cbUnhandledNALU(void *ptr, const uint8_t *buf, int32_t size)
{
    // Empty
}

NvMediaStatus cbAllocPictureBuffer(void *ptr, NvMediaRefSurface **p)
{
    int i;
    VideoDemoTestCtx *ctx = (VideoDemoTestCtx*)ptr;

    if(!ctx) {
        LOG_ERR("cbAllocPictureBuffer: Invalid VideoDemoTestCtx\n");
        return NVMEDIA_STATUS_ERROR;
    }

    *p = (NvMediaRefSurface *) NULL;

    for (i = 0; i < ctx->nBuffers; i++) {
        if (!ctx->RefFrame[i].refCount) {
            *p = (NvMediaRefSurface *) &ctx->RefFrame[i];
            ctx->RefFrame[i].refCount++;
            ctx->RefFrame[i].index = i;
            LOG_DBG("Allocated buffer for picture index: %d Ptr:%p Surface:%p\n", i, *p, ctx->RefFrame[i].videoSurface);
            return NVMEDIA_STATUS_OK;
        }
    }

    LOG_ERR("Alloc picture failed\n");
    return NVMEDIA_STATUS_ERROR;
}

void cbRelease(void *ptr, NvMediaRefSurface *p)
{
    FrameBuffer* buffer = (FrameBuffer*)p;

    if(!buffer) {
        LOG_ERR("cbRelease: Invalid FrameBuffer\n");
        return;
    }

    LOG_DBG("Releasing picture: %d index: %d\n", buffer->frameNum, buffer->index);
    if (buffer->refCount > 0)
        buffer->refCount--;
}

void cbAddRef(void *ptr, NvMediaRefSurface *p)
{
    FrameBuffer* buffer = (FrameBuffer*)p;

    if(!buffer) {
        LOG_ERR("cbAddRef: Invalid FrameBuffer\n");
        return;
    }

    LOG_DBG("Adding reference to picture: %d\n", buffer->frameNum);
    buffer->refCount++;
}

NvMediaStatus cbGetBackwardUpdates(void *ptr, NvMediaVP9BackwardUpdates *backwardUpdate)
{
    NvMediaStatus status;
    VideoDemoTestCtx *ctx = (VideoDemoTestCtx*)ptr;

    if(!ctx) {
        LOG_ERR("cbGetBackwardUpdates: Invalid VideoDemoTestCtx\n");
        return NVMEDIA_STATUS_ERROR;
    }

    status = NvMediaIDEGetBackwardUpdates(ctx->decoder, (void *)backwardUpdate);
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("cbGetBackwardUpdates : Failed to get Video backward updates : %d\n", status);
    }

    return status;
}

NvMediaParserClientCb TestClientCb =
{
    &cbBeginSequence,
    &cbDecodePicture,
    &cbDisplayPicture,
    &cbUnhandledNALU,
    &cbAllocPictureBuffer,
    &cbRelease,
    &cbAddRef,
    NULL,
    NULL,
    NULL,
    NULL,
    &cbGetBackwardUpdates,
    NULL
};

int Init(VideoDemoTestCtx *ctx, TestArgs *testArgs)
{
    struct stat st;
    bool enableVC1APInterlaced = true;
    float defaultFrameRate = 30.0;
    ctx->aspectRatio            = testArgs->aspectRatio;
    ctx->frameTimeUSec          = testArgs->frameTimeUSec;
    ctx->loop                   = testArgs->loop;
    ctx->numFramesToDecode      = testArgs->numFramesToDecode;
    ctx->eCodec                 = testArgs->eCodec;
    ctx->OutputYUVFilename      = testArgs->OutputYUVFilename;
    ctx->filename               = testArgs->filename;
    ctx->showDecodeTimimg       = testArgs->showDecodeTimimg;
    ctx->instanceId             = testArgs->instanceId;
    ctx->generateCRC            = testArgs->generateCRC;
    ctx->checkCRC               = testArgs->checkCRC;
    ctx->cropCRC                = testArgs->cropCRC;
    ctx->decProfiling           = testArgs->decProfiling;
    ctx->alternateCreateAPI     = testArgs->alternateCreateAPI;
    ctx->CRCResult              = false;
    ctx->CRCGenComplete         = false;
    ctx->YUVSaveComplete        = false;
    ctx->setAnnexBStream        = testArgs->setAnnexBStream;
    ctx->av1annexBStream        = testArgs->av1annexBStream;
    ctx->setOperatingPoint      = testArgs->setOperatingPoint;
    ctx->setOutputAllLayers     = testArgs->setOutputAllLayers;
    ctx->av1OutputAllLayers     = testArgs->av1OutputAllLayers;
    ctx->av1OperatingPoint      = testArgs->av1OperatingPoint;
    ctx->setMaxRes              = testArgs->setMaxRes;
    ctx->enableMaxRes           = testArgs->enableMaxRes;
    ctx->syncModule = NULL;
    ctx->bufModule = NULL;
    ctx->cpuWaitContext = NULL;

    LOG_MSG("Init: Opening Input file %s\n", testArgs->filename);
    ctx->file = fopen(ctx->filename, "rb");
    if (!ctx->file) {
        LOG_ERR("Init: Failed to open stream %s\n", testArgs->filename);
        return -1;
    }

    if((ctx->generateCRC) || (ctx->checkCRC))
    {
        ctx->fpCrc = fopen(testArgs->crcFilePath, ctx->generateCRC ? "wt" : "rb");
        if (ctx->fpCrc == NULL) {
            LOG_ERR("Init: Invalid CRC file %s specified!\n", testArgs->crcFilePath);
            return -1;
        }
        else
            if (ctx->checkCRC)
                LOG_MSG("Init: Opening CRC file %s for crc check\n", testArgs->crcFilePath);
            else
                LOG_MSG("Init: Opening CRC file %s for crc gen\n", testArgs->crcFilePath);
    }

    memset(&ctx->nvsi, 0, sizeof(ctx->nvsi));
    ctx->lDispCounter = 0;

    if(stat(ctx->filename, &st) == -1) {
        fclose(ctx->file);
        LOG_ERR("Init: cannot determine size of stream %s\n", ctx->filename);
        return -1;
    }
    ctx->fileSize = st.st_size;

    ctx->bRCVfile = Strcasestr(ctx->filename, ".rcv")  != NULL;

    // create video parser
    memset(&ctx->nvdp, 0, sizeof(NvMediaParserParams));
    ctx->nvdp.pClient = &TestClientCb;
    ctx->nvdp.pClientCtx = ctx;
    ctx->nvdp.uErrorThreshold = 50;
    ctx->nvdp.uReferenceClockRate = 0;
    ctx->nvdp.eCodec = ctx->eCodec;

    LOG_DBG("Init: Creating parser\n");
    ctx->parser = NvMediaParserCreate(&ctx->nvdp);
    if (!ctx->parser) {
        LOG_ERR("Init: NvMediaParserCreate failed\n");
        return -1;
    }
    NvMediaParserSetAttribute(ctx->parser, NvMParseAttr_EnableVC1APInterlaced, sizeof(bool), &enableVC1APInterlaced);
    NvMediaParserSetAttribute(ctx->parser, NvMParseAttr_SetDefaultFramerate, sizeof(float), &defaultFrameRate);
    if (ctx->setAnnexBStream)
    {
        // program AnnexB stream for av1
        NvMediaParserSetAttribute(ctx->parser, NvMParseAttr_av1AnnexBDecode, sizeof(uint8_t), &ctx->av1annexBStream);
    }
    if (ctx->setOperatingPoint)
    {
        NvMediaParserSetAttribute(ctx->parser, NvMParseAttr_av1SetOperatingPoint, sizeof(uint8_t), &ctx->av1OperatingPoint);
    }
    if (ctx->setOutputAllLayers)
    {
        NvMediaParserSetAttribute(ctx->parser, NvMParseAttr_av1SetOutputAllLayers, sizeof(uint8_t), &ctx->av1OutputAllLayers);
    }
    if (ctx->setMaxRes)
    {
        NvMediaParserSetAttribute(ctx->parser, NvMParseAttr_SetMaxRes, sizeof(uint8_t), &ctx->enableMaxRes);
    }

    if(ctx->OutputYUVFilename) {
        LOG_MSG("Init: Opening Output YUV file %s\n", ctx->OutputYUVFilename);
        FILE *file = fopen(ctx->OutputYUVFilename, "w");
        if(!file) {
            LOG_ERR("Init: unable to open output YUV file %s\n", ctx->OutputYUVFilename);
            return -1;
        }
        fclose(file);
    }

    return 0;
}

void Deinit(VideoDemoTestCtx *ctx)
{

    uint32_t i;

    NvMediaParserDestroy(ctx->parser);
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (ctx->file)
        fclose(ctx->file);

    if (ctx->decoder)
    {
        for(i = 0; i < MAX_DEC_BUFFERS; i++) {
            if (ctx->RefFrame[i].videoSurface)
            {
                NvSciSyncFenceClear(&ctx->RefFrame[i].preFence);
                status = NvMediaIDEUnregisterNvSciBufObj(ctx->decoder, ctx->RefFrame[i].videoSurface);
                if (status != NVMEDIA_STATUS_OK)
                {
                    LOG_ERR("main:failed to uregister NvSciBufObj\n");
                }
                NvSciBufObjFree(ctx->RefFrame[i].videoSurface);
                ctx->RefFrame[i].videoSurface = NULL;
            }
        }
        if (ctx->eofSyncObj != NULL)
        {
            NvMediaIDEUnregisterNvSciSyncObj(ctx->decoder, ctx->eofSyncObj);
            NvSciSyncObjFree(ctx->eofSyncObj);
            ctx->eofSyncObj = NULL;
        }
        NvMediaIDEDestroy(ctx->decoder);
        ctx->decoder = NULL;
    }

    if (NULL != ctx->cpuWaitContext) {
        NvSciSyncCpuWaitContextFree(ctx->cpuWaitContext);
        ctx->cpuWaitContext = NULL;
    }
    if (NULL != ctx->bufModule) {
        NvSciBufModuleClose(ctx->bufModule);
        ctx->bufModule = NULL;
    }

    if (NULL != ctx->syncModule) {
        NvSciSyncModuleClose(ctx->syncModule);
        ctx->syncModule = NULL;
    }

    if(ctx->fpCrc)
        fclose(ctx->fpCrc);
}

int StreamVC1SimpleProfile(VideoDemoTestCtx *ctx)
{
    uint8_t *bits;
    uint8_t header[256 + 32];
    uint32_t readSize = 0;
    int32_t frameCount = 0;
    RCVFileHeader RCVHeader;
    NvMediaParserSeqInfo nvsi, *pnvsi;
    NvMediaParserParams *pnvdp = &ctx->nvdp;
    float defaultFrameRate = 30.0;
    uint32_t len;

    rewind(ctx->file);
    fread(header, 5, 1, ctx->file);
    if (header[3] == 0xC5)
        readSize = 32;
    readSize += header[4];
    fread(header + 5, readSize - 5, 1, ctx->file);

    ctx->bVC1SimpleMainProfile = true;     //setting it for Simple/Main profile
    LOG_DBG("VC1 Simple/Main profile clip\n");
    len = ParseRCVHeader(&RCVHeader, header, readSize);
    LOG_DBG("ParseRCVHeader : len = %d \n", len);
    // Close previous instance
    NvMediaParserDestroy(ctx->parser);
    pnvsi = &nvsi;
    pnvsi->eCodec = NVMEDIA_VIDEO_CODEC_VC1;
    pnvsi->bProgSeq = true;
    pnvsi->uDisplayWidth = RCVHeader.lMaxCodedWidth;
    pnvsi->uDisplayHeight = RCVHeader.lMaxCodedHeight;
    pnvsi->uCodedWidth = (pnvsi->uDisplayWidth + 15) & ~15;
    pnvsi->uCodedHeight = (pnvsi->uDisplayHeight + 15) & ~15;
    pnvsi->eChromaFormat = 1;
    pnvsi->uBitrate = RCVHeader.lBitRate;
    pnvsi->fFrameRate = (RCVHeader.lFrameRate && RCVHeader.lFrameRate != -1) ? (float)RCVHeader.lFrameRate : 0;
    pnvsi->uDARWidth = pnvsi->uDisplayWidth;
    pnvsi->uDARHeight = pnvsi->uDisplayHeight;
    pnvsi->eVideoFormat = NvMVideoFormat_Unspecified;
    pnvsi->eColorPrimaries = NvMColorPrimaries_Unspecified;
    pnvsi->eTransferCharacteristics = NvMTransferCharacteristics_Unspecified;
    pnvsi->eMatrixCoefficients = NvMMatrixCoeffs_Unspecified;
    pnvsi->uSequenceHeaderSize = RCVHeader.cbSeqHdr;
    if (pnvsi->uSequenceHeaderSize > 0)
        memcpy(pnvsi->SequenceHeaderData, RCVHeader.SeqHdrData, pnvsi->uSequenceHeaderSize);
    pnvdp->pExternalSeqInfo = pnvsi;
    ctx->parser = NvMediaParserCreate(pnvdp);
    if (!ctx->parser) {
        LOG_ERR("NvMediaParserCreate failed\n");
    }
    NvMediaParserSetAttribute(ctx->parser, NvMParseAttr_SetDefaultFramerate, sizeof(float), &defaultFrameRate);

    bits = malloc(RCV_MAX_FRAME_SIZE);
    if (!bits)
        return -1;

    while (!feof(ctx->file) && !ctx->stopDecoding && (signal_stop == 0u)) {
        size_t len;
        NvMediaBitStreamPkt packet;
        uint32_t timeStamp;

        memset(&packet, 0, sizeof(NvMediaBitStreamPkt));

        // Read frame length
        len = fread(&readSize, 4, 1, ctx->file);
        // Check end of file
        if (!len)
            break;
        readSize &= 0x00FFFFFF;
        // Read time stamp
        len = fread(&timeStamp, 4, 1, ctx->file);
        if (!len)
            break;
        if (readSize) {
            //  Read frame data
            len = fread(bits, 1, readSize, ctx->file);
        } else {
            // Skipped P-Frame
            bits[0] = 0;
            len = 1;
        }

        LOG_DBG("Frame: %d readSize: %d actual read length: %d timeStamp: %d\n", frameCount, readSize, len, timeStamp);

        packet.uDataLength = (uint32_t) len;
        packet.pByteStream = bits;

        packet.bEOS = feof(ctx->file) ? true : false;
        LOG_DBG("StreamVC1SimpleProfile: EOS %d is sent...\n", packet.bEOS);
        if (NvMediaParserParse(ctx->parser, &packet) != NVMEDIA_STATUS_OK)
            return -1;
        frameCount++;
    }

    if (frameCount != RCVHeader.lNumFrames) {
        LOG_ERR("Actual (%d) and RCV header (%d) frame count does not match\n",
            frameCount, RCVHeader.lNumFrames);
    }

    NvMediaParserFlush(ctx->parser);

    free(bits);

    return 0;
}

static int StreamVP8(VideoDemoTestCtx *ctx)
{
    int i;
    uint8_t *bits;
    uint32_t FrameSize;
    uint32_t numFrames;
    uint32_t frameRateNum;
    uint32_t frameRateDen;
    float frameRate;
    uint32_t frameCount;
    bool Vp8IvfFileHdrRead;
    uint32_t readSize = READ_SIZE;

    bits = malloc(readSize);
    if (!bits) {
        LOG_ERR("StreamVP8: Failed allocating memory for file buffer\n");
        return -1;
    }

    for(i = 0; (i < ctx->loop) || (ctx->loop == -1); i++) {
        Vp8IvfFileHdrRead = false;
        frameCount = 0;
        numFrames = 0;
        ctx->lDispCounter = 0;
        while(!feof(ctx->file) && !ctx->stopDecoding && !signal_stop) {
            size_t len;
            NvMediaBitStreamPkt packet;
            memset(&packet, 0, sizeof(NvMediaBitStreamPkt));

            if(Vp8IvfFileHdrRead == false) {
                if(fread(bits, 1, IVF_FILE_HDR_SIZE, ctx-> file) != IVF_FILE_HDR_SIZE) {
                    LOG_ERR("StreamVP8: Failed to read IVF file header\n");
                    free(bits);
                    return -1;
                }
                if(!((bits[0] == 'D') && (bits[1] == 'K') && (bits[2] == 'I') && (bits[3] == 'F'))) {
                    LOG_ERR("StreamVP8: It is not a valid IVF file \n");
                    free(bits);
                    return -1;
                }
                Vp8IvfFileHdrRead = true;
                LOG_DBG("StreamVP8: It is a valid IVF file \n");

                frameRateNum = u32(bits + 16);
                frameRateDen = u32(bits + 20);
                if(frameRateDen)
                    frameRate = (frameRateNum * 1.0)/ frameRateDen;
                else {
                    LOG_INFO("StreamVP8: Value of time scale in IVF heder is zero. Using default frame rate\n");
                    frameRate = 0;
                }
                if(frameRate)
                    NvMediaParserSetAttribute(ctx->parser, NvMParseAttr_SetFramerate, sizeof(float), &frameRate);

                numFrames = u32(bits + 24);
                if(!numFrames) {
                    numFrames = 1000; //try to continue with Frame Decoding since this is not a fatal error.
                }

                LOG_DBG("StreamVP8:Frame Rate: %f \t Frame Count: %d \n",frameRate,numFrames);
                if (ctx->numFramesToDecode <= 0)
                {
                    ctx->numFramesToDecode = numFrames;
                }
            }

            if(fread(bits, 1, IVF_FRAME_HDR_SIZE, ctx->file) == IVF_FRAME_HDR_SIZE) {
                FrameSize = (bits[3]<<24) + (bits[2]<<16) + (bits[1]<<8) + bits[0];
                if(FrameSize > readSize) {
                    bits = realloc(bits, FrameSize);
                    readSize = FrameSize;
                }
                len = fread(bits, 1, FrameSize, ctx->file);
                packet.uDataLength = (uint32_t) len;
                packet.pByteStream = bits;
                frameCount++;
                LOG_DBG("StreamVP8: FrameCount = %d   Frame size= %d\n",frameCount, FrameSize);
            } else {
                FrameSize = 0;
                packet.uDataLength = 0;
                packet.pByteStream = NULL;
            }

            packet.bEOS = feof(ctx->file) ? true : false;
            LOG_DBG("StreamVP8: EOS %d is sent...\n", packet.bEOS);
            packet.bPTSValid = 0; // (pts != (uint32_t)-1);
            packet.llPts = 0; // packet.bPTSValid ? (1000 * pts / 9)  : 0;    // 100 ns scale

            if ((frameCount > (uint32_t)ctx->numFramesToDecode) || (packet.bEOS) || (packet.uDataLength == 0))
            {
                signal_stop = 1;
                break;
            }
            if (NvMediaParserParse(ctx->parser, &packet) != NVMEDIA_STATUS_OK)
                return -1;
        }

        NvMediaParserFlush(ctx->parser);
        rewind(ctx->file);

        if(ctx->loop != 1 && !signal_stop) {
            if(ctx->stopDecoding) {
                ctx->stopDecoding = false;
                ctx->decodeCount = 0;
                ctx->totalDecodeTime = 0;
            }
            LOG_MSG("loop count: %d/%d \n", i+1, ctx->loop);
        } else
            break;
    }
    free(bits);

    return 0;
}

static int Decode_orig(VideoDemoTestCtx *ctx)
{
    uint8_t *bits;
    int i;
    uint32_t readSize = READ_SIZE;

    bits = malloc(readSize);
    if (!bits) {
        LOG_ERR("Decode_orig: Failed allocating memory for file buffer\n");
        return -1;
    }

    LOG_DBG("Decode_orig: Starting %d loops of decode\n", ctx->loop);

    for(i = 0; (i < ctx->loop) || (ctx->loop == -1); i++) {
        LOG_DBG("Decode_orig: loop %d out of %d\n", i, ctx->loop);
        if(ctx->bRCVfile) {
            uint8_t header[32] = { 0, };
            if(fread(header, 32, 1, ctx->file)) {
                int i;
                uint32_t startCode = false;
                // Check start code
                for(i = 0; i <= (32 - 4); i++) {
                    if(!header[i + 0] && !header[i + 1] && header[i + 2] == 0x01 &&
                        (header[i + 3] == 0x0D || header[i + 3] == 0x0F)) {
                        startCode = true;
                        break;
                    }
                }
                if(!startCode) {
                    StreamVC1SimpleProfile(ctx);
                }
            }
        } else {
            while (!feof(ctx->file) && !ctx->stopDecoding && !signal_stop) {
                size_t len;
                NvMediaBitStreamPkt packet;
                memset(&packet, 0, sizeof(NvMediaBitStreamPkt));
                len = fread(bits, 1, readSize, ctx->file);
                packet.uDataLength = (uint32_t) len;
                packet.pByteStream = bits;
                packet.bEOS = feof(ctx->file) ? true : false;
                LOG_DBG("Decode_orig: EOS %d is sent...\n", packet.bEOS);
                packet.bPTSValid = 0; // (pts != (uint32_t)-1);
                packet.llPts = 0; // packet.bPTSValid ? (1000 * pts / 9)  : 0;    // 100 ns scale
                if (NvMediaParserParse(ctx->parser, &packet) != NVMEDIA_STATUS_OK) {
                    LOG_ERR("Decode_orig: NvMediaParserParse returned with failure\n");
                    return -1;
                }
            }
            NvMediaParserFlush(ctx->parser);
            LOG_DBG("Decode_orig: Finished decoding. Flushing parser and display\n");
        }

        rewind(ctx->file);

        if(ctx->loop != 1 && !signal_stop) {
            if(ctx->stopDecoding) {
                ctx->stopDecoding = false;
                ctx->decodeCount = 0;
                ctx->totalDecodeTime = 0;
            }
            LOG_MSG("loop count: %d/%d \n", i+1, ctx->loop);
        } else
            break;

    }

    free(bits);

    return 0;
}

int Decode(VideoDemoTestCtx *ctx)
{
    if(ctx->eCodec == NVMEDIA_VIDEO_CODEC_AV1)
    {
        return StreamVP8(ctx);
    }
    else
    {
        if((ctx->eCodec == NVMEDIA_VIDEO_CODEC_VP8) || (ctx->eCodec == NVMEDIA_VIDEO_CODEC_VP9))
            return StreamVP8(ctx);
        else
            return Decode_orig(ctx);
    }
}

static void sig_handler(int sig)
{
    LOG_INFO("sig_handler: Received Signal: %d\n", sig);
    signal_stop = 1;
}

int main(int argc, char *argv[])
{
    VideoDemoTestCtx ctx;
    TestArgs testArgs;
    int status;

    memset(&ctx, 0, sizeof(ctx));
    memset(&testArgs, 0, sizeof(testArgs));

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    LOG_DBG("main: Parsing command line arguments\n");
    status = ParseArgs(argc, argv, &testArgs);
    if(status) {
        if (status != 1)
            PrintUsage();
        return -1;
    }

    LOG_DBG("main: Initializing test context\n");
    if(Init(&ctx, &testArgs)) {
        LOG_ERR("Init failed\n");
        return -1;
    }

    LOG_DBG("main: Starting decode process\n");
    if(Decode(&ctx)) {
        LOG_ERR("Decode failed\n");
        Deinit(&ctx);
        LOG_MSG("\ntotal failures: 1\n");
        return -1;
    }

    LOG_DBG("main: Deinitializing\n");
    Deinit(&ctx);

    if(ctx.showDecodeTimimg) {
        //get decoding time info
        LOG_MSG("\nTotal Decoding time for %d frames: %.3f ms\n", ctx.decodeCount, ctx.totalDecodeTime);
        LOG_MSG("Decoding time per frame %.4f ms \n", ctx.totalDecodeTime / ctx.decodeCount);
    }

    LOG_MSG("Total %d frames decoded\n", ctx.FrameCount);

    if ((ctx.generateCRC == true) && (ctx.CRCGenComplete == true))
    {
        LOG_MSG("CRC file is generated at %s\n", testArgs.crcFilePath);
    }

    if ((ctx.YUVSaveComplete == true) && (ctx.OutputYUVFilename))
    {
        LOG_MSG("YUV file is saved at %s\n", ctx.OutputYUVFilename);
    }

    if ((ctx.checkCRC == true))
    {
        if (ctx.CRCResult == true)
        {
            LOG_MSG("\n*** Test Passed: DECODING PROCESS ENDED SUCCESSFULY***\n");
            LOG_MSG(" total failures: 0\n");
        }
        else
        {
            LOG_MSG("***** Test: Failed ***** \n");
            LOG_MSG(" total failures: 1\n");
        }
    }
    else
    {
        LOG_MSG("\n*** Test Passed: DECODING PROCESS ENDED SUCCESSFULY***\n");
        LOG_MSG("\ntotal failures: 0\n");
    }

    return 0;
}
