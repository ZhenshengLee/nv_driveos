/*
 * Copyright (c) 2017-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

/* standard headers */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <signal.h>

/* Nvidia headers */
#include "cmdline.h"
#include "log_utils.h"
#include "nvmedia_ldc.h"
#include "nvmedia_ldc_sci.h"
#if !NV_IS_SAFETY
#include "nvmedia_ldc_util.h"
#endif

static size_t Fread(void *ptr, size_t size, size_t count, FILE *stream)
{
    return fread(ptr, size, count, stream);
}

static size_t Fwrite(void *ptr, size_t size, size_t count, FILE *stream)
{
    return fwrite(ptr, size, count, stream);
}

static NvMediaBool quit_flag = NVMEDIA_FALSE;
static void
SigHandler (int signum) {
    signal(SIGINT, SIG_IGN);
    signal(SIGTERM, SIG_IGN);
    signal(SIGQUIT, SIG_IGN);
    signal(SIGHUP, SIG_IGN);
    signal(SIGKILL, SIG_IGN);
    signal(SIGSTOP, SIG_IGN);

    quit_flag = NVMEDIA_TRUE;

    signal(SIGINT, SIG_DFL);
    signal(SIGTERM, SIG_DFL);
    signal(SIGQUIT, SIG_DFL);
    signal(SIGHUP, SIG_DFL);
    signal(SIGKILL, SIG_DFL);
    signal(SIGSTOP, SIG_DFL);
}

static void
SigSetup (void) {
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = SigHandler;
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);
    sigaction(SIGQUIT, &action, NULL);
    sigaction(SIGHUP, &action, NULL);
    sigaction(SIGKILL, &action, NULL);
    sigaction(SIGSTOP, &action, NULL);
}

/* Checks NvMediaStatus and jumps to fail */
#define CHK_STATUS_AND_RETURN(function, status)                     \
    if (status != NVMEDIA_STATUS_OK) {                              \
        LOG_ERR("%s failed! status = %x\n", function, status);      \
        ret = -1;                                                   \
        goto ldc_cleanup;                                           \
    }

/* Checks NvSci status and jumps to fail */
#define CHK_SCI_STATUS_AND_RETURN(function, status)                 \
    if (status != NvSciError_Success) {                             \
        LOG_ERR("%s failed! status = %x\n", function, status);      \
        ret = -1;                                                   \
        goto ldc_cleanup;                                           \
    }

/* Checks ret and jumps to fail */
#define CHK_AND_RETURN(function)                                    \
    if (ret != 0) {                                                 \
        LOG_ERR("%s failed!\n", function);                          \
        goto ldc_cleanup;                                           \
    }

/* Global variables */

/* gRefTnr3Params contains the typical/recommended settings for different lighting conditions */
NvMediaLdcTnrParameters gRefTnr3Params[] = {
    {
    /* Bright */
/* spatialSigmaLuma */   2,
/* spatialSigmaChroma */ 2,
/* rangeSigmaLuma */     15,
/* rangeSigmaChroma */   45,
/* sadMultiplier */      1.0f,
/* sadWeightLuma */      0.5f,
/* alphaSmoothEnable */  0,
/* alphaIncreaseCap */   0.125f,
/* alphaScaleIIR */      0.5f,
/* alphaMaxLuma */       0.9277f,
/* alphaMinLuma */       0.0,
/* alphaMaxChroma */     0.9277f,
/* alphaMinChroma */     0.0,
/* betaX1 */             0.3906f,
/* betaX2 */             1.0f,
/* maxBeta */            1.0f,
/* minBeta */            0.3906f
    },

    {
    /* Medium */
/* spatialSigmaLuma */   3,
/* spatialSigmaChroma */ 3,
/* rangeSigmaLuma */     20,
/* rangeSigmaChroma */   60,
/* sadMultiplier */      0.7143f,
/* sadWeightLuma */      0.5f,
/* alphaSmoothEnable */  1,
/* alphaIncreaseCap */   0.125f,
/* alphaScaleIIR */      0.5,
/* alphaMaxLuma */       0.8789f,
/* alphaMinLuma */       0.0,
/* alphaMaxChroma */     0.8789f,
/* alphaMinChroma */     0.1953,
/* betaX1 */             0.3906f,
/* betaX2 */             1.0f,
/* maxBeta */            1.0f,
/* minBeta */            0.3906f
    },
    {
    /* Low */
/* spatialSigmaLuma */   4,
/* spatialSigmaChroma */ 4,
/* rangeSigmaLuma */     40,
/* rangeSigmaChroma */   120,
/* sadMultiplier */      0.5f,
/* sadWeightLuma */      0.5f,
/* alphaSmoothEnable */  1,
/* alphaIncreaseCap */   0.125f,
/* alphaScaleIIR */      0.5,
/* alphaMaxLuma */       0.7324f,
/* alphaMinLuma */       0.0,
/* alphaMaxChroma */     0.7324f,
/* alphaMinChroma */     0.1953f,
/* betaX1 */             0.3906f,
/* betaX2 */             1.0f,
/* maxBeta */            1.0f,
/* minBeta */            0.3906f
    },
    {
    /* Very low */
/* spatialSigmaLuma */   6,
/* spatialSigmaChroma */ 6,
/* rangeSigmaLuma */     200,
/* rangeSigmaChroma */   200,
/* sadMultiplier */      0.3571f,
/* sadWeightLuma */      0.5f,
/* alphaSmoothEnable */  1,
/* alphaIncreaseCap */   0.125f,
/* alphaScaleIIR */      0.5,
/* alphaMaxLuma */       0.7324f,
/* alphaMinLuma */       0.0,
/* alphaMaxChroma */     0.7324f,
/* alphaMinChroma */     0.1953f,
/* betaX1 */             0.3906f,
/* betaX2 */             1.0f,
/* maxBeta */            1.0f,
/* minBeta */            0.3906f
    }
};

static NvMediaStatus
GetTimeMicroSec(uint64_t *uTime)
{
    struct timespec t;
#if !(defined(CLOCK_MONOTONIC) && defined(_POSIX_MONOTONIC_CLOCK) && _POSIX_MONOTONIC_CLOCK >= 0 && _POSIX_TIMERS > 0)
    struct timeval tv;
#endif

    if(!uTime)
        return NVMEDIA_STATUS_BAD_PARAMETER;

#if !(defined(CLOCK_MONOTONIC) && defined(_POSIX_MONOTONIC_CLOCK) && _POSIX_MONOTONIC_CLOCK >= 0 && _POSIX_TIMERS > 0)
    gettimeofday(&tv, NULL);
    t.tv_sec = tv.tv_sec;
    t.tv_nsec = tv.tv_usec*1000L;
#else
    clock_gettime(CLOCK_MONOTONIC, &t);
#endif

    *uTime = (uint64_t)t.tv_sec * 1000000LL + (uint64_t)t.tv_nsec / 1000LL;
    return NVMEDIA_STATUS_OK;
}

/* Reads Warp map described via config file and warp map file.
 * Returns
 *  0 => Success
 * -1 => Failure
 */
static int
sReadWarpMap(
        NvMediaLdcControlPoint **pWarpMapBuf, /* Pointer to buffer for storing the warp map */
        uint32_t numControlPoints, /* Number of control points to read from the file */
        char *WarpMapFile) /* Warp map file */
{
    size_t size = 0;
    FILE *f = NULL;

    if (!pWarpMapBuf) {
        LOG_ERR("Invalid Warp map buffer pointer\n");
        return -1;
    }

    /* Read the warp map file */
    size = numControlPoints * sizeof(float_t) * 2;
    if (!size) {
        LOG_ERR("Invalid numControlPoints\n");
        return -1;
    }

    *pWarpMapBuf = calloc(1, size);
    if (!*pWarpMapBuf) {
        LOG_ERR("Failed to allocate WarpMap\n");
        return -1;
    }

    f = fopen(WarpMapFile, "rb");
    if (!f) {
        LOG_ERR("Failed to open WarpMapFile:%s\n", WarpMapFile);
        return -1;
    }

    if (fread(*pWarpMapBuf, 1, size, f) != size) {
        LOG_ERR("Failed to read WarpMapFile:%s\n", WarpMapFile);
        fclose(f);
        return -1;
    }

    fclose(f);

    return 0;
}

static void *pixelMasksData;

/* Reads bit mask map from bit mask file.
 * Returns,
 *  0 => Success
 * -1 => Failure
 */
static int
sReadBitMaskMap(
        NvMediaLdcMaskMapParameters *pBitMaskMap, /* NvMedia LDC Bit Mask Map */
        char *bitMaskFile)                        /* Bit Mask map file */
{
    /* Read bit map file */
    size_t size = (size_t) (pBitMaskMap->width) * (size_t) (pBitMaskMap->height);
    FILE *f = NULL;

    if (size) {
        pixelMasksData = calloc(1, size);
        if (!pixelMasksData) {
            LOG_ERR("Failed to allocate BitMaskMap mapPtr\n");
            return -1;
        }
    }

    f = fopen(bitMaskFile, "rb");
    if (!f) {
        LOG_ERR("Failed to open bitMaskFile:%s\n", bitMaskFile);
        return -1;
    }

    if (fread(pixelMasksData, 1, size, f) != size) {
        LOG_ERR("Failed to read bitMaskFile:%s\n", bitMaskFile);
        fclose(f);
        return -1;
    }

    fclose(f);

    pBitMaskMap->pixelMasks = pixelMasksData;

    return 0;
}

/* Creates an NvSciBuf
 * Supported color formats are:
 *  - Y plane 8 bit per pixel
 *  - YUV420 semi-planar
 *  - YUV420 planar, YV12
 */
static NvMediaStatus CreateSciBuf(uint32_t planes,
                                  uint32_t width,
                                  uint32_t height,
                                  NvMediaLdc *pLDC,
                                  NvSciBufModule sciBufModule,
                                  NvSciBufObj *buf,
                                  NvSciBufAttrList *sciBufAttrListReconciled) {
    NvSciBufAttrList sciBufAttrList = NULL;
    NvSciBufAttrList sciBufAttrListConflict = NULL;

    NvMediaStatus status = NVMEDIA_STATUS_OK;

    NvSciError sciSyncResult = NvSciError_Success;

    sciSyncResult = NvSciBufAttrListCreate(sciBufModule,
                                           &sciBufAttrList);
    if (sciSyncResult != NvSciError_Success) {
        status = NVMEDIA_STATUS_ERROR;
        LOG_ERR("NvSciBufAttrListCreate FAILED\n");
        goto createscibuf_cleanup;
    }

    status = NvMediaLdcFillNvSciBufAttrList(pLDC, sciBufAttrList);
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("NvMediaLdcFillNvSciBufAttrList FAILED\n");
        goto createscibuf_cleanup;
    }

    NvSciBufType bufType = NvSciBufType_Image;
    bool cpuAccess = true;
    NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
    uint32_t planeCount = planes;
    NvSciBufAttrValColorFmt planeColorFmt[3] = {0};
    uint32_t planeWidths[3] = {0};
    uint32_t planeHeights[3] = {0};
    NvSciBufAttrValImageScanType scanType = NvSciBufScan_ProgressiveType;
    planeColorFmt[0] = NvSciColor_Y8;
    planeWidths[0] = width;
    planeHeights[0] = height;
    if (planes == 2) {
        planeColorFmt[1] = NvSciColor_V8U8;
        planeWidths[1] = width / 2;
        planeHeights[1] = height / 2;
    }
    if (planes == 3) {
        planeColorFmt[1] = NvSciColor_U8;
        planeWidths[1] = width / 2;
        planeHeights[1] = height / 2;
        planeColorFmt[2] = NvSciColor_V8;
        planeWidths[2] = width / 2;
        planeHeights[2] = height / 2;
    }

    NvSciBufAttrKeyValuePair attrs[] = {
        { NvSciBufGeneralAttrKey_Types,
          &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess,
          &cpuAccess, sizeof(cpuAccess) },
        { NvSciBufImageAttrKey_Layout,
          &layout, sizeof(layout) },
        { NvSciBufImageAttrKey_PlaneCount,
          &planeCount, sizeof(planeCount) },
        { NvSciBufImageAttrKey_PlaneColorFormat,
          planeColorFmt, sizeof(NvSciBufAttrValColorFmt) * planeCount },
        { NvSciBufImageAttrKey_PlaneWidth,
          planeWidths, sizeof(uint32_t) * planeCount },
        { NvSciBufImageAttrKey_PlaneHeight,
          planeHeights, sizeof(uint32_t) * planeCount },
        { NvSciBufImageAttrKey_ScanType,
          &scanType, sizeof(scanType) },
    };

    sciSyncResult = NvSciBufAttrListSetAttrs(
                sciBufAttrList,
                attrs,
                sizeof(attrs)/sizeof(NvSciBufAttrKeyValuePair));

    if (sciSyncResult != NvSciError_Success) {
        status = NVMEDIA_STATUS_ERROR;
        LOG_ERR("NvSciBufAttrListSetAttrs FAILED\n");
        goto createscibuf_cleanup;
    }

    sciSyncResult = NvSciBufAttrListReconcile(&sciBufAttrList,
                                              1,
                                              sciBufAttrListReconciled,
                                              &sciBufAttrListConflict);
    if (sciSyncResult != NvSciError_Success) {
        status = NVMEDIA_STATUS_ERROR;
        LOG_ERR("NvSciBufAttrListReconcile FAILED\n");
        goto createscibuf_cleanup;
    }

    sciSyncResult = NvSciBufObjAlloc(*sciBufAttrListReconciled, buf);
    if (sciSyncResult != NvSciError_Success) {
        status = NVMEDIA_STATUS_ERROR;
        LOG_ERR("NvSciBufObjAlloc FAILED\n");
        goto createscibuf_cleanup;
    }

    status = NvMediaLdcRegisterNvSciBufObj(pLDC, *buf);

createscibuf_cleanup:

    if (sciBufAttrList) {
        NvSciBufAttrListFree(sciBufAttrList);
    }

    if (sciBufAttrListConflict) {
        NvSciBufAttrListFree(sciBufAttrListConflict);
    }

    return status;
}

/* Image Buffer attributes */
typedef struct {
    uint64_t size;
    NvSciBufAttrValImageLayoutType layout;
    uint32_t planecount;
    int32_t width[3];
    int32_t height[3];
    int32_t pitch[3];
    uint32_t bpp[3];
    uint64_t offset[3];
    uint8_t *cpuPtr;
} BufferAttrs;

/* Fills up the Image Buffer attributes of NvSciBufObj */
static NvMediaStatus
GetBufferAttrs(NvSciBufObj buf,
               NvSciBufAttrList *sciBufAttrListReconciled,
               BufferAttrs *attrs)
{
    NvSciError status = NvSciBufObjGetCpuPtr(buf, (void **)&attrs->cpuPtr);
    if (status != NvSciError_Success) {
        return NVMEDIA_STATUS_ERROR;
    }

    NvSciBufAttrKeyValuePair imgattrs[] = {
        { NvSciBufImageAttrKey_Size, NULL, 0 },
        { NvSciBufImageAttrKey_Layout, NULL, 0 },
        { NvSciBufImageAttrKey_PlaneCount, NULL, 0 },
        { NvSciBufImageAttrKey_PlaneWidth, NULL, 0 },
        { NvSciBufImageAttrKey_PlaneHeight, NULL, 0 },
        { NvSciBufImageAttrKey_PlanePitch, NULL, 0 },
        { NvSciBufImageAttrKey_PlaneBitsPerPixel, NULL, 0 },
        { NvSciBufImageAttrKey_PlaneOffset, NULL, 0}
    };

    status = NvSciBufAttrListGetAttrs(*sciBufAttrListReconciled,
                                      imgattrs,
                                      sizeof(imgattrs) /
                                              sizeof(NvSciBufAttrKeyValuePair));
    if(status != NvSciError_Success) {
        return NVMEDIA_STATUS_ERROR;
    }

    attrs->size = *((uint64_t*)(imgattrs[0].value));
    attrs->layout = *((NvSciBufAttrValImageLayoutType*)(imgattrs[1].value));
    attrs->planecount = *((uint32_t*)(imgattrs[2].value));

    memcpy(attrs->width,
           (int32_t*)(imgattrs[3].value),
           attrs->planecount * sizeof(int32_t));
    memcpy(attrs->height,
           (int32_t*)(imgattrs[4].value),
           attrs->planecount * sizeof(int32_t));
    memcpy(attrs->pitch,
           (int32_t*)(imgattrs[5].value),
           attrs->planecount * sizeof(int32_t));
    memcpy(attrs->bpp,
           (uint32_t*)(imgattrs[6].value),
           attrs->planecount * sizeof(uint32_t));
    memcpy(attrs->offset,
           (uint64_t*)(imgattrs[7].value),
           attrs->planecount * sizeof(uint64_t));

    return NVMEDIA_STATUS_OK;
}

/* Defines a function that operates on file: fread/fwrite */
typedef size_t (*FileFunction)(void *, size_t, size_t, FILE *);

/* Reads or writes a plane to or from a file */
static inline NvMediaStatus
ReadWritePlane(FILE *f,
               const BufferAttrs *attrs,
               uint8_t plane,
               bool semiplanar,
               FileFunction func)
{
    uint64_t offset = attrs->offset[plane];
    uint64_t width = attrs->width[plane] * (attrs->bpp[plane] >> 3);
    uint64_t pitch = attrs->pitch[plane];

    if (!semiplanar) {
        for (int j = 0; j < attrs->height[plane]; j++) {
            if (func(attrs->cpuPtr + offset, 1, width, f) != width) {
                return NVMEDIA_STATUS_ERROR;
            }
            offset += pitch;
        }
    } else { // semiplanar : read two planes into one
        for (int j = 0; j < attrs->height[plane]; j++) {
            for (int k = 0; k < attrs->width[plane]; k++) {
                if (func(attrs->cpuPtr + offset + k * 2 + 1, 1, 1, f) != 1) {
                    return NVMEDIA_STATUS_ERROR;
                }
            }
            offset += pitch;
        }
        offset = attrs->offset[plane];
        for (int j = 0; j < attrs->height[plane]; j++) {
            for (int k = 0; k < attrs->width[plane]; k++) {
                if (func(attrs->cpuPtr + offset + k * 2, 1, 1, f) != 1) {
                    return NVMEDIA_STATUS_ERROR;
                }
            }
            offset += pitch;
        }
    }
    return NVMEDIA_STATUS_OK;
}

/* Reads or writes planes to or from a file */
static inline NvMediaStatus
ReadWriteAllPlanes(FILE *f,
                   const BufferAttrs *attrs,
                   FileFunction func)
{
     NvMediaStatus status = NVMEDIA_STATUS_OK;

     if (attrs->planecount == 2) {
        status = ReadWritePlane(f, attrs, 0, false, func);
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("Error processing the YUV file");
            return status;
        }
        status = ReadWritePlane(f, attrs, 1, true, func);
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("Error processing the YUV file");
            return status;
        }
    } else {
        for (uint32_t i = 0; i < attrs->planecount; i++) {
            status = ReadWritePlane(f, attrs, i, false, func);
            if (status != NVMEDIA_STATUS_OK) {
                LOG_ERR("Error processing the YUV file");
                return status;
            }
        }
    }

    return status;
}

/* Reads YUV file to the NvSciBufObj memory.
 * Currently only 8bit per component YUV420 planar files are supported.
 * The supported NvSciBufObj formats are
 *   T_Y8___V8U8_N420 (semiplanar)
 *   T_Y8___U8___V8_N420 (YUV420 planar, YV12)
 */
static NvMediaStatus
ReadYUV(char *file,
        uint32_t frameIdx, /* frame number */
        NvSciBufObj buf,
        NvSciBufAttrList *sciBufAttrListReconciled)
{
    BufferAttrs attrs;
    memset(&attrs, 0, sizeof(attrs));
    NvMediaStatus status = GetBufferAttrs(buf,
                                          sciBufAttrListReconciled,
                                          &attrs);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    FILE *f = fopen(file, "rb");
    if (!f) {
        LOG_ERR("Can not open file");
        return NVMEDIA_STATUS_ERROR;
    }

    uint64_t ySize = attrs.width[0] * attrs.height[0] * (attrs.bpp[0] >> 3);
    uint64_t uSize = attrs.width[1] * attrs.height[1] * (attrs.bpp[1] >> 3);
    uint64_t vSize = attrs.width[2] * attrs.height[2] * (attrs.bpp[2] >> 3);

    if (frameIdx > 0) {
        if (fseek(f, frameIdx * (ySize + uSize + vSize), SEEK_SET) == -1) {
            LOG_ERR("Error seeking the YUV file");
            fclose(f);
            return NVMEDIA_STATUS_ERROR;
        }
    }

    status = ReadWriteAllPlanes(f, &attrs, Fread);

    fclose(f);
    return status;
}

/* Writes YUV file from the NvSciBufObj memory.
 * Currently only 8bit per component YUV420 planar files are supported.
 * The supported NvSciBufObj formats are
 *   T_Y8___V8U8_N420 (semiplanar)
 *   T_Y8___U8___V8_N420 (YUV420 planar, YV12)
 */
static NvMediaStatus
WriteYUV(char *file,
         NvSciBufObj buf,
         NvSciBufAttrList *sciBufAttrListReconciled,
         NvMediaBool append)
{
    BufferAttrs attrs;
    memset(&attrs, 0, sizeof(attrs));
    NvMediaStatus status = GetBufferAttrs(buf,
                                          sciBufAttrListReconciled,
                                          &attrs);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    FILE *f = fopen(file, append ? "ab" : "wb");
    if (!f) {
        LOG_ERR("Can not open file");
        return NVMEDIA_STATUS_ERROR;
    }

    status = ReadWriteAllPlanes(f, &attrs, Fwrite);

    fclose(f);
    return status;
}

/* Rounds n up to nearest multiple of m */
static uint32_t
RoundUp(uint32_t n, uint32_t m)
{
    return (n + m - 1) / m * m;
}

/* Main LDC test function. */
int main(int argc, char *argv[])
{
    TestArgs args;

    /* Surfaces */
    NvSciBufObj curr = NULL, prev = NULL, temp = NULL;
    NvSciBufObj out = NULL, xSob = NULL, dSample = NULL;

    /* NvMediaLDC handle */
    NvMediaLdc *pLDC = NULL;

    /* Dimensions */
    uint32_t srcWidth, srcHeight;
    uint32_t dstWidth, dstHeight;

    uint32_t ret = 0, frameIdx = 0;

    uint32_t numControlPoints = 0;
    NvMediaLdcControlPoint *controlPoints = NULL;

    /* NvSciSync */
    NvSciSyncModule sciSyncModule = NULL;

    NvSciSyncAttrList sciSyncAttrList = NULL;
    NvSciSyncAttrList sciSyncAttrListCpu = NULL;
    NvSciSyncAttrList syncReconciled = NULL;
    NvSciSyncAttrList syncConflict = NULL;

    NvSciSyncObj eofSyncObj = NULL;
    NvSciSyncCpuWaitContext waitCtx = NULL;
    NvSciSyncFence eofFence = NvSciSyncFenceInitializer;

    /* NvSciBuf */
    NvSciBufModule sciBufModule = NULL;

    NvSciBufAttrList sciBufAttrListSrc = NULL;
    NvSciBufAttrList sciBufAttrListDst = NULL;
    NvSciBufAttrList sciBufAttrListPrev = NULL;
    NvSciBufAttrList sciBufAttrListXSobel = NULL;
    NvSciBufAttrList sciBufAttrListXSobelDS = NULL;

    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvSciError sciSyncResult = NvSciError_Success;

    uint64_t totalOutTimeTaken = 0;

    NvMediaVersion version;

    SigSetup();

    /* Read configuration from command line and config file */
    memset(&args, 0, sizeof(TestArgs));

    /* ParseArgs parses the command line and the LDC configuration file
     * and populates all initParams and run time configuration
     * into appropriate structures within args.
     */
    if (ParseArgs(argc, argv, &args)) {
        PrintUsage();
        return -1;
    }

    /* Get copies of or reference to config populated by ParseArgs */
    NvMediaLdcParametersAttributes *pParamsAttrs = &args.paramsAttrs;
    NvMediaLdcMaskMapParameters *pMaskMap = &args.maskMapParams;

    pParamsAttrs->maxWarpMapWidth = 0;
    pParamsAttrs->maxWarpMapHeight = 0;
    pParamsAttrs->maxDstWidth = 0;
    pParamsAttrs->maxDstHeight = 0;

    srcWidth = args.srcWidth;
    srcHeight = args.srcHeight;
    dstWidth = args.dstWidth;
    dstHeight = args.dstHeight;

    NvMediaLdcIptParameters *pIpt = &args.iptParams;
    NvMediaLdcRegionParameters *pRegion = &(pIpt->regionParams);

    /* Read bit mask map file, if present */
    if (pParamsAttrs->enableMaskMap) {
        ret = sReadBitMaskMap(pMaskMap, args.bitMaskFile);
        CHK_AND_RETURN("sReadBitMaskMap");
    }

    /* MANDATORY STEP: Check version */
    status = NvMediaLdcGetVersion(&version);
    CHK_STATUS_AND_RETURN("NvMediaLdcGetVersion", status);
    if (status == NVMEDIA_STATUS_OK) {
        LOG_INFO("Library version: %u.%u.%u\n", version.major, version.minor, version.patch);
        LOG_INFO("Header version:  %u.%u.%u\n",
                 NVMEDIA_LDC_VERSION_MAJOR,
                 NVMEDIA_LDC_VERSION_MINOR,
                 NVMEDIA_LDC_VERSION_PATCH);
        if ((version.major != NVMEDIA_LDC_VERSION_MAJOR) ||
            (version.minor != NVMEDIA_LDC_VERSION_MINOR)) {
            LOG_ERR("Library and Header mismatch!\n");
        }
    }
    /* Check version done */

    /* MANDATORY STEP: Create LDC Handle */
    status = NvMediaLdcCreate(&pLDC, NULL);
    CHK_STATUS_AND_RETURN("NvMediaLdcCreate", status);
    LOG_INFO("NvMediaLdcCreate done. LDC Handle:%p\n", pLDC);
    /* Create LDC Handle done */

    /* MANDATORY STEP: Create parameters */
    NvMediaLdcParameters params = 0;

    if (args.generateWarpMap || args.applyWarpMap) {
        /* The maximum possible number of control points in a warp map is the number of pixels in
         * the destination, rounded up to the warp map processing tile size of 64x16 pixels. */
        pParamsAttrs->maxWarpMapWidth = RoundUp(dstWidth, 64);
        pParamsAttrs->maxWarpMapHeight = RoundUp(dstHeight, 16);
    }

    if (pParamsAttrs->enableTnr || pParamsAttrs->enableMaskMap) {
        pParamsAttrs->maxDstWidth = dstWidth;
        pParamsAttrs->maxDstHeight = dstHeight;
    }

    status = NvMediaLdcCreateParameters(pLDC, pParamsAttrs, &params);
    CHK_STATUS_AND_RETURN("NvMediaLdcCreateParameters", status);
    /* Create parameters done */

    if (args.generateWarpMap && args.applyWarpMap) {
        LOG_ERR("generateWarpMap and applyWarpMap can not be enabled"
                "at the same time.\n");
        goto ldc_cleanup;
    }

    /* Generate/Feed the warp mapping */
    if (args.generateWarpMap) {
#if NV_IS_SAFETY
        LOG_ERR("Warp map generation is not supported for this build.");
        goto ldc_cleanup;
#else
        /* NvMediaLDC will generate the mapping */
        status = NvMediaLdcGetNumControlPoints(pRegion, &numControlPoints);
        CHK_STATUS_AND_RETURN("NvMediaLdcGetNumControlPoints", status);
        controlPoints = (NvMediaLdcControlPoint*)malloc(
                sizeof(NvMediaLdcControlPoint) * numControlPoints);

        status = NvMediaLdcGenWarpMap(&args.Kin,
                                      &args.X,
                                      &args.Kout,
                                      &args.lensDistortion,
                                      &args.dstRect,
                                      pRegion,
                                      numControlPoints,
                                      controlPoints);
        CHK_STATUS_AND_RETURN("NvMediaLdcGenWarpMap", status);
        LOG_DBG("Num control points generated: %u\n", numControlPoints);
        LOG_INFO("NvMediaLdcGenWarpMap done.\n");
#endif
    }
    else if (args.applyWarpMap) {
        /* Read the Warp Map from file */
        numControlPoints = args.numControlPoints;
        ret = sReadWarpMap(&controlPoints, numControlPoints, args.warpMapFile);
        CHK_AND_RETURN("sReadWarpMap");
    }

    /* Populate a Warp Map */
    NvMediaLdcWarpMapParameters warpMapParams;
    if (args.generateWarpMap || args.applyWarpMap) {
        memcpy(&warpMapParams.regionParams,
               pRegion,
               sizeof(warpMapParams.regionParams));
        warpMapParams.numControlPoints = numControlPoints;
        warpMapParams.controlPoints = controlPoints;
    }

    /* MANDATORY STEP: SciSync init */
    sciSyncResult = NvSciSyncModuleOpen(&sciSyncModule);
    CHK_SCI_STATUS_AND_RETURN("NvSciSyncModuleOpen", sciSyncResult);

    sciSyncResult = NvSciSyncAttrListCreate(sciSyncModule, &sciSyncAttrList);
    CHK_SCI_STATUS_AND_RETURN("NvSciSyncAttrListCreate", sciSyncResult);

    status = NvMediaLdcFillNvSciSyncAttrList(pLDC,
                                             sciSyncAttrList,
                                             NVMEDIA_SIGNALER);
    CHK_STATUS_AND_RETURN("NvMediaLdcFillNvSciSyncAttrList", status);

    sciSyncResult = NvSciSyncAttrListCreate(sciSyncModule, &sciSyncAttrListCpu);
    CHK_SCI_STATUS_AND_RETURN("NvSciSyncAttrListCreate", sciSyncResult);

    bool cpuAccess = true;
    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
    NvSciSyncAttrKeyValuePair cpuAttrKvPairs[] = {
        { NvSciSyncAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess) },
        { NvSciSyncAttrKey_RequiredPerm, &cpuPerm, sizeof(cpuPerm) },
    };
    size_t cpuAttrCount = sizeof(cpuAttrKvPairs) / sizeof(cpuAttrKvPairs[0]);
    sciSyncResult = NvSciSyncAttrListSetAttrs(sciSyncAttrListCpu, cpuAttrKvPairs, cpuAttrCount);
    CHK_SCI_STATUS_AND_RETURN("NvSciSyncAttrListSetAttrs", sciSyncResult);

    NvSciSyncAttrList sciSyncAttrLists[] = { sciSyncAttrList, sciSyncAttrListCpu };
    size_t sciSyncAttrListCount = sizeof(sciSyncAttrLists) / sizeof(sciSyncAttrLists[0]);
    sciSyncResult = NvSciSyncAttrListReconcile(sciSyncAttrLists,
                                               sciSyncAttrListCount,
                                               &syncReconciled,
                                               &syncConflict);
    CHK_SCI_STATUS_AND_RETURN("NvSciSyncAttrListReconcile", sciSyncResult);

    sciSyncResult = NvSciSyncObjAlloc(syncReconciled, &eofSyncObj);
    CHK_SCI_STATUS_AND_RETURN("NvSciSyncObjAlloc", sciSyncResult);

    status = NvMediaLdcRegisterNvSciSyncObj(pLDC, NVMEDIA_EOFSYNCOBJ, eofSyncObj);
    CHK_STATUS_AND_RETURN("NvMediaLdcRegisterNvSciSyncObj", status);

    status = NvMediaLdcSetNvSciSyncObjforEOF(pLDC, params, eofSyncObj);
    CHK_STATUS_AND_RETURN("NvMediaLdcSetNvSciSyncObjforEOF", status);

    sciSyncResult = NvSciSyncCpuWaitContextAlloc(sciSyncModule, &waitCtx);
    CHK_SCI_STATUS_AND_RETURN("NvSciSyncCpuWaitContextAlloc", sciSyncResult);

    LOG_INFO("NvSciSync init is done\n");
    /* SciSync init done */

    /* MANDATORY STEP: SciBuf init */
    sciSyncResult = NvSciBufModuleOpen(&sciBufModule);
    CHK_SCI_STATUS_AND_RETURN("NvSciBufModuleOpen", sciSyncResult);
    /* SciBuf init done */

    /* MANDATORY STEP: Create and Register surfaces */
    LOG_INFO("Creating surfaces:\n");
    status = CreateSciBuf(2,
                          srcWidth,
                          srcHeight,
                          pLDC,
                          sciBufModule,
                          &curr,
                          &sciBufAttrListSrc);
    CHK_STATUS_AND_RETURN("CreateSciBuf curr", status);
    LOG_INFO("Input surface %p created\n", curr);

    if ((pParamsAttrs->enableTnr) && (args.numFrames > 1)) {
        status = CreateSciBuf(2,
                              dstWidth,
                              dstHeight,
                              pLDC,
                              sciBufModule,
                              &prev,
                              &sciBufAttrListPrev);
        CHK_STATUS_AND_RETURN("CreateSciBuf prev", status);
        LOG_INFO("Prevous surface %p created\n", curr);
    }

    status = CreateSciBuf(2,
                          dstWidth,
                          dstHeight,
                          pLDC,
                          sciBufModule,
                          &out,
                          &sciBufAttrListDst);
    CHK_STATUS_AND_RETURN("CreateSciBuf out", status);
    LOG_INFO("Output surface %p created\n", out);

    if (args.writeXSobel) {
        /* Create Xob Surface */
        status = CreateSciBuf(1,
                              dstWidth,
                              dstHeight,
                              pLDC,
                              sciBufModule,
                              &xSob,
                              &sciBufAttrListXSobel);
        CHK_STATUS_AND_RETURN("CreateSciBuf xSob", status);
        LOG_INFO("xSobel surface %p created\n", xSob);
    }

    if (args.writeXSobelDS) {
        /* Create Down Sample surface */
        status = CreateSciBuf(1,
                              dstWidth / 4,
                              dstHeight / 4,
                              pLDC,
                              sciBufModule,
                              &dSample,
                              &sciBufAttrListXSobelDS);
        CHK_STATUS_AND_RETURN("CreateSciBuf dSample", status);
        LOG_INFO("Downsampled surface %p created\n", dSample);
    }
    /* Create and Register surfaces done */

    /* MANDATORY STEP: Setting parameters */
    status = NvMediaLdcSetFilter(pLDC, params, args.filter);
    CHK_STATUS_AND_RETURN("NvMediaLdcSetFilter", status);
    LOG_DBG("  NvMediaLdcSetFilter called");

    status = NvMediaLdcSetGeometry(pLDC, params, &args.srcRect, &args.dstRect);
    CHK_STATUS_AND_RETURN("NvMediaLdcSetGeometry", status);
    LOG_DBG("  NvMediaLdcSetGeometry called");

    if (args.enableGeotrans) {
        status = NvMediaLdcSetIptParameters(pLDC, params, &args.iptParams);
        CHK_STATUS_AND_RETURN("NvMediaLdcSetIptParameters", status);
        LOG_DBG("  NvMediaLdcSetIptParameters called");
    }

    if (args.generateWarpMap || args.applyWarpMap) {
        status = NvMediaLdcSetWarpMapParameters(pLDC, params, &warpMapParams);
        CHK_STATUS_AND_RETURN("NvMediaLdcSetWarpMapParameters", status);
        LOG_DBG("  NvMediaLdcSetWarpMapParameters called");
    }

    if (pParamsAttrs->enableMaskMap) {
        status = NvMediaLdcSetMaskMapParameters(pLDC, params, &args.maskMapParams);
        CHK_STATUS_AND_RETURN("NvMediaLdcSetMaskMapParameters", status);
        LOG_DBG("  NvMediaLdcSetMaskMapParameters called");
    }

    if (pParamsAttrs->enableTnr) {
        status = NvMediaLdcSetTnrParameters(pLDC, params, &args.tnrParams);
        CHK_STATUS_AND_RETURN("NvMediaLdcSetTnrParameters", status);
        LOG_DBG("  NvMediaLdcSetTnrParameters called");
    }

    status = NvMediaLdcSetChecksumMode(pLDC, params, args.checksumMode);
    CHK_STATUS_AND_RETURN("NvMediaLdcSetChecksumMode", status);
    LOG_DBG("  NvMediaLdcSetChecksumMode called");
    /* Setting parameters done */

    /* Start main processing loop */
    for (
            frameIdx = 0;
            ((frameIdx < args.numFrames) && (quit_flag != NVMEDIA_TRUE));
            frameIdx++) {

        LOG_DBG("Processing frame:%u\n", frameIdx);

        /* Read input file */
        status = ReadYUV(args.inFile,
                         frameIdx, /* frame number */
                         curr,
                         /* In case of several frames, we might be using
                          * not the list the buffer has been created with.
                          * The assumption is in case of rotation, all the
                          * buffers have the same attributes.
                          */
                         &sciBufAttrListSrc);
        CHK_STATUS_AND_RETURN("ReadYUV", status);
        LOG_INFO("ReadImage(frame:%u from %s in to curr:%p) done\n", frameIdx, args.inFile, curr);

        /* MANDATORY STEP: Setting a source surface */
        status = NvMediaLdcSetSrcSurface(pLDC, params, curr);
        CHK_STATUS_AND_RETURN("NvMediaLdcSetSrcSurface", status);
        /* Setting a source surface done */

        /* Setting previous surface for TNR */
        if (prev) {
            status = NvMediaLdcSetPreviousSurface(pLDC, params, prev);
            CHK_STATUS_AND_RETURN("NvMediaLdcSetPreviousSurface", status);
        }

        /* MANDATORY STEP: Setting a destination surface */
        status = NvMediaLdcSetDstSurface(pLDC, params, out);
        CHK_STATUS_AND_RETURN("NvMediaLdcSetDstSurface", status);
        /* Setting a destination surface done */

        /* Setting an xSobel surface */
        if (args.writeXSobel) {
            status = NvMediaLdcSetXSobelDstSurface(pLDC, params, xSob);
            CHK_STATUS_AND_RETURN("NvMediaLdcSetXSobelDstSurface", status);
        }

        /* Setting a downsampled xSobel surface */
        if (args.writeXSobelDS) {
            status = NvMediaLdcSetDownsampledXSobelDstSurface(pLDC,
                                                              params,
                                                              dSample);
            CHK_STATUS_AND_RETURN("NvMediaLdcSetDownsampledXSobelDstSurface",
                                  status);
        }

        /* Update TNR3 Parameters */
        if (args.updateTnrParams) {
            /* Update TNR3 Parameters
             * The below code rotates through a pre-determined reference
             * settings for different lighting conditions to demo
             * the usage of API.
             * This illustrates how parameters can be updated in between
             * NvMediaLdcProcess() calls, any Set function can be called
             * in the same manner.
             * */
            NvMediaLdcTnrParameters *tnrParams = NULL;
            uint32_t numRef = sizeof(gRefTnr3Params) / sizeof(NvMediaLdcTnrParameters);
            tnrParams = &(gRefTnr3Params[frameIdx % numRef]);
            status = NvMediaLdcSetTnrParameters(pLDC, params, tnrParams);
            CHK_STATUS_AND_RETURN("NvMediaLdcSetTnrParameters", status);
            LOG_DBG("  NvMediaLdcSetTnrParameters called (predefined)");
        }

        /* MANDATORY STEP: NvMediaLdcProcess */
        NvMediaLdcResult result;
        status = NvMediaLdcProcess(pLDC, params, &result);
        CHK_STATUS_AND_RETURN("NvMediaLdcProcess", status);
        LOG_INFO("NvMediaLdcProcess(pLDC:%p) done\n", pLDC);
        /* NvMediaLdcProcess done */

        uint64_t timeEnd, timeStart = 0, timeTaken = 0;
        LOG_DBG("Waiting for NvMediaLdcProcess to write the frame: %u\n",
                frameIdx);

        GetTimeMicroSec(&timeStart);

        /* MANDATORY STEP: Wait for HW to be done with the surface */
        status = NvMediaLdcGetEOFNvSciSyncFence(pLDC, &result, &eofFence);
        CHK_STATUS_AND_RETURN("NvMediaLdcGetEOFNvSciSyncFence", status);
        status = NvSciSyncFenceWait(&eofFence, waitCtx, -1);
        CHK_STATUS_AND_RETURN("NvSciSyncFenceWait", status);
        NvSciSyncFenceClear(&eofFence);
        /* Wait for HW to be done with the surface done */

        GetTimeMicroSec(&timeEnd);
        timeTaken = (timeEnd - timeStart);
        totalOutTimeTaken += timeTaken;
        LOG_INFO("NvMediaImageGetStatus(out:%p) done in %llu us\n\n", out, timeTaken);

        /* Dump output(s) to file */
        if(out) {
            status = WriteYUV(args.outFile,
                              out,
                              &sciBufAttrListDst,
                              frameIdx ? NVMEDIA_TRUE : NVMEDIA_FALSE);
            CHK_STATUS_AND_RETURN("WriteImage(out)", status);
            LOG_INFO("WriteImage(from out:%p in to %s) done\n", out, args.outFile);
        }

        if(xSob) {
            status = WriteYUV(args.xSobelFile,
                              xSob,
                              &sciBufAttrListXSobel,
                              frameIdx ? NVMEDIA_TRUE : NVMEDIA_FALSE);
            CHK_STATUS_AND_RETURN("WriteImage(xSob)", status);
            LOG_INFO("WriteImage(from xSob:%p in to %s) done\n", xSob, args.xSobelFile);
        }

        if(dSample) {
            status = WriteYUV(args.xSobelDSFile,
                              dSample,
                              &sciBufAttrListXSobelDS,
                              frameIdx ? NVMEDIA_TRUE : NVMEDIA_FALSE);
            CHK_STATUS_AND_RETURN("WriteImage(dSample)", status);
            LOG_INFO("WriteImage(from dSample:%p in to %s) done\n", dSample, args.xSobelDSFile);
        }

        /* Print checksum if one was calculated */
        if (args.checksumMode != NVMEDIA_LDC_CHECKSUM_MODE_DISABLED)
        {
            NvMediaLdcChecksum chksum;
            status = NvMediaLdcGetChecksum(pLDC, &result, &chksum);
            CHK_STATUS_AND_RETURN("NvMediaLdcGetChecksum", status);
            char chksumStr[2 * NVMEDIA_LDC_CHECKSUM_NUM_BYTES + 1];
            memset(chksumStr, 0, sizeof(chksumStr));
            for (uint32_t i = 0; i < NVMEDIA_LDC_CHECKSUM_NUM_BYTES; ++i)
            {
                sprintf(chksumStr + 2 * i, "%02X", chksum.data[i]);
            }
            LOG_INFO("Checksum: %s (mode:%u, frame:%u)\n", chksumStr, args.checksumMode, frameIdx);
        }

        /* Swap out and prev */
        if (pParamsAttrs->enableTnr) {
            LOG_DBG("swap out:%p and prev:%p \n", out, prev);
            temp = prev;
            prev = out;
            out = temp;
        }
    }

    if (frameIdx) {
        LOG_INFO("\nAverage time taken to process a frame: %llu us\n", totalOutTimeTaken/frameIdx);
    }

    /* Cleanup before exit */
ldc_cleanup:
    LOG_INFO("Processed %u frames, cleaning up\n", frameIdx);

    /* MANDATORY STEP: Cleanup */
    NvSciSyncFenceClear(&eofFence);

    if (dSample) {
        NvMediaLdcUnregisterNvSciBufObj(pLDC, dSample);
        NvSciBufObjFree(dSample);
    }

    if (xSob) {
        NvMediaLdcUnregisterNvSciBufObj(pLDC, xSob);
        NvSciBufObjFree(xSob);
    }

    if (out) {
        NvMediaLdcUnregisterNvSciBufObj(pLDC, out);
        NvSciBufObjFree(out);
    }

    if (prev) {
        NvMediaLdcUnregisterNvSciBufObj(pLDC, prev);
        NvSciBufObjFree(prev);
    }

    if (curr) {
        NvMediaLdcUnregisterNvSciBufObj(pLDC, curr);
        NvSciBufObjFree(curr);
    }

    if (sciBufAttrListSrc) {
        NvSciBufAttrListFree(sciBufAttrListSrc);
    }

    if (sciBufAttrListDst) {
        NvSciBufAttrListFree(sciBufAttrListDst);
    }

    if (sciBufAttrListPrev) {
        NvSciBufAttrListFree(sciBufAttrListPrev);
    }

    if (sciBufAttrListXSobel) {
        NvSciBufAttrListFree(sciBufAttrListXSobel);
    }

    if (sciBufAttrListXSobelDS) {
        NvSciBufAttrListFree(sciBufAttrListXSobelDS);
    }

    if (sciBufModule) {
        NvSciBufModuleClose(sciBufModule);
    }

    if (waitCtx) {
        NvSciSyncCpuWaitContextFree(waitCtx);
    }

    if (eofSyncObj) {
        NvMediaLdcUnregisterNvSciSyncObj(pLDC, eofSyncObj);
        NvSciSyncObjFree(eofSyncObj);
    }

    if (syncReconciled) {
        NvSciSyncAttrListFree(syncReconciled);
    }

    if (syncConflict) {
        NvSciSyncAttrListFree(syncConflict);
    }

    if (sciSyncAttrListCpu) {
        NvSciSyncAttrListFree(sciSyncAttrListCpu);
    }

    if (sciSyncAttrList) {
        NvSciSyncAttrListFree(sciSyncAttrList);
    }

    if (sciSyncModule) {
        NvSciSyncModuleClose(sciSyncModule);
    }

    if(pLDC) {
        status = NvMediaLdcDestroyParameters(pLDC, params);
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("NvMediaLdcDestroyParameters failed!\n");
        }
        status = NvMediaLdcDestroy(pLDC);
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("NvMediaLDCDestroy failed!\n");
        }
        LOG_INFO("NvMediaLDCDestroy(pLDC:%p) done\n", pLDC);
    }
    /* Cleanup done */

    /* Free any memory allocted */
    if (pixelMasksData) {
        free(pixelMasksData);
    }

    if (controlPoints != NULL) {
        free(controlPoints);
    }

    return ret;
}
