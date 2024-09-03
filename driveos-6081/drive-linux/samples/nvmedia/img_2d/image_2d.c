/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "nvmedia_2d.h"
#include "nvmedia_2d_sci.h"
#include "config_parser.h"
#include "log_utils.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Maximum number of source layers to use for a 2D compose operation */
#define MAX_SRC_LAYER_COUNT 16

/* Maximum number of planes in a surface */
#define MAX_PLANE_COUNT 3

/* Structure holding allocated objects related to an NvSciSyncObj */
typedef struct
{
    NvSciSyncAttrList attrList;
    NvSciSyncAttrList attrListCpu;
    NvSciSyncAttrList attrListReconciled;
    NvSciSyncAttrList attrListConflict;
    NvSciSyncObj obj;
} SyncObj;

/* Structure holding allocated objects related to an NvSciBufObj */
typedef struct
{
    NvSciBufAttrList attrList;
    NvSciBufAttrList attrListReconciled;
    NvSciBufAttrList attrListConflict;
    NvSciBufObj obj;
} BufObj;

/* Structure holding attributes related to a surface */
typedef struct
{
    NvSciBufAttrValImageLayoutType layout;
    NvSciBufAttrValImageScanType scanType;
    uint32_t planeCount;
    NvSciBufAttrValColorFmt planeColorFmts[MAX_PLANE_COUNT];
    uint32_t planeWidths[MAX_PLANE_COUNT];
    uint32_t planeHeights[MAX_PLANE_COUNT];
    bool hasPlaneColorStds;
    NvSciBufAttrValColorStd planeColorStds[MAX_PLANE_COUNT];
} SurfaceAttrs;

/* Structure for holding file paths */
typedef struct
{
    char data[256];
} Filename;

/* Structure holding configuration for one source layer of a compose operation */
typedef struct
{
    bool isUsed;
    Filename filename;
    SurfaceAttrs surfaceAttrs;

    bool hasSrcRect;
    NvMediaRect srcRect;

    bool hasDstRect;
    NvMediaRect dstRect;

    bool hasTransform;
    NvMedia2DTransform transform;

    bool hasfilter;
    NvMedia2DFilter filter;

    bool hasBlendMode;
    NvMedia2DBlendMode blendMode;
    float constantAlpha;
} SrcLayer;

/* Enum specifying the different ways to read/write surface data from/to file */
typedef enum
{
    /* Use NvSci buffer r/w functionality */
    FILE_IO_MODE_NVSCI = 0,
    /* Copy surface data line-by-line discarding any padding */
    FILE_IO_MODE_LINE_BY_LINE,
} FileIOMode;

/* Structure holding all the configuration for a compose operation */
typedef struct
{
    SrcLayer srcLayers[MAX_SRC_LAYER_COUNT];
    Filename dstFilename;
    SurfaceAttrs dstSurfaceAttrs;
    FileIOMode fileIOMode;
} Config;

/* Read configuration from a file and validate it */
static int
ParseConfig(Config *config, char *filename);

/* Allocate NvSciSync objects */
static int
AllocateSyncObj(SyncObj *sync, NvSciSyncModule syncModule, NvMedia2D *handle);

/* Deallocate NvSciSync objects */
static void
DeallocateSyncObj(SyncObj *sync);

/* Allocate NvSciBuf objects based on surface attributes */
static int
AllocateBufObj(BufObj *buf, const SurfaceAttrs *attrs, NvSciBufModule bufModule, NvMedia2D *handle);

/* Deallocate NvSciBuf objects */
static void
DeallocateBufObj(BufObj *buf);

/* Read image data from a file to a buffer */
static int
ReadBufferFromFile(NvSciBufObj buffer, Filename *filename, FileIOMode mode);

/* Write image data from a buffer to a file */
static int
WriteBufferToFile(NvSciBufObj buffer, Filename *filename, FileIOMode mode);

int
main(int argc, char **argv)
{
    /* Return codes */
    int retval = 0;
    NvMediaStatus result = NVMEDIA_STATUS_OK;
    NvSciError sciResult = NvSciError_Success;

    /* Allocated objects */
    NvSciSyncModule syncModule = NULL;
    NvSciBufModule bufModule = NULL;
    NvMedia2D *handle = NULL;
    SyncObj eofSync;
    memset(&eofSync, 0, sizeof(eofSync));
    NvSciSyncCpuWaitContext waitCtx = NULL;
    BufObj srcBufs[MAX_SRC_LAYER_COUNT];
    memset(srcBufs, 0, sizeof(srcBufs));
    BufObj dstBuf;
    memset(&dstBuf, 0, sizeof(dstBuf));
    NvSciSyncFence eofFence = NvSciSyncFenceInitializer;

    if (argc != 2)
    {
        LOG_ERR("Usage: %s configfile", argv[0]);
        retval = 1;
        goto DeInit;
    }

    Config config;
    memset(&config, 0, sizeof(config));
    if (ParseConfig(&config, argv[1]) != 0)
    {
        LOG_ERR("ParseConfig failed");
        retval = 1;
        goto DeInit;
    }

    /******************************
     * Initialization phase
     *
     * The operations done in the initialization phase of an NvMedia 2D application are things that
     * are done once at the program startup, like allocating resources and registering handles. They
     * should not be done at the runtime phase anymore.
     */

    /* Create the NvSciSyncModule needed for creating other NvSciSync objects */
    sciResult = NvSciSyncModuleOpen(&syncModule);
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("NvSciSyncModuleOpen failed with %d", sciResult);
        retval = 1;
        goto DeInit;
    }

    /* Create the NvSciBufModule needed for creating other NvSciBuf objects */
    sciResult = NvSciBufModuleOpen(&bufModule);
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("NvSciBufModuleOpen failed with %d", sciResult);
        retval = 1;
        goto DeInit;
    }

    /* Create NvMedia 2D context */
    result = NvMedia2DCreate(&handle, NULL);
    if (result != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("NvMedia2DCreate failed with %d", result);
        retval = 1;
        goto DeInit;
    }

    /* Allocate the NvSciSyncObj used for end-of-frame synchronization */
    if (AllocateSyncObj(&eofSync, syncModule, handle) != 0)
    {
        LOG_ERR("AllocateSyncObj failed");
        retval = 1;
        goto DeInit;
    }

    /* Register the NvSciSyncObj with NvMedia 2D */
    result = NvMedia2DRegisterNvSciSyncObj(handle, NVMEDIA_EOFSYNCOBJ, eofSync.obj);
    if (result != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("NvMedia2DRegisterNvSciSyncObj failed with %d", result);
        retval = 1;
        goto DeInit;
    }

    /* Create a CPU wait context that is needed to wait the end-of-frame fence on CPU */
    sciResult = NvSciSyncCpuWaitContextAlloc(syncModule, &waitCtx);
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("NvSciSyncCpuWaitContextAlloc failed with %d", sciResult);
        retval = 1;
        goto DeInit;
    }

    /* Allocate, register and initialize source surfaces */
    for (int i = 0; i < MAX_SRC_LAYER_COUNT; ++i)
    {
        SrcLayer *layer = &config.srcLayers[i];

        if (!layer->isUsed)
        {
            continue;
        }

        if (AllocateBufObj(&srcBufs[i], &layer->surfaceAttrs, bufModule, handle) != 0)
        {
            LOG_ERR("AllocateBufObj for src surf %d failed", i);
            retval = 1;
            goto DeInit;
        }

        result = NvMedia2DRegisterNvSciBufObj(handle, srcBufs[i].obj);
        if (result != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("NvMedia2DRegisterNvSciBufObj for src surf %d failed with %d", i, result);
            retval = 1;
            goto DeInit;
        }

        if (ReadBufferFromFile(srcBufs[i].obj, &layer->filename, config.fileIOMode) != 0)
        {
            LOG_ERR("ReadBufferFromFile for src surf %d failed", i);
            retval = 1;
            goto DeInit;
        }
    }

    /* Allocate the destination surface */
    if (AllocateBufObj(&dstBuf, &config.dstSurfaceAttrs, bufModule, handle) != 0)
    {
        LOG_ERR("AllocateBufObj for dst surf failed");
        retval = 1;
        goto DeInit;
    }

    /* Register the destination surface with NvMedia 2D */
    result = NvMedia2DRegisterNvSciBufObj(handle, dstBuf.obj);
    if (result != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("NvMedia2DRegisterNvSciBufObj for dst surf failed with %d", result);
        retval = 1;
        goto DeInit;
    }

    /******************************
     * Runtime phase
     *
     * Runtime phase of an NvMedia 2D application contains the main processing logic of the program.
     * In real-world applications this usually involves some kind of a processing loop. Here we just
     * submit a single compose operation.
     */

    /* Acquire an empty NvMedia 2D parameters object */
    NvMedia2DComposeParameters params = 0;
    result = NvMedia2DGetComposeParameters(handle, &params);
    if (result != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("NvMedia2DGetComposeParameters failed with %d", result);
        retval = 1;
        goto DeInit;
    }

    /* Set the NvSciSyncObj to use for end-of-frame synchronization */
    result = NvMedia2DSetNvSciSyncObjforEOF(handle, params, eofSync.obj);
    if (result != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("NvMedia2DSetNvSciSyncObjforEOF failed with %d", result);
        retval = 1;
        goto DeInit;
    }

    /* Set the source layer parameters */
    for (int i = 0; i < MAX_SRC_LAYER_COUNT; ++i)
    {
        SrcLayer *layer = &config.srcLayers[i];

        if (!layer->isUsed)
        {
            continue;
        }

        result = NvMedia2DSetSrcNvSciBufObj(handle, params, i, srcBufs[i].obj);
        if (result != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("NvMedia2DSetSrcNvSciBufObj for src layer %d failed with %d", i, result);
            retval = 1;
            goto DeInit;
        }

        if (layer->hasSrcRect || layer->hasDstRect || layer->hasTransform)
        {
            result = NvMedia2DSetSrcGeometry(handle,
                                             params,
                                             i,
                                             layer->hasSrcRect ? &layer->srcRect : NULL,
                                             layer->hasDstRect ? &layer->dstRect : NULL,
                                             layer->hasTransform ? layer->transform
                                                                 : NVMEDIA_2D_TRANSFORM_NONE);
            if (result != NVMEDIA_STATUS_OK)
            {
                LOG_ERR("NvMedia2DSetSrcGeometry for src layer %d failed with %d", i, result);
                retval = 1;
                goto DeInit;
            }
        }

        if (layer->hasfilter)
        {
            result = NvMedia2DSetSrcFilter(handle, params, i, layer->filter);
            if (result != NVMEDIA_STATUS_OK)
            {
                LOG_ERR("NvMedia2DSetSrcFilter for src layer %d failed with %d", i, result);
                retval = 1;
                goto DeInit;
            }
        }

        if (layer->hasBlendMode)
        {
            result =
                NvMedia2DSetSrcBlendMode(handle, params, i, layer->blendMode, layer->constantAlpha);
            if (result != NVMEDIA_STATUS_OK)
            {
                LOG_ERR("NvMedia2DSetSrcBlendMode for src layer %d failed with %d", i, result);
                retval = 1;
                goto DeInit;
            }
        }
    }

    /* Set destination surface */
    result = NvMedia2DSetDstNvSciBufObj(handle, params, dstBuf.obj);
    if (result != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("NvMedia2DSetDstNvSciBufObj failed with %d", result);
        retval = 1;
        goto DeInit;
    }

    /* Submit the compose operation */
    NvMedia2DComposeResult composeResult;
    result = NvMedia2DCompose(handle, params, &composeResult);
    if (result != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("NvMedia2DCompose failed with %d", result);
        retval = 1;
        goto DeInit;
    }

    /* Get the end-of-frame fence for the compose operation */
    result = NvMedia2DGetEOFNvSciSyncFence(handle, &composeResult, &eofFence);
    if (result != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("NvMedia2DGetEOFNvSciSyncFence failed with %d", result);
        retval = 1;
        goto DeInit;
    }

    /* Block until the end-of-frame fence is signaled */
    sciResult = NvSciSyncFenceWait(&eofFence, waitCtx, -1);
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("NvSciSyncFenceWait failed with %d", sciResult);
        retval = 1;
        goto DeInit;
    }

    /* Write the destination surface to file */
    if (WriteBufferToFile(dstBuf.obj, &config.dstFilename, config.fileIOMode) != 0)
    {
        LOG_ERR("WriteBufferToFile failed");
        retval = 1;
        goto DeInit;
    }

DeInit:
    /******************************
     * De-initialization phase
     *
     * De-initialization phase of an NvMedia 2D application is done once at the program exit, and it
     * frees any resources allocated in the initialization phase.
     */

    NvSciSyncFenceClear(&eofFence);

    if (dstBuf.obj)
    {
        result = NvMedia2DUnregisterNvSciBufObj(handle, dstBuf.obj);
        if (result != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("NvMedia2DUnregisterNvSciBufObj for dst surf failed with %d", result);
        }
    }

    DeallocateBufObj(&dstBuf);

    for (int i = 0; i < MAX_SRC_LAYER_COUNT; ++i)
    {
        if (srcBufs[i].obj)
        {
            result = NvMedia2DUnregisterNvSciBufObj(handle, srcBufs[i].obj);
            if (result != NVMEDIA_STATUS_OK)
            {
                LOG_ERR("NvMedia2DUnregisterNvSciBufObj for src surf %d failed with %d", i, result);
            }
        }

        DeallocateBufObj(&srcBufs[i]);
    }

    if (waitCtx)
    {
        NvSciSyncCpuWaitContextFree(waitCtx);
        waitCtx = NULL;
    }

    if (eofSync.obj)
    {
        result = NvMedia2DUnregisterNvSciSyncObj(handle, eofSync.obj);
        if (result != NVMEDIA_STATUS_OK)
        {
            LOG_ERR("NvMedia2DUnregisterNvSciSyncObj failed with %d", result);
        }
    }

    DeallocateSyncObj(&eofSync);

    if (handle)
    {
        result = NvMedia2DDestroy(handle);
        if (result == NVMEDIA_STATUS_OK)
        {
            handle = NULL;
        }
        else
        {
            LOG_ERR("NvMedia2DDestroy failed with %d", result);
        }
    }

    if (bufModule)
    {
        NvSciBufModuleClose(bufModule);
        bufModule = NULL;
    }

    if (syncModule)
    {
        NvSciSyncModuleClose(syncModule);
        syncModule = NULL;
    }

    return retval;
}

typedef struct
{
    char data[32];
} ConfigFileStr;

typedef struct
{
    ConfigFileStr layout;
    ConfigFileStr scanType;
    ConfigFileStr colorFmts[MAX_PLANE_COUNT];
    ConfigFileStr colorStds[MAX_PLANE_COUNT];
    int widths[MAX_PLANE_COUNT];
    int heights[MAX_PLANE_COUNT];
} ConfigFileSurfaceAttrs;

typedef struct
{
    int left;
    int top;
    int right;
    int bottom;
} ConfigFileRect;

typedef struct
{
    Filename filename;
    ConfigFileSurfaceAttrs attrs;
    ConfigFileRect srcRect;
    ConfigFileRect dstRect;
    ConfigFileStr transform;
    ConfigFileStr filter;
    ConfigFileStr blendMode;
    float constAlpha;
} ConfigFileSrcLayer;

typedef struct
{
    ConfigFileSrcLayer src[MAX_SRC_LAYER_COUNT];
    Filename dstFilename;
    ConfigFileSurfaceAttrs dst;
    ConfigFileStr fileIOMode;
} ConfigFile;

static bool
IsSurfaceConfigured(const ConfigFileSurfaceAttrs *surf)
{
    for (int i = 0; i < MAX_PLANE_COUNT; ++i)
    {
        if (surf->colorFmts[i].data[0] || surf->colorStds[i].data[0] || surf->widths[i] >= 0 ||
            surf->heights[i] >= 0)
        {
            return true;
        }
    }

    return surf->layout.data[0] || surf->scanType.data[0];
}

static bool
IsRectConfigured(const ConfigFileRect *rect)
{
    return rect->left >= 0 || rect->top >= 0 || rect->right >= 0 || rect->bottom >= 0;
}

static bool
IsLayerConfigured(const ConfigFileSrcLayer *layer)
{
    return IsSurfaceConfigured(&layer->attrs) || IsRectConfigured(&layer->srcRect) ||
           IsRectConfigured(&layer->dstRect) || layer->filename.data[0] ||
           layer->transform.data[0] || layer->filter.data[0] || layer->blendMode.data[0] ||
           layer->constAlpha >= 0;
}

static int
SetPlaneConfig(SurfaceAttrs *attrs,
               const ConfigFileSurfaceAttrs *conf,
               int plane,
               bool firstPlaneHasColorStd)
{
    int retval = 0;

    if (strcmp(conf->colorFmts[plane].data, "U8V8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_U8V8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "U8_V8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_U8_V8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "V8U8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_V8U8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "V8_U8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_V8_U8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "U10V10") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_U10V10;
    }
    else if (strcmp(conf->colorFmts[plane].data, "V10U10") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_V10U10;
    }
    else if (strcmp(conf->colorFmts[plane].data, "U12V12") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_U12V12;
    }
    else if (strcmp(conf->colorFmts[plane].data, "V12U12") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_V12U12;
    }
    else if (strcmp(conf->colorFmts[plane].data, "U16V16") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_U16V16;
    }
    else if (strcmp(conf->colorFmts[plane].data, "V16U16") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_V16U16;
    }
    else if (strcmp(conf->colorFmts[plane].data, "Y8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_Y8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "Y10") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_Y10;
    }
    else if (strcmp(conf->colorFmts[plane].data, "Y12") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_Y12;
    }
    else if (strcmp(conf->colorFmts[plane].data, "Y16") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_Y16;
    }
    else if (strcmp(conf->colorFmts[plane].data, "U8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_U8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "V8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_V8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "U10") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_U10;
    }
    else if (strcmp(conf->colorFmts[plane].data, "V10") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_V10;
    }
    else if (strcmp(conf->colorFmts[plane].data, "U12") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_U12;
    }
    else if (strcmp(conf->colorFmts[plane].data, "V12") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_V12;
    }
    else if (strcmp(conf->colorFmts[plane].data, "U16") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_U16;
    }
    else if (strcmp(conf->colorFmts[plane].data, "V16") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_V16;
    }
    else if (strcmp(conf->colorFmts[plane].data, "A8Y8U8V8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_A8Y8U8V8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "Y8U8Y8V8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_Y8U8Y8V8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "Y8V8Y8U8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_Y8V8Y8U8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "U8Y8V8Y8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_U8Y8V8Y8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "V8Y8U8Y8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_V8Y8U8Y8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "A16Y16U16V16") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_A16Y16U16V16;
    }
    else if (strcmp(conf->colorFmts[plane].data, "A8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_A8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "B8G8R8A8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_B8G8R8A8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "A8R8G8B8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_A8R8G8B8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "A8B8G8R8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_A8B8G8R8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "A2R10G10B10") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_A2R10G10B10;
    }
    else if (strcmp(conf->colorFmts[plane].data, "A16B16G16R16") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_A16B16G16R16;
    }
    else if (strcmp(conf->colorFmts[plane].data, "R8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_R8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "G8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_G8;
    }
    else if (strcmp(conf->colorFmts[plane].data, "B8") == 0)
    {
        attrs->planeColorFmts[plane] = NvSciColor_B8;
    }
    else
    {
        LOG_ERR("Plane %d color format not %s",
                plane + 1,
                conf->colorFmts[plane].data[0] ? "valid" : "found");
        retval = 1;
    }

    if (firstPlaneHasColorStd)
    {
        if (strcmp(conf->colorStds[plane].data, "SRGB") == 0)
        {
            attrs->planeColorStds[plane] = NvSciColorStd_SRGB;
        }
        else if (strcmp(conf->colorStds[plane].data, "REC601_SR") == 0)
        {
            attrs->planeColorStds[plane] = NvSciColorStd_REC601_SR;
        }
        else if (strcmp(conf->colorStds[plane].data, "REC601_ER") == 0)
        {
            attrs->planeColorStds[plane] = NvSciColorStd_REC601_ER;
        }
        else if (strcmp(conf->colorStds[plane].data, "REC709_SR") == 0)
        {
            attrs->planeColorStds[plane] = NvSciColorStd_REC709_SR;
        }
        else if (strcmp(conf->colorStds[plane].data, "REC709_ER") == 0)
        {
            attrs->planeColorStds[plane] = NvSciColorStd_REC709_ER;
        }
        else if (strcmp(conf->colorStds[plane].data, "REC2020_RGB") == 0)
        {
            attrs->planeColorStds[plane] = NvSciColorStd_REC2020_RGB;
        }
        else if (strcmp(conf->colorStds[plane].data, "REC2020_SR") == 0)
        {
            attrs->planeColorStds[plane] = NvSciColorStd_REC2020_SR;
        }
        else if (strcmp(conf->colorStds[plane].data, "REC2020_ER") == 0)
        {
            attrs->planeColorStds[plane] = NvSciColorStd_REC2020_ER;
        }
        else if (strcmp(conf->colorStds[plane].data, "YcCbcCrc_SR") == 0)
        {
            attrs->planeColorStds[plane] = NvSciColorStd_YcCbcCrc_SR;
        }
        else if (strcmp(conf->colorStds[plane].data, "YcCbcCrc_ER") == 0)
        {
            attrs->planeColorStds[plane] = NvSciColorStd_YcCbcCrc_ER;
        }
        else if (strcmp(conf->colorStds[plane].data, "SENSOR_RGBA") == 0)
        {
            attrs->planeColorStds[plane] = NvSciColorStd_SENSOR_RGBA;
        }
        else if (strcmp(conf->colorStds[plane].data, "REQ2020PQ_ER") == 0)
        {
            attrs->planeColorStds[plane] = NvSciColorStd_REQ2020PQ_ER;
        }
        else
        {
            LOG_ERR("Plane %d color standard not %s",
                    plane + 1,
                    conf->colorStds[plane].data[0] ? "valid" : "found");
            retval = 1;
        }
    }
    else if (conf->colorStds[plane].data[0])
    {
        LOG_ERR("Plane %d has color standard configured but plane 1 does not", plane + 1);
        retval = 1;
    }

    if (conf->widths[plane] >= 0)
    {
        attrs->planeWidths[plane] = conf->widths[plane];
    }
    else
    {
        LOG_ERR("Plane %d width not found", plane + 1);
        retval = 1;
    }

    if (conf->heights[plane] >= 0)
    {
        attrs->planeHeights[plane] = conf->heights[plane];
    }
    else
    {
        LOG_ERR("Plane %d height not found", plane + 1);
        retval = 1;
    }

    return retval;
}

static int
SetSurfaceConfig(SurfaceAttrs *attrs, const ConfigFileSurfaceAttrs *conf)
{
    int retval = 0;

    if (strcmp(conf->layout.data, "BlockLinear") == 0)
    {
        attrs->layout = NvSciBufImage_BlockLinearType;
    }
    else if (strcmp(conf->layout.data, "PitchLinear") == 0)
    {
        attrs->layout = NvSciBufImage_PitchLinearType;
    }
    else
    {
        LOG_ERR("Layout not %s", conf->layout.data[0] ? "valid" : "found");
        retval = 1;
    }

    if (strcmp(conf->scanType.data, "Progressive") == 0)
    {
        attrs->scanType = NvSciBufScan_ProgressiveType;
    }
    else if (strcmp(conf->scanType.data, "Interlace") == 0)
    {
        attrs->scanType = NvSciBufScan_InterlaceType;
    }
    else
    {
        LOG_ERR("Scan type not %s", conf->scanType.data[0] ? "valid" : "found");
        retval = 1;
    }

    int planeCount = 1;
    bool firstPlaneHasColorStd = conf->colorStds[0].data[0];

    if (SetPlaneConfig(attrs, conf, 0, firstPlaneHasColorStd) != 0)
    {
        LOG_ERR("Plane config not valid for plane 1");
        retval = 1;
    }

    for (int i = 1; i < MAX_PLANE_COUNT; ++i)
    {
        if (conf->colorFmts[i].data[0] || conf->colorStds[i].data[0] || conf->widths[i] >= 0 ||
            conf->heights[i] >= 0)
        {
            if (i != planeCount)
            {
                LOG_ERR("Plane %d configured but plane %d not", i + 1, i);
                retval = 1;
                continue;
            }

            if (SetPlaneConfig(attrs, conf, i, firstPlaneHasColorStd) != 0)
            {
                LOG_ERR("Plane config not valid for plane %d", i + 1);
                retval = 1;
            }

            ++planeCount;
        }
    }

    attrs->planeCount = planeCount;
    attrs->hasPlaneColorStds = firstPlaneHasColorStd;

    return retval;
}

static int
SetRectConfig(NvMediaRect *rect, const ConfigFileRect *conf)
{
    int retval = 0;

    if (conf->left >= 0)
    {
        rect->x0 = conf->left;
    }
    else
    {
        LOG_ERR("Rectangle left coordinate not found");
        retval = 1;
    }

    if (conf->top >= 0)
    {
        rect->y0 = conf->top;
    }
    else
    {
        LOG_ERR("Rectangle top coordinate not found");
        retval = 1;
    }

    if (conf->right >= 0)
    {
        rect->x1 = conf->right;
    }
    else
    {
        LOG_ERR("Rectangle right coordinate not found");
        retval = 1;
    }

    if (conf->bottom >= 0)
    {
        rect->y1 = conf->bottom;
    }
    else
    {
        LOG_ERR("Rectangle bottom coordinate not found");
        retval = 1;
    }

    return retval;
}

static int
SetTransformConfig(NvMedia2DTransform *xform, const ConfigFileStr *conf)
{
    int retval = 0;

    if (strcmp(conf->data, "None") == 0)
    {
        *xform = NVMEDIA_2D_TRANSFORM_NONE;
    }
    else if (strcmp(conf->data, "Rotate90") == 0)
    {
        *xform = NVMEDIA_2D_TRANSFORM_ROTATE_90;
    }
    else if (strcmp(conf->data, "Rotate180") == 0)
    {
        *xform = NVMEDIA_2D_TRANSFORM_ROTATE_180;
    }
    else if (strcmp(conf->data, "Rotate270") == 0)
    {
        *xform = NVMEDIA_2D_TRANSFORM_ROTATE_270;
    }
    else if (strcmp(conf->data, "FlipHorizontal") == 0)
    {
        *xform = NVMEDIA_2D_TRANSFORM_FLIP_HORIZONTAL;
    }
    else if (strcmp(conf->data, "InvTranspose") == 0)
    {
        *xform = NVMEDIA_2D_TRANSFORM_INV_TRANSPOSE;
    }
    else if (strcmp(conf->data, "FlipVertical") == 0)
    {
        *xform = NVMEDIA_2D_TRANSFORM_FLIP_VERTICAL;
    }
    else if (strcmp(conf->data, "Transpose") == 0)
    {
        *xform = NVMEDIA_2D_TRANSFORM_TRANSPOSE;
    }
    else
    {
        retval = 1;
    }

    return retval;
}

static int
SetFilterConfig(NvMedia2DFilter *filter, const ConfigFileStr *conf)
{
    int retval = 0;

    if (strcmp(conf->data, "Off") == 0)
    {
        *filter = NVMEDIA_2D_FILTER_OFF;
    }
    else if (strcmp(conf->data, "Low") == 0)
    {
        *filter = NVMEDIA_2D_FILTER_LOW;
    }
    else if (strcmp(conf->data, "Medium") == 0)
    {
        *filter = NVMEDIA_2D_FILTER_MEDIUM;
    }
    else if (strcmp(conf->data, "High") == 0)
    {
        *filter = NVMEDIA_2D_FILTER_HIGH;
    }
    else
    {
        retval = 1;
    }

    return retval;
}

static int
SetBlendConfig(SrcLayer *layer, const ConfigFileSrcLayer *conf)
{
    int retval = 0;

    if (strcmp(conf->blendMode.data, "Disabled") == 0)
    {
        layer->blendMode = NVMEDIA_2D_BLEND_MODE_DISABLED;
    }
    else if (strcmp(conf->blendMode.data, "ConstantAlpha") == 0)
    {
        layer->blendMode = NVMEDIA_2D_BLEND_MODE_CONSTANT_ALPHA;
    }
    else if (strcmp(conf->blendMode.data, "StraightAlpha") == 0)
    {
        layer->blendMode = NVMEDIA_2D_BLEND_MODE_STRAIGHT_ALPHA;
    }
    else if (strcmp(conf->blendMode.data, "PremultipliedAlpha") == 0)
    {
        layer->blendMode = NVMEDIA_2D_BLEND_MODE_PREMULTIPLIED_ALPHA;
    }
    else
    {
        LOG_ERR("Blending not %s", conf->blendMode.data[0] ? "valid" : "found");
        retval = 1;
    }

    if (conf->constAlpha >= 0)
    {
        layer->constantAlpha = conf->constAlpha;
    }
    else
    {
        LOG_ERR("Constant alpha not found");
        retval = 1;
    }

    return retval;
}

static int
SetFileIOModeConfig(FileIOMode *mode, const ConfigFileStr *conf)
{
    int retval = 0;

    if (strcmp(conf->data, "NvSci") == 0)
    {
        *mode = FILE_IO_MODE_NVSCI;
    }
    else if (strcmp(conf->data, "LineByLine") == 0)
    {
        *mode = FILE_IO_MODE_LINE_BY_LINE;
    }
    else
    {
        retval = 1;
    }

    return retval;
}

static int
SetConfig(Config *config, const ConfigFile *confFile)
{
    int retval = 0;

    for (int i = 0; i < MAX_SRC_LAYER_COUNT; ++i)
    {
        SrcLayer *layer = &config->srcLayers[i];
        const ConfigFileSrcLayer *layerConf = &confFile->src[i];

        if (!IsLayerConfigured(layerConf))
        {
            layer->isUsed = false;
            continue;
        }

        layer->isUsed = true;

        if (layerConf->filename.data[0])
        {
            strcpy(layer->filename.data, layerConf->filename.data);
        }
        else
        {
            LOG_ERR("Input file missing for src layer %d", i + 1);
            retval = 1;
        }

        if (SetSurfaceConfig(&layer->surfaceAttrs, &layerConf->attrs) != 0)
        {
            LOG_ERR("Surface config not valid for src layer %d", i + 1);
            retval = 1;
        }

        if (IsRectConfigured(&layerConf->srcRect))
        {
            layer->hasSrcRect = true;
            if (SetRectConfig(&layer->srcRect, &layerConf->srcRect) != 0)
            {
                LOG_ERR("Src rect config not valid for src layer %d", i + 1);
                retval = 1;
            }
        }

        if (IsRectConfigured(&layerConf->dstRect))
        {
            layer->hasDstRect = true;
            if (SetRectConfig(&layer->dstRect, &layerConf->dstRect) != 0)
            {
                LOG_ERR("Dst rect config not valid for src layer %d", i + 1);
                retval = 1;
            }
        }

        if (layerConf->transform.data[0])
        {
            layer->hasTransform = true;
            if (SetTransformConfig(&layer->transform, &layerConf->transform) != 0)
            {
                LOG_ERR("Transform config not valid for src layer %d", i + 1);
                retval = 1;
            }
        }

        if (layerConf->filter.data[0])
        {
            layer->hasfilter = true;
            if (SetFilterConfig(&layer->filter, &layerConf->filter) != 0)
            {
                LOG_ERR("Filter config not valid for src layer %d", i + 1);
                retval = 1;
            }
        }

        if (layerConf->blendMode.data[0] || layerConf->constAlpha >= 0)
        {
            layer->hasBlendMode = true;
            if (SetBlendConfig(layer, layerConf) != 0)
            {
                LOG_ERR("Blend config not valid for src layer %d", i + 1);
                retval = 1;
            }
        }
    }

    if (confFile->dstFilename.data[0])
    {
        strcpy(config->dstFilename.data, confFile->dstFilename.data);
    }
    else
    {
        LOG_ERR("Output file missing");
        retval = 1;
    }

    if (SetSurfaceConfig(&config->dstSurfaceAttrs, &confFile->dst) != 0)
    {
        LOG_ERR("Surface config not valid for dst layer");
        retval = 1;
    }

    if (confFile->fileIOMode.data[0])
    {
        if (SetFileIOModeConfig(&config->fileIOMode, &confFile->fileIOMode) != 0)
        {
            LOG_ERR("File IO mode config not valid");
            retval = 1;
        }
    }

    return retval;
}

static ConfigParamsMap
ConfigParamStr(const char *name, ConfigFileStr *val)
{
    ConfigParamsMap param = { name, val->data, TYPE_CHAR_ARR,     0,    LIMITS_NONE,
                              0,    0,         sizeof(val->data), NULL, SECTION_NONE };
    return param;
}

static ConfigParamsMap
ConfigParamFilename(const char *name, Filename *val)
{
    ConfigParamsMap param = { name, val->data, TYPE_CHAR_ARR,     0,    LIMITS_NONE,
                              0,    0,         sizeof(val->data), NULL, SECTION_NONE };
    return param;
}

static ConfigParamsMap
ConfigParamInt(const char *name, int *val)
{
    ConfigParamsMap param = { name, val, TYPE_INT, -1, LIMITS_NONE, 0, 0, 0, NULL, SECTION_NONE };
    return param;
}

static ConfigParamsMap
ConfigParamFloat(const char *name, float *val)
{
    ConfigParamsMap param = { name, val, TYPE_FLOAT, -1, LIMITS_NONE, 0, 0, 0, NULL, SECTION_NONE };
    return param;
}

static ConfigParamsMap
ConfigParamSentinel(void)
{
    ConfigParamsMap param = { NULL, NULL, TYPE_UINT, 0, LIMITS_NONE, 0, 0, 0, NULL, SECTION_NONE };
    return param;
}

static int
ParseConfig(Config *config, char *filename)
{
    ConfigFile confFile;
    memset(&confFile, 0, sizeof(confFile));

    ConfigParamsMap paramsMap[] = {
        ConfigParamFilename("SrcLayer1InputFile", &confFile.src[0].filename),
        ConfigParamStr("SrcLayer1Layout", &confFile.src[0].attrs.layout),
        ConfigParamStr("SrcLayer1ScanType", &confFile.src[0].attrs.scanType),
        ConfigParamStr("SrcLayer1Plane1ColorFormat", &confFile.src[0].attrs.colorFmts[0]),
        ConfigParamStr("SrcLayer1Plane2ColorFormat", &confFile.src[0].attrs.colorFmts[1]),
        ConfigParamStr("SrcLayer1Plane3ColorFormat", &confFile.src[0].attrs.colorFmts[2]),
        ConfigParamStr("SrcLayer1Plane1ColorStandard", &confFile.src[0].attrs.colorStds[0]),
        ConfigParamStr("SrcLayer1Plane2ColorStandard", &confFile.src[0].attrs.colorStds[1]),
        ConfigParamStr("SrcLayer1Plane3ColorStandard", &confFile.src[0].attrs.colorStds[2]),
        ConfigParamInt("SrcLayer1Plane1Width", &confFile.src[0].attrs.widths[0]),
        ConfigParamInt("SrcLayer1Plane2Width", &confFile.src[0].attrs.widths[1]),
        ConfigParamInt("SrcLayer1Plane3Width", &confFile.src[0].attrs.widths[2]),
        ConfigParamInt("SrcLayer1Plane1Height", &confFile.src[0].attrs.heights[0]),
        ConfigParamInt("SrcLayer1Plane2Height", &confFile.src[0].attrs.heights[1]),
        ConfigParamInt("SrcLayer1Plane3Height", &confFile.src[0].attrs.heights[2]),
        ConfigParamInt("SrcLayer1SrcRectLeft", &confFile.src[0].srcRect.left),
        ConfigParamInt("SrcLayer1SrcRectTop", &confFile.src[0].srcRect.top),
        ConfigParamInt("SrcLayer1SrcRectRight", &confFile.src[0].srcRect.right),
        ConfigParamInt("SrcLayer1SrcRectBottom", &confFile.src[0].srcRect.bottom),
        ConfigParamInt("SrcLayer1DstRectLeft", &confFile.src[0].dstRect.left),
        ConfigParamInt("SrcLayer1DstRectTop", &confFile.src[0].dstRect.top),
        ConfigParamInt("SrcLayer1DstRectRight", &confFile.src[0].dstRect.right),
        ConfigParamInt("SrcLayer1DstRectBottom", &confFile.src[0].dstRect.bottom),
        ConfigParamStr("SrcLayer1Transform", &confFile.src[0].transform),
        ConfigParamStr("SrcLayer1Filtering", &confFile.src[0].filter),
        ConfigParamStr("SrcLayer1Blending", &confFile.src[0].blendMode),
        ConfigParamFloat("SrcLayer1BlendingConstantAlpha", &confFile.src[0].constAlpha),

        ConfigParamFilename("SrcLayer2InputFile", &confFile.src[1].filename),
        ConfigParamStr("SrcLayer2Layout", &confFile.src[1].attrs.layout),
        ConfigParamStr("SrcLayer2ScanType", &confFile.src[1].attrs.scanType),
        ConfigParamStr("SrcLayer2Plane1ColorFormat", &confFile.src[1].attrs.colorFmts[0]),
        ConfigParamStr("SrcLayer2Plane2ColorFormat", &confFile.src[1].attrs.colorFmts[1]),
        ConfigParamStr("SrcLayer2Plane3ColorFormat", &confFile.src[1].attrs.colorFmts[2]),
        ConfigParamStr("SrcLayer2Plane1ColorStandard", &confFile.src[1].attrs.colorStds[0]),
        ConfigParamStr("SrcLayer2Plane2ColorStandard", &confFile.src[1].attrs.colorStds[1]),
        ConfigParamStr("SrcLayer2Plane3ColorStandard", &confFile.src[1].attrs.colorStds[2]),
        ConfigParamInt("SrcLayer2Plane1Width", &confFile.src[1].attrs.widths[0]),
        ConfigParamInt("SrcLayer2Plane2Width", &confFile.src[1].attrs.widths[1]),
        ConfigParamInt("SrcLayer2Plane3Width", &confFile.src[1].attrs.widths[2]),
        ConfigParamInt("SrcLayer2Plane1Height", &confFile.src[1].attrs.heights[0]),
        ConfigParamInt("SrcLayer2Plane2Height", &confFile.src[1].attrs.heights[1]),
        ConfigParamInt("SrcLayer2Plane3Height", &confFile.src[1].attrs.heights[2]),
        ConfigParamInt("SrcLayer2SrcRectLeft", &confFile.src[1].srcRect.left),
        ConfigParamInt("SrcLayer2SrcRectTop", &confFile.src[1].srcRect.top),
        ConfigParamInt("SrcLayer2SrcRectRight", &confFile.src[1].srcRect.right),
        ConfigParamInt("SrcLayer2SrcRectBottom", &confFile.src[1].srcRect.bottom),
        ConfigParamInt("SrcLayer2DstRectLeft", &confFile.src[1].dstRect.left),
        ConfigParamInt("SrcLayer2DstRectTop", &confFile.src[1].dstRect.top),
        ConfigParamInt("SrcLayer2DstRectRight", &confFile.src[1].dstRect.right),
        ConfigParamInt("SrcLayer2DstRectBottom", &confFile.src[1].dstRect.bottom),
        ConfigParamStr("SrcLayer2Transform", &confFile.src[1].transform),
        ConfigParamStr("SrcLayer2Filtering", &confFile.src[1].filter),
        ConfigParamStr("SrcLayer2Blending", &confFile.src[1].blendMode),
        ConfigParamFloat("SrcLayer2BlendingConstantAlpha", &confFile.src[1].constAlpha),

        ConfigParamFilename("SrcLayer3InputFile", &confFile.src[2].filename),
        ConfigParamStr("SrcLayer3Layout", &confFile.src[2].attrs.layout),
        ConfigParamStr("SrcLayer3ScanType", &confFile.src[2].attrs.scanType),
        ConfigParamStr("SrcLayer3Plane1ColorFormat", &confFile.src[2].attrs.colorFmts[0]),
        ConfigParamStr("SrcLayer3Plane2ColorFormat", &confFile.src[2].attrs.colorFmts[1]),
        ConfigParamStr("SrcLayer3Plane3ColorFormat", &confFile.src[2].attrs.colorFmts[2]),
        ConfigParamStr("SrcLayer3Plane1ColorStandard", &confFile.src[2].attrs.colorStds[0]),
        ConfigParamStr("SrcLayer3Plane2ColorStandard", &confFile.src[2].attrs.colorStds[1]),
        ConfigParamStr("SrcLayer3Plane3ColorStandard", &confFile.src[2].attrs.colorStds[2]),
        ConfigParamInt("SrcLayer3Plane1Width", &confFile.src[2].attrs.widths[0]),
        ConfigParamInt("SrcLayer3Plane2Width", &confFile.src[2].attrs.widths[1]),
        ConfigParamInt("SrcLayer3Plane3Width", &confFile.src[2].attrs.widths[2]),
        ConfigParamInt("SrcLayer3Plane1Height", &confFile.src[2].attrs.heights[0]),
        ConfigParamInt("SrcLayer3Plane2Height", &confFile.src[2].attrs.heights[1]),
        ConfigParamInt("SrcLayer3Plane3Height", &confFile.src[2].attrs.heights[2]),
        ConfigParamInt("SrcLayer3SrcRectLeft", &confFile.src[2].srcRect.left),
        ConfigParamInt("SrcLayer3SrcRectTop", &confFile.src[2].srcRect.top),
        ConfigParamInt("SrcLayer3SrcRectRight", &confFile.src[2].srcRect.right),
        ConfigParamInt("SrcLayer3SrcRectBottom", &confFile.src[2].srcRect.bottom),
        ConfigParamInt("SrcLayer3DstRectLeft", &confFile.src[2].dstRect.left),
        ConfigParamInt("SrcLayer3DstRectTop", &confFile.src[2].dstRect.top),
        ConfigParamInt("SrcLayer3DstRectRight", &confFile.src[2].dstRect.right),
        ConfigParamInt("SrcLayer3DstRectBottom", &confFile.src[2].dstRect.bottom),
        ConfigParamStr("SrcLayer3Transform", &confFile.src[2].transform),
        ConfigParamStr("SrcLayer3Filtering", &confFile.src[2].filter),
        ConfigParamStr("SrcLayer3Blending", &confFile.src[2].blendMode),
        ConfigParamFloat("SrcLayer3BlendingConstantAlpha", &confFile.src[2].constAlpha),

        ConfigParamFilename("SrcLayer4InputFile", &confFile.src[3].filename),
        ConfigParamStr("SrcLayer4Layout", &confFile.src[3].attrs.layout),
        ConfigParamStr("SrcLayer4ScanType", &confFile.src[3].attrs.scanType),
        ConfigParamStr("SrcLayer4Plane1ColorFormat", &confFile.src[3].attrs.colorFmts[0]),
        ConfigParamStr("SrcLayer4Plane2ColorFormat", &confFile.src[3].attrs.colorFmts[1]),
        ConfigParamStr("SrcLayer4Plane3ColorFormat", &confFile.src[3].attrs.colorFmts[2]),
        ConfigParamStr("SrcLayer4Plane1ColorStandard", &confFile.src[3].attrs.colorStds[0]),
        ConfigParamStr("SrcLayer4Plane2ColorStandard", &confFile.src[3].attrs.colorStds[1]),
        ConfigParamStr("SrcLayer4Plane3ColorStandard", &confFile.src[3].attrs.colorStds[2]),
        ConfigParamInt("SrcLayer4Plane1Width", &confFile.src[3].attrs.widths[0]),
        ConfigParamInt("SrcLayer4Plane2Width", &confFile.src[3].attrs.widths[1]),
        ConfigParamInt("SrcLayer4Plane3Width", &confFile.src[3].attrs.widths[2]),
        ConfigParamInt("SrcLayer4Plane1Height", &confFile.src[3].attrs.heights[0]),
        ConfigParamInt("SrcLayer4Plane2Height", &confFile.src[3].attrs.heights[1]),
        ConfigParamInt("SrcLayer4Plane3Height", &confFile.src[3].attrs.heights[2]),
        ConfigParamInt("SrcLayer4SrcRectLeft", &confFile.src[3].srcRect.left),
        ConfigParamInt("SrcLayer4SrcRectTop", &confFile.src[3].srcRect.top),
        ConfigParamInt("SrcLayer4SrcRectRight", &confFile.src[3].srcRect.right),
        ConfigParamInt("SrcLayer4SrcRectBottom", &confFile.src[3].srcRect.bottom),
        ConfigParamInt("SrcLayer4DstRectLeft", &confFile.src[3].dstRect.left),
        ConfigParamInt("SrcLayer4DstRectTop", &confFile.src[3].dstRect.top),
        ConfigParamInt("SrcLayer4DstRectRight", &confFile.src[3].dstRect.right),
        ConfigParamInt("SrcLayer4DstRectBottom", &confFile.src[3].dstRect.bottom),
        ConfigParamStr("SrcLayer4Transform", &confFile.src[3].transform),
        ConfigParamStr("SrcLayer4Filtering", &confFile.src[3].filter),
        ConfigParamStr("SrcLayer4Blending", &confFile.src[3].blendMode),
        ConfigParamFloat("SrcLayer4BlendingConstantAlpha", &confFile.src[3].constAlpha),

        ConfigParamFilename("SrcLayer5InputFile", &confFile.src[4].filename),
        ConfigParamStr("SrcLayer5Layout", &confFile.src[4].attrs.layout),
        ConfigParamStr("SrcLayer5ScanType", &confFile.src[4].attrs.scanType),
        ConfigParamStr("SrcLayer5Plane1ColorFormat", &confFile.src[4].attrs.colorFmts[0]),
        ConfigParamStr("SrcLayer5Plane2ColorFormat", &confFile.src[4].attrs.colorFmts[1]),
        ConfigParamStr("SrcLayer5Plane3ColorFormat", &confFile.src[4].attrs.colorFmts[2]),
        ConfigParamStr("SrcLayer5Plane1ColorStandard", &confFile.src[4].attrs.colorStds[0]),
        ConfigParamStr("SrcLayer5Plane2ColorStandard", &confFile.src[4].attrs.colorStds[1]),
        ConfigParamStr("SrcLayer5Plane3ColorStandard", &confFile.src[4].attrs.colorStds[2]),
        ConfigParamInt("SrcLayer5Plane1Width", &confFile.src[4].attrs.widths[0]),
        ConfigParamInt("SrcLayer5Plane2Width", &confFile.src[4].attrs.widths[1]),
        ConfigParamInt("SrcLayer5Plane3Width", &confFile.src[4].attrs.widths[2]),
        ConfigParamInt("SrcLayer5Plane1Height", &confFile.src[4].attrs.heights[0]),
        ConfigParamInt("SrcLayer5Plane2Height", &confFile.src[4].attrs.heights[1]),
        ConfigParamInt("SrcLayer5Plane3Height", &confFile.src[4].attrs.heights[2]),
        ConfigParamInt("SrcLayer5SrcRectLeft", &confFile.src[4].srcRect.left),
        ConfigParamInt("SrcLayer5SrcRectTop", &confFile.src[4].srcRect.top),
        ConfigParamInt("SrcLayer5SrcRectRight", &confFile.src[4].srcRect.right),
        ConfigParamInt("SrcLayer5SrcRectBottom", &confFile.src[4].srcRect.bottom),
        ConfigParamInt("SrcLayer5DstRectLeft", &confFile.src[4].dstRect.left),
        ConfigParamInt("SrcLayer5DstRectTop", &confFile.src[4].dstRect.top),
        ConfigParamInt("SrcLayer5DstRectRight", &confFile.src[4].dstRect.right),
        ConfigParamInt("SrcLayer5DstRectBottom", &confFile.src[4].dstRect.bottom),
        ConfigParamStr("SrcLayer5Transform", &confFile.src[4].transform),
        ConfigParamStr("SrcLayer5Filtering", &confFile.src[4].filter),
        ConfigParamStr("SrcLayer5Blending", &confFile.src[4].blendMode),
        ConfigParamFloat("SrcLayer5BlendingConstantAlpha", &confFile.src[4].constAlpha),

        ConfigParamFilename("SrcLayer6InputFile", &confFile.src[5].filename),
        ConfigParamStr("SrcLayer6Layout", &confFile.src[5].attrs.layout),
        ConfigParamStr("SrcLayer6ScanType", &confFile.src[5].attrs.scanType),
        ConfigParamStr("SrcLayer6Plane1ColorFormat", &confFile.src[5].attrs.colorFmts[0]),
        ConfigParamStr("SrcLayer6Plane2ColorFormat", &confFile.src[5].attrs.colorFmts[1]),
        ConfigParamStr("SrcLayer6Plane3ColorFormat", &confFile.src[5].attrs.colorFmts[2]),
        ConfigParamStr("SrcLayer6Plane1ColorStandard", &confFile.src[5].attrs.colorStds[0]),
        ConfigParamStr("SrcLayer6Plane2ColorStandard", &confFile.src[5].attrs.colorStds[1]),
        ConfigParamStr("SrcLayer6Plane3ColorStandard", &confFile.src[5].attrs.colorStds[2]),
        ConfigParamInt("SrcLayer6Plane1Width", &confFile.src[5].attrs.widths[0]),
        ConfigParamInt("SrcLayer6Plane2Width", &confFile.src[5].attrs.widths[1]),
        ConfigParamInt("SrcLayer6Plane3Width", &confFile.src[5].attrs.widths[2]),
        ConfigParamInt("SrcLayer6Plane1Height", &confFile.src[5].attrs.heights[0]),
        ConfigParamInt("SrcLayer6Plane2Height", &confFile.src[5].attrs.heights[1]),
        ConfigParamInt("SrcLayer6Plane3Height", &confFile.src[5].attrs.heights[2]),
        ConfigParamInt("SrcLayer6SrcRectLeft", &confFile.src[5].srcRect.left),
        ConfigParamInt("SrcLayer6SrcRectTop", &confFile.src[5].srcRect.top),
        ConfigParamInt("SrcLayer6SrcRectRight", &confFile.src[5].srcRect.right),
        ConfigParamInt("SrcLayer6SrcRectBottom", &confFile.src[5].srcRect.bottom),
        ConfigParamInt("SrcLayer6DstRectLeft", &confFile.src[5].dstRect.left),
        ConfigParamInt("SrcLayer6DstRectTop", &confFile.src[5].dstRect.top),
        ConfigParamInt("SrcLayer6DstRectRight", &confFile.src[5].dstRect.right),
        ConfigParamInt("SrcLayer6DstRectBottom", &confFile.src[5].dstRect.bottom),
        ConfigParamStr("SrcLayer6Transform", &confFile.src[5].transform),
        ConfigParamStr("SrcLayer6Filtering", &confFile.src[5].filter),
        ConfigParamStr("SrcLayer6Blending", &confFile.src[5].blendMode),
        ConfigParamFloat("SrcLayer6BlendingConstantAlpha", &confFile.src[5].constAlpha),

        ConfigParamFilename("SrcLayer7InputFile", &confFile.src[6].filename),
        ConfigParamStr("SrcLayer7Layout", &confFile.src[6].attrs.layout),
        ConfigParamStr("SrcLayer7ScanType", &confFile.src[6].attrs.scanType),
        ConfigParamStr("SrcLayer7Plane1ColorFormat", &confFile.src[6].attrs.colorFmts[0]),
        ConfigParamStr("SrcLayer7Plane2ColorFormat", &confFile.src[6].attrs.colorFmts[1]),
        ConfigParamStr("SrcLayer7Plane3ColorFormat", &confFile.src[6].attrs.colorFmts[2]),
        ConfigParamStr("SrcLayer7Plane1ColorStandard", &confFile.src[6].attrs.colorStds[0]),
        ConfigParamStr("SrcLayer7Plane2ColorStandard", &confFile.src[6].attrs.colorStds[1]),
        ConfigParamStr("SrcLayer7Plane3ColorStandard", &confFile.src[6].attrs.colorStds[2]),
        ConfigParamInt("SrcLayer7Plane1Width", &confFile.src[6].attrs.widths[0]),
        ConfigParamInt("SrcLayer7Plane2Width", &confFile.src[6].attrs.widths[1]),
        ConfigParamInt("SrcLayer7Plane3Width", &confFile.src[6].attrs.widths[2]),
        ConfigParamInt("SrcLayer7Plane1Height", &confFile.src[6].attrs.heights[0]),
        ConfigParamInt("SrcLayer7Plane2Height", &confFile.src[6].attrs.heights[1]),
        ConfigParamInt("SrcLayer7Plane3Height", &confFile.src[6].attrs.heights[2]),
        ConfigParamInt("SrcLayer7SrcRectLeft", &confFile.src[6].srcRect.left),
        ConfigParamInt("SrcLayer7SrcRectTop", &confFile.src[6].srcRect.top),
        ConfigParamInt("SrcLayer7SrcRectRight", &confFile.src[6].srcRect.right),
        ConfigParamInt("SrcLayer7SrcRectBottom", &confFile.src[6].srcRect.bottom),
        ConfigParamInt("SrcLayer7DstRectLeft", &confFile.src[6].dstRect.left),
        ConfigParamInt("SrcLayer7DstRectTop", &confFile.src[6].dstRect.top),
        ConfigParamInt("SrcLayer7DstRectRight", &confFile.src[6].dstRect.right),
        ConfigParamInt("SrcLayer7DstRectBottom", &confFile.src[6].dstRect.bottom),
        ConfigParamStr("SrcLayer7Transform", &confFile.src[6].transform),
        ConfigParamStr("SrcLayer7Filtering", &confFile.src[6].filter),
        ConfigParamStr("SrcLayer7Blending", &confFile.src[6].blendMode),
        ConfigParamFloat("SrcLayer7BlendingConstantAlpha", &confFile.src[6].constAlpha),

        ConfigParamFilename("SrcLayer8InputFile", &confFile.src[7].filename),
        ConfigParamStr("SrcLayer8Layout", &confFile.src[7].attrs.layout),
        ConfigParamStr("SrcLayer8ScanType", &confFile.src[7].attrs.scanType),
        ConfigParamStr("SrcLayer8Plane1ColorFormat", &confFile.src[7].attrs.colorFmts[0]),
        ConfigParamStr("SrcLayer8Plane2ColorFormat", &confFile.src[7].attrs.colorFmts[1]),
        ConfigParamStr("SrcLayer8Plane3ColorFormat", &confFile.src[7].attrs.colorFmts[2]),
        ConfigParamStr("SrcLayer8Plane1ColorStandard", &confFile.src[7].attrs.colorStds[0]),
        ConfigParamStr("SrcLayer8Plane2ColorStandard", &confFile.src[7].attrs.colorStds[1]),
        ConfigParamStr("SrcLayer8Plane3ColorStandard", &confFile.src[7].attrs.colorStds[2]),
        ConfigParamInt("SrcLayer8Plane1Width", &confFile.src[7].attrs.widths[0]),
        ConfigParamInt("SrcLayer8Plane2Width", &confFile.src[7].attrs.widths[1]),
        ConfigParamInt("SrcLayer8Plane3Width", &confFile.src[7].attrs.widths[2]),
        ConfigParamInt("SrcLayer8Plane1Height", &confFile.src[7].attrs.heights[0]),
        ConfigParamInt("SrcLayer8Plane2Height", &confFile.src[7].attrs.heights[1]),
        ConfigParamInt("SrcLayer8Plane3Height", &confFile.src[7].attrs.heights[2]),
        ConfigParamInt("SrcLayer8SrcRectLeft", &confFile.src[7].srcRect.left),
        ConfigParamInt("SrcLayer8SrcRectTop", &confFile.src[7].srcRect.top),
        ConfigParamInt("SrcLayer8SrcRectRight", &confFile.src[7].srcRect.right),
        ConfigParamInt("SrcLayer8SrcRectBottom", &confFile.src[7].srcRect.bottom),
        ConfigParamInt("SrcLayer8DstRectLeft", &confFile.src[7].dstRect.left),
        ConfigParamInt("SrcLayer8DstRectTop", &confFile.src[7].dstRect.top),
        ConfigParamInt("SrcLayer8DstRectRight", &confFile.src[7].dstRect.right),
        ConfigParamInt("SrcLayer8DstRectBottom", &confFile.src[7].dstRect.bottom),
        ConfigParamStr("SrcLayer8Transform", &confFile.src[7].transform),
        ConfigParamStr("SrcLayer8Filtering", &confFile.src[7].filter),
        ConfigParamStr("SrcLayer8Blending", &confFile.src[7].blendMode),
        ConfigParamFloat("SrcLayer8BlendingConstantAlpha", &confFile.src[7].constAlpha),

        ConfigParamFilename("SrcLayer9InputFile", &confFile.src[8].filename),
        ConfigParamStr("SrcLayer9Layout", &confFile.src[8].attrs.layout),
        ConfigParamStr("SrcLayer9ScanType", &confFile.src[8].attrs.scanType),
        ConfigParamStr("SrcLayer9Plane1ColorFormat", &confFile.src[8].attrs.colorFmts[0]),
        ConfigParamStr("SrcLayer9Plane2ColorFormat", &confFile.src[8].attrs.colorFmts[1]),
        ConfigParamStr("SrcLayer9Plane3ColorFormat", &confFile.src[8].attrs.colorFmts[2]),
        ConfigParamStr("SrcLayer9Plane1ColorStandard", &confFile.src[8].attrs.colorStds[0]),
        ConfigParamStr("SrcLayer9Plane2ColorStandard", &confFile.src[8].attrs.colorStds[1]),
        ConfigParamStr("SrcLayer9Plane3ColorStandard", &confFile.src[8].attrs.colorStds[2]),
        ConfigParamInt("SrcLayer9Plane1Width", &confFile.src[8].attrs.widths[0]),
        ConfigParamInt("SrcLayer9Plane2Width", &confFile.src[8].attrs.widths[1]),
        ConfigParamInt("SrcLayer9Plane3Width", &confFile.src[8].attrs.widths[2]),
        ConfigParamInt("SrcLayer9Plane1Height", &confFile.src[8].attrs.heights[0]),
        ConfigParamInt("SrcLayer9Plane2Height", &confFile.src[8].attrs.heights[1]),
        ConfigParamInt("SrcLayer9Plane3Height", &confFile.src[8].attrs.heights[2]),
        ConfigParamInt("SrcLayer9SrcRectLeft", &confFile.src[8].srcRect.left),
        ConfigParamInt("SrcLayer9SrcRectTop", &confFile.src[8].srcRect.top),
        ConfigParamInt("SrcLayer9SrcRectRight", &confFile.src[8].srcRect.right),
        ConfigParamInt("SrcLayer9SrcRectBottom", &confFile.src[8].srcRect.bottom),
        ConfigParamInt("SrcLayer9DstRectLeft", &confFile.src[8].dstRect.left),
        ConfigParamInt("SrcLayer9DstRectTop", &confFile.src[8].dstRect.top),
        ConfigParamInt("SrcLayer9DstRectRight", &confFile.src[8].dstRect.right),
        ConfigParamInt("SrcLayer9DstRectBottom", &confFile.src[8].dstRect.bottom),
        ConfigParamStr("SrcLayer9Transform", &confFile.src[8].transform),
        ConfigParamStr("SrcLayer9Filtering", &confFile.src[8].filter),
        ConfigParamStr("SrcLayer9Blending", &confFile.src[8].blendMode),
        ConfigParamFloat("SrcLayer9BlendingConstantAlpha", &confFile.src[8].constAlpha),

        ConfigParamFilename("SrcLayer10InputFile", &confFile.src[9].filename),
        ConfigParamStr("SrcLayer10Layout", &confFile.src[9].attrs.layout),
        ConfigParamStr("SrcLayer10ScanType", &confFile.src[9].attrs.scanType),
        ConfigParamStr("SrcLayer10Plane1ColorFormat", &confFile.src[9].attrs.colorFmts[0]),
        ConfigParamStr("SrcLayer10Plane2ColorFormat", &confFile.src[9].attrs.colorFmts[1]),
        ConfigParamStr("SrcLayer10Plane3ColorFormat", &confFile.src[9].attrs.colorFmts[2]),
        ConfigParamStr("SrcLayer10Plane1ColorStandard", &confFile.src[9].attrs.colorStds[0]),
        ConfigParamStr("SrcLayer10Plane2ColorStandard", &confFile.src[9].attrs.colorStds[1]),
        ConfigParamStr("SrcLayer10Plane3ColorStandard", &confFile.src[9].attrs.colorStds[2]),
        ConfigParamInt("SrcLayer10Plane1Width", &confFile.src[9].attrs.widths[0]),
        ConfigParamInt("SrcLayer10Plane2Width", &confFile.src[9].attrs.widths[1]),
        ConfigParamInt("SrcLayer10Plane3Width", &confFile.src[9].attrs.widths[2]),
        ConfigParamInt("SrcLayer10Plane1Height", &confFile.src[9].attrs.heights[0]),
        ConfigParamInt("SrcLayer10Plane2Height", &confFile.src[9].attrs.heights[1]),
        ConfigParamInt("SrcLayer10Plane3Height", &confFile.src[9].attrs.heights[2]),
        ConfigParamInt("SrcLayer10SrcRectLeft", &confFile.src[9].srcRect.left),
        ConfigParamInt("SrcLayer10SrcRectTop", &confFile.src[9].srcRect.top),
        ConfigParamInt("SrcLayer10SrcRectRight", &confFile.src[9].srcRect.right),
        ConfigParamInt("SrcLayer10SrcRectBottom", &confFile.src[9].srcRect.bottom),
        ConfigParamInt("SrcLayer10DstRectLeft", &confFile.src[9].dstRect.left),
        ConfigParamInt("SrcLayer10DstRectTop", &confFile.src[9].dstRect.top),
        ConfigParamInt("SrcLayer10DstRectRight", &confFile.src[9].dstRect.right),
        ConfigParamInt("SrcLayer10DstRectBottom", &confFile.src[9].dstRect.bottom),
        ConfigParamStr("SrcLayer10Transform", &confFile.src[9].transform),
        ConfigParamStr("SrcLayer10Filtering", &confFile.src[9].filter),
        ConfigParamStr("SrcLayer10Blending", &confFile.src[9].blendMode),
        ConfigParamFloat("SrcLayer10BlendingConstantAlpha", &confFile.src[9].constAlpha),

        ConfigParamFilename("SrcLayer11InputFile", &confFile.src[10].filename),
        ConfigParamStr("SrcLayer11Layout", &confFile.src[10].attrs.layout),
        ConfigParamStr("SrcLayer11ScanType", &confFile.src[10].attrs.scanType),
        ConfigParamStr("SrcLayer11Plane1ColorFormat", &confFile.src[10].attrs.colorFmts[0]),
        ConfigParamStr("SrcLayer11Plane2ColorFormat", &confFile.src[10].attrs.colorFmts[1]),
        ConfigParamStr("SrcLayer11Plane3ColorFormat", &confFile.src[10].attrs.colorFmts[2]),
        ConfigParamStr("SrcLayer11Plane1ColorStandard", &confFile.src[10].attrs.colorStds[0]),
        ConfigParamStr("SrcLayer11Plane2ColorStandard", &confFile.src[10].attrs.colorStds[1]),
        ConfigParamStr("SrcLayer11Plane3ColorStandard", &confFile.src[10].attrs.colorStds[2]),
        ConfigParamInt("SrcLayer11Plane1Width", &confFile.src[10].attrs.widths[0]),
        ConfigParamInt("SrcLayer11Plane2Width", &confFile.src[10].attrs.widths[1]),
        ConfigParamInt("SrcLayer11Plane3Width", &confFile.src[10].attrs.widths[2]),
        ConfigParamInt("SrcLayer11Plane1Height", &confFile.src[10].attrs.heights[0]),
        ConfigParamInt("SrcLayer11Plane2Height", &confFile.src[10].attrs.heights[1]),
        ConfigParamInt("SrcLayer11Plane3Height", &confFile.src[10].attrs.heights[2]),
        ConfigParamInt("SrcLayer11SrcRectLeft", &confFile.src[10].srcRect.left),
        ConfigParamInt("SrcLayer11SrcRectTop", &confFile.src[10].srcRect.top),
        ConfigParamInt("SrcLayer11SrcRectRight", &confFile.src[10].srcRect.right),
        ConfigParamInt("SrcLayer11SrcRectBottom", &confFile.src[10].srcRect.bottom),
        ConfigParamInt("SrcLayer11DstRectLeft", &confFile.src[10].dstRect.left),
        ConfigParamInt("SrcLayer11DstRectTop", &confFile.src[10].dstRect.top),
        ConfigParamInt("SrcLayer11DstRectRight", &confFile.src[10].dstRect.right),
        ConfigParamInt("SrcLayer11DstRectBottom", &confFile.src[10].dstRect.bottom),
        ConfigParamStr("SrcLayer11Transform", &confFile.src[10].transform),
        ConfigParamStr("SrcLayer11Filtering", &confFile.src[10].filter),
        ConfigParamStr("SrcLayer11Blending", &confFile.src[10].blendMode),
        ConfigParamFloat("SrcLayer11BlendingConstantAlpha", &confFile.src[10].constAlpha),

        ConfigParamFilename("SrcLayer12InputFile", &confFile.src[11].filename),
        ConfigParamStr("SrcLayer12Layout", &confFile.src[11].attrs.layout),
        ConfigParamStr("SrcLayer12ScanType", &confFile.src[11].attrs.scanType),
        ConfigParamStr("SrcLayer12Plane1ColorFormat", &confFile.src[11].attrs.colorFmts[0]),
        ConfigParamStr("SrcLayer12Plane2ColorFormat", &confFile.src[11].attrs.colorFmts[1]),
        ConfigParamStr("SrcLayer12Plane3ColorFormat", &confFile.src[11].attrs.colorFmts[2]),
        ConfigParamStr("SrcLayer12Plane1ColorStandard", &confFile.src[11].attrs.colorStds[0]),
        ConfigParamStr("SrcLayer12Plane2ColorStandard", &confFile.src[11].attrs.colorStds[1]),
        ConfigParamStr("SrcLayer12Plane3ColorStandard", &confFile.src[11].attrs.colorStds[2]),
        ConfigParamInt("SrcLayer12Plane1Width", &confFile.src[11].attrs.widths[0]),
        ConfigParamInt("SrcLayer12Plane2Width", &confFile.src[11].attrs.widths[1]),
        ConfigParamInt("SrcLayer12Plane3Width", &confFile.src[11].attrs.widths[2]),
        ConfigParamInt("SrcLayer12Plane1Height", &confFile.src[11].attrs.heights[0]),
        ConfigParamInt("SrcLayer12Plane2Height", &confFile.src[11].attrs.heights[1]),
        ConfigParamInt("SrcLayer12Plane3Height", &confFile.src[11].attrs.heights[2]),
        ConfigParamInt("SrcLayer12SrcRectLeft", &confFile.src[11].srcRect.left),
        ConfigParamInt("SrcLayer12SrcRectTop", &confFile.src[11].srcRect.top),
        ConfigParamInt("SrcLayer12SrcRectRight", &confFile.src[11].srcRect.right),
        ConfigParamInt("SrcLayer12SrcRectBottom", &confFile.src[11].srcRect.bottom),
        ConfigParamInt("SrcLayer12DstRectLeft", &confFile.src[11].dstRect.left),
        ConfigParamInt("SrcLayer12DstRectTop", &confFile.src[11].dstRect.top),
        ConfigParamInt("SrcLayer12DstRectRight", &confFile.src[11].dstRect.right),
        ConfigParamInt("SrcLayer12DstRectBottom", &confFile.src[11].dstRect.bottom),
        ConfigParamStr("SrcLayer12Transform", &confFile.src[11].transform),
        ConfigParamStr("SrcLayer12Filtering", &confFile.src[11].filter),
        ConfigParamStr("SrcLayer12Blending", &confFile.src[11].blendMode),
        ConfigParamFloat("SrcLayer12BlendingConstantAlpha", &confFile.src[11].constAlpha),

        ConfigParamFilename("SrcLayer13InputFile", &confFile.src[12].filename),
        ConfigParamStr("SrcLayer13Layout", &confFile.src[12].attrs.layout),
        ConfigParamStr("SrcLayer13ScanType", &confFile.src[12].attrs.scanType),
        ConfigParamStr("SrcLayer13Plane1ColorFormat", &confFile.src[12].attrs.colorFmts[0]),
        ConfigParamStr("SrcLayer13Plane2ColorFormat", &confFile.src[12].attrs.colorFmts[1]),
        ConfigParamStr("SrcLayer13Plane3ColorFormat", &confFile.src[12].attrs.colorFmts[2]),
        ConfigParamStr("SrcLayer13Plane1ColorStandard", &confFile.src[12].attrs.colorStds[0]),
        ConfigParamStr("SrcLayer13Plane2ColorStandard", &confFile.src[12].attrs.colorStds[1]),
        ConfigParamStr("SrcLayer13Plane3ColorStandard", &confFile.src[12].attrs.colorStds[2]),
        ConfigParamInt("SrcLayer13Plane1Width", &confFile.src[12].attrs.widths[0]),
        ConfigParamInt("SrcLayer13Plane2Width", &confFile.src[12].attrs.widths[1]),
        ConfigParamInt("SrcLayer13Plane3Width", &confFile.src[12].attrs.widths[2]),
        ConfigParamInt("SrcLayer13Plane1Height", &confFile.src[12].attrs.heights[0]),
        ConfigParamInt("SrcLayer13Plane2Height", &confFile.src[12].attrs.heights[1]),
        ConfigParamInt("SrcLayer13Plane3Height", &confFile.src[12].attrs.heights[2]),
        ConfigParamInt("SrcLayer13SrcRectLeft", &confFile.src[12].srcRect.left),
        ConfigParamInt("SrcLayer13SrcRectTop", &confFile.src[12].srcRect.top),
        ConfigParamInt("SrcLayer13SrcRectRight", &confFile.src[12].srcRect.right),
        ConfigParamInt("SrcLayer13SrcRectBottom", &confFile.src[12].srcRect.bottom),
        ConfigParamInt("SrcLayer13DstRectLeft", &confFile.src[12].dstRect.left),
        ConfigParamInt("SrcLayer13DstRectTop", &confFile.src[12].dstRect.top),
        ConfigParamInt("SrcLayer13DstRectRight", &confFile.src[12].dstRect.right),
        ConfigParamInt("SrcLayer13DstRectBottom", &confFile.src[12].dstRect.bottom),
        ConfigParamStr("SrcLayer13Transform", &confFile.src[12].transform),
        ConfigParamStr("SrcLayer13Filtering", &confFile.src[12].filter),
        ConfigParamStr("SrcLayer13Blending", &confFile.src[12].blendMode),
        ConfigParamFloat("SrcLayer13BlendingConstantAlpha", &confFile.src[12].constAlpha),

        ConfigParamFilename("SrcLayer14InputFile", &confFile.src[13].filename),
        ConfigParamStr("SrcLayer14Layout", &confFile.src[13].attrs.layout),
        ConfigParamStr("SrcLayer14ScanType", &confFile.src[13].attrs.scanType),
        ConfigParamStr("SrcLayer14Plane1ColorFormat", &confFile.src[13].attrs.colorFmts[0]),
        ConfigParamStr("SrcLayer14Plane2ColorFormat", &confFile.src[13].attrs.colorFmts[1]),
        ConfigParamStr("SrcLayer14Plane3ColorFormat", &confFile.src[13].attrs.colorFmts[2]),
        ConfigParamStr("SrcLayer14Plane1ColorStandard", &confFile.src[13].attrs.colorStds[0]),
        ConfigParamStr("SrcLayer14Plane2ColorStandard", &confFile.src[13].attrs.colorStds[1]),
        ConfigParamStr("SrcLayer14Plane3ColorStandard", &confFile.src[13].attrs.colorStds[2]),
        ConfigParamInt("SrcLayer14Plane1Width", &confFile.src[13].attrs.widths[0]),
        ConfigParamInt("SrcLayer14Plane2Width", &confFile.src[13].attrs.widths[1]),
        ConfigParamInt("SrcLayer14Plane3Width", &confFile.src[13].attrs.widths[2]),
        ConfigParamInt("SrcLayer14Plane1Height", &confFile.src[13].attrs.heights[0]),
        ConfigParamInt("SrcLayer14Plane2Height", &confFile.src[13].attrs.heights[1]),
        ConfigParamInt("SrcLayer14Plane3Height", &confFile.src[13].attrs.heights[2]),
        ConfigParamInt("SrcLayer14SrcRectLeft", &confFile.src[13].srcRect.left),
        ConfigParamInt("SrcLayer14SrcRectTop", &confFile.src[13].srcRect.top),
        ConfigParamInt("SrcLayer14SrcRectRight", &confFile.src[13].srcRect.right),
        ConfigParamInt("SrcLayer14SrcRectBottom", &confFile.src[13].srcRect.bottom),
        ConfigParamInt("SrcLayer14DstRectLeft", &confFile.src[13].dstRect.left),
        ConfigParamInt("SrcLayer14DstRectTop", &confFile.src[13].dstRect.top),
        ConfigParamInt("SrcLayer14DstRectRight", &confFile.src[13].dstRect.right),
        ConfigParamInt("SrcLayer14DstRectBottom", &confFile.src[13].dstRect.bottom),
        ConfigParamStr("SrcLayer14Transform", &confFile.src[13].transform),
        ConfigParamStr("SrcLayer14Filtering", &confFile.src[13].filter),
        ConfigParamStr("SrcLayer14Blending", &confFile.src[13].blendMode),
        ConfigParamFloat("SrcLayer14BlendingConstantAlpha", &confFile.src[13].constAlpha),

        ConfigParamFilename("SrcLayer15InputFile", &confFile.src[14].filename),
        ConfigParamStr("SrcLayer15Layout", &confFile.src[14].attrs.layout),
        ConfigParamStr("SrcLayer15ScanType", &confFile.src[14].attrs.scanType),
        ConfigParamStr("SrcLayer15Plane1ColorFormat", &confFile.src[14].attrs.colorFmts[0]),
        ConfigParamStr("SrcLayer15Plane2ColorFormat", &confFile.src[14].attrs.colorFmts[1]),
        ConfigParamStr("SrcLayer15Plane3ColorFormat", &confFile.src[14].attrs.colorFmts[2]),
        ConfigParamStr("SrcLayer15Plane1ColorStandard", &confFile.src[14].attrs.colorStds[0]),
        ConfigParamStr("SrcLayer15Plane2ColorStandard", &confFile.src[14].attrs.colorStds[1]),
        ConfigParamStr("SrcLayer15Plane3ColorStandard", &confFile.src[14].attrs.colorStds[2]),
        ConfigParamInt("SrcLayer15Plane1Width", &confFile.src[14].attrs.widths[0]),
        ConfigParamInt("SrcLayer15Plane2Width", &confFile.src[14].attrs.widths[1]),
        ConfigParamInt("SrcLayer15Plane3Width", &confFile.src[14].attrs.widths[2]),
        ConfigParamInt("SrcLayer15Plane1Height", &confFile.src[14].attrs.heights[0]),
        ConfigParamInt("SrcLayer15Plane2Height", &confFile.src[14].attrs.heights[1]),
        ConfigParamInt("SrcLayer15Plane3Height", &confFile.src[14].attrs.heights[2]),
        ConfigParamInt("SrcLayer15SrcRectLeft", &confFile.src[14].srcRect.left),
        ConfigParamInt("SrcLayer15SrcRectTop", &confFile.src[14].srcRect.top),
        ConfigParamInt("SrcLayer15SrcRectRight", &confFile.src[14].srcRect.right),
        ConfigParamInt("SrcLayer15SrcRectBottom", &confFile.src[14].srcRect.bottom),
        ConfigParamInt("SrcLayer15DstRectLeft", &confFile.src[14].dstRect.left),
        ConfigParamInt("SrcLayer15DstRectTop", &confFile.src[14].dstRect.top),
        ConfigParamInt("SrcLayer15DstRectRight", &confFile.src[14].dstRect.right),
        ConfigParamInt("SrcLayer15DstRectBottom", &confFile.src[14].dstRect.bottom),
        ConfigParamStr("SrcLayer15Transform", &confFile.src[14].transform),
        ConfigParamStr("SrcLayer15Filtering", &confFile.src[14].filter),
        ConfigParamStr("SrcLayer15Blending", &confFile.src[14].blendMode),
        ConfigParamFloat("SrcLayer15BlendingConstantAlpha", &confFile.src[14].constAlpha),

        ConfigParamFilename("SrcLayer16InputFile", &confFile.src[15].filename),
        ConfigParamStr("SrcLayer16Layout", &confFile.src[15].attrs.layout),
        ConfigParamStr("SrcLayer16ScanType", &confFile.src[15].attrs.scanType),
        ConfigParamStr("SrcLayer16Plane1ColorFormat", &confFile.src[15].attrs.colorFmts[0]),
        ConfigParamStr("SrcLayer16Plane2ColorFormat", &confFile.src[15].attrs.colorFmts[1]),
        ConfigParamStr("SrcLayer16Plane3ColorFormat", &confFile.src[15].attrs.colorFmts[2]),
        ConfigParamStr("SrcLayer16Plane1ColorStandard", &confFile.src[15].attrs.colorStds[0]),
        ConfigParamStr("SrcLayer16Plane2ColorStandard", &confFile.src[15].attrs.colorStds[1]),
        ConfigParamStr("SrcLayer16Plane3ColorStandard", &confFile.src[15].attrs.colorStds[2]),
        ConfigParamInt("SrcLayer16Plane1Width", &confFile.src[15].attrs.widths[0]),
        ConfigParamInt("SrcLayer16Plane2Width", &confFile.src[15].attrs.widths[1]),
        ConfigParamInt("SrcLayer16Plane3Width", &confFile.src[15].attrs.widths[2]),
        ConfigParamInt("SrcLayer16Plane1Height", &confFile.src[15].attrs.heights[0]),
        ConfigParamInt("SrcLayer16Plane2Height", &confFile.src[15].attrs.heights[1]),
        ConfigParamInt("SrcLayer16Plane3Height", &confFile.src[15].attrs.heights[2]),
        ConfigParamInt("SrcLayer16SrcRectLeft", &confFile.src[15].srcRect.left),
        ConfigParamInt("SrcLayer16SrcRectTop", &confFile.src[15].srcRect.top),
        ConfigParamInt("SrcLayer16SrcRectRight", &confFile.src[15].srcRect.right),
        ConfigParamInt("SrcLayer16SrcRectBottom", &confFile.src[15].srcRect.bottom),
        ConfigParamInt("SrcLayer16DstRectLeft", &confFile.src[15].dstRect.left),
        ConfigParamInt("SrcLayer16DstRectTop", &confFile.src[15].dstRect.top),
        ConfigParamInt("SrcLayer16DstRectRight", &confFile.src[15].dstRect.right),
        ConfigParamInt("SrcLayer16DstRectBottom", &confFile.src[15].dstRect.bottom),
        ConfigParamStr("SrcLayer16Transform", &confFile.src[15].transform),
        ConfigParamStr("SrcLayer16Filtering", &confFile.src[15].filter),
        ConfigParamStr("SrcLayer16Blending", &confFile.src[15].blendMode),
        ConfigParamFloat("SrcLayer16BlendingConstantAlpha", &confFile.src[15].constAlpha),

        ConfigParamFilename("DstOutputFile", &confFile.dstFilename),
        ConfigParamStr("DstLayout", &confFile.dst.layout),
        ConfigParamStr("DstScanType", &confFile.dst.scanType),
        ConfigParamStr("DstPlane1ColorFormat", &confFile.dst.colorFmts[0]),
        ConfigParamStr("DstPlane2ColorFormat", &confFile.dst.colorFmts[1]),
        ConfigParamStr("DstPlane3ColorFormat", &confFile.dst.colorFmts[2]),
        ConfigParamStr("DstPlane1ColorStandard", &confFile.dst.colorStds[0]),
        ConfigParamStr("DstPlane2ColorStandard", &confFile.dst.colorStds[1]),
        ConfigParamStr("DstPlane3ColorStandard", &confFile.dst.colorStds[2]),
        ConfigParamInt("DstPlane1Width", &confFile.dst.widths[0]),
        ConfigParamInt("DstPlane2Width", &confFile.dst.widths[1]),
        ConfigParamInt("DstPlane3Width", &confFile.dst.widths[2]),
        ConfigParamInt("DstPlane1Height", &confFile.dst.heights[0]),
        ConfigParamInt("DstPlane2Height", &confFile.dst.heights[1]),
        ConfigParamInt("DstPlane3Height", &confFile.dst.heights[2]),

        ConfigParamStr("FileIOMode", &confFile.fileIOMode),

        ConfigParamSentinel(),
    };

    if (ConfigParser_InitParamsMap(paramsMap) != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("Failed to initialize config params map");
        return 1;
    }

    SectionMap sectionMap = { SECTION_NONE, "", 0, 0 };
    if (ConfigParser_ParseFile(paramsMap, 1, &sectionMap, filename) != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("Failed to parse config file");
        return 1;
    }

    if (SetConfig(config, &confFile) != 0)
    {
        LOG_ERR("Failed to set config from config file");
        return 1;
    }

    return 0;
}

static int
AllocateSyncObj(SyncObj *sync, NvSciSyncModule syncModule, NvMedia2D *handle)
{
    NvMediaStatus result = NVMEDIA_STATUS_OK;
    NvSciError sciResult = NvSciError_Success;

    /* Create an attribute list describing NvMedia 2D as a signaler */
    sciResult = NvSciSyncAttrListCreate(syncModule, &sync->attrList);
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("NvSciSyncAttrListCreate failed with %d", sciResult);
        return 1;
    }

    /* Fill NvMedia 2D mandatory attributes to the attribute list */
    result = NvMedia2DFillNvSciSyncAttrList(handle, sync->attrList, NVMEDIA_SIGNALER);
    if (result != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("NvMedia2DFillNvSciSyncAttrList failed with %d", result);
        return 1;
    }

    /* Create another attribute list describing a CPU waiter */
    sciResult = NvSciSyncAttrListCreate(syncModule, &sync->attrListCpu);
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("NvSciSyncAttrListCreate failed with %d", sciResult);
        return 1;
    }

    /* Add the attributes necessary for waiting on CPU */
    bool cpuAccess = true;
    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
    NvSciSyncAttrKeyValuePair cpuAttrKvPairs[] = {
        { NvSciSyncAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess) },
        { NvSciSyncAttrKey_RequiredPerm, &cpuPerm, sizeof(cpuPerm) },
    };
    sciResult = NvSciSyncAttrListSetAttrs(sync->attrListCpu,
                                          cpuAttrKvPairs,
                                          sizeof(cpuAttrKvPairs) / sizeof(cpuAttrKvPairs[0]));
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("NvSciSyncAttrListSetAttrs failed with %d", sciResult);
        return 1;
    }

    /* Reconcile the attribute lists */
    NvSciSyncAttrList attrLists[] = { sync->attrList, sync->attrListCpu };
    sciResult = NvSciSyncAttrListReconcile(attrLists,
                                           sizeof(attrLists) / sizeof(attrLists[0]),
                                           &sync->attrListReconciled,
                                           &sync->attrListConflict);
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("NvSciSyncAttrListReconcile failed with %d", sciResult);
        return 1;
    }

    /* Create the NvSciSyncObj */
    sciResult = NvSciSyncObjAlloc(sync->attrListReconciled, &sync->obj);
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("NvSciSyncObjAlloc failed with %d", sciResult);
        return 1;
    }

    return 0;
}

static void
DeallocateSyncObj(SyncObj *sync)
{
    if (sync->obj)
    {
        NvSciSyncObjFree(sync->obj);
        sync->obj = NULL;
    }

    if (sync->attrListConflict)
    {
        NvSciSyncAttrListFree(sync->attrListConflict);
        sync->attrListConflict = NULL;
    }

    if (sync->attrListReconciled)
    {
        NvSciSyncAttrListFree(sync->attrListReconciled);
        sync->attrListReconciled = NULL;
    }

    if (sync->attrListCpu)
    {
        NvSciSyncAttrListFree(sync->attrListCpu);
        sync->attrListCpu = NULL;
    }

    if (sync->attrList)
    {
        NvSciSyncAttrListFree(sync->attrList);
        sync->attrList = NULL;
    }
}

static int
AllocateBufObj(BufObj *buf, const SurfaceAttrs *attrs, NvSciBufModule bufModule, NvMedia2D *handle)
{
    NvMediaStatus result = NVMEDIA_STATUS_OK;
    NvSciError sciResult = NvSciError_Success;

    /* Create the attribute list describing the surface */
    sciResult = NvSciBufAttrListCreate(bufModule, &buf->attrList);
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("NvSciBufAttrListCreate failed with %d", sciResult);
        return 1;
    }

    /* Fill NvMedia 2D mandatory attributes to the attribute list */
    result = NvMedia2DFillNvSciBufAttrList(handle, buf->attrList);
    if (result != NVMEDIA_STATUS_OK)
    {
        LOG_ERR("NvMedia2DFillNvSciBufAttrList failed with %d", result);
        return 1;
    }

    /* Fill the surface attributes to the attribute list */
    NvSciBufType bufType = NvSciBufType_Image;
    bool cpuAccess = true;
    NvSciBufAttrKeyValuePair attrKvPairs[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess) },
        { NvSciBufImageAttrKey_Layout, &attrs->layout, sizeof(attrs->layout) },
        { NvSciBufImageAttrKey_ScanType, &attrs->scanType, sizeof(attrs->scanType) },
        { NvSciBufImageAttrKey_PlaneCount, &attrs->planeCount, sizeof(attrs->planeCount) },
        { NvSciBufImageAttrKey_PlaneColorFormat,
          attrs->planeColorFmts,
          attrs->planeCount * sizeof(attrs->planeColorFmts[0]) },
        { NvSciBufImageAttrKey_PlaneWidth,
          attrs->planeWidths,
          attrs->planeCount * sizeof(attrs->planeWidths[0]) },
        { NvSciBufImageAttrKey_PlaneHeight,
          attrs->planeHeights,
          attrs->planeCount * sizeof(attrs->planeHeights[0]) },
    };
    sciResult = NvSciBufAttrListSetAttrs(buf->attrList,
                                         attrKvPairs,
                                         sizeof(attrKvPairs) / sizeof(attrKvPairs[0]));
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("NvSciBufAttrListSetAttrs failed with %d", sciResult);
        return 1;
    }

    /* Fill the optional color standard attribute if the surface has it set */
    if (attrs->hasPlaneColorStds)
    {
        NvSciBufAttrKeyValuePair attr = { NvSciBufImageAttrKey_PlaneColorStd,
                                          attrs->planeColorStds,
                                          attrs->planeCount * sizeof(attrs->planeColorStds[0]) };
        sciResult = NvSciBufAttrListSetAttrs(buf->attrList, &attr, 1);
        if (sciResult != NvSciError_Success)
        {
            LOG_ERR("NvSciBufAttrListSetAttrs failed with %d", sciResult);
            return 1;
        }
    }

    /* Reconcile the attribute list */
    sciResult = NvSciBufAttrListReconcile(&buf->attrList,
                                          1,
                                          &buf->attrListReconciled,
                                          &buf->attrListConflict);
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("NvSciBufAttrListReconcile failed with %d", sciResult);
        return 1;
    }

    /* Create the NvSciBufObj for the surface */
    sciResult = NvSciBufObjAlloc(buf->attrListReconciled, &buf->obj);
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("NvSciBufObjAlloc failed with %d", sciResult);
        return 1;
    }

    return 0;
}

static void
DeallocateBufObj(BufObj *buf)
{
    if (buf->obj)
    {
        NvSciBufObjFree(buf->obj);
        buf->obj = NULL;
    }

    if (buf->attrListConflict)
    {
        NvSciBufAttrListFree(buf->attrListConflict);
        buf->attrListConflict = NULL;
    }

    if (buf->attrListReconciled)
    {
        NvSciBufAttrListFree(buf->attrListReconciled);
        buf->attrListReconciled = NULL;
    }

    if (buf->attrList)
    {
        NvSciBufAttrListFree(buf->attrList);
        buf->attrList = NULL;
    }
}

typedef struct
{
    void *buffer;
    uint64_t size;
    uint32_t planeCount;
    void *planePtrs[MAX_PLANE_COUNT];
    uint32_t planeSizes[MAX_PLANE_COUNT];
    uint32_t planePitches[MAX_PLANE_COUNT];
} PixelDataBuffer;

static bool
IsInterleavedYUV(NvSciBufAttrValColorFmt colorFmt)
{
    switch (colorFmt)
    {
    case NvSciColor_Y8U8Y8V8:
    case NvSciColor_Y8V8Y8U8:
    case NvSciColor_U8Y8V8Y8:
    case NvSciColor_V8Y8U8Y8:
        return true;
    default:
        return false;
    }
}

static int
AllocatePixelDataBuffer(PixelDataBuffer *px, NvSciBufObj sciBuf, FileIOMode mode)
{
    NvSciError sciResult = NvSciError_Success;

    NvSciBufAttrList attrList;
    sciResult = NvSciBufObjGetAttrList(sciBuf, &attrList);
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("Failed to get buffer attribute list");
        return 1;
    }

    enum
    {
        PLANE_COUNT_ATTR_IDX,
        PLANE_COLORFMT_ATTR_IDX,
        PLANE_WIDTH_ATTR_IDX,
        PLANE_HEIGHT_ATTR_IDX,
        PLANE_BPP_ATTR_IDX,
        ATTR_COUNT
    };

    NvSciBufAttrKeyValuePair attrs[ATTR_COUNT];
    attrs[PLANE_COUNT_ATTR_IDX].key = NvSciBufImageAttrKey_PlaneCount;
    attrs[PLANE_COLORFMT_ATTR_IDX].key = NvSciBufImageAttrKey_PlaneColorFormat;
    attrs[PLANE_WIDTH_ATTR_IDX].key = NvSciBufImageAttrKey_PlaneWidth;
    attrs[PLANE_HEIGHT_ATTR_IDX].key = NvSciBufImageAttrKey_PlaneHeight;
    attrs[PLANE_BPP_ATTR_IDX].key = NvSciBufImageAttrKey_PlaneBitsPerPixel;
    sciResult = NvSciBufAttrListGetAttrs(attrList, attrs, ATTR_COUNT);
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("Failed to get buffer attributes");
        return 1;
    }

    px->size = 0;
    px->planeCount = *((const uint32_t *)attrs[PLANE_COUNT_ATTR_IDX].value);

    for (uint32_t i = 0; i < px->planeCount; ++i)
    {
        uint32_t width = ((const uint32_t *)attrs[PLANE_WIDTH_ATTR_IDX].value)[i];
        uint32_t height = ((const uint32_t *)attrs[PLANE_HEIGHT_ATTR_IDX].value)[i];
        uint32_t bpp = ((const uint32_t *)attrs[PLANE_BPP_ATTR_IDX].value)[i];
        uint64_t pitchBits = width * bpp;
        px->planeSizes[i] = pitchBits * height / 8;
        px->planePitches[i] = pitchBits / 8;
        px->size += px->planeSizes[i];
    }

    /* NvSciBufObjGet/PutPixels() requires three separate planes for interleaved and semiplanar YUV
     * surfaces. Do special handling for them.*/
    if (mode == FILE_IO_MODE_NVSCI)
    {
        if (px->planeCount == 1 &&
            IsInterleavedYUV(
                ((const NvSciBufAttrValColorFmt *)attrs[PLANE_COLORFMT_ATTR_IDX].value)[0]))
        {
            px->planeCount = 3;
            uint32_t fullSize = px->planeSizes[0];
            px->planeSizes[0] = fullSize / 2;
            px->planeSizes[1] = fullSize / 4;
            px->planeSizes[2] = fullSize / 4;
            uint32_t fullPitch = px->planePitches[0];
            px->planePitches[0] = fullPitch / 2;
            px->planePitches[1] = fullPitch / 4;
            px->planePitches[2] = fullPitch / 4;
        }
        else if (px->planeCount == 2)
        {
            px->planeCount = 3;
            uint32_t fullChromaSize = px->planeSizes[1];
            px->planeSizes[1] = fullChromaSize / 2;
            px->planeSizes[2] = fullChromaSize / 2;
            uint32_t fullChromaPitch = px->planePitches[1];
            px->planePitches[1] = fullChromaPitch / 2;
            px->planePitches[2] = fullChromaPitch / 2;
        }
    }

    px->buffer = malloc(px->size);

    px->planePtrs[0] = px->buffer;
    for (uint32_t i = 1; i < px->planeCount; ++i)
    {
        px->planePtrs[i] = ((char *)px->planePtrs[i - 1]) + px->planeSizes[i - 1];
    }

    return 0;
}

static void
DeallocatePixelDataBuffer(PixelDataBuffer *px)
{
    if (px->buffer)
    {
        free(px->buffer);
        px->buffer = NULL;
    }
}

static int
ReadWritePixelDataLineByLine(NvSciBufObj sciBuf, PixelDataBuffer *px, bool write)
{
    NvSciError sciResult = NvSciError_Success;

    void *sciBufPtr;
    sciResult = NvSciBufObjGetCpuPtr(sciBuf, &sciBufPtr);
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("Failed to get buffer CPU pointer");
        return 1;
    }

    NvSciBufAttrList attrList;
    sciResult = NvSciBufObjGetAttrList(sciBuf, &attrList);
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("Failed to get buffer attribute list");
        return 1;
    }

    enum
    {
        PLANE_OFFSET_ATTR_IDX,
        PLANE_PITCH_ATTR_IDX,
        ATTR_COUNT
    };

    NvSciBufAttrKeyValuePair attrs[ATTR_COUNT];
    attrs[PLANE_OFFSET_ATTR_IDX].key = NvSciBufImageAttrKey_PlaneOffset;
    attrs[PLANE_PITCH_ATTR_IDX].key = NvSciBufImageAttrKey_PlanePitch;
    sciResult = NvSciBufAttrListGetAttrs(attrList, attrs, ATTR_COUNT);
    if (sciResult != NvSciError_Success)
    {
        LOG_ERR("Failed to get buffer attributes");
        return 1;
    }

    for (uint32_t plane = 0; plane < px->planeCount; ++plane)
    {
        uint64_t sciBufPlaneOffset = ((const uint64_t *)attrs[PLANE_OFFSET_ATTR_IDX].value)[plane];
        uint32_t sciBufPlanePitch = ((const uint32_t *)attrs[PLANE_PITCH_ATTR_IDX].value)[plane];
        uint32_t planeHeight = px->planeSizes[plane] / px->planePitches[plane];
        for (uint32_t line = 0; line < planeHeight; ++line)
        {
            char *sciBufLine = (char *)sciBufPtr + sciBufPlaneOffset + line * sciBufPlanePitch;
            char *pxBufLine = (char *)px->planePtrs[plane] + line * px->planePitches[plane];
            if (write)
            {
                memcpy(sciBufLine, pxBufLine, px->planePitches[plane]);
            }
            else
            {
                memcpy(pxBufLine, sciBufLine, px->planePitches[plane]);
            }
        }
    }

    return 0;
}

static int
ReadBufferFromFile(NvSciBufObj buffer, Filename *filename, FileIOMode mode)
{
    int retval = 0;
    NvSciError sciResult = NvSciError_Success;
    FILE *file = NULL;

    PixelDataBuffer px;
    memset(&px, 0, sizeof(px));
    if (AllocatePixelDataBuffer(&px, buffer, mode) != 0)
    {
        LOG_ERR("Failed to allocate pixel data buffer");
        retval = 1;
        goto ReadBufferFromFileEnd;
    }

    file = fopen(filename->data, "rb");
    if (!file)
    {
        LOG_ERR("Failed to open file %s", filename->data);
        retval = 1;
        goto ReadBufferFromFileEnd;
    }

    size_t bytesRead = fread(px.buffer, 1, px.size, file);
    if (bytesRead != px.size)
    {
        LOG_ERR("Expected to read %u bytes, got %u", px.size, bytesRead);
        retval = 1;
        goto ReadBufferFromFileEnd;
    }

    switch (mode)
    {
    case FILE_IO_MODE_NVSCI:
        sciResult = NvSciBufObjPutPixels(buffer,
                                         NULL,
                                         (const void **)px.planePtrs,
                                         px.planeSizes,
                                         px.planePitches);
        if (sciResult != NvSciError_Success)
        {
            LOG_ERR("Failed to write data to buffer using NvSci");
            retval = 1;
            goto ReadBufferFromFileEnd;
        }
        break;

    case FILE_IO_MODE_LINE_BY_LINE:
        if (ReadWritePixelDataLineByLine(buffer, &px, true) != 0)
        {
            LOG_ERR("Failed to write data to buffer line-by-line");
            retval = 1;
            goto ReadBufferFromFileEnd;
        }
        break;
    }

ReadBufferFromFileEnd:
    DeallocatePixelDataBuffer(&px);

    if (file)
    {
        fclose(file);
    }

    return retval;
}

static int
WriteBufferToFile(NvSciBufObj buffer, Filename *filename, FileIOMode mode)
{
    int retval = 0;
    NvSciError sciResult = NvSciError_Success;
    FILE *file = NULL;

    PixelDataBuffer px;
    memset(&px, 0, sizeof(px));
    if (AllocatePixelDataBuffer(&px, buffer, mode) != 0)
    {
        LOG_ERR("Failed to allocate pixel data buffer");
        retval = 1;
        goto WriteBufferToFileEnd;
    }

    file = fopen(filename->data, "wb");
    if (!file)
    {
        LOG_ERR("Failed to open file %s", filename->data);
        retval = 1;
        goto WriteBufferToFileEnd;
    }

    switch (mode)
    {
    case FILE_IO_MODE_NVSCI:
        sciResult =
            NvSciBufObjGetPixels(buffer, NULL, px.planePtrs, px.planeSizes, px.planePitches);
        if (sciResult != NvSciError_Success)
        {
            LOG_ERR("Failed to read data from buffer using NvSci");
            retval = 1;
            goto WriteBufferToFileEnd;
        }
        break;

    case FILE_IO_MODE_LINE_BY_LINE:
        if (ReadWritePixelDataLineByLine(buffer, &px, false) != 0)
        {
            LOG_ERR("Failed to read data from buffer line-by-line");
            retval = 1;
            goto WriteBufferToFileEnd;
        }
        break;
    }

    size_t bytesWritten = fwrite(px.buffer, 1, px.size, file);
    if (bytesWritten != px.size)
    {
        LOG_ERR("Expected to write %u bytes, did %u", px.size, bytesWritten);
        retval = 1;
        goto WriteBufferToFileEnd;
    }

WriteBufferToFileEnd:
    DeallocatePixelDataBuffer(&px);

    if (file)
    {
        fclose(file);
    }

    return retval;
}
