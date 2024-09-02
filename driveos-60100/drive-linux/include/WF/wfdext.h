/* Copyright (c) 2009 The Khronos Group Inc.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */

/*! \ingroup wfd
 *  \file wfdext.h
 *
 *  \brief Header file for defining available extension on a platform
 *
 *  See OpenWF specification for usage of this header file.
 */

#ifndef WFDEXT_H_
#define WFDEXT_H_

#ifdef __cplusplus
extern "C" {
#endif


#define WFD_VENDOR_INDEX        (0)
#define WFD_RENDERER_INDEX      (1)
#define WFD_VERSION_INDEX       (2)

#define WFD_PORT_PERMISSIBLE_LATENCY_MS_NV 0x7635

#ifdef WFD_WFDEXT_PROTOTYPES
WFD_API_CALL void WFD_APIENTRY
    wfdBindSourceToPipelineWithTimestampNV(const WFDDevice device,
                                           const WFDPipeline pipeline,
                                           const WFDSource source,
                                           const WFDTransition transition,
                                           const WFDRect *const region,
                                           const WFDuint64 timestamp);
#endif /* WFD_WFDEXT_PROTOTYPES */

typedef void (WFD_APIENTRY PFNWFDBINDSOURCETOPIPELINEWITHTIMESTAMPNV) (const WFDDevice device,
                                                                       const WFDPipeline pipeline,
                                                                       const WFDSource source,
                                                                       const WFDTransition transition,
                                                                       const WFDRect *const region,
                                                                       const WFDuint64 timestamp);

#ifdef WFD_WFDEXT_PROTOTYPES
WFD_API_CALL WFDErrorCode WFD_APIENTRY
    wfdDeviceSetSafeStateNV(const WFDDevice device);
#endif /* WFD_WFDEXT_PROTOTYPES */

typedef void (WFD_APIENTRY PFNWFDDEVICESETSAFESTATENV) (const WFDDevice device);

struct NvSciBufObjRefRec;
typedef struct NvSciBufObjRefRec* NvSciBufObj;
struct NvSciBufAttrListRec;
typedef struct NvSciBufAttrListRec* NvSciBufAttrList;
#ifdef WFD_WFDEXT_PROTOTYPES
WFD_API_CALL WFDSource WFD_APIENTRY
    wfdCreateSourceFromNvSciBufNVX(const WFDDevice device, const WFDPipeline pipeline,
                                   NvSciBufObj *const bufObj);
WFD_API_CALL WFDErrorCode WFD_APIENTRY
    wfdNvSciBufSetDisplayAttributesNVX(NvSciBufAttrList *const attrList);
#endif /* WFD_WFDEXT_PROTOTYPES */
typedef WFDSource (WFD_APIENTRY PFNWFDCREATESOURCEFROMNVSCIBUFNVX) (const WFDDevice device,
                   const WFDPipeline pipeline, NvSciBufObj *const bufObj);
typedef WFDErrorCode (WFD_APIENTRY PFNWFDNVSCIBUFSETDISPLAYATTRIBUTESNVX) (NvSciBufAttrList *const attrList);

struct NvSciSyncObjRec;
typedef struct NvSciSyncObjRec* NvSciSyncObj;
struct NvSciSyncAttrListRec;
typedef struct NvSciSyncAttrListRec* NvSciSyncAttrList;
typedef struct NvSciSyncFence NvSciSyncFence;
#ifdef WFD_WFDEXT_PROTOTYPES
WFD_API_CALL WFDErrorCode WFD_APIENTRY
    wfdNvSciSyncSetWaiterAttributesNVX(const NvSciSyncAttrList *const attrList);

WFD_API_CALL WFDErrorCode WFD_APIENTRY
    wfdNvSciSyncSetSignalerAttributesNVX(const NvSciSyncAttrList *const attrList);

WFD_API_CALL WFDErrorCode WFD_APIENTRY
    wfdBindNvSciSyncFenceToSourceNVX(const WFDDevice device, const WFDSource source,
                                     const NvSciSyncFence *const fence);

WFD_API_CALL WFDErrorCode WFD_APIENTRY
    wfdRegisterPostFlipNvSciSyncObjNVX(const WFDDevice device, const NvSciSyncObj *const obj);

WFD_API_CALL void WFD_APIENTRY
    wfdDeviceCommitWithNvSciSyncFenceNVX(const WFDDevice device,
                                         const WFDCommitType type,
                                         const WFDHandle handle,
                                         NvSciSyncFence *const fence);
#endif /* WFD_WFDEXT_PROTOTYPES */
typedef WFDErrorCode (WFD_APIENTRY PFNWFDNVSCISYNCSETSIGNALERATTRIBUTESNVX)
                     (NvSciSyncAttrList *const attrList);

typedef WFDErrorCode (WFD_APIENTRY PFNWFDNVSCISYNCSETWAITERATTRIBUTESNVX)
                     (NvSciSyncAttrList *const attrList);

typedef WFDErrorCode (WFD_APIENTRY PFNWFDBINDNVSCISYNCFENCETOSOURCENVX)
                     (const WFDDevice device, const WFDSource source, const NvSciSyncFence *const fence);

typedef WFDErrorCode (WFD_APIENTRY PFNWFDREGISTERPOSTFLIPNVSCISYNCOBJNVX)
                     (const WFDDevice device, const NvSciSyncObj *const obj);

typedef void (WFD_APIENTRY PFNWFDDEVICECOMMITWITHNVSCISYNCFENCENVX)
             (const WFDDevice device, const WFDCommitType type,
              const WFDHandle handle, NvSciSyncFence *const fence);

#define WFD_PIPELINE_POSTFENCE_SCANOUT_BEGIN_NVX 0x7731

#define WFD_PIPELINE_COMMIT_NON_BLOCKING_NVX 0x7730

#define WFD_PORT_MODE_H_TOTAL_NVX   0x7608
#define WFD_PORT_MODE_V_TOTAL_NVX   0x7609

#define WFD_PORT_DETECT_FROZEN_FRAME_NV 0x7636

#ifdef NV_STANDARD_BUILD

struct NvRmSurfaceRec;
typedef struct NvRmSurfaceRec NvRmSurface;

#ifdef WFD_WFDEXT_PROTOTYPES
WFD_API_CALL WFDSource WFD_APIENTRY
    wfdCreateSourceFromNvRmSurfaceNVX(const WFDDevice device, const WFDPipeline pipeline,
                                      const WFDint numSurfaces, const NvRmSurface *const surfaces);
#endif /* WFD_WFDEXT_PROTOTYPES */

typedef WFDSource (WFD_APIENTRY PFNWFDCREATESOURCEFROMNVRMSURFACENVX) (const WFDDevice device,
                   const WFDPipeline pipeline, const WFDint numSurfaces, const NvRmSurface *const surfaces);

struct NvRmFenceRec;
typedef struct NvRmFenceRec NvRmFence;

#ifdef WFD_WFDEXT_PROTOTYPES
WFD_API_CALL WFDErrorCode WFD_APIENTRY
    wfdBindNvRmFenceToSourceNVX(const WFDDevice device, const WFDSource source,
                                const NvRmFence *const fence);

WFD_API_CALL void WFD_APIENTRY
    wfdDeviceCommitWithNvRmFenceNVX(const WFDDevice device,
                                    const WFDCommitType type,
                                    const WFDHandle handle,
                                    NvRmFence *const fence);
#endif /* WFD_WFDEXT_PROTOTYPES */

typedef WFDErrorCode (WFD_APIENTRY PFNWFDBINDNVRMFENCETOSOURCENVX) (const WFDDevice device,
                      const WFDSource source, const NvRmFence *const fence);
typedef void (WFD_APIENTRY PFNWFDDEVICECOMMITWITHNVRMFENCENVX) (const WFDDevice device,
              const WFDCommitType type, const WFDHandle handle, NvRmFence *const fence);

#if defined(__QNX__)
#define WFD_PORT_CBABC_MODE_QNX 0x7670
typedef enum
{   WFD_PORT_CBABC_MODE_NONE_QNX   = 0x7671,
    WFD_PORT_CBABC_MODE_VIDEO_QNX  = 0x7672,
    WFD_PORT_CBABC_MODE_UI_QNX     = 0x7673,
    WFD_PORT_CBABC_MODE_PHOTO_QNX  = 0x7674,
    WFD_PORT_CBABC_MODE_32BIT_QNX  = 0x7FFFFFFF
} WFDPortCBABCModeQNX;

#define WFD_PIPELINE_BRIGHTNESS_QNX     0x7750
#define WFD_PIPELINE_CONTRAST_QNX       0x7751
#define WFD_PIPELINE_HUE_QNX            0x7752
#define WFD_PIPELINE_SATURATION_QNX     0x7753

#define WFD_PORT_MODE_ASPECT_RATIO_QNX  0x7606
#define WFD_PORT_MODE_PREFERRED_QNX     0x7607

#ifdef WFD_WFDEXT_PROTOTYPES
WFD_API_CALL WFDErrorCode WFD_APIENTRY
    wfdReadPixelsFromSourceQNX(WFDDevice device, WFDSource source, WFDint x, WFDint y,
                               WFDint width, WFDint height, WFDint format, void *data);
#endif /* WFD_WFDEXT_PROTOTYPES */
typedef WFDErrorCode (WFD_APIENTRY PFNWFDREADPIXELSFROMSOURCEQNX) (WFDDevice device, WFDSource source,
                      WFDint x, WFDint y, WFDint width,
                      WFDint height, WFDint format, void *data);

#define WFD_USAGE_DISPLAY_QNX      (1 << 0)
#define WFD_USAGE_READ_QNX         (1 << 1)
#define WFD_USAGE_WRITE_QNX        (1 << 2)
#define WFD_USAGE_NATIVE_QNX       (1 << 3)
#define WFD_USAGE_OPENGL_ES1_QNX   (1 << 4)
#define WFD_USAGE_OPENGL_ES2_QNX   (1 << 5)
#define WFD_USAGE_OPENGL_ES3_QNX   (1 << 11)
#define WFD_USAGE_OPENVG_QNX       (1 << 6)
#define WFD_USAGE_VIDEO_QNX        (1 << 7)
#define WFD_USAGE_CAPTURE_QNX      (1 << 8)
#define WFD_USAGE_ROTATION_QNX     (1 << 9)
#define WFD_USAGE_OVERLAY_QNX      (1 << 10)
#define WFD_USAGE_WRITEBACK_QNX    (1 << 31)
#define WFD_FORMAT_BYTE_QNX              1
#define WFD_FORMAT_RGBA4444_QNX          2
#define WFD_FORMAT_RGBX4444_QNX          3
#define WFD_FORMAT_RGBA5551_QNX          4
#define WFD_FORMAT_RGBX5551_QNX          5
#define WFD_FORMAT_RGB565_QNX            6
#define WFD_FORMAT_RGB888_QNX            7
#define WFD_FORMAT_RGBA8888_QNX          8
#define WFD_FORMAT_RGBX8888_QNX          9
#define WFD_FORMAT_YVU9_QNX             10
#define WFD_FORMAT_YUV420_QNX           11
#define WFD_FORMAT_NV12_QNX             12
#define WFD_FORMAT_YV12_QNX             13
#define WFD_FORMAT_UYVY_QNX             14
#define WFD_FORMAT_YUY2_QNX             15
#define WFD_FORMAT_YVYU_QNX             16
#define WFD_FORMAT_V422_QNX             17
#define WFD_FORMAT_AYUV_QNX             18
#define WFD_FORMAT_NV12_QC_SUPERTILE    ((1 << 16) | WFD_FORMAT_NV12_QNX)
#define WFD_FORMAT_NV12_QC_32M4KA       ((2 << 16) | WFD_FORMAT_NV12_QNX)
#ifdef WFD_WFDEXT_PROTOTYPES
WFD_API_CALL WFDErrorCode WFD_APIENTRY
    wfdCreateWFDEGLImagesQNX(WFDDevice device, WFDint width, WFDint height, WFDint format, WFDint usage, WFDint count, WFDEGLImage *images);
WFD_API_CALL WFDErrorCode WFD_APIENTRY
    wfdDestroyWFDEGLImagesQNX(WFDDevice device, WFDint count, WFDEGLImage *images);
#endif /* WFD_WFDEXT_PROTOTYPES */
typedef WFDErrorCode (WFD_APIENTRY PFNWFDCREATEWFDEGLIMAGESQNX) (WFDDevice device, WFDint width, WFDint height, WFDint usage, WFDint count, WFDEGLImage *images);
typedef WFDErrorCode (WFD_APIENTRY PFNWFDDESTROYWFDEGLIMAGESQNX) (WFDDevice device, WFDint count, WFDEGLImage *images);

#endif // defined(__QNX__)
#endif /* !NV_STANDARAD_BUILD */

#ifdef __cplusplus
}
#endif


#endif /* WFDEXT_H_ */
