/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file
 * @brief <b> NVIDIA Media Interface: Image JPEG Decode Processing (IJDP)</b>
 *
 * @b Description: This file contains the \ref 6x_image_jpeg_decode_api "Image
 * JPEG Decode Processing API".
 */

#ifndef NVMEDIA_IJPD_H
#define NVMEDIA_IJPD_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "nvmedia_core.h"
#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvmedia_common_encode_decode.h"

/**
 * @defgroup 6x_image_jpeg_decode_api Image JPEG Decoder
 *
 * The NvMediaIJPD object takes a JPEG bitstream and decompress it to image data.
 *
 * @ingroup 6x_nvmedia_image_top
 * @{
 */

/** @brief Major version number */
#define NVMEDIA_IJPD_VERSION_MAJOR   1
/** @brief Minor version number */
#define NVMEDIA_IJPD_VERSION_MINOR   0
/** @brief Patch version number */
#define NVMEDIA_IJPD_VERSION_PATCH   0

/**
 * Specifies the maximum number of times NvMediaIJPDInsertPreNvSciSyncFence()
 * can be called before each call to NvMediaIJPDFeedFrame().
 */
#define NVMEDIA_IJPD_MAX_PRENVSCISYNCFENCES        (16U)

/**
 * \hideinitializer
 * @brief JPEG decode set alpha
 */
#define NVMEDIA_JPEG_DEC_ATTRIBUTE_ALPHA_VALUE     (1 << 0)
/**
 * \hideinitializer
 * @brief JPEG decode set color standard
 */
#define NVMEDIA_JPEG_DEC_ATTRIBUTE_COLOR_STANDARD  (1 << 1)
/**
 * \hideinitializer
 * @brief JPEG decode render flag rotate 0
 */
#define NVMEDIA_IJPD_RENDER_FLAG_ROTATE_0          0
/**
 * \hideinitializer
 * @brief JPEG decode render flag rotate 90
 */
#define NVMEDIA_IJPD_RENDER_FLAG_ROTATE_90         1
/**
 * \hideinitializer
 * @brief JPEG decode render flag rotate 180
 */
#define NVMEDIA_IJPD_RENDER_FLAG_ROTATE_180        2
/**
 * \hideinitializer
 * @brief JPEG decode render flag rotate 270
 */
#define NVMEDIA_IJPD_RENDER_FLAG_ROTATE_270        3
/**
 * \hideinitializer
 * @brief JPEG decode render flag flip horizontal
 */
#define NVMEDIA_IJPD_RENDER_FLAG_FLIP_HORIZONTAL   (1 << 2)
/**
 * \hideinitializer
 * @brief JPEG decode render flag flip vertical
 */
#define NVMEDIA_IJPD_RENDER_FLAG_FLIP_VERTICAL     (1 << 3)
/**
 * \hideinitializer
 * @brief JPEG decode max number of app markers supported
 */
#define NVMEDIA_MAX_JPEG_APP_MARKERS               16

/** @brief Defines color standards.
 */
typedef enum {
    /** \hideinitializer @brief Specifies ITU BT.601 color standard. */
    NVMEDIA_IJPD_COLOR_STANDARD_ITUR_BT_601,
    /** \hideinitializer @brief Specifies ITU BT.709 color standard. */
    NVMEDIA_IJPD_COLOR_STANDARD_ITUR_BT_709,
    /** \hideinitializer @brief Specifies SMTE 240M color standard. */
    NVMEDIA_IJPD_COLOR_STANDARD_SMPTE_240M,
    /** \hideinitializer @brief Specifies ITU BT.601 color standard extended
     range. */
    NVMEDIA_IJPD_COLOR_STANDARD_ITUR_BT_601_ER,
    /** \hideinitializer @brief Specifies ITU BT.709 color standard extended
     range. */
    NVMEDIA_IJPD_COLOR_STANDARD_ITUR_BT_709_ER
} NvMediaIJPDColorStandard;

/**
 * Holds image JPEG decoder attributes.
 */
typedef struct {
/** @brief Specifies the color standard, defined in \ref NvMediaIJPDColorStandard.
 *  The corresponding attribute mask is \ref NVMEDIA_JPEG_DEC_ATTRIBUTE_COLOR_STANDARD
 */
    NvMediaIJPDColorStandard colorStandard;

/** @brief Specifies the alpha value. It can take one of 0 or 0xFF.
 *  The corresponding attribute mask is \ref NVMEDIA_JPEG_DEC_ATTRIBUTE_ALPHA_VALUE
 */
    uint32_t alphaValue;
} NVMEDIAJPEGDecAttributes;

/**
 * Holds image JPEG decoder marker Info.
 */
typedef struct {
/** @brief Specifies the App info marker.
 */
  uint16_t marker;
/** @brief Specifies the App info marker length.
*/
  uint16_t len;
/** @brief Specifies the App info marker data.
*/
  void    *pMarker;
} NvMediaJPEGAppMarkerInfo;

/**
 * Holds image JPEG decoder stream information.
 */
typedef struct {
/** @brief Specifies the stream-encoded width, in pixels.
 */
  uint16_t width;
/** @brief Specifies the stream-encoded height, in pixels.
 */
  uint16_t height;
/** @brief Specifies whether partial acceleration is needed for the stream.
 */
  uint8_t  partialAccel;
/** @brief Specifies the number of App merkers in the stream.
 */
  uint8_t  num_app_markers;
/** @brief Specifies the marker info.
 */
  NvMediaJPEGAppMarkerInfo appMarkerInfo[NVMEDIA_MAX_JPEG_APP_MARKERS];
} NVMEDIAJPEGDecInfo;

/**
 * @brief An opaque NvMediaIJPD object created by \ref NvMediaIJPDCreate.
 */
typedef struct NvMediaIJPD NvMediaIJPD;

/**
 * @brief Retrieves the version information for the NvMedia IJPD library.
 *
 * @pre None
 * @post None
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @param[in] version A pointer to a \ref NvMediaVersion structure
 *                    of the client.
 * @return ::NvMediaStatus The status of the operation.
 *
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if the pointer is invalid.
 */
NvMediaStatus
NvMediaIJPDGetVersion(
    NvMediaVersion *version
);

/**
 * Creates a JPEG decoder object capable of decoding a JPEG stream into
 * an image surface.
 *
 * @pre NvMediaIJPDGetVersion()
 * @pre NvMediaIJPDFillNvSciBufAttrList()
 * @post NvMediaIJPD object is created
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @param[in] maxWidth The maximum width of output surface to support.
 *      you can use NvMediaIJPDResize() to enlarge this limit for an existing decoder.
 * @param[in] maxHeight The maximum height of output surface to support. You can use
 * \c %NvMediaIJPDResize() to enlarge this limit for an existing decoder.
 * @param[in] maxBitstreamBytes The maximum JPEG bitstream size in bytes to support.
 *      Use \c %NvMediaIJPDResize() to enlarge this limit for an existing decoder.
 * @param[in] supportPartialAccel Indicates that the JPEG decode object supports
 *      partial acceleration.
 *      \n If it does, set this argument to the character '1' (true).
 *      \n If it does not, set this argument to the character '0' (false).
 * @param[in] instanceId The ID of the engine instance.
 *      The following instances are supported:
 *      - ::NVMEDIA_JPEG_INSTANCE_0
 *      - ::NVMEDIA_JPEG_INSTANCE_1     [Supported only on T23X]
 *      - ::NVMEDIA_JPEG_INSTANCE_AUTO  [Supported only on T23X]
 * @retval NvMediaIJPD The new image JPEG decoder handle or NULL if unsuccessful.
 */
NvMediaIJPD *
NvMediaIJPDCreate(
    uint16_t maxWidth,
    uint16_t maxHeight,
    uint32_t maxBitstreamBytes,
    bool supportPartialAccel,
    NvMediaJPEGInstanceId instanceId
);

/**
 * Destroys an NvMedia image JPEG decoder.
 *
 * @pre NvMediaIJPDUnregisterNvSciBufObj()
 * @pre NvMediaIJPDUnregisterNvSciSyncObj()
 * @post NvMediaIJPD object is destroyed
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * @param[in] decoder A pointer to the JPEG decoder to destroy.
 */
void NvMediaIJPDDestroy(NvMediaIJPD *decoder);

/**
 * @brief Resizes an existing image JPEG decoder.
 *
 * @pre NvMediaIJPDCreate()
 * @post NvMediaIJPD object is updated as specified
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @param[in] decoder A pointer to the JPEG decoder to use.
 * @param[in] maxWidth  The new maximum width of output surface to support.
 * @param[in] maxHeight The new maximum height of output surface to support.
 * @param[in] maxBitstreamBytes The new maximum JPEG bitstream size in bytes to support.
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is invalid.
 * - ::NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIJPDResize (
   NvMediaIJPD *decoder,
   uint16_t maxWidth,
   uint16_t maxHeight,
   uint32_t maxBitstreamBytes
);

/**
 * @brief Sets attributes of an existing image JPEG decoder.
 *
 * @pre NvMediaIJPDCreate()
 * @post Specified attribute is set
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @param[in] decoder  A pointer to the JPEG decoder to use.
 * @param[in] attributeMask An attribute mask.
 *  Supported mask are:
 * - ::NVMEDIA_JPEG_DEC_ATTRIBUTE_ALPHA_VALUE
 * - ::NVMEDIA_JPEG_DEC_ATTRIBUTE_COLOR_STANDARD
 * @param[in] attributes Attributes data.
 * Supported attribute structures:
 * - ::NVMEDIAJPEGDecAttributes
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL.
 * - ::NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIJPDSetAttributes(
   const NvMediaIJPD *decoder,
   uint32_t attributeMask,
   const void *attributes
);

/**
 * @brief A helper function that determines whether the JPEG decoder HW engine can decode
 * the input JPEG stream. Possible outcomes are:
 *
 * * __Decode possible__. If JPEG decoder supports decode of this stream,
 * this function returns \ref NVMEDIA_STATUS_OK and the \ref NVMEDIAJPEGDecInfo
 * info will be filled out. This function also determines
 * whether you must allocate the \ref NvMediaIJPD object when you call NvMediaIJPDCreate().
 * You specify that object with the \c %NvMediaIJPDCreate() \a supportPartialAccel parameter.
 *
 * * __Decode not possible__. If JPEG decoder cannot decode this stream,
 * this function returns \ref NVMEDIA_STATUS_NOT_SUPPORTED.
 *
 * @pre NvMediaIJPDCreate()
 * @post NVMEDIAJPEGDecInfo is populated with required information
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @param[in,out] info A pointer to the information to be filled.
 * @param[in] numBitstreamBuffers The number of bitstream buffers.
 * @param[in] bitstreams The bitstream buffer.
 *
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL.
 * - ::NVMEDIA_STATUS_NOT_SUPPORTED if stream not supported
 * - ::NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIJPDGetInfo (
   NVMEDIAJPEGDecInfo *info,
   uint32_t numBitstreamBuffers,
   const NvMediaBitstreamBuffer *bitstreams
);

/**
 * Decodes a JPEG image. The decode pipeline produces a
 * result equivalent to the following sequence:
 *
 * 1. Decodes the full JPEG image.
 * 2. Downscales the 8x8 block padded image by the \a downscaleLog2 factor.
 *    That is, a "width" by "height" JPEG is downscaled to:
 *
 *        ((width + 7) & ~7) >> downscaleLog2
 *
 *    by
 *
 *        ((height + 7) & ~7) >> downscaleLog2
 *
 * 3. From the downscaled image, removes the rectangle described by *srcRect*
 *    and optionally (a) mirrors the image horizontally and/or vertically and/or
 *    (b) rotates the image.
 * 4. Scales the transformed source rectangle to the *dstRect*
 *    on the output surface.
 *
 * @par Specifying Dimensions
 *
 * The JPEG decoder object must have *maxWidth* and *maxHeight* values
 * that are greater than or equal to the post-downscale JPEG image.
 * Additionally, it must have a *maxBitstreamBytes* value that is greater than
 * or equal to the total number of bytes in the bitstream buffers. You
 * set these values when you create the JPEG decoder object
 * with NvMediaIJPDCreate().
 * Alternatively, you can user NvMediaIJPDResize() to change the dimensions
 * of an existing JPEG decoder object.
 *
 * If the JPEG decoder object has inadequate dimensions, \c %NvMediaIJPDRender()
 * returns \ref NVMEDIA_STATUS_INSUFFICIENT_BUFFERING.
 *
 *
 * @par Supporting Partial Acceleration
 *
 * If the JPEG stream requires partial acceleration, created the JPEG
 * decoder object with *supportPartialAccel* set to '1'.
 * Otherwise,  the function returns \ref NVMEDIA_STATUS_BAD_PARAMETER.
 *
 * Use NvMediaIJPDGetInfo() to determine whether a
 * stream requires paritialAccel.
 *
 * @par Determining Supported JPEG Streams
 *
 * If the JPEG stream is not supported, the function returns
 * \ref NVMEDIA_STATUS_NOT_SUPPORTED.
 *
 * Use @c %NvMediaIJPDGetInfo() to determine whether a
 * stream is unsupported.
 *
 * @note \c %NvMediaIJPDRender() with the NVJPG 1.0 codec does not support rotation.
 *
 * @pre NvMediaIJPDRegisterNvSciBufObj()
 * @pre NvMediaIJPDRegisterNvSciSyncObj()
 * @pre NvMediaIJPDSetNvSciSyncObjforEOF()
 * @pre NvMediaIJPDInsertPreNvSciSyncFence()
 * @post JPEG decoding task is submitted
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] decoder  A pointer to the JPEG decoder to use.
 * @param[out] target NvSciBufObj that contains the decoded content, allocated
 *      with a call to \ref NvSciBufObjAlloc. Supported surface format attributes: \n
 *      Buffer Type: NvSciBufType_Image \n
 *      Surface Type: RGBA \n
 *      Bit Depth: 8 \n
 *      Layout: NvSciBufImage_PitchLinearType \n
 *      Scan Type: NvSciBufScan_ProgressiveType \n
 *      Plane base address alignment: 256 \n
 * @param[in] srcRect The source rectangle. The rectangle from the post-downscaled image to be
 *      transformed and scaled to the *dstRect*. You can achieve horizontal and/or
 *      vertical mirroring by swapping the left-right and/or top-bottom
 *      coordinates. If NULL, the full post-downscaled surface is implied.
 * @param[in] dstRect The destination rectangle on the output surface. If NULL, a
 *      rectangle the full size of the output surface is implied.
 * @param[in] downscaleLog2 A value clamped between 0 and 3 inclusive, gives downscale factors
 *      of 1 to 8.
 * @param[in] numBitstreamBuffers The number of bitstream buffers.
 * @param[in] bitstreams The bitstream buffer. \c %NvMediaIJPDRender()
 *     copies the data out
 *     of these buffers so the caller is free to reuse them as soon as
 *     \c %NvMediaIJPDRender() returns.
 * @param[in] flags Flags that specify a clockwise rotation of the source in
 *     degrees and horizontal and vertical flipping.
 *     If both are specified, the image is flipped before it is rotated.
 *     You can set the *flags* argument to any one of the following:
 *     \li NVMEDIA_RENDER_FLAG_ROTATE_0
 *     \li NVMEDIA_RENDER_FLAG_ROTATE_90
 *     \li NVMEDIA_RENDER_FLAG_ROTATE_180
 *     \li NVMEDIA_RENDER_FLAG_ROTATE_270
 *
 *     Additionally, you can use the bitwise OR operation to
 *     apply either or both of the following:
 *     \li NVMEDIA_RENDER_FLAG_FLIP_HORIZONTAL
 *     \li NVMEDIA_RENDER_FLAG_FLIP_VERTICAL
 * @param[in] instanceId The ID of the engine instance.
 *      The following instances are supported:
 *      - ::NVMEDIA_JPEG_INSTANCE_0
 *      - ::NVMEDIA_JPEG_INSTANCE_1     [Supported only on T23X]
 *      - ::NVMEDIA_JPEG_INSTANCE_AUTO  [Supported only on T23X]
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL.
 * - ::NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIJPDRender(
   const NvMediaIJPD *decoder,
   NvSciBufObj target,
   const NvMediaRect *srcRect,
   const NvMediaRect *dstRect,
   uint8_t downscaleLog2,
   uint32_t numBitstreamBuffers,
   const NvMediaBitstreamBuffer *bitstreams,
   uint32_t flags,
   NvMediaJPEGInstanceId instanceId
);

/**
 * Decodes a JPEG image into YUV format.
 * This function is similar to NvMediaIJPDRender() except that
 * the output surface is in YUV format, not RGBA format. Also, clipping
 * and scaling (other than downscaleLog2 scaling) are not supported, so
 * there are no source or destination rectangle parameters.
 *
 *   @note \c %NvMediaIJPDRenderYUV()
 *       with the NVJPG 1.0 codec has the following limitations:
 *   \li It supports chroma subsample conversion to 420 and 420H from any input
 *       format except 400.
 *   \li It does not simultaneously support downscaleLog2 and subsample conversion.
 *
 * @pre NvMediaIJPDRegisterNvSciBufObj()
 * @pre NvMediaIJPDRegisterNvSciSyncObj()
 * @pre NvMediaIJPDSetNvSciSyncObjforEOF()
 * @pre NvMediaIJPDInsertPreNvSciSyncFence()
 * @post JPEG decoding task is submitted
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] decoder  A pointer to the JPEG decoder to use.
 * @param[out] target NvSciBufObj that contains the decoded content, allocated
 *      with a call to \ref NvSciBufObjAlloc. Supported surface format attributes: \n
 *      Buffer Type: NvSciBufType_Image \n
 *      Sub-sampling type: YUV420, YUV422, YUV444 (planar and semi-planar) \n
 *      Bit Depth: 8 \n
 *      Layout: NvSciBufImage_PitchLinearType \n
 *      Scan Type: NvSciBufScan_ProgressiveType \n
 *      Plane base address alignment: 256 \n
 * @param[in] downscaleLog2 A value between 0 and 3 inclusive that gives downscale
 *       factors of 1 to 8.
 * @param[in] numBitstreamBuffers The number of bitstream buffers.
 * @param[in] bitstreams The bitstream buffer. \c %NvMediaIJPDRenderYUV()
 *       copies the data out
 *       of these buffers so the caller is free to reuse them as soon as
 *       \c %NvMediaIJPDRenderYUV() returns.
 * @param[in] flags Flags that specify a clockwise rotation of the source in degrees and horizontal
 *       and vertical flipping. If both are specified, flipping is performed before rotating.
 *     You can set the \a flags argument to any one of the following:
 *     \li NVMEDIA_RENDER_FLAG_ROTATE_0
 *     \li NVMEDIA_RENDER_FLAG_ROTATE_90
 *     \li NVMEDIA_RENDER_FLAG_ROTATE_180
 *     \li NVMEDIA_RENDER_FLAG_ROTATE_270
 *
 *     Additionally, you can use the bitwise OR operation to
 *     apply either or both of the following:
 *     \li NVMEDIA_RENDER_FLAG_FLIP_HORIZONTAL
 *     \li NVMEDIA_RENDER_FLAG_FLIP_VERTICAL
 * @param[in] instanceId The ID of the engine instance.
 *      The following instances are supported:
 *      - ::NVMEDIA_JPEG_INSTANCE_0
 *      - ::NVMEDIA_JPEG_INSTANCE_1     [Supported only on T23X]
 *      - ::NVMEDIA_JPEG_INSTANCE_AUTO  [Supported only on T23X]
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL.
 * - ::NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIJPDRenderYUV(
   const NvMediaIJPD *decoder,
   NvSciBufObj target,
   uint8_t downscaleLog2,
   uint32_t numBitstreamBuffers,
   const NvMediaBitstreamBuffer *bitstreams,
   uint32_t flags,
   NvMediaJPEGInstanceId instanceId
);

/**
 * @brief Registers \ref NvSciBufObj for use with a NvMediaIJPD handle.
 * NvMediaIJPD handle maintains a record of all the objects registered using this
 * API and only the registered NvSciBufObj handles are accepted when submitted
 * for decoding via \ref NvMediaIJPDRender. Even duplicated NvSciBufObj
 * objects need to be registered using this API prior.
 *
 * This needs to be used in tandem with NvMediaIJPDUnregisterNvSciBufObj(). The
 * pair of APIs for registering and unregistering NvSciBufObj are optional, but
 * it is highly recommended to use them as they ensure deterministic execution
 * of NvMediaIJPDRender().
 *
 * To ensure deterministic execution time of \ref NvMediaIJPDRender API:
 * - \ref NvMediaIJPDRegisterNvSciBufObj must be called for every input
 *   \ref NvSciBufObj that will be used with NvMediaIJPD
 * - All \ref NvMediaIJPDRegisterNvSciBufObj calls must be made before first
 *   \ref NvMediaIJPDRender API call.
 *
 * Registration of the buffer (output) is always with read-write permissions.
 *
 * Maximum of 32 \ref NvSciBufObj handles can be registered.
 *
 * @note This API is currently not supported and can be ignored
 *
 * @pre NvMediaIJPDCreate()
 * @pre NvMediaIJPDRegisterNvSciSyncObj()
 * @post NvSciBufObj is registered with NvMediaIJPD object
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @param[in] decoder A pointer to the NvMediaIJPD object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] bufObj NvSciBufObj object
 *      \inputrange A valid NvSciBufObj
 *
 * @return ::NvMediaStatus, the completion status of operation:
 * - ::NVMEDIA_STATUS_NOT_SUPPORTED if API is not functionally supported
 *
 **/
NvMediaStatus
NvMediaIJPDRegisterNvSciBufObj(
    const NvMediaIJPD   *decoder,
    NvSciBufObj         bufObj
);

/**
 * @brief Un-registers \ref NvSciBufObj which was previously registered with
 * \ref NvMediaIJPD using NvMediaIJPDRegisterNvSciBufObj().
 *
 * For all \ref NvSciBufObj handles registered with NvMediaIJPD using
 * \ref NvMediaIJPDRegisterNvSciBufObj API, \ref NvMediaIJPDUnregisterNvSciBufObj
 * must be called before calling \ref NvMediaIJPDDestroy API.
 * For unregistration to succeed, it should be ensured that none of the
 * submitted tasks on the bufObj are pending prior to calling
 * %NvMediaIJPDUnregisterNvSciBufObj(). In order to ensure this,
 * %NvMediaIJPDUnregisterNvSciSyncObj() should be called prior to this API on
 * all registered NvSciSyncObj. Post this NvMediaIJPDUnregisterNvSciBufObj() can
 * be successfully called on a valid NvSciBufObj.
 *
 * This needs to be used in tandem with NvMediaIJPDRegisterNvSciBufObj(). The
 * pair of APIs for registering and unregistering NvSciBufObj are optional, but
 * it is highly recommended to use them as they ensure deterministic execution
 * of NvMediaIJPDRender().
 *
 * To ensure deterministic execution time of \ref NvMediaIJPDRender API:
 * - \ref NvMediaIJPDUnregisterNvSciBufObj should be called only after the last
 *   \ref NvMediaIJPDRender call
 *
 * @note This API is currently not supported and can be ignored
 *
 * @pre NvMediaIJPDUnregisterNvSciSyncObj() [verify that processing is complete]
 * @post NvSciBufObj is un-registered from NvMediaIJPD object
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * @param[in] decoder A pointer to the NvMediaIJPD object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] bufObj NvSciBufObj object
 *      \inputrange A valid NvSciBufObj
 *
 * @return ::NvMediaStatus, the completion status of operation:
 * - ::NVMEDIA_STATUS_NOT_SUPPORTED if API is not functionally supported
 **/
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIJPDUnregisterNvSciBufObj(
    const NvMediaIJPD    *decoder,
    NvSciBufObj          bufObj
);

/**
 * @brief Fills the NvMediaIJPD specific NvSciBuf attributes which than then be
 * used to allocate an \ref NvSciBufObj that NvMediaIJPD can consume.
 *
 * This function assumes that @a attrlist is a valid \ref NvSciBufAttrList
 * created by the caller by a call to \ref NvSciBufAttrListCreate.
 *
 * @pre NvMediaIJPDGetVersion()
 * @post NvSciBufAttrList populated with NvMediaIJPD specific NvSciBuf
 *      attributes. The caller can then set attributes specific to the type of
 *      surface, reconcile attribute lists and allocate an NvSciBufObj.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @param[in] instanceId The ID of the engine instance.
 *      \inputrange The following instances are supported:
 *      - ::NVMEDIA_JPEG_INSTANCE_0
 *      - ::NVMEDIA_JPEG_INSTANCE_1     [Supported only on T23X]
 *      - ::NVMEDIA_JPEG_INSTANCE_AUTO  [Supported only on T23X]
 * @param[out] attrlist An %NvSciBufAttrList where NvMediaIJPD places the
 *      NvSciBuf attributes.
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a attrlist is NULL
 */
NvMediaStatus
NvMediaIJPDFillNvSciBufAttrList(
    NvMediaJPEGInstanceId     instanceId,
    NvSciBufAttrList          attrlist
);

/**
 * @brief Fills the NvMediaIJPD specific NvSciSync attributes.
 *
 * This function assumes that @a attrlist is a valid \ref NvSciSyncAttrList.
 *
 * This function sets the public attribute:
 * - \ref NvSciSyncAttrKey_RequiredPerm
 *
 * The application must not set this attribute.
 *
 * @pre NvMediaIJPDCreate()
 * @post NvSciSyncAttrList populated with NvMediaIJPD specific NvSciSync
 *        attributes
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] decoder A pointer to the NvMediaIJPD object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[out] attrlist A pointer to an %NvSciSyncAttrList structure where
 *                NvMedia places NvSciSync attributes.
 * @param[in] clienttype Indicates whether the NvSciSyncAttrList requested for
 *                an %NvMediaIJPD signaler or an %NvMediaIJPD waiter.
 *      \inputrange Entries in \ref NvMediaNvSciSyncClientType enumeration
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a attrlist is NULL,
 *         or any of the public attributes listed above are already set.
 * - ::NVMEDIA_STATUS_OUT_OF_MEMORY if there is not enough
 *         memory for the requested operation.
 */
NvMediaStatus
NvMediaIJPDFillNvSciSyncAttrList(
    const NvMediaIJPD           *decoder,
    NvSciSyncAttrList          attrlist,
    NvMediaNvSciSyncClientType clienttype
);

/**
 * @brief Registers an \ref NvSciSyncObj with NvMediaIJPD.
 *
 * Every NvSciSyncObj (even duplicate objects) used by %NvMediaIJPD
 * must be registered by a call to this function before it is used.
 * Only the exact same registered NvSciSyncObj can be passed to
 * NvMediaIJPDSetNvSciSyncObjforEOF(), NvMediaIJPDGetEOFNvSciSyncFence(), or
 * NvMediaIJPDUnregisterNvSciSyncObj().
 *
 * For a given %NvMediaIJPD handle,
 * one NvSciSyncObj can be registered as one \ref NvMediaNvSciSyncObjType only.
 * For each NvMediaNvSciSyncObjType, a maximum of 16 NvSciSyncObjs can
 * be registered.
 *
 * @pre NvMediaIJPDFillNvSciSyncAttrList()
 * @post NvSciSyncObj registered with NvMediaIJPD
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] decoder A pointer to the NvMediaIJPD object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] syncobjtype Determines how @a nvscisync is used by @a decoder.
 *      \inputrange Entries in \ref NvMediaNvSciSyncObjType enumeration
 * @param[in] nvscisync The NvSciSyncObj to be registered with @a decoder.
 *      \inputrange A valid NvSciSyncObj
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a decoder is NULL or
 *         @a syncobjtype is not a valid NvMediaNvSciSyncObjType.
 * - ::NVMEDIA_STATUS_NOT_SUPPORTED if @a nvscisync is not a
 *         compatible NvSciSyncObj which %NvMediaIJPD can support.
 * - ::NVMEDIA_STATUS_ERROR if the maximum number of NvSciScynObjs
 *         are already registered for the given @a syncobjtype, or
 *         if @a nvscisync is already registered with the same @a decoder
 *         handle for a different @a syncobjtype.
 */
NvMediaStatus
NvMediaIJPDRegisterNvSciSyncObj(
    const NvMediaIJPD           *decoder,
    NvMediaNvSciSyncObjType    syncobjtype,
    NvSciSyncObj               nvscisync
);

/**
 * @brief Unregisters an \ref NvSciSyncObj with NvMediaIJPD.
 *
 * Every %NvSciSyncObj registered with %NvMediaIJPD by
 * NvMediaIJPDRegisterNvSciSyncObj() must be unregistered before calling
 * NvMediaIJPDUnregisterNvSciBufObj() to unregister the NvSciBufObjs.
 *
 * Before the application calls this function, it must ensure that any
 * \ref NvMediaIJPDRender() operation that uses the NvSciSyncObj has
 * completed. If this function is called while NvSciSyncObj is still
 * in use by any %NvMediaIJPDRender() operation, the API returns
 * NVMEDIA_STATUS_PENDING to indicate the same. NvSciSyncFenceWait() API can
 * be called on the EOF NvSciSyncFence obtained post the last call to
 * NvMediaIJPDRender() to wait for the associated tasks to complete.
 * The EOF NvSciSyncFence would have been previously obtained via a call to
 * NvMediaIJPDGetEOFNvSciSyncFence(). The other option would be to call
 * NvMediaIJPDGetBits() till there is no more output to retrieve.
 *
 * @pre NvMediaIJPDRender()
 * @pre NvMediaIJPDGetBits() or NvSciSyncFenceWait() [verify that processing is
 *                                                     complete]
 * @post NvSciSyncObj un-registered with NvMediaIJPD
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * @note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] decoder A pointer to the NvMediaIJPD object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] nvscisync An NvSciSyncObj to be unregistered with @a decoder.
 *      \inputrange A valid NvSciSyncObj
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if decoder is NULL, or
 *         @a nvscisync is not registered with @a decoder.
 * - ::NVMEDIA_STATUS_PENDING if the \ref NvSciSyncObj is still in use, i.e.,
 *        the submitted task is still in progress. In this case, the application
 *        can choose to wait for operations to complete on the output surface
 *        using NvSciSyncFenceWait() or re-try the
 *        %NvMediaIJPDUnregisterNvSciBufObj() API call, until the status
 *        returned is not \ref NVMEDIA_STATUS_PENDING.
 * - ::NVMEDIA_STATUS_ERROR if @a decoder was destroyed before this function
 *         was called.
 */
NvMediaStatus
NvMediaIJPDUnregisterNvSciSyncObj(
    const NvMediaIJPD  *decoder,
    NvSciSyncObj      nvscisync
);

/**
 * @brief Specifies the \ref NvSciSyncObj to be used for an EOF
 * \ref NvSciSyncFence.
 *
 * To use NvMediaIJPDGetEOFNvSciSyncFence(), the application must call
 * %NvMediaIJPDSetNvSciSyncObjforEOF() before it calls NvMediaIJPDRender().
 *
 * %NvMediaIJPDSetNvSciSyncObjforEOF() currently may be called only once before
 * each call to %NvMediaIJPDRender(). The application may choose to call this
 * function only once before the first call to %NvMediaIJPDRender().
 *
 * @pre NvMediaIJPDRegisterNvSciSyncObj()
 * @post NvSciSyncObj to be used as EOF NvSciSyncFence is set
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] decoder A pointer to the NvMediaIJPD object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] nvscisyncEOF A registered NvSciSyncObj which is to be
 *                           associated with EOF \ref NvSciSyncFence.
 *      \inputrange A valid NvSciSyncObj
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a decoder is NULL, or if @a nvscisyncEOF
 *         is not registered with @a decoder as either type
 *         \ref NVMEDIA_EOFSYNCOBJ or \ref NVMEDIA_EOF_PRESYNCOBJ.
 */
NvMediaStatus
NvMediaIJPDSetNvSciSyncObjforEOF(
    const NvMediaIJPD      *decoder,
    NvSciSyncObj          nvscisyncEOF
);

/**
 * @brief Sets an \ref NvSciSyncFence as a prefence for an
 * NvMediaIJPDRender() %NvSciSyncFence operation.
 *
 * You must call %NvMediaIJPDInsertPreNvSciSyncFence() before you call
 * %NvMediaIJPDRender(). The %NvMediaIJPDRender() operation is started only
 * after the expiry of the @a prenvscisyncfence.
 *
 * For example, in this sequence of code:
 * \code
 * nvmstatus = NvMediaIJPDInsertPreNvSciSyncFence(handle, prenvscisyncfence);
 * nvmstatus = NvMediaIJPDRender(handle, arg2, arg3, ...);
 * \endcode
 * the %NvMediaIJPDRender() operation is assured to start only after the
 * expiry of @a prenvscisyncfence.
 *
 * You can set a maximum of \ref NVMEDIA_IJPD_MAX_PRENVSCISYNCFENCES prefences
 * by calling %NvMediaIJPDInsertPreNvSciSyncFence() before %NvMediaIJPDRender().
 * After the call to %NvMediaIJPDRender(), all NvSciSyncFences previously
 * inserted by %NvMediaIJPDInsertPreNvSciSyncFence() are removed, and they are not
 * reused for the subsequent %NvMediaIJPDRender() calls.
 *
 * @pre Pre-NvSciSync fence obtained from previous engine in the pipeline
 * @post Pre-NvSciSync fence is set
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] decoder A pointer to the NvMediaIJPD object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] prenvscisyncfence A pointer to %NvSciSyncFence.
 *      \inputrange Non-NULL - valid pointer address
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a decoder is not a valid %NvMediaIJPD
 *     handle, or @a prenvscisyncfence is NULL, or if @a prenvscisyncfence was not
 *     generated with an \ref NvSciSyncObj that was registered with @a decoder as
 *     either \ref NVMEDIA_PRESYNCOBJ or \ref NVMEDIA_EOF_PRESYNCOBJ type.
 * - ::NVMEDIA_STATUS_NOT_SUPPORTED if %NvMediaIJPDInsertPreNvSciSyncFence()
 *     has already been called at least %NVMEDIA_IJPD_MAX_PRENVSCISYNCFENCES times
 *     with the same @a decoder handle before an %NvMediaIJPDRender() call.
 */
NvMediaStatus
NvMediaIJPDInsertPreNvSciSyncFence(
    const NvMediaIJPD         *decoder,
    const NvSciSyncFence     *prenvscisyncfence
);

/**
 * @brief Gets EOF \ref NvSciSyncFence for an NvMediaIJPDRender() operation.
 *
 * The EOF %NvSciSyncFence associated with an %NvMediaIJPDRender() operation
 * is an %NvSciSyncFence. Its expiry indicates that the corresponding
 * %NvMediaIJPDRender() operation has finished.
 *
 * This function returns the EOF %NvSciSyncFence associated
 * with the last %NvMediaIJPDRender() call. %NvMediaIJPDGetEOFNvSciSyncFence()
 * must be called after an %NvMediaIJPDRender() call.
 *
 * For example, in this sequence of code:
 * \code
 * nvmstatus = NvMediaIJPDRender(handle, arg2, arg3, ...);
 * nvmstatus = NvMediaIJPDGetEOFNvSciSyncFence(handle, nvscisyncEOF, eofnvscisyncfence);
 * \endcode
 * expiry of @a eofnvscisyncfence indicates that the preceding
 * %NvMediaIJPDRender() operation has finished.
 *
 * @pre NvMediaIJPDSetNvSciSyncObjforEOF()
 * @pre NvMediaIJPDRender()
 * @post EOF NvSciSync fence for a submitted task is obtained
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] decoder A pointer to the NvMediaIJPD object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] eofnvscisyncobj    An EOF NvSciSyncObj associated with the
 *                                 NvSciSyncFence which is being requested.
 *      \inputrange A valid NvSciSyncObj
 * @param[out] eofnvscisyncfence A pointer to the EOF NvSciSyncFence.
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a decoder is not a valid %NvMediaIJPD
 *         handle, @a eofnvscisyncfence is NULL, or @a eofnvscisyncobj is not
 *         registered with @a decoder as type \ref NVMEDIA_EOFSYNCOBJ or
 *         \ref NVMEDIA_EOF_PRESYNCOBJ.
 * - ::NVMEDIA_STATUS_ERROR if the function was called before
 *         %NvMediaIJPDRender() was called.
 */
NvMediaStatus
NvMediaIJPDGetEOFNvSciSyncFence(
    const NvMediaIJPD        *decoder,
    NvSciSyncObj      eofnvscisyncobj,
    NvSciSyncFence    *eofnvscisyncfence
);

/**
 * @brief Specifies the \ref NvSciSyncObj to be used for an SOF
 * \ref NvSciSyncFence.
 *
 * This function is not supported.
 *
 * To use NvMediaIJPDGetSOFNvSciSyncFence(), the application must call
 * %NvMediaIJPDSetNvSciSyncObjforSOF() before it calls NvMediaIJPDRender().
 *
 * %NvMediaIJPDSetNvSciSyncObjforSOF() currently may be called only once before
 * each call to %NvMediaIJPDRender(). The application may choose to call this
 * function only once before the first call to %NvMediaIJPDRender().
 *
 * @pre N/A
 * @post N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] decoder A pointer to the NvMediaIJPD object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] nvscisyncSOF A registered NvSciSyncObj which is to be
 *                           associated with SOF \ref NvSciSyncFence.
 *      \inputrange A valid NvSciSyncObj
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a decoder is NULL, or if @a nvscisyncSOF
 *         is not registered with @a decoder as either type
 *         \ref NVMEDIA_SOFSYNCOBJ or \ref NVMEDIA_SOF_PRESYNCOBJ.
 */
NvMediaStatus
NvMediaIJPDSetNvSciSyncObjforSOF(
    const NvMediaIJPD        *decoder,
    NvSciSyncObj            nvscisyncSOF
);

/**
 * @brief Gets SOF \ref NvSciSyncFence for an NvMediaIJPDRender() operation.
 *
 * This function is not supported.
 *
 * The SOF %NvSciSyncFence associated with an %NvMediaIJPDRender() operation
 * is an %NvSciSyncFence. Its expiry indicates that the corresponding
 * %NvMediaIJPDRender() operation has started.
 *
 * This function returns the SOF %NvSciSyncFence associated
 * with the last %NvMediaIJPDRender() call. %NvMediaIJPDGetSOFNvSciSyncFence()
 * must be called after an %NvMediaIJPDRender() call.
 *
 * @pre N/A
 * @post N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] decoder A pointer to the NvMediaIJPD object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] sofnvscisyncobj    An SOF NvSciSyncObj associated with the
 *                                 NvSciSyncFence which is being requested.
 *      \inputrange A valid NvSciSyncObj
 * @param[out] sofnvscisyncfence A pointer to the SOF NvSciSyncFence.
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a decoder is not a valid %NvMediaIJPD
 *         handle, @a sofnvscisyncfence is NULL, or @a sofnvscisyncobj is not
 *         registered with @a decoder as type \ref NVMEDIA_SOFSYNCOBJ or
 *         \ref NVMEDIA_SOF_PRESYNCOBJ.
 * - ::NVMEDIA_STATUS_ERROR if the function was called before
 *         %NvMediaIJPDRender() was called.
 */
NvMediaStatus
NvMediaIJPDGetSOFNvSciSyncFence(
    const NvMediaIJPD        *decoder,
    NvSciSyncObj            sofnvscisyncobj,
    NvSciSyncFence          *sofnvscisyncfence
);


/*
 * @defgroup 6x_history_nvmedia_ijpd History
 * Provides change history for the NvMedia Image Jpeg Decode API.
 *
 * \section 6x_history_nvmedia_ijpd Version History
 *
 * <b> Version 1.0 </b> September 28, 2021
 * - Initial release
 */

/** @} */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_IJPD_H */
