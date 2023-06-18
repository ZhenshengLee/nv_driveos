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
 * @brief <b> NVIDIA Media Interface: The NvMedia Decode Processing API </b>
 *
 * This file contains the @ref 6x_decoder_api "Decode Processing API".
 */

#ifndef NVMEDIA_IDE_H
#define NVMEDIA_IDE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "nvmedia_common_decode.h"
#include "nvmedia_core.h"
#include "nvscibuf.h"
#include "nvscisync.h"

/**
 * @defgroup 6x_image_decoder_api Image Decoder
 * @ingroup 6x_nvmedia_image_top
 *
 * Defines and manages objects that decode video.
 *
 * The NvMediaIDE object decodes compressed video data, writing
 * the results to a @ref NvSciBufObj "target"
 *
 * A specific NvMedia implementation may support decoding multiple
 * types of compressed video data. However, NvMediaIDE objects
 * are able to decode a specific type of compressed video data.
 * This type must be specified during creation.
 *
 * @{
 */

/** @brief Major Version number */
#define NVMEDIA_IDE_VERSION_MAJOR   1
/** @brief Minor Version number */
#define NVMEDIA_IDE_VERSION_MINOR   0
/** @brief Patch Version number */
#define NVMEDIA_IDE_VERSION_PATCH   0

/**
 * Specifies the maximum number of times NvMediaIDEInsertPreNvSciSyncFence()
 * can be called before each call to NvMediaIDEFeedFrame().
 */
#define NVMEDIA_IDE_MAX_PRENVSCISYNCFENCES  (16U)



/**
 * @brief An opaque NvMediaIDE object created by
 * @ref NvMediaIDECreate
 */
typedef struct NvMediaIDE NvMediaIDE;

/**
 * @defgroup 6x_decoder_create_flag Decoder Creation Flag
 * Defines decoder flag bit masks for constructing the decoder.
 * @ingroup 6x_image_decoder_api
 * @{
 */

/**
 * \hideinitializer
 * @brief Defines 10-bit decode.
 */
#define NVMEDIA_IDE_10BIT_DECODE (1U<<0)

/**
 * \hideinitializer
 * @brief Rec_2020 color format for the decoded surface
 */

#define NVMEDIA_IDE_PIXEL_REC_2020 (1U<<1)

/**
 * \hideinitializer
 * @brief Use 16 bit surfaces if contents is higher than 8 bit.
 */

#define NVMEDIA_IDE_OUTPUT_16BIT_SURFACES (1U<<2)

/**
 * \hideinitializer
 * @brief Create decoder for encrypted content decoding
 */

#define NVMEDIA_IDE_ENABLE_AES  (1U<<3)

/**
 * \hideinitializer
 * @brief Create decoder to output in NV24 format.
 */

#define NVMEDIA_IDE_NV24_OUTPUT (1U<<4)

/**
 * \hideinitializer
 * @brief Enable decoder profiling support
 */

#define NVMEDIA_IDE_PROFILING   (1U<<5)

/**
 * \hideinitializer
 * @brief Enable decoder motion vector dump
 */

#define NVMEDIA_IDE_DUMP_MV     (1U<<6)

/**@} <!-- Ends decoder_create_flag sub-group --> */

/**
 * @brief Retrieves the version information for the NvMediaIDE library.
 *
 * @pre  None
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
 * @param[in] version A pointer to a @ref NvMediaVersion structure
 *                    of the client.
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if the pointer is invalid.
 */
NvMediaStatus
NvMediaIDEGetVersion(
    NvMediaVersion *version
);

/** @brief Creates an NvMediaIDE object.
 *
 * Creates a @ref NvMediaIDE object for the specified codec. Each
 * decoder object may be accessed by a separate thread. The object
 * must be destroyed with @ref NvMediaIDEDestroy().
 *
 * @pre NvMediaIDEGetVersion()
 * @post NvMediaIDE object is created.
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
 * @param[in] codec Codec type. The following types are supported:
 * - @ref NVMEDIA_VIDEO_CODEC_HEVC
 * - @ref NVMEDIA_VIDEO_CODEC_H264
 * - @ref NVMEDIA_VIDEO_CODEC_VC1
 * - @ref NVMEDIA_VIDEO_CODEC_VC1_ADVANCED
 * - @ref NVMEDIA_VIDEO_CODEC_MPEG1
 * - @ref NVMEDIA_VIDEO_CODEC_MPEG2
 * - @ref NVMEDIA_VIDEO_CODEC_MPEG4
 * - @ref NVMEDIA_VIDEO_CODEC_MJPEG
 * - @ref NVMEDIA_VIDEO_CODEC_VP8
 * - @ref NVMEDIA_VIDEO_CODEC_VP9
 * - @ref NVMEDIA_VIDEO_CODEC_AV1           [Supported only on T234 and further chips]
 * @param[in] width Decoder width in luminance pixels.
 * @param[in] height Decoder height in luminance pixels.
 * @param[in] maxReferences The maximum number of reference frames used.
 * This limits internal allocations.
 * @param[in] maxBitstreamSize The maximum size for bitstream.
 * This limits internal allocations.
 * @param[in] inputBuffering How many frames can be in flight at any given
 * time. If this value is 1, NvMediaIDERender() blocks until the
 * previous frame has finished decoding. If this is 2, \c NvMediaIDERender
 * blocks if two frames are pending but does not block if one is pending.
 * This value is clamped internally to between 1 and 8.
 * @param[in] flags Set the flags of the decoder.
 * The following flags are supported:
 * - ::NVMEDIA_IDE_10BIT_DECODE
 * @param[in] instanceId The ID of the engine instance.
 * The following instances are supported:
 * - ::NVMEDIA_DECODER_INSTANCE_0
 * - ::NVMEDIA_DECODER_INSTANCE_1       [Supported only on T194]
 * - ::NVMEDIA_DECODER_INSTANCE_AUTO    [Supported only on T194]
 * @return NvMediaIDE The created NvMediaIDE handle or NULL if unsuccessful.
 */

NvMediaIDE *
NvMediaIDECreate(
    NvMediaVideoCodec codec,
    uint16_t width,
    uint16_t height,
    uint16_t maxReferences,
    uint64_t maxBitstreamSize,
    uint8_t inputBuffering,
    uint32_t flags,
    NvMediaDecoderInstanceId instanceId
);

/**
 * @brief Create an NvMediaIDE object instance
 *
 * @pre NvMediaIDEGetVersion()
 * @pre NvMediaIDENvSciSyncGetVersion() [for use with IMGDEC-NvSciSync APIs]
 * @post NvMediaIDE object is created
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
 * @return ::NvMediaIDE The created NvMediaIDE handle or NULL if unsuccessful.
 */
NvMediaIDE *
NvMediaIDECreateCtx(
    void
);

/**
 * @brief Initialize an NvMediaIDE object instance
 * @pre NvMediaIDECreateCtx()
 * @post NvMediaIDE object is initialized.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * Initializes a @ref NvMediaIDE object for the specified codec. Each
 * decoder object may be accessed by a separate thread. The object
 * must be destroyed with @ref NvMediaIDEDestroy().
 *
 * @param[in] decoder The decoder to use.
 * @param[in] codec Codec type. The following types are supported:
 * - @ref NVMEDIA_VIDEO_CODEC_HEVC
 * - @ref NVMEDIA_VIDEO_CODEC_H264
 * - @ref NVMEDIA_VIDEO_CODEC_VC1
 * - @ref NVMEDIA_VIDEO_CODEC_VC1_ADVANCED
 * - @ref NVMEDIA_VIDEO_CODEC_MPEG1
 * - @ref NVMEDIA_VIDEO_CODEC_MPEG2
 * - @ref NVMEDIA_VIDEO_CODEC_MPEG4
 * - @ref NVMEDIA_VIDEO_CODEC_MJPEG
 * - @ref NVMEDIA_VIDEO_CODEC_VP8
 * - @ref NVMEDIA_VIDEO_CODEC_VP9
 * - @ref NVMEDIA_VIDEO_CODEC_AV1           [Supported only on T234 and further chips]
 * @param[in] width Decoder width in luminance pixels.
 * @param[in] height Decoder height in luminance pixels.
 * @param[in] maxReferences The maximum number of reference frames used.
 * This limits internal allocations.
 * @param[in] maxBitstreamSize The maximum size for bitstream.
 * This limits internal allocations.
 * @param[in] inputBuffering How many frames can be in flight at any given
 * time. If this value is 1, NvMediaIDERender() blocks until the
 * previous frame has finished decoding. If this is 2, \c NvMediaIDERender
 * blocks if two frames are pending but does not block if one is pending.
 * This value is clamped internally to between 1 and 8.
 * @param[in] flags Set the flags of the decoder.
 * The following flags are supported:
 * - ::NVMEDIA_IDE_10BIT_DECODE
 * @param[in] instanceId The ID of the engine instance.
 * The following instances are supported:
 * - ::NVMEDIA_DECODER_INSTANCE_0
 * - ::NVMEDIA_DECODER_INSTANCE_1       [Supported only on T194]
 * - ::NVMEDIA_DECODER_INSTANCE_AUTO    [Supported only on T194]
 *
 * @return ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - ::NVMEDIA_STATUS_ERROR if it is called after initialization of decoder.
 */
NvMediaStatus
NvMediaIDEInit(
    NvMediaIDE *decoder,
    NvMediaVideoCodec codec,
    uint16_t width,
    uint16_t height,
    uint16_t maxReferences,
    uint64_t maxBitstreamSize,
    uint8_t inputBuffering,
    uint32_t flags,
    NvMediaDecoderInstanceId instanceId
);

/** @brief Destroys an NvMediaIDE object.
 * @param[in] decoder The decoder to be destroyed.
 *
 * @pre NvMediaIDEUnregisterNvSciBufObj()
 * @pre NvMediaIDEUnregisterNvSciSyncObj()
 * @post NvMediaIDE object is destroyed
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * @return void
 */
NvMediaStatus
NvMediaIDEDestroy(
   const NvMediaIDE *decoder
);

/**
 * @brief Registers an @ref NvSciBufObj for use with an @ref NvMediaIde handle.
 * The NvMediaIde handle maintains a record of all the bufObjs registered using
 * this API.
 *
 * This is a optional API which needs to be called before NvMediaIDEProcessFrame()
 * \n All %NvMediaIDENvSciBufRegister() API calls must be made before first
 * %NvMediaIDEProcessFrame() API call.
 * Registration of the buffer is done with the same access permission as
 * that of the NvSciBufObj being registered. NvSciBufObj that need to be
 * registered with a reduced permission (Eg: Input buffer accesses being set to
 * read-only) can be done so by first duplicating the NvSciBufObj using
 * NvSciBufObjDupWithReducePerm() followed by a call the register the duplicated
 * NvSciBufObj.
 *
 *
 * Maximum of 192 NvSciBufObj handles can be registered using %NvMediaIDERegisterNvSciSyncObj() API.
 *
 * @pre NvMediaIDEInit()
 * @pre NvMediaIDERegisterNvSciSyncObj()
 * @post NvSciBufObj is registered with NvMediaIde object
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIde object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @param[in] decoder
 *      @ref NvMediaIde handle.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] bufObj
 *      An NvSciBufObj object.
 *      \inputrange A valid NvSciBufObj

 * @return ::NvMediaStatus, the completion status of operation:
 * - ::NVMEDIA_STATUS_OK if successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if ofa, bufObj or accessMode is invalid.
 * - ::NVMEDIA_STATUS_ERROR in following cases:
 *          - User registers more than 192 bufObjs.
 *          - User registers same bufObj with more than one accessModes.
 *          - User registers same bufObj multiple times.
 *
 **/
NvMediaStatus
NvMediaIDERegisterNvSciBufObj (
    const NvMediaIDE *decoder,
    NvSciBufObj        bufObj
);

/**
 * @brief Un-registers @ref NvSciBufObj which was previously registered with
 * @ref NvMediaIde using NvMediaIDERegisterNvSciBufObj().
 *
 * For all NvSciBufObj handles registered with NvMediaIde using
 * %NvMediaIDERegisterNvSciBufObj() API, %NvMediaIDEUnregisterNvSciBufObj()
 * must be called before calling NvMediaIDEDestroy() API.
 * For unregistration to succeed, it should be ensured that none of the
 * submitted tasks on the bufObj are pending prior to calling
 * %NvMediaIDEUnregisterNvSciBufObj(). In order to ensure this,
 * %NvMediaIDEUnregisterNvSciSyncObj() should be called prior to this API on
 * all registered NvSciSyncObj. Post this NvMediaIDEUnregisterNvSciBufObj() can
 * be successfully called on a valid NvSciBufObj.
 *
 * For deterministic execution of %NvMediaIDEProcessFrame() API,
 * %NvMediaIDEUnregisterNvSciBufObj() must be called only after last
 * %NvMediaIDEProcessFrame() call.
 *
 * @pre NvMediaIDEUnregisterNvSciSyncObj() [verify that processing is complete]
 * @post NvSciBufObj is un-registered from NvMediaIde object
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIde object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * @param[in] decoder
 *      @ref NvMediaIde handle.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] bufObj
 *      An NvSciBufObj object.
 *      \inputrange A valid NvSciBufObj
 * @return ::NvMediaStatus, the completion status of operation:
 * - ::NVMEDIA_STATUS_OK if successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if ofa or bufObj is invalid
 *           %NvMediaIDERegisterNvSciBufObj() API.
 * - ::NVMEDIA_STATUS_ERROR in following cases:
 *          - User unregisters an NvSciBufObj which is not previously
 *            registered using %NvMediaIDERegisterNvSciBufObj() API.
 *          - User unregisters an NvSciBufObj multiple times.
 **/
NvMediaStatus
NvMediaIDEUnregisterNvSciBufObj (
    const NvMediaIDE *decoder,
    NvSciBufObj       bufObj
);

/**
 * @brief Decodes a compressed field/frame and render the result
 *        into a @ref NvSciBufObj "target".
 *
 * @pre NvMediaIDERegisterNvSciSyncObj()
 * @pre NvMediaIDESetNvSciSyncObjforEOF()
 * @pre NvMediaIDEInsertPreNvSciSyncFence()
 * @post Decoding task is submitted
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Async
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] decoder The decoder object that will perform the
 *       decode operation.
 * @param[in] target NvSciBufObj that contains the decoded content, allocated
 *      with a call to @ref NvSciBufObjAlloc.
 *      characterizes the output surface into which the Decoded output is obtained \n
 *      Supported surface format attributes (common to all codecs): \n
 *      For content with bitdepth 10/12  Bit  the output  is MSB Aligned
 *      Non-safety build:</b> \n
 *          H.264, MPEG2, MPEG4, VP8, VC1: \n
 *              Sub-sampling type: YUV420, (Block Linear semi-planar) \n
 *              Bit Depth: 8 \n
 *          H.265/HEVC: \n
 *              Sub-sampling type: YUV420, YUV444 (Block Linear semi-planar) \n
 *              Bit Depth: 8, 10, 12 \n
 *          VP9: \n
 *              Sub-sampling type: YUV420, (Block Linear semi-planar) \n
 *              Bit Depth: 8, 10, 12 \n
 *          AV1: [Supported only on T234 and further chips] \n
 *              Sub-sampling type: YUV420, (Block Linear semi-planar) \n
 *              Bit Depth: 8, 10 \n
 * @param[in] pictureInfo A (pointer to a) structure containing
 *       information about the picture to be decoded. Note that
 *       the appropriate type of NvMediaPictureInfo* structure must
 *       be provided to match to profile that the decoder was
 *       created for.
 * @param[in] encryptParams A (pointer to a) structure containing
 *       information about encryption parameter used to decrypt the
 *       video content on the fly.
 * @param[in] numBitstreamBuffers The number of bitstream
 *       buffers containing compressed data for this picture.
 * @param[in] bitstreams An array of bitstream buffers.
 * @param[out] FrameStatsDump A (pointer to a) structure containing
 *       frame coding specific informations. This includes frame
 *       type, motion vector dumps,macroblock types and other details.
 * @param[in] instanceId The ID of the engine instance.
 * The following instances are supported if NVMEDIA_DECODER_INSTANCE_AUTO
 * was used in @ref NvMediaIDECreate API, else this parameter is ignored:
 * - ::NVMEDIA_DECODER_INSTANCE_0
 * - ::NVMEDIA_DECODER_INSTANCE_1      [Supported only on T194]
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_BAD_PARAMETER
 */
NvMediaStatus
NvMediaIDEDecoderRender(
    const NvMediaIDE *decoder,
    NvSciBufObj target,
    const NvMediaPictureInfo *pictureInfo,
    const void *encryptParams,
    uint32_t numBitstreamBuffers,
    const NvMediaBitstreamBuffer *bitstreams,
    NvMediaIDEFrameStats *FrameStatsDump,
    NvMediaDecoderInstanceId instanceId
);

/**
 * @brief This function is intended for use in low-latency decode mode.
 *  It is implemented only for H265 decoder. Error will be returned if it is
 *  called for any other codec.
 *
 *  Each set of buffers should contain exactly 1 slice data.
 *  For first slice of every frame, @ref NvMediaIDERender() function should be called.
 *  @ref NvMediaIDESliceDecode() function should be called for all subsequent
 *  slices of the frame.
 *
 *  Note that the ucode expects next slice data to be available within certain
 *  time (= 100msec). If data is not available within this time, it is assumed that
 *  the data is lost and error-concealment may be performed on the remaining portion
 *  of the frame.
 *
 * @pre NvMediaIDERegisterNvSciSyncObj()
 * @pre NvMediaIDESetNvSciSyncObjforEOF()
 * @pre NvMediaIDEInsertPreNvSciSyncFence()
 * @post Decoding task is submitted
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Async
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] decoder The decoder object that will perform the
 *       decode operation.
 * @param[in] target NvSciBufObj that contains the decoded content, allocated
 *      with a call to @ref NvSciBufObjAlloc.
 * @param[in] sliceDecData SliceDecode data info.
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_BAD_PARAMETER
 */

NvMediaStatus
NvMediaIDESliceDecode (
    const NvMediaIDE *decoder,
    const NvSciBufObj target,
    const NvMediaSliceDecodeData *sliceDecData
);

/**
 * @brief Retrieves the HW decode status available. This is to be used to get
 *  the decoding status. If application does not need decoding status then no need
 *  to call this function. This function should be called in decode order once
 *  decode is complete for target surface. This can be called from separate thread
 *  in decode order before the same index getting used. Syncronization can be
 *  acheived for @ref NvMediaIDERender and @ref
 *  NvMediaIDEGetFrameDecodeStatus using Semaphore.
 *  Semaphore can be signalled at the begining with inputBuffering times.
 *  Then client can wait on Semaphore before every @ref NvMediaIDERender
 *  This Semaphore will be signalled from thread after calling
 *  @ref NvMediaIDEGetFrameDecodeStatus for frame in decode order.
 *  For example: if inputBuffering is 3 when creating decoder using
 *  @ref NvMediaIDECreate then following sequence should be followed.
 *  NvMediaIDERender(DecodeOrder:0) : ringEntryIdx=0
 *  NvMediaIDERender(DecodeOrder:1) : ringEntryIdx=1
 *  NvMediaIDERender(DecodeOrder:2) : ringEntryIdx=2
 *  NvMediaIDEGetFrameDecodeStatus(0)
 *  NvMediaIDERender() : ringEntryIdx=0
 *  NvMediaIDEGetFrameDecodeStatus(1)
 *  NvMediaIDERender() : ringEntryIdx=1
 *  NvMediaIDEGetFrameDecodeStatus(2)
 *  NvMediaIDERender() : ringEntryIdx=2
 *  NvMediaIDEGetFrameDecodeStatus(0)
 *  Another example could be like this for above case
 *  NvMediaIDERender(DecodeOrder:0) : ringEntryIdx=0
 *  NvMediaIDERender(DecodeOrder:1) : ringEntryIdx=1
 *  NvMediaIDEGetFrameDecodeStatus(0)
 *  NvMediaIDERender(DecodeOrder:2) : ringEntryIdx=2
 *  NvMediaIDEGetFrameDecodeStatus(1)
 *  NvMediaIDERender() : ringEntryIdx=0
 *  NvMediaIDERender() : ringEntryIdx=1
 *  NvMediaIDEGetFrameDecodeStatus(2)
 *  NvMediaIDERender() : ringEntryIdx=2
 *  NvMediaIDEGetFrameDecodeStatus(0)
 *
 * @pre NvMediaIDERegisterNvSciSyncObj()
 * @pre NvMediaIDESetNvSciSyncObjforEOF()
 * @pre NvMediaIDEInsertPreNvSciSyncFence()
 * @post Decoding task status is returned.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] decoder The decoder object that will perform the
 *       decode operation.
 * @param[in] ringEntryIdx This is decoder order index.
 *       decode operation.
 * @param[out]  FrameStatus A pointer to @ref NvMediaIDEFrameStatus structure
 *       which will store current decoded frame status.
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_BAD_PARAMETER
 */

NvMediaStatus
NvMediaIDEGetFrameDecodeStatus(
    const NvMediaIDE *decoder,
    uint32_t ringEntryIdx,
    NvMediaIDEFrameStatus *FrameStatus
);

/**
 * @brief Fills the NvMediaIDE specific NvSciBuf attributes which than then be
 * used to allocate an @ref NvSciBufObj that NvMediaIDE can consume.
 *
 * This function assumes that @a attrlist is a valid @ref NvSciBufAttrList
 * created by the caller by a call to @ref NvSciBufAttrListCreate.
 *
 * @pre NvMediaIDEGetVersion()
 * @post NvSciBufAttrList populated with NvMediaIDE specific NvSciBuf
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
 *      - ::NVMEDIA_DECODER_INSTANCE_0
 *      - ::NVMEDIA_DECODER_INSTANCE_1       [Supported only on T194]
 *      - ::NVMEDIA_DECODER_INSTANCE_AUTO    [Supported only on T194]
 * @param[out] attrlist An %NvSciBufAttrList where NvMediaIDE places
 *      the NvSciBuf attributes.
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a attrlist is NULL
 */
NvMediaStatus
NvMediaIDEFillNvSciBufAttrList(
    NvMediaDecoderInstanceId  instanceId,
    NvSciBufAttrList          attrlist
);

/**
 * @brief Fills the NvMediaIDE specific NvSciSync attributes.
 *
 * This function assumes that @a attrlist is a valid @ref NvSciSyncAttrList.
 *
 * This function updates the input NvSciSyncAttrList with values equivalent to
 * the following public attribute key-values:
 * NvSciSyncAttrKey_RequiredPerm set to
 * - NvSciSyncAccessPerm_WaitOnly for @ref clienttype NVMEDIA_WAITER
 * - NvSciSyncAccessPerm_SignalOnly for @ref clienttype NVMEDIA_SIGNALER
 * - NvSciSyncAccessPerm_WaitSignal for @ref clienttype NVMEDIA_SIGNALER_WAITER
 * NvSciSyncAttrKey_PrimitiveInfo set to
 * - NvSciSyncAttrValPrimitiveType_Syncpoint
 *
 * The application must not set these attributes in the NvSciSyncAttrList passed
 * as an input to this function.
 *
 * @pre NvMediaIDECreate()
 * @post NvSciSyncAttrList populated with NvMediaIDE specific NvSciSync
 *        attributes
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @noteThis API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] decoder A pointer to the NvMediaIDE object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[out] attrlist A pointer to an %NvSciSyncAttrList structure where
 *                NvMedia places NvSciSync attributes.
 * @param[in] clienttype Indicates whether the NvSciSyncAttrList requested for
 *                an %NvMediaIDE signaler or an %NvMediaIDE waiter.
 *      \inputrange Entries in @ref NvMediaNvSciSyncClientType enumeration
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
NvMediaIDEFillNvSciSyncAttrList(
    const NvMediaIDE           *decoder,
    NvSciSyncAttrList          attrlist,
    NvMediaNvSciSyncClientType clienttype
);

/** @brief NvMediaIDE get backward updates counters for VP9
 *       adaptive entropy contexts.
 *
 * @pre NvMediaIDECreate() and only for VP9
 * @post Updates VP9 Entropy context.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @param[in] decoder A pointer to the decoder object that performs the
 *       decoding operation.
 * @param[in] backupdates A pointer to a structure that holds the
 *       backward update counters.
 */
NvMediaStatus
NvMediaIDEGetBackwardUpdates(
    const NvMediaIDE *decoder,
    void *backupdates
);

/**
 * @brief Registers an @ref NvSciSyncObj with NvMediaIDE.
 *
 * Every NvSciSyncObj (even duplicate objects) used by %NvMediaIDE
 * must be registered by a call to this function before it is used.
 * Only the exact same registered NvSciSyncObj can be passed to
 * NvMediaIDESetNvSciSyncObjforEOF(), NvMediaIDEGetEOFNvSciSyncFence(), or
 * NvMediaIDEUnregisterNvSciSyncObj().
 *
 * For a given %NvMediaIDE handle,
 * one NvSciSyncObj can be registered as one @ref NvMediaNvSciSyncObjType only.
 * For each NvMediaNvSciSyncObjType, a maximum of 16 NvSciSyncObjs can
 * be registered.
 *
 * @pre NvMediaIDEFillNvSciSyncAttrList()
 * @post NvSciSyncObj registered with NvMediaIDE
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @noteThis API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] decoder A pointer to the NvMediaIDE object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] syncobjtype Determines how @a nvscisync is used by @a decoder.
 *      \inputrange Entries in @ref NvMediaNvSciSyncObjType enumeration
 * @param[in] nvscisync The NvSciSyncObj to be registered with @a decoder.
 *      \inputrange A valid NvSciSyncObj
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a decoder is NULL or
 *         @a syncobjtype is not a valid NvMediaNvSciSyncObjType.
 * - ::NVMEDIA_STATUS_NOT_SUPPORTED if @a nvscisync is not a
 *         compatible NvSciSyncObj which %NvMediaIDE can support.
 * - ::NVMEDIA_STATUS_ERROR if the maximum number of NvSciScynObjs
 *         are already registered for the given @a syncobjtype, or
 *         if @a nvscisync is already registered with the same @a decoder
 *         handle for a different @a syncobjtype.
 */
NvMediaStatus
NvMediaIDERegisterNvSciSyncObj(
    const NvMediaIDE           *decoder,
    NvMediaNvSciSyncObjType    syncobjtype,
    NvSciSyncObj               nvscisync
);

/**
 * @brief Unregisters an @ref NvSciSyncObj with NvMediaIDE.
 *
 * Every %NvSciSyncObj registered with %NvMediaIDE by
 * NvMediaIDERegisterNvSciSyncObj() must be unregistered before calling
 * NvMediaIDEUnregisterNvSciBufObj() to unregister the NvSciBufObjs.
 *
 * Before the application calls this function, it must ensure that any
 * @ref NvMediaIDERender() operation that uses the NvSciSyncObj has
 * completed. If this function is called while NvSciSyncObj is still
 * in use by any %NvMediaIDERender() operation, the API returns
 * NVMEDIA_STATUS_PENDING to indicate the same. NvSciSyncFenceWait() API can
 * be called on the EOF NvSciSyncFence obtained post the last call to
 * NvMediaIDERender() to wait for the associated tasks to complete.
 * The EOF NvSciSyncFence would have been previously obtained via a call to
 * NvMediaIDEGetEOFNvSciSyncFence(). The other option would be to call
 * NvMediaIDEGetBits() till there is no more output to retrieve.
 *
 * @pre NvMediaIDERender()
 * @pre NvMediaIDEGetBits() or NvSciSyncFenceWait() [verify that processing is
 *                                                     complete]
 * @post NvSciSyncObj un-registered with NvMediaIDE
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * @noteThis API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] decoder A pointer to the NvMediaIDE object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] nvscisync An NvSciSyncObj to be unregistered with @a decoder.
 *      \inputrange A valid NvSciSyncObj
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if decoder is NULL, or
 *         @a nvscisync is not registered with @a decoder.
 * - @ref NVMEDIA_STATUS_PENDING if the @ref NvSciSyncObj is still in use, i.e.,
 *        the submitted task is still in progress. In this case, the application
 *        can choose to wait for operations to complete on the output surface
 *        using NvSciSyncFenceWait() or re-try the
 *        %NvMediaIDEUnregisterNvSciBufObj() API call, until the status
 *        returned is not @ref NVMEDIA_STATUS_PENDING.
 * - ::NVMEDIA_STATUS_ERROR if @a decoder was destroyed before this function
 *         was called.
 */
NvMediaStatus
NvMediaIDEUnregisterNvSciSyncObj(
    const NvMediaIDE  *decoder,
    NvSciSyncObj      nvscisync
);

/**
 * @brief Specifies the @ref NvSciSyncObj to be used for an EOF
 * @ref NvSciSyncFence.
 *
 * To use NvMediaIDEGetEOFNvSciSyncFence(), the application must call
 * %NvMediaIDESetNvSciSyncObjforEOF() before it calls NvMediaIDERender().
 *
 * %NvMediaIDESetNvSciSyncObjforEOF() currently may be called only once before
 * each call to %NvMediaIDERender(). The application may choose to call this
 * function only once before the first call to %NvMediaIDERender().
 *
 * @pre NvMediaIDERegisterNvSciSyncObj()
 * @post NvSciSyncObj to be used as EOF NvSciSyncFence is set
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @noteThis API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] decoder A pointer to the NvMediaIDE object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] nvscisyncEOF A registered NvSciSyncObj which is to be
 *                           associated with EOF @ref NvSciSyncFence.
 *      \inputrange A valid NvSciSyncObj
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a decoder is NULL, or if @a nvscisyncEOF
 *         is not registered with @a decoder as either type
 *         @ref NVMEDIA_EOFSYNCOBJ or @ref NVMEDIA_EOF_PRESYNCOBJ.
 */
NvMediaStatus
NvMediaIDESetNvSciSyncObjforEOF(
    const NvMediaIDE      *decoder,
    NvSciSyncObj          nvscisyncEOF
);

/**
 * @brief Sets an @ref NvSciSyncFence as a prefence for an
 * NvMediaIDERender() %NvSciSyncFence operation.
 *
 * You must call %NvMediaIDEInsertPreNvSciSyncFence() before you call
 * %NvMediaIDERender(). The %NvMediaIDERender() operation is started only
 * after the expiry of the @a prenvscisyncfence.
 *
 * For example, in this sequence of code:
 * @code
 * nvmstatus = NvMediaIDEInsertPreNvSciSyncFence(handle, prenvscisyncfence);
 * nvmstatus = NvMediaIDERender(handle, srcsurf, srcrect, picparams, instanceid);
 * @endcode
 * the %NvMediaIDERender() operation is assured to start only after the
 * expiry of @a prenvscisyncfence.
 *
 * You can set a maximum of @ref NVMEDIA_IDE_MAX_PRENVSCISYNCFENCES prefences
 * by calling %NvMediaIDEInsertPreNvSciSyncFence() before %NvMediaIDERender().
 * After the call to %NvMediaIDERender(), all NvSciSyncFences previously
 * inserted by %NvMediaIDEInsertPreNvSciSyncFence() are removed, and they are not
 * reused for the subsequent %NvMediaIDERender() calls.
 *
 * @pre Pre-NvSciSync fence obtained from previous engine in the pipeline
 * @post Pre-NvSciSync fence is set
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @noteThis API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] decoder A pointer to the NvMediaIDE object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] prenvscisyncfence A pointer to %NvSciSyncFence.
 *      \inputrange Non-NULL - valid pointer address
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a decoder is not a valid %NvMediaIDE
 *     handle, or @a prenvscisyncfence is NULL, or if @a prenvscisyncfence was not
 *     generated with an @ref NvSciSyncObj that was registered with @a decoder as
 *     either @ref NVMEDIA_PRESYNCOBJ or @ref NVMEDIA_EOF_PRESYNCOBJ type.
 * - ::NVMEDIA_STATUS_NOT_SUPPORTED if %NvMediaIDEInsertPreNvSciSyncFence()
 *     has already been called at least %NVMEDIA_IDE_MAX_PRENVSCISYNCFENCES times
 *     with the same @a decoder handle before an %NvMediaIDERender() call.
 */
NvMediaStatus
NvMediaIDEInsertPreNvSciSyncFence(
    const NvMediaIDE         *decoder,
    const NvSciSyncFence     *prenvscisyncfence
);

/**
 * @brief Gets EOF @ref NvSciSyncFence for an NvMediaIDERender() operation.
 *
 * The EOF %NvSciSyncFence associated with an %NvMediaIDERender() operation
 * is an %NvSciSyncFence. Its expiry indicates that the corresponding
 * %NvMediaIDERender() operation has finished.
 *
 * This function returns the EOF %NvSciSyncFence associated
 * with the last %NvMediaIDERender() call. %NvMediaIDEGetEOFNvSciSyncFence()
 * must be called after an %NvMediaIDERender() call.
 *
 * For example, in this sequence of code:
 * @code
 * nvmstatus = NvMediaIDERender(handle, srcsurf, srcrect, picparams, instanceid);
 * nvmstatus = NvMediaIDEGetEOFNvSciSyncFence(handle, nvscisyncEOF, eofnvscisyncfence);
 * @endcode
 * expiry of @a eofnvscisyncfence indicates that the preceding
 * %NvMediaIDERender() operation has finished.
 *
 * @pre NvMediaIDESetNvSciSyncObjforEOF()
 * @pre NvMediaIDERender()
 * @post EOF NvSciSync fence for a submitted task is obtained
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @noteThis API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] decoder A pointer to the NvMediaIDE object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] eofnvscisyncobj    An EOF NvSciSyncObj associated with the
 *                                 NvSciSyncFence which is being requested.
 *      \inputrange A valid NvSciSyncObj
 * @param[out] eofnvscisyncfence A pointer to the EOF NvSciSyncFence.
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a decoder is not a valid %NvMediaIDE
 *         handle, @a eofnvscisyncfence is NULL, or @a eofnvscisyncobj is not
 *         registered with @a decoder as type @ref NVMEDIA_EOFSYNCOBJ or
 *         @ref NVMEDIA_EOF_PRESYNCOBJ.
 * - ::NVMEDIA_STATUS_ERROR if the function was called before
 *         %NvMediaIDERender() was called.
 */
NvMediaStatus
NvMediaIDEGetEOFNvSciSyncFence(
    const NvMediaIDE        *decoder,
    NvSciSyncObj      eofnvscisyncobj,
    NvSciSyncFence    *eofnvscisyncfence
);

/*
 * @defgroup 6x_history_nvmedia_ide History
 * Provides change history for the NvMediaIDE API.
 *
 * \section 6x_history_nvmedia_ide Version History
 *
 * <b> Version 1.0 </b> September 28, 2021
 * - Initial release
 */
/** @} <!-- Ends image_decoder_api Image Decoder --> */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_IDE_H */
