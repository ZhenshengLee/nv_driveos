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
 * @brief <b> NVIDIA Media Interface: NvMedia Image Encode Processing API </b>
 *
 * This file contains the @ref 6x_image_encode_api "Image Encode Processing API".
 */

#ifndef NVMEDIA_IEP_H
#define NVMEDIA_IEP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "nvmedia_core.h"
#include "nvscisync.h"
#include "nvscibuf.h"
#include "nvmedia_common_encode.h"

/**
 * @defgroup 6x_image_encode_api Image Encoder
 *
 * The NvMediaIEP object takes uncompressed image data and turns it
 * into a codec specific bitstream. Only H.264, H.265 and VP9 encoding is supported.
 *
 * @ingroup 6x_nvmedia_image_top
 * @{
 */

/** @brief Major Version number */
#define NVMEDIA_IEP_VERSION_MAJOR   1
/** @brief Minor Version number */
#define NVMEDIA_IEP_VERSION_MINOR   0
/** @brief Patch Version number */
#define NVMEDIA_IEP_VERSION_PATCH   1

/**
 * Specifies the maximum number of times NvMediaIEPInsertPreNvSciSyncFence()
 * can be called before each call to NvMediaIEPFeedFrame().
 */
#define NVMEDIA_IEP_MAX_PRENVSCISYNCFENCES  (16U)

/**
 * @brief Image encode type
 */
typedef enum {
    /** @brief H.264 encode */
    NVMEDIA_IMAGE_ENCODE_H264,
    /** @brief HEVC codec */
    NVMEDIA_IMAGE_ENCODE_HEVC,
    /** @brief VP9 codec */
    NVMEDIA_IMAGE_ENCODE_VP9,
    /** @brief AV1 codec */
    NVMEDIA_IMAGE_ENCODE_AV1,
    NVMEDIA_IMAGE_ENCODE_END
} NvMediaIEPType;

/**
 * @brief Opaque NvMediaIEP object created by @ref NvMediaIEPCreate.
 */
typedef struct NvMediaIEP NvMediaIEP;

/**
 * @brief Retrieves the version information for the NvMedia IEP library.
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
 *   - Runtime: Yes
 *   - De-Init: Yes
 *
 * @param[in] version A pointer to a @ref NvMediaVersion structure
 *                    of the client.
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * -::NVMEDIA_STATUS_OK            Indicates a successful operation.
 * -::NVMEDIA_STATUS_BAD_PARAMETER Indicates an invalid or NULL argument.
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPGetVersion(
    NvMediaVersion *version
);

/**
 * @brief Create an NvMediaIEP object instance
 *
 * @pre NvMediaIEPGetVersion()
 * @pre NvMediaIEPFillNvSciBufAttrList()
 * @post NvMediaIEP object is created
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
 * Creates an NvMediaIEP object capable of turning a stream of YUV surfaces
 * characterized by @ref NvSciBufAttrList into a compressed bitstream of the
 * specified @ref NvMediaIEPType codec type. Surfaces are fed to the encoder
 * with @ref NvMediaIEPFeedFrame and generated bitstream buffers are retrieved
 * with @ref NvMediaIEPGetBits.
 *
 * @param[in] encodeType The video compression standard to be used for encoding
 *      \inputrange Entries in @ref NvMediaIEPType enumeration
 * @param[in] initParams The encoder initialization parameters
 *      \inputrange Pointer to a populated encode parameter structures:
 *      \n <b>QNX Safety build:</b>
 *      - ::NvMediaEncodeInitializeParamsH264 for NVMEDIA_IMAGE_ENCODE_H264
 *      \n <b>Non-safety build:</b>
 *      - ::NvMediaEncodeInitializeParamsH264 for NVMEDIA_IMAGE_ENCODE_H264
 *      - ::NvMediaEncodeInitializeParamsH265 for NVMEDIA_IMAGE_ENCODE_HEVC
 *      - ::NvMediaEncodeInitializeParamsVP9 for NVMEDIA_IMAGE_ENCODE_VP9
 *      - ::NvMediaEncodeInitializeParamsAV1 for NVMEDIA_IMAGE_ENCODE_AV1
 *              [Supported only from T234 onward]
 * @param[in] bufAttrList Pointer to a list of reconciled attributes that
 *      characterizes the input surface that needs to be encoded. \n
 *      Supported surface format attributes (common to all codecs): \n
 *      Buffer Type: NvSciBufType_Image \n
 *      Layout: NvSciBufImage_PitchLinearType \n
 *      Scan Type: NvSciBufScan_ProgressiveType \n
 *      Plane base address alignment: 256 \n
 *      Supported surface format attributes (codec specific): \n
 *      <b>QNX Safety build:</b> \n
 *          H.264: \n
 *              Sub-sampling type: YUV420, YUV444 (semi-planar) \n
 *              Bit Depth: 8 \n
 *      <b>Non-safety build:</b> \n
 *          H.264: \n
 *              Sub-sampling type: YUV420, YUV444 (semi-planar) \n
 *              Bit Depth: 8 \n
 *          H.265/HEVC: \n
 *              Sub-sampling type: YUV420, YUV444 (semi-planar) \n
 *              Bit Depth: 8, 10 \n
 *          VP9: \n
 *              Sub-sampling type: YUV420 (semi-planar) \n
 *              Bit Depth: 8, 12 \n
 *          AV1: \n
 *              Sub-sampling type: YUV420 (semi-planar) \n
 *              Bit Depth: 8, 10 \n
 *      \inputrange Non-NULL - valid pointer address obtained by a call to
 *      @ref NvSciBufObjGetAttrList called with a valid @ref NvSciBufObj that
 *      will contain the input content.
 * @param[in] maxBuffering
 *      Maximum number of frames outstanding at any given point in time that
 *      NvMediaIEP can hold before its output must be retrieved using
 *      @ref NvMediaIEPGetBits. If \a maxBuffering frames worth of encoded
 *      bitstream is yet to be retrived, @ref NvMediaIEPFeedFrame returns
 *      @ref NVMEDIA_STATUS_INSUFFICIENT_BUFFERING. In this case, encoded output
 *      of one or more frames must be retrived with @ref NvMediaIEPGetBits
 *      before feeding more frames using @ref NvMediaIEPFeedFrame
 *      \inputrange The values between 4 and 16, in increments of 1
 * @param[in] instanceId The ID of the NvENC HW engine instance
 *      \inputrange The following instances are supported:
 *      - ::NVMEDIA_ENCODER_INSTANCE_0
 *      - ::NVMEDIA_ENCODER_INSTANCE_1      [Supported only on T194]
 *      - ::NVMEDIA_ENCODER_INSTANCE_AUTO   [Supported only on T194]
 *      \n <b>Note</b>: @ref NVMEDIA_ENCODER_INSTANCE_AUTO initializes all
 *      supported NvENC HW engines
 *
 * @return The created @ref NvMediaIEP handle or NULL if unsuccessful.
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaIEP *
NvMediaIEPCreate(
    NvMediaIEPType encodeType,
    const void *initParams,
    NvSciBufAttrList bufAttrList,
    uint8_t maxBuffering,
    NvMediaEncoderInstanceId instanceId
);

#if !NV_IS_SAFETY
/**
 * @brief Create an NvMediaIEP object instance
 *
 * @note Supported only in non-safety build
 * @pre NvMediaIEPGetVersion()
 * @post NvMediaIEP object is created
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
 * @return @ref NvMediaIEP The new image encoder's handle or NULL if unsuccessful.
 */
NvMediaIEP *
NvMediaIEPCreateCtx(
    void
);

/**
 * @brief Initialize an NvMediaIEP object instance
 *
 * @note Supported only in non-safety build
 * @pre NvMediaIEPCreateCtx()
 * @post NvMediaIEP object is initialized.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIEP object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * Initializes an NvMediaIEP object capable of turning a stream of YUV surfaces
 * characterized by @ref NvSciBufAttrList into a compressed bitstream of the
 * specified @ref NvMediaIEPType codec type. Surfaces are fed to the encoder
 * with @ref NvMediaIEPFeedFrame and generated bitstream buffers are retrieved
 * with @ref NvMediaIEPGetBits.
 *
 * @param[in] encoder A pointer to the NvMediaIEP object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] encodeType The video compression standard to be used for encoding
 *      \inputrange Supported values from @ref NvMediaIEPType enumeration
 * @param[in] initParams The encoder initialization parameters
 *      \inputrange Pointer to a populated encode parameter structures:
 *      - <b>QNX Safety build:</b>
 *      - ::NvMediaEncodeInitializeParamsH264 for NVMEDIA_IMAGE_ENCODE_H264
 *      - <b>Non-safety build:</b>
 *      - ::NvMediaEncodeInitializeParamsH264 for NVMEDIA_IMAGE_ENCODE_H264
 *      - ::NvMediaEncodeInitializeParamsH265 for NVMEDIA_IMAGE_ENCODE_HEVC
 *      - ::NvMediaEncodeInitializeParamsVP9 for NVMEDIA_IMAGE_ENCODE_VP9
 *      - ::NvMediaEncodeInitializeParamsAV1 for NVMEDIA_IMAGE_ENCODE_AV1
 *              [Supported only from T234 onward]
 * @param[in] bufAttrList Pointer to a list of reconciled attributes that
 *      characterizes the input surface that needs to be encoded
 *      \inputrange Non-NULL - valid pointer address obtained by a call to
 *      @ref NvSciBufObjGetAttrList called with a valid @ref NvSciBufObj that
 *      will contain the input content
 *      \inputrange A valid NvSciBufAttrList
 * @param[in] maxBuffering
 *      Maximum number of frames outstanding at any given point in time that
 *      NvMediaIEP can hold before its output must be retrieved using
 *      @ref NvMediaIEPGetBits. If \a maxBuffering frames worth of encoded
 *      bitstream is yet to be retrived, @ref NvMediaIEPFeedFrame returns
 *      @ref NVMEDIA_STATUS_INSUFFICIENT_BUFFERING. In this case, encoded output
 *      of one or more frames must be retrived with @ref NvMediaIEPGetBits
 *      before feeding more frames using @ref NvMediaIEPFeedFrame
 *      \inputrange The values between 4 and 16, in increments of 1
 * @param[in] instanceId The ID of the NvENC HW engine instance
 *      \inputrange The following instances are supported:
 *      - ::NVMEDIA_ENCODER_INSTANCE_0
 *      - ::NVMEDIA_ENCODER_INSTANCE_1      [Supported only on T194]
 *      - ::NVMEDIA_ENCODER_INSTANCE_AUTO   [Supported only on T194]
 *      \n <b>Note</b>: @ref NVMEDIA_ENCODER_INSTANCE_AUTO initializes all
 *      supported NvENC HW engines
 *
 * @return ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - ::NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect system state.
 * - ::NVMEDIA_STATUS_OUT_OF_MEMORY in case of memory allocation failures.
 */
NvMediaStatus
NvMediaIEPInit(
    const NvMediaIEP *encoder,
    NvMediaIEPType encodeType,
    const void *initParams,
    NvSciBufAttrList bufAttrList,
    uint8_t maxBuffering,
    NvMediaEncoderInstanceId instanceId
);
#endif /* !NV_IS_SAFETY */

/**
 * @brief Destroys an NvMediaIEP object instance.
 *
 * @pre NvMediaIEPUnregisterNvSciBufObj()
 * @pre NvMediaIEPUnregisterNvSciSyncObj()
 * @post NvMediaIEP object is destroyed
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIEP object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * @param[in] encoder The NvMediaIEP object to destroy.
 *      \inputrange Non-NULL - valid pointer address
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
void NvMediaIEPDestroy(const NvMediaIEP *encoder);

/**
 * @brief Submits the specified frame for encoding
 *
 * The encoding process is asynchronous, as a result, the encoded output may
 * not be available when the API returns. Refer @ref NvMediaIEPBitsAvailable
 * and @ref NvMediaIEPGetBits for more details regarding how to retrieve
 * the encoded output.
 *
 * @pre NvMediaIEPRegisterNvSciBufObj()
 * @pre NvMediaIEPRegisterNvSciSyncObj()
 * @pre NvMediaIEPSetConfiguration() must be called at least once to configure
 *      NvMediaIEP
 * @pre NvMediaIEPSetNvSciSyncObjforEOF()
 * @pre NvMediaIEPInsertPreNvSciSyncFence()
 * @post Image encoding task is submitted
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIEP object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] encoder A pointer to the NvMediaIEP object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] bufObj Input bufObj that contains the input content that needs to
 *      be encoded, allocated with a call to @ref NvSciBufObjAlloc. The
 *      characteristics of the allocated NvSciBufObj should be equivalent to the
 *      bufAttrList passed in NvMediaIEPCreate. The size of the YUV surfaces
 *      contained in the NvSciBufObj need to be at least as large as the
 *      encoding width and height configured in the initialization parameters of
 *      @ref NvMediaIEPCreate.
 *      \inputrange A valid NvSciBufObj
 * @param[in] picParams Picture parameters used for the frame.
 *      \inputrange Supported picture parameter structures:
 *      - ::NvMediaEncodePicParamsH264 for NVMEDIA_IMAGE_ENCODE_H264
 *      - ::NvMediaEncodePicParamsH265 for NVMEDIA_IMAGE_ENCODE_HEVC
 *      - ::NvMediaEncodePicParamsVP9 for NVMEDIA_IMAGE_ENCODE_VP9
 *      - ::NvMediaEncodePicParamsAV1 for NVMEDIA_IMAGE_ENCODE_AV1
 * @param[in] instanceId The specific ID of the encoder engine instance where
 *      the encoding task needs to be submitted.
 *      \inputrange The following instances are supported:
 *      - ::NVMEDIA_ENCODER_INSTANCE_0
 *      - ::NVMEDIA_ENCODER_INSTANCE_1      [Supported only on T194]
 *      \n <b>Note</b>: If @ref NVMEDIA_ENCODER_INSTANCE_AUTO was \a not passed
 *          to @ref NvMediaIEPCreate/@ref NvMediaIEPInit API, then instanceId
 *          passed to the former needs to be passed to NvMediaIEPFeedFrame.
 *
 * @return @ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - ::NVMEDIA_STATUS_INSUFFICIENT_BUFFERING if the rate at which the input
 *      frames are being fed to NvMediaIEP using @ref NvMediaIEPFeedFrame is
 *      much greater than the rate at which the encoded output is being
 *      retrieved using @ref NvMediaIEPGetBits. This case typically happens
 *      when @ref NvMediaIEPGetBits has not been called frequently enough and
 *      the internal queue containing encoded/encoding frames awaiting to be
 *      retrieved is full. The \a maxBuffering argument passed to
 *      @ref NvMediaIEPCreate specifies the maximum number of calls that can be
 *      made to @ref NvMediaIEPFeedFrame after which at least one or more calls
 *      to @ref NvMediaIEPGetBits has to be made.
 * - ::NVMEDIA_STATUS_ERROR if there is a run-time error encountered during
 *      encoding
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPFeedFrame(
    const NvMediaIEP *encoder,
    NvSciBufObj bufObj,
    const void *picParams,
    NvMediaEncoderInstanceId instanceId
);

/**
 * @brief Sets the encoder configuration. The values in the configuration
 * take effect only at the start of the next GOP.
 *
 * @pre NvMediaIEPCreate()
 * @post NvMediaIEP object is configured
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIEP object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] encoder A pointer to the NvMediaIEP object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] configuration Configuration data.
 *      \inputrange Supported configuration structures:
 *      - ::NvMediaEncodeConfigH264 for NVMEDIA_IMAGE_ENCODE_H264
 *      - ::NvMediaEncodeConfigH265 for NVMEDIA_IMAGE_ENCODE_HEVC
 *      - ::NvMediaEncodeConfigVP9 for NVMEDIA_IMAGE_ENCODE_VP9
 *      - ::NvMediaEncodeConfigAV1 for NVMEDIA_IMAGE_ENCODE_AV1
 *
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPSetConfiguration(
    const NvMediaIEP *encoder,
    const void *configuration
);

/**
 * @brief Returns the bitstream for a slice or a frame.
 *
 *  It is safe to call the API to submit a task (@ref NvMediaIEPFeedFrame) and
 *  this function from two different threads.
 *
 *  The return value and behavior of @ref NvMediaIEPGetBits is the same as
 *  that of @ref NvMediaIEPBitsAvailable when called with
 *  @ref NVMEDIA_ENCODE_BLOCKING_TYPE_NEVER, except that when
 *  @ref NVMEDIA_STATUS_OK is returned, \a bitstreams is filled in addition
 *  to \a numBytes.
 *
 * @pre NvMediaIEPFeedFrame()
 * @post Encoded bitstream corresponding to the submitted task is retrieved.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIEP object.
 *   - Re-entrant: No
 *   - Async/Sync: Async
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @note This API writes encoded data corresponding to 1 input frame to
 * @ref bitstreams buffer passed by the application. In cases where there are
 * multiple encoder threads writing output to disk concurrently, the size of
 * the encoded bitstream data is large, or in general frequency of data write is
 * large, writing of the bitstream data to disk should be implemented/designed
 * such that the encoder submission thread does not get blocked. One way to do
 * this is to have two separate threads - one for input frame submission and the
 * other to check/wait on the output. Additionally, in some cases for reaching
 * maximum pipeline efficiency the application might need to reduce the number
 * of write calls to disk occurring, as 1 disk write occurs every frame per
 * encoder instance. One method to achieve this would be to have N successive
 * NvMediaIEPGetBitsEx() calls corresponding to one encoder instance
 * to write to a large contiguous buffer at defined offsets, post which the
 * data corresponding to 'N' frames can be written through 1 write call to disk.
 * This reduces the number of write calls to disk by a factor of N per encoder
 * instance.
 *
 * @param[in] encoder A pointer to the NvMediaIEP object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[out] numBytes Size of the filled bitstream.
 *      @note When extradata and MV_BUFFER_DUMP is enabled, the MV buffer data
 *      will be a part of the bitstream. But the \a numBytes will only reflect
 *      the size of the actual encoded output, and not that of the MV
 *      buffer dump.
 * @param[in] numBitstreamBuffers The length of the \a bitstreams array
 *      \inputrange 1 to 64 in steps of 1
 * @param[in,out] bitstreams A pointer to an array of type
 *      @ref NvMediaBitstreamBuffer, the length of the array being passed in
 *      \a numBitstreamBuffers. Encoded bitstream data will be filled in the
 *      entries of the array, starting from index 0 to
 *      (\a numBitstreamBuffers - 1), as and when each entry in the array gets
 *      filled up. The minimum combined bitstream size needs to be at least
 *      equal to the \a numBytesAvailable returned by
 *      @ref NvMediaIEPBitsAvailable call.
 *      \n Members of each @ref NvMediaBitstreamBuffer are to be set as follows:
 *      - ::NvMediaBitstreamBuffer.bitstream is assigned a pointer to an
 *          array of type uint8_t, where the encoded bitstream output will be
 *          written to by NvMediaIEP
 *      - ::NvMediaBitstreamBuffer.bitstreamSize is assigned with the size
 *          of the @ref NvMediaBitstreamBuffer.bitstream array
 *      - ::NvMediaBitstreamBuffer.bitstreamBytes will be populated by
 *          NvMediaIEP to indicate the number of encoded bitstream bytes that
 *          are populated in @ref NvMediaBitstreamBuffer.bitstream once the
 *          function call returns
 *      \inputrange Non-NULL - valid pointer address
 * @param[in,out] extradata Export encoding statistics. Pass a pointer
 *      to NvMediaEncodeOutputExtradata in order to populate encoding statistics.
 *      Refer NvMediaEncodeOutputExtradata documentation for more information on
 *      supported codecs, stats, etc. Pass NULL to disable this feature.
 *      @note This feature is not supported on QNX Safety.
 *      \inputrange Non-NULL - valid pointer address
 *
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - ::NVMEDIA_STATUS_PENDING if an encode is in progress but not yet completed.
 * - ::NVMEDIA_STATUS_NONE_PENDING if no encode is in progress.
 * - ::NVMEDIA_STATUS_INSUFFICIENT_BUFFERING if the size of the provided
 *         bitstream buffers are insufficient to hold the available output.
 * - ::NVMEDIA_STATUS_UNDEFINED_STATE if there was an internal error during
           encoding.
 * - ::NVMEDIA_STATUS_ERROR if there was an error during image encoding.
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPGetBits(
    const NvMediaIEP *encoder,
    uint32_t *numBytes,
    uint32_t numBitstreamBuffers,
    const NvMediaBitstreamBuffer *bitstreams,
    void *extradata
);

/**
 * @brief Returns the status of an encoding task submitted using
 * @ref NvMediaIEPFeedFrame, whose encoded output is to be retrieved next.
 *
 * The number of bytes of encoded output that is available (if ready), is also
 * retrieved along with the status. The specific behavior depends on the
 * specified @ref NvMediaBlockingType.
 *
 * It is safe to call the API to submit a task (@ref NvMediaIEPFeedFrame) and
 * this function from two different threads.
 *
 * @pre NvMediaIEPFeedFrame()
 * @post Status of the submitted encoding task is retrieved.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIEP object.
 *   - Re-entrant: No
 *   - Async/Sync: Async
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] encoder A pointer to the NvMediaIEP object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[out] numBytesAvailable
 *      The number of bytes of encoded output that is available. This output
 *      corresponds to the next encoding task in the queue for which
 *      the output is yet to be retrieved. The value is valid only when the
 *      return value from this API is @ref NVMEDIA_STATUS_OK.
 * @param[in] blockingType Blocking type.
 *      \inputrange The following are supported blocking types:
 * - ::NVMEDIA_ENCODE_BLOCKING_TYPE_NEVER
 *        This type never blocks so \a millisecondTimeout is ignored.
 *        The following are possible return values:  @ref NVMEDIA_STATUS_OK
 *        @ref NVMEDIA_STATUS_PENDING or @ref NVMEDIA_STATUS_NONE_PENDING.
 * \n
 * - ::NVMEDIA_ENCODE_BLOCKING_TYPE_IF_PENDING
 *        Blocks if an encoding task is pending, until it completes or till it
 *        times out. If no encoding task is pending, doesn't block and returns
 *        @ref NVMEDIA_STATUS_NONE_PENDING.
 *        Possible return values: @ref NVMEDIA_STATUS_OK,
 *        @ref NVMEDIA_STATUS_NONE_PENDING, @ref NVMEDIA_STATUS_TIMED_OUT.
 * @param[in] millisecondTimeout
 *       Timeout in milliseconds or @ref NVMEDIA_VIDEO_ENCODER_TIMEOUT_INFINITE
 *       if a timeout is not desired.
 *
 * @return @ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - ::NVMEDIA_STATUS_PENDING if an encode is in progress but not yet completed.
 * - ::NVMEDIA_STATUS_NONE_PENDING if no encode is in progress.
 * - ::NVMEDIA_STATUS_TIMED_OUT if the operation timed out.
 * - ::NVMEDIA_STATUS_UNDEFINED_STATE if there is an internal error during
 *         encoding
 * - ::NVMEDIA_STATUS_ERROR if there was an error during image encoding.
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPBitsAvailable(
    const NvMediaIEP *encoder,
    uint32_t *numBytesAvailable,
    NvMediaBlockingType blockingType,
    uint32_t millisecondTimeout
);


/**
 * @brief Gets the encoder attribute for the current encoding session.
 * This function can be called after passing the first frame for encoding.
 * It can be used to get header information (SPS/PPS/VPS) for the
 * current encoding session. Additionally, it can be extended for further
 * requirements, by implementing proper data structures.
 *
 * Before calling this function, you must pass the first frame for encoding.
 *
 * @pre NvMediaIEPFeedFrame() called at least once
 * @post Value of the required attribute is retrieved.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIEP object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] encoder A pointer to the NvMediaIEP object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] attrType Attribute type as defined in @ref NvMediaEncAttrType.
 *      \inputrange Supported values from @ref NvMediaEncAttrType enumeration
 * @param[in] attrSize Size of the data structure associated with attribute.
 *      \inputrange sizeof(NvMediaNalData)
 * @param[out] AttributeData A pointer to data structure associated with the attribute.
 *
 * @return @ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPGetAttribute(
    const NvMediaIEP *encoder,
    NvMediaEncAttrType attrType,
    uint32_t attrSize,
    void *AttributeData
);

/**
 * @brief Registers @ref NvSciBufObj for use with a NvMediaIEP handle.
 * NvMediaIEP handle maintains a record of all the objects registered using this
 * API and only the registered NvSciBufObj handles are accepted when submitted
 * for encoding via @ref NvMediaIEPFeedFrame. Even duplicated NvSciBufObj
 * objects need to be registered using this API prior.
 *
 * This is a mandatory API on QNX Safety Build to ensure deterministic execution
 * time of @ref NvMediaIEPFeedFrame. Although optional on other platform
 * configurations, it is highly recommended to use this API.
 *
 * Registration of the bufObj (input) is always with read-only permission.
 *
 * To ensure deterministic execution time of @ref NvMediaIEPFeedFrame API:
 * - @ref NvMediaIEPRegisterNvSciBufObj must be called for every input
 *   @ref NvSciBufObj that will be used with NvMediaIEP
 * - All @ref NvMediaIEPRegisterNvSciBufObj calls must be made before first
 *   @ref NvMediaIEPFeedFrame API call.
 *
 * Maximum of 32 @ref NvSciBufObj handles can be registered.
 *
 * @pre NvMediaIEPCreate()
 * @pre NvMediaIEPRegisterNvSciSyncObj()
 * @post NvSciBufObj is registered with NvMediaIEP object
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIEP object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @param[in] encoder A pointer to the NvMediaIEP object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] bufObj NvSciBufObj object
 *      \inputrange A valid NvSciBufObj
 *
 * @return @ref NvMediaStatus, the completion status of operation:
 * - @ref NVMEDIA_STATUS_OK if successful.
 * - @ref NVMEDIA_STATUS_BAD_PARAMETER if encoder, bufObj is invalid
 * - ::NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect system state.
 * - @ref NVMEDIA_STATUS_ERROR in following cases
 *          - user registers more than 32 bufObj.
 *          - user registers same bufObj multiple times.
 * - @ref NVMEDIA_STATUS_NOT_SUPPORTED if API is not functional
 *
 **/
NvMediaStatus
NvMediaIEPRegisterNvSciBufObj(
    const NvMediaIEP    *encoder,
    NvSciBufObj         bufObj
);

/**
 * @brief Un-registers @ref NvSciBufObj which was previously registered with
 * @ref NvMediaIEP using NvMediaIEPRegisterNvSciBufObj().
 *
 * For all @ref NvSciBufObj handles registered with NvMediaIEP using
 * @ref NvMediaIEPRegisterNvSciBufObj API, @ref NvMediaIEPUnregisterNvSciBufObj
 * must be called before calling @ref NvMediaIEPDestroy API. For unregistration
 * to succeed, it should be ensured that none of the submitted tasks on the
 * bufObj are pending prior to calling @ref NvMediaIEPUnregisterNvSciBufObj.
 * In order to ensure this, @ref NvMediaIEPGetBits API needs to be called
 * prior to unregistration, until the output of all the submitted tasks are
 * retrieved, following which @ref NvMediaIEPUnregisterNvSciSyncObj should be
 * called on all registered NvSciSyncObj.
 *
 * This is a mandatory API on QNX Safety Build to ensure deterministic execution
 * time of @ref NvMediaIEPFeedFrame. Although optional on other platform
 * configurations, it is highly recommended to use this API.
 *
 * To ensure deterministic execution time of @ref NvMediaIEPFeedFrame API:
 * - @ref NvMediaIEPUnregisterNvSciBufObj should be called only after the last
 *   @ref NvMediaIEPFeedFrame call
 *
 * @pre NvMediaIEPGetBits()
 * @pre NvMediaIEPUnregisterNvSciSyncObj() [verify that processing is complete]
 * @post NvSciBufObj is un-registered from NvMediaIEP object
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIEP object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * @param[in] encoder A pointer to the NvMediaIEP object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] bufObj NvSciBufObj object
 *      \inputrange A valid NvSciBufObj
 *
 * @return @ref NvMediaStatus, the completion status of operation:
 * - @ref NVMEDIA_STATUS_OK if successful.
 * - @ref NVMEDIA_STATUS_BAD_PARAMETER if encoder or bufObj is invalid
 * - ::NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect system state.
 * - @ref NVMEDIA_STATUS_ERROR in following cases:
 *          - User unregisters an NvSciBufObj which is not previously
 *            registered using %NvMediaIEPRegisterNvSciBufObj() API.
 *          - User unregisters an NvSciBufObj multiple times.
 **/
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPUnregisterNvSciBufObj(
    const NvMediaIEP    *encoder,
    NvSciBufObj         bufObj
);


/**
 * @brief Fills the NvMediaIEP specific NvSciBuf attributes which than then be
 * used to allocate an @ref NvSciBufObj that NvMediaIEP can consume.
 *
 * This function assumes that @a attrlist is a valid @ref NvSciBufAttrList
 * created by the caller by a call to @ref NvSciBufAttrListCreate.
 *
 * @pre NvMediaIEPGetVersion()
 * @post NvSciBufAttrList populated with NvMediaIEP specific NvSciBuf
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
 * @param[in] instanceId The ID of the NvENC HW engine instance
 *      \inputrange The following instances are supported:
 *      - ::NVMEDIA_ENCODER_INSTANCE_0
 *      - ::NVMEDIA_ENCODER_INSTANCE_1      [Supported only on T194]
 *      - ::NVMEDIA_ENCODER_INSTANCE_AUTO   [Supported only on T194]
 *      \n <b>Note</b>: @ref NVMEDIA_ENCODER_INSTANCE_AUTO initializes all
 *      supported NvENC HW engines
 * @param[out] attrlist A pointer to an %NvSciBufAttrList structure where
 *                NvMediaIEP places the NvSciBuf attributes.
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a attrlist is NULL
 * - ::NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect system state.
 */
NvMediaStatus
NvMediaIEPFillNvSciBufAttrList(
    NvMediaEncoderInstanceId  instanceId,
    NvSciBufAttrList          attrlist
);


/**
 * @brief Fills the NvMediaIEP specific NvSciSync attributes.
 *
 * This function assumes that @a attrlist is a valid @ref NvSciSyncAttrList.
 *
 * This function sets the public attribute:
 * - @ref NvSciSyncAttrKey_RequiredPerm
 *
 * The application must not set this attribute.
 *
 * @pre NvMediaIEPCreate()
 * @post NvSciSyncAttrList populated with NvMediaIEP specific NvSciSync
 *        attributes
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIEP object.
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
 * @param[in] encoder A pointer to the NvMediaIEP object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[out] attrlist A pointer to an %NvSciSyncAttrList structure where
 *                NvMedia places NvSciSync attributes.
 * @param[in] clienttype Indicates whether the NvSciSyncAttrList requested for
 *                an %NvMediaIEP signaler or an %NvMediaIEP waiter.
 *      \inputrange Entries in @ref NvMediaNvSciSyncClientType enumeration
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a attrlist is NULL,
 *         or any of the public attributes listed above are already set.
 * - ::NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect system state.
 * - ::NVMEDIA_STATUS_OUT_OF_MEMORY if there is not enough
 *         memory for the requested operation.
 */
NvMediaStatus
NvMediaIEPFillNvSciSyncAttrList(
    const NvMediaIEP           *encoder,
    NvSciSyncAttrList          attrlist,
    NvMediaNvSciSyncClientType clienttype
);


/**
 * @brief Registers an @ref NvSciSyncObj with NvMediaIEP.
 *
 * Every NvSciSyncObj (even duplicate objects) used by %NvMediaIEP
 * must be registered by a call to this function before it is used.
 * Only the exact same registered NvSciSyncObj can be passed to
 * NvMediaIEPSetNvSciSyncObjforEOF(), NvMediaIEPGetEOFNvSciSyncFence(), or
 * NvMediaIEPUnregisterNvSciSyncObj().
 *
 * For a given %NvMediaIEP handle,
 * one NvSciSyncObj can be registered as one @ref NvMediaNvSciSyncObjType only.
 * For each NvMediaNvSciSyncObjType, a maximum of 16 NvSciSyncObjs can
 * be registered.
 *
 * @pre NvMediaIEPFillNvSciSyncAttrList()
 * @post NvSciSyncObj registered with NvMediaIEP
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIEP object.
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
 * @param[in] encoder A pointer to the NvMediaIEP object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] syncobjtype Determines how @a nvscisync is used by @a encoder.
 *      \inputrange Entries in @ref NvMediaNvSciSyncObjType enumeration
 * @param[in] nvscisync The NvSciSyncObj to be registered with @a encoder.
 *      \inputrange A valid NvSciSyncObj
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a encoder is NULL or
 *         @a syncobjtype is not a valid NvMediaNvSciSyncObjType.
 * - ::NVMEDIA_STATUS_NOT_SUPPORTED if @a nvscisync is not a
 *         compatible NvSciSyncObj which %NvMediaIEP can support.
 * - ::NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect system state.
 * - ::NVMEDIA_STATUS_ERROR if the maximum number of NvSciScynObjs
 *         are already registered for the given @a syncobjtype, or
 *         if @a nvscisync is already registered with the same @a encoder
 *         handle for a different @a syncobjtype.
 */
NvMediaStatus
NvMediaIEPRegisterNvSciSyncObj(
    const NvMediaIEP           *encoder,
    NvMediaNvSciSyncObjType    syncobjtype,
    NvSciSyncObj               nvscisync
);

/**
 * @brief Unregisters an @ref NvSciSyncObj with NvMediaIEP.
 *
 * Every %NvSciSyncObj registered with %NvMediaIEP by
 * NvMediaIEPRegisterNvSciSyncObj() must be unregistered before calling
 * NvMediaIEPUnregisterNvSciBufObj() to unregister the NvSciBufObjs.
 *
 * Before the application calls this function, it must ensure that any
 * @ref NvMediaIEPFeedFrame() operation that uses the NvSciSyncObj has
 * completed. If this function is called while NvSciSyncObj is still
 * in use by any %NvMediaIEPFeedFrame() operation, the API returns
 * NVMEDIA_STATUS_PENDING to indicate the same. NvSciSyncFenceWait() API can
 * be called on the EOF NvSciSyncFence obtained post the last call to
 * NvMediaIEPFeedFrame() to wait for the associated tasks to complete.
 * The EOF NvSciSyncFence would have been previously obtained via a call to
 * NvMediaIEPGetEOFNvSciSyncFence(). The other option would be to call
 * NvMediaIEPGetBits() till there is no more output to retrieve.
 *
 * @pre NvMediaIEPFeedFrame()
 * @pre NvMediaIEPGetBits() or NvSciSyncFenceWait() [verify that processing is
 *                                                     complete]
 * @post NvSciSyncObj un-registered with NvMediaIEP
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIEP object.
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
 * @param[in] encoder A pointer to the NvMediaIEP object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] nvscisync An NvSciSyncObj to be unregistered with @a encoder.
 *      \inputrange A valid NvSciSyncObj
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if encoder is NULL, or
 *         @a nvscisync is not registered with @a encoder.
 * - @ref NVMEDIA_STATUS_PENDING if the @ref NvSciSyncObj is still in use, i.e.,
 *        the submitted task is still in progress. In this case, the application
 *        can choose to wait for operations to complete on the output surface
 *        using NvSciSyncFenceWait() or re-try the
 *        %NvMediaIEPUnregisterNvSciBufObj() API call, until the status
 *        returned is not @ref NVMEDIA_STATUS_PENDING.
 * - ::NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect system state.
 * - ::NVMEDIA_STATUS_ERROR if @a encoder was destroyed before this function
 *         was called.
 */
NvMediaStatus
NvMediaIEPUnregisterNvSciSyncObj(
    const NvMediaIEP  *encoder,
    NvSciSyncObj      nvscisync
);

/**
 * @brief Specifies the @ref NvSciSyncObj to be used for an EOF
 * @ref NvSciSyncFence.
 *
 * To use NvMediaIEPGetEOFNvSciSyncFence(), the application must call
 * %NvMediaIEPSetNvSciSyncObjforEOF() before it calls NvMediaIEPFeedFrame().
 *
 * %NvMediaIEPSetNvSciSyncObjforEOF() currently may be called only once before
 * each call to %NvMediaIEPFeedFrame(). The application may choose to call this
 * function only once before the first call to %NvMediaIEPFeedFrame().
 *
 * @pre NvMediaIEPRegisterNvSciSyncObj()
 * @post NvSciSyncObj to be used as EOF NvSciSyncFence is set
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIEP object.
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
 * @param[in] encoder A pointer to the NvMediaIEP object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] nvscisyncEOF A registered NvSciSyncObj which is to be
 *                           associated with EOF @ref NvSciSyncFence.
 *      \inputrange A valid NvSciSyncObj
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a encoder is NULL, or if @a nvscisyncEOF
 *         is not registered with @a encoder as either type
 *         @ref NVMEDIA_EOFSYNCOBJ or @ref NVMEDIA_EOF_PRESYNCOBJ.
 */
NvMediaStatus
NvMediaIEPSetNvSciSyncObjforEOF(
    const NvMediaIEP      *encoder,
    NvSciSyncObj          nvscisyncEOF
);

/**
 * @brief Sets an @ref NvSciSyncFence as a prefence for an
 * NvMediaIEPFeedFrame() %NvSciSyncFence operation.
 *
 * You must call %NvMediaIEPInsertPreNvSciSyncFence() before you call
 * %NvMediaIEPFeedFrame(). The %NvMediaIEPFeedFrame() operation is started only
 * after the expiry of the @a prenvscisyncfence.
 *
 * For example, in this sequence of code:
 * \code
 * nvmstatus = NvMediaIEPInsertPreNvSciSyncFence(handle, prenvscisyncfence);
 * nvmstatus = NvMediaIEPFeedFrame(handle, srcsurf, srcrect, picparams, instanceid);
 * \endcode
 * the %NvMediaIEPFeedFrame() operation is assured to start only after the
 * expiry of @a prenvscisyncfence.
 *
 * You can set a maximum of @ref NVMEDIA_IEP_MAX_PRENVSCISYNCFENCES prefences
 * by calling %NvMediaIEPInsertPreNvSciSyncFence() before %NvMediaIEPFeedFrame().
 * After the call to %NvMediaIEPFeedFrame(), all NvSciSyncFences previously
 * inserted by %NvMediaIEPInsertPreNvSciSyncFence() are removed, and they are not
 * reused for the subsequent %NvMediaIEPFeedFrame() calls.
 *
 * @pre Pre-NvSciSync fence obtained from previous engine in the pipeline
 * @post Pre-NvSciSync fence is set
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIEP object.
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
 * @param[in] encoder A pointer to the NvMediaIEP object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] prenvscisyncfence A pointer to %NvSciSyncFence.
 *      \inputrange Non-NULL - valid pointer address
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a encoder is not a valid %NvMediaIEP
 *     handle, or @a prenvscisyncfence is NULL, or if @a prenvscisyncfence was not
 *     generated with an @ref NvSciSyncObj that was registered with @a encoder as
 *     either @ref NVMEDIA_PRESYNCOBJ or @ref NVMEDIA_EOF_PRESYNCOBJ type.
 * - ::NVMEDIA_STATUS_NOT_SUPPORTED if %NvMediaIEPInsertPreNvSciSyncFence()
 *     has already been called at least %NVMEDIA_IEP_MAX_PRENVSCISYNCFENCES times
 *     with the same @a encoder handle before an %NvMediaIEPFeedFrame() call.
 */
NvMediaStatus
NvMediaIEPInsertPreNvSciSyncFence(
    const NvMediaIEP         *encoder,
    const NvSciSyncFence     *prenvscisyncfence
);

/**
 * @brief Gets EOF @ref NvSciSyncFence for an NvMediaIEPFeedFrame() operation.
 *
 * The EOF %NvSciSyncFence associated with an %NvMediaIEPFeedFrame() operation
 * is an %NvSciSyncFence. Its expiry indicates that the corresponding
 * %NvMediaIEPFeedFrame() operation has finished.
 *
 * This function returns the EOF %NvSciSyncFence associated
 * with the last %NvMediaIEPFeedFrame() call. %NvMediaIEPGetEOFNvSciSyncFence()
 * must be called after an %NvMediaIEPFeedFrame() call.
 *
 * For example, in this sequence of code:
 * \code
 * nvmstatus = NvMediaIEPFeedFrame(handle, srcsurf, srcrect, picparams, instanceid);
 * nvmstatus = NvMediaIEPGetEOFNvSciSyncFence(handle, nvscisyncEOF, eofnvscisyncfence);
 * \endcode
 * expiry of @a eofnvscisyncfence indicates that the preceding
 * %NvMediaIEPFeedFrame() operation has finished.
 *
 * @pre NvMediaIEPSetNvSciSyncObjforEOF()
 * @pre NvMediaIEPFeedFrame()
 * @post EOF NvSciSync fence for a submitted task is obtained
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIEP object.
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
 * @param[in] encoder A pointer to the NvMediaIEP object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] eofnvscisyncobj    An EOF NvSciSyncObj associated with the
 *                                 NvSciSyncFence which is being requested.
 *      \inputrange A valid NvSciSyncObj
 * @param[out] eofnvscisyncfence A pointer to the EOF NvSciSyncFence.
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a encoder is not a valid %NvMediaIEP
 *         handle, @a eofnvscisyncfence is NULL, or @a eofnvscisyncobj is not
 *         registered with @a encoder as type @ref NVMEDIA_EOFSYNCOBJ or
 *         @ref NVMEDIA_EOF_PRESYNCOBJ.
 * - ::NVMEDIA_STATUS_ERROR if the function was called before
 *         %NvMediaIEPFeedFrame() was called.
 */
NvMediaStatus
NvMediaIEPGetEOFNvSciSyncFence(
    const NvMediaIEP        *encoder,
    NvSciSyncObj      eofnvscisyncobj,
    NvSciSyncFence    *eofnvscisyncfence
);

/**
 * @brief Specifies the @ref NvSciSyncObj to be used for an SOF
 * @ref NvSciSyncFence.
 *
 * This function is not supported.
 *
 * To use NvMediaIEPGetSOFNvSciSyncFence(), the application must call
 * %NvMediaIEPSetNvSciSyncObjforSOF() before it calls NvMediaIEPFeedFrame().
 *
 * %NvMediaIEPSetNvSciSyncObjforSOF() currently may be called only once before
 * each call to %NvMediaIEPFeedFrame(). The application may choose to call this
 * function only once before the first call to %NvMediaIEPFeedFrame().
 *
 * @pre N/A
 * @post N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIEP object.s
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] encoder A pointer to the NvMediaIEP object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] nvscisyncSOF A registered NvSciSyncObj which is to be
 *                           associated with SOF @ref NvSciSyncFence.
 *      \inputrange A valid NvSciSyncObj
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a encoder is NULL, or if @a nvscisyncSOF
 *         is not registered with @a encoder as either type
 *         @ref NVMEDIA_SOFSYNCOBJ or @ref NVMEDIA_SOF_PRESYNCOBJ.
 */
NvMediaStatus
NvMediaIEPSetNvSciSyncObjforSOF(
    const NvMediaIEP        *encoder,
    NvSciSyncObj            nvscisyncSOF
);

/**
 * @brief Gets SOF @ref NvSciSyncFence for an NvMediaIEPFeedFrame() operation.
 *
 * This function is not supported.
 *
 * The SOF %NvSciSyncFence associated with an %NvMediaIEPFeedFrame() operation
 * is an %NvSciSyncFence. Its expiry indicates that the corresponding
 * %NvMediaIEPFeedFrame() operation has started.
 *
 * This function returns the SOF %NvSciSyncFence associated
 * with the last %NvMediaIEPFeedFrame() call. %NvMediaIEPGetSOFNvSciSyncFence()
 * must be called after an %NvMediaIEPFeedFrame() call.
 *
 * @pre N/A
 * @post N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIEP object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] encoder A pointer to the NvMediaIEP object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] sofnvscisyncobj    An SOF NvSciSyncObj associated with the
 *                                 NvSciSyncFence which is being requested.
 *      \inputrange A valid NvSciSyncObj
 * @param[out] sofnvscisyncfence A pointer to the SOF NvSciSyncFence.
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a encoder is not a valid %NvMediaIEP
 *         handle, @a sofnvscisyncfence is NULL, or @a sofnvscisyncobj is not
 *         registered with @a encoder as type @ref NVMEDIA_SOFSYNCOBJ or
 *         @ref NVMEDIA_SOF_PRESYNCOBJ.
 * - ::NVMEDIA_STATUS_ERROR if the function was called before
 *         %NvMediaIEPFeedFrame() was called.
 */
NvMediaStatus
NvMediaIEPGetSOFNvSciSyncFence(
    const NvMediaIEP        *encoder,
    NvSciSyncObj            sofnvscisyncobj,
    NvSciSyncFence          *sofnvscisyncfence
);

/*
 * @defgroup 6x_history_nvmedia_iep History
 * Provides change history for the NvMedia IEP API.
 *
 * \section 6x_history_nvmedia_iep Version History
 *
 * <b> Version 1.0 </b> September 28, 2021
 * - Initial release
 *
 * <b> Version 1.1 </b> August 03, 2022
 * - Added new quality preset API, NvMediaEncPreset
 *
 */

/** @} */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_IEP_H */
