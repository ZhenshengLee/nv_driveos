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
 * @file nvmedia_ijpe.h
 * @brief <b>NVIDIA Media Interface: Image JPEG Encode Processing API</b>
 *
 * @b Description: This file contains the @ref image_jpeg_encode_api "Image
 * JPEG Encode Processing API".
 */

#ifndef NVMEDIA_IJPE_H
#define NVMEDIA_IJPE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "nvmedia_core.h"
#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvmedia_common_encode_decode.h"
#include "nvmedia_common_encode.h"
/**
 * @defgroup 6x_image_jpeg_encode_api Image JPEG Encoder
 *
 * The NvMediaIJPE object takes uncompressed image data and turns it
 * into a jpeg bitstream.
 *
 * @ingroup 6x_nvmedia_image_top
 * @{
 */

/** @brief Major Version number */
#define NVMEDIA_IJPE_VERSION_MAJOR   1
/** @brief Minor Version number */
#define NVMEDIA_IJPE_VERSION_MINOR   0
/** @brief Patch Version number */
#define NVMEDIA_IJPE_VERSION_PATCH   0

/**
 * Specifies the maximum number of times NvMediaIJPEInsertPreNvSciSyncFence()
 * can be called before each call to NvMediaIJPEFeedFrame().
 */
#define NVMEDIA_IJPE_MAX_PRENVSCISYNCFENCES             (16U)

/**
 * \hideinitializer
 * @brief JPEG encoder flag empty
 */
#define NVMEDIA_JPEG_ENC_FLAG_NONE                      (0 << 0)
/**
 * \hideinitializer
 * @brief JPEG encoder flag to skip SOI marker. Normally, app inserts EXIF data in
 * jpeg image which occurs after SOI marker. This flag be used to skip SOI marker
 * insertion.
 */
#define NVMEDIA_JPEG_ENC_FLAG_SKIP_SOI                  (1 << 0)

/**
 * \hideinitializer
 * @brief Quality
 */
#define NVMEDIA_IMAGE_JPEG_ATTRIBUTE_QUALITY            (1 << 0)
/**
 * \hideinitializer
 * @brief Restart interval
 */
#define NVMEDIA_IMAGE_JPEG_ATTRIBUTE_RESTARTINTERVAL    (1 << 1)
/**
 * \hideinitializer
 * @brief encode frame target size
 */
#define NVMEDIA_IMAGE_JPEG_ATTRIBUTE_TARGETSIZE         (1 << 2)
/**
 * \hideinitializer
 * @brief Luma/Chroma quant table
 */
#define NVMEDIA_IMAGE_JPEG_ATTRIBUTE_QUANTTABLE         (1 << 3)
/**
 * \hideinitializer
 * @brief Huffmann table
 */
#define NVMEDIA_IMAGE_JPEG_ATTRIBUTE_HUFFTABLE          (1 << 4)

/**
 * @brief image JPEG encoder HuffmanTable.
 */
typedef struct {
   uint8_t length[16];
   uint8_t *values;
} NvMediaJPHuffmanTableSpecfication;

/**
 * @brief image JPEG encoder attributes.
 */
typedef struct {
/** @brief A value clamped to between 1 and 100, set to generate luma/chroma Quant tables.
 *  It is initialized to 0 at NvMediaIJPE creation, so encoder will use default standard
 *  Luma/Chroma quant tables. The corresponding attribute mask is @ref
 *  NVMEDIA_IMAGE_JPEG_ATTRIBUTE_QUALITY
 */
   uint8_t quality;
/** @brief A value to specify restart interval, initialized to 0 at NvMediaIJPE creation.
 *  The corresponding attribute mask is @ref NVMEDIA_IMAGE_JPEG_ATTRIBUTE_RESTARTINTERVAL
 */
   uint32_t restartInterval;
/** @brief A value to specify target encode size, initialized to 0xffffffff at NvMediaIJPE
 *  creation. The corresponding attribute mask is @ref NVMEDIA_IMAGE_JPEG_ATTRIBUTE_TARGETSIZE
 */
   uint32_t targetImageSize;
/** @brief A table for Luma quantization, initialized to default standard table at NvMediaIJPE
 *  creation. The corresponding attribute mask is @ref NVMEDIA_IMAGE_JPEG_ATTRIBUTE_QUANTTABLE
 */
   uint8_t lumaQuant[64];
/** @brief A table for Chroma quantization, initialized to default standard table at NvMediaIJPE
 *  creation. The corresponding attribute mask is @ref NVMEDIA_IMAGE_JPEG_ATTRIBUTE_QUANTTABLE
 */
   uint8_t chromaQuant[64];
/** @brief A Huffmann table for Luma DC sample, initialized to NULL for using default table
 *  in JPEG specification. The corresponding attribute mask is @ref
 *  NVMEDIA_IMAGE_JPEG_ATTRIBUTE_HUFFTABLE
 *  Only support in T210 or greater
 */
   NvMediaJPHuffmanTableSpecfication *lumaDC;
/** @brief A Huffmann table for Luma AC sample, initialized to NULL for using default table
 *  in JPEG specification. The corresponding attribute mask is @ref
 *  NVMEDIA_IMAGE_JPEG_ATTRIBUTE_HUFFTABLE
 *  Only support in T210 or greater
 */
   NvMediaJPHuffmanTableSpecfication *lumaAC;
/** @brief A Huffmann table for Chroma DC sample, initialized to NULL for using default table
 *  in JPEG specification. The corresponding attribute mask is @ref
 *  NVMEDIA_IMAGE_JPEG_ATTRIBUTE_HUFFTABLE
 *  Only support in T210 or greater
 */
   NvMediaJPHuffmanTableSpecfication *chromaDC;
/** @brief A Huffmann table for Chroma AC sample, initialized to NULL for using default table
 *  in JPEG specification. The corresponding attribute mask is @ref
 *  NVMEDIA_IMAGE_JPEG_ATTRIBUTE_HUFFTABLE
 *  Only support in T210 or greater
 */
   NvMediaJPHuffmanTableSpecfication *chromaAC;
} NvMediaJPEncAttributes;

/**
 * @brief An opaque NvMediaIJPE object created by @ref NvMediaIJPECreate.
 */
typedef struct NvMediaIJPE NvMediaIJPE;

/**
 * @brief Retrieves the version information for the NvMedia IJPE library.
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
 * @param[in] version A pointer to a @ref NvMediaVersion structure
 *                    of the client.
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if the pointer is invalid.
 */
NvMediaStatus
NvMediaIJPEGetVersion(
    NvMediaVersion *version
);

/**
 * @brief Creates a JPEG encoder object capable of turning a stream of surfaces
 * of the \a inputFormat into a JPEG stream.
 * Surfaces are fed to the encoder with NvMediaIJPEFeedFrame()
 * and bitstream buffers are retrieved with NvMediaIJPEGetBits().
 *
 * @pre NvMediaIJPEGetVersion()
 * @pre NvMediaIJPEFillNvSciBufAttrList()
 * @post NvMediaIJPE object is created
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
 * @param[in] bufAttrList NvSciBufAttrList that contains reconciled attributes
 *      that characterizes the input surface that needs to be encoded
 *      \inputrange An @ref NvSciBufObjGetAttrList called with a valid
 *      @ref NvSciBufObj that will contain the input content
 * @param[in] maxOutputBuffering
 *      This determines how many frames of encoded bitstream can be held
 *      by the @ref NvMediaIJPE object before it must be retrieved using
 *      NvMediaIJPEGetBits(). If \a maxOutputBuffering frames worth
 *      of encoded bitstream are yet unretrieved by @c %NvMediaIJPEGetBits(),
 *      then NvMediaIJPEFeedFrame() returns
 *      @ref NVMEDIA_STATUS_INSUFFICIENT_BUFFERING.  One or more frames must
 *      be retrieved with \c %NvMediaIJPEGetBits() before frame feeding
 *      can continue.
 *      \inputrange The values between 1 and 16, in increments of 1
 * @param[in] maxBitstreamBytes Determines the maximum bytes that JPEG
 *      encoder can produce for each feed frame.
 * @param[in] instanceId The ID of the engine instance.
 *      \inputrange The following instances are supported:
 *      - ::NVMEDIA_JPEG_INSTANCE_0
 *      - ::NVMEDIA_JPEG_INSTANCE_1     [Supported only on T23X]
 *      - ::NVMEDIA_JPEG_INSTANCE_AUTO  [Supported only on T23X]
 * @retval NvMediaIJPE The new image JPEG encoder device's handle
 * or \c NULL if unsuccessful.
 */
NvMediaIJPE *
NvMediaIJPECreate(
    //NvMediaSurfaceType inputFormat,
    NvSciBufAttrList bufAttrList,
    uint8_t maxOutputBuffering,
    uint32_t  maxBitstreamBytes,
    NvMediaJPEGInstanceId instanceId
);

/**
 * @brief Destroys an NvMedia image JPEG encoder
 *
 * @pre NvMediaIJPEUnregisterNvSciBufObj()
 * @pre NvMediaIJPEUnregisterNvSciSyncObj()
 * @post NvMediaIJPE object is destroyed
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * @param[in] encoder The JPEG encoder to destroy.
 */
void NvMediaIJPEDestroy(NvMediaIJPE *encoder);

/**
 * @brief Encodes the specified \a bufObj with input quality.
 * 
 * @pre NvMediaIJPERegisterNvSciBufObj()
 * @pre NvMediaIJPERegisterNvSciSyncObj()
 * @pre NvMediaIJPESetNvSciSyncObjforEOF()
 * @pre NvMediaIJPEInsertPreNvSciSyncFence()
 * @pre NvMediaIJPESetAttributes() must be called at least once to configure
 *      NvMediaIJPE
 * @post JPEG encoding task is submitted
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Async
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 * 
 * @param[in] encoder The encoder to use.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] bufObj Input bufObj that contains the input content that needs to
 *      be encoded, allocated with a call to @ref NvSciBufObjAlloc. The
 *      characteristics of the allocated NvSciBufObj should be equivalent to the
 *      bufAttrList passed in NvMediaIJPECreate. There is no limit on the size
 *      of this surface.
 *      \inputrange A valid NvSciBufObj
 * @param[in] quality
 *      This specifies the encode quality. JPEG encode will generate quant tables
 *      for luma and chroma according to the quality value
 *      \inputrange The values between 1 and 100, in increments of 1
 * @param[in] instanceId The ID of the engine instance.
 *      \inputrange The following instances are supported:
 *      - ::NVMEDIA_JPEG_INSTANCE_0
 *      - ::NVMEDIA_JPEG_INSTANCE_1     [Supported only on T23X]
 *      - ::NVMEDIA_JPEG_INSTANCE_AUTO  [Supported only on T23X]
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL or invalid.
 * - ::NVMEDIA_STATUS_INSUFFICIENT_BUFFERING if NvMediaIJPEGetBits()
 *  has not been called frequently enough and the maximum internal
 *  bitstream buffering (determined by \a maxOutputBuffering passed to
 *  NvMediaIJPECreate()) has been exhausted.
 */
NvMediaStatus
NvMediaIJPEFeedFrame(
    const NvMediaIJPE *encoder,
    NvSciBufObj bufObj,
    uint8_t quality,
    NvMediaJPEGInstanceId instanceId
);

/**
 * @brief Encodes the specified \a bufObj with input Luma and Chroma quant tables.
 * 
 * @pre NvMediaIJPERegisterNvSciBufObj()
 * @pre NvMediaIJPERegisterNvSciSyncObj()
 * @pre NvMediaIJPESetNvSciSyncObjforEOF()
 * @pre NvMediaIJPEInsertPreNvSciSyncFence()
 * @pre NvMediaIJPESetAttributes() must be called at least once to configure
 *      NvMediaIJPE
 * @post JPEG encoding task is submitted
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Async
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 * 
 * @param[in] encoder The encoder to use.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] bufObj Input bufObj that contains the input content that needs to
 *      be encoded, allocated with a call to @ref NvSciBufObjAlloc. The
 *      characteristics of the allocated NvSciBufObj should be equivalent to the
 *      bufAttrList passed in NvMediaIJPECreate. There is no limit on the size
 *      of this surface.
 *      \inputrange A valid NvSciBufObj
 * @param[in] lumaQuant
 *      This specifies Luma quant table used for encode
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] chromaQuant
 *      This specifies Chroma quant table used for encode
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] instanceId The ID of the engine instance.
 *      \inputrange The following instances are supported:
 *      - ::NVMEDIA_JPEG_INSTANCE_0
 *      - ::NVMEDIA_JPEG_INSTANCE_1     [Supported only on T23X]
 *      - ::NVMEDIA_JPEG_INSTANCE_AUTO  [Supported only on T23X]
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if successful.
 * - ::NVMEDIA_STATUS_INSUFFICIENT_BUFFERING
 *    if NvMediaIJPEGetBits() has not been called frequently enough and the
 *    maximum internal bitstream buffering (determined by \a maxOutputBuffering
 *    passed to @ref NvMediaIJPECreate()) has been exhausted.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL or invalid.
 */
NvMediaStatus
NvMediaIJPEFeedFrameQuant(
    const NvMediaIJPE *encoder,
    NvSciBufObj bufObj,
    uint8_t *lumaQuant,
    uint8_t *chromaQuant,
    NvMediaJPEGInstanceId instanceId
);

/**
 * @brief Encodes the specified \a bufObj with input Luma and Chroma quant tables
 *  and targetImageSize.
 *
 * @pre NvMediaIJPERegisterNvSciBufObj()
 * @pre NvMediaIJPERegisterNvSciSyncObj()
 * @pre NvMediaIJPESetNvSciSyncObjforEOF()
 * @pre NvMediaIJPEInsertPreNvSciSyncFence()
 * @pre NvMediaIJPESetAttributes() must be called at least once to configure
 *      NvMediaIJPE
 * @post JPEG encoding task is submitted
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Async
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 * 
 * @param[in] encoder The encoder to use.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] bufObj Input bufObj that contains the input content that needs to
 *      be encoded, allocated with a call to @ref NvSciBufObjAlloc. The
 *      characteristics of the allocated NvSciBufObj should be equivalent to the
 *      bufAttrList passed in NvMediaIJPECreate. There is no limit on the size
 *      of this surface.
 *      \inputrange A valid NvSciBufObj
 * @param[in] lumaQuant
 *      This specifies Luma quant table used for encode
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] chromaQuant
 *      This specifies Chroma quant table used for encode
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] targetImageSize
 *      This specifies target image size in bytes
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] instanceId The ID of the engine instance.
 *      \inputrange The following instances are supported:
 *      - ::NVMEDIA_JPEG_INSTANCE_0
 *      - ::NVMEDIA_JPEG_INSTANCE_1     [Supported only on T23X]
 *      - ::NVMEDIA_JPEG_INSTANCE_AUTO  [Supported only on T23X]
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if successful.
 * - ::NVMEDIA_STATUS_INSUFFICIENT_BUFFERING if NvMediaIJPEGetBits()
 *  has not been called frequently enough and the maximum internal
 *  bitstream buffering (determined by \a maxOutputBuffering passed to
 *  @ref NvMediaIJPECreate()) has been exhausted.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL or invalid.
 */
NvMediaStatus
NvMediaIJPEFeedFrameRateControl(
    const NvMediaIJPE *encoder,
    NvSciBufObj bufObj,
    uint8_t *lumaQuant,
    uint8_t *chromaQuant,
    uint32_t targetImageSize,
    NvMediaJPEGInstanceId instanceId
);

/**
 * @brief Sets the JPEG encoder attributes. These go into effect at
 * the next encode frame.
 * 
 *
 * @pre NvMediaIJPECreate()
 * @post NvMediaIJPE object is configured
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 * 
 * @param[in] encoder The encoder to use.
 * @param[in] attributeMask Attribute mask.
 * @param[in] attributes Attributes data.
 * \n Supported attribute structures:
 * - ::NvMediaJPEncAttributes
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if an input parameter is NULL.
 */
NvMediaStatus
NvMediaIJPESetAttributes(
    const NvMediaIJPE *encoder,
    uint32_t attributeMask,
    const void *attributes
);

/**
 * @brief Returns a frame's worth of bitstream
 *  into the provided \a buffer. \n numBytes returns the size of this
 *  bitstream.  It is safe to call this function from a separate thread.
 *  The return value and behavior is the same as that of
 *  NvMediaIJPEBitsAvailable() when called with @ref NVMEDIA_ENCODE_BLOCKING_TYPE_NEVER
 *  except that when @ref NVMEDIA_STATUS_OK is returned, the \a buffer will be
 *  filled in addition to the \a numBytes.
 *
 * Before calling this function:
 * 1. Call NvMediaIJPEBitsAvailable() to determine the number of bytes
 *    required for the next frame.
 * 2. Allocate a buffer that can hold the next frame.
 * 
 * @pre NvMediaIJPEBitsAvailable()
 * @pre NvMediaIJPEFeedFrame() or NvMediaIJPEFeedFrameQuant() or NvMediaIJPEFeedFrameRateControl()
 * @post Encoded bitstream corresponding to the submitted task is retrieved.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 * 
 * @param[in] encoder The encoder to use.
 * @param[out] numBytes
 *       Returns the size of the filled bitstream.
 * @param[in,out] buffer
 *      The buffer to be filled with the encoded data. If buffer is NULL, this function
 *      returns without copying the encoded bitstream.
 * @param[in] flags
 *      The flags for special handlings
 *      Current support flag @ref NVMEDIA_JPEG_ENC_FLAG_NONE or
 *                           @ref NVMEDIA_JPEG_ENC_FLAG_SKIP_SOI
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL.
 * - ::NVMEDIA_STATUS_PENDING if an encode is in progress but not yet completed.
 * - ::NVMEDIA_STATUS_NONE_PENDING if no encode is in progress.
 */
NvMediaStatus
NvMediaIJPEGetBits(
    const NvMediaIJPE *encoder,
    uint32_t *numBytes,
    void *buffer,
    uint32_t flags
);

/**
 * Performs the same operation as NvMediaIJPEGetBits() except
 * NvMediaIJPEGetBits() consumes an array of buffers. Please see NvMediaIJPEGetBits()
 * for usage.
 *
 * @pre NvMediaIJPEBitsAvailable()
 * @pre NvMediaIJPEFeedFrame() or NvMediaIJPEFeedFrameQuant() or NvMediaIJPEFeedFrameRateControl()
 * @post Encoded bitstream corresponding to the submitted task is retrieved.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 * 
 * @param[in] encoder The encoder to use.
 * @param[out] numBytes
 *       Returns the size of the filled bitstream.
 * @param[in] numBitstreamBuffers
 *       Number of buffers provided to encoder
 * @param[in,out] bitstreams
 *      Pointer pointing to the array of buffers of type NvMediaBitstreamBuffer.
 *      These buffer to be filled with the encoded data. If it is NULL, this function
 *      returns without copying the encoded bitstreams.
 * @param[in] flags
 *      The flags for special handlings
 *      Current support flag @ref NVMEDIA_JPEG_ENC_FLAG_NONE or
 *                           @ref NVMEDIA_JPEG_ENC_FLAG_SKIP_SOI
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL.
 * - ::NVMEDIA_STATUS_PENDING if an encode is in progress but not yet completed.
 * - ::NVMEDIA_STATUS_NONE_PENDING if no encode is in progress.
 */
NvMediaStatus
NvMediaIJPEGetBitsEx(
    const NvMediaIJPE *encoder,
    uint32_t *numBytes,
    uint32_t numBitstreamBuffers,
    const NvMediaBitstreamBuffer *bitstreams,
    uint32_t flags
);

/**
 * @brief Returns the encode status
 *  and number of bytes available for the next frame (if any).
 *  The specific behavior depends on the specified \a blockingType.
 *  It is safe to call this function from a separate thread.
 *
 * @pre NvMediaIJPEFeedFrame() or NvMediaIJPEFeedFrameQuant() or NvMediaIJPEFeedFrameRateControl()
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 * 
 * @param[in] encoder The encoder to use.
 * @param[in] numBytesAvailable
 *      The number of bytes available in the next encoded frame. This
 *      is valid only when the return value is @ref NVMEDIA_STATUS_OK.
 * @param[in] blockingType
 *      The following are the supported blocking types:
 * \li @ref NVMEDIA_ENCODE_BLOCKING_TYPE_NEVER
 *    \n  This type never blocks. As a result, \a millisecondTimeout is ignored.
 *        With this \a blockingType value, the following return values are possible:
 *        - ::NVMEDIA_STATUS_OK
 *        - ::NVMEDIA_STATUS_PENDING
 *        - ::NVMEDIA_STATUS_NONE_PENDING
 * \li @ref NVMEDIA_ENCODE_BLOCKING_TYPE_IF_PENDING
 *     \n Same as \c NVMEDIA_ENCODE_BLOCKING_TYPE_NEVER except that
 *        the function never returns with \c NVMEDIA_STATUS_PENDING.
 *        If an encode is pending, then
 *        this function blocks until the status changes to
 *        \c NVMEDIA_STATUS_OK or until the timeout occurs.
 *        With this \a blockingType value, the following return values are possible:
 *        - ::NVMEDIA_STATUS_OK
 *        - ::NVMEDIA_STATUS_NONE_PENDING
 *        - ::NVMEDIA_STATUS_TIMED_OUT
 * @param[in] millisecondTimeout
 *       Timeout in milliseconds or @ref NVMEDIA_VIDEO_ENCODER_TIMEOUT_INFINITE
 *       if a timeout is not desired.
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL.
 * - ::NVMEDIA_STATUS_PENDING if an encode is in progress but not yet completed.
 * - ::NVMEDIA_STATUS_NONE_PENDING if no encode is in progress.
 * - ::NVMEDIA_STATUS_TIMED_OUT if the operation timed out.
 */
NvMediaStatus
NvMediaIJPEBitsAvailable(
    const NvMediaIJPE *encoder,
    uint32_t *numBytesAvailable,
    NvMediaBlockingType blockingType,
    uint32_t millisecondTimeout
);

/**
 * @brief Registers @ref NvSciBufObj for use with a NvMediaIJPE handle.
 * NvMediaIJPE handle maintains a record of all the objects registered using this
 * API and only the registered NvSciBufObj handles are accepted when submitted
 * for encoding via @ref NvMediaIJPEFeedFrame. Even duplicated NvSciBufObj
 * objects need to be registered using this API prior.
 *
 * This needs to be used in tandem with NvMediaIJPEUnregisterNvSciBufObj(). The
 * pair of APIs for registering and unregistering NvSciBufObj are optional, but
 * it is highly recommended to use them as they ensure deterministic execution
 * of NvMediaIJPEFeedFrame().
 *
 * To ensure deterministic execution time of @ref NvMediaIJPEFeedFrame API:
 * - @ref NvMediaIJPERegisterNvSciBufObj must be called for every input
 *   @ref NvSciBufObj that will be used with NvMediaIJPE
 * - All @ref NvMediaIJPERegisterNvSciBufObj calls must be made before first
 *   @ref NvMediaIJPEFeedFrame API call.
 *
 * Registration of the bufObj (input) is always with read-only permission.
 *
 * Maximum of 32 @ref NvSciBufObj handles can be registered.
 *
 * @pre NvMediaIJPECreate()
 * @pre NvMediaIJPERegisterNvSciSyncObj()
 * @post NvSciBufObj is registered with NvMediaIJPE object
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @param[in] encoder A pointer to the NvMediaIJPE object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] bufObj NvSciBufObj object
 *      \inputrange A valid NvSciBufObj
 *
 * @return ::NvMediaStatus, the completion status of operation:
 * - ::NVMEDIA_STATUS_OK if successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if encoder, bufObj is invalid
 * - ::NVMEDIA_STATUS_ERROR in following cases
 *          - user registers more than 32 bufObj.
 *          - user registers same bufObj multiple times.
 * - ::NVMEDIA_STATUS_NOT_SUPPORTED if API is not functional
 *
 **/
NvMediaStatus
NvMediaIJPERegisterNvSciBufObj(
    const NvMediaIJPE   *encoder,
    NvSciBufObj         bufObj
);

/**
 * @brief Un-registers @ref NvSciBufObj which was previously registered with
 * @ref NvMediaIJPE using NvMediaIJPERegisterNvSciBufObj().
 *
 * For all @ref NvSciBufObj handles registered with NvMediaIJPE using
 * @ref NvMediaIJPERegisterNvSciBufObj API, @ref NvMediaIJPEUnregisterNvSciBufObj
 * must be called before calling @ref NvMediaIJPEDestroy API. For unregistration
 * to succeed, it should be ensured that none of the submitted tasks on the
 * bufObj are pending prior to calling @ref NvMediaIJPEUnregisterNvSciBufObj.
 * In order to ensure this, @ref NvMediaIJPEGetBits API needs to be called
 * prior to unregistration, until the output of all the submitted tasks are
 * retrieved, following which @ref NvMediaIJPEUnregisterNvSciSyncObj should be
 * called on all registered NvSciSyncObj.
 *
 * This needs to be used in tandem with NvMediaIJPERegisterNvSciBufObj(). The
 * pair of APIs for registering and unregistering NvSciBufObj are optional, but
 * it is highly recommended to use them as they ensure deterministic execution
 * of NvMediaIJPEFeedFrame().
 *
 * To ensure deterministic execution time of @ref NvMediaIJPEFeedFrame API:
 * - @ref NvMediaIJPEUnregisterNvSciBufObj should be called only after the last
 *   @ref NvMediaIJPEFeedFrame call
 *
 * @pre NvMediaIJPEGetBits()
 * @pre NvMediaIJPEUnregisterNvSciSyncObj() [verify that processing is complete]
 * @post NvSciBufObj is un-registered from NvMediaIJPE object
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * @param[in] encoder A pointer to the NvMediaIJPE object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] bufObj NvSciBufObj object
 *      \inputrange A valid NvSciBufObj
 *
 * @return ::NvMediaStatus, the completion status of operation:
 * - ::NVMEDIA_STATUS_OK if successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if encoder or bufObj is invalid
 * - ::NVMEDIA_STATUS_ERROR in following cases:
 *          - User unregisters an NvSciBufObj which is not previously
 *            registered using %NvMediaIJPERegisterNvSciBufObj() API.
 *          - User unregisters an NvSciBufObj multiple times.
 **/
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIJPEUnregisterNvSciBufObj(
    const NvMediaIJPE    *encoder,
    NvSciBufObj          bufObj
);


/**
 * @brief Fills the NvMediaIJPE specific NvSciBuf attributes which than then be
 * used to allocate an @ref NvSciBufObj that NvMediaIJPE can consume.
 *
 * This function assumes that @a attrlist is a valid @ref NvSciBufAttrList
 * created by the caller by a call to @ref NvSciBufAttrListCreate.
 *
 * @pre NvMediaIJPEGetVersion()
 * @post NvSciBufAttrList populated with NvMediaIJPE specific NvSciBuf
 *      attributes. The caller can then set attributes specific to the type of
 *      surface, reconcile attribute lists and allocate an NvSciBufObj.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
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
 * @param[out] attrlist A pointer to an %NvSciBufAttrList structure where
 *                NvMediaIJPE places the NvSciBuf attributes.
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a attrlist is NULL
 */
NvMediaStatus
NvMediaIJPEFillNvSciBufAttrList(
    NvMediaJPEGInstanceId     instanceId,
    NvSciBufAttrList          attrlist
);

/**
 * @brief Fills the NvMediaIJPE specific NvSciSync attributes.
 *
 * This function assumes that @a attrlist is a valid @ref NvSciSyncAttrList.
 *
 * This function sets the public attribute:
 * - @ref NvSciSyncAttrKey_RequiredPerm
 *
 * The application must not set this attribute.
 *
 * @pre NvMediaIJPECreate()
 * @post NvSciSyncAttrList populated with NvMediaIJPE specific NvSciSync
 *        attributes
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] encoder A pointer to the NvMediaIJPE object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[out] attrlist A pointer to an %NvSciSyncAttrList structure where
 *                NvMedia places NvSciSync attributes.
 * @param[in] clienttype Indicates whether the NvSciSyncAttrList requested for
 *                an %NvMediaIJPE signaler or an %NvMediaIJPE waiter.
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
NvMediaIJPEFillNvSciSyncAttrList(
    const NvMediaIJPE           *encoder,
    NvSciSyncAttrList          attrlist,
    NvMediaNvSciSyncClientType clienttype
);


/**
 * @brief Registers an @ref NvSciSyncObj with NvMediaIJPE.
 *
 * Every NvSciSyncObj (even duplicate objects) used by %NvMediaIJPE
 * must be registered by a call to this function before it is used.
 * Only the exact same registered NvSciSyncObj can be passed to
 * NvMediaIJPESetNvSciSyncObjforEOF(), NvMediaIJPEGetEOFNvSciSyncFence(), or
 * NvMediaIJPEUnregisterNvSciSyncObj().
 *
 * For a given %NvMediaIJPE handle,
 * one NvSciSyncObj can be registered as one @ref NvMediaNvSciSyncObjType only.
 * For each NvMediaNvSciSyncObjType, a maximum of 16 NvSciSyncObjs can
 * be registered.
 *
 * @pre NvMediaIJPEFillNvSciSyncAttrList()
 * @post NvSciSyncObj registered with NvMediaIJPE
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] encoder A pointer to the NvMediaIJPE object.
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
 *         compatible NvSciSyncObj which %NvMediaIJPE can support.
 * - ::NVMEDIA_STATUS_ERROR if the maximum number of NvSciScynObjs
 *         are already registered for the given @a syncobjtype, or
 *         if @a nvscisync is already registered with the same @a encoder
 *         handle for a different @a syncobjtype.
 */
NvMediaStatus
NvMediaIJPERegisterNvSciSyncObj(
    const NvMediaIJPE           *encoder,
    NvMediaNvSciSyncObjType    syncobjtype,
    NvSciSyncObj               nvscisync
);

/**
 * @brief Unregisters an @ref NvSciSyncObj with NvMediaIJPE.
 *
 * Every %NvSciSyncObj registered with %NvMediaIJPE by
 * NvMediaIJPERegisterNvSciSyncObj() must be unregistered before calling
 * NvMediaIJPEUnregisterNvSciBufObj() to unregister the NvSciBufObjs.
 *
 * Before the application calls this function, it must ensure that any
 * @ref NvMediaIJPEFeedFrame() operation that uses the NvSciSyncObj has
 * completed. If this function is called while NvSciSyncObj is still
 * in use by any %NvMediaIJPEFeedFrame() operation, the API returns
 * NVMEDIA_STATUS_PENDING to indicate the same. NvSciSyncFenceWait() API can
 * be called on the EOF NvSciSyncFence obtained post the last call to
 * NvMediaIJPEFeedFrame() to wait for the associated tasks to complete.
 * The EOF NvSciSyncFence would have been previously obtained via a call to
 * NvMediaIJPEGetEOFNvSciSyncFence(). The other option would be to call
 * NvMediaIJPEGetBits() till there is no more output to retrieve.
 *
 * @pre NvMediaIJPEFeedFrame()
 * @pre NvMediaIJPEGetBits() or NvSciSyncFenceWait() [verify that processing is
 *                                                     complete]
 * @post NvSciSyncObj un-registered with NvMediaIJPE
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * \note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] encoder A pointer to the NvMediaIJPE object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] nvscisync An NvSciSyncObj to be unregistered with @a encoder.
 *      \inputrange A valid NvSciSyncObj
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if encoder is NULL, or
 *         @a nvscisync is not registered with @a encoder.
 * - ::NVMEDIA_STATUS_PENDING if the @ref NvSciSyncObj is still in use, i.e.,
 *        the submitted task is still in progress. In this case, the application
 *        can choose to wait for operations to complete on the output surface
 *        using NvSciSyncFenceWait() or re-try the
 *        %NvMediaIJPEUnregisterNvSciBufObj() API call, until the status
 *        returned is not @ref NVMEDIA_STATUS_PENDING.
 * - ::NVMEDIA_STATUS_ERROR if @a encoder was destroyed before this function
 *         was called.
 */
NvMediaStatus
NvMediaIJPEUnregisterNvSciSyncObj(
    const NvMediaIJPE  *encoder,
    NvSciSyncObj      nvscisync
);

/**
 * @brief Specifies the @ref NvSciSyncObj to be used for an EOF
 * @ref NvSciSyncFence.
 *
 * To use NvMediaIJPEGetEOFNvSciSyncFence(), the application must call
 * %NvMediaIJPESetNvSciSyncObjforEOF() before it calls NvMediaIJPEFeedFrame().
 *
 * %NvMediaIJPESetNvSciSyncObjforEOF() currently may be called only once before
 * each call to %NvMediaIJPEFeedFrame(). The application may choose to call this
 * function only once before the first call to %NvMediaIJPEFeedFrame().
 *
 * @pre NvMediaIJPERegisterNvSciSyncObj()
 * @post NvSciSyncObj to be used as EOF NvSciSyncFence is set
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] encoder A pointer to the NvMediaIJPE object.
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
NvMediaIJPESetNvSciSyncObjforEOF(
    const NvMediaIJPE      *encoder,
    NvSciSyncObj          nvscisyncEOF
);

/**
 * @brief Sets an @ref NvSciSyncFence as a prefence for an
 * NvMediaIJPEFeedFrame() %NvSciSyncFence operation.
 *
 * You must call %NvMediaIJPEInsertPreNvSciSyncFence() before you call
 * %NvMediaIJPEFeedFrame(). The %NvMediaIJPEFeedFrame() operation is started only
 * after the expiry of the @a prenvscisyncfence.
 *
 * For example, in this sequence of code:
 * \code
 * nvmstatus = NvMediaIJPEInsertPreNvSciSyncFence(handle, prenvscisyncfence);
 * nvmstatus = NvMediaIJPEFeedFrame(handle, srcsurf, srcrect, picparams, instanceid);
 * \endcode
 * the %NvMediaIJPEFeedFrame() operation is assured to start only after the
 * expiry of @a prenvscisyncfence.
 *
 * You can set a maximum of @ref NVMEDIA_IJPE_MAX_PRENVSCISYNCFENCES prefences
 * by calling %NvMediaIJPEInsertPreNvSciSyncFence() before %NvMediaIJPEFeedFrame().
 * After the call to %NvMediaIJPEFeedFrame(), all NvSciSyncFences previously
 * inserted by %NvMediaIJPEInsertPreNvSciSyncFence() are removed, and they are not
 * reused for the subsequent %NvMediaIJPEFeedFrame() calls.
 *
 * @pre Pre-NvSciSync fence obtained from previous engine in the pipeline
 * @post Pre-NvSciSync fence is set
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] encoder A pointer to the NvMediaIJPE object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] prenvscisyncfence A pointer to %NvSciSyncFence.
 *      \inputrange Non-NULL - valid pointer address
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a encoder is not a valid %NvMediaIJPE
 *     handle, or @a prenvscisyncfence is NULL, or if @a prenvscisyncfence was not
 *     generated with an @ref NvSciSyncObj that was registered with @a encoder as
 *     either @ref NVMEDIA_PRESYNCOBJ or @ref NVMEDIA_EOF_PRESYNCOBJ type.
 * - ::NVMEDIA_STATUS_NOT_SUPPORTED if %NvMediaIJPEInsertPreNvSciSyncFence()
 *     has already been called at least %NVMEDIA_IJPE_MAX_PRENVSCISYNCFENCES times
 *     with the same @a encoder handle before an %NvMediaIJPEFeedFrame() call.
 */
NvMediaStatus
NvMediaIJPEInsertPreNvSciSyncFence(
    const NvMediaIJPE         *encoder,
    const NvSciSyncFence     *prenvscisyncfence
);

/**
 * @brief Gets EOF @ref NvSciSyncFence for an NvMediaIJPEFeedFrame() operation.
 *
 * The EOF %NvSciSyncFence associated with an %NvMediaIJPEFeedFrame() operation
 * is an %NvSciSyncFence. Its expiry indicates that the corresponding
 * %NvMediaIJPEFeedFrame() operation has finished.
 *
 * This function returns the EOF %NvSciSyncFence associated
 * with the last %NvMediaIJPEFeedFrame() call. %NvMediaIJPEGetEOFNvSciSyncFence()
 * must be called after an %NvMediaIJPEFeedFrame() call.
 *
 * For example, in this sequence of code:
 * \code
 * nvmstatus = NvMediaIJPEFeedFrame(handle, srcsurf, srcrect, picparams, instanceid);
 * nvmstatus = NvMediaIJPEGetEOFNvSciSyncFence(handle, nvscisyncEOF, eofnvscisyncfence);
 * \endcode
 * expiry of @a eofnvscisyncfence indicates that the preceding
 * %NvMediaIJPEFeedFrame() operation has finished.
 *
 * @pre NvMediaIJPESetNvSciSyncObjforEOF()
 * @pre NvMediaIJPEFeedFrame()
 * @post EOF NvSciSync fence for a submitted task is obtained
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] encoder A pointer to the NvMediaIJPE object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] eofnvscisyncobj    An EOF NvSciSyncObj associated with the
 *                                 NvSciSyncFence which is being requested.
 *      \inputrange A valid NvSciSyncObj
 * @param[out] eofnvscisyncfence A pointer to the EOF NvSciSyncFence.
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a encoder is not a valid %NvMediaIJPE
 *         handle, @a eofnvscisyncfence is NULL, or @a eofnvscisyncobj is not
 *         registered with @a encoder as type @ref NVMEDIA_EOFSYNCOBJ or
 *         @ref NVMEDIA_EOF_PRESYNCOBJ.
 * - ::NVMEDIA_STATUS_ERROR if the function was called before
 *         %NvMediaIJPEFeedFrame() was called.
 */
NvMediaStatus
NvMediaIJPEGetEOFNvSciSyncFence(
    const NvMediaIJPE        *encoder,
    NvSciSyncObj      eofnvscisyncobj,
    NvSciSyncFence    *eofnvscisyncfence
);

/**
 * @brief Specifies the @ref NvSciSyncObj to be used for an SOF
 * @ref NvSciSyncFence.
 *
 * This function is not supported.
 *
 * To use NvMediaIJPEGetSOFNvSciSyncFence(), the application must call
 * %NvMediaIJPESetNvSciSyncObjforSOF() before it calls NvMediaIJPEFeedFrame().
 *
 * %NvMediaIJPESetNvSciSyncObjforSOF() currently may be called only once before
 * each call to %NvMediaIJPEFeedFrame(). The application may choose to call this
 * function only once before the first call to %NvMediaIJPEFeedFrame().
 *
 * @pre N/A
 * @post N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] encoder A pointer to the NvMediaIJPE object.
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
NvMediaIJPESetNvSciSyncObjforSOF(
    const NvMediaIJPE        *encoder,
    NvSciSyncObj            nvscisyncSOF
);

/**
 * @brief Gets SOF @ref NvSciSyncFence for an NvMediaIJPEFeedFrame() operation.
 *
 * This function is not supported.
 *
 * The SOF %NvSciSyncFence associated with an %NvMediaIJPEFeedFrame() operation
 * is an %NvSciSyncFence. Its expiry indicates that the corresponding
 * %NvMediaIJPEFeedFrame() operation has started.
 *
 * This function returns the SOF %NvSciSyncFence associated
 * with the last %NvMediaIJPEFeedFrame() call. %NvMediaIJPEGetSOFNvSciSyncFence()
 * must be called after an %NvMediaIJPEFeedFrame() call.
 *
 * @pre N/A
 * @post N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] encoder A pointer to the NvMediaIJPE object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] sofnvscisyncobj    An SOF NvSciSyncObj associated with the
 *                                 NvSciSyncFence which is being requested.
 *      \inputrange A valid NvSciSyncObj
 * @param[out] sofnvscisyncfence A pointer to the SOF NvSciSyncFence.
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a encoder is not a valid %NvMediaIJPE
 *         handle, @a sofnvscisyncfence is NULL, or @a sofnvscisyncobj is not
 *         registered with @a encoder as type @ref NVMEDIA_SOFSYNCOBJ or
 *         @ref NVMEDIA_SOF_PRESYNCOBJ.
 * - ::NVMEDIA_STATUS_ERROR if the function was called before
 *         %NvMediaIJPEFeedFrame() was called.
 */
NvMediaStatus
NvMediaIJPEGetSOFNvSciSyncFence(
    const NvMediaIJPE        *encoder,
    NvSciSyncObj            sofnvscisyncobj,
    NvSciSyncFence          *sofnvscisyncfence
);

/*
 * @defgroup 6x_history_nvmedia_ijpe History
 * Provides change history for the NvMedia Image Jpeg Encode API.
 *
 * \section 6x_history_nvmedia_ijpe Version History
 *
 * <b> Version 1.0 </b> September 28, 2021
 * - Initial release
 *
 */

/** @} */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_IJPE_H */
