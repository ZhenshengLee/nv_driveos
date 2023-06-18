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


#ifndef NVMEDIA_DRM_H
#define NVMEDIA_DRM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

#include "nvmedia_common_decode.h"
#include "nvmedia_core.h"

/**
 * @file
 * @brief <b> NVIDIA Media Interface: Video Decryptor (DRM) API </b>
 *
 * @b Description: This file provides the Video Decryptor (DRM) API.
 */

/**
 * @defgroup 6x_nvmedia_drm_group Video Decryptor (DRM) API
 * @ingroup 6x_nvmedia_image_top
 *
 * @brief Decrypts and re-encrypts video data with supported formats for the hardware
 * in a video stream.
 * @{
 */

/**
 * \hideinitializer
 * @brief max number of subsample support
 */
#define NVMEDIA_MAX_NALS 256

/** @brief Defines the supported DRM formats. */
typedef enum {
    /** drm format: Netflix */
    NvMDRM_Netflix = 0,
    /** drm format: Widevine */
    NvMDRM_Widevine = 1,
    /** drm format: Ultravoilet */
    NvMDRM_Ultraviolet,
    /** drm format: Piff */
    NvMDRM_Piff,
    /** drm format: Marlin */
    NvMDRM_Marlin,
    /** drm format: Piff CBC */
    NvMDRM_PiffCbc,
    /** drm format: Piff CTC */
    NvMDRM_PiffCtr,
    /** drm format: Marlin CBC */
    NvMDRM_MarlinCbc,
    /** drm format: Marlin CTR */
    NvMDRM_MarlinCtr,
    /** drm format: Widevine CTR */
    NvMDRM_WidevineCtr,
    /** drm format: Clear data */
    NvMDRM_Clear = 0xf,
    /** drm format: To tell clear data processing in secure buffer */
    NvMDRM_ClearAsEncrypted,
    /** drm format: None: This should be the last element */
    NvMDRM_None
} NvMediaDRMType;

/**
 * @brief Holds encrypted metadata information that
 * the parser passes to the video decoder component. */
typedef struct
{
    /** Flag that specifies whether the buffer is encrypted. */
    uint32_t   enableEncryption;
    /** drm mode of encrypted content. */
    uint32_t      uDrmMode;
    /** Intialization vector of all subsamples */
    uint32_t      InitVector[NVMEDIA_MAX_NALS][4];
    /** intialization vectors are present or not for subsamples */
    uint32_t      IvValid[NVMEDIA_MAX_NALS];
    /** total bytes of encrypted data in input buffer */
    uint32_t      uBytesOfEncryptedData;
    /** encrypt blk count when pattern mode encryption is used */
    uint32_t      uEncryptBlkCnt;
    /** skip blk count when pattern mode encryption is used */
    uint32_t      uSkipBlkCnt;
    /** total number of subsamples for given buffer */
    uint32_t      uNumNals;
    /** keyslot number used where content key is written */
    uint32_t      KeySlotNumber;
    /** bytes of encrypted data for subsamples */
    uint32_t      BOED[NVMEDIA_MAX_NALS];
    /** bytes of clear data for subsamples */
    uint32_t      BOCD[NVMEDIA_MAX_NALS];
    /** encrypted metadata buffer of pass1 */
    uint32_t      *AesPass1OutputBuffer;
    /** bytes of non slice data in input buffer */
    uint32_t      non_slice_data;
}NvMediaEncryptParams;

/**
 * @brief Holds re-encrypted data information that the video decoder returns
 * to the parser.
 * \sa @ref pfnCbNvMediaDecodePicture
 */
typedef struct {
    /** clear hdr side after pass1 */
    uint32_t  uClearHeaderSize;
    /** clear hdr pointer to buffer after pass1 */
    uint8_t  *pClearHeaderPtr;
    /** encrypted metadata struture pointer after pass1 */
    uint32_t *pAesPass1OutputBuffer;
} NvMediaAESMetaData;

/** @brief Holds encryption intialization vector information. */
typedef struct {
    /** intialization vector */
    uint8_t         IV[16];
    /** intialization vector present or not */
    uint32_t     bIvValid;
} NvMediaAESIv;

/** @brief Holds encrypted metadata information that the client sends
 *  to the parser.
 *  \sa NvMediaParserSetEncryption()
 */
typedef struct _NvMediaAESParams
{
    /** drm mode of encrypted content */
    uint32_t uDrmMode;
    /** pointer to intialization vector array */
    uint8_t  *pIV;
    /** encrypt blk count when pattern mode encryption is used */
    uint32_t uEncryptBlkCnt;
    /** skip blk count when pattern mode encryption is used */
    uint32_t uSkipBlkCnt;
    /** keyslot number used where content key is written */
    uint32_t KeySlotNumber;
    /** pointer to bytes of encrypted data for subsamples */
    uint32_t *pBOED;
    /** pointer to bytes of clear data for subsamples */
    uint32_t *pBOCD;
    /** total number of subsamples for given buffer */
    uint32_t uMetadataCount;
    /** non aligned offset for encrypted buffer */
    uint32_t uNonAlignedOffset;
    /** initialization vector array */
    NvMediaAESIv IvSet[NVMEDIA_MAX_NALS];
} NvMediaAESParams;

/**
 * \hideinitializer
 * @brief Defines flags used for decryptor creation.
 */
#define NVMEDIA_VIDEO_DECRYPT_PROFILING             (1<<0)

/** @brief Holds the video decrypter object. */
typedef struct {
    /** @brief Codec type */
    NvMediaVideoCodec eCodec;
    /** @brief pass hw decode clock value for otf case */
    uint32_t  hwClockValue;
} NvMediaVideoDecrypter;

/** @brief Creates a video decrypter object.
 *
 * Creates a @ref NvMediaVideoDecrypter object for the specified codec @ref NvMediaVideoCodec.
 * Use NvMediaVideoDecrypterDestroy() to destroy this video decrypter object.
 *
 * @pre NvMediaIDEGetVersion()
 * @pre NvMediaIDENvSciSyncGetVersion()
 * @post NvMediaVideoDecrypter  Instance that was  created.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @param[in] codec Codec type. The following types are supported:
 * - ::NVMEDIA_VIDEO_CODEC_HEVC
 * - ::NVMEDIA_VIDEO_CODEC_H264
 * - ::NVMEDIA_VIDEO_CODEC_MPEG1
 * - ::NVMEDIA_VIDEO_CODEC_MPEG2
 * - ::NVMEDIA_VIDEO_CODEC_VP8
 * - ::NVMEDIA_VIDEO_CODEC_VP9
 * @param[in] maxBitstreamSize The maximum size for bitstream.
 * This limits internal allocations.
 * @param[in] flags Set the flags of the decoder.
 * @param[in] instanceId The ID of the engine instance.
 * The following instances are supported:
 * - ::NVMEDIA_DECODER_INSTANCE_0
 * - ::NVMEDIA_DECODER_INSTANCE_1
 * - ::NVMEDIA_DECODER_INSTANCE_AUTO
 * @return NvMediaVideoDecrypter The new video decoder decrypter handle or NULL if unsuccessful.
 */

NvMediaVideoDecrypter *
NvMediaVideoDecrypterCreate (
    NvMediaVideoCodec codec,
    uint32_t maxBitstreamSize,
    uint32_t flags,
    NvMediaDecoderInstanceId instanceId
);

/** @brief Destroys a video decoder decrypter object.
 *
 * @pre NvMediaVideoDecrypterCreate which creates the instance
 * @post None
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * @param[in] pDecrypter The video decoder decrypter to be destroyed.
 */
void
NvMediaVideoDecrypterDestroy(
    const NvMediaVideoDecrypter *pDecrypter
);


/**
 * @brief Decrypts the HDR of the encrypted content.
 *
 * The @ref NvMediaParserClientCb::DecryptHdr callback function calls this function.
 *
 * @pre NvMediaVideoDecrypterCreate which creates the instance
 * @pre instanceID  for NVDEC.
 * @post NvMediaStatus True or  False
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] pDecrypter A pointer to the decrypter object.
 * @param[in] pictureData A pointer to NvMediaParserPictureData.
 * @param[in] pBitstream A pointer to bitstream data.
 * @param[in] instanceId The ID of the engine instance.
 * The following instances are supported if NVMEDIA_DECODER_INSTANCE_AUTO
 * was used in @ref NvMediaIDECreate API, else this parameter is ignored:
 * - ::NVMEDIA_DECODER_INSTANCE_0
 * - ::NVMEDIA_DECODER_INSTANCE_1
 * @return @ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_ERROR
 */

NvMediaStatus
NvMediaVideoDecryptHeader(
    const NvMediaVideoDecrypter *pDecrypter,
    const void *pictureData,
    const NvMediaBitstreamBuffer *pBitstream,
    NvMediaDecoderInstanceId instanceId
);


/**
 * @brief Gets clear header data after pass1.
 *
 * @pre NvMediaVideoDecrypterCreate which creates the instance
 * @pre Call to NvMediaVideoDecryptHeader
 * @post Clear Header information that is width, height and codec type  etc.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * The @ref NvMediaParserClientCb::GetClearHdr callback function calls this function.
 * This is a blocking call, which means the output data is guaranteed to contain
 * clear header data along with re-entrypted data for pass2.
 * @param[in] pDecrypter The decrypter object.
 * @param[out] pictureData A pointer to @ref NvMediaParserPictureData.
 * @return @ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_ERROR
 */

NvMediaStatus
NvMediaVideoGetClearHeader(
    NvMediaVideoDecrypter *pDecrypter,
    void *pictureData
);

/** @} */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_DRM_H */
