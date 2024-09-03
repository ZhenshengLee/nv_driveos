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
 * @brief <b> NVIDIA Media Interface: NvMedia Image Encode Processing Output
 *            ExtraData </b>
 *
 * This file contains the output extradata definition for "Image Encode
 * Processing API".
 */

#ifndef NVMEDIA_IEP_OUTPUT_EXTRA_DATA_H
#define NVMEDIA_IEP_OUTPUT_EXTRA_DATA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/** Maximum number of reference pictures including current frame */
#define NVMEDIA_ENCODE_MAX_RPS_SIZE 17U

/**
 * @defgroup 6x_image_encode_extradata Image Encoder Output Extradata
 *
 * @brief Defines Image Encoder Output Extradata data types which can be used
 * with NvMedia IEP APIs to retrieve encoder statistics/configurations. This
 * feature is not available on the QNX Safety configuration.
 *
 * @ingroup 6x_image_encode_api
 * @{
 */

/** Token that indicates the start of @ref NvMediaEncodeMVBufferHeader */
#define MV_BUFFER_HEADER 0xFFFEFDFCU

/**
 * Enumeration of possible frame types - common to H264, H265
 */
typedef enum {
    /** P Frame */
    NVMEDIA_ENCODE_H26X_FRAME_TYPE_P=0,
    /** B Frame */
    NVMEDIA_ENCODE_H26X_FRAME_TYPE_B,
    /** I Frame */
    NVMEDIA_ENCODE_H26X_FRAME_TYPE_I,
    /** IDR Frame */
    NVMEDIA_ENCODE_H26X_FRAME_TYPE_IDR,
    /** @note This value is for internal use only. */
    NVMEDIA_ENCODE_H26X_FRAME_TYPE_END
} NvMediaEncodeH26xFrameType;

/**
 * Holds a encoder frames property. This is passed to client as
 * outputextra data. This is valid when external RPS control parameter is valid.
 */
typedef struct {
    /** Unique Frame ID */
    uint32_t ulFrameId;
    /** Is the frame an IDR frame */
    bool bIdrFrame;
    /** Is the frame a Long Term Ref Frame */
    bool bLTRefFrame;
    /** Picture order count of the frame */
    uint32_t  ulPictureOrderCnt;
    /** Frame Number of the frame */
    uint32_t  ulFrameNum;
    /** LongTermFrameIdx of the picture */
    uint32_t  ulLTRFrameIdx;
} NvMediaEncodeFrameFullProp;

/**
 * Holds the statistics from encode profiling.
 */
typedef struct {
    /** Hardware Cycle Count */
    uint32_t  ulCycleCount;
    /** Time taken for setting the preset in feedframe */
    uint32_t  ulPresetTime;
    /** Hardware Flush Time */
    uint32_t  ulFlushTime;
    /** Hardware Encode Time */
    uint32_t  ulEncodeTime;
    /** Time taken to fetch the encoded bitstream once a frame is passed for
     *  encoded */
    uint32_t  ulFetchTime;
} NvMediaFrameStats;

/**
 * Holds a codec-specific extradata output
 */
typedef union {
    /** Holds output for codec type H264 */
    struct {
        /** Frame type of the encoded frame */
        NvMediaEncodeH26xFrameType eFrameType;
        /** Is this a reference frame */
        bool bRefPic;
        /** Is this an intra refresh frame */
        bool bIntraRefresh;
        /** Count of the number of intra MBs */
        uint32_t uIntraMBCount;
        /** Count of the number of inter MBs */
        uint32_t uInterMBCount;
    } h264Extradata;
    /** Holds output for codec type H265 */
    struct {
        /** Frame type of the encoded frame. */
        NvMediaEncodeH26xFrameType eFrameType;
        /** Is this a reference frame. */
        bool bRefPic;
        /** Is this an intra refresh frame. */
        bool bIntraRefresh;
        /** Count of the number of intra 32x32 CUs. */
        uint32_t uIntraCU32x32Count;
        /** Count of the number of inter 32x32 CUs. */
        uint32_t uInterCU32x32Count;
        /** Count of the number of intra 16x16 CUs. */
        uint32_t uIntraCU16x16Count;
        /** Count of the number of inter 16x16 CUs. */
        uint32_t uInterCU16x16Count;
        /** Count of the number of intra 8x8 CUs. */
        uint32_t uIntraCU8x8Count;
        /** Count of the number of inter 8x8 CUs. */
        uint32_t uInterCU8x8Count;
    } h265Extradata;
} NvMediaEncodeCodecExData;

/**
  * Header format that defines motion vector output. This header will be
  * present in the encoded bitstream output at an offset defined by
  * @ref NvMediaEncodeOutputExtradata.MVBufferDumpStartOffset if
  * @ref NvMediaEncodeOutputExtradata.bMVbufferdump is set. The motion vector
  * output will immediately follow this header if \a buffersize is non-zero.
  *
  * @note Motion Vector output format in memory (v2.0):
  * @code
  * <Memory Address - low>
  * |-----------------------------| <- MVBufferDumpStartOffset
  * | NvMediaEncodeMVBufferHeader |
  * |-----------------------------|
  * |     NvMediaEncodeMVData 0   |
  * |-----------------------------|
  * |     NvMediaEncodeMVData 1   |
  * |-----------------------------|
  *                 .
  *                 .
  * <Memory Address - high>
  *
  * Number of NvMediaEncodeMVData =
  *     (NvMediaEncodeMVBufferHeader.width_in_blocksize *
  *      NvMediaEncodeMVBufferHeader.height_in_blocksize)
  * @endcode
  *
  * @note To enable v2.0 format of the MV Buffer dump,
  * NVMEDIA_ENCODE_CONFIG_H264_ENABLE_MV_BUFFER_DUMP_V2  should be bit-ORd in
  * @ref NvMediaEncodeConfigH264.features
  */
typedef struct
{
    /** Used to verify the integrity of the header. IEP will set this to
      * @ref MV_BUFFER_HEADER to indicate that the header is valid. */
    uint32_t MagicNum;
    /** Size of motion vector output (excluding header size), i.e., the size
      * of MV data in the bitstream post the header. */
    uint32_t buffersize;
    /** Macro Block size. */
    uint16_t blocksize;
    /** Input frame width in terms of \a blocksize. */
    uint16_t width_in_blocksize;
    /** Input frame height in terms of \a blocksize. */
    uint16_t height_in_blocksize;
    /** Reserved */
    uint16_t reserved;
} NvMediaEncodeMVBufferHeader;

/** Motion Vector format - motion vectors for each of the macro blocks are
  * dumped in this format contiguously in memory, beyond the
  * @ref NvMediaEncodeMVBufferHeader in the bitstream.
  */
typedef struct {
    /** X component of the motion vector pertaining to 1 macro block. */
    int32_t mv_x;
    /** Y component of the motion vector pertaining to 1 macro block. */
    int32_t mv_y;
} NvMediaEncodeMVData;

/**
 * Holds the encoder output extradata configuration.
 */
typedef struct {
    /** Size of this extradata structure. This needs to be filled correctly by
      * the client to sizeof(NvMediaEncodeOutputExtradata). This size is used
      * as sanity check before writing output extradata on this buffer. */
    uint32_t ulExtraDataSize;
    /** Format of input H264 data.
      * @note This feature is not supported. Use
      * @ref NvMediaEncodeCodecExData.eFrameType instead. */
    bool bkeyFrame;
    /** Slice end or frame end in the packet for application to handle packets
      * When slice encode is completed but not complete frame then it will set
      * bEndOfFrame to false. After frame encode is complete, it will
      * set bEndOfFrame to true.
      * @note This feature is not supported. */
    bool bEndOfFrame;
    /** Size of SPS/PPS header if it passed with output buffer. */
    uint32_t ulHdrSize;
    /** Average QP index of the encoded frame. */
    int16_t  AvgQP;
    /** Flag for vp8 reference frame information.
      * @note This feature is not supported. */
    bool bIsGoldenOrAlternateFrame;
    /** Whether Recon CRC for Recon frame is present.
      * @note This feature is not supported. */
    bool bValidReconCRC;
    /** Recon CRC for Y component when ReconCRC generation is enabled.
      * @note This feature is not supported. */
    uint32_t ulReconCRC_Y;
    /** Recon CRC for U component when ReconCRC generation is enabled.
      * @note This feature is not supported. */
    uint32_t ulReconCRC_U;
    /** Recon CRC for V component when ReconCRC generation is enabled.
      * @note This feature is not supported. */
    uint32_t ulReconCRC_V;
    /** Rate Control Feedback */
    /** Minimum QP used for this frame */
    uint32_t ulFrameMinQP;
    /** Maximum QP used for this frame */
    uint32_t ulFrameMaxQP;
    /** RPS Feedback */
    /** Reference Picture Set data output enabled.
      * @note This feature is not supported. */
    bool bRPSFeedback;
    /** frame id of reference frame to be used for motion search, ignored for IDR.
      * @note This feature is not supported. */
    uint32_t  ulCurrentRefFrameId;
    /** Number of valid entries in RPS.
      * @note This feature is not supported. */
    uint32_t  ulActiveRefFrames;
    /** RPS List including most recent frame if it is reference frame.
      * @note This feature is not supported. */
    NvMediaEncodeFrameFullProp RPSList[NVMEDIA_ENCODE_MAX_RPS_SIZE];
    /** Set if bitstream buffer contains MV Buffer dump.
      *
      * To enable this feature, NVMEDIA_ENCODE_CONFIG_H264_ENABLE_MV_BUFFER_DUMP
      * should be bit-ORd in @ref NvMediaEncodeConfigH264.features
      *
      * When this is set, @ref NvMediaEncodeMVBufferHeader and array of
      * @ref NvMediaEncodeMVData (one for each macro block - contiguously in
      * memory) will follow the encoded bitstream output at an offset defined in
      * @ref NvMediaEncodeOutputExtradata.MVBufferDumpStartOffset
      */
    bool bMVbufferdump;
    /** Size of the MV buffer, including @ref NvMediaEncodeMVBufferHeader and
      * @ref NvMediaEncodeMVData for each macroblock, as per the format defined
      * in NvMediaEncodeMVBufferHeader. This is set to a non-zero value only if
      * there was sufficient space in bitstream buffer to store the MV Buffer
      * data. */
    uint32_t MVBufferDumpSize;
    /** Encoded motion vector buffer dump start offset in the bitstream. This is
      * the location in the bitstream where the @ref NvMediaEncodeMVBufferHeader
      * starts, followed by @ref NvMediaEncodeMVData data. Motion vector dump is
      * appended at the end of encoded bistream buffer with 8 byte aligned
      * position. */
    uint32_t MVBufferDumpStartOffset;
    /** Encoder Profiling stats.
      * @note This feature is not supported. */
    NvMediaFrameStats FrameStats;
    /** hrdBitrate to be used to calculate RC stats.
      * @note This feature is not supported. */
    uint32_t ulHrdBitrate;
    /** vbvBufSize to be used to compute RC stats.
      * @note This feature is not supported. */
    uint32_t ulVbvBufSize;
    /** Codec Type */
    NvMediaVideoCodec codec;
    /** Codec specific extradata. */
    NvMediaEncodeCodecExData codecExData;
} NvMediaEncodeOutputExtradata;

/** @} <!-- Ends image_encode_extradata Encoder Output Extradata --> */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_IEP_OUTPUT_EXTRA_DATA_H */
