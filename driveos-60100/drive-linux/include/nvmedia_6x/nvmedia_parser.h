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


#ifndef NVMEDIA_PARSER_H
#define NVMEDIA_PARSER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

#include "nvmedia_common_decode.h"
#include "nvmedia_core.h"
#include "nvmedia_drm.h"

/**
 * @file
 * @brief <b> NVIDIA Media Interface: Parser </b>
 *
 * @b Description: This file provides the NvMedia Parser API.
 */

/**
 * \defgroup 6x_nvmedia_parser_group Parser API
 * \ingroup 6x_nvmedia_common_top
 *
 * @brief Provides parsing for decoder frame handling.
 * @{
 */

// RVCT compiler doesn't allow anonymous structures or unions by default.
#if defined(__CC_ARM)
#pragma anon_unions
#endif

/** @brief Defines video format types. */
typedef enum {
    /** video format: Component */
    NvMVideoFormat_Component     = 0,
    /** video format: PAL */
    NvMVideoFormat_PAL,
    /** video format: NTSC */
    NvMVideoFormat_NTSC,
    /** video format: SECAM */
    NvMVideoFormat_SECAM,
    /** video format: MAC */
    NvMVideoFormat_MAC,
    /** video format: Unspecified */
    NvMVideoFormat_Unspecified,
    /** video format: Reserved */
    NvMVideoFormat_Reserved6,
    /** video format: Reserved */
    NvMVideoFormat_Reserved7
} NvMediaVideoFormatType;

/** @brief Defines Color Range types. */
typedef enum {
    /** color range: limited */
    NvMColorRange_Reduced    = 0,
    /** color range: full */
    NvMColorRange_Full
} NvMediaColorRangeType;

/** @brief Defines color_primaries types. */
typedef enum {
    /** color primaries: Forbidden */
    NvMColorPrimaries_Forbidden      = 0,
    /** color primaries: BT_709 */
    NvMColorPrimaries_BT709,
    /** color primaries: Unspecified */
    NvMColorPrimaries_Unspecified,
    /** color primaries: Reserved */
    NvMColorPrimaries_Reserved,
    /** color primaries: BT470M */
    NvMColorPrimaries_BT470M,
    /** color primaries: BT470BG */
    NvMColorPrimaries_BT470BG,
    /** color primaries: SMPTE170M */
    NvMColorPrimaries_SMPTE170M,
    /** color primaries: SMPTE240M */
    NvMColorPrimaries_SMPTE240M,
    /** color primaries: GenericFilm */
    NvMColorPrimaries_GenericFilm,
    /** color primaries: BT2020 */
    NvMColorPrimaries_BT2020
} NvMediaColorPrimariesType;

/** @brief Defines matrix_coefficients types. */
typedef enum {
    /** matrix coefficients: Forbidden */
    NvMMatrixCoeffs_Forbidden=0,
    /** matrix coefficients: BT709 */
    NvMMatrixCoeffs_BT709,
    /** matrix coefficients: Unspecified */
    NvMMatrixCoeffs_Unspecified,
    /** matrix coefficients: Reserved */
    NvMMatrixCoeffs_Reserved,
    /** matrix coefficients: FCC */
    NvMMatrixCoeffs_FCC,
    /** matrix coefficients: BT470BG */
    NvMMatrixCoeffs_BT470BG,
    /** matrix coefficients: SMPTE170M */
    NvMMatrixCoeffs_SMPTE170M,
    /** matrix coefficients: SMPTE240 */
    NvMMatrixCoeffs_SMPTE240M,
    /** matrix coefficients: YCgCo */
    NvMMatrixCoeffs_YCgCo,
    /** matrix coefficients: BT2020 non-constant luminance system */
    NvMMatrixCoeffs_BT2020_nc,
    /** matrix coefficients: BT2020 constant luminance system */
    NvMMatrixCoeffs_BT2020_c,
} NvMediaMatrixCoeffsType;

/** @brief Defines transfer characteristics types. */
typedef enum {
    /** transfer characteristics: Forbidden */
    NvMTransferCharacteristics_Forbidden=0,
    /** transfer characteristics: BT709 */
    NvMTransferCharacteristics_BT709,
    /** transfer characteristics: Unspecified */
    NvMTransferCharacteristics_Unspecified,
    /** transfer characteristics: Reserved */
    NvMTransferCharacteristics_Reserved,
    /** transfer characteristics: BT470M */
    NvMTransferCharacteristics_BT470M,
    /** transfer characteristics: BT470BG */
    NvMTransferCharacteristics_BT470BG,
    /** transfer characteristics: SMPTE170M */
    NvMTransferCharacteristics_SMPTE170M,
    /** transfer characteristics: SMPTE240M */
    NvMTransferCharacteristics_SMPTE240M,
    /** transfer characteristics: Linear */
    NvMTransferCharacteristics_Linear,
    /** transfer characteristics: Log100 */
    NvMTransferCharacteristics_Log100,
    /** transfer characteristics: Log316 */
    NvMTransferCharacteristics_Log316,
    /** transfer characteristics: IEC61966_2_4 */
    NvMTransferCharacteristics_IEC61966_2_4,
    /** transfer characteristics: BT1361 */
    NvMTransferCharacteristics_BT1361,
    /** transfer characteristics: IEC61966_2_1 */
    NvMTransferCharacteristics_IEC61966_2_1,
    /** transfer characteristics: BT2020_1 */
    NvMTransferCharacteristics_BT2020_1,
    /** transfer characteristics: BT2020_2 */
    NvMTransferCharacteristics_BT2020_2,
    /** transfer characteristics: SMPTE2084 */
    NvMTransferCharacteristics_SMPTE2084,
    /** transfer characteristics: ST418_1 */
    NvMTransferCharacteristics_ST418_1,
    /** transfer characteristics: HLG */
    NvMTransferCharacteristics_HLG
} NvMediaTransferCharacteristicsType;

/** @brief Defines the parser decode modes. */
typedef enum
{
    /** normal decode mode */
    NvMParserDecodeMode_Normal = 0,
    /** only decode key frames */
    NvMParserDecodeMode_SyncToKeyFrame
} NvMediaParserDecodeMode;

/** @brief Defines the chroma formats of video content. */
typedef enum
{
    /** chroma format YUV400 (Only Y component) */
    NvMediaParserChromaFormat_400 = 0,
    /** chroma format YUV420 */
    NvMediaParserChromaFormat_420,
    /** chroma format YUV422 */
    NvMediaParserChromaFormat_422,
    /** chroma format YUV444 */
    NvMediaParserChromaFormat_444
} NvMediaParserChromaFormat;

/** @brief Defines parser attributes, which sets/gets decoder properties to parse
  * as per the client/app requirement.
  */
typedef enum
{
    /** Frame management: Disable DPB logic for low latency H264 playback */
    NvMParseAttr_h264DisableDpb = 100,
    /** Frame management: Decode IDR Frames only */
    NvMParseAttr_DecodeIDRFrameOnly,
    /** Frame management: 0: Normal decode, 1: KeyFrameDecode */
    NvMParseAttr_DecodeMode,
    /** Frame management: Drop MPEG2 B-Frames */
    NvMParseAttr_DropMPEG2BFrames,
    /** Frame management: Set default H.264 max_dec_frame_buffering */
    NvMParseAttr_MaxDecFrameBuffering,
    /** Frame management: Abort decoding when unsupported features or erroneous bitstream */
    NvMParseAttr_AbortDecodeOnParserError,
    /** Frame management: Disable DPB logic for low latency H265 playback */
    NvMParseAttr_h265DisableDpb,
    /** Frame management: DPB size is set to max number of reference frames */
    NvMParseAttr_DpbSize_SetTo_MaxRefFrames,

    /** Frame timing: Output timestamp in increasing order after filtering */
    NvMParseAttr_FilterTimestamps = 200,
    /** Frame timing: Use frame timing info for PTS calculation */
    NvMParseAttr_UseFrameTimingInfo,
    /** Frame timing: Set frame rate for PTS calculation (float type) */
    NvMParseAttr_SetFramerate,
    /** Frame timing: Set default frame rate for PTS calculation (float type) */
    NvMParseAttr_SetDefaultFramerate,

    /* Security group */

    /** Security: Enable m_pSlhData allocation for encrypted streams in parser. */
    NvMParseAttr_SecureDecode = 300,

    /** Other: Enable VC-1 advanced profile interlaced. */
    NvMParseAttr_EnableVC1APInterlaced = 400,
    /** Other: For setting max resolution for creating decoder. */
    NvMParseAttr_SetMaxRes,
    /* Other: Generate slice offsets with CompletePicture mode. */
    NvMParseAttr_GenerateSliceOffsets,
    /** Other: Enable error status reporting */
    NvMParseAttr_ErrorStatusReporting,
    /** Other: Get bitstream error for current input buffer. */
    NvMParseAttr_GetBitstreamError,
    /** Other: Indicate whether the previous frame was lost. */
    NvMParseAttr_PrevFrameLostFlag,
    /** Other: Enable slice-level decode for the current session.
     *  Applies only if slice-level decode is supported. */
    NvMParseAttr_SliceLevelDecode,
    /** Other: Set Operating Point in case of AV1 SVC stream.
     *  Applies only if operating Points is greater than 1. */
    NvMParseAttr_av1SetOperatingPoint,
    /** Other: Set outputAllLayers in case of AV1 SVC stream.
     *  Applies only if operating Points is greater than 1. */
    NvMParseAttr_av1SetOutputAllLayers,
    /** Other: Set AnnexB stream in case of AV1 SVC stream. */
    NvMParseAttr_av1AnnexBDecode,
} NvMediaParserAttr;

/** @brief Holds the bitstream packet input for parsing. */
typedef struct
{
    /** Pointer to byte stream data */
    const uint8_t *pByteStream;
    /** Data length for this packet */
    uint32_t uDataLength;
    /** Presentation Time Stamp for this packet (clock rate specified at initialization) */
    int64_t llPts;
    /** true if this is an End-Of-Stream packet (flush everything) */
    uint32_t bEOS;
    /** true if llPTS is valid (also used to detect frame boundaries for VC1 SP/MP) */
    uint32_t bPTSValid;
    /** true if DecMFT is signalling a discontinuity */
    uint32_t bDiscontinuity;
    /** true if this packet contains complete picture */
    uint32_t bCompletePicture;
    /** true if this packet contains complete Slice */
    uint32_t bCompleteSlice;
    /** true if this packet belongs to the same frame currently being decoded */
    uint32_t bDecodeNextSlice;
    /** if it is 0 then it is non-RTP */
    uint32_t uRTP_NALUSizeFieldWidthInBytes;
} NvMediaBitStreamPkt;

/** @brief Holds decoded reference frame information. */
typedef struct
{
    /** value 0 or 1 : whether the frame is present in DPB */
    uint8_t uPresentFlag;
    /** value 0 or 1 : whether the frame is an IDR */
    uint8_t uIDRFrame;
    /** value 0 or 1 : whether the frame is long term ref */
    uint8_t uLTRefFrame;
    /** value 0 or 1 : whether the frame is motion predicted for the current
     * frame as specified in slice header
     */
    uint8_t uPredicted;
    /** Picture order counter */
    uint32_t uPictureOrderCnt;
    /** Frame Num*/
    uint32_t uFrameNum;
    /** LongTermFrameIdx of the frame */
    uint32_t uLTRFrameIdx;
} NvMediaDecRefFrame;

/** @brief Holds decoded current frame information. */
typedef struct
{
    /** value 0 or 1 : indicating current frame is referenced or not */
    uint8_t uRefFrame;
    /** value 0 or 1 : indicating current frame IDR or not */
    uint8_t uIDRFrame;
    /** value 0 or 1 : indicating long term reference */
    uint8_t uLTRefFrame;
    /** reserved for 4-byte alignment */
    uint8_t uReserve0;
    /** picture order counter */
    uint32_t uPictureOrderCnt;
    /** frame number */
    uint32_t uFrameNum;
    /** LongTermFrameIdx of a picture */
    uint32_t uLTRFrameIdx;
} NvMediaDecCurrFrame;

/** @brief Holds decoded Picture Buffer information. */
typedef struct
{
    /** Specify the max number of bits used to derive frame number */
    uint32_t poc_fn_bits;
    /** Specify the property of current decoded frame */
    NvMediaDecCurrFrame currentFrame;
    /** number of valid entries in RPS */
    uint32_t uActiveRefFrames;
    /** RPS List @ref NvMediaDecRefFrame as signaled in slice header
     * excluding current frame, only first nActiveRefFrames are valid.
     * This can be different from list of frames in current DPB due to frame loss.
     */
    NvMediaDecRefFrame RPSList[16];
} NvMediaDPBInfo;

/**
 * \hideinitializer
 * @brief Defines the maximum size of the sequence header.
 */

#define MAX_SEQ_HDR_LEN   512U

/** @brief Holds video sequence information.
 *
 * @note video sequence information is passed to client using the @ref NvMediaParserClientCb::BeginSequence
 * callback. The client then creates a corresponding NvMedia codec component.
 */
typedef struct
{
    /* Codec information */

    /** Codec info: Specifies the compression standard */
    NvMediaVideoCodec eCodec;

    /* Frame information */

    /** Frame info: Specifies the frame rate of the bitstream */
    double fFrameRate;
    /** Frame info: Specifies the number of decode buffers required */
    uint32_t uDecodeBuffers;
    /** Frame info: Specifies whether the bitstream is progressive or not */
    uint8_t bProgSeq;
    /** Frame info: Specifies the Video bitrate (bps) */
    uint32_t uBitrate;

    /* Sequence information */

    /** Sequence info: Specifies the number of bytes in SequenceHeaderData */
    uint32_t uSequenceHeaderSize;
    /** Sequence info: Specifies the maximum size of bitstream buffer */
    uint32_t uMaxBitstreamSize;
    /** Sequence info: Specifies the raw sequence header data (codec-specific) */
    uint8_t SequenceHeaderData[MAX_SEQ_HDR_LEN];

    /* Frame size information */

    /** Frame size info: Specifies the coded Picture Width */
    uint16_t uCodedWidth;
    /** Frame size info: Specifies the coded Picture Height */
    uint16_t uCodedHeight;
    /** Frame size info: Specifies the displayed Horizontal Size */
    uint16_t uDisplayWidth;
    /** Frame size info: Specifies the displayed Vertical Size */
    uint16_t uDisplayHeight;
    /** Frame size info: Specifies the actual Width */
    uint16_t uActualWidth;
    /** Frame size info: Specifies the actual Height */
    uint16_t uActualHeight;
    /** Frame size info: Specifies the left offset for display */
    uint16_t uDisplayLeftOffset;
    /** Frame size info: Specifies the top offset for display */
    uint16_t uDisplayTopOffset;

    /* Aspect ratio information */

    /** Aspect ratio info: Displays Aspect Ratio = uDARWidth */
    uint16_t uDARWidth;
    /** Aspect ratio info: Displays Aspect Ratio = uDARHeight */
    uint16_t uDARHeight;
    /** Aspect ratio info: Specifies the sample Aspect Ratio = uSARWidth */
    uint16_t uSARWidth;
    /** Aspect ratio info: Specifies the sample Aspect Ratio = uSARHeight */
    uint16_t uSARHeight;

    /* Color information */

    /** Color info: Specifies the chroma format (0=4:0:0, 1=4:2:0, 2=4:2:2, 3=4:4:4) */
    NvMediaParserChromaFormat eChromaFormat;
    /** Color info: Specifies the Luma bit depth (0=8bit) */
    uint8_t uBitDepthLumaMinus8;
    /** Color info: Specifies the Chroma bit depth (0=8bit) */
    uint8_t uBitDepthChromaMinus8;
    /** Color info: Specifies the Video Format */
    NvMediaVideoFormatType eVideoFormat;
    /** Color info: Specifies the Video Range (0-255) vs (16-235) */
    NvMediaColorRangeType eVideoFullRangeFlag;
    /** Color info: Specifies the Color Primaries */
    NvMediaColorPrimariesType eColorPrimaries;
    /** Color info: Specifies the Transfer Characteristics */
    NvMediaTransferCharacteristicsType eTransferCharacteristics;
    /** Color info: Specifies the Matrix Coefficients */
    NvMediaMatrixCoeffsType eMatrixCoefficients;
    /** Color info: Specifies that mastering display data is present */
    uint32_t bMasteringDispDataPresent;
    /** Color info: Specifies Only: Mastering display data if present */
    NvMediaMasteringDisplayData MasteringDispData;
    /** Color info: Specifies Only: Content Light Level data if present */
    uint32_t bContentLightLevelInfoPresent;
    /** Color info: Specifies Only: Content Light Level data if present */
    NvMediaContentLightLevelInfo ContentLightLevelInfo;
    /** chroma info: chroma_sample_loc_type_top_field */
    uint8_t uChromaLocTopField;
    /** chroma info: chroma_sample_loc_type_bottom_field */
    uint8_t uChromaLocBottomField;
} NvMediaParserSeqInfo;


/** @brief Holds picture information for the current frame decode operation.
 *
 * @note This picture information is passed with @ref NvMediaParserClientCb::DecodePicture
 * callback. Client will call @ref NvMediaIDERender with
 * this information to NvMedia decode component to get the
 * current frame decoding done.
 */
typedef struct
{
    /** Display Aspect Ratio = uDARWidth */
    uint16_t uDARWidth;
    /** Display Aspect Ratio = uDARHeight */
    uint16_t uDARHeight;

    /** Left offset for display */
    uint16_t uDisplayLeftOffset;
    /** Top offset for display */
    uint16_t uDisplayTopOffset;

    /** Display Width */
    uint16_t uDisplayWidth;
    /** Display Height */
    uint16_t uDisplayHeight;

    /** Frame time in time_base units if NvMParseAttr_UseFrameTimingInfo is enabled */
    int64_t llFrameTime;
    /** Base time units per second if NvMParseAttr_UseFrameTimingInfo is enabled */
    int64_t llTimeBase;

    /** Current picture (output) */
    NvMediaRefSurface *pCurrPic;

    /** 0=frame picture, 1=field picture */
    int32_t field_pic_flag;
    /** 0=top field, 1=bottom field (ignored if field_pic_flag=0) */
    int32_t bottom_field_flag;
    /** Second field of a complementary field pair */
    int32_t second_field;
    /** Frame is progressive or not */
    int32_t progressive_frame;
    /** Frame pictures only*/
    int32_t top_field_first;
    /** For 3:2 pulldown (number of additional fields, 2=frame doubling, 4=frame tripling) */
    int32_t repeat_first_field;
    /** Frame is a reference frame */
    int32_t ref_pic_flag;
    /** Frame is entirely intra coded (no temporal dependencies) */
    int32_t intra_pic_flag;
    /** Chroma Format (should match sequence info) */
    int32_t chroma_format;
    /** picture order count (if known) */
    int32_t picture_order_count;

    /** Bitstream data*/
    /** Number of bytes in bitstream data buffer */
    uint32_t uBitstreamDataLen;
    /** Pointer to the bitstream data for this picture (slice-layer) */
    uint8_t *pBitstreamData;
    /** Number of slices in this picture */
    uint32_t uNumSlices;
    /** nNumSlices entries, contains offset of each slice within the bitstream data buffer */
    uint32_t *pSliceDataOffsets;
    /** Required only for encrypted h264 bitstreams*/
    uint32_t nal_ref_idc;
    /** Flag to indicate that slice level decoding is enabled */
    uint32_t bSliceDecode;

    /** NvMedia Picture Info */
    union {
        /** picture parameter information for mpeg1/2 @ref NvMediaPictureInfoMPEG1Or2 */
        NvMediaPictureInfoMPEG1Or2 mpeg2;
         /** picture parameter information for mpeg4 @ref NvMediaPictureInfoMPEG4Part2 */
        NvMediaPictureInfoMPEG4Part2 mpeg4;
        /** picture parameter information for h264 @ref NvMediaPictureInfoH264 */
        NvMediaPictureInfoH264 h264;
        /** picture parameter information for vc1 @ref NvMediaPictureInfoVC1 */
        NvMediaPictureInfoVC1 vc1;
        /** picture parameter information for VP8 @ref NvMediaPictureInfoVP8 */
        NvMediaPictureInfoVP8 vp8;
        /** picture parameter information for h265 @ref NvMediaPictureInfoH265 */
        NvMediaPictureInfoH265 hevc;
        /** picture parameter information for vp9 @ref NvMediaPictureInfoVP9 */
        NvMediaPictureInfoVP9 vp9;
        /* picture parameter information for av1 @ref NvMediaPictureInfoAV1*/
        NvMediaPictureInfoAV1 av1;
    } CodecSpecificInfo;

    /** Pass slice level data if slice level decoding enabled */
    NvMediaSliceDecodeData SliceData;

    /** Encrypted content decoding specific parameters */
    /** Contains the re-encrypted bitstream along with 256 bytes of clear bitstream of slice NAL. */
    NvMediaAESMetaData AesMetaData;
    /** contains encrypted metadata for pass1 */
    NvMediaEncryptParams encryptParams;
} NvMediaParserPictureData;

/** @defgroup 6x_abstraction_callbacks_group Abstraction of OS-Specific Client Callbacks
  * @{
  */
typedef int32_t (*pfnCbNvMediaBeginSequence)(void *, const NvMediaParserSeqInfo *);
typedef NvMediaStatus (*pfnCbNvMediaDecodePicture)(void *, NvMediaParserPictureData *);
typedef NvMediaStatus (*pfnCbNvMediaDisplayPicture)(void *, NvMediaRefSurface *, int64_t);
typedef void          (*pfnCbNvMediaUnhandledNALU)(void *, const uint8_t *, int32_t);
typedef NvMediaStatus (*pfnCbNvMediaAllocPictureBuffer)(void *, NvMediaRefSurface **);
typedef void          (*pfnCbNvMediaRelease)(void *, NvMediaRefSurface *);
typedef void          (*pfnCbNvMediaAddRef)(void *, NvMediaRefSurface *);
typedef NvMediaStatus (*pfnCbNvMediaCreateDecrypter)(void *, uint16_t width, uint16_t height);
typedef NvMediaStatus (*pfnCbNvMediaDecryptHdr)(void *, NvMediaParserPictureData *);
typedef NvMediaStatus (*pfnCbNvMediaSliceDecode)(void *, NvMediaSliceDecodeData *);
typedef NvMediaStatus (*pfnCbNvMediaGetClearHdr)(void *, NvMediaParserPictureData *);
typedef NvMediaStatus (*pfnCbNvMediaGetBackwardUpdates)(void *, NvMediaVP9BackwardUpdates *);
typedef void          (*pfnCbNvMediaGetDpbInfoForMetadata)(void *, NvMediaRefSurface *, NvMediaDPBInfo *);
/** @} */

typedef struct
{
    /** callback to provide sequence level information to client
      * Client will use this to create @ref NvMediaIDE based on codec type.
      * Client will use @ref NvMediaIDECreate
      * based on need. Clients use codec sequence information to create output buffers
      * required for decoding.
      */
    pfnCbNvMediaBeginSequence          BeginSequence;
    /** callback to provide one frame/field worth of data required to achieve decoding
      * using @ref NvMediaIDE component. Client will get @ref NvMediaParserPictureData
      * information. It will have to just update target surface pointer with
      * a reference to an @ref NvSciBufObj where the frame needs to be rendered.
      * Client will call @ref NvMediaIDERender
      * based on use case requirement.
      */
    pfnCbNvMediaDecodePicture          DecodePicture;
    /** Callback to pass frame-buffer with its timestamp in display order for display */
    pfnCbNvMediaDisplayPicture         DisplayPicture;
    /** Callback to report unhandled NALU */
    pfnCbNvMediaUnhandledNALU          UnhandledNALU;
    /** callback to get one free frame-buffer to do current frame decoding.
      * Increase the ref counting of buffer by 1 */
    pfnCbNvMediaAllocPictureBuffer     AllocPictureBuffer;
    /** Callback to tell client to decrement the buffer reference count by 1.
      * When the reference count reaches 0, the buffer can be used for further for decoding
      */
    pfnCbNvMediaRelease                Release;
    /** Callback to increase ref counting of buffer by 1 when surface is used
      * as reference
      */
    pfnCbNvMediaAddRef                 AddRef;
    /** Callback to create @ref NvMediaVideoDecrypter component to handle encrypted contents */
    pfnCbNvMediaCreateDecrypter        CreateDecrypter;
    /** Callback to pass encrypted data for decryption of frame-headers */
    pfnCbNvMediaDecryptHdr             DecryptHdr;
    /** Callback to pass slice level data for the slice level decoding use case.
      * Client will call @ref NvMediaIDESliceDecode() to pass the
      * information for slice level decoding */
    pfnCbNvMediaSliceDecode          SliceDecode;
    /** callback to get clear header data from NvMediaVideoDecrypter. This is called after
      * @ref DecryptHdr callback to get frame-headers data.
      */
    pfnCbNvMediaGetClearHdr            GetClearHdr;
    /** callback to get backward prabability data for VP9 codec. Client will call
      * @ref NvMediaIDEGetBackwardUpdates inside this callback to fill the
      * @ref NvMediaVP9BackwardUpdates structure
      */
    pfnCbNvMediaGetBackwardUpdates     GetBackwardUpdates;
    /** callback to pass internal DPB information to the client. Some clients need DPB information
      * to support error-resilience in the encoder. This callback supports conferencing
      * use cases.
      */
    pfnCbNvMediaGetDpbInfoForMetadata  GetDpbInfoForMetadata;
} NvMediaParserClientCb;

/** Holds initialization parameters for the decoder class. */
typedef struct
{
    /** Should always be present if using parsing functionality */
    NvMediaParserClientCb *pClient;
    /** Client context */
    void *pClientCtx;
    /** Ticks per second of PTS clock (0=default=10000000=10Mhz) */
    uint32_t uReferenceClockRate;
    /** Threshold for deciding to bypass of picture (0=do not decode, 100=always decode) */
    uint16_t uErrorThreshold;
    /** Codec type */
    NvMediaVideoCodec eCodec;
    /** Optional external sequence header data */
    NvMediaParserSeqInfo *pExternalSeqInfo;
} NvMediaParserParams;

/**
 * @brief  An opaque handle representing an NvMediaParser object.
 */
typedef void NvMediaParser;

/** @brief Creates a video decoder parser object.
 *
 * Creates a @ref NvMediaParser object for the specified codec @ref NvMediaVideoCodec.
 * This video parser object is to be destroyed with @ref NvMediaParserDestroy().
 *
 * @pre None
 * @post Video decoder parser object is created.
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
 * @param[in] pParserParams poiter to a @ref NvMediaParserParams structure
 * All the client callbacks are registerd inside this function. These callbacks are
 * used for different steps in decoding process.
 * @return NvMediaParser The new video decoder parser's handle or NULL if unsuccessful.
 */

NvMediaParser *
NvMediaParserCreate(
    const NvMediaParserParams *pParserParams
);

/** @brief Destroys a video decoder parser object.
 * @param[in] pParser The decoder parser to be destroyed.
 *
 * @pre NvMediaParserCreate()
 * @post Video decoder parser object is destroyed.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaParser object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 */
void
NvMediaParserDestroy(
    NvMediaParser *pParser
);

/**
 * @brief Parses or decodes bitstream data.
 *
 * @pre NvMediaParserCreate()
 * @post Bitstream is parsed for decoding.
 *
 * This function is called with bitstream packet @ref NvMediaBitStreamPkt
 * to do the decoding of current frame. If total bitstream data is not
 * sufficient for current frame decode then it returns after copying that
 * data in some internal buffer. Once one frame data is accumulated then
 * it calls different callback for different puspose.
 * @ref NvMediaParserClientCb::BeginSequence callback will be called when it detect new sequence
 * data and it will pass sequence information as defined in
 * @ref NvMediaParserSeqInfo.
 * @ref NvMediaParserClientCb::AllocPictureBuffer callback will be called to get frame buffer
 * for current frame.
 * @ref NvMediaParserClientCb::DecodePicture callback will be called when it has complete frame
 * data. It passes @ref NvMediaParserPictureData to its client.
 * @ref NvMediaParserClientCb::DisplayPicture callback will be called when frame is ready for
 * display. This passes the frame buffer and its timestamp.
 * @ref NvMediaParserClientCb::AddRef callback called to increase ref counting of frame buffer.
 * @ref NvMediaParserClientCb::Release callback called to decrease ref counting.
 * @ref NvMediaParserClientCb::GetBackwardUpdates callback called to get probability update
 * for VP9 codec.
 * @ref NvMediaParserClientCb::GetDpbInfoForMetadata callback is called if error status reporting
 * is enabled. This is pass DPB information to client.
 * The function calls the @ref NvMediaParserClientCb::SliceDecode callback
 * to pass slice-level data decoding.
 * This is used for specific hardware/codecs.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaParser object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] pParser The decoder parser object that will perform the
 *       decode parsing operation.
 * @param[in] pStreamPacket A (pointer to a) @ref NvMediaBitStreamPkt
 *       structure containing bitstream data.
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaParserParse(
    const NvMediaParser *pParser,
    const NvMediaBitStreamPkt *pStreamPacket
);

/**
 * @brief Performs the first pass handling for encrypted content data.
 *
 * This function is called with a bitstream packet @ref NvMediaBitStreamPkt
 * to do the first pass handling of encrypted frame data. This function
 * is called after NvMediaParserSetEncryption(), which
 * passes the all encryption metadata to the parser. It calls different callbacks
 * for different purposes.
 * The @ref NvMediaParserClientCb::CreateDecrypter callback is called only once for a given
 * session to create an NvMedia video decrypter component.
 * The @ref NvMediaParserClientCb::DecryptHdr callback is called for pass1 handling of encrypted
 * data. After this callback, it calls different callbacks based on
 * hardware/codecs.
 * @ref NvMediaParserClientCb::GetClearHdr callback is called to get the clear header data
 * after the NvMediaParserClientCb::DecryptHdr callback. Here it gets only frame header data to
 * create picture info data and to do buffer management.
 *
 *
 * @pre NvMediaParserCreate()
 * @post Bitstream is parsed for decoding.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaParser object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] pParser The decoder parser object that will perform the
 *       decode parsing operation.
 * @param[in] pStreamPacket A pointer to the bitstream data.
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaParserScan(
    const NvMediaParser *pParser,
    const NvMediaBitStreamPkt *pStreamPacket
);

/**
 * @brief Provides the parser object with encryption metadata for the
 * current buffer.
 *
 *
 * @pre NvMediaParserCreate()
 * @post pParser object is updated with encryption metadata.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaParser object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] pParser The decoder parser object that will perform the
 *       decode parsing operation.
 * @param[in] pAesParams A (pointer to a) encryption metadata structure
 *       as defined in @ref NvMediaAESParams
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaParserSetEncryption(
    const NvMediaParser *pParser,
    const NvMediaAESParams *pAesParams
);

/**
 * @brief Flushes the parser decode session.
 *
 * This function is called to flush the decoding session. This is used
 * before destroying parsing session to release all the buffers. This
 * is also used to handle discontinuity.
 *
 *
 * @pre NvMediaParserCreate()
 * @post pParser object will cleared/set to zero.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaParser object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] pParser The decoder parser object that will perform the
 *       decode parsing operation.
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_ERROR
 */
void
NvMediaParserFlush(
    const NvMediaParser *pParser
);

/**
 * @brief Sets attributes for the decode parser object.
 *
 *
 * @pre NvMediaParserCreate()
 * @post Parser attributes are set.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaParser object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * This function is called to set specific attributes.
 * @param[in] pParser The decoder parser object that will perform the
 *       decode parsing operation.
 * @param[in] eAttributeType attribute type defined in @ref NvMediaParserAttr
 * @param[in] uAttributeSize size of the data type
 * @param[in] pAttribute is the pointer to attribute.
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaParserSetAttribute(
    NvMediaParser *pParser,
    NvMediaParserAttr eAttributeType,
    uint32_t uAttributeSize,
    void *pAttribute
);

/**
 * @brief Gets attributes for the decode parser object.
 *
 * This function is called to get attribute set in parser component.
 * @param[in] pParser The decoder parser object that will perform the
 *       decode parsing operation.
 * @param[in] eAttributeType attribute type defined in @ref NvMediaParserAttr
 * @param[in] uAttributeSize size of the data type
 * @param[in] pAttribute is the pointer to attribute.
 *
 *
 * @pre NvMediaParserCreate()
 * @post Get parser attributes.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaParser object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @return ::NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK
 * - ::NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaParserGetAttribute(
    const NvMediaParser *pParser,
    NvMediaParserAttr eAttributeType,
    uint32_t uAttributeSize,
    void *pAttribute
);

/** @} */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_PARSER_H */
