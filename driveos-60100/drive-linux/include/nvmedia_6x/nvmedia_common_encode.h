/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @brief <b> NVIDIA Media Interface: Common Types for Encoding</b>
 *
 * @b Description: This file contains common types and definitions for
 * encode operations.
 */

#ifndef NVMEDIA_COMMON_ENCODE_H
#define NVMEDIA_COMMON_ENCODE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

#include "nvmedia_core.h"
#include "nvmedia_common_encode_decode.h"

/**
 * @ingroup 6x_image_encode_api
 * @{
 */

/**
 * \hideinitializer
 * @brief Infinite time-out for \ref NvMediaIEPBitsAvailable
 */
#define NVMEDIA_ENCODE_TIMEOUT_INFINITE         0xFFFFFFFFU

/**
 * \hideinitializer
 * @brief Infinite GOP length so that keyframes are not inserted automatically
 */
#define NVMEDIA_ENCODE_INFINITE_GOPLENGTH       0xFFFFFFFFU

/** Maximum number of Personal Identifiable Information (PII) regions */
#define NVMEDIA_ENCODE_MAX_PII_REGIONS 32U

/**
 * \hideinitializer
 * @brief Specifies the encoder instance ID
 */
typedef enum {
    /** @brief Specifies the encoder instance ID 0 */
    NVMEDIA_ENCODER_INSTANCE_0 = 0,
    /** @brief Specifies the encoder instance ID 1 */
    NVMEDIA_ENCODER_INSTANCE_1,
   /** @brief Specifies that the encoder instance ID
    * can be set dynamically during encode
    */
    NVMEDIA_ENCODER_INSTANCE_AUTO,
} NvMediaEncoderInstanceId;

/**
 * @brief Holds quantization parameters (QP) value for frames.
 */
typedef struct {
    /** QP value for P frames */
    int16_t qpInterP;
    /** QP value for B frames */
    int16_t qpInterB;
    /** QP value for Intra frames */
    int16_t qpIntra;

    int16_t reserved[3];
} NvMediaEncodeQP;

/**
 * @brief Holds Personal Identifiable Information (PII) regions
 * This feature is not supported in the QNX Safety build.
 */
typedef struct {
    /** PII region rectangle */
    NvMediaRect  piiRect;
} NvMediaEncPIIParams;

/**
 * Rate Control Modes
 * \hideinitializer
 */
typedef enum
{
    /** Constant bitrate mode. */
    NVMEDIA_ENCODE_PARAMS_RC_CBR          = 0,
    /** Constant QP mode. */
    NVMEDIA_ENCODE_PARAMS_RC_CONSTQP      = 1,
    /** Variable bitrate mode. */
    NVMEDIA_ENCODE_PARAMS_RC_VBR          = 2,
    /** Variable bitrate mode with MinQP.
        This value is not supported in the QNX Safety build.
        \deprecated This feature will be deprecated in a future release. */
    NVMEDIA_ENCODE_PARAMS_RC_VBR_MINQP    = 3,
    /** Constant bitrate mode with MinQP.
        This value is not supported in the QNX Safety build.
        \deprecated This feature will be deprecated in a future release. */
    NVMEDIA_ENCODE_PARAMS_RC_CBR_MINQP    = 4
} NvMediaEncodeParamsRCMode;

/**
 * Holds rate control configuration parameters.
 */
typedef struct {
    /** Holds the rate control mode. */
    NvMediaEncodeParamsRCMode rateControlMode;
    /** Specified number of B frames between two reference frames.
        \valuerange The values between 0 to 10, in increments of 1. */
    uint32_t numBFrames;
    union {
        struct {
            /** Holds the average bitrate (in bits/sec) used for encoding. */
            uint32_t averageBitRate;
            /** Holds the VBV(HRD) buffer size, in bits.
                Set to 0 (zero) to automatically determine the right VBV buffer
                size as per tier and level limits specified in the relevant
                coding standard. */
            uint32_t vbvBufferSize;
            /** Holds the VBV(HRD) initial delay in bits.
                Set to 0 (zero) to use the internally determined VBV initial
                delay.*/
            uint32_t vbvInitialDelay;
        } cbr; /**< Parameters for NVMEDIA_ENCODE_PARAMS_RC_CBR mode */
        struct {
            /** Holds the initial QP to be used for encoding, these values
                would be used for all frames in Constant QP mode. */
            NvMediaEncodeQP constQP;
        } const_qp; /**< Parameters for NVMEDIA_ENCODE_PARAMS_RC_CONSTQP mode */
        struct {
            /** Holds the average bitrate (in bits/sec) used for encoding. */
            uint32_t averageBitRate;
            /** Holds the maximum bitrate for the encoded output. */
            uint32_t maxBitRate;
            /** Holds the VBV(HRD) buffer size, in bits.
                Set to 0 (zero) to automatically determine the right VBV buffer
                size as per tier and level limits specified in the relevant
                coding standard. */
            uint32_t vbvBufferSize;
            /** Holds the VBV(HRD) initial delay in bits.
                Set to 0 (zero) to use the internally determined VBV initial
                delay.*/
            uint32_t vbvInitialDelay;
        } vbr; /**< Parameters for NVMEDIA_ENCODE_PARAMS_RC_VBR mode */
        struct {
            /** Holds the average bitrate (in bits/sec) used for encoding. */
            uint32_t averageBitRate;
            /** Holds the maximum bitrate for the encoded output. */
            uint32_t maxBitRate;
            /** Holds the VBV(HRD) buffer size, in bits.
                Set to 0 (zero) to automatically determine the right VBV buffer
                size as per tier and level limits specified in the relevant
                coding standard. */
            uint32_t vbvBufferSize;
            /** Holds the VBV(HRD) initial delay in bits.
                Set to 0 (zero) to use the internally determined VBV initial
                delay.*/
            uint32_t vbvInitialDelay;
            /** Holds the minimum QP used for rate control. */
            NvMediaEncodeQP minQP;
        } vbr_minqp; /**< Parameters for NVMEDIA_ENCODE_PARAMS_RC_VBR_MINQP mode */
        struct {
            /** Holds the average bitrate (in bits/sec) used for encoding. */
            uint32_t averageBitRate;
            /** Holds the VBV(HRD) buffer size, in bits.
                Set to 0 (zero) to automatically determine the right VBV buffer
                size as per tier and level limits specified in the relevant
                coding standard. */
            uint32_t vbvBufferSize;
            /** Holds the VBV(HRD) initial delay in bits.
                Set to 0 (zero) to use the internally determined VBV initial
                delay.*/
            uint32_t vbvInitialDelay;
            /** Holds the minimum QP used for rate control. */
            NvMediaEncodeQP minQP;
        } cbr_minqp; /**< Parameters for NVMEDIA_ENCODE_PARAMS_RC_CBR_MINQP mode */
    } params; /**< Rate Control parameters */
    /** Use constant QP at frame level or MB row level.
        \valuerange One of the following:
        - true for constant QP at frame level.
        - false for constant QP at MB row level.
        \note This field is taken into consideration when the rate control mode
        is anything other than constQP. */
    bool bConstFrameQP;
    /** Holds the max QP for encoding session when external picture RC hint
        is used.
        \if (SWDOCS_DRIVE_QNX_AGX_SDK || SWDOCS_DRIVE_LNX_AGX_SDK)
        \note This feature is not supported.
        \endif
        \internal This feature is only supported internally via the nvmmlite
        layer.
      */
    int8_t maxSessionQP;
    int8_t reserved[3];
 } NvMediaEncodeRCParams;

/**
 * Blocking type
 */
typedef enum {
    /** Never blocks */
    NVMEDIA_ENCODE_BLOCKING_TYPE_NEVER,
    /** Block only when operation is pending */
    NVMEDIA_ENCODE_BLOCKING_TYPE_IF_PENDING
} NvMediaBlockingType;

/**
 * @defgroup 6x_h264_encoder_api Data types for H.264 encoding
 * Types and declarations for H.264 Encoding.
 * @ingroup 6x_image_encode_api
 * @{
 */

/**
 * Input picture type
 * \hideinitializer
 */
typedef enum {
    /** Auto selected picture type */
    NVMEDIA_ENCODE_PIC_TYPE_AUTOSELECT      = 0,
    /** Forward predicted */
    NVMEDIA_ENCODE_PIC_TYPE_P               = 1,
    /** Bi-directionally predicted picture */
    NVMEDIA_ENCODE_PIC_TYPE_B               = 2,
    /** Intra predicted picture */
    NVMEDIA_ENCODE_PIC_TYPE_I               = 3,
    /** IDR picture */
    NVMEDIA_ENCODE_PIC_TYPE_IDR             = 4,
    /** Starts a new intra refresh cycle if intra
        refresh support is enabled otherwise it
        indicates a P frame */
    NVMEDIA_ENCODE_PIC_TYPE_P_INTRA_REFRESH = 5
} NvMediaEncodePicType;

/**
 * Defines encoding profiles.
 */
typedef enum {
    /** Automatic profile selection */
    NVMEDIA_ENCODE_PROFILE_AUTOSELECT  = 0,

    /** Baseline profile */
    NVMEDIA_ENCODE_PROFILE_BASELINE    = 66,
    /** Main profile */
    NVMEDIA_ENCODE_PROFILE_MAIN        = 77,
    /** Extended profile.
        This value is not supported in the QNX Safety build. */
    NVMEDIA_ENCODE_PROFILE_EXTENDED    = 88,
    /** High profile */
    NVMEDIA_ENCODE_PROFILE_HIGH        = 100,
    /** High10 profile */
    NVMEDIA_ENCODE_PROFILE_HIGH10      = 110,
    /** High422 profile */
    NVMEDIA_ENCODE_PROFILE_HIGH422     = 122,
    /** High 444 profile */
    /** H.264 High444 profile support lossless encode (trans-quant bypass) for
        YUV444 and YUV420 input. Lossy encode is not supported with High444
        profile. */
    NVMEDIA_ENCODE_PROFILE_HIGH444     = 244,
    /** Cavlc444_intra predictive profile */
    NVMEDIA_ENCODE_PROFILE_CAVLC444_INTRA= 44,
} NvMediaEncodeProfile;
 //check eprf
 //professional profile not part of above enum for h265?

/**
 * Defines extended encoding profiles.
 */
typedef enum {
    /** Automatic profile selection */
    NVMEDIA_ENCODE_EXT_PROFILE_AUTOSELECT  = 0,

    /** Baseline profile */
    NVMEDIA_ENCODE_EXT_PROFILE_BASELINE,
    /** Constrained Baseline profile */
    NVMEDIA_ENCODE_EXT_PROFILE_CONSTRAINED_BASELINE,
    /** Main profile */
    NVMEDIA_ENCODE_EXT_PROFILE_MAIN,
    /** Extended profile.
        This value is not supported in the QNX Safety build. */
    NVMEDIA_ENCODE_EXT_PROFILE_EXTENDED,
    /** High profile */
    NVMEDIA_ENCODE_EXT_PROFILE_HIGH,
    /** Progressive High profile */
    NVMEDIA_ENCODE_EXT_PROFILE_PROGRESSIVE_HIGH,
    /** Constrained High profile */
    NVMEDIA_ENCODE_EXT_PROFILE_CONSTRAINED_HIGH,
    /** High10 profile */
    NVMEDIA_ENCODE_EXT_PROFILE_HIGH10,
    /** Progressive High10 profile */
    NVMEDIA_ENCODE_EXT_PROFILE_PROGRESSIVE_HIGH10,
    /** High422 profile */
    NVMEDIA_ENCODE_EXT_PROFILE_HIGH422,
    /** High444 predictive profile */
    NVMEDIA_ENCODE_EXT_PROFILE_HIGH444_PREDICTIVE,
    /** High10_intra predictive profile */
    NVMEDIA_ENCODE_EXT_PROFILE_HIGH10_INTRA,
    /** High422_intra predictive profile */
    NVMEDIA_ENCODE_EXT_PROFILE_HIGH422_INTRA,
    /** High444_intra predictive profile */
    NVMEDIA_ENCODE_EXT_PROFILE_HIGH444_INTRA,
    /** Cavlc444_intra predictive profile */
    NVMEDIA_ENCODE_EXT_PROFILE_CAVLC444_INTRA,
} NvMediaEncodeExtProfile;

/**
 * Defines encoding levels for H264 encoder.
 * \hideinitializer
*/
typedef enum {
    /** Automatic level selection */
    NVMEDIA_ENCODE_LEVEL_AUTOSELECT         = 0,

    /** H.264 Level 1 */
    NVMEDIA_ENCODE_LEVEL_H264_1             = 10,
    /** H.264 Level 1b */
    NVMEDIA_ENCODE_LEVEL_H264_1b            = 9,
    /** H.264 Level 1.1 */
    NVMEDIA_ENCODE_LEVEL_H264_11            = 11,
    /** H.264 Level 1.2 */
    NVMEDIA_ENCODE_LEVEL_H264_12            = 12,
    /** H.264 Level 1.3 */
    NVMEDIA_ENCODE_LEVEL_H264_13            = 13,
    /** H.264 Level 2 */
    NVMEDIA_ENCODE_LEVEL_H264_2             = 20,
    /** H.264 Level 2.1 */
    NVMEDIA_ENCODE_LEVEL_H264_21            = 21,
    /** H.264 Level 2.2 */
    NVMEDIA_ENCODE_LEVEL_H264_22            = 22,
    /** H.264 Level 3 */
    NVMEDIA_ENCODE_LEVEL_H264_3             = 30,
    /** H.264 Level 3.1 */
    NVMEDIA_ENCODE_LEVEL_H264_31            = 31,
    /** H.264 Level 3.2 */
    NVMEDIA_ENCODE_LEVEL_H264_32            = 32,
    /** H.264 Level 4 */
    NVMEDIA_ENCODE_LEVEL_H264_4             = 40,
    /** H.264 Level 4.1 */
    NVMEDIA_ENCODE_LEVEL_H264_41            = 41,
    /** H.264 Level 4.2 */
    NVMEDIA_ENCODE_LEVEL_H264_42            = 42,
    /** H.264 Level 5.0 */
    NVMEDIA_ENCODE_LEVEL_H264_5             = 50,
    /** H.264 Level 5.1 */
    NVMEDIA_ENCODE_LEVEL_H264_51            = 51,
    /** H.264 Level 5.2 */
    NVMEDIA_ENCODE_LEVEL_H264_52            = 52,
    /** \note This value is for internal use only. */
    NVMEDIA_ENCODE_LEVEL_H264_END           = 255
} NvMediaEncodeLevel;

/**
 * Defines H265encoding profiles.
 */
typedef enum {
    /** Automatic profile selection */
    NVMEDIA_ENCODE_H265_PROFILE_AUTOSELECT  = 0,

    /** Main profile */
    NVMEDIA_ENCODE_H265_PROFILE_MAIN      = 1,
    /** Main10 */
    NVMEDIA_ENCODE_H265_PROFILE_MAIN10    = 2,
    /** Main still profile */
    NVMEDIA_ENCODE_H265_PROFILE_MAIN_STILLPICTURE  = 3,
    /** Format range exetnsions profiles */
    NVMEDIA_ENCODE_H265_PROFILES_FORMAT_RANGE_EXTENSIONS = 4,
    /** High throughput profiles */
    NVMEDIA_ENCODE_H265_PROFILES_HIGH_THROUGHPUT = 5,
    /** Screen content coding extensions profiles */
    NVMEDIA_ENCODE_H265_PROFILES_SCREEN_CONTENT_CODING_EXTENSIONS = 9,
    /** High throughput screen content coding extensions profiles */
    NVMEDIA_ENCODE_H265_PROFILES_HIGH_THROUGHPUT_SCREEN_CONTENT_CODING_EXTENSIONS = 11,
} NvMediaEncodeH265Profile;

/**
 * Defines encoding levels for H265 encoder.
 * \hideinitializer
*/
typedef enum {
    /** Automatic level selection */
    NVMEDIA_ENCODE_LEVEL_H265_AUTOSELECT    = 0,

    /** H.265 Level 1.0 */
    NVMEDIA_ENCODE_LEVEL_H265_1             = 30,
    /** H.265 Level 2.0 */
    NVMEDIA_ENCODE_LEVEL_H265_2             = 60,
    /** H.265 Level 2.1 */
    NVMEDIA_ENCODE_LEVEL_H265_21            = 63,
    /** H.265 Level 3.0 */
    NVMEDIA_ENCODE_LEVEL_H265_3             = 90,
    /** H.265 Level 3.1 */
    NVMEDIA_ENCODE_LEVEL_H265_31            = 93,
    /** H.265 Level 4.0 */
    NVMEDIA_ENCODE_LEVEL_H265_4             = 120,
    /** H.265 Level 4.1 */
    NVMEDIA_ENCODE_LEVEL_H265_41            = 123,
    /** H.265 Level 5.0 */
    NVMEDIA_ENCODE_LEVEL_H265_5             = 150,
    /** H.265 Level 5.1 */
    NVMEDIA_ENCODE_LEVEL_H265_51            = 153,
    /** H.265 Level 5.2 */
    NVMEDIA_ENCODE_LEVEL_H265_52            = 156,
    /** H.265 Level 6.0 */
    NVMEDIA_ENCODE_LEVEL_H265_6             = 180,
    /** H.265 Level 6.1 */
    NVMEDIA_ENCODE_LEVEL_H265_61            = 183,
    /** H.265 Level 6.2 */
    NVMEDIA_ENCODE_LEVEL_H265_62            = 186,
    /** \note This value is for internal use only. */
    NVMEDIA_ENCODE_LEVEL_H265_END           = 255

} NvMediaEncodeLevelH265;

/**
 * Defines encoding Picture encode flags.
 * \hideinitializer
*/
typedef enum {
    /** Writes the sequence and picture header in encoded bitstream of the
      * current picture. */
    NVMEDIA_ENCODE_PIC_FLAG_OUTPUT_SPSPPS      = (1 << 0),
    /** Indicates change in rate control parameters from the current picture
        onwards */
    NVMEDIA_ENCODE_PIC_FLAG_RATECONTROL_CHANGE = (1 << 1),
    /** Indicates that this frame is encoded with each slice completely
        independent of other slices in the frame.
        NVMEDIA_ENCODE_CONFIG_H264_ENABLE_CONSTRANED_ENCODING must be set to
        support this encode mode */
    NVMEDIA_ENCODE_PIC_FLAG_CONSTRAINED_FRAME  = (1 << 2)
} NvMediaEncodePicFlags;

/**
 * Defines encode preset level settings.
 */
typedef enum {
    /** Encoder Quality Preset HQ
     *  \n Relative comparison with respect to other presets:
     *  - Performance: Low
     *  - Quality: High */
    NVMEDIA_ENC_PRESET_HQ                      = 0x0,
    /** Encoder Quality Preset HP
     *  \n Relative comparison with respect to other presets:
     *  - Performance: Medium
     *  - Quality: Medium */
    NVMEDIA_ENC_PRESET_HP                      = 0x10,
    /** Encoder Quality Preset UHP
     *  \n Relative comparison with respect to other presets:
     *  - Performance: High
     *  - Quality: Low */
    NVMEDIA_ENC_PRESET_UHP                     = 0x20,
    /** Encoder Quality Preset Default.
     *  Default Encoder settings applied*/
    NVMEDIA_ENC_PRESET_DEFAULT                 = 0x7FFFFFFF
} NvMediaEncPreset;


/**
 * Defines H.264 entropy coding modes.
 * \hideinitializer
*/
typedef enum {
    /** Entropy coding mode is CAVLC */
    NVMEDIA_ENCODE_H264_ENTROPY_CODING_MODE_CAVLC = 0,
    /** Entropy coding mode is CABAC */
    NVMEDIA_ENCODE_H264_ENTROPY_CODING_MODE_CABAC = 1
} NvMediaEncodeH264EntropyCodingMode;

/**
 * Defines H.264 specific Bdirect modes.
 * \hideinitializer
 */
typedef enum {
    /** Spatial BDirect mode.
        This value is not supported in the QNX Safety build. */
    NVMEDIA_ENCODE_H264_BDIRECT_MODE_SPATIAL  = 0,
    /** Disable BDirect mode */
    NVMEDIA_ENCODE_H264_BDIRECT_MODE_DISABLE  = 1,
    /** Temporal BDirect mode.
        This value is not supported in the QNX Safety build. */
    NVMEDIA_ENCODE_H264_BDIRECT_MODE_TEMPORAL = 2
} NvMediaEncodeH264BDirectMode;

/**
 * Defines H.264 specific Adaptive Transform modes.
 * \hideinitializer
 */
typedef enum {
    /** Specifies that Adaptive Transform 8x8 mode is auto selected by the
        encoder driver. */
    NVMEDIA_ENCODE_H264_ADAPTIVE_TRANSFORM_AUTOSELECT = 0,
    /** Specifies that Adaptive Transform 8x8 mode is disabled. */
    NVMEDIA_ENCODE_H264_ADAPTIVE_TRANSFORM_DISABLE    = 1,
    /** Specifies that Adaptive Transform 8x8 mode must be used. */
    NVMEDIA_ENCODE_H264_ADAPTIVE_TRANSFORM_ENABLE     = 2
} NvMediaEncodeH264AdaptiveTransformMode;

/**
 * \hideinitializer
 * Defines motion prediction exclusion flags for H.264.
 * \deprecated This feature will be deprecated in a future release.
 */
typedef enum {
    /** Disable Intra 4x4 vertical prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_VERTICAL_PREDICTION             = (1 << 0),
    /** Disable Intra 4x4 horizontal prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_HORIZONTAL_PREDICTION           = (1 << 1),
    /** Disable Intra 4x4 DC prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_DC_PREDICTION                   = (1 << 2),
    /** Disable Intra 4x4 diagonal down left prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_DIAGONAL_DOWN_LEFT_PREDICTION   = (1 << 3),
    /** Disable Intra 4x4 diagonal down right prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_DIAGONAL_DOWN_RIGHT_PREDICTION  = (1 << 4),
    /** Disable Intra 4x4 vertical right prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_VERTICAL_RIGHT_PREDICTION       = (1 << 5),
    /** Disable Intra 4x4 horizontal down prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_HORIZONTAL_DOWN_PREDICTION      = (1 << 6),
    /** Disable Intra 4x4 vertical left prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_VERTICAL_LEFT_PREDICTION        = (1 << 7),
    /** Disable Intra 4x4 horizontal up prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_HORIZONTAL_UP_PREDICTION        = (1 << 8),

    /** Disable Intra 8x8 vertical prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_VERTICAL_PREDICTION             = (1 << 9),
    /** Disable Intra 8x8 horizontal prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_HORIZONTAL_PREDICTION           = (1 << 10),
    /** Disable Intra 8x8 DC prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_DC_PREDICTION                   = (1 << 11),
    /** Disable Intra 8x8 diagonal down left prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_DIAGONAL_DOWN_LEFT_PREDICTION   = (1 << 12),
    /** Disable Intra 8x8 diagonal down right prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_DIAGONAL_DOWN_RIGHT_PREDICTION  = (1 << 13),
    /** Disable Intra 8x8 vertical right prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_VERTICAL_RIGHT_PREDICTION       = (1 << 14),
    /** Disable Intra 8x8 horizontal down prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_HORIZONTAL_DOWN_PREDICTION      = (1 << 15),
    /** Disable Intra 8x8 vertical left prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_VERTICAL_LEFT_PREDICTION        = (1 << 16),
    /** Disable Intra 8x8 horizontal up prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_HORIZONTAL_UP_PREDICTION        = (1 << 17),

    /** Disable Intra 16x16 vertical prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_16x16_VERTICAL_PREDICTION           = (1 << 18),
    /** Disable Intra 16x16 horizontal prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_16x16_HORIZONTAL_PREDICTION         = (1 << 19),
    /** Disable Intra 16x16 DC prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_16x16_DC_PREDICTION                 = (1 << 20),
    /** Disable Intra 16x16 plane prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_16x16_PLANE_PREDICTION              = (1 << 21),

    /** Disable Intra chroma vertical prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_CHROMA_VERTICAL_PREDICTION          = (1 << 22),
    /** Disable Intra chroma horizontal prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_CHROMA_HORIZONTAL_PREDICTION        = (1 << 23),
    /** Disable Intra chroma DC prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_CHROMA_DC_PREDICTION                = (1 << 24),
    /** Disable Intra chroma plane prediction */
    NVMEDIA_ENCODE_DISABLE_INTRA_CHROMA_PLANE_PREDICTION             = (1 << 25),

    /** Disable Inter L0 partition 16x16 prediction */
    NVMEDIA_ENCODE_DISABLE_INTER_L0_16x16_PREDICTION                 = (1 << 26),
    /** Disable Inter L0 partition 16x8 prediction */
    NVMEDIA_ENCODE_DISABLE_INTER_L0_16x8_PREDICTION                  = (1 << 27),
    /** Disable Inter L0 partition 8x16 prediction */
    NVMEDIA_ENCODE_DISABLE_INTER_L0_8x16_PREDICTION                  = (1 << 28),
    /** Disable Inter L0 partition 8x8 prediction */
    NVMEDIA_ENCODE_DISABLE_INTER_L0_8x8_PREDICTION                   = (1 << 29)
} NvMediaEncodeH264MotionPredictionExclusionFlags;

/**
 * \hideinitializer
 * Defines motion search mode control flags for H.264.
 * \deprecated This feature will be deprecated in a future release.
 */
typedef enum {
    /** IP Search mode bit Intra 4x4 */
    NVMEDIA_ENCODE_ENABLE_IP_SEARCH_INTRA_4x4                = (1 << 0),
    /** IP Search mode bit Intra 8x8 */
    NVMEDIA_ENCODE_ENABLE_IP_SEARCH_INTRA_8x8                = (1 << 1),
    /** IP Search mode bit Intra 16x16 */
    NVMEDIA_ENCODE_ENABLE_IP_SEARCH_INTRA_16x16              = (1 << 2),
    /** Enable self_temporal_refine */
    NVMEDIA_ENCODE_ENABLE_SELF_TEMPORAL_REFINE               = (1 << 3),
    /** Enable self_spatial_refine */
    NVMEDIA_ENCODE_ENABLE_SELF_SPATIAL_REFINE                = (1 << 4),
    /** Enable coloc_refine */
    NVMEDIA_ENCODE_ENABLE_COLOC_REFINE                       = (1 << 5),
    /** Enable external_refine */
    NVMEDIA_ENCODE_ENABLE_EXTERNAL_REFINE                    = (1 << 6),
    /** Enable const_mv_refine */
    NVMEDIA_ENCODE_ENABLE_CONST_MV_REFINE                    = (1 << 7),
    /** Enable the flag set */
    NVMEDIA_ENCODE_MOTION_SEARCH_CONTROL_FLAG_VALID          = (1 << 31)
} NvMediaEncodeH264MotionSearchControlFlags;

/**
 * Specifies the frequency of the writing of Sequence and Picture parameters
 * for H.264.
 * \hideinitializer
 */
typedef enum {
    /** Repeating of SPS/PPS is disabled */
    NVMEDIA_ENCODE_SPSPPS_REPEAT_DISABLED          = 0,
    /** SPS/PPS is repeated for every intra frame */
    NVMEDIA_ENCODE_SPSPPS_REPEAT_INTRA_FRAMES      = 1,
    /** SPS/PPS is repeated for every IDR frame */
    NVMEDIA_ENCODE_SPSPPS_REPEAT_IDR_FRAMES        = 2
} NvMediaEncodeH264SPSPPSRepeatMode;

/**
 * Specifies the encoder get attribute type.
 * This can be extended to get other encoding parameter information.
 * \hideinitializer
 */
typedef enum {
    /** This attribute is used to get SPS data for current encoding session */
    NvMediaEncAttr_GetSPS = 1,
    /** This attribute is used to get PPS data for current encoding session */
    NvMediaEncAttr_GetPPS,
    /** This attribute is used to get VPS data for current h265 encoding
        session */
    NvMediaEncAttr_GetVPS
} NvMediaEncAttrType;

/**
 * Define H.264 pic_order_cnt_type
 * \hideinitializer
 */
typedef enum {
    /** Automatic level selection */
    NVMEDIA_ENCODE_H264_POC_TYPE_AUTOSELECT     = 0,
    /** Pic_order_cnt_type 0 */
    NVMEDIA_ENCODE_H264_POC_TYPE_0              = 1,
    /** Pic_order_cnt_type 2 */
    NVMEDIA_ENCODE_H264_POC_TYPE_2              = 2
} NvMediaEncodeH264POCType;

/**
 * \hideinitializer
 * @brief Maximum encoded header info size
 */
#define MAX_NON_SLICE_DATA_SIZE 2048U

/**
 * This is used to get header info (SPS/PPS/VPS) using GetAttribute call.
 * HeaderInfo type is defined in \ref NvMediaEncAttrType
 */
typedef struct {
    /** Nal size for header */
    uint32_t ulNalSize;
    /** header data passed on this buffer */
    uint8_t data[MAX_NON_SLICE_DATA_SIZE];
} NvMediaNalData;

/**
 * Holds H264 video usability information parameters.
 */
typedef struct {
    /** If set to true, it specifies that the \a aspectRatioIdc is
        present. */
    bool aspectRatioInfoPresentFlag;
    /** Holds the aspect ratio IDC
        (as defined in Annex E of the ITU-T Specification).*/
    uint8_t aspectRatioIdc;
    /** If \a aspectRatioIdc is Extended SAR then it indicates horizontal size
        of the sample aspect ratio (in arbitrary units). */
    uint16_t aspectSARWidth;
    /** If aspectRatioIdc is Extended SAR then it indicates vertical size
        of the sample aspect ratio (in the same arbitrary units as
        aspectSARWidth) */
    uint16_t aspectSARHeight;
    /** If set to true, it specifies that the overscanInfo is present */
    bool overscanInfoPresentFlag;
    /** Holds the overscan info (as defined in Annex E of the ITU-T
        Specification). */
    bool overscanAppropriateFlag;
    /** If set to true, it specifies  that the videoFormat,
        videoFullRangeFlag and colourDescriptionPresentFlag are present. */
    bool videoSignalTypePresentFlag;
    /** Holds the source video format
        (as defined in Annex E of the ITU-T Specification).*/
    uint8_t videoFormat;
    /** Holds the output range of the luma and chroma samples
        (as defined in Annex E of the ITU-T Specification). */
    bool videoFullRangeFlag;
    /** If set to true, it specifies that the colourPrimaries,
        transferCharacteristics and colourMatrix are present. */
    bool colourDescriptionPresentFlag;
    /** Holds color primaries for converting to RGB (as defined in Annex E of
        the ITU-T Specification). */
    uint8_t colourPrimaries;
    /** Holds the opto-electronic transfer characteristics to use
        (as defined in Annex E of the ITU-T Specification). */
    uint8_t transferCharacteristics;
    /** Holds the matrix coefficients used in deriving the luma and chroma from
        the RGB primaries (as defined in Annex E of the ITU-T Specification). */
    uint8_t colourMatrix;
    /** Holds that num_units_in_tick, time_scale and fixed_frame_rate_flag are
        present in the bitstream (as defined in Annex E of the ITU-T
        Specification). */
    bool timingInfoPresentFlag;
    /** Holds the bitstream restriction info (as defined in Annex E of the ITU-T
        Specification). */
    bool bitstreamRestrictionFlag;
} NvMediaEncodeConfigH264VUIParams;

/**
 * Holds an external motion vector hint with counts per block type.
 * \deprecated This feature will be deprecated in a future release.
 */
typedef struct {
    /** Holds the number of candidates per 16x16 block. */
    uint32_t   numCandsPerBlk16x16;
    /** Holds the number of candidates per 16x8 block. */
    uint32_t   numCandsPerBlk16x8;
    /** Holds the number of candidates per 8x16 block. */
    uint32_t   numCandsPerBlk8x16;
    /** Holds the number of candidates per 8x8 block. */
    uint32_t   numCandsPerBlk8x8;
} NvMediaEncodeExternalMeHintCountsPerBlocktype;

/**
 * Holds an External Motion Vector hint.
 * \deprecated This feature will be deprecated in a future release.
 */
typedef struct {
    /** Holds the x component of integer pixel MV (relative to current MB) S12.0. */
    int32_t mvx        : 12;
    /** Holds the y component of integer pixel MV (relative to current MB) S10.0 .*/
    int32_t mvy        : 10;
    /** Holds the reference index (31=invalid). Current we support only 1
        reference frame per direction for external hints, so refidx must be 0. */
    uint32_t refidx     : 5;
    /** Holds the direction of motion estimation . 0=L0 1=L1.*/
    uint32_t dir        : 1;
    /** Holds the block partition type. 0=16x16 1=16x8 2=8x16 3=8x8 (blocks in
        partition must be consecutive).*/
    uint32_t partType   : 2;
    /** Set to true for the last MV of (sub) partition  */
    uint32_t lastofPart : 1;
    /** Set to true for the last MV of macroblock. */
    uint32_t lastOfMB   : 1;
} NvMediaEncodeExternalMEHint;

/**
 * Defines H264 encoder configuration features.
 * \hideinitializer
 */
typedef enum {
    /** Enable to write access unit delimiter syntax in bitstream */
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_OUTPUT_AUD               = (1 << 0),
    /** Enable gradual decoder refresh or intra refresh.
       If the GOP structure uses B frames this will be ignored */
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_INTRA_REFRESH            = (1 << 1),
    /** Enable dynamic slice mode. Client must specify max slice
       size using the \ref NvMediaEncodeConfigH264.maxSliceSizeInBytes field.
       This setting is not supported in the QNX Safety build. */
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_DYNAMIC_SLICE_MODE       = (1 << 2),
    /** Enable constrainedFrame encoding where each slice in the
       constrained picture is independent of other slices. */
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_CONSTRANED_ENCODING      = (1 << 3),
    /** Enable lossless compression.
        \note This feature is not supported. */
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_LOSSLESS_COMPRESSION     = (1 << 4),
    /** Enable slice level output encoding. This enables delivery
      encoded data slice by slice to client to reduce encode latency.
      This setting is not supported in the QNX Safety build. */
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_SLICE_LEVEL_OUTPUT       = (1 << 5),
    /** Enable RTP mode output. NAL unit start code will be replaced
       with NAL size for the NAL units. */
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_RTP_MODE_OUTPUT          = (1 << 6),
    /** Enable support for external picture RC hint.
        \if (SWDOCS_DRIVE_QNX_AGX_SDK || SWDOCS_DRIVE_LNX_AGX_SDK)
        \note This feature is not supported.
        \endif
        \internal This feature is only supported internally via the nvmmlite
        layer.
      */
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_EXT_PIC_RC_HINT          = (1 << 7),
    /** Enable support for dynamic reference picture set RPS/DPB
       control support. After receiving an input buffer and result
       of previous frame encoding, based on real time information,
       Some client software determines properties for the new input
       buffer (long term/short term/non-referenced,
       frame number/poc/LT index).
       This setting is not supported in the QNX Safety build. */
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_DYNAMIC_RPS              = (1 << 8),
    /** Enable support for motion vector buffer dump. This will enable
       motion vector dump. Motion vector buffer will be appended at the
       end of encoded bitstream data.
       This setting is not supported in the QNX Safety build. */
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_MV_BUFFER_DUMP           = (1 << 9),
    /** Enable encoder profiling. Profiling information would be added
       as part of output extradata.
       This setting is not supported in the QNX Safety build. */
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_PROFILING                = (1 << 10),
    /** Enable support to use client provided Initial QP for all frame types */
    NVMEDIA_ENCODE_CONFIG_H264_INIT_QP                         = (1 << 11),
    /** Enable support to use client provided QP max for all frame types */
    NVMEDIA_ENCODE_CONFIG_H264_QP_MAX                          = (1 << 12),
    /** Enable support to use 4 byte start code in all the slices in a picture */
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_FOUR_BYTE_START_CODE     = (1 << 13),
    /** Enable ultra fast encoding. It overrides some of the quality settings
        to achieve ultra fast encoding. This is equivalent to setting
        \ref NVMEDIA_ENCODE_QUALITY_L0 as the \ref NvMediaEncodeQuality. */
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_ULTRA_FAST_ENCODE        = (1 << 14),
    /** Enable support for motion vector buffer dump in a simplified (V2)
        format. Motion vector buffer will be appended to the end of encoded
        bitstream data retrieved using \ref NvMediaIEPGetBits. Refer
        NvMediaEncodeOutputExtradata for more information regarding the
        format of the dumped MV output. Either
        NVMEDIA_ENCODE_CONFIG_H264_ENABLE_MV_BUFFER_DUMP_V2 or
        NVMEDIA_ENCODE_CONFIG_H264_ENABLE_MV_BUFFER_DUMP should be enabled at a
        given time. If both are enabled, V2 format will be used.

        It should be ensured that size of \ref NvMediaBitstreamBuffer passed to
        \ref NvMediaIEPGetBits has sufficient space to store the MV buffer
        dump. The \a numBytesAvailable returned by \ref NvMediaIEPBitsAvailable
        or \a numBytes returned by \ref NvMediaIEPGetBits does not take MV
        buffer dump size into account - represents only the size of the encoded
        bits that are available. While allocating bitstream buffers, the
        additional amount of space required in bitstream to accommodate MV
        buffer data needs to be added. This can be calculated as follows:
        \code
            mvBufferSize = sizeof(NvMediaEncodeMVBufferHeader) +
                ALIGN_256(numMacroBlocks * sizeof(NvMediaEncodeMVData))
            where,
            NumMacroBlocks =
                (ALIGN_16(InputWidth)/16) * (ALIGN_16(InputHeight)/16)

            bitstreamSize = ALIGN_8(bitsAvailable) + mvBufferSize
        \endcode
        ALIGN_N refers to an operation which returns a multiple of N, greater
        than or equal to a number, closest to the number.

        \note This setting is not supported in the QNX Safety build. */
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_MV_BUFFER_DUMP_V2         = (1 << 15),
} NvMediaEncodeH264Features;

/**
 * Holds an H264 encoder configuration.
 */
typedef struct {
    /** Holds bit-wise OR`ed configuration feature flags.
       See enum \ref NvMediaEncodeH264Features. */
    uint32_t features;
    /** Holds the number of pictures in one GOP. A low-latency application
       client can set @a goplength to \ref NVMEDIA_ENCODE_INFINITE_GOPLENGTH
       so that keyframes are not inserted automatically. */
    uint32_t gopLength;
    /** Holds the rate control parameters for the current encoding session. */
    NvMediaEncodeRCParams rcParams;
    /** Holds the frequency of the writing of Sequence and Picture
       parameters. */
    NvMediaEncodeH264SPSPPSRepeatMode repeatSPSPPS;
    /** Holds the IDR interval. If not set, the interval is made equal to
       @a gopLength. A low-latency application client can set the IDR interval
       to NVMEDIA_ENCODE_INFINITE_GOPLENGTH so that IDR frames are not inserted
       automatically. */
    uint32_t idrPeriod;
    /** Holds a number that is 1 less than the desired number of slices
       per frame. \ref NVMEDIA_ENCODE_CONFIG_H264_ENABLE_DYNAMIC_SLICE_MODE
       should NOT be set in \ref features for this setting to take effect.
       \valuerange 0 to 31 in increments of 1. */
    uint16_t numSliceCountMinus1;
    /** Holds the deblocking filter mode.
        \valuerange 0 to 2 in increments of 1. */
    uint8_t disableDeblockingFilterIDC;
    /** Holds the Adaptive Transform Mode. */
    NvMediaEncodeH264AdaptiveTransformMode adaptiveTransformMode;
    /** Holds the BDirect mode. */
    NvMediaEncodeH264BDirectMode bdirectMode;
    /** Holds the entropy coding mode. */
    NvMediaEncodeH264EntropyCodingMode entropyCodingMode;
    /** Holds the interval between frames that triggers a new
       intra refresh cycle. This cycle lasts for @a intraRefreshCnt frames.
       This value is used only if
       \ref NVMEDIA_ENCODE_CONFIG_H264_ENABLE_INTRA_REFRESH is
       set in @a features.
       \ref NVMEDIA_ENCODE_PIC_TYPE_P_INTRA_REFRESH picture type also triggers
       a new intra
       refresh cycle and resets the current intra refresh period.
       Set to zero to suppress triggering of automatic refresh cycles.
       In this case only NVMEDIA_ENCODE_PIC_TYPE_P_INTRA_REFRESH picture type
       can trigger a new refresh cycle. */
    uint32_t intraRefreshPeriod;
    /** Holds the number of frames over which intra refresh happens.
       This value must be less than or equal to @a intraRefreshPeriod.
       Set to zero to disable intra refresh functionality.
       If it is set to one, then
       after every intraRefreshPeriod frames the encoded P frame contains only
       intra predicted macroblocks.
       This value is used only if
       NVMEDIA_ENCODE_CONFIG_H264_ENABLE_INTRA_REFRESH is set in @a features. */
    uint32_t intraRefreshCnt;
    /** Holds the maximum slice size in bytes for dynamic slice mode.
       The client must set
       \ref NVMEDIA_ENCODE_CONFIG_H264_ENABLE_DYNAMIC_SLICE_MODE in @a features
       for this setting to take effect. */
    uint32_t maxSliceSizeInBytes;
    /** Holds the number of macroblocks per slice. Set to 0 if a fixed number
        of macroblocks not required or if @a maxSliceSizeInBytes or
        @a numSliceCountMinus1 is set to a non-zero value. */
    uint32_t numMacroblocksPerSlice;
    /** Holds the H.264 video usability information pamameters.
        Set to NULL if VUI is not needed. */
    NvMediaEncodeConfigH264VUIParams *h264VUIParameters;
    /** Holds bitwise OR`ed exclusion flags. See enum
        \ref NvMediaEncodeH264MotionPredictionExclusionFlags.
        \deprecated This feature will be deprecated in a future release. */
    uint32_t motionPredictionExclusionFlags;
    /** Holds pic_ordec_cnt_type. */
    NvMediaEncodeH264POCType pocType;
    /** Holds the initial QP parameters. The client must set
        \ref NVMEDIA_ENCODE_CONFIG_H264_INIT_QP in @a features to
        use this.
        \valuerange QP values can lie between 1 to 51 in increments of 1. */
    NvMediaEncodeQP initQP;
    /** Holds the maximum QP parameters. The client must set
        \ref NVMEDIA_ENCODE_CONFIG_H264_QP_MAX in @a features to use this.
        \valuerange QP values can lie between 1 to 51 in increments of 1, must
        be greater than \ref NvMediaEncodeRCParams::minQP. */
    NvMediaEncodeQP maxQP;
    /** Enable/disable weighted prediction.
        \valuerange 0 to disable, non-zero value to enable */
    uint8_t enableWeightedPrediction;
    /** Holds the encode quality pre-set. See enum NvMediaEncPreset.
        Recommended pre-setting is \ref NVMEDIA_ENC_PRESET_HP */
    NvMediaEncPreset encPreset;
} NvMediaEncodeConfigH264;

/**
 *  H.264 specific User SEI message
 */
typedef struct {
    /** SEI payload size in bytes. SEI payload must be byte aligned, as
        described in Annex D */
    uint32_t payloadSize;
    /** SEI payload types and syntax can be found in Annex D of the H.264
        Specification. */
    uint32_t payloadType;
    /** Pointer to user data. */
    uint8_t *payload;
} NvMediaEncodeH264SEIPayload;

/**
 * Holds H264-specific encode initialization parameters.
 */
typedef struct {
    /** Holds the encode width.
        \if (SWDOCS_DRIVE_QNX_AGX_SDK || SWDOCS_DRIVE_LNX_AGX_SDK)
        \valuerange 160 to 4096 in increments of 2.
        \endif
      */
    uint16_t encodeWidth;
    uint16_t reserved1;
    /** Holds the encode height.
        \if (SWDOCS_DRIVE_QNX_AGX_SDK || SWDOCS_DRIVE_LNX_AGX_SDK)
        \valuerange 64 to 4096 in increments of 2.
        \endif
      */
    uint16_t encodeHeight;
    uint16_t reserved2;
    /** Set this to true for limited-RGB (16-235) input.
        \note This feature is not supported. */
    bool enableLimitedRGB;
    /** Holds the numerator for frame rate used for encoding in frames per second
        ( Frame rate = frameRateNum / frameRateDen ).
        \valuerange Frame rate can lie between 1 and 60 (float value). */
    uint32_t frameRateNum;
    /** Holds the denominator for frame rate used for encoding in frames per second
        ( Frame rate = frameRateNum / frameRateDen ).
        \valuerange Frame rate can lie between 1 and 60 (float value). */
    uint32_t frameRateDen;
    /** Holds the encoding profile. Client is recommended to set this to
        \ref NVMEDIA_ENCODE_PROFILE_AUTOSELECT in order to enable the Encode
        interface to select the correct profile. If a specific profile needs to
        be configured, this field can be set to one of the supported
        enumerations from \ref NvMediaEncodeProfile. */
    uint8_t profile;
    /** Enables extended encoding profiles. */
    bool enableExtProfile;
    /** Holds the extended encoding profile. Client is recommended to set this
        in addition to profile to set extended profile. This field can be set to
        one of the supported enumerations from \ref NvMediaEncodeExtProfile. */
    NvMediaEncodeExtProfile extProfile;
    /** Holds the encoding level. Client is recommended to set this to
        NVMEDIA_ENCODE_LEVEL_AUTOSELECT in order to enable the Encode interface
        to select the correct level. If a specific level needs to be configured,
        this field can be set to one of the enumerations from
        \ref NvMediaEncodeLevel. If the specified level is lower than that
        which is required by the H264 standard, the encoder auto-corrects it to
        the minimum required level. */
    uint8_t level;
    /** Holds the maximum number of reference frames used for encoding.
        \valuerange 0
        \n For I-frame only mode (no reference frames), i.e.,
            \ref NvMediaEncodeInitializeParamsH264.enableAllIFrames is set to 1
            and \ref NvMediaEncodeRCParams.numBFrames is set to 0.
        \valuerange 1 to 8
        \n For encoding with I and P frame types (p-frames as reference), i.e.,
            \ref NvMediaEncodeInitializeParamsH264.enableAllIFrames is set to 0
            and \ref NvMediaEncodeRCParams.numBFrames is set to 0.
        \valuerange 2 to 8
        \n For encoding with I, P and B frame types (p and b-frames as
            reference), i.e.,
            \ref NvMediaEncodeInitializeParamsH264.enableAllIFrames is set to 0,
            \ref NvMediaEncodeRCParams.numBFrames is set to a non-zero value and
            \ref NvMediaEncodeInitializeParamsH264.useBFramesAsRef is set to 1.
        \valuerange 2 to 8
        \n For encoding with I, P and B frame types (only p-frames as
            reference), i.e.,
            \ref NvMediaEncodeInitializeParamsH264.enableAllIFrames is set to 0,
            \ref NvMediaEncodeRCParams.numBFrames is set to a non-zero value and
            \ref NvMediaEncodeInitializeParamsH264.useBFramesAsRef is set to 0.
        */
    uint8_t maxNumRefFrames;
    /** Set to true to enable external ME hints.
        Currently this feature is not supported if B frames are used.
        This setting is not supported in the QNX Safety build.
        \deprecated This feature will be deprecated in a future release. */
    bool enableExternalMEHints;
    /** If Client wants to pass external motion vectors in
        \ref NvMediaEncodePicParamsH264 meExternalHints buffer it must specify
        the maximum number of hint candidates per block per direction for the
        encode session. The \ref NvMediaEncodeInitializeParamsH264
        maxMEHintCountsPerBlock[0] is for L0 predictors and
        \ref NvMediaEncodeInitializeParamsH264 maxMEHintCountsPerBlock[1] is
        for L1 predictors. This client must also set
        \ref NvMediaEncodeInitializeParamsH264 enableExternalMEHints to
        true.
        \deprecated This feature will be deprecated in a future release. */
    NvMediaEncodeExternalMeHintCountsPerBlocktype maxMEHintCountsPerBlock[2];
    /** Enable support for recon CRC generation. NVMEDIA will
        allocate extra surface for recon CRC calculation. This can
        be enabled at run time for any frame by enabling recon CRC and
        passing recon CRC rectangle.
        \if (SWDOCS_DRIVE_QNX_AGX_SDK || SWDOCS_DRIVE_LNX_AGX_SDK)
        \note This feature is not supported.
        \endif
        \internal This feature is only supported internally via the nvmmlite
        layer.
      */
    bool enableReconCRC;
    /** If client want to do MVC encoding then this flag need to be set.
        \note This feature is not supported. */
    bool enableMVC;
    /** Enable region of interest encoding. Region of interest encoding
        parameters are passed with Input extra data parameters.
        \if (SWDOCS_DRIVE_QNX_AGX_SDK || SWDOCS_DRIVE_LNX_AGX_SDK)
        \note This feature is not supported.
        \endif
        \internal This feature is only supported internally via the nvmmlite
        layer.
      */
    bool enableROIEncode;
    /** Use slice encode to reduce latency in getting encoded buffers.
        This setting is not supported in the QNX Safety build. */
    bool enableSliceEncode;
    /** Enables B frames to be used as reference frames.
        \valuerange 0 to 1
        - 0: Disables the use of b-frames as reference frames.
        - 1: Enables the use of b-frames as reference frames.
        \note For this field to take effect
        \ref NvMediaEncodeRCParams.numBFrames needs to be set to a non-zero
        value and \ref NvMediaEncodeInitializeParamsH264.maxNumRefFrames should
        be set to an appropriate value. */
    uint8_t useBFramesAsRef;
    uint8_t reserved3[3];
    /** Enable 2 pass RC support. First pass RC can be run on full or quarter
        resolution.
        \note \a enableTwoPassRC is an internal feature. */
    bool enableTwoPassRC;
    /** Enable 2 pass RC with quarter resolution first pass.
        \if (SWDOCS_DRIVE_QNX_AGX_SDK || SWDOCS_DRIVE_LNX_AGX_SDK)
        \note This feature is not supported.
        \endif
        \internal This feature is only supported internally via the nvmmlite
        layer.
      */
    bool enableSourceHalfScaled;
    /** Number of views used for MVC.
        \note This feature is not supported. */
    uint32_t mvcNumViews             : 4;
    /** Enable external picture rate control.
        \note This feature is not supported. */
    uint32_t enableExternalPictureRC : 1;
    /** Encode all frames as I frames.
        \valuerange 0 to disable, non-zero value to enable */
    uint32_t enableAllIFrames : 1;
    /** Enables feature to allocate memory as per optimizations*/
    uint32_t enableMemoryOptimization : 1;
    /** Enables video anonymization of PII regions
        \valuerange 0 to 1
        - 0: Disables the video anonymization feature.
        - 1: Enables the video anonymization feature.
        This setting is not supported in the QNX Safety build. */
    uint32_t enableAnonEncode : 1;
    /** Add padding */
    uint32_t reserved                : 24;
} NvMediaEncodeInitializeParamsH264;

/**
 * H264 specific encoder picture params. Sent on a per frame basis.
 */
typedef struct {
    /** Holds input picture type. */
    NvMediaEncodePicType pictureType;
    /** Holds bit-wise OR`ed encode pic flags.
        See enum \ref NvMediaEncodePicFlags. */
    uint32_t encodePicFlags;
    /** Specifies the number of B-frames that follow the current frame.
        This number can be set only for reference frames and the frames
        that follow the current frame must be nextBFrames count of B-frames.
        B-frames are supported only if the profile is greater than
        NVMEDIA_ENCODE_PROFILE_BASELINE and the maxNumRefFrames is set to 2.
        Set to zero if no B-frames are needed. */
    uint32_t nextBFrames;
    /** Holds the rate control parameters from the current frame onward
        if the NVMEDIA_ENCODE_PIC_FLAG_RATECONTROL_CHANGE is set in the
        encodePicFlags.
        \note The \ref NvMediaEncodeRCParams.rateControlMode cannot be changed
        on a per-frame basis. Only the associated rate control parameters can be
        changed. */
    NvMediaEncodeRCParams rcParams;
    /** Holds the number of elements allocated in \ref seiPayloadArray array.
        Set to 0 if no SEI messages are needed */
    uint32_t seiPayloadArrayCnt;
    /** Array of SEI payloads which will be inserted for this frame. */
    NvMediaEncodeH264SEIPayload *seiPayloadArray;
    /** Holds the number of hint candidates per block per direction for the
        current frame. meHintCountsPerBlock[0] is for L0 predictors and
        meHintCountsPerBlock[1] is for L1 predictors. The candidate count in
        \ref NvMediaEncodePicParamsH264 meHintCountsPerBlock[lx] must never
        exceed \ref NvMediaEncodeInitializeParamsH264 maxMEHintCountsPerBlock[lx]
        provided during encoder initialization.
        \deprecated This feature will be deprecated in a future release. */
    NvMediaEncodeExternalMeHintCountsPerBlocktype meHintCountsPerBlock[2];
    /** Holds the pointer to ME external hints for the current frame.
        The size of ME hint buffer must be equal to number of macroblocks
        multiplied by the total number of candidates per macroblock.
        The total number of candidates per MB per direction =
        <pre>
        1*meHintCountsPerBlock[Lx].numCandsPerBlk16x16 +
        2*meHintCountsPerBlock[Lx].numCandsPerBlk16x8 +
        2*meHintCountsPerBlock[Lx].numCandsPerBlk8x16 +
        4*meHintCountsPerBlock[Lx].numCandsPerBlk8x8
        </pre>
        For frames using bidirectional ME, the total number of candidates for a
        single macroblock is the sum of the total number of candidates per MB
        for each direction (L0 and L1).

        If no external ME hints are needed, set this field to NULL.
        \deprecated This feature will be deprecated in a future release. */
    union
    {
        NvMediaEncodeExternalMEHint *meExternalHints;
        uint8_t *meHints;
    };
    /** Holds the numerator for frame rate used for encoding in frames per second
        ( Frame rate = frameRateNum / frameRateDen ).
        \valuerange Frame rate can lie between 1 and 60 (float value). */
    uint32_t frameRateNum;
    /** Holds the numerator for frame rate used for encoding in frames per second
        ( Frame rate = frameRateNum / frameRateDen ).
        \valuerange Frame rate can lie between 1 and 60 (float value). */
    uint32_t frameRateDen;
    /** Holds the viewId of current picture */
    uint32_t viewId;
    /** Holds the PII regions information
        This feature is not supported in the QNX Safety build. */
    NvMediaEncPIIParams PIIparams[NVMEDIA_ENCODE_MAX_PII_REGIONS];
    /** Holds the number of PII regions for current pic
        This feature is not supported in the QNX Safety build. */
    uint32_t numPIIRegions;
} NvMediaEncodePicParamsH264;
/** @} <!-- Ends h264_encoder_api H.264 Encoder -> */

/**
 * @defgroup 6x_h265_encoder_api Data types for H.265/HEVC encoding
 * Types and declarations for H.265/HEVC Encoding.
 * @ingroup 6x_image_encode_api
 * @{
 */
/**
 * Holds the H.265 video usability information parameters.
 */
typedef struct {
    /** If set to true, specifies the \a aspectRatioIdc is present. */
    bool aspectRatioInfoPresentFlag;
    /** Holds the aspect ratio IDC
        (as defined in Annex E of the ITU-T specification).*/
    uint8_t aspectRatioIdc;
    /** If \a aspectRatioIdc is Extended SAR it indicates horizontal size
        of the sample aspect ratio (in arbitrary units). */
    uint16_t aspectSARWidth;
    /** If \a aspectRatioIdc is Extended SAR it indicates vertical size
        of the sample aspect ratio (in the same arbitrary units as
        \a aspectSARWidth). */
    uint16_t aspectSARHeight;
    /** If set to true, it specifies that the \a overscanInfo is
        present. */
    bool overscanInfoPresentFlag;
    /** Holds the overscan info (as defined in Annex E of the ITU-T
        Specification). */
    bool overscanAppropriateFlag;
    /** If set to true, it specifies that the \a videoFormat,
        \a videoFullRangeFlag, and \a colourDescriptionPresentFlag are present.
        */
    bool videoSignalTypePresentFlag;
    /** Holds the source video format
        (as defined in Annex E of the ITU-T Specification).*/
    uint8_t videoFormat;
    /** Holds the output range of the luma and chroma samples
        (as defined in Annex E of the ITU-T Specification). */
    bool videoFullRangeFlag;
    /** If set to true, it specifies that the \a colourPrimaries,
        \a transferCharacteristics, and \a colourMatrix are present. */
    bool colourDescriptionPresentFlag;
    /** Holds color primaries for converting to RGB (as defined in Annex E of
        the ITU-T Specification). */
    uint8_t colourPrimaries;
    /** Holds the opto-electronic transfer characteristics to use
        (as defined in Annex E of the ITU-T Specification). */
    uint8_t transferCharacteristics;
    /** Holds the matrix coefficients used in deriving the luma and chroma from
        the RGB primaries (as defined in Annex E of the ITU-T Specification). */
    uint8_t matrixCoeffs;
    /** Holds that num_units_in_tick, time_scale and fixed_frame_rate_flag are
        present in the bitstream (as defined in Annex E of the ITU-T
        Specification). */
    bool vuiTimingInfoPresentFlag;
    /** Specified the bitstream restriction info (as defined in Annex E of the
        ITU-T Specification). */
    bool bitstreamRestrictionFlag;
} NvMediaEncodeConfigH265VUIParams;

/**
 * Defines H265 encoder configuration features.
 * \hideinitializer
 */
typedef enum {
    /** Enable to write access unit delimiter syntax in bitstream */
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_OUTPUT_AUD               = (1 << 0),
    /** Enable gradual decoder refresh or intra refresh.
       If the GOP structure uses B frames this will be ignored */
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_INTRA_REFRESH            = (1 << 1),
    /** Enable dynamic slice mode. Client must specify max slice
       size using the maxSliceSizeInBytes field. */
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_DYNAMIC_SLICE_MODE       = (1 << 2),
    /** Enable constrainedFrame encoding where each slice in the
       constrained picture is independent of other slices. */
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_CONSTRANED_ENCODING      = (1 << 3),
    /** Enable lossless compression. */
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_LOSSLESS_COMPRESSION     = (1 << 4),
    /** Enable slice level output encoding. This enables delivery
      encoded data slice by slice to client to reduce encode latency */
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_SLICE_LEVEL_OUTPUT       = (1 << 5),
    /** Enable RTP mode output. NAL unit start code will be replaced
       with NAL size for the NAL units. */
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_RTP_MODE_OUTPUT          = (1 << 6),
    /** Enable support for external picture RC hint.
        \if (SWDOCS_DRIVE_QNX_AGX_SDK || SWDOCS_DRIVE_LNX_AGX_SDK)
        \note This feature is not supported.
        \endif
        \internal This feature is only supported internally via the nvmmlite
        layer.
      */
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_EXT_PIC_RC_HINT          = (1 << 7),
    /** Enable support for dynamic reference picture set RPS/DPB
       control support. After receiving an input buffer and result
       of previous frame encoding, based on real time information,
       Some client software determines properties for the new input
       buffer (long term/short term/non-referenced,
       frame number/poc/LT index), */
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_DYNAMIC_RPS              = (1 << 8),
    /** Enable support for motion vector buffer dump. This will enable
       motion vector dump. Motion vector buffer will be appended at the
       end of encoded bitstream data */
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_MV_BUFFER_DUMP           = (1 << 9),
    /** Enable encoder profiling. Profiling information would be added
       as part of output extradata */
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_PROFILING                = (1 << 10),
    /** Enable ultra fast encoding. It overrides some of the quality settings
        to achieve ultra fast encoding. */
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_ULTRA_FAST_ENCODE        = (1 << 11),
    /** Enable support to use client provided Initial QP for all frame types */
    NVMEDIA_ENCODE_CONFIG_H265_INIT_QP                         = (1 << 12),
    /** Enable support to use client provided QP max for all frame types. */
    NVMEDIA_ENCODE_CONFIG_H265_QP_MAX                        = (1 << 13),
    /** Enable support to use 4 byte start code in all the slices in a picture */
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_FOUR_BYTE_START_CODE     = (1 << 14),
    /** Enable support for motion vector buffer dump in a simplified (V2)
        format. Motion vector buffer will be appended to the end of encoded
        bitstream data retrieved using \ref NvMediaIEPGetBits. Refer
        NvMediaEncodeOutputExtradata for more information regarding the
        format of the dumped MV output. Either
        NVMEDIA_ENCODE_CONFIG_H265_ENABLE_MV_BUFFER_DUMP_V2 or
        NVMEDIA_ENCODE_CONFIG_H265_ENABLE_MV_BUFFER_DUMP should be enabled at a
        given time. If both are enabled, V2 format will be used.

        It should be ensured that size of \ref NvMediaBitstreamBuffer passed to
        \ref NvMediaIEPGetBits has sufficient space to store the MV buffer
        dump. The \a numBytesAvailable returned by \ref NvMediaIEPBitsAvailable
        or \a numBytes returned by \ref NvMediaIEPGetBits does not take MV
        buffer dump size into account - represents only the size of the encoded
        bits that are available. While allocating bitstream buffers, the
        additional amount of space required in bitstream to accommodate MV
        buffer data needs to be added. This can be calculated as follows:
        \code
            mvBufferSize = sizeof(NvMediaEncodeMVBufferHeader) +
                ALIGN_256(numCTBs * sizeof(NvMediaEncodeMVData))
            where,
            NumCTBs =
                (ALIGN_32(InputWidth)/32) * (ALIGN_32(InputHeight)/32)

            bitstreamSize = ALIGN_8(bitsAvailable) + mvBufferSize
        \endcode
        ALIGN_N refers to an operation which returns a multiple of N, greater
        than or equal to a number, closest to the number.

        \note This setting is not supported in the QNX Safety build. */
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_MV_BUFFER_DUMP_V2         = (1 << 15),
} NvMediaEncodeH265Features;

/**
 * Holds the H265 encoder configuration parameters.
 */
typedef struct {
    /** Holds bit-wise OR`ed configuration feature flags.
      *  See the \ref NvMediaEncodeH265Features enum. */
    uint32_t features;
    /** Holds the number of pictures in one GOP. Low latency application
        client can set the \a goplength field to
        \ref NVMEDIA_ENCODE_INFINITE_GOPLENGTH so that keyframes are not
        inserted automatically. */
    uint32_t gopLength;
    /** Holds the rate control parameters for the current encoding session. */
    NvMediaEncodeRCParams rcParams;
    /** Holds the frequency of the writing of Sequence and Picture parameters. */
    NvMediaEncodeH264SPSPPSRepeatMode repeatSPSPPS;
    /** Holds the IDR interval. If not set, this is made equal to
        NvMediaEncodeConfigH265::gopLength. Low latency application client can
        set IDR interval to NVMEDIA_ENCODE_INFINITE_GOPLENGTH so that IDR frames
        are not inserted automatically. */
    uint32_t idrPeriod;
    /** Holds a number that is 1 less than the desired number of slices
       per frame. \ref NVMEDIA_ENCODE_CONFIG_H265_ENABLE_DYNAMIC_SLICE_MODE
       should NOT be set in \ref features for this setting to take effect.
       \valuerange 0 to 31 in increments of 1. */
    uint16_t numSliceCountMinus1;
    /** Holds disable the deblocking filter.
        \note This feature is not supported. */
    uint8_t disableDeblockingFilter;
    /** Holds enable weighted prediction.
        \note This feature is not supported. */
    uint8_t enableWeightedPrediction;
    /** Holds the interval between frames that trigger a new intra refresh cycle
       and this cycle lasts for intraRefreshCnt frames.
       This value is used only if the
       \ref NVMEDIA_ENCODE_CONFIG_H265_ENABLE_INTRA_REFRESH is set in features.
       The \ref NVMEDIA_ENCODE_PIC_TYPE_P_INTRA_REFRESH picture type also
       triggers a new intra-refresh cycle and resets the current intra-refresh
       period. Setting it to zero results in that no automatic refresh cycles
       are triggered. In this case only NVMEDIA_ENCODE_PIC_TYPE_P_INTRA_REFRESH
       picture type can trigger a new refresh cycle. */
    uint32_t intraRefreshPeriod;
    /** Holds the number of frames over which intra refresh will happen.
       This value must be less than or equal to intraRefreshPeriod. Setting it
       to zero turns off the intra refresh functionality. Setting it to one
       essentially means that after every intraRefreshPeriod frames the encoded
       P frame contains only intra predicted macroblocks.
       This value is used only if the
       NVMEDIA_ENCODE_CONFIG_H265_ENABLE_INTRA_REFRESH is set in features. */
    uint32_t intraRefreshCnt;
    /** Holds the maximum slice size in bytes for dynamic slice mode. The
       Client must set \ref NVMEDIA_ENCODE_CONFIG_H265_ENABLE_DYNAMIC_SLICE_MODE
       in features in order for this to take effect. */
    uint32_t maxSliceSizeInBytes;
    /** Number of CTU per slice. Set to 0 if fix number of macroblocks not
        required or  maxSliceSizeInBytes or numSliceCountMinus1 is set to
        non-zero value. */
    uint32_t numCTUsPerSlice;
    /** Holds the H265 video usability info pamameters. Set to NULL if VUI is
        not needed */
    NvMediaEncodeConfigH265VUIParams *h265VUIParameters;
    /** Holds Initial QP parameters. Client must set
        NVMEDIA_ENCODE_CONFIG_H265_INIT_QP in features to use this.
        \valuerange QP values can lie between 1 to 51 in increments of 1. */
    NvMediaEncodeQP initQP;
    /** Holds maximum QP parameters. Client must set
     *  NVMEDIA_ENCODE_CONFIG_H265_QP_MAX in features to use this.
        \valuerange QP values can lie between 1 to 51 in increments of 1, must
        be greater than \ref NvMediaEncodeRCParams::minQP. */
    NvMediaEncodeQP maxQP;
    /** Holds the encode quality pre-set. See enum NvMediaEncPreset.
        Recommended pre-setting is \ref NVMEDIA_ENC_PRESET_HP. */
    NvMediaEncPreset encPreset;
} NvMediaEncodeConfigH265;

/**
 * Holds H265-specific encode initialization parameters.
 */
typedef struct {
    /** Holds the encode width.
        \if (SWDOCS_DRIVE_QNX_AGX_SDK || SWDOCS_DRIVE_LNX_AGX_SDK)
        \valuerange 160 to 4096 in increments of 2.
        \endif
      */
    uint16_t encodeWidth;
    uint16_t reserved1;
    /** Holds the encode height.
        \if (SWDOCS_DRIVE_QNX_AGX_SDK || SWDOCS_DRIVE_LNX_AGX_SDK)
        \valuerange 64 to 4096 in increments of 2.
        \endif
      */
    uint16_t encodeHeight;
    uint16_t reserved2;
    /** Set this to true for limited-RGB (16-235) input.
        \note This feature is not supported. */
    bool enableLimitedRGB;
    /** Set this to true for slice level encode. */
    bool enableSliceLevelEncode;
    /** Holds the numerator for frame rate used for encoding in frames per second
        ( Frame rate = frameRateNum / frameRateDen ).
        \valuerange Frame rate can lie between 1 and 60 (float value). */
    uint32_t frameRateNum;
    /** Holds the denominator for frame rate used for encoding in frames per second
        ( Frame rate = frameRateNum / frameRateDen ).
        \valuerange Frame rate can lie between 1 and 60 (float value). */
    uint32_t frameRateDen;
    /** Holds the encoding profile to be set based on NvMediaEncodeH265Profile.
        Client is recommended to set this to \ref NVMEDIA_ENCODE_H265_PROFILE_AUTOSELECT
        in order to enable the Encode interface to select the correct profile.
        If a specific profile needs to be configured, this field can be set to one of
        the supported enumerations from \ref NvMediaEncodeH265Profile. */
    uint8_t profile;
    /** Holds the encoding level. Client is recommended to set this to
        NVMEDIA_ENCODE_LEVEL_H265_AUTOSELECT in order to enable the Encode
        interface to select the correct level. If a specific level needs to be
        configured, this field can be set to one of the enumerations from
        \ref NvMediaEncodeLevelH265. If the specified level is lower than that
        which is required by the H265 standard, the encoder auto-corrects it to
        the minimum required level. */
    uint8_t level;
    /** Holds the level tier information. This is set to 0 for main tier
        and 1 for high tier. This is valid only when level is not selected
        as NVMEDIA_ENCODE_LEVEL_H265_AUTOSELECT. */
    uint8_t levelTier;
    /** Holds the maximum number of reference frames used for encoding.
        \valuerange 0
        \n For I-frame only mode (no reference frames), i.e.,
            \ref NvMediaEncodeInitializeParamsH265.enableAllIFrames is set to 1
            and \ref NvMediaEncodeRCParams.numBFrames is set to 0.
        \valuerange 1 to 8
        \n For encoding with I and P frame types (p-frames as reference), i.e.,
            \ref NvMediaEncodeInitializeParamsH265.enableAllIFrames is set to 0
            and \ref NvMediaEncodeRCParams.numBFrames is set to 0.
        \valuerange 2 to 8
        \n For encoding with I, P and B frame types (p and b-frames as
            reference), i.e.,
            \ref NvMediaEncodeInitializeParamsH265.enableAllIFrames is set to 0,
            \ref NvMediaEncodeRCParams.numBFrames is set to a non-zero value and
            \ref NvMediaEncodeInitializeParamsH265.useBFramesAsRef is set to 1.
        \valuerange 2 to 8
        \n For encoding with I, P and B frame types (only p-frames as
            reference), i.e.,
            \ref NvMediaEncodeInitializeParamsH265.enableAllIFrames is set to 0,
            \ref NvMediaEncodeRCParams.numBFrames is set to a non-zero value and
            \ref NvMediaEncodeInitializeParamsH265.useBFramesAsRef is set to 0.
        */
    uint8_t maxNumRefFrames;
    /** Set to true to enable external ME hints.
        Currently this feature is not supported if B frames are used.
        \deprecated This feature will be deprecated in a future release. */
    bool enableExternalMEHints;
    /** If Client wants to pass external motion vectors in
        \ref NvMediaEncodePicParamsH265 meExternalHints buffer it must specify
        the maximum number of hint candidates per block per direction for the
        encode session. The \ref NvMediaEncodeInitializeParamsH265
        maxMEHintCountsPerBlock[0] is for L0 predictors and
        \ref NvMediaEncodeInitializeParamsH265 maxMEHintCountsPerBlock[1] is
        for L1 predictors. This client must also set
        \ref NvMediaEncodeInitializeParamsH265 enableExternalMEHints to
        true.
        \deprecated This feature will be deprecated in a future release. */
    NvMediaEncodeExternalMeHintCountsPerBlocktype maxMEHintCountsPerBlock[2];
    /** Enable support for recon CRC generation. NVMEDIA will
        allocate extra surface for recon CRC calculation. This can
        be enabled at run time for any frame by enabling recon CRC and
        passing recon CRC rectangle.
        \if (SWDOCS_DRIVE_QNX_AGX_SDK || SWDOCS_DRIVE_LNX_AGX_SDK)
        \note This feature is not supported.
        \endif
        \internal This feature is only supported internally via the nvmmlite
        layer.
      */
    bool enableReconCRC;
    /** If client want to do MVC encoding then this flag need to be set.
        \note This feature is not supported. */
    bool enableMVC;
    /** Enable region of interest encoding. Region of interest encoding
        parameters are passed with Input extra data parameters
        \note Setting input extra data is an internal feature. As a result,
        enableROIEncode is only supported internally. */
    bool enableROIEncode;
    /** use slice encode to reduce latency in getting encoded buffers. */
    bool enableSliceEncode;
    /** Enables B frames to be used as reference frames.
        \valuerange 0 to 1
        - 0: Disables the use of b-frames as reference frames.
        - 1: Enables the use of b-frames as reference frames.
        \note For this field to take effect
        \ref NvMediaEncodeRCParams.numBFrames needs to be set to a non-zero
        value and \ref NvMediaEncodeInitializeParamsH265.maxNumRefFrames should
        be set to an appropriate value. */
    uint32_t useBFramesAsRef;
    /** Enable 2 pass RC support. First pass RC can be run on full or quarter
        resolution.
        \note \a enableTwoPassRC is an internal feature. */
    bool enableTwoPassRC;
    /** Enable 2 pass RC with quarter resolution first pass.
        \if (SWDOCS_DRIVE_QNX_AGX_SDK || SWDOCS_DRIVE_LNX_AGX_SDK)
        \note This feature is not supported.
        \endif
        \internal This feature is only supported internally via the nvmmlite
        layer.
      */
    bool enableSourceHalfScaled;
    /** Number of views used for MV-Hevc.
        \note This feature is not supported. */
    uint32_t mvNumViews              : 4;
    /** Enable external picture rate control
        \note This feature is not supported. */
    uint32_t enableExternalPictureRC : 1;
    /** Encode all frames as I frames.
        \valuerange 0 to disable, non-zero value to enable */
    uint32_t enableAllIFrames : 1;
    /** Add padding */
    uint32_t reserved                : 26;
    /** Use ampDisable to enable or disable assymetric partition types.
     *  - 0: enable AMP block paritions
     *  - 1: disble AMP block paritions */
    bool ampDisable;
} NvMediaEncodeInitializeParamsH265;

/**
 *  Holds an H265-specific User SEI message.
 */
typedef struct {
    /** SEI payload size in bytes. SEI payload must be byte aligned, as
        described in Annex D */
    uint32_t payloadSize;
    /** SEI payload types and syntax can be found in Annex D of the H265
        Specification. */
    uint32_t payloadType;
    /** SEI nal_unit_type */
    uint32_t nalUnitType;
    /** pointer to user data */
    uint8_t *payload;
} NvMediaEncodeH265SEIPayload;

/**
 * Holds H265-specific encoder picture parameters. Sent on a per frame basis.
 */
typedef struct {
    /** Holds input picture type. */
    NvMediaEncodePicType pictureType;
    /** Holds bit-wise OR`ed encode pic flags.
        See enum \ref NvMediaEncodePicFlags. */
    uint32_t encodePicFlags;
    /** Specifies the number of B-frames that follow the current frame.
        This number can be set only for reference frames and the frames
        that follow the current frame must be nextBFrames count of B-frames.
        B-frames are supported only if the profile is greater than
        NVMEDIA_ENCODE_PROFILE_BASELINE and the maxNumRefFrames is set to 2.
        Set to zero if no B-frames are needed. */
    uint32_t nextBFrames;
    /** Holds the rate control parameters from the current frame onward
        if the NVMEDIA_ENCODE_PIC_FLAG_RATECONTROL_CHANGE is set in the
        encodePicFlags.
        \note The \ref NvMediaEncodeRCParams.rateControlMode cannot be changed
        on a per-frame basis. Only the associated rate control parameters can be
        changed. */
    NvMediaEncodeRCParams rcParams;
    /** Holds the number of elements allocated in \ref seiPayloadArray array.
        Set to 0 if no SEI messages are needed */
    uint32_t seiPayloadArrayCnt;
    /** Array of SEI payloads which will be inserted for this frame. */
    NvMediaEncodeH265SEIPayload *seiPayloadArray;
    /** Holds the number of hint candidates per block per direction for the
        current frame. meHintCountsPerBlock[0] is for L0 predictors and
        meHintCountsPerBlock[1] is for L1 predictors. The candidate count in
        \ref NvMediaEncodePicParamsH265 meHintCountsPerBlock[lx] must never
        exceed \ref NvMediaEncodeInitializeParamsH265 maxMEHintCountsPerBlock[lx]
        provided during encoder initialization.
        \deprecated This feature will be deprecated in a future release. */
    NvMediaEncodeExternalMeHintCountsPerBlocktype meHintCountsPerBlock[2];
    /** Holds the pointer to ME external hints for the current frame.
        The size of ME hint buffer must be equal to number of macroblocks
        multiplied by the total number of candidates per macroblock.
        The total number of candidates per MB per direction =
        <pre>
          1*meHintCountsPerBlock[Lx].numCandsPerBlk16x16 +
          2*meHintCountsPerBlock[Lx].numCandsPerBlk16x8 +
          2*meHintCountsPerBlock[Lx].numCandsPerBlk8x16 +
          4*meHintCountsPerBlock[Lx].numCandsPerBlk8x8
        </pre>
        For frames using bidirectional ME, the total number of candidates for
        a single macroblock is the sum of the total number of candidates per MB
        for each direction (L0 and L1).

        If no external ME hints are needed, set this field to NULL.
        \deprecated This feature will be deprecated in a future release. */
    NvMediaEncodeExternalMEHint *meExternalHints;
    /** Holds the numerator for frame rate used for encoding in frames per second
        ( Frame rate = frameRateNum / frameRateDen ).
        \valuerange Frame rate can lie between 1 and 60 (float value). */
    uint32_t frameRateNum;
    /** Holds the numerator for frame rate used for encoding in frames per second
        ( Frame rate = frameRateNum / frameRateDen ).
        \valuerange Frame rate can lie between 1 and 60 (float value). */
    uint32_t frameRateDen;
    /** Holds the viewId of current picture */
    uint32_t viewId;
} NvMediaEncodePicParamsH265;

/**@} <!-- Ends h265_encoder_api sub-group --> */

/**
 * @defgroup 6x_vp9_encoder_api Data types for VP9 encoding
 * Types and declarations for VP9 Encoding.
 * @ingroup 6x_image_encode_api
 * @{
 */

/**
 * Defines VP9 encoder configuration features.
 */
typedef enum {
    /** Enable to set loop filter parameters */
    NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_LOOP_FILTER_PARAMS               = (1 << 0),
    /** Enable to set quantization parameters */
    NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_QUANTIZATION_PARAMS              = (1 << 1),
    /** Enable transform mode */
    NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_TRANSFORM_MODE                   = (1 << 2),
    /** Enable high precision mv */
    NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_HIGH_PRECISION_MV                = (1 << 3),
    /** Disable error resiliency. Error resiliency is enabled by default */
    NVMEDIA_ENCODE_CONFIG_VP9_DISABLE_ERROR_RESILIENT                 = (1 << 4),
    /** Enable encoder profiling. Profiling information would be added
       as part of output extradata */
    NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_PROFILING                        = (1 << 5),
    /** Enable support to use client provided Initial QP for all frame types */
    NVMEDIA_ENCODE_CONFIG_VP9_INIT_QP                                 = (1 << 6),
    /** Enable support to use client provided QP max for all frame types */
    NVMEDIA_ENCODE_CONFIG_VP9_QP_MAX                                = (1 << 7)
} NvMediaEncodeVP9Features;

/**
 * Holds VP9 encoder configuration parameters.
 */
typedef struct {
    /** Holds bit-wise OR`ed configuration feature flags.
        See the \ref NvMediaEncodeVP9Features enum. */
    uint32_t features;
    /** Holds the number of pictures in one GOP. Low-latency application client
        can set \a goplength to NVMEDIA_ENCODE_INFINITE_GOPLENGTH so that
        keyframes are not inserted automatically. */
    uint32_t gopLength;
    /** Holds the rate control parameters for the current encoding session. */
    NvMediaEncodeRCParams rcParams;
    /** Holds the IDR interval. If not set, this is made equal to \a gopLength
        in NvMediaEncodeConfigVP9. Low-latency application client can set IDR
        interval to NVMEDIA_ENCODE_INFINITE_GOPLENGTH so that IDR frames are not
        inserted automatically. */
    uint32_t idrPeriod;

    /** Set the feature flag \ref NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_LOOP_FILTER_PARAMS
       to change the following parameters. */
    /** Specifies the type of the filter used.
        \valuerange One of the following filter types:
        - 0: Eight-tap smooth
        - 1: Eight-tap
        - 2: Eight-tap sharp
        - 3: Bilinear
        - 4: Switchable */
    uint32_t filter_type;
    /** Specifies the loop filter strength for each segment. */
    uint32_t filter_level;
    /** Specifies Sharpness level. */
    uint32_t sharpness_level;
    /** Specifies the Loop filter strength adjustments based on frame type
        (intra, inter). */
    int8_t ref_lf_deltas[4];
    /** Specifies the Loop filter strength adjustments based on mode
        (zero, new mv). */
    int8_t mode_lf_deltas[2];
    /** Set it to true if MB-level loop filter adjustment is on. */
    bool bmode_ref_lf_delta_enabled;
    /** Set it to true if MB-level loop filter adjustment delta values are
        updated. */
    bool bmode_ref_lf_delta_update;

    /** Set the feature flag
        \ref NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_QUANTIZATION_PARAMS to set the
        following parameters. */
    /** Specifies quant base index (used only when rc_mode = 0) for each segment
        0...255. This is for setting the feature flag
        \ref NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_QUANTIZATION_PARAMS. */
    uint32_t base_qindex;
    /** Specifies explicit qindex adjustment for y dccoefficient, -15...15.
        This is for setting the feature flag
        \ref NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_QUANTIZATION_PARAMS. */
    int32_t delta_y_dc_q;
    /** Specifies qindex adjustment for uv accoefficient, -15...15.
        This is for setting the feature flag
        \ref NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_QUANTIZATION_PARAMS. */
    int32_t delta_uv_dc;
    /** Specifies qindex adjustment for uv dccoefficient, -15...15.
        This is for setting the feature flag
        \ref NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_QUANTIZATION_PARAMS. */
    int32_t delta_uv_ac;

    /** Set the feature flag \ref NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_TRANSFORM_MODE
        to set the following parameter. */
    /** Specifies transform_mode.
        \valuerange One of the following modes:
        - 0: only4x4
        - 1: allow_8x8
        - 2: allow_16x16
        - 3: allow_32x32
        - 4:transform_mode_select. */
    uint32_t transform_mode;

    /** Set the feature flag \ref NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_HIGH_PRECISION_MV
       to set the following parameter. */
    /** Specifies to enable high precision MV.
        This is for setting the feature flag
        \ref NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_HIGH_PRECISION_MV. */
    uint32_t high_prec_mv;

    /** Set the feature flag \ref
        NVMEDIA_ENCODE_CONFIG_VP9_DISABLE_ERROR_RESILIENT to set the following
        parameter. */
    /** Enable Error resiliency.
        This is for setting the feature flag
        \ref NVMEDIA_ENCODE_CONFIG_VP9_DISABLE_ERROR_RESILIENT. */
    bool error_resilient;

    /** Holds Initial QP parameters. Client must set
        NVMEDIA_ENCODE_CONFIG_VP9_INIT_QP in features to use this. QP values
        should be within the range of 1 to 255 */
    NvMediaEncodeQP initQP;
    /** Holds maximum QP parameters. Client must set
        NVMEDIA_ENCODE_CONFIG_VP9_QP_MAX in features to use this. The maximum
        QP values must be within the range of 1 to 255 and must be set to a
        value greater than \ref NvMediaEncodeRCParams::minQP. */
    NvMediaEncodeQP maxQP;
} NvMediaEncodeConfigVP9;

/**
 * Holds VP9-specific encode initialization parameters.
 */
typedef struct {
    /** Holds the encode width.*/
    uint32_t encodeWidth;
    /** Holds the encode height.*/
    uint32_t encodeHeight;
    /** Holds a flag indicating whether input is limited-RGB (16-235).
        Set this to true for limited-RGB (16-235) input.
        \note This feature is not supported. */
    bool enableLimitedRGB;
    /** Holds the numerator for frame rate used for encoding in frames per second
        ( Frame rate = frameRateNum / frameRateDen ). */
    uint32_t frameRateNum;
    /** Holds the denominator for frame rate used for encoding in frames per second
        ( Frame rate = frameRateNum / frameRateDen ). */
    uint32_t frameRateDen;
    /** Holds the max reference numbers used for encoding. Allowed range is [0, 2].
        Values:
        - 0 allows only I frame encode
        - 1 allows I and IP encode
        - 2 allows I, IP and IBP encode */
    uint8_t maxNumRefFrames;
    /** Holds a flag indicating whether to enable or disable the external ME
        hints. Set to true to enable external ME hints.
        Currently this feature is not supported if B frames are used.
        \deprecated This feature will be deprecated in a future release. */
    bool enableExternalMEHints;
    /** If Client wants to pass external motion vectors in
        \ref NvMediaEncodePicParamsVP9 \a meExternalHints buffer it must
        specify the maximum number of hint candidates, per block and per
        direction, for the encode session. The \ref NvMediaEncodeInitializeParamsVP9
        \a maxMEHintCountsPerBlock[0] is for L0 predictors and
        \ref NvMediaEncodeInitializeParamsVP9 \a maxMEHintCountsPerBlock[1] is
        for L1 predictors. This client must also set
        \ref NvMediaEncodeInitializeParamsVP9 \a enableExternalMEHints to
        true.
        \deprecated This feature will be deprecated in a future release. */
    NvMediaEncodeExternalMeHintCountsPerBlocktype maxMEHintCountsPerBlock[2];
    /** Holds number of HW entropy cores for encoding. Supported cores are 1
        and 4 cores*/
    uint8_t numEpCores;
    /** Holds number of log2Rows used in a frame. Supported values are from 0
        to 4*/
    uint32_t log2TileRows;
    /** Holds number of log2Cols used in a frame. Supported values are from 0
        to 4*/
    uint32_t log2TileCols;
    /** Skip Chroma Processing. Supported values are 0 and 1. */
    uint32_t vp9SkipChroma;
} NvMediaEncodeInitializeParamsVP9;

/**
 * Holds VP9-specific encoder picture parameters, which are sent on a per
 * frame basis.
 */
typedef struct {
    /** Holds input picture type. */
    NvMediaEncodePicType pictureType;
    /** Holds bit-wise OR`ed encode pic flags.
        See enum \ref NvMediaEncodePicFlags. */
    uint32_t encodePicFlags;
    /** Holds the number of B-frames that follow the current frame.
        This number can be set only for reference frames and the frames
        that follow the current frame must be nextBFrames count of B-frames.
        B-frames are supported only if the profile is greater than
        NVMEDIA_ENCODE_PROFILE_BASELINE and the maxNumRefFrames is set to 2.
        Set to zero if no B-frames are needed. */
    uint32_t nextBFrames;
    /** Holds the rate control parameters from the current frame onward
        if the NVMEDIA_ENCODE_PIC_FLAG_RATECONTROL_CHANGE is set in the
        encodePicFlags. Please note that the rateControlMode cannot be changed
        on a per frame basis only the associated rate control parameters. */
    NvMediaEncodeRCParams rcParams;
    /** Holds the number of hint candidates per block per direction for the
        current frame. \a meHintCountsPerBlock[0] is for L0 predictors and
        \a meHintCountsPerBlock[1] is for L1 predictors. The candidate count in
        \ref NvMediaEncodePicParamsVP9 \a meHintCountsPerBlock[lx] must never
        exceed \ref NvMediaEncodeInitializeParamsVP9
        \a maxMEHintCountsPerBlock[lx] provided during encoder initialization.*/
    NvMediaEncodeExternalMeHintCountsPerBlocktype meHintCountsPerBlock[2];
    /** Holds the pointer to ME external hints for the current frame.
        The size of ME hint buffer must be equal to number of macroblocks
        multiplied by the total number of candidates per macroblock.
        The total number of candidates per MB per direction =
        <pre>
         1*meHintCountsPerBlock[Lx].numCandsPerBlk16x16 +
         2*meHintCountsPerBlock[Lx].numCandsPerBlk16x8 +
         2*meHintCountsPerBlock[Lx].numCandsPerBlk8x16 +
         4*meHintCountsPerBlock[Lx].numCandsPerBlk8x8
        </pre>
        For frames using bidirectional ME , the total number of candidates for
        a single macroblock is the sum of the total number of candidates per MB
        for each direction (L0 and L1).

        If no external ME hints are needed, set this field to NULL. */
    NvMediaEncodeExternalMEHint *meExternalHints;
} NvMediaEncodePicParamsVP9;

/**@} <!-- Ends vp9_encoder_api sub-group --> */

/**
 * @defgroup 6x_av1_encoder_api Data types for AV1 encoding
 * Types and declarations for AV1 Encoding.
 * @ingroup 6x_image_encode_api
 * @{
 */

/**
 * Defines AV1 encoder configuration features.
 */
typedef enum {
    /** Enable to set quantization parameters */
    NVMEDIA_ENCODE_CONFIG_AV1_ENABLE_QUANTIZATION_PARAMS              = (1 << 1),
    /** Disable CDF update during symbol decoding */
    NVMEDIA_ENCODE_CONFIG_AV1_DISABLE_CDF_UPDATE                      = (1 << 2),
    /** enable/disable CDF update at end of frame */
    NVMEDIA_ENCODE_CONFIG_AV1_FRAME_END_CDF_UPDATE                    = (1 << 3),
    /** Enable encoder profiling. Profiling information would be added
       as part of output extradata */
    NVMEDIA_ENCODE_CONFIG_AV1_ENABLE_PROFILING                        = (1 << 5),
    /** Enable support to use client provided Initial QP for all frame types */
    NVMEDIA_ENCODE_CONFIG_AV1_INIT_QP                                 = (1 << 6),
    /** Enable support to use client provided QP max for all frame types */
    NVMEDIA_ENCODE_CONFIG_AV1_QP_MAX                                  = (1 << 7)
} NvMediaEncodeAV1Features;

/**
 * Specifies the frequency of the writing of Sequence header
 * for AV1.
 * \hideinitializer
 */
typedef enum {
    /** Repeating of SPS is disabled */
    NVMEDIA_ENCODE_SEQUENCEHDR_REPEAT_DISABLED          = 0,
    /** SPS is repeated for every intra frame */
    NVMEDIA_ENCODE_SEQUENCEHDR_REPEAT_INTRA_FRAMES      = 1,
    /** SPS is repeated for every IDR frame */
    NVMEDIA_ENCODE_SEQUENCEHDR_REPEAT_IDR_FRAMES        = 2
} NvMediaEncodeAV1SeqHdrRepeatMode;

/**
 * Holds AV1 encoder configuration parameters.
 */
typedef struct {
    /** Holds bit-wise OR`ed configuration feature flags. TBD */
    uint32_t features;
    /** Holds the number of pictures in one GOP. Low-latency application
        client can set \a goplength to NVMEDIA_ENCODE_INFINITE_GOPLENGTH so
        that keyframes are not inserted automatically. */
    uint32_t gopLength;
    /** Holds the rate control parameters for the current encoding session. */
    NvMediaEncodeRCParams rcParams;
    /** Holds the IDR interval. If not set, this is made equal to \a
        gopLength in NvMediaEncodeConfigVP9. Low-latency application client
        can set IDR interval to NVMEDIA_ENCODE_INFINITE_GOPLENGTH so that IDR
        frames are not inserted automatically. */
    uint32_t idrPeriod;

    /* Set the feature flag \ref
       NVMEDIA_ENCODE_CONFIG_AV1_ENABLE_QUANTIZATION_PARAMS to set the
       following parameters */
    /** Specifies quant base index (used only when rc_mode = 0) for each segment
        0...255. This is for setting the feature flag
        \ref NVMEDIA_ENCODE_CONFIG_AV1_ENABLE_QUANTIZATION_PARAMS. */
    uint32_t base_qindex;
    /** Specifies explicit qindex adjustment for y dccoefficient, -15...15.
        This is for setting the feature flag
        \ref NVMEDIA_ENCODE_CONFIG_AV1_ENABLE_QUANTIZATION_PARAMS. */
    int32_t delta_y_dc_q;
    /** Specifies qindex adjustment for uv accoefficient, -15...15.
        This is for setting the feature flag
        \ref NVMEDIA_ENCODE_CONFIG_AV1_ENABLE_QUANTIZATION_PARAMS. */
    int32_t delta_uv_dc;
    /** Specifies qindex adjustment for uv dccoefficient, -15...15.
        This is for setting the feature flag
        \ref NVMEDIA_ENCODE_CONFIG_AV1_ENABLE_QUANTIZATION_PARAMS. */
    int32_t delta_uv_ac;

    /** Holds Initial QP parameters. Client must set
        NVMEDIA_ENCODE_CONFIG_VP9_INIT_QP in features to use this. QP values
        should be within the range of 1 to 255 */
    NvMediaEncodeQP initQP;
    /** Holds maximum QP parameters. Client must set
        NVMEDIA_ENCODE_CONFIG_VP9_QP_MAX in features to use this. The maximum
        QP values must be within the range of 1 to 255 and must be set to a
        value greater than \ref NvMediaEncodeRCParams::minQP. */
    NvMediaEncodeQP maxQP;
    /** Set to true to disable CDF update */
    NvMediaBool disableCdfUpdate;
    /** Holds the encode quality pre-set. See enum NvMediaEncPreset.
        Recommended pre-setting is \ref NVMEDIA_ENC_PRESET_HP. */
    NvMediaEncPreset encPreset;

    /** Holds the frequency of the writing of Sequence header. */
    NvMediaEncodeAV1SeqHdrRepeatMode repeatSeqHdr;
    /** Reserved Bytes */
    uint32_t reserved[17];
} NvMediaEncodeConfigAV1;

/**
 * Holds AV1-specific encode initialization parameters.
 */
typedef struct {
    /** Holds the encode width.*/
    uint32_t encodeWidth;
    /** Holds the encode height.*/
    uint32_t encodeHeight;
    /** Holds a flag indicating whether input is limited-RGB (16-235).
        Set this to true for limited-RGB (16-235) input. */
    bool enableLimitedRGB;
    /** Holds the numerator for frame rate used for encoding in frames per second
        ( Frame rate = frameRateNum / frameRateDen ). */
    uint32_t frameRateNum;
    /** Holds the denominator for frame rate used for encoding in frames per second
        ( Frame rate = frameRateNum / frameRateDen ). */
    uint32_t frameRateDen;
    /** Holds the encoding profile. Client is recommended to set this to
        \ref NVMEDIA_ENCODE_PROFILE_AUTOSELECT in order to enable the Encode
        interface to select the correct profile. If a specific profile needs to
        be configured, this field can be set to one of the supported
        enumerations from \ref NvMediaEncodeProfile. */
    uint8_t profile;
    /** Holds the encoding level. Client is recommended to set this to
        NVMEDIA_ENCODE_LEVEL_AUTOSELECT in order to enable the Encode interface
        to select the correct level. If a specific level needs to be configured,
        this field can be set to one of the enumerations from
        \ref NvMediaEncodeLevel. If the specified level is lower than that
        which is required by the H264 standard, the encoder auto-corrects it to
        the minimum required level. */
    uint8_t level;
    /** Holds the max reference numbers used for encoding. Allowed range is [0, 2].
        Values:
        - 0 allows only I frame encode
        - 1 allows I and IP encode
        - 2 allows I, IP and IBP encode */
    uint8_t maxNumRefFrames;
    /** Set to true to enable SSIM RDO*/
    bool enableSsimRdo;
    /** Set to true to enable Multiple tile mode*/
    bool enableTileEncode;
    /** Holds the log2 value of number of tiles used in a row*/
    uint8_t log2NumTilesInRow;
    /** Holds the log2 value of number of tiles used in a column*/
    uint8_t log2NumTilesInCol;
    /* Holds the frame restoration filter type*/
    uint8_t frameRestorationType;
    /* Set to true to enable bi-compound for B-frames (Currently not
       supported)*/
    bool enableBiCompound;
    /* Set to true to enable uni-compound for P/B-frames (only
       support in Pframes)*/
    bool enableUniCompound;
    /* Set to true to enable Internal High Bit depth*/
    bool enableInternalHighBitDepth;
    /** Holds a flag indicating whether to enable or disable the external ME
        hints. Set to true to enable external ME hints.
        Currently this feature is not supported if B frames are used. */
    bool enableExternalMEHints;
    /** If Client wants to pass external motion vectors in \ref
        NvMediaEncodePicParamsVP9 \a meExternalHints buffer it must specify
        the maximum number of hint candidates, per block and per direction,
        for the encode session. The \ref NvMediaEncodeInitializeParamsVP9
        \a maxMEHintCountsPerBlock[0] is for L0 predictors and
        \ref NvMediaEncodeInitializeParamsVP9 \a maxMEHintCountsPerBlock[1]
        is for L1 predictors. This client must also set
        \ref NvMediaEncodeInitializeParamsVP9 \a enableExternalMEHints to
        true. */
    NvMediaEncodeExternalMeHintCountsPerBlocktype maxMEHintCountsPerBlock[2];
    /** Reserved Bytes */
    uint32_t reserved[20];
} NvMediaEncodeInitializeParamsAV1;

/**
 * Holds AV1-specific encoder picture parameters, which are sent on a per
 * frame basis.
 */
typedef struct {
    /** Holds input picture type. */
    NvMediaEncodePicType pictureType;
    /** Holds bit-wise OR`ed encode pic flags.
        See enum \ref NvMediaEncodePicFlags. */
    uint32_t encodePicFlags;
    /** Holds the number of B-frames that follow the current frame.
        This number can be set only for reference frames and the frames
        that follow the current frame must be nextBFrames count of B-frames.
        B-frames are supported only if the profile is greater than
        NVMEDIA_ENCODE_PROFILE_BASELINE and the maxNumRefFrames is set to 2.
        Set to zero if no B-frames are needed. */
    uint32_t nextBFrames;
    /** Holds the rate control parameters from the current frame onward
        if the NVMEDIA_ENCODE_PIC_FLAG_RATECONTROL_CHANGE is set in the
        encodePicFlags. Please note that the rateControlMode cannot be
        changed on a per frame basis only the associated rate control parameters. */
    NvMediaEncodeRCParams rcParams;
    /** Holds the number of hint candidates per block per direction for the
        current frame. \a meHintCountsPerBlock[0] is for L0 predictors and
        \a meHintCountsPerBlock[1] is for L1 predictors. The candidate count
        in \ref NvMediaEncodePicParamsVP9 \a meHintCountsPerBlock[lx] must
        never exceed \ref NvMediaEncodeInitializeParamsVP9
        \a maxMEHintCountsPerBlock[lx] provided during encoder initialization. */
    NvMediaEncodeExternalMeHintCountsPerBlocktype meHintCountsPerBlock[2];
    /** Holds the pointer to ME external hints for the current frame.
        The size of ME hint buffer must be equal to number of macroblocks
        multiplied by the total number of candidates per macroblock.
        The total number of candidates per MB per direction =
        <pre>
         1*meHintCountsPerBlock[Lx].numCandsPerBlk16x16 +
         2*meHintCountsPerBlock[Lx].numCandsPerBlk16x8 +
         2*meHintCountsPerBlock[Lx].numCandsPerBlk8x16 +
         4*meHintCountsPerBlock[Lx].numCandsPerBlk8x8
        </pre>
        For frames using bidirectional ME , the total number of candidates for
        a single macroblock is the sum of the total number of candidates per MB
        for each direction (L0 and L1).

        If no external ME hints are needed, set this field to NULL. */
    NvMediaEncodeExternalMEHint *meExternalHints;
    /** Holds the numerator for frame rate used for encoding in frames per second
        ( Frame rate = frameRateNum / frameRateDen ).
        \valuerange Frame rate can lie between 1 and 60 (float value). */
    uint32_t frameRateNum;
    /** Holds the numerator for frame rate used for encoding in frames per second
        ( Frame rate = frameRateNum / frameRateDen ).
        \valuerange Frame rate can lie between 1 and 60 (float value). */
    uint32_t frameRateDen;
    /** Reserved Bytes */
    uint32_t reserved[18];
} NvMediaEncodePicParamsAV1;

/*@} <!-- Ends av1_encoder_api sub-group --> */

/** @} <!-- Ends common_types_encode_top group NvMedia Types and Structures
    common for encode operations --> */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_COMMON_ENCODE_H */
