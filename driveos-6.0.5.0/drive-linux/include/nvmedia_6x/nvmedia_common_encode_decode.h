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
 * @brief <b> NVIDIA Media Interface: Common Types for Encode and Decode</b>
 *
 * @b Description: This file contains common types and definitions for
 * decode and encode operations.
 */

#ifndef NVMEDIA_COMMON_ENCODE_DECODE_H
#define NVMEDIA_COMMON_ENCODE_DECODE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 * @defgroup 6x_common_types_top Encode/Decode: Common Types
 * @ingroup 6x_nvmedia_common_top
 *
 * @brief Defines common types and declarations for decode and encode operations.
 * @{
 */

/**
 * @brief Specifies NVJPG HW instance ID
 */
typedef enum  {
    /** @brief Specifies NVJPG HW instance ID 0 */
    NVMEDIA_JPEG_INSTANCE_0 = 0,
    /** @brief Specifies NVJPG HW instance ID 1 */
    NVMEDIA_JPEG_INSTANCE_1,
    /** @brief Specifies NVJPG HW instance ID AUTO */
    NVMEDIA_JPEG_INSTANCE_AUTO
} NvMediaJPEGInstanceId;

/**
 * @brief Video codec type
 */
typedef enum {
    /** @brief H.264 codec */
    NVMEDIA_VIDEO_CODEC_H264,
    /** @brief VC-1 simple and main profile codec */
    NVMEDIA_VIDEO_CODEC_VC1,
    /** @brief VC-1 advanced profile codec */
    NVMEDIA_VIDEO_CODEC_VC1_ADVANCED,
    /** @brief MPEG1 codec */
    NVMEDIA_VIDEO_CODEC_MPEG1,
    /** @brief MPEG2 codec */
    NVMEDIA_VIDEO_CODEC_MPEG2,
    /** @brief MPEG4 Part 2 codec */
    NVMEDIA_VIDEO_CODEC_MPEG4,
    /** @brief MJPEG codec */
    NVMEDIA_VIDEO_CODEC_MJPEG,
    /** @brief VP8 codec */
    NVMEDIA_VIDEO_CODEC_VP8,
    /** @brief H265 codec */
    NVMEDIA_VIDEO_CODEC_HEVC,
    /** @brief VP9 codec */
    NVMEDIA_VIDEO_CODEC_VP9,
    /** @brief H.264 Multiview Video Coding codec */
    NVMEDIA_VIDEO_CODEC_H264_MVC,
    /** @brief H265 Multiview Video Coding codec */
    NVMEDIA_VIDEO_CODEC_HEVC_MV,
    /** @brief AV1 Video Coding codec */
    NVMEDIA_VIDEO_CODEC_AV1,
    /** \note This value is for internal use only. */
    NVMEDIA_VIDEO_CODEC_END
} NvMediaVideoCodec;

/**
 * @brief Holds an application data buffer containing compressed video
 *        data.
 */
typedef struct {
    /** A pointer to the bitstream data bytes. */
    uint8_t *bitstream;
    /** The number of data bytes */
    uint32_t bitstreamBytes;
    /** Size of bitstream array */
    uint32_t bitstreamSize;
} NvMediaBitstreamBuffer;

/** @} <!-- Ends common_types_top group NvMedia Types and Structures common to
            Encode and Decode --> */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_COMMON_ENCODE_DECODE_H */
