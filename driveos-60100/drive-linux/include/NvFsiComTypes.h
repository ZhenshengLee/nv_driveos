/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/**
 * @file NvFsiComTypes.h
 * @brief NvFsiCom data types
 */

#ifndef NVFSICOMTYPES_H
#define NVFSICOMTYPES_H

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <sivc-instance.h>

#ifdef __cplusplus
using namespace Ivc;
#endif

/**
 * @brief Header size in FsiCom data frame
 */
#define NVFSICOM_HEADER_LEN 4U

/**
 * @brief Type definition for FsiCom channel ID
 */
typedef uint8_t NvFsiComChId_t;

/**
 * @brief Typedef representing array index of NvFsiComHandle_t
 */
typedef uint8_t NvFsiHandleIndex;

/**
 * @brief Structure representing IVC configuration for a channel
 */
typedef struct {
  struct sivc_queue queue;  /**< SIVC queue handle */
  uintptr_t  recv_base;  /**< Pointer to RX SIVC queue (64byte header + nFrames * frameSize) */
  uintptr_t  send_base;  /**< Pointer to TX SIVC queue (64byte header + nFrames * frameSize) */
  uint32_t nFrames;      /**< Total number frames configured for SIVC queue*/
  uint32_t FrameSize;    /**< Size of a frame in bytes */
} IvcChConfig_t;

#pragma pack(push,1)

/**
 * @brief Type definition for NvFsiCom handle
 */
typedef struct NvFsiComHandle
{
  NvFsiComChId_t ChId;  /**< Channel ID for FSICOM channel */
  uint8_t CoreId; /**< FSI Core ID for given FSICOM channel */
  IvcChConfig_t IvcQueue; /**< SIVC queue configuration for given FSICOM channel */
} NvFsiComHandle_t;

#pragma pack(pop)
#endif
