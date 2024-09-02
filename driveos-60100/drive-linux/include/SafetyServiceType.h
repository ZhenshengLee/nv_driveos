/*
 * Copyright (c) 2022-23, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef SafetyServiceType_H
#define SafetyServiceType_H

#include <stdint.h>
#include <stdbool.h>

/*=======================[type definitions]===================================*/

/**
 * @brief Type definition for error report frame
 * 
 * It contains an error code and additional information like attribute, timestamp
 * and source of an error.
 */
typedef struct
{
  /**
   * Error code indicates error reported by corresponding reporter_id
   */
  uint32_t ErrorCode;

  /**
   * Extra information for SEH to understand error
   */
  uint32_t Error_Attribute;

  /**
   * LSB 32-bit TSC counter when error is detected
   */
  uint32_t timestamp;

  /**
   * Indicates source of error
   */
  uint16_t ReporterId;
}SS_ErrorReportFrame_t;

/**
 * @brief Type definition for critical failure report
 *
 */
typedef struct
{
  /**
   * Error report frame
   */
  SS_ErrorReportFrame_t ErrorReportFrame;

  /**
   * System failure ID
   */
  uint16_t SystemFailureId;

  /**
   * System maturation state
   */
  uint8_t MaturationState;
} SS_NvSehCriticalFailure_t;

/**
 * @brief Enum for different return values / error codes
 *
 */
typedef enum
{
  /**
   * Precondition on API is not satisfied
   */
  SS_E_PRECON = -2,

  /**
   * Any other error has occured
   */
  SS_E_NOK = -1,

  /**
   * Success
   */
  SS_E_OK  = 0
}SS_ReturnType_t;

#endif
