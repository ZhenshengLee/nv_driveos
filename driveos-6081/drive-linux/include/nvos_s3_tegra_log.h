/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited
 */
#ifndef INCLUDED_NVOS_S3_LOG_H
#define INCLUDED_NVOS_S3_LOG_H

#include <stdint.h>
#if defined(__QNX__)
#include "qnx/nvos_log_codes.h"
#include "qnx/nvos_tegra_log_defs.h"
#elif defined(__linux__)
#include "linux/nvos_log_codes.h"
#include "linux/nvos_tegra_log_defs.h"
#elif NVOS_IS_HOS
#include "../../core-hos/include/hos/nvos_log_codes.h"
#include "../../core-hos/include/hos/nvos_tegra_log_defs.h"
#endif

#include "nvos_tegra_log_codes.h"

#if defined(__cplusplus)
extern "C"
{
#endif

/** @defgroup S3_Safe_Log S3 Safe Logging */

/**
 * @ingroup S3_Safe_Log
 * @brief This function logs a string in a log buffer.
 *
 *  SWUD ID: QNXBSP-NVOS-S3-100
 *
 * @param[in] module_id The user specific code that is associated the message.
 * @param[in] severity The severity level of this log.
 * @param[in] str The string that should be logged.
 *
 * @return N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: N/A
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
void NvOsDebugPrintStr(
    uint16_t module_id,
    uint8_t severity,
    const char *str);

/**
 * @ingroup S3_Safe_Log
 * @brief This function logs two different strings in a log buffer.
 *
 *  SWUD ID: QNXBSP-NVOS-S3-100
 *
 * @param[in] module_id The user specific code that is associated the message.
 * @param[in] severity The severity level of this log.
 * @param[in] str1 The string that should be logged.
 * @param[in] str2 The string that should be logged.
 *
 * @return N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: N/A
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
void NvOsDebugPrintStrStr(
    uint16_t module_id,
    uint8_t severity,
    const char *str1,
    const char *str2);

/**
 * @ingroup S3_Safe_Log
 * @brief This function logs a string and int32_t value in a log buffer.
 *
 *  SWUD ID: QNXBSP-NVOS-S3-101
 *
 * @param[in] module_id The user specific code that is associated the message.
 * @param[in] severity The severity level of this log.
 * @param[in] str The string that should be logged.
 * @param[in] val The int32_t value that should be logged.
 *
 * @return N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: N/A
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
void NvOsDebugPrintStrInt(
    uint16_t module_id,
    uint8_t severity,
    const char *str,
    int32_t val);

/**
 * @ingroup S3_Safe_Log
 * @brief This function logs a string and an uint32_t value in a log buffer.
 *
 *  SWUD ID: QNXBSP-NVOS-S3-102
 *
 * @param[in] module_id The user specific code that is associated the message.
 * @param[in] severity The severity level of this log.
 * @param[in] str The string that should be logged.
 * @param[in] val The unsigned int value that should be logged.
 *
 * @return N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: N/A
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
void NvOsDebugPrintStrUInt(
    uint16_t module_id,
    uint8_t severity,
    const char *str,
    uint32_t val);

/**
 * @ingroup S3_Safe_Log
 * @brief This function logs a string and int64_t value in a log buffer.
 *
 *  SWUD ID: QNXBSP-NVOS-S3-103
 *
 * @param[in] module_id The user specific code that is associated the message.
 * @param[in] severity The severity level of this log.
 * @param[in] str The string that should be logged.
 * @param[in] val The int64_t value that should be logged.
 *
 * @return N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: N/A
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
void NvOsDebugPrintStrSLong(
    uint16_t module_id,
    uint8_t severity,
    const char *str,
    int64_t val);

/**
 * @ingroup S3_Safe_Log
 * @brief This function logs a string and uint64_t value in a log buffer.
 *
 *  SWUD ID: QNXBSP-NVOS-S3-104
 *
 * @param[in] module_id The user specific code that is associated the message.
 * @param[in] severity The severity level of this log.
 * @param[in] str The string that should be logged.
 * @param[in] val The uint64_t value that should be logged.
 *
 * @return N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: N/A
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
void NvOsDebugPrintStrULong(
    uint16_t module_id,
    uint8_t severity,
    const char *str,
    uint64_t val);

/**
 * @ingroup S3_Safe_Log
 * @brief This function logs a string with two int32_t values in a log buffer.
 *
 *  SWUD ID: QNXBSP-NVOS-S3-105
 *
 * @param[in] module_id The user specific code that is associated the message.
 * @param[in] severity The severity level of this log.
 * @param[in] str The string that should be logged.
 * @param[in] value1 The int32_t value that should be logged.
 * @param[in] value2 The int32_t value that should be logged.
 *
 * @return N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: N/A
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
void NvOsDebugPrintStrWith2Int(
    uint16_t module_id,
    uint8_t severity,
    const char *str,
    int32_t value1,
    int32_t value2);

/**
 * @ingroup S3_Safe_Log
 * @brief This function logs a string with two int64_t values in a log buffer.
 *
 *  SWUD ID: QNXBSP-NVOS-S3-106
 *
 * @param[in] module_id The user specific code that is associated the message.
 * @param[in] severity The severity level of this log.
 * @param[in] str The string that should be logged.
 * @param[in] value1 The int64_t value that should be logged.
 * @param[in] value2 The int64_t value that should be logged.
 *
 * @return N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: N/A
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
void NvOsDebugPrintStrWith2SLong(
    uint16_t module_id,
    uint8_t severity,
    const char *str,
    int64_t value1,
    int64_t value2);

/**
 * @ingroup S3_Safe_Log
 * @brief This function logs a string with two uint32_t values in a log buffer.
 *
 *  SWUD ID: QNXBSP-NVOS-S3-107
 *
 * @param[in] module_id The user specific code that is associated the message.
 * @param[in] severity The severity level of this log.
 * @param[in] str The string that should be logged.
 * @param[in] value1 The uint32_t value that should be logged.
 * @param[in] value2 The uint32_t value that should be logged.
 *
 * @return N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: N/A
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
void NvOsDebugPrintStrWith2UInt(
    uint16_t module_id,
    uint8_t severity,
    const char *str,
    uint32_t value1,
    uint32_t value2);

/**
 * @ingroup S3_Safe_Log
 * @brief This function logs a string with two uint64_t values in a log buffer.
 *
 *  SWUD ID: QNXBSP-NVOS-S3-108
 *
 * @param[in] module_id The user specific code that is associated the message.
 * @param[in] severity The severity level of this log.
 * @param[in] str The string that should be logged.
 * @param[in] value1 The uint64_t value that should be logged.
 * @param[in] value2 The uint64_t value that should be logged.
 *
 * @return N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: N/A
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
void NvOsDebugPrintStrWith2ULong(
    uint16_t module_id,
    uint8_t severity,
    const char *str,
    uint64_t value1,
    uint64_t value2);

/**
 * @ingroup S3_Safe_Log
 * @brief This function logs a string and unsigned int value in hexadecimal to
 * a log buffer.
 *
 *  SWUD ID: QNXBSP-NVOS-S3-109
 *
 * @param[in] module_id The user specific code that is associated the message.
 * @param[in] severity The severity level of this log.
 * @param[in] str The string that should be logged.
 * @param[in] val The unsigned int value that should be logged.
 *
 * @return N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: N/A
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
void NvOsDebugPrintStrHexUInt(
    uint16_t module_id,
    uint8_t severity,
    const char *str,
    uint32_t val);

/**
 * @ingroup S3_Safe_Log
 * @brief This function logs a string and uint64_t value in hexadecimal to a
 * log buffer.
 *
 *  SWUD ID: QNXBSP-NVOS-S3-110
 *
 * @param[in] module_id The user specific code that is associated the message.
 * @param[in] severity The severity level of this log.
 * @param[in] str The string that should be logged.
 * @param[in] val The uint64_t value that should be logged.
 *
 * @return N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: N/A
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
void NvOsDebugPrintStrHexULong(
    uint16_t module_id,
    uint8_t severity,
    const char *str,
    uint64_t val);

/**
 * @ingroup S3_Safe_Log
 * @brief This function logs a string and two uint64_t values in hexadecimal to a
 * log buffer.
 *
 * @param[in] module_id The user specific code that is associated the message.
 * @param[in] severity The severity level of this log.
 * @param[in] str The string that should be logged.
 * @param[in] val1 The uint64_t value that should be logged.
 * @param[in] val2 The uint64_t value that should be logged.
 *
 * @return N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: N/A
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
void NvOsDebugPrintStrWith2HexULong(
    uint16_t module_id,
    uint8_t severity,
    const char *str,
    uint64_t val1,
    uint64_t val2);

/**
 * @ingroup S3_Safe_Log
 * @brief This function logs a string and float value to a log buffer.
 *
 *  SWUD ID: QNXBSP-NVOS-S3-111
 *
 * @param[in] module_id The user specific code that is associated the message.
 * @param[in] severity The severity level of this log.
 * @param[in] str The string that should be logged.
 * @param[in] val The float value that should be logged.
 *
 * @return N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: N/A
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
void NvOsDebugPrintStrFloat(
    uint16_t module_id,
    uint8_t severity,
    const char *str,
    float val);

/**
 * @ingroup S3_Safe_Log
 * @brief This function logs a string and two float values to a log buffer.
 *
 *  SWUD ID: QNXBSP-NVOS-S3-112
 *
 * @param[in] module_id The user specific code that is associated the message.
 * @param[in] severity The severity level of this log.
 * @param[in] str The string that should be logged.
 * @param[in] val1 The float value that should be logged.
 * @param[in] val2 The float value that should be logged.
 *
 * @return N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: N/A
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
void NvOsDebugPrintStrWith2Float(
    uint16_t module_id,
    uint8_t severity,
    const char *str,
    float val1,
    float val2);

/**
 * @ingroup S3_Safe_Log
 * @brief This function logs a string and a double value to a log buffer.
 *
 *  SWUD ID: QNXBSP-NVOS-S3-113
 *
 * @param[in] module_id The user specific code that is associated the message.
 * @param[in] severity The severity level of this log.
 * @param[in] str The string that should be logged.
 * @param[in] val The float value that should be logged.
 *
 * @return N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: N/A
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
void NvOsDebugPrintStrDouble(
    uint16_t module_id,
    uint8_t severity,
    const char *str,
    double val);

/**
 * @ingroup S3_Safe_Log
 * @brief This function logs a string and two double values to a log buffer.
 *
 *  SWUD ID: QNXBSP-NVOS-S3-114
 *
 * @param[in] module_id The user specific code that is associated the message.
 * @param[in] severity The severity level of this log.
 * @param[in] str The string that should be logged.
 * @param[in] val1 The double value that should be logged.
 * @param[in] val2 The double value that should be logged.
 *
 * @return N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: Yes
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: N/A
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
void NvOsDebugPrintStrWith2Double(
    uint16_t module_id,
    uint8_t severity,
    const char *str,
    double val1,
    double val2);

#if defined(__cplusplus)
}
#endif
#endif /* INCLUDED_NVOS_S3_LOG_H */

