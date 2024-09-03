/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CDD_NV_ERROR
#define CDD_NV_ERROR

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Log level enum
 *
 * The entries in this enum should match the entries
 * of INvSIPLDeviceBlockTrace::TraceLevel enum.
 */
typedef enum {
    LOG_LEVEL_NONE = 0,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_WARN,
    LOG_LEVEL_INFO,
    LOG_LEVEL_DEBUG
} C_LogLevel;

void SetCLogLevel(C_LogLevel level);

void PrintLogMsg(
    C_LogLevel eLevel,
    const char *pformat,
    ...
);

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* CDD_NV_ERROR */
