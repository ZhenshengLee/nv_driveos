/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 * All information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "cdd_nv_error.h"
#include <stdio.h>
#include <stdarg.h>

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY

/**
 * Variable to store log verbosity level
 */
static C_LogLevel level = LOG_LEVEL_NONE;

void SetCLogLevel(C_LogLevel eLevel)
{
    level = eLevel;
}

void PrintLogMsg(
    C_LogLevel eLevel,
    const char *pformat,
    ...
)
{
    if (level >= eLevel) {
        if (pformat == NULL) {
            return;
        }

        va_list args;
        va_start(args, pformat);
        vprintf(pformat, args);
        va_end(args);
    }
}

#else /* NV_IS_SAFETY */

void SetCLogLevel(C_LogLevel eLevel)
{
}

void PrintLogMsg(
    C_LogLevel eLevel,
    const char *pformat,
    ...
)
{
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif